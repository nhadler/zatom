"""Adapted from https://github.com/carlosinator/tabasco."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jvp_flash_attention.jvp_attention import JVPAttn
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from platonic_transformers.models.platoformer.rope import PlatonicRoPE

from zatom.models.architectures.platoformer import get_platonic_group
from zatom.models.architectures.platoformer.layer_norm_platonic import LayerNormPlatonic
from zatom.models.architectures.transformer.positional_encoder import (
    RotaryPositionalEmbeddings as SequenceRoPE,
)
from zatom.utils.typing_utils import typecheck

# # They call Platonic MHA "dynamic convolution", aliasing it here to prevent confusion.
# from platonic_transformers.models.platoformer.conv import PlatonicConv as MultiheadAttentionPlatonic


class ModernAttentionPlatonic(nn.Module):
    """More flexible reimplementation of the (dense) Platonic attention layer "PlatonicConv".
    Allows additionally for cross-attention and adds functionality from ModernAttention, e.g.
    sequence RoPE besides Platonic Euclidean RoPE and JVPAttn backend.

    Args:
        TODO TODO TODO: UPDATE WITH NEW ARGS
        TODO TODO TODO: UPDATE WITH NEW ARGS
        TODO TODO TODO: UPDATE WITH NEW ARGS
        TODO TODO TODO: UPDATE WITH NEW ARGS
        TODO TODO TODO: UPDATE WITH NEW ARGS
        dim:            Input dimension
        n_heads:        Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base:      Base frequency for rotary embeddings
        use_qk_norm:    Whether to apply RMS normalization to queries and keys
        use_sdpa:       Whether to use PyTorch's scaled dot-product attention
        jvp_attn:       Whether to use JVP-compatible attention
    """

    @typecheck
    def __init__(
        self,
        c_in: int,  # per group element
        c_out: int,  # per group element
        c_qk: int,  # per group element and head
        c_val: int,  # per group element and head
        n_heads: int,
        ### Platonic attention specific args
        solid_name: str,
        freq_sigma_platonic: float = 1.0,
        freq_init_platonic: str = "random",
        learned_freqs_platonic: bool = True,
        bias: bool = False,
        mean_aggregation: bool = False,
        linear_attention: bool = False,
        use_key: bool = False,
        ### Modern attention specific args
        context_length: Optional[int] = 2048,
        sequence_rope_base: Optional[int] = 10_000,
        use_qk_norm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()

        self.group = get_platonic_group(solid_name)
        self.G = G = self.group.G  # number of group elements
        self.H = H = n_heads
        self.c_in = c_in
        self.c_qk = c_qk
        self.c_val = c_val
        self.c_out = c_out

        self.mean_aggregation = mean_aggregation
        self.linear_attention = linear_attention
        self.use_key = use_key
        self.use_qk_norm = use_qk_norm
        self.use_sdpa = use_sdpa
        self.jvp_attn = jvp_attn
        assert not (jvp_attn and use_sdpa), "Either jvp_attn or use_sdpa can be True, not both."

        # Platonic linear projectors
        self.q_proj = PlatonicLinear(G * c_in, G * H * c_qk, solid_name, bias=bias)
        self.v_proj = PlatonicLinear(G * c_in, G * H * c_val, solid_name, bias=bias)
        self.o_proj = PlatonicLinear(G * H * c_val, G * c_out, solid_name, bias=bias)
        # key-projection is optional in Platonic transformers
        if freq_sigma_platonic is None or use_key:
            self.k_proj = PlatonicLinear(G * c_in, G * H * c_qk, solid_name, bias=bias)
        else:
            self.k_proj = None

        # Query/key normalization
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(c_qk)
            self.k_norm = nn.RMSNorm(c_qk)

        # Sequence axis RoPE
        if context_length is not None and sequence_rope_base is not None:
            # Rotary embeddings on half the dims per head
            self.sequence_rope = SequenceRoPE(
                dim=c_qk,
                max_seq_len=context_length,
                base=sequence_rope_base,
            )
        else:
            self.sequence_rope = None

        # Platonic RoPE for Euclidean coordinates
        if freq_sigma_platonic is not None:
            self.platonic_rope = PlatonicRoPE(
                embed_dim=G * H * c_qk,
                num_heads=H,
                head_dim=c_qk,
                solid_name=solid_name,
                spatial_dims=3,
                freq_sigma=freq_sigma_platonic,
                learned_freqs=learned_freqs_platonic,
                freq_init=freq_init_platonic,
            )
        else:
            self.rope_platonic = None

    @typecheck
    def forward(
        self,
        feat_Q: torch.Tensor,
        coords_Q: torch.Tensor,
        feat_KV: Optional[torch.Tensor] = None,
        coords_KV: Optional[torch.Tensor] = None,
        sequence_idxs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        avg_num_nodes: Optional[float] = 1.0,
    ) -> torch.Tensor:
        """Forward pass through the Platonic group equivariant modern attention layer.

        Args:
            feat_Q:        Query feature tensor,                                    [B, NQ, G*c_in]
            coords_Q:      Query Euclidean coordinates for Platonic RoPE,           [B, NQ, 3]
            feat_KV:       Key/value feature tensor (optional),                     [B, NKV, G*c_in]
                           For cross-attention. If None, set to feat_Q.
            coords_KV:     Key/value Euclidean coords for Platonic RoPE (optional), [B, NKV, 3]
                           For cross-attention. If None, set to coords_Q.
            sequence_idxs: Sequence index tensor for sequence RoPE,                 [B, NQ]
                           Valid only for self-attention.
            padding_mask:  Boolean mask for padding tokens (True means ignore)      [B, NKV]
            attn_mask:     Attention mask (False or -inf indicates masked entries)  [B, H, NQ, NKV]
            avg_num_nodes: Used to normalize the dynamic convolution kernel if
                           linear_attention is True and no padding mask is passed.

        Returns:
            Projected attention output tensor                                       [B, NQ, G*c_out]
        """
        # Self-attention:  set KV-tensors = Q-tensors
        if feat_KV is not None:
            assert self.sequence_rope is None, "SequenceRope is only valid for self-attention."
        if feat_KV is None:
            feat_KV = feat_Q
        if coords_KV is None:
            coords_KV = coords_Q

        B, NQ, _ = feat_Q.shape
        NKV = feat_KV.size(1)
        G, H = self.G, self.H
        device = feat_Q.device
        dtype = feat_Q.dtype

        q = self.q_proj(feat_Q)
        v = self.v_proj(feat_KV)
        if self.k_proj is not None:
            k = self.k_proj(feat_KV)
        else:
            k = torch.ones((B, NKV, G * H * self.c_qk), dtype=dtype, device=device)

        q = q.reshape(B, NQ, G * H, self.c_qk)
        k = k.reshape(B, NKV, G * H, self.c_qk)
        v = v.reshape(B, NKV, G * H, self.c_val)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply sequence RoPE
        if self.sequence_rope is not None:
            assert (
                sequence_idxs is not None
            ), "sequence_idxs must be provided when using sequence RoPE."
            q = self.sequence_rope(q, input_pos=sequence_idxs)
            k = self.sequence_rope(k, input_pos=sequence_idxs)

        # Apply Platonic RoPE
        # Note: the order of sequence/Platonic RoPE commutes since SO(2) is Abelian / phases add up
        if self.platonic_rope is not None:
            q = q.view(B, NQ, G, H, self.c_qk)
            k = k.view(B, NKV, G, H, self.c_qk)
            q = self.platonic_rope(q, coords_Q)
            k = self.platonic_rope(k, coords_KV)
            q = q.view(B, NQ, G * H, self.c_qk)
            k = k.view(B, NKV, G * H, self.c_qk)

        # TODO TODO TODO TODO:  CHECK WHETHER .expand WORKS WITH   F.SDPA / JVPAttn   (stash, pull, ipdb in AttnModern)
        # TODO TODO TODO TODO:  CHECK WHETHER .expand WORKS WITH   F.SDPA / JVPAttn   (stash, pull, ipdb in AttnModern)
        # TODO TODO TODO TODO:  CHECK WHETHER .expand WORKS WITH   F.SDPA / JVPAttn   (stash, pull, ipdb in AttnModern)
        # TODO TODO TODO TODO:  CHECK WHETHER .expand WORKS WITH   F.SDPA / JVPAttn   (stash, pull, ipdb in AttnModern)
        # TODO TODO TODO TODO:  CHECK WHETHER .expand WORKS WITH   F.SDPA / JVPAttn   (stash, pull, ipdb in AttnModern)
        # TODO TODO TODO TODO:  if it works:  benchmark
        # TODO TODO TODO TODO:  if it works:  benchmark
        # TODO TODO TODO TODO:  if it works:  benchmark
        # TODO TODO TODO TODO:  if it works:  benchmark
        # TODO TODO TODO TODO:  if it works:  benchmark

        # Expand masks to (B, H, NQ, NKV), then merge
        # Note: expand doesn't allocate new memory, it just sets zero stride for broadcasted axes
        if attn_mask is not None:
            attn_mask = attn_mask.expand(B, H, NQ, NKV)

        if padding_mask is not None:
            padding_mask = padding_mask[:, None, None, :].expand(B, H, NQ, NKV)

            if attn_mask is None:
                attn_mask = ~padding_mask

            elif attn_mask.is_floating_point():
                # Create a float mask: 0.0 where we keep, -inf where we mask
                padding_mask_float = torch.zeros((B, 1, 1, NKV), device=device, dtype=dtype)
                padding_mask_float.masked_fill_(padding_mask, float("-inf"))
                attn_mask = attn_mask + padding_mask_float

            else:  # boolean attention mask
                attn_mask = attn_mask & (~padding_mask)

        ### Softmax attention
        if not self.linear_attention:
            q = q.transpose(1, 2)  # (B, G*H, NQ,  c_qk)
            k = k.transpose(1, 2)  # (B, G*H, NKV, c_qk)
            v = v.transpose(1, 2)  # (B, G*H, NKV, c_val)

            # Expand mask over group axis.
            # If the stride of the heads axis is zero, reshape is a view and the G*H has stride zero
            # as well, which makes the implementation memory IO efficient.
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).expand(B, G, H, NQ, NKV)
                attn_mask = attn_mask.reshape(B, G * H, NQ, NKV)

            if self.use_sdpa:
                # Use PyTorch's optimized scaled dot-product attention (SDPA)
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            elif self.jvp_attn:
                output = JVPAttn.fwd_dual(q, k, v, attn_mask=attn_mask)
            else:
                # Use manual implementation for comparison or environments where SDPA is not available
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                if attn_mask is not None and attn_mask.is_floating_point():
                    scores = scores + attn_mask
                elif attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                scores = F.softmax(scores.float(), dim=-1).type_as(q)
                output = torch.matmul(scores, v)

            output = output.transpose(1, 2).flatten(-2, -1)  # (B, NQ, G*H*c_val)

        ### Linear attention  (dynamic convolution)
        else:
            q = q.view(B, NQ, G, H, self.c_qk)
            k = k.view(B, NKV, G, H, self.c_qk)
            v = v.view(B, NKV, G, H, self.c_val)

            if attn_mask is not None:
                assert (
                    not attn_mask.is_floating_point()
                ), "attn_mask needs to be boolean if linear_attention is True"
                assert (attn_mask.size(2) == 1) or (
                    attn_mask.stride(2) == 0
                ), "Linear attention currently doesn't support query dependent masks"

                attn_mask = attn_mask[:, :, 0, :].transpose(1, 2)  # (B, NKV, H)
                v = v * attn_mask[:, :, None, :, None]
                k = k * attn_mask[:, :, None, :, None]

            kv_conv_kernel = torch.einsum("bnghc,bnghd->bghcd", k, v)

            if self.mean_aggregation and padding_mask is not None:
                num_nodes = (~padding_mask[:, 0, 0, :]).sum(dim=1).float().view(B, 1, 1, 1, 1)
            else:
                num_nodes = avg_num_nodes
            kv_conv_kernel = kv_conv_kernel / num_nodes

            output = torch.einsum("bnghc,bghcd->bnghd", q, kv_conv_kernel)  # (B, NQ, G, H, c_val)
            output = output.flatten(-3, -1)  # (B, NQ, G*H*c_val)

        return self.o_proj(output)  # (B, NQ, G*c_out)


class AttentionPlatonic(nn.Module):
    """TODO A wrapper around PyTorch's MultiheadAttention module with a simplified interface.

    This class provides a more convenient interface for using multi-head attention
    in transformer architectures, handling the reshaping and masking operations.

    Args:
        mha_platonic: MultiheadAttentionPlatonic instance

        dim:          Input and output dimension
        num_heads:    Number of attention heads
        dropout:      Dropout probability for attention weights
        bias:         Whether to include bias terms in the projection layers
        batch_first:  Whether input tensors are in batch-first format (batch, seq, features)
    """

    @typecheck
    def __init__(self, mha_platonic: MultiheadAttentionPlatonic):
        super().__init__()
        self.mha_platonic = mha_platonic

    @typecheck
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the multi-head attention layer.

        Args:
            query:            Query tensor of shape [batch_size, seq_len_q, dim]
            key:              Key   tensor of shape [batch_size, seq_len_k, dim] (defaults to query if None)
            value:            Value tensor of shape [batch_size, seq_len_v, dim] (defaults to key if None)
            key_padding_mask: Boolean mask for keys to ignore (True means ignore)
                              Shape: [batch_size, seq_len_k]
            attn_mask:        Mask to prevent attention to certain positions
                              Shape: [seq_len_q, seq_len_k] or [batch_size, seq_len_q, seq_len_k]
            need_weights:     Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim]
            (and optionally attention weights if need_weights=True)
        """
        key = query if key is None else key
        value = key if value is None else value

        attn_output, attn_weights = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        if need_weights:
            return attn_output, attn_weights

        return attn_output


class AttentionBlockPlatonic(nn.Module):
    """A block of Platonic self-attention layers with layer normalization.

    Args:
        dim:          Input dimension
        num_heads:    Number of attention heads
        norm:         LayerNormPlatonic instance
        dropout:      Dropout probability
        bias:         Whether to use bias in linear projections
        batch_first:  Whether input is batch-first (batch, seq, features)
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        norm: LayerNormPlatonic,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm = norm

        self.attention = AttentionPlatonic(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the attention block.

        Args:
            x:                Input tensor of shape [batch_size, seq_len, dim]
            pos:              Euclidean coordinates tensor of shape [batch_size, seq_len, 3],
                              required for Platonic RoPE.
            key_padding_mask: Boolean mask for keys to ignore (True means ignore)
                              Shape: [batch_size, seq_len]
            attn_mask:        Mask to prevent attention to certain positions
                              Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            need_weights:     Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
            (and optionally attention weights if need_weights=True)
        """
        x = self.norm(x)
        x = self.attention(
            x,
            pos,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        return x
