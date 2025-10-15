"""Transformer encoder.

Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer.
"""

from typing import Type

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from zatom.models.architectures.encoders.custom_transformer import (
    Block,
    LayerNorm,
    Mlp,
    build_attention_mask,
)
from zatom.utils.typing_utils import Bool, Float, Int, typecheck


# Modules
class TransformerEncoder(nn.Module):
    """Standard Transformer encoder.

    Args:
        d_model: Dimension of the model.
        nhead: Number of attention heads.
        num_layers: Number of layers.
        context_length: Context length for the attention mechanism.
        rope_base: Base for the rotary positional encoding.
        mlp_ratio: Ratio of the hidden dimension to the embedding dimension in the feedforward network.
        proj_drop: Dropout rate for the projection layer.
        attn_drop: Dropout rate for the attention layer.
        qkv_bias: Whether to use bias in the query, key, and value projections.
        qk_norm: Whether to use normalization on the query and key projections.
        scale_attn_norm: Whether to scale the attention normalization.
        scale_mlp_norm: Whether to scale the MLP normalization.
        proj_bias: Whether to use bias in the projection layer.
        flex_attn: Whether to use flex-attention.
        fused_attn: Whether to use fused (i.e., Flash) attention.
        jvp_attn: Whether to use Jacobian-vector product (JVP) attention.
        checkpoint_activations: Whether to checkpoint activations.
        act_layer: Type of activation layer to use.
        norm_layer: Type of normalization layer to use.
        mlp_layer: Type of MLP layer to use.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 8,
        context_length: int | None = 2048,
        rope_base: int | None = 10_000,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.1,
        attn_drop: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = False,
        flex_attn: bool = False,
        fused_attn: bool = True,
        jvp_attn: bool = False,
        checkpoint_activations: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__()

        assert (
            sum([flex_attn, fused_attn, jvp_attn]) <= 1
        ), "Only one of flex_attn, fused_attn, or jvp_attn can be True."

        self.d_model = d_model
        self.nhead = nhead
        self.context_length = context_length
        self.flex_attn = flex_attn
        self.jvp_attn = jvp_attn

        self.atom_type_embedder = nn.Linear(d_model * 2, d_model, bias=True)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.lengths_scaled_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.angles_radians_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Embeddings
        self.transformer = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=nhead,
                    context_length=context_length,
                    rope_base=rope_base,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_attn_norm=scale_attn_norm,
                    scale_mlp_norm=scale_mlp_norm,
                    proj_bias=proj_bias,
                    flex_attn=flex_attn,
                    fused_attn=fused_attn,
                    jvp_attn=jvp_attn,
                    checkpoint_activations=checkpoint_activations,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(num_layers)
            ]
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    @typecheck
    def forward(
        self,
        atom_types: Float["b m c2"],  # type: ignore
        pos: Float["b m 3"],  # type: ignore
        frac_coords: Float["b m 3"],  # type: ignore
        lengths_scaled: Float["b 1 3"],  # type: ignore
        angles_radians: Float["b 1 3"],  # type: ignore
        token_idx: Int["b m"],  # type: ignore
        mask: Bool["b m"],  # type: ignore
        attn_mask: Bool["b h m m"] | Float["b 1 m m"] | None = None,  # type: ignore
        seq_idx: Int["b m"] | None = None,  # type: ignore
    ) -> Float["b m c"]:  # type: ignore
        """Forward pass for the Transformer encoder.

        Args:
            atom_types: Combined input and predicted (double channel) atom embeddings for the batch.
            pos: Cartesian coordinates of atoms in the batch.
            frac_coords: Fractional coordinates of atoms in the batch.
            lengths_scaled: Lattice lengths tensor (with a singular global value for each batch element).
            angles_radians: Lattice angles tensor (with a singular global value for each batch element).
            token_idx: Indices of tokens in the batch.
            mask: Attention mask for the batch.
            attn_mask: Pre-computed attention mask for the batch (optional).
            seq_idx: Indices of unique token sequences in the batch (optional unless using sequence packing).

        Returns:
            Encoded token features.
        """
        batch_size, num_tokens, _ = atom_types.shape

        x = self.atom_type_embedder(atom_types)  # [B, M, D * 2] -> [B, M, D]
        x += self.pos_embedder(pos)
        x += self.frac_coords_embedder(frac_coords)
        x += self.lengths_scaled_embedder(lengths_scaled)
        x += self.angles_radians_embedder(angles_radians)

        # Create the attention mask
        if attn_mask is None:
            if seq_idx is None:
                seq_idx = torch.ones_like(token_idx)

            def padded_document_mask_mod(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
            ) -> torch.Tensor:
                """Create a padded document mask for the attention mechanism.

                Args:
                    b: Batch index.
                    h: Head index (not used in this implementation).
                    q_idx: Index of the query token.
                    kv_idx: Index of the key-value tokens.

                Returns:
                    A boolean tensor value.
                """
                seq_ids = seq_idx
                non_padding_mask = (seq_ids[b, q_idx] != 0) & (seq_ids[b, kv_idx] != 0)
                document_mask = seq_ids[b, q_idx] == seq_ids[b, kv_idx]
                return non_padding_mask & document_mask

            attn_mask = (
                create_block_mask(
                    mask_mod=padded_document_mask_mod,
                    B=batch_size,
                    H=None,
                    Q_LEN=num_tokens,
                    KV_LEN=num_tokens,
                    device=self.device,
                )
                if self.flex_attn
                else build_attention_mask(
                    mask, seq_idx, dtype=torch.bool if self.jvp_attn else x.dtype
                )
            )
            if self.jvp_attn:
                attn_mask = attn_mask.expand(-1, self.nhead, -1, -1)  # [B, H, N, N]

        # Transformer blocks
        for block in self.transformer:
            x = block(x, pos_ids=token_idx, attn_mask=attn_mask)  # [B, M, D]

        return x
