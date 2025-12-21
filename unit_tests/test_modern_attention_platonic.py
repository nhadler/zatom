import unittest

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.attention_platonic import (
    ModernAttentionPlatonic,
)


class TestModernAttentionPlatonic(unittest.TestCase):
    """Unit tests for ModernAttentionPlatonic."""

    def test_equivariance_modern_attention_platonic(self):
        """Equivariance unit tests for ModernAttentionPlatonic."""

        c_in, c_out, c_qk, c_val = 128, 256, 32, 16
        B, NQ, NKV, H = 16, 33, 42, 8
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        for solid_name in PLATONIC_GROUPS_3D:
            group = get_platonic_group(solid_name)
            G = group.G

            # Random input tensors
            feat_Q = torch.randn((B, NQ, G, c_in), device=device, dtype=dtype)
            feat_KV = torch.randn((B, NKV, G, c_in), device=device, dtype=dtype)
            coords_Q = torch.randn((B, NQ, 3), device=device, dtype=dtype)
            coords_KV = torch.randn((B, NKV, 3), device=device, dtype=dtype)

            sequence_lens_Q = torch.randint(NQ // 3, NQ, (B,), device=device, dtype=dtype)
            sequence_lens_KV = torch.randint(NKV // 3, NKV, (B,), device=device, dtype=dtype)
            sequence_lens_Q[B // 2] = NQ
            sequence_lens_KV[B // 2] = NKV
            padding_mask_Q = ~(
                torch.arange(NQ, device=device, dtype=dtype)[None, :] < sequence_lens_Q[:, None]
            )
            padding_mask_KV = ~(
                torch.arange(NKV, device=device, dtype=dtype)[None, :] < sequence_lens_KV[:, None]
            )
            avg_num_nodes = (~padding_mask_Q).sum(1).mean(0)

            attn_mask = torch.randn((B, H, NQ, NKV), device=device, dtype=dtype) < 1
            sequence_idxs_Q = torch.arange(NQ, device=device, dtype=dtype).repeat(
                B, 1
            )  # (B,NQ), only for self-attention

            # Loop over grid of all setting combinations that change the logic
            for use_sdpa, jvp_attn in ((True, False), (False, True), (False, False)):
                for freq_sigma_platonic in (1, None):
                    for linear_attention in (False, True):
                        for use_key in (False, True):
                            for use_qk_norm in (False, True):
                                for ctx_len, seq_rope_base in ((None, None), (2048, 10_000)):

                                    attn_module = ModernAttentionPlatonic(
                                        c_in,
                                        c_out,
                                        c_qk,
                                        c_val,
                                        H,
                                        solid_name,
                                        freq_sigma_platonic=freq_sigma_platonic,
                                        mean_aggregation=True,
                                        linear_attention=linear_attention,
                                        use_key=use_key,
                                        context_length=ctx_len,
                                        sequence_rope_base=seq_rope_base,
                                        use_qk_norm=use_qk_norm,
                                        use_sdpa=use_sdpa,
                                        jvp_attn=jvp_attn,
                                    ).to(device, dtype)

                                    # Loop over grid of all valid input tensor combinations
                                    for cross_attn in (False, True):
                                        for sequence_idxs_ in (None, sequence_idxs_Q):
                                            for padded_sequence in (False, True):
                                                for attn_mask_ in (None, attn_mask):

                                                    feat_KV_ = feat_KV if cross_attn else None
                                                    coords_KV_ = coords_KV if cross_attn else None
                                                    if padded_sequence and cross_attn:
                                                        padding_mask_KV_ = padding_mask_KV
                                                    elif padded_sequence and not cross_attn:
                                                        padding_mask_KV_ = padding_mask_Q
                                                    else:
                                                        padding_mask_KV_ = None

                                                    # Test equivariance for each group element
                                                    out = attn_module(
                                                        feat_Q.flatten(-2),
                                                        coords_Q.flatten(-2),
                                                        feat_KV_.flatten(-2),
                                                        coords_KV_.flatten(-2),
                                                        sequence_idxs_,
                                                        padding_mask_KV_,
                                                        attn_mask_,
                                                        avg_num_nodes,
                                                    ).view(B, NQ, G, c_out)

                                                    for g in range(G):
                                                        g_indices = group.cayley_table[g, :]
                                                        g_element = group.elements[g].to(
                                                            device, dtype
                                                        )

                                                        # G-transform original output
                                                        g_out = out[:, :, g_indices, :]

                                                        # G-transform input features
                                                        g_feat_Q = feat_Q[
                                                            :, :, g_indices, :
                                                        ].flatten(-2)
                                                        g_feat_KV_ = (
                                                            feat_KV_[:, :, g_indices, :].flatten(
                                                                -2
                                                            )
                                                            if feat_KV_ is not None
                                                            else None
                                                        )

                                                        # G-transform coordinates
                                                        g_coords_Q = torch.einsum(
                                                            "ji,...j ->...i", g_element, coords_Q
                                                        )
                                                        g_coords_KV_ = (
                                                            torch.einsum(
                                                                "ji,...j ->...i",
                                                                g_element,
                                                                coords_KV_,
                                                            )
                                                            if coords_KV_ is not None
                                                            else None
                                                        )

                                                        # Compute output for transformed inputs
                                                        out_g = attn_module(
                                                            g_feat_Q.flatten(-2),
                                                            g_coords_Q.flatten(-2),
                                                            g_feat_KV_.flatten(-2),
                                                            g_coords_KV_.flatten(-2),
                                                            sequence_idxs_,
                                                            padding_mask_KV_,
                                                            attn_mask_,
                                                            avg_num_nodes,
                                                        ).view(B, NQ, G, c_out)

                                                        self.assertTrue(
                                                            torch.allclose(
                                                                out_g, g_out, atol=1e-5
                                                            ),
                                                            "\nEquivariance test failed for:\n"
                                                            + f"  G:                      {solid_name}\n"
                                                            + f"  g:                      {g}\n"
                                                            + "\n"
                                                            + f"  use_sdpa, jvp_attn:     {use_sdpa}, {jvp_attn}\n"
                                                            + f"  freq_sigma_platonic:    {freq_sigma_platonic}\n"
                                                            + f"  linear_attention:       {linear_attention}\n"
                                                            + f"  use_key:                {use_key}\n"
                                                            + f"  use_qk_norm:            {use_qk_norm}\n"
                                                            + f"  ctx_len, seq_rope_base: {ctx_len}, {seq_rope_base}\n"
                                                            + "\n"
                                                            + f"  cross_attn:             {cross_attn}\n"
                                                            + f"  sequence_idxs_:         {sequence_idxs_}\n"
                                                            + f"  padded_sequence:        {padded_sequence}\n"
                                                            + f"  attn_mask_:             {attn_mask_}\n"
                                                            + "\n"
                                                            + f"  max difference:         {torch.max(torch.abs(out_g - g_out))}",
                                                        )


if __name__ == "__main__":
    unittest.main()
