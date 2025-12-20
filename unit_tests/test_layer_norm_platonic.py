import unittest

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.transformer_platonic import (
    LayerNormPlatonic,
)


class TestLayerNormPlatonic(unittest.TestCase):
    """Unit tests for LayerNormPlatonic."""

    def test_against_standard_layer_norm(self):
        """Check for correctness against non-functional implementation for
        normalize_per_g=False."""
        solid_name = "octahedron"
        group = get_platonic_group(solid_name)
        num_G = group.G
        C = 8 * num_G  # divisible by num_G
        B, M = 16, 33
        eps = 1e-5
        x = torch.randn(B, M, C)

        norm = LayerNormPlatonic(
            solid_name=solid_name,
            hidden_dim=C,
            normalize_per_g=False,
            elementwise_affine=True,
            eps=eps,
        )

        norm_baseline = nn.LayerNorm(
            normalized_shape=(C // num_G,),
            eps=eps,
            elementwise_affine=True,
        )

        # randomize parameters
        norm.weight.data = norm_baseline.weight.data = torch.randn_like(norm.weight)
        norm.bias.data = norm_baseline.bias.data = torch.randn_like(norm.bias)

        y_norm = norm(x)
        z_norm = norm_baseline(x.view(B, M, num_G, C // num_G)).flatten(-2)
        self.assertTrue(torch.allclose(y_norm, z_norm))

    def test_equivariance_layer_norm_platonic(self):
        """Equivariance unit tests for Platonic LayerNorm."""

        for solid_name in PLATONIC_GROUPS_3D:

            group = get_platonic_group(solid_name)
            num_G = group.G
            C = 8 * num_G  # divisible by num_G
            B, M = 16, 33
            x = torch.randn(B, M, C)

            for normalize_per_g in (True, False):
                for elementwise_affine in (True, False):

                    norm = LayerNormPlatonic(
                        solid_name=solid_name,
                        hidden_dim=C,
                        normalize_per_g=normalize_per_g,
                        elementwise_affine=elementwise_affine,
                    )

                    # randomize parameters
                    if norm.weight is not None:
                        norm.weight.data = torch.randn_like(norm.weight)
                    if norm.bias is not None:
                        norm.bias.data = torch.randn_like(norm.bias)

                    norm_x = norm(x)

                    for g in range(num_G):
                        g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep

                        norm_g_x = norm(
                            x.view(B, M, num_G, C // num_G)[:, :, g_indices, :].flatten(-2)
                        )
                        g_norm_x = norm_x.view(B, M, num_G, C // num_G)[
                            :, :, g_indices, :
                        ].flatten(-2)

                        self.assertTrue(
                            torch.allclose(norm_g_x, g_norm_x, atol=1e-5),
                            "\nEquivariance test failed for:\n"
                            + f"  G:              {solid_name}\n"
                            + f"  max difference: {torch.max(torch.abs(norm_g_x - g_norm_x))}",
                        )


if __name__ == "__main__":
    unittest.main()
