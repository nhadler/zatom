import torch
import torch.nn as nn
from platonic_transformers.models.platoformer.ape import PlatonicAPE
from platonic_transformers.models.platoformer.io import lift_vectors
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from torch import Tensor

from zatom.models.architectures.platoformer import PLATONIC_GROUPS_3D
from zatom.models.architectures.transformer.positional_encoder import PositionalEncoding
from zatom.utils.typing_utils import typecheck


class PlatonicSinusoidAPE(PlatonicAPE, PositionalEncoding):
    """Aliasing PlatonicAPE to more descriptive PlatonicSinusoidAPE and adding out_dim method."""

    @typecheck
    def out_dim(self) -> int:
        return self.embed_dim


class PlatonicLinearAPE(PositionalEncoding):
    """Group-Equivariant Absolute Position Encoding via lift_vectors followed by PlatonicLinear.

    Maps Euclidean vector batches (.., spatial_dims) to regular rep embedding batches (.., embed_dim),
    where embed_dim = G * embed_dim//G.

    This module was added as a simple Platonic group equivariant counterpart of the *linear*
    pos_embed submodule of TransformerModule. Its interface and in/out shapes are similar to the
    more complicated sinusoidal embedding based PlatonicSinusoidAPE from platoformer.

    Args:
        embed_dim (int):    The total dimension of the output embedding.
                            Must be divisible by the group size G.
        spatial_dims (int): The number of spatial dimensions of the input positions (e.g., 3).
        solid_name (str):   Name of the Platonic solid ('tetrahedron', 'octahedron', 'icosahedron')
                            to define the symmetry group.
    """

    @typecheck
    def __init__(self, embed_dim: int, spatial_dims: int, solid_name: str):
        super().__init__()

        try:
            self.group = PLATONIC_GROUPS_3D[solid_name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown solid '{solid_name}'. Available options are {list(PLATONIC_GROUPS_3D.keys())}"
            )

        if embed_dim % self.group.G != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by group size G ({self.group.G})."
            )

        self.embed_dim = embed_dim
        self.spatial_dims = spatial_dims

        self.platonic_linear = PlatonicLinear(
            in_features=spatial_dims * self.group.G,  # after lifting pos
            out_features=embed_dim,
            solid=solid_name,
        )

    @typecheck
    def forward(self, pos: Tensor) -> Tensor:
        """Compute group-equivariant absolute position embeddings via lift_vectors followed by
        PlatonicLinear.

        Args:
            pos (Tensor): Position tensor of shape (..., spatial_dims).

        Returns:
            Tensor: The calculated position embedding of shape (..., embed_dim).
        """
        assert pos.size(-1) == self.spatial_dims

        # Lift position vectors to regular representations.
        # lift_vectors assumes input shape (..., C, dim) and produces output shape (..., G, C*dim).
        # Since pos has shape (..., 3), we first need to unsqueeze a channel axis of size 1.
        pos = pos.unsqueeze(-2)  # (..., 1, dim)
        pos_lift = lift_vectors(pos, self.group)  # (..., G, dim)
        pos_lift = pos_lift.flatten(-2, -1)  # (..., G*dim)

        # PlatonicLinear layer map to embed_dim  (grouped into G*num_regular_reps)
        embedding = self.platonic_linear(pos_lift)  # (..., embed_dim)

        return embedding

    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.embed_dim
