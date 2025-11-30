"""Adapted from https://github.com/carlosinator/tabasco."""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from platonic_transformers.models.platoformer.ape import PlatonicAPE
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.io import lift_vectors
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from torch import Tensor

from zatom.utils.typing_utils import typecheck


class PositionalEncoding(ABC, nn.Module):
    """Abstract interface for modules that add positional information to tensors."""

    @typecheck
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Return positional encodings.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor containing positional encodings.
        """
        pass

    @typecheck
    @abstractmethod
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        pass


class SinusoidEncoding(PositionalEncoding):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        posenc_dim: Size of the embedding dimension.
        max_len: Maximum sequence length supported. `seq_len` passed to
            `forward` must not exceed this value.
        random_permute: If `True`, the positions are randomly permuted for each
            sample (useful as lightweight data augmentation but destroys absolute
            ordering).
    """

    @typecheck
    def __init__(self, posenc_dim: int, max_len: int = 100, random_permute: bool = False):
        super().__init__()

        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

        pos_embed = torch.zeros(max_len, self.posenc_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.posenc_dim, 2).float()
            * (-math.log(10 * self.max_len) / self.posenc_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)

        self.register_buffer("pos_embed", pos_embed, persistent=False)

    @typecheck
    def forward(self, batch_size: int, seq_len: int) -> Tensor:
        """Return positional embeddings of shape `(batch_size, seq_len, posenc_dim)`.

        Args:
            batch_size: Number of samples in the batch.
            seq_len: Length of the sequence.

        Returns:
            Tensor of shape `(batch_size, seq_len, posenc_dim)`.

        NOTE: `seq_len` must not exceed the `max_len` passed at construction.
        """
        pos_embed = self.pos_embed[:, :seq_len, :]
        pos_embed = pos_embed.expand(batch_size, -1, -1)

        if self.random_permute:
            # a fast way to do batched random permutations
            batch_size, seq_len, dim = pos_embed.shape
            perm = torch.argsort(torch.rand(batch_size, seq_len, device=pos_embed.device), dim=1)
            pos_embed = pos_embed.gather(1, perm.unsqueeze(-1).expand(batch_size, seq_len, dim))

        return pos_embed

    @typecheck
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.posenc_dim


class TimeFourierEncoding(PositionalEncoding):
    """Encoder for continuous timesteps in `[0, 1]`.

    Args:
        posenc_dim: Size of the embedding dimension.
        max_len: Maximum value of the input timestep after scaling.
        random_permute: If `True`, the positions are randomly permuted for each
            sample (useful as lightweight data augmentation but destroys absolute
            ordering).
    """

    @typecheck
    def __init__(self, posenc_dim: int, max_len: int = 100, random_permute: bool = False):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """Encode a tensor of timesteps.

        Args:
            t: 1-D tensor with values in `[0, 1]`.

        Returns:
            Tensor of shape `(B, posenc_dim)` with sine/cosine features.
        """
        t_scaled = t * self.max_len
        half_dim = self.posenc_dim // 2
        emb = math.log(self.max_len) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = torch.outer(t_scaled.float(), emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.posenc_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")

        assert emb.shape == (
            t.shape[0],
            self.posenc_dim,
        ), f"Expected shape ({t.shape[0], self.posenc_dim}), got {emb.shape}"

        return emb

    @typecheck
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.posenc_dim


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
            self.group = PLATONIC_GROUPS[solid_name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown solid '{solid_name}'. Available options are {list(PLATONIC_GROUPS.keys())}"
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
