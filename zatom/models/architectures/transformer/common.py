from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zatom.utils.typing_utils import typecheck


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class ChargeSpinEmbedding(nn.Module):
    """Charge or spin embedding module.

    Args:
        embedding_type (Literal["pos_emb", "lin_emb", "rand_emb"]): Type of embedding to use.
        embedding_target (Literal["charge", "spin"]): Target to embed.
        embedding_size (int): Size of the embedding.
        grad (bool): Whether to allow gradients for the embedding parameters.
        scale (float, optional): Scale for the positional embedding. Defaults to 1.0.
    """

    @typecheck
    def __init__(
        self,
        embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"],
        embedding_target: Literal["charge", "spin"],
        embedding_size: int,
        grad: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type
        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target
        assert embedding_size % 2 == 0, f"{embedding_size=} must be even"

        if self.embedding_target == "charge":
            # NOTE: 100 is a conservative upper bound
            self.target_dict = {str(x): x + 100 for x in range(-100, 101)}
        elif self.embedding_target == "spin":
            # NOTE: 100 is a conservative upper bound
            self.target_dict = {str(x): x for x in range(101)}

        if self.embedding_type == "pos_emb":
            # NOTE: Dividing by 2 because x_proj multiplies by 2
            if not grad:
                self.W = nn.Parameter(
                    torch.randn(embedding_size // 2) * scale, requires_grad=False
                )
            else:
                self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=True)
        elif self.embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(in_features=1, out_features=embedding_size)
            if not grad:
                for param in self.lin_emb.parameters():
                    param.requires_grad = False
        elif self.embedding_type == "rand_emb":
            self.rand_emb = nn.Embedding(len(self.target_dict), embedding_size)
            if not grad:
                for param in self.rand_emb.parameters():
                    param.requires_grad = False

        else:
            raise ValueError(f"Embedding type {self.embedding_type} not implemented.")

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for charge or spin embedding.

        Args:
            x (Tensor): Input tensor of shape (N,).

        Returns:
            Tensor: Embedded tensor of shape (N, embedding_size).
        """
        # NOTE: Null token for spin is 0,
        # while charge is default 0
        if self.embedding_type == "pos_emb":
            x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
            if self.embedding_target == "charge":
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            elif self.embedding_target == "spin":
                zero_idxs = torch.where(x == 0)[0]
                emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
                # NOTE: This sets the null spin embedding to zero
                emb[zero_idxs] = 0
                return emb
        elif self.embedding_type == "lin_emb":
            if self.embedding_target == "spin":
                x[x == 0] = -100
            return self.lin_emb(x.unsqueeze(-1).float())
        elif self.embedding_type == "rand_emb":
            return self.rand_emb(
                torch.tensor(
                    [self.target_dict[str(i)] for i in x.tolist()],
                    device=x.device,
                    dtype=torch.long,
                )
            )
        raise ValueError(f"Embedding type {self.embedding_type} not implemented.")
