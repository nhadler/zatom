import torch

from zatom.utils.typing_utils import typecheck


@typecheck
def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs: Probability distribution tensor.
        p: Probability threshold for top-p sampling.

    Returns:
        Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold `p`. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
