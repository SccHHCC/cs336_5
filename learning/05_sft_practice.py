"""Day 1: implement the SFT primitives yourself before viewing existing code."""

import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return entropy over the last (vocabulary) dimension.

    Input: [batch, sequence, vocabulary]; output: [batch, sequence].
    Use a numerically stable expression. Labels are not needed.
    """
    raise NotImplementedError("Implement compute_entropy")


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked elements along dim and divide by normalize_constant.

    dim=None means sum all dimensions. Use the given constant, even when
    it differs from the number of included elements.
    """
    raise NotImplementedError("Implement masked_normalize")


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Select label log-probabilities from causal LM logits.

    Return 'log_probs' with shape [batch, sequence]. Optionally also return
    'token_entropy'. Preserve gradients for training. The caller supplies
    shifted labels; do not shift them a second time.
    """
    raise NotImplementedError("Implement get_response_log_probs after the first review")


if __name__ == "__main__":
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]])
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    print("entropy:", compute_entropy(logits))
    print("normalized by row:", masked_normalize(values, mask, 2.0, dim=1))
    print("normalized total:", masked_normalize(values, mask, 2.0))
