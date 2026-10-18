"""
This module contains the sampling utilities used in speculative decoding.
"""

import torch


def standardize(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Cast a sampling method into a probability distribution (Section 2.2 of the paper).

    Args:
        logits: The raw model scores, shape (..., vocab_size). Operates on the last dimension.
        temperature: The sampling temperature. 0 means argmax, which gives a one-hot distribution.
        top_k: Keep only the `top_k` highest-scoring tokens before sampling.
        top_p: Keep the smallest highest-probability set whose cumulative probability reaches `top_p`.

    Returns:
        The probabilities, same shape as `logits`, summing to 1 along the last dimension.

    Raises:
        ValueError: If any sampling parameter is outside its valid range.
    """

    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)

    scaled_logits: torch.Tensor = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        threshold: torch.Tensor = torch.topk(scaled_logits, top_k, dim=-1).values[..., -1, None]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, -torch.inf)

    probs: torch.Tensor = torch.softmax(scaled_logits, dim=-1)
    if top_p is not None and top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_original_order = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
        probs = probs.masked_fill(remove_original_order, 0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs


def speculative_sample(p: torch.Tensor, q: torch.Tensor, x: int) -> tuple[int, bool]:
    """
    Speculative sampling for a single position (Section 2.3 of the paper).

    Given a token `x` sampled from `q`, return a token distributed according to `p`.

    Args:
        p: The target distribution, shape (vocab_size,).
        q: The draft distribution, shape (vocab_size,).
        x: The token sampled from `q`.

    Returns:
        A tuple of the resulting token and whether the draft token `x` was accepted.
    """

    # Accept x with probability min(1, p(x) / q(x)).
    r: float = torch.rand(()).item()
    if r * q[x] <= p[x]:
        return x, True

    # Otherwise, resample from the adjusted distribution norm(max(0, p - q)).
    residual: torch.Tensor = torch.clamp(p - q, min=0)
    total: torch.Tensor = residual.sum()
    if total <= 0:
        # Only reachable through floating point error when p == q; p is then the right distribution.
        residual, total = p, p.sum()

    return int(torch.multinomial(residual / total, 1).item()), False
