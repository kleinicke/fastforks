"""Dual-softmax matching in a single dense correlation matrix."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {
    "denominators": "pppppii",
    "probabilities": "pppppil",
    "count_matches": "ppppfii",
    "write_matches": "ppppppfii",
}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def normalize_(logits: torch.Tensor) -> torch.Tensor:
    """Replace each logit with the product of its row and column softmax probabilities."""
    if logits.ndim != 2 or logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous():  # noqa: PLR2004 - matrix
        msg = "Dual-softmax requires a dense contiguous matrix of CUDA float32 logits"
        raise ValueError(msg)
    if not logits.numel():
        return logits
    rows, cols = logits.shape
    gx, gy = (cols + 31) // 32, (rows + 127) // 128
    row_max, col_max = logits.amax(1), logits.amax(0)
    row_parts, col_parts = logits.new_empty((rows, gx)), logits.new_empty((gy, cols))
    lib = load(logits.device)
    lib.launch("denominators", (gx, gy, 1), (32, 8, 1), logits, row_max, col_max, row_parts, col_parts, rows, cols)
    lib.launch(
        "probabilities",
        *linear(logits.numel()),
        logits,
        row_max,
        col_max,
        row_parts.sum(1),
        col_parts.sum(0),
        cols,
        logits.numel(),
    )
    return logits


def matches(
    desc_a: torch.Tensor, desc_b: torch.Tensor, inv_temp: float, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every mutual maximum over the threshold, including ties, in row-major order."""
    a = desc_a / desc_a.norm(dim=-1, keepdim=True)
    b = desc_b / desc_b.norm(dim=-1, keepdim=True)
    logits = (a @ b.mT).mul_(inv_temp)
    del a, b
    p = normalize_(logits)
    rows, cols = p.shape
    if not p.numel():
        return p.new_empty((0, 2), dtype=torch.long), p.new_empty(0)
    row_max, col_max = p.amax(1), p.amax(0)
    counts = p.new_empty(rows, dtype=torch.long)
    lib = load(p.device)
    grid, block = ((rows + 7) // 8, 1, 1), (32, 8, 1)
    lib.launch("count_matches", grid, block, p, row_max, col_max, counts, threshold, rows, cols)
    ends = counts.cumsum(0)
    count = int(ends[-1].item())
    pairs, scores = p.new_empty((count, 2), dtype=torch.long), p.new_empty(count)
    if not count:
        return pairs, scores
    lib.launch("write_matches", grid, block, p, row_max, col_max, ends, pairs, scores, threshold, rows, cols)
    return pairs, scores
