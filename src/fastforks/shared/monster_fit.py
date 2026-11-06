"""MonSter's mono scale / shift fit as tensor ops (fastforks; shared by MonSter, MonSter++ and RT-MonSter++).

`scale_shift(mono, gt)` is upstream's `compute_scale_shift`: least squares `[scale, shift]` on the pixels with
gt > 0, mono > 0.01 and mono above its 20 % quantile (the sorted value at index int(0.2 * N)), A = X^T X + 1e-6 I,
written with masked sums and a closed-form 2 x 2 solve (no boolean indexing, no `.item()`, no `linalg.solve`, which
the ONNX exporter has no function for). The quantile is a full sort in eager mode; an exported graph finds the same
value by bisection (`shared.quantile`; TensorRT's TopK takes at most 3840 elements).
"""

from __future__ import annotations

import torch

from fastforks.shared.quantile import kth_smallest

_MIN_MONO = 1e-2  # upstream's mask: mono > 0.01


def _sorted_at(flat: torch.Tensor, k: int) -> torch.Tensor:
    """`torch.sort(flat)[0][k]` (0-d); an exported graph bisects on the value instead."""
    return kth_smallest(flat, k) if torch.compiler.is_exporting() else torch.sort(flat)[0][k]


def scale_shift(mono: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """compute_scale_shift(mono, gt) as 0-d tensors (same maths, rounding-level differences)."""
    flat = mono.reshape(-1)
    thr = _sorted_at(flat, int(0.2 * flat.numel()))
    m = ((gt > 0) & (mono > _MIN_MONO) & (mono > thr)).to(mono.dtype)
    eps = 1e-6
    xm = mono * m
    n = m.sum()
    b = xm.sum()
    u = (xm * gt).sum()
    v = (m * gt).sum()
    # det = a d - b^2 and d u - b v with a = sum m x^2 + eps, d = n + eps, written with centred sums. Same maths; the
    # raw form cancels to 0 / inf in fp32 when mono is nearly constant (tiny inputs).
    xc = (mono - b / n.clamp(min=1)) * m
    gc = (gt - v / n.clamp(min=1)) * m
    det = n * (xc * xc).sum() + eps * ((xm * mono).sum() + n) + eps * eps
    scale = (n * (xc * gc).sum() + eps * u) / det
    return scale, (v - b * scale) / (n + eps)


def fit(mono: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """The per-image mono fit of upstream's loop: mono[b] = scale * mono[b] + shift (fp32)."""
    out = []
    for b in range(disp.shape[0]):
        scale, shift = scale_shift(mono[b], disp[b])  # upstream .squeeze(1) is a no-op on [1, H, W]; ONNX rejects it
        out.append(scale * mono[b] + shift)
    return torch.stack(out)
