"""Fused bilinear resize + argmax of segmentation scores (`seg_up.cu`; fp16 / fp32 scores, fp32 math).

`up_argmax_h` / `_f`: per-pixel class of the scores resized through one or two bilinear steps (PyTorch
`upsample_bilinear2d`, align_corners=False), without the full-resolution scores (tiles through shared memory;
`up_argmax_rh` / `_rf`: two steps, rows; `up_argmax_ph` / `_pf`: channels-last scores, one thread per pixel).
`up_h` / `_f`: the resized scores themselves (float32 NCHW), with the same per-pixel math, so the labels-only call
equals the argmax of the logits call.
`up_add_h` / `up_add8_h` (via `up_add`): `relu?(base + resize(z))` in fp16 NHWC, the HRNet fuse-layer and SegFormer
decode-head term.
Used by `fastforks.shared.seginfer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core import kernels
from fastforks._core.kernels import cubin_library

if TYPE_CHECKING:
    import torch

# p: pointer, i: int32, l: int64 (must match the extern "C" kernels in seg_up.cu)
_SIG = "pp" + "iiiiiiiii" + "llll"
SIGNATURES = {
    "up_argmax_h": _SIG,
    "up_argmax_f": _SIG,
    "up_argmax_rh": _SIG,
    "up_argmax_rf": _SIG,
    "up_argmax_ph": _SIG,
    "up_argmax_pf": _SIG,
    "up_h": _SIG,
    "up_f": _SIG,
    "up_add_h": "ppp" + "i" * 10,
    "up_add8_h": "ppp" + "i" * 10,
}

_V8 = 8  # channels per thread of up_add8_h
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def up_add(z: torch.Tensor, base: torch.Tensor, out: torch.Tensor, mid: tuple[int, int], *, relu: bool) -> None:
    """`out = relu?(base + resize(z))` for fp16 dense NHWC tensors (z resized to `mid`, then to the base size).

    `base` / `out` are `[B, H, W, C]`, z `[B, h, w, C]`, C even; one step if `mid` is H x W. 8 channels per thread
    when C % 8 == 0 and the tensors are 16-byte aligned. The caller checks dtypes, layouts and that the first step
    does not shrink.
    """
    b, hh, ww, c = base.shape
    h, w = z.shape[1:3]
    steps = 1 if tuple(mid) == (hh, ww) else 2
    wide = c % _V8 == 0 and all(t.data_ptr() % 16 == 0 for t in (z, base, out))
    args = (z, base, out, b, c, h, w, mid[0], mid[1], hh, ww, steps, int(relu))
    n = b * hh * ww * (c // (_V8 if wide else 2))
    load(base.device).launch("up_add8_h" if wide else "up_add_h", *kernels.linear(n), *args)
