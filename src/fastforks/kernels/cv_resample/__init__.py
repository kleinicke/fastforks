"""CUDA resampling with the same coefficient tables as portable edge preprocessing."""

import functools
from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear
from fastforks.shared.cv2_resize import Kind, _area, _cubic

_SIGNATURE = "ppppppiiiiiiil"
SIGNATURES = {"cv_cubic": _SIGNATURE, "cv_area": _SIGNATURE, "u8_levels": "ppl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


@functools.lru_cache(maxsize=64)
def _tables(n: int, out: int, kind: Kind, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx, weights = (_cubic if kind == "cubic" else _area)(n, out)
    return idx.to(device), weights.to(device)


def resize(x: torch.Tensor, h: int, w: int, kind: Kind, *, levels: bool = True) -> torch.Tensor:
    """Apply horizontal and vertical taps without materializing tap-expanded images."""
    b, c, height, width = x.shape
    ix, wx = _tables(width, w, kind, x.device)
    iy, wy = _tables(height, h, kind, x.device)
    out = x.new_empty((b, c, h, w), dtype=torch.float32)
    rounding = int(levels) * (1 + int(kind == "area" and width % w == 0 and height % h == 0))
    load(x.device).launch(
        "cv_" + kind,
        *linear(out.numel()),
        x.float().contiguous(),
        ix,
        wx,
        iy,
        wy,
        out,
        height,
        width,
        h,
        w,
        ix.shape[1],
        iy.shape[1],
        rounding,
        out.numel(),
    )
    return out


def image_levels(image: torch.Tensor) -> torch.Tensor:
    """Round normalized image samples to the input detector's eight-bit levels."""
    x = image.float().contiguous()
    out = torch.empty_like(x)
    load(x.device).launch("u8_levels", *linear(x.numel()), x, out, x.numel())
    return out
