"""OpenCV-compatible fixed-point pose crops without tap-expanded images."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"topdown": "ppppiiiillll", "letterbox": "pppiiillll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def _image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[:2] != (1, 3) or not image.is_cuda:  # noqa: PLR2004 - NCHW
        msg = "Expected a single RGB CUDA image"
        raise ValueError(msg)
    return image.float()


def crop(image: torch.Tensor, centre: torch.Tensor, size: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return [P, 3, H, W] integer levels using the same affine rounding as shared.topdown.crop."""
    image = _image(image)
    if centre.ndim != 2 or centre.shape[1] != 2 or size.shape != centre.shape:  # noqa: PLR2004 - xy
        msg = "Expected one (x, y) centre and (w, h) size per crop"
        raise ValueError(msg)
    if centre.device != image.device or size.device != image.device:
        msg = "Crop coordinates and the source image must share a device"
        raise ValueError(msg)
    out = image.new_empty((len(centre), 3, height, width))
    load(image.device).launch(
        "topdown",
        *linear(out.numel()),
        image,
        centre.float().contiguous(),
        size.float().contiguous(),
        out,
        *image.shape[-2:],
        height,
        width,
        *image.stride()[1:],
        out.numel(),
    )
    return out


def letterbox(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Sample at the RTMO column and row coordinates with grey border blending."""
    image = _image(image)
    if xy.device != image.device or xy.dtype != torch.float32 or xy.ndim != 2 or xy.shape[0] != 2:  # noqa: PLR2004 - (x, y) rows
        msg = "Expected two FP32 coordinate rows on the image device"
        raise ValueError(msg)
    size = xy.shape[1]
    out = image.new_empty((1, 3, size, size))
    load(image.device).launch(
        "letterbox",
        *linear(out.numel()),
        image,
        xy.contiguous(),
        out,
        *image.shape[-2:],
        size,
        *image.stride()[1:],
        out.numel(),
    )
    return out
