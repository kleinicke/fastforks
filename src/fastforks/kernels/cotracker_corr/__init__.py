"""CoTracker3 border-clamped feature windows and correlation resampling."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"gather": "pppppiiiil", "support": "ppppiiiil", "interpolate": "pppill"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_WINDOW = 8
_SUPPORT = 7


def window(feat: torch.Tensor, frames: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather 8-by-8 half-precision windows in x-major order with one fraction per coordinate."""
    time, height, width, channels = feat.shape
    if (
        frames.shape != coords.shape[:-1]
        or coords.shape[-1] != 2  # noqa: PLR2004 - XY
        or coords.dtype != torch.float32
        or frames.dtype != torch.int64
        or feat.dtype != torch.float16
        or frames.device != feat.device
        or coords.device != feat.device
    ):
        msg = "Expected matching int64 frames and float32 XY coordinates for the fp16 feature grid"
        raise ValueError(msg)
    out = feat.new_empty((*frames.shape, _WINDOW, _WINDOW, channels))
    fraction = torch.empty_like(coords, memory_format=torch.contiguous_format)
    if out.numel():
        load(feat.device).launch(
            "gather",
            *linear(out.numel()),
            feat.contiguous(),
            frames.contiguous(),
            coords.contiguous(),
            out,
            fraction,
            time,
            height,
            width,
            channels,
            out.numel(),
        )
    return out, fraction


def interpolate(g: torch.Tensor, fraction: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Interpolate 7-by-7 samples, optionally writing into rows of a padded correlation matrix."""
    channels = g.shape[-1]
    shape = (*g.shape[:-3], _SUPPORT, _SUPPORT, channels)
    if (
        g.shape[-3:-1] != (_WINDOW, _WINDOW)
        or g.dtype != torch.float16
        or fraction.shape != (*g.shape[:-3], 2)
        or fraction.dtype != torch.float32
        or fraction.device != g.device
    ):
        msg = "Expected fp16 8-by-8 windows and one float32 fraction pair per window"
        raise ValueError(msg)
    if out is None:
        out = g.new_empty(shape)
    if out.shape != shape or out.dtype != g.dtype or out.device != g.device:
        msg = "Mismatched CoTracker output storage"
        raise ValueError(msg)
    if out.numel():
        rows = out.view(-1, _SUPPORT * _SUPPORT * channels)
        if rows.stride(1) != 1:
            msg = "CoTracker output rows must be contiguous within each window"
            raise ValueError(msg)
        load(g.device).launch(
            "interpolate",
            *linear(out.numel()),
            g.contiguous(),
            fraction.contiguous(),
            out,
            channels,
            rows.stride(0),
            out.numel(),
        )
    return out


def support(feat: torch.Tensor, frames: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Read the 7-by-7 support directly into the channel-major matrix used by correlation GEMMs."""
    time, h, w, c = feat.shape
    if (
        frames.ndim != 1
        or coords.shape != (frames.numel(), 2)
        or feat.dtype != torch.float16
        or frames.dtype != torch.int64
        or coords.dtype != torch.float32
        or any(t.device != feat.device for t in (frames, coords))
    ):
        msg = "Expected one frame index and float32 coordinate pair per query on the feature device"
        raise ValueError(msg)
    out = feat.new_empty((frames.numel(), c, _SUPPORT * _SUPPORT))
    if out.numel():
        load(feat.device).launch(
            "support",
            *linear(out.numel()),
            feat.contiguous(),
            frames.contiguous(),
            coords.contiguous(),
            out,
            time,
            h,
            w,
            c,
            out.numel(),
        )
    return out
