"""Zero-padded WAFT-Stereo feature sampling into a strided half channel slice."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"stereo_warp4": "ppp" + "i" * 4 + "l" * 9 + "ff"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_MIN_EXTENT = 2  # the normalized input coordinates divide by the extent minus one


def sample_into(source: torch.Tensor, disparity: torch.Tensor, out: torch.Tensor) -> None:
    """Write the disparity warp directly into a channels-last destination."""
    n, c, h, w = source.shape
    if (
        min(n, c) < 1
        or min(h, w) < _MIN_EXTENT
        or c % 4
        or c * w >= 2**32
        or not source.is_cuda
        or disparity.device != source.device
        or out.device != source.device
        or disparity.shape != (n, 1, h, w)
        or out.shape != source.shape
        or disparity.dtype != torch.float32
        or any(
            t.dtype != torch.float16 or t.data_ptr() % 8 or t.stride(1) != 1 or any(t.stride(i) % 4 for i in (0, 2, 3))
            for t in (source, out)
        )
    ):
        msg = "Stereo warp needs aligned channels-last half features and one FP32 disparity map"
        raise ValueError(msg)
    grid = ((w * (c // 4) + 255) // 256, min(h, 65535), min(n, 65535))
    nb, _, ny, nx = source.stride()
    db, _, dy, dx = disparity.stride()
    ob, _, oy, ox = out.stride()
    load(source.device).launch(
        "stereo_warp4",
        grid,
        (256, 1, 1),
        source,
        disparity,
        out,
        n,
        c,
        h,
        w,
        nb,
        ny,
        nx,
        db,
        dy,
        dx,
        ob,
        oy,
        ox,
        1.0 / (h - 1),
        1.0 / (w - 1),
    )
