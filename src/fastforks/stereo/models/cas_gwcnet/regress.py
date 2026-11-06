"""Row-chunked "trilinear upsample of the cost volume -> softmax over disparity -> regression" (upstream eval tail).

Same values as upstream's `F.interpolate(cost, [D, H, W], mode="trilinear")` + softmax + `sum(p * disparities)`, but
each block of output rows upsamples only the low-resolution rows it depends on (plus one row of margin, so the clamping
at block borders matches the full computation). The full-resolution volumes never exist. Plain PyTorch; from our
cascade-stereo fork; adapted from `gwcnet/regress.py` (model families do not import each other): the disparity of each
hypothesis is an input (per pixel in the cascade's later stages).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def upsample_softmax_regress(
    cost: torch.Tensor, values: torch.Tensor, height: int, width: int, rows: int = 64
) -> torch.Tensor:
    """Disparity `[B, height, width]` float32 from a cost volume `[B, 1, d, h, w]` (any float dtype).

    `values` are the `D` hypotheses' disparities, float32 `[B, D, height, width]` or `[1, D, 1, 1]` (the same at every
    pixel); `D / d`, `height / h` and `width / w` must be integers (the upsampling factors).

    Raises:
        ValueError: The sizes are not integer multiples of the volume's.
    """
    b, _, d, h, w = cost.shape
    depth = values.shape[1]
    sd, sh, sw = depth // d, height // h, width // w
    if (sd * d, sh * h, sw * w) != (depth, height, width):
        msg = f"output {depth}x{height}x{width} is not an integer multiple of the volume {d}x{h}x{w}"
        raise ValueError(msg)
    if torch.compiler.is_exporting():  # one block: a traced loop would keep the traced height's blocks
        up = F.interpolate(cost.float(), size=(depth, height, width), mode="trilinear", align_corners=False)
        return torch.sum(F.softmax(up.squeeze(1), dim=1) * values, 1)
    per_pixel = values.shape[2] != 1
    out = torch.empty((b, height, width), device=cost.device, dtype=torch.float32)
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        i0 = max(0, math.floor((y0 + 0.5) / sh - 0.5) - 1)
        i1 = min(h, math.floor((y1 - 0.5) / sh - 0.5) + 2)
        up = F.interpolate(
            cost[:, :, :, i0:i1].float(),
            scale_factor=(sd, sh, sw),
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ).squeeze(1)  # output rows i0 * sh .. i1 * sh
        a = y0 - i0 * sh
        p = F.softmax(up[:, :, a : a + (y1 - y0)], dim=1)
        out[:, y0:y1] = torch.sum(p * (values[:, :, y0:y1] if per_pixel else values), 1)
    return out
