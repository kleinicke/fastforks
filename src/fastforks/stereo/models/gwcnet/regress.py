"""Row-chunked "trilinear upsample of the cost volume -> softmax over disparity -> soft-argmin" (upstream test tail).

Same values as upstream's `F.interpolate(cost, [D, H, W], mode="trilinear")` + softmax + regression, but each block
of output rows upsamples only the low-resolution rows it depends on (plus one row of margin, so the clamping at block
borders matches the full computation). The full-resolution volumes never exist (Middlebury-F: 192 x 2964 x 1988 fp32
is 4.5 GB each for the upsampled cost and its softmax). Plain PyTorch; from our stereo-zoo fork (`zoo_fastgen`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def upsample_softmax_regress(
    cost: torch.Tensor, max_disp: int, height: int, width: int, rows: int = 64
) -> torch.Tensor:
    """Disparity `[B, H, W]` float32 from a cost volume `[B, 1, d, h, w]` (any float dtype).

    `max_disp / d`, `height / h` and `width / w` must be integers (the upsampling factors).

    Raises:
        ValueError: The sizes are not integer multiples of the volume's.
    """
    b, _, d, h, w = cost.shape
    sd, sh, sw = max_disp // d, height // h, width // w
    if (sd * d, sh * h, sw * w) != (max_disp, height, width):
        msg = f"output {max_disp}x{height}x{width} is not an integer multiple of the volume {d}x{h}x{w}"
        raise ValueError(msg)
    values = torch.arange(max_disp, device=cost.device, dtype=torch.float32).view(1, max_disp, 1, 1)
    if torch.compiler.is_exporting():  # one block: a traced loop would keep the traced height's blocks
        up = F.interpolate(cost.float(), size=(max_disp, height, width), mode="trilinear", align_corners=False)
        return torch.sum(F.softmax(up.squeeze(1), dim=1) * values, 1)
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
        out[:, y0:y1] = torch.sum(p * values, 1)
    return out
