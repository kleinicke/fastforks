"""mmpose's top-down person crop, shared by the top-down pose families (RTMPose, HRNet).

Per person box (`x0, y0, x1, y1`), `GetBBoxCenterScale` + `TopdownAffine`: the box centre, its size x 1.25 with the
shorter side widened to the network's aspect; output pixel `(u, v)` samples
`centre + ((u, v) - (w, h) / 2) * size / (w, h)` bilinearly, the way OpenCV `warpAffine` does: source coordinates in
1/32 px, pixels outside the image count as 0 and blend in at the border, the crop rounded to uint8.
Not bit-exact with OpenCV 4.10 (900 random boxes, measured): rows and columns whose coordinate sits at a 1/32
rounding tie land one step off (OpenCV rounds a row's offset and step together, `fixed` one by one), 0.4 % of the
pixels of a noise image (up to 11 levels) and 0.02 % of a smooth one (up to 5); `crop` itself is the exact bilinear
blend at its coordinates. The networks normalise the crops with the mmpose means / stds in 0..255 (`MEAN`, `STD`).
"""

from __future__ import annotations

import torch

PADDING = 1.25  # fixed: reference crop padding (GetBBoxCenterScale)
MEAN = (123.675, 116.28, 103.53)  # fixed: reference data_preprocessor mean (RGB, 0..255)
STD = (58.395, 57.12, 57.375)  # fixed: reference data_preprocessor std (RGB, 0..255)
WARP_BITS = 5  # fixed: OpenCV INTER_BITS (bilinear source coordinates in 1/32 px)
AB_BITS = 10  # fixed: OpenCV AB_BITS (warpAffine fixed point before the 1/32 rounding)
# Left / right partner of each keypoint by label set (`flip_indices` of the reference dataset metas)
# fmt: off
FLIP = {  # fixed: the label sets' keypoint order
    "coco-keypoints-17": (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
    "coco-wholebody-133": (  # body, feet, face, left hand, right hand
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19, 39, 38, 37, 36, 35, 34,
        33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 50, 51, 52, 53, 58, 57,
        56, 55, 54, 68, 67, 66, 65, 70, 69, 62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72, 71, 82, 81, 80, 79, 78,
        87, 86, 85, 84, 83, 90, 89, 88,
        *range(112, 133), *range(91, 112),
    ),
    "halpe-26": (
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24,
    ),
    "coco-wholebody-hand-21": tuple(range(21)),  # one hand: the mirrored crop keeps every keypoint
    "lapa-106": (
        *range(32, -1, -1), 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54,
        *range(65, 54, -1), 79, 78, 77, 76, 75, 82, 81, 80, 83, 70, 69, 68, 67, 66, 73, 72, 71, 74,
        *range(90, 83, -1), *range(95, 90, -1), *range(100, 95, -1), 103, 102, 101, 105, 104,
    ),
    "ap10k-17": (1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13),
}
# fmt: on


def crop_boxes(boxes: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(centres `[P, 2]`, crop sizes `[P, 2]` w, h in pixels) of `x0, y0, x1, y1` boxes for a `height x width` net."""
    wh = (boxes[:, 2:] - boxes[:, :2]) * PADDING
    centre = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    aspect = width / height
    w, h = wh.unbind(1)
    wide = w > h * aspect
    return centre, torch.stack([torch.where(wide, w, h * aspect), torch.where(wide, w / aspect, h)], 1)


def fixed(offset: torch.Tensor, step: torch.Tensor, n: int) -> torch.Tensor:
    """Source coordinates `offset + step * i` (i < n) as OpenCV `warpAffine` rounds a column: 1/1024, then 1/32."""
    i = torch.arange(n, device=offset.device, dtype=torch.float32)  # float32: exact integers here (< 2^24)
    one = float(1 << AB_BITS)
    q = (offset[:, None] * one).round() + (step[:, None] * i * one).round()
    shift = 1 << (AB_BITS - WARP_BITS)
    return torch.floor((q + shift // 2) / shift) / (1 << WARP_BITS)


def crop(image: torch.Tensor, centre: torch.Tensor, size: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crops `[P, 3, height, width]` in 0..255 (integers) of `image` `[1, 3, H, W]` in [0, 1] (module docstring)."""
    ih, iw = image.shape[-2:]
    s = size[:, 0] / width  # source pixels per crop pixel (the same along y: the aspect is the network's)
    x = fixed(centre[:, 0] - s * (width / 2), s, width)  # [P, w]
    y = fixed(centre[:, 1] - s * (height / 2), s, height)  # [P, h]
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0  # multiples of 1/32: the tap weights below are exact in float32, as OpenCV's integers
    pixels = image[0].flatten(1)  # [3, H * W]; levels of the gathered taps only (no full-size copy)
    out = torch.zeros(3, centre.shape[0], height, width, device=image.device)
    for dy, ty in ((0, 1 - fy), (1, fy)):
        yy = y0 + dy
        wy = (ty * ((yy >= 0) & (yy < ih)))[:, :, None]  # pixels outside the image count as 0
        row = yy.clamp(0, ih - 1).long()[:, :, None] * iw  # int64: a float32 index is inexact above 2^24 pixels
        for dx, tx in ((0, 1 - fx), (1, fx)):
            xx = x0 + dx
            wx = (tx * ((xx >= 0) & (xx < iw)))[:, None, :]
            idx = row + xx.clamp(0, iw - 1).long()[:, None, :]  # [P, h, w]
            out += (pixels[:, idx].float() * 255).round_() * (wy * wx)
    return torch.floor(out.transpose(0, 1) + 0.5)  # OpenCV rounds half up to uint8
