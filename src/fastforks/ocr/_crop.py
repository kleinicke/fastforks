"""Text-line crops as PaddleOCR / PaddleX cut them (`CropByPolys`, quad mode), on the device, bit-exact.

Upstream: `cv2.minAreaRect` of the int quad -> `cv2.boxPoints` -> `get_mini_boxes` order -> `get_rotate_crop_image`
(crop size from the corner distances, `cv2.getPerspectiveTransform`, `cv2.warpPerspective` with `INTER_CUBIC` and
`BORDER_REPLICATE`, `np.rot90` if the crop is at least 1.5 times as high as wide). This reproduces OpenCV's
arithmetic: the 8 x 8 LU solve and the 3 x 3 inverse in float64, the per-block coordinate sums of
`WarpPerspectiveInvoker` (float64, rounded to 1/32 pixel), the 16-bit bicubic table (`initInterTab2D`, A = -0.75)
and the fixed-point sum of `remapBicubic`. Without float64 on the device (MPS) the coordinates are computed on the
CPU.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from fastforks._core.upload import device_tensor
from fastforks.ocr._db import hull_order, min_area_rect, mini_boxes

__all__ = ["crop_quads", "crop_sources", "perspective", "rotate_tall", "sort_quads"]

TAB = 32  # fixed: OpenCV INTER_TAB_SIZE (1/32 pixel)
SCALE = 1 << 15  # fixed: OpenCV INTER_REMAP_COEF_SCALE
BLOCK = 32  # fixed: WarpPerspectiveInvoker BLOCK_SZ
LINE_GAP = 10  # fixed: SortQuadBoxes swaps neighbours less than 10 px apart in y
TALL = 1.5  # fixed: get_rotate_crop_image rotates crops with h / w >= 1.5


def sort_quads(quads: np.ndarray) -> np.ndarray:
    """PaddleX `SortQuadBoxes`: the order (indices) of `[N, 4, 2]` quads, top to bottom, then left to right per line.

    Sorted by the first corner's (y, x), then neighbours less than 10 px apart in y swap while the later one is
    further left (upstream's bubble pass).
    """
    n = len(quads)
    order = sorted(range(n), key=lambda i: (int(quads[i, 0, 1]), int(quads[i, 0, 0])))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            a, b = quads[order[j + 1], 0], quads[order[j], 0]
            if abs(int(a[1]) - int(b[1])) < LINE_GAP and int(a[0]) < int(b[0]):
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    return np.array(order, dtype=np.int64)


def crop_sources(quads: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Source corners `[N, 4, 2]` float32 and crop sizes `(h, w)` of int quads, as `get_minarea_rect_crop`."""
    hulls = [hull_order([(int(x), int(y)) for x, y in q]) for q in quads.astype(np.int32)]
    src, _ = mini_boxes(min_area_rect(hulls))
    sizes = []
    for p in src:
        w = int(max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[2] - p[3])))
        h = int(max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2])))
        sizes.append((h, w))
    return src, sizes


def _lu_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """OpenCV `LUImpl` (partial pivoting, float64) of `a x = b`."""
    m = len(a)
    for i in range(m):
        k = max(range(i, m), key=lambda j: (abs(a[j][i]), -j))  # first row with the largest |pivot|
        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
        d = -1 / a[i][i]
        for j in range(i + 1, m):
            alpha = a[j][i] * d
            for c in range(i + 1, m):
                a[j][c] += alpha * a[i][c]
            b[j] += alpha * b[i]
    for i in range(m - 1, -1, -1):
        s = b[i]
        for c in range(i + 1, m):
            s -= a[i][c] * b[c]
        b[i] = s / a[i][i]
    return b


def perspective(src: np.ndarray, h: int, w: int) -> list[float]:
    """Row-major 3 x 3 map from crop pixels to `src` pixels, as `warpPerspective` inverts `getPerspectiveTransform`."""
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    a = [[0.0] * 8 for _ in range(8)]
    b = [0.0] * 8
    for i in range(4):
        sx, sy, dx, dy = src[i, 0], src[i, 1], dst[i, 0], dst[i, 1]  # float32: OpenCV's Point2f products
        a[i][0] = a[i + 4][3] = float(sx)
        a[i][1] = a[i + 4][4] = float(sy)
        a[i][2] = a[i + 4][5] = 1.0
        a[i][6] = float(-(sx * dx))
        a[i][7] = float(-(sy * dx))
        a[i + 4][6] = float(-(sx * dy))
        a[i + 4][7] = float(-(sy * dy))
        b[i], b[i + 4] = float(dx), float(dy)
    m = [*_lu_solve(a, b), 1.0]
    s = [m[0:3], m[3:6], m[6:9]]
    det = (
        s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1])
        - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
        + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0])
    )
    d = 1.0 / det
    return [
        (s[1][1] * s[2][2] - s[1][2] * s[2][1]) * d,
        (s[0][2] * s[2][1] - s[0][1] * s[2][2]) * d,
        (s[0][1] * s[1][2] - s[0][2] * s[1][1]) * d,
        (s[1][2] * s[2][0] - s[1][0] * s[2][2]) * d,
        (s[0][0] * s[2][2] - s[0][2] * s[2][0]) * d,
        (s[0][2] * s[1][0] - s[0][0] * s[1][2]) * d,
        (s[1][0] * s[2][1] - s[1][1] * s[2][0]) * d,
        (s[0][1] * s[2][0] - s[0][0] * s[2][1]) * d,
        (s[0][0] * s[1][1] - s[0][1] * s[1][0]) * d,
    ]


@functools.cache
def _cubic_table() -> np.ndarray:
    """OpenCV's `BicubicTab_i`: `[32 * 32, 16]` int16 weights (sum 2^15) per 1/32-pixel fraction (y, x)."""
    f32 = np.float32
    a = f32(-0.75)
    tab = np.zeros((TAB, 4), f32)
    for i in range(TAB):
        x = f32(i) * f32(1.0 / TAB)
        x1, x2 = x + f32(1), f32(1) - x
        c0 = ((a * x1 - f32(5) * a) * x1 + f32(8) * a) * x1 - f32(4) * a
        c1 = (((a + f32(2)) * x - (a + f32(3))) * x) * x + f32(1)
        c2 = (((a + f32(2)) * x2 - (a + f32(3))) * x2) * x2 + f32(1)
        tab[i] = c0, c1, c2, f32(1) - c0 - c1 - c2
    out = np.zeros((TAB * TAB, 16), np.int16)
    for i in range(TAB):
        for j in range(TAB):
            # saturate_cast<short>(cvRound(.)): ties to even, 1.0 -> 32767 (the sum fix-up below adds the rest)
            v = np.rint((tab[i][:, None] * tab[j][None, :]) * f32(SCALE)).clip(-SCALE, SCALE - 1).astype(np.int32)
            diff = int(v.sum()) - SCALE
            if diff:
                mk, big = (2, 2), (2, 2)  # OpenCV searches taps 2..3 (ksize / 2 onwards)
                for k1 in (2, 3):
                    for k2 in (2, 3):
                        if v[k1, k2] < v[mk]:
                            mk = (k1, k2)
                        elif v[k1, k2] > v[big]:
                            big = (k1, k2)
                v[big if diff < 0 else mk] -= diff
            out[i * TAB + j] = v.reshape(-1)
    return out


def _coords(
    mats: list[list[float]], sizes: list[tuple[int, int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integer source pixel (x, y) and table index of every crop pixel (all crops concatenated, row-major)."""
    f64 = torch.float64
    m = device_tensor(mats, dtype=f64, device=device)
    parts = []
    for i, (h, w) in enumerate(sizes):
        bh = min(BLOCK // 2, h)
        bw = min(BLOCK * BLOCK // bh, w)
        col = torch.arange(w, device=device)
        start = (col // bw * bw).to(f64)[None]
        x1 = (col % bw).to(f64)[None]
        y = torch.arange(h, device=device, dtype=f64)[:, None]
        r = m[i]
        x0 = r[0] * start + r[1] * y + r[2]  # per block row start, as WarpPerspectiveInvoker
        y0 = r[3] * start + r[4] * y + r[5]
        w0 = r[6] * start + r[7] * y + r[8]
        ww = w0 + r[6] * x1
        ww = torch.where(ww != 0, TAB / torch.where(ww != 0, ww, 1), 0)
        lim = float(2**31 - 1)
        xs = ((x0 + r[0] * x1) * ww).clamp(-lim - 1, lim).round()  # saturate_cast<int>: ties to even
        ys = ((y0 + r[3] * x1) * ww).clamp(-lim - 1, lim).round()
        parts.append(torch.stack((xs.reshape(-1), ys.reshape(-1))))
    xy = torch.cat(parts, 1).to(torch.int64)
    alpha = (xy[1] & (TAB - 1)) * TAB + (xy[0] & (TAB - 1))
    pix = (xy >> 5).clamp(-(2**15), 2**15 - 1)  # saturate_cast<short>
    return pix[0], pix[1], alpha


def crop_quads(image: torch.Tensor, src: np.ndarray, sizes: list[tuple[int, int]]) -> list[torch.Tensor]:
    """`get_rotate_crop_image` without the rotation: `[3, h, w]` uint8 crops of a `[3, H, W]` uint8 image.

    Args:
        image: Source image, uint8 (any channel order).
        src: Source corners `[N, 4, 2]` float32 (top-left, top-right, bottom-right, bottom-left), `crop_sources`.
        sizes: Crop sizes `(h, w)`, `crop_sources`.
    """
    if not sizes:
        return []
    dev = image.device if image.device.type != "mps" else torch.device("cpu")  # MPS has no float64
    mats = [perspective(p, *s) for p, s in zip(src, sizes, strict=True)]
    sx, sy, alpha = (t.to(image.device) for t in _coords(mats, sizes, dev))
    hgt, wid = image.shape[-2:]
    table = torch.from_numpy(_cubic_table()).to(image.device, torch.int32)[alpha]  # [P, 16]
    flat = image.reshape(image.shape[0], -1)
    acc = torch.zeros(image.shape[0], sx.shape[0], dtype=torch.int32, device=image.device)
    for k1 in range(4):
        row = (sy + (k1 - 1)).clamp_(0, hgt - 1) * wid  # BORDER_REPLICATE
        for k2 in range(4):
            idx = row + (sx + (k2 - 1)).clamp_(0, wid - 1)
            acc += flat[:, idx].to(torch.int32) * table[:, k1 * 4 + k2]
    out = ((acc + SCALE // 2) >> 15).clamp_(0, 255).to(torch.uint8)
    crops = out.split([h * w for h, w in sizes], 1)
    return [c.view(-1, h, w) for c, (h, w) in zip(crops, sizes, strict=True)]


def rotate_tall(crop: torch.Tensor) -> torch.Tensor:
    """`np.rot90` (counter-clockwise) of a `[C, h, w]` crop with `h / w >= 1.5`, as `get_rotate_crop_image`."""
    h, w = crop.shape[-2:]
    return torch.rot90(crop, 1, (-2, -1)) if h * 1.0 / w >= TALL else crop
