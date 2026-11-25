# ruff: noqa: PLR2004 - small geometric constants of the OpenCV / Clipper code reproduced here
"""DB text-detection boxes as PaddleOCR / PaddleX `DBPostProcess` (quad boxes, `score_mode="fast"`), in NumPy.

Upstream runs OpenCV and pyclipper per contour of the binary map; this reproduces their arithmetic (float32 where
OpenCV computes in float32) so the boxes come out the same:

- contours: one per 8-connected component of the map (`components`); the convex hull of a component's pixels equals
  that of its outer contour, ordered as `cv2.convexHull` orders a contour's hull. Upstream's `RETR_LIST` also returns
  hole contours; they are dropped here (a hole's box is scored on the hole and rarely passes `box_thresh`)
- `cv2.minAreaRect` (rotating calipers in float32) -> `cv2.boxPoints` -> upstream `get_mini_boxes`
- `box_score_fast`: mean of the map inside `cv2.fillPoly` of the truncated box
- `unclip`: pyclipper's round-join offset of the truncated box (Clipper 6.4.2 `ClipperOffset`); the final union of a
  convex polygon with itself only drops collinear points and changes the start, which `cv2.convexHull` undoes
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Rows", "boxes_from_rows", "hull_order", "min_area_rect", "mini_boxes", "offset_round"]

F32 = np.float32
Rows = tuple[np.ndarray, np.ndarray, np.ndarray]  # one component: rows y (ascending), leftmost x, rightmost x
ARC_TOLERANCE = 0.25  # fixed: pyclipper's default (Clipper def_arc_tolerance)


def _cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Strict convex hull (no collinear points) of integer points, positive area (`cv2.convexHull` orientation)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts
    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _area2(poly: Sequence[tuple[int, int]]) -> int:
    return sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, [*poly[1:], poly[0]], strict=True))


def hull_order(path: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """`cv2.convexHull(path)` of a simple polygon.

    Hull vertices in path order, rotated to start at the first one (path with positive area), or reversed from the
    last one (negative area).
    """
    hull = _hull(path)
    if len(hull) < 3:
        return hull
    first = {}
    for i, p in enumerate(path):
        first.setdefault(p, i)
    ordered = sorted(hull, key=first.__getitem__)
    return ordered if _area2(ordered) > 0 else ordered[::-1]


def _union_hull(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """`cv2.convexHull` of pyclipper's offset result.

    Clipper's union returns the (convex) polygon with positive area, ending at its top-most, then right-most vertex.
    """
    hull = _hull(points)
    if len(hull) < 3:
        return hull
    k = hull.index(min(hull, key=lambda p: (p[1], -p[0]))) + 1
    return hull[k:] + hull[:k]


def _component_hull(rows: Rows) -> list[tuple[int, int]]:
    """Hull of a component as `cv2.convexHull` returns it for its outer contour.

    The contour is traced from the first pixel in raster order (negative area); the hull has positive area and ends
    at that first pixel.
    """
    y, x0, x1 = (v.tolist() for v in rows)
    hull = _hull([*zip(x0, y, strict=True), *zip(x1, y, strict=True)])
    if len(hull) < 3:
        return hull[::-1]  # a segment comes out from its last point
    k = hull.index((x0[0], y[0])) + 1
    return hull[k:] + hull[:k]


def min_area_rect(  # noqa: PLR0915 - OpenCV rotatingCalipers, step by step
    hulls: Sequence[Sequence[tuple[int, int]]],
) -> np.ndarray:
    """`cv2.minAreaRect` of hulls already in `cv2.convexHull` order: `[N, 5]` float32 (cx, cy, w, h, angle in deg).

    Vectorised over the hulls; every step is OpenCV's `rotatingCalipers` operation in float32 (or double where OpenCV
    computes in double). Hulls with fewer than 3 points give OpenCV's degenerate rectangles (size 0 or a segment).
    """
    out = np.zeros((len(hulls), 5), F32)
    idx = [i for i, h in enumerate(hulls) if len(h) > 2]
    for i, h in enumerate(hulls):
        if len(h) == 2:
            (ax, ay), (bx, by) = h
            dx, dy = float(F32(bx) - F32(ax)), float(F32(by) - F32(ay))
            angle = F32(math.atan2(dy, dx))
            out[i] = (F32((ax + bx) * F32(0.5)), F32((ay + by) * F32(0.5)), F32(math.sqrt(dx * dx + dy * dy)), 0, 0)
            out[i, 4] = F32(float(angle * F32(180)) / math.pi)
        elif len(h) == 1:
            out[i, :2] = h[0]
    if not idx:
        return out
    count = np.array([len(hulls[i]) for i in idx])
    m, n = len(idx), int(count.max())
    p = np.zeros((m, n, 2), F32)
    for r, i in enumerate(idx):
        p[r, : count[r]] = hulls[i]
    rows = np.arange(m)
    valid = np.arange(n)[None] < count[:, None]
    nxt = np.where(np.arange(n)[None] + 1 < count[:, None], np.arange(n)[None] + 1, 0)
    vect = np.where(valid[..., None], p[rows[:, None], nxt] - p, F32(0))  # float32 differences
    d64 = vect.astype(np.float64)
    with np.errstate(divide="ignore"):
        inv_len = (1.0 / np.sqrt(d64[..., 0] ** 2 + d64[..., 1] ** 2)).astype(F32)
    big = F32(np.inf)
    xs, ys = np.where(valid, p[..., 0], big), np.where(valid, p[..., 1], big)
    left, bottom = xs.argmin(1), ys.argmin(1)
    right = np.where(valid, p[..., 0], -big).argmax(1)
    top = np.where(valid, p[..., 1], -big).argmax(1)
    seq = np.stack([bottom, right, top, left], 1)
    minarea = np.full(m, np.finfo(F32).max, F32)
    best = np.zeros((m, 6), F32)  # base_a, base_b, width, height and the left / bottom point indices
    for k in range(n):
        live = k < count
        v = vect[rows[:, None], seq]  # [m, 4, 2]
        rot = np.stack(
            [v[:, 0], np.stack([v[:, 1, 1], -v[:, 1, 0]], 1), -v[:, 2], np.stack([-v[:, 3, 1], v[:, 3, 0]], 1)], 1
        )
        main = np.zeros(m, np.int64)
        for j in (1, 2, 3):
            cur = rot[rows, main]
            right_of = rot[:, j, 1] * cur[:, 0] + -rot[:, j, 0] * cur[:, 1] < 0  # firstVecIsRight(rot[j], rot[main])
            main = np.where(right_of, j, main)
        pindex = seq[rows, main]
        lead_x = vect[rows, pindex, 0] * inv_len[rows, pindex]
        lead_y = vect[rows, pindex, 1] * inv_len[rows, pindex]
        base_a = np.choose(main, [lead_x, lead_y, -lead_x, -lead_y])
        base_b = np.choose(main, [lead_y, -lead_x, -lead_y, lead_x])
        bumped = seq[rows, main] + 1
        seq = np.where(
            live[:, None] & (np.arange(4)[None] == main[:, None]), np.where(bumped == count, 0, bumped)[:, None], seq
        )
        q = p[rows[:, None], seq]
        width = (q[:, 1, 0] - q[:, 3, 0]) * base_a + (q[:, 1, 1] - q[:, 3, 1]) * base_b
        height = -(q[:, 2, 0] - q[:, 0, 0]) * base_b + (q[:, 2, 1] - q[:, 0, 1]) * base_a
        area = width * height
        take = live & (area <= minarea)
        minarea = np.where(take, area, minarea)
        new = np.stack([base_a, base_b, width, height, seq[:, 3].astype(F32), seq[:, 0].astype(F32)], 1)
        best = np.where(take[:, None], new, best)
    a1, b1, width, height = best[:, 0], best[:, 1], best[:, 2], best[:, 3]
    a2, b2 = -b1, a1
    pl = p[rows, best[:, 4].astype(np.int64)]
    pb = p[rows, best[:, 5].astype(np.int64)]
    c1 = a1 * pl[:, 0] + pl[:, 1] * b1
    c2 = a2 * pb[:, 0] + pb[:, 1] * b2
    idet = F32(1) / (a1 * b2 - a2 * b1)
    px = (c1 * b2 - c2 * b1) * idet
    py = (a1 * c2 - a2 * c1) * idet
    o1x, o1y, o2x, o2y = a1 * width, b1 * width, a2 * height, b2 * height
    o1 = (o1x.astype(np.float64), o1y.astype(np.float64))
    o2 = (o2x.astype(np.float64), o2y.astype(np.float64))
    angle = np.arctan2(o1[1], o1[0]).astype(F32)
    out[idx] = np.stack(
        [
            px + (o1x + o2x) * F32(0.5),
            py + (o1y + o2y) * F32(0.5),
            np.sqrt(o1[0] ** 2 + o1[1] ** 2).astype(F32),
            np.sqrt(o2[0] ** 2 + o2[1] ** 2).astype(F32),
            ((angle * F32(180)).astype(np.float64) / math.pi).astype(F32),
        ],
        1,
    )
    return out


def box_points(rect: np.ndarray) -> np.ndarray:
    """`cv2.boxPoints` of `[N, 5]` rectangles -> `[N, 4, 2]` float32."""
    cx, cy, w, h, angle = rect.T
    rad = angle.astype(np.float64) * math.pi / 180.0
    b = np.cos(rad).astype(F32) * F32(0.5)
    a = np.sin(rad).astype(F32) * F32(0.5)
    p0 = np.stack([cx - a * h - b * w, cy + b * h - a * w], 1)
    p1 = np.stack([cx + a * h - b * w, cy - b * h - a * w], 1)
    c2 = np.stack([F32(2) * cx, F32(2) * cy], 1)
    return np.stack([p0, p1, c2 - p0, c2 - p1], 1)


def mini_boxes(rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upstream `get_mini_boxes`: corners `[N, 4, 2]` (top-left, top-right, bottom-right, bottom-left), short sides."""
    pts = box_points(rect)
    order = np.argsort(pts[..., 0], axis=1, kind="stable")
    pts = np.take_along_axis(pts, order[..., None], 1)
    swap_l = ~(pts[:, 1, 1] > pts[:, 0, 1])
    swap_r = ~(pts[:, 3, 1] > pts[:, 2, 1])
    i1, i4 = np.where(swap_l, 1, 0), np.where(swap_l, 0, 1)
    i2, i3 = np.where(swap_r, 3, 2), np.where(swap_r, 2, 3)
    r = np.arange(len(pts))
    box = np.stack([pts[r, i1], pts[r, i2], pts[r, i3], pts[r, i4]], 1)
    return box, np.minimum(rect[:, 2], rect[:, 3])


def _clip_line(w: int, h: int, p: list[int], q: list[int]) -> bool:
    """OpenCV `clipLine` (Cohen-Sutherland, integer results) of `p`, `q` to a `w` x `h` image, in place."""
    right, bottom = w - 1, h - 1
    x1, y1 = p
    x2, y2 = q
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if c1 & 12:
            a = 0 if c1 < 8 else bottom
            x1 += int(float(a - y1) * (x2 - x1) / (y2 - y1))
            y1 = a
            c1 = (x1 < 0) + (x1 > right) * 2
        if c2 & 12:
            a = 0 if c2 < 8 else bottom
            x2 += int(float(a - y2) * (x2 - x1) / (y2 - y1))
            y2 = a
            c2 = (x2 < 0) + (x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                a = 0 if c1 == 1 else right
                y1 += int(float(a - x1) * (y2 - y1) / (x2 - x1))
                x1 = a
                c1 = 0
            if c2:
                a = 0 if c2 == 1 else right
                y2 += int(float(a - x2) * (y2 - y1) / (x2 - x1))
                x2 = a
                c2 = 0
    p[:], q[:] = (x1, y1), (x2, y2)
    return (c1 | c2) == 0


def _outside(w: int, h: int, p: list[int], q: list[int]) -> bool:
    return not (0 <= p[0] < w and 0 <= q[0] < w and 0 <= p[1] < h and 0 <= q[1] < h)


def _line(mask: np.ndarray, p: tuple[int, int], q: tuple[int, int]) -> None:
    """OpenCV's 8-connected `Line` (clipped to the mask, drawn left to right)."""
    h, w = mask.shape
    a, b = list(p), list(q)
    if _outside(w, h, a, b) and not _clip_line(w, h, a, b):
        return
    (x0, y0), (x1, y1) = a, b
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    dx, dy = x1 - x0, y1 - y0
    sy = 1 if dy >= 0 else -1
    dy = abs(dy)
    vert = dy > dx
    if vert:
        dx, dy = dy, dx
    # OpenCV steps `err = dx - 2 dy`, `err -= 2 dy` and `err += 2 dx` on a diagonal step (`err < 0`); the diagonal
    # steps before step i are then (2 dy i + dx - 1) // (2 dx) in closed form (one array write, not a loop per pixel)
    i = np.arange(dx + 1)
    k = (2 * dy * i + dx - 1) // (2 * dx) if dx else i
    if vert:
        mask[y0 + sy * i, x0 + k] = 1
    else:
        mask[y0 + sy * k, x0 + i] = 1


def _cdiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def fill_poly(mask: np.ndarray, poly: np.ndarray) -> None:
    """`cv2.fillPoly(mask, [poly], 1)` of one integer polygon.

    OpenCV's outline lines (`CollectPolyEdges`) and scanline spans (`FillEdgeCollection`), both clipped to the mask.
    """
    h, w = mask.shape
    shift, half = 16, 1 << 15
    pts = [(int(x), int(y)) for x, y in poly]
    edges = []  # (y0, y1, x at y0 in 16.16 fixed point, dx per row)
    for (x0, y0), (x1, y1) in zip([pts[-1], *pts[:-1]], pts, strict=True):
        _line(mask, (x0, y0), (x1, y1))
        c0, c1 = [x0 << shift, y0], [x1 << shift, y1]
        t0, t1 = [x0, y0], [x1, y1]
        if _outside(w, h, t0, t1):
            _clip_line(w, h, t0, t1)
            if t0[1] != t1[1]:
                c0, c1 = [t0[0] << shift, t0[1]], [t1[0] << shift, t1[1]]
        else:
            c0[0] += half
            c1[0] += half
        if y0 == y1:
            continue
        dx = _cdiv(c1[0] - c0[0], c1[1] - c0[1])
        if y0 < y1:
            edges.append((y0, y1, c0[0] + (y0 - c0[1]) * dx, dx))
        else:
            edges.append((y1, y0, c1[0] + (y1 - c1[1]) * dx, dx))
    if len(edges) < 2:
        return
    for y in range(max(min(e[0] for e in edges), 0), min(max(e[1] for e in edges), h)):
        xs = sorted(x + (y - y0) * dx for y0, y1, x, dx in edges if y0 <= y < y1)
        for left, right in zip(xs[::2], xs[1::2], strict=False):
            a, b = left >> shift, right >> shift
            if a < w and b >= 0:
                mask[y, max(a, 0) : min(b, w - 1) + 1] = 1


def box_score(pred: np.ndarray, box: np.ndarray) -> float:
    """Upstream `box_score_fast`: mean of `pred` inside the filled, truncated box."""
    h, w = pred.shape
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    fill_poly(mask, (box - np.array([xmin, ymin], F32)).astype(np.int32))
    sel = mask.astype(bool)
    n = int(sel.sum())
    return float(pred[ymin : ymax + 1, xmin : xmax + 1][sel].astype(np.float64).sum() / n) if n else 0.0


def _round(v: float) -> int:
    return int(v - 0.5) if v < 0 else int(v + 0.5)  # Clipper Round: half away from zero, then truncate


def offset_round(  # noqa: C901 - Clipper ClipperOffset.DoOffset (round joins), step by step
    path: Sequence[tuple[int, int]], delta: float
) -> list[tuple[int, int]]:
    """Clipper 6.4.2 `ClipperOffset` of one closed polygon, round joins, before its final union (`delta > 0`)."""
    src = [path[0]]
    for p in path[1:]:
        if p != src[-1]:
            src.append(p)
    while len(src) > 1 and src[-1] == src[0]:
        src.pop()
    if len(src) < 3:
        return []
    if _area2(src) < 0:  # FixOrientations: Clipper's Orientation() of the (only) path must hold
        src.reverse()
    y = min(ARC_TOLERANCE, abs(delta) * ARC_TOLERANCE)
    steps = math.pi / math.acos(1 - y / abs(delta))
    steps = min(steps, abs(delta) * math.pi)
    sin, cos = math.sin(2 * math.pi / steps), math.cos(2 * math.pi / steps)
    per_rad = steps / (2 * math.pi)
    n = len(src)
    normals = []
    for j in range(n):
        (ax, ay), (bx, by) = src[j], src[(j + 1) % n]
        dx, dy = float(bx - ax), float(by - ay)
        f = 1.0 / math.sqrt(dx * dx + dy * dy)
        normals.append((dy * f, -dx * f))
    out: list[tuple[int, int]] = []
    k = n - 1
    for j in range(n):
        (kx, ky), (jx, jy) = normals[k], normals[j]
        sx, sy = src[j]
        sin_a = kx * jy - jx * ky
        if abs(sin_a * delta) < 1.0 and kx * jx + jy * ky > 0:
            out.append((_round(sx + kx * delta), _round(sy + ky * delta)))
            continue  # upstream returns before `k = j`
        sin_a = max(-1.0, min(1.0, sin_a))
        if sin_a * delta < 0:
            out += [
                (_round(sx + kx * delta), _round(sy + ky * delta)),
                (sx, sy),
                (_round(sx + jx * delta), _round(sy + jy * delta)),
            ]
        else:
            a = math.atan2(sin_a, kx * jx + ky * jy)
            count = max(_round(per_rad * abs(a)), 1)
            x, yy = kx, ky
            for _ in range(count):
                out.append((_round(sx + x * delta), _round(sy + yy * delta)))
                x, yy = x * cos - sin * yy, x * sin + yy * cos
            out.append((_round(sx + jx * delta), _round(sy + jy * delta)))
        k = j
    return out


def _contour_area(box: np.ndarray) -> float:
    prev = box[-1]
    a = 0.0
    for p in box:
        a += float(prev[0]) * float(p[1]) - float(prev[1]) * float(p[0])
        prev = p
    return abs(a * 0.5)


def _arc_length(box: np.ndarray) -> float:
    prev = box[-1]
    total = 0.0
    for p in box:
        d = p - prev  # float32
        total += float(np.sqrt(d[0] * d[0] + d[1] * d[1]))
        prev = p
    return total


def boxes_from_rows(  # noqa: PLR0913 - DBPostProcess settings
    pred: np.ndarray,
    comps: Sequence[Rows],
    size: tuple[int, int],
    *,
    box_thresh: float,
    unclip_ratio: float,
    max_candidates: int,
    min_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream `boxes_from_bitmap`: int16 quads `[N, 4, 2]` in `size` (h, w) pixels and their float scores `[N]`."""
    comps = comps[:max_candidates]
    if not comps:
        return np.zeros((0, 4, 2), np.int16), np.zeros(0)
    box, sside = mini_boxes(min_area_rect([_component_hull(c) for c in comps]))
    keep, scores, grown = [], [], []
    for i in np.flatnonzero(sside >= min_size):
        score = box_score(pred, box[i])
        if box_thresh > score:
            continue
        b = box[i]
        distance = _contour_area(b) * unclip_ratio / _arc_length(b)
        path = [(int(x), int(y)) for x, y in b]  # pyclipper truncates float coordinates
        grown.append(_union_hull(offset_round(path, distance)))
        keep.append(i)
        scores.append(score)
    if not keep:
        return np.zeros((0, 4, 2), np.int16), np.zeros(0)
    box2, sside2 = mini_boxes(min_area_rect(grown))
    h, w = pred.shape
    dest_h, dest_w = size
    ok = sside2 >= min_size + 2
    scale = np.array([dest_w / w, dest_h / h])
    scaled = box2.astype(np.float64) * scale  # upstream's image size is a float64 array: float64 scaling
    final = np.clip(np.round(scaled), 0, [dest_w, dest_h]).astype(np.int16)
    return final[ok], np.array(scores)[ok]
