"""Tensor conversions between ComfyUI and fastforks (torch only, no ComfyUI import).

ComfyUI passes images as IMAGE `[B, H, W, C]` float32 in `[0, 1]` and masks as MASK `[B, H, W]` float32 in `[0, 1]`;
fastforks takes channel-first images `[3, H, W]` in `[0, 1]` and returns maps on the model's device.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

_RGB = 3
_COLOURS = [  # OpenPose-like limb colours
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]


def frame(image: torch.Tensor, i: int, device: torch.device) -> torch.Tensor:
    """Image `i` of a ComfyUI IMAGE batch as a fastforks input `[3, H, W]` on `device` (alpha dropped, gray to RGB)."""
    x = image[i].to(device=device, dtype=torch.float32)
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, _RGB)
    return x[..., :_RGB].permute(2, 0, 1)


def to_image(maps: Sequence[torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
    """RGB maps of fastforks `[3, H, W]` in `[0, 1]` (one per image) -> IMAGE `[B, H, W, 3]` float32."""
    return torch.stack([m.detach().float().permute(1, 2, 0) for m in maps]).clamp(0, 1).to(device).contiguous()


def to_mask(maps: Sequence[torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
    """Masks or alpha maps `[H, W]` (bool or float in `[0, 1]`, one per image) -> MASK `[B, H, W]` float32."""
    return torch.stack([m.detach().float() for m in maps]).clamp(0, 1).to(device).contiguous()


def with_alpha(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """IMAGE `[B, H, W, C]` and MASK `[B, H, W]` -> RGBA IMAGE `[B, H, W, 4]` (alpha = mask)."""
    rgb = image[..., :_RGB].expand(*image.shape[:-1], _RGB) if image.shape[-1] == 1 else image[..., :_RGB]
    return torch.cat([rgb, mask.to(rgb).unsqueeze(-1)], dim=-1)


def from_mask(mask: torch.Tensor, i: int, device: torch.device) -> torch.Tensor:
    """Mask `i` of a MASK batch (or the only one) as `[H, W]` float32 on `device`."""
    m = mask if mask.ndim == 2 else mask[min(i, mask.shape[0] - 1)]  # noqa: PLR2004 - one unbatched mask
    return m.to(device=device, dtype=torch.float32)


def depth_image(depth: torch.Tensor | None, inverse_depth: torch.Tensor | None) -> torch.Tensor:
    """A depth map as a gray RGB map `[3, H, W]`: inverse depth scaled to `[0, 1]` (near = white).

    Uses `inverse_depth` when the model returns it (relative models), else `1 / depth`; non-finite and non-positive
    depths are black. Each image is scaled by its own minimum and maximum, as ControlNet depth preprocessors do.
    """
    if inverse_depth is not None:
        inv = inverse_depth.float()
    elif depth is not None:
        d = depth.float()
        inv = torch.where(torch.isfinite(d) & (d > 0), 1 / d, torch.zeros_like(d))
    else:
        msg = "the model returned neither depth nor inverse_depth"
        raise ValueError(msg)
    valid = torch.isfinite(inv)
    if not valid.any():
        return torch.zeros(_RGB, *inv.shape, device=inv.device)
    lo, hi = inv[valid].min(), inv[valid].max()
    scaled = torch.where(valid, (inv - lo) / (hi - lo).clamp_min(torch.finfo(torch.float32).tiny), 0)
    return scaled.unsqueeze(0).expand(_RGB, -1, -1)


def normals_image(normals: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit normals `[H, W, 3]` (x right, y down, z forward) -> normal-map colours `[3, H, W]`.

    The usual normal-map convention: red = right, green = up, blue = towards the camera, each `0.5 + 0.5 * n`;
    a surface facing the camera is light blue.
    """
    n = torch.nan_to_num(normals.float()) * normals.new_tensor([1.0, -1.0, -1.0])
    return (0.5 + 0.5 * n).clamp(0, 1).permute(2, 0, 1)


def union_mask(masks: torch.Tensor | None, boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """One `[H, W]` float mask covering every instance: its mask if the model returns masks, else its box."""
    h, w = size
    if masks is not None:
        return masks.any(0).float() if len(masks) else torch.zeros(h, w, device=masks.device)
    out = torch.zeros(h, w, device=boxes.device)
    for x0, y0, x1, y1 in boxes.round().long().tolist():
        out[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = 1
    return out


def points(text: str) -> list[list[float]]:
    """`"x,y; x,y"` (or a JSON list of `[x, y]`) -> `[[x, y], ...]` in pixels; empty text gives `[]`."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        pts = json.loads(text)
    else:
        pts = [[float(v) for v in p.split(",")] for p in text.replace("\n", ";").split(";") if p.strip()]
    if any(len(p) != 2 for p in pts):  # noqa: PLR2004 - (x, y)
        msg = f"points must be 'x,y; x,y; ...' in pixels, got {text!r}"
        raise ValueError(msg)
    return [[float(x), float(y)] for x, y in pts]


def box(text: str) -> list[float] | None:
    """`"x0,y0,x1,y1"` in pixels (or a JSON list) -> `[x0, y0, x1, y1]`; empty text gives `None`."""
    text = text.strip().strip("[]")
    if not text:
        return None
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:  # noqa: PLR2004 - x0, y0, x1, y1
        msg = f"box must be 'x0,y0,x1,y1' in pixels, got {text!r}"
        raise ValueError(msg)
    return values


def person_boxes(detections: str, i: int) -> list[list[float]] | None:
    """The person boxes of image `i` from a detection node's JSON output; `None` for empty text.

    The JSON is a list (one entry per image) of lists of dicts with `box` and `class_name`; boxes of every class are
    used if no detection is named `person`.
    """
    if not detections.strip():
        return None
    per_image: list[list[dict[str, Any]]] = json.loads(detections)
    dets = per_image[min(i, len(per_image) - 1)]
    people = [d for d in dets if d.get("class_name") == "person"] or dets
    return [list(map(float, d["box"])) for d in people]


def draw_pose(  # noqa: PLR0913 - the fields of one image's pose result
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    skeleton: Sequence[tuple[int, int]],
    size: tuple[int, int],
    *,
    threshold: float,
    width: int,
) -> torch.Tensor:
    """Skeletons on black, `[3, H, W]` in `[0, 1]`: limbs and keypoints scoring at least `threshold`.

    Args:
        keypoints: `[P, K, 2]` pixels.
        scores: `[P, K]`.
        skeleton: Limbs as keypoint index pairs.
        size: `(H, W)`.
        threshold: Minimum keypoint score to draw.
        width: Line width in pixels.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415 - only this node draws

    h, w = size
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)
    for kp, sc in zip(keypoints.tolist(), scores.tolist(), strict=True):
        for j, (a, b) in enumerate(skeleton):
            if sc[a] >= threshold and sc[b] >= threshold:
                draw.line([tuple(kp[a]), tuple(kp[b])], fill=_COLOURS[j % len(_COLOURS)], width=width)
        for (x, y), s in zip(kp, sc, strict=True):
            if s >= threshold:
                draw.ellipse([x - width, y - width, x + width, y + width], fill=(255, 255, 255))
    return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255


def to_json(obj: object) -> str:
    """Tensors and tuples as plain JSON lists (for STRING outputs)."""

    def plain(o: object) -> object:
        if isinstance(o, torch.Tensor):
            return o.tolist()
        msg = f"not JSON serialisable: {type(o).__name__}"
        raise TypeError(msg)

    return json.dumps(obj, default=plain)
