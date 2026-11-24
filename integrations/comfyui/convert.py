"""Tensor conversions between ComfyUI and fastforks (torch only, no ComfyUI import).

ComfyUI passes images as IMAGE `[B, H, W, C]` float32 in `[0, 1]` and masks as MASK `[B, H, W]` float32 in `[0, 1]`;
fastforks takes channel-first images `[3, H, W]` in `[0, 1]` and returns maps on the model's device.
"""

from __future__ import annotations

import difflib
import functools
import json
import math
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


def upload(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """`x` on `device` as float32; CPU -> CUDA through pinned memory, asynchronously.

    A pageable upload waits for the copy, which empties the GPU's launch queue per image (costly under Windows'
    WDDM). Plain torch, not `fastforks._core.upload`: the pack also runs with an older fastforks.
    """
    if x.device.type == "cpu" and torch.device(device).type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    return x.to(device=device, dtype=torch.float32)


def frame(image: torch.Tensor, i: int, device: torch.device) -> torch.Tensor:
    """Image `i` of a ComfyUI IMAGE batch as a fastforks input `[3, H, W]` on `device` (alpha dropped, gray to RGB).

    On CUDA the next image is uploaded meanwhile (`_Ahead`), so a loop over a batch does not wait for each upload.
    """
    x = _AHEAD.take(image, i, i + 1, device)[0]
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, _RGB)
    return x[..., :_RGB].permute(2, 0, 1)


def frames(image: torch.Tensor, start: int, stop: int, device: torch.device) -> torch.Tensor:
    """Images `start:stop` of a ComfyUI IMAGE batch as one fastforks batch `[B, 3, H, W]` on `device` (as `frame`)."""
    x = _AHEAD.take(image, start, stop, device)
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, -1, _RGB)
    return x[..., :_RGB].permute(0, 3, 1, 2)


class _Ahead:
    """Uploads of CPU IMAGE batches to CUDA: the next slice (same length) is prepared while the caller runs this one.

    A worker thread copies the slice to pinned memory and starts its upload on a side stream; the caller's stream waits
    for that copy only when it takes the slice. A frame of 1080p float32 is 25 MB: pinning and uploading it took
    3.4 ms on an RTX 5060 Ti under Windows, in line with the model's work. One slice is held ahead per IMAGE (a stereo
    node reads two), for at most `_HELD` IMAGEs; `drop` frees them (the end of a node's loop).
    """

    _HELD = 4

    def __init__(self) -> None:
        self._pool: Any = None
        self._held: dict[int, tuple[torch.Tensor, int, int, torch.device, Any]] = {}  # by id(IMAGE), which it holds

    def take(self, image: torch.Tensor, start: int, stop: int, device: torch.device) -> torch.Tensor:
        """`image[start:stop]` as float32 on `device`; starts the upload of `image[stop:2 * stop - start]`."""
        device = torch.device(device)
        if image.device.type != "cpu" or device.type != "cuda":
            return upload(image[start:stop], device)
        held = self._held.pop(id(image), None)
        if held is not None and held[0] is image and held[1:4] == (start, stop, device):
            x, event = held[4].result()
            stream = torch.cuda.current_stream(device)
            stream.wait_event(event)
            x.record_stream(stream)  # made on the side stream, used (and freed) on the caller's
        else:
            x = upload(image[start:stop], device)
        nxt = min(stop + (stop - start), image.shape[0])
        if nxt > stop:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415 - only for CUDA uploads

                self._pool = ThreadPoolExecutor(1, thread_name_prefix="fastforks-upload")
            while len(self._held) >= self._HELD:
                del self._held[next(iter(self._held))]
            job = self._pool.submit(_side_upload, image[stop:nxt], device)
            self._held[id(image)] = (image, stop, nxt, device, job)
        return x

    def drop(self) -> None:
        """Forget the slices held ahead (each upload finishes on the side stream; then its memory is freed)."""
        self._held.clear()


def _side_upload(x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.cuda.Event]:
    """`x` (CPU) uploaded to `device` on a side stream through pinned memory, and the event that marks the copy done."""
    stream = _side_stream(device)
    with torch.cuda.stream(stream):
        y = x.pin_memory().to(device, torch.float32, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    return y, event


@functools.cache
def _side_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


_AHEAD = _Ahead()


def drop_ahead() -> None:
    """Free the upload `frame` / `frames` prepared for a next slice that will not be asked for."""
    _AHEAD.drop()


def to_image(maps: Sequence[torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
    """RGB maps of fastforks `[3, H, W]` in `[0, 1]` (one per image) -> IMAGE `[B, H, W, 3]` float32."""
    return torch.stack([m.detach().float().permute(1, 2, 0) for m in maps]).clamp_(0, 1).to(device).contiguous()


def to_mask(maps: Sequence[torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
    """Masks or alpha maps `[H, W]` (bool or float in `[0, 1]`, one per image) -> MASK `[B, H, W]` float32."""
    return torch.stack([m.detach().float() for m in maps]).clamp_(0, 1).to(device).contiguous()


def with_alpha(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """IMAGE `[B, H, W, C]` and MASK `[B, H, W]` -> RGBA IMAGE `[B, H, W, 4]` (alpha = mask)."""
    rgb = image[..., :_RGB].expand(*image.shape[:-1], _RGB) if image.shape[-1] == 1 else image[..., :_RGB]
    return torch.cat([rgb, mask.to(rgb).unsqueeze(-1)], dim=-1)


def from_mask(mask: torch.Tensor, i: int, device: torch.device) -> torch.Tensor:
    """Mask `i` of a MASK batch (or the only one) as `[H, W]` float32 on `device`."""
    m = mask if mask.ndim == 2 else mask[min(i, mask.shape[0] - 1)]  # noqa: PLR2004 - one unbatched mask
    return m.to(device=device, dtype=torch.float32)


def depth_image(depth: torch.Tensor | None, inverse_depth: torch.Tensor | None, *, clip: float = 0.0) -> torch.Tensor:
    """A depth map as a gray RGB map `[3, H, W]`: inverse depth scaled to `[0, 1]` (near = white).

    Uses `inverse_depth` when the model returns it (relative models), else `1 / depth`; non-finite and non-positive
    depths are black. Each image is scaled by its own minimum and maximum, as ControlNet depth preprocessors do; with
    `clip`, by the `clip` and `1 - clip` quantiles of its valid pixels instead, so that a few outliers do not squash
    the rest of the map to one gray (the pixels beyond them saturate to black / white).
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
    if clip:
        if not valid.any():
            return torch.zeros(_RGB, *inv.shape, device=inv.device)
        lo, hi = _range(inv[valid], clip)
    else:  # min / max of the valid pixels without a host sync (none valid: every pixel is black below)
        lo, hi = torch.where(valid, inv, math.inf).amin(), torch.where(valid, inv, -math.inf).amax()
    scaled = ((inv - lo) / (hi - lo).clamp_min(torch.finfo(torch.float32).tiny)).clamp(0, 1)
    return torch.where(valid, scaled, 0).unsqueeze(0).expand(_RGB, *inv.shape)


def _range(values: torch.Tensor, clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The `clip` and `1 - clip` quantiles of a non-empty 1-D tensor (its minimum and maximum for `clip` 0)."""
    k = int(clip * values.numel())
    if k == 0:
        return values.min(), values.max()
    ordered = values.sort().values  # sort runs on every device (torch.quantile is capped at 16M elements)
    return ordered[k], ordered[-1 - k]


def normals_image(normals: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit normals `[H, W, 3]` (x right, y down, z forward) -> normal-map colours `[3, H, W]`.

    The usual normal-map convention: red = right, green = up, blue = towards the camera, each `0.5 + 0.5 * n`;
    a surface facing the camera is light blue.
    """
    n = torch.nan_to_num(normals.float())
    n = torch.cat([n[..., :1], -n[..., 1:]], dim=-1)  # y up, z towards the camera (no host-to-device sign tensor)
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


def _floats(values: Sequence[Any], form: str) -> list[float]:
    """`values` as finite floats; else a `ValueError` with the message `form` (it names the node input)."""
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        out = []
    if not out or not all(math.isfinite(v) for v in out):
        raise ValueError(form)
    return out


def points(text: str, name: str = "points") -> list[list[float]]:
    """`"x,y; x,y"` (or a JSON list of `[x, y]`) -> `[[x, y], ...]` in pixels; empty text gives `[]`.

    A malformed `text` raises a `ValueError` that names the node input `name` and shows the expected form.
    """
    text = text.strip()
    if not text:
        return []
    form = f"{name}: write points as 'x,y; x,y' in pixels (e.g. '120,80; 300,210'), got {text!r}"
    try:
        rows = json.loads(text) if text.startswith("[") else [p.split(",") for p in text.replace("\n", ";").split(";")]
    except json.JSONDecodeError as e:
        raise ValueError(form) from e
    if not isinstance(rows, list):
        raise ValueError(form)  # noqa: TRY004 - one message for every malformed value
    rows = [p for p in rows if not (isinstance(p, list) and all(str(v).strip() == "" for v in p))]  # "1,2;" -> 1,2
    if any(not isinstance(p, list) or len(p) != 2 for p in rows):  # noqa: PLR2004 - (x, y)
        raise ValueError(form)
    return [_floats(p, form) for p in rows]


def box(text: str, name: str = "box") -> list[float] | None:
    """`"x0,y0,x1,y1"` in pixels (or a JSON list) -> `[x0, y0, x1, y1]`; empty text gives `None`."""
    text = text.strip().strip("[]").strip()
    if not text:
        return None
    form = f"{name}: write a box as 'x0,y0,x1,y1' in pixels (left, top, right, bottom), got {text!r}"
    values = text.split(",")
    if len(values) != 4:  # noqa: PLR2004 - x0, y0, x1, y1
        raise ValueError(form)
    return _floats(values, form)


# COCO classes of the bodies a pose label set describes; hand and face models take every box (a hand / face detector's)
_ANIMALS = frozenset({"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"})
_PEOPLE = frozenset({"person"})
SUBJECTS: dict[str, frozenset[str]] = {
    "coco-keypoints-17": _PEOPLE,
    "coco-wholebody-133": _PEOPLE,
    "halpe-26": _PEOPLE,
    "ap10k-17": _ANIMALS,
}


def subject_boxes(detections: str, i: int, label_set: str) -> list[list[float]] | None:
    """The boxes of image `i` a pose model of `label_set` estimates, from a detection node's JSON; `None`: empty text.

    The JSON is a list (one entry per image) of lists of dicts with `box` and `class_name`. Body models take the
    detections of their subject (people, or the COCO animals for AP-10K); boxes of every class are used if none is
    one, and for hand and face models.
    """
    if not detections.strip():
        return None
    form = (
        "detections: connect the 'detections' output of a Detect node (a JSON list with one list of"
        f" {{box, class_name}} entries per image), got {detections.strip()[:80]!r}"
    )
    try:
        per_image: list[list[dict[str, Any]]] = json.loads(detections)
        if not isinstance(per_image, list) or not per_image:
            raise TypeError  # noqa: TRY301 - one message for every malformed shape
        dets = per_image[min(i, len(per_image) - 1)]
        subject = SUBJECTS.get(label_set)
        picked = [d for d in dets if d.get("class_name") in subject] if subject else []
        return [_floats(d["box"], form) for d in picked or dets]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
        raise ValueError(form) from e


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
    return torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div_(255)


def to_json(obj: object) -> str:
    """Tensors and tuples as plain JSON lists (for STRING outputs)."""

    def plain(o: object) -> object:
        if isinstance(o, torch.Tensor):
            return o.tolist()
        msg = f"not JSON serialisable: {type(o).__name__}"
        raise TypeError(msg)

    return json.dumps(obj, default=plain)


@functools.cache
def palette(n: int) -> torch.Tensor:
    """`[n, 3]` float colours in `[0, 1]`, one per class: the PASCAL VOC bit colormap (class 0 is black). Read-only."""
    out = torch.zeros(n, _RGB)
    for c in range(n):
        v, rgb = c, [0, 0, 0]
        for bit in range(8):
            for k in range(_RGB):
                rgb[k] |= ((v >> k) & 1) << (7 - bit)
            v >>= _RGB
        out[c] = torch.tensor(rgb, dtype=torch.float32) / 255
    return out


def class_ids(names: str, class_names: Sequence[str]) -> list[int]:
    """Comma-separated class names (case-insensitive) -> their indices; unknown names raise `ValueError`."""
    index = {n.lower(): i for i, n in enumerate(class_names)}
    wanted = [n.strip().lower() for n in names.split(",") if n.strip()]
    if unknown := [n for n in wanted if n not in index]:
        msg = "classes: " + unknown_classes(unknown, class_names)
        raise ValueError(msg)
    return [index[n] for n in wanted]


_SHOWN_CLASSES = 12


def unknown_classes(unknown: Sequence[str], class_names: Sequence[str]) -> str:
    """The message for class names a model does not have: the close matches, else the first of its classes."""
    lower = {n.lower(): n for n in class_names}
    close = [m for u in unknown for m in difflib.get_close_matches(u.lower(), lower, n=2)]
    msg = f"unknown classes {list(unknown)} (this model has {len(class_names)} classes)"
    if close:
        return f"{msg}; did you mean {', '.join(dict.fromkeys(lower[m] for m in close))}?"
    shown = ", ".join(class_names[:_SHOWN_CLASSES])
    return f"{msg}, e.g. {shown}{', ...' if len(class_names) > _SHOWN_CLASSES else ''}; names are case-insensitive"


def class_presence(labels: torch.Tensor, n: int) -> torch.Tensor:
    """Which of `n` classes appear in a label map: `[n + 1]` bool on its device, read without a host sync.

    The last entry flags indices outside `[0, n)`; `present_names` raises for it once the batch is read.
    """
    seen = torch.zeros(n + 1, dtype=torch.bool, device=labels.device)
    flat = labels.flatten()
    return seen.index_fill_(0, torch.where((flat >= 0) & (flat < n), flat, n), value=True)  # `[i] = True` syncs


def present_names(seen: torch.Tensor, class_names: Sequence[str]) -> list[str]:
    """`class_presence` rows `[B, n + 1]` -> per image, its classes' names (comma-separated, in class order)."""
    if seen[:, len(class_names)].any():
        msg = f"the model returned class indices outside its {len(class_names)} classes"
        raise ValueError(msg)
    return [", ".join(class_names[c] for c in row.nonzero().flatten().tolist()) for row in seen[:, :-1]]


def labels_image(labels: torch.Tensor, colours: torch.Tensor) -> torch.Tensor:
    """Class indices `[H, W]` -> colour map `[3, H, W]` (`colours[c]` per class).

    Pass `colours` on the labels' device: an upload per image waits for the device's queued work.
    """
    return colours.to(labels.device)[labels].permute(2, 0, 1)


def flow_image(flow: torch.Tensor) -> torch.Tensor:
    """Optical flow `[2, H, W]` (dx, dy pixels) -> the Middlebury colour-wheel image `[3, H, W]`.

    Hue = direction, saturation = speed scaled by the largest finite speed of the image (white = no motion).
    """
    dx, dy = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
    wheel = _colour_wheel(flow.device)
    n = wheel.shape[0]
    rad = torch.sqrt(dx**2 + dy**2)
    rad = rad / rad.max().clamp_min(torch.finfo(torch.float32).eps)
    fk = (torch.atan2(-dy, -dx) / torch.pi + 1) / 2 * (n - 1)
    k0 = fk.floor().long().clamp(0, n - 1)
    k1 = (k0 + 1) % n
    f = (fk - k0).unsqueeze(-1)
    col = (1 - f) * wheel[k0] + f * wheel[k1]
    col = 1 - rad.unsqueeze(-1) * (1 - col)
    return col.clamp(0, 1).permute(2, 0, 1)


@functools.cache
def _colour_wheel(device: torch.device) -> torch.Tensor:
    """The Baker et al. colour wheel `[55, 3]` in `[0, 1]` on `device`: red, yellow, green, cyan, blue, magenta, red.

    Cached: an upload per call would wait for the device's queued work. Read-only.
    """
    stops = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0)]
    counts = [15, 6, 4, 11, 13, 6]
    rows = []
    for a, b, n in zip(stops, stops[1:], counts, strict=False):
        start, end = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
        rows.append(start + (end - start) * (torch.arange(n, dtype=torch.float32).unsqueeze(-1) / n))
    return torch.cat(rows).to(device)


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp: IMAGE `[B, H, W, C]` sampled at `x + flow(x)`, flow `[B, H, W, 2]` -> `[B, H, W, C]`.

    With the flow from frame t to frame t + 1, warping frame t + 1 gives frame t; outside the image is black.
    """
    b, h, w, _ = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    gx = (xs + flow[..., 0]) / max(w - 1, 1) * 2 - 1
    gy = (ys + flow[..., 1]) / max(h - 1, 1) * 2 - 1
    out = torch.nn.functional.grid_sample(
        image.to(flow.device, torch.float32).permute(0, 3, 1, 2),
        torch.stack([gx, gy], dim=-1).expand(b, h, w, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out.permute(0, 2, 3, 1).contiguous()


def grid_points(n: int, size: tuple[int, int]) -> list[list[float]]:
    """An `n x n` grid of `(x, y)` pixel points at cell centres of an image of `size` `(H, W)`."""
    h, w = size
    return [[(j + 0.5) * w / n, (i + 0.5) * h / n] for i in range(n) for j in range(n)]


def draw_tracks(  # noqa: PLR0913 - one video's tracks
    frame: torch.Tensor, tracks: torch.Tensor, visible: torch.Tensor, t: int, *, trail: int, width: int
) -> torch.Tensor:
    """Frame `[3, H, W]` with every point's track of the last `trail` frames and its dot where it is visible.

    Args:
        frame: The image to draw on, `[3, H, W]` in `[0, 1]`.
        tracks: `[T, Q, 2]` pixels.
        visible: `[T, Q]` bool.
        t: The frame index.
        trail: How many earlier frames of each track to draw as a line.
        width: Line width in pixels.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415 - only this node draws

    canvas = Image.fromarray((frame.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy())
    draw = ImageDraw.Draw(canvas)
    pts, vis = tracks[max(t - trail, 0) : t + 1].tolist(), visible[max(t - trail, 0) : t + 1].tolist()
    for q in range(tracks.shape[1]):
        colour = _COLOURS[q % len(_COLOURS)]
        line = [tuple(p[q]) for p, v in zip(pts, vis, strict=True) if v[q]]
        if len(line) > 1:
            draw.line(line, fill=colour, width=width)
        if vis[-1][q]:
            x, y = pts[-1][q]
            draw.ellipse([x - width - 1, y - width - 1, x + width + 1, y + width + 1], fill=colour)
    return torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div_(255)
