"""Example outputs for the website's model cards: one fixed, freely licensed input per task through every variant.

    python tools/examples.py TASK [REF,...] --out DIR [--mode auto] [--device cuda] [--force]

Needs scikit-image (its sample images are the inputs). Writes `DIR/<ref>.webp` (the rendered output; classification
has none, its top five are the example) and `DIR/inputs/<input>.webp`, and appends one row per ref to
`docs/data/examples.jsonl` (latest row per ref wins): `ref`, `task`, `input`, `mode`, `dtype`, `device`, `date`, plus
per task `top` (classification: five `[class, probability]`), `classes` (segmentation: the largest classes and their
share of the image), `certain` (matching: share of pixels of the first view with
certainty above 0.5), `objects` (detection: `[class, score]` above `SCORE`) or `prompt` (promptable segmentation).
Failures are rows with `error`. Refs that already have an image and a row are skipped unless `--force`.

Inputs (see `INPUTS`): scikit-image's public-domain / CC0 photos, and its downscaled Middlebury 2014 motorcycle pair
for stereo, flow, matching and multi-view (calibrated multi-view models get the pair's published calibration).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import DATA

if TYPE_CHECKING:
    from collections.abc import Callable

ROWS = Path(os.environ.get("EXAMPLES_ROWS") or Path(__file__).resolve().parents[1] / "docs/data/examples.jsonl")
SCORE = 0.5  # detection boxes drawn and listed
KEYPOINT = 0.3  # pose keypoints drawn
BODY = 17  # COCO keypoints; more is whole-body (DWPose), drawn thinner
MAX_WIDTH = 640
QUALITY = 80

# name: (scikit-image loader, title, credit, licence, source)
INPUTS: dict[str, tuple[str, str, str, str, str]] = {
    "astronaut": ("astronaut", "Astronaut Eileen Collins", "NASA", "public domain", "https://flic.kr/p/r9qvLn"),
    "chelsea": (
        "chelsea",
        "Chelsea the cat",
        "Stéfan van der Walt",
        "CC0",
        "https://scikit-image.org/docs/stable/api/skimage.data.html",
    ),
    "coffee": (
        "coffee",
        "A cup of coffee",
        "Rachel Michetti",
        "CC0",
        "https://scikit-image.org/docs/stable/api/skimage.data.html",
    ),
    "motorcycle": (
        "stereo_motorcycle",
        "Motorcycle, Middlebury 2014 stereo (left and right view, downscaled 4x)",
        "Scharstein et al., Middlebury College",
        "research dataset, no licence stated",
        "https://vision.middlebury.edu/stereo/data/scenes2014/",
    ),
}
_SK = "https://scikit-image.org/docs/stable/api/skimage.data.html"
_SKCRED = "scikit-image sample image"
INPUTS |= {
    "astronaut-face": (
        "astronaut",
        "Astronaut Eileen Collins, face crop",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "rocket": ("rocket", "Rocket launch", "SpaceX", "public domain", _SK),
    "page": ("page", "Scanned text page, upscaled 3x", "scikit-image", "CC0", _SK),
    "page-line": ("page", "One text line of the scanned page, upscaled 3x", "scikit-image", "CC0", _SK),
    "astronaut-lowres": (
        "astronaut",
        "Astronaut, downscaled (4x, or 2x for the x2 models) with blur and noise",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "astronaut-blur": (
        "astronaut",
        "Astronaut with synthetic motion blur",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "astronaut-noise": (
        "astronaut",
        "Astronaut with synthetic Gaussian noise (sigma 25)",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "astronaut-rain": (
        "astronaut",
        "Astronaut with synthetic rain streaks",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "astronaut-hole": (
        "astronaut",
        "Astronaut with a masked hole (red)",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "astronaut-face-lowres": (
        "astronaut",
        "Astronaut's face, 4x downscaled, blurred, noisy",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "coffee-pan": (
        "coffee",
        "Two frames of a synthetic pan over the coffee photo (24 px apart)",
        "Rachel Michetti",
        "CC0",
        _SK,
    ),
    "astronaut-pan": (
        "astronaut",
        "Eight frames of a synthetic pan over the astronaut photo (6 px right, 2 px down per frame), 6 query points",
        "NASA",
        "public domain",
        "https://flic.kr/p/r9qvLn",
    ),
    "gallery": (
        "astronaut",
        "Query: a rotated crop of the coffee photo; database: nine scikit-image sample photos",
        _SKCRED,
        "public domain / CC0",
        _SK,
    ),
    "ct-s0013": (
        "",
        "CT case s0013 of TotalSegmentator v3.0.0 (test split), axial and coronal slice",
        "Wasserthal et al., TotalSegmentator, Radiology: AI 2023",
        "CC-BY-4.0",
        "https://doi.org/10.5281/zenodo.6802613",
    ),
}
TASK_INPUT = {
    "classification": "chelsea",
    "features": "chelsea",
    "depth": "coffee",
    "segmentation": "coffee",
    "promptable_segmentation": "coffee",
    "detection": "astronaut",
    "pose": "astronaut",
    "stereo": "motorcycle",
    "flow": "motorcycle",
    "multiview": "motorcycle",
    "matching": "motorcycle",
    "edges": "coffee",
    "matting": "astronaut",
    "restoration": "astronaut",
    "retrieval": "gallery",
    "calibration": "rocket",
    "ocr": "page",
    "interpolation": "coffee-pan",
    "tracking": "astronaut-pan",
    "volume": "ct-s0013",
}
# Variants whose task input is another one: faces for the face models, a text line for the line recogniser.
INPUT_OF = {
    "bisenet-face-parsing": "astronaut-face",
    "segformer-b5-face-parsing": "astronaut-face",
    "segformer-b2-clothes": "astronaut",
    "parseq": "page-line",
    "parseq-tiny": "page-line",
}
FACE = (150, 20, 350, 220)  # the astronaut's face (x0, y0, x1, y1)
OPEN_VOCAB = ["person", "flag", "helmet"]  # open-vocabulary detectors' queries on the astronaut
VOLUME = DATA / "totalsegmentator_subset" / "s0013" / "ct.nii.gz"
DEGRADED_BY_REF = {
    "nafnet-gopro": "blur",
    "restormer-deblur": "blur",
    "nafnet-sidd": "noise",
    "restormer-sidd": "noise",
    "restormer-derain": "rain",
    "big-lama": "hole",
}
PERSON = (15.0, 10.0, 370.0, 511.0)  # the astronaut's box (x0, y0, x1, y1), the pose models' person box
POINT = (290.0, 235.0)  # a foreground point on the coffee cup
CONCEPT = "cup"
# Middlebury 2014 motorcycle at 1/4 size (scikit-image's docstring): focal, principal point, right-view offset, baseline
FOCAL, CX, CY, DOFFS, BASELINE = 994.978, 311.193, 254.877, 31.086, 0.193
DEPTH_RANGE = (1.5, 8.0)  # metres, around the scene's 2-5 m


def _rgb(a: np.ndarray) -> np.ndarray:
    a = a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a[..., :3])  # noqa: PLR2004


def _sk(name: str) -> np.ndarray:
    from skimage import data  # noqa: PLC0415  # ty: ignore[unresolved-import] - tool-only dependency

    return _rgb(getattr(data, name)())


def _up(a: np.ndarray, factor: float, resample: Image.Resampling = Image.Resampling.BICUBIC) -> np.ndarray:
    im = Image.fromarray(a)
    return np.asarray(im.resize((round(im.width * factor), round(im.height * factor)), resample))


def _pan(img: np.ndarray, n: int, dx: int, dy: int, size: tuple[int, int]) -> list[np.ndarray]:
    """`n` crops of `size` (h, w) from `img`, each `(dx, dy)` further along: a pan, exact integer motion."""
    h, w = size
    return [np.ascontiguousarray(img[i * dy : i * dy + h, i * dx : i * dx + w]) for i in range(n)]


def _images(name: str) -> list[np.ndarray]:  # noqa: PLR0911 - one return per input kind
    from skimage import data  # noqa: PLC0415  # ty: ignore[unresolved-import] - tool-only dependency

    if name == "astronaut-face":
        x0, y0, x1, y1 = FACE
        return [_up(_sk("astronaut")[y0:y1, x0:x1], 2.56)]
    if name == "page":
        return [_up(_sk("page"), 3)]
    if name == "page-line":
        return [_up(_sk("page"), 3)[20:100]]
    if name == "coffee-pan":
        return _pan(_sk("coffee"), 2, 24, 0, (400, 576))
    if name == "astronaut-pan":
        return _pan(_sk("astronaut"), 8, 6, 2, (456, 456))
    if name == "gallery":
        return [_gallery_query()]
    if name == "ct-s0013":
        return []
    out = getattr(data, INPUTS[name][0])()
    return [np.ascontiguousarray(a[..., :3]) for a in out[:2]] if isinstance(out, tuple) else [_rgb(out)]


GALLERY = ["astronaut", "chelsea", "coffee", "rocket", "coins", "hubble_deep_field", "camera", "retina", "brick"]


def _gallery_query() -> np.ndarray:
    """The query: the coffee photo's centre, rotated by 8 degrees (the database holds the original)."""
    c = Image.fromarray(_sk("coffee"))
    c = c.rotate(8, Image.Resampling.BICUBIC, expand=False).crop((100, 50, 500, 350))
    return np.asarray(c)


# --- colours -------------------------------------------------------------------------------------------------------


def turbo(x: np.ndarray) -> np.ndarray:
    """Google's polynomial approximation of the Turbo colormap: `[...]` in [0, 1] -> uint8 `[..., 3]`."""
    x = np.clip(x, 0, 1)[..., None]
    c = np.array(
        [
            [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
            [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
            [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
        ]
    )
    rgb = sum(c[:, i] * x**i for i in range(6))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _normalise(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = x[np.isfinite(x) & (valid if valid is not None else True)]
    lo, hi = np.percentile(v, (2, 98)) if v.size else (0.0, 1.0)
    return np.nan_to_num((x - lo) / max(hi - lo, 1e-9), nan=0.0)


def _wheel() -> np.ndarray:
    """The Middlebury flow colour wheel (Baker et al.), `[55, 3]` in [0, 255]."""
    segments = [(15, (255, 0, 0), (255, 255, 0)), (6, (255, 255, 0), (0, 255, 0)), (4, (0, 255, 0), (0, 255, 255))]
    segments += [(11, (0, 255, 255), (0, 0, 255)), (13, (0, 0, 255), (255, 0, 255)), (6, (255, 0, 255), (255, 0, 0))]
    rows = [np.array(a) + (np.array(b) - np.array(a)) * i / n for n, a, b in segments for i in range(n)]
    return np.array(rows, dtype=np.float64)


def flow_colour(flow: np.ndarray) -> np.ndarray:
    """`[2, H, W]` flow -> uint8 `[H, W, 3]`: hue = direction, saturation = magnitude (normalised by its 98th pct)."""
    u, v = flow
    mag = np.hypot(u, v)
    mag = mag / max(float(np.percentile(mag, 98)), 1e-9)
    wheel = _wheel()
    angle = np.arctan2(-v, -u) / np.pi
    k = (angle + 1) / 2 * (len(wheel) - 1)
    k0 = np.floor(k).astype(int)
    k1 = (k0 + 1) % len(wheel)
    f = (k - k0)[..., None]
    col = ((1 - f) * wheel[k0] + f * wheel[k1]) / 255
    m = np.clip(mag, 0, 1)[..., None]
    col = 1 - m * (1 - col)
    return (col * 255).astype(np.uint8)


def palette(i: int) -> tuple[int, int, int]:
    """A stable, well-spread colour per class / instance index (golden-ratio hues)."""
    import colorsys  # noqa: PLC0415

    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, 0.65 + 0.25 * (i % 2), 0.95)
    return round(r * 255), round(g * 255), round(b * 255)


def _blend(img: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = img.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, np.float32)
    return out.astype(np.uint8)


def _font(size: int = 15) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, colour: tuple[int, int, int]) -> None:
    font = _font()
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=font)
    draw.rectangle((x0 - 3, y0 - 2, x1 + 3, y1 + 2), fill=colour)
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def _contract(model: Any) -> Any:  # noqa: ANN401
    return importlib.import_module(f"fastforks.{model.card.task}").model_card(model.variant.id).contract()


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


# --- per task: run and render --------------------------------------------------------------------------------------


def _depth(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    extra: dict[str, Any] = {}
    kw: dict[str, Any] = {}
    if _contract(model).inputs.get("prompt_depth") == "required":
        # A synthetic LiDAR-like prompt (24 x 32 px, 0.35 to 0.7 m, farther towards the top): no real sensor
        h, w = 24, 32
        kw["prompt_depth"] = np.repeat(np.linspace(0.7, 0.35, h, dtype=np.float32)[:, None], w, 1)
        extra["prompt"] = "synthetic 24x32 depth, a tilted plane from 0.35 m (bottom) to 0.7 m (top)"
    r = model(imgs[0], **kw)
    if r.depth is None and r.inverse_depth is None and r.normals is not None:  # normal estimators: normals as colour
        n = _np(r.normals)
        return ((np.clip(n, -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8), {"kind": "normals"}
    inv = _np(r.inverse_depth) if r.inverse_depth is not None else 1 / np.maximum(_np(r.depth), 1e-6)
    valid = _np(r.mask) > 0 if r.mask is not None else None
    out = turbo(_normalise(inv, valid))
    if valid is not None:
        out[~valid] = 0
    return out, {"kind": str(r.kind), **extra}


def _stereo(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    d = _np(model(imgs[0], imgs[1]).disparity)
    return turbo(_normalise(d)), {}


def _flow(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    return flow_colour(_np(model(imgs[0], imgs[1]).flow)), {}


def _multiview(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    runner = model._runner  # noqa: SLF001 - which cameras the model requires
    kw: dict[str, Any] = {}
    if runner.intrinsics == "required":
        k = np.array([[FOCAL, 0, CX, 0, FOCAL, CY, 0, 0, 1], [FOCAL, 0, CX + DOFFS, 0, FOCAL, CY, 0, 0, 1]], np.float32)
        kw["intrinsics"] = k.reshape(2, 3, 3)
    if runner.cam_to_world == "required":
        c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, 0)
        c2w[1, 0, 3] = BASELINE
        kw["cam_to_world"] = c2w
    if runner.depth_range == "required":
        kw["depth_range"] = DEPTH_RANGE
    r = model(imgs, **kw)
    depth = _np(r.depth[0])
    valid = (depth > 0) & np.isfinite(depth)
    if r.mask is not None:
        valid &= _np(r.mask[0]) > 0
    out = turbo(_normalise(1 / np.where(valid, depth, np.inf), valid))
    out[~valid] = 0
    return out, {"cameras": sorted(kw)}


LINES = 100  # matching: matches drawn


def _matching(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    """The first view above the second, `LINES` of the most certain-weighted sampled matches joined by lines."""
    r = model(imgs[0], imgs[1])
    p1, p2 = (_np(p) for p in (r.matches(LINES) if hasattr(r, "warp") else r.matches()))  # dense: sampled, sparse: all
    if len(p1) > LINES:
        keep = np.random.default_rng(0).choice(len(p1), LINES, replace=False)
        p1, p2 = p1[keep], p2[keep]
    a, b = imgs
    w = max(a.shape[1], b.shape[1])
    canvas = np.zeros((a.shape[0] + b.shape[0], w, 3), np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[a.shape[0] :, : b.shape[1]] = b
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    colours = turbo(_normalise(p1[:, 0])[None])[0]
    for (x1, y1), (x2, y2), c in zip(p1, p2, colours, strict=True):
        col = tuple(int(v) for v in c)
        draw.line([(x1, y1), (x2, y2 + a.shape[0])], fill=col, width=2)
        for x, y in ((x1, y1), (x2, y2 + a.shape[0])):
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    certainty = getattr(r, "certainty", None)
    certain = float((_np(certainty) > 0.5).mean()) if certainty is not None else None  # noqa: PLR2004
    return np.asarray(pil), {} if certain is None else {"certain": round(certain, 4)}


def _classification(model: Any, imgs: list[np.ndarray]) -> tuple[None, dict[str, Any]]:  # noqa: ANN401
    return None, {"top": [[n, round(float(p), 4)] for n, p in model(imgs[0]).top(5)]}


def _features(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0])
    gh, gw = r.grid
    tokens = r.last_hidden_state[-gh * gw :].float()
    x = tokens - tokens.mean(0)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    pcs = _np(x @ v[:3].T).reshape(gh, gw, 3)
    rgb = np.stack([np.clip(_normalise(pcs[..., i]), 0, 1) for i in range(3)], -1)
    h, w = imgs[0].shape[:2]
    small = Image.fromarray((rgb * 255).astype(np.uint8))
    return np.asarray(small.resize((w, h), Image.Resampling.BICUBIC)), {"grid": [gh, gw]}


def _segmentation(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0])
    labels = _np(r.labels).astype(np.int64)
    out = imgs[0].copy()
    ids, counts = np.unique(labels, return_counts=True)
    for i in ids:
        out = _blend(out, labels == i, palette(int(i)), 0.6)
    order = np.argsort(-counts)[:6]
    classes = [[r.class_names[ids[j]], round(float(counts[j]) / labels.size, 3), palette(int(ids[j]))] for j in order]
    return out, {"classes": classes}


def _instances(
    img: np.ndarray,
    r: Any,  # noqa: ANN401
    keep: np.ndarray,
    unnamed: str = "object",
) -> tuple[np.ndarray, list[list[Any]]]:
    out = img.copy()
    objects = []
    for n, i in enumerate(np.flatnonzero(keep)):
        colour = palette(int(r.labels[i]) if r.labels is not None else n)
        if r.masks is not None:
            out = _blend(out, _np(r.masks[i]) > 0, colour, 0.45)
        name = r.class_names[int(r.labels[i])] if r.labels is not None and r.class_names else unnamed
        objects.append([name, round(float(r.scores[i]), 3), colour, [round(float(c), 1) for c in r.boxes[i]]])
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    for name, score, colour, box in objects:
        draw.rectangle(box, outline=colour, width=3)
        _label(draw, (box[0] + 2, max(box[1] - 18, 0)), f"{name} {score:.2f}", colour)
    return np.asarray(pil), [o[:2] for o in objects]


def _detection(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    vocab = model.open_vocabulary
    r = model(imgs[0], text=OPEN_VOCAB) if vocab else model(imgs[0])
    out, objects = _instances(imgs[0], r, _np(r.scores) >= SCORE)
    return out, {"objects": objects, **({"prompt": "text " + ", ".join(f'"{t}"' for t in OPEN_VOCAB)} if vocab else {})}


def _promptable(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    if "text" in getattr(model, "prompts", ()):
        r = model(imgs[0], text=CONCEPT)
        out, objects = _instances(imgs[0], r, _np(r.scores) >= SCORE, CONCEPT)
        return out, {"prompt": f'text "{CONCEPT}"', "objects": objects}
    r = model(imgs[0], input_points=[list(POINT)], input_labels=[1])
    best = int(torch.argmax(r.iou_scores))
    out = _blend(imgs[0], _np(r.masks[best]) > 0, (64, 160, 255), 0.55)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    x, y = POINT
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 220, 0), outline=(0, 0, 0), width=2)
    return np.asarray(pil), {"prompt": "one foreground point", "iou": round(float(r.iou_scores[best]), 3)}


PEOPLE = 0.5  # pose: person score a bottom-up model's detection needs to be drawn


def _pose(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    # Top-down models get the fixed person box; bottom-up ones (RTMO, RF-DETR keypoint) find the people themselves.
    r = model(imgs[0], boxes=torch.tensor([PERSON])) if model.needs_boxes else model(imgs[0])
    kp, sc = _np(r.keypoints), _np(r.keypoint_scores)
    if not model.needs_boxes:  # their own detections: only the people they are confident about
        keep = _np(r.scores) > PEOPLE
        kp, sc = kp[keep], sc[keep]
    pil = Image.fromarray(imgs[0])
    draw = ImageDraw.Draw(pil)
    small = kp.shape[1] > BODY
    for p in range(kp.shape[0]):
        for n, (a, b) in enumerate(r.skeleton):
            if sc[p, a] > KEYPOINT and sc[p, b] > KEYPOINT:
                draw.line((*kp[p, a], *kp[p, b]), fill=palette(n), width=2 if small else 4)
        rad = 2 if small else 4
        for (x, y), s in zip(kp[p], sc[p], strict=True):
            if s > KEYPOINT:
                draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(255, 255, 255), outline=(0, 0, 0))
    return np.asarray(pil), {"keypoints": int(kp.shape[1]), "shown": int((sc > KEYPOINT).sum()), "people": len(kp)}


# --- edges, matting, restoration, retrieval, calibration, OCR, interpolation, tracking, volume ---


def _edges(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0])
    e = (np.clip(_np(r.edges), 0, 1) * 255).astype(np.uint8)
    extra = {} if r.lines is None else {"lines": len(r.lines)}
    return np.repeat(e[..., None], 3, -1), extra


def _matting(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    """The cut-out foreground on a flat green; trimap / mask models get a box around the astronaut as their guide."""
    img = imgs[0]
    h, w = img.shape[:2]
    kw: dict[str, Any] = {}
    guide = _contract(model).inputs
    box = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = (int(v) for v in PERSON)
    box[y0:y1, x0:x1] = 255
    if guide.get("trimap") == "required":  # foreground: the inner box, unknown: a 40 px band around it
        tri = np.zeros((h, w), np.uint8)
        tri[max(y0 - 40, 0) : y1 + 40, max(x0 - 40, 0) : x1 + 40] = 128
        tri[y0 + 40 : y1 - 40, x0 + 40 : x1 - 40] = 255
        kw["trimap"] = tri
    if guide.get("mask") == "required":
        kw["mask"] = box
    # video matting (MatAnyone) takes the image as a one-frame video
    a = _np(model.video([img], mask=box).alpha[0]) if "mask" in kw else _np(model(img, **kw).alpha)
    a = np.clip(a, 0, 1)[..., None]
    out = (a * img + (1 - a) * np.array([40, 180, 90])).astype(np.uint8)
    extra = {"guide": "a person box" + (" as trimap" if "trimap" in kw else "")} if kw else {}
    return out, extra


def _degrade(ref: str, scale: int, face: bool) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:  # noqa: FBT001
    """`(clean, degraded, input name, call kwargs)` for a restoration variant: what it is meant to repair."""
    rng = np.random.default_rng(0)
    if face:
        clean = _images("astronaut-face")[0]
        small = _up(clean, 0.25, Image.Resampling.BOX)
        low = np.clip(_up(small, 4).astype(np.float32) + rng.normal(0, 4, clean.shape), 0, 255).astype(np.uint8)
        return clean, low, "astronaut-face-lowres", {}
    clean = _sk("astronaut")
    kind = DEGRADED_BY_REF.get(ref)
    x = clean.astype(np.float32)
    if kind == "blur":
        k = 21
        x = np.mean([np.roll(x, i - k // 2, axis=1) for i in range(k)], axis=0)  # horizontal motion blur
        return clean, np.clip(x, 0, 255).astype(np.uint8), "astronaut-blur", {}
    if kind == "noise":
        return clean, np.clip(x + rng.normal(0, 25, x.shape), 0, 255).astype(np.uint8), "astronaut-noise", {}
    if kind == "rain":
        img = Image.fromarray(clean)
        draw = ImageDraw.Draw(img)
        for _ in range(350):
            px, py = rng.integers(0, 512, 2)
            draw.line((px, py, px - 4, py + int(rng.integers(12, 30))), fill=(215, 215, 225), width=1)
        return clean, np.asarray(img), "astronaut-rain", {}
    if kind == "hole":
        mask = np.zeros(clean.shape[:2], np.uint8)
        mask[150:300, 380:500] = 255  # the flag's lower part and the background beside the shoulder
        shown = clean.copy()
        shown[mask > 0] = (255, 0, 0)
        return clean, shown, "astronaut-hole", {"mask": mask}
    small = _up(clean, 1 / scale, Image.Resampling.BICUBIC)
    if "real" in ref:  # the "real-world" variants are trained on noise and blur on top of the downscaling
        small = np.clip(small.astype(np.float32) + rng.normal(0, 3, small.shape), 0, 255).astype(np.uint8)
    return clean, small, "astronaut-lowres", {}


def _restoration(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401, ARG001 - the clean image is built from the variant
    ref = model.variant.id
    face = any(k in ref for k in ("gfpgan", "codeformer"))
    _clean, low, name, kw = _degrade(ref, model.scale, face)
    r = model(low, **kw)
    out = (np.clip(_np(r.image), 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    inp = _up(low, out.shape[0] / low.shape[0], Image.Resampling.NEAREST)
    side = np.concatenate([inp, out], 1)
    return side, {
        "input": name,
        "_inputs": [low],
        "scale": model.scale,
        "layout": "degraded input (left) | restored (right)",
    }


def _retrieval(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    from fastforks.retrieval import top_k  # noqa: PLC0415

    gallery = [_sk(n) for n in GALLERY]
    q = model(imgs[0]).descriptor
    db = torch.stack([model(g).descriptor for g in gallery])
    scores, index = top_k(q, db, k=len(gallery))
    order = [(int(i), float(s)) for i, s in zip(index[0], scores[0], strict=True)]
    tile = 200
    canvas = Image.new("RGB", (tile * 5, tile * 2 + 8), (30, 30, 30))
    q_img = Image.fromarray(imgs[0])
    q_img.thumbnail((tile, tile))
    canvas.paste(q_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    _label(draw, (4, 4), "query", (255, 220, 0))
    for rank, (i, s) in enumerate(order[:8]):
        t = Image.fromarray(gallery[i])
        t.thumbnail((tile, tile))
        x, y = tile * ((rank + 1) % 5), (tile + 8) * ((rank + 1) // 5)
        canvas.paste(t, (x, y))
        _label(draw, (x + 4, y + 4), f"#{rank + 1} {s:.2f}", palette(rank))
    return np.asarray(canvas), {"top": [[GALLERY[i], round(s, 3)] for i, s in order[:3]]}


def _calibration(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0])
    w = imgs[0].shape[1]
    k = _np(r.intrinsics)
    pil = Image.fromarray(imgs[0])
    draw = ImageDraw.Draw(pil)
    lines = [f"focal {k[0, 0]:.0f} px"]
    extra: dict[str, Any] = {"focal_px": round(float(k[0, 0]), 1)}
    if r.vfov is not None:
        extra["vfov_deg"] = round(float(np.degrees(_np(r.vfov))), 1)
        lines.append(f"vertical FoV {extra['vfov_deg']:.1f} deg")
    if r.gravity is not None:
        gx, gy, gz = _np(r.gravity)
        # horizon: the image points whose ray is perpendicular to gravity, gx (x - cx) + gy (y - cy) + gz f = 0
        cx, cy = k[0, 2], k[1, 2]
        ys = [cy - (gx * (x - cx) + gz * k[0, 0]) / gy for x in (0, w)] if abs(gy) > 1e-6 else None  # noqa: PLR2004
        if ys:
            draw.line((0, ys[0], w, ys[1]), fill=(255, 220, 0), width=3)
        if r.roll is not None:
            extra["roll_deg"] = round(float(np.degrees(_np(r.roll))), 1)
            extra["pitch_deg"] = round(float(np.degrees(_np(r.pitch))), 1) if r.pitch is not None else None
            lines.append(f"roll {extra['roll_deg']:.1f}, pitch {extra['pitch_deg']} deg")
    for i, text in enumerate(lines):
        _label(draw, (6, 6 + 24 * i), text, (255, 255, 255))
    return np.asarray(pil), extra


def _ocr(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0])
    pil = Image.fromarray(imgs[0])
    draw = ImageDraw.Draw(pil)
    for i, quad in enumerate(_np(r.boxes)):
        draw.polygon([tuple(p) for p in quad], outline=palette(i), width=2)
    return np.asarray(pil), {
        "lines": len(r.texts),
        "texts": r.texts[:12],
        "mean_score": round(float(_np(r.scores).mean()), 3),
    }


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else round(10 * float(np.log10(255**2 / mse)), 2)


def _interpolation(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    """The middle of a 24 px pan; the true middle frame is the crop 12 px along (the PSNR is against it)."""
    mid = _sk("coffee")[0:400, 12:588]
    out = (np.clip(_np(model(imgs[0], imgs[1]).frame), 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return out, {"psnr_vs_true_middle": _psnr(out, mid), "motion_px": 24}


def _tracking(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    n = len(imgs)
    q0 = np.array([[130, 120], [250, 100], [200, 250], [300, 350], [100, 380], [330, 200]], np.float32)  # on frame 0
    queries = np.concatenate([np.zeros((len(q0), 1), np.float32), q0], 1)
    r = model(imgs, queries)
    tracks = _np(r.tracks)  # [T, Q, 2]
    truth = q0[None] - np.arange(n, dtype=np.float32)[:, None, None] * np.array(
        [6, 2], np.float32
    )  # content moves left/up
    err = np.linalg.norm(tracks - truth, axis=-1)
    pil = Image.fromarray(imgs[-1]).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for q in range(len(q0)):
        pts = [tuple(p) for p in tracks[:, q]]
        draw.line(pts, fill=palette(q), width=3)
        x, y = pts[-1]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=palette(q), outline=(255, 255, 255), width=2)
    return np.asarray(pil), {"mean_error_px": round(float(err.mean()), 2), "frames": n, "points": len(q0)}


def _slice_rgb(ct: np.ndarray, lab: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """A CT slice `[a, b]` (soft-tissue window) with the label colours blended on, resampled to isotropic pixels."""
    g = ((np.clip(ct, -150, 250) + 150) / 400 * 255).astype(np.uint8)
    img = np.repeat(g[..., None], 3, -1)
    for i in np.unique(lab):
        if i:
            img = _blend(img, lab == i, palette(int(i)), 0.55)
    h = round(img.shape[0] * spacing[0] / min(spacing))
    w = round(img.shape[1] * spacing[1] / min(spacing))
    return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.NEAREST))


def _volume(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401, ARG001 - the CT is read from disk
    from fastforks.volume import read_nifti  # noqa: PLC0415

    vol, meta = read_nifti(VOLUME)
    r = model(vol, meta=meta)
    lab = _np(r.labels).astype(np.int64)
    sp = tuple(float(s) for s in meta.spacing)  # (d, h, w) mm
    z = int(np.argmax((lab > 0).sum((1, 2))))
    y = int(np.argmax((lab > 0).sum((0, 2))))
    axial = _slice_rgb(vol[z], lab[z], (sp[1], sp[2]))
    coronal = _slice_rgb(vol[:, y][::-1], lab[:, y][::-1], (sp[0], sp[2]))  # head up
    hgt = max(axial.shape[0], coronal.shape[0])
    pad = lambda a: np.pad(a, ((0, hgt - a.shape[0]), (0, 0), (0, 0)))  # noqa: E731
    out = np.concatenate([pad(axial), pad(coronal)], 1)
    plain = np.concatenate(
        [
            pad(_slice_rgb(vol[z], np.zeros_like(lab[z]), (sp[1], sp[2]))),
            pad(_slice_rgb(vol[:, y][::-1], np.zeros_like(lab[:, y]), (sp[0], sp[2]))),
        ],
        1,
    )
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    top = [
        [r.class_names.get(int(i), str(i)), round(float(c) * float(np.prod(sp)) / 1e6, 2), palette(int(i))]
        for i, c in sorted(zip(ids, counts, strict=True), key=lambda t: -t[1])[:6]
    ]
    return out, {
        "_inputs": [plain],
        "classes_litres": top,
        "labels_found": len(ids),
        "layout": "axial | coronal slice",
        "attribution": "TotalSegmentator dataset v3.0.0 case s0013, Wasserthal et al., Radiology: AI 2023, CC-BY-4.0",
    }


RENDER: dict[str, Callable[[Any, list[np.ndarray]], tuple[np.ndarray | None, dict[str, Any]]]] = {
    "depth": _depth,
    "stereo": _stereo,
    "flow": _flow,
    "multiview": _multiview,
    "matching": _matching,
    "classification": _classification,
    "features": _features,
    "segmentation": _segmentation,
    "detection": _detection,
    "promptable_segmentation": _promptable,
    "pose": _pose,
    "edges": _edges,
    "matting": _matting,
    "restoration": _restoration,
    "retrieval": _retrieval,
    "calibration": _calibration,
    "ocr": _ocr,
    "interpolation": _interpolation,
    "tracking": _tracking,
    "volume": _volume,
}


def _save(img: np.ndarray, path: Path) -> None:
    pil = Image.fromarray(img)
    if pil.width > MAX_WIDTH:
        pil = pil.resize((MAX_WIDTH, round(pil.height * MAX_WIDTH / pil.width)), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, "WEBP", quality=QUALITY, method=6)


def _done() -> set[str]:
    if not ROWS.exists():
        return set()
    today = _today()  # a failure of today is not retried (the next step of the same job would hit it again)
    rows = map(json.loads, ROWS.read_text().splitlines())
    return {r["ref"] for r in rows if "error" not in r or r.get("date") == today}


def _inputs_for(task: str, ref: str) -> tuple[str, list[np.ndarray]]:
    name = INPUT_OF.get(ref, TASK_INPUT[task])
    return name, _images(name)


def _save_inputs(out: Path, name: str, imgs: list[np.ndarray]) -> None:
    for i, img in enumerate(imgs[:2]):
        target = out / "inputs" / f"{name}{'-right' if i else ''}.webp"
        if not target.exists():
            _save(img, target)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=sorted(RENDER))
    p.add_argument("refs", nargs="?", default="", help="comma-separated variant ids (default: every variant)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--budget", type=float, default=0, help="seconds: start no new variant after this (0: no limit)")
    a = p.parse_args()

    start = time.monotonic()
    api = importlib.import_module(f"fastforks.{a.task}")
    refs = a.refs.split(",") if a.refs else api.list_models()
    done = _done()
    inflight = a.out / f".inflight-{a.task}"
    if inflight.exists():  # the previous step was killed (time limit) on this variant: record it, do not retry
        ref = inflight.read_text()
        inflight.unlink()
        row = {
            "ref": ref,
            "task": a.task,
            "input": TASK_INPUT[a.task],
            "date": _today(),
            "error": "killed: over the 175 s step limit",
        }
        with ROWS.open("a") as f:
            f.write(json.dumps(row) + "\n")
    for ref in refs:
        if ref in done and not a.force:
            continue
        if a.budget and time.monotonic() - start > a.budget:
            print("budget used up", flush=True)
            return 3
        name, imgs = _inputs_for(a.task, ref)
        _save_inputs(a.out, name, imgs)
        target = a.out / f"{ref}.webp"
        row: dict[str, Any] = {"ref": ref, "task": a.task, "input": name, "date": _today()}
        inflight.write_text(ref)
        try:
            model = api.load(ref, mode=a.mode, device=a.device)
            with torch.inference_mode():
                img, extra = RENDER[a.task](model, imgs)
            for i, inp in enumerate(extra.pop("_inputs", [])):
                _save(inp, a.out / "inputs" / f"{extra.get('input', name)}{'-right' if i else ''}.webp")
            if img is not None:
                _save(img, target)
            row |= {"mode": model.backend.mode, "dtype": str(model.backend.dtype), **extra}
            row["device"] = torch.cuda.get_device_name() if a.device.startswith("cuda") else a.device
        except Exception as e:  # noqa: BLE001 - one row per ref, failures recorded
            row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        finally:
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        inflight.unlink(missing_ok=True)
        print(json.dumps(row), flush=True)
        with ROWS.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


def _today() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


if __name__ == "__main__":
    sys.exit(main())
