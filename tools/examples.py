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
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Callable

ROWS = Path(__file__).resolve().parents[1] / "docs/data/examples.jsonl"
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
}
PERSON = (15.0, 10.0, 370.0, 511.0)  # the astronaut's box (x0, y0, x1, y1), the pose models' person box
POINT = (290.0, 235.0)  # a foreground point on the coffee cup
CONCEPT = "cup"
# Middlebury 2014 motorcycle at 1/4 size (scikit-image's docstring): focal, principal point, right-view offset, baseline
FOCAL, CX, CY, DOFFS, BASELINE = 994.978, 311.193, 254.877, 31.086, 0.193
DEPTH_RANGE = (1.5, 8.0)  # metres, around the scene's 2-5 m


def _images(name: str) -> list[np.ndarray]:
    from skimage import data  # noqa: PLC0415  # ty: ignore[unresolved-import] - tool-only dependency

    out = getattr(data, INPUTS[name][0])()
    return [np.ascontiguousarray(a[..., :3]) for a in out[:2]] if isinstance(out, tuple) else [out[..., :3]]


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


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


# --- per task: run and render --------------------------------------------------------------------------------------


def _depth(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0])
    inv = _np(r.inverse_depth) if r.inverse_depth is not None else 1 / np.maximum(_np(r.depth), 1e-6)
    valid = _np(r.mask) > 0 if r.mask is not None else None
    out = turbo(_normalise(inv, valid))
    if valid is not None:
        out[~valid] = 0
    return out, {"kind": str(r.kind)}


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
    p1, p2 = (_np(p) for p in r.matches(LINES))
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
    certain = float((_np(r.certainty) > 0.5).mean()) if r.certainty is not None else None  # noqa: PLR2004
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
    r = model(imgs[0])
    out, objects = _instances(imgs[0], r, _np(r.scores) >= SCORE)
    return out, {"objects": objects}


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


def _pose(model: Any, imgs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: ANN401
    r = model(imgs[0], boxes=torch.tensor([PERSON]))
    kp, sc = _np(r.keypoints), _np(r.keypoint_scores)
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
    return np.asarray(pil), {"keypoints": int(kp.shape[1]), "shown": int((sc > KEYPOINT).sum())}


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
    return {r["ref"] for r in map(json.loads, ROWS.read_text().splitlines()) if "error" not in r}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=sorted(RENDER))
    p.add_argument("refs", nargs="?", default="", help="comma-separated variant ids (default: every variant)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    api = importlib.import_module(f"fastforks.{a.task}")
    refs = a.refs.split(",") if a.refs else api.list_models()
    name = TASK_INPUT[a.task]
    imgs = _images(name)
    for i, img in enumerate(imgs):
        target = a.out / "inputs" / f"{name}{'-right' if i else ''}.webp"
        if not target.exists():
            _save(img, target)
    done = _done()
    for ref in refs:
        target = a.out / f"{ref}.webp"
        if not a.force and ref in done and (target.exists() or a.task == "classification"):
            continue
        row: dict[str, Any] = {
            "ref": ref,
            "task": a.task,
            "input": name,
            "date": dt.datetime.now(dt.UTC).date().isoformat(),
        }
        try:
            model = api.load(ref, mode=a.mode, device=a.device)
            with torch.inference_mode():
                img, extra = RENDER[a.task](model, imgs)
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
        print(json.dumps(row), flush=True)
        with ROWS.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
