"""Video depth accuracy on TUM RGB-D: one call per sequence, AbsRel / delta1 against the Kinect depth.

Sequences (`FASTFORKS_DATA`/multiview/tum_independent, CC BY 4.0): freiburg1_room, freiburg3_long_office_household,
freiburg3_structure_texture_far. `--frames N`: the first N frames of each (dataset `tum-video-N`), else whole sequences
(`tum-video`). Ground truth: the depth PNG nearest in time (< 20 ms, else the frame is not scored), metres = value /
5000, valid in (0, 10] m. Video Depth Anything's protocol (from DepthCrafter): one least-squares scale + shift per
sequence over all valid pixels, in disparity space for inverse-depth outputs (then depth = 1 / disparity, clipped to
[1e-3, 10] m); AbsRel and delta1 over all valid pixels of a sequence, then the mean over sequences. Metric outputs are
also scored without the fit (`abs_absrel`, `abs_d1`).

Tiers: `upstream` / `upstream16` (the original `infer_video_depth`, fp32 / its `fp32=False` autocast; `FF_VDA_REPO` =
a clone of DepthAnything/Video-Depth-Anything), `portable32`, `portable`, `kernels`.

    python tools/eval_video_depth.py video-depth-anything-s --tiers portable,kernels --frames 110 --out results/raw/x
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ("freiburg1_room", "freiburg3_long_office_household", "freiburg3_structure_texture_far")
MAX_DEPTH = 10.0
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": ("upstream", "float32", "fp32", "original fp32"),
    "upstream16": ("upstream-fp16", "float16", "authors flag", "original fp16"),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}


def load_sequence(name: str, frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    """uint8 RGB frames `[L, H, W, 3]` and ground-truth depth `[L, H, W]` (metres, 0 = not scored)."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "multiview" / "tum_independent" / f"rgbd_dataset_{name}"

    def listing(file: str) -> list[tuple[float, str]]:
        rows = (line.split() for line in (root / file).read_text().splitlines() if not line.startswith("#"))
        return [(float(t), f) for t, f in rows]

    rgb, dep = listing("rgb.txt")[:frames], listing("depth.txt")
    times = np.array([t for t, _ in dep])
    images, gts = [], []
    for t, f in rgb:
        images.append(cv2.imread(str(root / f))[..., ::-1])
        i = int(np.abs(times - t).argmin())
        if abs(times[i] - t) < 0.02:  # noqa: PLR2004 - 20 ms
            gts.append(cv2.imread(str(root / dep[i][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000)
        else:
            gts.append(np.zeros(images[-1].shape[:2], np.float32))
    return np.stack(images), np.stack(gts)


def _frames(pred: np.ndarray, gt: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Valid pixels of each frame as float64 (a frame at a time: a whole sequence in float64 does not fit in RAM)."""
    for p, g in zip(pred, gt, strict=True):
        valid = (g > 0) & (g <= MAX_DEPTH)
        yield p[valid].astype(np.float64), g[valid].astype(np.float64)


def _fit(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Least squares `t ~ s * p + b` over all pixels (float64 normal equations)."""
    a00 = a01 = n = b0 = b1 = 0.0
    for p, t in pairs:
        a00, a01, n, b0, b1 = a00 + p @ p, a01 + p.sum(), n + p.size, b0 + p @ t, b1 + t.sum()
    det = a00 * n - a01 * a01
    return (n * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det


def _scores(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    err = good = n = 0.0
    for pred, gt in pairs:
        err += float(np.sum(np.abs(pred - gt) / gt))
        good += float(np.sum(np.maximum(pred / gt, gt / pred) < 1.25))  # noqa: PLR2004
        n += gt.size
    return err / n, good / n


def score(pred: np.ndarray, gt: np.ndarray, *, inverse: bool) -> dict[str, float]:
    """`rel_absrel` / `rel_d1` after the sequence's scale + shift fit (and unaligned `abs_*` for metric depth)."""
    out = {}
    if inverse:
        s, b = _fit((p, 1 / g) for p, g in _frames(pred, gt))
        aligned = ((1 / np.clip(s * p + b, 1 / MAX_DEPTH, 1e3), g) for p, g in _frames(pred, gt))
    else:
        s, b = _fit(_frames(pred, gt))
        aligned = ((np.clip(s * p + b, 1e-3, MAX_DEPTH), g) for p, g in _frames(pred, gt))
        out["abs_absrel"], out["abs_d1"] = _scores((np.clip(p, 1e-3, None), g) for p, g in _frames(pred, gt))
    out["rel_absrel"], out["rel_d1"] = _scores(aligned)
    return out


_GOLDEN: list[Any] = []


def _original(ref: str, fp32: bool) -> Any:  # noqa: ANN401, FBT001 - the upstream model's call
    from fastforks._core import registry  # noqa: PLC0415

    if not _GOLDEN:  # the golden module's name is upstream's package name: import it once, before the checkout
        sys.path.insert(0, str(ROOT / "tools" / "golden"))
        from _common import reference_tree  # noqa: PLC0415

        _GOLDEN.append(__import__("video_depth_anything"))  # tools/golden/video_depth_anything.py
        tree = reference_tree(Path(os.environ["FF_VDA_REPO"]), _GOLDEN[0].REF_COMMIT)
        sys.path.remove(str(ROOT / "tools" / "golden"))
        del sys.modules["video_depth_anything"]
        sys.path.insert(0, str(tree))
    g = _GOLDEN[0]
    from video_depth_anything.video_depth import VideoDepthAnything  # noqa: PLC0415 # ty: ignore[unresolved-import] - reference checkout

    card, variant = registry.resolve("depth", ref)
    (weights,) = registry.weights(card, variant)
    net = VideoDepthAnything(**g.CONFIG, metric=bool(variant.params.get("metric", False)))
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net = net.cuda().eval()
    return lambda frames: net.infer_video_depth(frames, 30, input_size=518, device="cuda", fp32=fp32)[0]


def predictor(ref: str, tier: str) -> Any:  # noqa: ANN401
    """`frames [L, H, W, 3] uint8 -> depth [L, H, W]` numpy of `tier`."""
    if tier.startswith("upstream"):
        return _original(ref, tier == "upstream")
    from fastforks import depth  # noqa: PLC0415
    from fastforks._core import registry  # noqa: PLC0415

    mode = "kernels" if tier == "kernels" else "portable"
    m = depth.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))
    key = "depth" if registry.resolve("depth", ref)[1].params.get("metric") else "inverse_depth"

    def call(frames: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().div(255)
        return getattr(m(x), key).float().cpu().numpy()

    return call


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = {s: load_sequence(s, a.frames) for s in SEQUENCES}
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    dataset = f"tum-video-{a.frames}" if a.frames else "tum-video"
    from fastforks._core import registry  # noqa: PLC0415

    for ref in a.refs:
        inverse = not registry.resolve("depth", ref)[1].params.get("metric", False)
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn = predictor(ref, tier)
            fn(data[SEQUENCES[0]][0][:32])  # warm-up
            torch.cuda.reset_peak_memory_stats()
            per, seconds, frames = [], 0.0, 0
            for images, gt in data.values():
                torch.cuda.synchronize()
                t = time.perf_counter()
                pred = fn(images)
                torch.cuda.synchronize()
                seconds += time.perf_counter() - t
                frames += len(images)
                per.append(score(pred, gt, inverse=inverse))
            metrics = {k: round(float(np.mean([p[k] for p in per])), 4) for k in per[0]}
            row = {
                "ref": ref, "task": "depth", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": "python tools/eval_video_depth.py " + " ".join(sys.argv[1:]),
                "dataset": dataset, "n_images": frames, **metrics, "n": frames, "metrics": metrics,
                "per_sequence": dict(zip(SEQUENCES, per, strict=True)),
                "ms_per_img": round(seconds * 1e3 / frames, 2),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20), "column": column,
            }  # fmt: skip
            print(json.dumps(row), flush=True)
            with (a.out / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            del fn
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
