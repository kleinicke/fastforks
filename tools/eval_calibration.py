"""Calibration accuracy: roll / pitch / vertical field of view on LaMAR2k (GeoCalib's benchmark split).

`FASTFORKS_DATA`/calibration/lamar2k/{images.csv,images/} (the GeoCalib release `lamar2k.zip`: 2000 HoloLens / phone
images, 1920 x 1440 portrait; `images.csv`: ground-truth roll, pitch, vfov in radians), in the csv's order; `--every
10`: every 10th image (dataset `lamar2k-every10`). One call per image at its own size. Errors: absolute differences
in degrees; AUC at 1 / 5 / 10 degrees as upstream `siclib.utils.tools.compute_auc(min_error=1)` (the eval of
`siclib/eval/simple_pipeline.py`), in percent.

Tiers: `upstream` / `upstream16` (the authors' `GeoCalib.calibrate`, the family's `tools/golden/<family>.py`: fp32 /
`torch.autocast` float16; no precision option of its own; `FF_<FAMILY>_REPO` = the upstream checkout), `portable32`,
`portable`, `kernels`.

    python tools/eval_calibration.py geocalib-pinhole --tiers upstream,portable,kernels --every 10 --out results/raw/x
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (1, 5, 10)  # degrees
ANGLES = ("roll", "pitch", "vfov")
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": ("upstream", "float32", "fp32", "original fp32"),
    "upstream16": ("upstream-fp16", "float16", "autocast", "original fp16"),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}


def dataset(every: int = 1) -> tuple[list[Path], np.ndarray]:
    """Image paths in the csv's order, every `every`-th, and their ground truth `[N, 3]` (roll, pitch, vfov; rad)."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "calibration" / "lamar2k"
    with (root / "images.csv").open() as f:
        rows = list(csv.DictReader(f))[::every]
    gt = np.array([[float(r["roll"]), float(r["pitch"]), float(r["vfov"])] for r in rows])
    return [root / "images" / r["fname"] for r in rows], gt


def auc(errors: np.ndarray, t: float, min_error: float = 1.0) -> float:
    """The AUC of siclib `compute_auc(errors, [t], min_error)`, in percent."""
    e = np.sort(errors)
    recall = (np.arange(len(e)) + 1) / len(e)
    i = np.searchsorted(e, min_error, side="right")
    recall, e = np.r_[i / len(e), i / len(e), recall[i:]], np.r_[0, min_error, e[i:]]
    last = np.searchsorted(e, t, side="right")
    r, x = np.r_[recall[:last], recall[last - 1]], np.r_[e[:last], t]
    return round(float(np.trapezoid(r, x=x) / t) * 100, 2)


def read(p: Path) -> np.ndarray:
    """RGB uint8."""
    return np.ascontiguousarray(cv2.imread(str(p))[..., ::-1])


def _row(out: Any) -> torch.Tensor:  # noqa: ANN401 - a dict of batched outputs
    """`[3]` (roll, pitch, vfov) of the first image; NaN for an output the model does not give."""
    v = [out.get(k) for k in ANGLES]
    ref = next(x for x in v if x is not None)
    return torch.stack([ref[0].new_tensor(float("nan")) if x is None else x[0] for x in v])


def predictor(ref: str, tier: str) -> Any:  # noqa: ANN401
    """`[1, 3, H, W]` float CUDA in [0, 1] -> `[3]` (roll, pitch, vfov; rad; NaN if not given) of `tier`."""
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("calibration", ref)
    if tier.startswith("upstream"):
        sys.path.insert(0, str(ROOT / "tools" / "golden"))
        spec = importlib.util.spec_from_file_location(
            f"golden_{card.family}", ROOT / "tools" / "golden" / f"{card.family}.py"
        )
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        (weights,) = registry.weights(card, variant)
        net = g.network(Path(os.environ[f"FF_{card.family.upper()}_REPO"]), Path(weights)).cuda()

        def call(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=tier == "upstream16"):
                out = g.predict(net, x)
            return _row(out)

        return call
    from fastforks import calibration  # noqa: PLC0415

    mode = "kernels" if tier == "kernels" else "portable"
    m = calibration.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))

    def run(x: torch.Tensor) -> torch.Tensor:
        r = m(x)
        return _row({k: getattr(r, k) for k in ANGLES})

    return run


def predict(fn: Any, paths: list[Path], times: list[float]) -> np.ndarray:  # noqa: ANN401
    """`[N, 3]` float64 of `paths`, one call each (timed, decoding overlapped)."""
    out = []
    with ThreadPoolExecutor(4) as pool:
        for img in pool.map(read, paths):
            x = torch.from_numpy(img).cuda().permute(2, 0, 1)[None].float() / 255
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = fn(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            out.append(y.double().cpu().numpy())
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths, gt = dataset(a.every)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    name = f"lamar2k-every{a.every}" if a.every > 1 else "lamar2k"
    for ref in a.refs:
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn = predictor(ref, tier)
            for _ in range(3):
                fn(torch.rand(1, 3, 1920, 1440, device="cuda"))
            torch.cuda.reset_peak_memory_stats()
            times: list[float] = []
            err = np.degrees(np.abs(predict(fn, paths, times) - gt))
            given = [i for i in range(len(ANGLES)) if not np.isnan(err[:, i]).all()]  # a model without roll / pitch
            metrics = {f"{ANGLES[i]}_auc{t}": auc(err[:, i], t) for i in given for t in THRESHOLDS}
            row = {
                "ref": ref, "task": "calibration", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": "python tools/eval_calibration.py " + " ".join(sys.argv[1:]),
                "dataset": name, "n_images": len(paths), **metrics, "n": len(paths), "metrics": metrics,
                "median_err_deg": {ANGLES[i]: round(float(np.median(err[:, i])), 3) for i in given},
                "ms_per_img": round(float(np.median(times)) * 1e3, 2),
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
