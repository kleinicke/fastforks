"""Matting accuracy on DIS5K's validation split (DIS-VD): MAE and max F-beta of the alpha against the binary masks.

Data: `FASTFORKS_DATA`/matting/DIS5K/DIS-VD/{im,gt} (470 images, sorted by name; `--every 2`: every 2nd, dataset
`dis-vd-every2`). Each image at its own size, one call each. Scoring as BiRefNet's evaluation (py_sod_metrics): the
alpha saved as 8-bit (`round(255 * alpha)`), min-max normalised per image, the mask = gt > 128; MAE = mean |pred - gt|;
F-beta (beta^2 = 0.3) at the 256 thresholds of the 8-bit prediction, averaged over images per threshold, the maximum.

Tiers: `upstream` / `upstream16` (the HF `birefnet.py` of the weights, fp32 / the authors' `half()`, with the card's
pre- and post-processing, `tools/golden/birefnet.py`; `FF_BIREFNET_HF` = a directory with the HF snapshots
`<ref>/birefnet.py`), `portable32`, `portable`, `kernels`.

    python tools/eval_matting.py birefnet-lite --tiers portable,kernels --every 2 --out results/raw/x
"""

from __future__ import annotations

import argparse
import datetime as dt
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
BETA2 = 0.3
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": ("upstream", "float32", "fp32", "original fp32"),
    "upstream16": ("upstream-fp16", "float16", "authors half()", "original fp16"),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}


def items(every: int) -> list[tuple[Path, Path]]:
    """(image, mask) paths of DIS-VD, sorted by name, every `every`-th."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "matting" / "DIS5K" / "DIS-VD"
    ims = sorted((root / "im").iterdir())[::every]
    return [(p, root / "gt" / f"{p.stem}.png") for p in ims]


def score(alpha: torch.Tensor, gt: torch.Tensor) -> tuple[float, torch.Tensor]:
    """MAE and the F-beta curve over the 256 thresholds of one image on the GPU (py_sod_metrics' `MAE` / `Fmeasure`)."""
    pred = (alpha.double() * 255).round() / 255  # the saved 8-bit PNG
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    mask = gt > 128  # noqa: PLR2004 - py_sod_metrics
    mae = float((pred - mask.double()).abs().mean())
    p8 = (pred * 255).to(torch.uint8).long()
    fg = torch.bincount(p8[mask], minlength=256).flip(0).cumsum(0).double()  # predicted >= t, t = 255 .. 0
    bg = torch.bincount(p8[~mask], minlength=256).flip(0).cumsum(0).double()
    precision = fg / (fg + bg).clamp(min=1)
    recall = fg / max(int(mask.sum()), 1)
    den = BETA2 * precision + recall
    return mae, torch.where(den == 0, 0, (1 + BETA2) * precision * recall / den.clamp(min=1e-300))


def read(item: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
    """RGB image, grey mask."""
    im, gt = item
    return np.ascontiguousarray(cv2.imread(str(im))[..., ::-1]), cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)


def predictor(ref: str, tier: str) -> Any:  # noqa: ANN401
    """`[1, 3, H, W]` float CUDA in [0, 1] -> alpha `[H, W]` of `tier`."""
    if tier.startswith("upstream"):
        sys.path.insert(0, str(ROOT / "tools" / "golden"))
        from fastforks._core import registry  # noqa: PLC0415

        card, variant = registry.resolve("matting", ref)
        (weights,) = registry.weights(card, variant)
        if ref == "ben2":  # the authors' forward: fp16 = its own CUDA autocast
            import ben2  # noqa: PLC0415 - tools/golden/ben2.py

            net = ben2.network(Path(os.environ["FF_BEN2_REPO"]), Path(weights)).cuda()
            fp16 = tier == "upstream16"

            def call(x: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    return ben2.predict(net, x, autocast=fp16)[0]

            return call
        from birefnet import network, predict  # noqa: PLC0415 - tools/golden/birefnet.py

        net = network(Path(os.environ["FF_BIREFNET_HF"]) / ref, Path(weights)).cuda()
        if tier == "upstream16":
            net = net.half()

        def call(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return predict(net, x)[0]

        return call
    from fastforks import matting  # noqa: PLC0415

    mode = "kernels" if tier == "kernels" else "portable"
    m = matting.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))
    return lambda x: m(x).alpha[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = items(a.every)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    dataset = f"dis-vd-every{a.every}" if a.every > 1 else "dis-vd"
    for ref in a.refs:
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn = predictor(ref, tier)
            fn(torch.rand(1, 3, 512, 768, device="cuda"))  # warm-up
            torch.cuda.reset_peak_memory_stats()
            maes, curves, times = [], [], []
            with ThreadPoolExecutor(4) as pool:  # decode the next images while the GPU runs
                for img, gt in pool.map(read, data):
                    x = torch.from_numpy(img).cuda().permute(2, 0, 1)[None].float() / 255
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    alpha = fn(x)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                    mae, f = score(alpha.float(), torch.from_numpy(gt).cuda())
                    maes.append(mae)
                    curves.append(f)
            metrics = {
                "mae": round(float(np.mean(maes)), 4),
                "max_fbeta": round(float(torch.stack(curves).mean(0).max()), 4),
            }
            row = {
                "ref": ref, "task": "matting", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": "python tools/eval_matting.py " + " ".join(sys.argv[1:]),
                "dataset": dataset, "n_images": len(data), **metrics, "n": len(data), "metrics": metrics,
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
