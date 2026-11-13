"""Frame-interpolation accuracy: middle frames of DAVIS 2017 trainval 480p.

`FASTFORKS_DATA`/davis2017/DAVIS/JPEGImages/480p/<video>/<frame>.jpg (90 videos, sorted by name): per video the
triplets (0, 2 -> 1) and (20, 22 -> 21), 180 in all (dataset `davis2017-triplets180`). Each pair at its own size (most
480 x 854), one call each at timestep 0.5. Scoring: the output rounded to 8 bits, against the true middle frame in
[0, 1]; PSNR = -10 log10(MSE) per triplet, SSIM with an 11 x 11 Gaussian window (sigma 1.5, K = 0.01 / 0.03, data
range 1, valid region) averaged over the channels, both averaged over the triplets.

Tiers: `upstream` / `upstream16` (Practical-RIFE `Model.inference`, fp32 / inference_video.py's `--fp16`, as
`tools/_original.py` `_rife`: `FF_RIFE_REPO` = a hzwer/Practical-RIFE clone, `FF_RIFE_RELEASES` = one unpacked release
per variant id), `portable32`, `portable`, `kernels`.

    python tools/eval_interpolation.py rife-4.25 --tiers portable,kernels --out results/raw/x
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATASET = "davis2017-triplets180"
STARTS = (0, 20)  # first frame of each triplet per video
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": ("upstream", "float32", "fp32", "original fp32"),
    "upstream16": ("upstream-fp16", "float16", "inference_video.py --fp16", "original fp16"),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}


def triplets() -> list[tuple[Path, Path, Path]]:
    """(frame t, frame t + 2, frame t + 1) paths, sorted by video."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "davis2017" / "DAVIS" / "JPEGImages" / "480p"
    out = []
    for video in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted(video.glob("*.jpg"))
        out += [(frames[s], frames[s + 2], frames[s + 1]) for s in STARTS]
    return out


def read(item: tuple[Path, Path, Path]) -> list[np.ndarray]:
    """The three frames as RGB uint8."""
    return [np.ascontiguousarray(cv2.imread(str(p))[..., ::-1]) for p in item]


def _window(device: torch.device) -> torch.Tensor:
    g = torch.exp(-((torch.arange(11, device=device, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()


def score(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """PSNR and SSIM of `[3, H, W]` float images in [0, 1] (the prediction already rounded to 8 bits)."""
    p, g = pred.double()[None], gt.double()[None]
    psnr = float(-10 * torch.log10((p - g).square().mean()))
    w = _window(p.device)
    mu_p, mu_g = F.conv2d(p, w, groups=3), F.conv2d(g, w, groups=3)
    s_pp = F.conv2d(p * p, w, groups=3) - mu_p**2
    s_gg = F.conv2d(g * g, w, groups=3) - mu_g**2
    s_pg = F.conv2d(p * g, w, groups=3) - mu_p * mu_g
    c1, c2 = 0.01**2, 0.03**2
    ssim = (2 * mu_p * mu_g + c1) * (2 * s_pg + c2) / ((mu_p**2 + mu_g**2 + c1) * (s_pp + s_gg + c2))
    return psnr, float(ssim.mean())


def predictor(ref: str, tier: str) -> Any:  # noqa: ANN401
    """`[1, 3, H, W]` pair, float CUDA in [0, 1] -> the middle frame `[3, H, W]` of `tier`."""
    if tier.startswith("upstream"):
        sys.path.insert(0, str(ROOT / "tools"))
        import _original  # noqa: PLC0415 - tools/_original.py

        a = _original.LOCAL["rife"]
        call = a.build(ref, tier == "upstream16")
        return lambda x1, x2: call((x1[0] * 255).round().byte(), (x2[0] * 255).round().byte())[0].float()
    from fastforks import interpolation  # noqa: PLC0415

    mode = "kernels" if tier == "kernels" else "portable"
    m = interpolation.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))
    return lambda x1, x2: m(x1, x2).frame[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = triplets()
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    for ref in a.refs:
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn = predictor(ref, tier)
            warm = torch.rand(1, 3, 480, 854, device="cuda")
            fn(warm, warm)
            torch.cuda.reset_peak_memory_stats()
            scores, times = [], []
            with ThreadPoolExecutor(4) as pool:  # decode the next frames while the GPU runs
                for frames in pool.map(read, data):
                    x1, x2, gt = (torch.from_numpy(f).cuda().permute(2, 0, 1).float() / 255 for f in frames)
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    with torch.inference_mode():
                        out = fn(x1[None], x2[None])
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                    scores.append(score((out.clamp(0, 1) * 255).round() / 255, gt))
            psnr, ssim = np.mean(scores, 0)
            metrics = {"psnr": round(float(psnr), 2), "ssim": round(float(ssim), 4)}
            row = {
                "ref": ref, "task": "interpolation", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": "python tools/eval_interpolation.py " + " ".join(sys.argv[1:]),
                "dataset": DATASET, "n_triplets": len(data), **metrics, "n": len(data), "metrics": metrics,
                "ms_per_pair": round(float(np.median(times)) * 1e3, 2),
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
