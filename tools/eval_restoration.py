"""Restoration accuracy: DIV2K validation (super-resolution, bicubic LR -> HR) or the variant's own test set.

Super-resolution: `FASTFORKS_DATA`/restoration/div2k/DIV2K_valid_HR/08xx.png and
DIV2K_valid_LR_bicubic/X<s>/08xxx<s>.png (the 100 validation images, s = the model's scale; dataset `div2k-val-x<s>`;
`--every 2`: every 2nd, `div2k-val-x<s>-every2`), one pass each at its own size, no tiles. Scoring:
the output rounded to 8 bits, its Y channel (BT.601, `16 + 65.481 R + 128.553 G + 24.966 B` on [0, 1] RGB, / 255)
against the HR image's, `s` px border cropped; PSNR = -10 log10(MSE) and SSIM (Gaussian 11 x 11, sigma 1.5,
K = 0.01 / 0.03, data range 1, valid region) per image, averaged over the images.

Scale-1 variants (`SETS`, the sets and channels their papers report): `FASTFORKS_DATA`/restoration/<dir>/{input,target}/
*.png, no border: GoPro test (1111 frames, RGB), SIDD validation (the 1280 256 x 256 sRGB crops of
NAFNet's / Restormer's `.lmdb`, written out as PNGs, RGB), Rain100L / Rain100H test (Y); on RGB, PSNR over all
channels and SSIM the mean of the per-channel SSIMs (BasicSR's `calculate_psnr` / `calculate_ssim`, `test_y_channel`
false). Upstream's SIDD numbers score the float output; here every set scores the 8-bit one.

Tiers: `upstream` / `upstream16` (the original as `tools/_original.py` `LOCAL[family]` builds it, e.g. `RealESRGANer`
fp32 / inference_realesrgan.py's default half; `upstream16` only for originals with an fp16 option of their own),
`portable32`, `portable`, `kernels`.

    python tools/eval_restoration.py realesrgan-x4plus --tiers portable,kernels --out out/x
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from _prefetch import prefetch

ROOT = Path(__file__).resolve().parents[1]
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": ("upstream", "float32", "--fp32", "original fp32"),
    "upstream16": ("upstream-fp16", "float16", "inference_realesrgan.py (half)", "original fp16"),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}
HOW = {  # family -> tier -> `how`, where not Real-ESRGAN's
    "span": {"upstream": "BasicSR SRModel.test, float32"},
    "swinir": {"upstream": "main_test_swinir.py, float32"},
    "swin2sr": {"upstream": "main_test_swin2sr.py, float32"},
    "nafnet": {"upstream": "ImageRestorationModel.test, float32"},
    "hat": {"upstream": "hat_model.py test, float32"},
    "dat": {"upstream": "dat_model.py test, float32"},
    "drct": {"upstream": "drct_model.py test, float32"},
    "realplksr": {"upstream": "neosr test.py (net(lq), clamped), float32"},
    "codeformer": {"upstream": "inference_codeformer.py, float32"},
}


_GOPRO = ("gopro-test", "gopro/GoPro/test/GoPro", False)
_SIDD = ("sidd-val", "sidd/val", False)
SETS = {  # scale-1 variant -> [(dataset, directory under restoration/, Y channel)]
    "nafnet-gopro": [_GOPRO],
    "nafnet-sidd": [_SIDD],
    "restormer-deblur": [_GOPRO],
    "restormer-sidd": [_SIDD],
    "restormer-derain": [("rain100l", "rain/test/Rain100L", True), ("rain100h", "rain/test/Rain100H", True)],
}


def set_pairs(directory: str) -> list[tuple[Path, Path]]:
    """(input, target) paths of a scale-1 set."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "restoration" / directory
    return [(p, root / "target" / p.name) for p in sorted((root / "input").glob("*.png"))]


def pairs(scale: int) -> list[tuple[Path, Path]]:
    """(LR, HR) paths of the 100 validation images."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "restoration" / "div2k"
    hr = sorted((root / "DIV2K_valid_HR").glob("*.png"))
    return [(root / "DIV2K_valid_LR_bicubic" / f"X{scale}" / f"{p.stem}x{scale}.png", p) for p in hr]


def read(item: tuple[Path, Path]) -> list[np.ndarray]:
    """LR and HR as RGB uint8."""
    return [np.ascontiguousarray(cv2.imread(str(p))[..., ::-1]) for p in item]


def _y(rgb: torch.Tensor) -> torch.Tensor:
    """`[3, H, W]` RGB in [0, 1] -> `[1, 1, H, W]` BT.601 Y in [0, 1] (float64)."""
    w = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float64, device=rgb.device)
    return ((16 + (rgb.double() * w[:, None, None]).sum(0)) / 255)[None, None]


def score(pred: torch.Tensor, gt: torch.Tensor, border: int, *, y: bool = True) -> tuple[float, float]:
    """PSNR and SSIM on Y (or RGB) of `[3, H, W]` images in [0, 1] (the prediction already rounded to 8 bits)."""
    h, w = pred.shape[-2:]
    crop = (..., slice(border, h - border), slice(border, w - border))
    p, g = ((_y(t) if y else t.double()[:, None])[crop] for t in (pred, gt))  # RGB: channels as the batch
    psnr = float(-10 * torch.log10((p - g).square().mean()))
    k = torch.exp(-((torch.arange(11, device=p.device, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    k = k / k.sum()
    w = (k[:, None] * k[None, :])[None, None]
    mu_p, mu_g = F.conv2d(p, w), F.conv2d(g, w)
    s_pp = F.conv2d(p * p, w) - mu_p**2
    s_gg = F.conv2d(g * g, w) - mu_g**2
    s_pg = F.conv2d(p * g, w) - mu_p * mu_g
    c1, c2 = 0.01**2, 0.03**2
    ssim = (2 * mu_p * mu_g + c1) * (2 * s_pg + c2) / ((mu_p**2 + mu_g**2 + c1) * (s_pp + s_gg + c2))
    return psnr, float(ssim.mean())


def predictor(ref: str, tier: str) -> tuple[Any, int, str]:
    """(`[3, h, w]` float CUDA in [0, 1] -> `[3, sh, sw]` of `tier`, scale, family)."""
    from fastforks import restoration  # noqa: PLC0415
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = next((c, v) for c, v in registry.cards("restoration").values() if v.id == ref)
    scale = int({**card.params, **variant.params}["scale"])
    if tier.startswith("upstream"):
        sys.path.insert(0, str(ROOT / "tools"))
        import _original  # noqa: PLC0415 - tools/_original.py

        adapter = _original.LOCAL[card.family]
        call = adapter.build(ref, tier == "upstream16")
        return lambda x: call(*adapter.prep(((x * 255).round().byte(),))), scale, card.family
    mode = "kernels" if tier == "kernels" else "portable"
    m = restoration.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))
    return lambda x: m(x).image, scale, card.family


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    for ref in a.refs:
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn, scale, family = predictor(ref, tier)
            how = HOW.get(family, {}).get(tier, how)
            for name, directory, y in SETS.get(ref, [(f"div2k-val-x{scale}", None, True)]):
                data = (pairs(scale) if directory is None else set_pairs(directory))[:: a.every]
                dataset = name + (f"-every{a.every}" if a.every > 1 else "")
                border = scale if directory is None else 0
                try:
                    fn(torch.rand(3, 1024 // scale, 1536 // scale, device="cuda"))
                except torch.OutOfMemoryError:  # warm-up only: the images decide
                    torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                scores, times, error = [], [], None
                with closing(prefetch(read, data, workers=4)) as images:
                    for lr, hr in images:
                        x, gt = (torch.from_numpy(f).cuda().permute(2, 0, 1).float() / 255 for f in (lr, hr))
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        try:
                            with torch.inference_mode():
                                out = fn(x)
                        except torch.OutOfMemoryError as e:  # the row says so (fp32 originals on large images)
                            error = f"OOM, {lr.shape[0]}x{lr.shape[1]}: {str(e).splitlines()[0]}"
                            break
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t)
                        h, w = out.shape[-2:]
                        scores.append(score((out.clamp(0, 1) * 255).round() / 255, gt[:, :h, :w], border, y=y))
                metrics = {}
                if error is None:
                    psnr, ssim = np.mean(scores, 0)
                    metrics = {"psnr": round(float(psnr), 2), "ssim": round(float(ssim), 4)}
                row = {
                    "ref": ref, "task": "restoration", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                    "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "command": "python tools/eval_restoration.py " + " ".join(sys.argv[1:]),
                    "dataset": dataset, **metrics, "n": len(data), "metrics": metrics,
                    "ms_per_image": round(float(np.median(times)) * 1e3, 2) if error is None else None,
                    "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20), "column": column,
                    **({} if error is None else {"error": error}),
                }  # fmt: skip
                print(json.dumps(row), flush=True)
                with (a.out / "rows.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            del fn
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
