"""Metric depth of Prompt Depth Anything (original, portable, kernels) on NYUv2 and iBims-1 (simulated prompt).

    FASTFORKS_KERNEL_PATH=... python tools/eval_prompt_depth.py --ref prompt-depth-anything-l --out raw.jsonl
    python tools/eval_prompt_depth.py --ref any2full-b --tiers upstream,portable,kernels --out raw.jsonl   # others

Prompt = the ground-truth depth reduced to 192 x 256 (the size of an iPhone LiDAR map): mean of the valid GT pixels of
each cell, cells without valid pixels take the nearest filled cell. Metric AbsRel / delta1 (no scale fit) over the valid
GT pixels at the GT size; the prediction is resized bilinearly where its size differs. NYUv2 every `--every`th image,
iBims-1 all. Other families (Any2Full, LingBot-Depth, OMNI-DC): the same prompt and metrics; their original = the bench
adapter of `tools/_original.py` (`upstream` fp32, `upstream-fp16` = its `upstream16` tier). Prompt Depth Anything's
original = DepthAnything/PromptDA at the pinned commit (area resize to the network size, `predict`), fp32 or
`torch.autocast(float16)`. One JSON line per tier and dataset (the `docs/data/depth_eval.jsonl` row), plus one
`prompt-only` line: the bilinear upsampled prompt itself as a reference level.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import closing
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from _prefetch import prefetch
from eval_stereo_depth import depth_items

if TYPE_CHECKING:
    from collections.abc import Callable

_EPS = 1e-6  # valid-pixel count floor of the prompt pooling
_DELTA1 = 1.25  # the delta1 threshold
UP = Path(os.environ.get("PROMPTDA_REPO", "upstream/PromptDA"))
CKPT = {
    "prompt-depth-anything-l": ("vitl", "prompt_depth_anything_vitl.ckpt"),
    "prompt-depth-anything-s": ("vits", "prompt_depth_anything_vits.ckpt"),
}
SETS = {"nyu": "nyu-prompt-every{}", "ibims": "ibims-prompt"}


def make_prompt(gt: np.ndarray, hw: tuple[int, int] = (192, 256)) -> np.ndarray:
    """Valid-pixel mean of GT cells, holes filled from the nearest filled cell; float32 metres [h, w]."""
    g = torch.from_numpy(np.nan_to_num(gt, nan=0.0)).float()[None, None]
    m = (torch.from_numpy(np.isfinite(gt) & (gt > 0)).float())[None, None]
    num, den = F.adaptive_avg_pool2d(g * m, hw), F.adaptive_avg_pool2d(m, hw)
    p = (num / den.clamp_min(_EPS))[0, 0].numpy()
    empty = den[0, 0].numpy() < _EPS
    if empty.any():
        idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
        p = p[tuple(idx)]
    return p.astype(np.float32)


def net_size(h: int, w: int, max_size: int = 1008) -> tuple[int, int]:
    s = min(1.0, max_size / max(h, w))
    return int(h * s) // 14 * 14, int(w * s) // 14 * 14


class Tier:
    """One precision tier of one family: the original, portable or kernels runner behind one call."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        self.tier = tier
        self.gen = None
        if (
            tier.startswith("upstream") and ref not in CKPT
        ):  # other depth-completion families: the bench adapter of `tools/_original.py`
            import _original  # noqa: PLC0415

            self.gen = _original.load("depth", ref, "upstream16" if tier == "upstream-fp16" else tier)
            self.orig, self.mode, self.dtype = None, tier, self.gen.dtype
        elif tier.startswith("upstream"):
            sys.path.insert(0, str(UP))
            from promptda.promptda import PromptDA  # noqa: PLC0415

            enc, ck = CKPT[ref]
            self.orig = (
                PromptDA(
                    encoder=enc, ckpt_path=str(Path(os.environ.get("FF_WEIGHTS", "data/ff-weights")) / "prompt-da" / ck)
                )
                .eval()
                .cuda()
            )
            self.mode, self.dtype = tier, "float16" if tier == "upstream-fp16" else "float32"
        else:
            from fastforks import depth  # noqa: PLC0415

            mode, dtype = ("portable", "float32") if tier == "portable32" else (tier, None)
            self.orig, self.model = None, depth.load(ref, mode=mode, dtype=dtype, device="cuda")
            self.mode, self.dtype = mode, str(self.model.backend.dtype).removeprefix("torch.")

    def __call__(self, img: np.ndarray, prompt: np.ndarray) -> torch.Tensor:  # noqa: D102
        p = torch.from_numpy(prompt).cuda()
        im = torch.from_numpy(img).permute(2, 0, 1).contiguous().cuda()
        if self.gen is not None:
            return self.gen(*self.gen.prep((im, p))).float().squeeze()
        if self.orig is None:
            return self.model(im, prompt_depth=p).depth.float().squeeze()
        x = F.interpolate(im[None].float() / 255, net_size(*im.shape[-2:]), mode="area")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.tier == "upstream-fp16"):
            d = self.orig.predict(x, p[None, None])
        return d.float().squeeze()


def metrics(pred: torch.Tensor, gt: np.ndarray) -> tuple[float, float, int]:
    g = torch.from_numpy(gt).cuda()
    if tuple(pred.shape) != tuple(g.shape):
        pred = F.interpolate(pred[None, None], tuple(g.shape), mode="bilinear", align_corners=False)[0, 0]
    m = torch.isfinite(g) & (g > 0) & torch.isfinite(pred)
    a, b = pred.clamp(min=1e-3)[m], g[m]
    return (
        ((a - b).abs() / b).mean().item(),
        (torch.maximum(a / b, b / a) < _DELTA1).float().mean().item(),
        int(m.sum()),
    )


def read_prompt(load: Callable[[], tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the persistent prompt and its ground truth for the prompt-only reference."""
    _, gt = load()
    return make_prompt(gt), gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="prompt-depth-anything-l")
    ap.add_argument("--tiers", default="upstream,upstream-fp16,portable,kernels")
    ap.add_argument("--every", type=int, default=4, help="NYUv2: every N-th image")
    ap.add_argument(
        "--ibims-every", type=int, default=1, help="iBims-1: every N-th image (dataset `ibims-prompt-everyN` if > 1)"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "log", "-1", "--format=%h", "--", "src"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    cmd = "python " + " ".join(sys.argv)
    loaders = {ds: [load for _, load in depth_items(ds)][:: a.every if ds == "nyu" else a.ibims_every] for ds in SETS}
    prompts = {}
    base = {}
    for ds, samples in loaders.items():
        # Keep the small depth prompts for reuse across tiers. Decode full images and labels as each tier needs them.
        prompts[ds], r = [], []
        with closing(prefetch(read_prompt, samples, workers=2)) as inputs:
            for p, gt in inputs:
                prompts[ds].append(p)
                up = F.interpolate(
                    torch.from_numpy(p)[None, None].cuda(), gt.shape, mode="bilinear", align_corners=False
                )
                r.append(metrics(up[0, 0], gt))
        base[ds] = (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r])))
    for tier in a.tiers.split(","):
        pr = Tier(a.ref, tier)
        for ds, samples in loaders.items():
            n = len(samples)
            res, t = [], 0.0
            with closing(prefetch(lambda load: load(), samples, workers=2)) as inputs:
                first = next(inputs)
                pr(first[0], prompts[ds][0])  # warm-up
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for (img, gt), p in zip(chain((first,), inputs), prompts[ds], strict=True):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = pr(img, p)
                    torch.cuda.synchronize()
                    t += time.perf_counter() - t0
                    res.append(metrics(out, gt))
            rel, d1 = float(np.mean([x[0] for x in res])), float(np.mean([x[1] for x in res]))
            dsn = (
                SETS[ds].format(a.every)
                if ds == "nyu"
                else SETS[ds] + (f"-every{a.ibims_every}" if a.ibims_every > 1 else "")
            )
            row = {
                "ref": a.ref,
                "task": "depth",
                "tier": tier,
                "mode": pr.mode,
                "dtype": pr.dtype,
                "dataset": dsn,
                "n": n,
                "n_images": n,
                "met_absrel": round(rel, 4),
                "met_d1": round(d1, 4),
                "metrics": {"met_absrel": round(rel, 4), "met_d1": round(d1, 4)},
                "prompt_only": {"absrel": round(base[ds][0], 4), "d1": round(base[ds][1], 4)},
                "ms_per_img": round(1e3 * t / n, 2),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
                "commit": commit,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command": cmd,
            }
            print(json.dumps(row), flush=True)
            with Path(a.out).open("a") as f:
                f.write(json.dumps(row) + "\n")
        del pr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
