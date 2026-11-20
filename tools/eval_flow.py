"""Real-world optical-flow accuracy of fastforks models on KITTI 2015 train (200 pairs, flow_occ ground truth).

    python tools/eval_flow.py --refs raft-large,raft-small --modes portable,kernels --out docs/data/flow_eval.jsonl

Models come from `fastforks.flow.load(ref, mode=...)` and run on the full 375x1242-ish pair (frames _10 -> _11 of
image_2, no cropping or resizing). The ground truth is the 16-bit flow_occ PNG: u, v = (value - 2^15) / 64, valid where
the third channel is > 0. EPE = mean over pairs of the per-pair mean end-point error over the valid pixels; Fl-all = the
share of all valid pixels of all pairs (pooled) whose error is > 3 px and > 5 % of the GT magnitude, as in the RAFT and
SEA-RAFT evaluation code. ms is the median over the pairs after one untimed warm-up call (upload and download
included); MiB is the peak allocated / reserved memory of the calls (model load excluded). One row per (ref, mode),
in the format of the other docs/data/*_eval.jsonl files. The modes `upstream` / `upstream16` run the original code
(`tools/_original.py`: fp32 / its fp16 tier) on the same pairs.
`portable32` runs the portable model in float32 and records a distinct precision tier.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import closing
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import cv2
import numpy as np
import torch
from PIL import Image

from _data import DATA
from _prefetch import prefetch

if TYPE_CHECKING:
    from collections.abc import Iterable

DATASET = "kitti2015-train"
OUTLIER_PX, OUTLIER_REL = 3, 0.05  # KITTI Fl: error > 3 px and > 5 % of the ground-truth magnitude
FlowPair: TypeAlias = tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def read_kitti_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """KITTI 16-bit flow PNG -> (flow [H, W, 2] float32 in px, valid [H, W] bool)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # PIL cannot decode 16-bit 3-channel PNGs
    if bgr is None or bgr.dtype != np.uint16 or bgr.shape[-1:] != (3,):
        msg = f"{path}: not a 16-bit 3-channel KITTI flow PNG"
        raise ValueError(msg)
    raw = bgr[..., ::-1]  # cv2 order is B, G, R -> R = u, G = v, B = valid
    flow = (raw[..., :2].astype(np.float32) - 2**15) / 64
    return flow, raw[..., 2] > 0


def _pair_ids(root: Path, stop: int | None, start: int) -> list[str]:
    ids = sorted(p.name[:6] for p in (root / "flow_occ").glob("*_10.png"))[start:stop]
    if not ids:
        msg = f"no KITTI 2015 flow_occ files under {root}"
        raise FileNotFoundError(msg)
    return ids


def _read_pair(root: Path, pair_id: str) -> FlowPair:
    with (
        Image.open(root / "image_2" / f"{pair_id}_10.png") as a,
        Image.open(root / "image_2" / f"{pair_id}_11.png") as b,
    ):
        first, second = np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))
    return pair_id, first, second, *read_kitti_flow(root / "flow_occ" / f"{pair_id}_10.png")


def load_pairs(root: Path, stop: int | None, start: int = 0) -> list[FlowPair]:
    """Load a bounded slice for callers that need the decoded pairs together."""
    return [_read_pair(root, pair_id) for pair_id in _pair_ids(root, stop, start)]


ORIGINAL = {"upstream": ("upstream", "original fp32"), "upstream16": ("upstream-fp16", "original fp16")}


def _original(ref: str, tier: str):  # noqa: ANN202 - (call(a, b) -> flow [2, H, W], dtype, how, note)
    import _original as og  # noqa: PLC0415 - needs the reference checkout (FF_*_REPO)

    o = og.load("flow", ref, tier)

    def call(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        return o(*o.prep(tuple(torch.from_numpy(x).cuda().permute(2, 0, 1) for x in (a, b))))[0]

    return call, o.dtype, o.how, o.note


def evaluate(ref: str, mode: str, pairs: Iterable[FlowPair], base: dict) -> dict:
    import fastforks.flow as ff  # noqa: PLC0415 - after argparse, so --help is fast

    remaining = iter(pairs)
    first = next(remaining)
    if mode in ORIGINAL:
        call, dtype, how, note = _original(ref, mode)
        extra = {"note": note}
    else:
        backend_mode = "portable" if mode == "portable32" else mode
        model = ff.load(ref, mode=backend_mode, **({"dtype": "float32"} if mode == "portable32" else {}))
        dtype, how, call = str(model.backend.dtype), None, lambda a, b, m=model: m(a, b).flow
        extra = {"backend": {"mode": backend_mode, "dtype": dtype, "kernels": list(model.backend.kernels)}}
        del model
    call(first[1], first[2])  # warm-up (untimed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    epes, ms, n_out, n_val = [], [], 0, 0
    pair_stats = []
    for pair_id, a, b, gt, valid in chain((first,), remaining):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = call(a, b)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
        f = res.float().cpu().numpy().transpose(1, 2, 0)
        err = np.linalg.norm(f[valid] - gt[valid], axis=-1)
        mag = np.linalg.norm(gt[valid], axis=-1)
        epe = float(err.mean())
        outliers = int(((err > OUTLIER_PX) & (err > OUTLIER_REL * mag)).sum())
        valid_count = int(valid.sum())
        epes.append(epe)
        n_out += outliers
        n_val += valid_count
        pair_stats.append({"id": pair_id, "epe": epe, "outliers": outliers, "valid": valid_count, "ms": ms[-1]})
    metrics = {"epe": round(float(np.mean(epes)), 4), "fl_all": round(100 * n_out / n_val, 3)}
    docs_mode, column = ORIGINAL.get(mode, ("portable", "portable32") if mode == "portable32" else (mode, mode))
    row = {"ref": ref, "task": "flow", "tier": mode, "mode": docs_mode, "dtype": dtype, "how": how, "dataset": DATASET}
    row |= {"n": len(pair_stats), **metrics, "metrics": metrics, "ms_per_img": round(statistics.median(ms), 2)}
    row |= {"peak_alloc_mb": torch.cuda.max_memory_allocated() >> 20}
    row |= {"peak_reserved_mb": torch.cuda.max_memory_reserved() >> 20}
    del call
    torch.cuda.empty_cache()
    return {**row, **extra, **base, "column": column, "pair_stats": pair_stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refs", required=True, help="comma-separated model ids")
    ap.add_argument(
        "--modes",
        default="portable,kernels",
        help="comma-separated: portable32, portable, kernels, upstream, upstream16 (original code)",
    )
    ap.add_argument("--root", type=Path, default=DATA / "stereo/KITTI/training", help="KITTI 2015 training dir")
    ap.add_argument("--stop", type=int, default=None, help="first N pairs only (smoke runs)")
    ap.add_argument("--start", type=int, default=0, help="first pair index (inclusive), for bounded measurement chunks")
    ap.add_argument("--commit", required=True, help="git commit of the measured src (the run dir may lack .git)")
    ap.add_argument("--source-id", help="immutable identifier of the measured source, including any local changes")
    ap.add_argument("--out", required=True, help="jsonl to append to")
    a = ap.parse_args()
    if a.start < 0 or (a.stop is not None and a.stop <= a.start):
        ap.error("--start must be nonnegative and less than --stop")
    ids = _pair_ids(a.root, a.stop, a.start)
    base = {"commit": a.commit, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    if a.source_id:
        base["source_id"] = a.source_id
    base |= {"date": time.strftime("%Y-%m-%d %H:%M"), "command": " ".join(["python", *sys.argv])}
    for ref in a.refs.split(","):
        for mode in a.modes.split(","):
            with closing(prefetch(partial(_read_pair, a.root), ids, workers=2)) as pairs:
                row = evaluate(ref, mode, pairs, base)
            with Path(a.out).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(ref, mode, row["metrics"], row["ms_per_img"], row["peak_alloc_mb"], flush=True)


if __name__ == "__main__":
    main()
