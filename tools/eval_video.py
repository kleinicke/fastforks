"""DAVIS 2017 val J&F of the video-tracking models (`track`, SAM 2), per tier, on the same sequences.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_video.py run --models sam2.1-hiera-tiny \
        --tiers portable,kernels --every 3 --out raw/
    python tools/eval_video.py summarize --out raw/       # J&F table and rows for docs/data

Data: `$FASTFORKS_DATA/davis2017/DAVIS` (DAVIS-2017-trainval-480p.zip), the `ImageSets/2017/val.txt` sequences
sorted, every `--every`-th (default 3: 10 of 30). Semi-supervised protocol: every object of the first annotation is
given as its frame-0 mask (`input_masks [P, H, W]`), then tracked forward; the per-object masks become one label map,
the lower object id winning where they overlap (SAM 2 `tools/vos_inference.py`). Scores as `davis2017-evaluation`:
region J (IoU) and boundary F (boundary pixels matched within 0.8 % of the image diagonal) per object, averaged
over the frames between the first and the last, then over all objects; J&F = their mean.

Tiers: fastforks `portable`, `kernels` (card default dtype). The authors' `SAM2VideoPredictor` (in the 007 venv,
`bench_upstream` adapter tree): `upstream` (fp32) and `upstream16` (their bfloat16 autocast, as `vos_inference.py`);
`init_state` loads the JPEG folder with their own resize. Its CUDA connected-components extension is not built
there, so hole filling uses scipy (8-connectivity, as the extension) on the 256 x 256 masks.

Timing: one sequence per call (fastforks: `track` on the decoded frames; original: `init_state` + `add_new_mask` +
`propagate_in_video`), synchronised, per frame; after one untimed warm-up sequence. Each sequence appends one record
to `<out>/<model>__<tier>.jsonl`; a run resumes after the last finished sequence and stops after `--budget` seconds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from _data import DATA

ROOT = DATA / "davis2017/DAVIS"
BOUND_TH = 0.008


def sequences(every: int) -> list[str]:
    names = sorted((ROOT / "ImageSets/2017/val.txt").read_text().split())
    return names[::every]


def load_sequence(name: str) -> tuple[list[Path], list[np.ndarray], np.ndarray]:
    """JPEG paths, RGB frames and the `[T, H, W]` label maps of a sequence."""
    jpgs = sorted((ROOT / "JPEGImages/480p" / name).glob("*.jpg"))
    pngs = sorted((ROOT / "Annotations/480p" / name).glob("*.png"))
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in jpgs]
    return jpgs, frames, np.stack([np.asarray(Image.open(p)) for p in pngs])


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (`davis2017-evaluation` `_seg2bmap`, same-size case)."""
    seg = seg.astype(bool)
    e, s, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1], s[:-1, :], se[:-1, :-1] = seg[:, 1:], seg[1:, :], seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def j_score(gt: np.ndarray, pred: np.ndarray) -> float:
    union = np.logical_or(gt, pred).sum()
    return 1.0 if union == 0 else float(np.logical_and(gt, pred).sum() / union)


def f_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """Boundary F-measure (`davis2017-evaluation` `f_measure`, `bound_th` 0.008)."""
    from scipy import ndimage  # noqa: PLC0415

    r = int(np.ceil(BOUND_TH * np.linalg.norm(gt.shape)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = xx**2 + yy**2 <= r**2
    fb, gb = seg2bmap(pred), seg2bmap(gt)
    n_f, n_g = int(fb.sum()), int(gb.sum())
    if n_f == 0 or n_g == 0:
        return 1.0 if n_f == n_g else 0.0
    precision = (fb & ndimage.binary_dilation(gb, disk)).sum() / n_f
    recall = (gb & ndimage.binary_dilation(fb, disk)).sum() / n_g
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def label_map(masks: np.ndarray, ids: list[int]) -> np.ndarray:
    """`[P, H, W]` object masks -> one label map; the lower id wins (SAM 2 `vos_inference.py`)."""
    out = np.zeros(masks.shape[1:], np.uint8)
    for k in reversed(range(len(ids))):
        out[masks[k]] = ids[k]
    return out


def scores(gt: np.ndarray, pred: np.ndarray, ids: list[int]) -> tuple[list[float], list[float]]:
    """Per object: J and F averaged over the frames between the first and the last."""
    js, fs = [], []
    for i in ids:
        frames = range(1, gt.shape[0] - 1)
        js.append(float(np.mean([j_score(gt[t] == i, pred[t] == i) for t in frames])))
        fs.append(float(np.mean([f_score(gt[t] == i, pred[t] == i) for t in frames])))
    return js, fs


class Ours:
    """A fastforks tier: `track` with every object's frame-0 mask."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        import fastforks.promptable_segmentation as seg  # noqa: PLC0415

        self.model = seg.load(ref, mode=tier, device="cuda")
        self.backend = self.model.backend

    def __call__(self, jpgs: list[Path], frames: list[np.ndarray], first: np.ndarray, ids: list[int]) -> np.ndarray:
        """`[T, P, H, W]` bool masks of the objects `ids` (labels of `first`)."""
        _ = jpgs
        r = self.model.track(frames, input_masks=np.stack([first == i for i in ids]).astype(np.uint8))
        return r.masks.cpu().numpy()  # [T, P, H, W]


class Original:
    """The authors' `SAM2VideoPredictor` (see the module docstring)."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        from scipy import ndimage  # noqa: PLC0415

        import _original  # noqa: PLC0415
        import _original_coco  # noqa: PLC0415

        bu = _original.bench()
        a = bu.FAM["sam2"]
        a.setup()
        from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415  # ty: ignore[unresolved-import]
        from sam2.utils import misc  # noqa: PLC0415  # ty: ignore[unresolved-import]

        def components(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # the extension's outputs
            m = mask.cpu().numpy()[:, 0].astype(bool)
            labels, areas = np.zeros(m.shape, np.int32), np.zeros(m.shape, np.int32)
            for b in range(m.shape[0]):
                lab, _ = ndimage.label(m[b], np.ones((3, 3), bool))
                labels[b] = lab
                areas[b] = np.bincount(lab.ravel())[lab] * (lab > 0)
            return tuple(torch.from_numpy(x)[:, None].to(mask.device) for x in (labels, areas))

        misc.get_connected_components = components
        size = ref.removeprefix("sam2.1-hiera-")
        cfg = f"configs/sam2.1/sam2.1_hiera_{a.CFG[size]}.yaml"
        ckpt = str(bu.wfile(f"sam2.1_hiera_{size.replace('-', '_')}.pt"))
        self.pred = build_sam2_video_predictor(cfg, ckpt, device="cuda")
        self.t = _original_coco._Tier(tier, "sam2", [self.pred], "bfloat16")  # noqa: SLF001
        self.backend = self.t.backend

    def __call__(self, jpgs: list[Path], frames: list[np.ndarray], first: np.ndarray, ids: list[int]) -> np.ndarray:
        """`[T, P, H, W]` bool masks of the objects `ids` (labels of `first`)."""
        out = np.zeros((len(frames), len(ids), *first.shape), bool)
        with self.t.ctx(), torch.inference_mode():
            state = self.pred.init_state(video_path=str(jpgs[0].parent))
            for i in ids:
                self.pred.add_new_mask(state, 0, i, first == i)
            for t, obj_ids, masks in self.pred.propagate_in_video(state):
                for k, i in enumerate(obj_ids):
                    out[t, ids.index(i)] = (masks[k, 0] > 0).cpu().numpy()
        return out


def _f(out: Path, ref: str, tier: str) -> Path:
    return out / f"{ref}__{tier}.jsonl"


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _run(a: argparse.Namespace, ref: str, tier: str) -> bool:
    """Process sequences of one model and tier until the budget is spent; True if all are done."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = _f(out, ref, tier)
    done = {r["seq"] for r in _records(path)}
    todo = [s for s in sequences(a.every) if s not in done]
    if not todo:
        return True
    deadline = time.perf_counter() + a.budget
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = (Original if tier.startswith("upstream") else Ours)(ref, tier)
    build_s = round(time.perf_counter() - t0, 1)
    be = model.backend
    common = {
        "model": ref,
        "tier": tier,
        "mode": "upstream" if tier.startswith("upstream") else be.mode,
        "dtype": str(be.dtype).removeprefix("torch."),
        "how": getattr(be, "how", None),
        "backend": f"{be}",
        "build_s": build_s,
        "every": a.every,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "commit": subprocess.run(
            ["git", "-C", str(Path(__file__).parent.parent), "log", "-1", "--format=%h", "--", "src"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "command": " ".join(["python", *sys.argv]),
    }
    warm = load_sequence(todo[0])  # untimed warm-up: the first sequence
    model(warm[0], warm[1], warm[2][0], [int(i) for i in np.unique(warm[2][0]) if i])
    for name in todo:
        if time.perf_counter() > deadline:
            return False
        jpgs, frames, gt = load_sequence(name)
        ids = [int(i) for i in np.unique(gt[0]) if i]
        torch.cuda.synchronize()
        t = time.perf_counter()
        masks = model(jpgs, frames, gt[0], ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000 / len(frames)
        js, fs = scores(gt, np.stack([label_map(m, ids) for m in masks]), ids)
        rec = common | {
            "seq": name,
            "frames": len(frames),
            "objects": len(ids),
            "j": js,
            "f": fs,
            "ms_frame": round(ms, 3),
            "alloc_mb": torch.cuda.max_memory_allocated() >> 20,
            "res_mb": torch.cuda.max_memory_reserved() >> 20,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("model", "tier", "seq", "frames", "ms_frame")}), flush=True)
    return True


def _summarize(a: argparse.Namespace) -> None:
    for path in sorted(Path(a.out).glob("*.jsonl")):
        recs = _records(path)
        if not recs:
            continue
        last = recs[-1]
        every = last["every"]
        if {r["seq"] for r in recs} != set(sequences(every)):
            print(f"# incomplete: {path.name}")
            continue
        j = [x for r in recs for x in r["j"]]
        f = [x for r in recs for x in r["f"]]
        frames = sum(r["frames"] for r in recs)
        row = {
            "ref": last["model"],
            "task": "promptable_segmentation",
            "tier": last["tier"],
            "mode": "upstream-fp16" if last["tier"] == "upstream16" else last["mode"],
            "dtype": last["dtype"],
            "how": last["how"],
            "dataset": f"davis2017-val-every{every}",
            "n": len(j),
            "metrics": {
                "jf": round(50 * (statistics.mean(j) + statistics.mean(f)), 2),
                "j": round(100 * statistics.mean(j), 2),
                "f": round(100 * statistics.mean(f), 2),
            },
            "ms_frame": round(sum(r["ms_frame"] * r["frames"] for r in recs) / frames, 2),
            "frames": frames,
            "peak_alloc_mb": max(r["alloc_mb"] for r in recs),
            "peak_reserved_mb": max(r["res_mb"] for r in recs),
        } | {k: last[k] for k in ("backend", "build_s", "commit", "gpu", "torch", "date", "command")}
        print(json.dumps(row))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--tiers", default="portable,kernels")
    r.add_argument("--every", type=int, default=3)
    r.add_argument("--budget", type=float, default=220, help="seconds per invocation (one GPU lock)")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "summarize":
        _summarize(a)
        return
    finished = all([_run(a, m, t) for m in a.models for t in a.tiers.split(",")])  # noqa: C419 - run every pair
    sys.exit(0 if finished else 3)


if __name__ == "__main__":
    main()
