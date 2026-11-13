"""TAP-Vid DAVIS (query mode "first") of the point trackers, per tier, on the same sequences.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_tracking.py run --models tapnext \
        --tiers portable,kernels --every 3 --out raw/
    TAPNET_SRC=... python tools/eval_tracking.py run --models tapnext --tiers upstream,upstream16 --out raw/
    python tools/eval_tracking.py summarize --out raw/    # AJ / delta_avg / OA and rows for docs/data

Data: `$FASTFORKS_DATA/tapvid/tapvid_davis/npz/<name>.npz` (the 30 TAP-Vid DAVIS sequences as `video [T, H, W, 3]`
uint8, `points [N, T, 2]` normalised `(x, y)`, `occluded [N, T]`), sorted, every `--every`-th (default 3: 10 of 30).
Protocol as tapnet `evaluation_datasets.py`: frames resized to 256 x 256 (PIL Lanczos, as their `resize_video`),
every track with a visible frame queried at its first visible point (`sample_queries_first`), metrics of
`compute_tapvid_metrics(query_mode="first")` on the frames after the query frame, in 256 x 256 raster coordinates:
occlusion accuracy (OA), the share of visible points within 1/2/4/8/16 px (delta_avg) and the Jaccard at those
thresholds (AJ); a point counts as visible when its visibility is > 0.5 (TAPNext: logit > 0). Per video, then the
mean over videos (the benchmark's averaging).

Tiers: fastforks `portable`, `kernels` (card default dtype) on the 256 x 256 frames, queries in their pixel
coordinates (normalised x 256 - 0.5). The authors' PyTorch models (`$TAPNET_SRC`, a tapnet checkout with einops,
einshape and dm-tree on the path): `upstream` (fp32) and `upstream16` (float16 autocast), queries `(t, y, x)`
continuous. TAPNext online as their evaluation (the first frame with the queries, then one frame at a time with the
state); TAPIR / BootsTAPIR offline (the whole video at once; visible where `(1 - sigmoid(occlusion)) * (1 -
sigmoid(expected_dist)) > 0.5`). AllTracker (both tiers and the original) gets the original frames, queries in
their pixels, and resizes them itself to 384 x 512 (upstream's evaluation resizes the originals with cv2 bilinear);
the predictions are scaled to 256 x 256 for the metrics. Its original (`$ALLTRACKER_SRC`, an aharley/alltracker
checkout with einops) runs through `tools/golden/alltracker.py` `track`: forward + backward passes per query frame,
queries sampled bilinearly (upstream's evaluation runs forward only from frame 0 with queries rounded to pixels).

Timing: one sequence per call, synchronised, per frame; after one untimed warm-up sequence. Each sequence appends one
record to `<out>/<model>__<tier>.jsonl`; a run resumes after the last finished sequence and stops after `--budget`
seconds (exit code 3).
"""

from __future__ import annotations

import argparse
import json
import os
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

ROOT = DATA / "tapvid/tapvid_davis/npz"
SIZE = 256
THRESHOLDS = (1, 2, 4, 8, 16)
WEIGHT = {
    "tapnext": "bootstapnext_ckpt.npz",
    "tapnext-pp": "tapnextpp_ckpt.pt",
    "bootstapir": "bootstapir_checkpoint_v2.pt",
    "tapir": "tapir_checkpoint_panning.pt",
}
TAPIR_REFS = ("bootstapir", "tapir")
NATIVE_REFS = ("alltracker", "alltracker-tiny")  # fed the original frames (they resize to their own 384 x 512)


def sequences(every: int) -> list[str]:
    return sorted(p.stem for p in ROOT.glob("*.npz"))[::every]


def load_sequence(name: str, *, native: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frames, queries `[Q, 3]` `(t, x, y)` in their pixels, target points `[Q, T, 2]` (256 x 256) and occlusion.

    The frames are 256 x 256, or with `native` the original ones (the queries then in original pixels).
    """
    with np.load(ROOT / f"{name}.npz", allow_pickle=False) as z:
        video, points, occluded = z["video"], z["points"], z["occluded"]
    if native:
        frames = video
    else:
        frames = np.stack(
            [np.asarray(Image.fromarray(f).resize((SIZE, SIZE), Image.Resampling.LANCZOS)) for f in video]
        )
    keep = (~occluded).any(1)
    points, occluded = points[keep], occluded[keep]
    first = np.argmax(~occluded, 1)
    xy = points[np.arange(len(first)), first] * np.array(frames.shape[2:0:-1], dtype=np.float32)
    queries = np.concatenate([first[:, None], xy - 0.5], 1).astype(np.float32)
    return frames, queries, points * SIZE, occluded  # targets in continuous 256 x 256 raster coordinates


def metrics(first: np.ndarray, occ: np.ndarray, pts: np.ndarray, pred_occ: np.ndarray, pred: np.ndarray) -> dict:
    """`compute_tapvid_metrics(query_mode="first")` of one video (tracks `[Q, T, 2]`, occlusion `[Q, T]`)."""
    t = np.arange(occ.shape[1])
    ev = t[None] > first[:, None]
    vis, pvis = ~occ, ~pred_occ
    out = {"oa": float(((pred_occ == occ) & ev).sum() / ev.sum())}
    dist = ((pred - pts) ** 2).sum(-1)
    within, jac = [], []
    for th in THRESHOLDS:
        near = dist < th**2
        correct = near & vis
        within.append((correct & ev).sum() / (vis & ev).sum())
        fp = ((~vis & pvis) | (~near & pvis)) & ev
        jac.append((correct & pvis & ev).sum() / ((vis & ev).sum() + fp.sum()))
    out["delta_avg"] = float(np.mean(within))
    out["aj"] = float(np.mean(jac))
    return out


class Ours:
    """A fastforks tier."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        from fastforks import tracking  # noqa: PLC0415

        self.model = tracking.load(ref, mode=tier, device="cuda")
        self.backend = self.model.backend

    def __call__(self, frames: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tracks `[Q, T, 2]` in continuous coordinates and the predicted occlusion `[Q, T]`."""
        r = self.model(list(frames), queries)
        return r.tracks.cpu().numpy().transpose(1, 0, 2) + 0.5, ~r.visible.cpu().numpy().T


class OriginalAllTracker:
    """The authors' AllTracker through `tools/golden/alltracker.py` `track` (`$ALLTRACKER_SRC`, unpatched, CUDA)."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        sys.path.insert(0, str(Path(__file__).parent / "golden"))
        import alltracker as g  # noqa: PLC0415

        file, kwargs = g.MODELS[f"{ref}@1"]
        self.g = g
        weights = Path(os.environ["FASTFORKS_WEIGHTS_PATH"]) / "alltracker" / file
        self.model = g.load(Path(os.environ["ALLTRACKER_SRC"]), weights, kwargs).cuda()
        self.backend = _Upstream(tier)
        self.backend.name = "AllTracker"

    def __call__(self, frames: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tracks `[Q, T, 2]` in continuous coordinates and the predicted occlusion `[Q, T]`."""
        video = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255
        amp = torch.float16 if self.backend.dtype == torch.float16 else None
        with torch.inference_mode():
            tracks, vis = self.g.track(self.model, video, torch.from_numpy(queries).cuda(), amp=amp)
        return tracks.transpose(0, 1).cpu().numpy() + 0.5, (vis.T <= 0.5).cpu().numpy()  # noqa: PLR2004


class _Upstream:
    def __init__(self, tier: str) -> None:
        self.mode, self.dtype, self.how = "upstream", torch.float16 if tier == "upstream16" else torch.float32, None
        self.name = "TAPNext"

    def __str__(self) -> str:
        return f"upstream {self.name} (PyTorch, {'float16 autocast' if self.dtype == torch.float16 else 'float32'})"


class OriginalTapir:
    """The authors' PyTorch TAPIR / BootsTAPIR, offline (see the module docstring)."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        sys.path.insert(0, os.environ["TAPNET_SRC"])
        from tapnet.torch.tapir_model import TAPIR  # noqa: PLC0415  # ty: ignore[unresolved-import]

        kwargs = {"pyramid_level": 1} if ref == "bootstapir" else {"pyramid_level": 0, "extra_convs": False}
        model = TAPIR(**kwargs)
        file = Path(os.environ["FASTFORKS_WEIGHTS_PATH"]) / "tapir" / WEIGHT[ref]
        model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
        self.model = model.eval().cuda()
        self.backend = _Upstream(tier)
        self.backend.name = "BootsTAPIR" if ref == "bootstapir" else "TAPIR"

    def __call__(self, frames: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tracks `[Q, T, 2]` in continuous coordinates and the predicted occlusion `[Q, T]`."""
        video = torch.from_numpy(frames).cuda().float()[None] / 255 * 2 - 1  # [1, T, 256, 256, 3]
        q = torch.from_numpy(queries).cuda()
        qp = torch.cat([q[:, :1], (q[:, 1:] + 0.5).flip(-1)], 1)[None]  # (t, y, x) continuous
        amp = self.backend.dtype == torch.float16
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=amp):
            out = self.model(video, qp)
        vis = (1 - out["occlusion"].float().sigmoid()) * (1 - out["expected_dist"].float().sigmoid())
        return out["tracks"][0].float().cpu().numpy(), (vis[0] <= 0.5).cpu().numpy()  # noqa: PLR2004


class Original:
    """The authors' PyTorch TAPNext, online (see the module docstring)."""

    def __init__(self, ref: str, tier: str) -> None:  # noqa: D107
        sys.path.insert(0, os.environ["TAPNET_SRC"])
        from tapnet.tapnext.tapnext_torch import TAPNext  # noqa: PLC0415  # ty: ignore[unresolved-import]
        from tapnet.tapnext.tapnext_torch_utils import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
            restore_model_from_jax_checkpoint,
        )

        file = Path(os.environ["FASTFORKS_WEIGHTS_PATH"]) / "tapnext" / WEIGHT[ref]
        model = TAPNext(image_size=(SIZE, SIZE))
        if file.suffix == ".npz":
            model = restore_model_from_jax_checkpoint(model, str(file))
        else:
            sd = torch.load(file, map_location="cpu", weights_only=True)["state_dict"]
            model.load_state_dict({k.removeprefix("tapnext."): v for k, v in sd.items()})
        self.model = model.eval().cuda()
        self.backend = _Upstream(tier)

    def __call__(self, frames: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tracks `[Q, T, 2]` in continuous coordinates and the predicted occlusion `[Q, T]`."""
        video = torch.from_numpy(frames).cuda().float()[None] / 255 * 2 - 1  # [1, T, 256, 256, 3]
        q = torch.from_numpy(queries).cuda()
        qp = torch.cat([q[:, :1], (q[:, 1:] + 0.5).flip(-1)], 1)[None]  # (t, y, x) continuous
        amp = self.backend.dtype == torch.float16
        tracks, vis = [], []
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=amp):
            tr, _, vl, state = self.model(video=video[:, :1], query_points=qp)
            tracks.append(tr.float()), vis.append(vl.float())
            for t in range(1, video.shape[1]):
                tr, _, vl, state = self.model(video=video[:, t : t + 1], state=state)
                tracks.append(tr.float()), vis.append(vl.float())
        yx = torch.cat(tracks, 1)[0]  # [T, Q, 2]
        visible = torch.cat(vis, 1)[0, ..., 0] > 0
        return yx.flip(-1).transpose(0, 1).cpu().numpy(), ~visible.T.cpu().numpy()


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
    orig = OriginalTapir if ref in TAPIR_REFS else OriginalAllTracker if ref in NATIVE_REFS else Original
    model = (orig if tier.startswith("upstream") else Ours)(ref, tier)
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
    native = ref in NATIVE_REFS
    warm = load_sequence(todo[0], native=native)  # untimed warm-up: the first sequence
    model(warm[0], warm[1])
    for name in todo:
        if time.perf_counter() > deadline:
            return False
        frames, queries, pts, occ = load_sequence(name, native=native)
        torch.cuda.synchronize()
        t = time.perf_counter()
        pred, pred_occ = model(frames, queries)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000 / len(frames)
        pred = pred * (SIZE / np.array(frames.shape[2:0:-1]))  # to 256 x 256 (a no-op on 256 x 256 frames)
        rec = common | {
            "seq": name,
            "frames": len(frames),
            "points": len(queries),
            **metrics(queries[:, 0].astype(int), occ, pts, pred_occ, pred),
            "ms_frame": round(ms, 3),
            "alloc_mb": torch.cuda.max_memory_allocated() >> 20,
            "res_mb": torch.cuda.max_memory_reserved() >> 20,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("model", "tier", "seq", "frames", "aj", "ms_frame")}), flush=True)
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
        frames = sum(r["frames"] for r in recs)
        row = {
            "ref": last["model"],
            "task": "tracking",
            "tier": last["tier"],
            "mode": "upstream-fp16" if last["tier"] == "upstream16" else last["mode"],
            "dtype": last["dtype"],
            "how": last["how"],
            "dataset": "tapvid-davis-first" + (f"-every{every}" if every > 1 else ""),
            "n": len(recs),
            "metrics": {k: round(100 * statistics.mean(r[k] for r in recs), 2) for k in ("aj", "delta_avg", "oa")},
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
