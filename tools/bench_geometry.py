"""Benchmark stereo rectification: OpenCV CPU vs fastforks torch CPU vs fastforks torch GPU.

Each (size, backend) runs in its own process so the peak RSS is per case. The rig is a pinhole-opencv pair with
strong distortion (k1 = -0.28), RGB uint8 images, `rectify(rig, alpha=0)` (as OpenCV with alpha 0).

    uv run python tools/bench_geometry.py --backends cv2 torch-cpu   # CPU part
    uv run python tools/bench_geometry.py --backends torch-cuda      # GPU part (run under a GPU lock)

Times: `build` = map build (`cv2.initUndistortRectifyMap` x2 to CV_16SC2, OpenCV's fastest remap format;
fastforks: `rectify()` + the first `remap`, which builds and caches the maps), `build_warm` = the same for a second
`rectify()` in the process (kernels loaded), `steady` = median remap of both images over `--runs` runs after 4
warm-ups. GPU: `steady` with the images already on the GPU; `steady_np` numpy in / numpy out, including the
transfers. fastforks also times `depth`, `point_cloud(frame="camera")` and `unrectify` of a disparity map (numpy on the
CPU, a tensor on the GPU; `unrectify_first` includes its map build). Memory: peak RSS above the process baseline
(CPU) or `max_memory_allocated` / `max_memory_reserved` (GPU).
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from fastforks._core.bench import WARMUP
from fastforks.stereo import geometry as g

if TYPE_CHECKING:
    from collections.abc import Callable

SIZES = {"480x640": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080), "12MP": (4000, 3000)}  # name: (w, h)
DIST = (np.array([-0.28, 0.09, 0.001, -0.0005, -0.01]), np.array([-0.27, 0.08, -0.0007, 0.0004, -0.012]))


def _rss_mb() -> float:
    status = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    return next(int(line.split()[1]) for line in status if line.startswith("VmRSS")) / 1024


def _rig(w: int, h: int) -> g.StereoRig:
    f = 0.65 * w
    cams = [
        g.camera("pinhole-opencv", w, h, f, f, (w - 1) / 2 + dx, (h - 1) / 2, *d)
        for d, dx in zip(DIST, (3, -4), strict=True)
    ]
    angle = np.radians(1.5)
    rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return g.StereoRig(cams[0], cams[1], rot, np.array([-0.12, 0.002, 0.001]))


def _sync() -> None:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _timed(fn: Callable[[], object]) -> float:
    _sync()
    t = time.perf_counter()
    fn()
    _sync()
    return (time.perf_counter() - t) * 1e3


def _median(fn: Callable[[], object], runs: int) -> float:
    for _ in range(WARMUP):
        fn()
    return statistics.median(_timed(fn) for _ in range(runs))


def _bench_cv2(rig: g.StereoRig, left: np.ndarray, right: np.ndarray, runs: int) -> dict:
    rect = g.rectify(rig, alpha=0)
    maps = []

    def build() -> None:
        for cam, d, r, p in zip((rig.left, rig.right), DIST, (rect.R1, rect.R2), (rect.P1, rect.P2), strict=True):
            k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
            maps.append(cv2.initUndistortRectifyMap(k, d, r, p, cam.size[::-1], cv2.CV_16SC2))

    def step() -> None:
        for img, (m1, m2) in zip((left, right), maps, strict=True):
            cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return {"build_ms": _timed(build), "steady_ms": _median(step, runs)}


def _bench_torch(rig: g.StereoRig, left: np.ndarray, right: np.ndarray, runs: int, device: str) -> dict:
    holder: dict[str, g.Rectification] = {}

    def first() -> None:
        holder["rect"] = g.rectify(rig, alpha=0)
        holder["rect"].remap(left, right, device=device)

    out = {"build_ms": _timed(first)}
    out["build_warm_ms"] = _timed(lambda: g.rectify(rig, alpha=0).remap(left, right, device=device))
    rect = holder["rect"]
    disparity = np.full(rect.size, 40.0, dtype=np.float32)
    if device == "cpu":
        out["steady_ms"] = _median(lambda: rect.remap(left, right), runs)
    else:
        out["steady_np_ms"] = _median(lambda: rect.remap(left, right, device=device), runs)
        lt, rt = (torch.from_numpy(x).to(device).permute(2, 0, 1) for x in (left, right))
        torch.cuda.reset_peak_memory_stats()
        out["steady_ms"] = _median(lambda: rect.remap(lt, rt), runs)
        out["steady_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
        disparity = torch.from_numpy(disparity).to(device)
    out["depth_ms"] = _median(lambda: rect.depth(disparity), runs)
    out["point_cloud_ms"] = _median(lambda: rect.point_cloud(disparity, frame="camera"), runs)
    depth = rect.depth(disparity)
    out["unrectify_first_ms"] = _timed(lambda: rect.unrectify(depth))
    out["unrectify_ms"] = _median(lambda: rect.unrectify(depth), runs)
    return out


def _case(size: str, backend: str, runs: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    w, h = SIZES[size]
    rng = np.random.default_rng(0)
    left, right = (rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(2))
    rig = _rig(w, h)
    if backend == "torch-cuda":
        torch.zeros(1, device="cuda").sum().item()  # CUDA context outside the timing and the memory baseline
    base = _rss_mb()
    if backend == "cv2":
        out = _bench_cv2(rig, left, right, runs)
    else:
        out = _bench_torch(rig, left, right, runs, "cuda" if backend == "torch-cuda" else "cpu")
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 - base
    if backend == "torch-cuda":
        out["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
        out["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    return {"size": size, "backend": backend, "threads": threads, **out}


def main() -> None:
    """Run every (size, backend) case in a subprocess and print one JSON line per case."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--backends", nargs="+", default=["cv2", "torch-cpu", "torch-cuda"])
    parser.add_argument("--sizes", nargs="+", default=list(SIZES))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--case", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(_case(*args.case, args.runs, args.threads)))
        return
    for size in args.sizes:
        for backend in args.backends:
            cmd = [sys.executable, __file__, "--case", size, backend, "--runs", str(args.runs)]
            result = subprocess.run([*cmd, "--threads", str(args.threads)], capture_output=True, text=True, check=True)
            print(result.stdout.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
