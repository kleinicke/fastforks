"""Absolute-pose estimation on synthetic 2D-3D sets built from MegaDepth-1500 matches: ours vs OpenCV and PoseLib.

    python tools/eval_absolute.py --matches DIR --method opencv|poselib|portable|kernels [--outliers 0.3] [--n 1000]
        [--noise 1.0] [--threshold 2.0] [--jobs N] [--stop N] --out F

Protocol (MegaDepth-1500 has no depth maps, so the 3D points are triangulated, not lifted): per pair and sampling of
`tools/eval_matching.py match` (`N_RUNS` x `N_MATCHES` matches), the matches are triangulated with the ground-truth
poses and intrinsics (camera 1 is the world frame); matches whose triangulation lies in front of both cameras and
reprojects within 1 px in both views are kept as true 3D points. The query is image 2: the kept points are projected
with the ground-truth pose, `--noise` px Gaussian noise is added, and the first `--n` of them (shuffled) form the
set; a fraction `--outliers` of them, plus the slots a short pair leaves empty, get uniform random pixel positions
instead. All methods see the same sets (seeded per item), threshold `--threshold` px, confidence 0.99999, max 1000
iterations. Error = max(rotation error, atan(|t - t_gt| / median point depth)) in degrees; AUC@5/10/20 over all
items. CPU methods run one set per call in `--jobs` processes, GPU methods in batches of `--batch` (single-set
latency measured separately). Appends one JSON line per run to `--out`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_matching import LONG_SIDE, N_RUNS, load_pairs, pose_auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

Item = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]  # p2, X, k, r, t, depth, seed
REPROJ_PX = 1.0
TINY = 1e-12


def triangulate(x: np.ndarray, k1: np.ndarray, k2: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """DLT triangulation `[N, 3]` of pixel matches `[N, 4]` in the frame of camera 1 (`P1 = K1 [I | 0]`)."""
    p1 = k1 @ np.c_[np.eye(3), np.zeros(3)]
    p2 = k2 @ np.c_[r, t]
    rows = [
        x[:, 0, None] * p1[2] - p1[0],
        x[:, 1, None] * p1[2] - p1[1],
        x[:, 2, None] * p2[2] - p2[0],
        x[:, 3, None] * p2[2] - p2[1],
    ]
    a = np.stack(rows, 1)  # [N, 4, 4]
    v = np.linalg.svd(a)[2][:, -1]
    return v[:, :3] / np.where(np.abs(v[:, 3:]) > TINY, v[:, 3:], TINY)


def project(pts: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels `[N, 2]` and depths `[N]` of world points under `x_cam = R X + t`."""
    c = pts @ r.T + t
    uv = c[:, :2] / c[:, 2:]
    return uv * np.diag(k)[:2] + k[:2, 2], c[:, 2]


def make_item(  # noqa: PLR0913, PLR0917 - the pair and the options
    x: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    seed: int,
    n: int,
    outliers: float,
    noise: float,
) -> Item:
    """One synthetic 2D-3D set (see the module docstring)."""
    rng = np.random.default_rng(seed)
    pts = triangulate(x, k1, k2, r, t)
    uv1, z1 = project(pts, k1, np.eye(3), np.zeros(3))
    uv2, z2 = project(pts, k2, r, t)
    good = (z1 > 0) & (z2 > 0) & (np.linalg.norm(uv1 - x[:, :2], axis=1) < REPROJ_PX)
    good &= np.linalg.norm(uv2 - x[:, 2:], axis=1) < REPROJ_PX
    pts, uv2 = pts[good][:n], uv2[good][:n]
    size = np.array([2 * k2[0, 2], 2 * k2[1, 2]])
    m = len(pts)
    if m < n:  # a short pair: fill with random 3D points (behind nothing in particular) and random pixels
        extra = rng.permutation(len(x))[: n - m]
        pts = np.r_[pts, triangulate(x[extra], k1, k2, r, t)]
        uv2 = np.r_[uv2, rng.uniform(0, 1, (n - m, 2)) * size]
    uv2 = uv2 + rng.normal(0, noise, uv2.shape)
    bad = rng.uniform(size=n) < outliers
    bad[m:] = True
    uv2[bad] = rng.uniform(0, 1, (bad.sum(), 2)) * size
    order = rng.permutation(n)
    depth = float(np.median(z2[good][:n])) if m else 1.0
    return uv2[order], pts[order], k2, r, t, depth, seed


def pose_error(r: np.ndarray, t: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray, depth: float) -> float:
    e_r = np.rad2deg(np.abs(np.arccos(np.clip((np.trace(r.T @ gt_r) - 1) / 2, -1.0, 1.0))))
    e_t = np.rad2deg(np.arctan(np.linalg.norm(t - gt_t) / depth))
    return float(max(e_r, e_t))


def run_opencv(item: Item, *, thr: float) -> tuple[float, float]:
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
    p2, pts, k, r, t, depth, seed = item
    cv2.setRNGSeed(seed)
    t0 = time.perf_counter()
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        pts, p2, k, None, iterationsCount=1000, reprojectionError=thr, confidence=0.99999
    )
    dt = time.perf_counter() - t0
    if not ok:
        return 90.0, dt * 1e3
    return pose_error(cv2.Rodrigues(rvec)[0], tvec[:, 0], r, t, depth), dt * 1e3


def run_poselib(item: Item, *, thr: float) -> tuple[float, float]:
    import poselib  # noqa: PLC0415

    p2, pts, k, r, t, depth, seed = item
    cam = {
        "model": "PINHOLE",
        "width": int(2 * k[0, 2]),
        "height": int(2 * k[1, 2]),
        "params": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
    }
    opt = {"max_reproj_error": thr, "success_prob": 0.99999, "max_iterations": 1000, "min_iterations": 10, "seed": seed}
    t0 = time.perf_counter()
    pose, _ = poselib.estimate_absolute_pose(p2.copy(), pts.copy(), cam, opt, {})
    dt = time.perf_counter() - t0
    return pose_error(pose.R, pose.t, r, t, depth), dt * 1e3


def gpu(items: list[Item], mode: str, batch: int, latency_items: int, *, thr: float) -> tuple[list[float], dict]:
    import torch  # noqa: PLC0415

    from fastforks.matching import estimate_absolute_pose  # noqa: PLC0415

    dev = torch.device("cuda")
    p2 = torch.from_numpy(np.stack([it[0] for it in items])).float()
    pts = torch.from_numpy(np.stack([it[1] for it in items])).float()
    k = torch.from_numpy(np.stack([it[2] for it in items])).float()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):  # warm-up (kernel load, cuBLAS/cuSOLVER handles)
        estimate_absolute_pose(p2[:batch].to(dev), pts[:batch].to(dev), k[:batch].to(dev), threshold=thr, mode=mode)
    torch.cuda.synchronize()
    errors, times = [], []
    for s in range(0, len(items), batch):
        pb, xb, kb = p2[s : s + batch].to(dev), pts[s : s + batch].to(dev), k[s : s + batch].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pose = estimate_absolute_pose(pb, xb, kb, threshold=thr, mode=mode)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        rs, ts = pose.R.cpu().numpy(), pose.t.cpu().numpy()
        errors += [pose_error(rs[j], ts[j], *items[s + j][3:6]) for j in range(len(rs))]
    peak = {
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
    }
    lat = []
    for j in range(0, min(len(items), latency_items * N_RUNS), N_RUNS):
        pb, xb, kb = p2[j].to(dev), pts[j].to(dev), k[j].to(dev)
        ts_ = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            estimate_absolute_pose(pb, xb, kb, threshold=thr, mode=mode)
            torch.cuda.synchronize()
            ts_.append(time.perf_counter() - t0)
        lat.append(min(ts_) * 1e3)
    info = {
        "ms_per_set_batched": round(sum(times) * 1e3 / len(items), 3),
        "sets_per_s": round(len(items) / sum(times), 1),
        "ms_single_set_median": round(statistics.median(lat), 2),
        "batch": batch,
        "gpu": torch.cuda.get_device_name(),
        **peak,
    }
    return errors, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/home/florian/projects/data/matching/megadepth1500"))
    ap.add_argument("--matches", type=Path, required=True)
    ap.add_argument("--method", required=True, choices=("opencv", "poselib", "portable", "kernels"))
    ap.add_argument("--n", type=int, default=1000, help="2D-3D matches per set")
    ap.add_argument("--outliers", type=float, default=0.3, help="fraction replaced by random pixels")
    ap.add_argument("--noise", type=float, default=1.0, help="pixel noise (sigma)")
    ap.add_argument("--threshold", type=float, default=2.0, help="inlier threshold in pixels")
    ap.add_argument("--jobs", type=int, default=1, help="processes for the CPU methods")
    ap.add_argument("--batch", type=int, default=500, help="sets per call for the GPU methods")
    ap.add_argument("--latency-items", type=int, default=100)
    ap.add_argument("--stop", type=int, default=None, help="only the first pairs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--errors", type=Path, default=None, help="save the per-set errors (npy)")
    a = ap.parse_args()
    from PIL import Image  # noqa: PLC0415

    pairs = load_pairs(a.root)[: a.stop]
    kpts: dict[str, np.ndarray] = {}
    for p in sorted(a.matches.glob("*.npz")):
        kpts |= {k: v for k, v in np.load(p).items() if int(k) < len(pairs)}
    items: list[Item] = []
    short = 0
    for i, (name1, name2, k1, k2, r, t) in enumerate(pairs):
        (w1, h1), (w2, h2) = (Image.open(a.root / "images" / n).size for n in (name1, name2))
        k1s, k2s = k1.copy(), k2.copy()
        k1s[:2] *= LONG_SIDE / max(w1, h1)
        k2s[:2] *= LONG_SIDE / max(w2, h2)
        for run, x in enumerate(kpts[str(i)]):
            seed = i * N_RUNS + run
            items.append(make_item(x.astype(np.float64), k1s, k2s, r, t, seed, a.n, a.outliers, a.noise))
            short += items[-1][5] == 1.0
    t0 = time.perf_counter()
    if a.method in {"portable", "kernels"}:
        errors, info = gpu(items, a.method, a.batch, a.latency_items, thr=a.threshold)
    else:
        fn = {"opencv": partial(run_opencv, thr=a.threshold), "poselib": partial(run_poselib, thr=a.threshold)}[
            a.method
        ]
        if a.jobs > 1:
            with Pool(a.jobs) as pool:
                res = pool.map(fn, items, chunksize=8)
        else:
            res = [fn(it) for it in items]
        errors = [e for e, _ in res]
        wall = time.perf_counter() - t0
        info = {
            "jobs": a.jobs,
            "ms_per_set_single": round(statistics.median([d for _, d in res]), 2),
            "sets_per_s": round(len(items) / wall, 1),
            "ms_per_set_wall": round(wall * 1e3 / len(items), 3),
            "cpu": os.cpu_count(),
        }
    auc = [round(100 * v, 2) for v in pose_auc(np.array(errors), (5, 10, 20))]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    row = {
        "method": a.method,
        "n_points": a.n,
        "outliers": a.outliers,
        "noise_px": a.noise,
        "threshold": a.threshold,
        "n_pairs": len(pairs),
        "n": len(items),
        "empty_sets": short,
        "auc5": auc[0],
        "auc10": auc[1],
        "auc20": auc[2],
        "median_err_deg": round(float(np.median(errors)), 3),
        **info,
    }
    row |= {"commit": commit, "date": time.strftime("%Y-%m-%d %H:%M"), "command": " ".join(["python", *sys.argv])}
    with a.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    if a.errors:
        np.save(a.errors, np.array(errors))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
