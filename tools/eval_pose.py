"""Relative-pose estimation from saved matches: `fastforks.matching.estimate_pose` against OpenCV, USAC and PoseLib.

    python tools/eval_pose.py --matches DIR --method opencv|usac|poselib|portable|kernels [--jobs N] [--stop N] --out F
        [--problem essential|fundamental] [--threshold PX]

`--problem fundamental` estimates the fundamental matrix without the intrinsics (7-point RANSAC, pixel threshold,
default 1 px), then converts it with the intrinsics (`E = K2^T F K1`) and recovers the pose from it, so the same
pose AUC measures it.

`DIR` holds the `*.npz` of `tools/eval_matching.py match` (`N_RUNS` samplings of `N_MATCHES` matches per
MegaDepth-1500 pair). Protocol as `eval_matching.py score`: threshold 0.5 px over the mean focal length, confidence
0.99999, pose error = max(rotation, translation angle), AUC@5/10/20 over all pairs x samplings. CPU methods run one
pair per call in `--jobs` processes (single-threaded each: OpenCV's and PoseLib's RANSAC are), GPU methods in
batches of `--batch` pairs; the single-pair latency of the GPU methods is measured separately (median over pairs).
Appends one JSON line per run to `--out`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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

Item = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]  # x, k1, k2, r, t, seed


def angle_error(r: np.ndarray, t: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray) -> float:
    cos_t = np.dot(t, gt_t) / max(np.linalg.norm(t) * np.linalg.norm(gt_t), 1e-12)
    e_t = np.rad2deg(np.arccos(np.clip(cos_t, -1.0, 1.0)))
    e_r = np.rad2deg(np.abs(np.arccos(np.clip((np.trace(r.T @ gt_r) - 1) / 2, -1.0, 1.0))))
    return float(max(min(e_t, 180 - e_t), e_r))


def normalise(x: np.ndarray, k1: np.ndarray, k2: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, float]:
    thr = thr / (np.mean(np.abs(k1[:2, :2])) + np.mean(np.abs(k2[:2, :2])))
    p1 = (np.linalg.inv(k1[:2, :2]) @ (x[:, :2] - k1[None, :2, 2]).T).T
    p2 = (np.linalg.inv(k2[:2, :2]) @ (x[:, 2:] - k2[None, :2, 2]).T).T
    return p1, p2, thr


def cv_pose(
    e: np.ndarray | None, mask: np.ndarray | None, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2  # noqa: PLC0415

    if e is None:
        return None
    best, ret = 0, None
    for ei in np.split(e, len(e) // 3):
        n, r, t, _ = cv2.recoverPose(ei, p1, p2, np.eye(3), 1e9, mask=mask)
        if n > best:
            best, ret = n, (r, t[:, 0])
    return ret


def run_opencv(item: Item, *, usac: bool = False, fundamental: bool = False, thr: float = 0.5) -> tuple[float, float]:
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
    x, k1, k2, r, t, seed = item
    p1, p2, thr_n = normalise(x, k1, k2, thr)
    cv2.setRNGSeed(seed)
    method = cv2.USAC_MAGSAC if usac else cv2.RANSAC
    t0 = time.perf_counter()
    if fundamental:
        f, mask = cv2.findFundamentalMat(x[:, :2], x[:, 2:], method, thr, 0.99999)
        e = None if f is None else np.concatenate([k2.T @ fi @ k1 for fi in np.split(f, len(f) // 3)])
    else:
        e, mask = cv2.findEssentialMat(p1, p2, np.eye(3), method=method, threshold=thr_n, prob=0.99999)
    ret = cv_pose(e, mask, p1, p2)
    dt = time.perf_counter() - t0
    return (90.0 if ret is None else angle_error(*ret, r, t)), dt * 1e3


def run_poselib(item: Item, *, fundamental: bool = False, thr: float = 0.5) -> tuple[float, float]:
    import poselib  # noqa: PLC0415

    x, k1, k2, r, t, seed = item
    if fundamental:
        opt = {"max_epipolar_error": thr, "success_prob": 0.99999, "max_iterations": 1000, "min_iterations": 10}
        t0 = time.perf_counter()
        f, info = poselib.estimate_fundamental(x[:, :2].copy(), x[:, 2:].copy(), {**opt, "seed": seed}, {})
        p1, p2, _ = normalise(x, k1, k2, thr)
        ret = cv_pose(k2.T @ f @ k1, np.array(info["inliers"], dtype=np.uint8), p1, p2)
        dt = time.perf_counter() - t0
        return (90.0 if ret is None else angle_error(*ret, r, t)), dt * 1e3
    cams = [
        {
            "model": "PINHOLE",
            "width": int(2 * k[0, 2]),
            "height": int(2 * k[1, 2]),
            "params": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
        }
        for k in (k1, k2)
    ]
    opt = {
        "max_epipolar_error": thr,
        "success_prob": 0.99999,
        "max_iterations": 1000,
        "min_iterations": 10,
        "seed": seed,
    }
    t0 = time.perf_counter()
    pose, _ = poselib.estimate_relative_pose(x[:, :2].copy(), x[:, 2:].copy(), cams[0], cams[1], opt, {})
    dt = time.perf_counter() - t0
    return angle_error(pose.R, pose.t, r, t), dt * 1e3


def gpu(  # noqa: PLR0913 - the options
    items: list[Item], mode: str, batch: int, latency_pairs: int, *, fundamental: bool = False, thr: float = 0.5
) -> tuple[list[float], dict]:
    import torch  # noqa: PLC0415

    from fastforks.matching import estimate_fundamental, relative_pose  # noqa: PLC0415

    def estimate_pose(p1, p2, k1, k2, *, mode):  # noqa: ANN001, ANN202 - the pose, through F when `fundamental`
        if not fundamental:
            return relative_pose.estimate_pose(p1, p2, k1, k2, threshold=thr, mode=mode)
        fund = estimate_fundamental(p1, p2, threshold=thr, mode=mode)
        if p1.ndim == 2:  # noqa: PLR2004 - the private helpers take batches
            p1, p2, k1, k2 = p1[None], p2[None], k1[None], k2[None]
            fund = relative_pose.FundamentalMatrix(fund.F[None], fund.inliers[None], fund.ok[None])
        e = k2.transpose(-1, -2) @ fund.F @ k1
        x1, x2 = relative_pose._normalise(p1, k1), relative_pose._normalise(p2, k2)  # noqa: SLF001
        r, t, _ = relative_pose._recover_pose(e, x1, x2, fund.inliers)  # noqa: SLF001
        return relative_pose.RelativePose(r, t, e, fund.inliers, fund.ok)

    dev = torch.device("cuda")
    x = torch.from_numpy(np.stack([it[0] for it in items])).float()
    k1 = torch.from_numpy(np.stack([it[1] for it in items])).float()
    k2 = torch.from_numpy(np.stack([it[2] for it in items])).float()
    torch.cuda.reset_peak_memory_stats()
    # Warm-up (kernel load, cuBLAS/cuSOLVER handles).
    for _ in range(2):
        estimate_pose(
            x[:batch, :, :2].to(dev), x[:batch, :, 2:].to(dev), k1[:batch].to(dev), k2[:batch].to(dev), mode=mode
        )
    torch.cuda.synchronize()
    errors, times = [], []
    for s in range(0, len(items), batch):
        xb, kb1, kb2 = x[s : s + batch].to(dev), k1[s : s + batch].to(dev), k2[s : s + batch].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pose = estimate_pose(xb[..., :2], xb[..., 2:], kb1, kb2, mode=mode)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        rs, ts = pose.R.cpu().numpy(), pose.t.cpu().numpy()
        errors += [angle_error(rs[j], ts[j], items[s + j][3], items[s + j][4]) for j in range(len(rs))]
    peak = {
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
    }
    # Single-pair latency: median over the first pairs (every N_RUNS-th: different pairs), each 3 times.
    lat = []
    for j in range(0, min(len(items), latency_pairs * N_RUNS), N_RUNS):
        xb, kb1, kb2 = x[j].to(dev), k1[j].to(dev), k2[j].to(dev)
        ts_ = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            estimate_pose(xb[:, :2], xb[:, 2:], kb1, kb2, mode=mode)
            torch.cuda.synchronize()
            ts_.append(time.perf_counter() - t0)
        lat.append(min(ts_) * 1e3)
    info = {
        "ms_per_pair_batched": round(sum(times) * 1e3 / len(items), 3),
        "pairs_per_s": round(len(items) / sum(times), 1),
    }
    info |= {
        "ms_single_pair_median": round(statistics.median(lat), 2),
        "batch": batch,
        "gpu": torch.cuda.get_device_name(),
        **peak,
    }
    return errors, info


def kernel_build() -> dict:
    """The RANSAC kernel binary loaded here: its file (named by the build), SHA-256 prefix, and where it came from."""
    import torch  # noqa: PLC0415

    from fastforks._core import kernels  # noqa: PLC0415
    from fastforks.kernels import ransac  # noqa: PLC0415

    manifest = kernels.read_manifest(Path(ransac.__file__).with_name("manifest.toml"))
    path, target = kernels.binary(manifest, torch.device("cuda"))  # checked against the manifest's SHA-256
    spec = manifest.files[target]
    env = [Path(d).resolve() for d in os.environ.get(kernels.KERNEL_PATH_ENV, "").split(os.pathsep) if d]
    local = any(path.resolve().is_relative_to(d) for d in env)
    return {"file": spec.name, "sha256": spec.sha256[:16], "source": "local build" if local else "published"}


def provenance(method: str, matches: Path) -> dict:
    """What a row needs to be trusted later: code state, library versions, machine, load, the exact matches."""
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    git = partial(subprocess.run, cwd=ROOT, capture_output=True, text=True, check=False)
    versions = {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__}
    if method == "poselib":
        import poselib  # noqa: PLC0415

        versions["poselib"] = poselib.__version__
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                ln.split(":", 1)[1].strip()
                for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")
            ),
            cpu,
        )
    sha = hashlib.sha256()
    for f in sorted(matches.glob("*.npz")):
        sha.update(f.read_bytes())
    return {
        "src_dirty": bool(git(["git", "status", "--porcelain", "--", "src", "tools"]).stdout.strip()),
        "versions": versions,
        "cpu_model": cpu,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "loadavg_1min": round(os.getloadavg()[0], 2),
        "matches": matches.name,
        "matches_sha256": sha.hexdigest()[:16],
        "kernel": kernel_build() if method == "kernels" else None,
        # Paths reduced to their last component: the row says which files, not where they lived on the machine.
        "command": " ".join(["python", *(Path(x).name if x.startswith("/") else x for x in sys.argv)]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/home/florian/projects/data/matching/megadepth1500"))
    ap.add_argument("--matches", type=Path, required=True)
    ap.add_argument("--method", required=True, choices=("opencv", "usac", "poselib", "portable", "kernels"))
    ap.add_argument("--jobs", type=int, default=1, help="processes for the CPU methods")
    ap.add_argument("--batch", type=int, default=500, help="pairs per call for the GPU methods")
    ap.add_argument("--latency-pairs", type=int, default=100)
    ap.add_argument("--stop", type=int, default=None, help="only the first pairs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--errors", type=Path, default=None, help="save the per-pair errors (npy)")
    ap.add_argument("--problem", default="essential", choices=("essential", "fundamental"))
    ap.add_argument("--threshold", type=float, default=None, help="pixels; default 0.5 (essential), 1.0 (fundamental)")
    a = ap.parse_args()
    from PIL import Image  # noqa: PLC0415

    pairs = load_pairs(a.root)[: a.stop]
    kpts: dict[str, np.ndarray] = {}
    for p in sorted(a.matches.glob("*.npz")):
        kpts |= {k: v for k, v in np.load(p).items() if int(k) < len(pairs)}
    items: list[Item] = []
    for i, (name1, name2, k1, k2, r, t) in enumerate(pairs):
        (w1, h1), (w2, h2) = (Image.open(a.root / "images" / n).size for n in (name1, name2))
        k1s, k2s = k1.copy(), k2.copy()
        k1s[:2] *= LONG_SIDE / max(w1, h1)
        k2s[:2] *= LONG_SIDE / max(w2, h2)
        rng = np.random.default_rng(i)
        for run, x in enumerate(kpts[str(i)]):
            items.append((x[rng.permutation(len(x))].astype(np.float64), k1s, k2s, r, t, i * N_RUNS + run))
    fundamental = a.problem == "fundamental"
    thr = a.threshold if a.threshold is not None else (1.0 if fundamental else 0.5)
    prov = provenance(a.method, a.matches)
    t0 = time.perf_counter()
    if a.method in {"portable", "kernels"}:
        errors, info = gpu(items, a.method, a.batch, a.latency_pairs, fundamental=fundamental, thr=thr)
    else:
        fn = {
            "opencv": partial(run_opencv, fundamental=fundamental, thr=thr),
            "usac": partial(run_opencv, usac=True, fundamental=fundamental, thr=thr),
            "poselib": partial(run_poselib, fundamental=fundamental, thr=thr),
        }[a.method]
        if a.jobs > 1:
            with Pool(a.jobs) as pool:
                res = pool.map(fn, items, chunksize=8)
        else:
            res = [fn(it) for it in items]
        errors = [e for e, _ in res]
        wall = time.perf_counter() - t0
        info = {
            "jobs": a.jobs,
            "ms_per_pair_single": round(statistics.median([d for _, d in res]), 2),
            "pairs_per_s": round(len(items) / wall, 1),
        }
        info |= {"ms_per_pair_wall": round(wall * 1e3 / len(items), 3), "cpu": os.cpu_count()}
    auc = [round(100 * v, 2) for v in pose_auc(np.array(errors), (5, 10, 20))]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    row = {
        "method": a.method,
        "problem": a.problem,
        "threshold": thr,
        "n_pairs": len(pairs),
        "n": len(items),
        "auc5": auc[0],
        "auc10": auc[1],
        "auc20": auc[2],
        **info,
    }
    row |= {k: v for k, v in prov.items() if k not in row or v is not None} | {
        "commit": commit,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    with a.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    if a.errors:
        np.save(a.errors, np.array(errors))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
