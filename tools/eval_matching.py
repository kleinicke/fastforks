"""Two-view pose accuracy of fastforks matchers on MegaDepth-1500 (AUC@5/10/20 of the relative pose error).

    python tools/eval_matching.py match --ref romav2 --mode portable --work DIR [--start 0 --stop 300]   # GPU
    python tools/eval_matching.py score --ref romav2 --mode portable --work DIR --commit C --out x.jsonl  # CPU

`--every N` (both steps): only every N-th pair (pair index % N == 0), dataset "megadepth1500-everyN"; slow variants
(float32) are compared on this subset pair for pair. A full-run chunk counts toward every subset.

The 1500 pairs of scenes 0015 and 0022 (LoFTR's test split; the glue-factory `megadepth1500` layout:
`pairs_calibrated.txt` with K1, K2, R, t of each pair, `images/`), as upstream RoMa v2's `benchmarks/mega1500.py`:

- `match` (GPU, resumable in chunks): the model matches the pair at full size in both directions (two calls: image1
  -> image2 and image2 -> image1), with the options of upstream's "mega1500" setting (`coarse_size` 800, `fine_size`
  1024). Upstream's balanced sampling then draws 5000 matches five times (seeded): certainties above 0.05 count as 1
  (upstream `threshold`), matches with a normalised coordinate beyond 1 - 1 / fine_size are dropped, 20000 are drawn
  by certainty, then 5000 by the inverse of their kernel density (std 0.1, density < 10 excluded). Keypoints in pixels
  of the images scaled to 1200 px on the long side (upstream's convention, the intrinsics scaled alike). Writes
  `DIR/<ref>-<mode>/<first>.npz` per chunk and, in a JSON next to it, the time of each pair's two calls (upload of the
  uint8 images and the outputs on the GPU included) and their peak torch memory (the model loaded; sampling excluded,
  reserved memory includes what earlier pairs left in the cache).
- Sparse matchers (LightGlue): one call image1 -> image2 at the protocol of LightGlue's MegaDepth-1500 table
  (glue-factory: `long_side` 1600, `max_keypoints` 2048); all its matches, once (`N_RUNS` 1), in the same 1200 px
  frame. `upstream` / `upstream16`: upstream `match_pair` (`tools/_original.py` `_lightglue`, `mp` off / on; needs
  `FF_LIGHTGLUE_REPO` and kornia).
- `score` (CPU): per sampling OpenCV `findEssentialMat` RANSAC (threshold 0.5 px over the mean focal length, confidence
  0.99999) and `recoverPose`, error = max(angle error of R, of t up to sign); 90 degrees if RANSAC fails. AUC@5/10/20
  over the 7500 errors (upstream `pose_auc`). Appends one row to `--out`.

Modes: `portable32` (float32), `portable`, `kernels` (card default dtype) through `fastforks.matching.load`;
`upstream` / `upstream16`: the original code (`tools/golden/romav2.py` `build`; fp32 with every autocast off and
matmul precision "highest" / upstream's `enable_amp=True` bfloat16) through the same two calls and the same sampling;
`upstreambench`: upstream's own benchmark path (`apply_setting("mega1500")`: one bidirectional bf16 call, upstream
`sample` on its 1024x1024 output grid), the reference for this protocol. The original needs `FF_ROMAV2_REPO` and
`FF_DINOV3_REPO` (git checkouts with the pinned commits) and einops.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _data import DATA

DATASET = "megadepth1500"
OPTIONS = {"coarse_size": 800, "fine_size": 1024}  # upstream setting "mega1500"
N_MATCHES, N_RUNS, LONG_SIDE, THRESHOLD, MIN_DENSITY = 5000, 5, 1200, 0.05, 10
FIVE = 5  # pose_error: the essential matrix needs 5 correspondences
SPARSE_OPTIONS = {"long_side": 1600, "max_keypoints": 2048}  # glue-factory's megadepth1500 setting of LightGlue
ORIGINAL = {"upstream": ("upstream", "original fp32"), "upstream16": ("upstream-fp16", "original fp16")}


def load_pairs(root: Path) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(image1, image2, K1, K2, R, t) per line of `pairs_calibrated.txt`; R, t map camera 1 to camera 2."""
    out = []
    for line in (root / "pairs_calibrated.txt").read_text().splitlines():
        a, b, *v = line.split()
        x = np.array(v, dtype=np.float64)
        out.append((a, b, x[:9].reshape(3, 3), x[9:18].reshape(3, 3), x[18:27].reshape(3, 3), x[27:30]))
    return out


def read(path: Path) -> torch.Tensor:
    """RGB uint8 `[3, H, W]` on the GPU."""
    from PIL import Image  # noqa: PLC0415

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).cuda()


def kde(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Upstream `kde` (half precision, Gaussian kernel)."""
    x = x.half()
    return (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp().sum(dim=-1)


def normalised(warp: torch.Tensor, h2: int, w2: int) -> torch.Tensor:
    """`[N, 4]` (x1, y1, x2, y2) in [-1, 1] of a pixel warp `[2, H1, W1]` into an `h2 x w2` image."""
    h1, w1 = warp.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(h1, device=warp.device), torch.arange(w1, device=warp.device), indexing="ij")
    grid = torch.stack(((xs + 0.5) * (2 / w1) - 1, (ys + 0.5) * (2 / h1) - 1), dim=-1)
    size = torch.tensor((w2, h2), device=warp.device, dtype=torch.float32)
    to = (warp.permute(1, 2, 0).float() + 0.5) * (2 / size) - 1
    return torch.cat((grid, to), dim=-1).reshape(-1, 4)


def sample(  # noqa: PLR0913, PLR0917 - the sampling options
    matches: torch.Tensor,
    certainty: torch.Tensor,
    limit: float,
    gen: torch.Generator,
    n: int = N_MATCHES,
    sampling: str = "balanced",
) -> torch.Tensor:
    """Upstream `RoMaV2.sample` (balanced) of `[N, 4]` normalised matches with their `[N]` certainties: the indices.

    `sampling`: "plain" draws by certainty only, "fast" balances with `MatchingResult.matches`' subsampled density.
    """
    c = torch.where(certainty > THRESHOLD, 1.0, certainty) * matches.abs().amax(dim=-1).le(limit).float()
    from fastforks.matching._api import _density  # noqa: PLC0415

    if sampling == "plain":
        return torch.multinomial(c, n, replacement=False, generator=gen)
    ind = torch.multinomial(c, 4 * n, replacement=False, generator=gen)
    m = matches[ind]
    density = kde(m) if sampling == "balanced" else _density(m[None])[0]
    p = 1 / (density + 1)
    p[density < MIN_DENSITY] = 1e-7  # upstream: at least about 10 close neighbours
    return ind[torch.multinomial(p, min(n, len(m)), replacement=False, generator=gen)]


def sparse(ref: str) -> bool:
    """Whether `ref` is a sparse matcher (index pairs of keypoints, no warp)."""
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("matching", ref)
    return "pairs" in (variant.outputs or card.outputs)


def _fastforks(ref: str, mode: str) -> tuple[Any, str, dict]:
    import fastforks.matching as ff  # noqa: PLC0415

    dtype = "float32" if mode == "portable32" else None
    options = SPARSE_OPTIONS if sparse(ref) else OPTIONS
    model = ff.load(ref, mode=mode.removesuffix("32"), dtype=dtype, options=options)
    backend = {"mode": model.backend.mode, "dtype": str(model.backend.dtype), "kernels": list(model.backend.kernels)}

    def call(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = model(a, b)
        return r.matches() if model.sparse else (r.warp, r.certainty)

    return call, backend["dtype"], {"backend": backend}


def _original(ref: str, mode: str) -> tuple[Any, str, dict]:
    import os  # noqa: PLC0415

    from _original import _golden  # noqa: PLC0415  # tools/
    from fastforks._core import registry  # noqa: PLC0415

    if sparse(ref):
        import types  # noqa: PLC0415

        from _original import _lightglue, _xfeat  # noqa: PLC0415  # tools/

        fp16 = mode == "upstream16"
        if registry.resolve("matching", ref)[0].family == "xfeat":  # no fp16 option of its own: autocast
            run = _xfeat(types.SimpleNamespace(), ref, options=SPARSE_OPTIONS)

            def matched(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                with torch.autocast("cuda", torch.float16, enabled=fp16):
                    r = run(a, b)
                return r["keypoints1"][r["pairs"][:, 0]].float(), r["keypoints2"][r["pairs"][:, 1]].float()

            note = "upstream XFeat" + (", torch.autocast float16" if fp16 else "")
            return matched, "float16" if fp16 else "float32", {"note": note}
        match_pair = _lightglue(types.SimpleNamespace(), ref, fp16, options=SPARSE_OPTIONS)

        def matched(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            f1, f2, m = match_pair(a, b)
            return f1["keypoints"][m["matches"][:, 0]], f2["keypoints"][m["matches"][:, 1]]

        note = "upstream match_pair + LightGlue" + (", mp=True (fp16)" if fp16 else "")
        return matched, "float16" if fp16 else "float32", {"note": note}
    g = _golden("romav2")

    card, variant = registry.resolve("matching", ref)
    tree = g.reference_tree(Path(os.environ["FF_ROMAV2_REPO"]), g.REF_COMMIT)
    dino = g.reference_tree(Path(os.environ["FF_DINOV3_REPO"]), g.DINOV3_COMMIT)
    amp = mode != "upstream"
    model = g.build(tree, dino, registry.weights(card, variant, allow_download=False)[0], amp=amp).cuda()
    note = "upstream RoMaV2.match, " + ("enable_amp=True (bf16 autocast)" if amp else "every autocast off, fp32")
    note += ", local correlation: upstream's plain-torch fallback (fused-local-corr not installed)"
    if mode == "upstreambench":
        model.apply_setting("mega1500")  # bidirectional, threshold 0.05: sampled by upstream below
        return model, "bfloat16", {"note": note + "; upstream mega1500 setting and sample()"}
    g.configure(model, OPTIONS["coarse_size"], OPTIONS["fine_size"])

    def call(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = model.match(a[None].float() / 255, b[None].float() / 255)
        out = g.to_api(preds, tuple(a.shape[-2:]), tuple(b.shape[-2:]))
        return out["warp"][0], out["certainty"][0]

    return call, "bfloat16" if amp else "float32", {"note": note}


def match(a: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - one loop over the pairs
    stop = a.stop if a.stop is not None else sys.maxsize
    pairs = [(i, p) for i, p in enumerate(load_pairs(a.root)) if a.start <= i < stop and i % a.every == 0]
    out = a.work / f"{a.ref}-{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    original = a.mode in ORIGINAL or a.mode == "upstreambench"
    call, dtype, extra = (_original if original else _fastforks)(a.ref, a.mode)
    limit = 1 - 1 / OPTIONS["fine_size"]
    is_sparse = sparse(a.ref)
    # The legacy set (balanced, N_MATCHES) keeps the plain keys and is drawn first; extra sets under "<sampling><n>/".
    sets = [("", "balanced", N_MATCHES)]
    sets += [(f"{s}{n}/", s, n) for s in a.sampling for n in a.n_matches if (s, n) != ("balanced", N_MATCHES)]
    kpts, ms, peak, first = {}, [], [0, 0], None
    for i, (name1, name2, *_) in pairs:
        im1, im2 = read(a.root / "images" / name1), read(a.root / "images" / name2)
        (h1, w1), (h2, w2) = im1.shape[-2:], im2.shape[-2:]
        s1, s2 = LONG_SIDE / max(h1, w1), LONG_SIDE / max(h2, w2)
        size = torch.tensor((w1 * s1, h1 * s1, w2 * s2, h2 * s2), device="cuda")
        gen = torch.Generator("cuda").manual_seed(i)
        with torch.inference_mode():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            if a.mode == "upstreambench":
                preds = call.match(im1[None].float() / 255, im2[None].float() / 255)
            elif is_sparse:
                p1, p2 = call(im1, im2)
            else:
                warp_ab, cert_ab = call(im1, im2)
                warp_ba, cert_ba = call(im2, im1)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000)
            peak = [max(peak[0], torch.cuda.max_memory_allocated()), max(peak[1], torch.cuda.max_memory_reserved())]
            runs = []
            if is_sparse:  # all matches, once; pixel centres -> the corner frame of the dense path's coordinates
                scale = torch.tensor((s1, s1, s2, s2), device="cuda")
                runs.append(((torch.cat((p1, p2), 1).float() + 0.5) * scale).cpu().numpy())
            elif a.mode != "upstreambench":
                both = torch.cat((normalised(warp_ab, h2, w2), normalised(warp_ba, h1, w1)[:, [2, 3, 0, 1]]))
                cert = torch.cat((cert_ab.reshape(-1), cert_ba.reshape(-1)))
            if is_sparse:
                kpts[str(i)] = np.stack(runs)
            for key, sampling, n in () if is_sparse else sets:
                runs, certs = [], []
                for _ in range(N_RUNS):
                    if a.mode == "upstreambench":
                        torch.manual_seed(int(torch.randint(2**31, (1,), generator=gen, device="cuda")))
                        m = call.sample(preds, N_MATCHES)[0]
                    else:
                        ind = sample(both, cert, limit, gen, n, sampling)
                        m = both[ind]
                        certs.append(cert[ind].float().cpu().numpy())
                    runs.append(((m + 1) / 2 * size).float().cpu().numpy())
                kpts[f"{key}{i}"] = np.stack(runs)  # [runs, matches, 4]: x1, y1, x2, y2 in pixels of the 1200 px images
                if a.certainty:
                    kpts[f"{key}c{i}"] = np.stack(certs)  # [runs, matches]
        if len(ms) == 1 and first is None:
            first = ms.pop()  # the first call includes start-up: not in the median
    np.savez(out / f"{a.start:04d}.npz", **kpts)
    info = {"start": a.start, "n": len(pairs), "dtype": dtype, "ms": ms, "first_ms": first, **extra}
    info |= {"peak_alloc_mb": peak[0] >> 20, "peak_reserved_mb": peak[1] >> 20, "gpu": torch.cuda.get_device_name()}
    (out / f"{a.start:04d}.json").write_text(json.dumps(info))
    print(a.mode, a.start, len(pairs), f"{statistics.median(ms) if ms else first:.0f} ms/pair", flush=True)


def pose_auc(errors: np.ndarray, thresholds: tuple[int, ...]) -> list[float]:
    """Upstream `pose_auc`: area under the recall curve of the errors up to each threshold, normalised."""
    errors = np.r_[0.0, np.sort(errors)]
    recall = np.r_[0.0, (np.arange(len(errors) - 1) + 1) / (len(errors) - 1)]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r, e = np.r_[recall[:last], recall[last - 1]], np.r_[errors[:last], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs


def pose_error(k1: np.ndarray, k2: np.ndarray, x: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray) -> float:
    """max(rotation, translation-direction angle error) in degrees of upstream's `estimate_pose_cv2_ransac`."""
    import cv2  # noqa: PLC0415

    if len(x) < FIVE:  # OpenCV raises on fewer points than the five-point solver needs
        return 90.0
    thr = 0.5 / (np.mean(np.abs(k1[:2, :2])) + np.mean(np.abs(k2[:2, :2])))
    p1 = (np.linalg.inv(k1[:2, :2]) @ (x[:, :2] - k1[None, :2, 2]).T).T
    p2 = (np.linalg.inv(k2[:2, :2]) @ (x[:, 2:] - k2[None, :2, 2]).T).T
    e, mask = cv2.findEssentialMat(p1, p2, np.eye(3), threshold=thr, prob=0.99999)
    if e is None:
        return 90.0
    best, ret = 0, None
    for ei in np.split(e, len(e) // 3):
        n, r, t, _ = cv2.recoverPose(ei, p1, p2, np.eye(3), 1e9, mask=mask)
        if n > best:
            best, ret = n, (r, t[:, 0])
    if ret is None:
        return 90.0
    r, t = ret
    cos_t = np.dot(t, gt_t) / (np.linalg.norm(t) * np.linalg.norm(gt_t))
    e_t = np.rad2deg(np.arccos(np.clip(cos_t, -1.0, 1.0)))
    e_r = np.rad2deg(np.abs(np.arccos(np.clip((np.trace(r.T @ gt_r) - 1) / 2, -1.0, 1.0))))
    return float(max(min(e_t, 180 - e_t), e_r))


def score(a: argparse.Namespace) -> None:
    import cv2  # noqa: PLC0415

    pairs = load_pairs(a.root)[: a.stop]  # --stop: smoke runs on the first pairs
    index = range(0, len(pairs), a.every)
    work = a.work / f"{a.ref}-{a.mode}"
    kpts: dict[str, np.ndarray] = {}
    infos = [json.loads(p.read_text()) for p in sorted(work.glob("*.json"))]
    for p in sorted(work.glob("*.npz")):
        kpts |= {k: v for k, v in np.load(p).items() if int(k) < len(pairs) and int(k) % a.every == 0}
    missing = sorted(set(index) - {int(k) for k in kpts})
    if missing:
        msg = f"{work}: {len(missing)} pairs not matched yet (first {missing[0]})"
        raise SystemExit(msg)
    errors = []
    for i in index:
        name1, name2, k1, k2, r, t = pairs[i]
        from PIL import Image  # noqa: PLC0415

        (w1, h1), (w2, h2) = (Image.open(a.root / "images" / n).size for n in (name1, name2))
        k1s, k2s = k1.copy(), k2.copy()
        k1s[:2] *= LONG_SIDE / max(w1, h1)
        k2s[:2] *= LONG_SIDE / max(w2, h2)
        rng = np.random.default_rng(i)
        for run, x in enumerate(kpts[str(i)]):
            cv2.setRNGSeed(i * N_RUNS + run)
            errors.append(pose_error(k1s, k2s, x[rng.permutation(len(x))], r, t))
    auc = [round(100 * v, 2) for v in pose_auc(np.array(errors), (5, 10, 20))]
    metrics = {"auc5": auc[0], "auc10": auc[1], "auc20": auc[2]}
    ms = [m for info in infos for m in info["ms"]]
    docs_mode, column = ORIGINAL.get(a.mode, (a.mode, None if a.mode in {"portable32", "upstreambench"} else a.mode))
    row = {"ref": a.ref, "task": "matching", "tier": a.mode, "mode": docs_mode, "dtype": infos[0]["dtype"]}
    dataset = DATASET if a.every == 1 else f"{DATASET}-every{a.every}"
    fp16 = "autocast" if "autocast" in infos[0].get("note", "") else "authors flag"
    row |= {"how": {"upstream": "fp32", "upstream16": fp16}.get(a.mode), "dataset": dataset, "n": len(index)}
    row |= {**metrics, "metrics": metrics, "ms_per_img": round(statistics.median(ms), 1)}
    row |= {"peak_alloc_mb": max(i["peak_alloc_mb"] for i in infos)}
    row |= {"peak_reserved_mb": max(i["peak_reserved_mb"] for i in infos)}
    row |= {k: infos[0][k] for k in ("backend", "note") if k in infos[0]}
    row |= {"commit": a.commit, "gpu": infos[0]["gpu"], "torch": torch.__version__}
    row |= {"date": time.strftime("%Y-%m-%d %H:%M"), "command": " ".join(["python", *sys.argv]), "column": column}
    with Path(a.out).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(a.ref, a.mode, metrics, row["ms_per_img"], row["peak_alloc_mb"], row["peak_reserved_mb"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("match", "score"))
    ap.add_argument("--ref", default="romav2")
    ap.add_argument("--mode", required=True, help="portable32, portable, kernels, upstream, upstream16, upstreambench")
    ap.add_argument("--root", type=Path, default=DATA / "matching" / DATASET, help="the megadepth1500 directory")
    ap.add_argument("--work", type=Path, required=True, help="directory of the sampled matches")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="only every N-th pair (a fixed subset)")
    ap.add_argument("--n-matches", type=int, nargs="+", default=[N_MATCHES], help="match: extra set sizes")
    ap.add_argument("--sampling", nargs="+", default=["balanced"], choices=("balanced", "plain", "fast"), help="match")
    ap.add_argument("--certainty", action="store_true", help="match: also save the certainty of every match")
    ap.add_argument("--commit", help="score: git commit of the measured src")
    ap.add_argument("--out", help="score: jsonl to append to")
    a = ap.parse_args()
    match(a) if a.step == "match" else score(a)


if __name__ == "__main__":
    main()
