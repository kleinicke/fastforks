"""Stereo and monocular-depth accuracy of one model in every tier, the protocol of `docs/data/README.md`.

    python tools/eval_stereo_depth.py run stereo raft-stereo-eth3d --tiers portable32,portable,kernels --out raw/
    python tools/eval_stereo_depth.py run depth depth-pro --tiers upstream,upstream16,upstreamhalf --out raw/
    python tools/eval_stereo_depth.py rows --out raw/ > rows.jsonl      # summary rows (docs/data schema)

Tiers: `portable32` (portable, dtype float32), `portable` / `kernels` (card default dtype) of the library;
`upstream` / `upstream16` / `upstreamhalf` = the authors' code in fp32 / mixed precision / `.half()` (`_original.py`;
run them with the family's reference venv). Every tier gets the same uint8 CUDA images; per-item rows go to
`<out>/<task>/<ref>__<tier>.jsonl` (resumable: `--max-seconds`, exit code 3 = call again).

Stereo: ETH3D two-view training (27 pairs, valid = finite GT < 512, bad > 1 px) and Middlebury 2014 trainingH, the 15
official training scenes (valid = finite GT < 1000, bad > 2 px), all pixels, the files and masks of the RAFT-Stereo
evaluation code. Depth: NYUv2 (654) and iBims-1 (100) of the MoGe eval set, ETH3D (27) / Middlebury trainingH (15)
left images with depth = f * B / (disp + doffs); least-squares scale + shift in the prediction's space (disparity for
inverse-depth outputs), clamp to the GT range, AbsRel and delta1 < 1.25; metric models also the raw metric AbsRel /
delta1. A prediction of another size is resized bilinearly to the GT (noted). Non-finite predictions count as 0 (the
fraction is recorded; > 1 % of an item's pixels makes the tier an `error` row). ms = CUDA-synchronised call, mean over
items without the process's first call; peak = max allocated / reserved during the calls (inputs and model included).
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _data import DATA

STEREO = DATA / "stereo"
MONO = DATA / "mono/moge_eval"
DATASETS = {"stereo": ("eth3d", "middH"), "depth": ("nyu", "ibims", "eth3d", "middH")}
BAD = {"eth3d": (1.0, 512.0), "middH": (2.0, 1000.0)}
NONFINITE_MAX = 0.01


def read_pfm(path: str) -> np.ndarray:
    with Path(path).open("rb") as f:
        hdr = f.readline().strip()
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        d = np.fromfile(f, "<f4" if scale < 0 else ">f4")
    d = d.reshape(h, w, 3)[..., 0] if hdr == b"PF" else d.reshape(h, w)
    return np.flipud(d).astype(np.float32).copy()


def read_depth_png(path: str) -> np.ndarray:
    """MoGe's depth PNG (log-encoded uint16 between `near` and `far`; 0 = invalid, 65535 = inf)."""
    im = Image.open(path)
    near, far = float(im.info["near"]), float(im.info["far"])
    d = np.array(im)
    bad = (d == 0) | (d == 65535)  # noqa: PLR2004
    t = (d.astype(np.float32) - 1) / 65533
    out = near ** (1 - t) * far**t
    if "unit" in im.info:
        out *= float(im.info["unit"])
    out[bad] = np.nan
    return out.astype(np.float32)


def read_calib(path: str) -> tuple[np.ndarray, float, float]:
    c = dict(line.strip().split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)
    k = np.array([[float(x) for x in r.split()] for r in c["cam0"].strip("[]").split(";")])
    return k, float(c["baseline"]) / 1000.0, float(c.get("doffs", 0))


def rgb(path: str) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(img[..., :3])


def stereo_items(ds: str) -> list[tuple[str, Any]]:
    """[(key, loader)]; loader() -> (left, right uint8 HWC, GT disparity HxW)."""
    if ds == "eth3d":
        ims = sorted(glob.glob(f"{STEREO}/ETH3D/two_view_training/*/im0.png"))  # noqa: PTH207
        gts = sorted(glob.glob(f"{STEREO}/ETH3D/two_view_training_gt/*/disp0GT.pfm"))  # noqa: PTH207
        pairs = list(zip(ims, gts, strict=True))
    else:
        root = f"{STEREO}/Middlebury/MiddEval3"
        names = Path(f"{root}/official_train.txt").read_text().split()
        ims = sorted(f"{root}/trainingH/{n}/im0.png" for n in names)
        pairs = [(i, i.replace("im0.png", "disp0GT.pfm")) for i in ims]
    return [
        (Path(i).parent.name, lambda i=i, g=g: (rgb(i), rgb(i.replace("im0", "im1")), read_pfm(g))) for i, g in pairs
    ]


def depth_items(ds: str) -> list[tuple[str, Any]]:
    """[(key, loader)]; loader() -> (uint8 HWC, GT depth HxW float32 m, nan = invalid)."""
    items: list[tuple[str, Any]] = []
    if ds in {"nyu", "ibims"}:
        root = f"{MONO}/{'NYUv2' if ds == 'nyu' else 'iBims-1'}"
        for k in Path(f"{root}/.index.txt").read_text().split():

            def ld(k: str = k, root: str = root) -> tuple[np.ndarray, np.ndarray]:
                gt = read_depth_png(f"{root}/{k}/depth.png")
                gt[~np.isfinite(gt)] = np.nan
                return rgb(f"{root}/{k}/image.jpg"), gt

            items.append((k, ld))
        return items
    if ds == "eth3d":
        scenes = sorted(glob.glob(f"{STEREO}/ETH3D/two_view_training/*/"))  # noqa: PTH207
        gtdir = lambda s: s.replace("two_view_training/", "two_view_training_gt/")  # noqa: E731
    else:
        scenes = sorted(glob.glob(f"{STEREO}/Middlebury/MiddEval3/trainingH/*/"))  # noqa: PTH207
        gtdir = lambda s: s  # noqa: E731
    for s in scenes:

        def ld2(s: str = s) -> tuple[np.ndarray, np.ndarray]:
            disp = read_pfm(f"{gtdir(s)}/disp0GT.pfm")
            k, b, doffs = read_calib(f"{s}/calib.txt")
            with np.errstate(divide="ignore", invalid="ignore"):
                gt = k[0, 0] * b / (disp + doffs)
            gt[~np.isfinite(disp) | (disp <= 0)] = np.nan
            return rgb(f"{s}/im0.png"), gt.astype(np.float32)

        items.append((Path(s).name, ld2))
    return items


def _hw(pred: Any, hw: tuple[int, int]) -> tuple[Any, bool]:  # noqa: ANN401
    """Prediction as float32 CUDA [H, W] at the GT size (bilinear if it differs) and whether it was resized."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    p = torch.as_tensor(np.asarray(pred) if not torch.is_tensor(pred) else pred).float().cuda().squeeze()
    if tuple(p.shape) == hw:
        return p, False
    return F.interpolate(p[None, None], hw, mode="bilinear", align_corners=False)[0, 0], True


def _metrics(p: Any, g: Any, m: Any) -> tuple[float, float]:  # noqa: ANN401
    import torch  # noqa: PLC0415

    a, b = p[m], g[m]
    return ((a - b).abs() / b).mean().item(), (torch.maximum(a / b, b / a) < 1.25).float().mean().item()  # noqa: PLR2004


def _align(pred: Any, gt: Any, m: Any, space: str) -> Any:  # noqa: ANN401
    """Least-squares scale + shift in the prediction's space, then clamp to the GT depth range (004 protocol)."""
    import torch  # noqa: PLC0415

    p, g = pred[m].double(), gt[m].double()
    t = g if space == "depth" else 1.0 / g
    sol = torch.linalg.lstsq(torch.stack([p, torch.ones_like(p)], 1), t[:, None]).solution[:, 0]
    a = pred.double() * sol[0] + sol[1]
    lo, hi = g.min(), g.max()
    return (a.clamp(lo, hi) if space == "depth" else 1.0 / a.clamp(1.0 / hi, 1.0 / lo)).float()


class Predictor:
    """One tier of one model: `call(uint8 HWC images...) -> (map, space)`; prep untimed, the call timed."""

    def __init__(self, task: str, ref: str, tier: str) -> None:  # noqa: D107
        import torch  # noqa: PLC0415

        self.task, self.tier, self.note = task, tier, ""
        self.kind = "relative"
        if tier.startswith("upstream"):
            import _original  # noqa: PLC0415

            self.orig = _original.load(task, ref, tier)
            _, _, variant = _original.full_ref(task, ref)
            self.kind = variant.output_info.get("kind", "relative") if task == "depth" else ""
            self.space = "disparity" if self.orig.adapter.key == "inverse_depth" else "depth"
            self.mode, self.dtype, self.how, self.note = "upstream", self.orig.dtype, self.orig.how, self.orig.note
            self.backend = f"original {self.how}"
            return
        self.orig = None
        mod = __import__(f"fastforks.{task}", fromlist=["load"])
        mode = "portable" if tier == "portable32" else tier
        kw = {"dtype": "float32"} if tier == "portable32" else {}
        self.model = mod.load(ref, mode=mode, device="cuda", **kw)
        self.mode, self.how = mode, None
        self.dtype = str(self.model.backend.dtype).removeprefix("torch.")
        self.backend = str(self.model.backend)
        if task == "depth":
            self.kind = self.model.variant.output_info["kind"]
        del torch

    def prep(self, ims: list[np.ndarray]) -> tuple[Any, ...]:  # noqa: D102
        import torch  # noqa: PLC0415

        xs = tuple(torch.from_numpy(i).permute(2, 0, 1).contiguous().cuda() for i in ims)
        return self.orig.prep(xs) if self.orig is not None else xs

    def call(self, args: tuple[Any, ...]) -> tuple[Any, str]:  # noqa: D102
        if self.orig is not None:
            return self.orig(*args), self.space
        r = self.model(*args)
        if self.task == "stereo":
            return r.disparity, "disparity"
        if r.inverse_depth is not None:
            return r.inverse_depth, "disparity"
        if r.depth is not None:
            return r.depth, "depth"
        return r.points[..., 2], "depth"


def _item(pred: Any, space: str, gt: np.ndarray, ds: str, task: str, kind: str) -> dict[str, Any]:  # noqa: ANN401, PLR0913, PLR0917
    import torch  # noqa: PLC0415

    g = torch.from_numpy(gt).cuda()
    p, resized = _hw(pred, tuple(g.shape))
    fin = torch.isfinite(p)
    row: dict[str, Any] = {"nonfinite": round(1 - fin.float().mean().item(), 6)}
    if resized:
        row["resized"] = True
    p = torch.where(fin, p, torch.zeros_like(p))
    if task == "stereo":
        thr, vmax = BAD[ds]
        v = torch.isfinite(g) & (g < vmax)
        e = (p - g).abs()[v]
        return row | {"epe": e.mean().item(), "bad": (e > thr).float().mean().item(), "n_px": int(v.sum())}
    m = torch.isfinite(g) & (g > 0) & fin
    row["rel_absrel"], row["rel_d1"] = _metrics(_align(p, g, m, space), g, m)
    if kind == "metric":
        md = (p if space == "depth" else 1.0 / p).clamp(min=1e-3)
        row["met_absrel"], row["met_d1"] = _metrics(md, g, m & torch.isfinite(md))
    return row | {"n_px": int(m.sum())}


def _commit() -> str:
    """The last library commit (src/) of this checkout: the fastforks main commit the run measures."""
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "log", "-1", "--format=%h", "--", "src"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def run(a: argparse.Namespace) -> int:
    """Every ref in `a.refs` in every tier, under one time budget; 3 = budget hit (call again)."""
    t0 = time.perf_counter()
    for ref in a.refs:
        if run_one(a, ref, t0) == 3:  # noqa: PLR2004
            return 3
    return 0


def run_one(a: argparse.Namespace, ref: str, t0: float) -> int:  # noqa: C901, PLR0915
    """One ref; tiers `upstreamhalf` is skipped where the authors' code has its own fp16 option."""
    import torch  # noqa: PLC0415

    out = Path(a.out) / a.task
    out.mkdir(parents=True, exist_ok=True)
    src = stereo_items if a.task == "stereo" else depth_items
    items = [(ds, k, ld) for ds in DATASETS[a.task] for k, ld in src(ds)[: a.limit or None]]
    again = False
    for tier in a.tiers.split(","):
        if tier == "upstreamhalf":
            import _original  # noqa: PLC0415

            if _original.own_option(a.task, ref):
                continue
        per = out / f"{ref}__{tier}.jsonl"
        rows = [json.loads(x) for x in per.read_text().splitlines()] if per.exists() else []
        if any("error" in r for r in rows):
            continue
        done = {(r["ds"], r["key"]) for r in rows if "key" in r}
        todo = [x for x in items if (x[0], x[1]) not in done]
        if not todo:
            continue
        if time.perf_counter() - t0 > a.max_seconds - a.build_reserve:
            again = True
            break
        f = per.open("a")
        meta: dict[str, Any] = {"meta": True, "ref": ref, "tier": tier, "commit": _commit(),
                                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "command": " ".join(["python", "tools/eval_stereo_depth.py", *sys.argv[1:]])}  # fmt: skip  # noqa: E501
        try:
            tb = time.perf_counter()
            pr = Predictor(a.task, ref, tier)
            meta |= {"mode": pr.mode, "dtype": pr.dtype, "how": pr.how, "note": pr.note, "backend": pr.backend,
                     "kind": pr.kind, "build_s": round(time.perf_counter() - tb, 1)}  # fmt: skip
        except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
            f.write(json.dumps(meta | {"error": f"build: {type(e).__name__}: {e}"[:400]}) + "\n")
            f.close()
            print("ERROR", ref, tier, type(e).__name__, str(e)[:300], flush=True)
            continue
        f.write(json.dumps(meta) + "\n")
        first = True
        for ds, key, ld in todo:
            if time.perf_counter() - t0 > a.max_seconds:
                again = True
                break
            data = ld()
            gt = data[-1]
            args = pr.prep(list(data[:-1]))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tc = time.perf_counter()
            try:
                pred, space = pr.call(args)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"ds": ds, "key": key, "error": f"call: {type(e).__name__}: {e}"[:400]}) + "\n")
                print("ERROR", ref, tier, key, type(e).__name__, str(e)[:300], flush=True)
                break
            ms = 1e3 * (time.perf_counter() - tc)
            row = {"ds": ds, "key": key, "ms": round(ms, 2), "first": first,
                   "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                   "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}  # fmt: skip
            row |= _item(pred, space, gt, ds, a.task, pr.kind)
            first = False
            f.write(json.dumps(row) + "\n")
            f.flush()
            pred = args = None
        f.close()
        del pr
        gc.collect()
        torch.cuda.empty_cache()
        if again:
            break
    return 3 if again else 0


def summary_rows(out: Path) -> list[dict[str, Any]]:
    """One docs row per model, tier and dataset from the per-item files under `out`."""
    res = []
    for per in sorted(out.glob("*/*__*.jsonl")):
        task = per.parent.name
        ref, tier = per.stem.split("__")
        rows = [json.loads(x) for x in per.read_text().splitlines()]
        metas = [r for r in rows if r.get("meta")]
        if not metas:
            continue
        meta = metas[-1]
        base = {"ref": ref, "task": task, "tier": tier, "mode": meta.get("mode"), "dtype": meta.get("dtype"),
                "how": meta.get("how"), **{k: meta[k] for k in ("commit", "gpu", "torch", "date", "command")}}  # fmt: skip  # noqa: E501
        if meta.get("note"):
            base["note"] = meta["note"]
        err = next((r["error"] for r in rows if "error" in r), None)
        for ds in DATASETS[task]:
            rs = [r for r in rows if r.get("ds") == ds and "key" in r and "error" not in r]
            want = len((stereo_items if task == "stereo" else depth_items)(ds))
            row = dict(base, dataset=ds)
            if err and len(rs) < want:
                res.append(row | {"error": err, "n": len(rs)})
                continue
            if len(rs) < want:
                continue  # unfinished
            bad_nf = sum(r["nonfinite"] > NONFINITE_MAX for r in rs)
            if bad_nf:
                row["error"] = f"non-finite output (> 1 % of the pixels) on {bad_nf} of {len(rs)} items"
            timed = [r["ms"] for r in rs if not r["first"]] or [r["ms"] for r in rs]
            mets = {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0] if k.endswith(("absrel", "d1"))}
            if task == "stereo":
                mets = {"bad": round(100 * float(np.mean([r["bad"] for r in rs])), 2),
                        "epe": round(float(np.mean([r["epe"] for r in rs])), 4)}  # fmt: skip
                row |= {"n_pairs": len(rs), "bad_thr": BAD[ds][0]}
            else:
                row["n_images"] = len(rs)
            if any(r.get("resized") for r in rs):
                row["note"] = "; ".join(x for x in (row.get("note"), "prediction resized to the GT size") if x)
            row |= mets
            row |= {"n": len(rs), "metrics": mets, "ms_per_img": round(float(np.mean(timed)), 2),
                    "mean_ms_after_first": round(float(np.mean(timed)), 2),
                    "peak_alloc_mb": max((r["peak_alloc_mb"] for r in rs if r["peak_alloc_mb"] is not None), default=None),  # noqa: E501
                    "peak_reserved_mb": max((r["peak_reserved_mb"] for r in rs if r["peak_reserved_mb"] is not None),
                                            default=None)}  # fmt: skip
            res.append(row)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rows"])
    ap.add_argument("task", nargs="?", choices=list(DATASETS))
    ap.add_argument("refs", nargs="*")
    ap.add_argument("--tiers", default="portable32,portable,kernels")
    ap.add_argument("--max-seconds", type=float, default=150.0)
    ap.add_argument("--build-reserve", type=float, default=60.0, help="no new tier after max-seconds minus this")
    ap.add_argument("--limit", type=int, default=0, help="first N items per dataset (pilot runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "rows":
        for r in summary_rows(Path(a.out)):
            print(json.dumps(r))
        return
    sys.exit(run(a))


if __name__ == "__main__":
    main()
