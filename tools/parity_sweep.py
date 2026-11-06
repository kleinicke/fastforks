"""Parity sweep: every family of every task on real inputs at two sizes, CPU fp32 vs portable CUDA vs kernels (public API).

  python tools/parity_sweep.py list                          # "task/family ref" of the families with local weights
  python tools/parity_sweep.py cpu FAMILY... --work DIR      # CPU float32 references (no GPU); writes DIR/<fam>.meta.json
  python tools/parity_sweep.py gpu FAMILY... --work DIR --out x.jsonl   # portable CUDA + kernels vs the reference; resumable
  python tools/parity_sweep.py report x.jsonl                # markdown table

FAMILY is "task/family" as printed by `list`. Per family one ref (the first with local weights) runs at two sizes: a
non-multiple-of-32 size and a larger realistic one (720x1280, capped at the largest size the model's `size_range()`
says fits in 16 GB). Inputs: real stereo pairs (KITTI crop, Middlebury), a COCO photo for single-image tasks, 4 DTU
views for multiview. Classification / features have fixed crops: two photos / two feature sizes instead.
The reference is portable float32 on the CPU (skipped if one call takes more than 60 s), else portable float32 on CUDA.
Errors are measured as the golden tests do (`tests/test_golden.py`, `golden.toml` tolerances) plus relative to the
reference output's range; classification / segmentation add top-1 / label agreement, detection a matched-IoU check.
"""

# ruff: noqa: ANN401, PLR2004, SLF001, PERF401, FURB171, E501, N818  # a one-off sweep tool: loose typing, long report lines
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import importlib
import json
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

import fastforks
from _data import DATA

if TYPE_CHECKING:
    from collections.abc import Callable

COCO = DATA / "coco" / "val2017"
PHOTOS = ("000000000139.jpg", "000000000285.jpg")
KITTI = DATA / "stereo" / "crops"
MIDD = DATA / "stereo" / "Middlebury" / "MiddEval3" / "testH" / "Australia"
DTU = DATA / "multiview" / "dtu" / "cache" / "scan1" / "cache_640x512.npz"
CPU_LIMIT_S = 60
TASKS = (
    "stereo",
    "depth",
    "multiview",
    "classification",
    "segmentation",
    "features",
    "promptable_segmentation",
    "detection",
)
STANDARD = {  # the task's output order: the first one present is the primary map (tolerances apply to it)
    "stereo": ("disparity", "confidence"),
    "depth": ("depth", "inverse_depth", "points", "intrinsics", "confidence", "mask", "normals"),
    "multiview": ("depth", "world_points", "confidence", "mask", "intrinsics", "cam_to_world"),
    "classification": ("logits",),
    "segmentation": ("logits",),
    "features": ("last_hidden_state", "pooler_output", "hidden_states"),
    "promptable_segmentation": ("pred_masks", "iou_scores", "boxes", "scores", "masks"),
}
MASK_CLASSIFICATION = {"eomt", "mask2former", "oneformer"}
INSTANCE_TASKS = {"detection"}
CAPPED = {"features": (112, 392)}
Case = tuple[str, Any]  # (name, size: (h, w) or feature size int or photo index)


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def api(task: str) -> Any:
    return importlib.import_module(f"fastforks.{task}")


def families() -> dict[str, list[str]]:
    """`{"task/family": [refs with local weights]}` (the size-table tool's check: no downloads)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import bench_sizes  # noqa: PLC0415

    return bench_sizes._families()


# ---------------------------------------------------------------- inputs


def fit(img: Image.Image, h: int, w: int) -> np.ndarray:
    """Centre crop to the aspect ratio of `h x w`, resize to it (uint8 RGB)."""
    iw, ih = img.size
    s = max(h / ih, w / iw)
    cw, ch = round(w / s), round(h / s)
    x0, y0 = (iw - cw) // 2, (ih - ch) // 2
    return np.asarray(img.convert("RGB").crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.BICUBIC))


def photo(i: int, h: int | None = None, w: int | None = None) -> np.ndarray:
    img = Image.open(COCO / PHOTOS[i]).convert("RGB")
    return np.asarray(img) if h is None else fit(img, h, w or h)


def stereo_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    if (h, w) == (360, 1200):
        return tuple(np.asarray(Image.open(KITTI / f"kitti_000005_10_1200x360_{s}.png").convert("RGB")) for s in "lr")  # type: ignore[return-value]
    return fit(Image.open(MIDD / "im0.png"), h, w), fit(Image.open(MIDD / "im1.png"), h, w)


def views(h: int, w: int) -> tuple[np.ndarray, dict[str, Any]]:
    d = np.load(DTU)
    ids = [24, *d["pairs"][24][:3].tolist()]
    imgs = np.stack([np.asarray(Image.fromarray(x).resize((w, h), Image.Resampling.BICUBIC)) for x in d["images"][ids]])
    k = d["K"][ids].astype(np.float64).copy()
    k[:, 0] *= w / d["images"].shape[2]
    k[:, 1] *= h / d["images"].shape[1]
    return imgs, {"intrinsics": k, "cam_to_world": d["c2w"][ids], "depth_range": (425.0, 935.0)}


def pick_sizes(task: str, m: Any) -> list[Case]:
    """Two cases: a non-multiple-of-32 size and 720x1280, never above what fits in 16 GB (`m.size_range()`)."""
    r = m.size_range()
    if task == "classification":
        return [("photo0", 0), ("photo1", 1)]
    if task == "features":
        return [(f"size{s}", s) for s in CAPPED["features"]]
    lo = max(r.min_size, r.reject_below)
    if task == "stereo":
        small = (360, 1200)
    elif task == "multiview":
        small = (360, 456)
    else:
        h = max(360, lo + 8)
        small = (h + 6 if h % 32 == 0 else h, h + 130 + (6 if (h + 130) % 32 == 0 else 0))
    big = (512, 640) if task == "multiview" else (720, 1280)
    if r.tested_max and big[0] * big[1] > r.tested_max[0] * r.tested_max[1]:
        big = r.tested_max
    cases = [("small", small)]
    if big[0] * big[1] > small[0] * small[1] * 1.2 and min(big) >= lo:
        cases.append(("large", big))
    return [(n, hw) for n, hw in cases if min(hw) >= lo]


def make_call(task: str, m: Any, case: Case) -> Callable[[], Any]:  # noqa: PLR0911 - one branch per task
    _, arg = case
    if task == "stereo":
        left, right = stereo_pair(*arg)
        return lambda: m(left, right)
    if task == "multiview":
        imgs, kw = views(*arg)
        calibrated = m.card.family in {"mvsanywhere", "patchmatchnet"}
        return lambda: m(list(imgs), **(kw if calibrated else {}))
    if task == "classification":
        img = photo(arg)
        return lambda: m(img)
    if task == "features":
        img = photo(0)
        layers = (0, 1) if "hidden_states" in m.variant.outputs else ()
        return lambda: m(img, size=arg, layers=layers)
    img = photo(0, *arg)
    if task == "segmentation":
        return lambda: m(img, logits=True)
    if task in {"detection", "depth"}:
        return lambda: m(img)
    h, w = arg  # promptable segmentation
    if "text" in m.prompts:
        return lambda: {"": m(img, text="person")}

    def sam() -> dict[str, Any]:
        emb = m.get_image_embeddings(img)
        pt = {"input_points": [[w // 2, h // 2]], "input_labels": [1]}
        return {
            "pt_": m(emb, **pt, multimask_output=True),
            "box_": m(emb, **pt, input_boxes=[w // 4, h // 4, 3 * w // 4, 3 * h // 4], multimask_output=False),
        }

    return sam


# ---------------------------------------------------------------- outputs


def _add(out: dict[str, torch.Tensor], k: str, v: Any) -> None:
    if isinstance(v, torch.Tensor):
        out[k] = v.detach().float().cpu()  # bool and integer maps as numbers
    elif isinstance(v, (tuple, list)) and v and all(isinstance(x, torch.Tensor) for x in v):
        for i, x in enumerate(v):
            out[f"{k}.{i}"] = x.detach().float().cpu()


def outputs(r: Any) -> dict[str, torch.Tensor]:
    """All tensors of a result (or of `{prefix: result}`), by name."""
    out: dict[str, torch.Tensor] = {}
    for prefix, res in r.items() if isinstance(r, dict) else [("", r)]:
        if isinstance(res, list):  # a batch result: first image
            res = res[0]  # noqa: PLW2901
        for f in dataclasses.fields(res):
            _add(out, prefix + f.name, getattr(res, f.name))
        for k, v in (getattr(res, "extras", None) or {}).items():
            _add(out, prefix + k, v)
    return out


def tolerance(task: str, ref: str, mode: str) -> dict[str, Any]:
    """The family's golden.toml `[tolerance]` with the ref's overrides, as `tests/test_golden.py::tolerance`.

    `real_gpu_mean` / `real_gpu_p99` replace `gpu_mean` / `gpu_p99` here only: the bound on real pairs up to 720x1280
    for families whose CUDA output is precision-sensitive there (the golden tests keep the synthetic pair's bound).
    """
    fam = api(task).model_card(ref).family
    path = Path(fastforks.__file__).parent / task / "models" / fam / "golden.toml"
    if not path.is_file():
        return {}
    d = tomllib.loads(path.read_text())
    entry = d.get("refs", {}).get(f"{ref}@1", {})
    own = {k: v for k, v in entry.items() if k.startswith(("gpu_mean", "gpu_p99", "gpu_extra"))}
    tol = d.get("tolerance", {}) | own
    tol |= {k.removeprefix("real_"): v for k, v in tol.items() if k.startswith("real_gpu_")}
    return tol | {k: tol[f"{k}_{mode}"] for k in ("gpu_mean", "gpu_p99") if f"{k}_{mode}" in tol}


def _p99(x: np.ndarray) -> float:
    flat = x.ravel()
    return float(np.percentile(flat[:: max(1, flat.size // 4_000_000)], 99))


def compare(task: str, ref: str, mode: str, want: dict[str, np.ndarray], got: dict[str, torch.Tensor]) -> list[dict]:
    """One record per output present in both; the golden-test verdict for the primary and `gpu_extra` outputs."""
    tol = tolerance(task, ref, mode)
    extra = {k: (mean, p99) for k, mean, p99 in tol.get("gpu_extra", [])}
    keys = [k for k in want if k in got]
    base = lambda k: k.removeprefix("pt_").removeprefix("box_").split(".")[0]  # noqa: E731
    order = [s for s in STANDARD[task] if any(base(k) == s for k in keys)]
    primary = next((k for k in keys if base(k) == order[0]), None) if order else None
    rows = []
    for k in keys:
        w, g = want[k].astype(np.float64), got[k].numpy().astype(np.float64)
        if w.shape != g.shape:
            rows.append({"key": k, "verdict": f"FLAG: shape {g.shape} vs {w.shape}"})
            continue
        fin = np.isfinite(w) & np.isfinite(g)
        bad = int((~fin).sum() - (~np.isfinite(w)).sum())  # non-finite in the test output only
        d = np.abs(np.where(fin, g - w, 0.0))
        rng = float(w[np.isfinite(w)].max() - w[np.isfinite(w)].min()) if np.isfinite(w).any() else 0.0
        rec: dict[str, Any] = {
            "key": k,
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "range": rng,
            "rel_mean": float(d.mean() / rng) if rng else None,
            "rel_max": float(d.max() / rng) if rng else None,
            "nonfinite_new": bad,
        }
        verdict = "info"
        nz = np.abs(w)[(w != 0) & np.isfinite(w)]
        if k == primary and "gpu_mean" in tol:
            unit = float(np.median(np.abs(w[np.isfinite(w)]))) if tol.get("relative") and np.isfinite(w).any() else 1.0
            e = d / (unit or 1.0)
            rec |= {
                "err_mean": float(e.mean()),
                "err_p99": _p99(e),
                "tol_mean": tol["gpu_mean"],
                "tol_p99": tol["gpu_p99"],
            }
            ok = rec["err_mean"] < tol["gpu_mean"] and rec["err_p99"] < tol["gpu_p99"]
            verdict = "ok" if ok else "FLAG"
        elif base(k) in extra:
            mean_t, p99_t = extra[base(k)]
            e = d / (float(np.median(nz)) if nz.size else 1.0)
            rec |= {"err_mean": float(e.mean()), "err_p99": _p99(e), "tol_mean": mean_t, "tol_p99": p99_t}
            verdict = "ok" if rec["err_mean"] < mean_t and rec["err_p99"] < p99_t else "FLAG"
        if bad:
            verdict = "FLAG"
        rec["verdict"] = verdict
        if (
            base(k) == "logits" and w.ndim >= 2
        ):  # classification: top-1 / top-5; segmentation: label agreement per pixel
            axis = -3 if task == "segmentation" else -1
            rec["top1_agree"] = float((w.argmax(axis) == g.argmax(axis)).mean())
            if task == "classification":
                srt = np.sort(w, axis=-1)
                rec["ref_margin"] = float((srt[..., -1] - srt[..., -2]).min())
                t5w, t5g = np.argsort(-w, axis=-1)[..., :5], np.argsort(-g, axis=-1)[..., :5]
                rec["top5_overlap"] = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(t5w, t5g, strict=True)]))
            if rec["top1_agree"] < 1.0 and task == "classification":
                rec["verdict"] = "FLAG: top-1 differs"
        rows.append(rec)
    return rows


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def compare_instances(want: dict[str, np.ndarray], got: dict[str, torch.Tensor], thr: float = 0.5) -> list[dict]:
    """Detections of score >= thr: greedy same-label box matching; counts, matched IoU, box / score / mask differences."""
    wb, ws = want["boxes"], want["scores"]
    gb, gs = got["boxes"].numpy(), got["scores"].numpy()
    wl = want.get("labels", np.zeros(len(ws)))
    gl = got["labels"].numpy() if "labels" in got else np.zeros(len(gs))
    wi, gi = np.where(ws >= thr)[0], np.where(gs >= thr)[0]
    iou = box_iou(wb[wi], gb[gi]) if len(wi) and len(gi) else np.zeros((len(wi), len(gi)))
    iou = iou * (wl[wi][:, None] == gl[gi][None])
    pairs, used = [], set()
    for i in range(len(wi)):
        j = int(np.argmax(iou[i])) if len(gi) else -1
        if j >= 0 and iou[i, j] > 0.5 and j not in used:
            used.add(j)
            pairs.append((wi[i], gi[j], float(iou[i, j])))
    rec: dict[str, Any] = {"key": "instances", "n_ref": len(wi), "n_test": len(gi), "matched": len(pairs)}
    if pairs:
        ious = np.array([p[2] for p in pairs])
        rec |= {
            "mean_iou": float(ious.mean()),
            "min_iou": float(ious.min()),
            "max_abs": float(max(np.abs(wb[a] - gb[b]).max() for a, b, _ in pairs)),
            "mean_abs": float(np.mean([np.abs(wb[a] - gb[b]).mean() for a, b, _ in pairs])),
            "score_max_abs": float(max(abs(ws[a] - gs[b]) for a, b, _ in pairs)),
        }
        wm = want.get("masks")
        if wm is not None and wm.ndim >= 3 and "masks" in got:
            gm = got["masks"].numpy() > 0.5
            mi = [((wm[a] > 0.5) & gm[b]).sum() / max(1, ((wm[a] > 0.5) | gm[b]).sum()) for a, b, _ in pairs]
            rec["mean_mask_iou"], rec["min_mask_iou"] = float(np.mean(mi)), float(np.min(mi))
        rec["rel_mean"] = None
    ok = len(wi) == len(gi) == len(pairs) and rec.get("min_iou", 0) >= 0.95
    rec["verdict"] = "ok" if ok else "FLAG"
    return [rec]


# ---------------------------------------------------------------- models and phases


def choose_ref(task: str, fam: str, avail: dict[str, list[str]]) -> str | None:
    refs = avail.get(f"{task}/{fam}", [])
    return refs[0].split("@")[0] if refs else None


def options_for(task: str, ref: str) -> dict[str, Any]:
    card = api(task).model_card(ref)
    return {"seed": 0} if "seed" in card.options else {}


def load(task: str, ref: str, mode: str, device: str, *, fp32: bool = False) -> Any:
    kw: dict[str, Any] = {"dtype": "float32"} if fp32 else {}
    return api(task).load(ref, mode=mode, device=device, options=options_for(task, ref) or None, **kw)


def run_case(task: str, m: Any, case: Case, *, calls: int = 1) -> dict[str, torch.Tensor]:
    f = make_call(task, m, case)
    out: dict[str, torch.Tensor] = {}
    for _ in range(calls):
        torch.manual_seed(0)
        out = outputs(f())
        if task == "segmentation" and m.card.family in MASK_CLASSIFICATION and "logits" in out:
            out["logits"] = out[
                "logits"
            ].exp()  # their public logits are log class probabilities (floor 1e-30): as the goldens
        if m.device.type == "cuda":
            torch.cuda.synchronize()
    return out


def meta_path(work: Path, fam: str) -> Path:
    return work / f"{fam.replace('/', '__')}.meta.json"


def ref_path(work: Path, fam: str, case: str, kind: str) -> Path:
    return work / f"{fam.replace('/', '__')}__{case}__{kind}.npz"


def save(path: Path, out: dict[str, torch.Tensor]) -> None:
    np.savez(path, **{k: v.numpy() for k, v in out.items()})


def cmd_cpu(a: argparse.Namespace) -> int:
    torch.set_num_threads(8)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    avail = families()

    class Slow(Exception):
        pass

    def alarm(*_: object) -> None:
        raise Slow

    signal.signal(signal.SIGALRM, alarm)
    for name in a.families:
        task, fam = name.split("/")
        ref = choose_ref(task, fam, avail)
        meta: dict[str, Any] = {"family": name, "task": task, "ref": ref, "cases": []}
        if ref is None:
            meta["skipped"] = "weights not available locally (no download attempted)"
        else:
            try:
                m = load(task, ref, "portable", "cpu")
                cases = pick_sizes(task, m)
                slow = False
                for case in cases:
                    rec: dict[str, Any] = {"size": case[0], "arg": case[1], "cpu": "ok"}
                    if slow:
                        rec["cpu"] = "skipped: a smaller size took > 60 s"
                    else:
                        t0 = time.monotonic()
                        signal.alarm(CPU_LIMIT_S)
                        try:
                            out = run_case(task, m, case)
                            save(ref_path(work, name, case[0], "cpu"), out)
                        except Slow:
                            slow, rec["cpu"] = True, f"skipped: one CPU call took > {CPU_LIMIT_S} s"
                        except Exception as e:  # noqa: BLE001
                            slow, rec["cpu"] = True, f"error: {type(e).__name__}: {str(e)[:200]}"
                        finally:
                            signal.alarm(0)
                        rec["cpu_s"] = round(time.monotonic() - t0, 1)
                    meta["cases"].append(rec)
                del m
            except Exception as e:  # noqa: BLE001
                meta["skipped"] = f"load: {type(e).__name__}: {str(e)[:200]}"
        meta_path(work, name).write_text(json.dumps(meta))
        gc.collect()
        print(name, meta.get("skipped") or [(c["size"], c["cpu"]) for c in meta["cases"]], flush=True)
    return 0


def _read(out: Path) -> list[dict]:
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip("\0 \n")] if out.is_file() else []


def _append(out: Path, rec: dict) -> None:
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def cmd_gpu(a: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0915
    from fastforks.kernels import cudnn_hacc  # noqa: PLC0415

    work, out = Path(a.work), Path(a.out)
    gpu, commit = torch.cuda.get_device_name(), _commit()
    rows = _read(out)
    done = {(r["family"], r["size"], r["tier"]) for r in rows if r["status"] != "started"}
    started: dict[tuple, int] = {}
    for r in rows:
        if r["status"] == "started":
            k = (r["family"], r["size"], r["tier"])
            started[k] = started.get(k, 0) + 1
    rc = 0
    for name in a.families:
        task = name.split("/")[0]
        meta = json.loads(meta_path(work, name).read_text())
        ref = meta["ref"]
        base = {"family": name, "task": task, "ref": ref, "gpu": gpu, "commit": commit, "date": now()}
        if meta.get("skipped"):
            if (name, "-", "-") not in done:
                _append(out, base | {"size": "-", "tier": "-", "status": "skipped", "why": meta["skipped"]})
            continue
        refs: dict[str, tuple[dict[str, np.ndarray], str]] = {}
        need32 = []
        for c in meta["cases"]:
            p = ref_path(work, name, c["size"], "cpu")
            if c["cpu"] == "ok" and p.is_file():
                refs[c["size"]] = (dict(np.load(p)), "cpu32")
            else:
                need32.append(c)
        # CUDA float32 references for the sizes the CPU could not do
        for c in need32:
            p = ref_path(work, name, c["size"], "cuda32")
            if p.is_file():
                refs[c["size"]] = (dict(np.load(p)), "cuda32")
        todo32 = [c for c in need32 if c["size"] not in refs]
        if todo32:
            try:
                m = load(task, ref, "portable", "cuda", fp32=True)
                for c in todo32:
                    case = (c["size"], tuple(c["arg"]) if isinstance(c["arg"], list) else c["arg"])
                    try:
                        o = run_case(task, m, case)
                        save(ref_path(work, name, c["size"], "cuda32"), o)
                        refs[c["size"]] = ({k: v.numpy() for k, v in o.items()}, "cuda32")
                    except torch.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                del m
            except Exception as e:  # noqa: BLE001
                _append(out, base | {"size": "-", "tier": "cuda32", "status": "error", "why": str(e)[:200]})
            gc.collect()
            torch.cuda.empty_cache()
        for tier, mode in (("cuda", "portable"), ("kernels", "kernels")):
            cases = [c for c in meta["cases"] if (name, c["size"], tier) not in done and c["size"] in refs]
            if not cases:
                continue
            if mode not in api(task).model_card(ref).modes:
                for c in cases:
                    _append(
                        out,
                        base
                        | {"size": c["size"], "tier": tier, "status": "skipped", "why": "no kernels for this family"},
                    )
                continue
            try:
                m = load(task, ref, mode, "cuda")
            except Exception as e:  # noqa: BLE001 - gated repo, missing kernel build, OOM at load
                for c in cases:
                    _append(
                        out,
                        base
                        | {
                            "size": c["size"],
                            "tier": tier,
                            "status": "skipped",
                            "why": f"{type(e).__name__}: {str(e)[:200]}",
                        },
                    )
                continue
            dtype = str(m.backend.dtype)
            for c in cases:
                key = (name, c["size"], tier)
                rec = base | {
                    "size": c["size"],
                    "arg": c["arg"],
                    "tier": tier,
                    "dtype": dtype,
                    "backend": str(m.backend),
                }
                if started.get(key, 0) >= 2:
                    _append(out, rec | {"status": "timeout", "why": "killed by the step timeout twice"})
                    continue
                _append(out, rec | {"status": "started"})
                case = (c["size"], tuple(c["arg"]) if isinstance(c["arg"], list) else c["arg"])
                want, ref_tier = refs[c["size"]]
                try:
                    t0 = time.monotonic()
                    got = run_case(task, m, case)
                    first = got
                    if tier == "kernels":  # call 2 with every cudnn_hacc plan built, as the golden test
                        old, cudnn_hacc.BUILD_AFTER = cudnn_hacc.BUILD_AFTER, 0
                        try:
                            got = run_case(task, m, case)
                        finally:
                            cudnn_hacc.BUILD_AFTER = old
                    for call, g in (("first", first), ("built", got)) if tier == "kernels" else (("only", got),):
                        res = (
                            compare_instances(want, g)
                            if task in INSTANCE_TASKS or "text" in getattr(m, "prompts", ())
                            else compare(task, ref, mode, want, g)
                        )
                        for r in res:
                            _append(
                                out,
                                rec
                                | r
                                | {
                                    "status": "ok",
                                    "ref_tier": ref_tier,
                                    "call": call,
                                    "s": round(time.monotonic() - t0, 1),
                                },
                            )
                except torch.OutOfMemoryError as e:
                    _append(out, rec | {"status": "oom", "why": str(e).split("\n")[0][:200]})
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    _append(out, rec | {"status": "error", "why": f"{type(e).__name__}: {str(e)[:300]}"})
                    if "CUDA" in str(e):
                        return 1
            del m
            gc.collect()
            torch.cuda.empty_cache()
    return rc


def cmd_report(a: argparse.Namespace) -> int:
    rows = _read(Path(a.jsonl))
    lines = [
        "| task | ref | size | tier | ref tier | output | max abs | mean abs | rel mean | rel max | extra | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    f = lambda v: "" if v is None else f"{v:.3g}"  # noqa: E731
    skipped = []
    for r in rows:
        if r["status"] in {"started"}:
            continue
        if r["status"] != "ok":
            skipped.append(f"- {r['family']} ({r['ref']}) {r['size']} {r['tier']}: {r['status']}: {r.get('why', '')}")
            continue
        if r["verdict"] == "info" or (r.get("call") == "first" and r["verdict"] == "ok"):
            continue
        ex = []
        for k in (
            "top1_agree",
            "top5_overlap",
            "ref_margin",
            "n_ref",
            "n_test",
            "matched",
            "mean_iou",
            "min_iou",
            "mean_mask_iou",
            "err_mean",
            "tol_mean",
            "err_p99",
            "tol_p99",
        ):
            if k in r:
                ex.append(f"{k}={f(r[k])}")
        size = r["size"] if r["size"].startswith(("photo", "size")) else f"{r['size']} {r['arg'][0]}x{r['arg'][1]}"
        tier = r["tier"] + (" (1st call)" if r.get("call") == "first" else "")
        lines.append(
            f"| {r['task']} | {r['ref']} | {size} | {tier} | {r['ref_tier']} | {r['key']} | {f(r.get('max_abs'))} | {f(r.get('mean_abs'))} | {f(r.get('rel_mean'))} | {f(r.get('rel_max'))} | {' '.join(ex)} | {r['verdict']} |"
        )
    print("\n".join(lines))
    print("\n## Skipped / failed\n" + "\n".join(skipped))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for n in ("cpu", "gpu"):
        s = sub.add_parser(n)
        s.add_argument("families", nargs="+")
        s.add_argument("--work", required=True)
        if n == "gpu":
            s.add_argument("--out", required=True)
    s = sub.add_parser("report")
    s.add_argument("jsonl")
    a = p.parse_args()
    if a.cmd == "list":
        avail = families()
        for t in TASKS:
            fams = sorted({api(t).model_card(r).family for r in api(t).list_models()})
            for fam in fams:
                print(f"{t}/{fam}", choose_ref(t, fam, avail) or "-")
        return 0
    return {"cpu": cmd_cpu, "gpu": cmd_gpu, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
