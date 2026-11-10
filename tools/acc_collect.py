"""Collect the 2026-11-05 accuracy campaign's raw files into docs/data rows (docs/data/README.md, "Accuracy").

    python tools/acc_collect.py --raw RAW [--write [--task TASK ...]] [--md summary.md]

RAW holds the eval tools' raw output: `stereo/`, `depth/` (eval_stereo_depth.py / eval_foreign.py per-item files),
`classification/cls.jsonl`, `segmentation/seg.jsonl` (chunk records), `coco_rows.jsonl` (eval_coco.py summarize
--rows), `features/features.jsonl`, `multiview/mv.jsonl`, `speed/*.jsonl` (speed_original.py, eval_foreign.py --speed).
One row per model, tier and dataset: ref, task, tier, mode, dtype, how, dataset, n, metrics, ms_per_img,
peak_alloc_mb, peak_reserved_mb, commit, gpu, torch, date, command (+ note / error), and `column`, the docs column the
row fills: "original fp32", "original fp16", "portable", "kernels" (null: a tier the docs do not show, e.g.
`portable32` / `upstreamhalf` of the stereo runs).

"original fp16" = `upstream16`: the authors' own precision option where the code has one, else autocast float16
(`upstreamac` where the own option fails), as measured, also where it loses accuracy (the point of the column; user
2026-11-06). `--write` replaces docs/data/<task>_eval.jsonl and appends the original speed rows (fp32 + fp16) to
docs/data/speed.jsonl (rows already there are not repeated); `--task` limits both to the tasks measured completely.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_stereo_depth import summary_rows

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("stereo", "depth", "classification", "segmentation", "detection", "promptable_segmentation", "features",
         "multiview")  # fmt: skip
# primary metric: (key in metrics, +1 higher is better / -1 lower is better, scale to points)
PRIMARY = {
    "stereo": ("bad", -1, 1.0),
    "depth": ("rel_d1", 1, 100.0),
    "classification": ("top1", 1, 1.0),
    "segmentation": ("miou", 1, 1.0),
    "detection": ("box_map", 1, 1.0),
    "promptable_segmentation": ("miou_best", 1, 1.0),
    "features": ("top1", 1, 1.0),
    "multiview": ("d125", 1, 1.0),
}
TOL = 0.3  # points: "within noise" for the fp16 choice, and the loss flag of the report
COLUMN = {"upstream": "original fp32", "portable": "portable", "kernels": "kernels"}
DOCS_MODE = {  # docs/data mode -> column
    "upstream": "original fp32",
    "upstream-fp16": "original fp16",
    "portable": "portable",
    "kernels": "kernels",
}
COLS = ("original fp32", "original fp16", "portable", "kernels")
# docs mode per tier: one row per (mode, dataset)
MODE = {"upstream": "upstream", "portable": "portable", "kernels": "kernels"}
FP16 = ("upstream16", "upstreamac")
KEEP = ("commit", "gpu", "torch", "date", "command")
# a port that stands for an original without a build for this GPU (MegEngine 1.13: no sm_120): its rows fill the
# original's columns; the original's own (CPU) rows keep tier `<tier>-cpu`, shown in the report only
PORTS = {"crestereo-eth3d-pt": "crestereo-eth3d"}


def _jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _sd(raw: Path) -> list[dict[str, Any]]:
    out = []
    for r in summary_rows(raw):
        mets = r.get("metrics", {})
        out.append(r | {"metrics": mets})
    return out


def _chunks(recs: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["model"], r["tier"]].append(r)
    out = []
    for (ref, tier), rs in sorted(by.items()):
        last = rs[-1]
        row: dict[str, Any] = {"ref": ref, "task": task, "tier": tier, "mode": "upstream" if tier.startswith("upstream")
                               else ("portable" if tier == "portable32" else tier), "dtype": last.get("dtype"),
                               "how": last.get("how")}  # fmt: skip
        if task == "segmentation":
            row["dataset"] = last.get("dataset", "ade20k-val")
        else:
            row["dataset"] = f"imagenet-val-subset{last.get('subset')}-seed{last.get('seed')}"
        if last.get("backend"):
            row["note"] = last["backend"]
        row |= {k: last.get(k) for k in KEEP}
        good = {(r["start"], r["stop"]): r for r in rs if "error" not in r}  # last record of a chunk wins
        spans = sorted(good)
        if any(b0 < a1 for (_, a1), (b0, _) in pairwise(spans)):  # a rerun with another --chunk: images twice
            msg = f"{task} {ref} {tier}: overlapping chunks {spans}; drop the stale chunk size's records"
            raise SystemExit(msg)
        err = next((r["error"] for r in rs if "error" in r and (r["start"], r["stop"]) not in good), None)
        ok = list(good.values())
        if err:
            out.append(row | {"error": err, "n": sum(r.get("images", 0) for r in ok)})
            continue
        n = sum(r["images"] for r in ok)
        ms = sum(r["ms_per_image"] * r["images"] for r in ok) / n
        if task == "segmentation":
            inter = np.sum([r["intersect"] for r in ok], 0)
            union = np.sum([r["union"] for r in ok], 0)
            area = np.sum([r["label_area"] for r in ok], 0)
            seen = union > 0
            mets = {"miou": round(100 * float((inter[seen] / union[seen]).mean()), 2),
                    "aacc": round(100 * float(inter.sum() / area.sum()), 2)}  # fmt: skip
            if ref.startswith("sapiens"):
                mets["person_iou"] = round(100 * float(inter[1] / max(union[1], 1)), 2)
            nf = sum(r.get("nonfinite_images", 0) for r in ok)
            if nf:
                row["nonfinite_images"] = nf
        else:
            mets = {"top1": round(100 * sum(r["top1_correct"] for r in ok) / n, 2),
                    "top5": round(100 * sum(r["top5_correct"] for r in ok) / n, 2)}  # fmt: skip
        out.append(row | {"n": n, "metrics": mets, "ms_per_img": round(ms, 3),
                          "peak_alloc_mb": max(r["alloc_mb"] for r in ok),
                          "peak_reserved_mb": max(r["res_mb"] for r in ok)})  # fmt: skip
    return out


def _coco(raw: Path) -> list[dict[str, Any]]:
    out = []
    for r in _jl(raw / "coco_rows.jsonl"):
        if "ms" in r:
            r["ms_per_img"] = r["ms"]
        elif "ms_embed" in r:
            r["ms_per_img"] = round(r["ms_embed"] + r["ms_prompts"], 2)
        if "box_map" in r["metrics"] and r["task"] == "promptable_segmentation":
            r["dataset"] += "-text"  # SAM 3: class-name text prompts, mAP; not the point-prompt mIoU of the same images
        out.append(r)
    return out


def _features(raw: Path) -> list[dict[str, Any]]:
    out = []
    for r in _jl(raw / "features/features.jsonl"):
        t = r["tier"]
        row = {"ref": r["model"], "task": "features", "tier": t,
               "mode": "upstream" if t.startswith("upstream") else ("portable" if t == "portable32" else t),
               "dtype": r.get("dtype"), "how": r.get("how"),
               "dataset": f"imagenet-val-knn-train{r['train']}-test{r['test']}-seed{r['seed']}-k{r['k']}",
               **{k: r.get(k) for k in KEEP}}  # fmt: skip
        if "error" in r:
            out.append(row | {"error": r["error"]})
            continue
        out.append(row | {"n": r["test"], "metrics": {"top1": r["top1"]}, "ms_per_img": round(1e3 / r["img_s"], 3),
                          "peak_alloc_mb": r["alloc_mb"], "peak_reserved_mb": r["res_mb"]}
                   | ({"nonfinite_images": r["nonfinite"]} if r.get("nonfinite") else {}))  # fmt: skip
    return out


def _multiview(raw: Path) -> list[dict[str, Any]]:
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _jl(raw / "multiview/mv.jsonl"):
        last[r["ref"], r["mode"]] = r
    out = []
    for (ref, t), r in sorted(last.items()):
        be = r.get("backend", {})
        row = {"ref": ref, "task": "multiview", "tier": t,
               "mode": "upstream" if t.startswith("upstream") else ("portable" if t == "portable32" else t),
               "dtype": str(be.get("dtype", "")).removeprefix("torch.") or None, "how": be.get("how"),
               "dataset": f"dtu-scans{'-'.join(map(str, r.get('scans', [])))}-view{r.get('ref_view')}",
               **{k: r.get(k) for k in KEEP}}  # fmt: skip
        if be.get("note"):
            row["note"] = be["note"]
        if "error" in r:
            out.append(row | {"error": r["error"]})
            continue
        out.append(row | {"n": len(r.get("scans", [])), "metrics": r["metrics"], "ms_per_img": r["ms"],
                          "peak_alloc_mb": r["peak_alloc_mib"],
                          "peak_reserved_mb": r["peak_reserved_mib"]})  # fmt: skip
    return out


def _primary(task: str, row: dict[str, Any]) -> float | None:
    key, _, scale = PRIMARY[task]
    m = row.get("metrics") or {}
    if task == "segmentation" and "person_iou" in m:
        key = "person_iou"  # Sapiens: the person-vs-background protocol
    if task == "promptable_segmentation" and key not in m:
        key = "box_map"
    v = m.get(key)
    return None if v is None or "error" in row else float(v) * scale


def _ports(rows: list[dict[str, Any]]) -> None:
    has = {PORTS[r["ref"]] for r in rows if r["ref"] in PORTS}
    for r in rows:
        if r["ref"] in has and r["tier"].startswith("upstream"):
            r["tier"] += "-cpu"
        if r["ref"] in PORTS:
            r["ref"] = PORTS[r["ref"]]


def choose(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Set `column` / `mode` on the rows; map ref -> its fp16 original tier.

    `upstream16`, or `upstreamac` where the authors' own option fails (c-foundationstereo-s: "CUDA error: invalid
    argument").
    """
    _ports(rows)
    failed = {r["ref"] for r in rows if r["tier"] == "upstream16" and "error" in r}
    ok = {r["ref"] for r in rows if r["tier"] == "upstreamac" and "error" not in r}
    picked = {r["ref"]: "upstreamac" if r["ref"] in failed & ok else "upstream16" for r in rows if r["tier"] in FP16}
    for r in rows:
        fp16 = r["tier"] == picked.get(r["ref"])
        r["column"] = "original fp16" if fp16 else COLUMN.get(r["tier"])
        r["mode"] = "upstream-fp16" if fp16 else MODE.get(r["tier"], r["tier"])
    return picked


def losses(task: str, rows: list[dict[str, Any]]) -> list[str]:
    """Original fp16, portable and kernels rows that lose more than TOL points against original fp32."""
    key, sign, _ = PRIMARY[task]
    base = {(r["ref"], r["dataset"]): _primary(task, r) for r in rows if r["tier"] == "upstream"}
    out = []
    for r in rows:
        if r["column"] not in ("original fp16", "portable", "kernels"):
            continue
        p, b = _primary(task, r), base.get((r["ref"], r["dataset"]))
        if p is not None and b is not None and sign * (p - b) < -TOL:
            out.append(f"{task} {r['ref']} {r['tier']} {r['dataset']}: {key} {b:.2f} -> {p:.2f}")
    return out


def speed_rows(raw: Path, picked: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """The original speed rows of the fp32 tier and the chosen fp16 tier."""
    out = []
    for f in sorted((raw / "speed").glob("*.jsonl")):
        for r in _jl(f):
            r["ref"] = PORTS.get(r["ref"], r["ref"])
            t = r.get("tier")
            if t == "upstream":  # the docs' fp32 original convention
                out.append(r | {"mode": "upstream-default"})
            elif t == picked.get(r.get("task", ""), {}).get(r["ref"]):
                out.append(r)
    return out


def write(rows: dict[str, list[dict[str, Any]]], sp: list[dict[str, Any]], tasks: list[str] | tuple[str, ...]) -> None:
    """Replace docs/data/<task>_eval.jsonl (earlier rows of a model column not re-measured are kept), add speed rows."""
    for t in tasks:
        f = ROOT / "docs/data" / f"{t}_eval.jsonl"
        have = {(r["ref"], r["column"]) for r in rows[t] if r.get("column")}
        old = [r for r in _jl(f) if r["mode"] in DOCS_MODE and (r["ref"], DOCS_MODE[r["mode"]]) not in have]
        f.write_text("".join(json.dumps(r) + "\n" for r in old + rows[t]))
    sj = ROOT / "docs/data/speed.jsonl"
    key = ("ref", "mode", "tier", "input", "date")
    have_sp = {tuple(r.get(k) for k in key) for r in _jl(sj)}
    new = [r for r in sp if r.get("task") in tasks and tuple(r.get(k) for k in key) not in have_sp]
    with sj.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in new)
    print(f"wrote {sum(len(rows[t]) for t in tasks)} eval rows, appended {len(new)} speed rows", file=sys.stderr)


def main() -> None:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--md", default="")
    ap.add_argument("--task", action="append", choices=TASKS, help="--write only these tasks (default: all)")
    a = ap.parse_args()
    raw = Path(a.raw)
    sd = _sd(raw)
    rows = {
        "stereo": [r for r in sd if r["task"] == "stereo"],
        "depth": [r for r in sd if r["task"] == "depth"],
        "classification": _chunks(_jl(raw / "classification/cls.jsonl"), "classification"),
        "segmentation": _chunks(_jl(raw / "segmentation/seg.jsonl"), "segmentation"),
        "detection": [r for r in _coco(raw) if r["task"] == "detection"],
        "promptable_segmentation": [r for r in _coco(raw) if r["task"] == "promptable_segmentation"],
        "features": _features(raw),
        "multiview": _multiview(raw),
    }
    for r in (r for rs in rows.values() for r in rs if "error" in r):
        if "out of memory" in r["error"].lower() and not r["error"].startswith("OOM"):
            r["error"] = f"OOM, {r.get('dataset')}: {r['error']}"
    picked = {t: choose(rows[t]) for t in TASKS}
    md = []
    for t in TASKS:
        refs = sorted({r["ref"] for r in rows[t]})
        errs = [r for r in rows[t] if "error" in r]
        md += [f"## {t}: {len(refs)} models, {len(rows[t])} rows, {len(errs)} errors", ""]
        md += [f"- error {r['ref']} {r['tier']} {r.get('dataset')}: {r['error'][:200]}" for r in errs]
        md += [f"- loss > {TOL} pt: {x}" for x in losses(t, rows[t])]
        md += ["", f"| model | dataset | {' | '.join(COLS)} | fp16 how |", "|---" * 8 + "|"]
        cell: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        how: dict[str, str] = {}
        for r in rows[t]:
            if r.get("column"):
                p = _primary(t, r)
                cell[r["ref"], r.get("dataset", "")][r["column"]] = "err" if "error" in r else f"{p:.2f}"
                if r["column"] == "original fp16":
                    how[r["ref"]] = f"{r.get('how')} ({r.get('dtype')})"
        for (ref, ds), c in sorted(cell.items()):
            vals = [c.get(k, "") for k in COLS]
            md.append(f"| {ref} | {ds} | " + " | ".join(vals) + f" | {how.get(ref, '')} |")
        md.append("")
    sp = speed_rows(raw, picked)
    md += [f"## speed: {len(sp)} original rows (fp32 + chosen fp16)", ""]
    md += [f"- {r['ref']} {r['tier']} {r.get('input')}: " + (r["error"] if "error" in r else
           f"{r['ms']} ms, {r.get('peak_alloc_mb')} / {r.get('peak_reserved_mb')} MB") for r in sp]  # fmt: skip
    text = "\n".join(md)
    if a.md:
        Path(a.md).write_text(text + "\n")
    else:
        print(text)
    if a.write:
        write(rows, sp, a.task or TASKS)


if __name__ == "__main__":
    main()
