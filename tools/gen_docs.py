"""Generate the model pages of the docs (`docs/models/<task>.md`) and the core's model index from the cards.

The index `src/fastforks/<task>/known_models.toml` maps every model id outside the core wheel to its distribution
(dists.toml), so the core can name the package to install for a model that is not installed.

    uv run python tools/gen_docs.py          # write
    uv run python tools/gen_docs.py --check  # exit 1 if a page is out of date

tests/test_docs.py runs the check.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import goldens
from fastforks._core import licences, registry
from fastforks._core.sizes import card_size_lines
from fastforks._core.tasks import spec

if TYPE_CHECKING:
    from fastforks._core.cards import ModelCard, Variant

ROOT = Path(__file__).resolve().parents[1]
_COMMERCIAL = {"yes": "yes", "no": "**no**", "unclear": "unclear (treat as no)"}
# Measured accuracy / speed / memory per task: rows written by the experiment's eval scripts (docs/data/README.md).
# task -> dataset id -> column title (also the column order); stereo titles state the bad-pixel threshold of _BAD_THR.
_DATASETS = {
    "stereo": {
        "eth3d": "ETH3D two-view train, bad > 1 px % / EPE",
        "middH": "Middlebury 2014 trainingH, bad > 2 px % / EPE",
    },
    "depth": {
        "nyu": "NYUv2 test, AbsRel / delta1",
        "ibims": "iBims-1, AbsRel / delta1",
        "eth3d": "ETH3D two-view train left, AbsRel / delta1",
        "middH": "Middlebury 2014 trainingH left, AbsRel / delta1",
    },
    "segmentation": {
        "ade20k-val": "ADE20K val, every 4th image, mIoU / aAcc %",
        "coco-val2017-persons-every10": "COCO val2017 persons, every 10th image, person IoU %",
    },
    "detection": {"coco-val2017-every10": "COCO val2017, every 10th image, box mAP"},
    "promptable_segmentation": {
        "coco-val2017-every10": "COCO val2017, every 10th image, point prompts, mIoU best of 3 / single mask %",
        "coco-val2017-every10-text": "COCO val2017, every 10th image, class-name prompts, box / mask mAP",
    },
    "features": {
        "imagenet-val-knn-train10000-test5000-seed0-k20": "ImageNet-1k val, kNN top-1 % (k = 20, 10000 labelled / "
        "5000 test images)"
    },
    "multiview": {"dtu-scans1-9-24-33-75-view24": "DTU scans 1, 9, 24, 33, 75, view 24 + 7 sources, AbsRel / delta1 %"},
    "classification": {"imagenet-val-subset5000-seed0": "ImageNet-1k val, 5000 random images, top-1 / top-5 %"},
}
_BAD_THR = {"eth3d": 1.0, "middH": 2.0}  # stereo
# task or (task, dataset) -> ((metric, format, title suffix), ...), count key, noun; a metric with a suffix is optional:
# shown where the row has it, its suffix appended to the dataset title where a row of the table has it
_Metric = tuple[str, str, str]
_METRICS: dict[str | tuple[str, str], tuple[tuple[_Metric, ...], str, str]] = {
    "stereo": ((("bad", "{:.2f}", ""), ("epe", "{:.3f}", "")), "n_pairs", "pairs"),
    "depth": ((("rel_absrel", "{:.4f}", ""), ("rel_d1", "{:.4f}", "")), "n_images", "images"),
    "segmentation": ((("miou", "{:.2f}", ""), ("aacc", "{:.2f}", "")), "n", "images"),
    ("segmentation", "coco-val2017-persons-every10"): ((("person_iou", "{:.2f}", ""),), "n", "images"),
    "detection": ((("box_map", "{:.2f}", ""), ("mask_map", "{:.2f}", " / mask mAP")), "n", "images"),
    "promptable_segmentation": ((("miou_best", "{:.2f}", ""), ("miou_single", "{:.2f}", "")), "n", "images"),
    ("promptable_segmentation", "coco-val2017-every10-text"): (
        (("box_map", "{:.2f}", ""), ("mask_map", "{:.2f}", "")),
        "n",
        "images",
    ),
    "features": ((("top1", "{:.2f}", ""),), "n", "images"),
    "multiview": ((("absrel", "{:.4f}", ""), ("d125", "{:.2f}", ""), ("mae_mm", "{:.2f}", " / MAE mm")), "n", "scans"),
    "classification": ((("top1", "{:.2f}", ""), ("top5", "{:.2f}", "")), "n", "images"),
}
# task or (task, dataset) -> protocol sentence of the measured tables (task first, then each shown dataset)
_PROTOCOL = {
    "depth": "AbsRel / delta1 after a least-squares scale + shift fit (in disparity space for inverse-depth models), "
    "mean over images. ",
    ("segmentation", "ade20k-val"): "Each image at its own size, argmax of the logits. ",
    ("segmentation", "coco-val2017-persons-every10"): "Any body part counts as person; ground truth = the COCO person "
    "masks. ",
    "detection": "pycocotools COCOeval of the top 100 detections per image. ",
    ("promptable_segmentation", "coco-val2017-every10"): "One point per ground-truth instance (the interior point "
    "farthest from the border); IoU of the best-scored of 3 masks / of the single mask. ",
    ("promptable_segmentation", "coco-val2017-every10-text"): "One text prompt (the COCO class name) per class in the "
    "image; the top 100 instances go to pycocotools COCOeval. ",
    "features": "Two disjoint seeded subsets of the val folders; each image shorter side 256 (bicubic), centre crop "
    "224, the CLS / pooled feature; cosine kNN, vote weighted by similarity. ",
    "multiview": "640x512 crops, the 7 most co-visible views as sources; GT = depth rendered from the structured-light "
    "points. AbsRel / delta1 over GT pixels, after per-image median scaling unless the model takes the cameras "
    "(calibrated MVS: cameras + depth range 425-935 mm, error in mm). ",
    "classification": "fastforks: the model's own resize / crop on the GPU; original: the library's pretrained loader "
    "and its PIL eval transform. ",
}
# family -> what its original columns ran where that differs from the original code under Details (note column)
_ORIGINAL_IS = {
    "crestereo": "original = the community PyTorch port (MegEngine 1.13 has no sm_120 build)",
    "sam3": "original = transformers `Sam3Model`, the port this code follows",
}
_MODES = ("upstream", "upstream-default", "portable", "kernels")
_EVAL_MODES = ("upstream", "upstream-fp16", "portable", "kernels")
_EVAL_LABEL = {"upstream": "original fp32", "upstream-fp16": "original fp16"}
_PROTO = "Original = the original code named under Details, same checkpoint and protocol."


def _label(mode: str) -> str:
    """User-facing name of a speed/eval mode ('upstream' stays the internal key)."""
    return {"upstream": "original", "upstream-default": "original (fp32)"}.get(mode, mode)


# eval dataset id -> training-data id (_core/datasets.toml) of the same public images (NYU / iBims: no id, none known)
_EVAL_TRAIN_ID = {"eth3d": "eth3d", "middH": "middlebury"}
_DAGGER = "\u2020"
_NOTE = {
    _DAGGER: "trained on this dataset's public images; not a zero-shot result",
    "?": "possibly in the training data",
}
_LEGEND = {
    _DAGGER: f"{_DAGGER} trained or fine-tuned on this dataset's public images; not a zero-shot result.",
    "?": "? training data not stated; overlap with this dataset's public images cannot be excluded.",
}


def _eval_rows(task: str) -> dict[str, list[dict]]:
    path = ROOT / "docs" / "data" / f"{task}_eval.jsonl"
    rows: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines() if path.is_file() else []:
        row = json.loads(line)
        if not row.get("cold"):  # cold-start rows (`cold: true`, no timings) are read by _startup
            rows.setdefault(row["ref"], []).append(row)
    return rows


def _speed_rows() -> dict[str, list[dict]]:
    """docs/data/speed.jsonl by model id (without revision)."""
    path = ROOT / "docs" / "data" / "speed.jsonl"
    rows: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines() if path.is_file() else []:
        row = json.loads(line)
        if not row.get("cold") or "ms" in row:  # speed_check rows: a fresh process, warm timings included
            rows.setdefault(row["ref"], []).append(row)
    return rows


NM = "not measured"
DASH = "\u2013"
_SPEED_MODES = ("upstream-default", "upstream", "portable", "kernels")  # fixed line order inside a table cell


_PLAIN_NOTES = {"original mixed_precision flag", "original default fp32"}  # convention, said once in the intro


def _note(row: dict) -> str:
    """The row's note without the convention parts (the intro says what original and original (fp32) are)."""
    text = "; ".join(p for p in row.get("note", "").split("; ") if p not in _PLAIN_NOTES)
    return text.replace("fork-port: ", "").replace("golden", "reference").replace("@1", "")


def _peak(row: dict | None) -> str:
    if row is None or "peak_alloc_mb" not in row:
        return DASH
    res = row.get("peak_reserved_mb")  # None: the framework has no reserved figure (TensorFlow)
    return f"{row['peak_alloc_mb']} / " + (DASH if res is None else str(res))


def _ms_str(ms: object) -> str:
    """A time in ms to 3 significant digits (all digits from 1000 ms): `3.48`, `20.1`, `214`."""
    if not isinstance(ms, int | float) or isinstance(ms, bool) or not math.isfinite(ms) or ms == 0:
        return str(ms)
    return f"{ms:.{max(0, 2 - math.floor(math.log10(abs(ms))))}f}"


def _ms_cell(row: dict | None) -> str:
    """A speed row's time; an fp32 original (fp16 unsupported or non-finite) is marked."""
    if row is None:
        return DASH
    fp32 = row["mode"] == "upstream" and row.get("protocol", "").endswith("float32")
    return f"{_ms_str(row['ms'])} (fp32)" if fp32 else _ms_str(row["ms"])


def _faster(*rows: dict | None) -> dict | None:
    """The faster of the timed rows (speed-ups use the faster of original fp16 / fp32: a conservative headline)."""
    return min((r for r in rows if r), key=lambda r: r["ms"], default=None)


def _orig(fp16: dict | None, fp32: dict | None) -> dict | None:
    """The original row a speed-up uses; an fp32 row with a `dtype` is no stand-in for a missing fp16 original."""
    return _faster(fp16, None if fp16 is None and fp32 and fp32.get("dtype") else fp32)


_COMPILE_LABEL = {"compile": "portable + torch.compile", "tensorrt": "TensorRT fp16"}  # speed_compile.jsonl modes


@functools.cache
def _compile_rows() -> dict[tuple[str, str], dict]:
    """docs/data/speed_compile.jsonl: the latest row per (model id, mode)."""
    path = ROOT / "docs" / "data" / "speed_compile.jsonl"
    return {(r["ref"], r["mode"]): r for r in map(json.loads, path.read_text().splitlines() if path.is_file() else [])}


def _compile_lines(model_id: str, size: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """The compile / TensorRT cell lines (label, time, memory) of one variant and input, and their one-time setup."""
    lines, setup = [], []
    for mode, label in _COMPILE_LABEL.items():
        r = _compile_rows().get((model_id, mode))
        if r is None or r["input"] != size:
            continue
        if "fails" in r:
            lines.append((label, f"fails: {_fail_reason(r['fails'])}", DASH))
            continue
        engine = mode == "tensorrt" and "device_mb" in r
        lines.append((label, _ms_str(r["ms"]), f"{r['device_mb']} (engine)" if engine else _peak(r)))
        first = r["compile_s"] if mode == "compile" else r["onnx_export_s"] + r["engine_build_s"]
        setup.append(f"{label} {first:.0f} s" + (" (cold Inductor cache)" if mode == "compile" else " (ONNX + engine)"))
    return lines, setup


_COMPUTE_KEYS = ("max_disp", "iters", "valid_iters", "resize")  # variant params that change the work per call


def _fmt_param(x: object) -> str:
    """A variant parameter for a label: lists as `a x b`, e.g. the mmseg resize [`mmseg`, 2048, 512] as `2048x512`."""
    if isinstance(x, list | tuple):
        nums = [str(i) for i in x if isinstance(i, int)]
        return "x".join(nums) if nums else " ".join(map(str, x))
    return str(x)


def setting(variants: list, v: Variant) -> str:
    """The compute setting that tells `v` from the other variants of its family, e.g. `max_disp 768`; else ''."""
    parts = [
        f"{k} {_fmt_param(v.params[k])}"
        for k in _COMPUTE_KEYS
        if k in v.params and len({repr(w.params.get(k)) for w in variants}) > 1
    ]
    return ", ".join(parts)


def labelled(variants: list, v: Variant) -> str:
    """`id`, followed by the compute setting where variants of the family differ in one."""
    return f"`{v.id}`" + (f" ({setting(variants, v)})" if setting(variants, v) else "")


def same_as(ids: list[str], cells: dict[str, object]) -> dict[str, str]:
    """{id: the earlier id with identical measured cells}: variants of identical network and settings share a row."""
    first: dict[str, str] = {}
    out = {}
    for i in ids:
        key = repr(cells[i])
        if key in first and cells[i]:
            out[i] = first[key]
        else:
            first.setdefault(key, i)
    return out


def _same_note(dups: dict[str, str]) -> list[str]:
    """One note per shown variant: the variants whose identical measurements are left out."""
    by: dict[str, list[str]] = {}
    for i, first in dups.items():
        by.setdefault(first, []).append(i)
    return [
        f"- {_ids_of(ids)}: same network and settings as `{first}`, measured once (rows not repeated)."
        for first, ids in by.items()
    ]


def _ratio(slow: dict | None, fast: dict | None) -> str:
    return f"{slow['ms'] / fast['ms']:.1f}x" if slow and fast else DASH


def _md_table(head: list[str], rows: list[list[str]], keep: int = 2) -> list[str]:
    """Markdown table lines; a column whose cells are all a dash (nothing measured for the task) is left out.

    The first `keep` columns (model, commercial use) always stay.
    """
    use = [i for i in range(len(head)) if i < keep or any(r[i] != DASH for r in rows)]
    lines = ["| " + " | ".join(head[i] for i in use) + " |", "|" + "---|" * len(use)]
    return lines + ["| " + " | ".join(r[i] for i in use) + " |" for r in rows]


def _scene(scene: str) -> str:
    """Short scene name of a speed row: 'KITTI 2015', 'Middlebury 2014 half', 'synthetic'."""
    if scene.startswith("KITTI 2015"):
        return "KITTI 2015"
    if scene.startswith("Middlebury 2014 trainingH"):
        return "Middlebury 2014 half"
    if scene.startswith("synthetic"):
        return "synthetic"
    return scene.split(",", maxsplit=1)[0]


def _pixels(size: str) -> int:
    h, w = size.split("x")
    return int(h) * int(w)


def _ids_of(ids: list[str]) -> str:
    return ", ".join(f"`{i}`" for i in ids)


_STARTUP_HEAD = (
    "load: from the load call until the model is on the GPU; first call: the whole first inference call, including "
    "one normal inference (see the speed table for the warm time)."
)
_LATER = {
    "stereo": "994x1482",
    "segmentation": "1024x2048",
    "promptable_segmentation": "1024x1024",
}  # not the sweep's first size
_WARM = "\u2020"
_WARM_NOTE = (
    f"{_WARM} measured in the speed sweep after the portable load (or an earlier size) in the same process: "
    "file cache and kernels are warm, so the time is lower than a fresh start."
)


def _cold_rows() -> dict[str, list[dict]]:
    """Cold-start rows (`cold: true`) of docs/data/speed.jsonl by id; none until the release pass records them."""
    path = ROOT / "docs" / "data" / "speed.jsonl"
    out: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines() if path.is_file() else []:
        row = json.loads(line)
        if row.get("cold"):
            out.setdefault(row["ref"], []).append(row)
    return out


def _startup(task: str, model_id: str, rows: list[dict]) -> dict[str, tuple[dict, str, str]]:
    """Startup row per mode: `{mode: (row, load mark, first-call mark)}`.

    One row per (id, mode), independent of the table size: a `cold: true` row if there is one, else the row at the
    task's standard size, else the largest size with a first call (the input is then shown). A mark flags a time
    measured warm (kernels load after the portable one; a size after an earlier one in the same process).
    """
    all_rows = [r for r in [*rows, *_cold_rows().get(model_id, [])] if "first_call_s" in r]
    std = _std_input(task, {})
    out = {}
    for mode in ("upstream", "portable", "kernels"):
        mine = [r for r in all_rows if r["mode"] == mode or (mode == "upstream" and r["mode"] == "upstream-default")]
        if mode == "upstream" and any(r["mode"] == "upstream" for r in mine):
            mine = [r for r in mine if r["mode"] == "upstream"]
        if not mine:
            continue
        cold = [r for r in mine if r.get("cold")]
        r = (cold or [x for x in mine if x["input"] == std] or [max(mine, key=lambda x: _pixels(x["input"]))])[0]
        warm = not r.get("cold") and mode != "upstream"
        out[mode] = (
            r,
            _WARM if warm and mode == "kernels" else "",
            _WARM if warm and r["input"] == _LATER.get(task) else "",
        )
    return out


def _startup_notes(task: str, measured: list, rows: dict[str, list[dict]]) -> list[str]:
    """Load / first-call times per variant and the timing protocol."""
    notes = []
    used = False
    for v in measured:
        by = _startup(task, v.id, rows[v.id])
        if by:
            parts = ", ".join(
                f"{_label(m)} {r.get('load_s', DASH)}{lm} s / {r['first_call_s']}{fm} s"
                + (f" ({r['input']})" if r["input"] != _std_input(task, {}) else "")
                for m, (r, lm, fm) in by.items()
            )
            notes.append(f"- `{v.id}` load / first call: {parts}")
            used = used or any(lm or fm for _, lm, fm in by.values())
    if used:
        notes.append(f"- {_WARM_NOTE}")
    protos: dict[str, list[str]] = {}
    for v in measured:
        sample = next((r for r in rows[v.id] if not r["mode"].startswith("upstream")), rows[v.id][0])
        protos.setdefault(sample["protocol"], []).append(v.id)
    for proto, ids in protos.items():
        who = "" if len(protos) == 1 else f"{_ids_of(ids)}: "
        notes.append(f"- Protocol, {who}{proto}")
    return notes


def _env_note(rows: dict[str, list[dict]]) -> str:
    """GPU / torch of all speed rows of a task, for the page intro."""
    rs = [r for v in rows.values() for r in v]
    gpus = sorted({r["gpu"].removeprefix("NVIDIA GeForce ") for r in rs})
    torches = sorted({r["torch"] for r in rs if r.get("torch")})
    return ", ".join(gpus) + "".join(f", torch {t}" for t in torches)


def _speed(card: ModelCard, rows: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    """Speed and peak memory: (the table, the notes); one table row per variant and input, backends as cell lines."""
    measured = [v for v in card.variants if v.id in rows]
    if not measured:
        return [f"Speed and memory {NM} for: {_ids_of([v.id for v in card.variants])}."], []
    lines = [
        "| variant | input | backend | time (ms) | peak memory (MiB, allocated / reserved) |",
        "|---|---|---|---|---|",
    ]
    measured_all = list(card.variants)
    notes = []
    given: dict[tuple[str, str], list[str]] = {}
    dups = same_as(
        [v.id for v in measured],
        {v.id: sorted((r["mode"], r["input"], r["ms"]) for r in rows[v.id]) for v in measured},
    )
    notes += _same_note(dups)
    for v in measured:
        if v.id in dups:
            continue
        sizes = sorted({r["input"] for r in rows[v.id]}, key=_pixels)
        std = _std_input(card.task, {})
        for size in [std] if std in sizes else sizes[-1:] if std != _OWN else sizes:
            by = {r["mode"]: r for r in rows[v.id] if r["input"] == size}
            modes = [m for m in _SPEED_MODES if m in by or m in ("upstream", "portable", "kernels")]
            cells = [by.get(m) for m in modes]
            scene = _scene(next(iter(by.values())).get("scene", ""))
            inp = f"{scene} ({size})" if scene else size
            extra, setup = _compile_lines(v.id, size)
            names = "<br>".join([*(_label(m) for m in modes), *(e[0] for e in extra)])
            ms = "<br>".join([*(_ms_str(r["ms"]) if r else DASH for r in cells), *(e[1] for e in extra)])
            mem = "<br>".join([*(_peak(r) for r in cells), *(e[2] for e in extra)])
            lines.append(f"| {labelled(measured_all, v)} | {inp} | {names} | {ms} | {mem} |")
            if setup:
                notes.append(f"- `{v.id}` {size}, first call: {', '.join(setup)}.")
            for m in modes:
                if (r := by.get(m)) and _note(r):
                    given.setdefault((v.id, _note(r)), []).append(f"{_label(m)} {size}")
    notes += [f"- `{vid}` {', '.join(where)}: {text}" for (vid, text), where in given.items()]
    notes += _startup_notes(card.task, measured, rows)
    missing = [v.id for v in card.variants if v.id not in rows]
    if missing:
        notes.append(f"- Speed and memory {NM} for: {_ids_of(missing)}.")
    return lines, notes


def parity_text(card: ModelCard, tol: dict) -> str:
    """The checked precision per family: CPU float32, and the GPU check at the card's default dtypes."""
    px = card.task == "stereo"  # disparity in pixels; every other output is compared in its own units

    def lim(x: float) -> str:
        return f"{x * 100:g} %" if tol.get("relative") else f"{x} px" if px else f"{x}"

    dts = {"portable": card.portable_cuda_dtype, "kernels": card.kernels_dtype}
    gpu = (
        f"GPU, {dts['portable']} (the default), portable and kernels"
        if dts["portable"] == dts["kernels"]
        else f"GPU, portable {dts['portable']} and kernels {dts['kernels']} (the defaults)"
    )
    of = " of the reference's median |value|" if tol.get("relative") else ""
    of += "" if px else f" (absolute error of the output `{card.outputs[0]}`)"
    return (
        f"CPU, float32: max |diff| <= {tol['cpu_atol']} vs the original. "
        f"{gpu}: mean |diff| <= {lim(tol['gpu_mean'])} and p99 <= {lim(tol['gpu_p99'])}{of}."
    )


def _parity(card: ModelCard) -> list[str]:
    """Parity with the original code: the family's tolerances and the recorded GPU error of every kernels variant."""
    store = goldens.load(card.family, card.task)
    tol = store["tolerance"]
    lines = [
        "",
        "### Parity",
        "",
        parity_text(card, tol)
        + (
            f" On real pairs up to 720x1280 the limits are mean {tol['real_gpu_mean']} / "
            f"p99 {tol['real_gpu_p99']} (precision-sensitive there, see Limitations)."
            if "real_gpu_mean" in tol
            else ""
        ),
        "",
        "| variant | kernels error vs original, mean / p99 | GPU, torch | kernels used |",
        "|---|---|---|---|",
    ]
    for v in card.variants:
        rec = store["validated"].get(v.ref)
        if "kernels" not in card.modes:
            lines.append(f"| `{v.id}` | no kernels backend | - | - |")
        elif rec is None or "mean_px" not in rec:
            lines.append(f"| `{v.id}` | {NM} | - | - |")
        else:
            gpu = rec["gpu"].removeprefix("NVIDIA GeForce ") + ", torch " + rec["torch"]
            kernels = ", ".join(f"`{k}`" for k in rec["kernels"])
            lines.append(f"| `{v.id}` | {rec['mean_px']} / {rec['p99_px']} | {gpu} | {kernels} |")
    return lines


def _gaps(task: str) -> list[dict[str, str]]:
    """Per family of `task`: what the model page shows as 'not measured' (the GPU to-do list)."""
    speed, acc = _speed_rows(), _eval_rows(task)
    out = []
    for card in {c.family: c for c, _ in registry.cards(task).values()}.values():
        ids = [v.id for v in card.variants]
        store = goldens.load(card.family, card.task)
        sp = [r for i in ids for r in speed.get(i, [])]
        ac = [r for i in ids for r in acc.get(i, [])]
        n_sp = sum(i in speed for i in ids)
        n_ac = sum(i in acc for i in ids)
        n_par = (
            sum("mean_px" in store["validated"].get(v.ref, {}) for v in card.variants)
            if "kernels" in card.modes
            else None
        )
        out.append(
            {
                "family": f"{task}/{card.family}",
                "variants": str(len(ids)),
                "speed": f"{n_sp}/{len(ids)}" if n_sp < len(ids) else "ok",
                "speed upstream": "ok" if any(r["mode"] == "upstream" for r in sp) else "missing",
                "peak mem upstream": "ok"
                if any(r["mode"] == "upstream" and "peak_alloc_mb" in r for r in sp)
                else "missing",
                "peak reserved": "ok"
                if all("peak_reserved_mb" in r for r in sp if not r["mode"].startswith("upstream")) and sp
                else "missing",
                "accuracy": f"{n_ac}/{len(ids)}" if n_ac < len(ids) else "ok",
                "accuracy upstream": "ok" if any(r["mode"] == "upstream" for r in ac) else "missing",
                "kernels parity": "no kernels"
                if n_par is None
                else (f"{n_par}/{len(ids)}" if n_par < len(ids) else "ok"),
                "limitations": "ok" if card.limitations else "missing",
            }
        )
    return out


def _accuracy(task: str, by: dict[tuple[str, str], dict], mode: str, dataset: str) -> str:
    r = by.get((mode, dataset))
    if r is None:
        return ""
    metrics, count, noun = _METRICS.get((task, dataset)) or _METRICS[task]
    err = r.get("error", "")
    if err.startswith("OOM"):  # the original at this size does not fit the GPU
        return "out of memory"
    if err.startswith("call: ") and mode.startswith("upstream"):  # the original itself fails on this data
        return "fails (" + ": ".join(err.split(": ")[1:3]).lstrip("_") + ")"
    if err and not err.startswith("non-finite"):
        msg = f"{r['ref']} {mode} {dataset}: no cell for the error {err!r}"
        raise ValueError(msg)
    m = r.get("metrics") or r
    n, full = r.get(count), by.get(("portable", dataset), r).get(count)
    cell = " / ".join(fmt.format(m[k]) for k, fmt, opt in metrics if not opt or k in m)
    cell += f" ({n} {noun})" if n and full and n < full else ""
    return cell + (" (NaN / inf output)" if err else "")


def _overlap(card: ModelCard, v: Variant, dataset: str) -> str:
    """Mark of an accuracy cell: dagger if the training data holds the eval images, `?` if unstated, else ''."""
    train = _EVAL_TRAIN_ID.get(dataset)
    if train is None:
        return ""
    ids = card.variant_licence(v).training_data
    return _DAGGER if train in ids else "?" if licences.UNSTATED in ids else ""


def _where(gpu: str, torch: str, commit: str, date: str) -> str:
    """Where a row set was measured; rows without a commit (history of the development repository) omit it."""
    return f"{gpu}, torch {torch}, " + (f"fastforks {commit}, " if commit else "") + date


def _span(days: set[str]) -> str:
    """`2026-11-05`, or `2026-11-05 to 2026-11-06` for rows of several days."""
    return " to ".join(dict.fromkeys((min(days), max(days))))


def _suffixes(task: str, dataset: str, rows: list[dict]) -> list[str]:
    """Title suffixes of the optional metrics of `dataset` that one of `rows` has."""
    metrics = (_METRICS.get((task, dataset)) or _METRICS[task])[0]
    return [opt for k, _, opt in metrics if opt and any(k in (r.get("metrics") or r) for r in rows)]


def _measured(task: str, card: ModelCard, rows: dict[str, list[dict]]) -> list[str]:
    """Accuracy (original fp32 / fp16, portable, kernels) of the measured variants (speed: the speed tables)."""
    measured = [v for v in card.variants if v.id in rows]
    if not measured:
        return []
    mine = [r for v in measured for r in rows[v.id]]
    titles = _DATASETS[task]
    sets = [d for d in titles if any(r["dataset"] == d for r in mine)]
    meta: dict[str, set[tuple[str, str]]] = {}  # mode -> {(gpu, torch)}
    commits: dict[str, set[str]] = {}  # mode -> fastforks commits (a campaign may move to a fix commit)
    days: dict[str, set[str]] = {}  # mode -> dates (a run may cross midnight)
    for r in mine:
        if not r["mode"].startswith("upstream"):
            meta.setdefault(r["mode"], set()).add((r["gpu"], r["torch"]))
            commits.setdefault(r["mode"], set()).add(r["commit"])
            days.setdefault(r["mode"], set()).add(r["date"][:10])
    thr = {r["bad_thr"] for r in mine if "bad_thr" in r and r["bad_thr"] != _BAD_THR.get(r["dataset"])}
    if any(len(m) > 1 for m in meta.values()) or thr or len(sets) != len({r["dataset"] for r in mine}):
        msg = f"{card.family}: rows of a mode need one gpu / torch, known datasets and thresholds: {meta} {thr}"
        raise ValueError(msg)
    per_mode = {
        mode: (*next(iter(m)), " / ".join(sorted(commits[mode])), _span(days[mode])) for mode, m in meta.items()
    }
    per_mode = {mode: w for mode, w in per_mode.items() if mode in _MODES}  # the modes the table shows
    if len(set(per_mode.values())) <= 1:
        where = _where(*next(iter(per_mode.values()))) + ". " if per_mode else ""
    else:  # e.g. a kernels backend re-measured after a kernel change
        where = "; ".join(f"{mode}: {_where(*per_mode[mode])}" for mode in _MODES if mode in per_mode) + ". "
    shown = [m for m in _EVAL_MODES if any(r["mode"] == m for r in mine)]
    lines = [
        "",
        "### Measured",
        "",
        _PROTOCOL.get(task, "") + "".join(_PROTOCOL.get((task, d), "") for d in sets) + _PROTO,
        "",
        f"| variant | dataset | {' | '.join(_EVAL_LABEL.get(m, m) for m in shown)} | note |",
        f"|---|---|{'---|' * len(shown)}---|",
    ]
    for v in measured:
        by = {(r["mode"], r["dataset"]): r for r in rows[v.id]}
        for d in sets:
            if not any(k[1] == d for k in by):
                continue
            title = titles[d] + "".join(_suffixes(task, d, [r for r in rows[v.id] if r["dataset"] == d]))
            cells = [_accuracy(task, by, m, d) for m in shown]
            note = "; ".join(n for n in (_NOTE.get(_overlap(card, v, d), ""), _ORIGINAL_IS.get(card.family, "")) if n)
            lines.append(f"| `{v.id}` | {title} | {' | '.join(cells)} | {note} |")
    lines += ["", "Measured on " + where.rstrip(". ") + "."] if where else []
    return lines


def _default_gpu(speed: dict[str, list[dict]]) -> str:
    """The GPU most speed rows were measured on."""
    gpus = [r["gpu"].removeprefix("NVIDIA GeForce ") for rs in speed.values() for r in rs]
    return statistics.mode(gpus) if gpus else "the benchmark GPU"


def _name(v: Variant) -> str:
    """The name shown in the docs: without the tag where the default tag resolves (`alexnet`), else the id."""
    return v.id.split(".", 1)[0] if v.default_tag else v.id


def _option(card: ModelCard, name: str) -> str:
    """`name` = default with its bounds and what it changes if not the outputs (`ModelCard.option_bounds`)."""
    notes = [b for b in (card.option_bounds(name), card.option_effect.get(name, "")) if b]
    notes = [f"{n} only" if n in {"speed", "precision"} else n for n in notes]
    notes += [f"also `{a}`" for a, n in card.option_aliases.items() if n == name]
    return f"`{name}` = {card.options[name]}" + (f" ({'; '.join(notes)})" if notes else "")


def _output(card: ModelCard, name: str) -> str:
    """An output name of a card; those that are not a field of the result live in its `extras`."""
    return name if name in spec(card.task).standard else f"{name} (in `extras`)"


def _variant(v: Variant) -> str:
    return f"`{v.id}`" + (f" (default of `{v.id.split('.', 1)[0]}`)" if v.default_tag else "")


def _data(ids: tuple[str, ...]) -> str:
    return ", ".join(
        "not stated by the authors" if i == licences.UNSTATED else licences.datasets()[i].name for i in ids
    )


def data_text(ids: tuple[str, ...], note: str | None) -> str:
    """Training data line: the dataset names with the card's note; the note alone when no dataset is named."""
    if not note:
        return _data(ids)
    return note if ids == (licences.UNSTATED,) else f"{_data(ids)} ({note})"


def _plain() -> dict[str, str]:
    """docs/data/plain.toml: family -> plain-language description."""
    path = ROOT / "docs" / "data" / "plain.toml"
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def _first_sentences(text: str, n: int = 2) -> str:
    """The first `n` sentences of a card summary."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return " ".join(parts[:n])


def _licence_line(card: ModelCard) -> str:
    """One line: commercial use per group of variants with the reason (from the card's status)."""
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for v in card.variants:
        st = card.status(v)
        groups.setdefault((st.commercial, tuple(st.reasons)), []).append(v.id)
    parts = []
    for (com, reasons), ids in groups.items():
        text = _COMMERCIAL[com] + (f" ({'; '.join(reasons)})" if reasons else "")
        parts.append(text if len(groups) == 1 else f"{_ids_of(ids)}: {text}")
    return "Commercial use: " + "; ".join(parts) + "."


def _std_input(task: str, _speed: dict[str, list[dict]] | None = None) -> str:
    """The task's one standard input (_STD_SIZE); multiview models each run their own number of views and size."""
    return _OWN if task == "multiview" else _STD_SIZE.get(task, _STD_DEFAULT)


def _at(rows: list[dict], size: str) -> str:
    """The input to show for one model: the standard size, else the largest input its kernels row has."""
    ins = {r["input"] for r in rows if r["mode"] == "kernels"}
    return size if size in ins or not ins else max(ins, key=_pixels)


def _mem_ratio(mem: object, *originals: dict | None) -> str:
    """`, 0.62x`: kernels peak reserved / the smallest original peak reserved (conservative), else empty."""
    orig = [r["peak_reserved_mb"] for r in originals if r and isinstance(r.get("peak_reserved_mb"), int | float)]
    if not orig or not isinstance(mem, int | float) or isinstance(mem, bool):
        return ""
    x = mem / min(orig)
    return f", {x:.2g}x" if x < 1 else f", {x:.1f}x"


def _summary_table(task: str, speed: dict[str, list[dict]], rows: dict[str, list[dict]], size: str) -> list[str]:
    """One row per model id; speed cells are the speed rows at `size`, else the fixed-size table's cells."""
    fixed = _fixed_sizes()
    ov = {r["id"]: r for r in _overview_rows(task, fixed, speed, rows)}
    acc_col = _OVERVIEW_ACC.get(task)
    head = ["model", "commercial"] + ([f"{acc_col[2]} \u2193"] if acc_col else [])
    head += ["kernels time (ms),<br>speed-up vs original", "kernels peak memory (MiB, reserved),<br>ratio to original"]
    body: list[list[str]] = []
    marks = set()
    for card, v in registry.cards(task).values():
        r = ov[v.id]
        at = (
            next((x["input"] for x in speed.get(v.id, []) if x["mode"] == "kernels"), size)
            if size == _OWN
            else _at(speed.get(v.id, []), size)
        )
        by = {x["mode"]: x for x in speed.get(v.id, []) if x["input"] == at}
        mx, orig = by.get("kernels"), _orig(by.get("upstream"), by.get("upstream-default"))
        cell = fixed.get((v.id, "kernels"), {}).get(at)
        if mx:
            ms, mem = mx["ms"], mx.get("peak_reserved_mb", _DASH)
        elif isinstance(cell, tuple):
            ms, mem = cell[0], cell[2]
        else:
            ms, mem = _DASH, _DASH
        speedup = f", {orig['ms'] / mx['ms']:.1f}x" if mx and orig else ""
        mem = f"{mem}{_mem_ratio(mem, by.get('upstream'), by.get('upstream-default'))}"
        cells = [
            f"[`{_name(v)}`](#{card.family})"
            + (f" ({setting(card.variants, v)})" if setting(card.variants, v) else ""),
            card.status(v).commercial,
        ]
        if acc_col:
            err = _DASH if r["error"] is None else r["fmt"].format(r["error"]) + (f" {r['mark']}" if r["mark"] else "")
            marks.add(r["mark"])
            cells.append(err)
        body.append([*cells, str(ms) + speedup + (f" ({at})" if size not in {at, _OWN} else ""), mem])
    lines = _md_table(head, body)
    for m in (_DAGGER, "?"):
        lines += ["", _LEGEND[m]] if m in marks else []
    return lines


TASK_IO = {
    "stereo": "Input: a rectified left / right image pair. Output: a disparity map (pixels, left view).",
    "depth": (
        "Input: one RGB image. Output: a depth or inverse-depth map. `kind` (a card field, `result.kind`) says what "
        "the numbers mean: `metric` = metres; `scale_invariant`, `affine_invariant` and `affine_invariant_inverse` "
        "are relative (unknown scale or shift, not metres). Metric models that use the focal length take "
        "`model(image, intrinsics=K)` (K `[3, 3]`, pixels of the input): `metric3d-vit-large` and `da3metric-large` "
        "need it for correct metres (`da3metric-large`: metres = output x focal / 300; without `intrinsics=` a 300 px "
        "focal length is assumed, wrong for a real camera). `unidepth-v2-vitl14`, `unik3d-vitl`, `depth-pro` and "
        "MoGe estimate the camera themselves (given `intrinsics=` replace `fx`); `zoedepth-nk` and the relative "
        "models (`da3mono-large` too) take none."
    ),
    "classification": "Input: one RGB image. Output: ImageNet-1k class logits.",
    "detection": "Input: one RGB image. Output: boxes, class labels, scores and instance masks.",
    "features": "Input: one RGB image. Output: dense patch features and a pooled image feature.",
    "multiview": (
        "Input: several images of one scene (posed and calibrated for multi-view stereo, unposed otherwise). "
        "Output: depth maps and, where the model predicts them, cameras and 3-D points."
    ),
    "segmentation": "Input: one RGB image. Output: per-pixel class labels (logits with `logits=True`).",
    "promptable_segmentation": "Input: one image plus a point, box or text prompt. Output: masks with quality scores.",
}


_TITLE = {"depth": "Monocular depth"}  # page titles that differ from the task name


def _title(task: str) -> str:
    return _TITLE.get(task, task.capitalize().replace("_", " "))


def render(task: str) -> str:
    """Markdown page of all models of `task`."""
    families = {card.family: card for card, _ in registry.cards(task).values()}
    rows, speed = _eval_rows(task), _speed_rows()
    std = _std_input(task, speed)
    basis = {_OWN: "each model's own benchmark input (see Benchmarks)"}.get(std, f"one input, {std}")
    gpu = _env_note({i: rs for i, rs in speed.items() if i in {v.id for c in families.values() for v in c.variants}})
    lines = [
        f"# {_title(task)} models (one image)" if task == "depth" else f"# {_title(task)} models",
        "",
        "<!-- Generated by tools/gen_docs.py from the model cards; do not edit. -->",
        "",
        (
            f"{len(registry.cards(task))} {task.replace('_', ' ')} models. Each section says what the model is, "
            "whether it may be used commercially and how fast it is; licences and checks are under Details."
        ),
        "",
        TASK_IO[task],
        "",
        (
            f"Measured on {gpu or _default_gpu(speed)}; `fastforks bench MODEL --size HxW` measures yours. "
            f"The summary uses {basis}: kernels time with its speed-up over the faster original run, and kernels "
            "peak reserved memory with its ratio to the smaller original one. "
            "Where measured (one variant per family), a variant's table also lists portable under `torch.compile` "
            "and the fp16 TensorRT engine of portable ([export](../export.md); memory: the engine's device memory). "
            f"{_DASH} = not measured."
            + (f" The accuracy column is lower-is-better ({_OVERVIEW_ACC[task][2]})." if task in _OVERVIEW_ACC else "")
        ),
        "",
        *_summary_table(task, speed, rows, std),
    ]
    plain = _plain()
    used = {i for card, v in registry.cards(task).values() for i in card.status(v).training_data} - {licences.UNSTATED}
    for card in families.values():
        lic = card.licence
        table, notes = _speed(card, speed)
        lines += [
            "",
            f'<a id="{card.family}"></a>',
            "",
            f"## {card.name}",
            "",
            plain.get(card.family) or _first_sentences(card.summary),
            "",
            _licence_line(card),
            "",
            *table,
            "",
            "<details><summary>Details</summary>",
            "",
            f"- Original code: <{card.upstream}> at `{card.upstream_commit[:12]}`",
            f"- Code licence: {lic.code} ({lic.code_class}); files: {', '.join(f'`{f}`' for f in lic.files)}",
            f"- Weights licence: {licences.describe_weights(lic.weights, lic.code)}",
            *(
                f"  - `{v.id}`: {licences.describe_weights(vl.weights, vl.code)}"
                for v in card.variants
                if v.licence and (vl := card.variant_licence(v)) and vl.weights != lic.weights
            ),
            *(
                [f"- Licence evidence: {lic.evidence if ' ' in lic.evidence else f'<{lic.evidence}>'}"]
                if lic.evidence
                else []
            ),
            f"- Training data: {data_text(lic.training_data, lic.training_note)}",
            *(
                f"  - `{v.id}`: {data_text(vl.training_data, vl.training_note)}"
                for v in card.variants
                if v.licence
                and (vl := card.variant_licence(v))
                and (vl.training_data, vl.training_note) != (lic.training_data, lic.training_note)
            ),
            *(f"- Licence note: {n}" for n in card.licence_notes),
            *([f"- Options: {', '.join(_option(card, k) for k in card.options)}"] if card.options else []),
            f"- Kernel libraries: {', '.join(f'`{k}`' for k in card.kernel_libs) or 'none'}"
            + (
                f" ({card.no_kernels_reason})"
                if card.no_kernels_reason
                else " (no kernels backend)"
                if not card.kernel_libs
                else ""
            ),
            f"- Variants: {', '.join(_variant(v) for v in card.variants)}",
            f"- Outputs: {', '.join(_output(card, o) for o in card.outputs)}"
            + "".join(f"; {k}: `{val}`" for k, val in card.output_info.items()),
            *card_size_lines(card),
            *notes,
            *_measured(task, card, rows),
            *_parity(card),
            *(["", "### Limitations", "", *(f"- {x}" for x in card.limitations)] if card.limitations else []),
            "",
            "</details>",
        ]
    lines += [
        "",
        "## All models",
        "",
        "<details><summary>Modes, licences, training data and outputs per model</summary>",
        "",
        (
            "Commercial status per variant: `yes` if the code and the weights licence both allow commercial use (the "
            "training data is listed but does not change it; weights without a licence of their own are assumed to "
            "carry the code licence); `no` if either forbids it; `unclear` for custom licence terms (treat as no)."
        ),
        "",
        "| model | modes | code licence | weights licence | training data | commercial | outputs |",
        "|---|---|---|---|---|---|---|",
    ]
    for card, v in registry.cards(task).values():
        st = card.status(v)
        lines.append(
            f"| [`{_name(v)}`](#{card.family}) | {', '.join(card.modes)} "
            f"| {card.licence.code} | {st.weights} | {', '.join(st.training_data)} | {_COMMERCIAL[st.commercial]}"
            f"{' (' + '; '.join(st.reasons) + ')' if st.reasons else ''} "
            f"| {', '.join(card.outputs)} |"
        )
    lines += ["", "</details>"]
    lines += ["", "## Training datasets", "", "| id | dataset | licence | commercial use |", "|---|---|---|---|"]
    lines += [
        f"| `{d.id}` | [{d.name}]({d.url}) | {d.licence} | {d.commercial} |"
        for d in licences.datasets().values()
        if d.id in used
    ]
    return "\n".join(lines) + "\n"


def known_models(task: str) -> str:
    """The model id -> distribution index of `task` (TOML): the models that are not in the core wheel."""
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    by_class = {
        cls: name for name, spec in dists.items() if task in spec.get("tasks", (task,)) for cls in spec["code_classes"]
    }
    lines = [
        "# Generated by tools/gen_docs.py from the model cards and dists.toml; do not edit.",
        "# Every model id of the task outside the core wheel -> its distribution (named in 'not installed' errors).",
    ]
    ids = {**registry.cards(task), **registry.default_tags(task)}  # untagged timm names too
    lines += [
        f'"{model_id}" = "{by_class[card.licence.code_class]}"'
        for model_id, (card, _) in sorted(ids.items())
        if card.licence.code_class in by_class
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- overview page
OVERVIEW = ROOT / "docs" / "overview.md"
_SIZE_BEGIN, _SIZE_END = "<!-- sizes:begin", "<!-- sizes:end -->"
# one standard size per task for the overview, the model pages' summary, the startup table and the speed-ups
_STD_SIZE = {
    "stereo": "994x1482",
    "depth": "994x1482",
    "segmentation": "1024x2048",
    "classification": "224x224",
    "features": "224x224",
    "promptable_segmentation": "1024x1024",
    "detection": "800x1344",
}
_STD_DEFAULT = "1080x1920"
_OWN = "own input"  # multiview: every model runs its own number of views and size, no common input
# task -> (dataset, metric key, column title, format); lower is better. Tasks without an entry have no real-data eval.
_OVERVIEW_ACC = {
    "stereo": ("eth3d", "bad", "ETH3D bad > 1 px (%)", "{:.2f}"),
    "depth": ("nyu", "rel_absrel", "NYUv2 AbsRel (scale+shift aligned)", "{:.4f}"),
}
_PICK_NOTES = {
    "depth": (
        "The error is AbsRel after a per-image scale and shift alignment for every model: it ranks relative depth "
        "quality, not metric accuracy (metric error is not measured here). Best relative, commercially usable: the "
        "most accurate pick above. Metric (metres), commercially usable: `metric3d-vit-large` and `da3metric-large` "
        "(give `intrinsics=`), `moge-2-vitl-normal`, `zoedepth-nk`; non-commercial: `unidepth-v2-vitl14`. "
        "See the [depth page](models/depth.md) for `kind` and `intrinsics=`."
    ),
}
_TRADEOFF = 1.25  # best trade-off: the fastest model within this factor of the best error
_DASH = "\u2013"
_CELL = re.compile(r"^([\d.]+) ms(?: \(n=\d+\))? / (\d+) / (\d+)$")


def _fixed_sizes() -> dict[tuple[str, str], dict[str, tuple[float, int, int] | str]]:
    """Fixed-size table of docs/benchmarks.md: `{(id, mode): {size column: (ms, alloc MiB, reserved MiB) | status}}`."""
    text = (ROOT / "docs" / "benchmarks.md").read_text().split(_SIZE_BEGIN, 1)[1].split(_SIZE_END, 1)[0]
    head: list[str] = []
    out: dict[tuple[str, str], dict[str, tuple[float, int, int] | str]] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if cells[:2] == ["model", "mode"]:
            head = [c.removeprefix("12 MP ") for c in cells[2:]]
        elif head and len(cells) == len(head) + 2 and cells[1] in ("portable", "kernels"):
            row = {}
            for size, cell in zip(head, cells[2:], strict=True):
                m = _CELL.match(cell)
                row[size] = (float(m[1]), int(m[2]), int(m[3])) if m else cell
            out[cells[0].split("@")[0], cells[1]] = row
    return out


def _slug(title: str) -> str:
    """GitHub / mkdocs anchor of a heading."""
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def _ms(v: tuple[float, int, int] | str | None) -> str:
    return _DASH if v is None or v in ("-", "SKIPPED") else f"{v[0]:.1f}" if isinstance(v, tuple) else str(v)


def _fixed_cell(speed_row: dict | None, table_cell: object) -> object:
    """(ms, allocated, reserved) of a speed row of the same input if there is one, else the fixed-size table's cell."""
    if speed_row and "peak_alloc_mb" in speed_row:
        return (speed_row["ms"], speed_row["peak_alloc_mb"], speed_row["peak_reserved_mb"])
    return table_cell


def _overview_rows(task: str, fixed: dict, speed: dict[str, list[dict]], acc: dict[str, list[dict]]) -> list[dict]:
    """One dict per variant of `task`: card data, fixed-size cells, original code and accuracy."""
    size = _OWN if task == "multiview" else _STD_SIZE.get(task, _STD_DEFAULT)
    ids = {v.id for card, _ in registry.cards(task).values() for v in card.variants}
    originals = [r for i in ids for r in speed.get(i, []) if r["mode"] in ("upstream", "upstream-default")]
    orig_size = (
        size
        if any(r["input"] == size for r in originals)
        else max((r["input"] for r in originals), key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]), default="")
    )
    dataset, key, _, fmt = _OVERVIEW_ACC.get(task, (None, None, None, None))
    rows = []
    seen = set()
    for card, _ in registry.cards(task).values():
        for v in card.variants:
            if v.id in seen:
                continue
            seen.add(v.id)
            mine = {(r["mode"], r["input"]): r for r in speed.get(v.id, [])}
            at = _at(speed.get(v.id, []), size)  # a speed row of the same input beats the older cell
            if size == _OWN:
                at = next((i for m, i in mine if m == "kernels"), size)
            mx, pt = (
                _fixed_cell(mine.get((m, at)), fixed.get((v.id, m), {}).get(size)) for m in ("kernels", "portable")
            )
            orig = _orig(mine.get(("upstream", orig_size)), mine.get(("upstream-default", orig_size)))
            max_o = mine.get(("kernels", orig_size))
            if orig and max_o:
                speedup = (orig["ms"] / max_o["ms"], f"original ({orig_size})")
            elif isinstance(mx, tuple) and isinstance(pt, tuple):
                speedup = (pt[0] / mx[0], "portable")
            else:
                speedup = None
            a = {r["mode"]: r for r in acc.get(v.id, []) if r["dataset"] == dataset}
            a_row = a.get("kernels") or a.get("portable")
            mark = _overlap(card, v, dataset) if a_row and dataset else ""
            best = mx if isinstance(mx, tuple) else pt if isinstance(pt, tuple) else None
            rows.append(
                {
                    "id": v.id,
                    "name": _name(v),
                    "link": f"models/{task}.md#{_slug(card.name)}",
                    "commercial": card.status(v).commercial,
                    "kernels": mx,
                    "portable": pt,
                    "original": orig["ms"] if orig else None,
                    "speedup": speedup,
                    "error": a_row[key] if a_row else None,
                    "mark": mark,
                    "ms": best[0] if best else None,
                    "at": at,
                }
            )
    rows.sort(key=lambda r: (r["ms"] is None, r["ms"] or 0.0, r["id"]))
    return [{**r, "std": size, "orig_size": orig_size, "fmt": fmt} for r in rows]


def _picks(rows: list[dict]) -> list[str]:
    """The 'which model' bullets of one task, by fixed rules over the table rows."""
    timed = [r for r in rows if r["ms"] is not None]
    scored = [r for r in rows if r["error"] is not None and not r["mark"]]

    def best(pool: list[dict], key: str) -> dict | None:
        return min(pool, key=lambda r: (r[key], r["ms"] if r["ms"] is not None else 1e18, r["id"]), default=None)

    def fmt(r: dict, *, error: bool = False) -> str:
        ms = f"{r['ms']:.1f} ms" if r["ms"] is not None else NM
        err = f", error {r['fmt'].format(r['error'])}" if error and r["error"] is not None else ""
        return f"[`{r['name']}`]({r['link']}) ({ms}{err})"

    out = []
    fast = best(timed, "ms")
    if fast:
        out.append(f"- Fastest: {fmt(fast)}.")
    if scored:
        top = best(scored, "error")
        out.append(f"- Most accurate: {fmt(top, error=True)}.")
        near = [r for r in scored if r["ms"] is not None and r["error"] <= _TRADEOFF * top["error"]]
        trade = best(near, "ms")
        if trade and trade is not top:
            out.append(f"- Best trade-off: {fmt(trade, error=True)}.")
    com = [r for r in timed if r["commercial"] == "yes"]
    if com:
        out.append(f"- Fastest commercially usable: {fmt(best(com, 'ms'))}.")
        com_scored = [r for r in scored if r["commercial"] == "yes"]
        if com_scored:
            out.append(f"- Most accurate commercially usable: {fmt(best(com_scored, 'error'), error=True)}.")
    return out or ["- No timed model."]


def _overview_task(task: str, fixed: dict, speed: dict, acc: dict) -> list[str]:
    rows = _overview_rows(task, fixed, speed, acc)
    std, orig_size = rows[0]["std"], rows[0]["orig_size"]  # the task's standard size, the size of its original rows
    acc_col = _OVERVIEW_ACC.get(task)
    lines = [
        f"## {_title(task)}",
        "",
        f"{len(rows)} models; "
        + ("each at its own benchmark input (views x size in Benchmarks)." if std == _OWN else f"size {std}."),
        "",
        *_picks(rows),
        *(["", _PICK_NOTES[task]] if task in _PICK_NOTES else []),
        "",
    ]
    head = ["model", "commercial use"] + ([acc_col[2]] if acc_col else [])
    head += [
        "original time (ms, its size)",
        "portable time (ms)",
        "kernels time (ms)",
        "kernels speed-up",
        "kernels peak memory (MiB, allocated)",
    ]
    marks = set()
    body = []
    for r in rows:
        cells = [f"[`{r['name']}`]({r['link']})", r["commercial"]]
        if acc_col:
            err = _DASH if r["error"] is None else r["fmt"].format(r["error"]) + (f" {r['mark']}" if r["mark"] else "")
            marks.add(r["mark"])
            cells.append(err)
        sp = r["speedup"]
        cells += [
            _DASH if r["original"] is None else f"{r['original']:.1f} ({orig_size})",
            _ms(r["portable"]),
            _ms(r["kernels"]) + (f" ({r['at']})" if r["at"] != std and std != _OWN and r["kernels"] else ""),
            _DASH if sp is None else f"{sp[0]:.1f}x vs {sp[1]}",
            str(r["kernels"][1]) if isinstance(r["kernels"], tuple) else _DASH,
        ]
        body.append(cells)
    lines += _md_table(head, body)
    for m in (_DAGGER, "?"):
        lines += ["", _LEGEND[m]] if m in marks else []
    return lines


def overview() -> str:
    """The overview page: which model to use and the speed-ups, per task, from the cards and docs/data."""
    fixed, speed = _fixed_sizes(), _speed_rows()
    gpus = [r["gpu"].removeprefix("NVIDIA GeForce ") for rs in speed.values() for r in rs]
    gpu = statistics.mode(gpus) if gpus else "the benchmark GPU"
    lines = [
        "# Which model to use",
        "",
        "<!-- Generated by tools/gen_docs.py from the model cards and docs/data; do not edit. -->",
        "",
        (
            f"Measured on one {gpu} at the size in each section header (median of warm runs); "
            "`fastforks bench MODEL --size HxW` measures your GPU. Each model links to its model page (licences, "
            "accuracy, more sizes); [benchmarks](benchmarks.md) explains the method."
        ),
        "",
        (
            "- `kernels` = our kernels, `portable` = plain PyTorch, `original` = the original code named on the model "
            "page (size in brackets)."
        ),
        (
            "- speed-up: `kernels` against the faster of original fp16 / fp32 where both were timed at the original's "
            "size, else against `portable` (labelled)."
        ),
        "- commercial use: `yes` only if the code and the weights licence both allow it.",
        (
            f"- Accuracy is a check on public images, not a leaderboard: {_DAGGER} = trained on that data, "
            "? = training data not stated; the most-accurate picks skip both. Best trade-off = the fastest model "
            f"within {_TRADEOFF:g}x of the lowest error."
        ),
        (
            "- Peak memory = the kernels run's peak allocated GPU memory (CUDA, weights included); the model pages' "
            "summary tables list peak reserved (what the caching allocator holds, larger). "
            f"{_DASH} = not measured."
        ),
    ]
    for task in registry.TASKS:
        lines += ["", *_overview_task(task, fixed, speed, _eval_rows(task))]
    return "\n".join(lines) + "\n"


_TABLE_BEGIN, _TABLE_END = "<!-- tables:begin", "<!-- tables:end -->"


def _speed_table(by: dict[str, dict[str, dict]]) -> list[str]:
    """Table of one (task, input), per model; a column nothing was measured for is left out."""
    names = ["original fp32", "original", "portable", "kernels", "vs original", "vs portable"]
    groups = [("time (ms): ", 4), ("kernels speed-up: ", 2), ("peak memory (MiB, allocated / reserved): ", 3)]
    body = []
    for model_id, m in by.items():
        o32, o, pt, k = (m.get(mode) for mode in _SPEED_MODES)
        cells = [_ms_cell(o32), _ms_cell(o), _ms_cell(pt), _ms_cell(k), _ratio(_orig(o, o32), k), _ratio(pt, k)]
        body.append([model_id, *cells, _peak(o), _peak(pt), _peak(k)])
    cols = [(groups[0][0], n) for n in names[:4]] + [(groups[1][0], n) for n in names[4:]]
    cols += [(groups[2][0], n) for n in ("original", "portable", "kernels")]
    use = [i for i in range(len(cols)) if any(r[i + 1] != DASH for r in body)]
    head, seen = ["model"], set()
    for i in use:
        g, n = cols[i]
        head.append(n if g in seen else g + n)
        seen.add(g)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join([r[0], *(r[i + 1] for i in use)]) + " |" for r in body]


def _bench_block() -> str:
    """Speed and memory tables per task and input, from docs/data/speed.jsonl (the block of docs/benchmarks.md)."""
    speed = _speed_rows()
    tasks = {v.id: t for t in registry.TASKS for _, v in registry.cards(t).values()}
    groups: dict[tuple[str, str, str], dict[str, dict[str, dict]]] = {}
    for model_id, rows in sorted(speed.items()):
        for r in rows:
            if model_id in tasks and r["mode"] in _SPEED_MODES:
                key = (tasks[model_id], r["input"], _scene(r["scene"]))
                groups.setdefault(key, {}).setdefault(model_id, {})[r["mode"]] = r
    order = {t: i for i, t in enumerate(registry.TASKS)}
    out = [f"{_TABLE_BEGIN} (tools/gen_docs.py, from docs/data/speed.jsonl) -->"]
    for key in sorted(groups, key=lambda k: (order[k[0]], -len(groups[k]), k[1:])):
        task, size, scene = key
        by = groups[key]
        if not any("kernels" in m or "portable" in m for m in by.values()):
            continue
        out += ["", f"### {task.capitalize().replace('_', ' ')}, {scene}, {size}", "", *_speed_table(by)]
    out += [
        "",
        "### Startup and first call",
        "",
        f"{_STARTUP_HEAD} Seconds, at each task's standard size (another size is shown in brackets).",
        "",
    ]
    out += [
        ("| model | load (s): original | portable | kernels | first call (s): original | portable | kernels |"),
        "|---|---|---|---|---|---|---|",
    ]
    used = False
    for task in registry.TASKS:
        for model_id in sorted(i for i in speed if tasks.get(i) == task):
            by = _startup(task, model_id, speed[model_id])
            cells = [
                (
                    f"{r[f]:.2f}{lm if f == 'load_s' else fm}"
                    + (f" ({r['input']})" if r["input"] != _std_input(task, {}) else "")
                )
                if (r := (by.get(mode) or (None,))[0]) and f in r
                else DASH
                for f in ("load_s", "first_call_s")
                for mode in ("upstream", "portable", "kernels")
                for lm, fm in [(by[mode][1], by[mode][2]) if mode in by else ("", "")]
            ]
            used = used or any(_WARM in c for c in cells)
            if any(c != DASH for c in cells):
                out.append(f"| {model_id} | " + " | ".join(cells) + " |")
    if used:
        out += ["", _WARM_NOTE]
    out.append(_TABLE_END)
    return "\n".join(out)


_COMPILE_BEGIN, _COMPILE_END = "<!-- compile:begin", "<!-- compile:end -->"


_FAIL_REASONS = (  # (pattern in the recorded error, short reason for the table); first match wins
    (r"cannot parse .*_aa and operator: Resize", "TensorRT: antialiased resize not supported"),
    (r"cannot parse .*operator: (\w+)", r"TensorRT parser: \1 node"),
    (r"engine build failed.*\[myelin", "TensorRT bug: Myelin internal error"),
    (r"TensorRT engine build failed", "TensorRT engine build failed"),
    (r"wrong output: an engine output is all NaN", "TensorRT: NaN output (onnxruntime on the same ONNX is right)"),
    (r"TorchExportError", "ONNX export: torch.export failed"),
    (r"ConversionError", "ONNX export: conversion failed"),
    (r"doesn't match model \((\d+)\)", r"ONNX export: fixed \1 px input"),
    (r"compilation subprocess exited", "Inductor: compile worker crashed"),
    (r"Guard failed on the same frame", "Dynamo guard error (torch bug)"),
    (r"version_counter for inference tensor", "Inductor: inference tensors"),
    (r"inference_mode expects no kwargs", "Dynamo: inference_mode in model code"),
    (r"expected size to be one of int", "Dynamo: numpy size in model code"),
    (r"CUDA driver error: invalid argument", "CUDA driver error (invalid argument)"),
)


def _fail_reason(msg: str) -> str:
    """Short table reason for a recorded `fails` message (the full text stays in the jsonl)."""
    for pat, short in _FAIL_REASONS:
        if m := re.search(pat, msg):
            return m.expand(short)
    return re.sub(r"\x1b\[\d+m|/tmp/\S+/", "", msg.removeprefix("run: "))[:60]


def _compile_ms(r: dict | None) -> str:
    """The time cell of a compile / tensorrt row: `N ms`, `fails: reason`, or a dash."""
    if r is None:
        return DASH
    return f"fails: {_fail_reason(r['fails'])}" if "fails" in r else f"{_ms_str(r['ms'])} ms"


def _compile_block() -> str:
    """torch.compile and TensorRT table of docs/benchmarks.md, from docs/data/speed_compile.jsonl (+ speed.jsonl)."""
    latest = _compile_rows()
    speed = _speed_rows()
    tasks = {v.id: t for t in registry.TASKS for _, v in registry.cards(t).values()}
    order = {t: i for i, t in enumerate(registry.TASKS)}
    head = (
        "| model | input | time: portable | compile | TensorRT fp16 | kernels"
        " | preparation: compile (cold first call) | ONNX + engine build"
        " | peak MiB (alloc / reserved): portable | compile | kernels | TensorRT (device) |"
    )
    out = [f"{_COMPILE_BEGIN} (tools/gen_docs.py, from docs/data/speed_compile.jsonl) -->"]
    if not latest:
        return "\n".join([*out, "not measured yet", _COMPILE_END])
    out += [head, "|" + "---|" * (head.count("|") - 1)]
    for ref in sorted({k[0] for k in latest}, key=lambda i: (order.get(tasks.get(i, ""), 99), i)):
        comp, trt = latest.get((ref, "compile")), latest.get((ref, "tensorrt"))
        size = next(r["input"] for r in (comp, trt) if r)
        at = {r["mode"]: r for r in speed.get(ref, []) if r["input"] == size}
        built = trt is not None and "fails" not in trt
        ms = {m: f"{_ms_str(at[m]['ms'])} ms" if m in at else DASH for m in ("portable", "kernels")}
        cells = [
            ref,
            size,
            ms["portable"],
            _compile_ms(comp),
            _compile_ms(trt),
            ms["kernels"],
            DASH if comp is None or "fails" in comp else f"{_ms_str(comp['compile_s'])} s",
            f"{_ms_str(trt['onnx_export_s'] + trt['engine_build_s'])} s" if built else DASH,
            _peak(at.get("portable")),
            _peak(comp) if comp and "fails" not in comp else DASH,
            _peak(at.get("kernels")),
            str(trt["device_mb"]) if built and "device_mb" in trt else DASH,
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join([*out, _COMPILE_END])


_LIBS_BEGIN, _LIBS_END = "<!-- libraries:begin", "<!-- libraries:end -->"


def _libs_block() -> str:
    """Kernel library -> families table of docs/OTHER_GPUS.md, from the cards' `[modes.kernels] kernels`."""
    fams: dict[str, set[str]] = {}
    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            for lib in card.kernel_libs:
                fams.setdefault(lib, set()).add(card.family)
    rows = [f"| `{lib}` | {', '.join(sorted(f))} |" for lib, f in sorted(fams.items())]
    return "\n".join(
        [
            f"{_LIBS_BEGIN} (tools/gen_docs.py, from the model cards) -->",
            "| library | families |",
            "|---|---|",
            *rows,
            _LIBS_END,
        ]
    )


_CAPS_BEGIN, _CAPS_END = "<!-- capabilities:begin", "<!-- capabilities:end -->"


def _caps_block() -> str:
    """Capability table of docs/OTHER_GPUS.md: families with the same support are one row (from the cards)."""
    short = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    groups: dict[tuple[str, ...], list[str]] = {}
    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            has = "kernels" in card.modes
            mps_k = bool(card.mps_kernel_libs)
            sig = (
                short[card.portable_cuda_dtype],
                short[card.kernels_dtype] if has else "-",
                "/".join(short[d] for d in card.portable_mps_dtypes) or "-",
                (", ".join(card.mps_kernel_libs) + " " + short[(card.mps_kernels_dtype or card.kernels_dtype)])
                if mps_k
                else "-",
            )
            groups.setdefault(sig, []).append(card.family)
    rows = [
        f"| {', '.join(sorted(set(f)))} | " + " | ".join(sig) + " |"
        for sig, f in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    return "\n".join(
        [
            f"{_CAPS_BEGIN} (tools/gen_docs.py, from the model cards) -->",
            "| families | CUDA portable | CUDA kernels | MPS portable (validated) | MPS kernels (libraries, dtype) |",
            "|---|---|---|---|---|",
            *rows,
            _CAPS_END,
        ]
    )


def _with_block(text: str, begin: str, end: str, block: str) -> str:
    head, rest = text.split(begin, 1)
    return head + block + rest.split(end, 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gaps", action="store_true", help="print the family x missing-item table and exit")
    args = ap.parse_args()
    check = args.check
    if args.gaps:
        rows = [g for t in registry.TASKS for g in _gaps(t)]
        cols = list(rows[0])
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        print("\n".join("| " + " | ".join(r[c] for c in cols) + " |" for r in rows))
        return
    bench, gpus = ROOT / "docs" / "benchmarks.md", ROOT / "docs" / "OTHER_GPUS.md"
    outputs = {
        bench: _with_block(
            _with_block(bench.read_text(), _TABLE_BEGIN, _TABLE_END, _bench_block()),
            _COMPILE_BEGIN,
            _COMPILE_END,
            _compile_block(),
        ),
        OVERVIEW: overview(),
        gpus: _with_block(
            _with_block(gpus.read_text(), _LIBS_BEGIN, _LIBS_END, _libs_block()), _CAPS_BEGIN, _CAPS_END, _caps_block()
        ),
    }
    for task in registry.TASKS:
        outputs[ROOT / "docs" / "models" / f"{task}.md"] = render(task)
        outputs[ROOT / "src" / "fastforks" / task / "known_models.toml"] = known_models(task)
    stale = []
    for path, text in outputs.items():
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        elif not path.is_file() or path.read_text() != text:
            stale.append(str(path.relative_to(ROOT)))
    if stale:
        sys.exit(f"out of date (run tools/gen_docs.py): {', '.join(stale)}")


if __name__ == "__main__":
    main()
