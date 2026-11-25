"""Structured per-family data for the model pages of the website (fastforks.dev/models/<family>).

The docs pages (tools/gen_docs.py) are markdown; the website's model cards need the same facts as data: the card
(licences, weights files, upstream, kernels), every measured speed row of the reference GPU (original fp32 / fp16,
portable, kernels, torch.compile, TensorRT), the survey devices, the accuracy rows per dataset and mode, and the
parity record. Nothing here is computed beyond picking rows; missing data stays missing (the site says "not
measured").

    uv run python tools/site_data.py --out DIR   # DIR/<family>.json, one file per family, plus DIR/index.json
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import _devices
import examples
import gen_docs as g
import goldens
from fastforks._core import licences, registry
from fastforks._core.cards import planned_reason
from fastforks._core.sizes import size_range
from fastforks._core.tasks import spec

_ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from fastforks._core.cards import ModelCard, Variant
    from fastforks._core.files import FileSpec

# Speed modes of docs/data/speed.jsonl -> the site's mode key (the eval rows use the same keys)
_SPEED_KEY = {"upstream-default": "original32", "upstream": "original16", "portable": "portable", "kernels": "kernels"}
_EVAL_KEY = {
    "upstream": "original32",
    "upstream-fp32-strict": "original32strict",
    "upstream-fp16": "original16",
    "portable32": "portable32",
    "portable": "portable",
    "kernels": "kernels",
    "compile": "compile",
    "tensorrt": "tensorrt",
    "upstreamhalf": "originalhalf",
    "upstream-cpu": "originalcpu",
}
_HOSTS = {
    "huggingface.co": "Hugging Face",
    "drive.google.com": "Google Drive",
    "dl.fbaipublicfiles.com": "Meta (fbaipublicfiles)",
    "download.openmmlab.com": "OpenMMLab",
    "download.pytorch.org": "PyTorch",
    "github.com": "GitHub",
    "www.dropbox.com": "Dropbox",
    "dropbox.com": "Dropbox",
}
_KEEP_SPEED = (
    "ms",
    "ms_min",
    "noise",
    "runs",
    "peak_alloc_mb",
    "peak_reserved_mb",
    "load_s",
    "first_call_s",
    "import_s",
    "scene",
    "date",
    "torch",
    "protocol",
)


def _host(url: str | None) -> str | None:
    if not url:
        return None
    net = urlparse(url).netloc
    return _HOSTS.get(net) or (
        "Alibaba Cloud OSS" if net.endswith("aliyuncs.com") else "Google Drive" if "google" in net else net
    )


def _file(f: FileSpec) -> dict[str, Any]:
    return {
        "name": f.name,
        "sha256": f.sha256,
        "size": f.size,
        "url": f.url,
        "host": _host(f.url),
        "member": f.member,
        "archive_size": f.archive_size,
        "mirror": f.mirror,
    }


def _data(ids: tuple[str, ...]) -> list[dict[str, Any]]:
    known = licences.datasets()
    return [
        {"id": i, "name": "not stated by the authors"}
        if i == licences.UNSTATED
        else {
            "id": i,
            "name": known[i].name,
            "licence": known[i].licence,
            "commercial": known[i].commercial,
            "url": known[i].url,
        }
        for i in ids
    ]


def _licence(card: ModelCard, v: Variant) -> dict[str, Any]:
    lic, st = card.variant_licence(v), card.status(v)
    return {
        "code": lic.code,
        "code_class": lic.code_class,
        "weights": licences.describe_weights(lic.weights, lic.code),
        "weights_assumed": lic.weights == licences.ASSUMED,
        "evidence": lic.evidence,
        "training_data": _data(lic.training_data),
        "training_note": lic.training_note,
        "commercial": st.commercial,
        "reasons": list(st.reasons),
        "notes": list(st.notes),
    }


def _speed_row(r: dict[str, Any]) -> dict[str, Any]:
    out = {k: r[k] for k in _KEEP_SPEED if k in r}
    dtype = _devices.dtype_of(r)
    out["dtype"] = "float16" if r["mode"] == "upstream" and dtype == "default" else dtype
    if note := g._note(r):  # noqa: SLF001 - shared docs helper
        out["note"] = note
    if r.get("commit"):
        out["commit"] = r["commit"][:12]
    return out


def _native_speed_input(card: ModelCard, v: Variant) -> str | None:
    """The input normally shown on a fixed-size model page."""
    if card.task == "classification" and v.io.crop:
        return f"{v.io.crop}x{v.io.crop}"
    if card.task == "ocr" and card.family == "parseq":
        # PARSeq recognises a single word crop. PP-OCRv5 takes a whole page.
        return "32x128"
    return None


def _speed(v: Variant, task: str, speed: dict[str, list[dict]], *, native: str | None = None) -> dict[str, Any]:
    """Reference-GPU rows of one variant: per input, the four modes plus torch.compile / TensorRT."""
    rows = speed.get(v.id, [])
    std = g._std_input(task, {})  # noqa: SLF001
    inputs = sorted({r["input"] for r in rows} | g._compiled_inputs(v.id), key=g._pixels)  # noqa: SLF001
    preferred = next((size for size in (native, std) if size in inputs), inputs[-1] if inputs else None)
    out = []
    for size in inputs:
        by = {}
        for r in rows:
            if r["input"] != size or r["mode"] not in _SPEED_KEY:
                continue
            row = _speed_row(r)
            key = _SPEED_KEY[r["mode"]]
            # Some historical FP32 rows used `upstream`, including the two LocoTrack checkpoints.
            by["original32" if key == "original16" and row["dtype"] == "float32" else key] = row
        oom = [_SPEED_KEY[m] for m in _SPEED_KEY if (v.id, m, size) in g._oom()]  # noqa: SLF001
        extra = {}
        for mode in ("compile", "tensorrt"):
            c = g._compile_rows().get((v.id, mode, size))  # noqa: SLF001
            if c is None:
                continue
            if "fails" in c:
                reason = planned_reason(v, mode) if c["fails"] == g.PLANNED else None
                extra[mode] = (
                    {"fails": reason, "planned": True} if reason else {"fails": g._fail_reason(c["fails"])}  # noqa: SLF001
                )
                continue
            if mode == "tensorrt":
                alloc, reserved = g._trt_peaks(c) or (None, None)  # noqa: SLF001 - shared docs helper
            else:
                alloc, reserved = c.get("peak_alloc_mb"), c.get("peak_reserved_mb")
            extra[mode] = {
                "ms": c["ms"],
                "peak_alloc_mb": alloc,
                "peak_reserved_mb": reserved,
                "device_mb": c.get("device_mb"),
                "engine_dtype": c.get("engine_dtype", "float16") if mode == "tensorrt" else None,
                "part": g._backend_part(c).strip(" ()"),  # noqa: SLF001 - shared docs helper
                "standalone": bool(c.get("standalone_build")),
                "setup_s": g._setup_seconds(c),  # noqa: SLF001 - shared docs helper
                "date": c.get("date"),
            }
        extra |= _pose_parts(v, size)
        scene = next((r.get("scene", "") for r in rows if r["input"] == size), "")
        out.append(
            {
                "input": size,
                "scene": g._scene(scene),  # noqa: SLF001 - shared docs helper
                "standard": size == std,
                "native": size == native,
                "modes": by,
                "oom": oom,
                **extra,
            }
        )
    return {"standard": std, "preferred": preferred, "inputs": out}


def _pose_parts(v: Variant, size: str) -> dict[str, Any]:
    """Pose split timings of one variant and input (speed_compile.jsonl): the original's network alone, our steps."""
    rows = g._compile_rows()  # noqa: SLF001
    out: dict[str, Any] = {}
    net = {}  # the original's network alone (crops made untimed), side "upstream"
    for mode, key in (("upstream-network-default", "original32"), ("upstream-network", "original16")):
        c = rows.get((v.id, mode, size))
        if c is not None and "ms" in c:
            net[key] = _speed_row(c) | {"timed": c.get("timed", "")}
    if net:
        out["original_network"] = net
    parts = {}  # where our call spends its time (crop / network / rest), modes "<mode>-sections"
    for mode in ("portable", "kernels"):
        c = rows.get((v.id, f"{mode}-sections", size))
        if c is not None:
            parts[mode] = {k: c[k] for k in ("ms", "crop_ms", "network_ms", "rest_ms", "network_has_decode")}
    if parts:
        out["sections"] = parts
    return out


def _devices_of(v: Variant) -> list[dict[str, Any]]:
    """Survey rows of one variant on every shown non-reference device: each (mode, dtype) row."""
    out = []
    for dev in _devices.shown():
        if dev.reference:
            continue
        rows = [r for r in _rows_of(dev) if r.get("ref") == v.id]
        if not rows:
            continue
        cells = []
        for r in rows:
            mem = (
                {"alloc": r.get("peak_alloc_mb"), "reserved": r.get("peak_reserved_mb")}
                if r.get("memory_kind", "torch-peak") == "torch-peak"
                else {"footprint": r.get("peak_footprint_mb")}
            )
            cells.append(
                {
                    "mode": r["mode"],
                    "dtype": _devices.dtype_of(r),
                    "status": _devices.status_of(r),
                    "golden": r.get("golden"),
                    "input": r.get("input"),
                    "ms": r.get("ms"),
                    "memory_kind": r.get("memory_kind", "torch-peak"),
                    **mem,
                    "error": r.get("error"),
                }
            )
        out.append(
            {
                "slug": dev.slug,
                "name": dev.name,
                "backend": dev.backend,
                "torch": rows[0].get("torch"),
                "date": max(r.get("date", "") for r in rows),
                "rows": cells,
            }
        )
    return out


_DEVICE_ROWS: dict[str, list[dict[str, Any]]] = {}


def _rows_of(dev: _devices.Device) -> list[dict[str, Any]]:
    if dev.slug not in _DEVICE_ROWS:
        _DEVICE_ROWS[dev.slug] = _devices.rows(dev)
    return _DEVICE_ROWS[dev.slug]


def _quality(card: ModelCard, v: Variant, rows: dict[str, list[dict]]) -> list[dict[str, Any]]:
    """Accuracy per dataset of one variant: every measured mode with its metrics and where it ran."""
    task = card.task
    mine = rows.get(v.id, [])
    out = []
    for d, title in g._DATASETS.get(task, {}).items():  # noqa: SLF001 - some tasks have no accuracy protocol yet
        ds = [r for r in mine if r["dataset"] == d]
        if not ds:
            continue
        metrics, count, noun = g._METRICS.get((task, d)) or g._METRICS[task]  # noqa: SLF001
        modes = {}
        for r in ds:
            key = _EVAL_KEY.get(r["mode"])
            # Some evaluation collectors store the precision-specific tier
            # separately from the generic mode. Keep both measurements visible.
            if r["mode"] == "portable" and r.get("tier") == "portable32":
                key = "portable32"
            elif r["mode"] == "upstream":
                if r.get("tier") in ("upstreamhalf", "upstreambf16"):
                    key = "originalhalf"
                elif r.get("tier") == "upstream16" or r.get("dtype") in ("float16", "bfloat16"):
                    key = "original16"
            if key is None:
                continue
            m = r.get("metrics") or r
            modes[key] = {
                "dtype": r.get("dtype"),
                "tier": r.get("tier"),
                "metrics": {k: m[k] for k, _, _ in metrics if k in m},
                "n": r.get(count) or r.get("n"),
                "error": r.get("error") or None,
                "ms_per_img": r.get("ms_per_img"),
                "peak_alloc_mb": r.get("peak_alloc_mb"),
                "peak_reserved_mb": r.get("peak_reserved_mb"),
                "gpu": r.get("gpu", "").removeprefix("NVIDIA GeForce "),
                "torch": r.get("torch"),
                "commit": (r.get("commit") or "")[:12],
                "date": (r.get("date") or "")[:10],
                "how": r.get("how"),
                "note": r.get("note"),
                "version": eval_version(task, r),
            }
        portable = modes.get("portable")
        if portable and portable["dtype"] == "float32" and portable["metrics"] and not portable["error"]:
            # A default portable FP32 run is already an FP32 measurement; do not claim a second run.
            note = ". ".join(
                x for x in ((portable["note"] or "").rstrip("."), "Same measurement as portable (float32).") if x
            )
            modes.setdefault("portable32", {**portable, "note": note})
        mark = g._overlap(card, v, d)  # noqa: SLF001
        out.append(
            {
                "id": d,
                "title": title,
                "noun": noun,
                "metrics": [
                    {"key": k, "format": f, "optional": bool(opt), "lower": _lower(k)} for k, f, opt in metrics
                ],
                "protocol": (g._PROTOCOL.get(task, "") + g._PROTOCOL.get((task, d), "")).strip(),  # noqa: SLF001
                "overlap": {g._DAGGER: "train", "?": "unknown"}.get(mark),  # noqa: SLF001
                "bad_thr": g._BAD_THR.get(d) if task == "stereo" else None,  # noqa: SLF001
                "modes": modes,
            }
        )
    return out


_LOWER = {
    "absrel",
    "ang_mean",
    "ang_median",
    "bad",
    "dtssd",
    "epe",
    "fl_all",
    "mad",
    "mae",
    "mae_mm",
    "met_absrel",
    "mse",
    "nme_pct",
    "rel_absrel",
    "rel_l2",
    "sad",
}


def _lower(key: str) -> bool:
    return key in _LOWER


def _parity(card: ModelCard) -> dict[str, Any]:
    store = goldens.load(card.family, card.task)
    tol = dict(store["tolerance"])
    per = {}
    for v in card.variants:
        rec = store["validated"].get(v.ref) or {}
        per[v.id] = (
            {
                "mean": rec["mean_px"],
                "p99": rec["p99_px"],
                "gpu": rec["gpu"].removeprefix("NVIDIA GeForce "),
                "torch": rec["torch"],
                "date": rec.get("date"),
                "kernels": list(rec.get("kernels", [])),
            }
            if "mean_px" in rec
            else None
        )
    return {
        "text": g.parity_text(card, tol),
        "tolerance": tol,
        "relative": bool(tol.get("relative")),
        "unit": "px" if card.task == "stereo" else "",
        "validated": per,
    }


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True, check=True).stdout


# The version the models and results not in any release yet will ship in.
UNRELEASED = "0.2.0"


@functools.cache
def _tags() -> tuple[str, ...]:
    """Release tags, oldest first."""
    return tuple(_git("tag", "--list", "v*", "--sort=v:refname").split())


@functools.cache
def _tag_ids(tag: str) -> frozenset[str]:
    """The variant ids the cards of one release list."""
    ids: set[str] = set()
    for path in _git("ls-tree", "-r", "--name-only", tag).split():
        if path.endswith("/card.toml"):
            ids |= {v["id"] for v in tomllib.loads(_git("show", f"{tag}:{path}")).get("variants", [])}
    return frozenset(ids)


def released_ids() -> tuple[str | None, set[str]]:
    """The latest release tag and the variant ids its cards list (all ids empty without a tag)."""
    if not _tags():
        return None, set()
    return _tags()[-1].removeprefix("v"), set(_tag_ids(_tags()[-1]))


def added_in(variant_id: str) -> str:
    """The first release whose cards list the variant, else the coming one."""
    tag = next((t for t in _tags() if variant_id in _tag_ids(t)), None)
    return tag.removeprefix("v") if tag else UNRELEASED


def _row_key(r: dict[str, Any]) -> str:
    return json.dumps(r, sort_keys=True)


@functools.cache
def _eval_since(task: str) -> dict[str, str]:
    """Each accuracy row of a task (by its full content) -> the first release whose eval file holds it."""
    out: dict[str, str] = {}
    for t in _tags():
        try:
            text = _git("show", f"{t}:docs/data/{task}_eval.jsonl")
        except subprocess.CalledProcessError:  # the task had no eval file in that release
            continue
        for line in text.splitlines():
            out.setdefault(_row_key(json.loads(line)), t.removeprefix("v"))
    return out


def eval_version(task: str, row: dict[str, Any]) -> str:
    """The release an accuracy row first shipped in: rows measured since the last release count as the coming one."""
    return _eval_since(task).get(_row_key(row), UNRELEASED)


_EXAMPLE_KEEP = (
    "top", "classes", "objects", "prompt", "certain", "mode", "dtype", "device", "date",
    "texts", "layout", "attribution", "scale", "psnr_vs_true_middle", "mean_error_px",
    "focal_px", "vfov_deg", "roll_deg", "pitch_deg", "classes_litres",
)  # fmt: skip


@functools.cache
def _examples() -> dict[str, dict[str, Any]]:
    """Latest successful row per ref of docs/data/examples.jsonl (tools/examples.py)."""
    path = _ROOT / "docs/data/examples.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return {r["ref"]: r for r in rows if "error" not in r}


def _example(v: Variant, rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The card's example: our output on the task's fixed input (images under the site's /examples/)."""
    r = rows.get(v.id)
    if r is None:
        return None
    _, title, credit, licence, source = examples.INPUTS[r["input"]]
    views = 2 if r["task"] in {"stereo", "flow", "multiview", "matching", "interpolation"} else 1
    inputs = [f"/examples/inputs/{r['input']}{'-right' if i else ''}.webp" for i in range(views)]
    return {
        "image": None if r["task"] == "classification" else f"/examples/{v.id}.webp",
        "input": {"name": r["input"], "title": title, "credit": credit, "licence": licence, "source": source},
        "inputs": inputs,
        **{k: r[k] for k in _EXAMPLE_KEEP if k in r},
    }


def _variant(
    card: ModelCard, v: Variant, speed: dict, rows: dict, released: tuple[str | None, set[str]]
) -> dict[str, Any]:
    r = size_range(v, card)
    return {
        "id": v.id,
        "released": released[0] if v.id in released[1] else None,
        "added_in": added_in(v.id),
        "ref": v.ref,
        "name": g._name(v),  # noqa: SLF001
        "setting": g.setting(list(card.variants), v),
        "default_tag": v.default_tag,
        "params": dict(v.params),
        "licence": _licence(card, v),
        "weights": [_file(f) for f in v.weights],
        "input_size": r.describe(tested=False),
        "tested_max": (
            {
                "size": f"{r.tested_max[0]}x{r.tested_max[1]}",
                "alloc_mib": r.tested_alloc_mib,
                "reserved_mib": r.tested_reserved_mib,
            }
            if r.tested_max
            else None
        ),
        "speed": _speed(v, card.task, speed, native=_native_speed_input(card, v)),
        "streaming": g.stream_rows().get(v.id, []),
        "devices": _devices_of(v),
        "quality": _quality(card, v, rows),
        "example": _example(v, _examples()),
    }


def family(
    card: ModelCard, speed: dict, rows: dict, plain: dict[str, str], released: tuple[str | None, set[str]]
) -> dict[str, Any]:
    st = spec(card.task)
    return {
        "family": card.family,
        "task": card.task,
        "name": card.name,
        "plain": plain.get(card.family, ""),
        "summary": card.summary,
        "tier": card.tier,
        "upstream": card.upstream,
        "upstream_commit": card.upstream_commit,
        "original_is": g._ORIGINAL_IS.get(card.family),  # noqa: SLF001
        "papers": [
            {"title": x.title, "url": x.url, "date": x.date, "venue": x.venue, "variants": list(x.variants)}
            for x in card.papers
        ],
        "official": [
            {"repo": x.repo, "framework": x.framework, "reason": x.reason, "variants": list(x.variants)}
            for x in card.official
        ],
        "modes": list(card.modes),
        "portable_dtype": card.portable_cuda_dtype,
        "kernels_dtype": card.kernels_dtype,
        "kernel_libs": list(card.kernel_libs),
        "mps_kernel_libs": list(card.mps_kernel_libs),
        "mps_dtypes": list(card.portable_mps_dtypes),
        "no_kernels_reason": card.no_kernels_reason,
        "no_original_reason": card.no_original_reason,
        "options": [
            {"name": k, "default": val, "bounds": card.option_bounds(k), "effect": card.option_effect.get(k, "output")}
            for k, val in card.options.items()
        ],
        "outputs": [{"name": o, "extra": o not in st.standard} for o in card.outputs],
        "output_info": dict(card.output_info),
        "licence_files": list(card.licence.files),
        "licence_notes": list(card.licence_notes),
        "limitations": list(card.limitations),
        "task_io": g.TASK_IO.get(card.task, ""),
        "install": {
            "extras": [*card.extras, *(["nc"] if card.licence.code_class != "permissive" else [])],
            "noncommercial_weights": [
                v.id
                for v in card.variants
                if licences.weights_commercial(card.variant_licence(v).weights_licence) == "no"
            ],
        },
        "variants": [_variant(card, v, speed, rows, released) for v in card.variants],
        "parity": _parity(card),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    speed, plain = g._speed_rows(), g._plain()  # noqa: SLF001
    ref = _devices.reference()
    released = released_ids()
    index = []
    for task in registry.TASKS:
        rows = g._eval_rows(task)  # noqa: SLF001
        for card in {c.family: c for c, _ in registry.cards(task).values()}.values():
            data = family(card, speed, rows, plain, released)
            (args.out / f"{card.family}.json").write_text(json.dumps(data, default=str) + "\n", encoding="utf-8")
            index.append(
                {
                    "family": card.family,
                    "task": task,
                    "name": card.name,
                    "variants": [{k: v[k] for k in ("id", "name", "released", "added_in")} for v in data["variants"]],
                }
            )
    meta = {
        "release": released[0],
        "reference": {"slug": ref.slug, "name": ref.name, "software": ref.software},
        "families": index,
    }
    (args.out / "index.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    print(f"{len(index)} families -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
