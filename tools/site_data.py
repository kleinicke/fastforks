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
    "upstream-fp16": "original16",
    "portable32": "portable32",
    "portable": "portable",
    "kernels": "kernels",
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
    out["dtype"] = _devices.dtype_of(r) if r["mode"] != "upstream" or r.get("dtype") else "float16"
    if note := g._note(r):  # noqa: SLF001 - shared docs helper
        out["note"] = note
    if r.get("commit"):
        out["commit"] = r["commit"][:12]
    return out


def _speed(v: Variant, task: str, speed: dict[str, list[dict]]) -> dict[str, Any]:
    """Reference-GPU rows of one variant: per input, the four modes plus torch.compile / TensorRT."""
    rows = speed.get(v.id, [])
    std = g._std_input(task, {})  # noqa: SLF001
    inputs = sorted({r["input"] for r in rows}, key=g._pixels)  # noqa: SLF001
    out = []
    for size in inputs:
        by = {_SPEED_KEY[r["mode"]]: _speed_row(r) for r in rows if r["input"] == size and r["mode"] in _SPEED_KEY}
        oom = [_SPEED_KEY[m] for m in _SPEED_KEY if (v.id, m, size) in g._oom()]  # noqa: SLF001
        extra = {}
        for mode in ("compile", "tensorrt"):
            c = g._compile_rows().get((v.id, mode))  # noqa: SLF001
            if c is None or c["input"] != size:
                continue
            if "fails" in c:
                extra[mode] = {"fails": g._fail_reason(c["fails"])}  # noqa: SLF001
                continue
            extra[mode] = {
                "ms": c["ms"],
                "peak_alloc_mb": c.get("peak_alloc_mb"),
                "peak_reserved_mb": c.get("peak_reserved_mb"),
                "device_mb": c.get("device_mb"),
                "engine_dtype": c.get("engine_dtype", "float16") if mode == "tensorrt" else None,
                "part": g._engine_part(v.id).strip(" ()") if mode == "tensorrt" else "",  # noqa: SLF001
                "setup_s": c.get("compile_s")
                if mode == "compile"
                else c.get("onnx_export_s", 0) + c.get("engine_build_s", 0),
                "date": c.get("date"),
            }
        extra |= _pose_parts(v, size)
        scene = next((r.get("scene", "") for r in rows if r["input"] == size), "")
        out.append({"input": size, "scene": g._scene(scene), "standard": size == std, "modes": by, "oom": oom, **extra})  # noqa: SLF001
    return {"standard": std, "inputs": out}


def _pose_parts(v: Variant, size: str) -> dict[str, Any]:
    """Pose split timings of one variant and input (speed_compile.jsonl): the original's network alone, our steps."""
    rows = g._compile_rows()  # noqa: SLF001
    out: dict[str, Any] = {}
    net = {}  # the original's network alone (crops made untimed), side "upstream"
    for mode, key in (("upstream-network-default", "original32"), ("upstream-network", "original16")):
        c = rows.get((v.id, mode))
        if c is not None and c["input"] == size and "ms" in c:
            net[key] = _speed_row(c) | {"timed": c.get("timed", "")}
    if net:
        out["original_network"] = net
    parts = {}  # where our call spends its time (crop / network / rest), modes "<mode>-sections"
    for mode in ("portable", "kernels"):
        c = rows.get((v.id, f"{mode}-sections"))
        if c is not None and c["input"] == size:
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
    for d, title in g._DATASETS[task].items():  # noqa: SLF001
        ds = [r for r in mine if r["dataset"] == d]
        if not ds:
            continue
        metrics, count, noun = g._METRICS.get((task, d)) or g._METRICS[task]  # noqa: SLF001
        modes = {}
        for r in ds:
            key = _EVAL_KEY.get(r["mode"])
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
            }
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


_LOWER = {"bad", "epe", "rel_absrel", "absrel", "mae_mm", "fl_all"}  # lower is better; every other metric: higher


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


def released_ids() -> tuple[str | None, set[str]]:
    """The latest release tag and the variant ids its cards list (all ids empty without a tag)."""

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True, check=True).stdout

    tags = git("tag", "--list", "v*", "--sort=-v:refname").split()
    if not tags:
        return None, set()
    ids: set[str] = set()
    for path in git("ls-tree", "-r", "--name-only", tags[0]).split():
        if path.endswith("/card.toml"):
            ids |= {v["id"] for v in tomllib.loads(git("show", f"{tags[0]}:{path}")).get("variants", [])}
    return tags[0].removeprefix("v"), ids


_EXAMPLE_KEEP = ("top", "classes", "objects", "prompt", "mode", "dtype", "device", "date")


@functools.cache
def _examples() -> dict[str, dict[str, Any]]:
    """Latest successful row per ref of docs/data/examples.jsonl (tools/examples.py)."""
    path = _ROOT / "docs/data/examples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {r["ref"]: r for r in rows if "error" not in r}


def _example(v: Variant, rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The card's example: our output on the task's fixed input (images under the site's /examples/)."""
    r = rows.get(v.id)
    if r is None:
        return None
    _, title, credit, licence, source = examples.INPUTS[r["input"]]
    views = 2 if r["task"] in {"stereo", "flow", "multiview"} else 1
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
        "speed": _speed(v, card.task, speed),
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
            (args.out / f"{card.family}.json").write_text(json.dumps(data, default=str) + "\n")
            index.append(
                {
                    "family": card.family,
                    "task": task,
                    "name": card.name,
                    "variants": [{k: v[k] for k in ("id", "name", "released")} for v in data["variants"]],
                }
            )
    meta = {
        "release": released[0],
        "reference": {"slug": ref.slug, "name": ref.name, "software": ref.software},
        "families": index,
    }
    (args.out / "index.json").write_text(json.dumps(meta) + "\n")
    print(f"{len(index)} families -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
