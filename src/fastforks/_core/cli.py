"""`fastforks` command line: list, inspect, benchmark and export models; pre-fetch weights and kernel binaries.

    fastforks list [PATTERN] [--task TASK] [--exclude PATTERN ...] [--commercial yes|no|unclear]
                   [--tier stable|experimental|deprecated] [--runnable]
    fastforks info MODEL
    fastforks download [MODEL ...] [--kernels | --kernels-only] [--arch sm_XX[,sm_YY] | --all-archs] [--commercial-only]
                       [--dry-run | --check]
    fastforks bench MODEL [--mode auto|portable|kernels] [--dtype float32|float16|bfloat16] [--device DEV] [--size HxW]
                    [--views N] [--runs N]
    fastforks export MODEL --format onnx|trt [--size HxW [--size-range MINxMIN:MAXxMAX]] [--fp16] [--output PATH]
    fastforks doctor
    fastforks env
    fastforks stereo run LEFT RIGHT --model MODEL --out DIR [--rig CALIB [--size HxW] [--focal PX]]
                         [--mode auto|portable|kernels] [--device DEV]
                         [--format .tif|.npy|.pfm|.png] [--bits N] [--skip-existing] [--skip-errors]

`download` without models fetches every installed model. `--kernels-only` fetches only the kernel binaries, no weights
(every library of the installed models, or just the named models' libraries). `--kernels` adds the binaries of the
kernel libraries the models' kernels backend needs, for the current GPU, for named targets with
`--arch` (`sm_89`, `8.9`, `native`; picked as at run time, so `sm_86` gives what an sm_86 GPU would load; no GPU
needed), or for every built target with
`--all-archs`. It prints each model's licence class; `--commercial-only` skips models not cleared for commercial
use. `--dry-run` lists the sizes and fetches nothing; `--check` fetches nothing either and exits 1 if a file cannot be
resolved without network (FASTFORKS_CACHE, the search paths, the Hugging Face / torch hub caches), and prints where each
resolves from (a last `RUN` of a Dockerfile, where those other places are empty). `--arch` needs `--kernels` or
`--kernels-only`. A file that cannot be fetched (a gated repo without access, a host that is down) does not stop
the rest: every other model and binary is still fetched, the failures are printed as `MISSING` lines at the end, and
the exit code is 1. With `FASTFORKS_OFFLINE=1` (or `HF_HUB_OFFLINE=1`) no model download is attempted at all: a file
that is not local raises an error that names the file and the command.

`bench` times one model of any task on synthetic input (the task's `bench_inputs`: an image, a pair for stereo,
`--views` views for multi-view; median wall time and peak CUDA memory; on Apple MPS the Metal driver's memory after the
runs). By default it times `portable` and the kernels backend one after the other and prints a table; `--mode` times one
backend.
`export` writes the portable model of any task to ONNX, or builds a TensorRT engine for this GPU (`trt`) and checks it
against the portable model; `--fp16` keeps the model's validated reduced precision (else float32). Without `--size`
it uses the task's default export size where there is one (classification: the card's crop; features: 224x224);
stereo, depth, segmentation and detection need `--size`, and the error names the model's valid size range.
`stereo run` runs a stereo model over many pairs (`StereoModel.run_many`: decoding, GPU work and writing overlap):
LEFT and RIGHT are directories (their images, in natural order) or quoted glob patterns (`'left/**/*.png'`), matched
by position; one map per pair goes to DIR under the left image's name. With `--rig` (a calibration file, see
`fastforks.stereo.geometry.read_calib`) the pairs are rectified and the output is depth at the original left pixels.
`env` prints the versions, the GPU, and per kernel library the binary kernels would use on it and whether it is local.
`doctor` lists the optional libraries (installed version or the extra to install), says which models cannot run
yet and why, then prints `env`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import gc
import importlib
import importlib.resources
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from fastforks._core import registry
from fastforks._core.cards import LIBRARY_EXTRA, Commercial, Tier
from fastforks._core.errors import BackendUnavailableError, FastforksError, LicenceError, OptionsError, install_hint
from fastforks._core.files import OFFLINE_ENV, _human_size, cache_root, env_flag, locate, offline, present
from fastforks._core.kernels import KERNEL_PATH_ENV, for_driver, read_manifest, select_target
from fastforks._core.sizes import EXAMPLE_SIZE, size_range

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from fastforks._core.cards import ModelCard, Variant
    from fastforks._core.files import FileSpec
    from fastforks._core.model import Backend


def _all_cards() -> dict[str, tuple[ModelCard, Variant]]:
    return {mid: entry for task in registry.TASKS for mid, entry in registry.cards(task).items()}


def _list(args: argparse.Namespace) -> None:
    ids = registry.list_models(
        [args.task] if args.task else registry.TASKS,
        args.pattern,
        exclude_filters=args.exclude or (),
        commercial=args.commercial,
        tier=args.tier,
        runnable=True if args.runnable else None,
    )
    width = max(map(len, registry.TASKS))
    entries = sorted(map(registry.resolve_any, ids), key=lambda e: registry.TASKS.index(e[0].task))  # grouped by task
    if args.format != "text":
        _list_machine(entries, args.format)
        entries = []
    for card, v in entries:
        summary = card.summary if len(card.summary) <= _SUMMARY else card.summary[: _SUMMARY - 3] + "..."
        st = card.status(v)
        commercial = (
            "yes (copyleft)"  # GPL code: commercial use allowed, with the licence's obligations
            if st.commercial == "yes" and st.code_class == "copyleft"
            else st.commercial
            if st.commercial == "yes"
            else f"{st.commercial} ({st.reasons[0].split(':')[0]})"
            if st.reasons
            else st.commercial
        )
        lic = card.licence.code[:14]  # the code licence; the commercial flag also covers the weights
        print(f"{v.ref:34} {card.task:{width}} {card.tier:12} commercial={commercial:16} {lic:14} {summary}")
    if registry.commercial_only() and args.commercial is None:
        print(f"{registry.COMMERCIAL_ONLY_ENV}=1: showing only models cleared for commercial use", file=sys.stderr)
    if not ids and args.pattern != "*":  # exit 0 like the hint below: a filter, not an error
        close = difflib.get_close_matches(args.pattern.lower(), list(_all_cards()), n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        print(f"no model matches {args.pattern!r} (with these filters){hint}", file=sys.stderr)
    if not ids and args.commercial == "yes":
        print(
            "no installed model is cleared for commercial use (code and weights licence both allow it); models with"
            " custom or unclear licence terms: fastforks list --commercial unclear",
            file=sys.stderr,
        )


_SUMMARY = 70
_LIST_FIELDS = (
    "id", "task", "tier", "commercial", "commercial_reasons", "code_licence", "weights_licence", "training_data",
    "notes", "revision", "weights_urls", "weights_sha256",
)  # fmt: skip


def _list_machine(entries: Sequence[tuple[ModelCard, Variant]], fmt: str) -> None:
    """`list --format json|csv`: one record per model with the audit fields (licences, data notes, weights, hashes)."""
    rows = []
    for card, v in entries:
        st = card.status(v)
        rows.append(
            {
                "id": v.ref,
                "task": card.task,
                "tier": card.tier,
                "commercial": st.commercial,
                "commercial_reasons": list(st.reasons),
                "code_licence": card.licence.code,
                "weights_licence": st.weights,
                "training_data": list(st.training_data),
                "notes": list(st.notes),
                "revision": v.revision,
                "weights_urls": [w.url for w in v.weights if w.url],
                "weights_sha256": [w.sha256 for w in v.weights],
            }
        )
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return
    out = csv.DictWriter(sys.stdout, fieldnames=_LIST_FIELDS, lineterminator="\n")
    out.writeheader()
    for r in rows:
        out.writerow({k: ";".join(map(str, val)) if isinstance(val, list) else val for k, val in r.items()})


def _kernel_libs_text(card: ModelCard) -> str:
    parts = [f"CUDA: {', '.join(card.kernel_libs)}"] if card.kernel_libs else []
    parts += [f"Apple MPS: {', '.join(card.mps_kernel_libs)}"] if card.mps_kernel_libs else []
    return f" (kernel libs: {'; '.join(parts)})" if parts else ""


def _option_text(card: ModelCard, name: str, value: object) -> str:
    """`iters=32 (>= 1)`, plus the effect (`speed`, `precision`) of an option that does not change the results."""
    bounds, effect = card.option_bounds(name), card.option_effect.get(name)
    aliases = [a for a, n in card.option_aliases.items() if n == name]
    return (
        f"{name}={value}"
        + (f" ({bounds})" if bounds else "")
        + (f" [{effect} only]" if effect else "")
        + (f" [also: {', '.join(aliases)}]" if aliases else "")
    )


def _auto_text(v: Variant) -> str:
    from fastforks._core.model import _auto_portable  # noqa: PLC0415

    return "; auto: portable on CUDA (kernels measured no faster)" if v.id in _auto_portable() else ""


def _apple_text(card: ModelCard) -> str:
    """The `apple gpu` row of `info`: what `mode="auto"` picks there, and the dtypes."""
    if not card.portable_mps_dtypes:
        return "not validated"
    dt = ", ".join(card.portable_mps_dtypes)
    if card.mps_kernel_libs:
        kd = card.mps_kernels_dtype or card.kernels_dtype
        return f"auto: kernels (Metal, {kd}); portable: {dt} (float32 is the default)"
    return f"auto: portable; dtypes {dt} (float32 is the default)"


def _info(args: argparse.Namespace) -> None:
    card, v = registry.resolve_any(args.model)
    lic, st = card.variant_licence(v), card.status(v)
    rows = {
        "model": f"{v.id} ({card.name}, {card.task}, {card.tier})",
        "summary": card.summary,
        "original code": f"{card.upstream} @ {card.upstream_commit}",
        "licence": f"code {lic.code} ({st.code_class}), weights {st.weights}, commercial {st.commercial}"
        + (
            f" ({'; '.join(st.reasons)})"
            if st.reasons
            else " (copyleft: its obligations apply)"
            if st.code_class == "copyleft"
            else ""
        ),
        "licence notes": "; ".join(st.notes) or "none",
        "training data": ", ".join(lic.training_data) + (f" ({lic.training_note})" if lic.training_note else ""),
        "options": ", ".join(_option_text(card, k, val) for k, val in card.options.items()) or "none",
        "modes": ", ".join(card.modes) + _kernel_libs_text(card) + _auto_text(v),
        "apple gpu": _apple_text(card),
        "outputs": ", ".join(v.outputs) + "".join(f", {k} = {val}" for k, val in v.output_info.items()),
        "input size": size_range(v, card).describe(),
        "revision": f"{v.revision} (pin with `{v.ref}`)",
        "weights": ", ".join(f"{w.name} ({(w.size or 0) / 2**20:.0f} MB)" for w in v.weights),
        "weights url": "; ".join(w.url or "none (provide the file)" for w in v.weights),
        "weights sha256": "; ".join(w.sha256 for w in v.weights),
        "requires": f"{', '.join(card.requires)} ({install_hint(*card.extras)})" if card.requires else "nothing",
    }
    for k, val in rows.items():
        print(f"{k:14} {val}")


def _parse_arch(text: str) -> tuple[int, int]:
    """`sm_89`, `8.9`, `89` or `native` (the visible GPU) as a compute capability."""
    t = text.strip().lower()
    if t == "native":
        import torch  # noqa: PLC0415 - only needed to query the GPU

        if not torch.cuda.is_available():
            msg = "--arch native needs a visible CUDA GPU; name the target instead, e.g. --arch sm_89"
            raise FastforksError(msg)
        return torch.cuda.get_device_capability()
    m = re.fullmatch(r"(?:sm_)?(\d+)\.(\d)|(?:sm_)?(\d+)(\d)", t)
    if not m:
        extra = (
            " (arch-specific targets like sm_90a are not in the kernel manifests; use sm_90)" if t.endswith("a") else ""
        )
        msg = f"--arch: cannot read {text!r}; use e.g. sm_89, 8.9 or native{extra}"
        raise FastforksError(msg)
    return int(m[1] or m[3]), int(m[2] or m[4])


def _kernel_files(lib: str, *, archs: Sequence[str], all_archs: bool) -> list[tuple[str, FileSpec]]:
    """`(tag:target, file)` pairs to fetch; libraries without a manifest (e.g. cuDNN-backed) have no binaries.

    For this GPU only the build its driver loads; with `--arch` / `--all-archs` (for other machines) the older-driver
    `compat` build too.
    """
    path = Path(str(importlib.resources.files("fastforks.kernels") / lib / "manifest.toml"))
    if not path.is_file():
        return []
    manifest = read_manifest(path)
    builds = [manifest] if manifest.compat is None else [manifest, manifest.compat]
    if all_archs:
        return [(f"{m.tag}:{t}", f) for m in builds for t, f in m.files.items()]
    caps = [_parse_arch(a) for a in archs]
    if not archs:
        import torch  # noqa: PLC0415 - only needed to query the GPU

        if not torch.cuda.is_available():
            msg = f"no CUDA device to pick {lib} binaries for; name the targets with --arch sm_XX or use --all-archs"
            raise FastforksError(msg)
        caps = [torch.cuda.get_device_capability()]
        builds = [for_driver(manifest)]
    targets: dict[str, tuple[int, int]] = {}
    for cap in caps:
        target = select_target(manifest.files, cap)
        if target is None:
            msg = f"{lib}: no binary for sm_{cap[0]}{cap[1]} (built: {', '.join(manifest.files)})"
            raise FastforksError(msg)
        targets.setdefault(target, cap)
    return [(f"{m.tag}:{t}", m.files[t]) for m in builds for t in targets]


def _entries(args: argparse.Namespace) -> list[tuple[ModelCard, Variant]]:
    """The models to handle; with `--commercial-only` the cleared ones (naming a skipped one is an error).

    Models whose non-commercial weights are not enabled (`registry.nc_weights_blocked`) are skipped, or an error if
    named.
    """
    entries = [registry.resolve_any(m) for m in args.models] if args.models else list(_all_cards().values())
    env_only = registry.commercial_only()
    only = args.commercial_only or env_only
    by = "--commercial-only" if args.commercial_only else f"{registry.COMMERCIAL_ONLY_ENV}=1"
    if only:
        skipped = [v.ref for card, v in entries if card.status(v).commercial != "yes"]
        for ref in skipped:
            print(f"skipping {ref}: commercial use not cleared (see `fastforks info {ref}`)", file=sys.stderr)
        if skipped and args.models:
            msg = f"{by}: not cleared for commercial use, so excluded: {', '.join(skipped)} (drop them from the list)"
            raise (LicenceError if env_only else FastforksError)(msg)
        entries = [(card, v) for card, v in entries if card.status(v).commercial == "yes"]
    # --dry-run / --check fetch nothing: they list non-commercial models (`_weights` notes what they need)
    gate = not (args.dry_run or args.check)
    blocked = [(card, v) for card, v in entries if gate and registry.nc_weights_blocked(card, v)]
    if blocked and args.models:
        msg = f"{', '.join(v.ref for _, v in blocked)}: non-commercial weights; set {registry.NONCOMMERCIAL_ENV}=1 to "
        msg += "download them"
        raise LicenceError(msg)
    if blocked:
        print(
            f"skipping {len(blocked)} models with non-commercial weights (set {registry.NONCOMMERCIAL_ENV}=1 to include"
            " them)",
            file=sys.stderr,
        )
        entries = [(card, v) for card, v in entries if not registry.nc_weights_blocked(card, v)]
    if not only and not args.models and not args.check:
        n = sum(card.status(v).commercial != "yes" for card, v in entries)
        print(
            f"warning: no model list given; {n} of {len(entries)} models are not cleared for commercial use and are"
            " fetched too. Name your models, or use --commercial-only, before baking weights into an image you share",
            file=sys.stderr,
        )
    return entries


def _kernel_entries(args: argparse.Namespace) -> list[tuple[ModelCard, Variant]]:
    """The models whose kernel libraries `--kernels-only` fetches; no weights are touched, so no licence filtering."""
    return [registry.resolve_any(m) for m in args.models] if args.models else list(_all_cards().values())


def _apple_gpu_only() -> bool:
    """No CUDA device but an Apple GPU: kernels mode here uses Metal kernels, compiled on the device."""
    import torch  # noqa: PLC0415 - only needed to query the GPU

    return not torch.cuda.is_available() and torch.backends.mps.is_available()


def _metal_notes(entries: Sequence[tuple[ModelCard, Variant]], *, apple_gpu: bool) -> None:
    """Say that Metal kernels have nothing to fetch (and, on an Apple GPU, which models have no kernels there)."""
    for card, v in entries:
        if card.mps_kernel_libs and (apple_gpu or not card.kernel_libs):
            libs = ", ".join(card.mps_kernel_libs)
            print(f"{v.ref}: Metal kernels ({libs}) are compiled locally on first use; nothing to fetch")
        elif apple_gpu and card.kernel_libs:
            print(f"{v.ref}: no kernels for Apple GPUs (runs portable); --arch sm_XX fetches CUDA binaries for others")


def _download(args: argparse.Namespace) -> int:
    archs = [a for part in args.arch or [] for a in part.split(",") if a]
    kernels = args.kernels or args.kernels_only
    if args.arch is not None and not (kernels and archs):
        msg = "--arch needs --kernels (or --kernels-only) and a non-empty target list (e.g. --kernels --arch sm_89)"
        raise FastforksError(msg)
    if offline() and not (args.check or args.dry_run):
        msg = f"offline mode ({OFFLINE_ENV} / HF_HUB_OFFLINE) is set; unset it to download (--check works offline)"
        raise FastforksError(msg)
    entries = _kernel_entries(args) if args.kernels_only else _entries(args)
    apple_gpu = kernels and not archs and not args.all_archs and _apple_gpu_only()
    if kernels:
        _metal_notes(entries, apple_gpu=apple_gpu)
    plan = (  # resolved before the first fetch, so a typo in --arch fails fast and nothing is half-fetched
        [
            (lib, tag, spec)
            for lib in sorted({lib for card, _ in entries for lib in card.kernel_libs})
            for tag, spec in _kernel_files(lib, archs=archs, all_archs=args.all_archs)
        ]
        if kernels and not apple_gpu
        else []
    )
    missing = [] if args.kernels_only else _weights(args, entries)
    for lib, tag, spec in plan:
        missing += _kernel(args, lib, tag, spec)
    for line in missing:
        print(f"MISSING {line}", file=sys.stderr)
    return 1 if missing else 0


def _size_text(n: int) -> str:
    return _human_size(n) if n else "size unknown"


def _weights(args: argparse.Namespace, entries: Sequence[tuple[ModelCard, Variant]]) -> list[str]:
    """Print each model's licence class and size, fetch its weights (`--check`: look only); returns what is missing.

    A model that fails to download does not stop the others: it is reported as missing at the end.
    """
    missing: list[str] = []
    for card, v in entries:
        st = card.status(v)
        # what goes over the wire: an archive once (its members share a url), not the members' sizes
        wire = {w.url or w.name: (w.archive_size if w.member else w.size) or 0 for w in v.weights}
        size = _size_text(sum(wire.values()))
        print(f"{v.ref}: licence {card.licence.code}, weights {st.weights}, commercial {st.commercial}, {size}")
        if bad := registry.problems(card):
            why = ", ".join(f"{x} ({w})" for x, w in bad.items())
            print(f"note: {v.ref} needs {why}: {install_hint(*card.extras)}", file=sys.stderr)
        nc = registry.nc_weights_blocked(card, v)
        if nc and args.dry_run:
            print(f"{v.ref}: needs {registry.NONCOMMERCIAL_ENV}=1 to download (non-commercial weights)")
        if args.check:  # a file that resolves locally is ok; the gate only matters for files that would be fetched
            found = _missing(v.ref, lambda card=card, v=v: registry.weights(card, v, allow_download=False))
            missing += [f"{m}; needs {registry.NONCOMMERCIAL_ENV}=1 to download" for m in found] if nc else found
        elif not args.dry_run:
            try:
                paths = registry.weights(card, v)
            except FastforksError as e:  # keep going: the other models still download, MISSING lists this one
                missing.append(f"{v.ref}: {e}")
                continue
            for p in paths:
                print(f"{v.ref}: {p}")
    return missing


def _kernel(args: argparse.Namespace, lib: str, tag: str, spec: FileSpec) -> list[str]:
    """Fetch (or `--dry-run`: size, `--check`: look for) one kernel binary; returns what is missing."""
    if args.dry_run:
        print(f"{tag}: {_size_text(spec.size or 0)}")
    elif args.check:
        return _missing(tag, lambda: [locate(spec, group=lib, env_var=KERNEL_PATH_ENV, allow_download=False)])
    else:
        try:
            path = locate(spec, group=lib, env_var=KERNEL_PATH_ENV)
        except FastforksError as e:
            return [f"{tag}: {e}"]
        print(f"{tag}: {path}")
    return []


def _missing(what: str, probe: Callable[[], Sequence[Path]]) -> list[str]:
    """`[]` if `probe()` finds the files without network (printing where each resolves from), else what is missing."""
    try:
        paths = probe()
    except FastforksError as e:
        return [f"{what}: {e}"]
    print(f"{what}: ok ({', '.join(map(str, paths))})")
    return []


def _bounds(text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    lo, sep, hi = text.partition(":")
    if not sep:
        msg = f"size range must be MINxMIN:MAXxMAX, e.g. 320x320:1088x1920, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return _size(lo), _size(hi)


def _dyn(bounds: tuple[tuple[int, int], tuple[int, int]] | None) -> str:
    return "" if bounds is None else f", dynamic {bounds[0][0]}x{bounds[0][1]}:{bounds[1][0]}x{bounds[1][1]}"


def _size(text: str) -> tuple[int, int]:
    h, sep, w = text.lower().partition("x")
    if not sep or not h.isdigit() or not w.isdigit() or int(h) < 1 or int(w) < 1:
        msg = f"size must be HxW with both >= 1, e.g. 375x1242, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(h), int(w)


def _positive(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        msg = f"must be an integer >= 1, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(text)


def _bench_mode(
    args: argparse.Namespace, card: ModelCard, mode: str
) -> tuple[str, Backend, float, tuple[int | None, int] | None]:
    """Load `card`'s model in `mode`, time it; returns ref, backend, median ms and peak `(allocated, reserved)` MiB.

    On MPS (no peak counters) reserved is the Metal driver's memory after the timed runs: its allocator keeps freed
    buffers, so that covers the peak; allocated is `None` there. No memory on the CPU.
    """
    import torch  # noqa: PLC0415 - heavy, only for these commands

    from fastforks._core.bench import BenchSize, bench_inputs, median_ms  # noqa: PLC0415

    model = importlib.import_module(f"fastforks.{card.task}").load(
        args.model, mode=mode, dtype=args.dtype, device=args.device
    )
    inputs, kwargs = bench_inputs(model, BenchSize(*args.size, views=args.views))
    cuda = model.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    ms = median_ms(lambda: model(*inputs, **kwargs), device=model.device.type, runs=args.runs)
    mem = None
    if cuda:
        mem = (
            round(torch.cuda.max_memory_allocated(model.device) / 2**20),
            round(torch.cuda.max_memory_reserved(model.device) / 2**20),
        )
    elif model.device.type == "mps":
        mem = (None, round(torch.mps.driver_allocated_memory() / 2**20))
    return model.variant.ref, model.backend, ms, mem


def _bench(args: argparse.Namespace) -> None:
    """Time one backend (`--mode`), or portable and kernels side by side where both can run (the default)."""
    card, _ = registry.resolve_any(args.model)
    if args.mode is not None:
        ref, backend, ms, mem = _bench_mode(args, card, args.mode)
        alloc, reserved = mem or (None, None)
        peak = f", peak {alloc} MiB allocated" if alloc is not None else ""
        peak += f"{' /' if peak else ', peak'} {reserved} MiB reserved" if reserved is not None else ""
        print(
            f"{ref} {args.size[0]}x{args.size[1]} {backend.mode} {backend.dtype}: {ms:.1f} ms"
            f" (median of {args.runs}){peak}"
        )
        return
    import torch  # noqa: PLC0415

    from fastforks._core.model import resolve_device  # noqa: PLC0415

    dev = resolve_device(args.device, card)
    modes = ["portable"]
    if "kernels" in card.modes and dev.type in ("cuda", "mps") and card.kernel_libs_on(dev):
        modes.append("kernels")
    if len(modes) == 1:  # nothing to compare: the one-line form, with a hint
        args.mode = "portable"
        _bench(args)
        print("(no kernels backend on this model or device; --mode kernels needs a supported CUDA GPU or Apple MPS)")
        return
    rows = []
    for mode in modes:
        try:
            _, backend, ms, mem = _bench_mode(args, card, mode)
        except (BackendUnavailableError, OptionsError) as e:  # e.g. float32 where kernels compute in float16
            rows.append((mode, "-", "not available: " + str(e).split("\n")[0], "", ""))
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # the next backend's peak must not include this one's
        alloc, reserved = mem or (None, None)
        rows.append(
            (
                mode,
                backend.dtype,
                f"{ms:.1f}",
                "-" if alloc is None else str(alloc),
                "-" if reserved is None else str(reserved),
            )
        )
    head = ("mode", "dtype", "ms", "peak allocated MiB", "peak reserved MiB")
    widths = [max(len(r[i]) for r in (head, *rows)) for i in range(len(head))]
    print(f"{args.model}, input {args.size[0]}x{args.size[1]}, median of {args.runs} runs")
    for r in (head, *rows):
        print("  ".join(f"{c:>{w}}" if i > 1 else f"{c:{w}}" for i, (c, w) in enumerate(zip(r, widths, strict=True))))


_QUERY_SCORE = 0.3  # detection transformers: queries above this score are compared (matched by box)


def _parity(
    want: dict[str, Any], got: dict[str, Any], *, stereo: bool, scores: Callable[..., Any] | None = None
) -> str:
    """The CLI's TensorRT check: every output of `module()` (`want`, by export name) vs the engine's (`got`).

    Reports the output with the largest max |difference| relative to its portable range (stereo: in px). With
    `scores` (`module().query_scores`: logits -> per-query score) the `logits` / `boxes` queries scoring above 0.3 are
    matched by box instead: an encoder top-k (RT-DETR) picks and orders the queries, so near-tied picks swap.
    """
    if missing := sorted(set(want) - set(got)):
        return f"the engine has no output(s) {missing} (engine outputs: {sorted(got)})"
    if scores is not None and {"logits", "boxes"} <= set(want):
        return _matched_queries(want, got, scores)
    diffs = {n: (got[n].float() - w.float()).abs() for n, w in want.items() if n in got}
    if stereo:
        d = diffs["disparity"] if "disparity" in diffs else next(iter(diffs.values()))
        return (
            f"mean |d disparity| {d.mean():.4f} px, max {d.max():.3f} px, > 1 px {(d > 1).float().mean() * 100:.3f} %"
        )

    def rel(n: str) -> float:
        return float(diffs[n].max()) / max(float(want[n].float().abs().max()), 1e-12) if diffs[n].numel() else 0.0

    name = max(diffs, key=rel)
    d, w = diffs[name], want[name].float()
    lo, hi = (w.min(), w.max()) if w.numel() else (0.0, 0.0)
    worst = f", worst of {len(diffs)} outputs" if len(diffs) > 1 else ""
    return f"mean |d {name}| {d.mean():.4g}, max {d.max():.4g} (portable {name} in [{lo:.4g}, {hi:.4g}]{worst})"


def _matched_queries(want: dict[str, Any], got: dict[str, Any], scores: Callable[..., Any]) -> str:
    """`_parity` for query outputs: each portable query above `_QUERY_SCORE` vs the engine's query nearest by box."""
    import torch  # noqa: PLC0415 - heavy, only for export

    sw, sg = scores(want["logits"].float()), scores(got["logits"].float())  # [B, Q]
    bw, bg = want["boxes"].float(), got["boxes"].float()
    keep = sw > _QUERY_SCORE
    n_w, n_g = int(keep.sum()), int((sg > _QUERY_SCORE).sum())
    if not n_w:
        return f"no query scores > {_QUERY_SCORE} on the synthetic image ({n_g} TensorRT); run the model on real images"
    d_box = d_score = 0.0
    for b in range(bw.shape[0]):
        if keep[b].any():
            j = torch.cdist(bw[b][keep[b]], bg[b], p=1).argmin(-1)  # the engine's nearest query by box
            d_box = max(d_box, float((bw[b][keep[b]] - bg[b][j]).abs().max()))
            d_score = max(d_score, float((sw[b][keep[b]] - sg[b][j]).abs().max()))
    return (
        f"{n_w} / {n_g} queries score > {_QUERY_SCORE} (portable / TensorRT), matched by box: max |d boxes| "
        f"{d_box:.3g}, max |d score| {d_score:.3g}"
    )


def _export(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415 - heavy, only for these commands

    from fastforks._core.bench import median_ms  # noqa: PLC0415
    from fastforks._core.export import (  # noqa: PLC0415
        ENGINE_NOTE,
        TensorRTEngine,
        flat_outputs,
        output_names,
        resolve_size,
    )

    if args.output:  # before the export work: a missing parent directory is created, other OS errors are reported
        if Path(args.output).is_dir():
            msg = f"--output {args.output} is a directory: name the file, e.g. {Path(args.output) / 'model.onnx'}"
            raise FastforksError(msg)
        try:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"cannot write {args.output}: {e}"
            raise FastforksError(msg) from e
    card, _ = registry.resolve_any(args.model)
    if args.format == "trt" and not torch.cuda.is_available():
        msg = "a TensorRT engine needs a CUDA GPU (build it on the machine that will run it)"
        raise FastforksError(msg)
    api = importlib.import_module(f"fastforks.{card.task}")
    model = api.load(args.model, mode="portable", dtype=None if args.fp16 else "float32")
    if args.fp16 and model.backend.dtype == "float32":
        print(f"note: --fp16 has no effect on {model.device.type}: float32 export", file=sys.stderr)
    h, w = args.size or resolve_size(model, None, None)
    bounds = args.size_range
    suffix = "onnx" if args.format == "onnx" else "engine"
    out = Path(args.output or f"{model.variant.id}_{h}x{w}_{model.backend.dtype}.{suffix}")
    if args.format == "onnx":
        model.export_onnx(out, h, w, size_range=bounds)
        print(f"wrote {out} ({model.backend}, {h}x{w}{_dyn(bounds)})")
        return
    model.export_tensorrt(out, h, w, size_range=bounds)
    engine = TensorRTEngine(out, model.device)
    inputs = model.example_inputs(h, w)
    module = model.module()
    # a side stream: TensorRT warns on (and may serialise with) the legacy default stream
    side = (
        torch.cuda.stream(torch.cuda.Stream(model.device)) if model.device.type == "cuda" else contextlib.nullcontext()
    )
    with torch.inference_mode(), side:
        ref = module(*inputs)  # tuple / dict outputs (detection, multi-view): every tensor, by its export name
        want = dict(zip(output_names(model, module, ref), flat_outputs(ref), strict=True))
        got = engine.run(*inputs)
        t_trt = median_ms(lambda: engine.run(*inputs), device="cuda")
        t_ref = median_ms(lambda: module(*inputs), device="cuda")
    print(
        f"wrote {out}: TensorRT engine of {model.variant.ref} at {h}x{w}{_dyn(bounds)}, {model.backend.dtype}"
        f" ({ENGINE_NOTE})"
    )
    what = {"stereo": "pair", "flow": "pair", "matching": "pair", "interpolation": "pair", "tracking": "video"}.get(
        card.task, "image"
    )
    stats = _parity(want, got, stereo=card.task == "stereo", scores=getattr(module, "query_scores", None))
    print(f"check vs portable (synthetic {what}): {stats}; {t_trt:.1f} ms TensorRT vs {t_ref:.1f} ms portable")


def _stereo_run(args: argparse.Namespace) -> int:
    import time  # noqa: PLC0415

    from fastforks import stereo  # noqa: PLC0415 - heavy, only for this command
    from fastforks.stereo.pipeline import check_inputs, check_output_dir  # noqa: PLC0415

    check_inputs(args.left, args.right)  # cheap input errors before the model loads
    check_output_dir(args.out)
    rig = args.rig
    if args.size is not None:  # the file stores no image size: read it with the size given here
        if rig is None:
            msg = "--size is the image size for the --rig calibration file; pass --rig"
            raise FastforksError(msg)
        from fastforks.stereo.geometry import read_calib  # noqa: PLC0415

        try:
            rig = read_calib(rig, size=args.size)
        except FileNotFoundError as e:
            msg = f"calibration file not found: {rig}"
            raise FastforksError(msg) from e
    model = stereo.load(args.model, mode=args.mode, device=args.device)
    t0 = time.perf_counter()
    done = existing = failed = 0
    with model.run_many(
        args.left,
        args.right,
        rig=rig,
        focal=args.focal,
        save_to=args.out,
        save_format=args.format,
        bits=args.bits,
        max_depth=args.max_depth,
        skip_existing=args.skip_existing,
        on_error="skip" if args.skip_errors else "raise",
    ) as results:
        for r in results:
            if r.error is not None:
                failed += 1
                print(f"fastforks: failed pair {r.index} ({r.left_path}, {r.right_path}): {r.error}", file=sys.stderr)
            elif r.existing:
                existing += 1
            else:
                done += 1
        stats = results.stats
    total = time.perf_counter() - t0
    first = f", first result after {stats.first_result_s:.2f} s" if stats.first_result_s is not None else ""
    extra = (f", {existing} existing" if existing else "") + (f", {failed} failed" if failed else "")
    print(
        f"{model.variant.ref} {model.backend}: {done} pairs -> {args.out} in {total:.2f} s"
        f" ({done / total:.2f} pairs/s{first}){extra}"
    )
    return 1 if failed else 0


def _kernel_status(lib: str, arch: tuple[int, int] | None, driver: tuple[int, int] | None, cudnn: str) -> str:
    """Which binary of `lib` kernels would use on a GPU of compute capability `arch`, and whether it is local."""
    path = Path(str(importlib.resources.files("fastforks.kernels") / lib / "manifest.toml"))
    if lib == "cudnn_hacc":  # cuDNN runs the convs; its binary only builds the plans faster (ctypes without it)
        builder = _binary_status(lib, path, arch, driver) if path.is_file() else "none"
        return f"cuDNN plans via the cuDNN backend API ({cudnn}); plan builder: {builder}"
    if not path.is_file():
        return f"no binaries: cuDNN plans via the cuDNN backend API ({cudnn})"
    return _binary_status(lib, path, arch, driver)


def _binary_status(lib: str, path: Path, arch: tuple[int, int] | None, driver: tuple[int, int] | None) -> str:
    full = read_manifest(path)
    manifest = full if driver is None else for_driver(full, driver)
    if arch is None:
        return f"built for {', '.join(manifest.files)}; no CUDA GPU"
    target = select_target(manifest.files, arch)
    if target is None:
        return f"no binary for sm_{arch[0]}{arch[1]} (built: {', '.join(manifest.files)})"
    if driver is not None and driver < manifest.toolkit:
        need = ".".join(map(str, manifest.toolkit))
        return f"{target}, but needs a driver for CUDA >= {need} (have {driver[0]}.{driver[1]})"
    try:
        locate(manifest.files[target], group=lib, env_var=KERNEL_PATH_ENV, allow_download=False)
    except FastforksError:
        return f"{target}, not downloaded yet (fastforks download --kernels)"
    jit = " (PTX, JIT-compiled on first load)" if target.startswith("compute_") else ""
    old = "" if manifest is full else f" (its CUDA {'.'.join(map(str, manifest.toolkit))} build for this driver)"
    return f"{target}{jit}{old}, local"


def _cpu_text(torch: ModuleType, *, gpu: bool) -> str:
    """`env`'s one CPU line: threads, cores, the torch build, and what a model runs on without a GPU (CUDA or MPS)."""
    build = "CPU-only" if torch.version.cuda is None and torch.version.hip is None else f"CUDA {torch.version.cuda}"
    text = f"{torch.get_num_threads()} torch threads, {os.cpu_count()} cores, torch {build} build"
    return text if gpu else f"{text}; no GPU: models run portable on the CPU (float32)"


def _flag_text(name: str) -> str:
    try:
        return "on" if env_flag(name) else "off"
    except OptionsError as e:  # a bug report should show the bad value, not die on it
        return f"invalid ({e})"


def _cache_text() -> str:
    """The cache directory and its size (one walk of the cache)."""
    try:
        root = cache_root()
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.is_dir() else 0
    except (OSError, FastforksError) as e:
        return f"unusable: {e}"
    return f"{root} ({_human_size(size)})"


def _env(_args: argparse.Namespace) -> None:
    import platform  # noqa: PLC0415

    import torch  # noqa: PLC0415 - heavy, only for these commands

    from fastforks import __version__  # noqa: PLC0415
    from fastforks._core.kernels import driver_version  # noqa: PLC0415

    try:
        driver: tuple[int, int] | None = driver_version()
    except (OSError, RuntimeError):  # no libcuda.so.1, or a failing cuDriverGetVersion
        driver = None
    cuda = torch.cuda.is_available()
    arch = torch.cuda.get_device_capability() if cuda else None
    cards = [card for card, _ in _all_cards().values()]
    rows = {
        "fastforks": __version__,
        "python": f"{platform.python_version()} ({sys.platform} {platform.machine()})",
        "torch": f"{torch.__version__} (CUDA {torch.version.cuda or '-'}, cuDNN {torch.backends.cudnn.version()})",
    }
    rows["cpu"] = _cpu_text(torch, gpu=cuda or torch.backends.mps.is_available())
    if sys.platform == "darwin" or torch.backends.mps.is_available():  # Apple GPU first on a Mac
        mps = torch.backends.mps.is_available()
        mac = platform.mac_ver()[0]
        rows["apple gpu"] = f"MPS available (macOS {mac or '?'})" if mps else "MPS not available"
        metal = sorted({lib for card in cards for lib in card.mps_kernel_libs})
        rows |= {f"metal {lib}": "Metal source, compiled on first use" for lib in metal} if mps else {}
    if not cuda and driver is None:  # no NVIDIA driver at all (e.g. a Mac): per-library CUDA rows say nothing
        rows["cuda"] = "not available (no NVIDIA driver); CUDA kernels unused"
    else:
        try:
            from fastforks.kernels.cudnn_hacc import _backend  # noqa: PLC0415 - loads libcudnn.so.9

            cudnn = f"cuDNN {_backend.version_str(_backend.version())}"
        except Exception as e:  # noqa: BLE001 - a broken install (missing libcudnn ...) is what `env` should show
            cudnn = f"unusable: {type(e).__name__}: {e}"
        rows["driver"] = f"supports CUDA {driver[0]}.{driver[1]}" if driver else "no NVIDIA driver (libcuda.so.1) found"
        rows["gpu"] = f"{torch.cuda.get_device_name()}, sm_{arch[0]}{arch[1]}" if arch else "no usable CUDA GPU"
        libs = sorted({lib for card in cards for lib in card.kernel_libs})
        by_status: dict[str, list[str]] = {}  # identical lines (most libraries) collapse into one
        for lib in libs:
            by_status.setdefault(_kernel_status(lib, arch, driver, cudnn), []).append(lib)
        for text, group in by_status.items():
            rows[f"kernels {group[0]}" if len(group) == 1 else f"kernels ({len(group)} libraries)"] = text
    rows["cache"] = _cache_text()
    rows["switches"] = ", ".join(
        f"{name.removeprefix('FASTFORKS_').lower()} {_flag_text(name)}"
        for name in (OFFLINE_ENV, registry.NONCOMMERCIAL_ENV, registry.COMMERCIAL_ONLY_ENV)
    )
    for k, val in rows.items():
        print(f"{k:22} {val}")


def _add_stereo(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stereo", help="stereo commands").add_subparsers(required=True)
    p = p.add_parser("run", help="a stereo model over many pairs: two folders (or globs) in, maps out")
    p.add_argument("left", help="directory of left images or a quoted glob pattern ('left/**/*.png')")
    p.add_argument("right", help="directory of right images or a glob pattern, matched to LEFT by position")
    p.add_argument("--model", required=True, help="stereo model id (see `fastforks list --task stereo`)")
    p.add_argument("--out", required=True, help="output directory (one file per pair, named after the left image)")
    p.add_argument("--rig", help="calibration file: rectify the pairs and write depth at the original left pixels")
    p.add_argument(
        "--size",
        type=_size,
        help="HxW of the images, for a --rig calibration file that stores no image size (e.g. OpenCV yaml)",
    )
    p.add_argument(
        "--focal", type=float, help="focal length in pixels of the rectified pairs (default: the left camera's)"
    )
    p.add_argument(
        "--mode",
        choices=("auto", "portable", "kernels"),
        default="auto",
        help="backend (default auto: kernels if it can run and is faster here)",
    )
    p.add_argument(
        "--device", help="e.g. cpu, cuda, cuda:1, mps (default: cuda if available, else mps if validated, else cpu)"
    )
    p.add_argument(
        "--format",
        choices=(".tif", ".npy", ".pfm", ".png"),
        default=".tif",
        help="float32 TIFF (default), .npy, .pfm, or 16-bit KITTI disparity .png (not with --rig)",
    )
    p.add_argument("--bits", type=int, help="TIFF: keep this many mantissa bits (smaller files; default lossless)")
    p.add_argument(
        "--max-depth",
        type=float,
        help="with --rig: write depth beyond this many metres as NaN (default: keep all; disparity ~0 is huge depth)",
    )
    p.add_argument("--skip-existing", action="store_true", help="skip pairs whose output file exists (resume a run)")
    p.add_argument("--skip-errors", action="store_true", help="report unreadable pairs and go on instead of stopping")
    p.set_defaults(run=_stereo_run)


def _sfm(args: argparse.Namespace) -> None:
    from fastforks.matching import sfm  # noqa: PLC0415 - heavy, only for this command

    options = {
        "matcher": args.matcher,
        "retrieval": None if args.retrieval == "none" else args.retrieval,
        "k": args.k,
        "threshold": args.threshold,
        "single_camera": args.single_camera,
        "mode": args.mode,
        "device": args.device,
    }
    if args.reconstruct:
        rec = sfm.reconstruct(args.images, args.out, mapper=args.mapper, **options)
        stats, target = rec.database, Path(args.out) / "database.db"
    else:
        stats, target = sfm.colmap_database(args.images, args.out, **options), Path(args.out)
    times = ", ".join(f"{name} {s:.1f} s" for name, s in stats.seconds.items())
    print(f"{stats.images} images, {stats.pairs} pairs, {stats.verified} verified ({times}) -> {target}")
    if args.reconstruct:
        built = f"{rec.registered}/{stats.images} images, {rec.points} points"
        print(f"{rec.mapper} mapper {rec.seconds:.1f} s: {built} -> {rec.model}")


def _add_sfm(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sfm", help="an image folder -> a COLMAP database (pairs, matches, verified geometries)")
    p.add_argument("images", help="image folder (pass the same folder as the mapper's --image_path)")
    p.add_argument("out", help="COLMAP database file to write (must not exist); with --reconstruct: the output folder")
    p.add_argument("--reconstruct", action="store_true", help="also run pycolmap's mapper: out/database.db, out/sparse")
    p.add_argument("--mapper", choices=("incremental", "global"), default="incremental", help="with --reconstruct")
    p.add_argument("--matcher", default="aliked-lightglue", help="sparse matching model id (default aliked-lightglue)")
    p.add_argument("--retrieval", default="megaloc", help="retrieval model id that picks the pairs, or none: all pairs")
    p.add_argument("--k", type=int, default=20, help="pairs per image (default 20)")
    p.add_argument("--threshold", type=float, default=4.0, help="RANSAC inlier threshold in pixels (default 4)")
    p.add_argument("--single-camera", action="store_true", help="one camera for all images of the same size")
    p.add_argument("--mode", choices=("auto", "portable", "kernels"), default="auto", help="backend (default auto)")
    p.add_argument("--device", help="e.g. cpu, cuda, cuda:1 (default: as `load`)")
    p.set_defaults(run=_sfm)


_SHOWN = 6  # model ids `doctor` names before "... (N more)"


def _doctor(args: argparse.Namespace) -> None:
    """Which optional libraries are installed (versions), which models they unlock, then `env`."""
    import importlib.metadata as md  # noqa: PLC0415

    libs = {
        **{lib: LIBRARY_EXTRA[lib] for lib in LIBRARY_EXTRA},
        "onnx": "onnx",
        "onnxscript": "onnx",
        "tensorrt": "tensorrt",
        "isal": "",
    }
    for lib, extra in libs.items():
        if importlib.util.find_spec(lib) is None:
            hint = f"missing: {install_hint(extra)}" if extra else "missing (optional: faster TIFF writing)"
            print(f"{lib:22} {hint}")
            continue
        try:
            ver = md.version(lib)
        except md.PackageNotFoundError:
            ver = "installed"
        print(f"{lib:22} {ver}")
    blocked = registry.list_models(registry.TASKS, runnable=False)
    cards = _all_cards()
    total = len(registry.list_models(registry.TASKS))
    if offline():  # downloads are forbidden: only models whose weights are local can run
        usable = [m for m in registry.list_models(registry.TASKS, runnable=True) if _weights_local(*cards[m])]
        print(f"{'models':22} {len(usable)} of {total} can run now (offline: weights must be local)")
    else:
        print(f"{'models':22} {total - len(blocked)} of {total} can run (weights download on first use)")
    no_libs = [m for m in blocked if registry.missing(cards[m][0])]
    nc_weights = [m for m in blocked if m not in no_libs]
    nc_remedy = (
        f"non-commercial weights, not local: set {registry.NONCOMMERCIAL_ENV}=1 to download them, or pass weights=PATH"
    )
    for ids, remedy in ((no_libs, "install the libraries marked missing above"), (nc_weights, nc_remedy)):
        if ids:
            shown = ", ".join(ids[:_SHOWN]) + (f", ... ({len(ids) - _SHOWN} more)" if len(ids) > _SHOWN else "")
            print(f"{'cannot run yet':22} {shown}: {remedy}")
    _env(args)
    if no_libs:
        step = f"install the missing libraries, e.g. {install_hint(*cards[no_libs[0]][0].extras)}"
    elif offline():
        step = f"fill {registry.WEIGHTS_PATH_ENV} or the cache (`fastforks download MODEL` on a machine with network)"
    else:
        step = "fastforks download MODEL [--kernels] to fetch weights (and kernel binaries) ahead of first use"
    print(f"{'next step':22} {step}")


def _weights_local(card: ModelCard, v: Variant) -> bool:
    return all(present(spec, card.family, registry.WEIGHTS_PATH_ENV) for spec in v.weights)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    ap = argparse.ArgumentParser(prog="fastforks", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("list", help="installed models")
    p.add_argument("pattern", nargs="?", default="*", help="shell-style pattern on the model id (case-insensitive)")
    p.add_argument("--task", choices=registry.TASKS, help="only this task's models")
    p.add_argument("--exclude", action="append", metavar="PATTERN", help="drop ids matching PATTERN (repeatable)")
    p.add_argument("--commercial", choices=get_args(Commercial), help="only models with this commercial-use status")
    p.add_argument("--tier", choices=get_args(Tier), help="only models of this support tier")
    p.add_argument(
        "--runnable",
        action="store_true",
        help=(
            "only models whose optional libraries (such as timm) are installed and whose non-commercial opt-in is set;"
            " weight files and FASTFORKS_OFFLINE are not checked"
        ),
    )
    p.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="json / csv: one record per model with licences, data notes, weights URLs and SHA-256 (for audits)",
    )
    p.set_defaults(run=_list)
    p = sub.add_parser("info", help="model card of one model")
    p.add_argument("model", help="model id, e.g. depth-anything-v2-s (see `fastforks list`)")
    p.set_defaults(run=_info)
    p = sub.add_parser("download", help="fetch weights (and kernel binaries) into the cache")
    p.add_argument("models", nargs="*", help="model ids (default: all installed models)")
    p.add_argument("--kernels", action="store_true", help="also the kernels-backend kernel binaries")
    p.add_argument(
        "--kernels-only",
        action="store_true",
        help="only the kernel binaries (implies --kernels), no weights; without models every library",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--arch",
        action="append",
        metavar="SM[,SM]",
        help="kernel binaries for these targets (sm_89, 8.9, native), as the runtime picks them; no GPU needed",
    )
    g.add_argument("--all-archs", action="store_true", help="kernel binaries for every target, not just this GPU")
    p.add_argument("--commercial-only", action="store_true", help="skip models not cleared for commercial use")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print licence class and sizes, fetch nothing")
    g.add_argument(
        "--check",
        action="store_true",
        help="fetch nothing; exit 1 if a file cannot be resolved without network (prints where from)",
    )
    p.set_defaults(run=_download)
    p = sub.add_parser("bench", help="time one model on synthetic input (an image; a pair for stereo)")
    p.add_argument("model", help="model id (see `fastforks list`)")
    p.add_argument(
        "--mode",
        choices=("auto", "portable", "kernels"),
        help="time only this backend (default: portable and kernels side by side, if both can run)",
    )
    p.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), help="compute dtype (default: the validated one)"
    )
    p.add_argument(
        "--device", help="e.g. cpu, cuda, cuda:1, mps (default: cuda if available, else mps if validated, else cpu)"
    )
    p.add_argument(
        "--size",
        type=_size,
        default=(375, 1242),
        help="HxW (default 375x1242, the KITTI image size of the speed tables; use a size that fits the model's task)",
    )
    p.add_argument("--views", type=_positive, default=8, help="views per scene for multi-view models (default 8)")
    p.add_argument(
        "--runs", type=_positive, default=10, help="timed runs after warm-up; the median is reported (default 10)"
    )
    p.set_defaults(run=_bench)
    p = sub.add_parser("export", help="portable model to ONNX or a TensorRT engine (with a parity check)")
    p.add_argument("model", help="model id (see `fastforks list`)")
    p.add_argument(
        "--format",
        choices=("onnx", "trt"),
        required=True,
        help="onnx file or TensorRT engine (needs the tensorrt extra)",
    )
    p.add_argument(
        "--size",
        type=_size,
        help=f"HxW, e.g. {EXAMPLE_SIZE}; fixed export size (needed for stereo, depth, segmentation, detection)",
    )
    p.add_argument(
        "--size-range",
        type=_bounds,
        metavar="MINxMIN:MAXxMAX",
        help="dynamic height / width within these bounds (opt = --size); default: the fixed --size",
    )
    p.add_argument("--fp16", action="store_true", help="the validated reduced precision (default float32)")
    p.add_argument("--output", "-o", help="file to write (default MODEL_HxW_DTYPE.onnx|engine)")
    p.set_defaults(run=_export)
    sub.add_parser(
        "doctor", help="optional libraries (versions, what is missing), runnable models, then `env`"
    ).set_defaults(run=_doctor)
    sub.add_parser("env", help="versions, GPU and kernel binaries (for bug reports)").set_defaults(run=_env)
    _add_stereo(sub)
    _add_sfm(sub)
    return _run(ap.parse_args(argv))


def _run(args: argparse.Namespace) -> int:
    try:
        status = args.run(args) or 0
        sys.stdout.flush()  # a closed pipe (`fastforks list | head`) surfaces here, not as exit 120 at shutdown
    except FastforksError as e:
        print(f"fastforks: error: {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # Python's "Note on SIGPIPE": no second error from the flush at interpreter exit
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
