"""Core A/B of the hand-written MLX networks against the exported-graph translation they replaced.

    python tools/mlx_core_ab.py list [--only REF ...] [--family FAMILY ...]        # the hand-ported refs (no GPU)
    python tools/mlx_core_ab.py run [--only REF ...] [--family FAMILY ...] [--dtypes float16 float32]
                                    [--modes portable] [--out FILE] [--cache-mb 2048] [--force] [--no-prime]
    python tools/mlx_core_ab.py table [--out FILE]                                 # the markdown table of the rows

The bar: a hand-written MLX network is at least as fast (warm) and as lean (peak) as the exported-graph translation
(`shared/mlx_graph`) of its PyTorch network, on the network alone. The speed survey's rows include loading, the
conversion cache and the pre- and post-processing, so a slower core can hide behind a faster load; this tool measures
the network call itself. A ref is hand-ported when its family's model module has `GRAPH_RUNNERS` (the runner its MLX
port had before the hand port) or it is a libnet classifier written out in `libnet.mlx_cnn` / `mlx_vit_cls`.
`FASTFORKS_MLX_GRAPH=1` (`fastforks._core.mlx.graph_switch`) builds that graph runner instead of the hand one.

Per (ref, mode, dtype) each arm (hand, graph) runs in a fresh process, both under the same MLX buffer cache limit
(`--cache-mb`, `FASTFORKS_MLX_CACHE_MB`), on the survey's input of the ref (`tools/speed_survey.py`: the floor's
scene where it exists locally, else the synthetic bench input; the same arrays in both arms). The runner's own
pre- and post-processing run as usual; every outermost call of an MLX network (an `MlxModule` / `GraphNet` of the
graph port, a class of a family's `mlx*` module of the hand port; `*_post` modules excluded) is timed with its inputs
evaluated before and its outputs evaluated after (synchronised), and its peak memory is `mx.get_peak_memory()` after
an `mx.reset_peak_memory()`. Recorded per arm:

- `net_first_s`, `net_ms` (median of MPS_RUNS after MPS_WARMUP warm-up calls, the survey's protocol), `net_peak_mb`
  (the largest network peak over the timed calls), `net_calls` (network calls per model call), `net_inputs` (the
  shapes the network got on the first call);
- `resident_mb`: MLX active memory after the warm calls with the outputs dropped (the weights and what the network
  keeps), less the active memory before `load()`;
- end to end: `load_s`, `first_s`, `ms` (the same protocol, without the network timers), `peak_footprint_mb`.

Before each measured process a throwaway one loads the arm and calls it once (`prime_load_s`, `prime_first_s`), so
both arms are measured with a warm MLX conversion cache (`_core/mlx_cache.py`) and warm OS file cache: a cold
conversion does not skew either arm (`--no-prime` skips it).

The two arms' outputs are compared per result key (`max_abs`, `max_rel` = max |hand - graph| / max |graph|): a sanity
check, not a golden. A ref is flagged where the hand network is more than 2 % slower warm (`net_ms`) or heavier
(`net_peak_mb`) than the graph one. Rows go to `--out` (JSONL, a row replaces the earlier one of the same ref / mode /
dtype; finished rows are skipped unless `--force`); `table` prints them (docs/plans/mlx-core-ab.md). Run it through
the GPU lock on a shared Mac, one family per call.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "plans" / "mlx-core-ab.jsonl"
ARMS = ("hand", "graph")
SLOWER = 1.02  # flag: hand / graph warm time above this
HEAVIER = 1.02  # flag: hand / graph network peak above this
TIMEOUT_S = 1800
CACHE_MB = 2048  # both arms: the largest request a port gets on a 16 GB Mac (1/8 of its memory)
# modules named mlx* that are the MLX runtime or the graph translation, not a family's hand-written network
_INFRA = frozenset(
    {
        "fastforks._core.mlx",
        "fastforks._core.mlx_cast",
        "fastforks._core.mlx_cache",
        "fastforks.shared.mlx_port",
        "fastforks.shared.mlx_graph",
        "fastforks.shared.mlx_wrap",
        "fastforks.shared.mlx_weights",
        "fastforks.shared.mlx_plan_cache",
        "fastforks.shared.mlx_dtype_report",
        "fastforks.shared.mlx_seg",
        "fastforks.shared.mlx_band",
    }
)


# Jobs (no MLX in this process: every measurement is a worker).
def hand_ported(card: Any) -> bool:  # noqa: ANN401 - a ModelCard
    """Whether the card's MLX runner is a hand-written network with a graph runner behind `FASTFORKS_MLX_GRAPH=1`."""
    from fastforks._core import registry  # noqa: PLC0415

    models = registry.import_model(card)
    if getattr(models, "GRAPH_RUNNERS", None):
        return True
    from fastforks.shared import libnet  # noqa: PLC0415

    runner = models.RUNNERS.get("portable_mlx")
    if not (isinstance(runner, type) and issubclass(runner, libnet.Mlx)):
        return False
    from fastforks.shared.libnet import mlx_cnn  # noqa: PLC0415 - imports mlx (no GPU work)

    return card.family in mlx_cnn.FAMILIES or card.family in libnet.VIT_FAMILIES


def jobs(
    only: Iterable[str] = (),
    families: Iterable[str] = (),
    modes: Iterable[str] = ("portable",),
    dtypes: Iterable[str] = ("float16", "float32"),
) -> list[dict[str, Any]]:
    """One job per hand-ported (ref, mode, dtype) the card validates on MLX, at the survey's input of the ref."""
    sys.path.insert(0, str(TOOLS))
    from speed_survey import _cards, _std_inputs  # noqa: PLC0415 - the survey's refs and inputs

    only, families, dtypes = set(only), set(families), list(dtypes)
    cards, inputs = _cards(), _std_inputs()
    seen: dict[str, bool] = {}
    out = []
    for ref in sorted(inputs):
        card = cards[ref.split("@")[0]][0]
        if (only and ref not in only) or (families and card.family not in families) or not card.portable_mlx_dtypes:
            continue
        if card.family not in seen:
            seen[card.family] = hand_ported(card)
        if not seen[card.family]:
            continue
        inp = inputs[ref]
        for mode in modes:
            have = card.mlx_dtypes(mode) if mode in card.modes else ()
            out += [
                {"ref": ref, "family": card.family, "task": inp["task"], "input": inp["label"], "scene": inp["scene"]}
                | {"mode": mode, "dtype": d}
                for d in dtypes
                if d in have
            ]
    return out


# Worker: one arm of one job in a fresh process.
class _Timers:
    """Times the outermost MLX network calls while `on` (see the module docstring)."""

    def __init__(self, mx: Any) -> None:  # noqa: ANN401 - mlx.core
        self.mx = mx
        self.on = False
        self.depth = 0
        self.reset()

    def reset(self) -> None:
        self.seconds = 0.0
        self.peak = 0
        self.calls = 0
        self.inputs: list[str] = []
        self.outputs: list[Any] = []
        self.keep = False  # keep the first network call's input shapes and outputs

    def arrays(self, tree: Any, *, torch_too: bool = False) -> list[Any]:  # noqa: ANN401
        """The `mx.array` leaves of a nested call argument / result (and the torch tensors with `torch_too`)."""
        import torch  # noqa: PLC0415

        if isinstance(tree, self.mx.array) or (torch_too and isinstance(tree, torch.Tensor)):
            return [tree]
        if isinstance(tree, dict):
            tree = list(tree.values())
        if isinstance(tree, (list, tuple)):
            return [a for t in tree for a in self.arrays(t, torch_too=torch_too)]
        return []

    def numpy(self, a: Any) -> Any:  # noqa: ANN401
        """An `mx.array` / torch tensor as a float32 numpy array."""
        import numpy as np  # noqa: PLC0415

        if isinstance(a, self.mx.array):
            return np.array(a.astype(self.mx.float32))
        return a.detach().cpu().float().numpy()

    def shapes(self, tree: Any) -> list[str]:  # noqa: ANN401
        """Shape and dtype of the array / tensor leaves of the network's arguments."""
        import torch  # noqa: PLC0415

        if isinstance(tree, (self.mx.array, torch.Tensor)):
            return [f"{'x'.join(map(str, tree.shape))}:{str(tree.dtype).rsplit('.', 1)[-1]}"]
        if isinstance(tree, dict):
            tree = list(tree.values())
        if isinstance(tree, (list, tuple)):
            return [s for t in tree for s in self.shapes(t)]
        return []

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """`fn` (an unbound `__call__`) timed when it is an outermost network call."""
        timers = self

        def call(obj: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if not timers.on or timers.depth:
                return fn(obj, *args, **kwargs)
            mx = timers.mx
            timers.depth += 1
            try:
                mx.eval(*timers.arrays((args, kwargs)))
                mx.synchronize()
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                out = fn(obj, *args, **kwargs)
                mx.eval(*timers.arrays(out))
                mx.synchronize()
                timers.seconds += time.perf_counter() - t0
                timers.peak = max(timers.peak, mx.get_peak_memory())
                timers.calls += 1
                if timers.calls == 1 and timers.keep:
                    timers.inputs = timers.shapes((args, kwargs))
                    timers.outputs = [timers.numpy(a) for a in timers.arrays(out, torch_too=True)]
                return out
            finally:
                timers.depth -= 1

        call.__wrapped__ = fn  # ty: ignore[unresolved-attribute]
        return call

    def patch(self, cls: type) -> None:
        if "__call__" in cls.__dict__ and not hasattr(cls.__dict__["__call__"], "__wrapped__"):
            cls.__call__ = self.wrap(cls.__dict__["__call__"])  # ty: ignore[invalid-assignment]

    def patch_hand(self) -> int:
        """Time every class with a `__call__` of the loaded hand-written MLX modules; the number patched."""
        n = 0
        for name, mod in list(sys.modules.items()):
            leaf = name.rpartition(".")[2]
            if not name.startswith("fastforks.") or not leaf.startswith("mlx") or name in _INFRA:
                continue
            if leaf.endswith("_post"):
                continue
            for obj in list(vars(mod).values()):
                if isinstance(obj, type) and obj.__module__ == name and "__call__" in obj.__dict__:
                    self.patch(obj)
                    n += 1
        return n


def _flat(tree: Any, prefix: str = "") -> dict[str, Any]:  # noqa: ANN401
    """The tensor leaves of a result as float32 numpy arrays, by dotted key."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if isinstance(tree, torch.Tensor):
        return {prefix or "out": tree.detach().cpu().float().numpy().astype(np.float32)}
    if isinstance(tree, np.ndarray) and tree.dtype != object:
        return {prefix or "out": tree.astype(np.float32)}
    if hasattr(tree, "__dataclass_fields__"):
        tree = {k: getattr(tree, k) for k in tree.__dataclass_fields__}
    if isinstance(tree, (list, tuple)):
        tree = dict(enumerate(tree))
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            out |= _flat(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    return {}


def worker(spec: dict[str, Any]) -> int:  # noqa: PLR0915 - one linear measurement protocol
    """One arm of one job: the row fields of `spec["arm"]` to `spec["row"]`, the outputs to `spec["npz"]`."""
    os.environ["FASTFORKS_MLX_GRAPH"] = "1" if spec["arm"] == "graph" else "0"
    os.environ["FASTFORKS_MLX_CACHE_MB"] = str(spec["cache_mb"])
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(TOOLS))
    from fastforks._core.mlx import mx  # noqa: PLC0415
    from fastforks.shared import mlx_graph  # noqa: PLC0415
    from speed_survey import MPS_RUNS, MPS_WARMUP, _footprint_peak, _inputs  # noqa: PLC0415

    torch.manual_seed(0)
    timers = _Timers(mx)
    for cls in (mlx_graph.MlxModule, mlx_graph.GraphNet):
        timers.patch(cls)
    row: dict[str, Any] = {}
    try:
        task = __import__(f"fastforks.{spec['task']}", fromlist=["load"])
        base = mx.get_active_memory()
        t0 = time.perf_counter()
        model = task.load(spec["ref"], mode=spec["mode"], device="mlx", dtype=spec["dtype"])
        row["load_s"] = round(time.perf_counter() - t0, 3)
        row["runner"] = type(model._runner).__name__  # noqa: SLF001 - which runner the switch built
        row["hand_classes"] = timers.patch_hand()
        args, kw, scene = _inputs(model, {**spec, "device": "mlx"})
        row["scene"] = scene

        def call() -> Any:  # noqa: ANN401
            with torch.inference_mode():
                return model(*args, **kw)

        timers.on = timers.keep = True
        t0 = time.perf_counter()
        out = call()
        timers.keep = False
        row["first_s"] = round(time.perf_counter() - t0, 3)
        row["net_first_s"] = round(timers.seconds, 3)
        if spec.get("prime"):  # the priming process: the conversion cache (and its plans) are now on disk
            Path(spec["row"]).write_text(json.dumps(row))
            return 0
        row["net_calls"] = timers.calls
        row["net_inputs"] = timers.inputs
        row["net_outputs"] = [f"{'x'.join(map(str, o.shape))}" for o in timers.outputs]
        if not timers.calls:
            msg = "no MLX network call seen (no MlxModule / GraphNet / hand-written class was called)"
            raise RuntimeError(msg)  # noqa: TRY301 - recorded as the arm's error
        net_out = {f"net.{i}": o for i, o in enumerate(timers.outputs)}
        np.savez(spec["npz"], **_flat(out), **net_out)
        del out, net_out
        timers.outputs = []
        for _ in range(MPS_WARMUP):
            call()
        ts, peaks = [], []
        for _ in range(MPS_RUNS):
            timers.reset()
            call()
            ts.append(1e3 * timers.seconds)
            peaks.append(timers.peak)
        row["net_ms"] = round(statistics.median(ts), 3)
        row["net_ms_min"] = round(min(ts), 3)
        row["net_peak_mb"] = round(max(peaks) / 2**20, 1)
        timers.on = False
        e2e = []
        for _ in range(MPS_RUNS):
            t0 = time.perf_counter()
            call()
            e2e.append(1e3 * (time.perf_counter() - t0))
        row["ms"] = round(statistics.median(e2e), 3)
        gc.collect()
        row["resident_mb"] = round((mx.get_active_memory() - base) / 2**20, 1)
        row["peak_footprint_mb"] = round(_footprint_peak() / 2**20)
    except Exception as e:  # noqa: BLE001 - recorded as the arm's error
        row["error"] = " ".join(f"{type(e).__name__}: {e}".split())[:600]
    Path(spec["row"]).write_text(json.dumps(row))
    return 0


# Orchestration.
def _read(out: Path) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def _key(r: dict[str, Any]) -> tuple[str, str, str]:
    return r["ref"], r["mode"], r["dtype"]


def _merge(out: Path, row: dict[str, Any]) -> None:
    rows = [r for r in _read(out) if _key(r) != _key(row)] + [row]
    rows.sort(key=_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _compare(hand: Path, graph: Path) -> dict[str, Any]:
    """Per result key max |hand - graph| and that over max |graph|, the keys whose shapes differ.

    Also the elements compared (`n`); a differing shape is, say, a different number of detections. The network outputs
    (`net.*`, from the first network call) are only compared when both arms give the same number of arrays; a
    channels-last hand array is moved to the graph's channels-first layout where that makes the shapes match. The
    result keys go to `max_*`, the network outputs to `net_max_*`.
    """
    import numpy as np  # noqa: PLC0415

    if not (hand.exists() and graph.exists()):
        return {}
    a, b = np.load(hand), np.load(graph)
    agree: dict[str, Any] = {"max_abs": 0.0, "max_rel": 0.0, "n": 0, "differ": []}
    agree |= {"net_max_abs": 0.0, "net_max_rel": 0.0, "net_n": 0}
    net_a, net_b = [k for k in a.files if k.startswith("net.")], [k for k in b.files if k.startswith("net.")]
    keys = [(k, k) for k in sorted((set(a.files) | set(b.files)) - {*net_a, *net_b})]
    if len(net_a) == len(net_b):
        keys += list(zip(sorted(net_a, key=_natural), sorted(net_b, key=_natural), strict=True))
    for ka, kb in keys:
        pre = "net_" if ka.startswith("net.") else ""
        if ka not in a.files or kb not in b.files:
            agree["differ"].append(ka)
            continue
        x, y = a[ka].astype(np.float64), b[kb].astype(np.float64)
        if x.shape != y.shape and x.ndim > 2 and np.moveaxis(x, -1, 1).shape == y.shape:  # noqa: PLR2004
            x = np.moveaxis(x, -1, 1)
        if x.shape != y.shape:
            agree["differ"].append(ka)
            continue
        if not x.size:
            continue
        diff = np.abs(x - y)
        d = float(np.nanmax(diff))
        scale = max(float(np.nanmax(np.abs(y))), 1e-12)
        agree[f"{pre}off"] = agree.get(f"{pre}off", 0) + int((diff > 1e-2 * scale).sum())
        agree[f"{pre}max_abs"] = max(agree[f"{pre}max_abs"], d)
        agree[f"{pre}max_rel"] = max(agree[f"{pre}max_rel"], d / scale)
        agree[f"{pre}n"] += x.size
    for k in ("max_abs", "max_rel", "net_max_abs", "net_max_rel"):
        agree[k] = float(f"{agree[k]:.3g}")
    return agree


def _natural(k: str) -> tuple[int, str]:
    """`net.10` after `net.9`."""
    tail = k.rpartition(".")[2]
    return (int(tail), k) if tail.isdigit() else (1 << 30, k)


def _flags(row: dict[str, Any]) -> list[str]:
    h, g = row.get("hand", {}), row.get("graph", {})
    if "net_ms" not in h or "net_ms" not in g:
        return []
    out = []
    if h["net_ms"] > SLOWER * g["net_ms"]:
        out.append("slower")
    if h["net_peak_mb"] > HEAVIER * g["net_peak_mb"]:
        out.append("heavier")
    return out


def _worker_cmd(spec: dict[str, Any]) -> list[str]:
    return [sys.executable, __file__, "worker", json.dumps(spec)]


def _prime(spec: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """Load the arm and call it once in a throwaway process: the measured one then finds a warm conversion cache.

    Returns that process's `load_s` / `first_s` as `prime_load_s` / `prime_first_s` (`_core/mlx_cache.py`).
    """
    subprocess.run(_worker_cmd(spec | {"prime": True}), env=env, timeout=TIMEOUT_S, check=False)
    with contextlib.suppress(OSError, json.JSONDecodeError):
        r = json.loads(Path(spec["row"]).read_text())
        Path(spec["row"]).unlink()
        return {f"prime_{k}": r[k] for k in ("load_s", "first_s") if k in r}
    return {}


def run(a: argparse.Namespace) -> int:
    """Measure both arms of every job not yet in `--out` (all with `--force`)."""
    out = Path(a.out)
    done = {_key(r) for r in _read(out) if "net_ms" in r.get("hand", {}) and "net_ms" in r.get("graph", {})}
    todo = jobs(a.only, a.family, a.modes, a.dtypes)
    if not todo:
        print("no hand-ported job matches", file=sys.stderr)
        return 1
    env = {**os.environ, "FASTFORKS_OFFLINE": "1"}
    with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
        for job in todo:
            if _key(job) in done and not a.force:
                continue
            row: dict[str, Any] = {k: job[k] for k in ("ref", "family", "task", "input", "mode", "dtype")}
            row["cache_mb"] = a.cache_mb
            npz = {}
            for arm in ARMS:
                spec = {**job, "arm": arm, "cache_mb": a.cache_mb}
                spec |= {"row": f"{tmp}/{arm}.json", "npz": f"{tmp}/{arm}.npz"}
                npz[arm] = Path(spec["npz"])
                for p in (Path(spec["row"]), npz[arm]):
                    p.unlink(missing_ok=True)
                try:
                    primed = _prime(spec, env) if a.prime else {}
                    subprocess.run(_worker_cmd(spec), env=env, timeout=TIMEOUT_S, check=False)
                    arm_row = json.loads(Path(spec["row"]).read_text()) | primed
                except subprocess.TimeoutExpired:
                    arm_row = {"error": f"timeout after {TIMEOUT_S} s"}
                except (OSError, json.JSONDecodeError):
                    arm_row = {"error": "the worker crashed (no row; out of memory?)"}
                row[arm] = arm_row
            row["agree"] = _compare(npz["hand"], npz["graph"])
            row["flags"] = _flags(row)
            row["mlx"] = _mlx_version()
            _merge(out, row)
            print(_line(row), flush=True)
            for p in npz.values():
                p.unlink(missing_ok=True)
    return 0


def _mlx_version() -> str:
    import importlib.metadata  # noqa: PLC0415

    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version("mlx")
    return ""


def _ratio(h: float | None, g: float | None) -> str:
    return f"{h / g:.2f}" if h is not None and g else "-"


def _num(v: Any, fmt: str) -> str:  # noqa: ANN401
    return format(v, fmt) if isinstance(v, (int, float)) else "-"


def _agree(agree: dict[str, Any], pre: str) -> str:
    """Max relative difference, and the share of elements off by more than 1 % of the graph's largest magnitude."""
    off = agree.get(f"{pre}off", 0) / agree[f"{pre}n"]
    return _num(agree.get(f"{pre}max_rel"), ".2g") + (f" ({100 * off:.2g} % off)" if off else "")


def _line(r: dict[str, Any]) -> str:
    h, g = r.get("hand", {}), r.get("graph", {})
    if "net_ms" not in h or "net_ms" not in g:
        err = h.get("error") or g.get("error") or "missing"
        side = "hand" if "error" in h else "graph"
        return f"{r['ref']:40s} {r['dtype']:8s} {side} failed: {err}"
    return (
        f"{r['ref']:40s} {r['dtype']:8s} net {h['net_ms']:9.2f} / {g['net_ms']:9.2f} ms"
        f" ({_ratio(h['net_ms'], g['net_ms'])}), peak {h['net_peak_mb']:7.0f} / {g['net_peak_mb']:7.0f} MB"
        f" ({_ratio(h['net_peak_mb'], g['net_peak_mb'])}), e2e {h['ms']:9.2f} / {g['ms']:9.2f} ms"
        f" {' '.join(r['flags'])}"
    )


def table(rows: list[dict[str, Any]]) -> str:
    """The markdown table of the rows (hand / graph per cell), then the flagged and the failed rows."""
    head = (
        "| ref | dtype | net warm ms | x | net first s | net peak MB | x | resident MB | e2e warm ms | load s"
        " | e2e first s | net rel | result rel | flag |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines, failed = [], []
    for r in sorted(rows, key=_key):
        h, g = r.get("hand", {}), r.get("graph", {})
        if "net_ms" not in h or "net_ms" not in g:
            failed.append(r)
            continue

        def pair(k: str, fmt: str, h: dict[str, Any] = h, g: dict[str, Any] = g) -> str:
            return f"{_num(h.get(k), fmt)} / {_num(g.get(k), fmt)}"

        agree = r.get("agree", {})
        net_rel = _agree(agree, "net_") if agree.get("net_n") else "-"
        rel = _agree(agree, "") if agree.get("n") else "empty"
        rel += f" ({len(agree['differ'])} keys differ)" if agree.get("differ") else ""
        lines.append(
            f"| {r['ref']} | {r['dtype'].replace('float', 'fp')} | {pair('net_ms', '.1f')}"
            f" | {_ratio(h['net_ms'], g['net_ms'])} | {pair('net_first_s', '.2f')} | {pair('net_peak_mb', '.0f')}"
            f" | {_ratio(h['net_peak_mb'], g['net_peak_mb'])} | {pair('resident_mb', '.0f')} | {pair('ms', '.1f')}"
            f" | {pair('load_s', '.2f')} | {pair('first_s', '.2f')} | {net_rel} | {rel}"
            f" | {' '.join(r.get('flags', []))} |"
        )
    text = head + "\n".join(lines) + "\n"
    if failed:
        text += "\nNot measured (one arm failed):\n\n"
        for r in failed:
            h, g = r.get("hand", {}), r.get("graph", {})
            side, err = ("hand", h.get("error")) if "error" in h else ("graph", g.get("error"))
            text += f"- {r['ref']} {r['dtype']}: {side}: {err}\n"
    return text


def main() -> int:
    """Command line (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "run"):
        s = sub.add_parser(name)
        s.add_argument("--only", nargs="*", default=[])
        s.add_argument("--family", nargs="*", default=[])
        s.add_argument("--modes", nargs="*", default=["portable"])
        s.add_argument("--dtypes", nargs="*", default=["float16", "float32"])
        if name == "run":
            s.add_argument("--out", default=str(OUT))
            s.add_argument("--cache-mb", type=int, default=CACHE_MB)
            s.add_argument("--tmp", default=None, help="directory for the arms' outputs (default: the system temp)")
            s.add_argument("--force", action="store_true")
            s.add_argument(
                "--no-prime", dest="prime", action="store_false", help="skip the warm-up process before each arm"
            )
    t = sub.add_parser("table")
    t.add_argument("--out", default=str(OUT))
    w = sub.add_parser("worker")
    w.add_argument("spec")
    a = p.parse_args()
    if a.cmd == "worker":
        return worker(json.loads(a.spec))
    if a.cmd == "list":
        for j in jobs(a.only, a.family, a.modes, a.dtypes):
            print(f"{j['ref']:40s} {j['family']:20s} {j['mode']:9s} {j['dtype']:8s} {j['input']}")
        return 0
    if a.cmd == "table":
        print(table(_read(Path(a.out))), end="")
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
