"""Core A/B of the hand-written MLX networks against the exported-graph translation they replaced.

    python tools/mlx_core_ab.py list [--only REF ...] [--family FAMILY ...]        # the hand-ported refs (no GPU)
    python tools/mlx_core_ab.py run [--only REF ...] [--family FAMILY ...] [--dtypes float16 float32]
                                    [--modes portable] [--out FILE] [--cache-mb 2048] [--force] [--no-prime]
                                    [--flagged] [--repeats 2] [--no-shipped] [--keep-quality]
    python tools/mlx_core_ab.py quick [REF ...] [--family F ...] [--dtypes float16] [--scale 0.5] [--no-golden]
                                      [--rebaseline] [--session TAG] [--lock gpu2.sh [--chunk N]]  # iterating
    python tools/mlx_core_ab.py table [--out FILE]                                 # the markdown table of the rows
    python tools/mlx_core_ab.py md [--out FILE] [--md FILE]      # rewrite the generated part of mlx-core-ab.md

The bar: a hand-written MLX network is at least as fast (warm) and as lean (peak) as the exported-graph translation
(`shared/mlx_graph`) of its PyTorch network, on the network alone. The speed survey's rows include loading, the
conversion cache and the pre- and post-processing, so a slower core can hide behind a faster load; this tool measures
the network call itself. A ref is hand-ported when its family's model module has `GRAPH_RUNNERS` (the runner its MLX
port had before the hand port) or it is a libnet classifier written out in `libnet.mlx_cnn` / `mlx_vit_cls`.
`FASTFORKS_MLX_GRAPH=1` (`fastforks._core.mlx.graph_switch`) builds that graph runner instead of the hand one.

Per (ref, mode, dtype) each arm (hand, graph) runs in a fresh process, both under the same MLX buffer cache limit
(`--cache-mb`, `FASTFORKS_MLX_CACHE_MB`), on the survey's input of the ref (`tools/speed_survey.py`: the floor's
scene where it exists locally, else the synthetic bench input; the same arrays in both arms). The runner's own pre-
and post-processing run as usual; every outermost call of an MLX network (an `MlxModule` / `GraphNet` of the graph
port; `__call__` or a public method of a class of a family's `mlx*` / `*_mlx` module of the hand port, `*_post`
modules excluded; `SCOPE`'s methods only where the hand port runs more in MLX than the graph port) is timed with its
inputs evaluated before and its outputs evaluated after (synchronised), and its peak memory is `mx.get_peak_memory()`
after an `mx.reset_peak_memory()`. Recorded per arm:

- `net_first_s`, `net_ms` (median of MPS_RUNS after MPS_WARMUP warm-up calls, the survey's protocol), `net_peak_mb`
  (the largest network peak over the timed calls), `net_calls` (network calls per model call), `net_inputs` (the
  shapes the network got on the first call);
- `net_call_ms`, `net_call_peak_mb`: each network call's share, where a model call makes several (an arm that
  runs more of the model in MLX makes more of them: compare the calls both arms make);
- `resident_mb`: MLX active memory after the warm calls with the outputs dropped (the weights and what the network
  keeps), less the active memory before `load()`;
- end to end: `load_s`, `first_s`, `ms` (the same protocol, without the network timers), `peak_footprint_mb`.

The method (docs/plans/mlx-core-ab.md, "Method"):

- arms: `hand` and `graph` at `--cache-mb` (the same, large buffer cache in both: a diagnostic of the networks), and
  `shipped`, the hand network at its own `request_cache` (as users run it; `--no-shipped` skips it);
- `--repeats` processes per arm, interleaved (hand, graph, shipped, then shipped, graph, hand, ...); each arm's samples
  are pooled (`net_ms` the median of all, `net_ms_reps` per process) and `ci` holds the bootstrap 95 % interval of
  hand / graph (and shipped / graph) warm time, resampling processes, then samples;
- a network under `SMALL_MS` is timed over a loop of `net_loop` model calls per sample;
- the host copies `to_mx` / `to_torch` inside a network call are left out of its time in both arms (`net_copy_ms`);
- `net_transient_mb`: the network's peak over the active memory before the call (what the call allocates);
- flags: `slower` when the interval lies above `SLOWER` (rows without one: the point ratio), `heavier` on the transient
  peak (above `HEAVIER` x and `TRANSIENT_FLOOR_MB` apart), `worse` as before, `shipped-slower` when the shipped arm's
  interval lies above `SLOWER`; `md` also lists float16 hand arms keeping more than `ISLAND` of their float32 resident
  memory and counts the rows whose arms make a different number of network calls (`net_ms` sums them per model call);
- `--flagged` re-measures the rows flagged under these rules, `--keep-quality` carries their golden check over.

Two tiers. `run` (the full method below) is for a borderline quick result and the final audit. While porting, `quick`
gives an answer in seconds to a minute: one process for all its refs, each model loaded once; only the hand arm is
timed, against the graph arm's baseline of the same (ref, mode, dtype, input size, MLX version), measured once per
session (`--session`, `FASTFORKS_AB_SESSION`, default the date) and cached in `BASELINE_DIR` (`FASTFORKS_AB_BASELINE`);
float16 and the input at `QUICK_SCALE` of the survey's (synthetic) by default. Each network sample is divided by the
time of a fixed reference net (`_ref_net`) timed right before and after it, so a GPU that is busier than when the
baseline was taken (another job in the light lane) shifts both. The hand arm runs in rounds of `QUICK_ROUND` samples
and stops once the bootstrap interval of hand / graph lies wholly beyond 1 +- `QUICK_BAND` (`slower` / `faster`) or
wholly inside it (`same`), at most `--rounds`; `open` is the borderline case the full `run` decides. Prints warm ms,
the interval and verdict, the transient peaks, the golden errors of both arms (`--no-golden` skips them) and the wall
time. Run it under `gpu2.sh --light` (or `--lock gpu2.sh`, which also splits it into `--chunk` refs per lock call):
its timings are indicative only and never go into the rows.

Under `gpu2.sh --timed` (tools/gpu_lock.py) a worker holds the exclusive GPU lock only around its timed calls.

Before each measured process a throwaway one loads the arm and calls it once (`prime_load_s`, `prime_first_s`), so
both arms are measured with a warm MLX conversion cache (`_core/mlx_cache.py`) and warm OS file cache: a cold
conversion does not skew either arm (`--no-prime` skips it).

The two arms' outputs are compared per result key (`max_abs`, `max_rel` = max |hand - graph| / max |graph|): a sanity
check, not a golden. After the measurements each arm runs the ref's golden check on the golden's own input and options
(`quality`: tests/test_golden.py `gpu_errors`, the measures `test_mlx` bounds: mean / p99 |diff| of the primary
output against the stored golden, the other bounded outputs, the sparse matchers' recall; `{"none": ...}` without a
golden). A ref is flagged where the hand network is more than 2 % slower warm (`net_ms`) or heavier (`net_peak_mb`)
than the graph one, or `worse`: clearly further from the golden than the graph arm (`_worse`). Rows go to `--out`
(JSONL, a row replaces the earlier one of the same ref / mode / dtype; finished rows are skipped unless `--force`);
`table` prints them, `md` rewrites the generated part of docs/plans/mlx-core-ab.md. Run it through the GPU lock on a
shared Mac, one family per call.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import merge_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "plans" / "mlx-core-ab.jsonl"
ARMS = ("hand", "graph")  # at `--cache-mb` (a diagnostic: the same, large buffer cache in both)
SHIPPED = "shipped"  # the hand arm at its own `request_cache`, as it ships
SLOWER = 1.02  # flag: hand / graph warm time above this
HEAVIER = 1.02  # flag: hand / graph network peak above this
WORSE_X = 1.25  # quality flag `worse`: the hand arm's golden error above this times the graph arm's ...
WORSE_BUDGET = 0.05  # ... and above it by more than this share of the golden's bound (both, for the mean or the p99)
WORSE_RECALL = 0.02  # sparse matchers: recall or match agreement lower by more than this
ACCEPT_X = 0.90  # an accepted peak trade: hand / graph warm at most this (and the e2e peak conditions, `accepted`)
ACCEPT_HEAD = (
    "| ref | dtype | net warm ms hand / graph | x | transient MB hand / graph | x | e2e peak MB | best MPS MB"
    " | MLX fp32 MB |"
)
SURVEY = ROOT / "docs" / "data" / "speed" / "apple-m1-pro-mlx-16gb.jsonl"  # end-to-end MLX rows (accepted trades)
SURVEY_MPS = ROOT / "docs" / "data" / "speed" / "apple-m1-pro-16gb.jsonl"  # the MPS rows they are held against
MD = ROOT / "docs" / "plans" / "mlx-core-ab.md"
GENERATED = "<!-- generated by `tools/mlx_core_ab.py md`: everything below this line -->"
TIMEOUT_S = 1800
SMALL_MS = 10.0  # a network under this is timed over a loop of model calls per sample ...
LOOP_MS = 20.0  # ... enough for this much network time ...
LOOP_MAX = 64  # ... at most this many
REPEATS = 2  # processes per arm, interleaved (`run`)
CI_LEVEL = 0.95  # the bootstrap interval a flag must clear
BOOT = 2000  # bootstrap resamples
TRANSIENT_FLOOR_MB = 4.0  # `heavier` also needs the transient peaks this far apart
ISLAND = 0.6  # resident float16 / float32 of the hand arm above this: float32 islands in a float16 load ...
ISLAND_MIN_MB = 32.0  # ... where the float32 arm keeps at least this much (below, the buffers that do not scale rule)
CACHE_MB = 2048  # both arms: the largest request a port gets on a 16 GB Mac (1/8 of its memory)
# modules named mlx* / *_mlx that are the MLX runtime or the graph translation, not a family's hand-written network
_INFRA = frozenset(
    {
        "fastforks._core.mlx",
        "fastforks._core.mlx_cast",
        "fastforks._core.mlx_cache",
        "fastforks.tracking._mlx",
        "fastforks.volume._mlx",
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
# families whose hand port runs more of the model in MLX than the graph port translated: only these hand methods are
# timed, the part the graph port covers (R-CNN: backbone, FPN and RPN head, not the box and mask heads; XFeat: the
# extractor, not LightGlue; LoFTR: backbone, coarse transformer and its matching, not the fine refinement)
SCOPE = {
    "rcnn": (("Net", "dense"),),
    "xfeat": (("MlxDense", "__call__"),),
    "loftr": (("MlxLoFTR", "_dense"),),
}


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
        if ref.split("@")[0] not in cards:  # a floor row of a model this checkout has no card for
            continue
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
        self.transient = 0  # the largest peak over the active memory before the call (bytes)
        self.copy = 0.0  # host copies (`to_mx` / `to_torch`) inside timed calls, left out of `seconds`
        self.calls = 0
        self.each: list[tuple[float, int]] = []  # seconds and peak bytes of every timed call, in order
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
                try:
                    mx.eval(*timers.arrays((args, kwargs)))
                except ValueError:  # traced by `mx.compile` / `mx.vmap`: the compiled caller is the network call
                    return fn(obj, *args, **kwargs)
                mx.synchronize()
                base = mx.get_active_memory()
                mx.reset_peak_memory()
                copy0 = timers.copy
                t0 = time.perf_counter()
                out = fn(obj, *args, **kwargs)
                mx.eval(*timers.arrays(out))
                mx.synchronize()
                dt, peak = time.perf_counter() - t0 - (timers.copy - copy0), mx.get_peak_memory()
                timers.seconds += dt
                timers.peak = max(timers.peak, peak)
                timers.transient = max(timers.transient, peak - base)
                timers.each.append((dt, peak))
                timers.calls += 1
                if timers.calls == 1 and timers.keep:
                    timers.inputs = timers.shapes((args, kwargs))
                    timers.outputs = [timers.numpy(a) for a in timers.arrays(out, torch_too=True)]
                return out
            finally:
                timers.depth -= 1

        call.__wrapped__ = fn  # ty: ignore[unresolved-attribute]
        return call

    def patch_copies(self) -> int:
        """Leave the host copies `to_mx` / `to_torch` out of the timed calls, in both arms alike.

        The graph arm's `GraphNet.__call__` copies its torch inputs in and its outputs out (and casts them to float32)
        inside the timed call, where a hand network often takes and gives `mx.array`s; so every binding of the two
        functions in the loaded `fastforks` modules (and `_core.mlx`'s own, for imports inside functions) is wrapped:
        inside a timed call the array is evaluated first (that is the network's work) and the copy's time goes to
        `copy`, which `wrap` takes off the call's time. Returns the number of bindings wrapped.
        """
        from fastforks._core import mlx as core  # noqa: PLC0415

        timers, mx = self, self.mx
        orig_mx, orig_torch = core.to_mx, core.to_torch

        def to_mx(t: Any) -> Any:  # noqa: ANN401
            if not (timers.on and timers.depth):
                return orig_mx(t)
            t0 = time.perf_counter()
            a = orig_mx(t)
            mx.eval(a)
            timers.copy += time.perf_counter() - t0
            return a

        def to_torch(a: Any) -> Any:  # noqa: ANN401
            if not (timers.on and timers.depth):
                return orig_torch(a)
            mx.eval(a)
            mx.synchronize()
            t0 = time.perf_counter()
            t = orig_torch(a)
            timers.copy += time.perf_counter() - t0
            return t

        n = 0
        for name, mod in list(sys.modules.items()):
            if not name.startswith("fastforks."):
                continue
            for attr, orig, new in (("to_mx", orig_mx, to_mx), ("to_torch", orig_torch, to_torch)):
                if vars(mod).get(attr) is orig:
                    setattr(mod, attr, new)
                    n += 1
        return n

    def patch(self, cls: type, name: str = "__call__") -> bool:
        fn = cls.__dict__.get(name)
        if not callable(fn) or isinstance(fn, (staticmethod, classmethod, type)) or hasattr(fn, "__wrapped__"):
            return False
        setattr(cls, name, self.wrap(fn))
        return True

    def patch_hand(self, family: str) -> int:
        """Time `__call__` and the public methods of every class of the loaded hand-written MLX modules.

        The entry points differ per family (`__call__`, `logits`, `dense`, `maps`, `embed`, ...); the depth counter
        times only the outermost one. A family in `SCOPE` times only the methods listed there. Returns the number of
        methods patched.
        """
        n = 0
        if family in SCOPE:
            for name, mod in list(sys.modules.items()):
                leaf = name.rpartition(".")[2]
                if name.startswith("fastforks.") and name not in _INFRA and leaf.startswith("mlx"):
                    for cls, meth in SCOPE[family]:
                        obj = vars(mod).get(cls)
                        if isinstance(obj, type) and obj.__module__ == name:
                            n += self.patch(obj, meth)
            return n
        for name, mod in list(sys.modules.items()):
            leaf = name.rpartition(".")[2]
            hand = leaf.startswith("mlx") or leaf.endswith("_mlx")
            if not name.startswith("fastforks.") or not hand or name in _INFRA or leaf.endswith("_post"):
                continue
            for obj in list(vars(mod).values()):
                if isinstance(obj, type) and obj.__module__ == name:
                    names = ["__call__"] + [k for k in obj.__dict__ if not k.startswith("_")]
                    n += sum(self.patch(obj, k) for k in names)
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


def worker(spec: dict[str, Any]) -> int:
    """One arm of one job: the row fields of `spec["arm"]` to `spec["row"]`, the outputs to `spec["npz"]`.

    `spec["cache_mb"]` None (the `shipped` arm): the port's own buffer cache (`request_cache`), no forced limit.
    """
    os.environ["FASTFORKS_MLX_GRAPH"] = "1" if spec["arm"] == "graph" else "0"
    if spec.get("cache_mb") is None:
        os.environ.pop("FASTFORKS_MLX_CACHE_MB", None)
    else:
        os.environ["FASTFORKS_MLX_CACHE_MB"] = str(spec["cache_mb"])
    sys.path.insert(0, str(TOOLS))
    import gpu_lock  # noqa: PLC0415

    with gpu_lock.session(f"mlx_core_ab {spec['ref']} {spec['dtype']} {spec['arm']}"):
        return _worker(spec, gpu_lock)


def _loops(net_ms: float) -> int:
    """Model calls per timed sample: one, or enough for `LOOP_MS` of network time under `SMALL_MS`.

    The per-sample mean of a small network is then less noisy; the evaluation and synchronisation around each
    network call are the same in both arms.
    """
    if net_ms >= SMALL_MS or net_ms <= 0:
        return 1
    return max(1, min(LOOP_MAX, math.ceil(LOOP_MS / net_ms)))


def _worker(spec: dict[str, Any], gpu_lock: Any) -> int:  # noqa: ANN401, C901, PLR0915 - one linear protocol
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

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
        row["hand_classes"] = timers.patch_hand(spec["family"])
        row["copy_bindings"] = timers.patch_copies()
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
        with gpu_lock.timed():  # the exclusive GPU lock (under `gpu2.sh --timed`) for the timed calls only
            for _ in range(MPS_WARMUP):
                timers.reset()
                call()
            loops = _loops(1e3 * timers.seconds)
            ts, peaks, transients, copies, each = [], [], [], [], []
            for _ in range(MPS_RUNS):
                timers.reset()
                for _ in range(loops):
                    call()
                ts.append(1e3 * timers.seconds / loops)
                copies.append(1e3 * timers.copy / loops)
                peaks.append(timers.peak)
                transients.append(timers.transient)
                each.append(timers.each[: len(timers.each) // loops])
            timers.on = False
            e2e = []
            for _ in range(MPS_RUNS):
                t0 = time.perf_counter()
                for _ in range(loops):
                    call()
                e2e.append(1e3 * (time.perf_counter() - t0) / loops)
        row["net_ms"] = round(statistics.median(ts), 3)
        row["net_ms_min"] = round(min(ts), 3)
        row["net_ms_runs"] = [round(t, 4) for t in ts]
        row["net_loop"] = loops
        row["net_copy_ms"] = round(statistics.median(copies), 3)
        row["net_peak_mb"] = round(max(peaks) / 2**20, 1)
        row["net_transient_mb"] = round(max(transients) / 2**20, 1)
        if len(each[0]) > 1 and len({len(e) for e in each}) == 1:  # several network calls: each one's share
            row["net_call_ms"] = [round(1e3 * statistics.median(t for t, _ in c), 3) for c in zip(*each, strict=True)]
            row["net_call_peak_mb"] = [round(max(p for _, p in c) / 2**20, 1) for c in zip(*each, strict=True)]
        row["ms"] = round(statistics.median(e2e), 3)
        row["ms_runs"] = [round(t, 3) for t in e2e]
        gc.collect()
        row["resident_mb"] = round((mx.get_active_memory() - base) / 2**20, 1)
        row["peak_footprint_mb"] = round(_footprint_peak() / 2**20)
        if spec.get("quality", True):  # after every measurement: its own call (and load, with the golden's options)
            held = [model]
            del model  # only `_quality` holds it: a golden with options loads it again
            row["quality"] = _quality(held, spec)
    except Exception as e:  # noqa: BLE001 - recorded as the arm's error
        row["error"] = " ".join(f"{type(e).__name__}: {e}".split())[:600]
    Path(spec["row"]).write_text(json.dumps(row))
    return 0


def _quality(held: list[Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The arm (the model in `held`, which this takes) against the stored golden of its ref.

    In the golden test's own measures (`test_golden.gpu_errors`), on the golden's input and options (a model with
    options is loaded again with them), with the mode's GPU tolerances:

    `mean` / `p99` of the primary output's |diff| in the test's unit and its bounds (`bound_mean` / `bound_p99`), the
    other bounded outputs in `rows` ([mean, p99, bound mean, bound p99]), and the sparse matchers' `recall` / `match`.
    `{"none": why}` without a golden; `{"error": ...}` when the call or the comparison fails (e.g. another number of
    detections than the golden).
    """
    import pytest  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from merge_check import _test_golden  # noqa: PLC0415 - tools/ is on the path (the worker put it there)

    os.environ.setdefault(registry.NONCOMMERCIAL_ENV, "1")  # as the golden tests: goldens may need such weights
    tg = _test_golden()
    model = held.pop()
    ref = model.variant.ref
    if ref not in tg.GOLDEN:
        return {"none": "no golden"}
    try:
        want = tg.golden(ref)
    except (pytest.skip.Exception, pytest.fail.Exception) as e:  # the golden file is missing
        return {"none": " ".join(str(e).split())[:200]}
    try:
        opts = tg.options(ref, want)
        if opts:  # the golden's options: load again, without the timed copy
            model = None
            gc.collect()
            model = tg.API[tg.TASK[ref]].load(ref, mode=spec["mode"], device="mlx", dtype=spec["dtype"], **opts)
        with torch.inference_mode():
            got, _ = tg.forward(model, ref, want)
        want = tg.outputs(ref, want)
        if "pairs" not in want:
            differ = [k for k in want.keys() & got.keys() if tuple(got[k].shape) != want[k].shape]
            if differ:
                k = differ[0]
                return {"error": f"{k}: shape {tuple(got[k].shape)}, golden {want[k].shape}"}
        q = tg.gpu_errors(ref, spec["mode"], want, got)
    except Exception as e:  # noqa: BLE001 - recorded
        return {"error": " ".join(f"{type(e).__name__}: {e}".split())[:300]}
    mean, p99, bound_mean, bound_p99 = q["rows"][q["primary"]]
    out: dict[str, Any] = {"golden": ref, "primary": q["primary"], "mean": _sig(mean), "p99": _sig(p99)}
    out |= {"bound_mean": bound_mean, "bound_p99": bound_p99}
    out["rows"] = {k: [_sig(v) for v in r] for k, r in q["rows"].items() if k != q["primary"]}
    out |= {k: _sig(q[k]) for k in ("recall", "match", "sure") if k in q}
    return out


def _sig(v: float) -> float:
    return float(f"{v:.4g}")


# Quick tier: one process, the hand arm against a cached graph baseline (for iterating on a port; `run` is the gate).
QUICK_BAND = 0.10  # decided once the interval lies wholly outside 1 +- this (or wholly inside it)
QUICK_SCALE = 0.5  # the input side, as a share of the survey's (a smaller input: faster rounds)
QUICK_ROUND = 3  # timed samples per round
QUICK_MIN_ROUNDS, QUICK_MAX_ROUNDS = 2, 8
QUICK_BASE_ROUNDS = 3  # rounds of the graph baseline (measured once per session, ref, mode, dtype and size)
REF_TRIES = 3  # reference net calls per reference sample (the fastest counts)
QUICK_CHUNK = 0  # refs per process under `--lock` (`quick_chunks`; 0: all in one)
BASELINE_DIR = Path(os.environ.get("FASTFORKS_AB_BASELINE", "/Volumes/Daten/ff-mlx/ab-baseline"))


def _small(label: str, scale: float) -> str:
    """The input label `HxW` scaled by `scale`, each side a multiple of 32 and at least 64 (`1.0`: unchanged)."""
    if scale >= 1:
        return label
    return "x".join(str(max(64, round(int(v) * scale / 32) * 32)) for v in label.split("x"))


def _verdict(ci: list[float], band: float = QUICK_BAND) -> str:
    """`slower` / `faster` (the interval wholly beyond 1 +- `band`), `same` (wholly inside it), else `open`."""
    if not ci:
        return "open"
    lo, hi = ci
    if lo > 1 + band:
        return "slower"
    if hi < 1 - band:
        return "faster"
    if lo >= 1 - band and hi <= 1 + band:
        return "same"
    return "open"


def _session() -> str:
    """The baseline session: `FASTFORKS_AB_SESSION`, else today's date (a baseline is measured once a day)."""
    return os.environ.get("FASTFORKS_AB_SESSION") or time.strftime("%Y-%m-%d")


def _baseline_path(job: dict[str, Any], size: str, mlx: str, session: str, root: Path = BASELINE_DIR) -> Path:
    """The cached graph baseline of (ref, mode, dtype, input size) under MLX `mlx` in `session`."""
    return root / session / f"{job['ref']}--{job['mode']}--{job['dtype']}--{size}--mlx{mlx}.json"


def _normalized(net: list[float], ref: list[float]) -> list[float]:
    """Each network sample over the reference net's time around it (the mean of the samples before and after it).

    `ref` has one sample more than `net` (ref, net, ref, net, ..., ref): a GPU that runs slower for a while (another
    job, heat) slows both, so the ratio drifts less than the network time.
    """
    return [n / ((ref[i] + ref[i + 1]) / 2) for i, n in enumerate(net)]


def _ref_net(mx: Any) -> Callable[[], float]:  # noqa: ANN401 - mlx.core
    """A fixed small float16 network (matmuls and a convolution): its call time in ms (the drift reference).

    The fastest of `REF_TRIES` calls: right after a large network the first small call can take twice as long (the
    big network's buffers being released), which is no drift of the GPU; a GPU shared with another job stays slower
    over all of them.
    """
    keys = mx.random.split(mx.random.key(0), 6)
    x = mx.random.normal((256, 512), key=keys[0]).astype(mx.float16)
    ws = [(mx.random.normal((512, 512), key=k) / 23).astype(mx.float16) for k in keys[1:5]]
    img = mx.random.normal((1, 96, 96, 32), key=keys[5]).astype(mx.float16)
    wc = (mx.random.normal((32, 3, 3, 32), key=keys[5]) / 17).astype(mx.float16)
    mx.eval(x, img, wc, *ws)

    def once() -> float:
        t0 = time.perf_counter()
        for _ in range(4):
            y = x
            for w in ws:
                y = mx.maximum(y @ w, 0)
            z = mx.conv2d(img, wc, padding=1)
            mx.eval(y, z)
        mx.synchronize()
        return 1e3 * (time.perf_counter() - t0)

    def call() -> float:
        return min(once() for _ in range(REF_TRIES))

    for _ in range(3):
        once()
    return call


def _quick_rounds(
    sample: Callable[[], float], ref: Callable[[], float], rounds: int, base: list[list[float]] | None
) -> tuple[list[list[float]], list[float]]:
    """Normalized samples per round (`QUICK_ROUND` each), and the interval against `base`.

    With a `base`, stops once `_verdict` is decided (after `QUICK_MIN_ROUNDS`), at most `rounds`; without one, runs
    `rounds` rounds (the baseline itself).
    """
    got: list[list[float]] = []
    ci: list[float] = []
    for r in range(rounds):
        refs, nets = [ref()], []
        for _ in range(QUICK_ROUND):
            nets.append(sample())
            refs.append(ref())
        got.append(_normalized(nets, refs))
        if base is not None and r + 1 >= QUICK_MIN_ROUNDS:
            ci = _ci(got, base)
            if _verdict(ci) != "open":
                break
    return got, ci


def _quick_arm(
    job: dict[str, Any], arm: str, ctx: dict[str, Any], rounds: int, base: list[list[float]] | None
) -> dict[str, Any]:
    """Load one arm in this process, call it, time it in rounds (`_quick_rounds`), check its golden; drop it.

    `job["input"]` is the (smaller) input size; `ctx`: `timers` (`_Timers`), `ref` (`_ref_net`), `golden` (bool).
    """
    import torch  # noqa: PLC0415

    from speed_survey import MPS_WARMUP, _inputs  # noqa: PLC0415

    timers, ref = ctx["timers"], ctx["ref"]
    mx = timers.mx
    os.environ["FASTFORKS_MLX_GRAPH"] = "1" if arm == "graph" else "0"  # read at load time (`graph_switch`)
    task = __import__(f"fastforks.{job['task']}", fromlist=["load"])
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    model = task.load(job["ref"], mode=job["mode"], device="mlx", dtype=job["dtype"])
    out["load_s"] = round(time.perf_counter() - t0, 2)
    if arm == "hand":
        timers.patch_hand(job["family"])
    timers.patch_copies()
    args, kw, _ = _inputs(model, {**job, "scene": ""})

    def call() -> None:
        with torch.inference_mode():
            model(*args, **kw)

    timers.on = True
    t0 = time.perf_counter()
    call()
    out["first_s"] = round(time.perf_counter() - t0, 2)
    if not timers.calls:
        msg = "no MLX network call seen"
        raise RuntimeError(msg)
    for _ in range(MPS_WARMUP):
        timers.reset()
        call()
    loops = _loops(1e3 * timers.seconds)
    transients: list[int] = []

    ms: list[float] = []
    refs: list[float] = []

    def sample() -> float:
        timers.reset()
        for _ in range(loops):
            call()
        transients.append(timers.transient)
        ms.append(1e3 * timers.seconds / loops)
        return ms[-1]

    def ref_ms() -> float:
        refs.append(ref())
        return refs[-1]

    out["norm"], out["ci"] = _quick_rounds(sample, ref_ms, rounds, base)
    timers.on = False
    out |= {"ref_ms": round(statistics.median(refs), 4), "net_ms": round(statistics.median(ms), 3)}
    out |= {"net_transient_mb": round(max(transients) / 2**20, 1), "samples": len(ms)}
    if ctx["golden"]:
        held = [model]
        del model
        out["quality"] = _quality(held, {**job, "arm": arm})
    else:
        del model
    gc.collect()
    mx.clear_cache()
    return out


def quick(a: argparse.Namespace) -> int:
    """The quick tier (module docstring): per job the hand arm against the session's graph baseline, one process."""
    sys.path.insert(0, str(TOOLS))
    import gpu_lock  # noqa: PLC0415

    todo = jobs(a.refs, a.family, a.modes, a.dtypes)
    if not todo:
        print("no hand-ported job matches", file=sys.stderr)
        return 1
    os.environ["FASTFORKS_OFFLINE"] = "1"
    os.environ["FASTFORKS_MLX_CACHE_MB"] = str(a.cache_mb)
    with gpu_lock.session("mlx_core_ab quick"):
        from fastforks._core.mlx import mx  # noqa: PLC0415
        from fastforks.shared import mlx_graph  # noqa: PLC0415

        timers = _Timers(mx)
        for cls in (mlx_graph.MlxModule, mlx_graph.GraphNet):
            timers.patch(cls)
        ctx = {"timers": timers, "ref": _ref_net(mx), "golden": a.golden}
        mlx = _mlx_version()
        for job in todo:
            t0 = time.perf_counter()
            size = _small(job["input"], a.scale)
            path = _baseline_path(job, size, mlx, a.session or _session(), Path(a.baseline_dir))
            try:
                base = None if a.rebaseline else json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                base = None
            try:
                if base is None:
                    with gpu_lock.timed():  # the exclusive lock (under `--timed`) for the baseline only
                        base = _quick_arm(job | {"input": size}, "graph", ctx, a.base_rounds, None)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(base))
                    base["new"] = True
                hand = _quick_arm(job | {"input": size}, "hand", ctx, a.rounds, base["norm"])
            except Exception as e:  # noqa: BLE001 - one job's failure, the others go on
                print(f"{job['ref']:32s} {job['dtype']:8s} {size:>9s}  error: {' '.join(str(e).split())[:200]}")
                continue
            print(_quick_line(job, size, hand, base, time.perf_counter() - t0), flush=True)
    return 0


def quick_chunks(a: argparse.Namespace) -> int:
    """`quick` with `--lock`: one `quick` process per `--chunk` refs, each under the lock (`gpu2.sh --light`).

    `--chunk N` splits the refs into one process (and lock call) per N; this process holds no lock.
    """
    refs = list(dict.fromkeys(j["ref"] for j in jobs(a.refs, a.family, a.modes, a.dtypes)))
    if not refs:
        print("no hand-ported job matches", file=sys.stderr)
        return 1
    prefix = [a.lock, "--light"] if Path(a.lock).name == "gpu2.sh" else [a.lock]
    opts = ["--modes", *a.modes, "--dtypes", *a.dtypes, "--scale", str(a.scale), "--rounds", str(a.rounds)]
    opts += ["--base-rounds", str(a.base_rounds), "--baseline-dir", a.baseline_dir, "--cache-mb", str(a.cache_mb)]
    opts += ["--session", a.session or _session()]
    opts += [*(() if a.golden else ("--no-golden",)), *(("--rebaseline",) if a.rebaseline else ())]
    worst = 0
    n = a.chunk if a.chunk > 0 else len(refs)
    for i in range(0, len(refs), n):
        cmd = [*prefix, sys.executable, __file__, "quick", *refs[i : i + n], *opts]
        worst = max(worst, subprocess.run(cmd, check=False).returncode)
    return worst


def _quick_line(job: dict[str, Any], size: str, hand: dict[str, Any], base: dict[str, Any], wall: float) -> str:
    """One printed row: warm ratio, its interval and verdict, transient peaks, golden errors, wall time."""
    ratio = statistics.median(x for r in hand["norm"] for x in r) / statistics.median(
        x for r in base["norm"] for x in r
    )
    ci = hand["ci"]
    ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "[-]"
    tx = (hand["net_transient_mb"] + 1) / (base["net_transient_mb"] + 1)
    out = f"{job['ref']:32s} {job['dtype']:8s} {size:>9s}  warm {hand['net_ms']:9.2f} / {base['net_ms']:9.2f} ms"
    out += f" x{ratio:.3f} {ci_s} {_verdict(ci):6s}"
    if "ref_ms" in base:  # the raw ratio and how far the reference net moved since the baseline
        out += f" (raw x{hand['net_ms'] / base['net_ms']:.3f}, ref drift x{hand['ref_ms'] / base['ref_ms']:.2f})"
    out += f"  transient {hand['net_transient_mb']:7.1f} /"
    out += f" {base['net_transient_mb']:7.1f} MB x{tx:.2f}"
    hq, gq = hand.get("quality", {}), base.get("quality", {})
    if "mean" in hq:
        out += (
            f"  golden mean {hq['mean']:.3g} (graph {gq.get('mean', float('nan')):.3g}, bound {hq['bound_mean']:.3g})"
        )
        out += " WORSE" if _worse(hq, gq) else ""
    elif hq:
        out += f"  golden {next(iter(hq.values()))!s:.60s}"
    out += f"  {hand['samples']} samples, {wall:.0f} s{' (new baseline)' if base.get('new') else ''}"
    return out


# Orchestration.
def _read(out: Path) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def _key(r: dict[str, Any]) -> tuple[str, str, str]:
    return r["ref"], r["mode"], r["dtype"]


def _merge(out: Path, row: dict[str, Any]) -> None:
    out = merge_rows.fragment(out)  # FASTFORKS_ROWS_AGENT: the agent's fragment of the shared file
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


def _transient(arm: dict[str, Any]) -> float:
    """The network's transient peak (MB): its peak over the active memory before the call.

    Rows measured before `net_transient_mb` estimate it as the network peak less the resident memory (which also
    holds the weights the call does not allocate).
    """
    if "net_transient_mb" in arm:
        return arm["net_transient_mb"]
    return max(arm["net_peak_mb"] - arm.get("resident_mb", 0.0), 0.0)


def _tx(r: dict[str, Any]) -> float:
    """Hand / graph transient peak, each plus 1 MB (a network that allocates next to nothing gives no infinite x)."""
    return (_transient(r["hand"]) + 1) / (_transient(r["graph"]) + 1)


def _ci(a: list[list[float]], b: list[list[float]], seed: int = 0) -> list[float]:
    """The `CI_LEVEL` bootstrap interval of median(a) / median(b), samples grouped by process.

    Two levels, as the noise has two: the processes are resampled (with replacement), then the samples within each
    drawn process, so a process that ran slow throughout widens the interval instead of passing for five agreeing
    samples.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    a_, b_ = [np.asarray(x, dtype=np.float64) for x in a if x], [np.asarray(x, dtype=np.float64) for x in b if x]
    if not a_ or not b_:
        return []

    def draw(groups: list[Any]) -> float:
        pick = rng.integers(len(groups), size=len(groups))
        return float(np.median(np.concatenate([rng.choice(groups[i], size=len(groups[i])) for i in pick])))

    ratios = np.array([draw(a_) / draw(b_) for _ in range(BOOT)])
    tail = 100 * (1 - CI_LEVEL) / 2
    return [round(float(np.percentile(ratios, tail)), 4), round(float(np.percentile(ratios, 100 - tail)), 4)]


def _reps(arm: dict[str, Any]) -> list[list[float]]:
    """The arm's timed samples per process (`net_ms_reps`), `[]` for rows measured before them."""
    if "net_ms_reps" in arm:
        return arm["net_ms_reps"]
    return [arm["net_ms_runs"]] if "net_ms_runs" in arm else []


def _clears(row: dict[str, Any], key: str, x: float) -> bool:
    """Whether `row["ci"][key]` lies above `x` (the point ratio of rows without an interval)."""
    ci = row.get("ci", {}).get(key)
    if ci:
        return ci[0] > x
    other = row.get("shipped" if key == "shipped" else "hand", {})
    return other["net_ms"] > x * row["graph"]["net_ms"]


def _flags(row: dict[str, Any]) -> list[str]:
    """`slower` (warm, the interval above `SLOWER`), `heavier` (transient peak), `worse` (golden), `shipped-slower`.

    `shipped-slower`: the hand network at its own buffer cache (the `shipped` arm) slower than the graph one at
    `--cache-mb`, its interval above `SLOWER`: a port whose `request_cache` is too small for it.
    """
    h, g = row.get("hand", {}), row.get("graph", {})
    if "net_ms" not in h or "net_ms" not in g:
        return []
    out = []
    if _clears(row, "net_ms", SLOWER):
        out.append("slower")
    ht, gt = _transient(h), _transient(g)
    if ht > HEAVIER * gt and ht - gt > TRANSIENT_FLOOR_MB:
        out.append("heavier")
    if _worse(h.get("quality", {}), g.get("quality", {})):
        out.append("worse")
    if "net_ms" in row.get("shipped", {}) and _clears(row, "shipped", SLOWER):
        out.append("shipped-slower")
    return out


def _calls_differ(row: dict[str, Any]) -> bool:
    """The arms make a different number of network calls per model call (their `net_ms` sums differ in scope)."""
    return row["hand"].get("net_calls") != row["graph"].get("net_calls")


def _pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One arm's processes as one arm row: the timed samples pooled, the per-process ones kept.

    `net_ms` / `ms` the median of all samples, `net_peak_mb` / `net_transient_mb` the largest, the one-off figures
    (`load_s`, `first_s`, `resident_mb`, ...) the median over the processes; the rest (and `quality`, which only the
    last process measures) from the last process.
    """
    ok = [r for r in rows if "net_ms" in r]
    if not ok:
        return rows[-1] if rows else {"error": "no process ran"}
    out = dict(ok[-1])
    med = statistics.median
    out["net_ms_reps"] = [r["net_ms_runs"] for r in ok]
    out["ms_reps"] = [r["ms_runs"] for r in ok]
    flat = [t for r in ok for t in r["net_ms_runs"]]
    out["net_ms"] = round(med(flat), 3)
    out["net_ms_min"] = round(min(flat), 3)
    out["ms"] = round(med([t for r in ok for t in r["ms_runs"]]), 3)
    for k in ("net_peak_mb", "net_transient_mb"):
        out[k] = max(r[k] for r in ok)
    for k in ("load_s", "first_s", "net_first_s", "net_copy_ms", "resident_mb", "peak_footprint_mb"):
        if all(k in r for r in ok):
            out[k] = round(med(r[k] for r in ok), 3)
    for k in ("net_ms_runs", "ms_runs"):
        out.pop(k, None)
    out["repeats"] = len(ok)
    if len(ok) < len(rows):
        out["failed_repeats"] = [r.get("error", "?") for r in rows if "net_ms" not in r]
    return out


def _worse(h: dict[str, Any], g: dict[str, Any]) -> bool:
    """Whether the hand arm is clearly further from the golden than the graph arm.

    Both arms measure the same golden in the same unit, so the noise between them is the float16 rounding of two
    different op orders: `worse` needs the hand error (mean or p99 of the primary output) above `WORSE_X` times the
    graph's and above it by more than `WORSE_BUDGET` of the golden's bound, so two errors far inside the bound that
    differ in relative terms only do not flag; sparse matchers also flag a recall or match agreement lower by more than
    `WORSE_RECALL`. A hand arm whose golden check could not run (another number of detections) where the graph's could
    is `worse` too. Rows without the quality fields (measured before them) never flag.
    """
    if "error" in h and "mean" in g:
        return True
    if "mean" not in h or "mean" not in g:
        return False
    for m in ("mean", "p99"):
        a, b, bound = h[m], g[m], h[f"bound_{m}"]
        if a == a and b == b and a > WORSE_X * b and a - b > WORSE_BUDGET * bound:  # noqa: PLR0124 - NaN: empty maps
            return True
    return any(k in h and k in g and h[k] < g[k] - WORSE_RECALL for k in ("recall", "match"))


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


def run(a: argparse.Namespace) -> int:  # noqa: C901, PLR0912 - one loop over jobs, arms and repeats
    """Measure both arms of every job not yet in `--out` (all with `--force`)."""
    out = Path(a.out)
    old = {_key(r): r for r in _read(out)}
    done = {k for k, r in old.items() if "net_ms" in r.get("hand", {}) and "net_ms" in r.get("graph", {})}
    if a.flagged:  # re-measure the rows flagged now (under the current rules), measured or not
        done = {k for k in done if not _flags(old[k])}
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
            arms = [*ARMS, *((SHIPPED,) if a.shipped else ())]
            spec = {arm: {**job, "arm": arm, "cache_mb": None if arm == SHIPPED else a.cache_mb} for arm in arms}
            primed = {arm: _prime(_files(spec[arm], tmp, "prime"), env) for arm in ARMS} if a.prime else {}
            got: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
            for rep in range(a.repeats):  # interleaved: H G S, then S G H, ...
                for arm in arms if rep % 2 == 0 else arms[::-1]:
                    quality = arm != SHIPPED and rep == a.repeats - 1 and not a.keep_quality
                    got[arm].append(_arm(_files(spec[arm], tmp, str(rep)) | {"quality": quality}, env))
            for arm in arms:
                row[arm] = _pool(got[arm]) | primed.get(arm, {})
            if a.keep_quality:  # the quality of the earlier row (it does not depend on the timing method)
                for arm in ARMS:
                    if "quality" in old.get(_key(row), {}).get(arm, {}):
                        row[arm]["quality"] = old[_key(row)][arm]["quality"]
            row["agree"] = _compare(Path(f"{tmp}/hand-0.npz"), Path(f"{tmp}/graph-0.npz"))
            row["ci"] = {}
            if _reps(row["hand"]) and _reps(row["graph"]):
                row["ci"]["net_ms"] = _ci(_reps(row["hand"]), _reps(row["graph"]))
                if _reps(row.get(SHIPPED, {})):
                    row["ci"][SHIPPED] = _ci(_reps(row[SHIPPED]), _reps(row["graph"]))
            row["flags"] = _flags(row)
            row["mlx"] = _mlx_version()
            _merge(out, row)
            print(_line(row), flush=True)
            for p in Path(tmp).iterdir():
                p.unlink(missing_ok=True)
    return 0


def _files(spec: dict[str, Any], tmp: str, tag: str) -> dict[str, Any]:
    """`spec` with its row and outputs files in `tmp` (`<arm>-<tag>.json` / `.npz`), old ones removed."""
    out = spec | {"row": f"{tmp}/{spec['arm']}-{tag}.json", "npz": f"{tmp}/{spec['arm']}-{tag}.npz"}
    for k in ("row", "npz"):
        Path(out[k]).unlink(missing_ok=True)
    return out


def _arm(spec: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """One measured process of one arm: its row (or the error)."""
    try:
        subprocess.run(_worker_cmd(spec), env=env, timeout=TIMEOUT_S, check=False)
        return json.loads(Path(spec["row"]).read_text())
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {TIMEOUT_S} s"}
    except (OSError, json.JSONDecodeError):
        return {"error": "the worker crashed (no row; out of memory?)"}


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


def _quality_cell(h: dict[str, Any], g: dict[str, Any]) -> str:
    """Golden error of the primary output, hand / graph (mean |diff| in the golden test's unit), and the bound."""
    hq, gq = h.get("quality"), g.get("quality")
    if not hq and not gq:
        return "-"
    if "none" in (hq or gq or {}):
        return "no golden"

    def one(q: dict[str, Any] | None) -> str:
        if not q:
            return "-"
        if "error" in q:
            return "error"
        return f"{q['mean']:.3g}" + (f" r{q['recall']:.3f}" if "recall" in q else "")

    bound = next((q["bound_mean"] for q in (hq, gq) if q and "bound_mean" in q), None)
    return f"{one(hq)} / {one(gq)}" + (f" ({bound:g})" if bound is not None else "")


def table(rows: list[dict[str, Any]]) -> str:
    """The markdown table of the rows (hand / graph per cell), then the failed rows."""
    head = (
        "| ref | dtype | net warm ms | x | 95 % CI | shipped ms | net first s | net peak MB | transient MB | x"
        " | resident MB | e2e peak MB | e2e warm ms | load s | e2e first s | net rel | result rel"
        " | golden mean (bound) | flag |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
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
        ci = r.get("ci", {}).get("net_ms")
        est = "" if "net_transient_mb" in h else "~"  # estimated: measured before the transient field
        sh = r.get(SHIPPED, {})
        flag = " ".join(_flags(r)) + (" calls-differ" if _calls_differ(r) else "")
        lines.append(
            f"| {r['ref']} | {_dt(r)} | {pair('net_ms', '.1f')} | {_ratio(h['net_ms'], g['net_ms'])}"
            f" | {f'{ci[0]:.2f}-{ci[1]:.2f}' if ci else '-'} | {_num(sh.get('net_ms'), '.1f')}"
            f" | {pair('net_first_s', '.2f')} | {pair('net_peak_mb', '.0f')}"
            f" | {est}{_transient(h):.0f} / {est}{_transient(g):.0f} | {_tx(r):.2f} | {pair('resident_mb', '.0f')}"
            f" | {pair('peak_footprint_mb', '.0f')} | {pair('ms', '.1f')} | {pair('load_s', '.2f')}"
            f" | {pair('first_s', '.2f')} | {net_rel} | {rel} | {_quality_cell(h, g)} | {flag.strip()} |"
        )
    text = head + "\n".join(lines) + "\n"
    if failed:
        text += "\nNot measured (one arm failed):\n\n"
        for r in failed:
            h, g = r.get("hand", {}), r.get("graph", {})
            side, err = ("hand", h.get("error")) if "error" in h else ("graph", g.get("error"))
            text += f"- {r['ref']} {r['dtype']}: {side}: {err}\n"
    return text


def _dt(r: dict[str, Any]) -> str:
    return r["dtype"].replace("float", "fp")


def _measured(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in sorted(rows, key=_key) if "net_ms" in r.get("hand", {}) and "net_ms" in r.get("graph", {})]


def _x(r: dict[str, Any], k: str) -> float:
    return r["hand"][k] / r["graph"][k]


def e2e_peaks(survey: Path = SURVEY, mps: Path = SURVEY_MPS) -> dict[str, dict[tuple[str, str], float]]:
    """The survey's process peaks: `mlx[(ref, dtype)]` (portable), `mps[(ref, "")]` (the best non-upstream MPS row)."""
    out: dict[str, dict[tuple[str, str], float]] = {"mlx": {}, "mps": {}}
    for r in _read(survey):
        if r.get("status") == "ok" and r.get("mode") == "portable" and r.get("peak_footprint_mb"):
            out["mlx"][r["ref"], r.get("dtype", "")] = r["peak_footprint_mb"]
    for r in _read(mps):
        if r.get("status") == "ok" and r.get("mode") != "upstream" and r.get("peak_footprint_mb"):
            k = (r["ref"], "")
            out["mps"][k] = min(out["mps"].get(k, 1e12), r["peak_footprint_mb"])
    return out


def accepted(r: dict[str, Any], peaks: dict[str, dict[tuple[str, str], float]]) -> tuple[float, float, float] | None:
    """A flagged `heavier` row accepted as a peak trade: (e2e peak, best MPS, MLX float32), else None.

    The hand network clearly faster (warm at most `ACCEPT_X` of the graph's), not `slower` or `worse`, and its
    end-to-end MLX peak (survey row, same dtype) below the best MPS row and at most its own MLX float32 row.
    """
    flags = _flags(r)
    if flags != ["heavier"] or _x(r, "net_ms") > ACCEPT_X:
        return None
    e2e = peaks["mlx"].get((r["ref"], r["dtype"]))
    fp32 = peaks["mlx"].get((r["ref"], "float32"))
    best = peaks["mps"].get((r["ref"], ""))
    if e2e is None or fp32 is None or best is None or not (e2e < best and e2e <= fp32):
        return None
    return e2e, best, fp32


def _islands(ok: list[dict[str, Any]]) -> list[str]:
    """The resident check: refs whose float16 hand arm keeps more than `ISLAND` of its float32 resident memory."""
    by = {_key(r): r for r in ok}
    pairs = []
    for (ref, mode, dtype), r in by.items():
        f32 = by.get((ref, mode, "float32"))
        if dtype == "float16" and f32 and f32["hand"].get("resident_mb", 0) >= ISLAND_MIN_MB:
            pairs.append((r["hand"].get("resident_mb", 0.0) / f32["hand"]["resident_mb"], r, f32))
    isl = sorted((p for p in pairs if p[0] > ISLAND), key=lambda p: -p[0])
    out = ["", "## Resident: float16 against float32", ""]
    out.append(
        f"The hand arm's resident memory (weights and what the network keeps) in float16 over float32: about 0.5"
        f" where the weights are all float16. Of the {len(pairs)} refs keeping at least {ISLAND_MIN_MB:.0f} MB in"
        f" float32, {len(isl)} are above {ISLAND} (float32 islands, or buffers that do not scale with the dtype),"
        " largest first."
    )
    out += ["", "| ref | resident MB fp16 / fp32 | x | graph fp16 / fp32 |", "|---|---|---|---|"]
    for x, r, f32 in isl:
        g16, g32 = r["graph"].get("resident_mb"), f32["graph"].get("resident_mb")
        out.append(
            f"| {r['ref']} | {r['hand']['resident_mb']:.0f} / {f32['hand']['resident_mb']:.0f} | {x:.2f}"
            f" | {_num(g16, '.0f')} / {_num(g32, '.0f')} |"
        )
    return out


def md(rows: list[dict[str, Any]], peaks: dict[str, dict[tuple[str, str], float]]) -> str:  # noqa: PLR0915 - one page
    """The generated part of docs/plans/mlx-core-ab.md: counts, flagged, accepted, per family, quality, full table."""
    ok = _measured(rows)
    flagged = [r for r in ok if _flags(r)]
    acc = {_key(r): a for r in flagged if (a := accepted(r, peaks))}
    miss = [r for r in flagged if _key(r) not in acc]
    families = sorted({r["family"] for r in rows})
    beyond = [r for r in miss if max(_x(r, "net_ms"), _tx(r)) > 1.10]  # noqa: PLR2004
    worse = [r for r in ok if "worse" in _flags(r)]
    with_q = [r for r in ok if "mean" in r["hand"].get("quality", {}) and "mean" in r["graph"].get("quality", {})]
    no_golden = [r for r in ok if "none" in r["hand"].get("quality", {})]
    out = ["## Counts", ""]
    out.append(f"{len(rows)} rows over {len(families)} families; {len(ok)} measured in both arms.")
    out.append("")
    n_slow, n_heavy = sum("slower" in _flags(r) for r in ok), sum("heavier" in _flags(r) for r in ok)
    n_ship = sum("shipped-slower" in _flags(r) for r in ok)
    new = [r for r in ok if "net_transient_mb" in r["hand"]]
    with_ci = [r for r in ok if r.get("ci", {}).get("net_ms")]
    out.append(
        f"- {len(flagged)} rows flagged ({n_slow} slower, {n_heavy} heavier, {len(worse)} worse,"
        f" {n_ship} shipped-slower)."
    )
    out.append(
        f"- {len(new)} rows measured with the transient peak and the copies left out, {len(with_ci)} with a"
        f" bootstrap interval; the other {len(ok) - len(with_ci)} flag `slower` on the point ratio and `heavier` on"
        " the estimated transient peak (net peak less resident)."
    )
    out.append(f"- {sum(_calls_differ(r) for r in ok)} rows whose arms make a different number of network calls.")
    out.append(f"- {len(acc)} of them accepted as a peak trade; {len(miss)} miss the bar ({len(beyond)} beyond 10 %).")
    if ok:
        med = statistics.median
        out.append(
            f"- Median hand / graph: warm {med(_x(r, 'net_ms') for r in ok):.2f},"
            f" peak {med(_x(r, 'net_peak_mb') for r in ok):.2f}, transient {med(_tx(r) for r in ok):.2f}."
        )
    out.append(
        f"- Quality: {len(with_q)} rows with a golden error in both arms, {len(no_golden)} without a golden,"
        f" {len(worse)} hand arms `worse` than the graph's."
    )
    out += ["", "## Flagged beyond 10 %", ""]
    out += ["Not accepted, hand / graph warm or transient peak above 1.10, worst first.", ""]
    out += [
        "| ref | dtype | net warm ms hand / graph | x | 95 % CI | transient MB hand / graph | x | flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(beyond, key=lambda r: -max(_x(r, "net_ms"), _tx(r))):
        h, g = r["hand"], r["graph"]
        ci = r.get("ci", {}).get("net_ms")
        out.append(
            f"| {r['ref']} | {_dt(r)} | {h['net_ms']:.1f} / {g['net_ms']:.1f} | {_x(r, 'net_ms'):.2f}"
            f" | {f'{ci[0]:.2f}-{ci[1]:.2f}' if ci else '-'}"
            f" | {_transient(h):.0f} / {_transient(g):.0f} | {_tx(r):.2f} | {' '.join(_flags(r))} |"
        )
    out += ["", "## Accepted: faster, peak trade", ""]
    out.append(
        f"`heavier` only, warm at most {ACCEPT_X:.2f} of the graph's, the end-to-end MLX peak (survey row) below the"
        " best MPS row and at most the ref's MLX float32 row (`accepted`)."
    )
    out += [
        "",
        ACCEPT_HEAD,
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in flagged:
        if (a := acc.get(_key(r))) is not None:
            h, g = r["hand"], r["graph"]
            out.append(
                f"| {r['ref']} | {_dt(r)} | {h['net_ms']:.1f} / {g['net_ms']:.1f} | {_x(r, 'net_ms'):.2f}"
                f" | {_transient(h):.0f} / {_transient(g):.0f} | {_tx(r):.2f}"
                f" | {a[0]:.0f} | {a[1]:.0f} | {a[2]:.0f} |"
            )
    out += _islands(ok)
    out += ["", "## Per family", ""]
    out.append("Flagged rows / rows (accepted in brackets), the worst hand / graph ratio of the family (warm, peak).")
    out += [
        "",
        "| family | flagged (accepted) | slower | heavier | worse | worst warm x | worst transient x |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam in families:
        fr = [r for r in ok if r["family"] == fam]
        if not fr:
            continue
        fl = [r for r in fr if _flags(r)]
        n_acc = sum(_key(r) in acc for r in fl)
        cnt = {f: sum(f in _flags(r) for r in fr) for f in ("slower", "heavier", "worse")}
        out.append(
            f"| {fam} | {len(fl)} ({n_acc}) / {len(fr)} | {cnt['slower']} | {cnt['heavier']} | {cnt['worse']}"
            f" | {max(_x(r, 'net_ms') for r in fr):.2f} | {max(_tx(r) for r in fr):.2f} |"
        )
    out += ["", "## Quality: hand arms further from the golden", ""]
    out.append(
        f"Rule (`_worse`): hand golden error (primary output, mean or p99) above {WORSE_X} x the graph's and above it"
        f" by more than {WORSE_BUDGET:.0%} of the golden's bound; sparse matchers also recall / match agreement lower"
        f" by more than {WORSE_RECALL}."
    )
    out += [
        "",
        "| ref | dtype | output | mean hand / graph | p99 hand / graph | bound mean / p99 |",
        "|---|---|---|---|---|---|",
    ]
    for r in worse:
        hq, gq = r["hand"]["quality"], r["graph"]["quality"]
        if "error" in hq:
            out.append(f"| {r['ref']} | {_dt(r)} | - | hand: {hq['error']} | - | - |")
            continue
        out.append(
            f"| {r['ref']} | {_dt(r)} | {hq['primary']} | {hq['mean']:.4g} / {gq['mean']:.4g}"
            f" | {hq['p99']:.4g} / {gq['p99']:.4g} | {hq['bound_mean']:g} / {hq['bound_p99']:g} |"
        )
    out += ["", "## Full table", ""]
    out.append(
        "hand / graph per cell; `x` = hand / graph; `net rel` = max |hand - graph| / max |graph| over the first network"
        " call's outputs (where both arms give the same number of arrays), `result rel` the same over the model's"
        " results (a sanity check); `golden mean` = each arm's mean |diff| of the primary output against the stored"
        " golden on the golden's input (tests/test_golden.py `gpu_errors`, the bound in brackets; `rN` the sparse"
        " matchers' keypoint recall)."
    )
    out += ["", table(rows)]
    return "\n".join(out)


def _quick_parser(sub: Any) -> None:  # noqa: ANN401 - argparse's subparsers action
    """The `quick` subcommand's options."""
    q = sub.add_parser("quick", help="the quick tier: hand arm against the session's cached graph baseline")
    q.add_argument("refs", nargs="*", default=[])
    q.add_argument("--family", nargs="*", default=[])
    q.add_argument("--modes", nargs="*", default=["portable"])
    q.add_argument("--dtypes", nargs="*", default=["float16"])
    q.add_argument("--scale", type=float, default=QUICK_SCALE, help="input side share of the survey's (1: as is)")
    q.add_argument("--rounds", type=int, default=QUICK_MAX_ROUNDS, help="most rounds of the hand arm")
    q.add_argument("--base-rounds", type=int, default=QUICK_BASE_ROUNDS, help="rounds of a new graph baseline")
    q.add_argument("--no-golden", dest="golden", action="store_false", help="skip the golden check")
    q.add_argument("--rebaseline", action="store_true", help="measure the graph baseline again")
    q.add_argument("--session", default=None, help="baseline session tag (default: FASTFORKS_AB_SESSION or the date)")
    q.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    q.add_argument("--cache-mb", type=int, default=CACHE_MB)
    q.add_argument("--chunk", type=int, default=QUICK_CHUNK, help="refs per process under --lock (0: all in one)")
    q.add_argument("--lock", default=None, help="run each chunk under this lock script (gpu.sh / gpu2.sh: --light)")


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
            s.add_argument("--flagged", action="store_true", help="also re-measure the rows flagged now")
            s.add_argument("--repeats", type=int, default=REPEATS, help="processes per arm, interleaved")
            s.add_argument("--no-shipped", dest="shipped", action="store_false", help="skip the as-shipped arm")
            s.add_argument("--keep-quality", action="store_true", help="carry the earlier row's golden check over")
            s.add_argument(
                "--no-prime", dest="prime", action="store_false", help="skip the warm-up process before each arm"
            )
    _quick_parser(sub)
    t = sub.add_parser("table")
    t.add_argument("--out", default=str(OUT))
    t = sub.add_parser("md", help="rewrite the generated part of docs/plans/mlx-core-ab.md")
    t.add_argument("--out", default=str(OUT))
    t.add_argument("--md", default=str(MD))
    w = sub.add_parser("worker")
    w.add_argument("spec")
    a = p.parse_args()
    if a.cmd == "worker":
        return worker(json.loads(a.spec))
    if a.cmd == "list":
        for j in jobs(a.only, a.family, a.modes, a.dtypes):
            print(f"{j['ref']:40s} {j['family']:20s} {j['mode']:9s} {j['dtype']:8s} {j['input']}")
        return 0
    if a.cmd == "quick":
        return quick_chunks(a) if a.lock else quick(a)
    if a.cmd == "table":
        print(table(_read(Path(a.out))), end="")
        return 0
    if a.cmd == "md":
        path = Path(a.md)
        text = path.read_text(encoding="utf-8")
        head = text.split(GENERATED)[0].rstrip("\n")
        path.write_text(f"{head}\n\n{GENERATED}\n\n{md(_read(Path(a.out)), e2e_peaks())}", encoding="utf-8")
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
