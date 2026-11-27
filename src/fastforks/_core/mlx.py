"""Apple MLX backend (`load(..., device="mlx")`): the one module that imports `mlx`.

MLX is an optional extra (`pip install 'fastforks[mlx]'`, Apple silicon only). Importing this module without it raises
`MissingDependencyError`; the families' MLX code imports `mx` from here, and only when a model is loaded on `"mlx"`.

What lives here:
- `mx`: `mlx.core`.
- `gpu()`: the Metal device (architecture string, generation, memory), from `mx.device_info()`.
- `TUNING` / `tuning()`: every per-GPU-generation tunable of the MLX ports in one table (the M1 values are the
  default); `FASTFORKS_MLX_TUNING` and `tuned()` override single fields for `tools/mlx_tune.py`.
- `graph_switch()`: `FASTFORKS_MLX_GRAPH=1` (diagnostics only) builds every hand-written MLX network as the
  exported-graph translation of its PyTorch network instead, as before its hand port (`tools/mlx_core_ab.py`).
- `setup()`: caps MLX's buffer cache (unbounded by default: 10.7 GB and swap on a 16 GB Mac).
- `convert_cached()` / `converted()`: the arrays of a variant in the port's layout and run dtype, converted once and
  cached as safetensors with their provenance (`fastforks._core.mlx_cache`); a hit needs neither torch nor the
  checkpoint.
  `FASTFORKS_MLX_CACHE_SRC=0` (development) leaves the port sources' hash out of the file name, so editing a
  converter does not convert again (bump the layout tag, or delete the file, when the stored arrays change).
- `dtype_audit()`: `FASTFORKS_MLX_DTYPE_AUDIT=1` lists, after each float16 / bfloat16 load, the float32 arrays over
  1 MB the runner holds (the fp32 islands; on stderr).
- `to_mx()` / `to_torch()`: the host <-> MLX hand-over (one copy each way; the memory is unified). torch is imported
  by `to_torch` only: this module itself does not import it.
- `metal_kernel()` / `source_tag()`: for the MLX kernel libraries (`fastforks.kernels.*_mlx`).

The task APIs stay as they are: images are converted, scaled, padded and cropped by the same PyTorch code as for every
other device, on the CPU; a runner turns its prepared CPU tensors into `mx.array`s, runs, and returns CPU tensors.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import itertools
import json
import os
import platform
import re
import threading
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fastforks._core import mlx_cache
from fastforks._core.errors import MissingDependencyError, OptionsError, install_hint

# MLX frees a command buffer's intermediates only when the buffer completes: with its default size (40-50 MB of
# inputs per buffer on an M1) a deep convolutional trunk holds every activation at once. Smaller buffers cost no speed
# (a 32-layer 64-channel trunk at 512^2: 1.06 GB -> 0.26 GB peak). MLX reads this when it is imported.
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "8")

try:
    import mlx.core as mx
except ImportError as e:  # not installed, or not Apple silicon (the wheel exists only there)
    msg = f"device='mlx' needs Apple silicon and the mlx package: {install_hint('mlx')}"
    raise MissingDependencyError(msg) from e

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from pathlib import Path

    import torch

    from fastforks._core.runner import Context

__all__ = [
    "DTYPES",
    "SDPA_BLOCK_BYTES",
    "SDPA_FUSED_DIMS",
    "TUNING",
    "Converted",
    "Gpu",
    "Tuning",
    "attention",
    "cache_request_limit",
    "convert_cached",
    "converted",
    "gpu",
    "graph_switch",
    "has_neural_accelerators",
    "metal_kernel",
    "mx",
    "request_cache",
    "setup",
    "source_tag",
    "to_mx",
    "to_torch",
    "tuned",
    "tuning",
    "tuning_row",
    "version",
]

_MB = 1 << 20
_NEURAL_ACCELERATOR_GENERATION = 17  # the M5's GPU (applegpu_g17*)
_NEURAL_ACCELERATOR_MACOS = (26, 2)  # MLX's own gate for the accelerated matmul / attention


@dataclass(frozen=True)
class Gpu:
    """The Metal device MLX runs on.

    Attributes:
        name: Marketing name, e.g. `"Apple M1 Pro"`.
        architecture: `mx.device_info()["architecture"]`, e.g. `"applegpu_g13s"`.
        generation: The number in it: 13 = M1, 14 = M2, 15 = M3, 16 = M4, 17 = M5; 0 if it cannot be read.
        tier: The trailing letter (`p` phone, `g` base, `s` Pro, `c` Max, `d` Ultra), or `""`.
        memory: Unified memory in bytes.
        macos: `(major, minor)` of the running macOS.
    """

    name: str
    architecture: str
    generation: int
    tier: str
    memory: int
    macos: tuple[int, ...]


def gpu() -> Gpu:
    """The Metal device, read from `mx.device_info()` on every call (cheap)."""
    info = mx.device_info()
    arch = str(info.get("architecture", ""))
    m = re.search(r"g(\d+)([a-z]?)$", arch)
    macos = tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2])
    return Gpu(
        name=str(info.get("device_name", "")),
        architecture=arch,
        generation=int(m.group(1)) if m else 0,
        tier=m.group(2) if m else "",
        memory=int(info.get("memory_size", 0)),
        macos=macos,
    )


def has_neural_accelerators() -> bool:
    """Whether MLX runs matmul and attention on the GPU's neural accelerators (M5 and later, macOS 26.2+).

    Hand-written kernels for those operations should step aside there; nothing in the library does yet (no such
    machine to measure on), the query is here for the ports that will.
    """
    g = gpu()
    return g.generation >= _NEURAL_ACCELERATOR_GENERATION and g.macos >= _NEURAL_ACCELERATOR_MACOS


@dataclass(frozen=True)
class Tuning:
    """The tunables of the MLX ports. One value set per GPU generation in `TUNING`; the defaults are the M1 values.

    Every field has a micro-benchmark in `tools/mlx_tune.py` (the name in brackets), which prints the value to put
    into `TUNING` for the machine it ran on.

    Attributes:
        cache_fraction: MLX buffer cache limit as a fraction of the unified memory [cache].
        cache_min_mb: Lower bound of that limit [cache].
        cache_cap_mb: Upper bound of that limit [cache].
        threadgroup: Threads per threadgroup of the 1-D Metal kernels (correlation, GRU, norm, upsampling)
            [threadgroup].
        warp_lane: Channels one thread of the warp kernel interpolates [lane].
        conv_pad: Run 96-channel 3x3 convolution stages 128 wide with zero filters (CREStereo's encoder): on the M1,
            MLX's convolution with 96 outputs is 3-6x slower than with 128. Costs memory [conv-pad].
        winograd: The GRU's 1x5 / 5x1 convolutions as Winograd F(4, 5) with one float32 matmul (kernels mode); off:
            MLX's convolution [winograd].
        batch_pair: Run both images of a stereo pair through the encoder in one batch; off: one at a time (same
            speed on the M1, lower peak memory) [batch-pair].
        compile_steps: `mx.compile` the per-iteration step and the encoders [compile].
    """

    cache_fraction: float = 1 / 128
    cache_min_mb: int = 128
    cache_cap_mb: int = 2048
    threadgroup: int = 256
    warp_lane: int = 8
    conv_pad: bool = True
    winograd: bool = True
    batch_pair: bool = False
    compile_steps: bool = True

    def cache_limit(self, memory: int) -> int:
        """The buffer cache limit in bytes for a machine with `memory` bytes.

        128 MB at 16 GB (1/128 of the memory, at least 128 MB), capped at 2 GB.
        """
        return int(min(self.cache_cap_mb * _MB, max(self.cache_min_mb * _MB, memory * self.cache_fraction)))


# The one table of per-generation tunables. Keys: a full architecture string (`"applegpu_g16s"`), a generation
# (`"applegpu_g16"`: every tier of it), or `"default"`. The most specific match wins. `"default"` holds the values
# measured on the M1 Pro (applegpu_g13s, 16 GB) and is what every GPU without an entry runs, so a port works
# everywhere before it is tuned anywhere else. To tune a new Mac: `python tools/mlx_tune.py` there, then add a row
# with the fields that differ, e.g. `"applegpu_g16": dataclasses.replace(_M1, threadgroup=512, batch_pair=True)`.
_M1 = Tuning()
TUNING: dict[str, Tuning] = {
    "default": _M1,
}

_ENV = "FASTFORKS_MLX_TUNING"
_OVERRIDE: dict[str, object] = {}


def _parse(text: str) -> dict[str, object]:
    """`"winograd=0,threadgroup=128"` -> field values, typed as the `Tuning` defaults."""
    out: dict[str, object] = {}
    fields = {f.name: f for f in dataclasses.fields(Tuning)}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, _, value = item.partition("=")
        if name not in fields:
            msg = f"{_ENV}: unknown tunable {name!r} (have: {', '.join(fields)})"
            raise OptionsError(msg)
        kind = type(getattr(_M1, name))
        try:
            out[name] = value.lower() in {"1", "true", "on", "yes"} if kind is bool else kind(value)
        except ValueError as e:
            msg = f"{_ENV}: {name}={value!r} is not a {kind.__name__}"
            raise OptionsError(msg) from e
    return out


def tuning_row() -> str:
    """The key of the `TUNING` row this GPU runs with: its architecture, its generation, or `"default"`."""
    g = gpu()
    family = re.sub(r"[a-z]$", "", g.architecture) if g.generation else g.architecture
    return next(k for k in (g.architecture, family, "default") if k in TUNING)


def tuning() -> Tuning:
    """The tunables in effect: the `TUNING` row of this GPU, then `FASTFORKS_MLX_TUNING`, then `tuned()` overrides.

    Read when a model is built (and when a compiled step is traced), not per call.
    """
    return _tuning(TUNING[tuning_row()], os.environ.get(_ENV, ""), tuple(_OVERRIDE.items()))


@functools.lru_cache(maxsize=8)
def _tuning(row: Tuning, env: str, override: tuple[tuple[str, object], ...]) -> Tuning:
    """`tuning()` for one table row, environment value and override set: kernel launchers call it per op."""
    return dataclasses.replace(row, **(_parse(env) | dict(override)))


def version() -> str:
    """The installed MLX version (`"0.32.4"`)."""
    from importlib.metadata import version as dist_version  # noqa: PLC0415 - only for `env` and the survey rows

    return dist_version("mlx")


@contextlib.contextmanager
def tuned(**fields: object) -> Generator[None]:
    """Override tunables for the models built inside the block (`tools/mlx_tune.py`; not thread-safe)."""
    unknown = set(fields) - {f.name for f in dataclasses.fields(Tuning)}
    if unknown:
        msg = f"unknown tunable(s) {sorted(unknown)}"
        raise OptionsError(msg)
    before = dict(_OVERRIDE)
    _OVERRIDE.update(fields)
    try:
        yield
    finally:
        _OVERRIDE.clear()
        _OVERRIDE.update(before)


GRAPH_ENV = "FASTFORKS_MLX_GRAPH"


def graph_switch() -> bool:
    """Whether `FASTFORKS_MLX_GRAPH=1`: the hand-written MLX networks are built as their exported-graph translation.

    A diagnostic switch for the core A/B (`tools/mlx_core_ab.py`), off by default: a family with a hand-written network
    keeps the runner its MLX port had before (`GRAPH_RUNNERS` in its model module; the libnet classifiers: the generic
    `GraphNet` of `libnet.Mlx`), so both networks can be measured on the same inputs. Any other value than `1` (unset,
    empty, `0`) leaves every load as it is.
    """
    return os.environ.get(GRAPH_ENV) == "1"


_CACHE_ENV = "FASTFORKS_MLX_CACHE_MB"  # sweeps: this exact buffer cache limit, whatever the models ask for
_CACHE_SHARE = 8  # a model's `cache_mb` request is honoured up to 1/8 of the unified memory (2 GB at 16 GB)
_cache_lock = threading.RLock()  # re-entrant: a finalizer can run (garbage collection) while a load holds it
_cache_requests: dict[int, int] = {}  # live models' requests in bytes, by token
_cache_tokens = itertools.count()


def _base_cache_limit() -> int:
    return tuning().cache_limit(gpu().memory)


def cache_request_limit(mb: int, memory: int) -> int:
    """The bytes a model's request for `mb` MB of buffer cache gets on a machine with `memory` bytes.

    Capped at 1/8 of the unified memory, so that a port tuned on a 16 GB Mac (RAFT-Stereo: 2 GB) does not keep 2 GB of
    freed buffers resident on an 8 GB one (1 GB there); 0 if the memory size cannot be read (the base limit stays).
    """
    return min(mb * _MB, memory // _CACHE_SHARE)


def _apply_cache_limit() -> int:
    """Set MLX's buffer cache limit: the tuning's base limit, or the largest request of a live model if larger."""
    with _cache_lock:
        if forced := os.environ.get(_CACHE_ENV):
            limit = int(forced) * _MB
            mx.set_cache_limit(limit)
            return limit
        base = _base_cache_limit()
        while True:
            requests = list(_cache_requests.values())
            limit = max([base, *requests])
            mx.set_cache_limit(limit)
            if requests == list(_cache_requests.values()):  # no owner was collected (on this thread) meanwhile
                return limit


def _release_cache(token: int) -> None:
    with _cache_lock:
        if _cache_requests.pop(token, None) is not None:
            _apply_cache_limit()


def request_cache(owner: object, mb: int) -> int:
    """Raise MLX's buffer cache limit to at least `mb` MB (see `cache_request_limit`) for as long as `owner` lives.

    The limit is process-wide: it is the largest request over the live owners (never below the tuning's base limit),
    so a later load that asks for less does not lower it, and it drops back when the owner is collected. Returns the
    limit in effect, in bytes.
    """
    with _cache_lock:
        if mb > 0:
            token = next(_cache_tokens)
            _cache_requests[token] = cache_request_limit(mb, gpu().memory)
            weakref.finalize(owner, _release_cache, token).atexit = False  # nothing to restore at exit
        return _apply_cache_limit()


def setup() -> None:
    """Cap MLX's buffer cache for this process (idempotent; every MLX load calls it).

    Never below what a live model asked for with `request_cache`: a second load keeps the first one's larger cache.
    """
    _apply_cache_limit()


def source_tag(name: str, *files: Path) -> str:
    """Tag of an MLX kernel library, `name@<first 12 hex of the SHA-256 of its sources>:mlx` (as `Backend.kernels`)."""
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.read_bytes())
    return f"{name}@{digest.hexdigest()[:12]}:mlx"


def metal_kernel(**spec: Any) -> Callable[..., list[mx.array]]:  # noqa: ANN401 - `mx.fast.metal_kernel`'s arguments
    """`mx.fast.metal_kernel(**spec)`, typed: MLX's stubs declare the kernel object as `object`."""
    return cast("Callable[..., list[mx.array]]", mx.fast.metal_kernel(**spec))


def to_mx(t: torch.Tensor) -> mx.array:
    """A CPU tensor as an `mx.array` of the same dtype and shape (one copy into MLX's buffer)."""
    return mx.array(t.detach().contiguous().numpy())


def to_torch(a: mx.array) -> torch.Tensor:
    """An `mx.array` (evaluated here) as a float32 CPU tensor (one copy out of MLX's buffer)."""
    import torch  # noqa: PLC0415 - this module stays torch-free until a tensor is asked for

    a = a.astype(mx.float32)
    mx.eval(a)
    return torch.from_numpy(np.array(a))


SDPA_FUSED_DIMS = (64, 72, 80, 96, 128)
"""Head sizes `mx.fast.scaled_dot_product_attention` runs fused (measured, MLX 0.32); others materialise the scores."""
SDPA_BLOCK_BYTES = 128 << 20
"""Float32 scores per query block of an unfused attention whose whole score matrix would exceed twice this."""


def attention(q: mx.array, k: mx.array, v: mx.array, *, scale: float, mask: mx.array | str | None = None) -> mx.array:
    """`mx.fast.scaled_dot_product_attention` of `[B, H, L, D]` heads, never materialising a large score matrix.

    MLX fuses only the head sizes `SDPA_FUSED_DIMS`; any other materialises the float32 scores. A smaller head size is
    zero-padded to the next fused one (the same scores, the padded output columns zero, cut off) where the scores
    outweigh the padded copies (`lq * lk > 2 * fused * (lq + lk)`; measured: 12 x 12 windows are leaner and in float32
    faster unpadded, 16384 x 256 SegFormer scores 2.5 times leaner padded), a larger one runs in
    query blocks of `SDPA_BLOCK_BYTES` scores one after the other (`mx.depends`). The graph translation's attention
    (`mlx_graph`) and the hand-written networks share this.
    """
    d = q.shape[-1]
    if d in SDPA_FUSED_DIMS or q.ndim != 4 or isinstance(mask, str):  # noqa: PLR2004 - [B, H, L, D]
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if d < SDPA_FUSED_DIMS[-1] and k.shape[-1] == d and v.shape[-1] == d:
        fused = next(f for f in SDPA_FUSED_DIMS if f > d)
        lq, lk = q.shape[-2], k.shape[-2]
        if lq * lk <= 2 * fused * (lq + lk):  # the padded copies would outweigh the scores (short windows: 12 x 12)
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        pad = [(0, 0)] * 3 + [(0, fused - d)]
        out = mx.fast.scaled_dot_product_attention(
            mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad), scale=scale, mask=mask
        )
        return out[..., :d]
    b, h, lq, _ = q.shape
    row = b * h * k.shape[-2] * 4  # bytes of one query's float32 scores
    if lq * row <= 2 * SDPA_BLOCK_BYTES:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    blk = max(1, SDPA_BLOCK_BYTES // row)
    per_query = isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == lq  # noqa: PLR2004
    outs: list[mx.array] = []
    for i in range(0, lq, blk):
        qi = q[:, :, i : i + blk]
        if outs:
            qi = mx.depends(qi, outs[-1])
        mi = mask[..., i : i + blk, :] if per_query else mask
        outs.append(mx.fast.scaled_dot_product_attention(qi, k, v, scale=scale, mask=mi))
    return mx.concatenate(outs, axis=2)


DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
"""Run dtype names (the `ctx.dtype` strings) -> MLX dtypes."""

_FP16_MAX = 65504.0


@dataclass(frozen=True)
class Converted:
    """What `convert_cached` returns.

    Attributes:
        arrays: The arrays by name, as stored (lazy reads of the cache file on a hit).
        config: The port's JSON configuration stored with them ({} if it gave none).
        hit: Read from the cache (False: converted by this call).
    """

    arrays: dict[str, mx.array]
    config: dict[str, Any]
    hit: bool


def _keeper(fp32: tuple[str, ...] | Callable[[str], bool]) -> Callable[[str], bool]:
    if isinstance(fp32, tuple):
        return lambda k: k.startswith(fp32)
    return fp32


def _store(
    arrays: dict[str, mx.array], dtype: str, keep: Callable[[str], bool], what: str
) -> tuple[dict[str, mx.array], list[str]]:
    """The floating arrays in `dtype` (the `keep` ones float32), evaluated one by one; and the float32 islands.

    Raises:
        ValueError: A finite value does not fit float16 (store bfloat16, or keep that array float32).
    """
    target = DTYPES[dtype]
    out: dict[str, mx.array] = {}
    for k in list(arrays):
        a = arrays.pop(k)  # one by one: the source of a cast is released when it is evaluated
        if mx.issubdtype(a.dtype, mx.floating):
            want = mx.float32 if keep(k) else target
            if a.dtype != want:
                if want == mx.float16 and a.dtype != mx.bfloat16 and a.size:
                    big = mx.max(mx.where(mx.isfinite(a), mx.abs(a), 0)).item()
                    if big > _FP16_MAX:
                        msg = (
                            f"{what}: {k} holds {big:.4g}, beyond float16's range: store it in bfloat16 or keep it "
                            "float32 (`fp32`)"
                        )
                        raise ValueError(msg)
                a = a.astype(want)
        mx.eval(a)
        out[k] = a
    islands = sorted(k for k, a in out.items() if a.dtype == mx.float32 and target != mx.float32)
    return out, islands


def convert_cached(  # noqa: PLR0913 - the builder, its cache key and its storage options
    ctx: Context,
    layout: str,
    build: Callable[[], tuple[dict[str, mx.array], Mapping[str, Any]]],
    *,
    dtype: str = "float32",
    fp32: tuple[str, ...] | Callable[[str], bool] = (),
    weights: int = 0,
    sources: tuple[Path, ...] = (),
) -> Converted:
    """The arrays of `ctx.variant` as a port runs them: converted by `build` once, then read from the cache.

    The file (`fastforks._core.mlx_cache`) is named by the variant, revision, weight file hash, layout and `dtype`;
    its metadata holds the provenance and `build`'s configuration. A hit reads neither the checkpoint nor imports
    torch: the arrays are lazy reads of the file (memory-mapped, each read when evaluated). The user's files of
    `load(weights=...)` are converted on every load and never cached. Written atomically (a per-process temporary
    file, then a rename).

    Args:
        ctx: The runner's context.
        layout: Name and version of the port's layout, plus every tunable that changes it (`"vit-cls-v2"`); change it
            whenever `build` changes (or give `sources`).
        build: Reads the checkpoint (`ctx.state_dict(...)`, the twin) and returns the arrays by name, every load-time
            transform applied, and a JSON-able configuration the port needs besides them ({} for none).
        dtype: The run dtype the floating arrays are stored in (integer arrays keep theirs).
        fp32: Array names (prefixes, or a predicate) stored float32 whatever `dtype` is: tables the port computes
            with in float32 (RoPE angles, an attention's logit scale).
        weights: Index of the weight file `build` reads (its hash is in the file name and metadata).
        sources: Files whose contents define the layout (the port's conversion code): their hash joins the layout
            tag, so a changed converter never reads an old file; `prune` removes the superseded one.

    Raises:
        ValueError: An array does not fit float16 (see `fp32`).
    """
    keep = _keeper(fp32)
    if ctx.user_weights is not None:
        arrays, config = build()
        return Converted(_store(dict(arrays), dtype, keep, ctx.variant.ref)[0], dict(config), hit=False)
    tag = layout
    if sources and os.environ.get(CACHE_SRC_ENV) != "0":
        digest = hashlib.sha256()
        for f in sources:
            digest.update(f.read_bytes())
        tag = f"{layout}-{digest.hexdigest()[:8]}"
    spec = ctx.variant.weights[weights]
    name = mlx_cache.file_name(ctx.variant.id, ctx.variant.revision, spec.sha256, tag, dtype)
    path = mlx_cache.cache_dir() / ctx.card.task / ctx.card.family / name
    if path.is_file():
        loaded, meta = cast("tuple[dict[str, mx.array], dict[str, str]]", mx.load(str(path), return_metadata=True))
        if meta.get("format") == mlx_cache.FORMAT and meta.get("dtype") == dtype:
            # read now: a lazy load captured by `mx.compile` is traced into the graph, which then holds its own copy
            # of the weights beside these (eva02-large: 1.2 GB resident instead of 0.6)
            mx.eval(loaded)
            return Converted(dict(loaded), json.loads(meta.get("config", "{}")), hit=True)
        del loaded  # not ours (another format): converted again below and replaced
    arrays, config = build()
    stored, islands = _store(dict(arrays), dtype, keep, ctx.variant.id)
    from fastforks import __version__  # noqa: PLC0415 - torch-free; only on a miss

    meta = {
        "format": mlx_cache.FORMAT,
        "variant": ctx.variant.id,
        "task": ctx.card.task,
        "family": ctx.card.family,
        "revision": str(ctx.variant.revision),
        "weights": spec.name,
        "weights_sha256": spec.sha256,
        "layout": tag,
        "layout_name": layout,
        "dtype": dtype,
        "fp32": json.dumps(islands),
        "fastforks": __version__,
        "mlx": version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": json.dumps(dict(config), sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # per process and thread; ends in the suffix `mx.save_safetensors` wants
    tmp = path.with_name(f".{os.getpid()}.{threading.get_ident()}.{path.name}")
    try:
        mx.save_safetensors(str(tmp), stored, metadata=meta)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return Converted(stored, dict(config), hit=False)


CACHE_SRC_ENV = "FASTFORKS_MLX_CACHE_SRC"  # "0": the sources' hash left out of the cache key (development)
AUDIT_ENV = "FASTFORKS_MLX_DTYPE_AUDIT"  # "1": `dtype_audit` after every reduced-precision load
AUDIT_MIN_BYTES = 1 << 20


def _children(o: object) -> list[tuple[str, object]]:
    """What `_arrays_of` walks into from `o`: (path suffix, object); nothing for torch objects, modules, classes."""
    import types  # noqa: PLC0415 - audit only

    if (type(o).__module__ or "").startswith("torch") or isinstance(
        o, (str, bytes, int, float, bool, type(None), type, types.ModuleType, types.MethodType)
    ):
        return []
    if isinstance(o, dict):
        return [(f"[{k!r}]", v) for k, v in list(o.items())]
    if isinstance(o, (list, tuple, set, frozenset)):
        return [(f"[{i}]", v) for i, v in enumerate(o)]
    if isinstance(o, functools.partial):
        return [(".func", o.func), (".args", o.args), (".keywords", o.keywords)]
    if isinstance(o, types.FunctionType):
        items: list[tuple[str, object]] = [(".__defaults__", o.__defaults__ or ())]
        for i, c in enumerate(o.__closure__ or ()):
            with contextlib.suppress(ValueError):  # an empty cell
                items.append((f".<cell {i}>", c.cell_contents))
        return items
    return [(f".{k}", v) for k, v in list(vars(o).items())] if hasattr(o, "__dict__") else []


def _arrays_of(root: object, limit: int = 14) -> dict[int, tuple[str, mx.array]]:
    """Every `mx.array` reachable from `root` (attributes, containers, partials, closures), by id, with a path."""
    found: dict[int, tuple[str, mx.array]] = {}
    seen: set[int] = set()
    stack: list[tuple[object, str, int]] = [(root, "", 0)]
    while stack:
        o, path, depth = stack.pop()
        if depth > limit or id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, mx.array):
            found.setdefault(id(o), (path, o))
        else:
            stack.extend((v, path + k, depth + 1) for k, v in _children(o))
    return found


def dtype_audit(runner: object, what: str = "", *, min_bytes: int = AUDIT_MIN_BYTES) -> list[tuple[str, int]]:
    """The float32 arrays over `min_bytes` reachable from a loaded runner, largest first, as `(path, bytes)`.

    Printed on stderr with the float32 share of all the arrays found. For a float16 port these are its fp32 islands
    (weights kept float32, float32 copies, position or RoPE tables): each should be listed in the port's notes or
    be small. Arrays built on the first call (per-size tables, lazily built networks) are not seen here.
    """
    import sys  # noqa: PLC0415 - audit only

    found = _arrays_of(runner)
    total = sum(a.nbytes for _, a in found.values())
    f32 = sorted(((p, a.nbytes) for p, a in found.values() if a.dtype == mx.float32), key=lambda r: -r[1])
    big = [(p, n) for p, n in f32 if n > min_bytes]
    share = sum(n for _, n in f32)
    head = f"{what}: {len(found)} arrays, {total / 2**20:.1f} MB; float32 {share / 2**20:.1f} MB"
    lines = [
        f"[dtype audit] {head} ({share / max(total, 1):.0%}); {len(big)} over {min_bytes / 2**20:g} MB",
        *(f"[dtype audit]   {n / 2**20:8.1f} MB  {p or '<runner>'}" for p, n in big),
    ]
    sys.stderr.write("".join(line + "\n" for line in lines))
    return big


def converted(  # noqa: PLR0913 - `convert_cached`'s, kept for the existing ports
    ctx: Context,
    layout: str,
    build: Callable[[], dict[str, mx.array]],
    *,
    weights: int = 0,
    dtype: str = "float32",
    fp32: tuple[str, ...] | Callable[[str], bool] = (),
) -> dict[str, mx.array]:
    """`convert_cached` for a `build` without configuration: the arrays only (float32 unless `dtype` says)."""
    return convert_cached(ctx, layout, lambda: (build(), {}), dtype=dtype, fp32=fp32, weights=weights).arrays
