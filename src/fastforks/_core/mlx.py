"""Apple MLX backend (`load(..., device="mlx")`): the one module that imports `mlx`.

MLX is an optional extra (`pip install 'fastforks[mlx]'`, Apple silicon only). Importing this module without it raises
`MissingDependencyError`; the families' MLX code imports `mx` from here, and only when a model is loaded on `"mlx"`.

What lives here:
- `mx`: `mlx.core`.
- `gpu()`: the Metal device (architecture string, generation, memory), from `mx.device_info()`.
- `TUNING` / `tuning()`: every per-GPU-generation tunable of the MLX ports in one table (the M1 values are the
  default); `FASTFORKS_MLX_TUNING` and `tuned()` override single fields for `tools/mlx_tune.py`.
- `setup()`: caps MLX's buffer cache (unbounded by default: 10.7 GB and swap on a 16 GB Mac).
- `converted()`: the weights of a variant in the port's layout, converted once and cached as safetensors.
- `to_mx()` / `to_torch()`: the host <-> MLX hand-over (one copy each way; the memory is unified).
- `metal_kernel()` / `source_tag()`: for the MLX kernel libraries (`fastforks.kernels.*_mlx`).

The task APIs stay as they are: images are converted, scaled, padded and cropped by the same PyTorch code as for every
other device, on the CPU; a runner turns its prepared CPU tensors into `mx.array`s, runs, and returns CPU tensors.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import platform
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks._core.errors import MissingDependencyError, OptionsError, install_hint
from fastforks._core.files import cache_root

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
    from collections.abc import Callable, Generator
    from pathlib import Path

    from fastforks._core.runner import Context

__all__ = [
    "TUNING",
    "Gpu",
    "Tuning",
    "converted",
    "gpu",
    "has_neural_accelerators",
    "metal_kernel",
    "mx",
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

    cache_fraction: float = 1 / 32
    cache_min_mb: int = 512
    cache_cap_mb: int = 2048
    threadgroup: int = 256
    warp_lane: int = 8
    conv_pad: bool = True
    winograd: bool = True
    batch_pair: bool = False
    compile_steps: bool = True

    def cache_limit(self, memory: int) -> int:
        """The buffer cache limit in bytes for a machine with `memory` bytes: 512 MB at 16 GB, capped at 2 GB."""
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
    return dataclasses.replace(TUNING[tuning_row()], **(_parse(os.environ.get(_ENV, "")) | _OVERRIDE))


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


def setup() -> None:
    """Cap MLX's buffer cache for this process (idempotent; every MLX load calls it)."""
    mx.set_cache_limit(tuning().cache_limit(gpu().memory))


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
    a = a.astype(mx.float32)
    mx.eval(a)
    return torch.from_numpy(np.array(a))


def converted(
    ctx: Context, layout: str, build: Callable[[], dict[str, mx.array]], *, weights: int = 0
) -> dict[str, mx.array]:
    """The float32 parameters of `ctx.variant` in a port's layout: converted by `build` once, then read from the cache.

    The cache file is `<FASTFORKS_CACHE>/mlx/<task>/<family>/<variant>-r<revision>-<sha256[:12]>-<layout>.safetensors`
    (the hash is that of the variant's recorded weight file): a new checkpoint, revision or layout is a new file,
    and a cache hit needs neither the original checkpoint nor PyTorch's unpickler. The user's files of
    `load(weights=...)` are converted on every load and never cached.

    Args:
        ctx: The runner's context.
        layout: Name and version of the port's parameter layout, plus every tunable that changes it (e.g.
            `"v1-w128"`); change it whenever `build` changes.
        build: Reads the checkpoint (`ctx.state_dict(...)`) and returns the float32 arrays by name.
        weights: Index of the weight file `build` reads (for the hash in the file name).
    """
    if ctx.user_weights is not None:
        return build()
    sha = ctx.variant.weights[weights].sha256[:12]
    name = f"{ctx.variant.id}-r{ctx.variant.revision}-{sha}-{layout}.safetensors"
    path = cache_root() / "mlx" / ctx.card.task / ctx.card.family / name
    if path.is_file():
        return dict(cast("dict[str, mx.array]", mx.load(str(path))))  # a safetensors file loads as a dict
    arrays = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{os.getpid()}.{path.name}")  # `mx.save_safetensors` wants the suffix
    try:
        mx.save_safetensors(str(tmp), arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return arrays
