"""Apple MLX ports of a family's PyTorch runner: its network runs through `fastforks.shared.mlx_graph`.

`port(Portable)` is the family's `portable_mlx` / `kernels_mlx` runner: the portable runner is built on the CPU in
float32 (its pre- and post-processing stays PyTorch on the CPU), then each network attribute (`attrs`, default
`net`) is replaced by an `MlxModule`, which exports that module once per input signature, converts its parameters
to the run dtype and replays the graph with `mx.compile` on the GPU. Kernels mode (`ctx.mode == "kernels"`) swaps in
our Metal kernels (`fastforks.kernels.graph_mlx`) for depthwise convolution and global pooling.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import gc
import os
import weakref
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from fastforks._core.runner import Context, Runner

__all__ = ["port", "run_dtype", "twin_dtype"]

POST_ON_MLX: contextvars.ContextVar[bool] = contextvars.ContextVar("mlx_post", default=False)
"""Set while an MLX runner computes: segmentation's logits resize and argmax run on MLX (`fastforks.shared.mlx_seg`)."""

_RUN_DTYPE: contextvars.ContextVar[str | None] = contextvars.ContextVar("mlx_run_dtype", default=None)


def run_dtype(ctx: Context) -> str:
    """The dtype the network runs in: while an MLX runner builds its float32 PyTorch twin, the MLX run dtype.

    A runner that rescales weights for float16 (to keep activations finite) asks this instead of `ctx.dtype`.
    """
    return _RUN_DTYPE.get() or ctx.dtype


def twin_dtype() -> str | None:
    """The MLX run dtype while an MLX runner builds its PyTorch twin, else None.

    A runner whose checkpoint is float16 can keep it as it is for a float16 replay (the twin's weights then stay the
    memory-mapped checkpoint: no float32 copy) by building its float16 graph instead of the float32 one.
    """
    return _RUN_DTYPE.get()


class LazyTwinError(RuntimeError):
    """A parameter of the lazily built PyTorch twin was never given a value (the twin is rebuilt the eager way)."""


_INITS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
)


@contextlib.contextmanager
def _lazy_twin() -> Iterator[None]:
    """While the PyTorch twin is built: its weights are never copied or randomly initialised, they are the checkpoint's.

    `load_state_dict` assigns the checkpoint's tensors to the parameters when their dtype fits (safetensors and zip
    checkpoints are memory-mapped, so those pages are clean file cache, not the process's footprint; a converted copy
    stays an ordinary copy), and the random initialisers of `torch.nn.init` do nothing (the parameter keeps untouched
    pages). The values are exactly those of the eager build. A parameter that keeps its skipped initialisation (never
    loaded, never written) would be garbage: that raises `LazyTwinError` and the caller builds the eager way.
    """
    skipped: dict[int, tuple[weakref.ref[torch.Tensor], int]] = {}
    orig_load = torch.nn.Module.load_state_dict
    orig_inits = {name: getattr(torch.nn.init, name) for name in _INITS}

    def load(self: torch.nn.Module, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:  # noqa: ANN401, FBT001, FBT002
        if assign:
            return orig_load(self, state_dict, strict, assign)
        own = self.state_dict()
        fixed = {}
        for k, v in state_dict.items():
            t = own.get(k)
            fixed[k] = v.to(t.dtype) if isinstance(v, torch.Tensor) and t is not None and v.dtype != t.dtype else v
        return orig_load(self, fixed, strict, True)

    def skip(name: str) -> Callable[..., Any]:
        def init(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401, ARG001
            skipped[id(tensor)] = (weakref.ref(tensor), tensor._version)  # noqa: SLF001
            return tensor

        init.__name__ = name
        return init

    torch.nn.Module.load_state_dict = load  # ty: ignore[invalid-assignment]
    for name in _INITS:
        setattr(torch.nn.init, name, skip(name))
    try:
        with torch.inference_mode(mode=False):  # hosts such as ComfyUI build under inference mode: no version counters
            yield
        for ref, version in skipped.values():
            t = ref()
            if t is not None and t._version == version:  # noqa: SLF001
                msg = "a parameter keeps its skipped initialisation"
                raise LazyTwinError(msg)
    finally:
        torch.nn.Module.load_state_dict = orig_load  # ty: ignore[invalid-assignment]
        for name, fn in orig_inits.items():
            setattr(torch.nn.init, name, fn)


class _Method(torch.nn.Module):
    """A runner method (tensors in, tensors out) as a module that owns the runner's network: what MLX exports."""

    def __init__(self, fn: Callable[..., Any], net: torch.nn.Module, *, prune: bool = False) -> None:
        super().__init__()
        self.net = net
        self.fn = fn
        self.prune_unused = prune  # `GraphNet`: export with only the weights a fake run of `fn` reads

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.fn(*args, **kwargs)


def _resolve(obj: Any, path: str) -> Any:  # noqa: ANN401
    """`obj.a.b` for the dotted `path` `"a.b"`."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _per_variant(value: int | Mapping[str, int], variant_id: str) -> int:
    """`value`, or its entry for `variant_id` (key `""` for the others)."""
    return value if isinstance(value, int) else value.get(variant_id, value.get("", 0))


def port(  # noqa: C901 - one builder for the four wrapping styles
    portable: type[Runner],
    *,
    attrs: tuple[str, ...] = ("net",),
    whole: bool | str = False,
    methods: tuple[str, ...] = (),
    net_attr: str = "net",
    kernels_fp16_only: bool = False,
    fp32_methods: tuple[str, ...] = (),
    compile: bool = True,  # noqa: A002 - mirrors `MlxModule`
    cache_mb: int | Mapping[str, int] = 0,
    chunk_mb: int | Mapping[str, int] = 0,
    release: bool = False,
    pipeline: bool = False,
    prune: bool = False,
    half_wino_f32: bool | tuple[tuple[int, int], ...] = False,
    fp32_inputs: tuple[str, ...] = (),
) -> type[Runner]:
    """The MLX runner of `portable`: same call, the module(s) named by `attrs` run on MLX in `ctx.dtype`.

    `methods`: names of runner methods (pure tensor computation on `self.net`) that run on MLX as one graph each; the
    instance's method is replaced by the MLX call (use with `attrs=()` when the runner reads inside its network);
    `net_attr` names the module they compute with.

    `whole=True`: the runner's `module()` (the network with its pre- and post-processing, image in, image out) runs on
    MLX as one graph and `__call__` returns `{"image": ...}` from it; for a runner whose forward is not `net(x)`.
    A string names the output instead (`whole="disparity"`; stereo pairs in, one map out).

    `fp32_methods`: names among `methods` that always run in float32 (a stage whose activations overflow float16).

    `compile=False`: replay lazily without `mx.compile` (a graph whose fused elementwise kernels exceed Metal's
    argument limit).

    `cache_mb`: raise the MLX buffer cache limit to at least this many MB. An iterative net (a GRU loop) allocates the
    same large temporaries every step; with the small default cache each one is a fresh Metal allocation (zero-filled
    pages), which costs more than the step's arithmetic (RAFT-Stereo 994x1482: 7.3 s at 128 MB, 4.0 s at 2 GB).

    A mapping gives the value per variant id (key `""`: the default for the others): the Winograd nets differ in how
    much cache their transients need (RealESRGAN x2plus 128 MB, x4plus 768 MB), and the cache is resident memory.

    `chunk_mb`: the MLX graph's checkpoint interval in MB of new activations (0: the default 256; a larger one evaluates
    less often, which the big-activation Winograd nets need at a small cache); a mapping as for `cache_mb`.

    `release=True`: the `methods`' graphs hand the twin's tensors they moved to MLX back (stand-ins, as `attrs` always
    do); only for a runner that computes nothing else with `net_attr`'s weights.

    `pipeline=True`: queue each checkpoint segment before waiting for the previous one (the GPU never idles while the
    next segment's graph is built, at the cost of one more segment's activations in flight).

    `prune=True`: each of the `methods` is exported with only the weights it reads (found by a fake-tensor run, about
    a tenth of an export); for methods that each use a small part of a large `net_attr` (every registered weight is a
    lifted placeholder, about 0.5 ms of export each, read or not).

    `kernels_fp16_only=True`: the kernels mode uses its Metal kernels in float16 only (float32 runs as portable).

    `half_wino_f32=True`: in float16, the 3x3 convs MLX would run as Winograd (its half Winograd's error is ~17x the
    direct conv's) run in float32 (portable mode; kernels mode has its own Winograd); a tuple of `(cin, cout)` pairs
    limits that to those convs.

    `fp32_inputs`: argument names of the wrapped modules / methods whose floating tensors enter a float16 replay as
    float32 (camera intrinsics and poses, depth bounds: the replay would round them before any float32 island).
    """
    out_key = "image" if whole is True else whole

    class Mlx(portable):  # ty: ignore[unsupported-base]
        __doc__ = f"Apple MLX: `{portable.__name__}` with its network run by the exported-graph translation."

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner in float32 on the CPU, then hand its network(s) to MLX."""
            token = _RUN_DTYPE.set(ctx.dtype)
            try:
                try:
                    with _lazy_twin():
                        super().__init__(dataclasses.replace(ctx, dtype="float32"))
                except LazyTwinError:
                    gc.collect()
                    super().__init__(dataclasses.replace(ctx, dtype="float32"))
            finally:
                _RUN_DTYPE.reset(token)
            cache = _per_variant(cache_mb, ctx.variant.id)
            chunk = _per_variant(chunk_mb, ctx.variant.id)
            if cache or os.environ.get("FASTFORKS_MLX_CACHE_MB"):
                from fastforks._core.mlx import mx  # noqa: PLC0415 - MLX only

                if forced := os.environ.get("FASTFORKS_MLX_CACHE_MB"):  # sweeps: this exact limit, declared or not
                    mx.set_cache_limit(int(forced) << 20)
                else:
                    mx.set_cache_limit(max(cache << 20, mx.set_cache_limit(0)))  # returns the previous limit
            from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - MLX only (imports mlx)

            kernels = ctx.mode == "kernels" and (ctx.dtype == "float16" or not kernels_fp16_only)
            compiled = (
                compile and os.environ.get("FASTFORKS_MLX_COMPILE") != "0"
            )  # sweeps: replay lazily for every port
            for name in methods:
                owner, _, leaf = name.rpartition(".")
                host = _resolve(self, owner) if owner else self
                dtype = "float32" if name in fp32_methods else ctx.dtype
                if isinstance(host, torch.nn.Module):  # a method of a submodule: it owns the weights it computes with
                    mod = MlxModule(
                        _Method(getattr(host, leaf), host, prune=prune),
                        dtype,
                        kernels=kernels,
                        compile=compiled,
                        chunk_mb=chunk,
                        release=release,
                        pipeline=pipeline,
                        half_wino_f32=half_wino_f32,
                        fp32_inputs=fp32_inputs,
                    )
                    object.__setattr__(host, leaf, mod)  # the instance entry beats the class's function
                else:
                    net = getattr(self, net_attr)
                    setattr(
                        host,
                        leaf,
                        MlxModule(
                            _Method(getattr(host, leaf), net, prune=prune),
                            dtype,
                            kernels=kernels,
                            compile=compiled,
                            chunk_mb=chunk,
                            release=release,
                            pipeline=pipeline,
                            half_wino_f32=half_wino_f32,
                            fp32_inputs=fp32_inputs,
                        ),
                    )
            if whole:
                self._whole = MlxModule(
                    self.module(),
                    ctx.dtype,
                    kernels=kernels,
                    release=True,
                    compile=compiled,
                    chunk_mb=chunk,
                    pipeline=pipeline,
                    half_wino_f32=half_wino_f32,
                    fp32_inputs=fp32_inputs,
                )
                return
            for name in attrs:
                owner, _, leaf = name.rpartition(".")
                host = _resolve(self, owner) if owner else self
                mod = MlxModule(
                    getattr(host, leaf),
                    ctx.dtype,
                    kernels=kernels,
                    release=True,
                    compile=compiled,
                    chunk_mb=chunk,
                    pipeline=pipeline,
                    half_wino_f32=half_wino_f32,
                    fp32_inputs=fp32_inputs,
                )
                if isinstance(host, torch.nn.Module):  # the MLX callable is no child module
                    host._modules.pop(leaf, None)  # noqa: SLF001 - the module registry
                    object.__setattr__(host, leaf, mod)
                else:
                    setattr(host, leaf, mod)

        if hasattr(portable, "labels"):  # segmentation: the logits' resize and argmax run on MLX too

            def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                """The runner's call with the post-processing on MLX."""
                token = POST_ON_MLX.set(True)
                try:
                    return super().__call__(*args, **kwargs)
                finally:
                    POST_ON_MLX.reset(token)

            def labels(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                """The labels-only call with the post-processing on MLX."""
                token = POST_ON_MLX.set(True)
                try:
                    return super().labels(*args, **kwargs)
                finally:
                    POST_ON_MLX.reset(token)

        if whole:

            def __call__(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
                """The module's output on MLX."""
                return {out_key: self._whole(*inputs)}

    Mlx.__name__ = Mlx.__qualname__ = f"Mlx{portable.__name__}"
    return Mlx
