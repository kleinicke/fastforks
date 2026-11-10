"""Task-agnostic loading: model ids, backends (`portable` / `kernels` / `auto`), precision, devices and options.

Every task package (`fastforks.stereo`, `fastforks.depth`, ...) exposes the same `load`, `list_models` and
`model_card`; they call `load_model`, `list_models` and `model_card` here with their task name and their `Model`
subclass, which adds the task's call signature and result type.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import functools
import importlib
import json
import logging
import math
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args

import torch

from fastforks._core import mps, registry, warmup
from fastforks._core.arguments import bind_error
from fastforks._core.cards import Dtype  # noqa: TC001 - re-exported to the task APIs, which annotate with it
from fastforks._core.checkpoint import mismatch_details, torch_mismatch
from fastforks._core.deferred_init import deferred_init, float32_default
from fastforks._core.errors import (
    ArgumentError,
    BackendUnavailableError,
    CardError,
    CheckpointMismatchError,
    InputError,
    KernelUnavailableError,
    OptionsError,
    WeightsError,
)
from fastforks._core.kernels import begin_call, checked, cuda_device
from fastforks._core.runner import Context, Runner
from fastforks._core.sizes import SizeRange, size_range
from fastforks._core.tasks import spec

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence
    from typing import Self

    from fastforks._core.cards import Commercial, Mode, ModelCard, Tier, Variant
    from fastforks._core.export import SizeBounds

_KERNELS_DEVICE: list[
    torch.device
] = []  # the one CUDA device `kernels` runs on in this process (kernel state is global)
_KERNELS_DEVICE_LOCK = threading.Lock()

ModeArg = Literal["portable", "kernels", "auto"]

_FORKED = False  # this process is a `fork()` child (set by the at-fork hook below)


def _mark_forked() -> None:
    global _FORKED  # noqa: PLW0603
    _FORKED = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_mark_forked)


class ReadOnlyDict(Mapping[str, Any]):
    """A read-only `dict` copy (`model.options`, every result's `options`) that pickles and deep-copies.

    `MappingProxyType` does neither, so results could not leave a worker process, be cached or `torch.save`d.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] = MappingProxyType({}), /, **changes: Any) -> None:  # noqa: ANN401
        """Copy `data`, with `changes` set."""
        self._data = {**data, **changes}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401 - an option value
        """The value of `key`."""
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """The keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of entries."""
        return len(self._data)

    def __repr__(self) -> str:
        """As the dict's."""
        return repr(self._data)

    def __reduce__(self) -> tuple[type[ReadOnlyDict], tuple[dict[str, Any]]]:
        """Pickle / copy as the class around a plain dict."""
        return type(self), (self._data,)


@dataclass(frozen=True)
class Backend:
    """What actually ran.

    Attributes:
        mode: `"portable"` (plain PyTorch) or `"kernels"` (custom kernels).
        device: Device string, e.g. `"cuda:0"`.
        dtype: Main compute dtype.
        kernels: Kernel libraries used by `kernels`, as `name@build:target`. A library may hand single geometries it
            does not support to PyTorch; those are counted in `fastforks._core.kernels.fallbacks()`.
        checkpoint: The user's weight files of `load(weights=...)` as `path:size:mtime_ns`, in the variant's order;
            empty for the variant's recorded weights. Results and embeddings of two checkpoints compare unequal.
        library: The fastforks version that produced the result (provenance: not part of `==` or `repr`).
        weights: SHA-256 of the variant's recorded weight files, in load order; empty with `load(weights=...)`, whose
            files `checkpoint` names (provenance: not part of `==` or `repr`). Kernel versions are in `kernels`.
    """

    mode: Literal["portable", "kernels"]
    device: str
    dtype: Dtype
    kernels: tuple[str, ...] = ()
    checkpoint: tuple[str, ...] = ()
    library: str = field(default="", compare=False, repr=False)
    weights: tuple[str, ...] = field(default=(), compare=False, repr=False)


torch.serialization.add_safe_globals([ReadOnlyDict, Backend])  # results `torch.save`d by users load with weights_only

# Calls on CUDA (and MPS) devices run one at a time per process: kernels runners (and some portable ones) set
# process-global torch state around their GEMMs and attention (`torch.backends.cuda.matmul.allow_fp16_accumulation`, the
# SDPA backend switches of `torch.nn.attention.sdpa_kernel`); two calls in parallel threads would see each other's
# settings and return different outputs (FoundationStereo kernels: up to ~1 px). Apple MPS calls take the same lock:
# PyTorch's MPS backend does not encode commands from several threads safely, and `mps._quiet_nd_padding` uses the
# process-global `warnings.catch_warnings()`. Reentrant: a model may call another one.
_GPU_CALLS = threading.RLock()


@contextlib.contextmanager
def _deterministic_cudnn() -> Generator[None]:
    """`torch.backends.cudnn.deterministic = True` for one CUDA call (restored after; under `_GPU_CALLS`).

    For runners with `deterministic_cudnn`: cuDNN's default transposed-conv algorithms accumulate with atomics, and
    the GRU / cost-volume families amplify the last-bit differences (portable IGEV++ fp32 at Middlebury-H: up to 16 px
    max, 0.0017 px mean between two calls on the same input).
    """
    old = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = old


_TORCH_DIR = str(Path(torch.__file__).parent) + os.sep  # frames of torch itself (`_drop_trace_frames`)


def _drop_trace_frames(error: BaseException) -> None:
    """Free the torch frames of a failed compiled call now, keeping torch's dispatch key set as it is.

    AOT autograd holds python dispatcher guards in frame locals. Freed later with the traceback (a caught error, a
    notebook's last exception), they set the PythonDispatcher key again: every later op of the process raises
    "PythonDispatcherTLS was not set" (torch 2.14). Clearing torch's frames under a key-set guard runs them here
    instead; the traceback keeps its lines, and user frames keep their locals.
    """
    seen: set[int] = set()
    todo: list[BaseException | None] = [error]
    with torch._C._PreserveDispatchKeyGuard():  # noqa: SLF001
        while todo:
            e = todo.pop()
            if e is None or id(e) in seen:
                continue
            seen.add(id(e))
            tb = e.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code.co_filename.startswith(_TORCH_DIR):
                    with contextlib.suppress(RuntimeError):  # a frame still running
                        tb.tb_frame.clear()
                tb = tb.tb_next
            todo += [e.__cause__, e.__context__]


def _call_error(
    fn: Callable[..., Any], model: object, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> ArgumentError | None:
    """An `ArgumentError` (with a did-you-mean) if the arguments do not bind to `fn`'s signature, else `None`.

    Only called after a `TypeError`, so a successful call pays nothing for it.
    """
    return bind_error(fn, f"{type(model).__name__}(...)", (model, *args), kwargs, skip=1)


def _typo_checked(fn: Callable[..., Any]) -> Callable[..., Any]:
    """`fn` (a model's `__call__`) with a keyword typo reported as `ArgumentError`; nothing runs on the success path."""

    @functools.wraps(fn)
    def call(self: object, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - the signature of `fn`
        try:
            return fn(self, *args, **kwargs)
        except TypeError as e:
            if isinstance(e, ArgumentError):
                raise
            err = _call_error(fn, self, args, kwargs)
            if err is None:
                raise
            raise err from e

    return call


def _import_dynamo() -> None:
    """Import `torch._dynamo` on a short-lived thread the first time.

    Its first import leaves a frame cycle (`torch.fx.wrap` keeps `inspect.currentframe()`); on the calling thread
    that cycle pins the caller's frames, and so the model, until `gc.collect()`.
    """
    if "torch._dynamo" not in sys.modules:
        t = threading.Thread(target=importlib.import_module, args=("torch._dynamo",))
        t.start()
        t.join()  # an import error is raised again by the caller's own import
    while _AFTER_DYNAMO:
        with contextlib.suppress(IndexError):  # another thread took the last one
            _AFTER_DYNAMO.pop()()


_AFTER_DYNAMO: list[Callable[[], None]] = []  # `eager_in_compile` wrappers still to make, once dynamo is imported
_Fn = TypeVar("_Fn", bound="Callable[..., Any]")


def eager_in_compile(fn: _Fn) -> _Fn:
    """`fn`, run eagerly (untraced) when it is called inside a `Model.compile()` trace.

    `torch.compiler.disable` imports `torch._dynamo`, so the disabled function is not made here (at the import of
    the caller's module) but by `_import_dynamo`, which `compile()` and every kernels load call first. A trace that
    starts without it (a caller's own `torch.compile`) traces `fn`.
    """
    disabled: Callable[..., Any] | None = None

    def arm() -> None:
        nonlocal disabled
        disabled = torch.compiler.disable(fn)

    if "torch._dynamo" in sys.modules:
        arm()
    else:
        _AFTER_DYNAMO.append(arm)

    @functools.wraps(fn)
    def call(*a: Any, **k: Any) -> Any:  # noqa: ANN401 - the wrapped function's own signature
        return disabled(*a, **k) if disabled is not None and torch.compiler.is_compiling() else fn(*a, **k)

    return cast("_Fn", call)


def _kernels_entry_points(runner: Runner) -> None:
    """Wrap the kernels runner's entry points: deferred output checks, and a clean graph break in `torch.compile`.

    Each call runs under `checked` (the overflow fallback of `fastbb.Flash64`: one sync per call while a
    launcher probes, else none). Dynamo would otherwise trace into the kernel launchers (RecursionError, then a silent
    eager run); outside a trace the check costs one `is_compiling()` call.
    """
    _import_dynamo()  # `torch.compiler.disable` imports it
    base = type(runner)

    def opaque(name: str) -> Callable[..., Any]:
        eager = getattr(base, name)

        @functools.wraps(eager)
        def run(self: Any, *a: Any, **k: Any) -> Any:  # noqa: ANN401 - the runner's bound method
            return checked(eager, self, *a, **k)

        disabled = torch.compiler.disable(run)

        @functools.wraps(eager)
        def call(self: Any, *a: Any, **k: Any) -> Any:  # noqa: ANN401 - the runner's bound method
            return disabled(self, *a, **k) if torch.compiler.is_compiling() else run(self, *a, **k)

        return call

    names = {"__module__": base.__module__, "__qualname__": base.__qualname__}
    runner.__class__ = type(base.__name__, (base,), names | {name: opaque(name) for name in base.compiled_methods})


class Model:
    """Base of the loaded-model classes of all tasks (`StereoModel`, `DepthModel`, ...).

    Not an `nn.Module`: calls take images of any supported type and size and handle the task's pre- and
    post-processing; the task class's `module()` is the model as an `nn.Module` for `torch.compile` / `torch.export`.

    An instance is not reentrant: calls from several threads are serialised. Calls on CUDA devices are serialised
    across all models of the process (they set process-global torch flags); CPU calls of different instances run in
    parallel.

    Attributes:
        card: The model family's card.
        variant: The loaded variant.
        backend: The backend every call runs on.
        options: Load options that differ from the validated defaults.
        device: The device the model runs on.
        weights: The user's files of `load(weights=...)` (not hash-checked; the golden parity does not apply), or
            `None` for the variant's recorded weights.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Wrap a `__call__` defined by a subclass (see `_typo_checked`)."""
        super().__init_subclass__(**kwargs)
        if (fn := cls.__dict__.get("__call__")) is not None:
            cls.__call__ = _typo_checked(fn)  # ty: ignore[invalid-assignment]

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load_model`)."""
        self.card = runner.ctx.card
        self.variant = runner.ctx.variant
        self.backend = backend
        self.options: Mapping[str, Any] = ReadOnlyDict(options)
        self.device = runner.ctx.device
        self.weights = runner.ctx.user_weights
        self._runner = runner
        self._lock = threading.Lock()
        self._pid = os.getpid()  # loaded in this process; see `_warn_if_forked`
        self._fork_warned = False
        self._min_size = 0  # the documented minimum (`size_range().reject_below`), read on the first call
        if backend.mode == "kernels":
            _kernels_entry_points(runner)

    def __repr__(self) -> str:
        """Model ref, backend and the user's weight files, if any."""
        mine = "" if self.weights is None else f", weights={', '.join(map(str, self.weights))!r}"
        return f"{type(self).__name__}({self.variant.ref!r}, {self.backend}{mine})"

    def __reduce__(self) -> tuple[Any, ...]:
        """Refuse to pickle (or copy): a model holds device memory, locks and compiled state; load one per process."""
        msg = (
            f"a fastforks model cannot be pickled or copied: load the model in each worker: "
            f"fastforks.{self.card.task}.load({self.variant.id!r}) (results pickle)"
        )
        raise TypeError(msg)

    def module(self) -> torch.nn.Module:
        """The network as an `nn.Module` for `torch.compile` / `torch.export` (inputs and outputs: the task's runner).

        Some submodules may be per-instance subclasses of their original class (fastforks overrides their `forward`
        without a reference cycle): `isinstance`, `state_dict`, `torch.save` and export work as usual, but exact type
        checks (`type(m) is nn.Conv2d`, dicts keyed by `type(m)`) need `fastforks.base_type(m)`.

        Raises:
            BackendUnavailableError: The backend has no exportable module (e.g. `kernels`).
        """
        try:
            return self._runner.module()
        except NotImplementedError as e:
            msg = f"{self.variant.ref} ({self.backend.mode}) has no exportable module: {e}"
            raise BackendUnavailableError(msg) from e

    def compile(self, **kwargs: Any) -> Self:  # noqa: ANN401 - `torch.compile`'s keyword arguments
        """Compile the portable network in place with `torch.compile`; later calls keep the normal high-level API.

        The runner's entry points (`Runner.compiled_methods`: the network call with its precision and, for stereo,
        padding and cropping; segmentation's labels-only call; the multi-view per-scene steps; promptable
        segmentation's image encoder and prompt decoder) are compiled; reading the images and building the result stay
        eager. The first call of each input size compiles (seconds to minutes); other sizes recompile. Like
        `nn.Module.compile`, it returns the same object, and a second `compile()` replaces the first (new arguments).

        If the first call of an entry point fails in `torch.compile` (a dynamo / inductor error, e.g. an unsupported
        op or no C compiler), the model is eager again and the error is re-raised with a note saying so; errors of
        later calls (another input size) are raised as they are, and the model stays compiled.

        Args:
            **kwargs: Passed to `torch.compile` (`fullgraph`, `mode`, `dynamic`, `backend`, ...).

        Returns:
            `self`.

        Raises:
            BackendUnavailableError: The backend has no exportable module (`kernels`).
        """
        if self.backend.mode != "portable":
            msg = f"{self.variant.ref} ({self.backend.mode}): only the portable backend compiles (kernels are opaque)"
            raise BackendUnavailableError(msg)
        _import_dynamo()
        from torch._dynamo.exc import TorchDynamoException  # noqa: PLC0415 - dynamo is slow to import

        with self._lock:  # no call runs during the swap; two compile() calls do not both wrap
            runner = self._runner
            base: type[Runner] = getattr(runner, "_ff_eager", type(runner))  # a repeat compile() replaces the first
            runner.__class__ = base
            self.module().requires_grad_(requires_grad=False)  # inference only; raises if there is no module
            ref = self.variant.ref
            compiled = {name: torch.compile(getattr(runner, name), **kwargs) for name in base.compiled_methods}
            untried = set(compiled)  # entry points whose first compiled call has not succeeded yet

            def forward(name: str) -> Callable[..., Any]:
                eager = getattr(base, name)

                def call(self: Any, *a: Any, **k: Any) -> Any:  # noqa: ANN401 - the compiled bound method
                    if torch.compiler.is_compiling():  # called from another compiled entry point: trace it inline
                        return eager(self, *a, **k)
                    try:
                        out = compiled[name](*a, **k)
                    except TorchDynamoException as e:
                        _drop_trace_frames(e)
                        if name in untried:
                            self.__class__ = base  # under the model lock: calls run in `_session`
                            e.add_note(f"fastforks: {ref} runs eager again; compile() retries with other arguments")
                        raise
                    except BaseException as e:
                        _drop_trace_frames(e)
                        raise
                    untried.discard(name)
                    return out

                return call

            namespace = {name: forward(name) for name in compiled} | {"_ff_eager": base}
            runner.__class__ = type(base.__name__, (base,), namespace)
        return self

    def export_size(self) -> tuple[int, int] | None:
        """Default `(height, width)` of `export_onnx` / `export_tensorrt`, or `None` if the task has none.

        Classification (the card's crop), features (224) and promptable segmentation (the encoder side: 1024, SAM 3
        1008) have one; stereo, depth, segmentation and detection do not: give a size inside `size_range()`.
        """
        side = size_range(self.variant, self.card).encoder if self.card.task == "promptable_segmentation" else 0
        return (side, side) if side else spec(self.card.task).export_size

    def size_range(self) -> SizeRange:
        """Input sizes this model accepts: min size, pad multiple, fixed size, and the largest size measured."""
        return size_range(self.variant, self.card)

    def release_graphs(self) -> None:
        """Free the CUDA graphs a kernels runner keeps for small input shapes (RT-MonSter++, DINOv3); else a no-op."""
        release = getattr(self._runner, "release_graphs", None)
        if release is not None:
            release()

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """Example inputs of `module()` at this size on the model's device, as the exports trace and check them.

        Synthetic textured RGB float32 `[batch, 3, height, width]` in `[0, 1]`, one per `TaskSpec.inputs` (stereo: a
        pair with an 8 px disparity; the pixels `fastforks bench` uses). Tasks whose `module()` takes transformed inputs
        override this.

        Args:
            height: Image height in pixels; default (with `width`): `export_size()` where the task has one.
            width: Image width in pixels.
            batch: Batch size.

        Returns:
            One tensor per input of `module()`.

        Raises:
            InputError: Only one of `height` and `width` is given, or the task has no default size.
        """
        from fastforks._core.bench import synthetic_image, synthetic_pair  # noqa: PLC0415 - not needed for inference
        from fastforks._core.export import resolve_size  # noqa: PLC0415

        height, width = resolve_size(self, height, width)

        names = spec(self.card.task).inputs
        if names in {("left", "right"), ("image1", "image2")}:  # image pairs: a textured, shifted pair
            views = synthetic_pair(height, width, self.device)
        else:
            views = (synthetic_image(height, width, self.device),) * len(names)
        return tuple(v.expand(batch, -1, -1, -1).contiguous() for v in views)

    def export_onnx(
        self,
        path: str | Path,
        height: int | None = None,
        width: int | None = None,
        *,
        batch: int = 1,
        size_range: SizeBounds | None = None,
    ) -> Path:
        """Write the portable model (`module()`) at a fixed size to ONNX; needs `pip install "fastforks-core[onnx]"`.

        Inputs (named as in `TaskSpec.inputs`; stereo `left`, `right`, else `image`) are `[batch, 3, height, width]`
        float32 as `module()` takes them; the output is named after the variant's main output (`disparity`,
        `inverse_depth`, `logits`, ...). Multi-view: `images` `[batch = views, 3, height, width]` plus the camera inputs
        the model requires, in `TaskSpec.inputs` order; one output per map of `module()`, named as its keys (fixed size
        only). The graph keeps this model's compute dtype (`load(dtype=...)`); export on the device you will run on.
        Graphs above 2 GB keep their weights in `<path>.data` next to the file.

        Args:
            path: ONNX file to write.
            height: Input height of the exported graph (fixed); default: `export_size()`; required if the task has none.
            width: Input width (fixed); default: `export_size()`; required if the task has none.
            batch: Batch size (fixed).
            size_range: `((min_h, min_w), (max_h, max_w))`: make height and width dynamic axes within these bounds
                (traced at `(height, width)`); default: the fixed size.

        Returns:
            `path`.

        Raises:
            BackendUnavailableError: The model runs `kernels` (custom kernels cannot be exported), or `onnx` is missing.
            InputError: The size is below the model's minimum, or only one of height / width is given.
        """
        from fastforks._core.export import export_onnx  # noqa: PLC0415 - optional dependencies

        return export_onnx(self, path, height, width, batch=batch, size_range=size_range)

    def export_tensorrt(
        self,
        path: str | Path,
        height: int | None = None,
        width: int | None = None,
        *,
        batch: int = 1,
        size_range: SizeBounds | None = None,
    ) -> Path:
        """`export_onnx` plus a TensorRT engine build for this GPU; run it with `TensorRTEngine(path)`.

        Needs `pip install "fastforks-core[onnx,tensorrt]"`. The engine only runs on this GPU model and TensorRT
        version.

        Args:
            path: Engine file to write.
            height: Input height of the engine (fixed); default: `export_size()`; required if the task has none.
            width: Input width (fixed); default: `export_size()`; required if the task has none.
            batch: Batch size (fixed).
            size_range: `((min_h, min_w), (max_h, max_w))`: a dynamic-shape engine with an optimization profile (min,
                opt = `(height, width)`, max); not faster than a fixed engine, see export.md.

        Returns:
            `path`.

        Raises:
            BackendUnavailableError: The model runs `kernels`, `onnx` / `tensorrt` is missing, or the built engine
                outputs NaN everywhere (usually a TensorRT bug; the engine is removed; use `export_onnx` + onnxruntime).
            InputError: The size is below the model's minimum.
            RuntimeError: TensorRT cannot parse or build the graph.
        """
        from fastforks._core.export import export_tensorrt  # noqa: PLC0415 - optional dependencies

        return export_tensorrt(self, path, height, width, batch=batch, size_range=size_range)

    @contextlib.contextmanager
    def _session(self) -> Generator[None]:
        """Locks, CUDA device guard, inference mode, no inherited autocast, deterministic cuDNN if the runner asks.

        `Runner.deterministic_cudnn` asks for deterministic cuDNN.
        """
        if _FORKED:
            self._warn_if_forked()
        cuda = self.device.type == "cuda"
        on_device = torch.cuda.device(self.device) if cuda else contextlib.nullcontext()
        if cuda:
            warmup.wait(self.device)
        det = _deterministic_cudnn() if cuda and self._runner.deterministic_cudnn else contextlib.nullcontext()
        with (
            _GPU_CALLS if cuda or self.device.type == "mps" else contextlib.nullcontext(),
            self._lock,
            on_device,
            det,
            torch.inference_mode(),
            _no_autocast(self.device),
        ):
            begin_call()
            yield

    def _warn_if_forked(self) -> None:
        """Once per model: it was loaded in the parent of this `fork()`ed process (TF-3)."""
        if self._pid == os.getpid() or self._fork_warned:
            return
        self._fork_warned = True
        if self.device.type == "cpu" and torch.get_num_threads() > 1:
            logging.getLogger(f"fastforks.{self.card.task}").warning(
                "%s was loaded before this process was forked: torch's CPU thread pool can deadlock in the child "
                "(libgomp does not survive fork). Load the model inside each worker, start workers with 'spawn', or "
                "call torch.set_num_threads(1) first",
                self.variant.ref,
            )
        elif self.device.type != "cpu":
            logging.getLogger(f"fastforks.{self.card.task}").warning(
                "%s was loaded before this process was forked: a CUDA / MPS context cannot be used in a forked child. "
                "Load the model inside each worker and start workers with 'spawn'",
                self.variant.ref,
            )

    def _check_size(self, x: torch.Tensor) -> None:
        """Raise `InputError` if `x` (`[..., H, W]`) is below the shortest side the card / size probe allow.

        Thin non-square inputs between that and the pad multiple (48x640 for a multiple of 64) are padded, not rejected.
        """
        if not self._min_size:
            r = self.size_range()
            self._min_size = r.reject_below or r.min_size
        if min(x.shape[-2:]) < self._min_size:
            msg = (
                f"{self.variant.ref} needs images of at least {self._min_size} px (H and W), got "
                f"{tuple(x.shape[-2:])}; resize or pad the input (size_range() says what is accepted)"
            )
            raise InputError(msg)

    def _batched(
        self, fn: Callable[..., dict[str, torch.Tensor]], *inputs: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """`fn` on chunks of at most `runner.batch` inputs (dim 0; `None` stays `None`) in one session, concatenated."""
        n = next(x for x in inputs if x is not None).shape[0]
        step = self._runner.batch or n
        with self._session():
            outs = [fn(*(None if x is None else x[i : i + step] for x in inputs)) for i in range(0, n, step)]
        return {k: torch.cat([o[k] for o in outs]) if len(outs) > 1 else outs[0][k] for k in outs[0]}

    def _check_outputs(
        self, out: Mapping[str, torch.Tensor], shapes: Mapping[str, tuple[int, ...]], *, cast: bool = True
    ) -> dict[str, torch.Tensor]:
        """Check the runner's maps against the card and the task spec; cast them to the result dtypes.

        Args:
            out: What the runner returned.
            shapes: Expected shape of every map in `out`.
            cast: Return the maps outside the spec's `bool_outputs` as float32 (no copy if they already are); `False`
                returns them unchanged, for callers that reduce a large map first (segmentation logits).

        Returns:
            The maps, cast as asked.

        Raises:
            CardError: The maps are not the card's, miss a required one, have a wrong shape, or are `torch.bool`
                exactly when not in the spec's `bool_outputs` (a runner, card or spec bug, not a user error).
        """
        task = spec(self.card.task)
        outputs, required = self.variant.outputs, task.required
        if not set(out) <= set(outputs) or not required & out.keys():
            msg = (
                f"{self.variant.ref}: runner returned {sorted(out)}; card outputs are {list(outputs)}, at least one of"
                f" {sorted(required)} is required"
            )
            raise CardError(msg)
        for k, v in out.items():
            want = shapes[k]
            if tuple(v.shape) != want:
                msg = f"{self.variant.ref}: output {k!r} has shape {tuple(v.shape)}, expected {want}"
                raise CardError(msg)
            if (k in task.bool_outputs) != (v.dtype == torch.bool):
                want_dtype = "torch.bool" if k in task.bool_outputs else "numbers (torch.bool only for masks)"
                msg = f"{self.variant.ref}: runner returned a {v.dtype} {k}; {k} maps must be {want_dtype}"
                raise CardError(msg)
        if not cast:
            return dict(out)
        return {k: v if k in task.bool_outputs else v.float() for k, v in out.items()}


M = TypeVar("M", bound=Model)


def list_models(  # noqa: PLR0913 - the filters of every task's `list_models`
    task: str,
    filter: str | Sequence[str] = "*",  # noqa: A002 - mirrors timm.list_models
    *,
    exclude_filters: str | Sequence[str] = (),
    commercial: Commercial | None = None,
    tier: Tier | None = None,
    family: str | None = None,
    runnable: bool | None = None,
) -> list[str]:
    """Sorted ids of the installed models of `task`; see `fastforks.list_models` for the arguments.

    Raises:
        OptionsError: `commercial` or `tier` is not one of its allowed values.
    """
    return registry.list_models(
        (task,),
        filter,
        exclude_filters=exclude_filters,
        commercial=commercial,
        tier=tier,
        family=family,
        runnable=runnable,
    )


def model_card(task: str, model: str) -> ModelCard:
    """Card of a model id of `task` (`"s2m2-s"` or `"s2m2-s@1"`).

    `io`, `outputs` and `output_info` are those of the id's variant (e.g. `kind = "affine_invariant"` for
    `da3mono-large`, whose family card says `"metric"`) and `variant_ref` is the resolved `"<id>@<revision>"`; the
    other fields are the family's. `card.contract()` adds the resolved inputs, outputs, options, modes and export
    stages without loading weights.

    Raises:
        ModelNotFoundError: Unknown id or revision.
    """
    return registry.variant_card(*registry.resolve(task, model))


_AMPERE = 8  # first GPU generation (compute capability major) with bfloat16 hardware


@functools.cache
def _auto_portable() -> frozenset[str]:
    """Model ids for which `auto` stays on `portable`: the kernels measured no faster (`tools/gen_auto_mode.py`)."""
    data = json.loads(Path(__file__).with_name("auto_portable.json").read_text())
    return frozenset(data["portable"])


def dtype_name(dtype: str | torch.dtype | None) -> str | None:
    """`"float16"` for `torch.float16` (any `torch.dtype` -> its name); anything else unchanged (validated later)."""
    return str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else dtype


def _check_args(mode: str, dtype: str | torch.dtype | None) -> tuple[bool, Dtype | None]:
    """Validate `mode` and `dtype` (a string or the torch dtype); returns `(fp32, reduced dtype asked for)`."""
    if mode not in get_args(ModeArg):
        msg = f"mode must be 'portable', 'kernels' or 'auto', got {mode!r}"
        raise OptionsError(msg)
    name = dtype_name(dtype)
    if name not in {None, "float32", "float16", "bfloat16"}:
        msg = f"dtype must be 'float32', 'float16', 'bfloat16' (or the torch dtype) or None, got {dtype!r}"
        raise OptionsError(msg)
    fp32 = name == "float32"
    return fp32, None if fp32 else cast("Dtype", name)


def _no_auto_kernels(
    card: ModelCard, variant: Variant, dev: torch.device, *, fp32: bool, want: Dtype | None
) -> str | None:
    """Why `mode='auto'` skips kernels for `variant` on `dev`, or None to try it."""
    kernels_dtype = card.kernels_dtype_on(dev)
    if "kernels" not in card.modes:
        return "no kernels backend"
    if fp32 and kernels_dtype != "float32":
        return "dtype='float32'"
    if dev.type == "mps" and not card.mps_kernel_libs:
        return "no Metal kernels for Apple MPS"
    if want not in {None, kernels_dtype}:
        return f"kernels computes in {kernels_dtype}, not {want}"
    if dev.type != "mps" and variant.id in _auto_portable():  # measured on CUDA
        return "kernels are not faster than portable on the measured inputs"
    return None


def load_model(  # noqa: PLR0913 - the arguments of every task's `load`
    task: str,
    model: str,
    cls: type[M],
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
) -> M:
    """Load `model` of `task` as a `cls`; see the task's `load` for the arguments and errors."""
    log = logging.getLogger(f"fastforks.{task}")
    card, variant = registry.resolve(task, model)
    registry.require_commercial(card, variant)
    registry.require(card, variant.ref)
    fp32, want = _check_args(mode, dtype)  # every argument is checked before `weights` logs its warning
    dev = resolve_device(device, card)
    opts = check_options(card, {} if options is None else options, variant.ref)
    files = None if weights is None else user_weights(variant, weights, log)
    if fp32 and mode == "kernels" and card.kernels_dtype_on(dev) != "float32":
        msg = "kernels runs reduced-precision kernels here; use mode='portable' with dtype='float32'"
        raise OptionsError(msg)
    if mode == "auto":
        reason = _no_auto_kernels(card, variant, dev, fp32=fp32, want=want)
        if reason is None:
            try:
                m = _build(cls, card, variant, mode="kernels", device=dev, options=opts, want=want, weights=files)
            except BackendUnavailableError as e:
                reason = f"kernels unavailable ({e})"
            else:
                log.info("%s: using kernels", variant.ref)
                return m
        # a kernels that failed on a GPU it supports is a WARNING; by design (fp32, no kernels, other dtype, CPU) INFO
        failed = reason.startswith("kernels unavailable") and dev.type in {"cuda", "mps"}
        level = logging.WARNING if failed else logging.INFO
        log.log(level, "%s: %s; using portable", variant.ref, reason)
        mode = "portable"
    if mode not in card.modes:
        msg = f"{variant.ref} has no {mode!r} backend (has: {', '.join(card.modes)})"
        if mode == "kernels" and card.no_kernels_reason:
            msg += f": {card.no_kernels_reason}"
        raise BackendUnavailableError(msg)
    return _build(cls, card, variant, mode=mode, device=dev, options=opts, fp32=fp32, want=want, weights=files)


def user_weights(variant: Variant, path: str | os.PathLike[str], log: logging.Logger | None = None) -> tuple[Path, ...]:
    """The files of `load(weights=path)` for `variant`, one per weight file of the variant, in its order.

    A variant with one weight file takes that file, or a directory holding it under its recorded name; one with
    several takes a directory holding each as `<dir>/<name>` (names may have sub-directories, e.g. a diffusers
    layout). Weights that ship inside an archive are given as the extracted file. Nothing is hash-checked.

    Args:
        variant: The variant to load.
        path: The user's file or directory.
        log: Gets one WARNING that the files are unverified.

    Raises:
        WeightsError: `path` does not exist, or does not hold the files the variant needs.
    """
    p = Path(os.fspath(path)).expanduser()
    names = [s.name for s in variant.weights]
    if not p.exists():
        msg = f"weights={str(path)!r}: no such file or directory"
        raise WeightsError(msg)
    if p.is_file():
        if len(names) == 1:
            return _unverified((p,), variant, log)
        msg = (
            f"weights={str(path)!r} is a file, but {variant.ref} has {len(names)} weight files; pass a directory"
            f" holding them as {', '.join(f'<dir>/{n}' for n in names)}"
        )
        raise WeightsError(msg)
    files = tuple(p / n for n in names)
    missing = [n for n, f in zip(names, files, strict=True) if not f.is_file()]
    if missing:
        hint = " (or pass the file itself)" if len(names) == 1 else ""
        msg = f"weights={str(path)!r}: {variant.ref} needs {', '.join(missing)} in this directory{hint}"
        raise WeightsError(msg)
    return _unverified(files, variant, log)


def _unverified(files: tuple[Path, ...], variant: Variant, log: logging.Logger | None) -> tuple[Path, ...]:
    if log is not None:
        log.warning(
            "%s: unverified weights from %s (not hash-checked; the golden parity of %s does not apply)",
            variant.ref,
            ", ".join(map(str, files)),
            variant.ref,
        )
    return files


def resolve_device(device: str | torch.device | None, card: ModelCard | None = None) -> torch.device:
    """The device for `load`, with an index for CUDA; an unknown or unusable device is an `OptionsError`.

    The default is CUDA if available, else Apple MPS for a family validated on it (`card.portable_mps_dtypes`),
    else the CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            return cuda_device(torch.device("cuda"))
        on_mps = card is not None and bool(card.portable_mps_dtypes) and torch.backends.mps.is_available()
        return torch.device("mps" if on_mps else "cpu")
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as e:
        msg = f"invalid device {device!r} (e.g. 'cpu', 'cuda', 'cuda:1', 'mps'): {e}"
        raise OptionsError(msg) from e
    if dev.type == "cuda":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count == 0:
            msg = (
                f"device {device!r}: torch {torch.__version__} has no usable CUDA GPU"
                " (install a CUDA build of PyTorch, or use device='cpu')"
            )
            raise OptionsError(msg)
        if dev.index is not None and dev.index >= count:
            msg = f"device {device!r}: this machine has {count} CUDA device(s)"
            raise OptionsError(msg)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        msg = f"device {device!r}: MPS is not available in this torch build / on this machine"
        raise OptionsError(msg)
    return cuda_device(dev)


def claim_kernels_device(device: torch.device) -> None:
    """Kernel handles and buffers of `kernels` are process-wide: allow one CUDA device per process."""
    with _KERNELS_DEVICE_LOCK:
        if not _KERNELS_DEVICE:
            _KERNELS_DEVICE.append(device)
        elif _KERNELS_DEVICE[0] != device:
            msg = (
                f"kernels already runs on {_KERNELS_DEVICE[0]} in this process; use one process per GPU (got {device})"
            )
            raise BackendUnavailableError(msg)


def check_options(card: ModelCard, options: Mapping[str, Any], model: str | None = None) -> dict[str, Any]:
    """The options that differ from the card defaults; raises on unknown keys, wrong types and bad values."""
    if not isinstance(options, Mapping):
        msg = f"options must be a dict such as {{'iters': 8}}, got {type(options).__name__}"
        raise OptionsError(msg)
    unknown = set(options) - set(card.options) - set(card.option_aliases)
    if unknown:
        hints = []
        for k in sorted(unknown, key=str):
            close = difflib.get_close_matches(str(k), [*card.options, *card.option_aliases], n=1)
            hints.append(f"{k!r} (did you mean {card.option_aliases.get(close[0], close[0])!r}?)" if close else repr(k))
        msg = (
            f"unknown option(s) {', '.join(hints)} for {model or card.family}; "
            f"valid options: {', '.join(card.options) or 'none'} (defaults {dict(card.options)})"
        )
        raise OptionsError(msg)
    renamed = {card.option_aliases.get(k, k): v for k, v in options.items()}
    if len(renamed) < len(options):
        both = sorted(k for k in options if k in card.option_aliases and card.option_aliases[k] in options)
        msg = f"option(s) {both} given twice: each is an alias of another given option ({model or card.family})"
        raise OptionsError(msg)
    options = renamed
    for k, v in options.items():
        default = card.options[k]
        int_for_float = isinstance(default, float) and type(v) is int  # exact type: not a module (no bool)
        if type(v) is not type(default) and not int_for_float:  # exact type: not a module
            msg = f"option {k!r} must be {type(default).__name__}, got {type(v).__name__}"
            raise OptionsError(msg)
        if isinstance(v, float) and not math.isfinite(v):  # NaN passes every `<` check
            msg = f"option {k!r} must be finite, got {v}"
            raise OptionsError(msg)
        if k in card.option_choices and v not in card.option_choices[k]:
            msg = f"option {k!r} must be {card.option_bounds(k)}, got {v!r}"
            raise OptionsError(msg)
        if (k in card.option_min and v < card.option_min[k]) or (k in card.option_max and v > card.option_max[k]):
            msg = f"option {k!r} must be {card.option_bounds(k)}, got {v}"
            raise OptionsError(msg)
    return {k: v for k, v in options.items() if v != card.options[k]}


def make_params(cls: type[Any], card: ModelCard, variant: Variant, options: Mapping[str, Any]) -> object:
    """The runner's `Params` from the card `[params]`, the variant's params, the option defaults and `options`."""
    values = {**card.params, **variant.params, **card.options, **options}
    names = {f.name for f in dataclasses.fields(cls)}
    if set(values) != names:
        msg = f"{card.family}: card params / options {sorted(values)} do not match {cls.__qualname__} {sorted(names)}"
        raise CardError(msg)
    return cls(**values)


def compute_dtype(  # noqa: PLR0913 - the resolved arguments of `_build`
    card: ModelCard,
    runner: type[Runner],
    mode: Mode,
    device: torch.device,
    *,
    fp32: bool = False,
    want: Dtype | None = None,
) -> Dtype:
    """Main compute dtype: the card's for kernels; for portable the card's validated CUDA dtype, else float32.

    `want` (`load(dtype=)`, never float32: that is `fp32`) must be what the backend implements.

    Raises:
        OptionsError: `want` is not available for this mode and device.
    """
    if want is not None:
        if mode == "kernels":
            if want != card.kernels_dtype_on(device):
                msg = f"{card.family} kernels computes in {card.kernels_dtype_on(device)}, not {want}"
                raise OptionsError(msg)
            return want
        have = {"cuda": runner.dtypes, "mps": card.portable_mps_dtypes}.get(device.type, ())
        if want not in have:
            where = f"portable on {device.type}" if device.type != "cuda" else "portable"
            msg = f"{card.family} {where} cannot compute in {want}; dtype must be one of {list(have) or ['float32']}"
            raise OptionsError(msg)
        if device.type == "cuda" and want != card.portable_cuda_dtype:
            logging.getLogger("fastforks").warning(
                "%s: portable %s is not the validated dtype (%s)", card.family, want, card.portable_cuda_dtype
            )
        return want
    if mode == "kernels":
        return card.kernels_dtype_on(device)
    if device.type != "cuda" or fp32:
        return "float32"
    return _validated_cuda_dtype(card, runner, device)


def _validated_cuda_dtype(card: ModelCard, runner: type[Runner], device: torch.device) -> Dtype:
    """The card's portable CUDA dtype; float32 instead of bfloat16 on GPUs older than Ampere (no bf16 hardware)."""
    if card.portable_cuda_dtype not in runner.dtypes:
        msg = f"{card.family}: portable cuda_dtype {card.portable_cuda_dtype!r} not implemented ({list(runner.dtypes)})"
        raise CardError(msg)
    if card.portable_cuda_dtype == "bfloat16" and torch.cuda.get_device_capability(device)[0] < _AMPERE:
        # torch only emulates bf16 there, slowly; float16 overflows for these models
        logging.getLogger("fastforks").info(
            "%s: %s has no bfloat16 hardware; computing in float32", card.family, torch.cuda.get_device_name(device)
        )
        return "float32"
    return card.portable_cuda_dtype


def _no_autocast(device: torch.device) -> contextlib.AbstractContextManager:
    """Turn off the caller's autocast on the CPU and on `device` (if on), a no-op context when none is on (usual).

    The results and their reported precision are those without it; runners that use autocast enter their own.
    """
    on = [t for t in dict.fromkeys(("cpu", device.type)) if torch.is_autocast_enabled(t)]
    if not on:
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    for t in on:
        stack.enter_context(torch.autocast(t, enabled=False))
    return stack


def runner_class(card: ModelCard, mode: Mode, device: torch.device | None = None) -> type[Runner]:
    """The family's runner for `mode` on `device` (imports the model code); kernels on MPS: `RUNNERS["kernels_mps"]`."""
    key = "kernels_mps" if mode == "kernels" and device is not None and device.type == "mps" else mode
    return registry.import_model(card).RUNNERS[key]


def _mismatch(e: Exception, files: tuple[Path, ...], ref: str) -> CheckpointMismatchError | None:
    """`e` as a `CheckpointMismatchError` naming the user's `files` if it says their state dict does not fit `ref`.

    Only `load_strict`'s and a strict `Module.load_state_dict`'s errors qualify; files that cannot be read already
    raise a `WeightsError` naming them (`Context.state_dict`), and any other error (CUDA, a kernel) is not
    about the weights.
    """
    what = f"weights={', '.join(map(str, files))} for {ref}"
    if isinstance(e, KeyError):  # a family that reads a tensor by name from the user's file
        return CheckpointMismatchError(what, f"the checkpoint has no {e.args[0] if e.args else '?'!r}")
    if isinstance(e, CheckpointMismatchError):
        return CheckpointMismatchError(what, e.details)
    if (keys := torch_mismatch(e)) is not None:
        return CheckpointMismatchError(what, mismatch_details(*keys))
    return None


def _kernel_tags(card: ModelCard, variant: Variant, device: torch.device) -> tuple[str, ...]:
    """Load the kernel libraries of the kernels backend on `device`; their tags."""
    libs = card.kernel_libs_on(device)
    if not libs:  # a family with CUDA kernels only, on MPS
        msg = f"{variant.ref} (mode='kernels'): no Metal kernels for Apple MPS; use mode='portable'"
        raise BackendUnavailableError(msg)
    try:
        return tuple(importlib.import_module(f"fastforks.kernels.{n}").load(device).tag for n in libs)
    except KernelUnavailableError as e:
        raise KernelUnavailableError(e.library, e.reason, model=variant.ref) from e


def adapt_to_device(runner: Runner, mode: Mode, device: torch.device) -> None:
    """Device-specific changes after a build: on Apple MPS, `mps.adapt` on every `runner.networks()` (portable)."""
    if device.type == "mps":
        mps.sync_at_exit()
        if mode == "portable":
            for net in runner.networks():
                mps.adapt(net)


def _build(  # noqa: PLR0913 - the resolved arguments of `load_model`
    cls: type[M],
    card: ModelCard,
    variant: Variant,
    *,
    mode: Mode,
    device: torch.device,
    options: Mapping[str, Any],
    fp32: bool = False,
    want: Dtype | None = None,
    weights: tuple[Path, ...] | None = None,
) -> M:
    if device.type == "cuda":
        warmup.start(device)  # CUDA library start-up overlaps with the build below
    # Kernel libraries are checked before any weights are fetched, so `kernels` fails fast.
    tags = _kernel_tags(card, variant, device) if mode == "kernels" else ()
    rcls = runner_class(card, mode, device)
    dtype = compute_dtype(card, rcls, mode, device, fp32=fp32, want=want)
    ctx = Context(card, variant, mode, device, dtype, make_params(rcls.Params, card, variant, options), tags, weights)
    try:
        # the checkpoint overwrites the initial values: skip the random fill; the build runs at the requested dtype
        # whatever the caller's `torch.set_default_dtype`
        with deferred_init(), float32_default():
            runner = rcls(ctx)  # a failing build (e.g. missing weights) must not pin the process to this device
    except (RuntimeError, KeyError) as e:  # CheckpointMismatchError is a WeightsError, a RuntimeError
        error = None if weights is None else _mismatch(e, weights, variant.ref)
        if error is None:
            raise
        raise error from e
    if mode == "kernels" and device.type == "cuda":
        claim_kernels_device(device)
    adapt_to_device(runner, mode, device)
    backend = Backend(
        mode,
        str(device),
        dtype,
        tags,
        _checkpoint_id(weights),
        library=_library_version(),
        weights=() if weights else tuple(f.sha256 for f in variant.weights),
    )
    model = cls(runner, backend, options)
    _settle(model, card, mode, device)
    return model


def _library_version() -> str:
    from fastforks import __version__  # noqa: PLC0415 - the package imports this module

    return __version__


def _checkpoint_id(files: tuple[Path, ...] | None) -> tuple[str, ...]:
    """`Backend.checkpoint` of the user's weight files (a stat each: no hashing of multi-GB files)."""
    out = []
    for f in files or ():
        st = f.stat()
        out.append(f"{f.resolve()}:{st.st_size}:{st.st_mtime_ns}")
    return tuple(out)


def _settle(model: Model, card: ModelCard, mode: Mode, device: torch.device) -> None:
    """The device-side end of a build: the cuDNN plan workspace's owner, and the build's cached blocks freed."""
    if mode == "kernels" and "cudnn_hacc" in card.kernel_libs:
        from fastforks.kernels import cudnn_hacc  # noqa: PLC0415 - only kernels models

        cudnn_hacc.attach(model)  # frees the plan workspace with the last such model
    if device.type == "cuda":
        # The build's transient blocks (fp32 BN-fold probes that pick large cuDNN workspaces, checkpoint copies before
        # the cast) would stay cached and count in the reserved memory for good (regnety_160 3990 -> 250 MiB).
        torch.cuda.empty_cache()
