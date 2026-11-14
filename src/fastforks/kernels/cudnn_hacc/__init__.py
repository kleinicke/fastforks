"""fp16-accumulate convolutions through cuDNN's backend API.

PyTorch always asks cuDNN for fp32 accumulation; on GeForce cards that runs the tensor cores at half rate
(RTX 5060 Ti: ~47 vs ~80 TFLOPS). Here fp16 Conv2d/Conv3d (groups 1, any dilation) run as a cuDNN conv-forward graph
with HALF compute, channels-last in and out; fp16 ConvTranspose2d/3d (output padding 0, dilation 1) as its data
gradient; other geometries raise ValueError when their plan is built. The graphs go through cuDNN's backend API by
ctypes (`_backend`), on the `libcudnn.so.9` PyTorch loaded: no extra package.

Any input size, no per-size compilation or tuning: plans that need runtime compilation (NVRTC) are skipped and the
first precompiled plan in cuDNN's heuristic order (mode A, then FALLBACK) is used (no timing; the order the cuDNN
frontend's `create_execution_plans` gives). Engines 0 (reference, very slow), 8 (crashes on sm_120) and 28 (legacy
hhgemm, 5x slower than torch on 3x1 convs) are skipped. A geometry without a half-compute plan runs as a torch
convolution: that is part of the validated kernels numerics; such calls are counted in
`fastforks._core.kernels.fallbacks()`. `load` runs one small conv as a self-test, so a cuDNN that cannot run
half-compute plans at all raises instead of turning every conv into a torch convolution.

Host cost per new geometry: ~0.07 ms with this library's binary (`plan_build.cu`: graph, first heuristic config and
plan in one native call, most of it cuDNN's heuristic), ~0.15 ms through ctypes without it (same calls, same plans).
The heuristic query finalizes only its first config (then the first `_FIRST_CONFIGS`; all ~40 cost ~1.3 ms, which the
frontend's `create_execution_plans` always pays) and all of them only if none of those gives a usable plan; the plan
is the same as with the full list. The weight tensor and conv descriptors are reused across input sizes. A plan's
knob choices are read only for its `name`. (Building in threads does not help: cuDNN's heuristic queries do not run
concurrently, and a build from a stored engine + knobs costs more than the query.)

Per-shape dispatch: by default (`BUILD_AFTER = 0`) a geometry (shapes + strides) gets its plan on its first call, so
every call of a model returns the same output, the first call at a new size included (it pays the plan builds: ~1-13
ms for a stereo net, twice that through ctypes). With `BUILD_AFTER = n > 0` (a module setting read on every model
call; set it before loading, as graph models fix their capture call, `capture_after()`, at load) a geometry runs as a
torch fp16 convolution (fp32 accumulation; not counted as a fallback) in the first `n` model calls
(`fastforks._core.kernels.begin_call`) that use it, and its plan is built in the next one; a geometry called
`BUILD_CALLS` times without a plan (iterative refiners outside a model call) gets its plan as well. A stream of new
input sizes (ETH3D, Middlebury: each size 1-3 times) then pays no plan builds, but the first `n` calls at a size
differ slightly from the later ones (mean ~0.01 px on a disparity map, single pixels up to several px). Inside a CUDA
graph capture a missing plan is built at once, so captured graphs always hold the half-compute plans; models that
capture graphs capture a shape on its 4th call (`capture_after()`), after the plans. Plans are kept per geometry in an
LRU cache of `MAX_PLANS` entries (an unbounded cache grows with every new input size); each keeps the variant packs
(data + workspace pointers) of its last `_PACKS` calls, so a call with the same buffers (a captured graph, PyTorch's
caching allocator) costs one `cudnnBackendExecute`.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import KernelUnavailableError
from fastforks._core.kernels import call_index, cuda_device, current_stream, note_fallback, shared_library
from fastforks.kernels.cudnn_hacc import _backend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_NAME = "cudnn_hacc"
log = logging.getLogger(__name__)

_SKIP_ENGINES = (0, 8, 28)
_FIRST_CONFIGS = (1, 4)  # heuristic configs fetched first (finalizing all ~40 costs ~40x more); all only if none works
_DESCS_MAX = 1024  # weight tensor + conv descriptors kept for new input sizes
MAX_PLANS = 512
BUILD_AFTER = 0  # model calls of a geometry that run it as a torch convolution before its plan is built
BUILD_CALLS = 64  # calls of a geometry without a plan after which its plan is built in any case
CAPTURE_AFTER = 3  # eager model calls of a shape before models that replay CUDA graphs capture it
_SEEN_MAX = 4096  # geometries remembered as seen (LRU)
_PACKS = 4  # variant packs kept per plan
_X, _W, _Y = 1, 2, 3  # tensor uids of a conv graph


class _Defer:
    """`_plan` result for a geometry seen fewer than `BUILD_AFTER` times: run the torch convolution, no plan yet."""


_DEFER = _Defer()


class _Plan:
    """A built plan: the backend plan, its workspace bytes, output shape, name and its last variant packs."""

    __slots__ = ("cfg", "engine", "out_shape", "packs", "plan", "workspace")

    def __init__(
        self, plan: _backend.Desc, workspace: int, out_shape: list[int], cfg: _backend.Desc, engine: int
    ) -> None:
        """`plan` from `_backend.plan` of the engine config `cfg` (engine index `engine`)."""
        self.plan, self.workspace, self.out_shape, self.cfg, self.engine = plan, workspace, out_shape, cfg, engine
        self.packs: dict[tuple[int, int, int, int], _backend.Desc] = {}

    @property
    def name(self) -> str:
        """The plan's name as the cuDNN frontend gives it (`eng<index>_k<knob>=<value>...`)."""
        return _backend.name(self.engine, _backend.knobs(self.cfg))

    def run(self, handle: _backend.Handle, pointers: tuple[int, int, int, int]) -> None:
        """Execute on `(x, w, y, workspace)` device pointers (the caller holds the instance lock)."""
        vp = self.packs.get(pointers)
        if vp is None:
            vp = self.packs[pointers] = _backend.pack(pointers)
            if len(self.packs) > _PACKS:
                del self.packs[next(iter(self.packs))]
        _backend.execute(handle, self.plan, vp)


class CudnnHalfConv:
    """Plan cache and handle of one process.

    Attributes:
        tag: `cudnn_hacc@cudnn<backend version>`.
    """

    def __init__(self) -> None:
        """Create the cuDNN handle.

        Raises:
            KernelUnavailableError: cuDNN's library (`libcudnn.so.9`) cannot be loaded, is older than
                `_backend.MIN_VERSION` or gives no handle.
        """
        try:
            v = _backend.version()
            if v < _backend.MIN_VERSION:
                raise KernelUnavailableError(
                    _NAME,
                    f"needs cuDNN >= {_backend.version_str(_backend.MIN_VERSION)}, torch loaded cuDNN "
                    f"{_backend.version_str(v)}: install a newer torch (or nvidia-cudnn-cu12/13 9.10+)",
                )
            self.handle = _backend.create_handle()
            self.tag = f"cudnn_hacc@cudnn{v}"
        except (OSError, AttributeError, _backend.CudnnError) as e:
            raise KernelUnavailableError(_NAME, f"cuDNN 9's backend API is not usable: {e}") from e
        self._builder = _builder()
        self._plans: dict[tuple[Any, ...], _Plan | None] = {}  # insertion order = LRU order
        self._descs: dict[tuple[Any, ...], _backend.Desc] = {}  # `_backend.op_graph`'s size-independent descriptors
        # geometry -> [last model call, model calls, calls] while it has no plan
        self._seen: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
        self._freed = -1  # the model call that last freed the deferred GEMMs' cuBLAS workspaces
        self._deferred = False  # a geometry ran as a torch convolution since the last free (BUILD_AFTER > 0)
        self._stream = -1  # the stream the handle is bound to
        # the handle's stream binding + execute, and the plan caches, are shared by all threads
        self._lock = threading.Lock()

    def _build(  # noqa: PLR0913 - one geometry: shapes, output strides, padding, stride, dilation, kind
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y_stride: Sequence[int] | None,
        padding: Sequence[int],
        stride: Sequence[int],
        *,
        dgrad: bool = False,
        dilation: Sequence[int] | None = None,
        y_shape: Sequence[int] | None = None,
    ) -> _Plan | None:
        """The half-compute plan of one geometry, None if cuDNN has none."""
        dil = list(dilation or [1] * (w.dim() - 2))
        _check_geometry(x, w, padding, stride, dil, dgrad=dgrad)  # once per geometry (the plan key holds all of it)
        yshape = _out_shape(x, w, padding, stride, dgrad=dgrad, dilation=dil)
        if y_shape is not None and list(y_shape) != yshape:
            msg = f"cudnn_hacc conv_into: y is {list(y_shape)}, the conv gives {yshape}"
            raise ValueError(msg)
        ystride = list(y_stride or _cl_stride(yshape))
        if len(self._descs) > _DESCS_MAX:
            self._descs.clear()
        xd, yd = (list(x.shape), list(x.stride())), (yshape, ystride)
        try:
            descs = _backend.conv_descs(
                w=(list(w.shape), list(w.stride())),
                padding=padding,
                stride=stride,
                dilation=dil,
                uid=_W,
                cache=self._descs,
            )
            if self._builder is not None:  # the first heuristic config in one native call (same plan)
                r = self._builder.plan(
                    self.handle, x=xd, y=yd, descs=descs, dgrad=dgrad, mode=_backend.HEUR_A, skip=_SKIP_ENGINES
                )
                if r is not None:
                    return _Plan(r[0], r[1], yshape, r[2], r[3])
                log.debug("cudnn_hacc: no native plan for %s -> %s, querying through ctypes", xd, yd)
            g = _backend.op_graph(
                self.handle,
                x=xd,
                w=(list(w.shape), list(w.stride())),
                y=yd,
                padding=list(padding),
                stride=list(stride),
                dilation=dil,
                uids=(_X, _W, _Y),
                dgrad=dgrad,
                cache=self._descs,
            )
        except _backend.CudnnError as e:  # no half-compute graph for this geometry
            log.debug("cudnn_hacc: no half-compute graph for %s * %s: %s", tuple(x.shape), tuple(w.shape), e)
            return None
        return self._query(g, yshape)

    def _query(self, g: _backend.Desc, yshape: list[int]) -> _Plan | None:
        """The first usable plan in heuristic order (see the module docstring), None if there is none."""
        for mode in (_backend.HEUR_A, _backend.HEUR_FALLBACK):  # FALLBACK only if A has no usable plan
            tried = 0
            for limit in (*_FIRST_CONFIGS, None):  # the first config, then a few, all of them only if none works
                try:
                    cfgs = _backend.heuristics(g, mode, limit)
                except _backend.CudnnError as e:
                    log.debug("cudnn_hacc: heuristics %s found no plan: %s", mode, e)
                    break
                for cfg in cfgs[tried:]:
                    try:
                        engine, notes = _backend.engine_info(cfg)
                        if engine in _SKIP_ENGINES or _backend.RUNTIME_COMPILATION in notes:
                            continue
                        p, ws = _backend.plan(self.handle, cfg)
                    except _backend.CudnnError:
                        continue  # try the next config
                    return _Plan(p, ws, yshape, cfg, engine)
                if limit is None or len(cfgs) < limit:
                    break  # that was all of them
                tried = len(cfgs)
        return None

    def _plan(self, key: tuple[Any, ...], build: Callable[[], _Plan | None]) -> _Plan | _Defer | None:
        """The cached plan of `key`; `_DEFER` while the geometry is new (see the module docstring)."""
        with self._lock:
            if key in self._plans:
                e = self._plans.pop(key)
            elif self._defer(key):
                return _DEFER
            else:
                e = build()
                if self._deferred and self._freed != call_index():  # once per model call that builds plans
                    self._freed, self._deferred = call_index(), False
                    _free_gemm_workspaces()
            self._plans[key] = e  # most recently used last
            if len(self._plans) > MAX_PLANS:
                del self._plans[next(iter(self._plans))]
        if e is None:
            note_fallback("cudnn_hacc", f"{key[0]} {' '.join(map(str, key[1:3]))}")
        return e

    def _defer(self, key: tuple[Any, ...]) -> bool:
        """Count one call of a geometry without a plan; True: run it as a torch convolution this time."""
        if BUILD_AFTER <= 0 or torch.cuda.is_current_stream_capturing():
            return False  # a captured graph keeps what it records: always the half-compute plan
        call = call_index()
        seen = self._seen.pop(key, None) or [-1, 0, 0]
        if seen[0] != call:
            seen[0], seen[1] = call, seen[1] + 1
        seen[2] += 1
        if seen[1] > BUILD_AFTER or seen[2] > BUILD_CALLS:
            return False
        self._deferred = True
        self._seen[key] = seen  # most recently seen last
        if len(self._seen) > _SEEN_MAX:
            self._seen.popitem(last=False)
        return True

    def _execute(self, e: _Plan, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> None:
        stream = current_stream()
        # The workspace comes from the caching allocator per call (under capture: the graph's pool, alive as long as the
        # graph). Steady state reuses the same block, so the variant pack still hits; a workspace cached per stream
        # stayed resident at the largest plan's size and raised the peak (MVSplat 8 views: +360 MB reserved).
        ws = torch.empty(max(e.workspace, 1), device=x.device, dtype=torch.uint8)
        with self._lock:
            if stream != self._stream:
                _backend.set_stream(self.handle, stream)
                self._stream = stream
            e.run(self.handle, (x.data_ptr(), w.data_ptr(), y.data_ptr(), ws.data_ptr()))

    def conv(  # noqa: PLR0913, PLR0917 - the conv geometry
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        padding: Sequence[int],
        stride: Sequence[int],
        dilation: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """fp16 conv (groups 1) with fp16 accumulation; `x` any layout, `w` channels-last fp16. Returns NHWC."""
        mf = _memory_format(x.dim())
        x = x.contiguous(memory_format=mf)
        dil = tuple(dilation or (1,) * (w.dim() - 2))
        _check_inputs(x, w)
        key = ("conv", x.shape, w.shape, w.stride(), tuple(padding), tuple(stride), dil, x.device.index)
        e = self._plan(key, lambda: self._build(x, w, None, padding, stride, dilation=dil))
        if not isinstance(e, _Plan):
            return _torch_conv(x, w, b, padding, stride, dil).contiguous(memory_format=mf)
        y = torch.empty(e.out_shape, device=x.device, dtype=x.dtype, memory_format=mf)
        self._execute(e, x, w, y)
        if b is not None:
            y.add_(b.view(1, -1, *([1] * (x.dim() - 2))))
        return y

    def deconv(self, x: torch.Tensor, w: torch.Tensor, padding: Sequence[int], stride: Sequence[int]) -> torch.Tensor:
        """fp16 transposed conv (no bias, groups 1, dilation 1, output padding 0), fp16 accumulation, `w` `[C, K, ...]`.

        Returns:
            The channels-last output.
        """
        mf = _memory_format(x.dim())
        x = x.contiguous(memory_format=mf)
        _check_inputs(x, w)
        key = ("dgrad", x.shape, w.shape, w.stride(), tuple(padding), tuple(stride), x.device.index)
        e = self._plan(key, lambda: self._build(x, w, None, padding, stride, dgrad=True))
        if not isinstance(e, _Plan):
            deconv = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d  # noqa: PLR2004 - NCHW
            return deconv(x, w, None, stride, padding).contiguous(memory_format=mf)
        y = torch.empty(e.out_shape, device=x.device, dtype=x.dtype, memory_format=mf)
        self._execute(e, x, w, y)
        return y

    def conv_into(self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, padding: int, stride: int) -> None:
        """2-D fp16 conv (fp16 accumulation, no bias) written into `y`, all as strided `[1, C, H, W]` views.

        `x` and `y` may be channel slices of packed NHWC buffers (`buf[..., c0:c1].permute(2, 0, 1)[None]`): cuDNN
        reads and writes them in place, without copies. `w` is a channels-last `[K, C, R, S]` view.
        """
        _check_inputs(x, w)
        pad, st = (padding, padding), (stride, stride)
        key = ("conv_into", x.shape, w.shape, x.stride(), w.stride(), y.stride(), padding, stride, x.device.index)
        e = self._plan(key, lambda: self._build(x, w, y.stride(), pad, st, y_shape=y.shape))
        if not isinstance(e, _Plan):
            y.copy_(_torch_conv(x, w, None, pad, st))
            return
        self._execute(e, x, w, y)


def _torch_conv(  # noqa: PLR0913, PLR0917 - the conv geometry
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int] | None = None,
) -> torch.Tensor:
    """The conv as torch runs it (fp32 accumulation); a pointwise conv as a cuBLAS GEMM over the channels.

    PyTorch's cuDNN path costs ~1-3 ms of host time on the first call of a new 1x1 shape, a GEMM ~10 us.
    """
    if _pointwise(x, w, padding, stride):
        nd = x.dim()
        xm = x.permute(0, *range(2, nd), 1).reshape(-1, x.shape[1])  # a view for channels-last x
        wm = w.reshape(w.shape[0], w.shape[1]).t()
        y = torch.mm(xm, wm) if b is None else torch.addmm(b, xm, wm)
        return y.view(x.shape[0], *x.shape[2:], w.shape[0]).permute(0, nd - 1, *range(1, nd - 1))
    conv = F.conv2d if x.dim() == 4 else F.conv3d  # noqa: PLR2004 - NCHW
    return conv(x, w, b, stride, padding, dilation or 1)


def _free_gemm_workspaces() -> None:
    """Free the cuBLAS workspaces (~32 MB per stream) the deferred pointwise GEMMs allocated.

    Called when plans are built, outside a capture, after a device synchronize (no kernel still uses them); PyTorch's
    own CUDA graph capture frees them the same way. A later cuBLAS call takes its workspace from the allocator again.
    """
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
        getattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None)()  # noqa: SLF001 - no public API


def _pointwise(x: torch.Tensor, w: torch.Tensor, padding: Sequence[int], stride: Sequence[int]) -> bool:
    return math.prod(w.shape[2:]) == 1 and w.shape[1] == x.shape[1] and not any(padding) and all(s == 1 for s in stride)


def _check_inputs(x: torch.Tensor, w: torch.Tensor) -> None:
    """Plans are built for fp16 data: other dtypes must not reuse (or build) them."""
    if x.dtype != torch.float16 or w.dtype != torch.float16:
        msg = f"cudnn_hacc runs fp16 only, got {x.dtype} * {w.dtype}"
        raise TypeError(msg)


def _out_shape(  # noqa: PLR0913 - the conv geometry
    x: torch.Tensor,
    w: torch.Tensor,
    padding: Sequence[int],
    stride: Sequence[int],
    *,
    dgrad: bool,
    dilation: Sequence[int] | None = None,
) -> list[int]:
    """`[N, C_out, *spatial]` of a conv (`w` = `[K, C, ...]`) or a transposed conv (`dgrad`, `w` = `[C, K, ...]`)."""
    sp = range(w.dim() - 2)
    d = dilation or [1] * len(sp)
    if dgrad:
        return [
            x.shape[0],
            w.shape[1],
            *[(x.shape[2 + i] - 1) * stride[i] - 2 * padding[i] + w.shape[2 + i] for i in sp],
        ]
    return [
        x.shape[0],
        w.shape[0],
        *[(x.shape[2 + i] + 2 * padding[i] - d[i] * (w.shape[2 + i] - 1) - 1) // stride[i] + 1 for i in sp],
    ]


def _check_geometry(  # noqa: PLR0913 - the conv geometry
    x: torch.Tensor,
    w: torch.Tensor,
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    *,
    dgrad: bool,
) -> None:
    """Raise for a geometry the graphs do not describe (groups != 1, wrong ranks, an empty output).

    Instead of a wrong result; runs when a geometry's plan is built (calls before that run as torch convolutions).
    """
    nd = w.dim() - 2
    c_in = w.shape[0] if dgrad else w.shape[1]
    bad = (
        x.dim() != w.dim()
        or x.shape[1] != c_in
        or not len(padding) == len(stride) == len(dilation) == nd
        or any(p < 0 for p in padding)
        or any(s < 1 for s in stride)
        or any(d < 1 for d in dilation)
        or min(_out_shape(x, w, padding, stride, dgrad=dgrad, dilation=dilation)) < 1
    )
    if bad:
        msg = (
            f"cudnn_hacc {'deconv' if dgrad else 'conv'}: unsupported geometry x {list(x.shape)} * w {list(w.shape)},"
            f" padding {list(padding)}, stride {list(stride)}, dilation {list(dilation)} (groups 1 only:"
            f" x needs {c_in} channels)"
        )
        raise ValueError(msg)


def _memory_format(dim: int) -> torch.memory_format:
    return torch.channels_last if dim == 4 else torch.channels_last_3d  # noqa: PLR2004 - NCHW


def _cl_stride(shape: Sequence[int]) -> list[int]:
    """Channels-last strides of `[N, C, *spatial]`."""
    _, c, *sp = shape
    st = [0] * len(shape)
    acc = c
    st[1] = 1
    for i in range(len(sp) - 1, -1, -1):
        st[2 + i] = acc
        acc *= sp[i]
    st[0] = acc
    return st


def capture_after() -> int:
    """Eager calls of a shape before a graph-replaying model on these plans captures it (`shared.cuda_graphs`).

    A capture does not change the math (a captured graph holds the same plans); deferring it to the 4th call keeps calls
    2-3 of a size (a stream of new sizes: each 1-3 times) from paying a capture they never replay, and a capture
    after `BUILD_AFTER` torch calls records the half-compute plans.
    """
    return max(CAPTURE_AFTER, BUILD_AFTER)


def _builder() -> _backend.Builder | None:
    """The native one-call plan build (`plan_build.cu`), None without its binary: the same plans through ctypes."""
    try:
        lib, _ = shared_library(
            Path(__file__).with_name("manifest.toml"), torch.device("cuda", torch.cuda.current_device())
        )
        return _backend.Builder(lib)
    except (KernelUnavailableError, AttributeError, OSError) as e:
        log.debug("cudnn_hacc: plans built through ctypes (%s)", e)
        return None


@functools.cache
def _instance() -> CudnnHalfConv:
    return CudnnHalfConv()


@functools.cache
def _self_test(device: torch.device) -> None:
    """One 3x3 conv through a half-compute plan, compared with torch: cuDNN must work on `device` at all."""
    with torch.cuda.device(device):
        g = torch.Generator(device).manual_seed(0)
        x = torch.randn(1, 16, 8, 8, device=device, generator=g).half().contiguous(memory_format=torch.channels_last)
        w = torch.randn(16, 16, 3, 3, device=device, generator=g).half().contiguous(memory_format=torch.channels_last)
        conv = _instance()
        y = None
        try:
            e = conv._build(x, w, None, (1, 1), (1, 1))  # noqa: SLF001 - same module
            if e is not None:
                y = torch.empty(e.out_shape, device=device, dtype=x.dtype, memory_format=torch.channels_last)
                conv._execute(e, x, w, y)  # noqa: SLF001 - same module
        except Exception as err:  # noqa: BLE001 - any cuDNN failure here means kernels cannot run
            e, reason = None, f"{type(err).__name__}: {err}"
        else:
            reason = "no half-compute plan for a 3x3 fp16 conv"
        if (
            e is not None
            and y is not None
            and not torch.allclose(y.float(), F.conv2d(x.float(), w.float(), padding=1), atol=0.25)
        ):
            e, reason = None, "a 3x3 fp16 conv gave a wrong result"
        if e is None:
            arch = "sm_{}{}".format(*torch.cuda.get_device_capability(device))
            raise KernelUnavailableError(_NAME, f"unusable on {arch} ({conv.tag}): {reason}")


def load(device: torch.device) -> CudnnHalfConv:
    """Return the process-wide instance after a self-test on `device` (once per device).

    Raises:
        KernelUnavailableError: Not a CUDA device, or cuDNN's backend API cannot run fp16 half-compute convolutions
            here.
    """
    if device.type != "cuda":
        raise KernelUnavailableError(_NAME, f"needs a CUDA device, got {device}")
    _self_test(cuda_device(device))
    return _instance()


def deconv_half_acc(x: torch.Tensor, w: torch.Tensor, padding: Sequence[int], stride: Sequence[int]) -> torch.Tensor:
    """`CudnnHalfConv.deconv` of the process-wide instance."""
    return _instance().deconv(x, w, padding, stride)


def conv_half_acc(  # noqa: PLR0913, PLR0917 - the conv geometry
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int] | None = None,
) -> torch.Tensor:
    """`CudnnHalfConv.conv` of the process-wide instance."""
    return _instance().conv(x, w, b, padding, stride, dilation)
