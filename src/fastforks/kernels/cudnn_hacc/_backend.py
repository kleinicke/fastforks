"""The cuDNN backend API through ctypes: half-compute conv graphs, heuristics, engine configs and plans.

Compared with the cuDNN frontend's python API: no extra package, and a heuristic query can finalize just its first
configs (the frontend's `create_execution_plans` finalizes all of them, ~10x the host time). Calls through ctypes
release the GIL, so other python threads keep running while cuDNN builds.

The library is the `libcudnn.so.9` PyTorch loaded (one cuDNN per process); the constants are the cuDNN 9 ABI values.
"""

from __future__ import annotations

import ctypes
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# descriptor types
_CONVOLUTION, _ENGINE, _ENGINECFG, _ENGINEHEUR, _EXECUTION_PLAN, _KNOB_CHOICE = 1, 2, 3, 4, 5, 7
_OP_CONV_FWD, _OP_CONV_BWD_DATA, _OPERATIONGRAPH, _VARIANT_PACK, _TENSOR = 10, 12, 15, 16, 17
# attribute value types
_T_HANDLE, _T_DATA_TYPE, _T_BOOLEAN, _T_INT64, _T_FLOAT, _T_VOID_PTR = 0, 1, 2, 3, 4, 6
_T_CONVOLUTION_MODE, _T_HEUR_MODE, _T_KNOB_TYPE, _T_BACKEND_DESCRIPTOR, _T_BEHAVIOR_NOTE = 7, 8, 9, 15, 19
# attributes
_CONV_COMP_TYPE, _CONV_MODE, _CONV_DILATIONS, _CONV_STRIDES, _CONV_POST_PAD, _CONV_PRE_PAD, _CONV_DIMS = range(100, 107)
_HEUR_MODE, _HEUR_GRAPH, _HEUR_RESULTS = 200, 201, 202
_CFG_ENGINE, _CFG_KNOB_CHOICES = 300, 302
_PLAN_HANDLE, _PLAN_CONFIG, _PLAN_WORKSPACE = 400, 401, 402
_KNOB_TYPE, _KNOB_VALUE = 600, 601
_FWD_ALPHA, _FWD_BETA, _FWD_DESC, _FWD_W, _FWD_X, _FWD_Y = range(700, 706)
_BWD_ALPHA, _BWD_BETA, _BWD_DESC, _BWD_W, _BWD_DX, _BWD_DY = range(706, 712)
_GRAPH_HANDLE, _GRAPH_OPS = 800, 801
_TENSOR_ALIGNMENT, _TENSOR_DATA_TYPE, _TENSOR_DIMS, _TENSOR_STRIDES = 900, 901, 902, 903
_TENSOR_UID, _TENSOR_VIRTUAL = 906, 907
_PACK_UIDS, _PACK_POINTERS, _PACK_WORKSPACE = 1000, 1001, 1003
_ENGINE_GRAPH, _ENGINE_INDEX, _ENGINE_BEHAVIOR = 1300, 1301, 1305
# values
_HALF = 2
_CROSS_CORRELATION = 1
HEUR_A, HEUR_FALLBACK = 3, 2
RUNTIME_COMPILATION = 0
_NOTES = 32
_KNOB_TYPES = 64  # > CUDNN_KNOB_TYPE_COUNTS (46 in cuDNN 9.24)
_STATUS_NOT_SUPPORTED = 3000
MIN_VERSION = 91000  # 9.10.0: the oldest cuDNN run (9.10.2, 9.19.0, 9.24.0); torch >= 2.8 wheels ship >= 9.10

_VP = ctypes.c_void_p
_I64 = ctypes.c_int64


class CudnnError(RuntimeError):
    """A cuDNN backend call failed."""

    def __init__(self, status: int, what: str) -> None:
        """`status` from cuDNN, `what` the failing call."""
        super().__init__(f"{what}: cuDNN status {status} ({_lib().name(status)})")
        self.status = status


def _mapped_path(soname: str) -> str:
    """Absolute path of the already loaded `soname` (from `/proc/self/maps`), else `soname` itself."""
    try:
        with open("/proc/self/maps") as f:  # noqa: PTH123
            for line in f:
                path = line.rsplit(maxsplit=1)[-1]
                if path.startswith("/") and path.rsplit("/", 1)[-1].startswith(soname):
                    return path
    except OSError:
        pass
    return soname


class _Lib:
    """`libcudnn.so.9` with the backend API's signatures."""

    def __init__(self) -> None:
        import torch  # noqa: PLC0415 - loads cuDNN first

        torch.backends.cudnn.version()  # loads the library torch links: dlopen by soname then finds that copy
        lib = ctypes.CDLL(_mapped_path("libcudnn.so.9"))
        i32 = ctypes.c_int32
        sig = {
            "cudnnCreate": [ctypes.POINTER(_VP)],
            "cudnnDestroy": [_VP],
            "cudnnSetStream": [_VP, _VP],
            "cudnnBackendCreateDescriptor": [i32, ctypes.POINTER(_VP)],
            "cudnnBackendDestroyDescriptor": [_VP],
            "cudnnBackendFinalize": [_VP],
            "cudnnBackendSetAttribute": [_VP, i32, i32, _I64, _VP],
            "cudnnBackendGetAttribute": [_VP, i32, i32, _I64, ctypes.POINTER(_I64), _VP],
            "cudnnBackendExecute": [_VP, _VP, _VP],
        }
        for f, args in sig.items():
            fn = getattr(lib, f)
            fn.argtypes, fn.restype = args, i32
        lib.cudnnGetErrorString.argtypes, lib.cudnnGetErrorString.restype = [i32], ctypes.c_char_p
        lib.cudnnGetVersion.argtypes, lib.cudnnGetVersion.restype = [], ctypes.c_size_t
        self.lib = lib
        self.create = lib.cudnnBackendCreateDescriptor
        self.destroy = lib.cudnnBackendDestroyDescriptor
        self.finalize = lib.cudnnBackendFinalize
        self.set = lib.cudnnBackendSetAttribute
        self.get = lib.cudnnBackendGetAttribute
        self.execute = lib.cudnnBackendExecute
        self.set_stream = lib.cudnnSetStream

    def name(self, status: int) -> str:
        return (self.lib.cudnnGetErrorString(status) or b"?").decode()


_LIB: _Lib | None = None
_LIB_LOCK = threading.Lock()


def _lib() -> _Lib:
    global _LIB  # noqa: PLW0603 - one library per process
    if _LIB is not None:  # the hot path: no lock once loaded
        return _LIB
    with _LIB_LOCK:
        if _LIB is None:
            _LIB = _Lib()
        return _LIB


def _check(status: int, what: str) -> None:
    if status:
        raise CudnnError(status, what)


class Desc:
    """An owned backend descriptor (destroyed with the object)."""

    __slots__ = ("__weakref__", "keep", "ptr")

    def __init__(self, kind: int) -> None:
        """Create a descriptor of backend type `kind`."""
        self.ptr = _VP()
        _check(_lib().create(kind, ctypes.byref(self.ptr)), f"create descriptor {kind}")
        self.keep: list[Desc | Handle] = []  # what this one references (cuDNN does not own it)

    @classmethod
    def adopt(cls, ptr: int, *keep: Desc | Handle) -> Desc:
        """Own the descriptor `ptr` created elsewhere (`Builder`), which references `keep`."""
        d = cls.__new__(cls)
        d.ptr, d.keep = _VP(ptr), list(keep)
        return d

    def set(self, attr: int, typ: int, values: ctypes.Array | ctypes._SimpleCData, n: int = 1) -> Desc:  # type: ignore[type-arg]
        _check(_lib().set(self.ptr, attr, typ, n, ctypes.byref(values)), f"set attribute {attr}")
        return self

    def ints(self, attr: int, values: Sequence[int]) -> Desc:
        return self.set(attr, _T_INT64, (_I64 * len(values))(*values), len(values))

    def enum(self, attr: int, typ: int, value: int) -> Desc:
        return self.set(attr, typ, ctypes.c_int32(value))

    def desc(self, attr: int, *ds: Desc) -> Desc:
        self.keep.extend(ds)
        return self.set(attr, _T_BACKEND_DESCRIPTOR, (_VP * len(ds))(*(d.ptr for d in ds)), len(ds))

    def finalize(self) -> Desc:
        _check(_lib().finalize(self.ptr), "finalize")
        return self

    def get_int(self, attr: int) -> int:
        v, n = _I64(), _I64()
        _check(_lib().get(self.ptr, attr, _T_INT64, 1, ctypes.byref(n), ctypes.byref(v)), f"get attribute {attr}")
        return v.value

    def __del__(self) -> None:
        if self.ptr and _LIB is not None:
            _LIB.destroy(self.ptr)
            self.ptr = _VP()


def version() -> int:
    """The version number of cuDNN (92400 = 9.24.0)."""
    return int(_lib().lib.cudnnGetVersion())


def version_str(v: int) -> str:
    """`92400` -> `9.24.0`."""
    return f"{v // 10000}.{v // 100 % 100}.{v % 100}"


class Handle:
    """An owned cuDNN handle, destroyed when it and every graph / plan built on it are gone."""

    __slots__ = ("__weakref__", "ptr")

    def __init__(self) -> None:
        """`cudnnCreate` on the current device."""
        self.ptr = _VP()
        _check(_lib().lib.cudnnCreate(ctypes.byref(self.ptr)), "cudnnCreate")

    def __del__(self) -> None:
        if self.ptr and _LIB is not None:
            _LIB.lib.cudnnDestroy(self.ptr)
            self.ptr = _VP()


def create_handle() -> Handle:
    """A new cuDNN handle (`CudnnError` if cuDNN gives none)."""
    return Handle()


def set_stream(handle: Handle, stream: int) -> None:
    _check(_lib().set_stream(handle.ptr, stream), "cudnnSetStream")


def _tensor(uid: int, dims: Sequence[int], strides: Sequence[int]) -> Desc:
    t = Desc(_TENSOR).ints(_TENSOR_UID, [uid]).enum(_TENSOR_DATA_TYPE, _T_DATA_TYPE, _HALF)
    t.ints(_TENSOR_ALIGNMENT, [16]).ints(_TENSOR_DIMS, dims).ints(_TENSOR_STRIDES, strides)
    t.set(_TENSOR_VIRTUAL, _T_BOOLEAN, ctypes.c_bool())
    return t.finalize()


def conv_descs(  # noqa: PLR0913 - the weight and conv geometry
    *,
    w: tuple[Sequence[int], Sequence[int]],
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    uid: int,
    cache: dict[tuple[object, ...], Desc] | None = None,
) -> tuple[Desc, Desc]:
    """The finalized weight tensor and conv descriptors of `op_graph`, from `cache` if it has them."""
    wkey = ("w", uid, tuple(w[0]), tuple(w[1]))
    ckey = ("conv", tuple(padding), tuple(stride), tuple(dilation))
    wt = cache.get(wkey) if cache is not None else None
    if wt is None:
        wt = _tensor(uid, *w)
        if cache is not None:
            cache[wkey] = wt
    cd = cache.get(ckey) if cache is not None else None
    if cd is None:
        cd = Desc(_CONVOLUTION).ints(_CONV_DIMS, [len(padding)]).enum(_CONV_COMP_TYPE, _T_DATA_TYPE, _HALF)
        cd.enum(_CONV_MODE, _T_CONVOLUTION_MODE, _CROSS_CORRELATION).ints(_CONV_DILATIONS, dilation)
        cd.ints(_CONV_STRIDES, stride).ints(_CONV_PRE_PAD, padding).ints(_CONV_POST_PAD, padding).finalize()
        if cache is not None:
            cache[ckey] = cd
    return wt, cd


def op_graph(  # noqa: PLR0913 - one conv: three tensors, padding, stride, dilation, kind
    handle: Handle,
    *,
    x: tuple[Sequence[int], Sequence[int]],
    w: tuple[Sequence[int], Sequence[int]],
    y: tuple[Sequence[int], Sequence[int]],
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    uids: tuple[int, int, int],
    dgrad: bool,
    cache: dict[tuple[object, ...], Desc] | None = None,
) -> Desc:
    """The finalized operation graph of one half-compute conv (`dgrad`: the transposed conv `x -> y`), groups 1.

    `cache` (the caller's, bounded by it) keeps the weight tensor and conv descriptors, which do not change with the
    input size: a finalized descriptor may be referenced by any number of graphs.
    """
    xt, yt = _tensor(uids[0], *x), _tensor(uids[2], *y)
    wt, cd = conv_descs(w=w, padding=padding, stride=stride, dilation=dilation, uid=uids[1], cache=cache)
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    if dgrad:  # the data gradient: dy = the transposed conv's input, dx = its output
        op = Desc(_OP_CONV_BWD_DATA).desc(_BWD_DESC, cd).desc(_BWD_W, wt).desc(_BWD_DY, xt).desc(_BWD_DX, yt)
        op.set(_BWD_ALPHA, _T_FLOAT, one).set(_BWD_BETA, _T_FLOAT, zero)
    else:
        op = Desc(_OP_CONV_FWD).desc(_FWD_DESC, cd).desc(_FWD_X, xt).desc(_FWD_W, wt).desc(_FWD_Y, yt)
        op.set(_FWD_ALPHA, _T_FLOAT, one).set(_FWD_BETA, _T_FLOAT, zero)
    op.finalize()
    g = Desc(_OPERATIONGRAPH).set(_GRAPH_HANDLE, _T_HANDLE, handle.ptr).desc(_GRAPH_OPS, op)
    g.keep.append(handle)
    return g.finalize()


def heuristics(graph: Desc, mode: int, limit: int | None = None) -> list[Desc]:
    """The first `limit` (default all) engine configs of `graph` in cuDNN's heuristic order.

    Most of a query's cost is finalizing the configs it returns (~1.3 ms for all ~40, ~0.13 ms for 4).
    """
    h = Desc(_ENGINEHEUR).desc(_HEUR_GRAPH, graph).enum(_HEUR_MODE, _T_HEUR_MODE, mode).finalize()
    lib, n = _lib(), _I64()
    _check(lib.get(h.ptr, _HEUR_RESULTS, _T_BACKEND_DESCRIPTOR, 0, ctypes.byref(n), None), "heuristic count")
    cfgs = [Desc(_ENGINECFG) for _ in range(n.value if limit is None else min(n.value, limit))]
    arr = (_VP * len(cfgs))(*(c.ptr for c in cfgs))
    _check(lib.get(h.ptr, _HEUR_RESULTS, _T_BACKEND_DESCRIPTOR, len(cfgs), ctypes.byref(n), arr), "heuristic results")
    for c in cfgs:
        c.keep.append(graph)
    return cfgs[: n.value]


def engine_info(cfg: Desc) -> tuple[int, tuple[int, ...]]:
    """`(engine index, behavior notes)` of an engine config."""
    lib, n = _lib(), _I64()
    eng = Desc(_ENGINE)
    _check(lib.get(cfg.ptr, _CFG_ENGINE, _T_BACKEND_DESCRIPTOR, 1, ctypes.byref(n), ctypes.byref(eng.ptr)), "engine")
    notes = (ctypes.c_int32 * _NOTES)()
    _check(lib.get(eng.ptr, _ENGINE_BEHAVIOR, _T_BEHAVIOR_NOTE, _NOTES, ctypes.byref(n), notes), "behavior notes")
    return eng.get_int(_ENGINE_INDEX), tuple(notes[: n.value])


def knobs(cfg: Desc) -> tuple[tuple[int, int], ...]:
    """`((knob type, value), ...)` of an engine config, sorted (~0.2 ms: one descriptor per knob type)."""
    lib, n = _lib(), _I64()
    descs = [Desc(_KNOB_CHOICE) for _ in range(_KNOB_TYPES)]  # cuDNN gives no count for this attribute
    arr = (_VP * _KNOB_TYPES)(*(k.ptr for k in descs))
    _check(lib.get(cfg.ptr, _CFG_KNOB_CHOICES, _T_BACKEND_DESCRIPTOR, _KNOB_TYPES, ctypes.byref(n), arr), "knobs")
    choices = []
    for k in descs[: n.value]:
        t, m = ctypes.c_int32(), _I64()
        _check(lib.get(k.ptr, _KNOB_TYPE, _T_KNOB_TYPE, 1, ctypes.byref(m), ctypes.byref(t)), "knob type")
        choices.append((t.value, k.get_int(_KNOB_VALUE)))
    return tuple(sorted(choices))


def plan(handle: Handle, cfg: Desc) -> tuple[Desc, int]:
    """The finalized execution plan of an engine config and its workspace bytes."""
    p = Desc(_EXECUTION_PLAN).set(_PLAN_HANDLE, _T_HANDLE, handle.ptr).desc(_PLAN_CONFIG, cfg)
    p.keep.append(handle)
    p.finalize()
    return p, p.get_int(_PLAN_WORKSPACE)


def name(engine: int, knobs: Sequence[tuple[int, int]]) -> str:
    """The frontend's plan name: `eng<index>_k<type>=<value>...`."""
    return f"eng{engine}" + "".join(f"_k{t}={v}" for t, v in knobs)


class Builder:
    """`plan_build.cu`: the first config's plan of one conv in one native call (~0.07 ms instead of ~0.15 ms).

    It makes the same backend calls as `op_graph` + `heuristics(graph, mode, 1)` + `engine_info` + `plan`, in the same
    order, on the same library, so it gives the same plan.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        """`lib`: the loaded `cudnn_hacc` shared library."""
        cu = _lib()
        fn = lib.ffh_build
        p64, i32 = ctypes.POINTER(_I64), ctypes.c_int
        fn.argtypes = [_VP, _VP, i32, p64, _VP, _VP, i32, i32, p64, i32, ctypes.POINTER(_VP), p64, p64]
        fn.restype = i32
        self._fn, self._lib = fn, lib
        self._api = (_VP * 5)(*(ctypes.cast(f, _VP) for f in (cu.create, cu.destroy, cu.finalize, cu.set, cu.get)))

    def plan(  # noqa: PLR0913 - one conv
        self,
        handle: Handle,
        *,
        x: tuple[Sequence[int], Sequence[int]],
        y: tuple[Sequence[int], Sequence[int]],
        descs: tuple[Desc, Desc],
        dgrad: bool,
        mode: int,
        skip: Sequence[int],
    ) -> tuple[Desc, int, Desc, int] | None:
        """`(plan, workspace bytes, engine config, engine index)` of the first config of heuristic `mode`.

        None: a backend call failed or the first config is not usable (an engine in `skip`, runtime compilation, no
        plan); the caller's full query decides.

        `descs` = `conv_descs(...)`; the tensors have the uids 1 (x) and 3 (y), the weight tensor its own.
        """
        nd = len(x[0])
        keep, ws, engine = (_VP * 7)(), _I64(), _I64()
        status = self._fn(
            self._api,
            handle.ptr,
            nd,
            (_I64 * (4 * nd))(*x[0], *x[1], *y[0], *y[1]),
            descs[0].ptr,
            descs[1].ptr,
            dgrad,
            mode,
            (_I64 * len(skip))(*skip),
            len(skip),
            keep,
            ctypes.byref(ws),
            ctypes.byref(engine),
        )
        if status:
            return None
        xt, yt = Desc.adopt(keep[0]), Desc.adopt(keep[1])
        op = Desc.adopt(keep[2], descs[1], xt, descs[0], yt)
        cfg = Desc.adopt(keep[5], Desc.adopt(keep[3], op, handle))
        return Desc.adopt(keep[6], cfg, handle), ws.value, cfg, engine.value


_UIDS = (_I64 * 3)(1, 2, 3)


def pack(pointers: tuple[int, int, int, int]) -> Desc:
    """The variant pack of data pointers `(x, w, y)` (uids 1, 2, 3) and the workspace pointer."""
    vp = Desc(_VARIANT_PACK)
    lib = _lib()
    _check(lib.set(vp.ptr, _PACK_UIDS, _T_INT64, 3, _UIDS), "uids")
    _check(lib.set(vp.ptr, _PACK_POINTERS, _T_VOID_PTR, 3, (_VP * 3)(*pointers[:3])), "pointers")
    _check(lib.set(vp.ptr, _PACK_WORKSPACE, _T_VOID_PTR, 1, ctypes.byref(_VP(pointers[3]))), "workspace")
    return vp.finalize()


def execute(handle: Handle, p: Desc, vp: Desc) -> None:
    """Run plan `p` with variant pack `vp` on the handle's stream."""
    _check(_lib().execute(handle.ptr, p.ptr, vp.ptr), "execute")
