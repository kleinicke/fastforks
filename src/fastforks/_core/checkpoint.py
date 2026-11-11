"""Checkpoint reading: the only place that unpickles anything other than `weights_only` data.

Some released checkpoints pickle more than tensors (an OmegaConf training config, a yacs `CfgNode`, numpy scalars), so
`torch.load(weights_only=True)` rejects them. `restricted_load` reads them with an unpickler that resolves exactly the
`(module, name)` pairs of a fixed table and refuses everything else, so a checkpoint cannot run code while it loads.
Config objects load as inert stand-ins that keep nothing. Plain pickles that are not torch files (MegEngine `.mge`
checkpoints: a dict of numpy arrays) are read by the same unpickler, with `NUMPY_ARRAYS` as the extra globals. Every
family reads its weights through `fastforks._core.runner.Context.state_dict`, which calls `load_state_dict` here.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import pickle
import re
import types
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.errors import CheckpointMismatchError, WeightsError

try:  # numpy's array pickle (numpy 2 moved the module; pickles name either)
    from numpy._core.multiarray import _reconstruct
    from numpy._core.multiarray import scalar as _scalar
except ImportError:  # numpy 1
    from numpy.core.multiarray import _reconstruct
    from numpy.core.multiarray import scalar as _scalar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "NUMPY_ARRAYS",
    "CheckpointMismatchError",
    "Inert",
    "empty_bytes",
    "latin1_encode",
    "load_state_dict",
    "load_strict",
    "mismatch_details",
    "numeric_scalar",
    "restricted_load",
    "torch_mismatch",
    "torchscript_state_dict",
]

_STORAGES = ("Float", "Double", "Half", "BFloat16", "Long", "Int", "Short", "Char", "Byte", "Bool")
_BASE: dict[tuple[str, str], object] = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,  # noqa: SLF001 - torch's own tensor pickle
    **{("torch", f"{s}Storage"): getattr(torch, f"{s}Storage") for s in _STORAGES},
    **{
        (m, n): getattr(builtins, n)
        for m in ("builtins", "__builtin__")
        for n in ("list", "dict", "set", "tuple", "int")
    },
    ("__builtin__", "long"): int,  # python 2 pickles
}
# storage class -> element dtype (without `Storage.dtype`, which warns that TypedStorage is deprecated)
_STORAGE_DTYPES: dict[object, torch.dtype] = {
    getattr(torch, f"{s}Storage"): getattr(torch, d)
    for s, d in zip(
        _STORAGES,
        ("float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8", "uint8", "bool"),
        strict=True,
    )
}
# torch's legacy (pre-1.6) format starts with this pickled magic number; other non-zip files are plain pickles
_LEGACY_MAGIC = b"\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19"
# numeric numpy arrays (an object array would need further globals, which stay refused)
NUMPY_ARRAYS: dict[tuple[str, str], object] = {
    **{(m, "_reconstruct"): _reconstruct for m in ("numpy.core.multiarray", "numpy._core.multiarray")},
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


_LATIN1 = frozenset({"latin1", "latin-1", "iso-8859-1", "l1"})
_MAX_CHECKPOINT_BYTES = 8 << 30  # larger than every released checkpoint (the biggest is ~1.2 GB)


def latin1_encode(obj: str, encoding: str = "latin1") -> bytes:
    """The `_codecs.encode` of a protocol-2 pickle of `bytes`: str -> bytes in latin-1, no other codec."""
    if not isinstance(obj, str) or encoding not in _LATIN1:
        msg = (
            "the checkpoint calls _codecs.encode with something other than a str and latin-1; it is not a plain "
            "weights file. Re-save the tensors with torch.save(model.state_dict(), path)"
        )
        raise WeightsError(msg)
    return obj.encode("latin-1")


def empty_bytes(*args: object) -> bytes:
    """The `__builtin__.bytes` of a protocol-2 pickle of `b""` (`_codecs.encode` covers every other `bytes`)."""
    if args:
        msg = (
            "the checkpoint calls bytes with arguments; it is not a plain weights file. Re-save the tensors with "
            "torch.save(model.state_dict(), path)"
        )
        raise WeightsError(msg)
    return b""


def numeric_scalar(dtype: np.dtype, obj: object) -> object:
    """`numpy.core.multiarray.scalar` for numeric dtypes only (an object dtype would take an arbitrary object)."""
    if not isinstance(dtype, np.dtype) or dtype.hasobject or not isinstance(obj, bytes):
        msg = (
            "the checkpoint builds a numpy scalar of an object dtype or from a non-bytes value; it is not a plain "
            "weights file. Re-save the tensors with torch.save(model.state_dict(), path)"
        )
        raise WeightsError(msg)
    return _scalar(dtype, obj)


class Inert:
    """Stand-in for a config object saved in a checkpoint: keeps nothing, runs nothing."""

    def __init__(self, *_: object, **__: object) -> None:
        """Accept and drop any constructor arguments."""

    def __setstate__(self, _: object) -> None:
        """Accept and drop the saved state."""


def _module(table: Mapping[tuple[str, str], object], path: Path) -> types.SimpleNamespace:
    """A `pickle_module` for `torch.load` whose unpickler resolves only `table` (and has no unrestricted `load`)."""

    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - pickle protocol
            try:
                return table[module, name]
            except KeyError:
                msg = (
                    f"{path.name}: the checkpoint references {module}.{name}, which is not on the allow-list; "
                    f"re-save the weights with torch.save(model.state_dict(), path)"
                )
                raise WeightsError(msg) from None

    def load(file: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - torch's legacy reader passes its own arguments
        return Unpickler(file, **kwargs).load()

    return types.SimpleNamespace(Unpickler=Unpickler, load=load, __name__="fastforks_restricted_pickle")


def restricted_load(path: Path, extra: Mapping[tuple[str, str], object] | None = None) -> Any:  # noqa: ANN401 - whatever the file holds
    """Read a checkpoint on the CPU; only tensors, plain containers and the `extra` globals may be unpickled.

    Without `extra` this is `torch.load(weights_only=True)`. With it, a pickle module that resolves exactly the base
    table plus `extra` reads the file. Zip files are memory-mapped; the legacy (pre-1.6) format is detected and read
    normally. A file that is neither (a plain pickle, e.g. MegEngine's `.mge`) is read by the same restricted
    unpickler; it needs `extra` (`NUMPY_ARRAYS` for numpy arrays).

    Args:
        path: Checkpoint file.
        extra: Further allowed globals, `{(module, name): object}`: the object is what the pickle gets (use `Inert` for
            config classes).

    Returns:
        The unpickled object.

    Raises:
        WeightsError: The checkpoint references a global that is not allowed, or cannot be read (including a malformed
            state for an allowed global).
    """
    if path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        msg = f"{path.name}: larger than {_MAX_CHECKPOINT_BYTES >> 30} GiB, refusing to read it"
        raise WeightsError(msg)
    legacy = not zipfile.is_zipfile(path)
    try:
        if legacy and _plain_pickle(path):
            if not extra:
                msg = f"{path.name}: not a torch checkpoint (a plain pickle); reading it needs its allowed globals"
                raise WeightsError(msg)
            with path.open("rb") as f:
                return _module({**_BASE, **extra}, path).load(f)
        if not extra:  # torch's own `weights_only` unpickler is the stricter one; ours is only for the extra globals
            return torch.load(path, map_location="cpu", mmap=not legacy, weights_only=True)
        table = {**_BASE, **extra}
        return torch.load(
            path, map_location="cpu", mmap=not legacy, weights_only=False, pickle_module=_module(table, path)
        )
    except pickle.UnpicklingError as e:
        msg = f"{path.name}: cannot read the checkpoint safely: {e}"
        raise WeightsError(msg) from e
    except (TypeError, ValueError, AttributeError) as e:  # a malformed state for an allowed global
        msg = f"{path.name}: malformed checkpoint: {e}"
        raise WeightsError(msg) from e


def _is_safetensors(path: Path) -> bool:
    """Whether `path` is a safetensors file (by content: cached weight files are named by hash, without suffix)."""
    with path.open("rb") as f:  # an 8-byte little-endian header length, then the JSON header
        size, brace = f.read(8), f.read(1)
    return brace == b"{" and int.from_bytes(size, "little") < path.stat().st_size


def _plain_pickle(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(_LEGACY_MAGIC)) != _LEGACY_MAGIC


def load_state_dict(
    path: Path,
    *,
    key: str | tuple[str, ...] | None = None,
    strip: str = "module.",
    extra: Mapping[tuple[str, str], object] | None = None,
) -> dict[str, torch.Tensor]:
    """The flat state dict of a checkpoint file (see `restricted_load`; `.safetensors` files need no unpickling).

    numpy arrays become tensors.

    Args:
        path: Checkpoint file.
        key: Entry of the checkpoint that holds the state dict (`"model"`, `"state_dict"`; a tuple for nested entries,
            e.g. `("ema", "module")`), or `None` if the file is the state dict.
        strip: Prefix removed from every key (`DataParallel` saves with `"module."`); `""` keeps the keys.
        extra: Further allowed globals, see `restricted_load`.

    Raises:
        WeightsError: Unsafe or unreadable file, no state dict under `key`, or a non-numeric numpy array.
    """
    if _is_safetensors(path):
        from safetensors.torch import load_file  # noqa: PLC0415 - only for safetensors weights

        ckpt: object = load_file(path)
    elif _is_torchscript(path):
        ckpt = torchscript_state_dict(path)
    else:
        ckpt = restricted_load(path, extra)
    for k in (key,) if isinstance(key, str) else key or ():
        if not isinstance(ckpt, Mapping) or k not in ckpt:
            msg = (
                f"{path.name}: the checkpoint has no entry {key!r}; expected a state dict (or a dict holding it under "
                f"{key!r}). Re-save with torch.save(model.state_dict(), path), see usage 'Already have the weights?'"
            )
            raise WeightsError(msg)
        ckpt = ckpt[k]
    if not isinstance(ckpt, Mapping):
        msg = f"{path.name}: expected a state dict, got {type(ckpt).__name__}"
        raise WeightsError(msg)
    return {
        (k.removeprefix(strip) if strip else k): _tensor(v, f"{path.name}: {k}") if isinstance(v, np.ndarray) else v
        for k, v in ckpt.items()
    }


def _is_torchscript(path: Path) -> bool:
    """Whether `path` is a TorchScript archive (a torch zip with serialized code next to `data.pkl`)."""
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as z:
        return any("/code/__torch__/" in n for n in z.namelist())


_SCRIPT_LISTS = frozenset(
    ("build_intlist", "build_tensorlist", "build_doublelist", "build_boollist", "build_genlist", "restore_type_tag")
)


def _script_value(value: object, *_type: object) -> object:
    return value


class _ScriptObject:
    """Stand-in for a TorchScript module object: keeps its attribute dict (tensors, sub-objects), runs nothing."""

    def __setstate__(self, state: object) -> None:
        self.attrs = state if isinstance(state, Mapping) else {}


class _ScriptUnpickler(pickle.Unpickler):
    """`data.pkl` of a TorchScript archive: module classes -> `_ScriptObject`, tensors from the mapped file."""

    def __init__(self, raw: bytes, path: Path, root: str, infos: dict[str, zipfile.ZipInfo]) -> None:
        super().__init__(io.BytesIO(raw))
        self.path, self.root, self.infos = path, root, infos
        self.mm = np.memmap(path, dtype=np.uint8, mode="c")

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - pickle protocol
        if module.startswith("__torch__."):
            return _ScriptObject
        if module == "torch.jit._pickle" and name in _SCRIPT_LISTS:  # typed-list wrappers: keep the value as is
            return _script_value
        try:
            return _BASE[module, name]
        except KeyError:
            msg = f"{self.path.name}: the archive references {module}.{name}, which is not on the allow-list"
            raise WeightsError(msg) from None

    def persistent_load(self, pid: Any) -> Any:  # noqa: ANN401 - pickle protocol
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):  # noqa: PLR2004 - torch's id
            msg = f"{self.path.name}: unexpected persistent id {pid!r}"
            raise WeightsError(msg)
        _, kind, key, _, numel = pid
        dtype = _STORAGE_DTYPES.get(kind)
        if dtype is None:
            msg = f"{self.path.name}: tensor data {key!r} has an unsupported storage type {kind!r}"
            raise WeightsError(msg)
        info = self.infos.get(f"{self.root}data/{key}")
        if info is None or info.compress_type != zipfile.ZIP_STORED or int(numel) * dtype.itemsize > info.file_size:
            msg = f"{self.path.name}: tensor data {key!r} is missing, compressed or too short"
            raise WeightsError(msg)
        header = self.mm[info.header_offset : info.header_offset + 30]  # zip local header: name / extra lengths
        start = info.header_offset + 30 + int(header[26:28].view("<u2")[0]) + int(header[28:30].view("<u2")[0])
        data = torch.from_numpy(self.mm[start : start + int(numel) * dtype.itemsize])
        return torch.TypedStorage(wrap_storage=data.untyped_storage(), dtype=dtype, _internal=True)


def _script_tensors(o: object, prefix: str, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for k, v in (o.attrs if isinstance(o, _ScriptObject) else {}).items():
        if isinstance(v, torch.Tensor):
            out[prefix + k] = v
        elif isinstance(v, _ScriptObject):
            _script_tensors(v, f"{prefix}{k}.", out)
    return out


def torchscript_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Every tensor attribute of a TorchScript archive, named as `torch.jit.load(path).state_dict()` names them.

    The archive's TorchScript code is never loaded or run: `data.pkl` is read with a restricted unpickler, every
    `__torch__.*` class becomes an inert attribute holder, and tensors are views of the memory-mapped file
    (copy-on-write: writes stay private). Only uncompressed, little-endian archives (as torch writes them) are read.

    Raises:
        WeightsError: Unsafe or unreadable archive.
    """
    with zipfile.ZipFile(path) as z:
        infos = {i.filename: i for i in z.infolist()}
        pkl = next((n for n in infos if n.count("/") == 1 and n.endswith("/data.pkl")), None)
        root = (pkl or "").removesuffix("data.pkl")
        if pkl is None or (f"{root}byteorder" in infos and z.read(f"{root}byteorder") != b"little"):
            msg = (
                f"{path.name}: not a little-endian TorchScript archive with a data.pkl; export it again with "
                f"torch.jit.save, or re-save the weights as a state dict with torch.save(model.state_dict(), path)"
            )
            raise WeightsError(msg)
        raw = z.read(pkl)
    try:
        obj = _ScriptUnpickler(raw, path, root, infos).load()
    except (pickle.UnpicklingError, TypeError, ValueError, AttributeError) as e:
        msg = f"{path.name}: cannot read the TorchScript archive safely: {e}"
        raise WeightsError(msg) from e
    return _script_tensors(obj, "", {})


def _tensor(a: np.ndarray, what: str) -> torch.Tensor:
    """A numeric numpy array as a tensor (byte-swapped to native order if needed)."""
    if a.dtype.kind not in "biuf":
        msg = f"{what} has dtype {a.dtype}, not a numeric one"
        raise WeightsError(msg)
    return torch.from_numpy(a if a.dtype.isnative else a.astype(a.dtype.newbyteorder("=")))


_SHOWN = 5  # keys listed per kind in a mismatch message
_TORCH_SHAPE = re.compile(
    r"size mismatch for (\S+): copying a param with shape torch\.Size\((\[.*?\])\) from checkpoint,"
    r" the shape in current model is torch\.Size\((\[.*?\])\)"
)


def mismatch_details(
    missing: Sequence[str],
    unexpected: Sequence[str],
    shapes: Sequence[tuple[str, str, str]],
    dtypes: Sequence[tuple[str, str, str]] = (),
) -> str:
    """The counts and the first few missing, unexpected and reshaped keys, e.g. `"1 unexpected keys (extra.weight)"`.

    Args:
        missing: Network tensors the checkpoint lacks.
        unexpected: Checkpoint tensors the network does not have.
        shapes: `(key, checkpoint shape, network shape)` of tensors whose shapes differ.
        dtypes: `(key, checkpoint dtype, network dtype)` of tensors whose dtypes cannot be cast (float vs integer).
    """

    def first(keys: Sequence[str]) -> str:
        return ", ".join(keys[:_SHOWN]) + (", ..." if len(keys) > _SHOWN else "")

    parts = [
        f"{len(missing)} missing keys ({first(missing)})" if missing else "",
        f"{len(unexpected)} unexpected keys ({first(unexpected)})" if unexpected else "",
        f"{len(shapes)} shape mismatches ({first([f'{k}: checkpoint {c} vs network {n}' for k, c, n in shapes])})"
        if shapes
        else "",
        f"{len(dtypes)} dtype mismatches ({first([f'{k}: checkpoint {c} vs network {n}' for k, c, n in dtypes])})"
        if dtypes
        else "",
    ]
    return "; ".join(p for p in parts if p)


def torch_mismatch(error: BaseException) -> tuple[list[str], list[str], list[tuple[str, str, str]]] | None:
    """The missing, unexpected and reshaped keys of a strict `Module.load_state_dict` error, else `None`."""
    text = str(error)
    if "Error(s) in loading state_dict" not in text and "size mismatch for" not in text:
        return None
    missing: list[str] = []
    unexpected: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith("Missing key(s) in state_dict:"):
            missing = re.findall(r'"([^"]+)"', line)
        elif line.startswith("Unexpected key(s) in state_dict:"):
            unexpected = re.findall(r'"([^"]+)"', line)
    return missing, unexpected, _TORCH_SHAPE.findall(text)


def check_fits(
    own: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    what: str,
    *,
    not_in_checkpoint: tuple[str, ...] = (),
) -> None:
    """Check that `state` fits a network's state dict `own` (keys, shapes, float vs integer) without loading it.

    For converters that read a checkpoint into their own layout (e.g. a kernels engine): `own` may come from the
    network built on the meta device. Raises `CheckpointMismatchError` as `load_strict`.
    """
    missing = [k for k in own if k not in state and not (not_in_checkpoint and k.endswith(not_in_checkpoint))]
    unexpected = [k for k in state if k not in own]
    shapes = [
        (k, str(list(v.shape)), str(list(own[k].shape)))
        for k, v in state.items()
        if k in own and v.shape != own[k].shape
    ]
    dtypes = [
        (k, str(v.dtype).removeprefix("torch."), str(own[k].dtype).removeprefix("torch."))
        for k, v in state.items()
        if k in own and v.is_floating_point() != own[k].is_floating_point()
    ]
    if missing or unexpected or shapes or dtypes:
        raise CheckpointMismatchError(what, mismatch_details(missing, unexpected, shapes, dtypes))


def load_strict(  # noqa: PLR0913 - the exceptions a family may declare
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    what: str,
    *,
    unused: tuple[str, ...] = (),
    not_in_checkpoint: tuple[str, ...] = (),
    assign: bool = False,
    dtype: torch.dtype | None = None,
) -> None:
    """Load `state` into `module`: every key and shape must match, except for the declared exceptions.

    Floating-point tensors are cast to the network's dtype (to `dtype` if given), so a float32 fine-tune of a
    float16 checkpoint (or the reverse) loads like the original; with `assign=True` the cast is explicit, as
    `Module.load_state_dict` would keep the checkpoint's dtype.

    Args:
        module: The network.
        state: The checkpoint's state dict (keys already renamed to the network's).
        what: The checkpoint in error messages, e.g. `"s2m2-s@1 checkpoint"`.
        unused: Key prefixes of checkpoint tensors the network does not use; those not in the network are dropped.
        not_in_checkpoint: Key suffixes of network tensors the checkpoint lacks (unused at inference); they keep
            their initial values.
        assign: See `Module.load_state_dict`.
        dtype: The floating-point dtype of the loaded tensors, e.g. that of the recorded checkpoint when the network
            is built on the meta device in float32; default: each network tensor's dtype.

    Raises:
        CheckpointMismatchError: Missing, unexpected, reshaped tensors or a floating-point tensor where the network
            has an integer one (or the reverse); the message lists the first few of each.
    """
    own = module.state_dict()
    try:
        state = _fitting(own, state, unused, what, not_in_checkpoint)
    except CheckpointMismatchError:
        state = _module_state(own, state, unused, not_in_checkpoint) or _fitting(
            own, state, unused, what, not_in_checkpoint
        )
    if assign:
        state = {k: _cast(v, dtype or own[k].dtype) for k, v in state.items()}
    module.load_state_dict(state, strict=not not_in_checkpoint, assign=assign)


def _fitting(
    own: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    unused: tuple[str, ...],
    what: str,
    not_in_checkpoint: tuple[str, ...],
) -> Mapping[str, torch.Tensor]:
    """`state` without the declared unused tensors, checked against `own` (`CheckpointMismatchError` if it misfits)."""
    if unused:
        state = {k: v for k, v in state.items() if k in own or not k.startswith(unused)}
    check_fits(own, state, what, not_in_checkpoint=not_in_checkpoint)
    return state


def _module_state(
    own: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    unused: tuple[str, ...],
    not_in_checkpoint: tuple[str, ...],
) -> Mapping[str, torch.Tensor] | None:
    """A misfitting `state` saved from `model.module()`: its keys without the `Forward` wrappers' `net.` prefixes.

    Only tried after the keys did not fit (no cost on the normal path); `None` if no prefix depth fits.
    """
    while state and all(k.startswith("net.") for k in state):
        state = {k.removeprefix("net."): v for k, v in state.items()}
        with contextlib.suppress(CheckpointMismatchError):
            return _fitting(own, state, unused, "", not_in_checkpoint)
    return None


def _cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """`tensor` in `dtype` if it is floating point (a no-op if it already is), else unchanged."""
    return tensor.to(dtype) if tensor.is_floating_point() and tensor.dtype != dtype else tensor
