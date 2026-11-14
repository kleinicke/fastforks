# ruff: noqa: PLR2004 - protobuf wire-format keys
"""Paddle inference files (`inference.json` + `inference.pdiparams`) -> named numpy arrays, then torch state dicts.

`inference.pdiparams` is Paddle's `save_combine` stream: per tensor a uint32 version, a uint64 LoD level count (+ LoD
data), a uint32 tensor version, an int32 `TensorDesc` protobuf size, the protobuf (field 1 data type, field 2 dims)
and the raw little-endian data, in the sorted order of the parameter names that `inference.json` (PIR program)
declares. Names are Paddle's per-layer-type counters (`conv2d_12.w_0`, `batch_norm2d_3.w_1`, ...), numbered in
creation order; `state_dict` maps them to a torch module's parameters by type and creation order and checks every
shape.
"""

from __future__ import annotations

import json
import re
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import torch

_DTYPES = {5: np.float32, 4: np.float16, 3: np.int64, 2: np.int32, 6: np.float64}  # paddle VarType


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << s
        s += 7
        if b < 0x80:
            return v, i


def _desc(buf: bytes) -> tuple[int, list[int]]:
    """`TensorDesc` protobuf -> (data type, dims)."""
    dtype, dims, i = 0, [], 0
    while i < len(buf):
        key, i = _varint(buf, i)
        if key == 0x08:  # data_type
            dtype, i = _varint(buf, i)
        elif key == 0x10:  # dims, unpacked
            d, i = _varint(buf, i)
            dims.append(d - (1 << 64) if d >= 1 << 63 else d)
        elif key == 0x12:  # dims, packed
            n, i = _varint(buf, i)
            end = i + n
            while i < end:
                d, i = _varint(buf, i)
                dims.append(d - (1 << 64) if d >= 1 << 63 else d)
        else:
            msg = f"unexpected TensorDesc field {key:#x}"
            raise ValueError(msg)
    return dtype, dims


def param_shapes(program: Path) -> dict[str, tuple[int, ...]]:
    """Parameter name -> shape, from a PIR `inference.json`."""
    ops = json.loads(program.read_text())["program"]["regions"][0]["blocks"][0]["ops"]
    return {o["A"][3]: tuple(o["O"]["TT"]["D"][1]) for o in ops if o.get("#") == "p"}


def read(program: Path, params: Path) -> dict[str, np.ndarray]:
    """All parameters of a Paddle inference model by name (shapes checked against the program)."""
    shapes = param_shapes(program)
    buf = params.read_bytes()
    out: dict[str, np.ndarray] = {}
    pos = 0
    for name in sorted(shapes):
        (_version,) = struct.unpack_from("<I", buf, pos)
        (lod_levels,) = struct.unpack_from("<Q", buf, pos + 4)
        pos += 12
        for _ in range(lod_levels):
            (n,) = struct.unpack_from("<Q", buf, pos)
            pos += 8 + n
        (_tversion, size) = struct.unpack_from("<Ii", buf, pos)
        pos += 8
        dtype, dims = _desc(buf[pos : pos + size])
        pos += size
        dt = np.dtype(_DTYPES[dtype])
        count = int(np.prod(dims, dtype=np.int64))
        arr = np.frombuffer(buf, dt, count, pos).reshape(dims)
        pos += count * dt.itemsize
        if tuple(dims) != shapes[name]:
            msg = f"{params.name}: {name} has shape {dims}, the program says {shapes[name]}"
            raise ValueError(msg)
        out[name] = arr
    if pos != len(buf):
        msg = f"{params.name}: {len(buf) - pos} trailing bytes"
        raise ValueError(msg)
    return out


def _key(name: str) -> tuple[str, int, str]:
    m = re.fullmatch(r"(.*)_(\d+)\.(\w+)", name)
    if m is None:
        msg = f"unexpected Paddle parameter name {name!r}"
        raise ValueError(msg)
    return m[1], int(m[2]), m[3]


# Paddle layer type -> (torch parameter suffix per Paddle suffix)
_SUFFIX = {
    "conv2d": {"w_0": "weight", "b_0": "bias"},
    "conv2d_transpose": {"w_0": "weight", "b_0": "bias"},
    "batch_norm2d": {"w_0": "weight", "b_0": "bias", "w_1": "running_mean", "w_2": "running_var"},
    "batch_norm": {"w_0": "weight", "b_0": "bias", "w_1": "running_mean", "w_2": "running_var"},
    "linear": {"w_0": "weight", "b_0": "bias"},
    "layer_norm": {"w_0": "weight", "b_0": "bias"},
    "learnable_affine_block": {"w_0": "scale", "w_1": "bias"},
}


def state_dict(
    arrays: dict[str, np.ndarray],
    modules: Mapping[str, Sequence[tuple[str, torch.nn.Module]]],
) -> dict[str, np.ndarray]:
    """Torch state dict from Paddle arrays.

    Args:
        arrays: `read`'s output.
        modules: Paddle layer type -> the torch modules of that type `(qualified name, module)` in Paddle's creation
            order (the program's numbering, gaps for pruned layers skipped).

    Returns:
        Torch name -> array (Paddle's `Linear` weights transposed to torch's `[out, in]`).
    """
    groups: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for name, arr in arrays.items():
        kind, idx, suffix = _key(name)
        groups.setdefault(kind, {}).setdefault(idx, {})[suffix] = arr
    if set(groups) != set(modules):
        msg = f"layer types differ: Paddle {sorted(groups)}, torch {sorted(modules)}"
        raise ValueError(msg)
    out: dict[str, np.ndarray] = {}
    for kind, layers in groups.items():
        idx = sorted(layers)
        mods = modules[kind]
        if len(idx) != len(mods):
            msg = f"{kind}: {len(idx)} Paddle layers, {len(mods)} torch modules"
            raise ValueError(msg)
        for i, (qual, mod) in zip(idx, mods, strict=True):
            for suffix, arr in layers[i].items():
                attr = _SUFFIX[kind][suffix]
                value = arr.T if kind == "linear" and suffix == "w_0" else arr
                want = tuple(getattr(mod, attr).shape)
                if value.shape != want:
                    if value.size == int(np.prod(want)) and kind == "learnable_affine_block":
                        value = value.reshape(want)
                    else:
                        msg = f"{kind}_{i}.{suffix} {value.shape} != torch {qual}.{attr} {want}"
                        raise ValueError(msg)
                out[f"{qual}.{attr}" if qual else attr] = value
    return out
