"""Reader for TensorFlow frozen graphs (`GraphDef` protobuf files): constants and the node list, without TensorFlow.

Only what HITNet's loader needs is decoded: per node its name, op and inputs, and the value of `Const` nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

# tensorflow/core/framework/types.proto
_DTYPES = {1: np.float32, 2: np.float64, 3: np.int32, 9: np.int64}
_VALUE_FIELDS = {5: ("<f4", 4), 6: ("<f8", 8)}  # TensorProto.float_val / double_val (packed fixed-width)
_VARINT_FIELDS = {7: np.int32, 10: np.int64}  # TensorProto.int_val / int64_val (packed varints)
_MAX_ELEMENTS = 1 << 28  # fixed: bound on one constant (1 GiB of float32); HITNet's largest has 1.2e5


@dataclass(frozen=True)
class Node:
    """One graph node: `inputs` are the names it reads (`name`, `name:1`, `^control`)."""

    name: str
    op: str
    inputs: tuple[str, ...]
    value: np.ndarray | None  # the tensor of a `Const` node


def read_graph(path: Path) -> dict[str, Node]:
    """Nodes of the frozen graph in `path` by name.

    Raises:
        ValueError: Not a `GraphDef` this reader understands (truncated data, unsupported constant type or shape).
    """
    data = path.read_bytes()
    try:
        return {n.name: n for n in (_node(body) for field, body in _fields(data) if field == 1)}
    except (IndexError, KeyError, ValueError) as e:  # UnicodeDecodeError is a ValueError
        msg = f"{path}: not a readable GraphDef ({e})"
        raise ValueError(msg) from e


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = out = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf: bytes):
    """Fields of one message: length-delimited ones as `bytes`, varints as `int`, fixed-width ones as raw bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire in (1, 2, 5):
            n, i = _varint(buf, i) if wire == 2 else (8 if wire == 1 else 4, i)
            if i + n > len(buf):
                msg = f"truncated field {field}"
                raise ValueError(msg)
            value, i = buf[i : i + n], i + n
        else:
            msg = f"unsupported protobuf wire type {wire}"
            raise ValueError(msg)
        yield field, value


def _node(body: bytes) -> Node:
    """NodeDef: name = 1, op = 2, input = 3, attr = 5 (map entry: key = 1, AttrValue = 2; tensor = 8)."""
    name = op = ""
    inputs: list[str] = []
    value = None
    for field, v in _fields(body):
        if field == 1:
            name = v.decode()
        elif field == 2:
            op = v.decode()
        elif field == 3:
            inputs.append(v.decode())
        elif field == 5 and op == "Const":
            entry = dict(_fields(v))
            if entry.get(1) == b"value":
                tensor = dict(_fields(entry[2])).get(8)
                value = _tensor(tensor) if tensor is not None else None
    return Node(name, op, tuple(inputs), value)


def _tensor(body: bytes) -> np.ndarray:
    """TensorProto: dtype = 1, tensor_shape = 2, tensor_content = 4, float_val = 5, double_val = 6, int_val = 7,
    int64_val = 10. A short `*_val` list is the value repeated over the shape (TF's constant folding); repeated values
    may be packed or one field per element."""
    dtype, shape, content, parts = np.float32, (), None, []
    for field, v in _fields(body):
        if field == 1:
            if v not in _DTYPES:
                msg = f"unsupported constant dtype {v}"
                raise ValueError(msg)
            dtype = _DTYPES[v]
        elif field == 2:
            shape = tuple(next((s for f, s in _fields(d) if f == 1), 0) for g, d in _fields(v) if g == 2)
        elif field == 4:
            content = v
        elif field in _VALUE_FIELDS:
            parts.append(np.frombuffer(v, dtype=_VALUE_FIELDS[field][0]))
        elif field in _VARINT_FIELDS:
            parts.append(np.array(_packed_varints(v), dtype=np.int64))
    n = math.prod(shape)
    if any(d < 0 for d in shape) or n > _MAX_ELEMENTS:
        msg = f"constant shape {shape} out of range"
        raise ValueError(msg)
    if content is not None:
        return np.frombuffer(content, dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape).astype(dtype)
    if not parts:
        return np.zeros(shape, dtype)
    vals = np.concatenate(parts).astype(dtype)
    return np.resize(vals, n).reshape(shape) if vals.size != n else vals.reshape(shape)


def _packed_varints(v: bytes | int) -> list[int]:
    """Signed values of a packed varint list (`bytes`) or of one unpacked varint (`int`)."""
    out, i = [v] if isinstance(v, int) else [], 0
    while isinstance(v, bytes) and i < len(v):
        x, i = _varint(v, i)
        out.append(x)
    return [x - (1 << 64) if x >= 1 << 63 else x for x in out]
