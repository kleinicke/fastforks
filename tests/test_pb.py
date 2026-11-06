"""HITNet's GraphDef reader on hand-made protobuf bytes."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from fastforks.stereo.models.hitnet.net.pb import read_graph


def varint(x: int) -> bytes:
    out = bytearray()
    while True:
        b, x = x & 0x7F, x >> 7
        out.append(b | (0x80 if x else 0))
        if not x:
            return bytes(out)


def field(num: int, value: bytes | int, wire: int = 2) -> bytes:
    key = varint(num << 3 | wire)
    if isinstance(value, int):  # wire 0
        return key + varint(value)
    return key + (value if wire == 5 else varint(len(value)) + value)


def const(name: str, tensor: bytes) -> bytes:
    value = field(1, b"value") + field(2, field(8, tensor))
    return field(1, field(1, name.encode()) + field(2, b"Const") + field(5, value))


def shape(*dims: int) -> bytes:
    return field(2, b"".join(field(2, field(1, d, wire=0)) for d in dims))


def test_packed_unpacked_and_broadcast_values(tmp_path):
    floats = [1.5, -2.0, 3.25]
    packed = field(1, 1, wire=0) + shape(3) + field(5, struct.pack("<3f", *floats))
    unpacked = field(1, 1, wire=0) + shape(3) + b"".join(field(5, struct.pack("<f", f), wire=5) for f in floats)
    ints = field(1, 3, wire=0) + shape(2) + field(7, 7, wire=0) + field(7, 2**64 - 5, wire=0)
    splat = field(1, 1, wire=0) + shape(2, 2) + field(5, struct.pack("<f", 0.5))
    path = tmp_path / "g.pb"
    path.write_bytes(const("p", packed) + const("u", unpacked) + const("i", ints) + const("s", splat))
    nodes = read_graph(path)
    np.testing.assert_array_equal(nodes["p"].value, floats)
    np.testing.assert_array_equal(nodes["u"].value, floats)
    np.testing.assert_array_equal(nodes["i"].value, [7, -5])
    np.testing.assert_array_equal(nodes["s"].value, np.full((2, 2), 0.5, np.float32))


@pytest.mark.parametrize(
    "data",
    [
        const("big", field(1, 1, wire=0) + shape(1 << 20, 1 << 20)),  # would allocate 4 TiB
        const("neg", field(1, 1, wire=0) + shape(2**64 - 1)),  # dim -1
        field(1, b"x" * 10)[:-3],  # truncated
        const("dt", field(1, 7, wire=0)),  # string constant
        field(1, field(1, b"\xff\xfe")),  # bad UTF-8 name
    ],
)
def test_malformed_graphs_are_value_errors(tmp_path, data):
    path = tmp_path / "bad.pb"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="not a readable GraphDef"):
        read_graph(path)
