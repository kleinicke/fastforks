"""torch.fx helpers shared by the passes."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import fx, nn

from fastforks._core.patch import base_type, is_base

from . import ops

if TYPE_CHECKING:
    from collections.abc import Callable

# activation code of the fused ops: 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu, 6 gelu (tanh)
ACTS: dict[object, int] = {
    nn.ReLU: 1,
    nn.ReLU6: 2,
    nn.SiLU: 3,
    nn.Hardswish: 4,
    F.relu: 1,
    F.relu6: 2,
    F.silu: 3,
    F.hardswish: 4,
    torch.relu: 1,
}
ADDS = (operator.add, torch.add, operator.iadd)


# ops that write into an argument: their graph nodes have no users but must stay
INPLACE: frozenset[Callable[..., object]] = frozenset(
    {ops.conv_fused_into, ops.conv_tc_into, ops.pw_into, ops.chan_gather, ops.mbconv_into}
)


def dce(g: fx.Graph) -> None:
    """Dead-code elimination that keeps the in-place ops."""
    g.eliminate_dead_code(is_impure_node=lambda n: n.is_impure() or n.target in INPLACE)


def act_of(node: fx.Node, mods: dict[str, nn.Module]) -> int | None:
    """Activation code of `node` if it is a supported activation call, else None."""
    if node.op == "call_module":
        m = mods[str(node.target)]
        if is_base(m, nn.GELU):
            return 5 if m.approximate == "none" else 6
        return ACTS.get(base_type(m))
    if node.op == "call_function":
        return ACTS.get(node.target)
    if node.op == "call_method" and node.target in {"relu", "relu_"}:
        return 1
    return None


def attr(gm: fx.GraphModule, node: object) -> torch.Tensor | None:
    """Tensor behind a `get_attr` node (parameter or buffer); None for None / other nodes."""
    if not isinstance(node, fx.Node) or node.op != "get_attr":
        return None
    mod, _, name = str(node.target).rpartition(".")
    t = getattr(gm.get_submodule(mod) if mod else gm, name)
    return t if isinstance(t, torch.Tensor) else None


def finish(gm: fx.GraphModule) -> None:
    """Dce + lint + recompile."""
    dce(gm.graph)
    gm.graph.lint()
    gm.recompile()


def ok_conv(m: object, *, stem_ok: bool = False) -> bool:
    """Dense fp16 Conv2d the fused conv covers (groups 1, no dilation, zero padding, C / K multiples of 8)."""
    return (
        is_base(m, nn.Conv2d)
        and m.groups == 1
        and m.dilation == (1, 1)
        and m.padding_mode == "zeros"
        and not isinstance(m.padding, str)
        and (m.in_channels % 8 == 0 or (stem_ok and m.in_channels < 8))  # noqa: PLR2004
        and m.out_channels % 8 == 0
        and m.weight.dtype == torch.half
    )


def pair(v: int | tuple[int, ...] | list[int]) -> list[int]:
    """Two-element list of an int or pair."""
    return list(v) if isinstance(v, (tuple, list)) else [v, v]
