"""fx graph rewrites of BN-folded fp16 CNN graphs onto the fvm kernels (kernels mode).

Order (`fuse_cnn`): conv [+ residual] [+ act] -> fused conv, depthwise convs, unfoldable BN [+ relu], concat buffers,
merged concat convs, max / avg pool, stem conv + pool, then expand 1x1 + depthwise in one kernel, EfficientViT
attention, grouped convs, depthwise with channel multiplier, half2 depthwise, RegNet grouped convs + SE scale,
small-C dense convs on tensor cores and narrow 1x1 convs on cuBLASLt.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import fx, nn

from fastforks._core.patch import is_base, on_load

from . import ops
from .fold import module_uses
from .graph import ADDS, act_of, attr, dce, finish, ok_conv, pair

if TYPE_CHECKING:
    from collections.abc import Sequence

KC = 32  # expanded-channel granularity of the fvm_mb kernels
MERGE_M = 8192  # cat2: one merged conv up to this many output pixels
XDW_MAX_CIN = 64  # expand + depthwise fusion only for narrow inputs (halo tile size)
DW2_MAX_TAPS = 7
_BIN = 2  # binary op argument count
_TC_MAX_ACT = 4  # conv_tc epilogue: none, relu, relu6, silu, hardswish


def _rup(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def _flag(gm: fx.GraphModule, name: str, value: int) -> None:
    """Plain (non-tensor) attribute on the GraphModule: pass counts and the input-format flags read by FastModel."""
    object.__setattr__(gm, name, value)


def _mods(gm: fx.GraphModule) -> dict[str, nn.Module]:
    return dict(gm.named_modules())


def _module(mods: dict[str, nn.Module], node: fx.Node) -> nn.Module | None:
    return mods.get(str(node.target)) if node.op == "call_module" else None


def _users(node: fx.Node) -> list[fx.Node]:
    return list(node.users)


def _take_act(node: fx.Node, mods: dict[str, nn.Module]) -> tuple[fx.Node, int]:
    """(last node, act code): the single activation consumer of `node` if there is one."""
    us = _users(node)
    if len(us) == 1:
        a = act_of(us[0], mods)
        if a is not None:
            return us[0], a
    return node, 0


# --- stem -------------------------------------------------------------------------------------------------------


def direct_stem(gm: fx.GraphModule) -> bool:
    """Graph input -> one Conv2d (C <= 4, R*S <= 25, K in STEM_K) [-> act]  ==>  ops.stem_direct (raw input)."""
    mods = _mods(gm)
    ph = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(ph) != 1 or len(ph[0].users) != 1:
        return False
    u = next(iter(ph[0].users))
    c = _module(mods, u)
    if not (
        isinstance(c, nn.Conv2d)
        and is_base(c, nn.Conv2d)
        and c.groups == 1
        and c.dilation == (1, 1)
        and c.padding_mode == "zeros"
        and not isinstance(c.padding, str)
        and c.in_channels <= 4  # noqa: PLR2004
        and c.kernel_size[0] * c.kernel_size[1] <= 25  # noqa: PLR2004
        and c.out_channels in ops.STEM_K
        and c.weight.dtype == torch.half
    ):
        return False
    _flag(gm, "stem_in_chans", c.in_channels)  # the kernel reads any C: FastModel checks the input against it
    c.register_buffer("w_t", c.weight.data.permute(2, 3, 1, 0).contiguous())
    last, act = _take_act(u, mods)
    g = gm.graph
    with g.inserting_before(last.next):
        w = g.get_attr(f"{u.target}.w_t")
        b = g.get_attr(f"{u.target}.bias") if c.bias is not None else None
        args = (ph[0], w, b, list(c.stride), list(c.padding), act)
        new = g.call_function(ops.stem_direct, args)
    last.replace_all_uses_with(new)
    new.args = args
    dce(g)
    return True


def pad_stem(gm: fx.GraphModule) -> bool:
    """Graph input -> one Conv2d with C < 8: zero-pad its weight to C = 8 (input from ops.input_nhwc8)."""
    mods = _mods(gm)
    ph = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(ph) != 1 or len(ph[0].users) != 1:
        return False
    u = next(iter(ph[0].users))
    c = _module(mods, u)
    if not isinstance(c, nn.Conv2d) or not ok_conv(c, stem_ok=True) or c.in_channels >= 8:  # noqa: PLR2004
        return False
    if module_uses(gm.graph)[str(u.target)] != 1:  # a second call reads fewer than 8 channels
        return False
    _flag(gm, "stem_in_chans", c.in_channels)  # the padded conv takes any C <= 8: FastModel checks the input against it
    w = torch.zeros(c.out_channels, 8, *c.kernel_size, dtype=c.weight.dtype, device=c.weight.device)
    w[:, : c.in_channels] = c.weight.data
    c.weight = nn.Parameter(w, requires_grad=False)
    c.in_channels = 8
    return True


# --- conv fusion ------------------------------------------------------------------------------------------------


def fold_layer_scale(gm: fx.GraphModule) -> int:
    """Conv2d -> mul(per-channel constant gamma)  ==>  gamma folded into the conv (timm LayerScale2d).

    Lets the following residual add fuse into the conv epilogue. Dropout calls (eval: identity) are dropped first.
    """
    mods = _mods(gm)
    g = gm.graph
    for node in list(g.nodes):
        if isinstance(_module(mods, node), nn.Dropout):
            node.replace_all_uses_with(node.args[0])
            g.erase_node(node)
    params = dict(gm.named_parameters())
    uses = module_uses(g)
    n = 0
    for node in list(g.nodes):
        m = _module(mods, node)
        if not is_base(m, nn.Conv2d) or len(node.users) != 1:
            continue
        u = next(iter(node.users))
        if not (
            u.op == "call_function" and u.target in {operator.mul, torch.mul} and len(u.args) == _BIN and not u.kwargs
        ):
            continue
        other = u.args[1] if u.args[0] is node else u.args[0]
        if not (isinstance(other, fx.Node) and other.op == "get_attr"):
            continue
        gam = params.get(str(other.target))
        if gam is None or gam.numel() != m.out_channels or uses[str(node.target)] != 1:
            continue
        with torch.no_grad():
            gv = gam.detach().reshape(-1).float()
            m.weight.data = (m.weight.float() * gv.view(-1, 1, 1, 1)).to(m.weight.dtype)
            if m.bias is not None:
                m.bias.data = (m.bias.float() * gv).to(m.bias.dtype)
        u.replace_all_uses_with(node)
        g.erase_node(u)
        n += 1
    dce(g)
    return n


def pad_conv_pairs(gm: fx.GraphModule) -> int:
    """Conv2d A (out % 8 != 0) -> [act] -> Conv2d B: zero-pad the channels between them to a multiple of 8.

    SE squeeze / excite convs, so both run on the fused conv. Exact: act(0) = 0, zero weights.
    """
    mods = _mods(gm)
    uses = module_uses(gm.graph)
    n = 0
    for node in list(gm.graph.nodes):
        a = _module(mods, node)
        if not is_base(a, nn.Conv2d) or a.groups != 1 or a.out_channels % 8 == 0 or len(node.users) != 1:
            continue
        u = next(iter(node.users))
        if act_of(u, mods) is not None and len(u.users) == 1:
            u = next(iter(u.users))
        b = _module(mods, u)
        if not is_base(b, nn.Conv2d) or b.groups != 1 or b.in_channels != a.out_channels:
            continue
        if uses[str(node.target)] != 1 or uses[str(u.target)] != 1:
            continue
        c = a.out_channels
        cp = _rup(c, 8)
        with torch.no_grad():
            wa = torch.zeros((cp, *a.weight.shape[1:]), dtype=a.weight.dtype, device=a.weight.device)
            wa[:c] = a.weight
            a.weight = nn.Parameter(wa, requires_grad=False)
            if a.bias is not None:
                ba = torch.zeros(cp, dtype=a.bias.dtype, device=a.bias.device)
                ba[:c] = a.bias
                a.bias = nn.Parameter(ba, requires_grad=False)
            wb = torch.zeros((b.out_channels, cp, *b.weight.shape[2:]), dtype=b.weight.dtype, device=b.weight.device)
            wb[:, :c] = b.weight
            b.weight = nn.Parameter(wb, requires_grad=False)
        a.out_channels = cp
        b.in_channels = cp
        n += 1
    return n


def fuse_graph(gm: fx.GraphModule) -> int:
    """Conv -> [add(other)] -> [act]  ==>  ops.conv_fused(x, w, b, other, pad, stride, act). Returns the count."""
    _flag(gm, "nls", fold_layer_scale(gm))
    _flag(gm, "npad", pad_conv_pairs(gm))
    _flag(gm, "raw_input", direct_stem(gm))
    _flag(gm, "pad8", False if gm.raw_input else pad_stem(gm))
    mods = _mods(gm)
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        conv = _module(mods, node)
        if not isinstance(conv, nn.Conv2d) or not ok_conv(conv):
            continue
        conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last)
        last, res = node, None
        us = _users(last)
        if (
            len(us) == 1
            and us[0].op == "call_function"
            and us[0].target in ADDS
            and len(us[0].args) == 2  # noqa: PLR2004
            and not us[0].kwargs
        ):
            u = us[0]
            other = u.args[1] if u.args[0] is last else u.args[0]
            if isinstance(other, fx.Node) and other is not last:
                res, last = other, u
        last, act = _take_act(last, mods)
        with g.inserting_before(last.next):
            w = g.get_attr(f"{node.target}.weight")
            b = g.get_attr(f"{node.target}.bias") if conv.bias is not None else None
            args = (node.args[0], w, b, res, list(conv.padding), list(conv.stride), act)
            new = g.call_function(ops.conv_fused, args)
        last.replace_all_uses_with(new)
        new.args = args
        n += 1
    finish(gm)
    return n


def _ok_dw(m: object) -> bool:
    return (
        is_base(m, nn.Conv2d)
        and m.groups == m.in_channels == m.out_channels
        and m.in_channels % 8 == 0
        and m.dilation == (1, 1)
        and m.padding_mode == "zeros"
        and not isinstance(m.padding, str)
        and m.weight.dtype == torch.half
    )


def fuse_dw(gm: fx.GraphModule) -> int:
    """Depthwise Conv2d [-> act]  ==>  ops.dwconv (weight as an [R, S, C] buffer)."""
    mods = _mods(gm)
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        conv = _module(mods, node)
        if not isinstance(conv, nn.Conv2d) or not _ok_dw(conv):
            continue
        conv.register_buffer("w_rsc", conv.weight.data[:, 0].permute(1, 2, 0).contiguous())
        last, act = _take_act(node, mods)
        with g.inserting_before(last.next):
            w = g.get_attr(f"{node.target}.w_rsc")
            b = g.get_attr(f"{node.target}.bias") if conv.bias is not None else None
            args = (node.args[0], w, b, list(conv.stride), list(conv.padding), act)
            new = g.call_function(ops.dwconv, args)
        last.replace_all_uses_with(new)
        new.args = args
        n += 1
    dce(g)
    return n


def _bn_affine(m: nn.BatchNorm2d, _keys: object) -> None:
    """The eval BatchNorm as one per-channel scale / shift (`_sc`, `_sh`; derived, so not in the state dict)."""
    rv, rm = m.running_var, m.running_mean
    assert rv is not None  # noqa: S101 - checked by the caller
    assert rm is not None  # noqa: S101
    with torch.no_grad():
        sc = (m.weight.float() if m.affine else 1) / torch.sqrt(rv.float() + m.eps)
        sh = (m.bias.float() if m.affine else 0) - rm.float() * sc
    m.register_buffer("_sc", sc.contiguous(), persistent=False)
    m.register_buffer("_sh", sh.contiguous(), persistent=False)


def replace_bn_act(gm: fx.GraphModule) -> int:
    """Remaining (unfoldable, e.g. pre-activation) eval BatchNorm2d [-> ReLU]  ==>  ops.affine_act."""
    mods = _mods(gm)
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        m = _module(mods, node)
        if not is_base(m, nn.BatchNorm2d) or m.num_features % 8:
            continue
        if m.running_var is None or m.running_mean is None:
            continue
        _bn_affine(m, None)
        on_load(m, _bn_affine)  # a later load of the running stats rederives scale / shift
        last, act = node, 0
        us = _users(node)
        if len(us) == 1 and act_of(us[0], mods) == 1:
            last, act = us[0], 1
        with g.inserting_before(last.next):
            s_ = g.get_attr(f"{node.target}._sc")
            t_ = g.get_attr(f"{node.target}._sh")
            new = g.call_function(ops.affine_act, (node.args[0], s_, t_, act))
        last.replace_all_uses_with(new)
        n += 1
    dce(g)
    return n


def fuse_into(gm: fx.GraphModule) -> int:
    """aten.copy_(buf[idx], conv_fused(...))  ==>  ops.conv_fused_into(..., out=buf[idx])  (DenseNet buffers)."""
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten.copy_.default:
            continue
        view, val = node.args[:2]
        if not (isinstance(val, fx.Node) and val.target is ops.conv_fused and len(val.users) == 1):
            continue
        with g.inserting_before(node):
            g.call_function(ops.conv_fused_into, (*val.args[:7], view))
        node.replace_all_uses_with(view)
        g.erase_node(node)
        g.erase_node(val)
        n += 1
    return n


def cat_buffer(x: torch.Tensor, w: torch.Tensor, pad: Sequence[int], stride: Sequence[int], ctot: int) -> torch.Tensor:
    """Channels-last fp16 buffer [N, ctot, P, Q] for the outputs of convs that read x."""
    n, _, h, wd = x.shape
    p = (h + 2 * pad[0] - w.shape[2]) // stride[0] + 1
    q = (wd + 2 * pad[1] - w.shape[3]) // stride[1] + 1
    return torch.empty((n, ctot, p, q), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)


def _cat_dim(node: fx.Node) -> object:
    return node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)


def _cat_parts(node: fx.Node) -> list[fx.Node] | None:
    if not (node.op == "call_function" and node.target is torch.cat and node.args and _cat_dim(node) == 1):
        return None
    out: list[fx.Node] = []
    for a in node.args[0]:  # ty: ignore[not-iterable] - torch.cat's tensor list
        if not isinstance(a, fx.Node):
            return None
        if a.target is torch.cat and len(a.users) == 1:
            sub = _cat_parts(a)
            if sub is None:
                return None
            out += sub
        elif a.target is ops.conv_fused and len(a.users) == 1:
            out.append(a)
        else:
            return None
    return out


def cat_into(gm: fx.GraphModule) -> int:
    """torch.cat([conv_fused | cat of conv_fused, ...], 1)  ==>  one buffer, every conv writes its channel slice.

    ops.conv_fused_into: no concatenation copies (SqueezeNet Fire, Inception A / C / E).
    """
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.cat:
            continue
        if any(u.target is torch.cat for u in node.users):  # handled as part of the outer cat
            continue
        ps = _cat_parts(node)
        if not ps or len(ps) < 2:  # noqa: PLR2004
            continue
        ks: list[int] = []
        for c in ps:
            w = attr(gm, c.args[1])
            if w is None or w.shape[0] % 8:
                break
            ks.append(w.shape[0])
        if len(ks) != len(ps):
            continue
        order = {x: i for i, x in enumerate(g.nodes)}
        first = min(ps, key=lambda c: order[c])
        with g.inserting_before(first):
            buf = g.call_function(cat_buffer, (first.args[0], first.args[1], first.args[4], first.args[5], sum(ks)))
        off = 0
        for c, k in zip(ps, ks, strict=True):
            with g.inserting_before(c):
                view = g.call_function(operator.getitem, (buf, (slice(None), slice(off, off + k))))
                g.call_function(ops.conv_fused_into, (*c.args[:7], view))
            off += k
        node.replace_all_uses_with(buf)
        dce(g)
        n += 1
    return n


def merge_cat_convs(gm: fx.GraphModule) -> int:  # noqa: C901, PLR0915 - one pattern match
    """torch.cat([conv_fused(x, ...), ...], 1) of convs reading the same x  ==>  one merged conv.

    Same stride / act, no residual, square odd kernels with 'same' padding: ONE conv with the smaller kernels
    zero-embedded at the centre of the largest (SqueezeNet Fire);
    two convs (1x1 + KxK): ops.cat2 decides merged / separate by the problem size.
    """
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.cat or _cat_dim(node) != 1:
            continue
        ps = node.args[0]
        if not isinstance(ps, (list, tuple)) or len(ps) < 2:  # noqa: PLR2004
            continue
        if not all(isinstance(p, fx.Node) and p.target is ops.conv_fused and len(p.users) == 1 for p in ps):
            continue
        nodes: list[fx.Node] = list(ps)
        x0 = nodes[0].args[0]
        ws = [attr(gm, p.args[1]) for p in nodes]
        if any(w is None for w in ws):
            continue
        wts: list[torch.Tensor] = ws  # ty: ignore[invalid-assignment] - checked above
        ok = all(
            p.args[0] is x0
            and p.args[3] is None
            and list(p.args[5]) == list(nodes[0].args[5])  # ty: ignore[invalid-argument-type]
            and p.args[6] == nodes[0].args[6]
            for p in nodes
        )
        ok = ok and all(
            w.shape[2] == w.shape[3] and w.shape[2] % 2 == 1 and list(p.args[4]) == [w.shape[2] // 2] * 2  # ty: ignore[invalid-argument-type]
            for w, p in zip(wts, nodes, strict=True)
        )
        if not ok or list(nodes[0].args[5]) != [1, 1]:  # ty: ignore[invalid-argument-type]
            continue
        kk = max(w.shape[2] for w in wts)
        ktot = sum(w.shape[0] for w in wts)
        wm = torch.zeros(ktot, wts[0].shape[1], kk, kk, dtype=wts[0].dtype, device=wts[0].device)
        bm = torch.zeros(ktot, dtype=wts[0].dtype, device=wts[0].device)
        off = 0
        for w, p in zip(wts, nodes, strict=True):
            o = (kk - w.shape[2]) // 2
            wm[off : off + w.shape[0], :, o : o + w.shape[2], o : o + w.shape[2]] = w
            b = attr(gm, p.args[2])
            if b is not None:
                bm[off : off + w.shape[0]] = b
            off += w.shape[0]
        name = f"_merged{n}"
        mod = nn.Module()
        mod.register_buffer("weight", wm.contiguous(memory_format=torch.channels_last))
        mod.register_buffer("bias", bm)
        gm.add_submodule(name, mod)
        two = len(nodes) == 2 and wts[0].shape[2] == 1 and wts[1].shape[2] == kk  # noqa: PLR2004
        with g.inserting_before(node):
            wn = g.get_attr(f"{name}.weight")
            bn = g.get_attr(f"{name}.bias")
            p0, p1 = nodes[0], nodes[-1]
            if two:
                args = (x0, wn, bn, p0.args[1], p0.args[2], p1.args[1], p1.args[2], p0.args[6], MERGE_M)
                new = g.call_function(ops.cat2, args)
            else:
                args = (x0, wn, bn, None, [kk // 2] * 2, list(p0.args[5]), p0.args[6])  # ty: ignore[invalid-argument-type]
                new = g.call_function(ops.conv_fused, args)
        node.replace_all_uses_with(new)
        g.erase_node(node)
        for p in nodes:
            if not p.users:
                g.erase_node(p)
        n += 1
    finish(gm)
    return n


def replace_maxpool(gm: fx.GraphModule) -> int:
    """nn.MaxPool2d (no dilation / indices)  ==>  ops.maxpool."""
    mods = _mods(gm)
    n = 0
    for node in list(gm.graph.nodes):
        m = _module(mods, node)
        if not is_base(m, nn.MaxPool2d) or m.dilation not in {1, (1, 1)} or m.return_indices:
            continue
        st = m.stride if m.stride is not None else m.kernel_size
        with gm.graph.inserting_before(node):
            args = (node.args[0], pair(m.kernel_size), pair(st), pair(m.padding), bool(m.ceil_mode))
            new = gm.graph.call_function(ops.maxpool, args)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        n += 1
    return n


_AVG_ARGS = ("input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override")


def replace_avgpool(gm: fx.GraphModule) -> int:
    """nn.AvgPool2d / F.avg_pool2d (constant geometry, no ceil_mode / divisor_override)  ==>  ops.avgpool."""
    mods = _mods(gm)
    n = 0
    for node in list(gm.graph.nodes):
        geo = None
        m = _module(mods, node)
        if is_base(m, nn.AvgPool2d):
            if not m.ceil_mode and m.divisor_override is None:
                st = m.stride if m.stride is not None else m.kernel_size
                geo = (m.kernel_size, st, m.padding, m.count_include_pad)
        elif node.op == "call_function" and node.target is F.avg_pool2d:
            a = dict(zip(_AVG_ARGS, node.args, strict=False))
            a.update(node.kwargs)
            if (
                not a.get("ceil_mode", False)
                and a.get("divisor_override") is None
                and all(not isinstance(a.get(k), fx.Node) for k in ("kernel_size", "stride", "padding"))
            ):
                geo = (
                    a["kernel_size"],
                    a.get("stride") or a["kernel_size"],
                    a.get("padding", 0),
                    a.get("count_include_pad", True),
                )
        if geo is None:
            continue
        with gm.graph.inserting_before(node):
            args = (node.args[0], pair(geo[0]), pair(geo[1]), pair(geo[2]), bool(geo[3]))
            new = gm.graph.call_function(ops.avgpool, args)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        n += 1
    return n


def fuse_stem_pool(gm: fx.GraphModule) -> int:
    """ops.stem_direct -> ops.maxpool (square, only consumer)  ==>  ops.stem_pool."""
    g = gm.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target is not ops.stem_direct or len(node.users) != 1:
            continue
        u = next(iter(node.users))
        if u.op != "call_function" or u.target is not ops.maxpool:
            return 0
        _, k, s, p, ceil = u.args
        if k[0] != k[1] or s[0] != s[1] or p[0] != p[1]:
            return 0
        x, w, b, st, pad, act = node.args
        with g.inserting_before(u):
            new = g.call_function(ops.stem_pool, (x, w, b, st, pad, act, k[0], s[0], p[0], bool(ceil)))
        u.replace_all_uses_with(new)
        g.erase_node(u)
        g.erase_node(node)
        g.lint()
        gm.recompile()
        return 1
    return 0


def fuse_all(gm: fx.GraphModule) -> int:
    """Base rewrites (fused conv, depthwise, BN+act, concat buffers, pools, stem + pool); returns the conv count."""
    n = fuse_graph(gm)
    n += fuse_dw(gm)
    _flag(gm, "nbnact", replace_bn_act(gm))
    _flag(gm, "ninto", fuse_into(gm))
    _flag(gm, "nmergecat", merge_cat_convs(gm))
    _flag(gm, "ncatinto", cat_into(gm))
    _flag(gm, "npool", replace_maxpool(gm) + replace_avgpool(gm))
    _flag(gm, "nstempool", fuse_stem_pool(gm))
    gm.graph.lint()
    gm.recompile()
    return n


# --- fvm_mb passes ----------------------------------------------------------------------------------------------


def _is_pw(gm: fx.GraphModule, node: object) -> bool:
    """ops.conv_fused 1x1 / stride 1 / pad 0."""
    if not (isinstance(node, fx.Node) and node.op == "call_function" and node.target is ops.conv_fused):
        return False
    w = attr(gm, node.args[1])
    return w is not None and w.shape[2:] == (1, 1) and list(node.args[4]) == [0, 0] and list(node.args[5]) == [1, 1]  # ty: ignore[invalid-argument-type]


def fuse_xdw(gm: fx.GraphModule) -> int:
    """conv_fused 1x1 (expand, act1, no residual) -> dwconv (act2)  ==>  ops.xdw.

    The expanded tensor lives only in shared memory; the project conv stays a fused conv. Only for
    Cin <= XDW_MAX_CIN, depthwise kernels up to XDW_MAX_TAPS x XDW_MAX_TAPS and relu-family activations.
    """
    g = gm.graph
    n = 0
    for d in list(g.nodes):
        if not (d.op == "call_function" and d.target is ops.dwconv):
            continue
        e = d.args[0]
        if not (_is_pw(gm, e) and isinstance(e, fx.Node) and len(e.users) == 1 and e.args[3] is None):
            continue
        w1, b1 = attr(gm, e.args[1]), attr(gm, e.args[2])
        wd, bd = attr(gm, d.args[1]), attr(gm, d.args[2])
        if w1 is None or b1 is None or wd is None or bd is None:
            continue
        r, s, ch = wd.shape
        cin = w1.shape[1]
        if cin % 8 or ch % 8 or cin > XDW_MAX_CIN or r > ops.XDW_MAX_TAPS or s > ops.XDW_MAX_TAPS:
            continue
        if int(e.args[6]) in {3, 5, 6} or int(d.args[5]) in {3, 5, 6}:
            continue
        cinp, ep = _rup(cin, 16), _rup(ch, KC)
        dev, dt = w1.device, w1.dtype
        m = nn.Module()
        w1p = torch.zeros(ep, cinp, device=dev, dtype=dt)
        w1p[:ch, :cin] = w1.reshape(ch, cin)
        b1p = torch.zeros(ep, device=dev, dtype=dt)
        b1p[:ch] = b1
        wdp = torch.zeros(r, s, ep, device=dev, dtype=dt)
        wdp[:, :, :ch] = wd
        bdp = torch.zeros(ep, device=dev, dtype=dt)
        bdp[:ch] = bd
        for k, v in (("w1", w1p), ("b1", b1p), ("wd", wdp), ("bd", bdp)):
            m.register_buffer(k, v)
        name = f"xdw{n}"
        gm.add_submodule(name, m)
        with g.inserting_before(d):
            ga = [g.get_attr(f"{name}.{k}") for k in ("w1", "b1", "wd", "bd")]
            acts = [int(e.args[6]), int(d.args[5])]
            new = g.call_function(ops.xdw, (e.args[0], *ga, ch, list(d.args[3]), list(d.args[4]), acts, 0))
        d.replace_all_uses_with(new)
        g.erase_node(d)
        g.erase_node(e)
        n += 1
    finish(gm)
    return n


def fuse_gconv(gm: fx.GraphModule) -> int:
    """Grouped (not depthwise) Conv2d [-> act], equal in / out group width % 16 == 0  ==>  ops.gconv."""
    mods = _mods(gm)
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        m = _module(mods, node)
        if not (
            isinstance(m, nn.Conv2d)
            and is_base(m, nn.Conv2d)
            and 1 < m.groups < m.in_channels
            and m.in_channels == m.out_channels
            and (m.in_channels // m.groups) % 16 == 0
            and m.dilation == (1, 1)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
            and m.weight.dtype == torch.half
        ):
            continue
        m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        last, act = _take_act(node, mods)
        with g.inserting_before(last.next):
            w = g.get_attr(f"{node.target}.weight")
            b = g.get_attr(f"{node.target}.bias") if m.bias is not None else None
            args = (node.args[0], w, b, list(m.stride), list(m.padding), m.groups, act)
            new = g.call_function(ops.gconv, args)
        last.replace_all_uses_with(new)
        new.args = args
        n += 1
    finish(gm)
    return n


def fuse_dwmult(gm: fx.GraphModule) -> int:
    """Conv2d with groups == C_in and C_out = mult * C_in (mult > 1) [-> act]  ==>  ops.dwmult (FastViT)."""
    mods = _mods(gm)
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        m = _module(mods, node)
        if not (
            isinstance(m, nn.Conv2d)
            and is_base(m, nn.Conv2d)
            and m.groups == m.in_channels
            and m.out_channels % m.in_channels == 0
            and m.out_channels > m.in_channels
            and m.out_channels % 8 == 0
            and m.dilation == (1, 1)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
            and m.weight.dtype == torch.half
        ):
            continue
        m.register_buffer("w_rsk", m.weight.data[:, 0].permute(1, 2, 0).contiguous())
        last, act = _take_act(node, mods)
        with g.inserting_before(last.next):
            w = g.get_attr(f"{node.target}.w_rsk")
            b = g.get_attr(f"{node.target}.bias") if m.bias is not None else None
            args = (node.args[0], w, b, list(m.stride), list(m.padding), act)
            new = g.call_function(ops.dwmult, args)
        last.replace_all_uses_with(new)
        new.args = args
        n += 1
    finish(gm)
    return n


# --- fvm_r3 passes ----------------------------------------------------------------------------------------------


def fuse_regnet(gm: fx.GraphModule) -> tuple[int, int]:
    """sigmoid(s) * x  ==>  ops.se_scale; ops.gconv  ==>  ops.gconv2. Returns (n_gconv2, n_se)."""
    mods = _mods(gm)
    g = gm.graph
    ng = nse = 0

    def is_sig(n: object) -> bool:
        return isinstance(n, fx.Node) and (
            isinstance(_module(mods, n), nn.Sigmoid)
            or (n.op == "call_function" and n.target in {torch.sigmoid, F.sigmoid})
        )

    for node in list(g.nodes):  # SE scale first (the pattern needs the sigmoid node)
        if not (node.op == "call_function" and node.target in {operator.mul, torch.mul} and len(node.args) == 2):  # noqa: PLR2004
            continue
        a, b = node.args
        s_, x_ = (a, b) if is_sig(a) else (b, a) if is_sig(b) else (None, None)
        if not isinstance(s_, fx.Node) or len(s_.users) != 1 or not isinstance(x_, fx.Node):
            continue
        with g.inserting_before(node):
            new = g.call_function(ops.se_scale, (x_, s_.args[0]))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        g.erase_node(s_)
        nse += 1
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not ops.gconv:
            continue
        with g.inserting_before(node):
            y = g.call_function(ops.gconv2, tuple(node.args))
        node.replace_all_uses_with(y)
        g.erase_node(node)
        ng += 1
    finish(gm)
    return ng, nse


def to_conv_tc(gm: fx.GraphModule) -> int:
    """Small-C KxK fused convs  ==>  ops.conv_tc / conv_tc_into (gconv2 kernel: faster than CUTLASS for small C).

    K > 1, no residual, act <= 4, C % 16 == 0, C <= TC_MAX_C, K % 16 == 0.
    """
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in {ops.conv_fused, ops.conv_fused_into}:
            continue
        a = node.args
        w = attr(gm, a[1])
        if (
            w is None
            or w.shape[2:] == (1, 1)
            or a[3] is not None
            or a[6] > _TC_MAX_ACT
            or w.shape[1] % 16
            or w.shape[1] > ops.TC_MAX_C
            or w.shape[0] % 16
        ):
            continue
        with g.inserting_before(node):
            if node.target is ops.conv_fused_into:
                new = g.call_function(ops.conv_tc_into, (a[0], a[1], a[2], list(a[5]), list(a[4]), a[6], a[7]))
            else:
                new = g.call_function(ops.conv_tc, (a[0], a[1], a[2], list(a[5]), list(a[4]), a[6]))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        n += 1
    g.lint()
    gm.recompile()
    return n


def to_pw(gm: fx.GraphModule) -> int:
    """Narrow 1x1 stride-1 fused convs  ==>  ops.pw / pw_into.

    Act none / relu, no residual, min(C, K) <= 256; cuBLASLt for big narrow GEMMs, fused conv otherwise (decided
    per call by the size rule).
    """
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in {ops.conv_fused, ops.conv_fused_into}:
            continue
        a = node.args
        w = attr(gm, a[1])
        if (
            w is None
            or w.shape[2:] != (1, 1)
            or list(a[4]) != [0, 0]
            or list(a[5]) != [1, 1]
            or a[3] is not None
            or a[6] not in {0, 1}
            or min(w.shape[0], w.shape[1]) > ops.PW_MAX_NARROW
        ):
            continue
        with g.inserting_before(node):
            if node.target is ops.conv_fused_into:
                new = g.call_function(ops.pw_into, (a[0], a[1], a[2], None, a[6], a[7]))
            else:
                new = g.call_function(ops.pw, (a[0], a[1], a[2], None, a[6]))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        n += 1
    g.lint()
    gm.recompile()
    return n


def to_dw2(gm: fx.GraphModule) -> int:
    """ops.dwconv (taps <= 7)  ==>  ops.dwconv2 (half2 accumulation)."""
    n = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target is ops.dwconv:
            w = attr(gm, node.args[1])
            if w is not None and max(w.shape[:2]) > DW2_MAX_TAPS:  # weight [R, S, C]
                continue
            node.target = ops.dwconv2
            n += 1
    gm.recompile()
    return n
