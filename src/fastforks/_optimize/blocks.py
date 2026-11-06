"""Architecture-specific rewrites of CNN blocks.

torchvision DenseNet (concat-free), torchvision ShuffleNetV2 (one fused kernel per unit, kernels) and timm / MIT
EfficientViT LiteMLA (kernels).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import torch
from torch import fx, nn

from fastforks._core.patch import override

from . import ops

_Blk = Any  # a timm / torchvision / transformers block patched with our buffers: untyped to the checker


def _modules(model: nn.Module) -> Iterator[_Blk]:
    return model.modules()


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

_CL = torch.channels_last


# --- DenseNet (torchvision _DenseBlock), plain PyTorch -------------------------------------------------------------


def _copy(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # the aten op (not Tensor.copy_): fx records it as an impure call_function, dead-code elimination keeps it
    return torch.ops.aten.copy_.default(dst, src)


def _dense_block(self: _Blk, x: torch.Tensor) -> torch.Tensor:
    layers = list(self.values())
    k = layers[0].conv2.out_channels
    n, c0, h, w = x.shape
    buf = torch.empty((n, c0 + k * len(layers), h, w), device=x.device, dtype=x.dtype, memory_format=_CL)
    _copy(buf[:, :c0], x)
    c = c0
    for lay in layers:
        y = lay.relu1(lay.norm1(buf[:, :c]))
        y = lay.relu2(lay.norm2(lay.conv1(y)))
        _copy(buf[:, c : c + k], lay.conv2(y))
        c += k
    return buf


def patch_densenet(model: nn.Module) -> int:
    """One [N, C_total, H, W] buffer per dense block instead of O(L^2) concatenations; returns the count."""
    n = 0
    for m in _modules(model):
        if type(m).__name__ == "_DenseBlock":
            override(m, _dense_block)
            n += 1
    return n


# --- ShuffleNetV2 (torchvision), kernels ------------------------------------------------------------------------------
# Every unit writes a fresh buffer with the physical layout [P1 | P2 | B2 | B1] (each part Q = ceil8(C / 4) channels,
# zero padded): P1 + P2 = the logical channels the NEXT unit passes through, B2 + B1 = the ones its branch reads. Every
# physical channel is tracked as a logical index (-1 = zero pad); the first conv of each consumer is built with permuted
# input columns, so channel split + shuffle cost nothing at run time.


def _fold64(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    rv, rm = bn.running_var, bn.running_mean
    assert rv is not None  # noqa: S101 - torchvision eval BN
    assert rm is not None  # noqa: S101
    sc = bn.weight.double() / torch.sqrt(rv.double() + bn.eps)
    w = conv.weight.double() * sc.view(-1, 1, 1, 1)
    b = bn.bias.double() - rm.double() * sc
    if conv.bias is not None:
        b = b + conv.bias.double() * sc
    return w, b


def _r(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def _pw64(  # noqa: PLR0913, PLR0917 - weight + bias + two index maps + two sizes
    w: torch.Tensor, b: torch.Tensor, rows: Sequence[int], cols: Sequence[int], rows_pad: int, cols_pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense 1x1 weight [rows_pad, cols_pad] from logical w [O, I]; rows[i] / cols[j] = logical index or -1."""
    wo = torch.zeros(rows_pad, cols_pad, dtype=torch.float64)
    bo = torch.zeros(rows_pad, dtype=torch.float64)
    w = w.reshape(w.shape[0], w.shape[1])
    r, c = torch.tensor(rows), torch.tensor(cols)
    ri, ci = (r >= 0).nonzero().flatten(), (c >= 0).nonzero().flatten()
    wo[ri[:, None], ci[None, :]] = w[r[ri][:, None], c[ci][None, :]]
    bo[ri] = b[r[ri]]
    return wo, bo


def _dw64(w: torch.Tensor, b: torch.Tensor, chans: Sequence[int], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise weight [R, S, pad] + bias from logical w [C, 1, R, S]; chans[i] = logical channel or -1."""
    wd = torch.zeros(*w.shape[2:], pad, dtype=torch.float64)
    bd = torch.zeros(pad, dtype=torch.float64)
    c = torch.tensor(chans)
    ci = (c >= 0).nonzero().flatten()
    wd[:, :, ci] = w[c[ci], 0].permute(1, 2, 0)
    bd[ci] = b[c[ci]]
    return wd, bd


def _halves(h: int, q: int, f0: Callable[[int], int], f1: Callable[[int], int]) -> list[int]:
    return [f0(i) if i < h else -1 for i in range(q)] + [f1(i) if i < h else -1 for i in range(q)]


class FastShuffleNetV2(nn.Module):
    """torchvision ShuffleNetV2 rebuilt on fvm kernels.

    big=False: one fused mbconv kernel per unit (best at small batch); big=True: same layout and weights, standard
    kernels per unit (fused 1x1 reading the channel slice in place, depthwise, fused 1x1 into the output slice, gather).
    """

    stem_wt: torch.Tensor
    stem_b: torch.Tensor
    w5: torch.Tensor
    b5: torch.Tensor

    raw_input = True  # the stem reads the fp32 NCHW input directly

    def __init__(self, tv: _Blk, *, big: bool = False) -> None:  # noqa: PLR0915 - one layout walk
        """Build from an eval torchvision ShuffleNetV2 (CPU copy, fp32)."""
        super().__init__()
        self.big = big
        dt = torch.float16
        w, b = _fold64(tv.conv1[0], tv.conv1[1])
        self.stem_in_chans = w.shape[1]  # FastModel checks the input channels (the stem kernel reads any)
        self.register_buffer("stem_wt", w.to(dt).permute(2, 3, 1, 0).contiguous())  # [R, S, 3, K]
        self.register_buffer("stem_b", b.to(dt))
        pin = list(range(w.shape[0]))  # physical -> logical channel of the current buffer
        self.units = nn.ModuleList()
        self.meta: list[tuple[bool, int, int, int]] = []  # per unit: (downsample, Q, cin_off, cin)
        for stage in (tv.stage2, tv.stage3, tv.stage4):
            for blk in stage:
                u: _Blk = nn.Module()
                b2 = blk.branch2
                w21, b21 = _fold64(b2[0], b2[1])
                wd2, bd2 = _fold64(b2[3], b2[4])
                w23, b23 = _fold64(b2[5], b2[6])
                bf = w23.shape[0]
                h = bf // 2
                q = _r(h, 8)
                cphys = len(pin)
                ds = blk.stride > 1
                ep = _r(bf, 32)
                rows2 = _halves(h, q, lambda i: i, lambda i, h=h: h + i)  # P2 slots then B2 slots
                if ds:
                    cols, cin_off, cin = list(pin), 0, cphys
                else:
                    half = cphys // 2
                    clog = sum(1 for p in pin if p >= 0)
                    cols = [p - clog // 2 if p >= 0 else -1 for p in pin[half:]]
                    cin_off, cin = half, cphys - half
                w1, b1 = _pw64(w21, b21, list(range(bf)), cols, ep, _r(cin, 16))
                wd, bd = _dw64(wd2, bd2, list(range(bf)), ep)
                w3, b3 = _pw64(w23, b23, rows2, list(range(bf)), 2 * q, ep)
                for k, v in (("w1", w1), ("b1", b1), ("wd", wd), ("bd", bd), ("w2", w3), ("b2", b3)):
                    u.register_buffer(k, v.to(dt))
                u.register_buffer("colmap", torch.tensor([q + 8 * j for j in range(2 * q // 8)], dtype=torch.int32))
                outer = [8 * j for j in range(q // 8)] + [3 * q + 8 * j for j in range(q // 8)]
                if ds:
                    wd1, bd1 = _fold64(blk.branch1[0], blk.branch1[1])
                    w12, b12 = _fold64(blk.branch1[2], blk.branch1[3])
                    ep1 = _r(cphys, 32)
                    wd1p, bd1p = _dw64(wd1, bd1, list(pin), ep1)
                    w12p, b12p = _pw64(w12, b12, rows2, list(pin) + [-1] * (ep1 - cphys), 2 * q, ep1)
                    for k, v in (("wd1", wd1p), ("bd1", bd1p), ("w12", w12p), ("b12", b12p)):
                        u.register_buffer(k, v.to(dt))
                    u.register_buffer("colmap1", torch.tensor(outer, dtype=torch.int32))
                else:  # passthrough x1 = input logical 0 .. C/2-1 -> P1 (x1[i], i < h) and B1 (x1[h + i])
                    where = {p: k for k, p in enumerate(pin) if p >= 0}
                    gsrc = _halves(h, q, lambda i, wh=where: wh[i], lambda i, wh=where, h=h: wh[h + i])
                    u.register_buffer("gsrc", torch.tensor(gsrc, dtype=torch.int32))
                    u.register_buffer("gdst", torch.tensor(outer, dtype=torch.int32))
                if big:  # standard-kernel weights: [K, C, 1, 1] / [R, S, C] views of the same values
                    u.register_buffer("k1", u.w1[:, :cin].reshape(ep, cin, 1, 1).contiguous(memory_format=_CL))
                    u.register_buffer("k3", u.w2.reshape(2 * q, ep, 1, 1).contiguous(memory_format=_CL))
                    if ds:
                        u.register_buffer("kd1", u.wd1[:, :, :cphys].contiguous())
                        u.register_buffer("bd1c", u.bd1[:cphys].contiguous())
                        w12c = u.w12[:, :cphys]
                        u.register_buffer("k12a", w12c[:q].reshape(q, cphys, 1, 1).contiguous(memory_format=_CL))
                        u.register_buffer("k12b", w12c[q:].reshape(q, cphys, 1, 1).contiguous(memory_format=_CL))
                        u.register_buffer("b12a", u.b12[:q].contiguous())
                        u.register_buffer("b12b", u.b12[q:].contiguous())
                self.units.append(u)
                self.meta.append((ds, q, cin_off, cin))
                # output physical map: P1 -> 2i, P2 -> 2i+1, B2 -> 2(h+i)+1, B1 -> 2(h+i)
                pin = _halves(h, q, lambda i: 2 * i, lambda i: 2 * i + 1) + _halves(
                    h, q, lambda i, h=h: 2 * (h + i) + 1, lambda i, h=h: 2 * (h + i)
                )
        w5, b5 = _fold64(tv.conv5[0], tv.conv5[1])
        w5p, b5p = _pw64(w5, b5, list(range(w5.shape[0])), pin, w5.shape[0], len(pin))
        self.register_buffer("w5", w5p.to(dt).view(w5p.shape[0], -1, 1, 1).contiguous(memory_format=_CL))
        self.register_buffer("b5", b5p.to(dt))
        self.fc = nn.Linear(tv.fc.in_features, tv.fc.out_features).to(dt)
        self.fc.load_state_dict({k: v.to(dt) for k, v in tv.fc.state_dict().items()})

    def _units(self) -> Iterator[_Blk]:
        return iter(self.units)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        return ops.stem_pool(x, self.stem_wt, self.stem_b, [2, 2], [1, 1], 1, 3, 2, 1, False)  # noqa: FBT003

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [N, classes] (fp16) from the fp32 NCHW image batch."""
        if self.big:
            return self._forward_big(x)
        x = self._stem(x)
        for u, (ds, q, cin_off, cin) in zip(self._units(), self.meta, strict=True):
            n, _, hh, ww = x.shape
            ho, wo = ((hh + 1) // 2, (ww + 1) // 2) if ds else (hh, ww)
            out = torch.empty((n, 4 * q, ho, wo), device=x.device, dtype=x.dtype, memory_format=_CL)
            if ds:
                ops.mbconv_into(
                    x,
                    None,
                    None,
                    u.wd1,
                    u.bd1,
                    u.w12,
                    u.b12,
                    [2, 2],
                    [1, 1],
                    [0, 0, 1],
                    0,
                    cin,
                    out,
                    u.colmap1,
                    None,
                    None,
                )
                ops.mbconv_into(
                    x,
                    u.w1,
                    u.b1,
                    u.wd,
                    u.bd,
                    u.w2,
                    u.b2,
                    [2, 2],
                    [1, 1],
                    [1, 0, 1],
                    cin_off,
                    cin,
                    out,
                    u.colmap,
                    None,
                    None,
                )
            else:
                ops.mbconv_into(
                    x,
                    u.w1,
                    u.b1,
                    u.wd,
                    u.bd,
                    u.w2,
                    u.b2,
                    [1, 1],
                    [1, 1],
                    [1, 0, 1],
                    cin_off,
                    cin,
                    out,
                    u.colmap,
                    u.gsrc,
                    u.gdst,
                )
            x = out
        x = ops.conv_fused(x, self.w5, self.b5, None, [0, 0], [1, 1], 1)
        return self.fc(x.mean((2, 3)))

    def _forward_big(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stem(x)
        for u, (ds, q, cin_off, cin) in zip(self._units(), self.meta, strict=True):
            n, _, hh, ww = x.shape
            if ds:
                out = torch.empty(
                    (n, 4 * q, (hh + 1) // 2, (ww + 1) // 2), device=x.device, dtype=x.dtype, memory_format=_CL
                )
                d1 = ops.dwconv(x, u.kd1, u.bd1c, [2, 2], [1, 1], 0)
                ops.conv_fused_into(d1, u.k12a, u.b12a, None, [0, 0], [1, 1], 1, out[:, :q])
                ops.conv_fused_into(d1, u.k12b, u.b12b, None, [0, 0], [1, 1], 1, out[:, 3 * q :])
                h = ops.conv_fused(x, u.k1, u.b1, None, [0, 0], [1, 1], 1)
                h = ops.dwconv(h, u.wd, u.bd, [2, 2], [1, 1], 0)
            else:
                out = torch.empty((n, 4 * q, hh, ww), device=x.device, dtype=x.dtype, memory_format=_CL)
                h = ops.conv_fused(x[:, cin_off : cin_off + cin], u.k1, u.b1, None, [0, 0], [1, 1], 1)
                h = ops.dwconv(h, u.wd, u.bd, [1, 1], [1, 1], 0)
                ops.chan_gather(x, out, u.gsrc, u.gdst)
            ops.conv_fused_into(h, u.k3, u.b2, None, [0, 0], [1, 1], 1, out[:, q : 3 * q])
            x = out
        x = ops.conv_fused(x, self.w5, self.b5, None, [0, 0], [1, 1], 1)
        return self.fc(x.mean((2, 3)))


def build_shufflenet(tv: _Blk, device: torch.device) -> tuple[FastShuffleNetV2, FastShuffleNetV2]:
    """(small-batch, big-batch) fast ShuffleNetV2 pair from a torchvision model (built from an fp32 CPU copy)."""
    import copy  # noqa: PLC0415

    cpu = copy.deepcopy(tv).float().eval().cpu()
    return FastShuffleNetV2(cpu).to(device), FastShuffleNetV2(cpu, big=True).to(device)


# --- EfficientViT LiteMLA (timm efficientvit_mit), kernels ------------------------------------------------------------


def _lmla_ok(m: _Blk) -> bool:
    if len(m.aggreg) != 1 or m.dim != 32 or not isinstance(m.kernel_func, nn.ReLU):  # noqa: PLR2004
        return False
    if not all(isinstance(s, nn.Identity) for s in (m.qkv.norm, m.qkv.act, m.proj.act)):
        return False
    dw, gp = m.aggreg[0]
    return gp.groups * 32 == dw.weight.shape[0] and dw.weight.shape[0] % 32 == 0


class FastLiteMLA(nn.Module):
    """qkv 1x1 -> dw KxK -> grouped 1x1 -> ReLU linear attention over both tensors (fp32) -> proj (BN folded) + res."""

    w_qkv: torch.Tensor
    b_qkv: torch.Tensor | None
    wd: torch.Tensor
    bd: torch.Tensor | None
    w2c: torch.Tensor
    b2g: torch.Tensor | None
    w_p: torch.Tensor
    b_p: torch.Tensor

    def __init__(self, m: _Blk) -> None:
        """Take the weights of an eval timm LiteMLA (fp16, already on its device)."""
        super().__init__()
        cw = m.qkv.conv.weight
        dt, dev = cw.dtype, cw.device
        self.register_buffer("w_qkv", cw.detach().contiguous(memory_format=_CL))
        self.register_buffer("b_qkv", m.qkv.conv.bias.detach() if m.qkv.conv.bias is not None else None)
        dw, gp = m.aggreg[0]
        c = dw.weight.shape[0]
        self.k = dw.kernel_size[0]
        self.groups = gp.groups
        self.register_buffer("wd", dw.weight.detach()[:, 0].permute(1, 2, 0).contiguous())
        self.register_buffer("bd", dw.bias.detach() if dw.bias is not None else None)
        self.register_buffer("w2c", gp.weight.detach().contiguous(memory_format=_CL))  # [C, 32, 1, 1]
        self.register_buffer("b2g", gp.bias.detach() if gp.bias is not None else None)
        pw, bn = m.proj.conv, m.proj.norm
        with torch.no_grad():
            w = pw.weight.float()
            b = pw.bias.float() if pw.bias is not None else torch.zeros(w.shape[0], device=dev)
            if isinstance(bn, nn.BatchNorm2d):
                rv, rm = bn.running_var, bn.running_mean
                assert rv is not None  # noqa: S101 - eval BN
                assert rm is not None  # noqa: S101
                sc = bn.weight.float() / torch.sqrt(rv.float() + bn.eps)
                w, b = w * sc.view(-1, 1, 1, 1), (b - rm.float()) * sc + bn.bias.float()
        self.register_buffer("w_p", w.to(dt).contiguous(memory_format=_CL))
        self.register_buffer("b_p", b.to(dt))
        self.eps = float(m.eps)
        self.c = c

    def forward(self, x: torch.Tensor, res: torch.Tensor | None = None) -> torch.Tensor:
        """LiteMLA(x) (+ res)."""
        qkv = ops.conv_fused(x, self.w_qkv, self.b_qkv, None, [0, 0], [1, 1], 0)
        p = self.k // 2
        d = ops.dwconv(qkv, self.wd, self.bd, [1, 1], [p, p], 0)
        agg = ops.gconv(d, self.w2c, self.b2g, [1, 1], [0, 0], self.groups, 0)
        a = ops.linattn2(qkv, agg, self.eps)
        return ops.conv_fused(a, self.w_p, self.b_p, res, [0, 0], [1, 1], 0)


def patch_litemla(gm: fx.GraphModule) -> int:
    """Swap LiteMLA leaf modules of an fx GraphModule for FastLiteMLA; fold the following residual add."""
    mods = dict(gm.named_modules())
    n = 0
    for node in list(gm.graph.nodes):
        m = mods.get(str(node.target)) if node.op == "call_module" else None
        if type(m).__name__ != "LiteMLA" or not _lmla_ok(m):
            continue
        gm.add_submodule(str(node.target), FastLiteMLA(m))
        x = node.args[0]
        us = list(node.users)
        if (
            len(us) == 1
            and us[0].op == "call_function"
            and us[0].target in {operator.add, torch.add}
            and set(us[0].args) == {node, x}
            and not us[0].kwargs
        ):
            node.args = (x, x)
            us[0].replace_all_uses_with(node)
            gm.graph.erase_node(us[0])
        n += 1
    gm.graph.lint()
    gm.recompile()
    return n
