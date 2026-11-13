# ruff: noqa: SLF001 - patches library modules in place
"""Block-level fast forwards for transformer-style classifiers.

timm ConvNeXt, ViT / DeiT / DINOv2, Swin, EVA-02 and HF transformers SigLIP. Same math as the
library forward; patched per instance with `override`.

portable: plain PyTorch (SDPA attention, LayerScale folded / as one addcmul, no copies).
kernels:      the token-wise linear layers run as fused fp16 GEMMs (fvm_conv: 1x1 conv over a [1, K, M, 1] view) with
          bias, GELU and the residual add in the epilogue; EVA RoPE and SwiGLU + LayerNorm as single kernels
          (fvm_extra).
"""

from __future__ import annotations

import itertools
import sys
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import is_base, on_load, override

from . import ops
from .conv_ops import lin  # re-exported: `transformer.lin`
from .fold import hooked

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    _PeRope = Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]  # `_pe_rope`'s signature

_Blk = Any  # a timm / torchvision / transformers block patched with our buffers: untyped to the checker


def _modules(model: nn.Module) -> Iterator[_Blk]:
    """The block candidates: modules without hooks below them (a block forward may bypass its children's calls)."""
    return (m for m in model.modules() if not any(hooked(d) for d in itertools.islice(m.modules(), 1, None)))


_SILU = 3
_GELU = 5
_GELU_TANH = 6
_QUICK_GELU = 1.702  # QuickGELU(x) = x * sigmoid(1.702 x) = SiLU(1.702 x) / 1.702 (CLIP)


def _w4(w: torch.Tensor) -> torch.Tensor:
    return w.detach().reshape(w.shape[0], w.shape[1], 1, 1).contiguous(memory_format=torch.channels_last)


def _is_id(m: object) -> bool:
    return m is None or is_base(m, nn.Identity)


def _timm(module: str, *names: str) -> tuple[Any, ...] | None:
    """The classes `names` of the imported `timm.<module>` (None if timm or one of them is absent: nothing to match)."""
    mod = sys.modules.get(f"timm.{module}")
    found = tuple(getattr(mod, n, None) for n in names)
    return None if mod is None or None in found else found


def _own_forwards(*mods: object) -> bool:
    """No instance-level `forward` replaced on any of `mods` (the class forward is the one matched)."""
    return not any("forward" in getattr(m, "__dict__", {}) for m in mods)


def _gelu(act: object) -> int:
    """_GELU / _GELU_TANH for torch's and timm's GELU modules (exact or tanh), else 0."""
    if is_base(act, nn.GELU):
        return {"none": _GELU, "tanh": _GELU_TANH}.get(act.approximate, 0)
    cls = _timm("layers.activations", "GELU", "GELUTanh")
    if cls is not None and is_base(act, cls[0]):
        return _GELU
    return _GELU_TANH if cls is not None and is_base(act, cls[1]) else 0


def _quick_gelu(act: object) -> bool:
    """Timm's QuickGELU (OpenAI / DFN / MetaCLIP CLIP towers)."""
    cls = _timm("layers.activations", "QuickGELU")
    return cls is not None and is_base(act, cls[0])


def _id_or(m: object, *classes: type) -> bool:
    """Identity, or exactly one of `classes` (eval DropPath / LayerScale)."""
    return _is_id(m) or any(is_base(m, c) for c in classes)


def _reg(mod: _Blk, bufs: dict[str, torch.Tensor | None]) -> None:
    for k, v in bufs.items():
        # buffers follow the module across .to(); detached: an fp32 weight passes `.float()` / `.to()` unchanged and
        # would stay a Parameter that requires grad (torch.export warns and treats it as a constant). A buffer made
        # from a parameter without a fold shares its storage (no copy of the weights: lean).
        mod.register_buffer(k, None if v is None else v.detach(), persistent=False)


def _alias(params: list[nn.Parameter], packed: torch.Tensor, rows: list[int] | None = None) -> None:
    """Point `params` at row slices of `packed` (which holds exactly their values) instead of their own storage.

    One copy of the weights serves the packed buffer and the parameters (state_dict, the portable forward).
    """
    at = 0
    for i, p in enumerate(params):
        if rows is not None:
            at = rows[i]
        p.data = packed[at : at + p.shape[0]]
        at += p.shape[0]


# --- ConvNeXt (timm ConvNeXtBlock, Linear MLP) ---------------------------------------------------------------------


def _cnx_ok(blk: _Blk) -> bool:
    """An unmodified timm ConvNeXtBlock with a Linear Mlp / GlobalResponseNormMlp (exactly `_cnx_portable`'s math)."""
    cls = _timm("models.convnext", "ConvNeXtBlock", "Mlp", "GlobalResponseNormMlp", "DropPath")
    if cls is None or not is_base(blk, cls[0]):
        return False
    m = blk.mlp
    return (
        not blk.use_conv_mlp
        and (is_base(m, cls[1]) or is_base(m, cls[2]))
        and _own_forwards(blk, m)
        and is_base(blk.conv_dw, nn.Conv2d)
        and is_base(m.fc1, nn.Linear)
        and is_base(m.fc2, nn.Linear)
        and _is_id(blk.shortcut)
        and _id_or(blk.drop_path, cls[3])
        and _gelu(m.act) != 0  # exact or tanh (WD ConvNeXt tagger)
        and m.fc1.bias is not None
        and m.fc2.bias is not None
        and _is_id(getattr(m, "norm", None))
    )


def _cnx_prep(blk: _Blk) -> None:
    m = blk.mlp
    with torch.no_grad():
        w2, b2 = m.fc2.weight.float(), m.fc2.bias.float()
        if blk.gamma is not None:
            g = blk.gamma.float()
            w2, b2 = w2 * g[:, None], b2 * g
        dt = m.fc2.weight.dtype
        if blk.gamma is None:
            _reg(blk, {"_w2": m.fc2.weight, "_b2": m.fc2.bias})
        else:
            _reg(blk, {"_w2": w2.to(dt).contiguous(), "_b2": b2.to(dt)})
    grn = getattr(m, "grn", None)
    if grn is not None:
        blk._grn_eps = grn.eps


def _grn(blk: _Blk, h: torch.Tensor) -> torch.Tensor:
    """h: [N, H, W, C]; timm GlobalResponseNorm (channels-last) as one addcmul: h * (1 + w * Nx) + b."""
    g = blk.mlp.grn
    gx = h.norm(p=2, dim=(1, 2), keepdim=True).float()
    nx = gx / (gx.mean(dim=-1, keepdim=True) + blk._grn_eps)
    s = (1 + g.weight.float() * nx).to(h.dtype)
    return torch.addcmul(g.bias.view(1, 1, 1, -1), h, s)


def _cnx_portable(self: _Blk, x: torch.Tensor) -> torch.Tensor:
    shortcut = x
    h = self.conv_dw(x).permute(0, 2, 3, 1)  # channels-last: a free [N, H, W, C] view
    h = self.norm(h)
    m = self.mlp
    h = m.act(m.fc1(h))
    if hasattr(self, "_grn_eps"):
        h = _grn(self, h)
    g = self.gamma
    if torch.is_grad_enabled() and any(p is not None and p.requires_grad for p in (m.fc2.weight, m.fc2.bias, g)):
        h = m.fc2(h)  # autograd: from the parameters (`_w2` / `_b2` are detached folded copies)
        h = h * g if g is not None else h
    else:
        h = F.linear(h, self._w2, self._b2)
    return shortcut + h.permute(0, 3, 1, 2)


def _cnx_kernels(self: _Blk, x: torch.Tensor) -> torch.Tensor:
    c = self.conv_dw
    shortcut = x.contiguous(memory_format=torch.channels_last)
    h = ops.dwconv(shortcut, self._w_rsc, c.bias, list(c.stride), list(c.padding), 0).permute(0, 2, 3, 1)
    h = self.norm(h)
    h = lin(h, self._w1, self.mlp.fc1.bias, act=self._act)
    if hasattr(self, "_grn_eps"):
        h = _grn(self, h)
    return lin(h, self._w2_4, self._b2, res=shortcut.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def patch_convnext(model: nn.Module, mode: str) -> int:
    """Patch every timm ConvNeXtBlock (Linear MLP, GELU, optional GRN); returns the count."""
    n = 0
    for blk in _modules(model):
        if not _cnx_ok(blk):
            continue
        _cnx_prep(blk)
        c = blk.conv_dw
        dw_ok = (
            c.groups == c.in_channels == c.out_channels
            and c.in_channels % 8 == 0
            and c.dilation == (1, 1)
            and not isinstance(c.padding, str)
        )
        if mode == "kernels" and dw_ok:
            _cnx_kprep(blk)
            override(blk, _cnx_kernels)
        else:
            override(blk, _cnx_portable)
        on_load(blk, _cnx_reload)
        n += 1
    return n


def _cnx_kprep(blk: _Blk) -> None:
    _reg(
        blk,
        {
            "_w_rsc": blk.conv_dw.weight.data[:, 0].permute(1, 2, 0).contiguous(),
            "_w1": _w4(blk.mlp.fc1.weight),
            "_w2_4": _w4(blk._w2),
        },
    )
    blk._act = _gelu(blk.mlp.act)


def _cnx_reload(blk: _Blk, _keys: object) -> None:
    _cnx_prep(blk)
    if "_w_rsc" in blk._buffers:
        with torch.no_grad():
            _cnx_kprep(blk)


# --- ViT (timm Block) ----------------------------------------------------------------------------------------------


def _gamma(ls: object) -> torch.Tensor | None:
    return getattr(ls, "gamma", None)


def _vit_portable(
    self: _Blk,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,  # noqa: FBT001, FBT002 - timm's signature
) -> torch.Tensor:
    a = self.attn
    b, n, c = x.shape
    h = a.num_heads
    qkv = F.linear(self.norm1(x), a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=a.scale)
    o = o.transpose(1, 2).reshape(b, n, c)
    if not _is_id(getattr(a, "norm", None)):
        o = a.norm(o)
    o = F.linear(o, a.proj.weight, a.proj.bias)
    g = _gamma(self.ls1)
    x = torch.addcmul(x, o, g) if g is not None else x + o
    m = self.mlp
    y = m.act(F.linear(self.norm2(x), m.fc1.weight, m.fc1.bias))
    if not _is_id(getattr(m, "norm", None)):
        y = m.norm(y)
    o = F.linear(y, m.fc2.weight, m.fc2.bias)
    g = _gamma(self.ls2)
    if torch.is_grad_enabled():  # norm2 saved x for backward: in place only without autograd (TG-2)
        return torch.addcmul(x, o, g) if g is not None else x + o
    return x.addcmul_(o, g) if g is not None else x.add_(o)


def _vit_kernels(
    self: _Blk,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,  # noqa: FBT001, FBT002 - timm's signature
) -> torch.Tensor:
    a = self.attn
    b, n, c = x.shape
    h = a.num_heads
    qkv = lin(self.norm1(x), self._w_qkv, a.qkv.bias).view(b, n, 3, h, c // h)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=a.scale)
    o = o.transpose(1, 2).reshape(b, n, c)
    x = lin(o, self._w_proj, self._b_proj, res=x)
    n2 = self.norm2
    h = n2(x) if self._ln2_w is None else F.layer_norm(x, n2.normalized_shape, self._ln2_w, self._ln2_b, n2.eps)
    y = lin(h, self._w_fc1, self._b_fc1, act=self._act)
    return lin(y, self._w_fc2, self._b_fc2, res=x)


def _vit_act(mod: _Blk) -> int:
    """Fused activation code of the block MLP, 0 if the block cannot take the fused path."""
    a, m = mod.attn, mod.mlp
    if not _is_id(getattr(a, "norm", None)) or not _is_id(getattr(m, "norm", None)):
        return 0
    if a.qkv.bias is None or a.proj.bias is None or m.fc1.bias is None or m.fc2.bias is None:
        return 0
    return _SILU if _quick_gelu(m.act) else _gelu(m.act)


def _vit_prep(mod: _Blk) -> None:
    """Fused-path buffers; without a fold a buffer shares the parameter's storage.

    Folds: LayerScale into proj / fc2; QuickGELU as SiLU on 1.702 x (the factor in norm2's affine and fc1's bias),
    fc2 / 1.702.
    """
    a, m = mod.attn, mod.mlp
    g1, g2 = _gamma(mod.ls1), _gamma(mod.ls2)
    quick = _quick_gelu(m.act)
    n2 = mod.norm2
    tln = _timm("layers.norm", "LayerNorm")
    ln2 = (
        quick
        and (is_base(n2, nn.LayerNorm) or (tln is not None and is_base(n2, tln[0])))
        and n2.weight is not None
        and n2.bias is not None
    )
    dt = a.qkv.weight.dtype
    bufs: dict[str, torch.Tensor | None] = {
        "_w_qkv": _w4(a.qkv.weight),
        "_w_proj": _w4(a.proj.weight),
        "_b_proj": a.proj.bias,
        "_ln2_w": None,
        "_ln2_b": None,
        "_w_fc1": _w4(m.fc1.weight),
        "_b_fc1": m.fc1.bias,
        "_w_fc2": _w4(m.fc2.weight),
        "_b_fc2": m.fc2.bias,
    }
    with torch.no_grad():
        if g1 is not None:
            bufs["_w_proj"] = _w4((a.proj.weight.float() * g1.float()[:, None]).to(dt))
            bufs["_b_proj"] = (a.proj.bias.float() * g1.float()).to(dt)
        if quick:
            bufs["_b_fc1"] = (m.fc1.bias.float() * _QUICK_GELU).to(dt)
            if ln2:
                bufs["_ln2_w"] = (n2.weight.float() * _QUICK_GELU).to(n2.weight.dtype)
                bufs["_ln2_b"] = (n2.bias.float() * _QUICK_GELU).to(n2.bias.dtype)
            else:
                bufs["_w_fc1"] = _w4((m.fc1.weight.float() * _QUICK_GELU).to(dt))
        if quick or g2 is not None:
            w2, b2 = m.fc2.weight.float(), m.fc2.bias.float()
            if quick:
                w2 = w2 / _QUICK_GELU
            if g2 is not None:
                w2, b2 = w2 * g2.float()[:, None], b2 * g2.float()
            bufs["_w_fc2"], bufs["_b_fc2"] = _w4(w2.to(dt)), b2.to(dt)
    _reg(mod, bufs)


def _vit_reload(mod: _Blk, _keys: object) -> None:
    _vit_prep(mod)


def _vit_ok(mod: _Blk) -> bool:
    """An unmodified timm vision_transformer `Block` with its `Attention` and `Mlp`: exactly `_vit_portable`'s math.

    Variants keep timm's forward: other attention / MLP classes (DiffAttention, SwiGLU, packed SwiGLU, ...), q / k
    norm, gated attention, a head width other than dim / heads, other LayerScale / DropPath modules, forwards replaced
    on the instance, and the many other timm modules that are also called `Block` (BEiT, Twins, PVT, ...).
    """
    cls = _timm("models.vision_transformer", "Block", "Attention", "Mlp", "LayerScale", "DropPath")
    if cls is None or not is_base(mod, cls[0]):
        return False
    a, m = mod.attn, mod.mlp
    if not (is_base(a, cls[1]) and is_base(m, cls[2]) and _own_forwards(mod, a, m)):
        return False
    if not all(is_base(x, nn.Linear) for x in (a.qkv, a.proj, m.fc1, m.fc2)):
        return False
    c = a.qkv.in_features
    return (
        a.gate is None
        and _is_id(a.q_norm)
        and _is_id(a.k_norm)
        and a.num_heads * a.head_dim == a.attn_dim == c
        and a.qkv.out_features == 3 * c
        and a.proj.in_features == a.proj.out_features == c
        and m.fc1.in_features == m.fc2.out_features == c
        and all(_id_or(x, cls[3]) for x in (mod.ls1, mod.ls2))
        and all(_id_or(x, cls[4]) for x in (mod.drop_path1, mod.drop_path2))
    )


def patch_vit_blocks(model: nn.Module, mode: str) -> int:
    """Patch every plain timm vision_transformer `Block` (see `_vit_ok`); returns the count."""
    n = 0
    for mod in _modules(model):
        if not _vit_ok(mod):
            continue
        act = _vit_act(mod) if mode == "kernels" else 0
        if act:
            _vit_prep(mod)
            mod._act = act
            override(mod, _vit_kernels)
            on_load(mod, _vit_reload)
        else:
            override(mod, _vit_portable)
        n += 1
    return n


# --- Swin (timm SwinTransformerStage) ------------------------------------------------------------------------------
# kernels keeps the residual stream of a stage in window-major token order t = n * nW + w (n: position in the window,
# w: window index after the cyclic shift): q / k / v come from fused GEMMs as [B, N, nW * heads, D] without copies,
# ONE SDPA call over nW * heads "heads" takes relative-position bias + shift mask as a precomputed broadcast bias,
# and roll + window partition / reverse become one gather per block.


def patch_swin_portable(model: nn.Module) -> int:
    """Timm's own SDPA attention path (`fused_attn = True`) instead of the manual q @ k^T path."""
    n = 0
    for m in _modules(model):
        cls = _timm("models.swin_transformer", "WindowAttention")
        if cls is not None and is_base(m, cls[0]) and hasattr(m, "fused_attn"):
            m.fused_attn = True
            n += 1
    return n


def _swin_order(h: int, w: int, ws: tuple[int, int], shift: tuple[int, int]) -> torch.Tensor:
    """idx[t] = raster index of the token at stream position t for one block."""
    wh, ww = ws
    nwh, nww = h // wh, w // ww
    t = torch.arange(h * w)
    n, wi = t // (nwh * nww), t % (nwh * nww)
    r = (wi // nww) * wh + n // ww
    c = (wi % nww) * ww + n % ww
    return ((r + shift[0]) % h) * w + (c + shift[1]) % w


def _swin_bias(blk: _Blk, h: int, w: int) -> tuple[torch.Tensor, int]:
    a = blk.attn
    n = blk.window_area
    nw = (h // blk.window_size[0]) * (w // blk.window_size[1])
    rel = a._get_rel_pos_bias()[0].float()
    heads = rel.shape[0]
    b = rel.unsqueeze(0).expand(nw, heads, n, n)
    if any(blk.shift_size):
        m = blk.attn_mask
        if m is None or m.shape[0] != nw:
            m = blk.get_attn_mask(torch.zeros(1, h, w, 1, device=rel.device))
        b = b + m.float().view(nw, 1, n, n)
    np_ = -(-n // 8) * 8  # 16-byte aligned rows for the efficient-attention kernel; callers slice [..., :N]
    out = torch.zeros(1, nw * heads, n, np_, dtype=a.qkv.weight.dtype, device=rel.device)
    out[..., :n] = b.reshape(1, nw * heads, n, n)
    return out, nw


def _swin_block(blk: _Blk, x: torch.Tensor, bias: torch.Tensor, nw: int) -> torch.Tensor:
    a, m = blk.attn, blk.mlp
    b, tokens, c = x.shape
    heads = a.num_heads
    d = c // heads
    n = tokens // nw
    h = blk.norm1(x)
    q = lin(h, blk._wq, blk._bq).view(b, n, nw * heads, d).transpose(1, 2)
    k = lin(h, blk._wk, blk._bk).view(b, n, nw * heads, d).transpose(1, 2)
    v = lin(h, blk._wv, blk._bv).view(b, n, nw * heads, d).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=a.scale).transpose(1, 2).reshape(b, tokens, c)
    x = lin(o, blk._wp, a.proj.bias, res=x)
    h = lin(blk.norm2(x), blk._w1, m.fc1.bias, act=blk._act)
    return lin(h, blk._w2, m.fc2.bias, res=x)


def _swin_stage(self: _Blk, x: torch.Tensor) -> torch.Tensor:
    x = self.downsample(x)
    b, h, w, c = x.shape
    if (h, w) != self._ff_hw:  # other resolution: timm path
        for blk in self.blocks:
            x = blk(x)
        return x
    x = x.reshape(b, h * w, c).index_select(1, self._ff_first)
    for blk in self.blocks:
        x = _swin_block(blk, x, blk._ff_bias[..., : blk.window_area], blk._ff_nw)
        x = x.index_select(1, blk._ff_next)
    return x.view(b, h, w, c)


def _swin_prep(st: _Blk) -> bool:
    blocks = list(st.blocks)
    h, w = blocks[0].input_resolution
    if any(h % b.window_size[0] or w % b.window_size[1] for b in blocks):
        return False
    dev = blocks[0].attn.qkv.weight.device
    orders = [_swin_order(h, w, b.window_size, b.shift_size).to(dev) for b in blocks]
    invs = [torch.argsort(o) for o in orders]
    _reg(st, {"_ff_first": orders[0]})
    for i, b in enumerate(blocks):
        nxt = invs[i][orders[i + 1]] if i < len(blocks) - 1 else invs[-1]
        bias, nw = _swin_bias(b, h, w)
        a, m = b.attn, b.mlp
        c = a.qkv.weight.shape[1]
        wq, bq = a.qkv.weight, a.qkv.bias.detach()
        _reg(
            b,
            {
                "_ff_next": nxt,
                "_ff_bias": bias,
                "_wq": _w4(wq[:c]),
                "_wk": _w4(wq[c : 2 * c]),
                "_wv": _w4(wq[2 * c :]),
                "_bq": bq[:c].contiguous(),
                "_bk": bq[c : 2 * c].contiguous(),
                "_bv": bq[2 * c :].contiguous(),
                "_wp": _w4(a.proj.weight),
                "_w1": _w4(m.fc1.weight),
                "_w2": _w4(m.fc2.weight),
            },
        )
        b._ff_nw = nw
        b._act = _gelu(m.act)
    st._ff_hw = (h, w)
    return True


def _swin_reload(st: _Blk, _keys: object) -> None:
    with torch.no_grad():
        _swin_prep(st)


def patch_swin_kernels(model: nn.Module) -> int:
    """Window-major fused path for every timm SwinTransformerStage at its native resolution; returns the count."""
    n = 0
    cls = _timm("models.swin_transformer", "SwinTransformerStage", "SwinTransformerBlock", "WindowAttention", "Mlp")
    if cls is None:
        return 0
    for st in _modules(model):
        if not is_base(st, cls[0]) or not _own_forwards(st):
            continue
        ok = all(
            is_base(b, cls[1])
            and is_base(b.attn, cls[2])
            and is_base(b.mlp, cls[3])
            and _own_forwards(b, b.attn, b.mlp)
            and _gelu(b.mlp.act) != 0  # exact or tanh
            and b.attn.qkv.bias is not None
            and b.attn.proj.bias is not None
            for b in st.blocks
        )
        if ok and _swin_prep(st):
            override(st, _swin_stage)
            on_load(st, _swin_reload)
            n += 1
    return n


# --- EVA-02 (timm EvaBlock) ----------------------------------------------------------------------------------------


def _eva_ok(blk: _Blk) -> bool:
    """An unmodified timm EvaBlock / EvaAttention (separate q / k / v, no q / k norm) with a SwiGLU MLP."""
    cls = _timm("models.eva", "EvaBlock", "EvaAttention", "SwiGLU")
    if cls is None or not (is_base(blk, cls[0]) and is_base(blk.attn, cls[1]) and is_base(blk.mlp, cls[2])):
        return False
    a, m = blk.attn, blk.mlp
    c = a.proj.out_features
    return (
        a.qkv is None
        and _own_forwards(blk, a, m)
        and a.num_heads * a.head_dim == c
        and all(p.in_features == p.out_features == c for p in (a.q_proj, a.k_proj, a.v_proj))
        and a.proj.in_features == c
        and is_base(m.act, nn.SiLU)
        and type(m.norm).__name__ == "LayerNorm"
        and _is_id(a.norm)
        and _is_id(a.q_norm)
        and _is_id(a.k_norm)
        and getattr(a, "gate", None) is None
        and not getattr(a, "rotate_half", False)
        and a.proj.bias is not None
        and m.fc2.bias is not None
    )


def _eva_prep(blk: _Blk) -> None:
    a, m = blk.attn, blk.mlp
    c = a.q_proj.weight.shape[1]
    dt, dev = a.q_proj.weight.dtype, a.q_proj.weight.device
    z = torch.zeros(c, dtype=dt, device=dev)
    with torch.no_grad():
        wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
        bqkv = torch.cat([p.bias if p.bias is not None else z for p in (a.q_proj, a.k_proj, a.v_proj)])
        hid = m.fc1_g.weight.shape[0]
        hp = -(-hid // 8) * 8  # [gate | x] halves padded to a multiple of 8
        w1 = torch.zeros(2 * hp, c, dtype=dt, device=dev)
        b1 = torch.zeros(2 * hp, dtype=dt, device=dev)
        w1[:hid], w1[hp : hp + hid] = m.fc1_g.weight, m.fc1_x.weight
        b1[:hid], b1[hp : hp + hid] = m.fc1_g.bias, m.fc1_x.bias
        w2 = torch.zeros(c, hp, dtype=torch.float32, device=dev)
        w2[:, :hid] = m.fc2.weight.float()
        b2 = m.fc2.bias.float()
        wp, bp = a.proj.weight.float(), a.proj.bias.float()
        if blk.gamma_1 is not None:
            wp, bp = wp * blk.gamma_1.float()[:, None], bp * blk.gamma_1.float()
        if blk.gamma_2 is not None:
            w2, b2 = w2 * blk.gamma_2.float()[:, None], b2 * blk.gamma_2.float()
        lw = torch.zeros(hp, dtype=dt, device=dev)
        lb = torch.zeros(hp, dtype=dt, device=dev)
        lw[:hid], lb[:hid] = m.norm.weight, m.norm.bias
        w2 = w2.to(dt)
        _alias([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], wqkv)
        _alias([m.fc1_g.weight, m.fc1_x.weight], w1, [0, hp])
        if blk.gamma_2 is None:
            m.fc2.weight.data = w2[:, :hid]
    _reg(
        blk,
        {
            "_wqkv": _w4(wqkv),
            "_bqkv": bqkv,
            "_wp": _w4(a.proj.weight) if blk.gamma_1 is None else _w4(wp.to(dt)),
            "_bp": a.proj.bias if blk.gamma_1 is None else bp.to(dt),
            "_w1": _w4(w1),
            "_b1": b1,
            "_w2": _w4(w2),
            "_b2": b2.to(dt),
            "_lw": lw,
            "_lb": lb,
        },
    )
    blk._hid, blk._eps = hid, m.norm.eps


def _eva_kernels(
    self: _Blk,
    x: torch.Tensor,
    rope: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,  # noqa: FBT001, FBT002 - timm's signature
) -> torch.Tensor:
    a = self.attn
    b, n, c = x.shape
    h = a.num_heads
    d = c // h
    qkv = lin(self.norm1(x), self._wqkv, self._bqkv)  # [B, N, 3C]
    if rope is not None:
        q, k = ops.rope_qk(qkv, rope, a.num_prefix_tokens, h)
    else:
        q, k = qkv[..., :c].reshape(b, n, h, d), qkv[..., c : 2 * c].reshape(b, n, h, d)
    v = qkv.view(b, n, 3, h, d)[:, :, 2]
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, is_causal=is_causal, scale=a.scale
    )
    x = lin(o.transpose(1, 2).reshape(b, n, c), self._wp, self._bp, res=x)
    y = lin(self.norm2(x), self._w1, self._b1)  # [B, N, 2 * hp] = [gate | x]
    y = ops.swiglu_ln(y, self._lw, self._lb, self._hid, self._eps)
    return lin(y, self._w2, self._b2, res=x)


def _eva_reload(blk: _Blk, _keys: object) -> None:
    _eva_prep(blk)


def patch_eva(model: nn.Module) -> int:
    """Fused path for every timm EvaBlock (separate q / k / v, 2-D RoPE, SwiGLU with inner LayerNorm)."""
    n = 0
    for blk in _modules(model):
        if _eva_ok(blk):
            _eva_prep(blk)
            override(blk, _eva_kernels)
            on_load(blk, _eva_reload)
            n += 1
    return n


# --- Perception Encoder (timm EvaBlock with AttentionRope and a GELU Mlp) -----------------------------------------


def _pe_ok(blk: _Blk) -> bool:
    """An unmodified timm EvaBlock: `AttentionRope` (fused qkv, interleaved RoPE, no q / k norm, no GQA) and `Mlp`."""
    cls = _timm("models.eva", "EvaBlock")
    lay = _timm("layers.attention", "AttentionRope")
    mlp = _timm("layers.mlp", "Mlp")
    dp = _timm("layers.drop", "DropPath")
    if cls is None or lay is None or mlp is None or not is_base(blk, cls[0]):
        return False
    a, m = blk.attn, blk.mlp
    if not (is_base(a, lay[0]) and is_base(m, mlp[0]) and _own_forwards(blk, a, m)):
        return False
    if a.qkv is None or not all(is_base(x, nn.Linear) for x in (a.qkv, a.proj, m.fc1, m.fc2)):
        return False
    c = a.qkv.in_features
    return (
        a.num_heads * a.head_dim == a.attn_dim == c
        and getattr(a, "num_kv_heads", a.num_heads) == a.num_heads
        and a.qkv.out_features == 3 * c
        and a.proj.in_features == a.proj.out_features == c
        and m.fc1.in_features == m.fc2.out_features == c
        and _is_id(a.q_norm)
        and _is_id(a.k_norm)
        and _is_id(a.norm)
        and _is_id(getattr(m, "norm", None))
        and getattr(a, "gate", None) is None
        and not getattr(a, "rotate_half", False)
        and all(p.bias is not None for p in (a.qkv, a.proj, m.fc1, m.fc2))
        and _gelu(m.act) != 0
        and all(_id_or(x, *(dp or ())) for x in (blk.drop_path1, blk.drop_path2))
    )


def _pe_prep(blk: _Blk) -> None:
    a, m = blk.attn, blk.mlp
    with torch.no_grad():
        wp, bp = a.proj.weight.float(), a.proj.bias.float()
        w2, b2 = m.fc2.weight.float(), m.fc2.bias.float()
        if blk.gamma_1 is not None:
            wp, bp = wp * blk.gamma_1.float()[:, None], bp * blk.gamma_1.float()
        if blk.gamma_2 is not None:
            w2, b2 = w2 * blk.gamma_2.float()[:, None], b2 * blk.gamma_2.float()
        dt = a.qkv.weight.dtype
        _reg(
            blk,
            {
                "_w_qkv": _w4(a.qkv.weight),
                "_w_proj": _w4(a.proj.weight) if blk.gamma_1 is None else _w4(wp.to(dt)),
                "_b_proj": a.proj.bias if blk.gamma_1 is None else bp.to(dt),
                "_w_fc1": _w4(m.fc1.weight),
                "_w_fc2": _w4(m.fc2.weight) if blk.gamma_2 is None else _w4(w2.to(dt)),
                "_b_fc2": m.fc2.bias if blk.gamma_2 is None else b2.to(dt),
            },
        )


def _pe_kernels(
    self: _Blk,
    x: torch.Tensor,
    rope: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,  # noqa: FBT001, FBT002 - timm's signature
) -> torch.Tensor:
    a = self.attn
    b, n, c = x.shape
    h = a.num_heads
    d = c // h
    qkv = lin(self.norm1(x), self._w_qkv, a.qkv.bias)  # [B, N, 3C]
    if rope is not None:
        q, k = ops.rope_qk(qkv, rope, a.num_prefix_tokens, h)
    else:
        q, k = qkv[..., :c].reshape(b, n, h, d), qkv[..., c : 2 * c].reshape(b, n, h, d)
    v = qkv.view(b, n, 3, h, d)[:, :, 2]
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, is_causal=is_causal, scale=a.scale
    )
    x = lin(o.transpose(1, 2).reshape(b, n, c), self._w_proj, self._b_proj, res=x)
    y = lin(self.norm2(x), self._w_fc1, self.mlp.fc1.bias, act=self._act)
    return lin(y, self._w_fc2, self._b_fc2, res=x)


def _pe_rope(qkv: torch.Tensor, rope: torch.Tensor, npt: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k `[B, H, N, D]` of `qkv` `[B, N, 3, H, D]` with timm's interleaved RoPE after the `npt` prefix tokens.

    timm's `apply_rot_embed_cat` on q and k together, in float32, then cast back (timm's `.type_as(v)`).
    """
    sin, cos = (t[:, None, None] for t in rope.chunk(2, -1))  # [N - npt, 1, 1, D]
    qk = qkv[:, npt:, :2]
    rot = torch.stack([-qk[..., 1::2], qk[..., ::2]], -1).reshape(qk.shape)  # timm's `rot`
    qk = (qk * cos + rot * sin).to(qkv.dtype)
    if npt:
        qk = torch.cat([qkv[:, :npt, :2], qk], 1)
    q, k = qk.permute(2, 0, 3, 1, 4).unbind(0)
    return q, k


def _pe_portable(
    self: _Blk,
    x: torch.Tensor,
    rope: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,  # noqa: FBT001, FBT002 - timm's signature
) -> torch.Tensor:
    return _pe_forward(self, x, rope, attn_mask, is_causal=is_causal, qk_rope=_pe_rope)


def _pe_own_rope(
    self: _Blk,
    x: torch.Tensor,
    rope: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,  # noqa: FBT001, FBT002 - timm's signature
) -> torch.Tensor:
    return _pe_forward(self, x, rope, attn_mask, is_causal=is_causal, qk_rope=vars(self)["_qk_rope"])


def _pe_forward(  # noqa: PLR0913 - timm's arguments + the RoPE function
    self: _Blk,
    x: torch.Tensor,
    rope: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    *,
    is_causal: bool,
    qk_rope: _PeRope,
) -> torch.Tensor:
    a = self.attn
    b, n, c = x.shape
    h = a.num_heads
    qkv = F.linear(self.norm1(x), a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h)
    if rope is not None:
        q, k = qk_rope(qkv, rope, a.num_prefix_tokens)
    else:
        q, k = qkv[:, :, :2].permute(2, 0, 3, 1, 4).unbind(0)
    v = qkv[:, :, 2].transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=a.scale)
    o = F.linear(o.transpose(1, 2).reshape(b, n, c), a.proj.weight, a.proj.bias)
    x = torch.addcmul(x, o, self.gamma_1) if self.gamma_1 is not None else x + o
    m = self.mlp
    o = F.linear(m.act(F.linear(self.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
    g = self.gamma_2
    if torch.is_grad_enabled():  # norm2 saved x for backward: in place only without autograd (TG-2)
        return torch.addcmul(x, o, g) if g is not None else x + o
    return x.addcmul_(o, g) if g is not None else x.add_(o)


def _pe_reload(blk: _Blk, _keys: object) -> None:
    _pe_prep(blk)


def pe_rope(model: nn.Module, qk_rope: _PeRope) -> int:
    """Give every portable-patched Perception Encoder block of `model` its own q / k RoPE (e.g. a Metal kernel).

    Args:
        model: A model `optimize` patched in portable mode.
        qk_rope: `_pe_rope`'s replacement: `(qkv [B, N, 3, H, D], rope [N - npt, 2 D] float32, npt)` -> q, k
            `[B, H, N, D]`.

    Returns:
        The number of blocks changed.
    """
    n = 0
    for blk in _modules(model):
        if dict(getattr(type(blk), "_fastforks_methods", ())).get("forward") is _pe_portable:
            object.__setattr__(blk, "_qk_rope", qk_rope)  # a plain function: not a submodule, no binding
            override(blk, _pe_own_rope)
            n += 1
    return n


def patch_pe(model: nn.Module, mode: str) -> int:
    """Patch every Perception Encoder EvaBlock (see `_pe_ok`): `_pe_kernels` in kernels mode, else `_pe_portable`."""
    n = 0
    for blk in _modules(model):
        if _pe_ok(blk) and mode != "kernels":
            override(blk, _pe_portable)
            n += 1
        elif _pe_ok(blk):
            _pe_prep(blk)
            blk._act = _gelu(blk.mlp.act)
            override(blk, _pe_kernels)
            on_load(blk, _pe_reload)
            n += 1
    return n


# --- HF transformers SiglipEncoderLayer ----------------------------------------------------------------------------

_HF_TANH = ("PytorchGELUTanh", "GELUTanh", "NewGELUActivation", "GELUPytorchTanh")
_HF_ACTS = {"gelu_pytorch_tanh": _GELU_TANH, "gelu_new": _GELU_TANH, "gelu": _GELU}


def _siglip_kernels(
    self: _Blk,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **_kwargs: object,
) -> torch.Tensor:
    a = self.self_attn
    x = hidden_states
    b, n, c = x.shape
    h, d = a.num_heads, a.head_dim
    qkv = lin(self.layer_norm1(x), self._wqkv, self._bqkv).view(b, n, 3, h, d)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    o = (
        F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=a.scale)
        .transpose(1, 2)
        .reshape(b, n, c)
    )
    x = lin(o, self._wo, a.out_proj.bias, res=x)
    y = lin(self.layer_norm2(x), self._w1, self.mlp.fc1.bias, act=self._act)
    return lin(y, self._w2, self.mlp.fc2.bias, res=x)


def patch_hf_siglip(model: nn.Module) -> int:
    """Fused path for every HF SiglipEncoderLayer (SigLIP / SigLIP 2 vision towers)."""
    n = 0
    for lay in _modules(model):
        if type(lay).__name__ != "SiglipEncoderLayer":
            continue
        a, m = lay.self_attn, lay.mlp
        an = type(m.activation_fn).__name__
        act = (
            _GELU_TANH
            if an in _HF_TANH
            else _GELU
            if an == "GELUActivation"
            else _HF_ACTS.get(getattr(getattr(m, "config", None), "hidden_act", ""), 0)
        )
        if not act or a.q_proj.bias is None:
            continue
        _siglip_prep(lay)
        lay._act = act
        override(lay, _siglip_kernels)
        on_load(lay, _siglip_reload)
        n += 1
    return n


def _siglip_prep(lay: _Blk) -> None:
    a, m = lay.self_attn, lay.mlp
    with torch.no_grad():
        wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
        _alias([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], wqkv)
        _reg(
            lay,
            {
                "_wqkv": _w4(wqkv),
                "_bqkv": torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).detach(),
                "_wo": _w4(a.out_proj.weight),
                "_w1": _w4(m.fc1.weight),
                "_w2": _w4(m.fc2.weight),
            },
        )


def _siglip_reload(lay: _Blk, _keys: object) -> None:
    _siglip_prep(lay)
