"""Kernels inference path for GREAT-IGEV (general: any input size, no per-size compilation / tuning).

The IGEV machinery of our IGEV-Stereo kernels path (`igev_stereo/kernels/inference.py`, copied: families do not import each
other): fp16 weights, BatchNorm as an fp32 affine in the conv epilogue, fp16-accumulate cuDNN / CUTLASS convs with
fused epilogues, fused ConvGRU, one-kernel geometry lookup, context encoder in row strips. GREAT's attention modules
run on these convs, plus our `igevpp` kernels:
  - LayerNorm as one kernel per call (fp32 statistics and affine),
  - outlook attention as two kernels (logits + softmax + window sum, fold): no k^4 logits tensor, no unfold,
  - volume attention + feed-forward as two kernels (softmax statistics over D, then a per-token update in registers),
  - the excitive-attention volume without the concat volume (cc_stem[0] split into a conv of each view, summed per
    disparity plane), the final volume softmax fused with the disparity regression.

prepare_kernels(model, device), then forward_kernels(model, left, right, iters) (images [1, 3, H, W], 0..255, /32).
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, override
from fastforks.kernels import igevpp as igevpp_kernels
from fastforks.kernels.cutlass_conv import ConvEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, bn_affine, conv_ep, conv_fused, cutlass, epilogue, fuse_ok, fused_cfg
from fastforks.shared.igev_blocks.net import extractor, submodule
from fastforks.shared.igev_blocks.net.submodule import context_upsample, disparity_regression

from ..net import transformers
from ..net.great_igev import CC_CHUNK, ExcitiveAttentionHourglassEncoder, ExcitiveAttentionVolume
from ..net.transformers import d_chunks, planes, tokens, volume_softmax
from ..portable import encode

_TAIL = 3  # cutlass_conv affine act 3: relu(res + relu(conv * s + t))
STRIP_PIXELS = 2 * 2**20  # context-encoder full-res stage processed in row strips of about this many pixels
STRIP_HALO = 16  # full-res rows of receptive field of conv1(7x7) + layer1 + layer2 (14), kept even


def _basic_conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:  # BasicConv: conv -> BN -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    if self.use_bn:
        return conv_ep(self.conv, x, self._s, self._t, act)
    return conv_ep(self.conv, x, None, None, act)


def _in_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    return mean, var


def _basic_conv_in_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # BasicConv_IN: conv -> InstanceNorm -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    y = conv_ep(self.conv, x)
    if not self.use_in:
        return epilogue(y, None, None, act) if act else y
    return cnn_kernels.instance_norm(y, self.IN.eps, act)


def _resblock_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # batch-norm ResidualBlock (context encoder): relu(x + relu(bn2(conv2(relu(bn1(conv1 x))))))
    y = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    if self.downsample is not None:
        x = conv_ep(self.downsample[0], x, self.downsample[1]._s, self.downsample[1]._t, 0)
    if getattr(self.conv2, "_fuse", False) and x.shape[0] == 1 and x.is_contiguous(memory_format=CL):
        o = conv_fused(self.conv2, y, self.norm2._s, self.norm2._t, _TAIL, res=x)  # relu(x + relu(bn(conv)))
        if o is not None:
            return o
    o = conv_ep(self.conv2, y, self.norm2._s, self.norm2._t, ACT["relu"])
    del y
    return epilogue(o, None, None, ACT["relu"], r=x)


def _stem(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    x = self.layer1(x)
    return self.layer2(x)


def _encoder_stem_forward(
    self: nn.Module, x: torch.Tensor, dual_inp: bool = False, num_layers: int = 3
) -> tuple[torch.Tensor, ...]:
    """MultiBasicEncoder with conv1 + norm1 + relu fused. The full-resolution part (conv1, layer1: 64 channels at
    input resolution, then layer2 to 1/2) runs in row strips with a halo: all ops there are local (eval BatchNorm),
    so the result is the same, but only one strip of full-res activations is alive (Midd-F: 3 x 0.78 GB -> ~0.8 GB)."""
    H, W = x.shape[-2:]
    rows = max(64, (STRIP_PIXELS // W) // 2 * 2)
    if rows + 2 * STRIP_HALO >= H or self.downsample != 2:
        x = _stem(self, x)
    else:
        out = None
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            a, b = max(0, r0 - STRIP_HALO), min(H, r1 + STRIP_HALO)
            y = _stem(self, x[:, :, a:b])
            if out is None:
                out = torch.empty(
                    (y.shape[0], y.shape[1], (H + 1) // 2, y.shape[3]), device=y.device, dtype=y.dtype, memory_format=CL
                )
            t = (r0 - a) // 2
            out[:, :, r0 // 2 : (r1 + 1) // 2] = y[:, :, t : t + (r1 + 1) // 2 - r0 // 2]
            del y
        x = out
    x = self.layer3(x)
    outputs04 = [f(x) for f in self.outputs04]
    if num_layers == 1:
        return (outputs04,)
    y = self.layer4(x)
    outputs08 = [f(y) for f in self.outputs08]
    if num_layers == 2:
        return (outputs04, outputs08)
    z = self.layer5(y)
    outputs16 = [f(z) for f in self.outputs16]
    return (outputs04, outputs08, outputs16)


# ------------------------------------------------------------------------------------------------ geometry lookup
class GeoLookup:
    """FastGeoEncodingVolume (num_levels 2) as one kernel: fp16 level-0 geometry and correlation volumes only, the
    kernel pools level 1 on the fly. Output [B, 2 * (C + 1) * (2r + 1), H, W] fp16 as an NHWC view."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, geo_volume: torch.Tensor, radius: int) -> None:
        b, c, d, h, w = geo_volume.shape
        self.shape, self.radius = (b, h, w), radius
        # correlation: [B*H, W1, D] @ [B*H, D, W2] (fp16 in, fp32 accumulate, fp16 out)
        f1 = fmap1.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        f2 = fmap2.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        self.geo = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, d).half().contiguous()
        self.corr = torch.bmm(f1.half(), f2.half()).view(b * h * w, w)

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        b, h, w = self.shape
        P, C, D = self.geo.shape
        out = torch.empty((P, 2 * (C + 1) * (2 * self.radius + 1)), device=disp.device, dtype=torch.float16)
        dsp = disp.reshape(-1).float().contiguous()
        cnn_kernels.kernels(disp.device).launch(
            "geo_lookup", *linear(P * 2 * (C + 1)), self.geo, C, D, self.corr, self.corr.shape[1], w, dsp, out, P,
            self.radius,
        )  # fmt: skip
        return out.view(b, h, w, -1).permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------ update block
def _conv_gru_fast(self: nn.Module, h: int, ctx: tuple[torch.Tensor, torch.Tensor], parts: list[torch.Tensor]) -> int:
    """ConvGRU with fused gates. ctx = (czr, cq): context + conv biases (prepared once); parts: the x inputs as
    tensors, ("pool2x", src) / ("interp", src) (written straight into the [h | x] conv input: no cat, no temporaries)
    or ("fill", fn, width). Updates h in place. The z|r conv reads [h | x] and writes z and r*h from its epilogue
    into a second buffer [r*h | x]; the q conv updates h in place from its epilogue (v8 layout)."""
    czr, cq = ctx
    b, Hd, hh, ww = h.shape
    k = cnn_kernels.kernels(h.device)
    fz = b == 1 and Hd % 8 == 0
    widths = [p.shape[1] if isinstance(p, torch.Tensor) else p[2] if p[0] == "fill" else p[1].shape[1] for p in parts]
    ctot = Hd + sum(widths)
    hx = torch.empty((b, ctot, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
    bufs = [hx, torch.empty_like(hx)] if fz else [hx]
    hx[:, :Hd].copy_(h)
    c0 = Hd
    for p, cw in zip(parts, widths, strict=True):
        if not isinstance(p, torch.Tensor) and p[0] == "fill":  # writes its channels into all buffers itself
            p[1](bufs, c0)
            c0 += cw
            continue
        if isinstance(p, torch.Tensor):
            for buf in bufs:
                buf[:, c0 : c0 + cw].copy_(p)
        else:
            kind, src = p
            if not (b == 1 and src.is_contiguous(memory_format=CL) and cw % 8 == 0):
                raise RuntimeError(f"fused GRU input needs batch 1, channels-last, channels % 8 == 0 (got {cw})")
            n = linear(hh * ww * cw // 8)
            if fz:  # one pass, both buffers
                k.launch(
                    kind + "_into2", *n, src, cw, src.shape[2], src.shape[3], hx.data_ptr() + 2 * c0,
                    bufs[1].data_ptr() + 2 * c0, ctot, hh, ww, cw,
                )  # fmt: skip
            else:
                k.launch(
                    kind + "_into", *n, src, cw, src.shape[2], src.shape[3], hx.data_ptr() + 2 * c0, ctot, hh, ww, cw
                )
        c0 += cw
    P = h.numel() // Hd
    if fz:
        rhx = bufs[1]
        zr = torch.empty((b, 2 * Hd, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
        e0 = ConvEpiArgs(0, 0, czr.data_ptr(), 2 * Hd, zr.data_ptr(), 2 * Hd, hx.data_ptr(), ctot, rhx.data_ptr())
        e1 = ConvEpiArgs(1, 0, cq.data_ptr(), Hd, zr.data_ptr(), 2 * Hd, h.data_ptr(), Hd)
        if not (
            cutlass(0, hx, self.convzr.weight, (hh, ww), e0)
            and cutlass(fused_cfg(P), rhx, self.convq.weight, (hh, ww), e1)
        ):
            raise RuntimeError("cutlass_conv: GRU conv geometry not supported")
        return h
    zr = cnn_kernels.conv_raw(self.convzr, hx)
    k.launch("gru_zr", *linear(P * Hd), zr, czr, h, Hd, hx, ctot, P, Hd)  # r*h -> hx[:, :Hd]
    q = cnn_kernels.conv_raw(self.convq, hx)
    k.launch("gru_update", *linear(P * Hd), q, cq, zr, h, Hd, P, Hd)
    return h


def prepare_context(model: nn.Module, inp_list: list[torch.Tensor]) -> torch.Tensor:
    """inp_list[i] = relu(cnet output) -> per level (czr, cq) with the GRU conv biases added once."""
    out = []
    grus = [model.update_block.gru04, model.update_block.gru08, model.update_block.gru16]
    for i, (x, conv) in enumerate(zip(inp_list, model.context_zqr_convs, strict=False)):
        c = conv_ep(conv, x)  # [B, 3*Hd, h, w]
        Hd = c.shape[1] // 3
        g = grus[i]
        bz, br, bq = g.convz.bias.float(), g.convr.bias.float(), g.convq.bias.float()
        czr = epilogue(c[:, : 2 * Hd].contiguous(memory_format=CL), None, torch.cat([bz, br]).contiguous(), 0)
        cq = epilogue(c[:, 2 * Hd :].contiguous(memory_format=CL), None, bq.contiguous(), 0)
        out.append((czr, cq))
    return out


def _motion_encoder_fast(self: nn.Module, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    cor = conv_ep(self.convc1, corr, act=ACT["relu"])
    cor = conv_ep(self.convc2, cor, act=ACT["relu"])
    d = conv_ep(self.convd1, disp, act=ACT["relu"])
    d = conv_ep(self.convd2, d, act=ACT["relu"])
    out = conv_ep(self.conv, torch.cat([cor, d], 1), act=ACT["relu"], keep_pad=True)  # 127 outputs padded to 128
    if out.shape[1] == self.conv.out_channels + 1:
        out[:, -1:].copy_(disp)  # == torch.cat([out, disp], 1) without the copy
        return out
    return torch.cat([out[:, : self.conv.out_channels], disp], 1).contiguous(memory_format=CL)


def _motion_fill_fused(self: nn.Module, disp: torch.Tensor, corr: torch.Tensor) -> tuple[str, torch.Tensor, int] | None:
    """Fused motion encoder: the motion conv (127 outputs + 1 zero pad) writes from its epilogue into the x slot of
    the GRU's [h | x] and [r*h | x] buffers; disp then overwrites the pad channel. Returns ("fill", fn, 128) for
    _conv_gru_fast, or None (geometry unsupported)."""
    enc = self
    _, _, hh, ww = disp.shape
    if not fuse_ok(enc.conv, hh, ww) or enc.conv.weight.shape[0] != 128:
        return None
    cor = conv_ep(enc.convc1, corr, act=ACT["relu"])
    d = conv_ep(enc.convd1, disp, act=ACT["relu"])
    cd = torch.empty((1, 128, hh, ww), device=cor.device, dtype=cor.dtype, memory_format=CL)
    # 64-channel convs stay cuDNN + epilogue (+ copy into the cat buffer): fused into the buffer was 1-4 ms slower
    for m, x, c0 in ((enc.convc2, cor, 0), (enc.convd2, d, 64)):
        cd[:, c0 : c0 + 64].copy_(conv_ep(m, x, act=ACT["relu"]))
    del cor, d

    def fill(bufs: list[torch.Tensor], c0: int) -> None:
        out2 = (bufs[1], c0) if len(bufs) > 1 else None
        if conv_fused(enc.conv, cd, None, enc.conv._bias32, ACT["relu"], out=(bufs[0], c0), out2=out2) is None:
            raise RuntimeError("fused motion conv unsupported")
        for b in bufs:
            b[:, c0 + 127 : c0 + 128].copy_(disp)

    return ("fill", fill, 128)


def update_block_fast(
    self: nn.Module, net: nn.Module, ctx: Sequence[Any], corr: torch.Tensor, disp: torch.Tensor, compute_mask: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BasicMultiUpdateBlock.forward for 3 GRU levels, inference (net updated in place)."""
    _conv_gru_fast(self.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(self.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    mf = _motion_fill_fused(self.encoder, disp, corr) if net[0].shape[0] == 1 else None
    if mf is None:
        mf = _motion_encoder_fast(self.encoder, disp, corr)
    _conv_gru_fast(self.gru04, net[0], ctx[0], [mf, ("interp", net[1])])
    d = conv_ep(self.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(self.disp_head.conv2, d)
    mask = conv_ep(self.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None
    return net, mask, delta


# ------------------------------------------------------------------------------------------------ GREAT attention
_OUTLOOK = {(7, 8), (5, 16), (3, 32), (3, 48)}  # (kernel size, channels) with an outlook kernel
_VOL = {8, 16, 32, 48}  # volume-attention channels with a kernel


def _layer_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """nn.LayerNorm over the last dim: one kernel (fp32 statistics + affine) for contiguous fp16 rows."""
    c = x.shape[-1]
    if not (x.dtype == torch.float16 and x.is_cuda and c % 8 == 0 and x.is_contiguous()):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    y = torch.empty_like(x)
    p = x.numel() // c
    igevpp_kernels.load(x.device).launch(
        "layer_norm_rows", *linear(p), x, c, y, c, self._w32, self._b32, p, c, self.eps
    )  # fmt: skip
    return y


def _outlook_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """OutlookAttentionLayer: LayerNorm kernel, to_v conv, then one kernel for logits (upstream's flat reshape),
    softmax + eps and the window sum, one for the fold (no k^4 logits tensor, no unfold)."""
    b, c, h, w = x.shape
    k = self.kernel_size
    if (k, c) not in _OUTLOOK or x.dtype != torch.float16 or self.padding != k // 2:
        return transformers.OutlookAttentionLayer.forward(self, x)
    kk, n = k * k, h * w
    xn = self.norm(tokens(x))
    v = tokens(self.to_v(planes(xn)))
    lib = igevpp_kernels.load(x.device)
    o = torch.empty((b, n, kk, c), device=x.device, dtype=x.dtype)
    lib.launch(
        f"outlook_k{k}_c{c}", *linear(b * n * kk), xn, c, v, c, self._wa, self._ba, o, b, h, w, self.scale, self.eps
    )  # fmt: skip
    del v
    out = torch.empty((b, h, w, c), device=x.device, dtype=x.dtype)
    lib.launch(f"outlook_fold_k{k}_c{c}", *linear(b * n * c // 8), o, out, c, b, h, w)
    del o
    return self.to_out(planes(out))


def _ffn_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """FeedForwardLayer: LayerNorm kernel, 1x1 expand with bias + SiLU in the epilogue, the three depthwise convs as
    one kernel into the concatenated buffer, 1x1 with bias + SiLU, + y, 1x1 out."""
    y = conv_ep(self.in_conv, planes(self.norm(tokens(x))), act=ACT["silu"])
    b, e, h, w = y.shape
    if e % 8 or y.dtype != torch.float16 or not y.is_contiguous(memory_format=CL):
        z = torch.cat([self.dw_conv_3(y), self.dw_conv_5(y), self.dw_conv_7(y)], dim=1)
    else:
        z = torch.empty((b, 3 * e, h, w), device=y.device, dtype=y.dtype, memory_format=CL)
        igevpp_kernels.load(y.device).launch(
            "dw357", *linear(b * h * w * e // 8), y, e, *self._dw, z, b, h, w, e
        )  # fmt: skip
    o = conv_ep(self.pointwise_conv[0], z, act=ACT["silu"])
    del z
    return conv_ep(self.out_conv, o.add_(y))


def _volume_block_forward(self: nn.Module, volume: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
    """VolumeTransformerBlock in two kernels: per-pixel softmax statistics over D, then per token the attention
    update + the feed-forward (LayerNorm, Linear, SiLU, Linear, residual) in registers, in place."""
    t = volume.permute(0, 2, 3, 4, 1)
    c = t.shape[-1]
    if c not in _VOL or t.dtype != torch.float16:
        return transformers.VolumeTransformerBlock.forward(self, volume, feats)
    if not t.is_contiguous():
        t = t.contiguous()
    a = self.volume_attn
    kv = a.to_kv(a.norm_context(tokens(feats))).float()
    g = ((kv[..., :c] @ self._wq) * self._gs).contiguous()
    vo = (kv[..., c:] @ self._wo).contiguous()
    del kv
    b, d, h, w, _ = t.shape
    hw = h * w
    st = torch.empty((b * hw, 2), device=t.device, dtype=torch.float32)
    lib = igevpp_kernels.load(t.device)
    lib.launch(f"vol_stats_c{c}", *linear(b * hw), t, g, st, b, d, hw, a.norm.eps)
    n = b * d * hw
    lib.launch(
        f"vol_update_c{c}", *linear(n, 128 if c >= 32 else 256), t, g, st, vo, self._prm, b, d, hw,
        a.norm.eps, self.feedforward[0].eps, a.eps, smem=self._prm.numel() * 4,
    )  # fmt: skip
    return t.permute(0, 4, 1, 2, 3)


def _eav_disparity_kernels(
    self: nn.Module, x: torch.Tensor, feats: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """ExcitiveAttentionHourglassEncoder.disparity with the final volume softmax fused with the disparity regression:
    -> (volume [B, 8, D, H, W], initial disparity [B, 1, H, W] fp32)."""
    conv = self.hourglass(x, feats)
    t = conv.permute(0, 2, 3, 4, 1)
    t = t if t.is_contiguous() else t.contiguous()
    b, d, h, w, c = t.shape
    g = (tokens(feats[0]).float() * 8**-0.5).contiguous()
    if c != 8 or t.dtype != torch.float16:
        return conv, disparity_regression(volume_softmax(t, g, 1e-6), d)
    disp = torch.empty((b, 1, h, w), device=t.device, dtype=torch.float32)
    igevpp_kernels.load(t.device).launch("vol_regress_c8", *linear(b * h * w), t, g, disp, b, d, h * w, 1e-6)
    return conv, disp


def _cost_volume_kernels(
    eav: nn.Module, left: torch.Tensor, right: torch.Tensor, context: torch.Tensor
) -> torch.Tensor:
    """ExcitiveAttentionVolume.cost_volume without the concat volume: cc_stem[0] (1x3x3 conv on [L | R shifted by
    d]) split into a conv of L and a conv of R (+ its right-edge variant), summed per disparity plane by a kernel
    with the BN affine + LeakyReLU; the 3x3x3 convs and fuse_stem as before (disparity chunks, 2-plane halo)."""
    b, c, h, w = left.shape
    if b != 1 or left.dtype != torch.float16 or c % 8:
        return base_type(eav).cost_volume(eav, left, right, context)
    d = eav.max_disp // 4
    cl3 = torch.channels_last_3d
    st0 = eav.cc_stem[0]
    am = tokens(conv_ep(eav._cc_l, left))  # [1, H, W, C]
    rp = torch.empty((1, c, h, w + 1), device=right.device, dtype=right.dtype, memory_format=CL)
    rp[:, :, :, :1] = 0
    rp[:, :, :, 1:] = right
    bm = tokens(conv_ep(eav._cc_r, rp))  # [1, H, W + 1, 2C]: conv_R at columns -1 .. W - 1
    del rp
    feat = eav.feat_stem(context).unsqueeze(2)
    lib = igevpp_kernels.load(left.device)
    out = None
    for d0, d1 in d_chunks(d, c * h * w, CC_CHUNK):
        a, e = max(0, d0 - 2), min(d, d1 + 2)
        y = torch.empty((1, e - a, h, w, c), device=left.device, dtype=left.dtype)
        lib.launch(
            "cc_stem0", *linear((e - a) * h * w * c // 8), am, am.stride(2), bm, bm.stride(2), st0._s, st0._t, y, a,
            e - a, h, w, c,
        )  # fmt: skip
        y = eav.cc_stem[2](eav.cc_stem[1](y.permute(0, 4, 1, 2, 3)))
        y = y[:, :, d0 - a : d1 - a].contiguous(memory_format=cl3)
        y = eav.fuse_stem(y.mul_(feat))
        if out is None:
            out = torch.empty((b, y.shape[1], d, h, w), device=y.device, dtype=y.dtype, memory_format=cl3)
        out[:, :, d0:d1] = y
        del y
    return out


def _prepare_attention(model: nn.Module) -> None:
    """fp32 copies (before the fp16 cast) for the attention kernels; the split cc_stem[0] convs; the overrides."""
    for m in model.modules():
        t = base_type(m)
        if t is nn.LayerNorm and m.weight is not None and m.bias is not None:
            m._w32, m._b32 = m.weight.detach().float().contiguous(), m.bias.detach().float().contiguous()
            override(m, _layer_norm_forward)
        elif t is transformers.OutlookAttentionLayer:
            kk = m.kernel_size**2
            m._wa = m.to_attn.weight.detach()[: kk * kk].reshape(kk * kk, -1).float().contiguous()
            m._ba = m.to_attn.bias.detach()[: kk * kk].float().contiguous()
            override(m, _outlook_forward)
        elif t is transformers.FeedForwardLayer:
            dws = (m.dw_conv_3, m.dw_conv_5, m.dw_conv_7)
            m._dw = [c.weight.detach().float().flatten(1).t().contiguous() for c in dws]  # tap-major [k*k, E]
            m._dw += [c.bias.detach().float().contiguous() for c in dws]
            override(m, _ffn_forward)
        elif t is transformers.VolumeTransformerBlock:
            a, ff = m.volume_attn, m.feedforward
            m._wq = a.to_q.weight.detach().float().contiguous()
            m._gs = (a.scale * a.norm.weight.detach().float()).contiguous()
            m._wo = a.to_out.weight.detach().float().t().contiguous()
            parts = (a.to_out.bias, ff[0].weight, ff[0].bias, ff[1].bias, ff[3].bias, ff[1].weight, ff[3].weight)
            m._prm = torch.cat([p.detach().float().reshape(-1) for p in parts]).contiguous()
            override(m, _volume_block_forward)
        elif t is ExcitiveAttentionVolume:
            wt = m.cc_stem[0].conv.weight.detach()[:, :, 0]  # [C, 2C, 3, 3]
            c = wt.shape[0]
            wr = wt[:, c:]
            # registered submodules (not plain attributes): cnn_kernels.prepare_convs must see them
            m._cc_l = nn.Conv2d(c, c, 3, padding=1, bias=False).to(wt.device)
            m._cc_l.weight.data = wt[:, :c].contiguous()
            m._cc_r = nn.Conv2d(c, 2 * c, 3, padding=1, bias=False).to(wt.device)
            m._cc_r.weight.data = torch.cat([wr, wr * wr.new_tensor([1.0, 1.0, 0.0])], 0).contiguous()
            override(m, _cost_volume_kernels, "cost_volume")
        elif t is ExcitiveAttentionHourglassEncoder:
            override(m, _eav_disparity_kernels, "disparity")


# ------------------------------------------------------------------------------------------------ forward
@torch.no_grad()
def forward_kernels(self: nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32) -> torch.Tensor:
    """`portable.forward_fast` with the kernels: fp16 NHWC, disparity state + lookup coordinates + softmax / regression /
    upsampling in fp32, no host syncs in the loop, upsampling features only after the last iteration."""
    im1 = (2 * (image1 / 255.0) - 1.0).half().contiguous(memory_format=CL)
    im2 = (2 * (image2 / 255.0) - 1.0).half().contiguous(memory_format=CL)
    match_left, match_right, volume, init_disp, stem_2x = encode(self, im1, im2)
    del im2
    geo_fn = GeoLookup(match_left, match_right, volume, self.args.corr_radius)
    del volume, match_left, match_right

    cnet_list = self.cnet(im1, num_layers=self.args.n_gru_layers)
    del im1
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    del cnet_list
    ctx = prepare_context(self, inp_list)

    b, _, h, w = init_disp.shape
    disp = init_disp.float()
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=torch.float16, memory_format=CL)
    mask_feat_4 = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        net_list, mask_feat_4, delta_disp = update_block_fast(
            self.update_block, net_list, ctx, geo_feat, disp_h, itr == iters - 1
        )
        disp = disp + delta_disp.float()

    xspx = self.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(self.spx_gru(xspx).float(), 1)
    return context_upsample(disp * 4.0, spx_pred).unsqueeze(1)


# ------------------------------------------------------------------------------------------------ prepare
def prepare_kernels(model: nn.Module, device: torch.device) -> None:
    """Load the kernel libraries for `device` and patch `model` (on `device`, eval) for forward_kernels."""
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    igevpp_kernels.load(device)
    model.eval()
    # merged z|r conv weights of the GRUs (before the dtype cast; biases go into the context, see prepare_context)
    for g in (model.update_block.gru04, model.update_block.gru08, model.update_block.gru16):
        zr = nn.Conv2d(g.convz.in_channels, 2 * g.convz.out_channels, 3, padding=1, bias=False)
        zr.weight.data = torch.cat([g.convz.weight.data, g.convr.weight.data], 0)
        g.convzr = zr.to(g.convz.weight.device)
    for m in model.modules():
        t = base_type(m)
        if t is submodule.BasicConv:
            if m.use_bn:
                m._s, m._t = bn_affine(m.bn, m.conv)
            override(m, _basic_conv_forward)
        elif t is submodule.BasicConv_IN:
            override(m, _basic_conv_in_forward)
        elif t is extractor.ResidualBlock and isinstance(m.norm1, nn.BatchNorm2d):
            m.norm1._s, m.norm1._t = bn_affine(m.norm1, m.conv1)
            m.norm2._s, m.norm2._t = bn_affine(m.norm2, m.conv2)
            if m.downsample is not None:
                m.downsample[1]._s, m.downsample[1]._t = bn_affine(m.downsample[1], m.downsample[0])
            override(m, _resblock_forward)
        elif t is extractor.MultiBasicEncoder:
            m.norm1._s, m.norm1._t = bn_affine(m.norm1, m.conv1)
            override(m, _encoder_stem_forward)
    _prepare_attention(model)
    cnn_kernels.patch_norms(model)  # remaining BatchNorms (incl. the backbone's BatchNormAct2d) and bare InstanceNorms
    cnn_kernels.prepare_convs(model)
    cnn_kernels.fuse_narrow(
        model.cnet
    )  # 64 / 96-channel context convs as CUTLASS narrow tiles (+ the ResidualBlock tail)
