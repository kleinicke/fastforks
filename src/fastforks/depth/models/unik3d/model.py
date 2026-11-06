"""UniK3D runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `UniK3D.infer` with the demo settings (`resolution_level` 9 by default, bilinear output
interpolation): ImageNet normalisation, zero padding to the aspect-ratio bounds [0.5, 2.5], bilinear resize to the
level's pixel range (sides multiples of 14), the network, then the 3-D points (rays * radius) and the confidence
bilinearly resized back to the padded size and cropped; depth is the points' z. Given pinhole intrinsics are shifted by
the padding and scaled with the actual resize per axis (upstream: by the resize factor before the sides are rounded to
multiples of 14: rays up to 7 px off at 480 x 640, level 9), and their rays replace the predicted camera (the angular
module is then skipped). UniK3D's own camera is a field of view + spherical-harmonic ray model, not a pinhole matrix,
so no intrinsics are returned. Upstream runs everything under fp16 autocast on CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import OptionsError
from fastforks._core.half import half_weights, pure_half
from fastforks._core.patch import override
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.shared.depth_anything_v2 import flash_attention, sdpa_attention

from .net import LAYERS, PATCH, UniK3DNet, camera_rays, embed_rays, get_paddings, get_resize_factor, pinhole_rays

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

LEVELS = 10  # resolution_level 0-9


@dataclass(frozen=True)
class Params:
    """Load options."""

    resolution_level: int  # 0-9: network input of 200k-240k ... 560k-600k pixels
    confidence: bool  # compute the confidence map


@dataclass(frozen=True)
class _Geometry:
    """Upstream `infer` pre-processing of one input size."""

    paddings: tuple[int, int, int, int]  # left, right, top, bottom
    padded: tuple[int, int]
    factor: float
    net: tuple[int, int]


def _check(ctx: Context) -> None:
    if not 0 <= ctx.params.resolution_level < LEVELS:
        msg = f"UniK3D: resolution_level must be 0-{LEVELS - 1}, got {ctx.params.resolution_level}"
        raise OptionsError(msg)


def _net(ctx: Context) -> UniK3DNet:
    return UniK3DNet().load(ctx.state_dict(), f"{ctx.variant.ref} checkpoint").eval()


def _prepare(
    ctx: Context, image: torch.Tensor, intrinsics: torch.Tensor | None
) -> tuple[_Geometry, torch.Tensor, torch.Tensor | None]:
    """Normalised, padded, resized network input (float32) and the given camera's rays `[B, H * W, 3]` (or `None`)."""
    paddings, padded = get_paddings((image.shape[-2], image.shape[-1]))
    factor, size = get_resize_factor(padded, ctx.params.resolution_level)
    g = _Geometry(paddings, padded, factor, size)
    x = F.pad(scale(image, ctx.io), paddings, value=0.0)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    rays = None
    if intrinsics is not None:
        k = intrinsics.float().clone()
        k[:, 0, 2] += paddings[0] + 0.5  # OpenCV pixel centres -> the network's corner-origin pixels
        k[:, 1, 2] += paddings[2] + 0.5
        k[:, 0] *= size[1] / padded[1]  # fastforks: the actual resize per axis (upstream: the factor before the
        k[:, 1] *= size[0] / padded[0]  # sides are rounded to multiples of 14)
        rays = pinhole_rays(k, *size).flatten(2).transpose(1, 2)
    return g, x, rays


def _postprocess(t: torch.Tensor, g: _Geometry) -> torch.Tensor:
    """Upstream `_postprocess`: bilinear to the padded size, crop the padding."""
    t = F.interpolate(t, size=g.padded, mode="bilinear", align_corners=False)
    left, right, top, bottom = g.paddings
    return t[..., top : g.padded[0] - bottom, left : g.padded[1] - right]


def _restore(
    g: _Geometry, logr: torch.Tensor, logc: torch.Tensor | None, rays: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Log-radius / log-confidence `[B, 1, H, W]` and rays `[B, H * W, 3]` -> outputs at the input size."""
    b = logr.shape[0]
    h, w = g.net
    radius = torch.exp(logr.float().clip(-8.0, 8.0) + 2.0)
    points = rays.float().transpose(1, 2).reshape(b, 3, h, w) * radius
    pts = _postprocess(points, g)
    out = {"depth": pts[:, 2], "points": pts.permute(0, 2, 3, 1)}
    if logc is not None:
        out["confidence"] = _postprocess(torch.exp(logc.float().clip(-8.0, 10.0)), g)[:, 0]
    return out


def _refl_conv(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """`nn.Conv2d` with 1-pixel reflect padding: slices + cat (the `F.pad` reflect kernel is slower)."""
    x = torch.cat([x[..., 1:2, :], x, x[..., -2:-1, :]], -2)
    x = torch.cat([x[..., 1:2], x, x[..., -2:-1]], -1)
    return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def _rcu(r: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - a ResidualConvUnit (untyped upstream module)
    """`ResidualConvUnit` in place on `x`: `x + gamma * conv2(act(conv1(act(x))))`."""
    o = r.conv2(F.leaky_relu_(r.conv1(F.leaky_relu(x, SLOPE)), SLOPE))
    return x.addcmul_(o, r.gamma.to(o.dtype))


def _radial_fast(
    self: Any,  # noqa: ANN401 - the RadialModule
    features: list[torch.Tensor],
    rays: torch.Tensor,
    size: tuple[int, int],
    *,
    confidence: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`RadialModule.forward` with the conv part in pure fp16 channels-last (the attention part keeps autocast).

    Each stride == kernel `ConvTranspose2d` is one GEMM added into the latents, the final 1x1 conv a Linear.
    """
    b, h, w, _ = features[0].shape
    emb = embed_rays(rays, size, (h, w))
    cond = [p(f.reshape(b, h * w, -1), emb) for p, f in zip(self.prompt_camera, features, strict=True)]
    lat = self.to_latents(cond[0])
    with torch.autocast("cuda", enabled=False):
        x = lat.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        for i, up in enumerate(self.ups):
            k, cout, wm, bias = self._ff_pf[i]
            y = F.linear(cond[i + 1].reshape(b * h * w, -1).to(wm.dtype), wm, bias)
            xv = x.permute(0, 2, 3, 1).view(b, h, k, w, k, cout)
            xv.add_(y.view(b, h, w, k, k, cout).permute(0, 1, 3, 2, 4, 5))
            for r in up.convs:
                x = _rcu(r, x)
            x = F.interpolate(up.up[0](x), scale_factor=2, mode="bilinear", align_corners=False)
        out = []
        for mlp, lr, hr in [(self.depth_mlp[-1], self.to_depth_lr, self.to_depth_hr)] + (
            [(self.confidence_mlp, self.to_confidence_lr, self.to_confidence_hr)] if confidence else []
        ):
            t = lr(mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            t = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
            t = F.leaky_relu_(hr[0](t), SLOPE)
            c = hr[2]
            out.append(
                F.linear(t.permute(0, 2, 3, 1), c.weight.view(c.weight.shape[0], -1), c.bias).permute(0, 3, 1, 2)
            )
    return out[0], out[1] if confidence else None


class Portable(Runner):
    """The network with fused SDPA attention: float32, or fp16 autocast (upstream: on CUDA)."""

    intrinsics = "optional"
    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        _check(ctx)
        self.net = _net(ctx).to(ctx.device)
        sdpa_attention(self.net)
        self.half = ctx.dtype == "float16"
        if self.half:  # fp16 encoder, fp16 decoder weights under autocast (no per-call weight casts)
            pure_half(self.net.pixel_encoder, channels_last=False)
            half_weights(self.net.pixel_decoder)
            for m in self.net.pixel_decoder.modules():
                if isinstance(m, nn.Conv2d) and m.padding_mode == "reflect" and m.padding == (1, 1):
                    override(m, _refl_conv)
            self._fast_decoder(self.net.pixel_decoder.radial_module)

    @staticmethod
    def _fast_decoder(rad: Any) -> None:  # noqa: ANN401 - the RadialModule
        """Pure fp16 channels-last conv part; the transposed convs as GEMM weights `[(i j cout), cin]`."""
        for m in (
            rad.ups,
            rad.depth_mlp,
            rad.to_depth_lr,
            rad.to_depth_hr,
            rad.confidence_mlp,
            rad.to_confidence_lr,
            rad.to_confidence_hr,
        ):
            m.half()
            for c in m.modules():
                if isinstance(c, nn.Conv2d):
                    c.weight.data = c.weight.data.contiguous(memory_format=torch.channels_last)
        pf = []
        for t in rad.process_features:
            k = t.kernel_size[0]
            cout = t.weight.shape[1]
            wm = t.weight.data.half().permute(2, 3, 1, 0).reshape(k * k * cout, -1).contiguous()
            pf.append((k, cout, wm, t.bias.data.half().repeat(k * k)))
        rad._ff_pf = pf  # noqa: SLF001
        override(rad, _radial_fast, "forward")

    def _run(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        g, x, rays = _prepare(self.ctx, image, intrinsics)
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.half):
            logr, logc, rays = self.net(x, rays, confidence=self.ctx.params.confidence)
        return _restore(g, logr, logc, rays)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Metric depth, points and confidence of a batch."""
        return self._run(image, intrinsics)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> metric depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image, None)["depth"])


# ------------------------------------------------------------------------------------------------ kernels
from fastforks.depth.models.depth_anything_v2.fused import FusedViT  # noqa: E402 - the dpt_fused ViT
from fastforks.kernels.dpt_fused import ACT_GELU, DptFused  # noqa: E402
from fastforks.kernels.dpt_fused import load as load_dpt_fused  # noqa: E402
from fastforks.kernels.fastbb import Flash64  # noqa: E402

SLOPE = 0.01  # nn.LeakyReLU default


class _ViT(FusedViT):
    """`FusedViT` with UniK3D's position table (resized by output size, `interpolate_offset` 0) and class tokens."""

    def interpolate_pos(self, h0: int, w0: int) -> torch.Tensor:
        """Upstream `interpolate_pos_encoding` for an `h0 x w0` grid (bicubic in fp32, by output size)."""
        pe = self.pos_fp32.to(self.vit.pos_embed.device)
        m = math.isqrt(pe.shape[1] - 1)
        p = F.interpolate(pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2), size=(h0, w0), mode="bicubic")
        return torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)

    def layer_tokens(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Normalised network input `[B, 3, H, W]` -> normalised tokens `[B, 1 + h * w, C]` (fp16) of `LAYERS`."""
        vit, k = self.vit, self.k
        size = (x.shape[-2], x.shape[-1])
        gh, gw = size[0] // PATCH, size[1] // PATCH
        cols = k.patches(x, size, PATCH, self.kp, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # identity resize: exact
        pos, cls = self._pos_tables(gh, gw)
        b = x.shape[0]
        t = torch.empty((b, gh * gw + 1, vit.embed_dim), device=x.device, dtype=torch.float16)
        t[:, 0] = cls
        for i in range(b):
            k.gemm(cols[i], self.w_patch, self.b_patch, res=pos, out=t[i, 1:], acc32=True)
        outs = []
        for i in range(LAYERS[-1] + 1):
            t = self._block(i, t)
            if i in LAYERS:
                outs.append(vit.norm(t))
        return outs


def _w(m: Any) -> torch.Tensor:  # noqa: ANN401 - a Linear
    return m.weight.detach().half().contiguous()


def _b(m: Any) -> torch.Tensor:  # noqa: ANN401 - a Linear / Conv2d with bias
    return m.bias.detach().half().contiguous()


def _conv_w(c: Any, gamma: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - a Conv2d
    """Conv2d -> (weight `[K, R, S, C]` fp16, bias fp16), optionally scaled per output channel by `gamma`."""
    w, b = c.weight.detach().float(), c.bias.detach().float()
    if gamma is not None:
        g = gamma.detach().float().flatten()
        w, b = w * g.view(-1, 1, 1, 1), b * g
    return w.permute(0, 2, 3, 1).contiguous().half(), b.half()


def _pad_reflect(x: torch.Tensor) -> torch.Tensor:
    """Reflect-pad a contiguous NHWC tensor by 1."""
    b, h, w, c = x.shape
    p = x.new_empty((b, h + 2, w + 2, c))
    p[:, 1:-1, 1:-1] = x
    p[:, 0, 1:-1] = x[:, 1]
    p[:, -1, 1:-1] = x[:, -2]
    p[:, :, 0] = p[:, :, 2]
    p[:, :, -1] = p[:, :, -3]
    return p


class _Head:
    """A log-depth / log-confidence head on NHWC fp16.

    LayerNorm + Linear, reflect 3x3 conv, bilinear (align corners) to the network size, reflect 3x3 conv + LeakyReLU,
    1x1 conv.
    """

    def __init__(self, mlp: Any, lr: Any, hr: Any, device: torch.device) -> None:  # noqa: ANN401 - Sequential / Conv2d
        """Convert the weights."""
        ln, lin = mlp[0], mlp[1]
        self.ln = (
            ln.normalized_shape,
            ln.weight.detach().half().to(device),
            ln.bias.detach().half().to(device),
            ln.eps,
        )
        self.lin = (_w(lin).to(device), _b(lin).to(device))
        self.lr = tuple(t.to(device) for t in _conv_w(lr))
        self.hr = tuple(t.to(device) for t in _conv_w(hr[0]))
        self.out = (hr[2].weight.detach().half().view(1, -1).to(device), hr[2].bias.detach().half().to(device))

    def __call__(self, k: DptFused, z: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Features `[B, h, w, C]` -> log map `[B, 1, H, W]` (fp16)."""
        shape, lw, lb, eps = self.ln
        t = k.gemm(F.layer_norm(z, shape, lw, lb, eps), *self.lin)
        y = k.conv(_pad_reflect(t), *self.lr)
        y = k.bilinear(y, size, align_corners=True)
        y = F.leaky_relu_(k.conv(_pad_reflect(y), *self.hr), SLOPE)
        return F.linear(y, *self.out).permute(0, 3, 1, 2)


class Kernels(Runner):
    """`dpt_fused` kernels: fused-epilogue GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 decoder.

    Encoder as `FusedViT`; the camera-prompt cross-attention layers as GEMMs (residuals in the epilogues) + flash
    attention; the kernel == stride deconvolutions as shuffling GEMMs; the residual conv units with LayerScale folded
    into the second conv. The camera head (only without intrinsics) and the ray embedding run in PyTorch.
    """

    intrinsics = "optional"
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        _check(ctx)
        dev = ctx.device
        k = load_dpt_fused(dev)  # before the weights: a kernel build / load failure costs no weight load
        net = _net(ctx)
        self.k, self.flash = k, Flash64("hh")
        dec: Any = net.pixel_decoder
        rm: Any = dec.radial_module
        self.vit = _ViT(net.pixel_encoder, dev, k)
        self.adapters = [(_w(a).to(dev), _b(a).to(dev)) for a in dec.input_adapter.input_adapters]
        self.camera = (dec.camera_token_adapter.to(dev), dec.angular_module.to(dev))  # fp32, run under autocast
        self.prompts = []
        for p in rm.prompt_camera:
            blk = p.layers[0]
            m = blk.mlp
            norms = [n.half().to(dev) for n in (blk.norm_attnx, blk.norm_attnctx, m.norm)]
            ws = [_w(blk.q), _w(blk.kv), _w(blk.out), _w(m.proj1), _b(m.proj1), _w(m.proj2), _b(m.proj2)]
            self.prompts.append((*norms, *(t.to(dev) for t in ws)))
        self.latents = (_w(rm.to_latents).to(dev), _b(rm.to_latents).to(dev))
        self.pf = []
        for pf in rm.process_features:
            s = pf.stride[0]
            w = pf.weight.detach().half()  # [Cin, Cout, s, s] -> rows (a, b, cout)
            wg = w.permute(2, 3, 1, 0).reshape(s * s * w.shape[1], w.shape[0]).contiguous().to(dev)
            self.pf.append((s, wg, pf.bias.detach().half().repeat(s * s).to(dev)))
        self.ups = []
        for u in rm.ups:
            rcus = [tuple(t.to(dev) for t in (*_conv_w(r.conv1), *_conv_w(r.conv2, r.gamma))) for r in u.convs]
            c = u.up[0]
            self.ups.append((rcus, c.weight.detach()[:, :, 0, 0].half().contiguous().to(dev), _b(c).to(dev)))
        self.depth = _Head(rm.depth_mlp[-1], rm.to_depth_lr, rm.to_depth_hr, dev)
        self.conf = _Head(rm.confidence_mlp, rm.to_confidence_lr, rm.to_confidence_hr, dev)

    def _prompt(self, p: tuple, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Camera-prompt block on `[B, N, C]` fp16 features (in place) with the ray embedding `emb`."""
        nx, nc, nm, wq, wkv, wo, w1, b1, w2, b2 = p
        k = self.k
        b, n, c = x.shape
        heads = c // 64
        kv = k.gemm(nc(emb), wkv).view(b, n, 2, heads, 64)
        q = k.gemm(nx(x), wq).view(b, n, heads, 64)
        o = flash_attention(q, kv[:, :, 0], kv[:, :, 1], 0.125, self.flash)
        k.gemm(o.view(b, n, c), wo, res=x, out=x)
        hid = k.gemm(nm(x), w1, b1, act=ACT_GELU)
        return k.gemm(hid, w2, b2, res=x, out=x)

    def _rcu(self, r: tuple, z: torch.Tensor) -> torch.Tensor:
        """`z + gamma * conv2(leaky(conv1(leaky(z))))`, in place on `z` (gamma folded into conv2)."""
        w1, b1, w2, b2 = r
        o = self.k.conv(F.leaky_relu(z, SLOPE), w1, b1, padding=1)
        return self.k.conv(F.leaky_relu_(o, SLOPE), w2, b2, padding=1, res=z, out=z)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Metric depth, points and confidence of a batch."""
        g, x, rays = _prepare(self.ctx, image, intrinsics)
        k = self.k
        b, size = x.shape[0], g.net
        h, w = size[0] // PATCH, size[1] // PATCH
        toks = self.vit.layer_tokens(x)
        feats = []
        for t, (wa, ba) in zip(toks, self.adapters, strict=True):
            f = t.new_empty((b, h * w, wa.shape[0]))
            for i in range(b):
                k.gemm(t[i, 1:], wa, ba, out=f[i])
            feats.append(f)
        if rays is None:
            adapter, angular = self.camera
            with torch.autocast("cuda", dtype=torch.float16):
                fov, sh = angular(torch.cat(adapter([t[:, :1] for t in toks]), 1), size)
                rays = camera_rays(fov, sh, *size)
        emb = embed_rays(rays.float(), size, (h, w)).half()
        cond = [self._prompt(p, f, emb) for p, f in zip(self.prompts, feats, strict=True)]
        z = k.gemm(cond[0], *self.latents).view(b, h, w, -1)
        for i, (rcus, wu, bu) in enumerate(self.ups):
            s, wg, bg = self.pf[i]
            if s == 1:
                k.gemm(cond[i + 1], wg, bg, res=z, out=z)
            else:
                z += k.gemm(cond[i + 1], wg, bg, shuffle=(h, w, s, s))
            for r in rcus:
                z = self._rcu(r, z)
            z = k.gemm(z, wu, bu)
            z = k.bilinear(z, (2 * z.shape[1], 2 * z.shape[2]), align_corners=False)
        logr = self.depth(k, z, size)
        logc = self.conf(k, z, size) if self.ctx.params.confidence else None
        return _restore(g, logr, logc, rays)


RUNNERS = {"portable": Portable, "kernels": Kernels}
