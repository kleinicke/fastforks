"""UniDepth V2 runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `UniDepthV2.infer` with the released config (`resolution_level` unset, bilinear output
interpolation): ImageNet normalisation, zero padding to the aspect-ratio bounds [0.5, 2.5], bilinear resize to
200k-600k pixels (sides multiples of 14), the network, then the 3-D points (rays * radius), confidence and intrinsics
bilinearly resized back to the padded size and cropped. Given intrinsics are shifted by the padding and scaled with
the resize per axis (upstream scales by the resize factor before the sides are rounded up to multiples of 14, which
moves the points of a 480 x 640 image by up to 3.6 px), and their rays replace the predicted camera (upstream still
predicts intrinsics then; the runners return the given ones instead). Predicted intrinsics are mapped back the same
way, per axis (upstream divides by the scalar factor: up to 4 / 10 px reprojection error at 480 x 640). Upstream runs
everything under fp16 autocast on CUDA.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate, half_weights
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.kernels import cudnn_hacc, dpt_fused
from fastforks.shared.depth_anything_v2 import use_sdpa_attention
from fastforks.shared.depth_anything_v2.fast_vit import FastViT

from .net.unidepthv2 import UniDepthV2, _postprocess, get_paddings, get_resize_factor
from .net.utils import PinholeCamera

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.depth._runner import Context

RATIO_BOUNDS = (0.5, 2.5)  # fixed: released config data.augmentations.shape_constraints.ratio_bounds
PIXELS_BOUNDS = (200000, 600000)  # fixed: released config shape_constraints pixels_min / pixels_max


def _config() -> dict[str, Any]:
    """The inference part of the released `config.json` (a fresh copy: `UniDepthV2.build` writes into it)."""
    return {
        "training": {},
        "data": {"augmentations": {"shape_constraints": {"ratio_bounds": list(RATIO_BOUNDS), "shape_mult": 14}}},
        "model": {
            "expansion": 4,
            "layer_scale": 1.0,
            "num_heads": 8,
            "pixel_decoder": {"depths": [2, 2, 2], "dropout": 0.0, "hidden_dim": 512, "kernel_size": 3, "out_dim": 64},
            "pixel_encoder": {
                "name": "dinov2_vitl14",
                "num_register_tokens": 0,
                "output_idx": [6, 12, 18, 24],
                "pretrained": None,
                "stacking_fn": "last",
                "use_norm": True,
                "frozen_stages": 0,
            },
        },
    }


@dataclass(frozen=True)
class Params:
    """Load options."""

    confidence: bool  # compute the confidence map


@dataclass(frozen=True)
class _Geometry:
    """Upstream `infer` pre-processing of one input size."""

    paddings: tuple[int, int, int, int]  # left, right, top, bottom
    padded: tuple[int, int]
    net: tuple[int, int]

    @property
    def scales(self) -> tuple[float, float]:
        """Actual resize per axis `(sx, sy)` padded -> network (the sides are rounded up to 14 px)."""
        return self.net[1] / self.padded[1], self.net[0] / self.padded[0]


def _geometry(h: int, w: int) -> _Geometry:
    paddings, padded = get_paddings((h, w), RATIO_BOUNDS)
    _, net = get_resize_factor(padded, PIXELS_BOUNDS)
    return _Geometry(tuple(paddings), padded, net)


def _net(ctx: Context) -> UniDepthV2:
    net = UniDepthV2(_config())
    net.load_state_dict(ctx.state_dict())
    return net.eval().to(ctx.device)


def _prepare(ctx: Context, image: torch.Tensor, intrinsics: torch.Tensor | None) -> tuple:
    """Normalised, padded, resized network input (float32) and the camera in network pixels (or `None`)."""
    g = _geometry(*image.shape[-2:])
    x = F.pad(scale(image, ctx.io), g.paddings, value=0.0)
    x = F.interpolate(x, size=g.net, mode="bilinear", align_corners=False)
    cam = None
    if intrinsics is not None:
        pl, _, pt, _ = g.paddings
        cam = PinholeCamera(intrinsics.float()).crop(left=-pl - 0.5, top=-pt - 0.5).resize(*g.scales)  # corner origin
    return g, x, cam


def _restore(
    g: _Geometry,
    points: torch.Tensor,
    confidence: torch.Tensor | None,
    k_net: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Network-size maps -> outputs at the input size (upstream `infer` post-processing)."""
    pts = _postprocess(points.float(), g.padded, g.paddings)
    out = {"depth": pts[:, 2], "points": pts.permute(0, 2, 3, 1)}
    if confidence is not None:
        out["confidence"] = _postprocess(confidence.float(), g.padded, g.paddings)[:, 0]
    if intrinsics is None and k_net is not None:
        sx, sy = g.scales  # fastforks: per axis (upstream `_postprocess_intrinsics`: the scalar resize factor)
        out["intrinsics"] = (
            PinholeCamera(k_net.float()).resize(1 / sx, 1 / sy).crop(g.paddings[0] + 0.5, g.paddings[2] + 0.5).K
        )
    return out


_fp16_accumulate = fp16_accumulate


def _pad_reflect(x: torch.Tensor, p: int) -> torch.Tensor:
    """`F.pad(x, (p,) * 4, mode="reflect")` for a channels-last `x`, channels-last out.

    torch's pad returns NCHW and the layout copy after it costs more than the pad: one interior copy plus the border
    strips instead.
    """
    n, c, h, w = x.shape
    y = torch.empty((n, c, h + 2 * p, w + 2 * p), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    y[:, :, p : p + h, p : p + w] = x
    y[:, :, :p, p : p + w] = x[:, :, 1 : p + 1].flip(2)
    y[:, :, p + h :, p : p + w] = x[:, :, h - p - 1 : h - 1].flip(2)
    y[:, :, :, :p] = y[:, :, :, p + 1 : 2 * p + 1].flip(3)
    y[:, :, :, p + w :] = y[:, :, :, w - 1 : w + p - 1].flip(3)
    return y


class _ViT(FastViT):
    """The shared fast ViT with UniDepth's token layout.

    No register tokens (the checkpoint holds an unused one) and the position table resized to the patch grid by size
    (`interpolate_offset` 0), as upstream.
    """

    def __init__(self, vit: Any, attn: str, fused: dpt_fused.DptFused | None) -> None:  # noqa: ANN401 - vendored ViT
        """As `FastViT`; upstream's `train()` override returns `None`, so `eval()` cannot be chained."""
        super().__init__(vit, attn, None, fused=fused)
        self.vit = vit

    def interpolate_pos(self, h0: int, w0: int) -> torch.Tensor:
        """Upstream `interpolate_pos_encoding` for an `h0 x w0` grid (bicubic in fp32, by output size)."""
        vit, pe = self.vit, self.pos_fp32.to(self.vit.pos_embed.device)
        n = pe.shape[1] - 1
        m = round(n**0.5)
        if n == h0 * w0 and h0 == w0:
            return pe
        p = F.interpolate(
            pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode="bicubic",
            antialias=vit.interpolate_antialias,
        )
        return torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens + cls token + positions, fp16."""
        vit = self.vit
        b, _, h, w = x.shape
        t = vit.patch_embed(x.half())
        t = torch.cat([vit.cls_token.expand(b, -1, -1), t], 1)
        return t + self.pos_embed(h // vit.patch_size, w // vit.patch_size)


def _conv_hacc(x: torch.Tensor, c: Any) -> torch.Tensor:  # noqa: ANN401 - a vendored Conv2d
    """fp16 channels-last conv (fp16 accumulation); reflect padding as a separate pad."""
    p = c.padding[0]
    if c.padding_mode == "reflect" and p > 0:
        x = _pad_reflect(x, p)
        p = 0
    return cudnn_hacc.conv_half_acc(x, c.weight, c.bias, [p, p], list(c.stride), list(c.dilation))


def _conv_plain(x: torch.Tensor, c: Any) -> torch.Tensor:  # noqa: ANN401 - a vendored Conv2d
    """fp16 channels-last conv; reflect padding as a separate pad."""
    p = c.padding[0]
    if c.padding_mode == "reflect" and p > 0:
        x = _pad_reflect(x, p)
        p = 0
    return F.conv2d(x, c.weight, c.bias, c.stride, p, c.dilation, c.groups).contiguous(
        memory_format=torch.channels_last
    )


def _interp_plain(x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
    """Bilinear `F.interpolate` of a channels-last map."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def _interp_nhwc(x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
    """Bilinear `F.interpolate` of a channels-last fp16 map with the dpt_fused NHWC kernel (8 channels per thread).

    torch's NHWC bilinear kernel takes about 8x longer on these maps.
    """
    y = dpt_fused.load(x.device).bilinear(x.permute(0, 2, 3, 1), size, align_corners=align_corners)
    return y.permute(0, 3, 1, 2)


class _SdpaViT(_ViT):
    """`_ViT` in plain PyTorch: SDPA attention (flash backend) and cuBLAS GEMMs with fp32 accumulation."""

    def attend(self, qkv: torch.Tensor, scale: float) -> torch.Tensor:
        """softmax(q k^T * scale) v of the packed qkv `[B, N, 3, H, 64]` -> `[B, N, H * 64]`."""
        b, n, _, h, d = qkv.shape
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        return F.scaled_dot_product_attention(q, k, v, scale=scale).transpose(1, 2).reshape(b, n, h * d)

    def accumulate(self) -> contextlib.AbstractContextManager[None]:
        """No fp16 accumulation: the portable path keeps the default GEMM precision."""
        return contextlib.nullcontext()


class _Fast:
    """The fp16 depth path: fp16 ViT, fp16 channels-last conv decoder, the decoder's attention part as upstream.

    Kernels and the fp16 Portable differ in the ViT (flash kernel or SDPA), the convolution and the GEMM accumulation
    context only. The decoder's attention part runs under fp16 autocast, as upstream (its Linear / Conv weights are
    stored in fp16, which gives the same results as fp32 weights).
    """

    def __init__(
        self, ctx: Context, net: UniDepthV2, vit: _ViT, conv: Callable[..., torch.Tensor], acc: Callable[[], Any]
    ) -> None:
        """Convert the decoder of `net` (`vit` has converted the encoder)."""
        self.ctx, self.net, self.vit, self._conv, self._acc = ctx, net, vit, conv, acc
        self.interp: Callable[..., torch.Tensor] = _interp_plain  # Kernels sets the NHWC kernel
        self.dec: Any = net.pixel_decoder
        self.head: Any = net.pixel_decoder.depth_layer
        half_weights(net.pixel_decoder, channels_last=False)
        self.layers = [j - 1 for _, j in net.slices_encoder_range]  # "last" stacking: the last block of each slice
        h = self.head
        convs = [h.ups, h.depth_mlp, h.to_depth_lr, h.to_depth_hr, h.confidence_mlp, h.to_confidence_lr]
        convs.append(h.to_confidence_hr)
        for m in convs:
            m.half()
            for c in m.modules():
                if isinstance(c, torch.nn.Conv2d):
                    c.weight.data = c.weight.data.contiguous(memory_format=torch.channels_last)
        # ConvTranspose2d with stride == kernel as one GEMM: W [Cin, Cout, k, k] -> [(i j Cout), Cin]
        self.pf = []
        for pf in h.process_features:
            k = pf.kernel_size[0]
            cin, cout = pf.weight.shape[:2]
            wm = pf.weight.detach().permute(2, 3, 1, 0).reshape(k * k * cout, cin).half().contiguous()
            b = pf.bias.detach().repeat(k * k).half() if pf.bias is not None else None
            self.pf.append((k, cout, wm, b))

    def _rcu(self, r: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored ResidualConvUnit
        """`x + gamma * conv2(act(conv1(act(x))))` (no norms), in place on `x`."""
        slope = r.activation.negative_slope
        o = self._conv(F.leaky_relu(x, slope), r.conv1)
        o = self._conv(F.leaky_relu_(o, slope), r.conv2)
        return x.addcmul_(o, r.gamma.view(1, -1, 1, 1))

    def _out(self, x: torch.Tensor, mlp: Any, lr: Any, hr: Any, size: tuple[int, int]) -> torch.Tensor:  # noqa: ANN401 - vendored Sequential
        """Upstream `depth_proj` / `confidence_proj` of the last level; returns the log map `[B, H, W, 1]`.

        LayerNorm + Linear, conv, upsample, conv, LeakyReLU, 1x1 conv (identity interpolations dropped).
        """
        ln, lin = mlp[0], mlp[1]
        t = F.linear(
            F.layer_norm(x.permute(0, 2, 3, 1), ln.normalized_shape, ln.weight, ln.bias, ln.eps), lin.weight, lin.bias
        )
        y = self._conv(t.permute(0, 3, 1, 2), lr)
        y = self.interp(y, size, align_corners=True)
        c0, act, c1 = hr[0], hr[1], hr[2]
        y = F.leaky_relu_(self._conv(y, c0), act.negative_slope)
        return F.linear(y.permute(0, 2, 3, 1), c1.weight.view(c1.weight.shape[0], -1), c1.bias)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Run a batch."""
        g, x, cam = _prepare(self.ctx, image, intrinsics)
        b, (nh, nw) = x.shape[0], g.net
        ps = self.vit.vit.patch_size
        h, w = nh // ps, nw // ps
        outs = self.vit.get_intermediate_layers(x, self.layers, return_class_token=True)
        dec, hd = self.dec, self.head
        k_net = None
        with torch.autocast("cuda", dtype=torch.float16), self._acc():
            feats = dec.input_adapter([t.view(b, h, w, -1) for t, _ in outs])
            if cam is None:  # predicted camera (upstream computes it also with a given camera, then ignores it)
                toks = dec.camera_token_adapter([c[:, None] for _, c in outs])
                dec.camera_layer.set_shapes((nh, nw))
                k_net, rays_flat = dec.run_camera(
                    torch.cat(toks, 1),
                    features=feats[0][..., None],
                    pos_embed=None,
                    original_shapes=(nh, nw),
                    rays_gt=None,
                )
                rays = rays_flat.view(b, nh, nw, 3).permute(0, 3, 1, 2)
            else:
                rays = cam.get_rays(shapes=(b, nh, nw))
                rays_flat = rays.flatten(2).transpose(1, 2)
            hd.set_shapes((h, w))
            hd.set_original_shapes((nh, nw))
            remb = hd.embed_rays(rays_flat)
            cond = [p(f.reshape(b, h * w, -1), remb) for p, f in zip(hd.prompt_camera, feats, strict=False)]
            lat = hd.to_latents(cond[0])
        with self._acc():
            z = lat.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            for i, u in enumerate(hd.ups):
                k, cout, wm, bias = self.pf[i]
                y = F.linear(cond[i + 1].reshape(b * h * w, -1).half(), wm, bias)  # [(b h w), (i j c)]
                zv = z.permute(0, 2, 3, 1).view(b, h, k, w, k, cout)
                zv.add_(y.view(b, h, w, k, k, cout).permute(0, 1, 3, 2, 4, 5))
                for r in u.convs:
                    z = self._rcu(r, z)
                z = self._conv(z, u.up[0])
                z = self.interp(z, (z.shape[2] * 2, z.shape[3] * 2), align_corners=False)
            logd = self._out(z, hd.depth_mlp[-1], hd.to_depth_lr, hd.to_depth_hr, (nh, nw))
            logc = None
            if self.ctx.params.confidence:
                logc = self._out(z, hd.confidence_mlp, hd.to_confidence_lr, hd.to_confidence_hr, (nh, nw))
        radius = torch.exp(logd.float().clamp(-8.0, 8.0) + 2.0).permute(0, 3, 1, 2)
        conf = None if logc is None else torch.exp(logc.float().clamp(-8.0, 8.0)).permute(0, 3, 1, 2)
        return _restore(g, rays.float() * radius, conf, k_net, intrinsics)


class Portable(Runner):
    """The vendored network with fused SDPA attention: float32, or on CUDA the fp16 path in plain PyTorch.

    float16 is the fork's `FastDepth`: fp16 ViT (SDPA attention, final norm only on the layers the decoder reads),
    fp16 channels-last conv decoder (`F.conv2d`, the stride == kernel ConvTranspose2d as one GEMM), the decoder's
    attention part as upstream under fp16 autocast with fp16 weights; no fp16-accumulating GEMMs. float32 runs the
    vendored network as is.
    """

    intrinsics = "optional"
    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        use_sdpa_attention(self.net)
        self.fast = None
        if ctx.dtype == "float16":
            self.fast = _Fast(
                ctx, self.net, _SdpaViT(self.net.pixel_encoder, "hh", None), _conv_plain, contextlib.nullcontext
            )

    def _run(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if self.fast is not None:
            return self.fast(image, intrinsics)
        g, x, cam = _prepare(self.ctx, image, intrinsics)
        _, o = self.net.encode_decode({"image": x, "camera": cam})
        conf = o["confidence"] if self.ctx.params.confidence else None
        return _restore(g, o["points"], conf, o["intrinsics"], intrinsics)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Metric depth, points, confidence and (without given intrinsics) estimated intrinsics of a batch."""
        return self._run(image, intrinsics)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> metric depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image, None)["depth"])


class Kernels(Runner):
    """fastbb ViT (flash attention, fp16-accumulating GEMMs) and an fp16 channels-last conv decoder.

    The decoder convs are cuDNN fp16-accumulate convs, the bilinear upsamples the dpt_fused NHWC kernel; the decoder's
    attention part runs as upstream under fp16 autocast.
    """

    intrinsics = "optional"
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert it."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        self.net = _net(ctx)
        vit = _ViT(self.net.pixel_encoder, "hh", dpt_fused.load(ctx.device))
        self.fast = _Fast(ctx, self.net, vit, _conv_hacc, _fp16_accumulate)
        self.fast.interp = _interp_nhwc

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Metric depth, points, confidence and (without given intrinsics) estimated intrinsics of a batch."""
        return self.fast(image, intrinsics)


RUNNERS = {"portable": Portable, "kernels": Kernels}
