"""AnyCalib runners: DINOv2 ViT-L/14 + light DPT decoder -> one ray per pixel -> pinhole intrinsics.

Upstream `AnyCalib.predict(image, cam_id="pinhole")`: the input is resized to about `resolution` pixels at its aspect
ratio (`set_im_size`: bicubic, centre crop to the target aspect ratio, antialiased bicubic downscale), the network
predicts tangent coordinates at the optical axis (convex 7x upsampling) mapped to unit rays, and the camera is fitted
to the rays: a linear least-squares fit (`Pinhole.fit`), then `gn_iters` Gauss-Newton steps (residuals in the tangent
space of the fitted rays); the result is mapped back to the input image (`reverse_scale_and_shift`). The vertical field
of view is upstream's `get_vfov` at the input height.

Differences from upstream: the fit runs on the whole batch at once (upstream: a Python loop over the images; the same
equations, `net/pinhole.py` and `net/gauss_newton.py` broadcast over the batch); an image whose linear fit fails gives
NaN intrinsics (upstream falls back to RANSAC, not ported); the inverse covariance of the fit is not computed;
the choice between the linear and the refined fit (the refined one unless its cost is higher) is a `torch.where`
instead of a host branch, so a call has no device-to-host sync.

`portable`: plain PyTorch, the network in float32 or float16 (SDPA attention); the head's upsampling, the ray map
and the fit always in float32. `kernels`: the `fastbb` ViT (flash attention, `dpt_fused` block GEMMs, fp16
accumulate), `cudnn_hacc` decoder convolutions, the network replayed from a CUDA graph per input shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks._core.patch import is_base, override
from fastforks.calibration._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc, dpt_fused
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.depth_anything_v2 import FastViT, fp16_accumulate, sdpa_attention, vit_large
from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer

from .net.dpt_light_decoder import LightDPTDecoder
from .net.gauss_newton import GaussNewtonCalib
from .net.manifolds import Unit3
from .net.pinhole import Pinhole
from .net.ray_decoder import ConvexTangentDecoder, cvx_upsample

if TYPE_CHECKING:
    from fastforks.calibration._runner import Context

_OUT_INDEX = (4, 11, 17, 23)  # upstream DINOV2_CFG["dinov2_vitl14"]["out_index"]: un-normed block outputs
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Params:
    """Card / variant `params`.

    Attributes:
        resolution: Network input pixels (sides rounded to multiples of `edge`).
        edge: Patch size.
        ar_min: Lower clamp of the network input's aspect ratio (H / W).
        ar_max: Upper clamp.
        gn_iters: Gauss-Newton steps.
        cuda_graphs: Option: kernels replays the network from a CUDA graph.
    """

    resolution: int
    edge: int
    ar_min: float
    ar_max: float
    gn_iters: int
    cuda_graphs: bool


def target_size(h: int, w: int, p: Params) -> tuple[int, int]:
    """Upstream `compute_target_size(RESOLUTION, clamp(h / w, *AR_RANGE))`: (height, width)."""
    ar = max(p.ar_min, min(h / w, p.ar_max))
    tw = sqrt(p.resolution / ar)
    th = ar * tw
    return round(th / p.edge) * p.edge, round(tw / p.edge) * p.edge


def preprocess(image: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream `AnyCalib.set_im_size` on `[B, 3, H, W]`: `(network input, scale_xy [2], shift_xy [2])`."""
    dev = image.device
    if tuple(image.shape[-2:]) == size:
        return image, torch.ones(2, device=dev), torch.zeros(2, device=dev)
    h, w = image.shape[-2:]
    ht, wt = size
    x = image
    s1 = (1.0, 1.0)
    if h < ht or w < wt:  # upsample (aspect ratio kept) until no side is shorter than the target
        x = F.interpolate(x, scale_factor=max(ht / h, wt / w), mode="bicubic", align_corners=False).clamp(0, 1)
        s1 = (x.shape[-1] / w, x.shape[-2] / h)
        h, w = x.shape[-2:]
    shift = [0.0, 0.0]
    ar = wt / ht
    if w / h > ar:  # crop the width
        c = round(w - h * ar)
        x = x[..., c // 2 : w - c + c // 2]
        shift[0] = -(c // 2)
    else:  # crop the height
        c = round(h - w / ar)
        x = x[..., c // 2 : h - c + c // 2, :]
        shift[1] = -(c // 2)
    h, w = x.shape[-2:]
    x = F.interpolate(x, size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    s2 = (wt / w, ht / h)
    scale = torch.tensor([s1[0] * s2[0], s1[1] * s2[1]], dtype=torch.float32, device=dev)
    return x, scale, torch.tensor([shift[0] * s2[0], shift[1] * s2[1]], dtype=torch.float32, device=dev)


class _Net(nn.Module):
    """Upstream `AnyCalib` without the calibrator: `vit` = `backbone.model`, the normalisation of `backbone`."""

    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        with mock.patch.object(DinoVisionTransformer, "init_weights", return_value=None):  # every tensor is assigned
            self.vit = vit_large(
                img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
                interpolate_antialias=False, interpolate_offset=0.1,
            )  # fmt: skip
        self.decoder = LightDPTDecoder(embed_dim=1024)
        self.head = ConvexTangentDecoder(in_channels=self.decoder.out_channels)
        self.register_buffer("image_mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(_STD).view(1, 3, 1, 1))


def build(ctx: Context) -> _Net:
    """The network on the CPU in float32 with the variant's weights."""
    net = _Net()
    state = {
        ("vit." + k.removeprefix("backbone.model."))
        if k.startswith("backbone.model.")
        else k.removeprefix("backbone."): v
        for k, v in ctx.state_dict().items()
    }
    load_strict(net, state, f"{ctx.variant.ref} checkpoint", not_in_checkpoint=("mask_token",), assign=True)
    return net.eval().requires_grad_(requires_grad=False)


def _maps(net: _Net, tokens: list[torch.Tensor], h: int, w: int) -> torch.Tensor:
    """Decoder + head on the block outputs (with cls) -> float32 rays `[B, 3, h, w]` of an `h x w` input."""
    p = net.vit.patch_size
    b, c = tokens[0].shape[0], tokens[0].shape[-1]
    feats = [t[:, 1:].reshape(b, h // p, w // p, c).permute(0, 3, 1, 2).contiguous() for t in tokens]
    x = net.decoder({"outputs": feats})
    head = net.head
    tangent = head.tangent_head(x).float()
    up: Any = head.upsampling_weights_head
    weights = up[3](up[2](up[1](up[0](x)))).float().softmax(2)  # conv, ReLU, conv, Unflatten; softmax in float32
    tangent = cvx_upsample(tangent, weights, head.up_factor)
    return Unit3.expmap_at_z1(tangent.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


_GN = GaussNewtonCalib({"max_iters": 0})  # its static methods; the loop is `fit`'s


def fit(rays: torch.Tensor, h: int, w: int, iters: int) -> torch.Tensor:
    """Pinhole `(fx, fy, cx, cy)` `[B, 4]` of rays `[B, h * w, 3]` (float32): linear fit, Gauss-Newton refinement.

    As upstream `Calibrator` with the `AnyCalib` defaults (`res_tangent="fitted"`, `solver="normal"`, all pixels);
    a failed linear fit gives NaN (upstream: RANSAC).
    """
    cam = Pinhole()
    x = torch.arange(w, device=rays.device, dtype=rays.dtype)
    y = torch.arange(h, device=rays.device, dtype=rays.dtype)
    coords = (torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1) + 0.5).view(h * w, 2)  # pixel centres
    init, info = cam.fit(coords, rays)
    ok = (info == 0) & init.isfinite().all(-1)
    init = torch.where(ok[:, None], init, torch.nan)
    params = init
    res, jac, valid = _GN.res_and_jac_in_fitted_tangent(cam, rays, params, coords)
    cost0 = _GN.compute_cost(res, None, valid)
    for _ in range(iters):
        params = params + _GN.solve_normal_eqs(jac, -res, None, valid)
        res, jac, valid = _GN.res_and_jac_in_fitted_tangent(cam, rays, params, coords)
    cost = _GN.compute_cost(res, None, valid)
    return torch.where((cost > cost0)[:, None], init, params)  # upstream keeps the linear fit if the cost grew


class _AnyCalib(Runner):
    """Pre-processing, network, camera fit, camera of the input image."""

    Params = Params
    batch = None
    net: _Net

    def _rays(self, x: torch.Tensor) -> torch.Tensor:
        """float32 rays `[B, 3, h, w]` of the network input `x` (float32 in [0, 1])."""
        raise NotImplementedError

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Pinhole intrinsics and vertical field of view of `[B, 3, H, W]` RGB in `[0, 1]`."""
        p = self.ctx.params
        h, w = image.shape[-2:]
        size = target_size(h, w, p)
        x, scale, shift = preprocess(image, size)
        rays = self._rays(x)
        b = rays.shape[0]
        with torch.autocast(image.device.type, enabled=False):
            params = fit(rays.permute(0, 2, 3, 1).reshape(b, -1, 3), *size, p.gn_iters)
            params = torch.cat([params[:, :2] / scale, (params[:, 2:] - shift) / scale], -1)
            vfov, _ = Pinhole().get_vfov(params, h)
        k = params.new_zeros(b, 3, 3)
        k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = *params.unbind(-1), 1.0
        return {"intrinsics": k, "vfov": vfov}

    def module(self) -> torch.nn.Module:
        """The network with its pre-processing and fit: `[B, 3, H, W]` in `[0, 1]` -> intrinsics `[B, 3, 3]`."""
        return Forward(self.net, lambda _net, image: self(image)["intrinsics"])


class Portable(_AnyCalib):
    """Plain PyTorch: the network in float32 or float16 (SDPA attention), head upsampling and fit in float32."""

    dtypes = ("float32", "float16")
    compiled_methods = ("_rays",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = build(ctx)
        sdpa_attention(net.vit)
        pos = net.vit.pos_embed.detach().clone()
        mean, std = net.image_mean, net.image_std
        net.to(self.dtype)
        net.vit.pos_embed = nn.Parameter(pos, requires_grad=False)  # fp32 position table, as upstream interpolates it
        net.image_mean, net.image_std = mean, std  # the normalisation in float32
        self.net = net.to(ctx.device)

    def _rays(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        t = net.vit.prepare_tokens_with_masks(((x - net.image_mean) / net.image_std).to(self.dtype))
        outs = []
        for i, blk in enumerate(net.vit.blocks):
            t = blk(t)
            if i in _OUT_INDEX:
                outs.append(t)
        return _maps(net, outs, *x.shape[-2:])


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(_AnyCalib):
    """`fastbb` ViT-L (flash attention, fused block GEMMs), `cudnn_hacc` decoder, a CUDA graph per input shape."""

    module = Runner.module  # fp16-accumulate GEMMs and convolutions are cuBLAS / cuDNN modes, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it to fp16, move it to the device."""
        super().__init__(ctx)
        net = build(ctx)
        mean, std = net.image_mean, net.image_std
        self.fast = FastViT(net.vit, "hh", None, fused=dpt_fused.load(ctx.device))  # converts the ViT to fp16
        net.decoder.half()
        net.head.half()
        self.net = net.to(ctx.device)
        self.fast.vit, self.fast.pos_fp32 = net.vit, self.fast.pos_fp32.to(ctx.device)
        net.image_mean, net.image_std = mean.to(ctx.device), std.to(ctx.device)
        cudnn_hacc.load(ctx.device)
        for m in (*net.decoder.modules(), *net.head.modules()):
            if is_base(m, torch.nn.Conv2d) and m.groups == 1 and m.padding_mode == "zeros" and m.padding != "same":
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        net, fast = self.net, self.fast
        with torch.autocast("cuda", enabled=False), fp16_accumulate():
            t = fast.tokens((x - net.image_mean) / net.image_std)
            outs = []
            for i, blk in enumerate(fast.blocks):
                t = fast.block(blk, t)
                if i in _OUT_INDEX:
                    outs.append(t)
            return _maps(net, outs, *x.shape[-2:])

    def _rays(self, x: torch.Tensor) -> torch.Tensor:
        out = None if self.graphs is None else self.graphs.run((x.device, tuple(x.shape)), x)
        return self._eager(x) if out is None else out

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
