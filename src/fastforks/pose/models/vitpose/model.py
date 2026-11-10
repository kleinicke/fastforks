"""ViTPose runners: person crops -> heatmaps -> DARK-refined keypoints in the image (HF `VitPoseImageProcessor`).

Per person box (`x0, y0, x1, y1`; the reference takes COCO `x, y, w, h`):

- crop: the box centre and size, the shorter side widened to the network's 3 : 4 aspect, x 1.25; the crop maps output
  pixel `(u, v)` to `centre - size / 2 + (u, v) * size / (out - 1)` (the unbiased warp of Huang et al.), bilinear,
  0 for points outside `[0, n - 1]` of the image (scipy `affine_transform`, order 1, constant mode: no blending at
  the border); then ImageNet normalisation
- decode: argmax per heatmap (score = its maximum), then DARK: the heatmap blurred with a Gaussian (sigma 0.8, radius
  5, scipy's half-sample `reflect` border), clipped to [0.001, 50], log, edge padded by 1; one Newton step from the
  argmax with central differences; mapped back with `centre - size / 2 + p * size / (heatmap - 1)`
- `scores`: the mean keypoint score (the reference gives none)

The reference marks keypoints whose heatmap maximum is <= 0 with -1 before the DARK step (and then reads neighbours of
another keypoint's heatmap); here they keep their argmax: the score says they were not found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.half import cublas_gemms, fp16_accumulate
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.pose._runner import BUCKETS, Forward, Runner, in_buckets
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import EPS, PATCH, PATCH_PAD, Arch, Block, ViTPose, convert

if TYPE_CHECKING:
    from fastforks.pose._runner import Context

PADDING = 1.25  # fixed: reference crop padding (box_to_center_and_scale)
SIGMA = 0.8  # fixed: reference DARK blur sigma
UPSAMPLE_OUT = 4  # fixed: architecture (heatmaps at 1/4 of the network input)
RADIUS = 5  # fixed: reference DARK blur radius (kernel 11)
CLIP = (0.001, 50.0)  # fixed: reference DARK clip before the log
GELU = 5  # fixed: fused-GEMM epilogue code of the exact (erf) GELU (`fastforks._optimize.transformer`)
# kernels: people per call padded to the next of these, so a CUDA graph per bucket serves every count up to the last
# (the calls are launch bound there: 192 tokens per person); more people run eagerly in one batch


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    fp32_accumulate: bool  # kernels: every GEMM through cuBLAS with fp32 accumulation (option)


def crop_boxes(boxes: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(centres `[P, 2]`, crop sizes `[P, 2]` w, h in pixels) of `x0, y0, x1, y1` boxes for a `height x width` net."""
    wh = boxes[:, 2:] - boxes[:, :2]
    centre = boxes[:, :2] + wh * 0.5
    aspect = width / height
    w, h = wh.unbind(1)
    w, h = torch.where(w > aspect * h, w, h * aspect), torch.where(w > aspect * h, w / aspect, h)
    return centre, torch.stack([w, h], 1) * PADDING


def crop(image: torch.Tensor, centre: torch.Tensor, size: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crops `[P, 3, height, width]` of `image` `[1, 3, H, W]` (see the module docstring)."""
    ih, iw = image.shape[-2:]
    dev, dt = image.device, torch.float32
    u = torch.arange(width, device=dev, dtype=dt) / (width - 1)
    v = torch.arange(height, device=dev, dtype=dt) / (height - 1)
    x0 = centre - size * 0.5
    x = x0[:, 0, None] + u[None] * size[:, 0, None]  # [P, w] source pixel columns
    y = x0[:, 1, None] + v[None] * size[:, 1, None]  # [P, h]
    gx = (x * (2 / (iw - 1)) - 1)[:, None, :].expand(-1, height, -1)
    gy = (y * (2 / (ih - 1)) - 1)[:, :, None].expand(-1, -1, width)
    p = centre.shape[0]
    out = F.grid_sample(
        image.float().expand(p, -1, -1, -1), torch.stack([gx, gy], -1), align_corners=True, padding_mode="zeros"
    )
    inside = ((x >= 0) & (x <= iw - 1))[:, None, :] & ((y >= 0) & (y <= ih - 1))[:, :, None]
    return out * inside[:, None].to(out.dtype)


def _blur(hm: torch.Tensor) -> torch.Tensor:
    """Scipy `gaussian_filter(sigma=SIGMA, radius=RADIUS)` over the last two axes, `reflect` (half-sample) border."""
    t = torch.arange(-RADIUS, RADIUS + 1, device=hm.device, dtype=hm.dtype)  # on the device: no copy in a graph
    k = torch.exp(-0.5 * (t / SIGMA) ** 2)
    k = k / k.sum()
    for axis in (-1, -2):
        n = hm.shape[axis]
        i = torch.arange(-RADIUS, n + RADIUS, device=hm.device)
        i = torch.where(i < 0, -i - 1, torch.where(i >= n, 2 * n - 1 - i, i))
        x = hm.index_select(axis, i).unfold(axis, 2 * RADIUS + 1, 1)  # [..., n, 11] windows along `axis`
        hm = (x * k).sum(-1)
    return hm


def decode(hm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(heatmap coordinates `[P, K, 2]` x, y after DARK, scores `[P, K]`) of float32 heatmaps `[P, K, h, w]`."""
    w = hm.shape[-1]
    score, idx = hm.flatten(2).max(-1)
    xi, yi = idx % w, idx // w
    lg = F.pad(_blur(hm).clamp(*CLIP).log(), (1, 1, 1, 1), mode="replicate").flatten(2)  # [P, K, (h + 2)(w + 2)]
    wp = w + 2
    c = (xi + 1) + (yi + 1) * wp

    def at(dx: int, dy: int) -> torch.Tensor:
        return lg.gather(2, (c + dx + dy * wp)[..., None])[..., 0]

    i0 = at(0, 0)
    dx = 0.5 * (at(1, 0) - at(-1, 0))
    dy = 0.5 * (at(0, 1) - at(0, -1))
    dxx = at(1, 0) - 2 * i0 + at(-1, 0)
    dyy = at(0, 1) - 2 * i0 + at(0, -1)
    dxy = 0.5 * (at(1, 1) - at(1, 0) - at(0, 1) + 2 * i0 - at(-1, 0) - at(0, -1) + at(-1, -1))
    eps = torch.finfo(torch.float32).eps
    a, d = dxx + eps, dyy + eps
    det = a * d - dxy * dxy
    ox = (d * dx - dxy * dy) / det
    oy = (a * dy - dxy * dx) / det
    coords = torch.stack([xi.float() - ox, yi.float() - oy], -1)
    return coords, score


class Portable(Runner):
    """Plain PyTorch: crops on the device, float16 network on CUDA, float32 decode."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (HF keys converted, BatchNorm folded) on the device."""
        super().__init__(ctx)
        a: Params = ctx.params
        net = ViTPose(a)
        net.load_state_dict(convert(ctx.state_dict(strip=""), a))
        self.net = net.eval().to(ctx.device, ctx.torch_dtype)
        self.mean = torch.tensor(IMAGENET_MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=ctx.device).view(1, 3, 1, 1)

    def heatmaps(self, crops: torch.Tensor) -> torch.Tensor:
        """float32 heatmaps `[P, K, h / 4, w / 4]` of crops `[P, 3, h, w]` in [0, 1]."""
        x = ((crops - self.mean) / self.std).to(self.ctx.torch_dtype)
        return self.net(x).float()

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Keypoints of the people in `boxes`."""
        assert boxes is not None  # noqa: S101 - the task API checks top-down calls
        a: Params = self.ctx.params
        centre, size = crop_boxes(boxes.float(), a.height, a.width)
        hm = self.heatmaps(crop(image, centre, size, a.height, a.width))
        coords, score = decode(hm)
        h, w = hm.shape[-2:]
        scale = torch.stack([size[:, 0] / (w - 1), size[:, 1] / (h - 1)], 1)  # no host copy
        kp = coords * scale[:, None] + (centre - size * 0.5)[:, None]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}

    def module(self) -> torch.nn.Module:
        """The network: crops `[B, 3, h, w]` RGB in [0, 1] at the network size -> heatmaps `[B, K, h / 4, w / 4]`."""

        def fn(_net: Any, crops: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - the wrapped network
            return self.heatmaps(crops)

        return Forward(self.net, fn)


def _w4(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight `[out, in]` as the fused GEMM's fp16 `[out, in, 1, 1]` channels-last filter."""
    return w.detach().half().reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


def _block(b: Block, w: tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
    """`Block.forward` with the token GEMMs fused with their bias / GELU / residual epilogues (fvm_conv).

    `w`: the qkv, proj, fc1, fc2 weights as `_w4` filters.
    """
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    bsz, n, c = x.shape
    h = b.heads
    y = F.layer_norm(x, (c,), b.norm1.weight, b.norm1.bias, EPS)
    w_qkv, w_proj, w_fc1, w_fc2 = w
    q, k, v = lin(y, w_qkv, b.qkv.bias).view(bsz, n, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, n, c)
    x = lin(o, w_proj, b.proj.bias, res=x)
    y = lin(F.layer_norm(x, (c,), b.norm2.weight, b.norm2.bias, EPS), w_fc1, b.fc1.bias, act=GELU)
    return lin(y, w_fc2, b.fc2.bias, res=x)


class Kernels(Portable):
    """Our kernels: fused fp16 GEMMs, flash attention, a CUDA graph per person bucket.

    fvm_conv token GEMMs with bias / GELU / residual epilogues (fp16 accumulation; option `fp32_accumulate`: cuBLAS
    with fp32), flash attention, the cuDNN fp16 channels-last decoder; network + heatmap decoding replay from a CUDA
    graph per person bucket (`BUCKETS`). Cropping and the mapping to the image run eagerly (one launch each).
    """

    def __init__(self, ctx: Context) -> None:
        """The portable network in fp16 with the fused-GEMM filters attached; the graph cache."""
        super().__init__(ctx)
        self.blocks = [b for b in self.net.blocks if isinstance(b, Block)]  # all of them (built above)
        self.w4 = [tuple(_w4(getattr(b, n).weight) for n in ("qkv", "proj", "fc1", "fc2")) for b in self.blocks]
        self.net.head.to(memory_format=torch.channels_last)
        self.gemm = cublas_gemms if ctx.params.fp32_accumulate else fp16_accumulate
        self.graphs = SmallShapeGraphs(self._decoded, pixels=BUCKETS[-1] * ctx.params.height * ctx.params.width)

    def _decoded(self, crops: torch.Tensor) -> torch.Tensor:
        """`[P, K, 3]` float32 (x, y in heatmap pixels after DARK, score) of crops in [0, 1] at the network size."""
        with self.gemm():
            x = ((crops - self.mean) / self.std).half()
            t = self.net.tokens(x)
            for b, w in zip(self.blocks, self.w4, strict=True):
                t = _block(b, w, t)
            hm = self.net.head(self.net.feature_map(t).contiguous(memory_format=torch.channels_last)).float()
        coords, score = decode(hm)
        return torch.cat([coords, score[..., None]], -1)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Keypoints of the people in `boxes`."""
        assert boxes is not None  # noqa: S101 - the task API checks top-down calls
        a: Params = self.ctx.params
        centre, size = crop_boxes(boxes.float(), a.height, a.width)
        crops = crop(image, centre, size, a.height, a.width)
        out = in_buckets(self.graphs, crops)
        h, w = a.height // UPSAMPLE_OUT, a.width // UPSAMPLE_OUT
        scale = torch.stack([size[:, 0] / (w - 1), size[:, 1] / (h - 1)], 1)  # no host copy
        kp = out[..., :2] * scale[:, None] + (centre - size * 0.5)[:, None]
        score = out[..., 2]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}


class MetalKernels(Portable):
    """Apple MPS: the portable network with its memory-bound parts as PyTorch GEMMs + Metal kernels (`pose_metal`).

    - patch embedding: the 16 x 16 stride-16 convolution as a reshape + GEMM (MPS's strided convolution is ~4x slower);
    - classic decoder: each transposed convolution as one GEMM of the tokens (channels last) + a Metal overlap-add
      with bias and ReLU (`deconv4s2_relu`), the 1 x 1 convolution as a GEMM; the simple decoder runs as portable;
    - DARK decoding: one Metal kernel (argmax, the blur at the 3 x 3 neighbourhood only, the Newton step).

    Same math as portable (float32 decode; fp16 GEMMs accumulate in float32 on MPS).
    """

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """The portable network; the GEMM forms of its convolutions; the Metal library."""
        super().__init__(ctx)
        from fastforks.kernels import pose_metal  # noqa: PLC0415 - needs the kernel library

        self.lib = pose_metal.load(ctx.device)
        net = self.net
        self.patch_w = net.patch.weight.reshape(net.patch.weight.shape[0], -1)  # [C, 3 * 16 * 16]: (c, ky, kx)
        self.deconvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if ctx.params.decoder == "classic":
            layers = [m for m in net.head.modules() if isinstance(m, torch.nn.ConvTranspose2d | torch.nn.Conv2d)]
            w, b = [m.weight for m in layers], [cast("torch.Tensor", m.bias) for m in layers]  # all with biases
            for i in range(2):  # the transposed convolutions: [in, out, 4, 4] -> [in, (ky, kx, out)]
                self.deconvs.append((w[i].permute(0, 2, 3, 1).reshape(w[i].shape[0], -1).contiguous(), b[i]))
            self.final = w[2][:, :, 0, 0], b[2]  # the 1 x 1 convolution

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """`ViTPose.tokens` with the patch convolution as reshape + GEMM (its padding: the zero rows / columns)."""
        net, b = self.net, x.shape[0]
        gh, gw = net.gh, net.gw
        x = F.pad(x, (PATCH_PAD,) * 4)[..., : gh * PATCH, : gw * PATCH]
        p = x.reshape(b, 3, gh, PATCH, gw, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, -1)
        return F.linear(p, self.patch_w, net.patch.bias) + net.pos

    def _decoded(self, crops: torch.Tensor) -> torch.Tensor:
        """`[P, K, 3]` float32 (x, y in heatmap pixels after DARK, score) of crops in [0, 1] at the network size."""
        from fastforks.kernels import pose_metal  # noqa: PLC0415 - loaded in __init__

        net = self.net
        t = self._tokens(((crops - self.mean) / self.std).to(self.ctx.torch_dtype))
        for blk in net.blocks:
            t = blk(t)
        k = self.ctx.params.keypoints
        if not self.deconvs:  # the simple decoder: bilinear upsampling + 3 x 3 convolution, as portable
            hm = net.head(net.feature_map(t)).float()
            return pose_metal.dark_decode(self.lib, hm, k, channels_last=False, sigma=SIGMA, radius=RADIUS, clip=CLIP)
        t, h, w = net.norm(t), net.gh, net.gw
        for wt, bias in self.deconvs:
            t = pose_metal.deconv4s2_relu(self.lib, t @ wt, bias, h, w)  # [P, 2h * 2w, 256]
            h, w = 2 * h, 2 * w
        hm = F.linear(t, *self.final).float().view(-1, h, w, k)
        return pose_metal.dark_decode(self.lib, hm, k, channels_last=True, sigma=SIGMA, radius=RADIUS, clip=CLIP)

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Keypoints of the people in `boxes`."""
        assert boxes is not None  # noqa: S101 - the task API checks top-down calls
        a: Params = self.ctx.params
        centre, size = crop_boxes(boxes.float(), a.height, a.width)
        out = self._decoded(crop(image, centre, size, a.height, a.width))
        h, w = a.height // UPSAMPLE_OUT, a.width // UPSAMPLE_OUT
        scale = torch.stack([size[:, 0] / (w - 1), size[:, 1] / (h - 1)], 1)  # no host copy
        kp = out[..., :2] * scale[:, None] + (centre - size * 0.5)[:, None]
        score = out[..., 2]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
