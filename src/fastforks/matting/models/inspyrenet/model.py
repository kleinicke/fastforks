"""InSPyReNet runners: weights -> network and the portable and kernels forward passes.

Inference as transparent-background's `Remover.process` (what ComfyUI's InSPyReNet nodes call), static resize: the
image resized to `size x size` (PIL bilinear, antialiased; here `F.interpolate` with `antialias=True` on floats),
ImageNet-normalised, the network's finest saliency map `d0` (`threshold=None`: one pyramid, no LR/HR blending),
sigmoid, min-max normalised, resized back to the image (bilinear, `align_corners=True`). Upstream normalises min-max
over the whole batch (it runs one image); here per image.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16. In fp16 the attention modules (`SICA`) compute their
context vectors (a sum over all positions of a stage: up to 156 049 on the golden image) and the key / value convs
on them in float32; those convs are never cast to float16 (their batch-norm running stats exceed its range).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate
from fastforks._core.mlx_cast import to_float32
from fastforks._core.patch import override
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.kernels import image_normalize
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap, swin_v1
from fastforks.shared.cnn_kernels import fold_bn
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import InSPyReNet_SwinB
from .net.modules.attention_module import SICA
from .net.modules.layers import Conv2d

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        size: Network input side (square; transparent-background's `base_size`).
    """

    size: int


def _sica(
    m: SICA, x: torch.Tensor, smap: torch.Tensor, lmap: torch.Tensor | None = None, *, nhwc: bool = False
) -> tuple[torch.Tensor, ...]:
    """`SICA.forward` with the context vector and the key / value convs in float32.

    `nhwc` (kernels): the query and output convs run channels-last (their weights moved by `Kernels`). cuDNN runs a
    fp16 conv in NHWC, so in NCHW each of the six converted its input and output; channels-last, the query tokens and
    the refined context are views of the conv / GEMM outputs, and only `x` and the outputs change layout.
    """
    b, _, h, w = x.shape
    p = torch.sigmoid(F.interpolate(smap, size=x.shape[-2:], mode="bilinear", align_corners=False)) - m.threshold
    prob = [torch.clip(p, 0, 1), torch.clip(-p, 0, 1), m.threshold - torch.abs(p)]
    if lmap is not None:  # SICA asserts that `lmap_in` modules get one
        lp = torch.sigmoid(F.interpolate(lmap, size=x.shape[-2:], mode="bilinear", align_corners=False)) - m.lthreshold
        prob += [torch.clip(lp, 0, 1), torch.clip(-lp, 0, 1)]
    shape = m.stage_size if m.stage_size is not None else (h, w)
    n = shape[0] * shape[1]
    at_size = shape == (h, w)
    f = x if at_size and x.is_contiguous() else F.interpolate(x, size=shape, mode="bilinear", align_corners=False)
    pr = torch.cat(prob, dim=1)
    if not at_size:
        pr = F.interpolate(pr, size=shape, mode="bilinear", align_corners=False)
    # Keep the upstream channel layout for the context projection.
    context = torch.bmm(to_float32(pr.reshape(b, m.ctx, n)), to_float32(f.view(b, n, -1))).permute(0, 2, 1).unsqueeze(3)
    if CTX_SCALE_ATTR in m.__dict__:  # a float16 MLX replay: its key / value convs read float16 (see `_scale_context`)
        context = context * (1 / m.__dict__[CTX_SCALE_ATTR])
    del f, pr, prob
    if nhwc:
        x = x.contiguous(memory_format=torch.channels_last)
        query = m.conv_query(x).permute(0, 2, 3, 1).reshape(b, h * w, m.depth)  # a view of the NHWC output
    else:
        query = m.conv_query(x).view(b, m.depth, -1).permute(0, 2, 1)
    key = m.conv_key(context).to(x.dtype).view(b, m.depth, -1)
    value = m.conv_value(context).to(x.dtype).view(b, m.depth, -1).permute(0, 2, 1)
    sim = F.softmax((m.depth**-0.5) * torch.bmm(query, key), dim=-1)
    del query, key
    if nhwc:
        context = torch.bmm(sim, value).view(b, h, w, -1).permute(0, 3, 1, 2)  # channels-last [B, C, H, W]
    else:
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
    del sim, value
    x = torch.cat([x, m.conv_out1(context)], dim=1)
    del context
    x = m.conv_out3(m.conv_out2(x))
    out = m.conv_out4(x)
    return (x.contiguous(), out.contiguous()) if nhwc else (x, out)


CTX_SCALE_ATTR = "_ff_ctx_scale"
CTX_SCALE = 4096.0  # fixed: float16 range; base checkpoint context sums reach 1.6e5, key / value BN variances 5.4e10


@torch.no_grad()
def _scale_context(m: SICA) -> None:
    """Divide `m`'s context vector by `CTX_SCALE` and its first key / value batch-norm statistics to match.

    A float16 MLX replay runs the key / value convs in float16: the context vector and those running statistics
    overflow it. Bias-free 1x1 convs are linear, so the batch norm output is unchanged (up to its eps).
    """
    m.__dict__[CTX_SCALE_ATTR] = CTX_SCALE
    for seq in (m.conv_key, m.conv_value):
        bn = seq[0].bn
        bn.running_mean.div_(CTX_SCALE)
        bn.running_var.div_(CTX_SCALE**2)


def _sica_nhwc(
    m: SICA, x: torch.Tensor, smap: torch.Tensor, lmap: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    """`_sica` with its query and output convs channels-last (kernels)."""
    return _sica(m, x, smap, lmap, nhwc=True)


def _fold_batchnorms(net: torch.nn.Module) -> int:
    """Fold each decoder `Conv2d` block's eval BatchNorm into its conv (float32, before any fp16 cast); how many."""
    n = 0
    for m in net.modules():
        if isinstance(m, Conv2d) and m.bn is not None:
            fold_bn(m.conv, m.bn)
            m.bn = None
            n += 1
    return n


def _net(ctx: Context, dtype: torch.dtype, *, fold: bool = False) -> torch.nn.Module:
    s = ctx.params.size
    net = InSPyReNet_SwinB(depth=64, pretrained=False, base_size=[s, s], threshold=None)
    net.load_state_dict(ctx.state_dict())
    torch.nn.Module.to(net, ctx.device)  # upstream's `to(device)` takes no dtype
    if fold and not _fold_batchnorms(net):
        msg = "InSPyReNet kernels: no Conv2d BatchNorm to fold (the module tree changed)"
        raise RuntimeError(msg)
    half = dtype == torch.float16 or mlx_wrap.twin_dtype() == "float16"  # also the float32 twin of a float16 replay
    sica = [m for m in net.modules() if isinstance(m, SICA)] if half else []
    keep = [(m, m.conv_key, m.conv_value) for m in sica]
    for m in sica:  # their batch-norm running stats overflow float16 (|mean| > 65504): converted, they turn NaN
        del m.conv_key, m.conv_value
    torch.nn.Module.to(net, dtype)
    for m, key, value in keep:
        m.conv_key, m.conv_value = key, value
        override(m, _sica)
        if mlx_wrap.twin_dtype() == "float16":
            _scale_context(m)
    net.image_pyramid.kernel = net.image_pyramid.kernel.to(ctx.device, dtype)  # a plain tensor, not a buffer
    return net.eval()


class Portable(Runner):
    """Upstream's modules in plain PyTorch, float32 or float16."""

    Params = Params
    sdpa_auto = "efficient,cudnn,math"  # Swin window attention: cuDNN rebuilds per shape (`fastforks.shared.sdpa`)
    dtypes = ("float32", "float16")
    batch = None
    fold = False  # kernels: the decoder's BatchNorms folded into their convs
    # network pixels (images x size^2) per call: the eager call is launch bound at 384 (RTX 5060 Ti, Windows, fp16,
    # 8 images: inspyrenet-fast 57.4 -> 13.9 ms per image), not at 1024 (115 vs 107 ms, 7.5 GB): one image there
    net_pixels = 1_300_000

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _net(ctx, self.dtype, fold=self.fold)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return scale(x, self.ctx.io)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_inspyre(x)["saliency"][3]  # ty: ignore[call-non-callable] - the upstream class

    def _nets(self, x: torch.Tensor) -> torch.Tensor:
        step = max(1, self.net_pixels // (x.shape[-2] * x.shape[-1]))
        return torch.cat([self._net(x[i : i + step]) for i in range(0, len(x), step)])

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False, antialias=True)
        y = torch.sigmoid(self._nets(self._normalize(x).to(self.dtype)).float())
        lo, hi = y.amin((1, 2, 3), keepdim=True), y.amax((1, 2, 3), keepdim=True)
        y = (y - lo) / (hi - lo + 1e-8)
        return {"alpha": F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=True)[:, 0]}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image, None)["alpha"])


class Kernels(Portable):
    """Fuse input normalization, gather-based Swin blocks, fp16-accumulate GEMMs, capture the fixed-size network."""

    module = Runner.module
    # the eval BatchNorms after the decoder convs (SICA, the context branches) were a pass each: 5 ms of 86 in the SICA
    # modules alone (1024 x 1024, RTX 5060 Ti)
    fold = True

    def __init__(self, ctx: Context) -> None:
        """Prepare the accelerated inference path on the model device."""
        super().__init__(ctx)
        swin_v1.fast(self.net.backbone)
        if self.dtype == torch.float16:
            for m in self.net.modules():
                if isinstance(m, SICA):
                    for conv in (m.conv_query, m.conv_out1, m.conv_out2, m.conv_out3, m.conv_out4):
                        conv.to(memory_format=torch.channels_last)
                    override(m, _sica_nhwc)
        self.mean = torch.tensor(IMAGENET_MEAN, device=ctx.device)
        self.std = torch.tensor(IMAGENET_STD, device=ctx.device)
        self.graphs = SmallShapeGraphs(super()._net, pixels=2**21)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return image_normalize.normalize(x, self.mean, self.std)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():  # also while a graph captures: its GEMMs keep the fp16-accumulate kernels
            return self.graphs(x)

    def release_graphs(self) -> None:
        """Release captured activation buffers."""
        self.graphs.release()


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OneImage, attrs=(), methods=("_net",)),
}
