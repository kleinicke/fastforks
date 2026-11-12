"""RoMa v2 runners: weights -> network and the portable forward pass.

Upstream `RoMaV2.match` without the settings object: both images are resized (bicubic, antialiased) to a
`coarse_size` square for the DINOv3 descriptor, the matcher and the first refinement, then to a `fine_size` square for
the second refinement (`fine_size = 0`: none). The warp of the first image into the second and its overlap logit are
then resized (bilinear) to the first image's input size; the warp goes from upstream's normalised coordinates to pixels
of the second image (pixel centres at integers), the overlap to its probability (sigmoid, as upstream `overlap_AB`
without the evaluation threshold). Upstream's precision matrices are not returned.

On CUDA the validated dtype is bfloat16 as upstream (`enable_amp=True`): the DINOv3 descriptor in bf16 weights under
bf16 autocast, and the multi-view ViT, the DPT head, the VGG features and the refiner blocks under bf16 autocast;
matching, the local correlation and the warps stay float32. float32 is upstream with `enable_amp=False` (and the DPT
head and VGG autocasts, which upstream cannot switch off, off). On CUDA the eval BatchNorms are folded
into their convs, the VGG and refiner activations are channels-last (the refiners' 1x1 convs as GEMMs) and the
linear / conv weights of the bf16 regions are stored in bf16 (`cast_autocast_weights`).
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.checkpoint import check_fits
from fastforks.matching._runner import Forward, Runner

from .net.refiner import Block, ConvRefiner
from .net.romav2 import RoMaV2, resize

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule
    from fastforks.matching._runner import Context

MULTIPLE = 16  # fixed: the descriptor's patch size (the coarse size) and the refiners' largest stride times 4
DESCRIPTOR_BLOCKS = 18  # fixed: upstream reads DINOv3 layers 11 and 17; the checkpoint's blocks 18-23 are unused


@dataclass(frozen=True)
class Params:
    """Load options."""

    coarse_size: int  # side of the square the descriptor, matcher and first refinement see (upstream H_lr = W_lr)
    fine_size: int  # side of the square of the second refinement (upstream H_hr = W_hr); 0 = no second refinement
    keep_descriptors: bool = False  # reuse the DINOv3 descriptors of the last two images (sequences, one-to-many)


class Descriptors:
    """`net.describe` that keeps the descriptors of the last two coarse images (`keep_descriptors`).

    An image that returns in the next pair (a sequence A-B, B-C; one image against many) skips the DINOv3 pass. The
    key is the resized image itself (compared on the GPU), so files, arrays and tensors all match.
    """

    def __init__(self, net: RoMaV2) -> None:
        """Nothing kept yet."""
        self.net = net
        self.kept: list[tuple[torch.Tensor, list[torch.Tensor]]] = []

    def __call__(self, img: torch.Tensor) -> list[torch.Tensor]:
        """`net.describe(img)`, from the kept ones if `img` is one of the last two images."""
        for i, (key, f) in enumerate(self.kept):
            if key.shape == img.shape and torch.equal(key, img):
                self.kept.append(self.kept.pop(i))
                return list(f)  # a copy: the matcher replaces the last entry of its list
        f = self.net.describe(img)
        self.kept = [*self.kept[-1:], (img, f)]
        return list(f)


def forward(
    net: RoMaV2, image1: torch.Tensor, image2: torch.Tensor, params: Params, describe: Descriptors | None = None
) -> dict[str, torch.Tensor]:
    """`{"warp": [B, 2, H1, W1], "certainty": [B, H1, W1]}` of `[0, 1]` RGB `[B, 3, H1, W1]`, `[B, 3, H2, W2]`."""
    lr, hr = (params.coarse_size, params.coarse_size), (params.fine_size, params.fine_size)
    fine = params.fine_size > 0
    lr1, lr2 = resize(image1, lr), resize(image2, lr)
    kept = {"f_A": describe(lr1), "f_B": describe(lr2)} if describe else {}
    out = net(lr1, lr2, resize(image1, hr) if fine else None, resize(image2, hr) if fine else None, **kept)
    warp = out["warp_AB"].float().permute(0, 3, 1, 2)  # [B, 2, h, w], normalised (-1 and 1: the image borders)
    overlap = out["confidence_AB"][..., 0].float().sigmoid()  # [B, h, w]
    (h1, w1), (h2, w2) = image1.shape[-2:], image2.shape[-2:]
    if warp.shape[-2:] != (h1, w1):
        warp = F.interpolate(warp, (h1, w1), mode="bilinear", align_corners=False)
        overlap = F.interpolate(overlap[:, None], (h1, w1), mode="bilinear", align_corners=False)[:, 0]
    size = torch.tensor((w2, h2), dtype=warp.dtype, device=warp.device).view(1, 2, 1, 1)
    return {"warp": (warp + 1) * (size / 2) - 0.5, "certainty": overlap}


def fold_batch_norms(net: RoMaV2) -> None:
    """Eval BatchNorms into the preceding conv: the VGG layers and the refiner blocks' depthwise convs (-8 % time)."""
    layers = net.refiner_features.layers
    for i, (conv, bn) in enumerate(itertools.pairwise(layers)):
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            layers[i], layers[i + 1] = fuse_conv_bn_eval(conv, bn), nn.Identity()
    for m in net.modules():
        if isinstance(m, Block) and isinstance(m.norm, nn.BatchNorm2d):
            m.conv_depthwise, m.norm = fuse_conv_bn_eval(m.conv_depthwise, m.norm), nn.Identity()


def cast_autocast_weights(net: RoMaV2, dtype: torch.dtype) -> None:
    """Linear / conv weights inside the autocast regions in `dtype`; norms stay float32.

    The multi-view ViT, the DPT head up to its float32 output conv, the VGG and the refiner blocks: autocast casts
    these weights to `dtype` on every call anyway, so the outputs are the same, without the float32 copies (-220 MB of
    weights, no cast per call).
    """
    regions: list[nn.Module] = [net.matcher.mv_vit, net.matcher.head, net.refiner_features]
    regions += [m for m in net.modules() if isinstance(m, Block)]
    out_conv = cast("nn.Module", net.matcher.head.scratch.output_conv2)
    skip = set(out_conv.modules())  # after upstream's `out.float()`
    for region in regions:
        for m in region.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and m not in skip:
                m.to(dtype)


class Portable(Runner):
    """Plain PyTorch; see the module docstring for the bf16 regions on CUDA."""

    dtypes = ("float32", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        if p.coarse_size % MULTIPLE or p.fine_size % MULTIPLE:
            sizes = f"{p.coarse_size}, {p.fine_size}"
            msg = f"{ctx.variant.ref}: coarse_size and fine_size must be multiples of {MULTIPLE}, not {sizes}"
            raise ValueError(msg)
        net = RoMaV2()
        sd = {
            k: v
            for k, v in ctx.state_dict().items()
            if not (k.startswith("f.blocks.") and int(k.split(".")[2]) >= DESCRIPTOR_BLOCKS)
        }
        check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        net.load_state_dict(sd)
        net.eval().requires_grad_(requires_grad=False).to(ctx.device)
        if ctx.dtype == "bfloat16":
            amp = torch.bfloat16
            net.f.to(amp)  # upstream wrap_model: the frozen descriptor in bf16
            net.f.amp = net.matcher.amp = net.matcher.head.amp = net.refiner_features.amp = amp
            for m in net.modules():  # the refiner blocks
                if isinstance(m, Block):
                    m.amp = amp
        if ctx.device.type == "cuda":  # channels-last refiner weights: cuDNN's depthwise 5x5, not ATen's (-17 % time)
            fold_batch_norms(net)  # not on CPU: the float32 reference parity (the refiners amplify rounding at borders)
            for m in net.modules():
                if isinstance(m, ConvRefiner):
                    m.to(memory_format=torch.channels_last)
            net.refiner_features.channels_last = True  # NHWC VGG: no layout conversions around cuDNN (-5 % time)
            net.refiner_features.to(memory_format=torch.channels_last)
            if ctx.dtype == "bfloat16":
                cast_autocast_weights(net, torch.bfloat16)
        self.net = net
        self.describe = Descriptors(net) if p.keep_descriptors else None

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Warp and certainty of a batch."""
        return forward(self.net, image1, image2, self.ctx.params, self.describe)

    def module(self) -> torch.nn.Module:
        """The network with the resizing; returns `{"warp", "certainty"}`."""
        params = self.ctx.params
        return Forward(self.net, lambda net, image1, image2: forward(net, image1, image2, params))


def _kernel_correlate(
    k: CubinModule, f_a: torch.Tensor, f_b: torch.Tensor, warp: torch.Tensor, radius: int
) -> torch.Tensor:
    """Upstream `local_correlation` by `romav2_local_corr_*`: `[B, K, H, W]` (a view of BHWC) of BHWC fp32 features.

    The same fp32 operands as the portable loop, without its `(2 * radius + 1) ** 2` grid_sample passes.
    """
    from fastforks.kernels import romav2 as lib  # noqa: PLC0415 - needs the kernel library

    b, h, w, c = f_a.shape
    f_a, f_b, warp = (x.float().contiguous() for x in (f_a, f_b, warp))
    out = torch.empty(b, h, w, (2 * radius + 1) ** 2, dtype=torch.float32, device=f_a.device)
    grid = (-(-b * h * w // lib.PIXELS_PER_BLOCK), 1, 1)
    k.launch(lib.LOCAL_CORR[c, radius], grid, (256, 1, 1), f_a, f_b, warp, out, b, h, w, 1 / c**0.5)
    return out.permute(0, 3, 1, 2)


class Kernels(Portable):
    """The portable pass with the refiners' local correlation as a kernel (`romav2`): no window tensor, one pass."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the correlation kernel to the refiners that correlate."""
        super().__init__(ctx)
        from fastforks.kernels import romav2  # noqa: PLC0415 - needs the kernel library

        k = romav2.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, ConvRefiner) and m.cfg.local_corr_radius is not None:  # (C, r): (192, 3), (48, 1)
                m.correlate = functools.partial(_kernel_correlate, k)

    def module(self) -> torch.nn.Module:
        """No exportable module: the correlation is a kernel launch."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
