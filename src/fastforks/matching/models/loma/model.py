"""Portable LoMa runners for the DaD detector, DeDoDe descriptors and LightGlue-style matcher.

As upstream's `LoMa.match` on image paths (its MegaDepth / ScanNet benchmarks): the DaD detector sees the image
resized to a long side of `long_side` (aspect kept, both sides floored to a multiple of 8; bicubic, antialiased,
clamped to [0, 1]: PIL's `resize` of upstream's `load_image` without its rounding to uint8), ImageNet-normalised; its
logits give a softmax over all pixels, non-maximum-suppressed in 3 x 3 windows, and the `max_keypoints` best pixels,
refined to sub-pixel by the softmax of their 3 x 3 logits, are the keypoints. The descriptor samples the square-
resized image at these keypoints. B, L, G and R use VGG19 and DINOv2 ViT-L/14. B128 uses VGG19 and B64 uses VGG11
with ImageNet normalisation. Nine LightGlue-style transformer layers (rotary self-attention, cross-attention)
refine the descriptions of both images, and the matches are the mutual maxima of the row-times-column softmax of
their projected similarity above `threshold`, scored by that product.
Keypoints are mapped back to input pixels (pixel centres at integers: upstream's `to_pixel_coords` minus 0.5).

On CUDA the validated dtype is bfloat16 as upstream on Ampere and newer (float16 where upstream falls back to it):
the VGG encoders and the refiners under autocast, the DINOv2 with weights and input in that dtype, the matcher under
autocast; logits, keypoint sampling, the description sampling and the match filter stay float32. On CUDA the eval
BatchNorms are folded into their convs and the convolutions run channels-last (cuDNN's depthwise 5 x 5); the DINOv2
runs fused `scaled_dot_product_attention` (as upstream) with its LayerScales folded into the linears before them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.checkpoint import load_strict
from fastforks._core.deferred_init import skipped_init
from fastforks._core.patch import override
from fastforks._core.tracing import in_onnx_export
from fastforks._core.upload import device_tensor
from fastforks.kernels import loma_descriptor
from fastforks.matching._dedode_kernels import prepare
from fastforks.matching._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.dedode import (
    DESCRIPTOR_B,
    DETECTOR_S,
    VGG,
    VGG11_BN,
    VGG19_BN,
    ConvRefiner,
    Decoder,
    Descriptor,
    Detector,
    VGGDINOv2,
    describe_keypoints,
)
from fastforks.shared.dedode import sparse as dedode_sparse
from fastforks.shared.depth_anything_v2 import DINOv2, fold_layer_scale, sdpa_attention

from .net.dad import sample_keypoints
from .net.loma import LoMaNet, filter_matches

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # fixed: DaD's `normalizer`
NMS_SIZE = 3  # fixed: upstream `DaD.Cfg.nms_size`
SUBPIXEL_TEMP = 0.5  # fixed: upstream `DaD.Cfg.subpixel_temp`
STRIDE = 8  # fixed: upstream `load_image` floors both sides to a multiple of 8
PREFIXES = {  # checkpoint prefix -> (submodule, the prefix it keeps)
    "_detector.": "detector",
    "_descriptor.encoder.layers.": "descriptor.encoder.layers",
    "_descriptor.encoder.vgg.": "descriptor.encoder.vgg",
    "_descriptor.encoder.frozen_dinov2.dinov2_vitl14.": "dinov2",
    "_descriptor.decoder.": "descriptor.decoder",
}


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    dim: int  # matcher embedding width
    input_dim: int  # descriptor output width
    descriptor: str  # dedode_g, dedode_b or dedode_b_vgg11
    normalize_descriptor: bool
    heads: int  # attention heads (loma-b: 4)
    layers: int  # transformer layers (loma-b: 9)
    threshold: float  # match score (product of the two softmaxes) a match needs (upstream 0.1)
    long_side: int  # the detector's image: long side (upstream 1024), aspect kept
    size: int  # the descriptor's image: side of the square (upstream 784)
    max_keypoints: int  # keypoints per image (upstream default 2048; its benchmarks 4096)


Described = tuple[torch.Tensor, torch.Tensor]  # normalised keypoints [N, 2], descriptions [N, dim]


def fold_batch_norms(net: nn.Module) -> None:
    """Eval BatchNorms into the conv before them (VGG blocks, refiner blocks)."""
    for seq in [m for m in net.modules() if isinstance(m, (nn.Sequential, nn.ModuleList))]:
        layers = list(seq)
        for i in range(len(layers) - 1):
            conv, bn = layers[i], layers[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                seq[i], seq[i + 1] = fuse_conv_bn_eval(conv, bn), nn.Identity()


class SmallDescriptor(Descriptor):
    """Upstream LoMa's DeDoDe-B with configurable output width and VGG encoder."""

    def __init__(self, params: Params) -> None:
        """Build the descriptor using the shared DeDoDe layers."""
        nn.Module.__init__(self)
        self.encoder = VGG({"dedode_b_vgg11": VGG11_BN, "dedode_b": VGG19_BN}[params.descriptor])
        self.decoder = Decoder(*DESCRIPTOR_B, num_prototypes=params.input_dim)


class Matcher(nn.Module):
    """LoMa on an RGB pair: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        self.detector = Detector(DETECTOR_S, VGG11_BN)
        self.dinov2 = DINOv2("vitl") if params.descriptor == "dedode_g" else None
        self.descriptor = Descriptor(self.dinov2) if self.dinov2 is not None else SmallDescriptor(params)
        self.matcher = LoMaNet(params.input_dim, params.dim, params.heads, params.layers)
        self.amp: torch.dtype | None = None
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def set_amp(self, amp: torch.dtype | None) -> None:
        """Upstream's autocast dtype for the encoders, refiners and the matcher; `None` = float32."""
        self.amp = amp
        for m in self.modules():
            if isinstance(m, (VGG, ConvRefiner)):
                m.amp = amp
        encoder = self.descriptor.encoder
        if isinstance(encoder, VGGDINOv2):
            encoder.dinov2[0].to(amp or torch.float32)
            encoder.dino_dtype = amp or torch.float32

    @staticmethod
    def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(image, size=size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)

    def describe(self, image: torch.Tensor) -> Described:
        """Keypoints (normalised coordinates) and their descriptions of one image."""
        p = self.params
        keypoints = self._detect(image)  # the detector's input and logits are freed before the descriptor runs
        resized = self._resize(image, (p.size, p.size))
        if p.normalize_descriptor:
            resized = (resized - self.mean) / self.std
        return keypoints, self._descriptions(resized, keypoints)

    def _descriptions(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        if (
            image.is_cuda
            and self.amp is not None
            and not (
                torch.compiler.is_compiling() or torch.jit.is_tracing() or in_onnx_export() or torch.is_grad_enabled()
            )
        ):
            return describe_keypoints(_deferred_descriptor(self.descriptor, image), keypoints)
        return describe_keypoints(self.descriptor(image), keypoints)

    def _detect(self, image: torch.Tensor) -> torch.Tensor:
        p = self.params
        h, w = image.shape[-2:]
        scale = p.long_side / max(h, w)
        det = self._resize(image, (int(scale * h) // STRIDE * STRIDE, int(scale * w) // STRIDE * STRIDE))
        logits = self.detector((det - self.mean) / self.std)
        return sample_keypoints(logits, p.max_keypoints, NMS_SIZE, SUBPIXEL_TEMP)

    @staticmethod
    def to_input(keypoints: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Normalised keypoints in input pixels (pixel centres at integers)."""
        h, w = image.shape[-2:]
        return (keypoints + 1) / 2 * device_tensor([w, h], device=keypoints.device) - 0.5

    def match(
        self, image1: torch.Tensor, image2: torch.Tensor, d1: Described, d2: Described
    ) -> dict[str, torch.Tensor]:
        """The pair's matches from both images' `describe` results."""
        device = image1.device.type
        with torch.autocast(device, enabled=self.amp is not None, dtype=self.amp or torch.float16):
            scores = self.matcher(d1[0][None], d2[0][None], d1[1][None], d2[1][None])
        m0, mscores0 = filter_matches(scores.float(), self.params.threshold)
        idx = torch.nonzero(m0 > -1)[:, 0]
        return {
            "keypoints1": self.to_input(d1[0], image1),
            "keypoints2": self.to_input(d2[0], image2),
            "pairs": torch.stack([idx, m0[idx]], dim=-1),
            "scores": mscores0[idx],
        }

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        return self.match(image1, image2, self.describe(image1), self.describe(image2))


def split_checkpoint(sd: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Upstream's single state dict by submodule (`PREFIXES` values and `"matcher"`) -> that module's state dict."""
    parts: dict[str, dict[str, torch.Tensor]] = {name: {} for name in [*PREFIXES.values(), "matcher"]}
    for k, v in sd.items():
        for prefix, name in PREFIXES.items():
            if k.startswith(prefix):
                parts[name][k[len(prefix) :]] = v
                break
        else:
            parts["matcher"][k] = v
    return parts


def _build_matcher(ctx: Context, *, assign: bool) -> Matcher:
    if assign:
        with skipped_init():
            net = Matcher(ctx.params)
    else:
        net = Matcher(ctx.params)
    parts = split_checkpoint(ctx.state_dict(0))
    modules = dict(net.named_modules())
    for name, sd in parts.items():
        if not sd and name not in modules:
            continue
        part = net.get_submodule(name)
        if assign:
            contiguous = {key: weight.contiguous() for key, weight in sd.items()}
            load_strict(part, contiguous, f"LoMa {name} weights", assign=True)
        else:
            part.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False)


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (one file: detector, descriptor, DINOv2, matcher) on `ctx.device`."""
    amp = None if ctx.dtype == "float32" else ctx.torch_dtype
    direct = (
        ctx.device.type == "cuda"
        and ctx.params.descriptor == "dedode_g"
        and amp is not None
        and ctx.user_weights is None
    )
    net = _build_matcher(ctx, assign=direct)
    if direct and net.dinov2 is not None:
        # Fold in float32 on the host before placing the DINO weights in their compute dtype.
        fold_layer_scale(net.dinov2)
        net.dinov2.to(device=ctx.device, dtype=amp)
    net.to(ctx.device)
    if net.dinov2 is not None:
        sdpa_attention(net.dinov2)
    if ctx.device.type == "cuda":  # not on CPU: the float32 reference parity
        if net.dinov2 is not None and not direct:
            fold_layer_scale(net.dinov2)
        fold_batch_norms(net)
        for m in net.modules():
            if isinstance(m, ConvRefiner):
                m.to(memory_format=torch.channels_last)
        net.detector.encoder.to(memory_format=torch.channels_last)
        net.descriptor.encoder.to(memory_format=torch.channels_last)
    net.set_amp(amp)
    return net


class Portable(Runner):
    """Plain PyTorch; see the module docstring for the autocast regions on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores."""
        return self.net(image1, image2)

    def describe(self, image: torch.Tensor) -> Described:
        """Keypoints and descriptions of one image: the per-image stage."""
        return self.net.describe(image)

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Described, Described]) -> dict:
        """`__call__` with the two images' `describe` results."""
        return self.net.match(image1, image2, *described)

    def module(self) -> torch.nn.Module:
        """Detector, descriptor and matcher (data-dependent match counts: compiles with graph breaks, no export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


class Kernels(Portable):
    """Accelerate the DeDoDe descriptor convolutions while preserving detector precision."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Prepare the loaded descriptor for CUDA."""
        super().__init__(ctx)
        prepare(self.net.descriptor, ctx.device)
        # These weights are consumed under autocast. Store them once in the inference dtype.
        for network in (self.net.detector, self.net.matcher):
            for layer in network.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layer.to(self.net.amp)
        override(self.net.detector, _detector_logits)
        override(self.net, _sample_descriptor, "_descriptions")


def _deferred_descriptor(network: Descriptor, image: torch.Tensor) -> torch.Tensor:
    """Resize the previous descriptor after computing the final convolution."""
    maps, sizes = network.encoder(image)
    features = list(maps)
    del maps
    scales = list(network.decoder.layers)
    final = len(scales) - 1
    descriptor = context = None
    for i, scale in enumerate(scales):
        delta, context = network.decoder(features.pop(), scale=scale, context=context)
        if i == final:
            if descriptor is None:
                return delta
            prior = F.interpolate(descriptor, size=delta.shape[-2:], mode="bilinear", align_corners=False)
            return delta.add_(prior)
        descriptor = delta if descriptor is None else delta.add_(descriptor)
        size = sizes[-2 - i]
        if i < final - 1:
            descriptor = F.interpolate(descriptor, size=size, mode="bilinear", align_corners=False)
        context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
    msg = "LoMa needs a descriptor with at least one scale"
    raise ValueError(msg)


def _detector_logits(detector: Detector, image: torch.Tensor) -> torch.Tensor:
    """Keep the multi-scale logits in float32 and resize the context in its next convolution's dtype."""
    maps, sizes = detector.encoder(image)
    features = list(maps)
    del maps
    logits = context = None
    for i, scale in enumerate(detector.decoder.layers):
        layer_input = features.pop()
        if context is not None:
            layer_input = torch.cat((layer_input, context), dim=1)
        del context
        result = detector.decoder.layers[scale](layer_input)
        del layer_input
        delta, context = result[:, : detector.decoder.num_prototypes], result[:, detector.decoder.num_prototypes :]
        del result
        logits = delta.float() if logits is None else logits.add_(delta)
        if features:
            size = sizes[-2 - i]
            logits = F.interpolate(logits, size, mode="bicubic", align_corners=False)
            context = F.interpolate(context, size, mode="bilinear", align_corners=False)
    assert logits is not None  # noqa: S101 - each supported detector has a nonempty decoder
    return logits


def _sample_descriptor(net: Matcher, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    return dedode_sparse.sample(net.descriptor, image, keypoints, blend=loma_descriptor.blend)


class _MlxPortable(Portable):
    """`Portable` for MLX: detector, descriptor, keypoint sampling and matcher in MLX operations (`mlx_net`).

    The two image resizes are PyTorch's on the CPU (bicubic with antialiasing, as `Matcher._resize`); the keypoints
    and descriptions stay on MLX between `describe` and `match` within a call, and only the match decisions are read
    on the host. The PyTorch networks are released once their weights are copied.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch networks on the CPU and copy their weights to MLX."""
        super().__init__(ctx)
        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self._mlx = mlx_net.MlxLoMa(self.net, mlx_wrap.run_dtype(ctx))
        for name in ("detector", "descriptor", "matcher", "dinov2"):  # their MLX copies hold the weights now
            setattr(self.net, name, None)

    def _features(self, image: torch.Tensor) -> tuple[Any, Any]:
        """Queue the keypoints (normalised) and descriptions of one image on MLX."""
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - imports mlx

        net, p = self.net, self.net.params
        image = image.float()
        h, w = image.shape[-2:]
        scale = p.long_side / max(h, w)
        det = net._resize(image, (int(scale * h) // STRIDE * STRIDE, int(scale * w) // STRIDE * STRIDE))  # noqa: SLF001
        det = to_mx((det - net.mean) / net.std).transpose(0, 2, 3, 1)
        kpts = self._mlx.keypoints(det, p.max_keypoints, SUBPIXEL_TEMP)
        mx.async_eval(kpts)
        resized = net._resize(image, (p.size, p.size))  # noqa: SLF001 - on the CPU while the keypoints are sampled
        if p.normalize_descriptor:
            resized = (resized - net.mean) / net.std
        desc = self._mlx.descriptions(to_mx(resized).transpose(0, 2, 3, 1), kpts)
        mx.async_eval(desc)
        return kpts, desc

    def _result(self, image1: torch.Tensor, image2: torch.Tensor, f1: Any, f2: Any) -> dict[str, torch.Tensor]:  # noqa: ANN401
        from fastforks._core.mlx import mx, to_torch  # noqa: PLC0415 - imports mlx

        m0, scores = self._mlx.match(f1[0], f2[0], f1[1], f2[1], self.net.params.threshold)
        mx.eval(m0, scores, f1[0], f2[0])
        m0, scores = np.array(m0).astype(np.int64), np.array(scores)
        idx = np.flatnonzero(m0 > -1)
        return {
            "keypoints1": self.net.to_input(to_torch(f1[0]), image1),
            "keypoints2": self.net.to_input(to_torch(f2[0]), image2),
            "pairs": torch.from_numpy(np.stack([idx, m0[idx]], axis=-1)),
            "scores": torch.from_numpy(scores[idx]),
        }

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores (CPU tensors)."""
        return self._result(image1, image2, self._features(image1), self._features(image2))

    def describe(self, image: torch.Tensor) -> Described:
        """Keypoints (normalised) and descriptions of one image as CPU tensors."""
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - imports mlx

        kpts, desc = self._features(image)
        return to_torch(kpts), to_torch(desc)

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Described, Described]) -> dict:
        """`__call__` with the two images' `describe` results."""
        from fastforks._core.mlx import to_mx  # noqa: PLC0415 - imports mlx

        f1, f2 = ((to_mx(k), to_mx(d)) for k, d in described)
        return self._result(image1, image2, f1, f2)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
