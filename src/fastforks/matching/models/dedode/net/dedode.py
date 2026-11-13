"""DeDoDe (Edstedt et al., 3DV 2024) and DeDoDe v2: upstream Parskatt/DeDoDe `DeDoDe/` (MIT), inference only.

fastforks changes: `encoder.py` (VGG19-BN features, VGG + frozen DINOv2), `decoder.py`, `detectors/dedode_detector.py`,
`descriptors/dedode_descriptor.py`, the model zoo's refiner sizes and the `utils.py` functions inference uses
(`sample_keypoints` with upstream's detect settings, `get_grid`, `dual_softmax_matcher`) in one file; torchvision's
`vgg19_bn().features[:40]` written out as the same layers (the same state-dict keys); the DINOv2 ViT-L/14 is
fastforks.shared.depth_anything_v2's (the same network; its weights are passed in, not downloaded); the autocast
dtype is an argument (`None` = float32) instead of always float16 on CUDA; path / PIL helpers and training code
removed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

VGG19_BN = (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M")  # torchvision cfg "E"


def vgg19_bn_features() -> nn.ModuleList:
    """torchvision `vgg19_bn().features[:40]`: conv-BN-ReLU blocks up to and including the 4th max-pool."""
    layers: list[nn.Module] = []
    c = 3
    for v in VGG19_BN:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(c, int(v), kernel_size=3, padding=1), nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
            c = int(v)
    return nn.ModuleList(layers)


def autocast(amp: torch.dtype | None) -> torch.autocast:
    """Upstream's `torch.autocast("cuda", ...)`: on with `amp`, off (float32) with `None`."""
    return torch.autocast("cuda", enabled=amp is not None, dtype=amp or torch.float16)


class VGG(nn.Module):
    """The features before each max-pool (strides 1, 2, 4, 8) and their sizes."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = vgg19_bn_features()
        self.amp: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Size]]:
        with autocast(self.amp):
            feats, sizes = [], []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


class VGGDINOv2(nn.Module):
    """VGG features + the frozen DINOv2 ViT-L/14's normalised patch tokens (stride 14, last)."""

    def __init__(self, dinov2: nn.Module) -> None:
        super().__init__()
        self.vgg = VGG()
        self.dinov2 = [dinov2]  # upstream: not a submodule (not in the descriptor's state dict)
        self.dino_dtype = torch.float32  # upstream: float16 weights and input with amp

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Size]]:
        feats, sizes = self.vgg(x)
        b, _, h, w = x.shape
        tokens = self.dinov2[0].forward_features(x.to(self.dino_dtype))["x_norm_patchtokens"]
        f = tokens.permute(0, 2, 1).reshape(b, 1024, h // 14, w // 14)
        return [*feats, f], [*sizes, f.shape[-2:]]


class ConvRefiner(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hidden_blocks: int) -> None:
        super().__init__()
        self.block1 = self.create_block(in_dim, hidden_dim, dw=False, kernel_size=1)
        self.hidden_blocks = nn.Sequential(
            *[self.create_block(hidden_dim, hidden_dim, dw=True, kernel_size=5) for _ in range(hidden_blocks)]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp: torch.dtype | None = None

    @staticmethod
    def create_block(in_dim: int, out_dim: int, *, dw: bool, kernel_size: int) -> nn.Sequential:
        conv1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_dim if dw else 1
        )
        return nn.Sequential(conv1, nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.Conv2d(out_dim, out_dim, 1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        with autocast(self.amp):
            x0 = self.block1(feats)
            x = (self.hidden_blocks(x0) + x0) / 1.4  # residual = True in every released model
            return self.out_conv(x)


# model zoo: scale -> (in, hidden, out without the prototypes); hidden blocks
DETECTOR_L = {"8": (512, 512, 256), "4": (512, 256, 128), "2": (256, 128, 64), "1": (128, 64, 1)}, 8
DESCRIPTOR_B = {"8": (512, 512, 256), "4": (512, 256, 128), "2": (256, 64, 32), "1": (96, 32, 1)}, 5
DESCRIPTOR_G = {"14": (1024, 768, 512), "8": (1024, 512, 256), "4": (512, 256, 128), "2": (256, 64, 32), "1": (96, 32, 1)}, 5


class Decoder(nn.Module):
    def __init__(self, layers: dict[str, tuple[int, int, int]], hidden_blocks: int, num_prototypes: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {s: ConvRefiner(i, h, o + num_prototypes, hidden_blocks) for s, (i, h, o) in layers.items()}
        )
        self.num_prototypes = num_prototypes

    def forward(
        self, features: torch.Tensor, context: torch.Tensor | None, scale: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        return stuff[:, : self.num_prototypes], stuff[:, self.num_prototypes :]


class Detector(nn.Module):
    """DeDoDe-L detector: keypoint logits `[B, 1, H, W]` (float32) of a normalised image."""

    def __init__(self) -> None:
        super().__init__()
        layers, blocks = DETECTOR_L
        self.encoder = VGG()
        self.decoder = Decoder(layers, blocks, num_prototypes=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features, sizes = self.encoder(images)
        logits: torch.Tensor | float = 0
        context = None
        scales = list(self.decoder.layers.keys())
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales, strict=True)):
            delta_logits, context = self.decoder(feature_map, context=context, scale=scale)
            logits = logits + delta_logits.float()
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False)
        assert isinstance(logits, torch.Tensor)
        return logits.float()


class Descriptor(nn.Module):
    """DeDoDe-B / -G descriptor: the dense 256-d description grid of a normalised image."""

    def __init__(self, dinov2: nn.Module | None = None) -> None:
        super().__init__()
        layers, blocks = DESCRIPTOR_B if dinov2 is None else DESCRIPTOR_G
        self.encoder = VGG() if dinov2 is None else VGGDINOv2(dinov2)
        self.decoder = Decoder(layers, blocks, num_prototypes=256)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features, sizes = self.encoder(images)
        descriptor: torch.Tensor | float = 0
        context = None
        scales = list(self.decoder.layers.keys())
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales, strict=True)):
            delta_descriptor, context = self.decoder(feature_map, scale=scale, context=context)
            descriptor = descriptor + delta_descriptor
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                descriptor = F.interpolate(descriptor, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        assert isinstance(descriptor, torch.Tensor)
        return descriptor


def get_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Upstream `get_grid` for batch 1: the pixel centres in normalised coordinates, `[H * W, 2]` (x, y)."""
    ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).reshape(h * w, 2)


def sample_keypoints(logits: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `detect`: softmax over all pixels, `sample_keypoints(use_nms=False, sample_topk=True,
    increase_coverage=True)` (no border removal: `dedode_detector_L`'s default). `[1, 1, H, W]` -> `[N, 2]`, `[N]`."""
    _, _, h, w = logits.shape
    scoremap = logits.reshape(1, h * w).softmax(dim=-1).reshape(1, h, w)
    weights = (-torch.linspace(-2, 2, steps=51, device=logits.device) ** 2).exp()[None, None]
    local_density_x = F.conv2d((scoremap[:, None] + 1e-6) * 10000, weights[..., None, :], padding=(0, 51 // 2))
    local_density = F.conv2d(local_density_x, weights[..., None], padding=(51 // 2, 0))[:, 0]
    scoremap = scoremap * (local_density + 1e-8) ** (-1 / 2)
    inds = torch.topk(scoremap.reshape(h * w), k=num_samples).indices
    return get_grid(h, w, logits.device)[inds], scoremap.reshape(h * w)[inds]


def describe_keypoints(grid: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    """Upstream `describe_keypoints`: bilinear samples of the description grid, `[N, 256]` float32."""
    out = F.grid_sample(grid.float(), keypoints[None, None], mode="bilinear", align_corners=False)
    return out[0, :, 0].mT


def dual_softmax_match(
    desc_a: torch.Tensor, desc_b: torch.Tensor, inv_temp: float, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `DualSoftMaxMatcher.match(normalize=True)`: mutual maxima of the dual softmax above `threshold`.

    Returns the index pairs `[K, 2]` and their dual-softmax probability `[K]`.
    """
    desc_a = desc_a / desc_a.norm(dim=-1, keepdim=True)
    desc_b = desc_b / desc_b.norm(dim=-1, keepdim=True)
    corr = (desc_a @ desc_b.mT) * inv_temp
    p = corr.softmax(dim=-2) * corr.softmax(dim=-1)
    keep = (p == p.max(dim=-1, keepdim=True).values) & (p == p.max(dim=-2, keepdim=True).values) & (p > threshold)
    pairs = torch.nonzero(keep)
    return pairs, p[pairs[:, 0], pairs[:, 1]]
