# fastforks: upstream `romatch/models/matcher.py`, `encoders.py` and `transformer/__init__.py` (Parskatt/RoMa) reduced
# to the inference path of `roma_outdoor`: the VGG19-BN + DINOv2 encoder, the Gaussian-process match at stride 16, the
# transformer classifier and the convolutional refiners. Module names and layout follow upstream, so the checkpoint
# keys (`encoder.cnn.layers.N`, `decoder.*`) load unchanged; the DINOv2 ViT-L/14 (its own weight file) is
# `encoder.dinov2` here (upstream keeps it outside the state dict). Training-only branches (dropout, covariance,
# learned embeddings, other bases and kernels) are removed; einops is replaced by reshapes. Autocast regions are those
# of upstream on CUDA, switchable through the `amp` attributes (None: float32 as upstream on the CPU).
import math

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.depth_anything_v2 import Block, SdpaAttention

from .local_correlation import local_correlation

TRAIN_RES = 560  # fixed: upstream's training resolution, the denominator of the refiners' displacement scale
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # fixed: upstream's input normalisation
IMAGENET_STD = (0.229, 0.224, 0.225)  # fixed: upstream's input normalisation


def autocast(x: torch.Tensor, dtype: torch.dtype | None):  # noqa: ANN201
    """`torch.autocast(x.device.type, dtype)`, or disabled for `dtype=None` (upstream's `get_autocast_params`)."""
    return torch.autocast(x.device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixel centres of an `h x w` image in normalised `(x, y)`, `[b, 2, h, w]` (upstream `get_placeholder_flow`)."""
    ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy))[None].expand(b, 2, h, w)


def imagenet(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def _vgg19bn_layers() -> list[nn.Module]:
    """`torchvision.models.vgg19_bn().features[:39]` from plain layers (same indices; the last max pool is unused)."""
    layers: list[nn.Module] = []
    c_in = 3
    for v in (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512):
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(c_in, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            c_in = v
    return layers


class VGG19(nn.Module):
    """Features at strides 1, 2, 4 and 8 (the inputs of the max pools) as `{stride: [B, C, H, W]}`."""

    amp: torch.dtype | None = None

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_vgg19bn_layers())

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        with autocast(x, self.amp):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale * 2
                x = layer(x)
            feats[scale] = x  # upstream: the input of features[39], the last max pool
            return feats


class CNNandDinov2(nn.Module):
    """The VGG pyramid plus the DINOv2 patch tokens as the stride-16 level (`dinov2`: the shared ViT-L/14)."""

    def __init__(self, dinov2: nn.Module) -> None:
        super().__init__()
        self.cnn = VGG19()
        self.dinov2 = dinov2
        self.dino_dtype = torch.float32  # upstream: the ViT in fp16 weights on CUDA (`amp_dtype`)

    def describe(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 1024, H / 14, W / 14]` normalised patch tokens of the last block (upstream `forward_features`)."""
        b, _, h, w = x.shape
        tokens = self.dinov2.get_intermediate_layers(x.to(self.dino_dtype), 1)[0]
        return tokens.permute(0, 2, 1).reshape(b, -1, h // 14, w // 14)

    def forward(self, x: torch.Tensor, upsample: bool = False) -> dict[int, torch.Tensor]:  # noqa: FBT001, FBT002
        feats = self.cnn(x)
        if not upsample:
            feats[16] = self.describe(x)
        return feats


class TransformerDecoder(nn.Module):
    """Five DINOv2 blocks over the stride-16 tokens, a linear classifier into 64 x 64 anchors + a certainty logit."""

    amp: torch.dtype | None = None

    def __init__(self, dim: int = 1024, out_dim: int = 64**2 + 1, depth: int = 5) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[Block(dim, 8, attn_class=SdpaAttention) for _ in range(depth)])
        self.to_out = nn.Linear(dim, out_dim)
        self.out_dim = out_dim

    def forward(self, gp_posterior: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast(gp_posterior, self.amp):
            x = torch.cat((gp_posterior, features), dim=1)
            b, c, h, w = x.shape
            z = self.blocks(x.reshape(b, c, h * w).permute(0, 2, 1))
            out = self.to_out(z).permute(0, 2, 1).reshape(b, self.out_dim, h, w)
            return out[:, :-1], out[:, -1:]


def cls_to_flow_refine(cls: torch.Tensor) -> torch.Tensor:
    """Upstream `cls_to_flow_refine`: probability-weighted anchor of the mode and its 4 neighbours, `[B, H, W, 2]`."""
    _, c, _, _ = cls.shape
    res = round(math.sqrt(c))
    anchors = torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=cls.device)
    gy, gx = torch.meshgrid(anchors, anchors, indexing="ij")
    g = torch.stack((gx, gy), dim=-1).reshape(c, 2)
    cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    index = torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1).clamp(0, c - 1).long()
    neighbours = torch.gather(cls, dim=1, index=index)[..., None]
    flow = sum(neighbours[:, i] * g[index[:, i]] for i in range(5))
    return flow / neighbours.sum(dim=1)


class GP(nn.Module):
    """The Gaussian-process match (cosine kernel, Fourier position basis), float32, Cholesky solve as upstream eval."""

    def __init__(self, gp_dim: int = 512, temperature: float = 0.2, sigma_noise: float = 0.1) -> None:
        super().__init__()
        self.pos_conv = nn.Conv2d(2, gp_dim, 1, 1)
        self.temperature, self.sigma_noise = temperature, sigma_noise

    def kernel(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Upstream `CosKernel`: `exp((cos(x, y) - 1) / T)`."""
        c = torch.einsum("bnd,bmd->bnm", x, y) / (x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps)
        return ((c - 1.0) / torch.tensor(self.temperature, device=c.device)).exp()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, _, h1, w1 = x.shape
        _, _, h2, w2 = y.shape
        f = torch.cos(8 * math.pi * self.pos_conv(grid(b, h2, w2, y.device)))
        d = f.shape[1]
        x, y, f = (t.float().flatten(2).transpose(1, 2) for t in (x, y, f))
        k_yy = self.kernel(y, y)
        k_xy = self.kernel(x, y)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        lower = torch.linalg.cholesky(k_yy + sigma_noise)
        mu_x = k_xy @ torch.cholesky_solve(f.reshape(b, h2 * w2, d), lower, upper=False)
        return mu_x.transpose(1, 2).reshape(b, d, h1, w1)


def _block(in_dim: int, out_dim: int, kernel_size: int = 5) -> nn.Sequential:
    """Upstream `ConvRefiner.create_block(dw=True)`: depthwise conv, BatchNorm, ReLU, 1x1 conv."""
    conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, 1, kernel_size // 2, groups=in_dim)
    return nn.Sequential(conv1, nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.Conv2d(out_dim, out_dim, 1, 1, 0))


class ConvRefiner(nn.Module):
    """Upstream `ConvRefiner` (dw, kernel 5, 8 hidden blocks, linear displacement embedding, correlation in image B)."""

    amp: torch.dtype | None = None

    def __init__(self, dim: int, emb_dim: int, local_corr_radius: int | None = None, hidden_blocks: int = 8) -> None:
        super().__init__()
        self.block1 = _block(dim, dim)
        self.hidden_blocks = nn.Sequential(*[_block(dim, dim) for _ in range(hidden_blocks)])
        self.out_conv = nn.Conv2d(dim, 3, 1, 1, 0)
        self.disp_emb = nn.Conv2d(2, emb_dim, 1, 1, 0)
        self.local_corr_radius = local_corr_radius
        self.correlate = local_correlation  # `(f_a, f_b, radius, warp [B, H, W, 2])` -> `[B, K, H, W]`

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, warp: torch.Tensor, scale_factor: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, hs, ws = x.shape
        with autocast(x, self.amp):
            x_hat = F.grid_sample(y, warp.permute(0, 2, 3, 1), align_corners=False, mode="bilinear")
            emb = self.disp_emb(40 / 32 * scale_factor * (warp - grid(b, hs, ws, x.device)))
            parts = [x, x_hat, emb]
            if self.local_corr_radius:
                parts.append(self.correlate(x, y, self.local_corr_radius, warp.permute(0, 2, 3, 1)))
            d = torch.cat(parts, dim=1)
            d = self.hidden_blocks(self.block1(d))
        d = self.out_conv(d.float())
        return d[:, :-1], d[:, -1:]


class Decoder(nn.Module):
    """Coarse match at stride 16, then refinement at strides 16, 8, 4, 2, 1 (8 to 1 in the upsampling pass)."""

    refine_init = 4  # fixed: upstream Decoder.refine_init
    amp: torch.dtype | None = None  # the projections' autocast

    def __init__(self) -> None:
        super().__init__()
        self.embedding_decoder = TransformerDecoder()
        self.gps = nn.ModuleDict({"16": GP()})
        dims = {"16": (1024, 512), "8": (512, 512), "4": (256, 256), "2": (128, 64), "1": (64, 9)}
        self.proj = nn.ModuleDict(
            {s: nn.Sequential(nn.Conv2d(i, o, 1, 1), nn.BatchNorm2d(o)) for s, (i, o) in dims.items()}
        )
        # scale: (displacement embedding, local correlation radius); input = 2 * proj + emb + window
        refiners = {"16": (128, 7), "8": (64, 3), "4": (32, 2), "2": (16, None), "1": (6, None)}
        self.conv_refiner = nn.ModuleDict(
            {
                s: ConvRefiner(2 * dims[s][1] + emb + (2 * r + 1) ** 2 if r else 2 * dims[s][1] + emb, emb, r)
                for s, (emb, r) in refiners.items()
            }
        )

    def forward(
        self,
        f1: dict[int, torch.Tensor],
        f2: dict[int, torch.Tensor],
        scale_factor: float,
        flow: torch.Tensor | None = None,
        certainty: torch.Tensor | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """`{scale: (flow [B, 2, h, w], certainty logit [B, 1, h, w])}`; `flow` given: the upsampling pass."""
        upsample = flow is not None
        scales = ["8", "4", "2", "1"] if upsample else ["16", "8", "4", "2", "1"]
        sizes = {scale: f.shape[-2:] for scale, f in f1.items()}
        h, w = sizes[1]
        coarsest = int(scales[0])
        if flow is None or certainty is None:
            flow = grid(f1[1].shape[0], *sizes[coarsest], f1[1].device)
            certainty = torch.zeros((), device=flow.device)
        else:
            flow = F.interpolate(flow, size=sizes[coarsest], align_corners=False, mode="bilinear")
            certainty = F.interpolate(certainty, size=sizes[coarsest], align_corners=False, mode="bilinear")
        out = {}
        for scale in scales:
            ins = int(scale)
            # upstream: under autocast on CUDA (`get_autocast_params` enables it there whatever it is asked), else
            # float32
            with autocast(f1[ins], self.amp):
                f1_s, f2_s = f1[ins], f2[ins]
                if self.amp is None:
                    f1_s, f2_s = f1_s.float(), f2_s.float()
                f1_s, f2_s = self.proj[scale](f1_s), self.proj[scale](f2_s)
            if scale == "16":
                cls, certainty = self.embedding_decoder(self.gps[scale](f1_s, f2_s), f1_s)
                flow = cls_to_flow_refine(cls).permute(0, 3, 1, 2)
            delta_flow, delta_certainty = self.conv_refiner[scale](f1_s, f2_s, flow, scale_factor)
            displacement = ins * torch.stack(
                (delta_flow[:, 0].float() / (self.refine_init * w), delta_flow[:, 1].float() / (self.refine_init * h)),
                dim=1,
            )
            flow = flow + displacement
            certainty = certainty + delta_certainty
            out[ins] = (flow, certainty)
            if scale != "1":
                flow = F.interpolate(flow, size=sizes[ins // 2], mode="bilinear")
                certainty = F.interpolate(certainty, size=sizes[ins // 2], mode="bilinear")
        return out


class RoMa(nn.Module):
    """Upstream `RegressionMatcher.match` of one direction on normalised, resized inputs.

    `forward(a, b, a_hi, b_hi, coarse_scale, fine_scale)`: `a`, `b` `[B, 3, h, w]` at the coarse size (a multiple of
    14), `a_hi`, `b_hi` at the fine size or None (no upsampling pass). Returns the warp `[B, 2, H, W]` (normalised
    `(x, y)` in image B, clamped to [-1, 1]) and the certainty `[B, H, W]` (sigmoid of the logit minus upstream's
    attenuation by the coarse certainty; 0 where the warp left the image) at the last size.
    """

    def __init__(self, dinov2: nn.Module) -> None:
        super().__init__()
        self.encoder = CNNandDinov2(dinov2)
        self.decoder = Decoder()

    def features(self, x: torch.Tensor, upsample: bool = False) -> tuple[dict, dict]:  # noqa: FBT001, FBT002
        """Both images' pyramids from one batched encoder pass (upstream `extract_backbone_features`)."""
        pyramid = self.encoder(x, upsample=upsample)
        return {s: f.chunk(2)[0] for s, f in pyramid.items()}, {s: f.chunk(2)[1] for s, f in pyramid.items()}

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_hi: torch.Tensor | None = None,
        b_hi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hs, ws = a.shape[-2:]
        out = self.decoder(*self.features(torch.cat((a, b))), scale_factor=math.sqrt(hs * ws) / TRAIN_RES)
        low_res_certainty = out[16][1]
        flow, certainty = out[1]
        if a_hi is not None and b_hi is not None:
            hs, ws = a_hi.shape[-2:]
            f1, f2 = self.features(torch.cat((a_hi, b_hi)), upsample=True)
            flow, certainty = self.decoder(f1, f2, math.sqrt(hs * ws) / TRAIN_RES, flow, certainty)[1]
        low = F.interpolate(low_res_certainty, size=(hs, ws), align_corners=False, mode="bilinear")
        certainty = (certainty - 0.5 * low * (low < 0)).sigmoid()[:, 0]  # upstream attenuate_cert
        certainty = certainty.masked_fill((flow.abs() > 1).any(dim=1), 0)
        return flow.clamp(-1, 1), certainty
