"""ViTPose / ViTPose++ network (HF transformers `VitPoseForPoseEstimation`, usyd-community checkpoints), inference only.

- patch embedding: 16 x 16 stride-16 conv with 2 px zero padding (a 16 x 12 grid at 256 x 192), plus the position
  table's patch rows and its (unused) cls row, summed once at load
- plain pre-norm ViT blocks (LayerNorm eps 1e-12, exact GELU); the qkv projections are packed into one GEMM
- ViTPose++ mixture-of-experts MLP: the shared `fc2` gives `dim - part` channels and one expert per training dataset
  the last `part`; the checkpoints carry only the COCO head, so the COCO expert (dataset index 0) is concatenated
  onto `fc2` at load: one dense `fc2`, exactly the reference's output for `dataset_index = 0`
- decoder: "classic" (two 4 x 4 stride-2 deconvolutions + BatchNorm + ReLU, then a 1 x 1 conv; BatchNorm folded into
  the deconvolutions) or "simple" (ReLU, 4x bilinear upsample, 3 x 3 conv); heatmaps at 1/4 of the input
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PATCH = 16  # fixed: architecture (patch size)
PATCH_PAD = 2  # fixed: architecture (the reference's patch conv padding)
EPS = 1e-12  # fixed: architecture (LayerNorm eps of the reference config)
MLP_RATIO = 4  # fixed: architecture
DECONV = 256  # fixed: architecture (classic decoder width)
BN_EPS = 1e-5  # fixed: architecture (torch BatchNorm2d default, as the reference)
UPSAMPLE = 4  # fixed: architecture (simple decoder scale factor)
COCO_EXPERT = 0  # fixed: checkpoint layout (dataset index of COCO in ViTPose++)


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    dim: int  # hidden size
    depth: int  # blocks
    heads: int  # attention heads
    part: int  # ViTPose++: expert channels of the MoE MLP (0: plain MLP)
    experts: int  # ViTPose++: experts (training datasets); 0 for plain ViTPose
    decoder: str  # "classic" | "simple"
    keypoints: int  # heatmap channels
    height: int  # network input height
    width: int  # network input width


class Block(nn.Module):
    """Pre-norm ViT block: x + attn(LN(x)), then x + mlp(LN(x))."""

    def __init__(self, a: Arch) -> None:
        """Layers of one block."""
        super().__init__()
        d = a.dim
        self.heads = a.heads
        self.norm1 = nn.LayerNorm(d, eps=EPS)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, eps=EPS)
        self.fc1 = nn.Linear(d, MLP_RATIO * d)
        self.fc2 = nn.Linear(MLP_RATIO * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, N, C]` -> `[B, N, C]`."""
        b, n, c = x.shape
        h = self.heads
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(o)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ViTPose(nn.Module):
    """Normalised crops `[B, 3, h, w]` -> heatmaps `[B, K, h / 4, w / 4]`."""

    def __init__(self, a: Arch) -> None:
        """Layers of `a` (weights from `convert`)."""
        super().__init__()
        self.arch = a
        self.gh, self.gw = a.height // PATCH, a.width // PATCH
        self.patch = nn.Conv2d(3, a.dim, PATCH, PATCH, padding=PATCH_PAD)
        self.pos = nn.Parameter(torch.zeros(1, self.gh * self.gw, a.dim))
        self.blocks = nn.ModuleList(Block(a) for _ in range(a.depth))
        self.norm = nn.LayerNorm(a.dim, eps=EPS)
        if a.decoder == "classic":
            self.head: nn.Module = nn.Sequential(
                nn.ConvTranspose2d(a.dim, DECONV, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(DECONV, DECONV, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(DECONV, a.keypoints, 1),
            )
        elif a.decoder == "simple":
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Upsample(scale_factor=UPSAMPLE, mode="bilinear", align_corners=False),
                nn.Conv2d(a.dim, a.keypoints, 3, padding=1),
            )
        else:
            msg = f"vitpose: unknown decoder {a.decoder!r} (classic, simple)"
            raise ValueError(msg)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens `[B, N, C]` with the position table added."""
        return self.patch(x).flatten(2).transpose(1, 2) + self.pos

    def feature_map(self, t: torch.Tensor) -> torch.Tensor:
        """Final LayerNorm of the tokens `[B, N, C]` as the `[B, C, gh, gw]` grid."""
        t = self.norm(t)
        return t.transpose(1, 2).reshape(t.shape[0], -1, self.gh, self.gw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Heatmaps of normalised crops at the network size."""
        t = self.tokens(x)
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.feature_map(t))


def _fold_bn(sd: dict[str, torch.Tensor], bn: str, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A bias-free ConvTranspose2d weight `[in, out, k, k]` followed by BatchNorm `bn` -> (weight, bias), float32."""
    s = sd[f"{bn}.weight"].double() / (sd[f"{bn}.running_var"].double() + BN_EPS).sqrt()
    b = sd[f"{bn}.bias"].double() - sd[f"{bn}.running_mean"].double() * s
    return (w.double() * s[None, :, None, None]).float(), b.float()


def convert(sd: dict[str, torch.Tensor], a: Arch) -> dict[str, torch.Tensor]:
    """The HF state dict -> `ViTPose`'s: packed qkv, COCO expert folded into fc2, position rows summed, BN folded."""
    out: dict[str, torch.Tensor] = {}
    e = "backbone.embeddings."
    out["patch.weight"] = sd[f"{e}patch_embeddings.projection.weight"]
    out["patch.bias"] = sd[f"{e}patch_embeddings.projection.bias"]
    pos = sd[f"{e}position_embeddings"]
    out["pos"] = pos[:, 1:] + pos[:, :1]
    for i in range(a.depth):
        p, q = f"backbone.encoder.layer.{i}.", f"blocks.{i}."
        att = f"{p}attention.attention."
        for t in ("weight", "bias"):
            out[f"{q}qkv.{t}"] = torch.cat([sd[f"{att}{n}.{t}"] for n in ("query", "key", "value")])
            out[f"{q}proj.{t}"] = sd[f"{p}attention.output.dense.{t}"]
            out[f"{q}norm1.{t}"] = sd[f"{p}layernorm_before.{t}"]
            out[f"{q}norm2.{t}"] = sd[f"{p}layernorm_after.{t}"]
            out[f"{q}fc1.{t}"] = sd[f"{p}mlp.fc1.{t}"]
            fc2 = sd[f"{p}mlp.fc2.{t}"]
            if a.experts:
                fc2 = torch.cat([fc2, sd[f"{p}mlp.experts.{COCO_EXPERT}.{t}"]])
            out[f"{q}fc2.{t}"] = fc2
    for t in ("weight", "bias"):
        out[f"norm.{t}"] = sd[f"backbone.layernorm.{t}"]
    if a.decoder == "classic":
        out["head.0.weight"], out["head.0.bias"] = _fold_bn(sd, "head.batchnorm1", sd["head.deconv1.weight"])
        out["head.2.weight"], out["head.2.bias"] = _fold_bn(sd, "head.batchnorm2", sd["head.deconv2.weight"])
        out["head.4.weight"], out["head.4.bias"] = sd["head.conv.weight"], sd["head.conv.bias"]
    else:
        out["head.2.weight"], out["head.2.bias"] = sd["head.conv.weight"], sd["head.conv.bias"]
    return out
