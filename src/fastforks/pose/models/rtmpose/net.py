"""RTMPose network (mmpose 1.3.2 `TopdownPoseEstimator`: CSPNeXt backbone + `RTMCCHead`), inference only.

- CSPNeXt (mmdet 3.3.0, arch P5): a 3-conv stem (stride 2), then four stages, each a 3 x 3 stride-2 conv and a
  `CSPLayer`; the last stage has an SPP bottleneck (max pools 5, 9, 13) before it. A `CSPLayer` splits into a main
  branch (1 x 1 conv, `CSPNeXtBlock`s: 3 x 3 conv, then 5 x 5 depthwise + 1 x 1 pointwise, residual) and a 1 x 1
  shortcut, concatenates them, applies channel attention (global average pool, 1 x 1 conv with bias, hard sigmoid,
  multiply) and a final 1 x 1 conv. Every conv is followed by BatchNorm (eps 1e-5) and SiLU; BatchNorm is folded into
  the convs at load. Depth and width scale with `deepen` / `widen` (t, s, m, l)
- RTMCC head: a 7 x 7 conv to one map per keypoint (stride 32), flattened to a token per keypoint; ScaleNorm + a
  linear layer to 256; one gated attention unit (GAU: ScaleNorm, SiLU(linear) -> u, v, a shared base that per-dim
  scales and offsets give q and k, relu(q k^T / sqrt(s))^2 attention, u * (attn v), linear; scaled residual); then
  two linear layers to the SimCC x / y logits (2 bins per input pixel)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# (in, out, blocks, identity shortcut in the blocks); the last stage also has the SPP bottleneck
STAGES = (
    (64, 128, 3, True),
    (128, 256, 6, True),
    (256, 512, 6, True),
    (512, 1024, 3, False),
)  # fixed: CSPNeXt P5 at 1.0
SPP_POOLS = (5, 9, 13)  # fixed: architecture (SPP bottleneck kernels of the last stage)
DW_KERNEL = 5  # fixed: architecture (CSPNeXtBlock depthwise kernel)
BN_EPS = 1e-5  # fixed: architecture (SyncBN default of the configs)
STRIDE = 32  # fixed: architecture (backbone output stride)
FINAL_KERNEL = 7  # fixed: architecture (head `final_layer_kernel_size`)
HIDDEN = 256  # fixed: architecture (GAU `hidden_dims`)
S = 128  # fixed: architecture (GAU shared q / k width)
EXPANSION = 2  # fixed: architecture (GAU `expansion_factor`: e = 512)
SPLIT = 2  # fixed: architecture (SimCC bins per input pixel, `simcc_split_ratio`)
NORM_EPS = 1e-5  # fixed: architecture (ScaleNorm clamp)


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    deepen: float  # CSPNeXt depth factor (blocks per stage)
    widen: float  # CSPNeXt width factor (channels)
    keypoints: int  # SimCC outputs (17 COCO keypoints)
    height: int  # network input height
    width: int  # network input width


class ConvAct(nn.Module):
    """mmcv `ConvModule` with its BatchNorm folded: conv (with the folded bias) + SiLU."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, groups: int = 1) -> None:
        """A `k x k` conv with `k // 2` padding."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU(conv(x)) (traced by `fastforks._optimize` as a conv module + SiLU)."""
        return F.silu(self.conv(x))


class DepthwiseSeparable(nn.Module):
    """5 x 5 depthwise conv + SiLU, then 1 x 1 conv + SiLU."""

    def __init__(self, c: int) -> None:
        """Both convs on `c` channels."""
        super().__init__()
        self.depthwise_conv = ConvAct(c, c, DW_KERNEL, groups=c)
        self.pointwise_conv = ConvAct(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pointwise(depthwise(x))."""
        return self.pointwise_conv(self.depthwise_conv(x))


class Block(nn.Module):
    """`CSPNeXtBlock` (expansion 1): dwsep(conv3x3(x)), plus x with the identity shortcut."""

    def __init__(self, c: int, *, identity: bool) -> None:
        """Layers on `c` channels."""
        super().__init__()
        self.conv1 = ConvAct(c, c, 3)
        self.conv2 = DepthwiseSeparable(c)
        self.identity = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The block's output."""
        y = self.conv2(self.conv1(x))
        return x + y if self.identity else y


class Attention(nn.Module):
    """Channel attention: x * hardsigmoid(conv1x1(mean over HW))."""

    def __init__(self, c: int) -> None:
        """The 1 x 1 conv with bias."""
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweighted channels."""
        return x * F.hardsigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class CSPLayer(nn.Module):
    """`CSPLayer` with CSPNeXt blocks and channel attention."""

    def __init__(self, c: int, blocks: int, *, identity: bool) -> None:
        """In and out `c` channels, `blocks` blocks on `c // 2`."""
        super().__init__()
        mid = c // 2
        self.main_conv = ConvAct(c, mid, 1)
        self.short_conv = ConvAct(c, mid, 1)
        self.blocks = nn.Sequential(*(Block(mid, identity=identity) for _ in range(blocks)))
        self.attention = Attention(2 * mid)
        self.final_conv = ConvAct(2 * mid, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """final(attention(cat(blocks(main(x)), short(x))))."""
        y = torch.cat([self.blocks(self.main_conv(x)), self.short_conv(x)], 1)
        return self.final_conv(self.attention(y))


class SPP(nn.Module):
    """SPP bottleneck: 1 x 1 conv to half, cat with max pools (stride 1, same size), 1 x 1 conv."""

    def __init__(self, c: int) -> None:
        """In and out `c` channels."""
        super().__init__()
        self.conv1 = ConvAct(c, c // 2, 1)
        self.conv2 = ConvAct(c // 2 * (len(SPP_POOLS) + 1), c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv2(cat(x, pools(x)))."""
        x = self.conv1(x)
        return self.conv2(torch.cat([x, *(F.max_pool2d(x, k, 1, k // 2) for k in SPP_POOLS)], 1))


def blocks_of(n: int, deepen: float) -> int:
    """Blocks of a stage: `max(round(n * deepen), 1)` (Python's round, as the reference)."""
    return max(round(n * deepen), 1)


class CSPNeXt(nn.Module):
    """The backbone: stem + 4 stages; the stride-32 map of the last stage."""

    def __init__(self, a: Arch) -> None:
        """Layers scaled by `a.deepen` / `a.widen`."""
        super().__init__()
        w = a.widen
        c0 = int(STAGES[0][0] * w // 2)
        self.stem = nn.Sequential(ConvAct(3, c0, 3, 2), ConvAct(c0, c0, 3), ConvAct(c0, int(STAGES[0][0] * w), 3))
        for i, (cin, cout, n, identity) in enumerate(STAGES):
            ci, co = int(cin * w), int(cout * w)
            layers: list[nn.Module] = [ConvAct(ci, co, 3, 2)]
            if i == len(STAGES) - 1:
                layers.append(SPP(co))
            layers.append(CSPLayer(co, blocks_of(n, a.deepen), identity=identity))
            self.add_module(f"stage{i + 1}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 1024 w, h / 32, w / 32]`."""
        x = self.stem(x)
        for i in range(len(STAGES)):
            x = getattr(self, f"stage{i + 1}")(x)
        return x


class ScaleNorm(nn.Module):
    """x / clamp(||x|| / sqrt(dim), eps) * g (one scalar gain)."""

    def __init__(self, dim: int) -> None:
        """The gain."""
        super().__init__()
        self.scale = dim**-0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised over the last axis."""
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=NORM_EPS) * self.g


class Scale(nn.Module):
    """Per-channel residual scale."""

    def __init__(self, dim: int) -> None:
        """The scales."""
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class GAU(nn.Module):
    """`RTMCCBlock` (self-attention, no relative bias, no RoPE): res_scale * x + GAU(x)."""

    def __init__(self) -> None:
        """Layers of the unit (`HIDDEN` in and out)."""
        super().__init__()
        self.e = EXPANSION * HIDDEN
        self.ln = ScaleNorm(HIDDEN)
        self.uv = nn.Linear(HIDDEN, 2 * self.e + S, bias=False)
        self.gamma = nn.Parameter(torch.ones(2, S))
        self.beta = nn.Parameter(torch.zeros(2, S))
        self.o = nn.Linear(self.e, HIDDEN, bias=False)
        self.res_scale = Scale(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, K, HIDDEN]` -> same."""
        u, v, base = F.silu(self.uv(self.ln(x))).split([self.e, self.e, S], 2)
        q, k = (base[:, :, None] * self.gamma + self.beta).unbind(2)
        kernel = F.relu(q @ k.transpose(1, 2) / math.sqrt(S)).square()
        return x * self.res_scale.scale + self.o(u * (kernel @ v))


class Head(nn.Module):
    """`RTMCCHead`: stride-32 features -> SimCC x logits `[B, K, w * 2]`, y logits `[B, K, h * 2]`."""

    def __init__(self, a: Arch) -> None:
        """Layers for the network input size."""
        super().__init__()
        flat = (a.height // STRIDE) * (a.width // STRIDE)
        self.final_layer = nn.Conv2d(int(STAGES[-1][1] * a.widen), a.keypoints, FINAL_KERNEL, 1, FINAL_KERNEL // 2)
        self.mlp = nn.Sequential(ScaleNorm(flat), nn.Linear(flat, HIDDEN, bias=False))
        self.gau = GAU()
        self.cls_x = nn.Linear(HIDDEN, a.width * SPLIT, bias=False)
        self.cls_y = nn.Linear(HIDDEN, a.height * SPLIT, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(x logits, y logits)."""
        t = self.gau(self.mlp(self.final_layer(x).flatten(2)))
        return self.cls_x(t), self.cls_y(t)


class RTMPose(nn.Module):
    """Normalised crops `[B, 3, h, w]` -> (SimCC x `[B, K, 2 w]`, SimCC y `[B, K, 2 h]`)."""

    def __init__(self, a: Arch) -> None:
        """Backbone and head."""
        super().__init__()
        self.backbone = CSPNeXt(a)
        self.head = Head(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SimCC logits."""
        return self.head(self.backbone(x))


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The mmpose keys as `RTMPose` keys: every `ConvModule`'s BatchNorm folded into its conv (weight and bias)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.endswith(".bn.num_batches_tracked") or ".bn." in k:
            continue
        if k.endswith(".conv.weight"):
            p = k[: -len(".conv.weight")]
            std = (sd[f"{p}.bn.running_var"].double() + BN_EPS).sqrt()
            g = sd[f"{p}.bn.weight"].double() / std
            out[k] = (v.double() * g.view(-1, 1, 1, 1)).float()
            out[f"{p}.conv.bias"] = (sd[f"{p}.bn.bias"].double() - sd[f"{p}.bn.running_mean"].double() * g).float()
        else:
            out[k] = v
    return out
