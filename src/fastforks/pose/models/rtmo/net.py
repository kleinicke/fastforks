"""RTMO network (mmpose 1.3.2 `BottomupPoseEstimator`: CSPDarknet, HybridEncoder, RTMOHead), inference only.

Rewritten for the folded checkpoint (`convert`): every BatchNorm sits in its conv, the Focus stem (four pixel phases
concatenated, then a 3 x 3 conv) is one 6 x 6 stride-2 conv, and each RepVGG block of the neck is one 3 x 3 conv.

- `Backbone`: CSPDarknet P5 with SPP, feature maps at strides 8, 16, 32
- `Neck`: 1 x 1 projections to 256 channels, one transformer encoder layer on the stride-32 map, a top-down and a
  bottom-up pass of CSPRep layers, 1 x 1 projections of the stride-16 and stride-32 maps
- `Head`: per map a classification branch on the first half of the channels and a pose branch on the second half:
  person logit, box (centre offset, log size), keypoint visibility logits and a pose vector per grid cell
- `DCC` (dynamic coordinate classifier, run on the kept people only): the pose vector -> one feature per keypoint
  (a gated attention unit across the keypoints) -> a 1-D distribution over x and y bins that span the person's box
  -> the expected coordinate
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

STAGES = ((64, 128, 3), (128, 256, 9), (256, 512, 9), (512, 1024, 3))  # fixed: CSPDarknet P5 (in, out, blocks)
SPP_KERNELS = (5, 9, 13)  # fixed: reference SPP max-pool sizes
HIDDEN = 256  # fixed: reference neck width (all variants)
HEADS = 8  # fixed: reference encoder attention heads
FFN = 1024  # fixed: reference encoder feed-forward width
TEMPERATURE = 10000.0  # fixed: reference sine position temperature of the encoder
REP_BLOCKS = 3  # fixed: reference CSPRep blocks before the depth factor
POSE_VEC = 512  # fixed: reference pose vector / projector width before the width factor
POSE_CONV = 36  # fixed: reference pose branch channels per group before the width factor
POSE_GROUPS = 8  # fixed: reference pose branch groups
STRIDES = (16, 32)  # fixed: the two maps the head reads
KPT_FEAT = 128  # fixed: reference DCC feature channels per keypoint
GAU_EXPAND = 256  # fixed: reference GAU expanded width (u, v)
BINS = (192, 256)  # fixed: reference DCC bins along x, y
BN_EPS_YOLOX = 1e-3  # fixed: reference BatchNorm eps of the backbone and the head convs
BN_EPS = 1e-5  # fixed: reference BatchNorm eps elsewhere (PyTorch default)
NORM_EPS = 1e-5  # fixed: reference ScaleNorm eps
SOFTMAX_EPS = 1e-8  # fixed: reference eps in the bin normalisation
LOGIT_CLAMP = 5e4  # fixed: reference clamp of the bin logits


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    deepen: float  # depth factor (blocks per stage)
    widen: float  # width factor (channels)
    keypoints: int  # keypoints per person (17 COCO keypoints)
    size: int  # network input side (square letterbox)


class ConvAct(nn.Module):
    """mmcv `ConvModule` with its BatchNorm folded: conv (with the folded bias) + SiLU."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, groups: int = 1) -> None:
        """A `k x k` conv with `k // 2` padding."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU(conv(x)) (traced by `fastforks._optimize` as a conv module + SiLU)."""
        return F.silu(self.conv(x))


class Stem(nn.Module):
    """The Focus stem as one conv: 6 x 6, stride 2, padding 2 (`convert` spreads the 3 x 3 phase kernel) + SiLU."""

    def __init__(self, cout: int) -> None:
        """RGB (the reference's BGR order is folded into the weights' input channels by `convert`) -> `cout`."""
        super().__init__()
        self.conv = nn.Conv2d(3, cout, 6, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU(conv(x))."""
        return F.silu(self.conv(x))


class Bottleneck(nn.Module):
    """`DarknetBottleneck` (expansion 1): conv3x3(conv1x1(x)), plus x with the identity shortcut."""

    def __init__(self, c: int, *, identity: bool) -> None:
        """Layers on `c` channels."""
        super().__init__()
        self.conv1 = ConvAct(c, c, 1)
        self.conv2 = ConvAct(c, c, 3)
        self.identity = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The block's output."""
        y = self.conv2(self.conv1(x))
        return x + y if self.identity else y


class CSPLayer(nn.Module):
    """`CSPLayer`: final(cat(blocks(main(x)), short(x)))."""

    def __init__(self, c: int, blocks: int, *, identity: bool) -> None:
        """`c` -> `c` channels, `blocks` bottlenecks on `c // 2`."""
        super().__init__()
        self.main_conv = ConvAct(c, c // 2, 1)
        self.short_conv = ConvAct(c, c // 2, 1)
        self.blocks = nn.Sequential(*(Bottleneck(c // 2, identity=identity) for _ in range(blocks)))
        self.final_conv = ConvAct(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output."""
        return self.final_conv(torch.cat([self.blocks(self.main_conv(x)), self.short_conv(x)], 1))


class SPP(nn.Module):
    """`SPPBottleneck`: conv1x1 to half, the map and its 5 / 9 / 13 max-pools concatenated, conv1x1 back."""

    def __init__(self, c: int) -> None:
        """`c` -> `c` channels."""
        super().__init__()
        self.conv1 = ConvAct(c, c // 2, 1)
        self.conv2 = ConvAct(c // 2 * (len(SPP_KERNELS) + 1), c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The pooled map."""
        x = self.conv1(x)
        return self.conv2(torch.cat([x, *(F.max_pool2d(x, k, 1, k // 2) for k in SPP_KERNELS)], 1))


class Backbone(nn.Module):
    """CSPDarknet P5: the stem and four stages; the maps of stages 2, 3, 4 (strides 8, 16, 32)."""

    def __init__(self, a: Arch) -> None:
        """Layers scaled by the width and depth factors."""
        super().__init__()
        self.stem = Stem(int(STAGES[0][0] * a.widen))
        stages = []
        for i, (cin, cout, blocks) in enumerate(STAGES, 1):
            c, last = int(cout * a.widen), i == len(STAGES)
            layers: list[nn.Module] = [ConvAct(int(cin * a.widen), c, 3, 2)]
            if last:
                layers.append(SPP(c))
            layers.append(CSPLayer(c, max(round(blocks * a.deepen), 1), identity=not last))
            stages.append(nn.Sequential(*layers))
        self.stage1, self.stage2, self.stage3, self.stage4 = stages

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feature maps at strides 8, 16, 32 of an image batch `[B, 3, S, S]`."""
        c3 = self.stage2(self.stage1(self.stem(x)))
        c4 = self.stage3(c3)
        return c3, c4, self.stage4(c4)


class Encoder(nn.Module):
    """One post-norm transformer encoder layer; queries and keys carry the positional encoding, values do not."""

    def __init__(self) -> None:
        """Self-attention (packed q, k, v projection as `nn.MultiheadAttention`) and the GELU feed-forward."""
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * HIDDEN, HIDDEN))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * HIDDEN))
        self.out_proj = nn.Linear(HIDDEN, HIDDEN)
        self.fc1 = nn.Linear(HIDDEN, FFN)
        self.fc2 = nn.Linear(FFN, HIDDEN)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Tokens `[B, N, 256]` with positions `[1, N, 256]`."""
        b, n, c = x.shape
        qk = F.linear(x + pos, self.in_proj_weight[: 2 * c], self.in_proj_bias[: 2 * c])
        v = F.linear(x, self.in_proj_weight[2 * c :], self.in_proj_bias[2 * c :])
        q, k, v = (t.view(b, n, HEADS, c // HEADS).transpose(1, 2) for t in (*qk.split(c, -1), v))
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = self.norm1(x + self.out_proj(y))
        return self.norm2(x + self.fc2(F.gelu(self.fc1(x))))


class CSPRep(nn.Module):
    """`CSPRepLayer`: conv3(reps(conv1(x)) + conv2(x)), each RepVGG block folded to one 3 x 3 conv + SiLU."""

    def __init__(self, c: int, blocks: int) -> None:
        """`2 * HIDDEN` -> `HIDDEN` channels with `blocks` blocks on `c`."""
        super().__init__()
        self.conv1 = ConvAct(2 * HIDDEN, c, 1)
        self.conv2 = ConvAct(2 * HIDDEN, c, 1)
        self.bottlenecks = nn.Sequential(*(ConvAct(c, c, 3) for _ in range(blocks)))
        self.conv3 = ConvAct(c, HIDDEN, 1) if c != HIDDEN else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output."""
        y = self.bottlenecks(self.conv1(x)) + self.conv2(x)
        return y if self.conv3 is None else self.conv3(y)


class Neck(nn.Module):
    """`HybridEncoder`: the stride-16 and stride-32 maps for the head."""

    def __init__(self, a: Arch) -> None:
        """Layers scaled by the width and depth factors."""
        super().__init__()
        cin = [int(s[1] * a.widen) for s in STAGES[1:]]
        hidden, blocks = int(HIDDEN * a.widen), round(REP_BLOCKS * a.deepen)
        self.input_proj = nn.ModuleList(nn.Conv2d(c, HIDDEN, 1) for c in cin)
        self.encoder = Encoder()
        self.lateral_convs = nn.ModuleList(ConvAct(HIDDEN, HIDDEN, 1) for _ in range(2))
        self.fpn_blocks = nn.ModuleList(CSPRep(hidden, blocks) for _ in range(2))
        self.downsample_convs = nn.ModuleList(ConvAct(HIDDEN, HIDDEN, 3, 2) for _ in range(2))
        self.pan_blocks = nn.ModuleList(CSPRep(hidden, blocks) for _ in range(2))
        self.projector = nn.ModuleList(nn.Conv2d(HIDDEN, int(POSE_VEC * a.widen), 1) for _ in STRIDES)

    def positions(self, h: int, w: int) -> torch.Tensor:
        """Sine positions `[1, h * w, 256]` (float32): (cos, sin) of the row, then of the column, 64 periods each."""
        device = self.encoder.in_proj_bias.device
        n = HIDDEN // 4
        d = TEMPERATURE ** (torch.arange(n, device=device, dtype=torch.float32) / n)
        ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] / d
        xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] / d
        ey = torch.cat([ys.cos(), ys.sin()], -1)[:, None].expand(h, w, -1)
        ex = torch.cat([xs.cos(), xs.sin()], -1)[None].expand(h, w, -1)
        return torch.cat([ey, ex], -1).reshape(1, h * w, -1)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """The two projected maps (strides 16, 32) of the backbone's three."""
        p3, p4, p5 = (proj(f) for proj, f in zip(self.input_proj, feats, strict=True))
        b, c, h, w = p5.shape
        p5 = (
            self.encoder(p5.flatten(2).transpose(1, 2), self.positions(h, w).to(p5.dtype))
            .transpose(1, 2)
            .reshape(b, c, h, w)
        )
        t5 = self.lateral_convs[0](p5)
        t4 = self.lateral_convs[1](self.fpn_blocks[0](torch.cat([F.interpolate(t5, scale_factor=2.0), p4], 1)))
        t3 = self.fpn_blocks[1](torch.cat([F.interpolate(t4, scale_factor=2.0), p3], 1))
        o4 = self.pan_blocks[0](torch.cat([self.downsample_convs[0](t3), t4], 1))
        o5 = self.pan_blocks[1](torch.cat([self.downsample_convs[1](o4), t5], 1))
        return self.projector[0](o4), self.projector[1](o5)


class Head(nn.Module):
    """`RTMOHeadModule`: per map the classification branch (first half of the channels) and the pose branch."""

    def __init__(self, a: Arch) -> None:
        """Layers scaled by the width factor."""
        super().__init__()
        half, vec = int(POSE_VEC * a.widen) // 2, int(POSE_VEC * a.widen)
        pose = POSE_GROUPS * int(a.widen * POSE_CONV)
        levels = range(len(STRIDES))
        self.conv_cls = nn.ModuleList(nn.Sequential(ConvAct(half, half, 3), ConvAct(half, half, 3)) for _ in levels)
        self.conv_pose = nn.ModuleList(
            nn.Sequential(ConvAct(half, pose, 3), *(ConvAct(pose, pose, 3, groups=POSE_GROUPS) for _ in range(3)))
            for _ in levels
        )
        self.out_cls = nn.ModuleList(nn.Conv2d(half, 1, 1) for _ in levels)
        self.out_bbox = nn.ModuleList(nn.Conv2d(pose, 4, 1) for _ in levels)
        self.out_kpt_vis = nn.ModuleList(nn.Conv2d(pose, a.keypoints, 1) for _ in levels)
        self.out_pose = nn.ModuleList(nn.Conv2d(pose, vec, 1) for _ in levels)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Per grid cell (the maps row by row, stride 16 first): `[B, N]`, `[B, N, 4]`, `[B, N, K]`, `[B, N, C]`."""
        outs: list[list[torch.Tensor]] = [[], [], [], []]
        for i, f in enumerate(feats):
            cls, reg = f.chunk(2, 1)
            pose = self.conv_pose[i](reg)
            maps = (self.out_cls[i](self.conv_cls[i](cls)), self.out_bbox[i](pose))
            for out, m in zip(outs, (*maps, self.out_kpt_vis[i](pose), self.out_pose[i](pose)), strict=True):
                out.append(m.flatten(2).transpose(1, 2))
        cls, box, vis, vec = (torch.cat(o, 1) for o in outs)
        return cls[..., 0], box, vis, vec


class DCC(nn.Module):
    """Dynamic coordinate classifier: pose vectors of the kept people -> keypoints (network input pixels)."""

    x_bins: torch.Tensor
    y_bins: torch.Tensor
    dim_t: torch.Tensor

    def __init__(self, a: Arch) -> None:
        """Layers for `a.keypoints` keypoints."""
        super().__init__()
        self.keypoints = a.keypoints
        self.pose_to_kpts = nn.Linear(int(POSE_VEC * a.widen), a.keypoints * KPT_FEAT)
        self.g = nn.Parameter(torch.empty(1))
        self.uv = nn.Linear(KPT_FEAT, 2 * GAU_EXPAND + KPT_FEAT, bias=False)
        self.gamma = nn.Parameter(torch.empty(2, KPT_FEAT))
        self.beta = nn.Parameter(torch.empty(2, KPT_FEAT))
        self.o = nn.Linear(GAU_EXPAND, KPT_FEAT, bias=False)
        self.res_scale = nn.Parameter(torch.empty(KPT_FEAT))
        self.pos_enc = nn.Parameter(torch.empty(a.keypoints, KPT_FEAT))
        self.x_fc = nn.Linear(KPT_FEAT, KPT_FEAT)
        self.y_fc = nn.Linear(KPT_FEAT, KPT_FEAT)
        self.register_buffer("x_bins", torch.empty(BINS[0]))
        self.register_buffer("y_bins", torch.empty(BINS[1]))
        self.register_buffer("dim_t", torch.empty(KPT_FEAT // 2))

    def features(self, vec: torch.Tensor) -> torch.Tensor:
        """Keypoint features `[P, K, 128]` of pose vectors `[P, C]` (the gated attention unit over the keypoints)."""
        x = self.pose_to_kpts(vec).unflatten(-1, (self.keypoints, KPT_FEAT))
        y = x / (x.norm(dim=-1, keepdim=True) * KPT_FEAT**-0.5).clamp(min=NORM_EPS) * self.g
        u, v, base = F.silu(self.uv(y)).split((GAU_EXPAND, GAU_EXPAND, KPT_FEAT), -1)
        q = base * self.gamma[0] + self.beta[0] + self.pos_enc
        k = base * self.gamma[1] + self.beta[1] + self.pos_enc
        kernel = F.relu(q @ k.transpose(-1, -2) / math.sqrt(KPT_FEAT)).square()
        return x * self.res_scale + self.o(u * (kernel @ v))

    def _axis(self, feats: torch.Tensor, fc: nn.Linear, bins: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Expected coordinate `[P, K]` over `bins` `[P, n]` (absolute); `offset` `[P, 1]` is the grid cell's."""
        f = (bins - offset)[..., None] / self.dim_t
        logits = feats @ fc(torch.cat([f.cos(), f.sin()], -1)).transpose(-1, -2)  # [P, K, n]
        logits = logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
        prob = (logits - logits.amax(-1, keepdim=True)).exp()
        prob = prob / (prob.sum(-1, keepdim=True) + SOFTMAX_EPS)
        return (prob * bins[:, None]).sum(-1)

    def forward(self, vec: torch.Tensor, centre: torch.Tensor, scale: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Keypoints `[P, K, 2]` of pose vectors `[P, C]`, box centres, padded box sizes and grid cells `[P, 2]`."""
        feats = self.features(vec)
        x = self._axis(feats, self.x_fc, self.x_bins * scale[:, :1] + centre[:, :1], grid[:, :1])
        y = self._axis(feats, self.y_fc, self.y_bins * scale[:, 1:] + centre[:, 1:], grid[:, 1:])
        return torch.stack([x, y], -1)


class RTMO(nn.Module):
    """Backbone, neck and head: an image batch -> the head's per-cell outputs."""

    def __init__(self, a: Arch) -> None:
        """The three parts."""
        super().__init__()
        self.backbone = Backbone(a)
        self.neck = Neck(a)
        self.head = Head(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """`Head.forward` of an image batch `[B, 3, S, S]` (RGB, 0..255)."""
        return self.head(self.neck(self.backbone(x)))


def _fold(
    w: torch.Tensor, sd: dict[str, torch.Tensor], bn: str, eps: float, own: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(weight, bias) of a conv / linear `w` (its bias `own`) followed by the BatchNorm at `bn`, in double precision."""
    scale = sd[bn + ".weight"].double() / (sd[bn + ".running_var"].double() + eps).sqrt()
    bias = sd[bn + ".bias"].double() - sd[bn + ".running_mean"].double() * scale
    if own is not None:
        bias = bias + own.double() * scale
    return w.double() * scale.view(-1, *([1] * (w.ndim - 1))), bias


def _unfocus(w: torch.Tensor) -> torch.Tensor:
    """The Focus conv `[O, 12, 3, 3]` on the four pixel phases (BGR each) as one `[O, 3, 6, 6]` conv on RGB.

    Phase order of the reference: top-left, bottom-left, top-right, bottom-right.
    """
    out = w.new_zeros(w.shape[0], 3, 6, 6)
    for phase, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        out[:, :, dy::2, dx::2] = w[:, 3 * phase : 3 * phase + 3]
    return out.flip(1)


RENAMES = (  # reference key prefix -> ours (first match)
    ("head.head_module.", "head."),
    ("head.dcc.gau.ln.g", "dcc.g"),
    ("head.dcc.gau.res_scale.scale", "dcc.res_scale"),
    ("head.dcc.gau.", "dcc."),
    ("head.dcc.spe.dim_t", "dcc.dim_t"),
    ("head.dcc.", "dcc."),
    ("neck.encoder.0.layers.0.self_attn.attn.", "neck.encoder."),
    ("neck.encoder.0.layers.0.ffn.layers.0.0.", "neck.encoder.fc1."),
    ("neck.encoder.0.layers.0.ffn.layers.1.", "neck.encoder.fc2."),
    ("neck.encoder.0.layers.0.norms.0.", "neck.encoder.norm1."),
    ("neck.encoder.0.layers.0.norms.1.", "neck.encoder.norm2."),
    ("neck.projector.convs.", "neck.projector."),
)
# Training only, and the neck's sine periods (`Neck.positions` computes them)
DROPPED = ("head.head_module.out_kpt_reg.", "head.loss_oks.", "head.dcc.sigma_fc.", "neck.sincos_pos_enc.")


def convert(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(network, DCC) state dicts of an mmpose RTMO checkpoint: BatchNorms, RepVGG branches and the Focus folded."""
    folded: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.startswith(DROPPED) or ".bn." in key or key.startswith("head.dcc.pose_to_kpts.1."):
            continue
        if key.endswith(".conv.weight") and key[:-11] + "bn.weight" in sd:
            yolox = key.startswith(("backbone.", "head.head_module."))
            w, b = _fold(value, sd, key[:-11] + "bn", BN_EPS_YOLOX if yolox else BN_EPS)
            folded[key], folded[key[:-6] + "bias"] = w, b
        elif key == "head.dcc.pose_to_kpts.0.weight":
            w, b = _fold(value, sd, "head.dcc.pose_to_kpts.1", BN_EPS, sd["head.dcc.pose_to_kpts.0.bias"])
            folded["head.dcc.pose_to_kpts.weight"], folded["head.dcc.pose_to_kpts.bias"] = w, b
        elif key != "head.dcc.pose_to_kpts.0.bias":
            folded[key] = value
    for key in [k for k in folded if ".branch_3x3." in k]:  # RepVGG: the 1 x 1 branch joins the 3 x 3 kernel's centre
        one = folded.pop(key.replace(".branch_3x3.", ".branch_1x1."))
        folded[key.replace(".branch_3x3.", ".")] = folded.pop(key) + (F.pad(one, (1, 1, 1, 1)) if one.ndim > 1 else one)
    stem = "backbone.stem.conv."
    folded[stem + "weight"] = _unfocus(folded.pop(stem + "conv.weight"))
    folded[stem + "bias"] = folded.pop(stem + "conv.bias")
    net: dict[str, torch.Tensor] = {}
    dcc: dict[str, torch.Tensor] = {}
    for key, value in folded.items():
        name = next((new + key[len(old) :] for old, new in RENAMES if key.startswith(old)), key)
        if name.startswith(("neck.input_proj.", "neck.projector.")):
            name = name.replace(".conv.", ".")
        (dcc if name.startswith("dcc.") else net)[name.removeprefix("dcc.")] = value.float()
    return net, dcc
