"""RT-DETRv2 (PResNet-vd backbone, hybrid encoder, deformable-attention decoder) for inference.

State-dict names follow the official checkpoint (`ema.module`), with every BatchNorm folded into the preceding
convolution's bias (`fold_bn`: `X.conv` + `X.norm` -> `X.conv` with a bias) and every RepVGG block merged into one 3x3
convolution. The anchors and the encoder position embedding are computed from the feature sizes (the official model
precomputes them for 640 x 640).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import CheckpointMismatchError
from fastforks.detection._attention import Attention, deformable_sample
from fastforks.detection._fold import fold_bn_into, merge_repvgg
from fastforks.detection._rtdetr import (
    BN_EPS,
    HEADS,
    LN_EPS,
    AifiLayer,
    ConvBn,
    CspRep,
    Mlp,
    anchors,
    inverse_sigmoid,
    sincos_position,
)

STRIDES = (8, 16, 32)  # fixed: feature strides of the three levels
POINTS = 4  # fixed: sampling points per level and head (`num_points: [4, 4, 4]`)
LEVELS = 3
EMB = 256  # decoder width (the encoder's `hidden_dim` is `Arch.enc_dim`)
SILU_MID = 2 * EMB  # fixed: query position head hidden width


@dataclass(frozen=True)
class Arch:
    """Network shape (card `[params]` + variant `params`)."""

    blocks: list[int]  # ResNet blocks per stage
    bottleneck: bool  # bottleneck blocks (50, 101) or basic blocks (18, 34)
    enc_dim: int  # hybrid encoder width
    enc_ffn: int  # AIFI feed-forward width
    expansion: float  # CSP layers' hidden width / output width
    dec_layers: int  # decoder layers in the checkpoint
    eval_layers: int  # decoder layers run at inference (`eval_idx` + 1)
    num_queries: int
    num_classes: int


class Short(nn.Module):
    """Variant-d shortcut of a strided block: 2 x 2 average pooling (ceil mode) then a 1 x 1 convolution."""

    def __init__(self, cin: int, cout: int) -> None:
        """Create the 1 x 1 convolution."""
        super().__init__()
        self.conv = ConvBn(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`conv(avg_pool(x))`."""
        return self.conv(F.avg_pool2d(x, 2, 2, 0, ceil_mode=True))


class Block(nn.Module):
    """ResNet-vd block (stride on the 3x3 of bottlenecks, on the first 3x3 of basic blocks)."""

    def __init__(self, cin: int, width: int, stride: int, *, shortcut: bool, bottleneck: bool) -> None:
        """Create the block; `width` is the stage width (the output has 4x of it for bottlenecks)."""
        super().__init__()
        cout = width * 4 if bottleneck else width
        self.bottleneck = bottleneck
        if bottleneck:
            self.branch2a = ConvBn(cin, width, 1, 1, "relu")
            self.branch2b = ConvBn(width, width, 3, stride, "relu")
            self.branch2c = ConvBn(width, cout, 1)
        else:
            self.branch2a = ConvBn(cin, width, 3, stride, "relu")
            self.branch2b = ConvBn(width, width, 3)
        self.short = None
        if not shortcut:
            self.short = Short(cin, cout) if stride > 1 else ConvBn(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`relu(branches(x) + shortcut(x))`."""
        res = x if self.short is None else self.short(x)
        if self.bottleneck:
            return self.branch2c(self.branch2b(self.branch2a(x)), res)
        return self.branch2b(self.branch2a(x), res)


class Stage(nn.Module):
    """One ResNet stage (`blocks` list, as in the checkpoint)."""

    def __init__(self, cin: int, width: int, n: int, stride: int, *, bottleneck: bool) -> None:
        """Create `n` blocks; the first has the stage stride and a projection shortcut."""
        super().__init__()
        cout = width * 4 if bottleneck else width
        self.blocks = nn.ModuleList(
            Block(cin if i == 0 else cout, width, stride if i == 0 else 1, shortcut=i > 0, bottleneck=bottleneck)
            for i in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks."""
        for b in self.blocks:
            x = b(x)
        return x


class PResNet(nn.Module):
    """ResNet-vd: 3-conv stem, max pool, four stages; returns strides 8, 16, 32."""

    def __init__(self, a: Arch) -> None:
        """Create the stem and stages."""
        super().__init__()
        self.conv1 = nn.Module()
        self.conv1.conv1_1 = ConvBn(3, 32, 3, 2, "relu")
        self.conv1.conv1_2 = ConvBn(32, 32, 3, 1, "relu")
        self.conv1.conv1_3 = ConvBn(32, 64, 3, 1, "relu")
        self.res_layers = nn.ModuleList()
        cin = 64
        for i, n in enumerate(a.blocks):
            width = 64 << i
            self.res_layers.append(Stage(cin, width, n, 1 if i == 0 else 2, bottleneck=a.bottleneck))
            cin = width * 4 if a.bottleneck else width

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Image -> stride 8, 16, 32 features."""
        x = self.conv1.conv1_3(self.conv1.conv1_2(self.conv1.conv1_1(x)))  # ty: ignore[call-non-callable]
        x = F.max_pool2d(x, 3, 2, 1)
        outs = []
        for i, stage in enumerate(self.res_layers):
            x = stage(x)
            if i > 0:
                outs.append(x)
        return outs


class HybridEncoder(nn.Module):
    """Projection, one AIFI layer on the stride-32 map, top-down FPN and bottom-up PAN of CSP-RepVGG blocks."""

    def __init__(self, a: Arch, in_channels: list[int]) -> None:
        """Create the encoder for backbone outputs with `in_channels`."""
        super().__init__()
        d = a.enc_dim
        self.input_proj = nn.ModuleList(ConvBn(c, d, 1) for c in in_channels)
        self.encoder = nn.ModuleList([nn.Module()])
        self.encoder[0].layers = nn.ModuleList([AifiLayer(d, a.enc_ffn)])
        self.lateral_convs = nn.ModuleList(ConvBn(d, d, 1, 1, "silu") for _ in range(2))
        self.fpn_blocks = nn.ModuleList(CspRep(2 * d, d, a.expansion) for _ in range(2))
        self.downsample_convs = nn.ModuleList(ConvBn(d, d, 3, 2, "silu") for _ in range(2))
        self.pan_blocks = nn.ModuleList(CspRep(2 * d, d, a.expansion) for _ in range(2))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Backbone features (strides 8, 16, 32) -> three `enc_dim` maps."""
        proj = [p(f) for p, f in zip(self.input_proj, feats, strict=True)]
        b, d, h, w = proj[2].shape
        pos = sincos_position(h, w, d, proj[2].device).to(proj[2].dtype)
        mem = proj[2].flatten(2).transpose(1, 2)
        for layer in self.encoder[0].layers:  # ty: ignore[not-iterable]
            mem = layer(mem, pos)  # ty: ignore[call-non-callable]
        proj[2] = mem.transpose(1, 2).reshape(b, d, h, w)
        inner = [proj[2]]
        for idx in (2, 1):
            high = self.lateral_convs[2 - idx](inner[0])
            inner[0] = high
            up = F.interpolate(high, scale_factor=2.0, mode="nearest")
            inner.insert(0, self.fpn_blocks[2 - idx](torch.cat([up, proj[idx - 1]], dim=1)))
        outs = [inner[0]]
        for i in range(2):
            down = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.cat([down, inner[i + 1]], dim=1)))
        return outs


class DeformableAttention(nn.Module):
    """`MSDeformableAttention` (default method): learned sampling offsets around 4-d reference boxes."""

    def __init__(self) -> None:
        """Create the projections and the per-point offset scale (a checkpoint buffer)."""
        super().__init__()
        n = LEVELS * POINTS
        self.sampling_offsets = nn.Linear(EMB, HEADS * n * 2)
        self.attention_weights = nn.Linear(EMB, HEADS * n)
        self.value_proj = nn.Linear(EMB, EMB)
        self.output_proj = nn.Linear(EMB, EMB)
        self.register_buffer("num_points_scale", torch.full((n,), 1.0 / POINTS))

    def forward(
        self, query: torch.Tensor, ref: torch.Tensor, value: torch.Tensor, shapes: list[tuple[int, int]]
    ) -> torch.Tensor:
        """`[B, Q, d]` queries with `[B, Q, 1, 4]` reference boxes over the flattened `[B, S, d]` value."""
        b, q = query.shape[:2]
        n = LEVELS * POINTS
        value = self.value_proj(value).reshape(b, -1, HEADS, EMB // HEADS)
        offsets = self.sampling_offsets(query).reshape(b, q, HEADS, n, 2)
        weights = F.softmax(self.attention_weights(query).reshape(b, q, HEADS, n), dim=-1)
        scale = self.num_points_scale.to(query.dtype).unsqueeze(-1)  # ty: ignore[call-non-callable]
        # reference boxes (cx, cy, w, h) in 0..1; the offset is a fraction of the box size
        loc = ref[:, :, None, :, :2] + offsets * scale * ref[:, :, None, :, 2:] * 0.5
        out = deformable_sample(value, loc, weights, shapes, POINTS)
        return self.output_proj(out)


class DecoderLayer(nn.Module):
    """Post-norm decoder layer: self-attention, deformable cross-attention over the memory, FFN."""

    def __init__(self) -> None:
        """Create the layer."""
        super().__init__()
        self.self_attn = Attention(EMB, HEADS)
        self.cross_attn = DeformableAttention()
        self.linear1, self.linear2 = nn.Linear(EMB, 1024), nn.Linear(1024, EMB)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(EMB, LN_EPS) for _ in range(3))

    def forward(
        self, x: torch.Tensor, ref: torch.Tensor, mem: torch.Tensor, shapes: list[tuple[int, int]], qpos: torch.Tensor
    ) -> torch.Tensor:
        """`[B, Q, d]` -> `[B, Q, d]`."""
        qk = x + qpos
        x = self.norm1(x + self.self_attn(qk, qk, x))
        x = self.norm2(x + self.cross_attn(x + qpos, ref, mem, shapes))
        return self.norm3(x + self.linear2(F.relu(self.linear1(x))))


class Decoder(nn.Module):
    """Input projection, query selection from the encoder output, and the iterative box refinement."""

    def __init__(self, a: Arch) -> None:
        """Create the projections, heads and layers."""
        super().__init__()
        self.input_proj = nn.ModuleList(ConvBn(a.enc_dim, EMB, 1) for _ in range(LEVELS))
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(DecoderLayer() for _ in range(a.dec_layers))
        self.query_pos_head = Mlp(4, SILU_MID, EMB, 2)
        self.enc_output = nn.Module()
        self.enc_output.proj, self.enc_output.norm = nn.Linear(EMB, EMB), nn.LayerNorm(EMB, LN_EPS)
        self.enc_score_head = nn.Linear(EMB, a.num_classes)
        self.enc_bbox_head = Mlp(EMB, EMB, 4, 3)
        self.dec_score_head = nn.ModuleList(nn.Linear(EMB, a.num_classes) for _ in range(a.dec_layers))
        self.dec_bbox_head = nn.ModuleList(Mlp(EMB, EMB, 4, 3) for _ in range(a.dec_layers))
        self.queries, self.eval_layers = a.num_queries, a.eval_layers

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder maps -> class logits `[B, Q, classes]` and boxes `[B, Q, 4]` (centre x / y, width, height; 0..1)."""
        proj = [p(f) for p, f in zip(self.input_proj, feats, strict=True)]
        shapes = [(int(f.shape[2]), int(f.shape[3])) for f in proj]
        mem = torch.cat([f.flatten(2).transpose(1, 2) for f in proj], dim=1)
        anc, valid = anchors(shapes, mem.device)
        # query selection in float32 (the ranking decides which memory slots the decoder sees)
        sel = (valid.to(mem.dtype) * mem).float()
        out = self.enc_output.norm(self.enc_output.proj(sel))  # ty: ignore[call-non-callable]
        logits = self.enc_score_head(out)
        coords = self.enc_bbox_head(out) + anc
        # fewer memory slots than queries below ~ 250 px
        idx = logits.max(-1).values.topk(min(self.queries, mem.shape[1]), dim=-1).indices
        pick = idx.unsqueeze(-1)
        ref = F.sigmoid(coords.gather(1, pick.expand(-1, -1, 4)))
        tgt = out.gather(1, pick.expand(-1, -1, EMB)).to(mem.dtype)
        for i, layer in enumerate(self.decoder.layers):  # ty: ignore[invalid-argument-type]
            qpos = self.query_pos_head(ref).to(mem.dtype)
            tgt = layer(tgt, ref.to(mem.dtype).unsqueeze(2), mem, shapes, qpos)
            boxes = F.sigmoid(self.dec_bbox_head[i](tgt.float()) + inverse_sigmoid(ref))
            if i == self.eval_layers - 1:
                return self.dec_score_head[i](tgt.float()), boxes
            ref = boxes
        msg = "eval_layers exceeds the decoder depth"
        raise ValueError(msg)


class RtDetr(nn.Module):
    """Image `[B, 3, h, w]` in 0..1 -> class logits `[B, 300, 80]` and boxes `[B, 300, 4]` (centre x / y, w, h)."""

    def __init__(self, a: Arch) -> None:
        """Create backbone, encoder and decoder."""
        super().__init__()
        self.backbone = PResNet(a)
        width = [128, 256, 512] if not a.bottleneck else [512, 1024, 2048]
        self.encoder = HybridEncoder(a, width)
        self.decoder = Decoder(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The network."""
        return self.decoder(self.encoder(self.backbone(x)))


def fold_bn(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint state dict -> the same for `RtDetr`: BatchNorm folded into the convs, RepVGG branches merged.

    `y = (conv(x) - mean) * w / sqrt(var + eps) + b` (computed in float64); the 3x3 and 1x1 RepVGG branches add up to
    one 3x3 convolution. Tensors that do not fit are left as they are for the strict load to report. The precomputed
    `anchors` / `valid_mask` buffers are dropped (computed from the sizes).
    """
    bns = [k.removesuffix(".running_var") for k in sd if k.endswith(".running_var")]
    out = {
        k: v
        for k, v in sd.items()
        if k.rsplit(".", 1)[0] not in bns and not k.endswith(("anchors", "valid_mask")) and "denoising" not in k
    }
    for bn in bns:
        conv = f"{bn.removesuffix('norm')}conv"
        if bn.rsplit(".", 1)[-1] != "norm":
            what, details = "rtdetrv2 checkpoint", f"BatchNorm {bn} has no convolution {conv}"
            raise CheckpointMismatchError(what, details)
        out.update(fold_bn_into(sd, bn, conv, BN_EPS))
    return merge_repvgg(out)
