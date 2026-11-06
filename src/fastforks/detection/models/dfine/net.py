"""D-FINE (HGNetV2 backbone, hybrid encoder, deformable decoder with fine-grained distribution refinement).

State-dict names follow the Hugging Face checkpoint (`ustc-community/dfine-*`) after `convert`: every BatchNorm folded
into the preceding convolution's bias, every RepVGG block merged into one 3x3 convolution, the separate q / k / v
projections joined, and only the heads the last decoder layer uses kept. The anchors and the AIFI position embedding
are computed from the feature sizes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import CheckpointMismatchError
from fastforks.detection._attention import Attention, sample_maps
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

BINS = 32  # fixed: `max_num_bins` (the distribution has BINS + 1 values per box side)
UP = 0.5  # fixed: `up`
OFFSET_SCALE = 0.5  # fixed: `decoder_offset_scale`
TOP_PROBS = 4  # fixed: `top_prob_values` of the location quality estimator
LQE_HIDDEN = 64  # fixed: `lqe_hidden_dim`
KERNELS = (3, 3, 5, 5)  # fixed: stage kernel sizes
LIGHT = (False, False, True, True)  # fixed: stages built from light blocks (1x1 + depthwise)
CLAMP = 65504.0  # fixed: float16 range, applied before the decoder layer's last norm
STRIDES = (8, 16, 32)  # fixed: the last `levels` of these are the feature strides


@dataclass(frozen=True)
class Arch:
    """Network shape (card `[params]` + variant `params`)."""

    stem: list[int]  # the two stem widths
    mid: list[int]  # per stage: width of the block layers
    out: list[int]  # per stage: output width
    blocks: list[int]  # per stage: blocks
    layers: int  # layers per block
    lab: bool  # learnable affine block after every activation
    levels: int  # feature levels used (the last ones of the backbone)
    enc_dim: int  # hybrid encoder width
    dec_dim: int  # decoder width
    enc_ffn: int  # AIFI feed-forward width
    dec_ffn: int  # decoder feed-forward width
    csp_blocks: int  # RepVGG blocks per CSP layer
    expansion: float  # width of the CSP branches relative to the encoder
    dec_layers: int
    points: list[int]  # deformable sampling points per level
    num_queries: int
    num_classes: int
    reg_scale: float  # curvature of the bin positions and the edge distance unit (checkpoint's `reg_scale`)


class Lab(nn.Module):
    """Learnable affine block: `scale * x + bias` (one scalar each)."""

    def __init__(self) -> None:
        """Create the parameters."""
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`scale * x + bias`."""
        return torch.addcmul(self.bias, x, self.scale)


class HgConv(ConvBn):
    """Convolution (+ folded BatchNorm), ReLU and the optional affine block (`HGNetV2ConvLayer`)."""

    def __init__(  # noqa: PLR0913 - a plain convolution's arguments
        self, cin: int, cout: int, k: int, stride: int = 1, groups: int = 1, *, lab: bool = False
    ) -> None:
        """Create the convolution."""
        super().__init__(cin, cout, k, stride, "relu", groups)
        self.lab = Lab() if lab else None

    def forward(self, x: torch.Tensor, res: torch.Tensor | None = None) -> torch.Tensor:
        """`lab(relu(conv(x)))`."""
        y = super().forward(x, res)
        return y if self.lab is None else self.lab(y)


class Light(nn.Module):
    """Light layer: 1x1 convolution without activation, then a depthwise `k` x `k` with ReLU."""

    def __init__(self, cin: int, cout: int, k: int, *, lab: bool) -> None:
        """Create the two convolutions."""
        super().__init__()
        self.conv1 = ConvBn(cin, cout, 1)
        self.conv2 = HgConv(cout, cout, k, groups=cout, lab=lab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`conv2(conv1(x))`."""
        return self.conv2(self.conv1(x))


class HgBlock(nn.Module):
    """`HGNetV2BasicLayer`: chained layers, all outputs concatenated and squeezed / excited, optional residual."""

    def __init__(self, a: Arch, i: int, j: int) -> None:
        """Create block `j` of stage `i`."""
        super().__init__()
        cin = [a.stem[1], *a.out][i] if j == 0 else a.out[i]
        mid, cout, k, lab = a.mid[i], a.out[i], KERNELS[i], a.lab
        self.residual = j > 0
        self.layers = nn.ModuleList(
            Light(cin if n == 0 else mid, mid, k, lab=lab)
            if LIGHT[i]
            else HgConv(cin if n == 0 else mid, mid, k, lab=lab)
            for n in range(a.layers)
        )
        self.aggregation = nn.Sequential(
            HgConv(cin + a.layers * mid, cout // 2, 1, lab=lab), HgConv(cout // 2, cout, 1, lab=lab)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""
        outs = [x]
        y = x
        for layer in self.layers:
            y = layer(y)
            outs.append(y)
        y = self.aggregation(torch.cat(outs, dim=1))
        return y + x if self.residual else y


class HgStage(nn.Module):
    """Depthwise stride-2 downsampling (not in the first stage) and the blocks."""

    def __init__(self, a: Arch, i: int) -> None:
        """Create stage `i`."""
        super().__init__()
        cin = [a.stem[1], *a.out][i]
        self.downsample = ConvBn(cin, cin, 3, 2, groups=cin) if i > 0 else None
        self.blocks = nn.ModuleList(HgBlock(a, i, j) for j in range(a.blocks[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stage."""
        if self.downsample is not None:
            x = self.downsample(x)
        for b in self.blocks:
            x = b(x)
        return x


class Embedder(nn.Module):
    """Stem: 3x3 stride 2, then a 2x2 conv branch and a max pool branch, joined by a 3x3 stride 2 and a 1x1."""

    def __init__(self, a: Arch) -> None:
        """Create the stem convolutions."""
        super().__init__()
        c1, c2 = a.stem
        self.stem1 = HgConv(3, c1, 3, 2, lab=a.lab)
        self.stem2a = HgConv(c1, c1 // 2, 2, lab=a.lab)
        self.stem2b = HgConv(c1 // 2, c1, 2, lab=a.lab)
        self.stem3 = HgConv(2 * c1, c1, 3, 2, lab=a.lab)
        self.stem4 = HgConv(c1, c2, 1, lab=a.lab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> stride-4 features."""
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        y = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        x = torch.cat([F.max_pool2d(x, 2, 1, 0, ceil_mode=True), y], dim=1)
        return self.stem4(self.stem3(x))


class HgNet(nn.Module):
    """HGNetV2: stem and four stages; returns the last `levels` stage outputs (strides 8 / 16 / 32)."""

    def __init__(self, a: Arch) -> None:
        """Create the stem and the stages."""
        super().__init__()
        self.levels = a.levels
        self.embedder = Embedder(a)
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList(HgStage(a, i) for i in range(4))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Image -> the last `levels` feature maps."""
        x = self.embedder(x)
        outs = []
        for stage in self.encoder.stages:  # ty: ignore[not-iterable]
            x = stage(x)  # ty: ignore[call-non-callable]
            outs.append(x)
        return outs[-self.levels :]


class RepNcspElan4(nn.Module):
    """`RepNCSPELAN4`: split, two CSP-RepVGG + 3x3 steps, concatenate everything, 1x1."""

    def __init__(self, d: int, a: Arch) -> None:
        """Create the block for encoder width `d` (input `2 * d`)."""
        super().__init__()
        c = round(a.expansion * d // 2)
        self.conv1 = ConvBn(2 * d, 2 * d, 1, 1, "silu")
        self.csp_rep1 = CspRep(d, c, 1.0, a.csp_blocks)
        self.conv2 = ConvBn(c, c, 3, 1, "silu")
        self.csp_rep2 = CspRep(c, c, 1.0, a.csp_blocks)
        self.conv3 = ConvBn(c, c, 3, 1, "silu")
        self.conv4 = ConvBn(2 * d + 2 * c, d, 1, 1, "silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 2d, H, W]` -> `[B, d, H, W]`."""
        a, b = self.conv1(x).chunk(2, dim=1)
        c = self.conv2(self.csp_rep1(b))
        e = self.conv3(self.csp_rep2(c))
        return self.conv4(torch.cat([a, b, c, e], dim=1))


class ScDown(nn.Module):
    """1x1 then depthwise 3x3 stride 2."""

    def __init__(self, d: int) -> None:
        """Create the convolutions."""
        super().__init__()
        self.conv1 = ConvBn(d, d, 1)
        self.conv2 = ConvBn(d, d, 3, 2, groups=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`conv2(conv1(x))`."""
        return self.conv2(self.conv1(x))


class HybridEncoder(nn.Module):
    """One AIFI layer on the last map, top-down FPN and bottom-up PAN of `RepNCSPELAN4` blocks."""

    def __init__(self, a: Arch) -> None:
        """Create the encoder."""
        super().__init__()
        d, n = a.enc_dim, a.levels - 1
        self.aifi = AifiLayer(d, a.enc_ffn)
        self.lateral_convs = nn.ModuleList(ConvBn(d, d, 1) for _ in range(n))
        self.fpn_blocks = nn.ModuleList(RepNcspElan4(d, a) for _ in range(n))
        self.downsample_convs = nn.ModuleList(ScDown(d) for _ in range(n))
        self.pan_blocks = nn.ModuleList(RepNcspElan4(d, a) for _ in range(n))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Projected backbone maps -> `levels` maps of the encoder width."""
        feats = list(feats)
        b, d, h, w = feats[-1].shape
        pos = sincos_position(h, w, d, feats[-1].device).to(feats[-1].dtype)
        mem = self.aifi(feats[-1].flatten(2).transpose(1, 2), pos)
        feats[-1] = mem.transpose(1, 2).reshape(b, d, h, w)
        inner = [feats[-1]]
        for i, (lat, blk) in enumerate(zip(self.lateral_convs, self.fpn_blocks, strict=True)):
            top = lat(inner[0])
            inner[0] = top
            up = F.interpolate(top, scale_factor=2.0, mode="nearest")
            inner.insert(0, blk(torch.cat([up, feats[len(feats) - 2 - i]], dim=1)))
        outs = [inner[0]]
        for i, (down, blk) in enumerate(zip(self.downsample_convs, self.pan_blocks, strict=True)):
            outs.append(blk(torch.cat([down(outs[-1]), inner[i + 1]], dim=1)))
        return outs


class DeformableAttention(nn.Module):
    """D-FINE's deformable attention: no value / output projection, a different number of points per level."""

    def __init__(self, d: int, points: list[int]) -> None:
        """Create the offset and weight projections and the per-point offset scale (a checkpoint buffer)."""
        super().__init__()
        self.points = points
        n = sum(points)
        self.sampling_offsets = nn.Linear(d, HEADS * n * 2)
        self.attention_weights = nn.Linear(d, HEADS * n)
        self.register_buffer("num_points_scale", torch.tensor([1.0 / p for p in points for _ in range(p)]))

    def forward(self, query: torch.Tensor, ref: torch.Tensor, maps: list[torch.Tensor]) -> torch.Tensor:
        """`[B, Q, d]` queries with `[B, Q, 1, 4]` reference boxes over the memory maps `[B * heads, dh, h, w]`."""
        b, q, _ = query.shape
        n = sum(self.points)
        offsets = self.sampling_offsets(query).reshape(b, q, HEADS, n, 2)
        weights = F.softmax(self.attention_weights(query).reshape(b, q, HEADS, n), dim=-1)
        scale = self.num_points_scale.to(query.dtype).unsqueeze(-1)  # ty: ignore[call-non-callable]
        loc = ref[:, :, None, :, :2] + offsets * scale * ref[:, :, None, :, 2:] * OFFSET_SCALE
        return sample_maps(maps, loc, weights, self.points)


class Gate(nn.Module):
    """`norm(g1 * residual + g2 * update)` with `g1, g2 = sigmoid(gate([residual, update]))`."""

    def __init__(self, d: int) -> None:
        """Create the gate."""
        super().__init__()
        self.gate = nn.Linear(2 * d, 2 * d)
        self.norm = nn.LayerNorm(d, LN_EPS)

    def forward(self, res: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gated sum."""
        g1, g2 = torch.sigmoid(self.gate(torch.cat([res, x], dim=-1))).chunk(2, dim=-1)
        return self.norm(g1 * res + g2 * x)


class DecoderLayer(nn.Module):
    """Self-attention, deformable cross-attention with a gate, ReLU feed-forward (all post-norm)."""

    def __init__(self, a: Arch) -> None:
        """Create the layer."""
        super().__init__()
        d = a.dec_dim
        self.self_attn = Attention(d, HEADS)
        self.self_attn_layer_norm = nn.LayerNorm(d, LN_EPS)
        self.encoder_attn = DeformableAttention(d, a.points)
        self.gateway = Gate(d)
        self.fc1, self.fc2 = nn.Linear(d, a.dec_ffn), nn.Linear(a.dec_ffn, d)
        self.final_layer_norm = nn.LayerNorm(d, LN_EPS)

    def forward(self, x: torch.Tensor, ref: torch.Tensor, maps: list[torch.Tensor], qpos: torch.Tensor) -> torch.Tensor:
        """`[B, Q, d]` -> `[B, Q, d]`."""
        qk = x + qpos
        x = self.self_attn_layer_norm(x + self.self_attn(qk, qk, x))
        x = self.gateway(x, self.encoder_attn(x + qpos, ref, maps))
        return self.final_layer_norm((x + self.fc2(F.relu(self.fc1(x)))).clamp(-CLAMP, CLAMP))


def weighting_function(reg_scale: float) -> torch.Tensor:
    """The non-uniform bin positions `W(n)` (BINS + 1 values from `-2 * UP * reg_scale` to `2 * UP * reg_scale`)."""
    step = (UP * reg_scale + 1) ** (2 / (BINS - 2))
    left = [-(step**i) + 1 for i in range(BINS // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, BINS // 2)]
    return torch.tensor([-2 * UP * reg_scale, *left, 0.0, *right, 2 * UP * reg_scale], dtype=torch.float32)


def distance2bbox(ref: torch.Tensor, dist: torch.Tensor, reg_scale: float) -> torch.Tensor:
    """Reference boxes (centre x / y, width, height) and edge distances (left, top, right, bottom) -> boxes (same)."""
    wh = ref[..., 2:] / reg_scale
    tl = ref[..., :2] - (0.5 * reg_scale + dist[..., :2]) * wh
    br = ref[..., :2] + (0.5 * reg_scale + dist[..., 2:]) * wh
    return torch.cat([(tl + br) / 2, br - tl], dim=-1)


class Decoder(nn.Module):
    """Decoder layers with iterative box refinement as distributions over edge distances (FDR) and the final heads."""

    def __init__(self, a: Arch) -> None:
        """Create the layers and heads (the last layer's class head and quality estimator only)."""
        super().__init__()
        d = a.dec_dim
        self.layers = nn.ModuleList(DecoderLayer(a) for _ in range(a.dec_layers))
        self.query_pos_head = Mlp(4, 2 * d, d, 2)
        self.pre_bbox_head = Mlp(d, d, 4, 3)
        self.bbox_embed = nn.ModuleList(Mlp(d, d, 4 * (BINS + 1), 3) for _ in range(a.dec_layers))
        self.class_embed = nn.Linear(d, a.num_classes)
        self.lqe = Mlp(4 * (TOP_PROBS + 1), LQE_HIDDEN, 1, 2)
        self.reg_scale = a.reg_scale
        self.project: torch.Tensor  # float32 in use: the module may be half
        self.register_buffer("project", weighting_function(a.reg_scale), persistent=False)

    def forward(
        self, tgt: torch.Tensor, ref: torch.Tensor, maps: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Queries `[B, Q, d]`, reference boxes `[B, Q, 4]`, memory maps -> class logits `[B, Q, classes]`, boxes."""
        prev = corners = 0.0
        ref_init = ref
        h = prob = ref  # set by the loop (at least one layer)
        for i, layer in enumerate(self.layers):
            qpos = self.query_pos_head(ref).clamp(-10, 10).to(tgt.dtype)
            tgt = layer(tgt, ref.to(tgt.dtype).unsqueeze(2), maps, qpos)
            h = tgt.float()
            if i == 0:
                ref_init = F.sigmoid(self.pre_bbox_head(h) + inverse_sigmoid(ref))
            corners = self.bbox_embed[i](h + prev) + corners
            prob = F.softmax(corners.unflatten(-1, (4, BINS + 1)), dim=-1)
            ref = distance2bbox(ref_init, prob @ self.project.float(), self.reg_scale)
            prev = h
        top = prob.topk(TOP_PROBS, dim=-1).values
        stat = torch.cat([top, top.mean(-1, keepdim=True)], dim=-1).flatten(-2)
        return self.class_embed(h) + self.lqe(stat), ref


class DFine(nn.Module):
    """Image `[B, 3, h, w]` in 0..1 -> class logits `[B, 300, 80]` and boxes `[B, 300, 4]` (centre x / y, w, h)."""

    def __init__(self, a: Arch) -> None:
        """Create backbone, encoder, query selection and decoder."""
        super().__init__()
        d = a.dec_dim
        self.backbone = HgNet(a)
        self.encoder_input_proj = nn.ModuleList(ConvBn(c, a.enc_dim, 1) for c in a.out[-a.levels :])
        self.encoder = HybridEncoder(a)
        self.decoder_input_proj = nn.ModuleList(ConvBn(a.enc_dim, d, 1) for _ in range(a.levels) if a.enc_dim != d)
        self.enc_output = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d, LN_EPS))
        self.enc_score_head = nn.Linear(d, a.num_classes)
        self.enc_bbox_head = Mlp(d, d, 4, 3)
        self.decoder = Decoder(a)
        self.queries, self.dim = a.num_queries, d

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Image -> class logits and boxes (centre x / y, width, height; 0..1)."""
        feats = [p(f) for p, f in zip(self.encoder_input_proj, self.backbone(x), strict=True)]
        feats = self.encoder(feats)
        if len(self.decoder_input_proj):
            feats = [p(f) for p, f in zip(self.decoder_input_proj, feats, strict=True)]
        shapes = [(int(f.shape[2]), int(f.shape[3])) for f in feats]
        mem = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], dim=1)
        anc, valid = anchors(shapes, mem.device)
        # query selection in float32 (the ranking decides which memory slots the decoder sees)
        out = self.enc_output((valid.to(mem.dtype) * mem).float())
        coords = self.enc_bbox_head(out) + anc
        idx = self.enc_score_head(out).max(-1).values.topk(min(self.queries, mem.shape[1]), dim=-1).indices
        pick = idx.unsqueeze(-1)
        ref = F.sigmoid(coords.gather(1, pick.expand(-1, -1, 4)))
        tgt = out.gather(1, pick.expand(-1, -1, self.dim)).to(mem.dtype)
        # the decoder samples the maps themselves (a split of `mem` gave NaN in TensorRT 11.4 fp16 engines)
        maps = [f.reshape(f.shape[0] * HEADS, -1, *f.shape[2:]) for f in feats]
        return self.decoder(tgt, ref, maps)


_RENAME = (
    (r"^model\.", ""),
    (r"^backbone\.model\.", "backbone."),
    (r"\.convolution\.", ".conv."),
    (r"\.normalization\.", ".norm."),
    (r"^((?:encoder|decoder)_input_proj\.\d+)\.0\.", r"\1.conv."),
    (r"^((?:encoder|decoder)_input_proj\.\d+)\.1\.", r"\1.norm."),
    (r"^encoder\.encoder\.0\.layers\.0\.fc1\.", "encoder.aifi.linear1."),
    (r"^encoder\.encoder\.0\.layers\.0\.fc2\.", "encoder.aifi.linear2."),
    (r"^encoder\.encoder\.0\.layers\.0\.self_attn_layer_norm\.", "encoder.aifi.norm1."),
    (r"^encoder\.encoder\.0\.layers\.0\.final_layer_norm\.", "encoder.aifi.norm2."),
    (r"^encoder\.encoder\.0\.layers\.0\.", "encoder.aifi."),
)


def _rename(key: str, last: int) -> str | None:
    """Checkpoint key -> `DFine` key (None: not used at inference)."""
    for pat, rep in _RENAME:
        key = re.sub(pat, rep, key)
    if m := re.match(r"^decoder\.(class_embed|lqe_layers)\.(\d+)\.(.*)$", key):
        if int(m[2]) != last:
            return None
        return (
            f"decoder.class_embed.{m[3]}" if m[1] == "class_embed" else f"decoder.lqe.{m[3].removeprefix('reg_conf.')}"
        )
    if key.endswith("num_batches_tracked") or key.startswith("denoising") or key in {"decoder.reg_scale", "decoder.up"}:
        return None
    return key


def convert(sd: dict[str, torch.Tensor], layers: int) -> dict[str, torch.Tensor]:
    """Checkpoint state dict -> the same for `DFine` (BatchNorm folded, RepVGG merged, q / k / v joined).

    Of the per-layer class heads and quality estimators only the last layer's are kept (the others, the denoising
    embedding and the constants `reg_scale` / `up` are not used at inference). Tensors that do not fit are left as
    they are for the strict load to report.
    """
    out = {n: v for k, v in sd.items() if (n := _rename(k, layers - 1)) is not None}
    bns = [k.removesuffix(".running_var") for k in out if k.endswith(".running_var")]
    for bn in bns:
        if not bn.endswith(".norm"):
            what, details = "d-fine checkpoint", f"BatchNorm {bn} is not the '.norm' of a convolution"
            raise CheckpointMismatchError(what, details)
    folded = {k: v for bn in bns for k, v in fold_bn_into(out, bn, f"{bn.removesuffix('norm')}conv", BN_EPS).items()}
    out = {k: v for k, v in out.items() if k.rsplit(".", 1)[0] not in bns} | folded
    for k in [k for k in out if k.endswith("self_attn.q_proj.weight")]:
        p = k.removesuffix("q_proj.weight")
        for s in ("weight", "bias"):
            names = [f"{p}{n}_proj.{s}" for n in "qkv"]
            if all(n in out for n in names):  # else the strict load reports it
                out[f"{p}in_proj_{s}"] = torch.cat([out.pop(n) for n in names])
    return merge_repvgg(out)
