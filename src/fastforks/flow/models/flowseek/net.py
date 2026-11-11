"""FlowSeek network (Poggi & Tosi, ICCV 2025) as mattpoggi/flowseek builds it (`core/flowseek.py`), size T.

Inference only.

Ported from mattpoggi/flowseek (Apache-2.0, see `LICENSE`; its SEA-RAFT parts are BSD-3-Clause, see
`LICENSE-BSD-3-Clause`) with the same module tree, so the released checkpoints load as they are (minus the batch-norm
tensors upstream stores twice, see `BasicBlock`). The Depth Anything V2 network is `fastforks.shared`'s. Changes: no
torchvision / huggingface_hub import and no Depth Anything checkpoint at construction (training-time
initialisation; the released checkpoints contain every weight); no `.cuda()` (runs on the input's device); the input
is RGB in [0, 1] instead of [0, 255]; the flow is upsampled (and the upsampling weights predicted) after the last
iteration only, which is upstream's `final`; the uncertainty (`info`) channels of the flow head are not returned;
with `fp16=True` the networks run under fp16 autocast while the motion bases, the correlation and its lookup stay
float32.

- Depth Anything V2 (Small, frozen) on each image resized to 518 x 518: the last DPT fusion feature (`path_1`) and
  the relative depth. Three stride-2 convs (`merge_head`) turn the feature, resized to the image size, into 1/8
  features.
- Motion bases: 8 flow fields (16 channels) a camera motion would cause, from the depth of image 1: 3 translations
  (scaled by the inverse depth) and 5 rotation terms, each divided by its norm. `bnet` encodes them.
- SEA-RAFT (see the SEA-RAFT family) with twice as wide hidden state and context: the context encoder's and
  `bnet`'s outputs are concatenated; the feature encoder's maps are concatenated with the Depth Anything features
  before the correlation.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.depth_anything_v2 import DepthAnythingV2

FACTOR = 8  # fixed: the encoders downsample by 8 (three stride-2 stages)
DA_SIZE = 518  # fixed: upstream resizes every image to 518 x 518 for Depth Anything
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Config:
    """Architecture of one FlowSeek size (the fields of upstream's `config/eval/*.json` that shape the network)."""

    blocks: tuple[int, int, int]  # BasicBlocks per stage: resnet18 (2, 2, 2)
    block_dims: tuple[int, int, int]
    initial_dim: int
    dim: int  # hidden state and context channels of each of the two branches (image pair, motion bases)
    radius: int  # correlation lookup radius
    levels: int  # correlation pyramid levels
    num_blocks: int  # ConvNeXt blocks in the update block
    da_features: int  # Depth Anything V2 DPT head width
    da_channels: tuple[int, int, int, int]


# fixed: upstream config/eval/flowseek-T.json and -S.json (resnet18, da_size vits; S = T with 12 iterations)
TINY = Config(
    blocks=(2, 2, 2),
    block_dims=(64, 128, 256),
    initial_dim=64,
    dim=128,
    radius=4,
    levels=4,
    num_blocks=2,
    da_features=64,
    da_channels=(48, 96, 192, 384),
)


def _conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0)


def _conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    """Two 3 x 3 conv-bn-relu with a residual (1 x 1 conv-bn projection when the shape changes) and a final relu."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        """Build the block."""
        super().__init__()
        self.conv1 = _conv3x3(cin, cout, stride)
        self.conv2 = _conv3x3(cout, cout)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or cin != cout:
            # upstream: downsample = Sequential(conv1x1, bn3) with bn3 shared, so its checkpoints hold the same
            # tensors as `bn3.*` and `downsample.1.*`; here the projection conv sits alone at `downsample.0`
            self.bn3 = nn.BatchNorm2d(cout)
            self.downsample = nn.Sequential(_conv1x1(cin, cout, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.bn3(self.downsample(x))
        return self.relu(x + y)


class ResNetFPN(nn.Module):
    """7 x 7 stride-2 stem (conv-bn-relu), three ResNet stages (strides 1, 2, 2), 1 x 1 output conv: stride 8."""

    def __init__(self, cfg: Config, cin: int, cout: int) -> None:
        """Build the encoder."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cfg.initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(cfg.initial_dim)
        self.relu = nn.ReLU(inplace=True)
        dims = (cfg.initial_dim, *cfg.block_dims)
        self.layer1, self.layer2, self.layer3 = (
            nn.Sequential(*(BasicBlock(dims[i] if j == 0 else dims[i + 1], dims[i + 1], s if j == 0 else 1)
                            for j in range(n)))
            for i, (n, s) in enumerate(zip(cfg.blocks, (1, 2, 2), strict=True))
        )  # fmt: skip
        self.final_conv = _conv1x1(cfg.block_dims[2], cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, cin, H, W]` -> `[B, cout, H / 8, W / 8]`."""
        x = self.relu(self.bn1(self.conv1(x)))
        return self.final_conv(self.layer3(self.layer2(self.layer1(x))))


class LayerNorm(nn.Module):
    """LayerNorm over the last (channel) dimension, with upstream's parameter names."""

    def __init__(self, dim: int, eps: float) -> None:
        """Build the norm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised `[..., C]`."""
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class ConvNextBlock(nn.Module):
    """7 x 7 depthwise conv, LayerNorm, Linear-GELU-Linear MLP, layer scale, residual, then a 1 x 1 conv to `cout`."""

    def __init__(self, dim: int, cout: int) -> None:
        """Build the block."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * cout)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * cout, dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.final = nn.Conv2d(dim, cout, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, dim, h, w]` -> `[B, cout, h, w]`."""
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return self.final(x + y.permute(0, 3, 1, 2))


class BasicMotionEncoder(nn.Module):
    """Encodes the correlation features and the current flow; the flow is appended to the output."""

    def __init__(self, corr_channels: int, dim: int) -> None:
        """Build the encoder."""
        super().__init__()
        self.convc1 = nn.Conv2d(corr_channels, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Motion features `[B, dim, h, w]`."""
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        return torch.cat([F.relu(self.conv(torch.cat([cor, flo], dim=1))), flow], dim=1)


class BasicUpdateBlock(nn.Module):
    """Motion encoder, then ConvNeXt blocks on [hidden | context | motion] -> new hidden state; all `dim` wide."""

    def __init__(self, dim: int, num_blocks: int, corr_channels: int) -> None:
        """Build the block."""
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_channels, dim)
        self.refine = nn.ModuleList(ConvNextBlock(3 * dim, dim) for _ in range(num_blocks))

    def forward(self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Updated hidden state."""
        inp = torch.cat([inp, self.encoder(flow, corr)], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net


def _coords(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixel coordinates `[B, 2, h, w]` (x, y), float32."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xs, ys])[None].expand(b, -1, -1, -1)


def _pyramid(fmap1: torch.Tensor, fmap2: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Correlation of fmap1 with fmap2 halved `l` times (bilinear), `[B * h * w, 1, h2, w2]` per level."""
    b, c, h, w = fmap1.shape
    f1 = fmap1.reshape(b, c, h * w).transpose(1, 2)
    out = []
    for _ in range(levels):
        h2, w2 = fmap2.shape[-2:]
        corr = torch.matmul(f1, fmap2.reshape(b, c, h2 * w2)) / math.sqrt(c)
        out.append(corr.reshape(b * h * w, 1, h2, w2))
        fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode="bilinear", align_corners=False)
    return out


def _lookup(pyramid: list[torch.Tensor], coords: torch.Tensor, radius: int) -> torch.Tensor:
    """Correlation windows around `coords` `[B, 2, h, w]` on every level -> `[B, levels * (2r + 1)^2, h, w]`."""
    side = 2 * radius + 1
    d = torch.linspace(-radius, radius, side, device=coords.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    delta = torch.stack([dy, dx], dim=-1).view(1, side, side, 2)  # as upstream: (dy, dx) offsets (x, y)
    b, _, h, w = coords.shape
    centroids = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
    out = []
    for i, corr in enumerate(pyramid):
        ch, cw = corr.shape[-2:]
        gx, gy = (centroids / 2**i + delta).split([1, 1], dim=-1)
        grid = torch.cat([2 * gx / (cw - 1) - 1, 2 * gy / (ch - 1) - 1], dim=-1)
        out.append(F.grid_sample(corr, grid, align_corners=True).view(b, h, w, -1))
    return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()


def _upsample(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flow at 1/8 -> full resolution: convex combination of 3 x 3 neighbours."""
    b, c, h, w = flow.shape
    m = torch.softmax(mask.view(b, 1, 9, FACTOR, FACTOR, h, w), dim=2)
    up = F.unfold(FACTOR * flow, kernel_size=3, padding=1).view(b, c, 9, 1, 1, h, w)
    up = torch.sum(m * up, dim=2)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * FACTOR, w * FACTOR)


def _unit(x: torch.Tensor) -> torch.Tensor:
    """`x` divided by its norm over everything but the batch."""
    return x / torch.linalg.vector_norm(x, dim=(1, 2, 3), keepdim=True)


def motion_bases(disp: torch.Tensor) -> torch.Tensor:
    """Upstream `create_bases`: `[B, 1, H, W]` relative inverse depth -> `[B, 16, H, W]` flow bases of camera motion.

    Three translations (x, y, z), scaled by `2 * disp`, and five rotation terms, each normalised over the image.
    """
    b, _, h, w = disp.shape
    ys = torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=disp.device)
    xs = torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=disp.device)
    u, v = torch.meshgrid(xs, ys, indexing="xy")
    u = (u - 0.5).expand(b, 1, h, w)
    v = (v - 0.5).expand(b, 1, h, w)
    aspect = w / h
    zero, one = torch.zeros_like(disp), torch.ones_like(disp)
    trans = [torch.cat([-one, zero], dim=1), torch.cat([zero, -one], dim=1), torch.cat([u, v], dim=1)]
    rot = [
        torch.cat([zero, one], dim=1),
        torch.cat([u * v, v * v], dim=1),
        torch.cat([-one, zero], dim=1),
        torch.cat([-u * u, -u * v], dim=1),
        torch.cat([-v / aspect, u * aspect], dim=1),
    ]
    return torch.cat([2 * disp * _unit(t) for t in trans] + [_unit(r) for r in rot], dim=1)


class FlowSeek(nn.Module):
    """FlowSeek: `(image1, image2)` RGB in `[0, 1]`, any size -> flow `[B, 2, H, W]` (image1 to image2)."""

    def __init__(self, cfg: Config = TINY) -> None:
        """Build the network."""
        super().__init__()
        self.cfg = cfg
        wide = 2 * cfg.dim
        self.cnet = ResNetFPN(cfg, 6, wide)
        self.dav2 = DepthAnythingV2("vits", features=cfg.da_features, out_channels=list(cfg.da_channels))
        f = cfg.da_features
        self.merge_head = nn.Sequential(
            nn.Conv2d(f, f // 2 * 3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f // 2 * 3, f * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 2, f * 2, 3, stride=2, padding=1),
        )
        self.bnet = ResNetFPN(cfg, 16, wide)
        self.init_conv = _conv3x3(wide, wide)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(wide, wide, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(wide, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(wide, wide, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(wide, 6, 3, padding=1)
        )
        self.fnet = ResNetFPN(cfg, 3, wide)
        self.update_block = BasicUpdateBlock(wide, cfg.num_blocks, cfg.levels * (2 * cfg.radius + 1) ** 2)
        self.fp16 = False  # fp16 autocast for the networks (CUDA)

    def _amp(self, device: torch.device) -> contextlib.AbstractContextManager:
        return torch.autocast(device.type, torch.float16) if self.fp16 else contextlib.nullcontext()

    def _mono(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream's `DepthAnythingV2.forward`: the last DPT fusion feature and the relative depth of `x`."""
        dav2, head = self.dav2, self.dav2.depth_head
        ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        feats = dav2.pretrained.get_intermediate_layers(
            x, dav2.intermediate_layer_idx[dav2.encoder], return_class_token=True
        )
        maps = [
            head.resize_layers[i](head.projects[i](t[0].permute(0, 2, 1).reshape(x.shape[0], -1, ph, pw)))
            for i, t in enumerate(feats)
        ]
        s = head.scratch
        rn = [s.layer1_rn(maps[0]), s.layer2_rn(maps[1]), s.layer3_rn(maps[2]), s.layer4_rn(maps[3])]
        path = s.refinenet4(rn[3], size=rn[2].shape[2:])
        path = s.refinenet3(path, rn[2], size=rn[1].shape[2:])
        path = s.refinenet2(path, rn[1], size=rn[0].shape[2:])
        path = s.refinenet1(path, rn[0])
        out = F.interpolate(s.output_conv1(path), (ph * 14, pw * 14), mode="bilinear", align_corners=True)
        return path, F.relu(s.output_conv2(out))

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow after `iters` updates, float32 `[B, 2, H, W]`."""
        cfg = self.cfg
        b, _, h, w = image1.shape
        device = image1.device
        both = torch.cat([image1, image2], dim=0)
        mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        small = F.interpolate(both, (DA_SIZE, DA_SIZE), mode="bilinear", align_corners=False)
        with self._amp(device):
            # upstream's normalisation (x / mean - std, not (x - mean) / std): the checkpoints were trained with it
            path, depth = self._mono(small / mean - std)
            mono = self.merge_head(F.interpolate(path.float(), (h, w), mode="bilinear", align_corners=False))
        bases = motion_bases(F.interpolate(depth[:b].float(), (h, w), mode="bilinear", align_corners=False))

        # upstream InputPadder ('sintel' mode): centred replicate padding to multiples of 8
        ph, pw = -h % FACTOR, -w % FACTOR
        pads = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        both = F.pad(2 * both - 1, pads, mode="replicate")
        bases = F.pad(bases, pads, mode="replicate")
        with self._amp(device):
            net, context = self.init_conv(self.cnet(torch.cat(both.chunk(2, dim=0), dim=1))).split(cfg.dim, dim=1)
            net_b, context_b = self.init_conv(self.bnet(bases)).split(cfg.dim, dim=1)
            net = torch.cat([net, net_b], dim=1)
            context = torch.cat([context, context_b], dim=1)
            flow = self.flow_head(net)[:, :2].float()
            fmaps = torch.cat([self.fnet(both), mono], dim=1)  # batch norm in eval mode: per-image as upstream
        fmap1, fmap2 = fmaps.float().chunk(2, dim=0)
        pyramid = _pyramid(fmap1, fmap2, cfg.levels)
        grid = _coords(b, fmap1.shape[-2], fmap1.shape[-1], device)
        for _ in range(iters):
            corr = _lookup(pyramid, grid + flow, cfg.radius)
            with self._amp(device):
                net = self.update_block(net, context, corr, flow)
                delta = self.flow_head(net)[:, :2]
            flow = flow + delta.float()
        with self._amp(device):
            mask = 0.25 * self.upsample_weight(net)
        up = _upsample(flow, mask.float())
        return up[..., pads[2] : pads[2] + h, pads[0] : pads[0] + w]
