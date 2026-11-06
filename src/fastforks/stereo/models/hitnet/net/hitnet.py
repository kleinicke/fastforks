"""HITNet network in plain PyTorch (NCHW, any input size).

Written from the architecture of the authors' released TensorFlow graphs (google-research/hitnet: `eth3d.pb`,
`middlebury_d400.pb`); layer names follow the TF variables.

Structure of a graph:
  feature U-Net (`fe_shared`): conv_down_1_0, then n x [conv_down_i (4x4 s2) + conv_down_1_{i+1}], conv_up_1_n/conv_up_2_n,
    then per scale: conv_upsample_i (2x2 transposed) -> crop -> concat skip -> conv_up_0_i (1x1) conv_up_1_i conv_up_2_i
  levels 0..L-1 (coarse to fine; level k uses the features of scale n-k, tiles of 4x4 pixels):
    init: 4x4/4 tile embedding (left) against 4x4/(4,1) embedding (right, every pixel), L1 cost over D disparities, argmin
    prop: per hypothesis warp the right features with the slanted tile plane (d, dx, dy) at offsets +1/0/-1, SAD
          -> 64-channel tile cost -> 1x1 conv; ResNet updates every hypothesis; the argmin of the predicted weight selects
    upsample: 2x tiles, d' = 2d +- 2dx +- 2dy, other channels nearest
  refinement r = 0..2 at scales 2..0 (optionally a warp SAD replaces the last 3 feature channels), tile expansion 2x.
Hypotheses, disparities, argmin decisions and warp positions stay float32; the convolutions run in the caller's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

SLOPE = 0.2  # fixed: LeakyReLU slope of every activation in the released graphs
TILE = 4  # fixed: tile size of the hypotheses (pixels)
HYP = 16  # fixed: hypothesis channels (d, dx, dy and 13 descriptor channels)


def lrelu(x: torch.Tensor) -> torch.Tensor:
    """LeakyReLU(0.2)."""
    return F.leaky_relu(x, SLOPE)


@dataclass(frozen=True)
class Refine:
    """One refinement stage: channel width, dilations of its residual blocks, scale of the disparity used for the
    warp SAD (`None`: no warp SAD, the features are used as they are)."""

    channels: int
    dilations: tuple[int, ...]
    sad_scale: float | None


class ResBlock(nn.Module):
    """Two dilated 3x3 convolutions with an identity shortcut."""

    def __init__(self, c: int, dilation: int) -> None:
        """Build the block with `c` channels."""
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, dilation, dilation)
        self.conv2 = nn.Conv2d(c, c, 3, 1, dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`lrelu(conv2(lrelu(conv1(x))) + x)`."""
        return lrelu(self.conv2(lrelu(self.conv1(x))) + x)


class ResNet(nn.Module):
    """3x3 convolution, residual blocks, 3x3 convolution without activation."""

    def __init__(self, cin: int, c: int, cout: int, dilations: Sequence[int]) -> None:
        """Build the network."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, 1, 1)
        self.res = nn.ModuleList([ResBlock(c, d) for d in dilations])
        self.convn = nn.Conv2d(c, cout, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual stack."""
        x = lrelu(self.conv1(x))
        for r in self.res:
            x = r(x)
        return self.convn(x)


class FeatureUNet(nn.Module):
    """Shared feature extractor; returns the features from the coarsest to the finest scale."""

    def __init__(self, cin: int, ch: Sequence[int]) -> None:
        """`ch`: feature channels per scale, finest first."""
        super().__init__()
        n = len(ch) - 1
        self.n = n
        self.conv_down_1 = nn.ModuleList([nn.Conv2d(cin if i == 0 else ch[i], ch[i], 3, 1, 1) for i in range(n + 1)])
        self.conv_down = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1) for i in range(n)])
        self.conv_up_1n = nn.Conv2d(ch[n], ch[n], 3, 1, 1)
        self.conv_up_2n = nn.Conv2d(ch[n], ch[n], 3, 1, 1)
        self.conv_upsample = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in range(n)])
        self.conv_up_0 = nn.ModuleList([nn.Conv2d(2 * ch[i], ch[i], 1) for i in range(n)])
        self.conv_up_1 = nn.ModuleList([nn.Conv2d(ch[i], ch[i], 3, 1, 1) for i in range(n)])
        self.conv_up_2 = nn.ModuleList([nn.Conv2d(ch[i], ch[i], 3, 1, 1) for i in range(n)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Features of the images `x`, coarse to fine."""
        skips = []
        x = lrelu(self.conv_down_1[0](x))
        skips.append(x)
        for i in range(self.n):
            x = lrelu(self.conv_down[i](x))
            x = lrelu(self.conv_down_1[i + 1](x))
            skips.append(x)
        x = lrelu(self.conv_up_2n(lrelu(self.conv_up_1n(x))))
        outs = [x]
        for i in range(self.n - 1, -1, -1):
            s = skips[i]
            u = lrelu(self.conv_upsample[i](x))[:, :, : s.shape[2], : s.shape[3]]
            x = lrelu(self.conv_up_0[i](torch.cat((u, s), 1)))
            x = lrelu(self.conv_up_2[i](lrelu(self.conv_up_1[i](x))))
            outs.append(x)
        return outs


def expand_tiles(d: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Tile planes -> per-pixel disparity at `TILE` times the tile resolution: `d + cx dx + cy dy` (centre offsets)."""
    n, _, h, w = d.shape
    r = torch.arange(TILE, device=d.device, dtype=d.dtype) - (TILE - 1) / 2
    cy, cx = torch.meshgrid(r, r, indexing="ij")
    e = d + cx.reshape(1, -1, 1, 1) * dx + cy.reshape(1, -1, 1, 1) * dy  # [n, 16, h, w]
    return e.view(n, TILE, TILE, h, w).permute(0, 3, 1, 4, 2).reshape(n, 1, h * TILE, w * TILE)


def upsample_hyp(hyp: torch.Tensor, dmul: float, coef: float) -> torch.Tensor:
    """2x tile upsampling: `d' = dmul d + coef (+-dx +- dy)` for the four children, other channels nearest."""
    d, dx, dy = hyp[:, 0:1], hyp[:, 1:2], hyp[:, 2:3]
    n, _, h, w = d.shape
    sx = torch.arange(2, device=d.device, dtype=d.dtype) * 2 - 1  # [-1, 1] without a host copy (CUDA graphs)
    e = dmul * d.unsqueeze(2).unsqueeze(2) + coef * (
        sx.view(1, 1, 1, 2, 1, 1) * dx.unsqueeze(2).unsqueeze(2) + sx.view(1, 1, 2, 1, 1, 1) * dy.unsqueeze(2).unsqueeze(2)
    )  # [n, 1, 2 (by), 2 (bx), h, w]
    e = e.permute(0, 1, 4, 2, 5, 3).reshape(n, 1, 2 * h, 2 * w)
    return torch.cat((e, F.interpolate(hyp[:, 1:], scale_factor=2, mode="nearest")), 1)


def init_cost(lt: torch.Tensor, rt: torch.Tensor, max_disp: int, budget: int = 256 << 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Best disparity and its cost: `cost(d) = sum_c |lt(x) - rt(clip(4x + 1 - d))|`, first minimum, row chunks.

    Replicate padding plus an `unfold` view instead of index gathers.
    """
    n, c, h, w = lt.shape
    rp = F.pad(rt, (max_disp - 1, 0, 0, 0), mode="replicate")  # index 4x + 1 - d -> 4x + D - d in rp
    win = rp.unfold(3, max_disp + 1, TILE)[..., :w, :]  # [n, c, h, w, D + 1], [.., x, j] = rp[4x + j], j = D - d
    rows = max(1, budget // max(1, n * c * w * max_disp * lt.element_size()))
    ds, cs = [], []
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        cost = (lt[:, :, y0:y1, :, None] - win[:, :, y0:y1, :, 1:]).abs().sum(1, dtype=torch.float32)  # j = 1..D
        cm, dm = cost.flip(-1).min(-1)  # d = 0..D-1
        ds.append(dm)
        cs.append(cm)
    return torch.cat(ds, 1).unsqueeze(1).float(), torch.cat(cs, 1).unsqueeze(1)


def warp_rows(fr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """TF `warp_1d`: `fr` `[n, C, H, W]` sampled at the horizontal positions `x` `[n, k, H, W]` (same rows), clipped
    linear interpolation, as one per-row `grid_sample`. Returns `[n, k, C, H, W]` float32."""
    n, c, h, w = fr.shape
    k = x.shape[1]
    src = fr.float().permute(0, 2, 1, 3).reshape(n * h, c, 1, w)  # the sample coordinates need float32
    gx = x.permute(0, 2, 1, 3).reshape(n * h, 1, k * w) * (2.0 / max(w - 1, 1)) - 1.0
    grid = torch.stack((gx, torch.zeros_like(gx)), -1)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)  # [nH, C, 1, kW]
    return out.reshape(n, h, c, k, w).permute(0, 3, 2, 1, 4)


def x_positions(like: torch.Tensor) -> torch.Tensor:
    """Pixel columns `[1, 1, 1, W]` as float32."""
    return torch.arange(like.shape[3], device=like.device, dtype=torch.float32).view(1, 1, 1, -1)


def tile_sad(hyp: torch.Tensor, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
    """64-channel tile cost of a hypothesis `[n, 16, h, w]` on the features at 4x its resolution: per pixel
    `[sum |fl|, SAD(+1), SAD(0), SAD(-1)]`, packed (by, bx, k)."""
    n, _, h, w = hyp.shape
    base = x_positions(fl) - expand_tiles(hyp[:, 0:1], hyp[:, 1:2], hyp[:, 2:3])
    sad = (fl.unsqueeze(1) - warp_rows(fr, torch.cat((base + 1, base, base - 1), 1))).abs().sum(2, dtype=torch.float32)
    cost = torch.cat((fl.abs().sum(1, keepdim=True, dtype=torch.float32), sad), 1)
    return cost.view(n, 4, h, TILE, w, TILE).permute(0, 3, 5, 1, 2, 4).reshape(n, TILE * TILE * 4, h, w)


def refine_sad(d: torch.Tensor, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
    """SAD of `fl` against `fr` warped by `d`, `d + 1` and `d - 1` (`d` already scaled): `[n, 3, H, W]`."""
    xs = x_positions(fl)
    pos = torch.cat((xs - d, xs - (d + 1), xs - (d - 1)), 1)
    return (fl.unsqueeze(1) - warp_rows(fr, pos)).abs().sum(2, dtype=torch.float32)


class LevelInit(nn.Module):
    """Tile embedding of both feature maps, initial disparity by cost minimisation, first hypothesis."""

    def __init__(self, cin: int, max_disp: int, cref: int) -> None:
        """`max_disp` disparities are searched at this level; `cref` is the channel count of the extra input."""
        super().__init__()
        self.max_disp = max_disp
        self.conv_reduce = nn.Conv2d(cin, HYP, TILE, TILE)
        self.conv_embedding = nn.Conv2d(HYP, HYP, 1)
        self.conv_hyp = nn.Conv2d(cref + 1, 13, 1)
        self.init_cost = init_cost  # the kernels runner on Apple MPS swaps in a Metal kernel

    def embed(self, fl: torch.Tensor, fr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left tile embedding (stride 4) and right embedding (stride (4, 1): every pixel)."""
        red = self.conv_reduce
        lt = lrelu(self.conv_embedding(lrelu(F.conv2d(fl, red.weight, red.bias, stride=TILE))))
        rt = lrelu(self.conv_embedding(lrelu(F.conv2d(F.pad(fr, (1, 2, 0, 0)), red.weight, red.bias, stride=(TILE, 1)))))
        return lt, rt

    def hypothesis(self, d: torch.Tensor, cmin: torch.Tensor, lt: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        """`[d, 0, 0, 13 descriptor channels]` from the best disparity, its cost and the left embedding."""
        p = lrelu(self.conv_hyp(torch.cat((cmin.to(lt.dtype), lt if ref is None else ref), 1)))
        z = torch.zeros_like(d)
        return torch.cat((d, z, z, p.float()), 1)

    def forward(self, fl: torch.Tensor, fr: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        """Hypothesis `[n, 16, h, w]` of the level."""
        lt, rt = self.embed(fl, fr)
        d, cmin = self.init_cost(lt, rt, self.max_disp)
        return self.hypothesis(d, cmin, lt, ref)


class LevelProp(nn.Module):
    """Propagation: tile costs of the hypotheses, ResNet update, selection among the hypotheses."""

    def __init__(self, hypotheses: int) -> None:
        """`hypotheses`: how many hypotheses the level carries (1 at the coarsest level, else 2)."""
        super().__init__()
        self.nh = hypotheses
        self.conv_neighbors = nn.Conv2d(TILE * TILE * 4, HYP, 1)
        self.prop = ResNet(2 * HYP * hypotheses, 2 * HYP, (HYP + 1) * hypotheses, [1, 1])
        self.tile_sad = tile_sad  # the kernels runner on Apple MPS swaps in a Metal kernel

    def select(self, hyps: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Updated hypotheses from the ResNet output `x`; the one with the smallest predicted weight (TF `ArgMin`)."""
        x = x.float()
        new = [hyps[i] + x[:, (HYP + 1) * i : (HYP + 1) * i + HYP] for i in range(self.nh)]
        if self.nh == 1:
            return new[0]
        sel = torch.cat([x[:, (HYP + 1) * i + HYP : (HYP + 1) * (i + 1)] for i in range(self.nh)], 1).argmin(1, keepdim=True)
        out = new[0]
        for i in range(1, self.nh):
            out = torch.where(sel == i, new[i], out)
        return out

    def forward(self, hyps: list[torch.Tensor], fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        """Propagated hypothesis."""
        parts = []
        for hp in hyps:
            parts += [hp, lrelu(self.conv_neighbors(self.tile_sad(hp, fl, fr)))]
        dtype = parts[1].dtype
        return self.select(hyps, self.prop(torch.cat([p.to(dtype) for p in parts], 1)))


class Refinement(nn.Module):
    """Residual refinement of the full hypothesis at one scale."""

    def __init__(self, cf: int, stage: Refine) -> None:
        """`cf`: feature channels at the scale."""
        super().__init__()
        self.sad_scale = stage.sad_scale
        self.conv1x1 = nn.Conv2d(cf + HYP, stage.channels, 1)
        self.refiner = ResNet(stage.channels, stage.channels, HYP, stage.dilations)
        self.refine_sad = refine_sad  # the kernels runner on Apple MPS swaps in a Metal kernel

    def forward(self, hyp: torch.Tensor, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        """Refined hypothesis."""
        if self.sad_scale is not None:  # WarpSadAndReplace: the last 3 feature channels become SADs at offsets 0, +1, -1
            sad = self.refine_sad(hyp[:, 0:1] * self.sad_scale, fl, fr)
            fl = torch.cat((fl[:, :-3], sad.to(fl.dtype)), 1)
        x = lrelu(self.conv1x1(torch.cat((fl, hyp.to(fl.dtype)), 1)))
        return hyp + self.refiner(x).float()


class HITNet(nn.Module):
    """The network. `forward` takes padded RGB images in 0..255 and returns the disparity `[N, 1, H, W]`."""

    def __init__(
        self,
        in_ch: int,
        ch: Sequence[int],
        disparities: Sequence[int],
        refine: Sequence[Refine],
    ) -> None:
        """Build from the architecture numbers of a variant.

        Args:
            in_ch: Input channels per image (1: gray, 3: RGB).
            ch: Feature channels per scale, finest first.
            disparities: Disparities searched per level, coarse to fine.
            refine: Refinement stages.
        """
        super().__init__()
        n = len(ch) - 1
        if len(disparities) != n + 1:
            msg = f"{len(disparities)} levels for {n + 1} scales"
            raise ValueError(msg)
        self.in_ch = in_ch
        self.fe = FeatureUNet(in_ch, ch)
        self.init = nn.ModuleList()
        self.prop = nn.ModuleList()
        for k, d in enumerate(disparities):
            s = n - k
            self.init.append(LevelInit(ch[s], d, 16 if k < 2 else ch[s + 2]))
            self.prop.append(LevelProp(1 if k == 0 else 2))
        self.refine = nn.ModuleList([Refinement(ch[2 - r], stage) for r, stage in enumerate(refine)])

    def prepare(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Both images as one batch `[2N, in_ch, H, W]` in the network's `[-1, 1]` input range (gray: BT.601 luma)."""
        x = torch.cat((left, right), 0)
        if self.in_ch == 1:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x = 0.299 * r + 0.587 * g + 0.114 * b
        return x * (2.0 / 255.0) - 1.0

    def levels(self, feats: list[torch.Tensor], n: int) -> torch.Tensor:
        """Hypothesis after all levels and refinements from the features of the left (first `n`) and right images."""
        fl, fr = [f[:n] for f in feats], [f[n:] for f in feats]
        hyp = torch.empty(0)
        for k, (init, prop) in enumerate(zip(self.init, self.prop, strict=True)):
            hi = init(fl[k], fr[k], None if k < 2 else fl[k - 2])
            hyp = prop([hi] if k == 0 else [hi, upsample_hyp(hyp, 2.0, 2.0)], fl[k], fr[k])
        base = len(self.init) - 3  # fixed: refinement r works at scale 2 - r
        for r, ref in enumerate(self.refine):
            hyp = ref(hyp, fl[base + r], fr[base + r])
            if r < len(self.refine) - 1:
                hyp = upsample_hyp(hyp, 1.0, 0.5)
        return hyp

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Disparity `[N, 1, H, W]` of padded images `[N, 3, H, W]` (0..255)."""
        x = self.prepare(left, right)
        return self.levels(self.fe(x), left.shape[0])[:, 0:1]
