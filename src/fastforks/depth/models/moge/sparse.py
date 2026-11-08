"""MoGe-3's sparse 3D refiner (upstream `Sparse3DUNet`) in plain PyTorch.

Upstream runs it on FlexGEMM (Triton, CUDA only). The same operations here, on any device:

- voxels: `[b, i, j, z]` integer coordinates; a level's voxels are kept sorted by their key `((b H + i) W + j) Z + z`
- submanifold 3x3x3 conv: `out[n] = bias + sum_v W[:, v] x[nbr(n, v)]` over the present neighbours `coord + d_v`,
  `d_v` in `product((-1, 0, 1), repeat=3)` over `(i, j, z)` (FlexGEMM's weight layout `[C_out, 3, 3, 3, C_in]`), as one
  gather + GEMM + index-add per tap over the precomputed neighbour pairs (each output once per tap: deterministic).
  On CUDA in fp16 the taps most voxels have (`_DENSE_SHARE`: the centre and the in-plane `dz = 0` ones) instead run as
  one gather + GEMM over a `[N, taps]` neighbour table (a zero row for a missing neighbour), the rest per tap
- mean pool 2x2x2 / stride 2: parent = `coord // 2`, mean over the present children (segment reduce); nearest
  upsampling copies the parent's features back to the skip level's voxels
- the bottleneck is fused with the dense encoder features at the parent pixel `(i, j)` of the deepest level

The finest level has one voxel per pixel of the point map, and most of its neighbours lie in the same z bin: its convs
run as a dense 3x3 2D conv with the `dz = 0` weights plus sparse corrections for the neighbour pixels in another bin
(`dz = +-1`: the weight difference; further away: no neighbour, minus the `dz = 0` term).

`SparseKernels` (kernels mode, CUDA fp16) instead keeps a neighbour table `[27, N]` per coarser level (the finest
level's convs find the neighbours from the z bins) and runs each conv as one gather implicit GEMM
(`fastforks.kernels.sparse_conv`, float32 accumulation) over the voxels sorted by their neighbour mask, with the
block's SiLU / residual in its epilogue and LayerNorm + SiLU as one kernel.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import CubinModule, linear

if TYPE_CHECKING:
    from collections.abc import Callable

_TAPS = list(itertools.product((-1, 0, 1), repeat=3))  # (di, dj, dz), FlexGEMM's order (dz fastest)
_CENTER = 13
# CUDA fp16: taps present at >= this share of a level's voxels join the one gather + GEMM; the rest run per tap.
# Measured on the RTX 5060 Ti (MoGe-3 refiner, fp16): 0.4 the fastest (994x1482 400 ms vs 443 ms per tap, 518x770 214
# vs 291); float32 and MPS stay per tap (the table gather loses there).
_DENSE_SHARE = 0.4
_GATHER_BLOCK = 1 << 24  # elements of one gathered `[rows, taps * C_in]` block
_LN_EPS = 1e-6  # fixed: upstream SparseResBlock3d LayerNorm


@dataclass
class Level:
    """The voxels of one level: coordinates `[N, 4]` (int64, sorted by key), extents and the neighbours.

    `nbr [27, N]`: per tap and voxel the input index (int32, -1: none); `pairs` the same as per-tap (output, input)
    index lists. The finest level instead has `grid` = (B, H, W) and `fixes`: the corrections of the dense conv,
    `(spatial tap, dz or None for "no neighbour", output indices, input indices)`.
    """

    coords: torch.Tensor
    extent: tuple[int, int, int]  # H, W, Z of this level
    nbr: torch.Tensor | None = None
    grid: tuple[int, int, int] | None = None
    fixes: list[tuple[int, int | None, torch.Tensor, torch.Tensor]] | None = None
    perm: torch.Tensor | None = None  # SparseKernels: the rows sorted by neighbour mask (int32)
    zbins: torch.Tensor | None = None  # SparseKernels, finest level: z [B, H, W] (int64) instead of nbr

    @functools.cached_property
    def pairs(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """`pairs[v]` = (output indices, input indices) of tap `v`; `None` for the centre tap (every voxel)."""
        assert self.nbr is not None  # noqa: S101 - a sparse level
        return _pairs(self.nbr)

    @functools.cached_property
    def dense(self) -> tuple[list[int], torch.Tensor]:
        """The taps present at >= `_DENSE_SHARE` of the voxels (the centre included) and their table.

        Table `[N, taps]`: the int64 input index, `N` for none (a zero row).
        """
        assert self.nbr is not None  # noqa: S101 - a sparse level
        n = self.nbr.shape[1]
        share = ((self.nbr >= 0).sum(1) / n).tolist()
        taps = [v for v in range(len(_TAPS)) if v == _CENTER or share[v] >= _DENSE_SHARE]
        nb = self.nbr[taps]
        return taps, torch.where(nb >= 0, nb, n).t().long().contiguous()


def _key(c: torch.Tensor, extent: tuple[int, int, int]) -> torch.Tensor:
    h, w, z = extent
    return ((c[:, 0] * h + c[:, 1]) * w + c[:, 2]) * z + c[:, 3]


def _sorted_counts(x: torch.Tensor, n: int) -> torch.Tensor:
    """`torch.bincount(x, minlength=n)` for sorted `x` in `[0, n)` (bincount takes ~0.6 s per 1M values on MPS)."""
    return torch.searchsorted(x.contiguous(), torch.arange(n + 1, device=x.device)).diff()


def _pairs(nbr: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    """`nbr [27, N]` (input index or -1) -> per-tap (output, input) index pairs (one host sync)."""
    tap, out = (nbr >= 0).nonzero(as_tuple=True)
    counts = _sorted_counts(tap, len(_TAPS)).tolist()
    inp = nbr[tap, out].long()
    pairs: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    start = 0
    for v, n in enumerate(counts):
        pairs.append(None if v == _CENTER else (out[start : start + n], inp[start : start + n]))
        start += n
    return pairs


def _grid_coords(z: torch.Tensor) -> torch.Tensor:
    b, h, w = z.shape
    dev = z.device
    ii, jj = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    return torch.stack(
        [torch.arange(b, device=dev).view(b, 1, 1).expand(b, h, w), ii.expand(b, h, w), jj.expand(b, h, w), z], -1
    ).reshape(-1, 4)


def grid_level(z: torch.Tensor) -> Level:
    """The finest level: one voxel per pixel of `z [B, H, W]` (int64 z bins), voxel index = `(b H + i) W + j`."""
    b, h, w = z.shape
    dev = z.device
    coords = _grid_coords(z)
    zp = F.pad(z[:, None].float(), (1, 1, 1, 1), value=float("nan"))[:, 0]  # out of the image: zero padding
    idx = torch.arange(b * h * w, device=dev).view(b, h, w)
    spatial = [(di, dj) for di, dj in itertools.product((-1, 0, 1), repeat=2) if (di, dj) != (0, 0)]
    classes: list[tuple[int, int | None]] = [(t, k) for t in range(len(spatial)) for k in (-1, 1, None)]
    nbr = torch.full((len(classes), b * h * w), -1, dtype=torch.int64, device=dev)
    for c, (t, k) in enumerate(classes):
        di, dj = spatial[t]
        dz = zp[:, 1 + di : 1 + di + h, 1 + dj : 1 + dj + w] - z  # z of the neighbour pixel - own z (NaN outside)
        hit = dz == k if k is not None else dz.abs() > 1
        nbr[c] = torch.where(hit, idx + (di * w + dj), -1).view(-1)
    cls, out = (nbr >= 0).nonzero(as_tuple=True)
    counts = _sorted_counts(cls, len(classes)).tolist()
    inp = nbr[cls, out]
    fixes = []
    start = 0
    for (t, k), n in zip(classes, counts, strict=True):
        if n:
            di, dj = spatial[t]
            fixes.append(((di + 1) * 3 + (dj + 1), k, out[start : start + n], inp[start : start + n]))
        start += n
    return Level(coords, (h, w, int(z.max()) + 1), grid=(b, h, w), fixes=fixes)


def sparse_level(coords: torch.Tensor, extent: tuple[int, int, int]) -> Level:
    """A level from voxel coordinates sorted by key: neighbours by binary search."""
    key = _key(coords, extent)
    n = key.numel()
    h, w, z = extent
    d = torch.tensor(_TAPS, device=coords.device)  # [27, 3]
    c = coords[None, :, 1:] + d[:, None]  # [27, N, 3] neighbour coordinates
    inside = ((c >= 0) & (c < torch.tensor([h, w, z], device=coords.device))).all(-1)
    want = key + ((d[:, 0] * w + d[:, 1]) * z + d[:, 2])[:, None]
    pos = torch.searchsorted(key, want).clamp_(max=n - 1)
    return Level(coords, extent, torch.where(inside & (key[pos] == want), pos, -1).int())


@dataclass
class Pool:
    """A 2x2x2 mean-pooling edge: the parent level and, per child voxel, its parent index; children sorted by parent."""

    level: Level
    parent: torch.Tensor  # [N_child] index into the parent level
    order: torch.Tensor  # children sorted by parent
    counts: torch.Tensor  # children per parent


def pool_level(child: Level, make: Callable[[torch.Tensor, tuple[int, int, int]], Level] = sparse_level) -> Pool:
    """The parent level of `child` (`coord // 2`, sorted by key, built by `make`) and the child -> parent map."""
    h, w, z = child.extent
    ext = ((h + 1) // 2, (w + 1) // 2, (z + 1) // 2)
    c = child.coords
    pc = torch.cat([c[:, :1], c[:, 1:] // 2], 1)
    keys, order = torch.sort(_key(pc, ext), stable=True)
    first = torch.ones_like(keys, dtype=torch.bool)
    first[1:] = keys[1:] != keys[:-1]
    seg = torch.cumsum(first, 0) - 1
    parent = torch.empty_like(seg)
    parent[order] = seg
    counts = _sorted_counts(seg, int(seg[-1]) + 1)
    return Pool(make(pc[order[first]], ext), parent, order, counts)


def _segment_mean(x: torch.Tensor, p: Pool) -> torch.Tensor:
    """Mean of the children's features per parent voxel: `[N_parent, C]`."""
    if x.device.type == "mps":  # no segment_reduce kernel there (torch 2.14): sum the children in parent order
        s = x.new_zeros(len(p.counts), x.shape[1]).index_add_(0, p.parent[p.order], x[p.order])
        return s / p.counts[:, None].to(x.dtype)
    return torch.segment_reduce(x[p.order], "mean", lengths=p.counts)  # ty: ignore[unresolved-attribute] - not in the stubs


def subm_conv(x: torch.Tensor, lv: Level, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Submanifold 3x3x3 conv of `x [N, C_in]` (weight `[C_out, 3, 3, 3, C_in]`) at `lv`'s voxels."""
    if lv.grid is not None:
        return _grid_conv(x, lv, weight, bias)
    w = weight.reshape(weight.shape[0], len(_TAPS), weight.shape[-1]).to(x.dtype)
    if x.is_cuda and x.dtype == torch.float16:
        taps, table = lv.dense
        out = _gather_mm(x, table, w[:, taps].reshape(w.shape[0], -1).t(), bias.to(x.dtype))
    else:
        taps = [_CENTER]
        out = torch.addmm(bias.to(x.dtype), x, w[:, _CENTER].t())
    for v, p in enumerate(lv.pairs):
        if v not in taps and p is not None and p[0].numel():
            out.index_add_(0, p[0], x[p[1]] @ w[:, v].t())
    return out


def _gather_mm(x: torch.Tensor, table: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """`bias + x[table] @ w` per row block (`table [N, taps]`, `N` = a zero row; `w [taps * C_in, C_out]`)."""
    xp = torch.cat([x, x.new_zeros(1, x.shape[1])])
    out = x.new_empty(table.shape[0], w.shape[1])
    rows = max(1, _GATHER_BLOCK // w.shape[0])
    for s in range(0, table.shape[0], rows):
        t = table[s : s + rows]
        torch.addmm(bias, xp[t].view(t.shape[0], -1), w, out=out[s : s + rows])
    return out


def _grid_conv(x: torch.Tensor, lv: Level, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """`subm_conv` at the finest level: dense 2D conv with the `dz = 0` weights + the corrections `lv.fixes`."""
    assert lv.grid is not None  # noqa: S101 - grid levels have a grid
    assert lv.fixes is not None  # noqa: S101 - and corrections
    b, h, w = lv.grid
    wt = weight.to(x.dtype)  # [C_out, 3 (i), 3 (j), 3 (z), C_in]
    xs = x.view(b, h, w, -1).permute(0, 3, 1, 2)  # NCHW view of NHWC memory (channels_last)
    out = F.conv2d(xs, wt[:, :, :, 1].permute(0, 3, 1, 2), bias.to(x.dtype), padding=1)
    out = out.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
    for t, k, o, i in lv.fixes:
        w0 = wt[:, t // 3, t % 3, 1]
        wk = -w0 if k is None else wt[:, t // 3, t % 3, k + 1] - w0
        out.index_add_(0, o, x[i] @ wk.t())
    return out


_CONV_TILES = {
    "subm_conv_n32": (32, 128),
    "subm_conv_n64": (64, 128),
    "subm_conv_n64s3": (64, 128),
    "subm_conv_n128": (128, 256),
}


class SparseKernels:
    """The refiner's levels, convs and LayerNorm + SiLU on the `sparse_conv` kernels (CUDA, fp16 weights / features)."""

    VARIANT: dict[int, str] = {}  # noqa: RUF012 - C_out -> conv kernel (dev A/B)

    def __init__(self, lib: CubinModule, net: nn.Module, *, sort: bool = True) -> None:
        """`lib`: `sparse_conv` on the device of `net` (fp16); `sort`: tiles of voxels with the same neighbour mask."""
        self.lib, self.sort = lib, sort
        self.wt: dict[SubmConv3d, torch.Tensor] = {}  # [27, C_in, C_out]: the B operand of each tap
        for m in net.modules():
            if isinstance(m, SubmConv3d):  # replaces the weight
                w = m.weight.detach()
                self.wt[m] = w.reshape(w.shape[0], len(_TAPS), w.shape[-1]).permute(1, 2, 0).contiguous()
                m.register_parameter("weight", None)

    def _level(
        self, coords: torch.Tensor, extent: tuple[int, int, int], nbr: torch.Tensor | None, mask: torch.Tensor
    ) -> Level:
        perm = torch.argsort(mask, stable=True).int() if self.sort else None
        return Level(coords, extent, nbr=nbr, perm=perm)

    def grid_level(self, z: torch.Tensor) -> Level:
        """`grid_level` without a neighbour table: the convs find a pixel's neighbours from the z bins themselves."""
        b, h, w = z.shape
        n = b * h * w
        z = z.contiguous()
        mask = torch.empty(n, dtype=torch.int32, device=z.device)
        self.lib.launch("grid_nbr", *linear(n), z, None, mask, b, h, w)
        lv = self._level(_grid_coords(z), (h, w, int(z.max()) + 1), None, mask)
        lv.zbins = z
        return lv

    def sparse_level(self, coords: torch.Tensor, extent: tuple[int, int, int]) -> Level:
        """`sparse_level` with a neighbour table (binary search over the sorted keys)."""
        coords = coords.contiguous()
        key = _key(coords, extent).contiguous()
        n = key.numel()
        nbr = torch.empty((len(_TAPS), n), dtype=torch.int32, device=coords.device)
        mask = torch.empty(n, dtype=torch.int32, device=coords.device)
        self.lib.launch("sparse_nbr", *linear(n), coords, key, nbr, mask, n, *extent)
        return self._level(coords, extent, nbr, mask)

    def conv(
        self, x: torch.Tensor, lv: Level, m: SubmConv3d, *, act: bool = False, res: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`act(subm_conv(x) (+ res))`, act = SiLU."""
        n, c_in = x.shape
        wt = self.wt[m]
        c_out = wt.shape[2]
        name = self.VARIANT.get(c_out) or ("subm_conv_n64" if c_out % 64 == 0 else "subm_conv_n32")
        bn, threads = _CONV_TILES[name]
        out = x.new_empty(n, c_out)
        grid = ((n + 63) // 64, c_out // bn, 1)
        h, w = lv.extent[:2]
        args = (x, lv.nbr, lv.perm, wt, m.bias, res, out, lv.zbins, n, c_in, c_out, int(act), h, w)
        self.lib.launch(name, grid, (threads, 1, 1), *args)
        return out

    def block(self, blk: nn.Module, x: torch.Tensor, lv: Level) -> torch.Tensor:
        """`SparseResBlock3d` with LayerNorm + SiLU fused and SiLU / residual in the conv epilogues."""
        assert isinstance(blk, SparseResBlock3d)  # noqa: S101 - the refiner's stages hold residual blocks
        n, c = x.shape
        h = torch.empty_like(x)
        ln = blk.norm1
        self.lib.launch("ln_silu", ((n + 3) // 4, 1, 1), (128, 1, 1), x, ln.weight, ln.bias, h, n, c, ln.eps)
        return self.conv(self.conv(h, lv, blk.conv1, act=True), lv, blk.conv2, res=x)


class SubmConv3d(nn.Module):
    """FlexGEMM `SubmanifoldConv3d(c, c, 3)` parameters: weight `[C_out, 3, 3, 3, C_in]`, bias."""

    def __init__(self, c_in: int, c_out: int) -> None:
        """Uninitialised parameters (loaded from the checkpoint)."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, 3, 3, 3, c_in))
        self.bias = nn.Parameter(torch.empty(c_out))

    def forward(self, x: torch.Tensor, lv: Level) -> torch.Tensor:
        """Conv at `lv`'s voxels."""
        return subm_conv(x, lv, self.weight, self.bias)


class SparseResBlock3d(nn.Module):
    """`x + conv2(silu(conv1(silu(LayerNorm(x)))))`."""

    def __init__(self, c: int) -> None:
        """Width `c`."""
        super().__init__()
        self.norm1 = nn.LayerNorm(c, eps=_LN_EPS)
        self.conv1, self.conv2 = SubmConv3d(c, c), SubmConv3d(c, c)

    def forward(self, x: torch.Tensor, lv: Level) -> torch.Tensor:
        """Residual block at `lv`'s voxels."""
        h = F.silu(self.norm1(x).type_as(x))
        return self.conv2(F.silu(self.conv1(h, lv)), lv) + x


class _Resample(nn.Module):
    """Upstream `PoolDown` / `NearestUp`: their `linear` (applied after pooling / before upsampling)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)


def _stage(c: int) -> nn.ModuleList:
    return nn.ModuleList([SparseResBlock3d(c)])


class SparseUNet(nn.Module):
    """Upstream `Sparse3DUNet` for the released MoGe-3 config (1 block per level and in the bottleneck, factors 2)."""

    def __init__(self, c_in: int, c_out: int, c_enc: int, channels: list[int]) -> None:
        """Widths per level, finest first."""
        super().__init__()
        n = len(channels)
        self.input_proj = nn.Linear(c_in, channels[0])
        self.encoder_fuse = nn.Linear(c_enc, channels[-1])
        self.fuse_proj = nn.Sequential(
            nn.Linear(2 * channels[-1], channels[-1]), nn.SiLU(), nn.Linear(channels[-1], channels[-1])
        )
        self.down_stages = nn.ModuleList(_stage(c) for c in channels)
        self.downsample_blocks = nn.ModuleList(_Resample(channels[i], channels[i + 1]) for i in range(n - 1))
        self.bottleneck_stage = _stage(channels[-1])
        self.upsample_blocks = nn.ModuleList(_Resample(channels[n - 1 - i], channels[n - 2 - i]) for i in range(n - 1))
        self.up_stages = nn.ModuleList(_stage(channels[n - 2 - i]) for i in range(n - 1))
        self.out_proj = nn.Linear(channels[0], c_out)
        self.kern: SparseKernels | None = None  # set by the kernels runner

    def forward(self, x: torch.Tensor, z: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        """Refine: `x [B*H*W, C_in]` per pixel (row-major), z bins `[B, H, W]`, encoder map `[B, C, H/16, W/16]`.

        Returns `[B*H*W, C_out]` (the dtype of the blocks' input).
        """
        k = self.kern
        levels = [k.grid_level(z) if k else grid_level(z)]
        block = k.block if k else (lambda blk, x, lv: blk(x, lv))
        pools: list[Pool] = []
        skips: list[torch.Tensor] = []
        x = self.input_proj(x)
        for i, stage in enumerate(self.down_stages):
            for blk in stage.children():
                x = block(blk, x, levels[i])
            if i < len(self.downsample_blocks):
                skips.append(x)
                p = pool_level(levels[i], k.sparse_level if k else sparse_level)
                pools.append(p)
                levels.append(p.level)
                x = _segment_mean(x, p)
                x = self.downsample_blocks[i].get_submodule("linear")(x)
        c = levels[-1].coords
        x = self.fuse_proj(torch.cat([x, self.encoder_fuse(enc[c[:, 0], :, c[:, 1], c[:, 2]].to(x.dtype))], -1))
        for blk in self.bottleneck_stage.children():
            x = block(blk, x, levels[-1])
        for i, (up, stage) in enumerate(zip(self.upsample_blocks, self.up_stages, strict=True)):
            t = len(levels) - 2 - i
            x = up.get_submodule("linear")(x)[pools[t].parent] + skips[t]
            for blk in stage.children():
                x = block(blk, x, levels[t])
        return self.out_proj(x)
