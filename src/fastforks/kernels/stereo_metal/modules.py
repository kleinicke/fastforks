"""The Metal kernels of `stereo_kernels.metal` as drop-in replacements for the portable stereo blocks (same math).

- `FusedConvGRU`: a `ConvGRU` (RAFT-Stereo, the IGEV family) with the z and r convolutions as one and the gates and
  the update as one kernel each.
- `FlatCorr`: RAFT-Stereo's `CorrBlockGather1D` with the pyramid in one buffer and the lookup as one kernel.
- `DispCorr`: DEFOM-Stereo's `CorrBlock1D` (pyramid and scale lookups around a disparity) on the `FlatCorr` buffer.
- `LineLookup`: the IGEV family's `FastGeoEncodingVolume` lookups (geometry and correlation pyramid) as one kernel
  per output.
- `WarpVolume`: CFNet's refinement-stage volume (warp, group-wise correlation, concatenation) as one kernel.
- `gwc_volume`: a group-wise correlation volume (LaS2) as one kernel.
- `win_rpe_attn`: NMRF / BridgeDepth's window attention with relative-position embeddings as one kernel.
- `HitnetCosts`: HITNet's level-init cost (cost and argmin, no cost volume) and its tile / refinement warp SADs.
- `DeformConv`: AANet's modulated deformable convolution with the sampling as one kernel.
- `CrestereoOps`: CREStereo's iterative group correlation (warp, local window) and GRU gates as kernels.
- `patch_igev`: the IGEV family's GRUs and lookups swapped for the above, in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F

from fastforks._core import mps
from fastforks._core.patch import base_type

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastforks._core.metal import MetalLibrary

_CORR_DIMS = 2  # a correlation volume is [P, W], a geometry volume [P, C, D]


class ConvGRULike(Protocol):
    """A `ConvGRU`: z, r and q convolutions, called `gru(h, cz, cr, cq, *x)`."""

    convz: torch.nn.Conv2d
    convr: torch.nn.Conv2d
    convq: torch.nn.Conv2d


class FusedConvGRU(torch.nn.Module):
    """`ConvGRU` with the z and r convolutions as one, gates and update in one kernel each (same math)."""

    def __init__(self, gru: ConvGRULike, lib: MetalLibrary) -> None:
        """Take over the weights of `gru`."""
        super().__init__()
        self.lib = lib
        convz, convr = gru.convz, gru.convr
        k = convz.weight.shape[-1]  # the ConvGRU's square kernel, `padding=k // 2`
        self.convzr = torch.nn.Conv2d(convz.in_channels, 2 * convz.out_channels, k, padding=k // 2)
        self.convzr.weight.data = torch.cat([convz.weight, convr.weight])
        self.convzr.bias = torch.nn.Parameter(torch.cat([cast("torch.Tensor", c.bias) for c in (convz, convr)]))
        self.convq = gru.convq

    def forward(
        self, h: torch.Tensor, cz: torch.Tensor, cr: torch.Tensor, cq: torch.Tensor, *x: torch.Tensor
    ) -> torch.Tensor:
        """The new state from state `h`, the context terms `cz`, `cr`, `cq` and the inputs `x`."""
        dt = self.convq.weight.dtype  # the convs' dtype: also the state's under autocast
        h, cz, cr, cq = (t.to(dt).contiguous() for t in (h, cz, cr, cq))
        hx = torch.cat([h, *x], dim=1).to(dt).contiguous()  # the kernels index the flat NCHW buffers
        zr = self.convzr(hx).contiguous()
        _, c, hh, ww = h.shape
        z = torch.empty_like(h)
        # hx's h channels become r * h: the input of the q convolution
        self.lib.kernel("gru_gates", dt)(z, hx, zr, cz, cr, h, c, hh * ww, hx.shape[1], threads=h.numel())
        q = self.convq(hx)
        out = torch.empty_like(h)
        self.lib.kernel("gru_update", dt)(out, h, z, q.contiguous(), cq, threads=h.numel())
        return out


def fuse_grus(net: torch.nn.Module, gru_type: type[torch.nn.Module], lib: MetalLibrary) -> int:
    """Replace every `gru_type` submodule of `net` by a `FusedConvGRU`, in place; the number replaced.

    Raises:
        ValueError: `net` has no `gru_type` submodule.
    """
    found = [(parent, name, m) for parent in net.modules() for name, m in parent.named_children()]
    grus = [(parent, name, m) for parent, name, m in found if base_type(m) is gru_type]
    if not grus:
        msg = f"no {gru_type.__name__} in {type(net).__name__} to fuse"
        raise ValueError(msg)
    for parent, name, m in grus:
        setattr(parent, name, FusedConvGRU(cast("ConvGRULike", m), lib))
    return len(grus)


class FlatCorr:
    """`CorrBlockGather1D` with the pyramid written into one flat buffer and the lookup as one kernel (same values)."""

    lib: MetalLibrary  # set by a subclass (RAFT-Stereo's `patch`)

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4) -> None:
        """All-pairs volume of `fmap1` x `fmap2` (scaled by 1/sqrt(D)) and its 1-D average-pooled levels."""
        b, d, h, w1 = fmap1.shape
        n, widths = b * h * w1, [fmap2.shape[-1]]
        for _ in range(1, num_levels):
            widths.append(widths[-1] // 2)
        self.radius, self.shape, self.levels = radius, (b, h, w1), num_levels
        self.vol = torch.empty(n * sum(widths), dtype=fmap1.dtype, device=fmap1.device)
        meta, o, prev = [], 0, None
        for w in widths:
            level = self.vol[o : o + n * w].view(n, w)
            if prev is None:
                torch.matmul(fmap1.permute(0, 2, 3, 1), fmap2.permute(0, 2, 1, 3), out=level.view(b, h, w1, w))
                level.mul_(d**-0.5)
            else:
                torch.mean(prev[:, : 2 * w].reshape(n, w, 2), dim=-1, out=level)
            meta += [o, w]
            o += n * w
            prev = level
        self.meta = torch.tensor(meta, dtype=torch.int32, device=fmap1.device)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """The `levels * (2 radius + 1)` interpolated correlations around each `coords` x, `[B, C, H, W1]`."""
        b, h, w1 = self.shape
        out = torch.empty(b, self.levels * (2 * self.radius + 1), h, w1, dtype=self.vol.dtype, device=self.vol.device)
        self.lib.kernel("corr_lookup", self.vol.dtype)(
            out, self.vol, coords.float().contiguous(), self.meta, h * w1, self.levels, self.radius, threads=out.numel()
        )
        return out


class DispCorr(FlatCorr):
    """DEFOM-Stereo's `CorrBlock1D` on the `FlatCorr` pyramid: either lookup around a disparity as one kernel."""

    def __init__(  # noqa: PLR0913, PLR0917 - the `CorrBlock1D` arguments
        self,
        lib: MetalLibrary,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int,
        radius: int,
        scale_list: Sequence[float],
        scale_radius: int,
    ) -> None:
        """All-pairs volume and pyramid of `fmap1` x `fmap2`; `scale_list`: the disparity scales of the scale lookup."""
        super().__init__(fmap1, fmap2, num_levels, radius)
        self.lib, self.scale_radius = lib, scale_radius
        dev = fmap1.device
        self.pyramid = torch.tensor([(lvl, 1.0) for lvl in range(num_levels)], dtype=torch.float32, device=dev)
        self.scales = torch.tensor([(0.0, s) for s in scale_list], dtype=torch.float32, device=dev)

    def lookup(self, disp: torch.Tensor, *, scaling: bool) -> torch.Tensor:
        """`[B, C, H, W1]`: the levels at x - disp, or (`scaling`) level 0 at x - scale * disp for each scale.

        C = levels * (2 radius + 1), or scales * (2 scale_radius + 1).
        """
        b, h, w1 = self.shape
        seg, r = (self.scales, self.scale_radius) if scaling else (self.pyramid, self.radius)
        out = torch.empty(b, len(seg) * (2 * r + 1), h, w1, dtype=self.vol.dtype, device=self.vol.device)
        self.lib.kernel("disp_lookup", self.vol.dtype)(
            out, self.vol, disp.float().contiguous(), self.meta, seg, h * w1, len(seg), r, threads=out.numel()
        )
        return out


class LineLookup:
    """The lookups of an IGEV-family `FastGeoEncodingVolume` as one Metal kernel per output (same values).

    Each output is a list of `(volume, l)` segments, concatenated over channels: a geometry volume `[P, C, D]` sampled
    at disparity / 2^l, or a correlation volume `[P, W]` sampled at x / 2^l - disparity / 2^l (x: the pixel's column),
    each at the `2 radius + 1` taps around that position. The volumes are copied into one geometry (`geo_dtype`:
    float32, or float16 for a float16 lookup) and one correlation buffer.
    """

    def __init__(  # noqa: PLR0913 - the layout of the outputs plus the lookup settings
        self,
        lib: MetalLibrary,
        shape: tuple[int, int, int],
        radius: int,
        out_dtype: torch.dtype,
        outputs: Sequence[Sequence[tuple[torch.Tensor, int]]],
        *,
        geo_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Lay out the volumes of `outputs` (`shape`: the batch and lookup grid `(B, H, W)`).

        Raises:
            ValueError: A correlation volume is not in the lookup's dtype (float32 lookups read float32 ones, float16
                lookups float16 ones).
        """
        self.lib, self.shape, self.radius, self.out_dtype = lib, shape, radius, out_dtype
        device, k = outputs[0][0][0].device, 2 * radius + 1
        vols: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])  # geometry, correlation
        sizes, self.seg, self.channels = [0, 0], [], []
        for segments in outputs:
            table, ch = [], 0
            for vol, level in segments:
                src = int(vol.dim() == _CORR_DIMS)
                if src and vol.dtype != out_dtype:
                    msg = f"LineLookup: a {out_dtype} lookup reads a {vol.dtype} correlation volume"
                    raise ValueError(msg)
                c = 1 if src else vol.shape[1]
                table += [ch, src, sizes[src], c, vol.shape[-1], level]
                vols[src].append(vol.reshape(-1).to(geo_dtype) if src == 0 else vol.reshape(-1))
                sizes[src] += vol.numel()
                ch += c * k
            self.seg.append(torch.tensor(table, dtype=torch.int32, device=device))
            self.channels.append(ch)
        empty = torch.zeros(1, dtype=geo_dtype, device=device)
        self.geo = torch.cat(vols[0]) if vols[0] else empty
        self.corr = torch.cat(vols[1]) if vols[1] else empty.to(out_dtype)
        self.name = "line_lookup" if geo_dtype == torch.float32 else "line_lookup_hgeo"

    def __call__(self, disp: torch.Tensor) -> list[torch.Tensor]:
        """The outputs at disparity `disp` (`[B, 1, H, W]`), each `[B, C, H, W]` (channels_last views)."""
        b, h, w = self.shape
        d = disp.reshape(-1).float().contiguous()
        kernel = self.lib.kernel(self.name, self.out_dtype)
        outs = []
        for seg, ct in zip(self.seg, self.channels, strict=True):
            out = torch.empty(b, h, w, ct, dtype=self.out_dtype, device=d.device)
            kernel(out, self.geo, self.corr, d, seg, seg.numel() // 6, ct, w, self.radius, threads=out.numel())
            outs.append(out.permute(0, 3, 1, 2))
        return outs


def gwc_volume(lib: MetalLibrary, left: torch.Tensor, right: torch.Tensor, depth: int, groups: int) -> torch.Tensor:
    """`[B, G, D, H, W]`: per group the mean of `left(x) * right(x - d)` over its channels, 0 for `x < d`."""
    b, c, h, w = left.shape
    if c % groups or right.shape != left.shape or right.dtype != left.dtype:
        msg = f"gwc_volume: {groups} groups, left {tuple(left.shape)} {left.dtype}, right {tuple(right.shape)}"
        msg += f" {right.dtype}"
        raise ValueError(msg)
    out = torch.empty(b, groups, depth, h, w, dtype=left.dtype, device=left.device)
    lib.kernel("gwc_volume", left.dtype)(
        out, left.contiguous(), right.contiguous(), groups, c, depth, h, w, threads=out.numel()
    )
    return out


def win_rpe_attn(  # noqa: PLR0913 - the window geometry
    lib: MetalLibrary,
    qkv: torch.Tensor,
    table: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    heads: int,
    ws: int,
    n: int,
    grid: tuple[int, int],
    shift: int,
    scale: float,
) -> torch.Tensor:
    """`fastbb.win_rpe_attn` (same arguments after `lib`, same math) in float16 or float32.

    Raises:
        ValueError: Shapes or dtypes the kernel cannot run.
    """
    hc, wc = grid
    t, c3 = ws * ws * n, qkv.shape[-1]
    if c3 != 96 * heads or hc % ws or wc % ws or qkv.numel() % (hc * wc * n * c3) or not 0 <= shift < min(hc, wc):
        msg = f"win_rpe_attn: head_dim 32 on whole windows, got {tuple(qkv.shape)}, grid {grid}, shift {shift}"
        raise ValueError(msg)
    if table.dtype != qkv.dtype or table.shape != ((2 * ws - 1) ** 2, c3):
        msg = f"win_rpe_attn: [{(2 * ws - 1) ** 2}, {c3}] {qkv.dtype} table, got {tuple(table.shape)} {table.dtype}"
        raise ValueError(msg)
    if mask is None:
        mask = qkv.new_zeros(1, t, t, dtype=torch.float32)
    elif mask.shape[1:] != (t, t):
        msg = f"win_rpe_attn: [nW, {t}, {t}] mask, got {tuple(mask.shape)}"
        raise ValueError(msg)
    out = qkv.new_empty(*qkv.shape[:-1], c3 // 3)
    meta = torch.tensor([heads, t, ws, n, hc, wc, shift, mask.shape[0]], dtype=torch.int32, device=qkv.device)
    lib.kernel("win_rpe_attn", qkv.dtype)(
        out, qkv.contiguous(), table.contiguous(), mask.float().contiguous(), meta, scale, threads=out.numel() // 32
    )
    return out


class HitnetCosts:
    """HITNet's `init_cost`, `tile_sad` and `refine_sad` as one kernel each (same values; float32 outputs)."""

    def __init__(self, lib: MetalLibrary) -> None:
        """Use `lib`'s `hit_*` kernels."""
        self.lib = lib

    def init_cost(self, lt: torch.Tensor, rt: torch.Tensor, max_disp: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The first minimizing disparity and its cost, float32 `[N, 1, h, w]` each."""
        n, c, h, w = lt.shape
        d, cost = (torch.empty(n, 1, h, w, dtype=torch.float32, device=lt.device) for _ in range(2))
        self.lib.kernel("hit_init_cost", lt.dtype)(
            d, cost, lt.contiguous(), rt.to(lt.dtype).contiguous(), c, h, w, rt.shape[3], max_disp, threads=d.numel()
        )
        return d, cost

    def tile_sad(self, hyp: torch.Tensor, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        """64-channel tile cost `[N, 64, h, w]` of hypothesis `hyp` `[N, 16, h, w]`."""
        n, _, h, w = hyp.shape
        c = fl.shape[1]
        out = torch.empty(n, 64, h, w, dtype=torch.float32, device=hyp.device)
        self.lib.kernel("hit_tile_sad", fl.dtype)(
            out,
            hyp.float().contiguous(),
            fl.contiguous(),
            fr.to(fl.dtype).contiguous(),
            c,
            h,
            w,
            fr.shape[3],
            threads=n * fl.shape[2] * fl.shape[3],
        )
        return out

    def refine_sad(self, d: torch.Tensor, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        """SADs `[N, 3, H, W]` at disparities `d`, `d + 1`, `d - 1`."""
        n, c, h, w = fl.shape
        out = torch.empty(n, 3, h, w, dtype=torch.float32, device=fl.device)
        self.lib.kernel("hit_refine_sad", fl.dtype)(
            out,
            d.float().contiguous(),
            fl.contiguous(),
            fr.to(fl.dtype).contiguous(),
            c,
            h,
            w,
            fr.shape[3],
            threads=n * h * w,
        )
        return out


class WarpVolume:
    """CFNet's `stage_volume` as one kernel: the warped right features are never materialized (same values)."""

    def __init__(self, lib: MetalLibrary) -> None:
        """Use `lib`'s `warp_volume`."""
        self.lib = lib

    def __call__(  # noqa: PLR0913 - CFNet's `StageVolume`
        self,
        net: object,  # noqa: ARG002 - upstream's volume calls into the network, this one does not
        feats: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ds: torch.Tensor,
        *,
        level: int,
        a: int,
        groups: int,
    ) -> torch.Tensor:
        """CFNet's volume `[B, G + 2 Cc + 1, D, h, W]` of feature rows `[a, a + h)` at samples `ds` (`[B, D, h, W]`)."""
        gl, gr, cl, cr = (f[k].contiguous() for k in (f"gw{level}", f"concat_feature{level}") for f in feats)
        b, d, h, w = ds.shape
        cc = cl.shape[1]
        out = torch.empty(b, groups + 2 * cc + 1, d, h, w, dtype=gl.dtype, device=gl.device)
        self.lib.kernel("warp_volume", gl.dtype)(
            out, gl, gr, cl, cr, ds.contiguous(), groups, gl.shape[1], cc, d, gl.shape[2], w, a, h, threads=out.numel()
        )
        return out


def patch_igev(net: torch.nn.Module, geo_type: type, gru_type: type[torch.nn.Module] | None, lib: MetalLibrary) -> None:
    """Switch an IGEV-family network (portable, NCHW) to the Metal kernels; in place.

    `net.geo_volume` (read by the family's forward pass) builds `geo_type` (its `FastGeoEncodingVolume`), then does
    its lookups as `LineLookup`s; IGEV++'s volume (extra geometry volumes `g1`, `g2` sampled at 1/2 and 1/4) returns
    its four outputs, the others one. Every `gru_type` (its `ConvGRU`; None for a family without one) becomes a
    `FusedConvGRU`. In float16 the
    3-D transposed convolutions run in float32, as in portable mode (MPS has no float16 kernel for them).
    """

    def geo_volume(*args: object, **kwargs: object) -> Callable[[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        v = geo_type(*args, **kwargs)
        levels = [(g, i) for i, g in enumerate(v.geo)], [(c, i) for i, c in enumerate(v.corr)]
        if hasattr(v, "g1"):  # IGEV++: geo0 levels, geo1, geo2, correlation levels
            return LineLookup(lib, v.shape, v.radius, v.out_dtype, [levels[0], [(v.g1, 1)], [(v.g2, 2)], levels[1]])
        segments = [s for pair in zip(*levels, strict=True) for s in pair]  # per level: geometry, then correlation
        lookup = LineLookup(lib, v.shape, v.radius, v.out_dtype, [segments])
        return lambda disp: lookup(disp)[0]

    object.__setattr__(net, "geo_volume", geo_volume)  # a plain attribute (Module's __setattr__ types modules)
    if gru_type is not None:
        fuse_grus(net, gru_type, lib)
    mps.adapt(net)


class DeformConv:
    """AANet's `modulated_deform_conv` with the sampling, masking and cast as one kernel into the column buffer."""

    chunk = 1 << 24  # elements of one chunk's column buffer, as in the portable function

    def __init__(self, lib: MetalLibrary) -> None:
        """Use `lib`'s `deform_cols` kernel."""
        self.lib = lib

    def __call__(  # noqa: PLR0913, PLR0917 - the signature of the portable `modulated_deform_conv`
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor | None,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        deformable_groups: int = 1,
    ) -> torch.Tensor:
        """`[B, O, Ho, Wo]` (a channels-last view) in the dtype of `x`; arguments as the portable function's.

        Args:
            x: Input `[B, C, H, W]`.
            offset: `[B, deformable_groups * 2 * K, Ho, Wo]`, (dy, dx) per group and tap.
            mask: `[B, deformable_groups * K, Ho, Wo]`, or None (no modulation).
            weight: `[O, C, kh, kw]`.
            bias: `[O]` or None.
            stride: Stride, int or (y, x).
            padding: Padding, int or (y, x).
            dilation: Dilation, int or (y, x).
            groups: Must be 1.
            deformable_groups: Offset groups over the channels.

        Raises:
            NotImplementedError: `groups > 1`.
        """
        if groups != 1:
            msg = "deformable convolution with groups > 1"
            raise NotImplementedError(msg)
        (sy, sx), (py, px), (dy, dx) = (p if isinstance(p, tuple) else (p, p) for p in (stride, padding, dilation))
        b, c, h, w = x.shape
        o, _, kh, kw = weight.shape
        k = kh * kw
        ho = (h + 2 * py - dy * (kh - 1) - 1) // sy + 1
        wo = (w + 2 * px - dx * (kw - 1) - 1) // sx + 1
        dt, dev = x.dtype, x.device
        wt = weight.reshape(o, c * k).t().to(dt)
        rows = max(1, min(ho, self.chunk // (c * k * wo)))
        out = torch.empty(b, ho, wo, o, device=dev, dtype=dt)
        cols = torch.empty(c * k * rows * wo, device=dev, dtype=dt)
        run = self.lib.kernel("deform_cols", dt)
        for i in range(b):
            src, off = x[i].contiguous(), offset[i].float().contiguous()
            m = off if mask is None else mask[i].to(dt).contiguous()  # unread without a mask
            for r0 in range(0, ho, rows):
                n = min(ho, r0 + rows) - r0
                geo = [c, h, w, ho, wo, kw, k, deformable_groups, sy, sx, py, px, dy, dx, r0, n, mask is not None]
                cn = cols[: c * k * n * wo]
                run(cn, src, off, m, torch.tensor(geo, dtype=torch.int32, device=dev), threads=cn.numel())
                torch.matmul(cn.view(c * k, n * wo).t(), wt, out=out[i, r0 : r0 + n].view(n * wo, o))
        if bias is not None:
            out.add_(bias.to(dt))
        return out.permute(0, 3, 1, 2)


class CrestereoOps:
    """CREStereo's portable `_corr_iter` and `_gru` with the Metal kernels (same signatures, same math).

    The correlation as a warp and a window kernel, the GRU gates and update as one kernel each.
    """

    groups = 4

    def __init__(self, lib: MetalLibrary) -> None:
        """Use `lib`'s `agcl_*` and `gru_*` kernels."""
        self.lib = lib

    def corr_iter(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        flow: torch.Tensor,
        small_patch: bool,  # noqa: FBT001 - upstream's flag
    ) -> torch.Tensor:
        """Group correlations `[N, 4 * 9, H, W]` over a 3 x 3 (`small_patch`) or 1 x 9 window, dtype of `fmap1`.

        `fmap2` is sampled in float32 (the portable path's fp32 copy; no cast when it is one).
        """
        n, c, h, w = fmap1.shape
        ry, rx = (1, 1) if small_patch else (0, 4)
        fmap1 = fmap1.contiguous()
        warped = torch.empty_like(fmap1)
        self.lib.kernel("agcl_warp", fmap1.dtype)(
            warped, fmap2.float().contiguous(), flow.float().contiguous(), c, h, w, threads=n * h * w
        )
        out = torch.empty(n, self.groups * 9, h, w, dtype=fmap1.dtype, device=fmap1.device)
        self.lib.kernel("agcl_window", fmap1.dtype)(
            out, fmap1, warped, self.groups, c, h, w, ry, rx, threads=out.numel()
        )
        return out

    def gru(  # noqa: PLR0913, PLR0917 - the portable `_gru`'s signature
        self,
        hidden: torch.Tensor,
        motion: torch.Tensor,
        w_zr: torch.Tensor,
        w_q: torch.Tensor,
        pad: tuple[int, int],
        c_zr: torch.Tensor,
        c_q: torch.Tensor,
    ) -> torch.Tensor:
        """One `SepConvGRU` half step (the portable `_gru`)."""
        dt, (_, c, h, w) = hidden.dtype, hidden.shape
        hidden, c_q = hidden.contiguous(), c_q.to(dt).contiguous()
        cz, cr = (t.to(dt).contiguous() for t in c_zr.chunk(2, 1))
        hx = torch.cat([hidden, motion.to(dt)], 1).contiguous()  # the kernels index the flat NCHW buffers
        zr = F.conv2d(hx, w_zr, padding=pad).contiguous()
        z = torch.empty_like(hidden)
        # hx's hidden channels become r * hidden: the input of the q convolution
        self.lib.kernel("gru_gates", dt)(z, hx, zr, cz, cr, hidden, c, h * w, hx.shape[1], threads=hidden.numel())
        q = F.conv2d(hx, w_q, padding=pad).contiguous()
        out = torch.empty_like(hidden)
        self.lib.kernel("gru_update", dt)(out, hidden, z, q, c_q, threads=hidden.numel())
        return out
