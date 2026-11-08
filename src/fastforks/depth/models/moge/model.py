"""MoGe runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `MoGeModel.infer` (`resolution_level` -> number of ViT tokens, `apply_mask=False`,
`force_projection=True`): the image is resized (bilinear, antialiased) to a 14-pixel patch grid of about
`1200 + 2400 * resolution_level / 9` tokens with the input's aspect ratio; the network predicts an affine point map
`(x / z, y / z, log z)` at 16x the token grid (MoGe-3 then refines `log z` three times with its sparse 3D UNet), a
validity mask and a metric scale. The maps are upsampled to the input size (bilinear) and post-processed in float32:

- the focal length and z shift that best make the points project to their pixels (upstream `recover_focal_shift`:
  least squares over a 64x64 nearest-neighbour subsample of the masked points); upstream's SciPy Levenberg-Marquardt
  stops at a relative cost change of 1e-3, here the minimum is solved to convergence (float64)
- with given intrinsics only the shift is solved, for the focal length of `fx` (upstream `fov_x`); the points are
  unprojected with square pixels and the principal point at the image centre, so intrinsics with `fy != fx` (> 1 %),
  skew, or a principal point off the centre (> 5 % of the size) raise `InputError`; the result's `intrinsics` is always
  the camera the points use
- mask = sigmoid > 0.5 and depth > 0; depth and points (unprojected from the depth) times the metric scale

The estimated intrinsics have the principal point at the image centre and square pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import InputError
from fastforks._core.half import half_weights
from fastforks._core.patch import base_type, override
from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import sparse_conv
from fastforks.shared.depth_anything_v2 import fold_layer_scale, sdpa_attention

from .net import PATCH, ConvStack, MoGeNet
from .sparse import Level, SparseKernels, SubmConv3d

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

    from .sparse import SparseUNet

_TOKENS = (1200, 3600)  # num_tokens_range of the released checkpoints
_REFINE_STEPS = 3  # MoGe-3 infer default
_DEPTH_RES = 256  # MoGe-3 refiner_depth_resolution: z bins per unit of log depth
_FIT = (64, 64)  # recover_focal_shift subsample


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: str  # "vits" or "vitl"
    version: int  # 2 or 3 (sparse refiner)
    resolution_level: int  # 0-9: 1200-3600 ViT tokens
    normals: bool  # also run the normal head


def grid(h: int, w: int, resolution_level: int) -> tuple[int, int]:
    """Token rows and columns for an `h x w` image (upstream `MoGeModel.forward`)."""
    n = int(_TOKENS[0] + resolution_level / 9 * (_TOKENS[1] - _TOKENS[0]))
    a = w / h
    return round((n / a) ** 0.5), round((n * a) ** 0.5)


def view_plane_uv(w: int, h: int, aspect: float, device: torch.device | str) -> torch.Tensor:
    """Upstream `normalized_view_plane_uv`: `[h, w, 2]` pixel centres, the half diagonal = 1."""
    u, v = _plane_axes(w, h, aspect, device)
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), -1)


def _plane_axes(w: int, h: int, aspect: float, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """The `u` (`[w]`) and `v` (`[h]`) axes of `view_plane_uv`."""
    sx, sy = aspect / (1 + aspect**2) ** 0.5, 1 / (1 + aspect**2) ** 0.5
    u = torch.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w, device=device)
    v = torch.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h, device=device)
    return u, v


def _nearest_index(n: int, m: int, device: torch.device | str) -> torch.Tensor:
    """The source indices `F.interpolate(mode="nearest")` reads for `m` of `n` samples (its own rounding, exactly)."""
    i = torch.arange(n, dtype=torch.float32, device=device)[None, None, None]  # exact below 2**24
    return F.interpolate(i, (1, m), mode="nearest")[0, 0, 0].long()


def _net(ctx: Context) -> MoGeNet:
    net = MoGeNet(ctx.params.arch, ctx.params.version)
    net.load_state_dict(ctx.state_dict(key="model", extra={}))
    return net.eval()


# ------------------------------------------------------------------------------------------------ post-processing
def _solve_shift(uv: torch.Tensor, xyz: torch.Tensor, w: torch.Tensor, focal: torch.Tensor | None) -> torch.Tensor:
    """Levenberg-Marquardt for `min_s |f * xy / (z + s) - uv|^2` over the weighted points, per batch item (float64).

    `f` is the given focal length or, if `None`, the optimal one for each `s` (upstream `solve_optimal_focal_shift`).
    """
    xy, z = xyz[..., :2], xyz[..., 2:]
    w = w[..., None]

    def residual(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = z + s[:, None, None]
        p = torch.where(w > 0, xy / d, 0)
        dp = torch.where(w > 0, -p / d, 0)
        if focal is None:
            a, b = (p * uv * w).sum((1, 2)), (p * p * w).sum((1, 2))
            da, db = (dp * uv * w).sum((1, 2)), 2 * (p * dp * w).sum((1, 2))
            f, df = a / b, (da * b - a * db) / (b * b)
        else:
            f, df = focal, torch.zeros_like(focal)
        r = (f[:, None, None] * p - uv) * w
        return r, (df[:, None, None] * p + f[:, None, None] * dp) * w

    s = torch.zeros(xyz.shape[0], dtype=torch.float64)
    lam = torch.full_like(s, 1e-3)
    r, jac = residual(s)
    cost = (r * r).sum((1, 2))
    for _ in range(100):
        g, hess = (jac * r).sum((1, 2)), (jac * jac).sum((1, 2))
        step = -g / (hess * (1 + lam)).clamp_min(1e-300)
        r2, j2 = residual(s + step)
        c2 = (r2 * r2).sum((1, 2))
        ok = torch.isfinite(c2) & (c2 <= cost)
        s = torch.where(ok, s + step, s)
        r, jac = torch.where(ok[:, None, None], r2, r), torch.where(ok[:, None, None], j2, jac)
        cost = torch.where(ok, c2, cost)
        lam = torch.where(ok, lam / 10, lam * 10)
        if torch.compiler.is_compiling():  # traced: all iterations (no host sync); converged steps change s < 1e-13
            continue
        if bool((step.abs() <= 1e-13 * (s.abs() + 1e-9)).all()):
            break
    return s


def focal_shift(
    points: torch.Tensor, mask: torch.Tensor, focal: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `recover_focal_shift` of affine points `[B, H, W, 3]` (float32) and mask `[B, H, W]` (bool).

    Returns (focal relative to the half diagonal, z shift), `[B]` float32 on the points' device.
    """
    b, h, w, _ = points.shape
    near = lambda t: F.interpolate(t, _FIT, mode="nearest")  # noqa: E731
    pl = near(points.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, -1, 3)
    u, v = _plane_axes(w, h, w / h, points.device)  # the uv grid at the nearest samples only, not at every pixel
    u, v = u[_nearest_index(w, _FIT[1], points.device)], v[_nearest_index(h, _FIT[0], points.device)]
    ul = torch.stack(torch.meshgrid(u, v, indexing="xy"), -1).reshape(1, -1, 2)
    ml = near(mask[:, None].float())[:, 0].reshape(b, -1) > 0
    pl, ul, wl = pl.cpu().double(), ul.cpu().double(), ml.cpu().double()  # cast on the CPU: MPS has no float64
    fd = focal.cpu().double() if focal is not None else None
    shift = _solve_shift(ul, pl, wl, fd)
    few = wl.sum(1) < 2  # noqa: PLR2004 - upstream: fewer than 2 points -> focal 1, shift 0
    shift = torch.where(few, 0, shift).float()
    if fd is None:
        p = pl[..., :2].float() / (pl[..., 2:].float() + shift[:, None, None])
        wf = wl[..., None].float()
        f = (p * ul.float() * wf).sum((1, 2)) / (p * p * wf).sum((1, 2))
        f = torch.where(few, 1, f)
    else:
        f = fd.float()
    return f.to(points.device), shift.to(points.device)


# MoGe accepts relative |fy - fx| / fx (and skew) and |c - size / 2| / size up to these (NYU: cy is 2.9 % off)
_FY_TOL, _CENTRE_TOL = 0.01, 0.05


def check_camera(k: torch.Tensor | None, h: int, w: int) -> None:
    """Reject given intrinsics that MoGe's camera (fx only, square pixels, centred principal point) cannot follow."""
    if k is None:
        return
    fx, fy, skew, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 1], k[:, 0, 2], k[:, 1, 2]
    off = (
        ((fy - fx).abs() > _FY_TOL * fx.abs())
        | (skew.abs() > _FY_TOL * fx.abs())
        | ((cx - (w - 1) / 2).abs() > _CENTRE_TOL * w)
        | ((cy - (h - 1) / 2).abs() > _CENTRE_TOL * h)
    )
    if bool(off.any()):
        i = int(off.nonzero()[0, 0])
        msg = (
            f"MoGe uses only fx of the intrinsics (square pixels, principal point at the image centre); intrinsics "
            f"{i} have fx={float(fx[i]):g}, fy={float(fy[i]):g}, skew={float(skew[i]):g}, "
            f"principal point ({float(cx[i]):g}, {float(cy[i]):g}) for a {h}x{w} image "
            f"(centre ({(w - 1) / 2:g}, {(h - 1) / 2:g})); "
            "crop the image so the principal point is centred, or pass K with fy = fx and the centre"
        )
        raise InputError(msg)


def finish(  # noqa: PLR0913, PLR0917 - the network outputs
    coords: torch.Tensor,
    mask: torch.Tensor,
    normal: torch.Tensor | None,
    scale: torch.Tensor,
    size: Sequence[int],
    k: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Network outputs (float32, NCHW at the network size: `(x/z, y/z, log z)`, mask logits, normals) -> the maps."""
    h, w = size
    a = w / h
    maps = torch.cat([coords, mask] + ([normal] if normal is not None else []), 1)
    maps = F.interpolate(maps, (h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    z = maps[..., 2:3].exp()
    points = torch.cat([maps[..., :2] * z, z], -1)
    valid = maps[..., 3] > 0  # sigmoid > 0.5
    focal = None if k is None else a / (1 + a * a) ** 0.5 * 2 * k[:, 0, 0] / w  # from fx (upstream fov_x)
    focal, shift = focal_shift(points, valid, focal)
    fx, fy = focal / 2 * (1 + a * a) ** 0.5 / a * w, focal / 2 * (1 + a * a) ** 0.5 * h  # pixels
    depth = points[..., 2] + shift[:, None, None]
    valid &= depth > 0
    depth = depth * scale[:, None, None]
    u = (torch.arange(w, device=depth.device) + 0.5 - w / 2)[None, None, :] / fx[:, None, None]
    v = (torch.arange(h, device=depth.device) + 0.5 - h / 2)[None, :, None] / fy[:, None, None]
    res = {"depth": depth, "points": torch.stack([u * depth, v * depth, depth], -1), "mask": valid}
    kk = torch.zeros((depth.shape[0], 3, 3), device=depth.device)  # the camera of the points, also for given K
    kk[:, 0, 0], kk[:, 1, 1], kk[:, 0, 2], kk[:, 1, 2], kk[:, 2, 2] = fx, fy, (w - 1) / 2, (h - 1) / 2, 1
    res["intrinsics"] = kk
    if normal is not None:
        res["normals"] = F.normalize(maps[..., 4:7], dim=-1)
    return res


def refine(
    refiner: SparseUNet, coords: torch.Tensor, enc: torch.Tensor, dtype: torch.dtype, steps: int = _REFINE_STEPS
) -> torch.Tensor:
    """MoGe-3: refine `log z` of `coords` `[B, 3, h, w]` (float32) `steps` times; `enc`: `[B, C, h / 16, w / 16]`."""
    b, _, h, w = coords.shape
    c = coords.permute(0, 2, 3, 1)
    for _ in range(steps):
        logz = c[..., 2]
        zq = torch.round(logz * _DEPTH_RES).long()
        zq -= zq.amin((1, 2), keepdim=True)
        out = refiner(c.reshape(-1, 3).to(dtype), zq, enc.to(dtype))
        c = torch.cat([c[..., :2], (logz + out.float().view(b, h, w))[..., None]], -1)
    return c.permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------ portable
def _rep_pad1(x: torch.Tensor) -> torch.Tensor:
    """Replicate-pad the last two dims by 1 (as `F.pad(mode="replicate")`, which is slower than slices + cat)."""
    x = torch.cat([x[..., :1, :], x, x[..., -1:, :]], -2)
    return torch.cat([x[..., :1], x, x[..., -1:]], -1)


def _rep_conv(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """`nn.Conv2d` with 1-pixel replicate padding (the conv itself runs unpadded)."""
    return F.conv2d(_rep_pad1(x), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class Portable(Runner):
    """The upstream modules with fused SDPA attention: float32, or fp16 autocast.

    In float16 the point / normal output convs, the metric-scale MLP and the post-processing stay in float32, and so
    does the MoGe-3 refiner except on CUDA (fp16 there, as upstream's `use_fp16` and the kernels mode run it).
    """

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    intrinsics = "optional"

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.device)
        sdpa_attention(self.net)
        fold_layer_scale(self.net)
        self.half = ctx.dtype == "float16"
        if self.half:
            # fp16 Linear / Conv weights (outputs bit-identical under the autocast below); the output convs stay fp32
            net = self.net
            half_weights(net.encoder, channels_last=False)
            for m in (net.neck, net.points_head, net.mask_head, net.normal_head):
                half_weights(m)
            net.points_head.output_blocks.float()
            net.normal_head.output_blocks.float()
            if net.refiner is not None and ctx.device.type == "cuda":
                net.refiner.half()
            for m in net.modules():
                if isinstance(m, nn.Conv2d) and m.padding_mode == "replicate" and m.padding == (1, 1):
                    override(m, _rep_conv)

    def _forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Network outputs at its size (see `finish`) and the metric scale `[B]`."""
        net, p = self.net, self.ctx.params
        b, _, h, w = image.shape
        rows, cols = grid(h, w, p.resolution_level)
        a = w / h
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=self.half):
            feat, cls = net.encoder(image, rows, cols)
            uv = [
                view_plane_uv(cols << i, rows << i, a, image.device).permute(2, 0, 1)[None].expand(b, -1, -1, -1)
                for i in range(5)
            ]
            enc = torch.cat([feat, uv[0].to(feat.dtype)], 1)
            levels = net.neck([enc, *uv[1:]])
            coords = net.points_head(levels, fp32_out=True)[-1].float()
            mask = net.mask_head(levels)[-1].float()
            normal = net.normal_head(levels, fp32_out=True)[-1].float() if p.normals else None
        with torch.autocast(image.device.type, enabled=False):
            scale = net.scale_head(cls.float())[:, 0].exp()
            if net.refiner is not None:
                rdt = next(net.refiner.parameters()).dtype
                coords = refine(net.refiner, coords, enc.float(), rdt)
        return coords, mask, normal, scale

    def _run(self, image: torch.Tensor, k: torch.Tensor | None) -> dict[str, torch.Tensor]:
        return finish(*self._forward(image), image.shape[-2:], k)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Depth, points, mask, intrinsics (and normals) of a batch."""
        check_camera(intrinsics, *image.shape[-2:])
        return self._run(image, intrinsics)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> metric depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image, None)["depth"])


# ------------------------------------------------------------------------------------------------ kernels
# kernels-only dependencies (after the portable part)
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from fastforks.depth.models.depth_anything_v2.fused import FusedViT, _linear_1x1, _resize  # noqa: E402 - shared DPT conversions
from fastforks.kernels.dpt_fused import DptFused  # noqa: E402
from fastforks.kernels.dpt_fused import load as load_dpt_fused  # noqa: E402
from fastforks.shared.depth_anything_v2.fast_vit import _flash  # noqa: E402


def _pad(x: torch.Tensor, *, relu: bool) -> torch.Tensor:
    """Replicate-pad an NHWC tensor by 1 (after a ReLU: `relu(pad(x)) = pad(relu(x))`)."""
    b, h, w, c = x.shape
    p = x.new_empty((b, h + 2, w + 2, c))
    inner = p[:, 1:-1, 1:-1]
    if relu:
        torch.clamp_min(x, 0, out=inner)
    else:
        inner.copy_(x)
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    p[:, :, 0] = p[:, :, 1]
    p[:, :, -1] = p[:, :, -2]
    return p


def _rconv(m: nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A 3x3 replicate-padded conv -> (weight `[K, 3, 3, C]` fp16, bias fp16), run on a `_pad`-ed input."""
    if not (
        isinstance(m, nn.Conv2d)
        and m.padding_mode == "replicate"
        and m.kernel_size == (3, 3)
        and m.padding == (1, 1)
        and m.stride == (1, 1)
        and m.groups == 1
        and m.bias is not None
    ):
        msg = f"dpt_fused MoGe: unsupported layer {m}"
        raise ValueError(msg)
    w = m.weight.detach().half().permute(0, 2, 3, 1).contiguous().to(device)
    return w, m.bias.detach().half().to(device)


def _fp32_1x1(m: nn.Conv2d, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A 1x1 conv as fp32 (weight `[C_in, C_out]`, bias)."""
    w, b = m.get_parameter("weight"), m.get_parameter("bias")
    return w.detach().float()[:, :, 0, 0].t().contiguous().to(device), b.detach().float().to(device)


class _FusedStack:
    """A ConvStack in NHWC fp16 on dpt_fused: 3x3 replicate convs as `_pad` + conv, level inputs in the epilogues."""

    def __init__(self, cs: ConvStack, k: DptFused, device: torch.device) -> None:
        """Convert the stack's weights."""
        self.k = k
        self.inp = [_linear_1x1(m, device) for m in cs.input_blocks]
        self.res = [
            [
                (_rconv(rb.get_submodule("layers.2"), device), _rconv(rb.get_submodule("layers.5"), device))
                for rb in seq.children()
            ]
            for seq in cs.res_blocks
        ]
        self.up: list[tuple[tuple | None, tuple[torch.Tensor, torch.Tensor]]] = []
        for r in cs.resamplers:
            first, conv = r.get_submodule("0"), r.get_submodule("1")
            pre = None if isinstance(first, nn.Upsample) else _resize(first, device)
            if pre is not None and pre[0] != "deconv":
                msg = f"dpt_fused MoGe: unsupported resampler {first}"
                raise ValueError(msg)
            self.up.append((pre, _rconv(conv, device)))
        ob = cs.output_blocks[-1]
        self.out = _fp32_1x1(ob, device) if isinstance(ob, nn.Conv2d) else None

    def _conv(self, x: torch.Tensor, c: tuple[torch.Tensor, torch.Tensor], **kw: Any) -> torch.Tensor:  # noqa: ANN401 - epilogue options
        return self.k.conv(x, c[0], c[1], **kw)

    def levels(self, x: torch.Tensor, adds: Sequence[torch.Tensor | None]) -> list[torch.Tensor]:
        """Level features from level 0's (input block applied) `x`; `adds[i]`: level i's input (i >= 1)."""
        k = self.k
        outs = []
        n = len(self.res)
        for i in range(n):
            for c1, c2 in self.res[i]:
                y = self._conv(_pad(x, relu=True), c1)
                x = self._conv(_pad(y, relu=True), c2, res=x, out=x)
            outs.append(x)
            if i < n - 1:
                pre, c = self.up[i]
                _, h, w, _ = x.shape
                if pre is None:
                    u = k.bilinear(x, (2 * h, 2 * w), align_corners=False)
                else:
                    _, wg, bg, s = pre
                    u = k.gemm(x, wg, bg, shuffle=(h, w, s, s))
                x = self._conv(_pad(u, relu=False), c, res=adds[i + 1])
        return outs

    def head(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        """Head over the neck's levels -> its fp32 output `[B, h, w, C_out]` (last level, fp32 1x1 conv)."""
        k = self.k
        adds = [k.gemm(f, w, b) for f, (w, b) in zip(levels, self.inp, strict=True)]
        x = self.levels(adds[0], adds)[-1]
        assert self.out is not None  # noqa: S101 - heads have an output conv
        w, b = self.out
        return torch.addmm(b, x.view(-1, x.shape[-1]).float(), w).view(*x.shape[:-1], -1)


class _MoGeViT(FusedViT):
    """`FusedViT` that also returns the last layer's normalised class token, tracked in float32.

    The metric scale comes from the class token alone, so its fp16 residual stream (and the fp16-accumulating GEMMs)
    gave a uniform ~1 % scale error. The class row keeps a float32 residual: each block's fused GEMMs still update the
    whole fp16 stream (the patch tokens attend to the class token's fp16 copy), and the class row's own increments are
    recomputed from its attention output with cuBLAS GEMVs (fp16 weights, fp32 accumulation) and added in float32.
    """

    def __init__(self, vit: Any, device: torch.device, k: DptFused) -> None:  # noqa: ANN401 - duck-typed DINOv2 module
        """`FusedViT`, plus float32 copies of the LayerNorm parameters the class row uses."""
        super().__init__(vit, device, k)
        self.ln32 = [(n.weight.float(), n.bias.float()) for n in (*(b.norm2 for b in self.blocks), self.vit.norm)]

    def _cls_block(self, i: int, c32: torch.Tensor, att0: torch.Tensor) -> torch.Tensor:
        """Class-token residual `[B, C]` float32 after block `i`, from its attention output `att0` `[B, C]` (fp16)."""
        blk = self.blocks[i]
        a, m = blk.attn, blk.mlp
        g1, g2 = self.gammas[i]
        p = F.linear(att0, a.proj.weight, a.proj.bias)  # fp16 increments, float32 sums
        x = c32 + p if g1 is None else torch.addcmul(c32, g1, p)
        y = F.layer_norm(x, x.shape[-1:], *self.ln32[i], blk.norm2.eps).half()
        f = F.linear(F.gelu(F.linear(y, m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        return x + f if g2 is None else torch.addcmul(x, g2, f)

    def features_cls(
        self, image: torch.Tensor, size: tuple[int, int], layers: Sequence[int]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Normalised patch tokens `[B, gh * gw, C]` (fp16) of `layers`, the last one's class token `[B, C]` (fp32)."""
        vit, k = self.vit, self.k
        ps = vit.patch_size
        gh, gw = size[0] // ps, size[1] // ps
        cols = k.patches(image, size, ps, self.kp, IMAGENET_MEAN, IMAGENET_STD)  # identity resize: exact
        pos, cls = self._pos_tables(gh, gw)
        b, c = image.shape[0], vit.embed_dim
        t = torch.empty((b, gh * gw + 1, c), device=image.device, dtype=torch.float16)
        for i in range(b):
            k.gemm(cols[i], self.w_patch, self.b_patch, res=pos, out=t[i, 1:], acc32=True)
        c32 = cls.float().expand(b, c)
        outs = []
        for i in range(max(layers) + 1):
            blk = self.blocks[i]
            a, m = blk.attn, blk.mlp
            g1, g2 = self.gammas[i]
            t[:, 0] = c32
            h = a.num_heads
            qkv = k.gemm(blk.norm1(t), a.qkv.weight, a.qkv.bias).view(b, t.shape[1], 3, h, c // h)
            att = _flash(qkv, (c // h) ** -0.5, self.attn)
            c32 = self._cls_block(i, c32, att.view(b, -1, c)[:, 0])
            k.gemm(att, a.proj.weight, a.proj.bias, scale=g1, res=t, out=t)
            hid = k.gemm(blk.norm2(t), m.fc1.weight, m.fc1.bias, act=k.gelu)
            k.gemm(hid, m.fc2.weight, m.fc2.bias, scale=g2, res=t, out=t)
            if i in layers:
                outs.append(vit.norm(t[:, 1:]))
        return outs, F.layer_norm(c32, c32.shape[-1:], *self.ln32[-1], vit.norm.eps)


_UV_SIZES = 2  # fixed: cache policy, view-plane tables: ~115 MB per input size (3600 tokens); video always hits


class Kernels(Runner):
    """`dpt_fused` kernels: fused-epilogue CUTLASS GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 neck.

    The input resize (bilinear, antialiased) and the metric-scale MLP run in PyTorch (fp32); the encoder's output
    projections are accumulating GEMMs, the constant view-plane inputs per-size bias tables in the epilogues; the
    point / normal / mask output convs and the post-processing run in fp32, the MoGe-3 refiner in fp16.
    """

    batch = None
    Params = Params
    intrinsics = "optional"

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        dev = ctx.device
        k = load_dpt_fused(dev)  # before the weights: a kernel build / load failure costs no weight load
        net = _net(ctx)
        self.k = k
        enc = net.encoder
        self.layers = list(enc.layers)
        self.vit = _MoGeViT(enc.backbone, dev, k)
        self.proj = [
            p.get_parameter("weight").detach()[:, :, 0, 0].half().contiguous().to(dev) for p in enc.output_projections
        ]
        self.proj_b = sum(p.get_parameter("bias").detach().float() for p in enc.output_projections).half().to(dev)  # ty: ignore[unresolved-attribute] - sum of tensors
        ib = net.neck.input_blocks
        w0 = ib[0].get_parameter("weight").detach()[:, :, 0, 0]
        self.neck_wf = w0[:, :-2].half().contiguous().to(dev)
        self.neck_uv = [
            (m.get_parameter("weight").detach()[:, -2:, 0, 0].float(), m.get_parameter("bias").detach().float())
            for m in ib
        ]  # CPU tables
        self.neck = _FusedStack(net.neck, k, dev)
        self.heads = {n: _FusedStack(getattr(net, n), k, dev) for n in ("points_head", "mask_head", "normal_head")}
        self.scale_head = net.scale_head.float().to(dev)
        self.refiner = net.refiner.half().to(dev) if net.refiner is not None else None
        if self.refiner is not None:
            self.refiner.kern = SparseKernels(sparse_conv.load(dev), self.refiner)
        self._uv: dict[tuple[int, int, float, int], list[torch.Tensor]] = {}

    def _uv_tables(self, rows: int, cols: int, a: float, b: int) -> list[torch.Tensor]:
        """Per level: the neck input conv of the view-plane uv (+ bias) as an fp16 NHWC `[B, h, w, C]` table.

        Keyed by the aspect too: inputs of one token grid differ in aspect (384 x 576 and 400 x 596: 49 x 73 tokens).
        """
        key = (rows, cols, a, b)
        if key not in self._uv:
            if len(self._uv) >= _UV_SIZES:
                del self._uv[next(iter(self._uv))]
            out = []
            for i, (w, bias) in enumerate(self.neck_uv):
                uv = view_plane_uv(cols << i, rows << i, a, "cpu")
                t = torch.addmm(bias, uv.reshape(-1, 2), w.t()).half().view(1, rows << i, cols << i, -1)
                out.append(t.expand(b, -1, -1, -1).contiguous().to(self.ctx.device))
            self._uv[key] = out
        return self._uv[key]

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Depth, points, mask, intrinsics (and normals) of a batch."""
        k, p = self.k, self.ctx.params
        b, _, h, w = image.shape
        check_camera(intrinsics, h, w)
        rows, cols = grid(h, w, p.resolution_level)
        a = w / h
        size = (rows * PATCH, cols * PATCH)
        x = F.interpolate(image, size, mode="bilinear", align_corners=False, antialias=True)
        with torch.autocast("cuda", enabled=False):
            feats, cls = self.vit.features_cls(x, size, self.layers)
            f0 = k.gemm(feats[0], self.proj[0], self.proj_b)
            for t, wp in zip(feats[1:], self.proj[1:], strict=True):
                k.gemm(t, wp, res=f0, out=f0)
            uv = self._uv_tables(rows, cols, a, b)
            n0 = k.gemm(f0, self.neck_wf, res=uv[0]).view(b, rows, cols, -1)
            levels = self.neck.levels(n0, uv)
            coords = self.heads["points_head"].head(levels).permute(0, 3, 1, 2)
            mask = self.heads["mask_head"].head(levels).permute(0, 3, 1, 2)
            normal = self.heads["normal_head"].head(levels).permute(0, 3, 1, 2) if p.normals else None
            scale = self.scale_head(cls.float())[:, 0].exp()
            if self.refiner is not None:
                uv0 = view_plane_uv(cols, rows, a, image.device).permute(2, 0, 1)[None].expand(b, -1, -1, -1)
                enc = torch.cat([f0.view(b, rows, cols, -1).permute(0, 3, 1, 2), uv0.half()], 1)
                coords = refine(self.refiner, coords, enc, torch.float16)
        return finish(coords, mask, normal, scale, (h, w), intrinsics)


# ----------------------------------------------------------------------------------------------- kernels, MPS
def _metal_subm_conv(self: SubmConv3d, x: torch.Tensor, lv: Level) -> torch.Tensor:
    """`SubmConv3d.forward` by the `subm_conv` Metal kernel; the finest level stays a dense conv."""
    if lv.nbr is None:
        return base_type(self).forward(self, x, lv)
    return vars(self)["metal"](x, lv.nbr)  # a `sparse_metal.SubmConv`


class MetalKernels(Portable):
    """Apple MPS: the portable network with the MoGe-3 refiner's sparse convolutions as a Metal kernel."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch the refiner's sparse convs to the Metal kernel."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only

        mps.adapt(self.net)  # the faster transposed convolutions the portable runner gets
        convs = [m for m in self.net.modules() if isinstance(m, SubmConv3d)]
        if not convs:  # MoGe-2 has no refiner: nothing to build a Metal library for
            return
        from fastforks.kernels import sparse_metal  # noqa: PLC0415 - needs the kernel library

        lib = sparse_metal.load(ctx.device)
        for m in convs:
            object.__setattr__(m, "metal", sparse_metal.SubmConv(lib, m.weight, m.bias))  # not a submodule
            override(m, _metal_subm_conv)


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}

__all__ = ["RUNNERS", "finish", "focal_shift", "grid", "view_plane_uv"]
