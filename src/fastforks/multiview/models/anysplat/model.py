"""AnySplat runners: weights -> network and the portable / kernels forward passes.

Upstream `EncoderAnySplat.forward` (eval, the released config: depth head, voxelize, no pruning) on all views at once:
the VGGT aggregator, camera head and depth head (AnySplat's fine-tuned copies), the world points unprojected from the
depth with the predicted cameras (upstream `batchify_unproject_depth_map_to_point_map`), and the Gaussian head
(`GSHead`: AnySplat's `VGGT_DPT_GS_Head`) whose per-pixel parameters are fused per voxel (`voxel_size`, upstream 0.002;
0 keeps one Gaussian per pixel) into the scene's Gaussians. Images are resized to the network size (long side
`long_side`, multiples of 14, bicubic antialiased; upstream crops 448 x 448) unless they already have it; maps are
resized back (bilinear), intrinsics rescaled to the input pixels (pixel centres at integer coordinates). The Gaussians
stay at the network resolution; their means are upstream's points (pixel index `u` against the principal point `W / 2`).

The fp16 and kernels paths are VGGT's (`../vggt/model.py`, copied: families do not import each other): fp16
aggregator with fused q/k-norm + RoPE, fp16 channels-last DPT heads with the last 1x1 conv in fp32, camera head fp32;
kernels: `fastbb` flash attention, fp16-accumulate GEMMs, cuDNN fp16-accumulate convs. Upstream runs the aggregator
under bf16 autocast and the heads in fp32.
"""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels.cudnn_hacc import conv_half_acc
from fastforks.multiview._runner import Forward, Runner, invert
from fastforks.shared.depth_anything_v2 import FastViT, vit_large
from fastforks.shared.rope2d import rope_forward

from .net.heads.camera_head import CameraHead
from .net.heads.dpt_head import DPTHead
from .net.models.aggregator import Aggregator, slice_expand_and_flatten
from .net.utils.pose_enc import pose_encoding_to_extri_intri

if TYPE_CHECKING:
    from fastforks.multiview._runner import Context

PATCH = 14  # fixed: the ViT patch size
EMBED = 1024  # fixed: the aggregator width (one checkpoint)
SH_DEGREE = 4  # fixed: the released config (gaussian_adapter.sh_degree)
SH = (SH_DEGREE + 1) ** 2  # fixed: SH coefficients per channel
GS_DIM = 1 + 3 + 4 + 3 * SH  # fixed: per pixel: opacity, scales, rotation (x y z w), SH; the head adds 1 confidence
RAW = "gaussians/raw"  # `forward`: per-pixel head output [G, GS_DIM + 1] (G = V * hn * wn, view-major)
POINTS = "gaussians/points"  # `forward`: per-pixel world points [G, 3] at the network size


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (both sides rounded to multiples of 14)
    voxel_size: float  # Gaussian fusion voxel edge in scene units (0: one Gaussian per pixel)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: long side `long_side`, both sides multiples of 14."""
    s = long_side / max(h, w)
    return (max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH))


class GSHead(DPTHead):
    """AnySplat's `VGGT_DPT_GS_Head` (eval forward; reimplemented on the VGGT `DPTHead`, same attribute names).

    The DPT fusion at 1/1 + a 7x7 conv of the images, then `output_conv2` -> raw parameters `[B, S, GS_DIM + 1, H, W]`
    (no activation: the last channel is the fusion confidence).
    """

    def __init__(self, dim_in: int) -> None:  # noqa: D107
        super().__init__(dim_in, output_dim=GS_DIM + 1, activation="norm_exp", conf_activation="expp1", features=256)
        self.input_merger = nn.Sequential(nn.Conv2d(3, 128, 7, 1, 3), nn.ReLU())
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, GS_DIM + 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(  # ty: ignore[invalid-method-override] - upstream's own signature
        self, tokens: list[torch.Tensor | None], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int = 8
    ) -> torch.Tensor:
        """Aggregator tokens + images `[B, S, 3, H, W]` -> `[B, S, GS_DIM + 1, H, W]`, in chunks of 8 views."""
        s = images.shape[1]
        if s <= frames_chunk_size:
            return self._gs(tokens, images, patch_start_idx, 0, s)
        chunks = range(0, s, frames_chunk_size)
        return torch.cat(
            [self._gs(tokens, images, patch_start_idx, i, min(i + frames_chunk_size, s)) for i in chunks], 1
        )

    def _gs(self, tokens: list[torch.Tensor | None], imgs: torch.Tensor, psi: int, s0: int, s1: int) -> torch.Tensor:
        imgs = imgs[:, s0:s1]
        b, s, _, h, w = imgs.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        out = []
        for i, layer in enumerate(self.intermediate_layer_idx):
            t = tokens[layer]
            if t is None:
                msg = f"AnySplat: aggregator layer {layer} is not cached"
                raise RuntimeError(msg)
            x = self.norm(t[:, s0:s1, psi:].reshape(b * s, -1, t.shape[-1]))
            x = self.projects[i](x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], ph, pw))
            if self.pos_embed:
                x = self._apply_pos_embed(x, w, h)
            out.append(self.resize_layers[i](x))
        x = F.interpolate(self.scratch_forward(out), size=(h, w), mode="bilinear", align_corners=True)
        x = x + self.input_merger(imgs.flatten(0, 1))
        if self.pos_embed:
            x = self._apply_pos_embed(x, w, h)
        scratch: Any = self.scratch
        x = scratch.output_conv2(x)
        return x.view(b, s, *x.shape[1:])


class Net(nn.Module):
    """AnySplat's encoder at inference (same attribute names, so the checkpoint keys match after `encoder.`)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.aggregator = Aggregator(img_size=518, patch_size=PATCH, embed_dim=EMBED)
        self.camera_head = CameraHead(dim_in=2 * EMBED)
        self.depth_head = DPTHead(dim_in=2 * EMBED, output_dim=2, activation="exp", conf_activation="expp1")
        self.gaussian_param_head = GSHead(2 * EMBED)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Images `[1, S, 3, H, W]` in `[0, 1]` -> pose encoding, depth + confidence, raw Gaussian parameters."""
        toks, psi = self.aggregator(images)
        pose = self.camera_head(toks)[-1]
        depth, depth_conf = self.depth_head(toks, images=images, patch_start_idx=psi)
        return {
            "pose_enc": pose,
            "depth": depth,
            "depth_conf": depth_conf,
            "gs": self.gaussian_param_head(toks, images, psi),
        }


def _net(ctx: Context) -> Net:
    sd = {k.removeprefix("encoder."): v for k, v in ctx.state_dict().items()}  # bf16 + fp32: cast on copy
    net = Net()
    net.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def unproject(depth: torch.Tensor, ext: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Upstream `batchify_unproject_depth_map_to_point_map`: depth `[S, H, W]`, w2c `[S, 3, 4]`, K -> `[S, H, W, 3]`."""
    s, h, w = depth.shape
    u = torch.arange(w, device=depth.device, dtype=depth.dtype)[None, None, :]
    v = torch.arange(h, device=depth.device, dtype=depth.dtype)[None, :, None]
    fu, fv, cu, cv = (k[:, i, j, None, None] for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
    cam = torch.stack(((u - cu) * depth / fu, (v - cv) * depth / fv, depth), -1)
    w2c = torch.eye(4, device=ext.device, dtype=ext.dtype).repeat(s, 1, 1)
    w2c[:, :3] = ext
    c2w = invert(w2c)
    return torch.einsum("sij,shwj->shwi", c2w[:, :3, :3], cam) + c2w[:, None, None, :3, 3]


def outputs(pred: dict[str, torch.Tensor], hn: int, wn: int, h0: int, w0: int) -> dict[str, torch.Tensor]:
    """Upstream predictions at the network size `hn x wn` -> the maps at the input size `h0 x w0` + per-pixel raw."""

    def up(x: torch.Tensor) -> torch.Tensor:  # [S, C, hn, wn] -> [S, C, h0, w0]
        x = x.float()
        return x if (hn, wn) == (h0, w0) else F.interpolate(x, (h0, w0), mode="bilinear", align_corners=False)

    ext, k = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (hn, wn))
    ext, k = ext[0], k[0]
    depth = pred["depth"][0, ..., 0].float()
    pts = unproject(depth, ext, k)
    ki = k.clone()
    ki[:, 0] *= w0 / wn
    ki[:, 1] *= h0 / hn
    ki[:, :2, 2] -= 0.5  # principal point at (W / 2, H / 2) in corner-origin pixels
    w2c = torch.eye(4, device=ext.device).repeat(ext.shape[0], 1, 1)
    w2c[:, :3] = ext
    gs = pred["gs"][0].float()
    return {
        "depth": up(depth[:, None])[:, 0],
        "confidence": up(pred["depth_conf"][0][:, None])[:, 0],
        "world_points": up(pts.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous(),
        "intrinsics": ki,
        "cam_to_world": invert(w2c),
        RAW: gs.permute(0, 2, 3, 1).reshape(-1, gs.shape[1]),
        POINTS: pts.reshape(-1, 3),
    }


def voxelize(
    pts: torch.Tensor, feats: torch.Tensor, conf: torch.Tensor, size: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `voxelizaton_with_fusion`: points `[N, 3]`, features `[N, C]` -> one per voxel (softmax of `conf`).

    Voxels in `torch.unique` order (sorted voxel indices), as upstream.
    """
    vox, inv = torch.unique((pts / size).round().int(), dim=0, return_inverse=True)
    m = vox.shape[0]
    cmax = conf.new_full((m,), -math.inf).scatter_reduce(0, inv, conf, "amax")
    e = torch.exp(conf - cmax[inv])
    wsum = conf.new_zeros(m).index_add_(0, inv, e)
    wt = (e / (wsum[inv] + 1e-6))[:, None]
    vp = pts.new_zeros(m, 3).index_add_(0, inv, pts * wt)
    return vp, feats.new_zeros(m, feats.shape[1]).index_add_(0, inv, feats * wt)


@functools.cache
def _sh_mask(device: torch.device) -> torch.Tensor:
    mask = torch.ones(SH, device=device)
    for deg in range(1, SH_DEGREE + 1):
        mask[deg**2 : (deg + 1) ** 2] = 0.1 * 0.25**deg
    return mask


def gaussians(raw: torch.Tensor, pts: torch.Tensor, voxel_size: float) -> dict[str, torch.Tensor]:
    """Per-pixel raw parameters `[N, GS_DIM + 1]` + points `[N, 3]` -> the `gaussians/*` fields (upstream adapter).

    `UnifiedGaussianAdapter` with the released config: opacity = sigmoid (the opacity mapping is the identity),
    scales = 0.001 softplus clamped to 0.3, normalised quaternions (upstream x y z w -> w x y z), SH x `sh_mask`.
    """
    feats, conf = raw[:, :GS_DIM], raw[:, GS_DIM]
    if voxel_size > 0:
        pts, feats = voxelize(pts, feats, conf, voxel_size)
    scales, rot, sh = feats[:, 1:].split((3, 4, 3 * SH), -1)
    rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)
    return {
        "gaussians/means": pts,
        "gaussians/scales": (0.001 * F.softplus(scales)).clamp_max(0.3),
        "gaussians/rotations": rot[:, [3, 0, 1, 2]],
        "gaussians/harmonics": sh.reshape(-1, 3, SH) * _sh_mask(sh.device),
        "gaussians/opacities": feats[:, 0].sigmoid(),
    }


class _Base(Runner):
    intrinsics = "unused"
    cam_to_world = "unused"
    depth_range = "unused"
    per_view = True
    min_views = 1
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = _net(ctx)

    def resize(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        h0, w0 = images.shape[-2:]
        hn, wn = network_size(h0, w0, self.ctx.params.long_side)
        if (hn, wn) != (h0, w0):
            images = F.interpolate(images, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return images, hn, wn

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Images `[1, S, 3, hn, wn]` -> upstream prediction dict."""
        raise NotImplementedError

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,  # noqa: ARG002 - unused input of the common signature
        cam_to_world: torch.Tensor | None = None,  # noqa: ARG002
        depth_range: torch.Tensor | None = None,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """The runner's inputs -> the maps + per-pixel raw Gaussians (`gaussians/raw`, `gaussians/points`)."""
        h0, w0 = images.shape[-2:]
        x, hn, wn = self.resize(images)
        return outputs(self.predict(x[None]), hn, wn, h0, w0)

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth, points, cameras of every view and the scene's Gaussians, in the frame of `images[0]`."""
        maps = self.forward(images, intrinsics, cam_to_world, depth_range)
        return maps | gaussians(maps.pop(RAW), maps.pop(POINTS), self.ctx.params.voxel_size)


class _FP32Out(nn.Module):
    """The last 1x1 conv of a DPT head in fp32 (the head runs fp16 up to here)."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x.float())


@functools.cache
def _swap_index(device: torch.device) -> torch.Tensor:
    return torch.tensor([*range(16, 32), *range(16), *range(48, 64), *range(32, 48)], device=device)


def _swap(t: torch.Tensor) -> torch.Tensor:
    """Swap the two 16-wide quarters inside each 32-wide RoPE half of the last (64) dim."""
    return t.index_select(-1, _swap_index(t.device))


def _fp16_accumulate(*, on: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if on else contextlib.nullcontext()


class _Fast(_Base):
    """VGGT's fast path: fp16 aggregator with fused q/k-norm + RoPE, fp16 DPT heads (see the module docstring)."""

    attn = "sdpa"
    gacc = False

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._aff: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dt = torch.float32 if ctx.dtype == "float32" else torch.float16
        agg: Any = self.net.aggregator
        b = agg.frame_blocks[0].attn
        if b.head_dim != 64 or b.rope is None or list(agg.aa_order) != ["frame", "global"] or agg.aa_block_size != 1:  # noqa: PLR2004 - fused qk path
            msg = "AnySplat fast path: unexpected aggregator structure"
            raise ValueError(msg)
        if ctx.dtype == "float32":
            return
        agg.half()
        heads: tuple[Any, ...] = (self.net.depth_head, self.net.gaussian_param_head)
        for h in heads:
            h.half().to(memory_format=torch.channels_last)
            oc = h.scratch.output_conv2  # [3x3 conv, ReLU, 1x1 conv]: only the last 1x1 conv in fp32
            h.scratch.output_conv2 = nn.Sequential(oc[0], oc[1], _FP32Out(oc[2])).eval()

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv2 patch tokens `[B*S, P, C]` (fp16) of normalised fp16 images."""
        return self.net.aggregator.patch_embed(x)["x_norm_patchtokens"]

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`[B, N, heads, 64]` q / k / v -> `[B, N, heads * 64]`."""
        b, n, h, d = q.shape
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        return o.transpose(1, 2).reshape(b, n, h * d)

    def qk_affine(self, a: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - vendored Attention
        key = id(a)
        if key not in self._aff:
            self._aff[key] = (
                torch.stack([a.q_norm.weight, a.k_norm.weight])[:, None],
                torch.stack([a.q_norm.bias, a.k_norm.bias])[:, None],
            )  # [2, 1, 64] (q, k)
        return self._aff[key]

    def rope_tables(self, pos: torch.Tensor, b: int, s: int, p: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token RoPE tables `[B, S*P, 1, 1, 64]`: cos and sign-folded sin (rot(x) = swap(x) * sign)."""
        rope: Any = self.net.aggregator.frame_blocks[0].attn.rope  # ty: ignore[unresolved-attribute]
        cc, sc = rope._compute_frequency_components(32, p + 1, pos.device, self.dt)  # noqa: SLF001 - pos <= p
        q = pos.view(b, s * p, 2)
        cos = torch.cat([F.embedding(q[..., 0], cc), F.embedding(q[..., 1], cc)], -1)
        sgn = torch.tensor([-1.0] * 16 + [1.0] * 16, device=pos.device, dtype=self.dt).repeat(2)
        sin = torch.cat([F.embedding(q[..., 0], sc), F.embedding(q[..., 1], sc)], -1) * sgn
        return cos[:, :, None, None], sin[:, :, None, None]

    def block(self, blk: Any, x: torch.Tensor, tab: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:  # noqa: ANN401 - vendored Block
        """One aggregator block: q/k LayerNorm + RoPE as `n*(w*cos) + swap(n)*(swap(w)*sin) + (b*cos + swap(b)*sin)`."""
        a = blk.attn
        bsz, n, c = x.shape
        nh, d = a.num_heads, c // a.num_heads
        xn = blk.norm1(x)
        cos, sin = tab
        qk = F.linear(xn, a.qkv.weight[: 2 * c], a.qkv.bias[: 2 * c])  # contiguous q|k: no copy for group_norm
        v = F.linear(xn, a.qkv.weight[2 * c :], a.qkv.bias[2 * c :]).view(bsz, n, nh, d)
        nrm = F.group_norm(qk.view(bsz * n, 2 * nh, d), 2 * nh, None, None, a.q_norm.eps).view(bsz, n, 2, nh, d)
        w, bb = self.qk_affine(a)
        qk = torch.addcmul(torch.addcmul(bb * cos, _swap(bb), sin), nrm, w * cos).addcmul_(_swap(nrm), _swap(w) * sin)
        o = F.linear(self.attention(qk[:, :, 0], qk[:, :, 1], v, a.scale), a.proj.weight, a.proj.bias)
        x = torch.addcmul(x, o, blk.ls1.gamma)
        m = blk.mlp
        o = F.linear(F.gelu(F.linear(blk.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        return torch.addcmul(x, o, blk.ls2.gamma) if torch.is_grad_enabled() else x.addcmul_(o, blk.ls2.gamma)

    def aggregate(self, images: torch.Tensor) -> tuple[list[torch.Tensor | None], int]:
        """Upstream `Aggregator.forward` in fp16 -> (cached token lists, patch start index)."""
        agg: Any = self.net.aggregator
        b, s, _, h, w = images.shape
        x = ((images - agg._resnet_mean) / agg._resnet_std).view(b * s, 3, h, w).to(self.dt)  # noqa: SLF001
        pt = self.patch_tokens(x)
        c = pt.shape[-1]
        tokens = torch.cat(
            [slice_expand_and_flatten(agg.camera_token, b, s), slice_expand_and_flatten(agg.register_token, b, s), pt],
            1,
        )
        pos = agg.position_getter(b * s, h // PATCH, w // PATCH, device=images.device) + 1
        pos = torch.cat([torch.zeros(b * s, agg.patch_start_idx, 2, device=images.device, dtype=pos.dtype), pos], 1)
        p = tokens.shape[1]
        tg = self.rope_tables(pos, b, s, p)  # global view
        tf = (tg[0].view(b * s, p, 1, 1, -1), tg[1].view(b * s, p, 1, 1, -1))  # frame view
        out: list[torch.Tensor | None] = []
        for i in range(agg.aa_block_num):
            tokens = self.block(agg.frame_blocks[i], tokens.view(b * s, p, c), tf)
            fr = tokens.view(b, s, p, c)
            if i in agg.cached_layer_indices:
                fr = fr.clone()
            tokens = self.block(agg.global_blocks[i], tokens.view(b, s * p, c), tg)
            out.append(torch.cat([fr, tokens.view(b, s, p, c)], -1) if i in agg.cached_layer_indices else None)
        return out, agg.patch_start_idx

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fast aggregator, fp32 camera head, fp16 depth and Gaussian heads."""
        net = self.net
        with torch.autocast("cuda", enabled=False), _fp16_accumulate(on=self.gacc):
            toks, psi = self.aggregate(x)
        last = toks[-1]
        if last is None:
            msg = "AnySplat: the last aggregator block is not cached"
            raise RuntimeError(msg)
        ct = [None] * (len(toks) - 1) + [last.float()]
        pose = net.camera_head(ct)[-1]
        xh = x.to(self.dt)
        depth, depth_conf = net.depth_head(toks, images=xh, patch_start_idx=psi)
        return {
            "pose_enc": pose,
            "depth": depth,
            "depth_conf": depth_conf,
            "gs": net.gaussian_param_head(toks, xh, psi),
        }


class Portable(_Fast):
    """Plain PyTorch: float32 = upstream modules; float16 (CUDA) = friendly (fp16 aggregator, SDPA)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        rope: Any = self.net.aggregator.rope
        if rope is not None:
            override(rope, rope_forward)

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream modules in fp32 (the fused path in fp32 on MPS), or the fp16 path."""
        upstream = self.ctx.dtype == "float32" and self.ctx.device.type != "mps"
        return self.net(x) if upstream else super().predict(x)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `forward` (Gaussians per pixel, raw).

        The voxel fusion (`torch.unique`) and the parameter activations of `__call__` stay outside: `gaussians/raw`
        `[G, 84]` and `gaussians/points` `[G, 3]` (`G = V * hn * wn`); `model.gaussians(raw, points, voxel_size)` of
        this family finishes them.
        """
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class _PatchEmbed(FastViT):
    """The shared fast DINOv2 path, with upstream VGGT's position interpolation (`size=` instead of `scale_factor=`)."""

    def interpolate_pos(self, h0: int, w0: int) -> torch.Tensor:
        """Upstream `interpolate_pos_encoding` (interpolate_offset 0: output size), fp32."""
        vit, pe = self.vit, self.pos_fp32.to(self.vit.pos_embed.device)
        m = int(math.sqrt(pe.shape[1] - 1))
        if m * m == h0 * w0 and h0 == w0:
            return pe
        p = F.interpolate(
            pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode="bicubic",
            antialias=vit.interpolate_antialias,
        )
        return torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)


def fast_patch_embed(pe: nn.Module, attn: str) -> _PatchEmbed:
    """The aggregator's DINOv2 (ViT-L/14, 4 registers) rebuilt as the shared ViT -> fast fp16 path (`fastbb` flash)."""
    vit = vit_large(
        img_size=518,
        patch_size=PATCH,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    )
    vit.load_state_dict({k: v.float() for k, v in pe.state_dict().items()}, strict=True)
    return _PatchEmbed(vit.to(next(pe.parameters()).device).eval(), attn, None)


def _hacc_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return conv_half_acc(x.half(), self.weight, self.bias, list(self.padding), list(self.stride))  # ty: ignore


def _hacc_convs(module: nn.Module) -> int:
    """Route every fp16 Conv2d (groups 1, dilation 1, zero padding) through cuDNN fp16-accumulate convs."""
    n = 0
    for m in module.modules():
        if (
            is_base(m, nn.Conv2d)
            and m.groups == 1
            and m.dilation == (1, 1)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
            and m.weight.dtype == torch.float16
        ):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            override(m, _hacc_forward)
            n += 1
    return n


class Kernels(_Fast):
    """`fastbb` flash attention (fp16 accumulation), fp16-accumulate GEMMs, cuDNN half-acc DPT convs."""

    gacc = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash, self.check_flash64 = fastbb.Flash64("hh"), fastbb.check_flash64
        cudnn_hacc.load(ctx.device)
        agg: Any = self.net.aggregator
        self.patch_embed = fast_patch_embed(agg.patch_embed, "hh")
        agg.patch_embed = nn.Identity().eval()  # the fast copy owns the weights now
        gs = self.net.gaussian_param_head
        convs = [self.net.depth_head, *(m for name, m in gs.named_children() if name != "input_merger")]
        if sum(_hacc_convs(m) for m in convs) == 0:  # input_merger (3 input channels) stays on cuDNN's default
            msg = "AnySplat kernels: no DPT conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The shared fast DINOv2 (`FastViT`) on the aggregator's patch embed weights."""
        return self.patch_embed.get_intermediate_layers(x, 1)[0]

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`fastbb` flash64 (fp16 in / out, fp16 P.V accumulation) on strided `[B, N, heads, 64]` views."""
        b, n, h, d = q.shape
        self.check_flash64(q, k, v)
        return self.flash(q, k, v, scale).view(b, n, h * d)


RUNNERS = {"portable": Portable, "kernels": Kernels}
