"""VGGT runners: weights -> network and the portable / kernels forward passes.

Upstream `VGGT.forward` (camera, depth and point heads; no tracking head) on all views at once. Images are resized to
the network size (long side `long_side`, both sides multiples of 14, bicubic antialiased: upstream preprocesses with
PIL bicubic to width 518) unless they already have it; maps are resized back (bilinear), the predicted intrinsics are
rescaled to the input pixels (pixel centres at integer coordinates; VGGT puts the principal point at the image centre)
and the predicted world-from-camera poses are those of upstream (world = the first camera, OpenCV axes; scale as
predicted: VGGT normalises the scene).

The fp16 and kernels paths are `vggt/fast.py` (`FastVGGT`): aggregator in pure fp16 (no autocast), q/k LayerNorm
+ 2D RoPE in 4 full passes (affine-free group_norm over q and k, LN affine folded into per-token RoPE tables), DPT heads
fp16 channels-last except the last 1x1 conv + activation (fp32), camera head fp32. Kernels: the `fastbb` head-dim-64
flash kernel with fp16 accumulation, fp16-accumulate GEMMs and cuDNN fp16-accumulate convs in the DPT heads.
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


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (both sides rounded to multiples of 14)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: long side `long_side`, both sides multiples of 14."""
    s = long_side / max(h, w)
    return (max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH))


class Net(nn.Module):
    """Upstream `VGGT` without the tracking head (same attribute names, so the checkpoint keys match)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.aggregator = Aggregator(img_size=518, patch_size=PATCH, embed_dim=EMBED)
        self.camera_head = CameraHead(dim_in=2 * EMBED)
        self.point_head = DPTHead(dim_in=2 * EMBED, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * EMBED, output_dim=2, activation="exp", conf_activation="expp1")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `VGGT.forward` (eval): images `[1, S, 3, H, W]` in `[0, 1]` -> pose encoding, depth, points."""
        toks, psi = self.aggregator(images)
        pose = self.camera_head(toks)[-1]
        depth, depth_conf = self.depth_head(toks, images=images, patch_start_idx=psi)
        pts, pts_conf = self.point_head(toks, images=images, patch_start_idx=psi)
        return {
            "pose_enc": pose,
            "depth": depth,
            "depth_conf": depth_conf,
            "world_points": pts,
            "world_points_conf": pts_conf,
        }


def _net(ctx: Context) -> Net:
    sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith("track_head.")}
    net = Net()
    net.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def outputs(pred: dict[str, torch.Tensor], hn: int, wn: int, h0: int, w0: int) -> dict[str, torch.Tensor]:
    """Upstream predictions at the network size `hn x wn` -> the runner's maps at the input size `h0 x w0`."""

    def up(x: torch.Tensor) -> torch.Tensor:  # [S, C, hn, wn] -> [S, C, h0, w0]
        x = x.float()
        return x if (hn, wn) == (h0, w0) else F.interpolate(x, (h0, w0), mode="bilinear", align_corners=False)

    ext, k = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (hn, wn))
    ext, k = ext[0], k[0].clone()
    k[:, 0] *= w0 / wn
    k[:, 1] *= h0 / hn
    k[:, :2, 2] -= 0.5  # VGGT: principal point at (W / 2, H / 2) in corner-origin pixels
    w2c = torch.eye(4, device=ext.device).repeat(ext.shape[0], 1, 1)
    w2c[:, :3] = ext
    return {
        "depth": up(pred["depth"][0].permute(0, 3, 1, 2))[:, 0],
        "confidence": up(pred["depth_conf"][0][:, None])[:, 0],
        "world_points": up(pred["world_points"][0].permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous(),
        "points_confidence": up(pred["world_points_conf"][0][:, None])[:, 0],
        "intrinsics": k,
        "cam_to_world": invert(w2c),
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
        """The runner's inputs -> the maps of `__call__`."""
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
        """Depth, points and cameras of every view, in the frame of `images[0]`."""
        return self.forward(images, intrinsics, cam_to_world, depth_range)


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
    """Swap the two 16-wide quarters inside each 32-wide RoPE half of the last (64) dim.

    An `index_select`, not `flip`: MPS's flip of the `[..., 2, 2, 16]` view is ~70x slower (torch 2.14).
    """
    return t.index_select(-1, _swap_index(t.device))


def _fp16_accumulate(*, on: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if on else contextlib.nullcontext()


class _Fast(_Base):
    """`FastVGGT`: fp16 aggregator with fused q/k-norm + RoPE, fp16 DPT heads (see the module docstring)."""

    attn = "sdpa"
    gacc = False

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._aff: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dt = torch.float32 if ctx.dtype == "float32" else torch.float16
        agg: Any = self.net.aggregator
        b = agg.frame_blocks[0].attn
        if b.head_dim != 64 or b.rope is None or list(agg.aa_order) != ["frame", "global"] or agg.aa_block_size != 1:  # noqa: PLR2004 - fused qk path
            msg = "VGGT fast path: unexpected aggregator structure"
            raise ValueError(msg)
        if ctx.dtype == "float32":
            return
        agg.half()
        heads: tuple[Any, ...] = (self.net.depth_head, self.net.point_head)
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
        """`FastVGGT.__call__`."""
        net = self.net
        with torch.autocast("cuda", enabled=False), _fp16_accumulate(on=self.gacc):
            toks, psi = self.aggregate(x)
        last = toks[-1]
        if last is None:
            msg = "VGGT: the last aggregator block is not cached"
            raise RuntimeError(msg)
        ct = [None] * (len(toks) - 1) + [last.float()]
        pose = net.camera_head(ct)[-1]
        xh = x.to(self.dt)
        depth, depth_conf = net.depth_head(toks, images=xh, patch_start_idx=psi)
        pts, pts_conf = net.point_head(toks, images=xh, patch_start_idx=psi)
        return {
            "pose_enc": pose,
            "depth": depth,
            "depth_conf": depth_conf,
            "world_points": pts,
            "world_points_conf": pts_conf,
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
        """Upstream modules in fp32 (the fused path in fp32 on MPS: ~12% faster there), or the fp16 path."""
        upstream = self.ctx.dtype == "float32" and self.ctx.device.type != "mps"
        return self.net(x) if upstream else super().predict(x)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class MetalKernels(Portable):
    """Apple MPS: the fused fp32 path with each block's q / k LayerNorm + RoPE as one Metal kernel (`vggt_metal`)."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import vggt_metal  # noqa: PLC0415 - needs the kernel library

        mps.adapt(self.net)  # the DPT heads' transposed convolutions, as the portable runner

        self.lib = vggt_metal.load(ctx.device)
        self.qk_norm_rope = vggt_metal.qk_norm_rope

    def block(self, blk: Any, x: torch.Tensor, tab: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:  # noqa: ANN401 - vendored Block
        """`_Fast.block` with the q / k LayerNorm + RoPE in one kernel that writes SDPA's layout."""
        a = blk.attn
        bsz, n, c = x.shape
        nh, d = a.num_heads, c // a.num_heads
        xn = blk.norm1(x)
        qk = F.linear(xn, a.qkv.weight[: 2 * c], a.qkv.bias[: 2 * c]).view(bsz, n, 2, nh, d)
        v = F.linear(xn, a.qkv.weight[2 * c :], a.qkv.bias[2 * c :]).view(bsz, n, nh, d).transpose(1, 2)
        w, bb = self.qk_affine(a)
        cos, sin = (t.reshape(-1, d) for t in tab)
        q, k = self.qk_norm_rope(self.lib, qk, cos, sin, (w.view(2, d), bb.view(2, d)), a.q_norm.eps)
        o = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(bsz, n, c)
        x = torch.addcmul(x, F.linear(o, a.proj.weight, a.proj.bias), blk.ls1.gamma)
        m = blk.mlp
        o = F.linear(F.gelu(F.linear(blk.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        return x.addcmul_(o, blk.ls2.gamma)


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
        if sum(_hacc_convs(h) for h in (self.net.depth_head, self.net.point_head)) == 0:
            msg = "VGGT kernels: no DPT conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The shared fast DINOv2 (`FastViT`) on the aggregator's patch embed weights."""
        return self.patch_embed.get_intermediate_layers(x, 1)[0]

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`fastbb` flash64 (fp16 in / out, fp16 P.V accumulation) on strided `[B, N, heads, 64]` views."""
        b, n, h, d = q.shape
        self.check_flash64(q, k, v)
        return self.flash(q, k, v, scale).view(b, n, h * d)


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
