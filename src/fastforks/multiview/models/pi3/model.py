"""Pi3 runners: weights -> network and the portable / kernels forward passes.

Upstream `Pi3.forward` on all views at once. Images are resized to the network size (about `pixel_limit` pixels at the
input aspect ratio, both sides multiples of 14, as upstream `load_images_as_tensor`; bicubic antialiased instead of
PIL Lanczos) unless they already have it; maps are resized back (bilinear). Pi3 predicts one camera-to-world pose per
view in no fixed frame; the runner re-expresses poses and points in the frame of the first view (OpenCV axes, scale
as predicted: Pi3 normalises the scene). `depth` is the z of upstream's `local_points`, `confidence` the sigmoid of
upstream's `conf` logit (upstream's demo keeps points with confidence > 0.1).

The fp16 and kernels paths (`_Fast`): encoder and decoder in pure fp16 (no autocast); in the 36 alternating
frame / global decoder blocks the q/k LayerNorm + 2D RoPE run as 4 full passes (affine-free group_norm over q and k,
LN affine folded into per-token RoPE tables), the 15 head-decoder blocks apply RoPE the same way (no q/k norm); the
patch heads (`LinearPts3d`) and the camera head run in fp32 (upstream disables autocast there). Kernels: the shared
fast DINOv2 encoder and the `fastbb` head-dim-64 flash kernel with fp16 accumulation, fp16-accumulate GEMMs.
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
from fastforks._core.patch import override
from fastforks.multiview._runner import Forward, Runner, invert
from fastforks.shared.depth_anything_v2 import FastViT, vit_large

from .net.layers.pos_embed import RoPE2D
from .net.pi3 import Pi3

if TYPE_CHECKING:
    from fastforks.multiview._runner import Context

PATCH = 14  # fixed: the ViT patch size
ROPE_FREQ = 100.0  # fixed: upstream pos_type "rope100"


@dataclass(frozen=True)
class Params:
    """Load options."""

    pixel_limit: int  # network pixel budget (upstream PIXEL_LIMIT)


def network_size(h: int, w: int, pixel_limit: int) -> tuple[int, int]:
    """Upstream `load_images_as_tensor`: the largest multiple-of-14 size of about `pixel_limit` pixels."""
    s = math.sqrt(pixel_limit / (w * h))
    wt, ht = w * s, h * s
    k, m = round(wt / PATCH), round(ht / PATCH)
    while (k * PATCH) * (m * PATCH) > pixel_limit:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, m) * PATCH, max(1, k) * PATCH


def _net(ctx: Context) -> Pi3:
    net = Pi3()
    net.load_state_dict(ctx.state_dict(), strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def outputs(pred: dict[str, torch.Tensor], hn: int, wn: int, h0: int, w0: int) -> dict[str, torch.Tensor]:
    """Upstream predictions `[1, S, hn, wn, C]` -> the runner's maps at `h0 x w0`, in the frame of view 0."""

    def up(x: torch.Tensor) -> torch.Tensor:  # [S, hn, wn, C] -> [S, C, h0, w0]
        x = x.float().permute(0, 3, 1, 2)
        return x if (hn, wn) == (h0, w0) else F.interpolate(x, (h0, w0), mode="bilinear", align_corners=False)

    c2w = pred["camera_poses"][0].float()
    to0 = invert(c2w[:1])  # world -> camera 0
    pts = pred["points"][0].float()
    pts = torch.einsum("ij,shwj->shwi", to0[0, :3, :3], pts) + to0[0, :3, 3]
    return {
        "depth": up(pred["local_points"][0][..., 2:])[:, 0],
        "confidence": torch.sigmoid(up(pred["conf"][0])[:, 0]),
        "world_points": up(pts).permute(0, 2, 3, 1).contiguous(),
        "cam_to_world": to0 @ c2w,
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
        for m in self.net.modules():  # conf_decoder is a deepcopy of point_decoder: its own RoPE2D
            if isinstance(m, RoPE2D):
                override(m, _rope_forward)

    def resize(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        h0, w0 = images.shape[-2:]
        hn, wn = network_size(h0, w0, self.ctx.params.pixel_limit)
        if (hn, wn) != (h0, w0):
            images = F.interpolate(images, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return images, hn, wn

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Images `[1, S, 3, hn, wn]` in `[0, 1]` -> upstream prediction dict."""
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


def _rope_forward(self: nn.Module, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Upstream `RoPE2D.forward` with a table sized by the token count, not `positions.max()`.

    Positions are below the token count and a longer table has the same rows, so the result is unchanged; the
    data-dependent size broke torch.compile / export and cost a device sync. Use with `override(rope, _rope_forward)`.
    """
    rope: Any = self
    d = tokens.size(3) // 2
    cos, sin = rope.get_cos_sin(d, tokens.size(-2) + 1, tokens.device, tokens.dtype)
    y, x = tokens.chunk(2, dim=-1)
    return torch.cat(
        (rope.apply_rope1d(y, positions[:, :, 0], cos, sin), rope.apply_rope1d(x, positions[:, :, 1], cos, sin)), -1
    )


@functools.cache
def _swap_index(device: torch.device) -> torch.Tensor:
    return torch.tensor([*range(16, 32), *range(16), *range(48, 64), *range(32, 48)], device=device)


def _swap(t: torch.Tensor) -> torch.Tensor:
    """Swap the two 16-wide quarters inside each 32-wide RoPE half of the last (64) dim (y half, x half)."""
    return t.index_select(-1, _swap_index(t.device))


def _sdpa_forward(self: Any, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ANN401 - vendored DINOv2 Attention
    """The encoder's `MemEffAttention` without xFormers, as SDPA (upstream: explicit softmax(q k^T) v)."""
    del attn_bias
    b, n, c = x.shape
    qkv = self.qkv(x).view(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale)
    return self.proj(o.transpose(1, 2).reshape(b, n, c))


class _Fast(_Base):
    """fp16 encoder + decoder with fused q/k-norm + RoPE, fp32 patch and camera heads (see the module docstring)."""

    gacc = False

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._aff: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dt = torch.float32 if ctx.dtype == "float32" else torch.float16
        net: Any = self.net  # vendored modules (untyped attributes)
        a0 = net.decoder[0].attn
        if a0.head_dim != 64 or net.camera_decoder.blocks[0].attn.head_dim != 64:  # noqa: PLR2004 - fused path
            msg = "Pi3 fast path: unexpected head dim"
            raise ValueError(msg)
        for b in net.encoder.blocks:
            override(b.attn, _sdpa_forward)
        if ctx.dtype == "float32":
            return
        for m in (net.encoder, net.decoder, net.point_decoder, net.conf_decoder, net.camera_decoder):
            m.half()
        net.register_token.data = net.register_token.data.half()

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv2 patch tokens `[B*S, P, C]` of normalised images."""
        return self.net.encoder(x, is_training=True)["x_norm_patchtokens"]

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`[B, N, heads, 64]` q / k / v -> `[B, N, heads * 64]`."""
        b, n, h, d = q.shape
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        return o.transpose(1, 2).reshape(b, n, h * d)

    def qk_affine(self, a: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - vendored AttentionRope
        key = id(a)
        if key not in self._aff:
            self._aff[key] = (
                torch.stack([a.q_norm.weight, a.k_norm.weight])[:, None],
                torch.stack([a.q_norm.bias, a.k_norm.bias])[:, None],
            )  # [2, 1, 64] (q, k)
        return self._aff[key]

    def rope_tables(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token RoPE tables `[B, N, 1, 1, 64]` of positions `[B, N, 2]`: cos and sign-folded sin."""
        n = pos.shape[1] + 1  # positions < token count: a static table size (no `pos.max()` sync)
        inv = 1.0 / (ROPE_FREQ ** (torch.arange(0, 32, 2, device=pos.device).float() / 32))
        f = torch.outer(torch.arange(n, device=pos.device).float(), inv).to(self.dt)  # upstream: angles in the dtype
        f = torch.cat([f, f], -1)
        cc, sc = f.cos(), f.sin()
        cos = torch.cat([F.embedding(pos[..., 0], cc), F.embedding(pos[..., 1], cc)], -1)
        sgn = torch.tensor([-1.0] * 16 + [1.0] * 16, device=pos.device, dtype=self.dt).repeat(2)
        sin = torch.cat([F.embedding(pos[..., 0], sc), F.embedding(pos[..., 1], sc)], -1) * sgn
        return cos[:, :, None, None], sin[:, :, None, None]

    def block(self, blk: Any, x: torch.Tensor, tab: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:  # noqa: ANN401 - vendored BlockRope
        """One `BlockRope`; q/k LayerNorm + RoPE as `n*(w*cos) + swap(n)*(swap(w)*sin) + (b*cos + swap(b)*sin)`."""
        a = blk.attn
        bsz, n, c = x.shape
        nh, d = a.num_heads, c // a.num_heads
        xn = blk.norm1(x)
        cos, sin = tab
        qk = F.linear(xn, a.qkv.weight[: 2 * c], a.qkv.bias[: 2 * c])  # contiguous q|k: no copy for group_norm
        v = F.linear(xn, a.qkv.weight[2 * c :], a.qkv.bias[2 * c :]).view(bsz, n, nh, d)
        if isinstance(a.q_norm, nn.LayerNorm):
            nrm = F.group_norm(qk.view(bsz * n, 2 * nh, d), 2 * nh, None, None, a.q_norm.eps).view(bsz, n, 2, nh, d)
            w, bb = self.qk_affine(a)
            qk = torch.addcmul(torch.addcmul(bb * cos, _swap(bb), sin), nrm, w * cos)
            qk = qk.addcmul_(_swap(nrm), _swap(w) * sin)
        else:  # the head decoders: RoPE only
            qk = qk.view(bsz, n, 2, nh, d)
            qk = torch.addcmul(qk * cos, _swap(qk), sin)
        o = F.linear(self.attention(qk[:, :, 0], qk[:, :, 1], v, a.scale), a.proj.weight, a.proj.bias)
        x = torch.addcmul(x, o, blk.ls1.gamma) if hasattr(blk.ls1, "gamma") else x + o
        m = blk.mlp
        o = F.linear(F.gelu(F.linear(blk.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        if hasattr(blk.ls2, "gamma"):
            return torch.addcmul(x, o, blk.ls2.gamma) if torch.is_grad_enabled() else x.addcmul_(o, blk.ls2.gamma)
        return x + o if torch.is_grad_enabled() else x.add_(o)

    def head(self, dec: Any, h: torch.Tensor, tab: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:  # noqa: ANN401 - vendored TransformerDecoder
        """Upstream `TransformerDecoder.forward` (5 RoPE blocks, per view)."""
        h = dec.projects(h)
        for blk in dec.blocks:
            h = self.block(blk, h, tab)
        return dec.linear_out(h)

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `Pi3.forward` with the fused blocks."""
        net: Any = self.net
        b, s, _, hn, wn = x.shape
        hp, wp = hn // PATCH, wn // PATCH
        with torch.autocast("cuda", enabled=False), _fp16_accumulate(on=self.gacc):
            xn = ((x - net.image_mean[None]) / net.image_std[None]).view(b * s, 3, hn, wn).to(self.dt)
            pt = self.patch_tokens(xn)
            c = pt.shape[-1]
            reg = net.register_token.expand(b, s, -1, -1).reshape(b * s, -1, c)
            hidden = torch.cat([reg, pt.to(self.dt)], 1)
            p = hidden.shape[1]
            pos = net.position_getter(b * s, hp, wp, x.device) + 1
            pos = torch.cat([torch.zeros(b * s, net.patch_start_idx, 2, device=x.device, dtype=pos.dtype), pos], 1)
            tf = self.rope_tables(pos)  # frame view [B*S, P, ...]
            tg = (tf[0].view(b, s * p, 1, 1, -1), tf[1].view(b, s * p, 1, 1, -1))  # global view
            last: list[torch.Tensor] = []
            nd = len(net.decoder)
            for i, blk in enumerate(net.decoder):
                if i % 2 == 0:
                    hidden = self.block(blk, hidden.view(b * s, p, c), tf)
                else:
                    hidden = self.block(blk, hidden.view(b, s * p, c), tg)
                if i + 1 in (nd - 1, nd):
                    last.append(hidden.view(b * s, p, c))  # blocks never write their input in place
            hidden = torch.cat(last, -1)
            point_h = self.head(net.point_decoder, hidden, tf)
            conf_h = self.head(net.conf_decoder, hidden, tf)
            cam_h = self.head(net.camera_decoder, hidden, tf)
            psi = net.patch_start_idx
            lp = net.point_head([point_h[:, psi:].float()], (hn, wn)).reshape(b, s, hn, wn, -1)
            xy, z = lp.split([2, 1], dim=-1)
            z = torch.exp(z)
            local = torch.cat([xy * z, z], dim=-1)
            conf = net.conf_head([conf_h[:, psi:].float()], (hn, wn)).reshape(b, s, hn, wn, -1)
            poses = net.camera_head(cam_h[:, psi:].float(), hp, wp).reshape(b, s, 4, 4)
            pts = torch.einsum("bnij,bnhwj->bnhwi", poses[..., :3, :3], local) + poses[..., None, None, :3, 3]
        return {"points": pts, "local_points": local, "conf": conf, "camera_poses": poses}


def _fp16_accumulate(*, on: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if on else contextlib.nullcontext()


class Portable(_Fast):
    """Plain PyTorch: float32 = upstream modules (SDPA in the encoder); float16 (CUDA) = the fused fp16 path."""

    dtypes = ("float32", "float16")

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `Pi3.forward` in fp32, or the fp16 path."""
        if self.ctx.dtype == "float32":
            with torch.autocast(x.device.type, enabled=False):
                return self.net(x)
        return super().predict(x)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class _Encoder(FastViT):
    """The shared fast DINOv2 path with upstream's position interpolation (`size=`: interpolate_offset 0)."""

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


class Kernels(_Fast):
    """The shared fast DINOv2, `fastbb` flash attention (fp16 accumulation), fp16-accumulate GEMMs."""

    gacc = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash, self.check_flash64 = fastbb.Flash64("hh"), fastbb.check_flash64
        enc = self.net.encoder
        vit = vit_large(
            img_size=518,
            patch_size=PATCH,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
        )
        sd = {k: v.float() for k, v in enc.state_dict().items()}
        vit.load_state_dict(sd, strict=False)  # Pi3 deleted the mask token
        self.encoder = _Encoder(vit.to(ctx.device).eval(), "hh", None)
        self.net.encoder = nn.Identity().eval()  # the fast copy owns the weights now

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The shared fast DINOv2 (`FastViT`) on Pi3's encoder weights."""
        return self.encoder.get_intermediate_layers(x, 1)[0]

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`fastbb` flash64 (fp16 in / out, fp16 P.V accumulation) on strided `[B, N, heads, 64]` views."""
        b, n, h, d = q.shape
        self.check_flash64(q, k, v)
        return self.flash(q, k, v, scale).view(b, n, h * d)


RUNNERS = {"portable": Portable, "kernels": Kernels}
