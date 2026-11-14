"""MegaFlow runners: weights -> network and the portable / kernels forward passes of a pair.

Upstream `MegaFlow.forward` on the pair as a two-frame video (`num_reg_refine` = `iters`): the frames are resized to
width 952 (`fix_width`) for the DINOv2 + alternating-attention aggregator, a DPT head fuses four of its layers with
the ResNet CNN's features into 128-channel features, a global softmax correlation gives the initial flow, and
`iters` RAFT-style updates on 9 x 9 local correlations of the CNN's 1/4 features refine it (convex 4x upsampling).

float32 runs the upstream modules. The float16 path (`_Fast`): aggregator in pure fp16 with q / k LayerNorm + 2D
RoPE fused as in the VGGT port (only the four layers the head reads are kept), head, CNN and update block in fp16
channels-last, the global correlation as one fp32 SDPA call (values: the pixel grid centred on the image, so no
`[HW, HW]` matrix), the local correlation in pixel chunks (no `[HW, C, 81]` window tensor), the temporal attention
of the update block as what it is on a pair (a length-1 sequence: `proj(v)`), no `torch.cuda.empty_cache()` per
iteration. Kernels: the `fastbb` head-dim-64 flash kernel and the shared fast DINOv2, fp16-accumulate GEMMs, cuDNN
fp16-accumulate convolutions and the correlation windows computed on the fly (`raft_flow`).
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

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.flow._runner import Forward, Runner
from fastforks.shared.depth_anything_v2 import FastViT, vit_large

from .net.megaflow import MegaFlow, slice_expand_and_flatten
from .net.model_utils import create_uv_grid, position_grid_to_embed, upsample_flow_with_mask

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context

PATCH = 14  # fixed: the ViT patch size
WIDTH = 952  # fixed: upstream `fix_width`, the aggregator's trained input width
LAYERS = (4, 11, 17, 23)  # fixed: the aggregator layers the flow head reads
RADIUS = 4  # fixed: trained with a local correlation window 9 x 9
UP = 4  # fixed: trained with refinement at 1/4 resolution, convex 4x upsampling
CHUNK = 8192  # fixed: memory bound; pixels per local-correlation chunk of the portable fp16 path


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # refinement updates (upstream `num_reg_refine`)


def _net(ctx: Context) -> MegaFlow:
    net = MegaFlow(fuse_cnn=True, use_temporal_attn=True, fix_width=True)  # upstream `from_pretrained`
    sd = ctx.state_dict()
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


@functools.cache
def _swap_index(device: torch.device) -> torch.Tensor:
    return torch.tensor([*range(16, 32), *range(16), *range(48, 64), *range(32, 48)], device=device)


def _swap(t: torch.Tensor) -> torch.Tensor:
    """Swap the two 16-wide quarters inside each 32-wide RoPE half of the last (64) dim."""
    return t.index_select(-1, _swap_index(t.device))


def _pos_embed(self: nn.Module, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:  # noqa: N803 - upstream names
    """Upstream `FlowFeature._apply_pos_embed` with the table in fp32 and the sum in `x`'s dtype (upstream promotes)."""
    del self
    pe = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W / H, dtype=torch.float32, device=x.device)
    pe = position_grid_to_embed(pe, x.shape[1]) * ratio
    return x + pe.permute(2, 0, 1)[None].to(x.dtype)


def _grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixel coordinates `[B, 2, H, W]` (x, y), fp32."""
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([x, y])[None].expand(b, -1, -1, -1)


def global_flow(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    """Upstream `global_correlation_softmax` as one fp32 SDPA call: softmax(f0 f1^T / sqrt(C)) (grid - centre).

    The values are the pixel grid minus the image centre (softmax rows sum to 1, so the centre cancels): smaller
    magnitudes than upstream's grid, padded to 8 channels.
    """
    b, _, h, w = f0.shape
    q = f0.flatten(2).transpose(1, 2).float()[:, None]
    k = f1.flatten(2).transpose(1, 2).float()[:, None]
    g = _grid(1, h, w, f0.device)[0].flatten(1).t()  # [HW, 2]
    g = g - torch.tensor([(w - 1) / 2, (h - 1) / 2], device=g.device)
    v = F.pad(g, (0, 6))[None, None].expand(b, 1, -1, -1)
    o = F.scaled_dot_product_attention(q, k, v)[:, 0, :, :2]  # default scale 1 / sqrt(C), as upstream
    return (o - g).transpose(1, 2).reshape(b, 2, h, w)


def _window(device: torch.device) -> torch.Tensor:
    """The 81 offsets `[81, 2]` (dx, dy) in upstream's channel order: channel `a * 9 + b` is (b - 4, a - 4)."""
    r = torch.arange(-RADIUS, RADIUS + 1, device=device, dtype=torch.float32)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([dx, dy], -1).view(-1, 2)


def local_corr(f0: torch.Tensor, f1: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Upstream `local_correlation_with_flow` (radius 4) in pixel chunks: `[B, 81, H, W]` in `f0`'s dtype.

    Sampling in fp32 (upstream: grid_sample of fp32 features at fp32 coordinates), one chunk of `CHUNK` pixels'
    windows at a time instead of the whole `[B, C, HW, 81]` tensor.
    """
    b, c, h, w = f0.shape
    xy = (_grid(b, h, w, flow.device) + flow.float()).flatten(2).transpose(1, 2)  # [B, HW, 2]
    norm = torch.tensor([2 / (w - 1), 2 / (h - 1)], device=flow.device)
    win = _window(flow.device)
    f1f = f1.float()
    a = f0.flatten(2).transpose(1, 2)  # [B, HW, C]
    out = []
    for s in range(0, h * w, CHUNK):
        g = (xy[:, s : s + CHUNK, None] + win) * norm - 1  # [B, m, 81, 2]
        wf = F.grid_sample(f1f, g, padding_mode="zeros", align_corners=True)  # [B, C, m, 81]
        out.append((torch.einsum("bcmk,bmc->bmk", wf, a[:, s : s + CHUNK].float()) / math.sqrt(c)).to(f0.dtype))
    return torch.cat(out, 1).view(b, h, w, -1).permute(0, 3, 1, 2)


def _temporal(blk: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored Block
    """The update block's temporal-attention `Block` on a pair: a length-1 sequence, so the attention is `proj(v)`."""
    a = blk.attn
    c = x.shape[1]
    t = x.permute(0, 2, 3, 1)  # NHWC (a view of a channels-last tensor)
    t = t + a.proj(F.linear(blk.norm1(t), a.qkv.weight[2 * c :], a.qkv.bias[2 * c :]))
    t = t + blk.mlp(blk.norm2(t))
    return t.permute(0, 3, 1, 2)


class _Fast(Runner):
    """float32: the upstream modules; float16: the fused fp16 path (see the module docstring)."""

    dtypes = ("float32", "float16")
    Params = Params
    gacc = False  # fp16-accumulate GEMMs

    def __init__(self, ctx: Context) -> None:
        """Build the network; for float16 convert it."""
        super().__init__(ctx)
        self.net = net = _net(ctx)
        self.fp16 = ctx.dtype == "float16"
        a = net.frame_blocks[0].attn
        u = net.refine
        if (
            a.head_dim != 64  # noqa: PLR2004 - fused qk path
            or a.rope is None
            or list(net.aa_order) != ["frame", "global"]
            or net.aa_block_size != 1
            or tuple(net.flow_head.intermediate_layer_idx) != LAYERS
            or not all(
                isinstance(b.ls1, nn.Identity) and isinstance(b.ls2, nn.Identity) for b in u.temporal_attn_refine
            )
        ):
            msg = "MegaFlow fast path: unexpected network structure"
            raise ValueError(msg)
        self._aff: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.lookup: Callable[..., torch.Tensor] | None = None
        if not self.fp16:
            return
        net.half()
        for m in (net.flow_head, net.backbone, net.refine_proj, net.refine):
            m.to(memory_format=torch.channels_last)
        override(net.flow_head, _pos_embed, "_apply_pos_embed")

    # ------------------------------------------------------------------------------------------------ aggregator
    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv2 patch tokens `[B*S, P, C]` (fp16) of normalised fp16 images."""
        return self.net.patch_embed(x)["x_norm_patchtokens"]

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
        rope: Any = self.net.rope
        cc, sc = rope._compute_frequency_components(32, p + 1, pos.device, torch.float16)  # noqa: SLF001 - pos <= p
        q = pos.view(b, s * p, 2)
        cos = torch.cat([F.embedding(q[..., 0], cc), F.embedding(q[..., 1], cc)], -1)
        sgn = torch.tensor([-1.0] * 16 + [1.0] * 16, device=pos.device, dtype=torch.float16).repeat(2)
        sin = torch.cat([F.embedding(q[..., 0], sc), F.embedding(q[..., 1], sc)], -1) * sgn
        return cos[:, :, None, None], sin[:, :, None, None]

    def block(self, blk: Any, x: torch.Tensor, tab: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:  # noqa: ANN401 - vendored Block
        """One aggregator block: q/k LayerNorm + RoPE as `n*(w*cos) + swap(n)*(swap(w)*sin) + (b*cos + swap(b)*sin)`."""
        a = blk.attn
        bsz, n, c = x.shape
        nh, d = a.num_heads, c // a.num_heads
        xn = blk.norm1(x)
        cos, sin = tab
        qk = F.linear(xn, a.qkv.weight[: 2 * c], a.qkv.bias[: 2 * c])
        v = F.linear(xn, a.qkv.weight[2 * c :], a.qkv.bias[2 * c :]).view(bsz, n, nh, d)
        nrm = F.group_norm(qk.view(bsz * n, 2 * nh, d), 2 * nh, None, None, a.q_norm.eps).view(bsz, n, 2, nh, d)
        w, bb = self.qk_affine(a)
        qk = torch.addcmul(torch.addcmul(bb * cos, _swap(bb), sin), nrm, w * cos).addcmul_(_swap(nrm), _swap(w) * sin)
        o = F.linear(self.attention(qk[:, :, 0], qk[:, :, 1], v, a.scale), a.proj.weight, a.proj.bias)
        x = torch.addcmul(x, o, blk.ls1.gamma)
        m = blk.mlp
        o = F.linear(F.gelu(F.linear(blk.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        return x.addcmul_(o, blk.ls2.gamma)

    def aggregate(self, x: torch.Tensor, b: int) -> list[torch.Tensor | None]:
        """Upstream `extract_feature` in fp16 on `[B*2, 3, h, w]`: the token lists of `LAYERS` (others `None`)."""
        net: Any = self.net
        s = x.shape[0] // b
        pt = self.patch_tokens(x)
        c = pt.shape[-1]
        tokens = torch.cat([slice_expand_and_flatten(net.register_token, b, s), pt], 1)
        pos = net.position_getter(b * s, x.shape[-2] // PATCH, x.shape[-1] // PATCH, device=x.device) + 1
        pos = torch.cat([torch.zeros(b * s, net.patch_start_idx, 2, device=x.device, dtype=pos.dtype), pos], 1)
        p = tokens.shape[1]
        tg = self.rope_tables(pos, b, s, p)  # global view
        tf = (tg[0].view(b * s, p, 1, 1, -1), tg[1].view(b * s, p, 1, 1, -1))  # frame view
        out: list[torch.Tensor | None] = [None] * len(net.frame_blocks)
        for i in range(max(LAYERS) + 1):
            tokens = self.block(net.frame_blocks[i], tokens.view(b * s, p, c), tf)
            fr = tokens.view(b, s, p, c).clone() if i in LAYERS else None
            tokens = self.block(net.global_blocks[i], tokens.view(b, s * p, c), tg)
            if fr is not None:
                out[i] = torch.cat([fr, tokens.view(b, s, p, c)], -1)
        return out

    # ------------------------------------------------------------------------------------------------ refinement
    def correlate(self, f0: torch.Tensor, f1: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """The local correlation of `f0` against `f1` as a function of the flow."""
        return functools.partial(local_corr, f0, f1)

    def update(
        self, h: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor, *, last: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Upstream `RAFTUpdateBlock.forward` on a pair -> (hidden state, flow update, upsampling mask if `last`)."""
        u: Any = self.net.refine
        mf = u.encoder(flow.half(), corr)
        inp = torch.cat([inp, mf], 1)
        for blk, ta in zip(u.refine, u.temporal_attn_refine, strict=True):
            h = _temporal(ta, blk(torch.cat([h, inp], 1)))
        return h, u.flow_head(h), 0.25 * u.mask(h) if last else None

    def fast(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """The float16 forward pass -> flow `[B, 2, H, W]` (fp32)."""
        net: Any = self.net
        b, _, h, w = image1.shape
        x = torch.stack([image1, image2], 1).flatten(0, 1)  # [B*2, 3, H, W]: frames of a batch item adjacent
        rh = round(h * WIDTH / w / PATCH) * PATCH  # upstream fix_width
        xr = F.interpolate(x, (rh, WIDTH), mode="bilinear", align_corners=True)
        with self.accumulate():
            toks = self.aggregate(xr.half(), b)
        cnn = net.backbone(x.half().contiguous(memory_format=torch.channels_last))
        feats = net.flow_head(
            toks, images=xr.view(b, 2, 3, rh, WIDTH), patch_start_idx=net.patch_start_idx, cnn_features=cnn
        )
        del toks
        f0 = feats[:, 0]
        hf, wf = f0.shape[-2:]
        flow = global_flow(f0, feats[:, 1])
        h4, w4 = h // UP, w // UP
        flow = F.interpolate(flow, (h4, w4), mode="bilinear", align_corners=True)
        flow = flow * torch.tensor([w4 / wf, h4 / hf], device=flow.device).view(1, 2, 1, 1)
        ref = F.interpolate(f0, (h4, w4), mode="bilinear", align_corners=True)
        hid, inp = net.refine_proj(ref).chunk(2, 1)
        c = cnn[-1].view(b, 2, *cnn[-1].shape[1:])
        corr_fn = self.correlate(c[:, 0], c[:, 1])
        mask = None
        for i in range(iters):
            hid, delta, mask = self.update(hid, inp, corr_fn(flow), flow, last=i == iters - 1)
            flow = flow + delta.float()
        if mask is None:
            msg = "MegaFlow: iters must be >= 1"
            raise ValueError(msg)
        return upsample_flow_with_mask(flow, mask.float(), upsample_factor=UP)

    def accumulate(self) -> contextlib.AbstractContextManager[None]:
        return fp16_accumulate() if self.gacc else contextlib.nullcontext()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow `[B, 2, H, W]` of a padded, normalised batch."""
        iters = self.ctx.params.iters
        if self.fp16:
            with torch.autocast(image1.device.type, enabled=False):
                return self.fast(image1, image2, iters)
        return self.net(torch.stack([image1, image2], 1), num_reg_refine=iters)["flow_preds"][-1][:, 0]

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self.forward(image1, image2)}


class Portable(_Fast):
    """Plain PyTorch: float32 = upstream modules; float16 (CUDA) = the fused fp16 path with SDPA attention."""

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        return Forward(self.net, lambda _net, image1, image2: self.forward(image1, image2))


class _PatchEmbed(FastViT):
    """The shared fast DINOv2 path with upstream's position interpolation (`size=` instead of `scale_factor=`)."""

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


def _fast_patch_embed(pe: nn.Module, attn: str) -> _PatchEmbed:
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
    from fastforks.kernels.cudnn_hacc import conv_half_acc  # noqa: PLC0415 - kernels only

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


def _kernel_lookup(k: CubinModule, f0: torch.Tensor, f1: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The 9 x 9 correlation window computed on the fly by `raft_flow`'s lookup: O(HW) memory, no window tensor.

    fp16 features, fp32 sums and bilinear weights (zero outside, as upstream's grid_sample); the window is written in
    fp16 in the kernel's channel order (x offset major), which the transposed `convc1` input channels absorb.
    """
    a, o = (f.half().permute(0, 2, 3, 1).contiguous() for f in (f0, f1))
    b, h, w, c = a.shape
    side = (2 * RADIUS + 1) ** 2

    def lookup(flow: torch.Tensor) -> torch.Tensor:
        coords = (_grid(b, h, w, flow.device) + flow).permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=flow.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c128_r4_l1",
                (-(-h * w // 8), 1, 1),
                (256, 1, 1),
                a[i],
                o[i],
                h,
                w,
                coords[i],
                out[i],
                side,
                1 / math.sqrt(c),
            )
        return out.permute(0, 3, 1, 2)  # channels-last NCHW view

    return lookup


class Kernels(_Fast):
    """`fastbb` flash attention + fast DINOv2, fp16-accumulate GEMMs and convs, on-the-fly correlation windows."""

    gacc = True

    def __init__(self, ctx: Context) -> None:
        """Build the network, convert it and bind the kernels."""
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, fastbb, raft_flow  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash, self.check_flash64 = fastbb.Flash64("hh"), fastbb.check_flash64
        cudnn_hacc.load(ctx.device)
        net: Any = self.net
        self.patch_embed = _fast_patch_embed(net.patch_embed, "hh")
        net.patch_embed = nn.Identity().eval()  # the fast copy owns the weights now
        if sum(_hacc_convs(m) for m in (net.flow_head, net.backbone)) == 0:
            msg = "MegaFlow kernels: no conv routed through cudnn_hacc"
            raise RuntimeError(msg)
        c1 = net.refine.encoder.convc1  # the kernel's window is x-offset major: transpose the input channels
        side = 2 * RADIUS + 1
        c1.weight.data = c1.weight.data.view(-1, side, side, 1, 1).transpose(1, 2).reshape(c1.weight.shape).contiguous()
        self.raft = raft_flow.load(ctx.device)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The shared fast DINOv2 (`FastViT`) on the aggregator's patch embed weights."""
        return self.patch_embed.get_intermediate_layers(x, 1)[0]

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`fastbb` flash64 (fp16 in / out, fp16 P.V accumulation) on strided `[B, N, heads, 64]` views."""
        b, n, h, d = q.shape
        self.check_flash64(q, k, v)
        return self.flash(q, k, v, scale).view(b, n, h * d)

    def correlate(self, f0: torch.Tensor, f1: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """The windows from the `raft_flow` lookup kernel."""
        return _kernel_lookup(self.raft, f0, f1)

    def module(self) -> torch.nn.Module:
        """No exportable module: kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
