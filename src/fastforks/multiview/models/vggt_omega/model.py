"""VGGT-Omega runners: weights -> network and the portable / kernels forward passes.

Upstream `VGGTOmega.forward` (camera and dense depth heads) on all views at once. Images are resized to the network
size (upstream "balanced" preprocessing: about `(area_side / 16)^2` patches per view at the input aspect ratio, both
sides multiples of 16, bicubic antialiased; upstream also centre-crops aspect ratios outside [0.5, 2], we do not) unless
they already have it; maps are resized back (bilinear), the predicted intrinsics are rescaled to the input pixels
(pixel centres at integer coordinates; the principal point is the image centre) and the predicted world-from-camera
poses are those of upstream (world = the first camera, OpenCV axes, scale as predicted).

The fp16 and kernels paths are `vggt_omega/fast.py` (`FastVGGTOmega`): DINOv3 patch
embed (24 blocks) and aggregator (24 frame + 24 inter-frame blocks) with fp16 weights / GEMMs / attention and an fp32
residual stream (upstream runs them under bf16 autocast with fp32 LayerNorms and RoPE), the k-bias mask folded into the
qkv bias, LayerScale + residual as addcmul into the fp32 stream; DPT dense head fp16 channels-last with the two final
1x1 convs (depth / confidence) as one fp32 linear; camera head fp32 as upstream. Kernels = `kernels`: the `fastbb`
head-dim-64 flash kernel with fp16 accumulation, fp16-accumulate GEMMs except the MLP fc2, and the Triton fusions as
the `vggt_omega` CUDA kernels (LayerNorm fp32 -> fp16, residual add + LayerNorm, in-place q / k LayerNorm + RoPE with
the softmax scale folded into q: the flash kernel accumulates q k^T in fp16, unscaled q overflows on real views).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks.multiview._runner import Forward, Runner, invert

from .net.models.aggregator import slice_expand_and_flatten
from .net.models.heads.utils import create_uv_grid, position_grid_to_embed
from .net.models.vggt_omega import VGGTOmega
from .net.utils.pose_enc import encoding_to_camera

if TYPE_CHECKING:
    from fastforks.multiview._runner import Context

PATCH = 16  # fixed: the ViT patch size
_POS_SIZES = 20  # fixed: cache policy, head position tables: five per input size (~11 MB near 384 x 512)
SMALL_WORK = 2.2e5  # fixed: B * N * C below this: torch SDPA (fastbb's threshold)


@dataclass(frozen=True)
class Params:
    """Load options."""

    area_side: int  # upstream "balanced" resolution: about (area_side / 16)^2 patches per view


def network_size(h: int, w: int, area_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: upstream `_balanced_target_shape` (multiples of 16)."""
    tokens = (area_side // PATCH) ** 2
    wp = math.sqrt(tokens / (h / max(w, 1)))
    hp = tokens / wp
    return max(1, round(hp)) * PATCH, max(1, round(wp)) * PATCH


def _net(ctx: Context) -> VGGTOmega:
    net = VGGTOmega(patch_size=PATCH, autocast=False)  # camera + depth heads, no text alignment (as the checkpoint)
    net.load_state_dict(ctx.state_dict(), strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def outputs(pred: dict[str, torch.Tensor], hn: int, wn: int, h0: int, w0: int) -> dict[str, torch.Tensor]:
    """Upstream predictions at the network size `hn x wn` -> the runner's maps at the input size `h0 x w0`."""

    def up(x: torch.Tensor) -> torch.Tensor:  # [S, C, hn, wn] -> [S, C, h0, w0]
        x = x.float()
        return x if (hn, wn) == (h0, w0) else F.interpolate(x, (h0, w0), mode="bilinear", align_corners=False)

    ext, k = encoding_to_camera(pred["pose_enc"].float(), (hn, wn))
    ext, k = ext[0], k[0].clone()
    k[:, 0] *= w0 / wn
    k[:, 1] *= h0 / hn
    k[:, :2, 2] -= 0.5  # upstream: principal point at (W / 2, H / 2) in corner-origin pixels
    w2c = torch.eye(4, device=ext.device).repeat(ext.shape[0], 1, 1)
    w2c[:, :3] = ext
    return {
        "depth": up(pred["depth"][0].permute(0, 3, 1, 2))[:, 0],
        "confidence": up(pred["depth_conf"][0][:, None])[:, 0],
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
        hn, wn = network_size(h0, w0, self.ctx.params.area_side)
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
        """Depth and cameras of every view, in the frame of `images[0]`."""
        return self.forward(images, intrinsics, cam_to_world, depth_range)


class _Blk:
    """Prepared parameters of one upstream `SelfAttentionBlock` (fp16 GEMM weights, fp32 norms and LayerScales)."""

    def __init__(self, b: Any) -> None:  # noqa: ANN401 - vendored block
        a = b.attn

        def h(t: torch.Tensor) -> torch.Tensor:
            return t.detach().half().contiguous()

        def f(t: torch.Tensor) -> torch.Tensor:
            return t.detach().float().contiguous()

        self.heads = a.num_heads
        self.eps1, self.eps2 = b.norm1.eps, b.norm2.eps
        self.n1 = (f(b.norm1.weight), f(b.norm1.bias))
        self.n2 = (f(b.norm2.weight), f(b.norm2.bias))
        bias = a.qkv.bias * a.qkv.bias_mask if hasattr(a.qkv, "bias_mask") else a.qkv.bias
        self.wqkv, self.bqkv = h(a.qkv.weight), h(bias)
        self.wp, self.bp = h(a.proj.weight), h(a.proj.bias)
        self.qn: tuple[torch.Tensor, ...] | None = None
        self.qnf: tuple[torch.Tensor, ...] | None = None  # fp32 copies (kernels)
        self.qk_eps = 1e-5
        if a.use_qk_norm:
            self.qn = (h(a.q_norm.weight), h(a.q_norm.bias), h(a.k_norm.weight), h(a.k_norm.bias))
            self.qk_eps = a.q_norm.eps
        self.scale = a.scale
        self.g1, self.g2 = f(b.ls1.gamma), f(b.ls2.gamma)
        m = b.mlp
        if not isinstance(m.act, nn.GELU) or m.act.approximate != "none":
            msg = "VGGT-Omega fast path: unexpected MLP activation"
            raise ValueError(msg)
        self.w1, self.b1, self.w2, self.b2 = h(m.fc1.weight), h(m.fc1.bias), h(m.fc2.weight), h(m.fc2.bias)


def _rot(x: torch.Tensor, p: int, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the tokens `>= p` of `x` `[B, N, heads, 64]` (sin / cos `[N - p, 64]` in x's dtype)."""
    xp = x[:, p:]
    r = torch.cat((-xp[..., 32:], xp[..., :32]), -1)
    y = xp * cos[None, :, None] + r * sin[None, :, None]
    return torch.cat((x[:, :p], y), 1) if p else y


def _fp16_accumulate(*, on: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if on else contextlib.nullcontext()


class _Fast(_Base):
    """`FastVGGTOmega` (see the module docstring); float32 keeps the upstream modules."""

    fast_attention = False  # fastbb flash attention for large enough calls
    gacc = False  # fp16-accumulate GEMMs (except the MLP fc2)
    dense_chunk = 8  # views per dense-head chunk (upstream frames_chunk_size)

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._pos: dict[tuple[Any, ...], torch.Tensor] = {}
        if ctx.dtype == "float32":
            return
        net: Any = self.net
        agg = net.aggregator
        pe = agg.patch_embed
        if not isinstance(pe.patch_embed.norm, nn.Identity) or pe.untie_cls_and_patch_norms:
            msg = "VGGT-Omega fast path: unexpected patch embed structure"
            raise ValueError(msg)
        self.pe_blocks = [_Blk(b) for b in pe.blocks]
        self.frame = [_Blk(b) for b in agg.frame_blocks]
        self.inter = [_Blk(b) for b in agg.inter_frame_blocks]
        self.pe_conv = (pe.patch_embed.proj.weight.detach().half(), pe.patch_embed.proj.bias.detach().half())
        self.pe_norm = (pe.norm.weight.detach().float(), pe.norm.bias.detach().float(), pe.norm.eps)
        for mod in (pe.blocks, agg.frame_blocks, agg.inter_frame_blocks):  # free the converted fp32 copies
            for p in mod.parameters():
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)
        dh = net.dense_head
        last = (dh.proj, dh.proj_conf)
        dh.proj = dh.proj_conf = None
        dh.half().to(memory_format=torch.channels_last)
        dh.proj, dh.proj_conf = last
        self.head_w = torch.cat([last[0].weight.flatten(1), last[1].weight.flatten(1)]).float()  # [2 r^2, 256]
        self.head_b = torch.cat([last[0].bias, last[1].bias]).float()
        self.head_norm = (dh.norm.weight.detach().float(), dh.norm.bias.detach().float(), dh.norm.eps)

    def ln(self, x: torch.Tensor, wb: tuple[torch.Tensor, torch.Tensor], eps: float) -> torch.Tensor:
        """LayerNorm fp32 `[.., C]` -> fp16 `[rows, C]` (contiguous, NHWC-ready)."""
        c = x.shape[-1]
        return F.layer_norm(x, (c,), wb[0], wb[1], eps).half().reshape(-1, c)

    def rope_tables(self, sin: torch.Tensor, cos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE sin / cos `[HW, 64]` in the dtype the q / k pass uses (fp16)."""
        return sin.half().contiguous(), cos.half().contiguous()

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`[B, N, heads, 64]` q / k / v -> `[B, N, heads * 64]` (torch SDPA)."""
        b, n, h, d = q.shape
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        return o.transpose(1, 2).reshape(b, n, h * d)

    def _qk_attn(self, p: _Blk, qkv: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, pre: int) -> Any:  # noqa: ANN401
        d = qkv.shape[-1]
        q, k, v = qkv.unbind(2)  # [B, N, heads, 64]
        if p.qn is not None:
            q = F.layer_norm(q, (d,), p.qn[0], p.qn[1], p.qk_eps)
            k = F.layer_norm(k, (d,), p.qn[2], p.qn[3], p.qk_eps)
        if rope is not None:
            sin, cos = rope
            q, k = _rot(q, pre, sin, cos), _rot(k, pre, sin, cos)
        return self.attention(q, k, v, p.scale)

    def block(
        self, p: _Blk, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None, pre: int = 0
    ) -> torch.Tensor:
        """One block: fp32 `[B, N, C]` (not modified) -> a new fp32 tensor."""
        b, n, c = x.shape
        qkv = F.linear(self.ln(x, p.n1, p.eps1), p.wqkv, p.bqkv).view(b, n, 3, p.heads, c // p.heads)
        o = F.linear(self._qk_attn(p, qkv, rope, pre), p.wp, p.bp)
        x = torch.addcmul(x, o.view(b, n, c), p.g1)
        y = F.gelu(F.linear(self.ln(x, p.n2, p.eps2), p.w1, p.b1))
        if self.gacc:  # fc2 accumulates in fp32 (`nofc2`)
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            o = F.linear(y, p.w2, p.b2)
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        else:
            o = F.linear(y, p.w2, p.b2)
        o = o.view(b, n, c)
        return torch.addcmul(x, o, p.g2) if torch.is_grad_enabled() else x.addcmul_(o, p.g2)

    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv3 normed patch tokens `[B*S, P, C]` (fp32) of normalised images."""
        pe: Any = self.net.aggregator.patch_embed
        t = F.conv2d(x.half(), *self.pe_conv, stride=pe.patch_size)
        bs, c, h, w = t.shape
        t = t.flatten(2).transpose(1, 2)
        cls = (pe.cls_token + 0 * pe.mask_token).float().expand(bs, -1, -1)
        tok = torch.cat([cls, pe.storage_tokens.float().expand(bs, -1, -1), t.float()], 1)
        sin, cos = pe.rope_embed(H=h, W=w)
        rope = self.rope_tables(sin, cos)
        pre = 1 + pe.n_storage_tokens
        for p in self.pe_blocks:
            tok = self.block(p, tok, rope, pre)
        w_, b_, eps = self.pe_norm
        return F.layer_norm(tok[:, pre:], (c,), w_, b_, eps)

    def aggregate(self, images: torch.Tensor) -> tuple[list[torch.Tensor | None], int]:
        """Upstream `Aggregator.forward` -> (cached token lists, patch start index)."""
        agg: Any = self.net.aggregator
        b, s, _, h, w = images.shape
        x = ((images - agg._resnet_mean) / agg._resnet_std).view(b * s, 3, h, w)  # noqa: SLF001
        pt = self.patch_embed(x)
        cam = slice_expand_and_flatten(agg.camera_token, b, s)
        reg = slice_expand_and_flatten(agg.register_token, b, s)
        tokens = torch.cat([cam, reg, pt], 1).float()
        _, n, c = tokens.shape
        sin, cos = agg.rope_embed(H=h // agg.patch_size, W=w // agg.patch_size)
        rope = self.rope_tables(sin, cos)
        pre = agg.patch_token_start
        out: list[torch.Tensor | None] = []
        for i in range(agg.depth):
            tokens = self.block(self.frame[i], tokens.view(b * s, n, c), rope, pre)
            fr = tokens.view(b, s, n, c)
            if agg.inter_frame_attention_types[i] == "global":
                t4 = self.block(self.inter[i], fr.reshape(b, s * n, c)).view(b, s, n, c)
            else:  # "register": only the camera + register tokens attend across views
                cr = self.block(self.inter[i], fr[:, :, :pre].reshape(b, s * pre, c)).view(b, s, pre, c)
                t4 = torch.cat([cr, fr[:, :, pre:]], 2)
            tokens = t4
            out.append(torch.cat([fr, t4], -1) if i in agg.cached_layer_indices else None)
        return out, pre

    def _pos_embed(self, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
        key = (tuple(x.shape[1:]), width, height)
        if key not in self._pos:
            if len(self._pos) >= _POS_SIZES:
                del self._pos[next(iter(self._pos))]
            pw, ph = x.shape[-1], x.shape[-2]
            pe = create_uv_grid(pw, ph, aspect_ratio=width / height, dtype=torch.float32, device=x.device)
            pe = position_grid_to_embed(pe, x.shape[1]) * 0.1
            self._pos[key] = pe.permute(2, 0, 1)[None].to(x.dtype).contiguous(memory_format=torch.channels_last)
        return x + self._pos[key]

    def _dense_chunk(  # noqa: PLR0913, PLR0917 - one chunk of the dense head
        self, toks: list[torch.Tensor | None], pre: int, b: int, s0: int, s1: int, h: int, w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dh: Any = self.net.dense_head
        ph, pw = h // dh.patch_size, w // dh.patch_size
        n = (s1 - s0) * b
        feats = []
        for fi, li in enumerate(dh.intermediate_layer_idx):
            t = toks[li]
            if t is None:
                msg = f"VGGT-Omega: aggregator layer {li} is not cached"
                raise RuntimeError(msg)
            y = self.ln(t[:, s0:s1, pre:], self.head_norm[:2], self.head_norm[2])  # fp16 NHWC rows
            y = dh.projects[fi](y.view(n, ph, pw, -1).permute(0, 3, 1, 2))
            feats.append(dh.resize_layers[fi](self._pos_embed(y, w, h)))
        fused = self._pos_embed(dh.scratch_forward(feats), w, h)
        r = dh.final_shuffle_factor
        o = F.linear(fused.permute(0, 2, 3, 1).float(), self.head_w, self.head_b)  # [n, h, w, 2 r^2] fp32
        o = F.pixel_shuffle(o.permute(0, 3, 1, 2), r)  # [n, 2, H, W]: depth, confidence logits
        depth = torch.exp(o[:, 0])[..., None]
        conf = 1.0 + torch.exp(o[:, 1])
        return depth.view(b, s1 - s0, h, w, 1), conf.view(b, s1 - s0, h, w)

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """`FastVGGTOmega.__call__`."""
        b, s, _, h, w = x.shape
        with torch.autocast("cuda", enabled=False), _fp16_accumulate(on=self.gacc):
            toks, pre = self.aggregate(x)
        pose = self.net.camera_head(toks, patch_token_start=pre)  # ty: ignore[call-non-callable] - built with the camera head
        c = self.dense_chunk
        ds, cs = zip(*[self._dense_chunk(toks, pre, b, i, min(i + c, s), h, w) for i in range(0, s, c)], strict=True)
        return {"pose_enc": pose, "depth": torch.cat(ds, 1), "depth_conf": torch.cat(cs, 1)}


class Portable(_Fast):
    """Plain PyTorch: float32 = upstream modules; float16 (CUDA) = friendly (fp16 GEMMs, SDPA)."""

    dtypes = ("float32", "float16")

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream modules in fp32, or the fp16 path."""
        return self.net(x) if self.ctx.dtype == "float32" else super().predict(x)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class Kernels(_Fast):
    """kernels: `fastbb` flash attention (fp16 accumulation), fp16-acc GEMMs (not fc2), `vggt_omega` fusions."""

    gacc = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash = fastbb.Flash64("hh")
        from fastforks.kernels import vggt_omega  # noqa: PLC0415 - kernels only

        self.kv = vggt_omega.load(ctx.device)
        self.max_c = vggt_omega.LN_MAX_C
        self.qk_rows = vggt_omega.QK_ROWS
        for p in (*self.pe_blocks, *self.frame, *self.inter):  # fp32 q / k norm parameters for the kernel
            p.qnf = None if p.qn is None else tuple(t.float().contiguous() for t in p.qn)

    def rope_tables(self, sin: torch.Tensor, cos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 tables (the kernel rotates in fp32, as the Triton kernel)."""
        return sin.float().contiguous(), cos.float().contiguous()

    def _ln_launch(self, name: str, rows: int, *args: Any) -> None:  # noqa: ANN401 - kernel args
        self.kv.launch(name, (rows, 1, 1), (256, 1, 1), *args)

    def ln(self, x: torch.Tensor, wb: tuple[torch.Tensor, torch.Tensor], eps: float) -> torch.Tensor:
        """`omega_ln_f2h`: fp32 `[.., C]` rows (strided, unit stride in C) -> fp16 `[rows, C]`."""
        c = x.shape[-1]
        if x.dim() > 3:  # noqa: PLR2004 - rows as [R, T] with strides (a view when the leading dims collapse)
            x = x.reshape(-1, x.shape[-2], c)
        r, t, sr, st = (x.shape[0], 1, x.stride(0), 0) if x.dim() == 2 else (*x.shape[:2], *x.stride()[:2])  # noqa: PLR2004
        if (
            x.dtype != torch.float32
            or x.stride(-1) != 1
            or c % 4
            or c > self.max_c
            or (sr % 4 or st % 4 or x.data_ptr() % 16)
        ):
            msg = f"VGGT-Omega kernels LayerNorm: unsupported input {tuple(x.shape)} {x.dtype} strides {x.stride()}"
            raise ValueError(msg)
        y = torch.empty(r * t, c, device=x.device, dtype=torch.float16)
        self._ln_launch("omega_ln_f2h", r * t, x, y, wb[0], wb[1], t, sr, st, c, eps)
        return y

    def _qk_attn(self, p: _Blk, qkv: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, pre: int) -> Any:  # noqa: ANN401
        """`omega_qk_norm_rope` in place on the packed qkv (q / k LayerNorm, RoPE, softmax scale into q), attention."""
        b, n, _, h, d = qkv.shape
        if d != 64 or not qkv.is_contiguous() or qkv.dtype != torch.float16:  # noqa: PLR2004 - the kernel's head dim
            msg = f"VGGT-Omega kernels: unsupported qkv {tuple(qkv.shape)} {qkv.dtype}"
            raise ValueError(msg)
        z = qkv  # placeholder pointer for absent tables
        qw, qb, kw, kb = p.qnf if p.qnf is not None else (z,) * 4
        sin, cos = rope if rope is not None else (z, z)
        if rope is not None and sin.shape != (n - pre, 64):
            msg = f"VGGT-Omega kernels: RoPE tables {tuple(sin.shape)} for {n} tokens, {pre} not rotated"
            raise ValueError(msg)
        rows = b * n * 2 * h
        grid = ((rows + self.qk_rows - 1) // self.qk_rows, 1, 1)
        self.kv.launch(
            "omega_qk_norm_rope", grid, (32 * self.qk_rows, 1, 1), qkv, sin, cos, qw, qb, kw, kb, rows, n, h, pre,
            p.qk_eps, p.scale, int(p.qnf is not None), int(rope is not None)
        )  # fmt: skip
        q, k, v = qkv.unbind(2)
        return self.attention(q, k, v, 1.0)

    def block(
        self, p: _Blk, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None, pre: int = 0
    ) -> torch.Tensor:
        """One block with the fused residual add + LayerNorm (`omega_add_ln`); fp32 `[B, N, C]` -> a new tensor."""
        b, n, c = x.shape
        if not x.is_contiguous():
            msg = "VGGT-Omega kernels: the residual stream must be contiguous"
            raise ValueError(msg)
        qkv = F.linear(self.ln(x, p.n1, p.eps1), p.wqkv, p.bqkv).view(b, n, 3, p.heads, c // p.heads)
        o = F.linear(self._qk_attn(p, qkv, rope, pre), p.wp, p.bp)
        x1 = torch.empty_like(x)
        y = torch.empty(b * n, c, device=x.device, dtype=torch.float16)
        self._ln_launch("omega_add_ln", b * n, x, o, p.g1, x1, y, p.n2[0], p.n2[1], c, p.eps2)
        hdn = F.gelu(F.linear(y, p.w1, p.b1))
        torch.backends.cuda.matmul.allow_fp16_accumulation = False  # fc2 accumulates in fp32 (`nofc2`)
        o = F.linear(hdn, p.w2, p.b2)
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        return x1.addcmul_(o.view(b, n, c), p.g2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """`fastbb` flash64 (fp16 in / out, fp16 P.V accumulation) on `[B, N, heads, 64]` views; SDPA when tiny."""
        b, n, h, d = q.shape
        if d != 64 or b * n * h * d < SMALL_WORK:  # noqa: PLR2004 - the kernel's head dim
            return super().attention(q, k, v, scale)
        return self.flash(q, k, v, scale).view(b, n, h * d)


RUNNERS = {"portable": Portable, "kernels": Kernels}
