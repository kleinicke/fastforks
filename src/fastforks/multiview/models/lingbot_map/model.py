"""LingBot-Map runners: weights -> network and the portable / kernels streaming passes.

Upstream `GCTStream.inference_streaming` (SDPA backend, KV cache, one frame per step): the first `scale_frames` views
run together (bidirectional global attention), then every further view attends to the cached keys / values of the
scale frames, of the last `window` frames and to the 6 special tokens of every frame that left the window. Images are
resized as for VGGT (long side `long_side`, multiples of 14, bicubic antialiased); maps are resized back (bilinear),
intrinsics rescaled to the input pixels and the predicted camera-to-world poses returned as they are (world = the first
camera, OpenCV axes, scale as predicted).

One streaming pass for every mode: frame blocks as VGGT's fused fp16 path (`vggt/model.py`), global blocks with the
q/k LayerNorm + 3D RoPE fused (q/k channels permuted per head so the interleaved RoPE pairs become two halves), the KV
cache as one preallocated buffer per layer whose live keys are one contiguous slice (no concatenation per frame), the
depth head per chunk of frames and the camera head once at the end over all camera tokens with the streaming mask
(each frame sees the scale frames and every earlier frame: one token per frame, so no frame leaves its window).
float32 runs it in fp32. Kernels: VGGT's (`fastbb` flash for the frame blocks and the scale frames, fp16-accumulate
GEMMs, cuDNN fp16-accumulate DPT convs, fast DINOv2 patch embed).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.multiview._runner import Forward, Runner
from fastforks.multiview.models.vggt import model as vggt
from fastforks.multiview.models.vggt.net.heads.camera_head import modulate
from fastforks.multiview.models.vggt.net.heads.head_act import activate_pose
from fastforks.multiview.models.vggt.net.utils.pose_enc import pose_encoding_to_extri_intri
from fastforks.shared.rope2d import rope_forward

from .net import PATCH, SPECIAL, Net

if TYPE_CHECKING:
    from fastforks.multiview._runner import Context

LAYERS = (4, 11, 17, 23)  # aggregator layers the heads read
CHUNK = 8  # fixed: frames per depth-head call (upstream `frames_chunk_size`)
FHW = (20, 22, 22)  # 3D RoPE: head-dim channels for frame index, row, column (upstream `WanRotaryPosEmbed`)
THETA = 10000.0  # fixed: upstream RoPE base


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (both sides rounded to multiples of 14)
    scale_frames: int  # first views processed together; every later view attends to them
    window: int  # KV-cache window in frames (older frames keep only their 6 special tokens)


def _net(ctx: Context) -> Net:
    net = Net()
    net.load_state_dict(ctx.state_dict(), strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def outputs(
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor], hn: int, wn: int, h0: int, w0: int
) -> dict[str, torch.Tensor]:
    """Pose encodings `[S, 9]`, depth, confidence `[S, hn, wn]` -> the runner's maps at the input size `h0 x w0`."""
    pose, depth, conf = pred

    def up(x: torch.Tensor) -> torch.Tensor:  # [S, hn, wn] -> [S, h0, w0]
        x = x.float()[:, None]
        return (x if (hn, wn) == (h0, w0) else F.interpolate(x, (h0, w0), mode="bilinear", align_corners=False))[:, 0]

    c2w3, k = pose_encoding_to_extri_intri(pose[None].float(), (hn, wn))
    c2w3, k = c2w3[0], k[0].clone()
    k[:, 0] *= w0 / wn
    k[:, 1] *= h0 / hn
    k[:, :2, 2] -= 0.5  # principal point at (W / 2, H / 2) in corner-origin pixels
    c2w = torch.eye(4, device=c2w3.device).repeat(c2w3.shape[0], 1, 1)
    c2w[:, :3] = c2w3  # LingBot-Map's pose encoding is camera-to-world (VGGT's is world-to-camera)
    return {"depth": up(depth), "confidence": up(conf), "intrinsics": k, "cam_to_world": c2w}


def _perm(device: torch.device) -> torch.Tensor:
    """Per-head channel order that turns RoPE's interleaved pairs into halves: [0, 2, .., 62, 1, 3, .., 63]."""
    return torch.cat([torch.arange(0, 64, 2), torch.arange(1, 64, 2)]).to(device)


class _Cache:
    """Per-layer KV buffers `[1, L, heads, 64]` whose live keys are one contiguous slice at every step.

    Layout: [special tokens of evicted frames (filled right to left) | scale frames | ring of `window` frames].
    """

    def __init__(self, n_layers: int, n_img: int, s: int, win: int, p: int, like: torch.Tensor) -> None:  # noqa: PLR0913, PLR0917
        self.s, self.win, self.p, self.evicted = s, win, p, 0
        self.ring = min(win, n_img - s)
        self.spec = SPECIAL * max(0, n_img - s - win)
        total = self.spec + p * (s + self.ring)
        self.kv = [(like.new_empty(1, total, 16, 64), like.new_empty(1, total, 16, 64)) for _ in range(n_layers)]

    def step(self, fr: range) -> tuple[slice, slice]:
        """Frames `fr` (the scale frames, or one later frame) -> (where their k / v go, the keys they attend to)."""
        s, p, spec = self.s, self.p, self.spec
        if fr.start < s:  # the scale frames: bidirectional, fill the scale slots
            put = slice(spec, spec + p * s)
        else:
            at = spec + p * (s + (fr.start - s) % self.win)
            if fr.start - s >= self.win:  # frame start - win leaves the window: keep its special tokens
                self.evicted += 1
                dst = slice(spec - SPECIAL * self.evicted, spec - SPECIAL * (self.evicted - 1))
                for kb, vb in self.kv:
                    kb[:, dst] = kb[:, at : at + SPECIAL]
                    vb[:, dst] = vb[:, at : at + SPECIAL]
            put = slice(at, at + p)
        end = spec + p * (s + min(self.ring, max(0, fr.stop - s)))
        return put, slice(spec - SPECIAL * self.evicted, end)


class _Fast(vggt._Fast):  # noqa: SLF001 - the VGGT fast path is this family's base
    """The streaming pass (see the module docstring); frame blocks, patch embed and DPT setup from VGGT."""

    Params = Params
    max_views = 1024  # upstream 3D RoPE table length

    def __init__(self, ctx: Context) -> None:
        Runner.__init__(self, ctx)
        self.net: Any = _net(ctx)
        self._aff = {}
        self.dt = torch.float32 if ctx.dtype == "float32" else torch.float16
        agg = self.net.aggregator
        b = agg.global_blocks[0].attn
        if b.head_dim != 64 or b.num_heads != 16:  # noqa: PLR2004 - fused 3D RoPE layout
            msg = "LingBot-Map: unexpected aggregator structure"
            raise ValueError(msg)
        p = _perm(ctx.device)
        for blk in agg.global_blocks:  # permute q / k channels inside each head (q.k is unchanged)
            a = blk.attn
            rows = (torch.arange(2 * a.num_heads, device=ctx.device)[:, None] * 64 + p).flatten()
            a.qkv.weight.data[: rows.numel()] = a.qkv.weight.data[rows]
            a.qkv.bias.data[: rows.numel()] = a.qkv.bias.data[rows]
            for n in (a.q_norm, a.k_norm):
                n.weight.data, n.bias.data = n.weight.data[p], n.bias.data[p]
        self._gaff = [
            (
                torch.stack([blk.attn.q_norm.weight, blk.attn.k_norm.weight])[:, None].to(self.dt),
                torch.stack([blk.attn.q_norm.bias, blk.attn.k_norm.bias])[:, None].to(self.dt),
            )
            for blk in agg.global_blocks
        ]  # [2, 1, 64] (q, k)
        if ctx.dtype == "float32":
            return
        agg.half()
        h = self.net.depth_head
        h.half().to(memory_format=torch.channels_last)
        oc = h.scratch.output_conv2  # [3x3 conv, ReLU, 1x1 conv]: only the last 1x1 conv in fp32
        h.scratch.output_conv2 = nn.Sequential(oc[0], oc[1], vggt._FP32Out(oc[2])).eval()  # noqa: SLF001
        self._gaff = [(w.half(), b.half()) for w, b in self._gaff]

    def gattention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """Global attention: `[1, N, heads, 64]` q, `[1, L, heads, 64]` k / v -> `[1, N, heads * 64]`."""
        b, n, h, d = q.shape
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        return o.transpose(1, 2).reshape(b, n, h * d)

    def rope3d(self, frames: range, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """3D RoPE tables `[len(frames) * P, 1, 1, 64]` for the permuted channels: cos and sign-folded sin."""
        dev = self.ctx.device
        om = [1.0 / THETA ** (torch.arange(0, d, 2, dtype=torch.float64, device=dev) / d) for d in FHW]
        i = torch.arange(SPECIAL, dtype=torch.float64, device=dev)  # special token t sits at (f, t, t)
        r = torch.arange(rows, dtype=torch.float64, device=dev) + SPECIAL
        c = torch.arange(cols, dtype=torch.float64, device=dev) + SPECIAL
        hh = torch.cat([i, r[:, None].expand(rows, cols).flatten()])
        ww = torch.cat([i, c[None].expand(rows, cols).flatten()])
        f = torch.arange(frames.start, frames.stop, dtype=torch.float64, device=dev)
        p = hh.numel()
        ang = torch.cat(
            [
                (f[:, None, None] * om[0]).expand(-1, p, -1),
                (hh[:, None] * om[1]).expand(len(frames), -1, -1),
                (ww[:, None] * om[2]).expand(len(frames), -1, -1),
            ],
            -1,
        ).flatten(0, 1)  # [F*P, 32]
        cos, sin = ang.cos().to(self.dt), ang.sin().to(self.dt)
        return torch.cat([cos, cos], -1)[:, None, None], torch.cat([-sin, sin], -1)[:, None, None]

    def gblock(
        self,
        i: int,
        x: torch.Tensor,
        tab: tuple[torch.Tensor, torch.Tensor],
        kv: tuple[torch.Tensor, torch.Tensor],
        slots: tuple[slice, slice],
    ) -> torch.Tensor:
        """Global block `i` on `[1, N, C]`: its k / v go to `kv[.][put]`, it attends to `kv[.][keys]`."""
        put, keys = slots
        blk = self.net.aggregator.global_blocks[i]
        a = blk.attn
        bsz, n, c = x.shape
        nh, d = a.num_heads, c // a.num_heads
        xn = blk.norm1(x)
        qk = F.linear(xn, a.qkv.weight[: 2 * c], a.qkv.bias[: 2 * c])
        nrm = F.group_norm(qk.view(bsz * n, 2 * nh, d), 2 * nh, None, None, a.q_norm.eps).view(bsz, n, 2, nh, d)
        w, bb = self._gaff[i]
        y = torch.addcmul(bb, nrm, w)
        cos, sin = tab
        y = torch.addcmul(y * cos, y.roll(32, -1), sin)
        kb, vb = kv
        kb[:, put] = y[:, :, 1]
        vb[:, put] = F.linear(xn, a.qkv.weight[2 * c :], a.qkv.bias[2 * c :]).view(bsz, n, nh, d)
        o = F.linear(self.gattention(y[:, :, 0], kb[:, keys], vb[:, keys], a.scale), a.proj.weight, a.proj.bias)
        x = torch.addcmul(x, o, blk.ls1.gamma)
        m = blk.mlp
        o = F.linear(F.gelu(F.linear(blk.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        return x.addcmul_(o, blk.ls2.gamma)

    def tokens(self, x: torch.Tensor, frames: range, s: int) -> torch.Tensor:
        """Normalised network images `[F, 3, h, w]` of `frames` -> tokens `[F, P, C]` (specials + patches)."""
        agg = self.net.aggregator
        pt = self.patch_tokens(x)
        idx = torch.tensor([0 if g == 0 else 1 for g in frames], device=x.device)
        sdx = torch.tensor([0 if g < s else 1 for g in frames], device=x.device)
        return torch.cat(
            [agg.camera_token[0, idx], agg.register_token[0, idx], agg.scale_token[0, sdx], pt.to(self.dt)], 1
        )

    def stream(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Images `[S, 3, hn, wn]` -> camera tokens `[S, 2C]` (fp32), depth and confidence `[S, hn, wn]`."""
        agg = self.net.aggregator
        prm = self.ctx.params
        n_img, _, h, w = images.shape
        s = min(prm.scale_frames, n_img)
        rows, cols = h // PATCH, w // PATCH
        p = SPECIAL + rows * cols
        c = agg.global_blocks[0].attn.qkv.in_features
        dev = images.device
        cache = _Cache(len(agg.global_blocks), n_img, s, prm.window, p, images.new_empty(0, dtype=self.dt))
        pos = agg.position_getter(1, rows, cols, device=dev) + 1
        pos = torch.cat([torch.zeros(1, SPECIAL, 2, device=dev, dtype=pos.dtype), pos], 1)
        tf = self.rope_tables(pos, 1, 1, p)  # frame blocks: [1, P, 1, 1, 64]
        x = ((images - agg._resnet_mean[0]) / agg._resnet_std[0]).to(self.dt)  # noqa: SLF001
        cam, depth, conf = [], [], []
        pend: list[list[torch.Tensor]] = [[] for _ in LAYERS]
        pend_img: list[torch.Tensor] = []

        def heads(force: bool) -> None:  # noqa: FBT001
            if not pend_img or (not force and sum(t.shape[0] for t in pend_img) < CHUNK):
                return
            toks: list[torch.Tensor | None] = [None] * len(agg.global_blocks)
            for j, li in enumerate(LAYERS):
                toks[li] = torch.cat(pend[j])[None]
                pend[j].clear()
            xi = torch.cat(pend_img)[None]
            pend_img.clear()
            d, cf = self.net.depth_head(toks, images=xi, patch_start_idx=SPECIAL)
            depth.append(d[0, ..., 0].float())
            conf.append(cf[0].float())

        for fr in [range(s)] + [range(g, g + 1) for g in range(s, n_img)]:
            nf = len(fr)
            t = self.tokens(x[fr.start : fr.stop], fr, s)
            tg = self.rope3d(fr, rows, cols)
            slots = cache.step(fr)
            for i in range(len(agg.frame_blocks)):
                t = self.block(agg.frame_blocks[i], t, tf)
                fo = t if i in LAYERS else None  # the global block does not write into its input
                t = self.gblock(i, t.view(1, nf * p, c), tg, cache.kv[i], slots).view(nf, p, c)
                if fo is not None:
                    pend[LAYERS.index(i)].append(torch.cat([fo, t], -1))
            cam.append(pend[-1][-1][:, 0].float())
            pend_img.append(images[fr.start : fr.stop].to(self.dt))
            heads(force=False)
        heads(force=True)
        return torch.cat(cam), torch.cat(depth), torch.cat(conf)

    def camera(self, tok: torch.Tensor, s: int) -> torch.Tensor:
        """Upstream `CameraCausalHead` on all camera tokens `[S, 2C]` at once with the streaming mask -> `[S, 9]`."""
        ch = self.net.camera_head
        n = tok.shape[0]
        g = torch.arange(n, device=tok.device)
        mask = (g[None] < s) | (g[None] <= g[:, None])  # query frame (row) sees key frame (column)
        pt = ch.token_norm(tok[None])
        pred = None
        for _ in range(4):  # upstream camera_num_iterations
            inp = ch.embed_pose(ch.empty_pose_tokens.expand(1, n, -1) if pred is None else pred)
            shift, scale, gate = ch.poseLN_modulation(inp).chunk(3, dim=-1)
            x = gate * modulate(ch.adaln_norm(pt), shift, scale) + pt
            for blk in ch.trunk:
                a = blk.attn
                qkv = a.qkv(blk.norm1(x)).view(1, n, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
                o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
                x = x + blk.ls1(a.proj(o.transpose(1, 2).reshape(1, n, -1)))
                x = x + blk.ls2(blk.mlp(blk.norm2(x)))
            delta = ch.pose_branch(ch.trunk_norm(x))
            pred = delta if pred is None else pred + delta
        out = activate_pose(pred, trans_act=ch.trans_act, quat_act=ch.quat_act, fl_act=ch.fl_act)
        return out[0]

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Network images `[S, 3, hn, wn]` -> pose encodings `[S, 9]`, depth, confidence `[S, hn, wn]`."""
        with torch.autocast(x.device.type, enabled=False):
            with fp16_accumulate() if self.gacc else contextlib.nullcontext():
                tok, depth, conf = self.stream(x)
            pose = self.camera(tok, min(self.ctx.params.scale_frames, x.shape[0]))
        return pose, depth, conf

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
        return outputs(self.run(x), hn, wn, h0, w0)


class Portable(_Fast):
    """Plain PyTorch streaming pass: float32, or float16 on CUDA (friendly: fp16 aggregator, SDPA)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        rope: Any = self.net.aggregator.rope
        if rope is not None:
            override(rope, rope_forward)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class Kernels(_Fast):
    """VGGT's kernels: `fastbb` flash (frame blocks, scale frames), fp16-accumulate GEMMs, cuDNN half-acc DPT convs."""

    gacc = True
    patch_tokens = vggt.Kernels.patch_tokens
    attention = vggt.Kernels.attention

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash, self.check_flash64 = fastbb.Flash64("hh"), fastbb.check_flash64
        cudnn_hacc.load(ctx.device)
        agg = self.net.aggregator
        self.patch_embed = vggt.fast_patch_embed(agg.patch_embed, "hh")
        agg.patch_embed = nn.Identity().eval()  # the fast copy owns the weights now
        if vggt._hacc_convs(self.net.depth_head) == 0:  # noqa: SLF001
            msg = "LingBot-Map kernels: no DPT conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def gattention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """Flash for the scale frames (as many keys as queries), SDPA for the streamed frames."""
        return self.attention(q, k, v, scale) if q.shape == k.shape else super().gattention(q, k, v, scale)


RUNNERS = {"portable": Portable, "kernels": Kernels}
