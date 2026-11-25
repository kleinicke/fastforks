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
import contextvars
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import OptionsError
from fastforks._core.half import fp16_accumulate
from fastforks.multiview._runner import Forward, Runner, invert
from fastforks.shared import mlx_wrap
from fastforks.shared.host_blocks import StreamedModuleList

from .net.models.aggregator import slice_expand_and_flatten
from .net.models.heads.utils import create_uv_grid, position_grid_to_embed
from .net.models.vggt_omega import VGGTOmega
from .net.utils.pose_enc import encoding_to_camera

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

    from fastforks.multiview._runner import Context

PATCH = 16  # fixed: the ViT patch size
_POS_SIZES = 20  # fixed: cache policy, head position tables: five per input size (~11 MB near 384 x 512)
SMALL_WORK = 2.2e5  # fixed: B * N * C below this: torch SDPA (fastbb's threshold)


@dataclass(frozen=True)
class Params:
    """Load options."""

    area_side: int  # upstream "balanced" resolution: about (area_side / 16)^2 patches per view
    stream: bool = False


def network_size(h: int, w: int, area_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: upstream `_balanced_target_shape` (multiples of 16)."""
    tokens = (area_side // PATCH) ** 2
    wp = math.sqrt(tokens / (h / max(w, 1)))
    hp = tokens / wp
    return max(1, round(hp)) * PATCH, max(1, round(wp)) * PATCH


def _net(ctx: Context) -> VGGTOmega:
    with torch.device("meta"):
        net = VGGTOmega(patch_size=PATCH, autocast=False)
    state = ctx.state_dict()
    for key, target in net.state_dict().items():
        if key in state and state[key].dtype != target.dtype:
            state[key] = state[key].to(target.dtype)  # preserve ordinary load_state_dict's dtype for user checkpoints
    net.load_state_dict(state, strict=True, assign=True)
    return net.eval().requires_grad_(requires_grad=False)


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
        if ctx.params.stream and ctx.device.type != "cuda":
            msg = "stream=True requires CUDA for VGGT-Omega block transfers"
            raise OptionsError(msg)
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


class _StagedCalls:
    """Stage blocks in their inference order, retaining upstream's indexed block calls for the FP32 path."""

    def __init__(self, blocks: Iterable[nn.Module], device: torch.device) -> None:
        self.source = StreamedModuleList(blocks, device)
        self.active: contextvars.ContextVar[Iterator[tuple[int, Callable[..., torch.Tensor]]]] = contextvars.ContextVar(
            "omega_blocks"
        )

    def _ordered(self) -> Generator[tuple[int, Callable[..., torch.Tensor]]]:
        yield from enumerate(self.source)

    @contextlib.contextmanager
    def session(self) -> Generator[None]:
        with contextlib.closing(self._ordered()) as blocks:
            token = self.active.set(blocks)
            try:
                yield
            finally:
                self.active.reset(token)


class _Stage(nn.Module):
    """One upstream block's position in the shared sequence of uploads."""

    def __init__(self, sequence: _StagedCalls, index: int) -> None:
        super().__init__()
        self.sequence, self.index = sequence, index

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401 - upstream or prepared block
        index, block = next(self.sequence.active.get())
        if index != self.index:
            msg = f"VGGT-Omega requested block {self.index} while staging block {index}"
            raise RuntimeError(msg)
        return block(*args, **kwargs)


class _Blk(nn.Module):
    """Prepared parameters of one upstream `SelfAttentionBlock` (fp16 GEMM weights, fp32 norms and LayerScales)."""

    def __init__(self, b: Any, device: torch.device, *, kernels: bool) -> None:  # noqa: ANN401 - vendored block
        super().__init__()
        a = b.attn

        def h(t: torch.Tensor) -> torch.Tensor:
            return nn.Buffer(t.detach().to(device=device, dtype=torch.float16).contiguous())

        def f(t: torch.Tensor) -> torch.Tensor:
            return nn.Buffer(t.detach().to(device=device, dtype=torch.float32).contiguous())

        self.heads = a.num_heads
        self.eps1, self.eps2 = b.norm1.eps, b.norm2.eps
        self.n1w, self.n1b = f(b.norm1.weight), f(b.norm1.bias)
        self.n2w, self.n2b = f(b.norm2.weight), f(b.norm2.bias)
        bias = a.qkv.bias * a.qkv.bias_mask if hasattr(a.qkv, "bias_mask") else a.qkv.bias
        self.wqkv, self.bqkv = h(a.qkv.weight), h(bias)
        self.wp, self.bp = h(a.proj.weight), h(a.proj.bias)
        self.qn: torch.Tensor | None
        self.qnf: torch.Tensor | None
        self.qk_eps = 1e-5
        if a.use_qk_norm:
            self.qn = h(torch.stack((a.q_norm.weight, a.q_norm.bias, a.k_norm.weight, a.k_norm.bias)))
            self.qnf = f(self.qn) if kernels else None  # preserve the half rounding before FP32 Q/K normalization
            self.qk_eps = a.q_norm.eps
        else:
            self.qn = self.qnf = None
        self.scale = a.scale
        self.g1, self.g2 = f(b.ls1.gamma), f(b.ls2.gamma)
        m = b.mlp
        if not isinstance(m.act, nn.GELU) or m.act.approximate != "none":
            msg = "VGGT-Omega fast path: unexpected MLP activation"
            raise ValueError(msg)
        self.w1, self.b1, self.w2, self.b2 = h(m.fc1.weight), h(m.fc1.bias), h(m.fc2.weight), h(m.fc2.bias)

    @property
    def n1(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.n1w, self.n1b

    @property
    def n2(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.n2w, self.n2b

    def forward(self, run: Callable[..., torch.Tensor], *args: Any) -> torch.Tensor:  # noqa: ANN401
        return run(self, *args)


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
        self.staged = None
        net: Any = self.net
        agg = net.aggregator
        pe = agg.patch_embed
        if ctx.dtype == "float32":
            if ctx.params.stream:
                calls = self._stage(pe.blocks, agg.frame_blocks, agg.inter_frame_blocks)
                pe.blocks, agg.frame_blocks, agg.inter_frame_blocks = (nn.ModuleList(group) for group in calls)
            net.to(ctx.device)
            return
        if not isinstance(pe.patch_embed.norm, nn.Identity) or pe.untie_cls_and_patch_norms:
            msg = "VGGT-Omega fast path: unexpected patch embed structure"
            raise ValueError(msg)
        target = torch.device("cpu") if ctx.params.stream else ctx.device
        groups = [
            [_Blk(b, target, kernels=self.gacc) for b in modules]
            for modules in (pe.blocks, agg.frame_blocks, agg.inter_frame_blocks)
        ]
        self.pe_blocks, self.frame, self.inter = self._stage(*groups) if ctx.params.stream else groups
        # The prepared weights replace the FP32 transformer. Only the heads and token/position parameters move next.
        pe.blocks, agg.frame_blocks, agg.inter_frame_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        dh = net.dense_head
        last = (dh.proj, dh.proj_conf)
        dh.proj = dh.proj_conf = None
        dh.half().to(memory_format=torch.channels_last)
        dh.proj, dh.proj_conf = last
        net.to(ctx.device)
        self.pe_conv = (pe.patch_embed.proj.weight.detach().half(), pe.patch_embed.proj.bias.detach().half())
        self.pe_norm = (pe.norm.weight.detach().float(), pe.norm.bias.detach().float(), pe.norm.eps)
        self.head_w = torch.cat([last[0].weight.flatten(1), last[1].weight.flatten(1)]).float()  # [2 r^2, 256]
        self.head_b = torch.cat([last[0].bias, last[1].bias]).float()
        self.head_norm = (dh.norm.weight.detach().float(), dh.norm.bias.detach().float(), dh.norm.eps)

    def _stage(self, patch: Any, frame: Any, inter: Any) -> tuple[list[_Stage], list[_Stage], list[_Stage]]:  # noqa: ANN401
        ordered = itertools.chain(patch, itertools.chain.from_iterable(zip(frame, inter, strict=True)))
        self.staged = _StagedCalls(ordered, self.ctx.device)
        calls = [_Stage(self.staged, i) for i in range(len(self.staged.source))]
        boundary = len(patch)
        return calls[:boundary], calls[boundary::2], calls[boundary + 1 :: 2]

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
        del qkv
        x = torch.addcmul(x, o.view(b, n, c), p.g1)
        del o
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
            tok = p(self.block, tok, rope, pre)
        w_, b_, eps = self.pe_norm
        return F.layer_norm(tok[:, pre:], (c,), w_, b_, eps)

    def _head_tokens(self, frame: torch.Tensor, inter: torch.Tensor) -> torch.Tensor:
        weights = self.head_norm
        return self.ln(torch.cat((frame, inter), dim=-1), weights[:2], weights[2])

    def aggregate(self, images: torch.Tensor) -> tuple[list[torch.Tensor | None], torch.Tensor]:
        """Retain normalized FP16 patch features and the final FP32 camera and register tokens."""
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
            tokens = self.frame[i](self.block, tokens.view(b * s, n, c), rope, pre)
            fr = tokens.view(b, s, n, c)
            if agg.inter_frame_attention_types[i] == "global":
                t4 = self.inter[i](self.block, fr.reshape(b, s * n, c)).view(b, s, n, c)
            else:  # "register": only the camera + register tokens attend across views
                cr = self.inter[i](self.block, fr[:, :, :pre].reshape(b, s * pre, c)).view(b, s, pre, c)
                t4 = torch.cat([cr, fr[:, :, pre:]], 2)
            tokens = t4
            if i in agg.cached_layer_indices:
                patches = self._head_tokens(fr[:, :, pre:], t4[:, :, pre:])
                out.append(patches.view(b, s, n - pre, 2 * c))
                del patches
            else:
                out.append(None)
            if i + 1 == agg.depth:
                return out, torch.cat((fr[:, :, :pre], t4[:, :, :pre]), dim=-1)
        msg = "VGGT-Omega aggregator has no blocks"
        raise RuntimeError(msg)

    def _pos_embed(self, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
        key = (tuple(x.shape[1:]), width, height)
        if key not in self._pos:
            if len(self._pos) >= _POS_SIZES:
                del self._pos[next(iter(self._pos))]
            pw, ph = x.shape[-1], x.shape[-2]
            pe = create_uv_grid(pw, ph, aspect_ratio=width / height, dtype=torch.float32, device=x.device)
            pe = position_grid_to_embed(pe, x.shape[1]) * 0.1
            self._pos[key] = pe.permute(2, 0, 1)[None].to(x.dtype).contiguous(memory_format=torch.channels_last)
        if x.is_cuda and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled()):
            return x.add_(self._pos[key])
        return x + self._pos[key]

    def _dense_chunk(  # noqa: PLR0913, PLR0917 - one chunk of the dense head
        self, toks: list[torch.Tensor | None], b: int, s0: int, s1: int, h: int, w: int
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
            y = dh.projects[fi](t[:, s0:s1].reshape(n, ph, pw, -1).permute(0, 3, 1, 2))
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
        stage = self.staged.session() if self.staged is not None else contextlib.nullcontext()
        with torch.autocast("cuda", enabled=False), _fp16_accumulate(on=self.gacc), stage:
            toks, camera = self.aggregate(x)
        pose = self.net.camera_head([camera], patch_token_start=camera.shape[2])  # ty: ignore[call-non-callable]
        c = self.dense_chunk
        ds, cs = zip(*[self._dense_chunk(toks, b, i, min(i + c, s), h, w) for i in range(0, s, c)], strict=True)
        return {"pose_enc": pose, "depth": torch.cat(ds, 1), "depth_conf": torch.cat(cs, 1)}


class Portable(_Fast):
    """Plain PyTorch: float32 = upstream modules; float16 (CUDA) = friendly (fp16 GEMMs, SDPA)."""

    dtypes = ("float32", "float16")

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream modules in fp32, or the fp16 path."""
        if self.ctx.dtype != "float32":
            return super().predict(x)
        with self.staged.session() if self.staged is not None else contextlib.nullcontext():
            return self.net(x)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        if self.staged is not None:
            msg = "Export requires resident VGGT-Omega weights; reload with stream=False"
            raise NotImplementedError(msg)
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class Kernels(_Fast):
    """kernels: `fastbb` flash attention (fp16 accumulation), fp16-acc GEMMs (not fc2), `vggt_omega` fusions."""

    gacc = True
    dense_chunk = 2  # bound full-resolution features while refining multiple views

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash = fastbb.Flash64("hh")
        from fastforks.kernels import vggt_omega  # noqa: PLC0415 - kernels only

        self.kv = vggt_omega.load(ctx.device)
        self.max_c = vggt_omega.LN_MAX_C
        self.qk_rows = vggt_omega.QK_ROWS

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

    def _head_tokens(self, frame: torch.Tensor, inter: torch.Tensor) -> torch.Tensor:
        tokens, width = frame.shape[-2:]
        a = frame.reshape(-1, tokens, width)
        b = inter.reshape_as(a)
        n = a.shape[0] * tokens
        out = torch.empty((n, 2 * width), dtype=torch.float16, device=a.device)
        w, bias, eps = self.head_norm
        self._ln_launch("omega_join_ln", n, a, b, out, w, bias, tokens, *a.stride()[:2], *b.stride()[:2], width, eps)
        return out

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
        del qkv
        x1 = torch.empty_like(x)
        y = torch.empty(b * n, c, device=x.device, dtype=torch.float16)
        self._ln_launch("omega_add_ln", b * n, x, o, p.g1, x1, y, p.n2[0], p.n2[1], c, p.eps2)
        del o
        hdn = F.linear(y, p.w1, p.b1)
        del y
        torch.ops.aten.gelu_(hdn)
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


class _Upstream(Portable):
    """The upstream network in any dtype (the MLX graph replaces `net` whole)."""

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream forward."""
        return self.net(x)


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": mlx_wrap.port(_Upstream)}
