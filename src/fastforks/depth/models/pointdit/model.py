"""PointDiT runners: DINOv3 features (`fastforks.features`) -> a few Euler steps of the point-map DiT.

As upstream's wild-image path (`engine.py`, `--eval_wild_images`): keep the aspect ratio and resize (bilinear,
`align_corners=False`, no antialias) to the multiple-of-16 size whose token count is closest to `tokens` (training:
512 x 512 = 1024 tokens), predict, and resize the point map back with nearest. `generate`: the ODE starts from zeros,
`steps` Euler steps on `linspace(0, 1, steps + 1)` (the last one returns the prediction itself). The DINOv3 features
(the patch tokens of 4 equally spaced blocks, final LayerNorm) are computed once per image.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn.functional as F

from fastforks._core import kernels, registry
from fastforks._core.checkpoint import check_fits, load_state_dict
from fastforks._core.errors import InputError
from fastforks._core.model import load_variant
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.features._api import TASK, FeaturesModel
from fastforks.kernels.dpt_fused import DEEP_K, DptFused
from fastforks.kernels.fastbb import Flash64
from fastforks.kernels.fastbb import load as load_kernels
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.depth_anything_v2.fast_vit import _flash, _fp16_accumulate

from .net import (
    CONFIGS,
    HEAD,
    PATCH,
    Block,
    PointDiT,
    checkpoint_shapes,
    encoder_layers,
    rms,
    rope_angles,
    sample,
    unpatchify,
)

if TYPE_CHECKING:
    from fastforks.depth._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: str  # "b" or "l"
    encoder: str  # fastforks.features DINOv3 id
    tokens: int  # target token count of the resize (upstream: (img_size / 16)^2)
    steps: int  # Euler steps (upstream eval: 3)


def network_size(h: int, w: int, tokens: int) -> tuple[int, int]:
    """Upstream's wild-image size: aspect kept, sides rounded to multiples of 16, about `tokens` patches."""
    f = (tokens * PATCH**2 / (h * w)) ** 0.5
    return max(PATCH, round(h * f / PATCH) * PATCH), max(PATCH, round(w * f / PATCH) * PATCH)


_GRAPHS = 8  # fixed: graph cache policy (sampler graphs and RoPE tables kept per model)
_GRAPH_WORK = 2**22  # fixed: largest token stream (B x N x C) sampled through a CUDA graph
_INT32 = 2**31 - 1  # fixed: `rope_qk_h` indexes in int32
_ALIGN = 64  # fixed: the FFN width is zero-padded to this multiple for the `dpt_fused` tiles (PointDiT-L: 2730)


def _pad_ffn(blk: Block) -> None:
    """Zero-pad the SwiGLU width to `_ALIGN`: padded gates are silu(0) * 0 = 0 and meet zero `w3` columns."""
    f = blk.w3.shape[1]
    pad = -f % _ALIGN
    if pad:
        w, b = blk.w12, blk.b12
        blk.w12 = torch.cat([F.pad(u, (0, 0, 0, pad)) for u in w.chunk(2, 0)]).contiguous()
        blk.b12 = torch.cat([F.pad(u, (0, pad)) for u in b.chunk(2, -1)], -1).contiguous()
        blk.w3 = F.pad(blk.w3, (0, pad)).contiguous()


class Portable(Runner):
    """Plain PyTorch: `fastforks.features` DINOv3 portable and the folded DiT (SDPA), float32 or fp16 on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    encoder_mode: ClassVar[Literal["portable", "kernels"]] = "portable"

    def __init__(self, ctx: Context) -> None:
        """Build the DINOv3 backbone from its checkpoint (read once, on the device) and the folded DiT."""
        super().__init__(ctx)
        p = ctx.params
        self.cfg = CONFIGS[p.arch]
        self.layers = encoder_layers(self.cfg)
        card, variant = registry.resolve(TASK, p.encoder)
        path = ctx.weights()[0]
        state = {k: v.to(ctx.device) for k, v in load_state_dict(path).items()}
        ln = (state["norm.weight"].clone(), state["norm.bias"].clone())
        dtype = None if self.encoder_mode == "kernels" else ctx.dtype
        self.encoder = load_variant(
            TASK, card, variant, FeaturesModel, mode=self.encoder_mode, device=ctx.device, dtype=dtype,
            weights=path, state=state,
        )  # fmt: skip
        del state
        sd = ctx.state_dict(1, key="model_ema1", strip="net.")
        with torch.device("meta"):
            own = {k: torch.empty(s) for k, s in checkpoint_shapes(self.cfg).items()}
        check_fits(own, sd, f"{ctx.variant.ref} checkpoint")
        sd = {k: v.to(ctx.device) for k, v in sd.items()}
        half = self.encoder_mode == "kernels" or ctx.dtype == "float16"
        with torch.no_grad():  # constants only
            self.net = PointDiT(sd, ln, self.cfg, p.steps).cast(torch.float16 if half else torch.float32)
        del sd

    def _sample(self, cond: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return sample(self.net, cond, h, w)

    def points(self, image: torch.Tensor) -> torch.Tensor:
        """`[B, 3, H, W]` in `[0, 1]` -> points `[B, H, W, 3]` float32 at the input size."""
        size = image.shape[-2:]
        nh, nw = network_size(*size, self.ctx.params.tokens)
        x = image if (nh, nw) == tuple(size) else F.interpolate(image, (nh, nw), mode="bilinear", align_corners=False)
        hidden = self.encoder._runner(scale(x, self.ctx.io), self.layers)["hidden_states"]  # noqa: SLF001
        gh, gw = nh // PATCH, nw // PATCH
        pts = unpatchify(self._sample(self.net.condition(hidden, gh, gw), gh, gw), gh, gw)
        if (nh, nw) != tuple(size):
            pts = F.interpolate(pts.permute(0, 3, 1, 2), size, mode="nearest").permute(0, 2, 3, 1)
        return pts.contiguous()

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant point map and its z of a batch."""
        pts = self.points(image)
        return {"points": pts, "depth": pts[..., 2].contiguous()}

    def module(self) -> torch.nn.Module:
        """Backbone + DiT: `[B, 3, H, W]` in `[0, 1]` -> the depth map (point z) `[B, H, W]`, as the model call."""
        nets = torch.nn.ModuleDict({"encoder": self.encoder._runner.module(), "dit": self.net})  # noqa: SLF001
        return Forward(nets, lambda _net, image: self.points(image)[..., 2]).eval()

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The backbone's networks and the DiT."""
        return (*self.encoder._runner.networks(), self.net)  # noqa: SLF001


class Kernels(Portable):
    """`fastforks.features` DINOv3 kernels; the DiT on `dpt_fused` GEMMs, fastbb RoPE + flash attention, CUDA graphs.

    fp16 weights and stream. The GEMM epilogues take the folded biases, the per-step gates (`scale`) and the
    residual; RoPE runs in place on the packed q / k (q also takes the softmax scale 2^-3) after their RMSNorm (PyTorch,
    float32 statistics); SwiGLU's gate in place. The whole sampler (`steps` x blocks) replays one CUDA graph per input
    shape, captured on the shape's second call.
    """

    encoder_mode = "kernels"

    def __init__(self, ctx: Context) -> None:
        """The portable setup in fp16 + the kernels."""
        super().__init__(ctx)
        for blk in self.net.layers:
            _pad_ffn(blk)
        self.kernels = load_kernels(ctx.device)
        self.k = DptFused(ctx.device)
        self.attn = Flash64("fh")
        self._cs: dict[tuple[int, int], torch.Tensor] = {}
        self._hw = (0, 0)  # the patch grid of the call in flight (set under `_lock`)
        self._cache = GraphCache(self._graph_sample, graphs=_GRAPHS)
        self._lock = threading.Lock()

    def _table(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """fp32 `[h * w, 2, 32]`: cos | sin of the first half (the second half repeats it), as `rope_qk_h` reads it."""
        if (h, w) not in self._cs:
            if len(self._cs) >= _GRAPHS:
                del self._cs[next(iter(self._cs))]
            a = rope_angles(h, w, device)
            self._cs[h, w] = torch.stack((a.cos(), a.sin()), 1).contiguous()
        return self._cs[h, w]

    def _block(self, blk: Block, s: int, x: torch.Tensor, cs: torch.Tensor, rope: tuple) -> torch.Tensor:
        b, n, _ = x.shape
        heads, k = self.cfg.heads, self.k
        qkv = k.gemm(rms(x, blk.a1[s]), blk.qkv_w, blk.qkv_b[s]).view(b, n, 3, heads, HEAD)
        qk = qkv[:, :, :2]
        qk.copy_(rms(qk, blk.qk_norm))
        self.kernels.launch("rope_qk_h", *rope[0], qkv, cs, n, 0, heads, rope[1], HEAD**-0.5)
        x = k.gemm(_flash(qkv, 1.0, self.attn), blk.proj_w, blk.proj_b, scale=blk.g1[s], res=x)
        g, u = k.gemm(rms(x, blk.a2[s]), blk.w12, blk.b12[s]).chunk(2, -1)
        hid = F.silu(g, inplace=True).mul_(u)
        if blk.w3.shape[1] >= DEEP_K:  # deep down GEMMs: cuBLAS (fp32 accumulate)
            return x.addcmul_(F.linear(hid, blk.w3, blk.b3), blk.g2h[s])
        return k.gemm(hid, blk.w3, blk.b3, scale=blk.g2[s], res=x, out=x)

    def _sample_k(self, cond: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """`sample` on the kernels: `cond` `[B, N, C]` float32 -> point tokens float32."""
        net, k = self.net, self.k
        cs = self._table(h, w, cond.device)
        b, n, _ = cond.shape
        total = b * n * 2 * self.cfg.heads * (HEAD // 4)  # q / k half2 pairs per half head
        grid = kernels.linear(total)
        if total + grid[0][0] * grid[1][0] > _INT32:  # the grid-stride index must not overflow
            msg = f"batch too large for the fastbb RoPE kernel ({b} x {n} tokens): split it"
            raise InputError(msg)
        rope = (grid, total)
        c16 = cond.half()
        z = None
        for s, (step, one_minus_t) in enumerate(net.schedule):
            x = c16 if z is None else k.gemm(z.half(), net.wp, res=c16, acc32=True)
            for blk in net.layers:
                x = self._block(blk, s, x, cs, rope)
            out = k.gemm(rms(x, 1.0), net.final_w[s], net.final_b[s], acc32=True).float()
            z = out.mul_(step / one_minus_t) if z is None else z.add_(out.sub_(z), alpha=step / one_minus_t)
        if z is None:  # steps >= 1 (the card's sampler)
            msg = "no sampling steps"
            raise ValueError(msg)
        return z

    def _graph_sample(self, cond: torch.Tensor) -> torch.Tensor:
        return self._sample_k(cond, *self._hw)

    def release_graphs(self) -> None:
        """Drop every CUDA graph and free its memory pool on the device (`torch.cuda.empty_cache`)."""
        self._cache.release()

    def _sample(self, cond: torch.Tensor, h: int, w: int) -> torch.Tensor:
        self._table(h, w, cond.device)  # outside the capture
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            z = None
            if cond.numel() <= _GRAPH_WORK:  # larger calls: launches are cheap next to the GEMMs, no graph memory
                with self._lock:  # `_hw` is the argument of the graph forward
                    self._hw = (h, w)
                    z = self._cache.run((tuple(cond.shape), h, w), cond)
            return self._sample_k(cond, h, w) if z is None else z

    def module(self) -> torch.nn.Module:
        """Kernels export nothing."""
        raise NotImplementedError

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """No portable networks."""
        return ()


RUNNERS = {"portable": Portable, "kernels": Kernels}
