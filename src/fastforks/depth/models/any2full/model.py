"""Any2Full runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `Any2Full.forward` (`resize_mode="resize"`, `init_scailing`): the image is scaled so that its
shorter side is at least `short_side` (518) with both sides rounded up to a multiple of 14 (bicubic, align_corners),
the depth map (nearest) to the same size. Missing depth (0, negative, NaN, inf) enters as zero disparity; the
disparity is normalised by the mean / std of its non-zero values. The network's disparity is fitted to the prompt's
disparity by least squares (scale and shift over the pixels with depth), inverted (disparity clamped to >= 1e-3,
depth to [1e-6, 1e3]) and resampled to the input size (bilinear, align_corners). Upstream fits with `pinverse` and
1e-10 noise; here the closed-form fit (float32, centred sums).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import latin1_encode, numeric_scalar
from fastforks._core.half import pure_half
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.depth.models.depth_anything_v2.fused import FusedHead, FusedViT
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.shared.depth_anything_v2 import fold_layer_scale, sdpa_attention

from .net import PATCH, Any2FullNet, PromptBranch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

_FIT_MIN = 1e-5  # fixed: upstream fits on prompt disparities above this
_DISP_MIN = 1e-3  # fixed: upstream clamps the fitted disparity to 1 / max_depth (1e3)
_DEPTH = (1e-6, 1e3)  # fixed: upstream min_depth / max_depth
_EPS = 1e-8  # fixed: upstream's 1 / (disparity + eps)
EXTRA: dict[tuple[str, str], object] = {  # the released files also pickle the epoch, best RMSE and optimizer state
    ("numpy", "dtype"): np.dtype,
    ("_codecs", "encode"): latin1_encode,
    **{(m, "scalar"): numeric_scalar for m in ("numpy.core.multiarray", "numpy._core.multiarray")},
}


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: str  # "vits", "vitb" or "vitl"
    short_side: int  # upstream resize_lower_size: the shorter network side is at least this


def network_size(h: int, w: int, short_side: int) -> tuple[int, int]:
    """Upstream `resize_to_multiple`: scale so the shorter side reaches `short_side`, round both sides up to 14."""
    s = max(short_side / h, short_side / w)
    return int(((h * s + PATCH - 1) // PATCH) * PATCH), int(((w * s + PATCH - 1) // PATCH) * PATCH)


def _net(ctx: Context) -> Any2FullNet:
    net = Any2FullNet(ctx.params.arch)
    net.load_state_dict(net.checkpoint_keys(ctx.state_dict(key="state_dict", extra=EXTRA)))
    return net.eval()


def prepare(
    image: torch.Tensor, prompt: torch.Tensor, short_side: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Image `[B, 3, H, W]` in [0, 1], depth `[B, 1, h, w]` -> network-size image, disparity, normalised disparity."""
    size = network_size(*image.shape[-2:], short_side)
    x = image if tuple(image.shape[-2:]) == size else F.interpolate(image, size, mode="bicubic", align_corners=True)
    d = prompt.float()
    if tuple(d.shape[-2:]) != size:
        d = F.interpolate(d, size, mode="nearest")
    disp = torch.where(torch.isfinite(d) & (d > 0), 1 / (d + _EPS), 0)
    m = disp != 0
    n = m.sum((1, 2, 3), keepdim=True)
    mean = torch.where(n > 0, disp.sum((1, 2, 3), keepdim=True) / n.clamp_min(1), 0)
    var = torch.where(m, (disp - mean) ** 2, 0).sum((1, 2, 3), keepdim=True) / (n - 1).clamp_min(1)
    std = torch.where((n > 1) & (var > 0), var.sqrt(), 1)  # upstream's fallbacks: one value / zero std -> 1
    return x, disp, (disp - mean) / std


def finish(pred: torch.Tensor, disp: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Network disparity `[B, 1, h, w]` + prompt disparity -> metric depth `[B, H, W]` (float32) at the input size."""
    pred = pred.float()
    sel = disp > _FIT_MIN
    n = sel.sum((1, 2, 3), keepdim=True).clamp_min(1)
    pm = torch.where(sel, pred, 0).sum((1, 2, 3), keepdim=True) / n
    tm = torch.where(sel, disp, 0).sum((1, 2, 3), keepdim=True) / n
    pc = torch.where(sel, pred - pm, 0)
    num = (pc * (disp - tm)).sum((1, 2, 3), keepdim=True)
    den = (pc * pc).sum((1, 2, 3), keepdim=True)
    a = torch.where(den > 0, num / den.clamp_min(1e-30), 0)
    fit = a * pred + (tm - a * pm)  # no prompt pixel: n -> 1, tm = 0, fit 0 (upstream's pinverse of an empty system)
    depth = (1 / (fit.clamp_min(_DISP_MIN) + _EPS)).clamp(*_DEPTH)
    if tuple(depth.shape[-2:]) != tuple(size):
        depth = F.interpolate(depth, tuple(size), mode="bilinear", align_corners=True)
    return depth[:, 0]


class Portable(Runner):
    """Plain PyTorch with fused SDPA attention: float32, or on CUDA fp16 weights without autocast.

    In fp16 the depth embedding (pooling, hole filling, its convs) and the head's 1-channel output conv stay float32.
    """

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = _net(ctx).to(ctx.device)
        sdpa_attention(net.pretrained)
        fold_layer_scale(net.pretrained)
        if ctx.dtype == "float16":
            for m in (net.pretrained, net.branches, net.prompt_norm, net.fusion, net.scale):
                m.half()
            pure_half(net.depth_head, fp32=[net.depth_head.scratch.output_conv2[2]])
        self.net = net

    def _run(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x, disp, dn = prepare(image, prompt, self.ctx.params.short_side)
        return finish(self.net(scale(x, self.ctx.io), dn), disp, image.shape[-2:])

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its sparse depth maps `[B, 1, h, w]` (0 / NaN: missing)."""
        return {"depth": self._run(image, prompt_depth)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `([B, 3, H, W] in [0, 1], depth [B, 1, h, w])` -> depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image, prompt: self._run(image, prompt))


def _linears(seq: torch.nn.Module) -> tuple[torch.nn.Linear, torch.nn.Linear]:
    """The two linears of a `Linear, GELU, Linear` sequential."""
    return cast("torch.nn.Linear", seq.get_submodule("0")), cast("torch.nn.Linear", seq.get_submodule("2"))


class Kernels(Runner):
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs (fp16 accumulation), flash attention, the NHWC fp16 DPT head.

    The ViT and head are Depth Anything V2's (`depth_anything_v2.fused`); the prompt branches and the feature fusion
    MLPs are fused-epilogue GEMMs, their attention SDPA (the first one masked). The depth embedding is the portable
    float32 module; the input resize is PyTorch (so the kernels' patch resize is the identity).
    """

    batch = None
    Params = Params
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        dev = ctx.device
        k = load_dpt_fused(dev)  # before the weights: a kernel build / load failure costs no weight load
        net = _net(ctx)
        self.k = k
        self.layers = net.layers
        self.vit = FusedViT(net.pretrained, dev, k, "hh")
        self.head = FusedHead(net.depth_head, dev, k)
        self.embed = net.embed.to(dev)
        self.branches = cast("list[PromptBranch]", list(net.branches))
        self.gamma2 = [br.ls2.gamma.detach().float().to(dev) for br in self.branches]
        for m in (net.branches, net.prompt_norm, net.fusion, net.scale):
            m.half().to(dev)
        self.prompt_norm = net.prompt_norm
        self.fusion = [_linears(s) for s in net.fusion]
        self.scale = [_linears(s) for s in net.scale]

    def _mlp(self, lin: tuple[torch.nn.Linear, torch.nn.Linear], x: torch.Tensor) -> torch.Tensor:
        """`Linear, GELU, Linear` as two GEMMs (GELU in the first one's epilogue)."""
        a, b = lin
        return self.k.gemm(self.k.gemm(x, a.weight, a.bias, act=self.vit.gelu), b.weight, b.bias)

    def _branch(self, j: int, x1: torch.Tensor, p: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """`PromptBranch.forward` with fused-epilogue GEMMs; updates `p` in place."""
        br, k = self.branches[j], self.k
        b, n, c = p.shape
        h = br.heads
        qk = k.gemm(x1, br.qk.weight, br.qk.bias).view(b, n, 2, h, c // h).permute(2, 0, 3, 1, 4)
        v = k.gemm(br.norm(p), br.bias.weight, br.bias.bias).view(b, n, h, c // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(qk[0], qk[1], v, attn_mask=mask).transpose(1, 2).reshape(b, n, c)
        k.gemm(o, br.proj.weight, br.proj.bias, res=p, out=p)
        hid = k.gemm(br.norm2(p), br.mlp.fc1.weight, br.mlp.fc1.bias, act=self.vit.gelu)
        return k.gemm(hid, br.mlp.fc2.weight, br.mlp.fc2.bias, scale=self.gamma2[j], res=p, out=p)

    @torch.inference_mode()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its sparse depth maps `[B, 1, h, w]` (0 / NaN: missing)."""
        vit, layers = self.vit, self.layers
        x, disp, dn = prepare(image, prompt_depth, self.ctx.params.short_side)
        size = (x.shape[-2], x.shape[-1])
        with torch.autocast("cuda", enabled=False):
            t = vit.embed_tokens(x.float(), size)  # identity resize: exact
            feats, pres = [], []
            for i in range(max(layers) + 1):
                if i in layers[1:]:
                    pres.append(t.clone())  # `_block` updates its input in place
                t = vit._block(i, t)  # noqa: SLF001 - the shared fused block
                if i in layers:
                    feats.append(vit.vit.norm(t[:, 1:]))
            p, has = self.embed(dn, pres[0].float())
            p = p.half()
            mask: torch.Tensor | None = has[:, None, None]
            for j, pre in enumerate(pres):
                p = self._branch(j, vit.blocks[layers[j + 1]].norm1(pre), p, mask)
                q = self.prompt_norm(p[:, 1:])
                f = self._mlp(self.fusion[j], torch.cat([feats[j + 1], q], -1))
                s = self._mlp(self.scale[j], q)
                feats[j + 1] = f.addcmul_(s, feats[j + 1]).add_(feats[j + 1])
                mask = None
            pred = self.head(feats, size[0] // PATCH, size[1] // PATCH)[:, None]
        return {"depth": finish(pred, disp, image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
