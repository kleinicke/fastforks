"""Metric3D v2 runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `hubconf.py` (`__main__`, `metric3d_vit_large`, config `vit.raft5.large`): keep-ratio bilinear
resize into 616 x 1064, ImageNet normalisation, padding with the mean (zero after normalisation), the network (ViT-L
DINOv2 with 4 register tokens, then the RAFT depth / normal decoder, 8 GRU iterations), crop and bilinear upsampling
to the input size. The network predicts depth for a canonical camera with a 1000 px focal length; given intrinsics
rescale it by `fx * s / 1000` (`s` the resize factor), without them the canonical camera is assumed (scale 1).
Depth is clamped to [0, 300] as upstream; normals (camera frame) are re-normalised after the upsampling.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks._core.half import fp16_accumulate, half_weights, pure_half
from fastforks._core.patch import override
from fastforks.depth._runner import Forward, NoParams, Runner, scale
from fastforks.kernels.fastbb import Flash64
from fastforks.shared.depth_anything_v2.fast_vit import FastViT

from .net.raft_depth_normal_dpt5 import RAFTDepthNormalDPT5, norm_normalize
from .net.vit_dino_reg import Attention, Mlp, vit_large_reg

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

INPUT = (616, 1064)  # fixed: config data_basic.vit_size (multiples of 28)
CANONICAL_FOCAL = 1000.0  # fixed: config data_basic.canonical_space.focal_length
MAX_DEPTH = 300.0  # fixed: hubconf clamp
BINS = 256  # fixed: decoder num_depth_regressor_anchor


def _config() -> types.SimpleNamespace:
    """The decoder part of the released config (`vit.raft5.large` on `dino_vit_large_reg.dpt_raft`)."""
    head = types.SimpleNamespace(
        in_channels=[1024] * 4,
        use_cls_token=True,
        feature_channels=[256, 512, 1024, 1024],
        decoder_channels=[128, 256, 512, 1024, 1024],
        up_scale=7,
        hidden_channels=[128] * 4,
        n_gru_layers=3,
        n_downsample=2,
        iters=8,
        slow_fast_gru=True,
        num_register_tokens=4,
    )
    return types.SimpleNamespace(
        model=types.SimpleNamespace(decode_head=head),
        data_basic=types.SimpleNamespace(depth_normalize=(0.1, 200)),
    )


class _Metric3D(nn.Module):
    """Upstream `DensePredModel` (`encoder` + `decoder`), the `depth_model` of the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = vit_large_reg(checkpoint=None, out_channels=[1024] * 4, drop_path_rate=0.0)
        self.decoder = RAFTDepthNormalDPT5(_config())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decoder(self.encoder(x))


def _net(ctx: Context) -> _Metric3D:
    with torch.device(ctx.device):  # random init on the target device (fast on CUDA), then the checkpoint
        net = _Metric3D()
    state = ctx.state_dict(key="model_state_dict", strip="depth_model.")
    load_strict(net, state, f"{ctx.variant.ref} checkpoint", not_in_checkpoint=("encoder.mask_token",))
    net.encoder.mask_token.data.zero_()  # not in the checkpoint; upstream: zeros, only used in training
    net.decoder.register_depth_expectation_anchor(BINS, 1)  # broadcasts over the batch
    return net.eval()


def _geometry(h: int, w: int) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Resize factor, resized size and padding (top, bottom, left, right) of an `h x w` input (upstream formulas)."""
    s = min(INPUT[0] / h, INPUT[1] / w)
    rh, rw = int(h * s), int(w * s)
    ph, pw = INPUT[0] - rh, INPUT[1] - rw
    return s, (rh, rw), (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)


def _prepare(ctx: Context, image: torch.Tensor) -> tuple[torch.Tensor, float, tuple[int, int, int, int]]:
    s, size, pad = _geometry(*image.shape[-2:])
    x = F.interpolate(scale(image, ctx.io), size=size, mode="bilinear", align_corners=False)
    return F.pad(x, (pad[2], pad[3], pad[0], pad[1]), value=0.0), s, pad


def _restore(
    image: torch.Tensor, intrinsics: torch.Tensor | None, s: float, pad: tuple[int, int, int, int], maps: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Network-size depth, confidence, normal (`[B, 1 + 1 + 3, 616, 1064]`) -> outputs at the input size."""
    m = maps.float()
    m = m[:, :, pad[0] : m.shape[2] - pad[1], pad[2] : m.shape[3] - pad[3]]
    m = F.interpolate(m, image.shape[-2:], mode="bilinear", align_corners=False)
    d = m[:, 0]
    if intrinsics is not None:
        d = d * (intrinsics[:, 0, 0].float() * s / CANONICAL_FOCAL)[:, None, None]
    n = m[:, 2:]
    n = n / n.norm(dim=1, keepdim=True)
    return {"depth": d.clamp(0.0, MAX_DEPTH), "normals": n.permute(0, 2, 3, 1), "confidence": m[:, 1]}


def _sdpa_forward(self: Any, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ANN401 - the vendored Attention
    """Upstream `Attention.forward` (eval, no bias) with fused scaled-dot-product attention."""
    if attn_bias is not None:
        msg = "attention bias is a training-only path"
        raise ValueError(msg)
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(b, n, c)
    return self.proj(x)


class Portable(Runner):
    """The vendored network with fused SDPA attention: float32, or on CUDA a fp16 ViT and an autocast decoder."""

    intrinsics = "optional"
    dtypes = ("float32", "float16")
    batch = None
    Params = NoParams

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        for m in self.net.encoder.modules():
            if isinstance(m, Attention):
                override(m, _sdpa_forward)
        self.half = ctx.dtype == "float16"
        if self.half:  # the ViT in plain fp16, the decoder under autocast with fp16 weights (identical to fp32 weights)
            pure_half(self.net.encoder, channels_last=False)
            half_weights(self.net.decoder)

    def _run(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        x, s, pad = _prepare(self.ctx, image)
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.half):
            o = self.net(x)
        maps = torch.cat([o["prediction"], o["confidence"], o["prediction_normal"][:, :3]], 1)
        return _restore(image, intrinsics, s, pad, maps)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Metric depth, normals and confidence of a batch."""
        return self._run(image, intrinsics)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> depth `[B, H, W]` (canonical camera)."""
        return Forward(self.net, lambda _net, image: self._run(image, None)["depth"])


_fp16_accumulate = fp16_accumulate


class _ViT(FastViT):
    """The shared fast ViT on Metric3D's vendored DINOv2-reg (its own `Mlp` / `Attention` classes, same maths)."""

    def __init__(self, vit: Any, attn: str) -> None:  # noqa: ANN401 - the vendored ViT
        """As `FastViT.__init__`, with the structure check against the vendored classes."""
        blocks = [b for chunk in vit.blocks for b in chunk if not isinstance(b, nn.Identity)]
        for b in blocks:
            mlp = b.mlp
            ok = isinstance(mlp, Mlp) and isinstance(mlp.act, nn.GELU) and mlp.act.approximate == "none"
            if not ok or b.tuning_mode is not None or b.attn.num_heads * 64 != vit.embed_dim:
                msg = "fast ViT: unexpected Metric3D block structure"
                raise ValueError(msg)
        self.blocks = blocks
        self.cache_pos = True
        self.pos_fp32 = vit.pos_embed.detach().float().clone()
        self.vit = vit.half().eval()
        self.attn, self.out_dtype = Flash64(attn), None
        self._pos = {}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Normed final tokens (cls, registers, patches), fp16: upstream `forward_features` (`multi_output` off)."""
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            t = self.tokens(x)
            for blk in self.blocks:
                t = self.block(blk, t)
            return self.vit.norm(t)


def _decode(dec: Any, tokens: torch.Tensor, b: int, gh: int, gw: int) -> torch.Tensor:  # noqa: ANN401 - the vendored decoder
    """Upstream `RAFTDepthNormalDPT5.forward` (eval): depth, confidence, normal `[B, 1 + 1 + 3, H, W]`.

    Without the NaN / Inf logging (each check is a device sync) and without upsampling the intermediate iterations
    (only the last one is returned).
    """
    r, c = dec.num_register_tokens, dec.in_channels[0]
    feats = [[tokens[:, 1 + r :].view(b, gh, gw, c), tokens[:, : 1 + r].reshape(b, 1, 1, c * (1 + r))]] * 4
    enc = dec.token2feature(feats)
    ref = dec.decoder_mono(enc)
    fm = ref[:, :-2]
    prob = dec.depth_regressor(fm).softmax(dim=1)
    d = (prob * dec.depth_expectation_anchor[0].view(1, -1, 1, 1)).sum(1, keepdim=True)
    depth = (dec.clamp(d) - dec.max_val) / dec.regress_scale
    normal = norm_normalize(torch.cat([dec.normal_predictor(fm), ref[:, -1:]], dim=1))
    flow = torch.cat((depth, ref[:, -2:-1], normal), dim=1).float()

    cnet = dec.context_feature_encoder(enc[::-1])
    net = [torch.tanh(x[0]) for x in cnet]
    inp = [torch.relu(x[1]) for x in cnet]
    zqr = dec.context_zqr_convs
    inp = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp, zqr, strict=False)]
    ub = dec.update_block
    mask = None
    for _ in range(dec.iters):
        net = ub(net, inp, iter32=True, iter16=False, iter08=False, update=False)
        net = ub(net, inp, iter32=True, iter16=True, iter08=False, update=False)
        net, mask, delta = ub(net, inp, None, flow, iter32=True, iter16=True)
        flow = flow + delta
    up = dec.upsample_flow(flow, mask)
    depth = dec.clamp(up[:, :1] * dec.regress_scale + dec.max_val)
    return torch.cat([depth, up[:, 1:2], norm_normalize(up[:, 2:])[:, :3]], 1)


class Kernels(Runner):
    """fastbb ViT (flash attention, fp16-accumulating GEMMs) and the decoder under fp16 autocast.

    The decoder skips upstream's NaN-logging syncs and the upsampling of intermediate iterations.
    """

    intrinsics = "optional"
    batch = None
    Params = NoParams

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert it."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.vit = _ViT(self.net.encoder, "hh")

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Metric depth, normals and confidence of a batch."""
        x, s, pad = _prepare(self.ctx, image)
        ps = self.net.encoder.patch_size
        tokens = self.vit.features(x)
        with torch.autocast("cuda", dtype=torch.float16):
            maps = _decode(self.net.decoder, tokens, x.shape[0], x.shape[2] // ps, x.shape[3] // ps)
        return _restore(image, intrinsics, s, pad, maps)


RUNNERS = {"portable": Portable, "kernels": Kernels}
