"""LingBot-Depth runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `MDMModel.infer` (`resolution_level` -> number of ViT tokens): the image is resized (bilinear,
antialiased) to a 14-pixel patch grid of about `1200 + 2400 * resolution_level / 9` tokens with the input's aspect
ratio, the depth map (nearest) to the same size. Depth values that are not finite or at most 1 cm are missing; the
others enter as log depth. The ViT sees the class token, every image patch and one depth token per patch that holds at
least one depth value (upstream's per-sample token selection; a patch whose only values are exactly 1 m counts as
empty, as upstream's validity test on the log depth). The last block's image tokens, projected and plus the class
token, feed MoGe's neck and the depth / mask heads; both maps are upsampled to the input size (bilinear). `depth`:
metres (`lingbot-depth`: the head's value, `-dc`: its exp), `mask`: sigmoid > 0.5 (upstream sets the depth outside
the mask to inf; here it is kept and the mask returned).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import half_weights
from fastforks._core.patch import override
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.depth._runner import Forward, Runner
from fastforks.depth.models.depth_anything_v2.fused import FusedViT
from fastforks.depth.models.moge.model import _FusedStack, _rep_conv, grid, neck_inputs, uv_tables, view_plane_uv
from fastforks.depth.models.moge.net import PATCH
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.shared.depth_anything_v2 import fold_layer_scale

from .net import LingBotDepthNet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

_MIN_DEPTH = 0.01  # fixed: upstream treats depth <= 1 cm as missing
_TABLES = 2  # fixed: cache policy, position / uv tables per input size (video always hits)


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    exp_output: bool  # the head predicts log depth (`-dc`), else metres
    resolution_level: int  # 0-9: 1200-3600 ViT tokens


def _net(ctx: Context) -> LingBotDepthNet:
    net = LingBotDepthNet()
    net.load_state_dict(ctx.state_dict(key="model"))
    return net.eval()


def depth_input(prompt: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth `[B, 1, h, w]` (metres, missing: 0 / NaN / inf) -> log depth and the patches with a depth token.

    Returns the log depth at the patch grid `[B, 1, 14 rows, 14 cols]` (0 where missing) and the patches that get a
    depth token `[B, rows * cols]` (bool).
    """
    d = F.interpolate(prompt.float(), (rows * PATCH, cols * PATCH), mode="nearest")
    ok = (d > _MIN_DEPTH) & torch.isfinite(d)
    logd = torch.where(ok, d, 1).log()
    pix = ok & (logd != 0)  # upstream marks raw 0 (here log 1 m) missing in the token test
    tok = pix.view(-1, rows, PATCH, cols, PATCH).any(4).any(2).flatten(1)
    return logd, tok


def finish(depth: torch.Tensor, mask: torch.Tensor, size: Sequence[int], *, exp: bool) -> dict[str, torch.Tensor]:
    """Head outputs (float32 `[B, 1, h, w]`: depth, mask logits) -> depth and mask at the input size."""
    m = F.interpolate(torch.cat([depth, mask], 1), tuple(size), mode="bilinear", align_corners=False)
    d = m[:, 0].exp() if exp else m[:, 0]
    return {"depth": d, "mask": m[:, 1] > 0}  # sigmoid > 0.5


# ------------------------------------------------------------------------------------------------ portable
def _block(blk: Any, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:  # noqa: ANN401 - DINOv2 block
    """A DINOv2 block (LayerScale folded) with SDPA attention and an optional key mask `[B, 1, 1, N]`."""
    a = blk.attn
    b, n, c = x.shape
    h = a.num_heads
    qkv = a.qkv(blk.norm1(x)).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
    x = x + blk.ls1(a.proj(o.transpose(1, 2).reshape(b, n, c)))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


class Portable(Runner):
    """The upstream modules with SDPA attention: float32, or fp16 autocast (the depth output conv in float32).

    The ViT's residual stream stays float32 in both (as upstream's autocast): an fp16 stream rounds the metric depth
    the tokens carry (NYUv2 AbsRel 0.0123 -> 0.0129).

    Depth tokens: eager calls keep only the patches with depth (padded to the batch's longest with a key mask), as
    upstream; `module()` (export) keeps every depth token and masks the empty ones (the same result, static shapes).
    """

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.device)
        fold_layer_scale(self.net)
        self.half = ctx.dtype == "float16"
        if self.half:
            net = self.net
            half_weights(net.encoder, channels_last=False)
            for m in (net.neck, net.depth_head, net.mask_head):
                half_weights(m)
            net.depth_head.output_blocks.float()
            for m in net.modules():
                if isinstance(m, nn.Conv2d) and m.padding_mode == "replicate" and m.padding == (1, 1):
                    override(m, _rep_conv)

    def _encode(self, image: torch.Tensor, prompt: torch.Tensor, rows: int, cols: int, *, gather: bool) -> torch.Tensor:
        """Encoder features plus the class token `[B, C, rows, cols]`."""
        enc = self.net.encoder
        vit = enc.backbone
        size = (rows * PATCH, cols * PATCH)
        x = F.interpolate(image, size, mode="bilinear", align_corners=False, antialias=True)
        x = (x - enc.get_buffer("image_mean")) / enc.get_buffer("image_std")
        logd, tok = depth_input(prompt, rows, cols)
        ti = vit.patch_embed(x)
        b, n, c = ti.shape
        pe = vit.interpolate_pos_encoding(ti.new_empty((1, n + 1, c)), *size)
        ti = ti + (pe[:, 1:] + 1)
        td = vit.depth_patch_embed.proj(logd).flatten(2).transpose(1, 2) + (pe[:, 1:] + 2)
        if gather:  # only the patches with depth, in order (attention ignores the order; the positions are added)
            cnt = tok.sum(1)
            counts = cnt.tolist()
            keep = max(counts)
            idx = torch.argsort((~tok).to(torch.uint8), dim=1, stable=True)[:, :keep]
            td = torch.gather(td, 1, idx[..., None].expand(-1, -1, c))
            valid = torch.arange(keep, device=x.device) < cnt[:, None]
            mask = None if min(counts) == keep else valid
        else:
            mask = tok
        cls = (vit.cls_token + pe[:, :1]).expand(b, -1, -1).float()
        t = torch.cat([cls, ti.float(), td.float()], 1)  # float32 residual stream (upstream's autocast keeps it)
        if mask is not None:
            mask = torch.cat([mask.new_ones((b, n + 1)), mask], 1)[:, None, None]
        for blk in vit.blocks:
            t = _block(blk, t, mask)
        t = vit.norm(t[:, : n + 1])
        f = enc.output_projections[0](t[:, 1:].transpose(1, 2).unflatten(2, (rows, cols)))
        return f + t[:, 0, :, None, None]

    def _run(self, image: torch.Tensor, prompt: torch.Tensor, *, gather: bool = True) -> dict[str, torch.Tensor]:
        net, p = self.net, self.ctx.params
        b, _, h, w = image.shape
        rows, cols = grid(h, w, p.resolution_level)
        a = w / h
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=self.half):
            feat = self._encode(image, prompt, rows, cols, gather=gather)
            uv = [
                view_plane_uv(cols << i, rows << i, a, image.device).permute(2, 0, 1)[None].expand(b, -1, -1, -1)
                for i in range(5)
            ]
            levels = net.neck([torch.cat([feat, uv[0].to(feat.dtype)], 1), *uv[1:]])
            depth = net.depth_head(levels, fp32_out=True)[-1].float()
            mask = net.mask_head(levels)[-1].float()
        return finish(depth, mask, (h, w), exp=p.exp_output)

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth and mask `[B, H, W]` of a batch and its depth maps `[B, 1, h, w]` (0 / NaN: missing)."""
        return self._run(image, prompt_depth)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `([B, 3, H, W] in [0, 1], depth [B, 1, h, w])` -> depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image, prompt: self._run(image, prompt, gather=False)["depth"])


# ------------------------------------------------------------------------------------------------ kernels
class Kernels(Runner):
    """`dpt_fused` kernels: fused-epilogue CUTLASS GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 neck.

    The ViT runs per sample on its own token count (class + image + the depth tokens it has), the patch embeddings and
    the output projection (class token in its bias) are GEMMs; the input resize (bilinear, antialiased) is PyTorch,
    the view-plane inputs of the neck per-size bias tables, the depth / mask output convs float32. The ViT's residual
    stream and LayerNorms are float32 (`FusedViT.block32`, see `Portable`).
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
        enc = net.encoder
        vit = enc.backbone
        dp = vit.depth_patch_embed.proj
        c, kin = dp.out_channels, PATCH * PATCH
        wd = torch.zeros(c, -(-kin // 8) * 8, dtype=torch.float16)
        wd[:, :kin] = dp.weight.detach().reshape(c, kin)
        self.w_depth, self.b_depth = wd.to(dev), dp.get_parameter("bias").detach().half().to(dev)
        del vit.depth_patch_embed
        self.vit = FusedViT(vit, dev, k, res32=True)
        proj = enc.output_projections[0]
        self.proj = proj.get_parameter("weight").detach()[:, :, 0, 0].half().contiguous().to(dev)
        self.proj_b = proj.get_parameter("bias").detach().float().to(dev)
        self.neck_wf, self.neck_uv = neck_inputs(net.neck, dev)
        self.neck = _FusedStack(net.neck, k, dev)
        self.heads = [_FusedStack(net.depth_head, k, dev), _FusedStack(net.mask_head, k, dev)]
        self._tables: dict[tuple[int, int, float, int], tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = {}

    def _size_tables(
        self, rows: int, cols: int, a: float, b: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Image and depth token positions (`pos + 1`, `pos + 2`, fp16 `[rows * cols, C]`) and the neck's uv tables."""
        key = (rows, cols, a, b)
        if key not in self._tables:
            if len(self._tables) >= _TABLES:
                del self._tables[next(iter(self._tables))]
            pe = self.vit.interpolate_pos(rows, cols)[0, 1:]
            uv = uv_tables(self.neck_uv, (rows, cols), a, b, self.ctx.device)
            self._tables[key] = ((pe + 1).half().contiguous(), (pe + 2).half().contiguous(), uv)
        return self._tables[key]

    @torch.inference_mode()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth and mask `[B, H, W]` of a batch and its depth maps `[B, 1, h, w]` (0 / NaN: missing)."""
        k, vit, p = self.k, self.vit, self.ctx.params
        b, _, h, w = image.shape
        rows, cols = grid(h, w, p.resolution_level)
        n = rows * cols
        size = (rows * PATCH, cols * PATCH)
        pos1, pos2, uv = self._size_tables(rows, cols, w / h, b)
        x = F.interpolate(image, size, mode="bilinear", align_corners=False, antialias=True)
        logd, tok = depth_input(prompt_depth, rows, cols)
        dcols = logd.half().view(b, rows, PATCH, cols, PATCH).permute(0, 1, 3, 2, 4).reshape(b, n, PATCH * PATCH)
        dcols = F.pad(dcols, (0, self.w_depth.shape[1] - PATCH * PATCH))
        with torch.autocast("cuda", enabled=False):
            icols = k.patches(x, size, PATCH, vit.kp, IMAGENET_MEAN, IMAGENET_STD)  # identity resize: exact
            cls = vit.vit.cls_token[0, 0] + vit.pos_embed(rows, cols)[0, 0]
            f0 = torch.empty((b, n, self.proj.shape[0]), device=image.device, dtype=torch.float16)
            for i in range(b):
                idx = tok[i].nonzero()[:, 0]
                t16 = torch.empty((1 + n + idx.numel(), vit.vit.embed_dim), device=image.device, dtype=torch.float16)
                k.gemm(icols[i], vit.w_patch, vit.b_patch, res=pos1, out=t16[1 : n + 1], acc32=True)
                if idx.numel():
                    k.gemm(dcols[i, idx], self.w_depth, self.b_depth, res=pos2[idx], out=t16[n + 1 :], acc32=True)
                t = t16.float()[None]  # the residual stream in float32, as upstream's autocast
                t[0, 0] = cls
                for j in range(len(vit.blocks)):
                    t = vit.block32(j, t)
                y = vit.norm32(t[0, : n + 1])
                k.gemm(y[1:], self.proj, (self.proj_b + y[0].float()).half(), out=f0[i])
            n0 = k.gemm(f0, self.neck_wf, res=uv[0]).view(b, rows, cols, -1)
            levels = self.neck.levels(n0, uv)
            depth, mask = (hd.head(levels).permute(0, 3, 1, 2) for hd in self.heads)
            return finish(depth, mask, (h, w), exp=p.exp_output)


RUNNERS = {"portable": Portable, "kernels": Kernels}
