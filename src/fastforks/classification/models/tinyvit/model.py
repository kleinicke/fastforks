"""TinyViT: the timm network with the variant's weights.

`portable` runs it through `fastforks._optimize` (`shared.libnet`: BatchNorm folded, fp16 channels-last). `kernels`
reads the weights out of the same timm network (BatchNorm folded with timm's own `ConvNorm.fuse`) and runs the
MBConv stages and the window-attention blocks as kernel calls (below).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate, graph_sized
from fastforks._core.runner import Runner
from fastforks.shared import libnet
from fastforks.shared.cuda_graphs import SmallShapeGraphs

if TYPE_CHECKING:
    from fastforks._core.runner import Context

# CUDA graphs while the call is launch bound or close to it: up to this many input pixels (batch x H x W)
_GRAPH_PIXELS = 1 << 20  # fixed: batch <= 20 at 224, <= 7 at 384, <= 4 at 512 (as ConvNeXt; not tuned per family)
_GRAPHS = 4  # fixed: graph cache policy (the crop is fixed, so a key is a batch size)
_GELU = 5  # fixed: fvm activation code
_IN_CHANNELS = 8  # fixed: `input_nhwc8` pads the image's channels to 8

_Conv = tuple[torch.Tensor, torch.Tensor, list[int], list[int]]  # weight, bias, stride, padding


def _h(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    return t.detach().half().contiguous().to(dev)


def _w4(w: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """Linear `[N, K]` or conv weight as a channels-last fp16 conv weight `[N, K, R, S]`."""
    w = w.detach().half()
    return (w[:, :, None, None] if w.dim() == 2 else w).contiguous(memory_format=torch.channels_last).to(dev)  # noqa: PLR2004


def _conv(cn: Any, dev: torch.device) -> _Conv:  # noqa: ANN401 - a timm ConvNorm (conv + BatchNorm)
    """BatchNorm folded (timm's `fuse`, fp32); dense weight channels-last, depthwise weight as `[R, S, C]`."""
    c = cn.fuse()
    w = c.weight
    if c.groups > 1:
        wk = _h(w[:, 0].permute(1, 2, 0), dev)
    else:
        wk = _w4(F.pad(w, (0, 0, 0, 0, 0, -w.shape[1] % _IN_CHANNELS)), dev)  # only the image conv is padded
    return wk, _h(c.bias, dev), list(c.stride), list(c.padding)


def _block(blk: Any, dev: torch.device) -> tuple[Any, ...]:  # noqa: ANN401 - a timm TinyVitBlock
    a, m = blk.attn, blk.mlp
    bias = a.attention_biases[:, a.attention_bias_idxs]  # [heads, N, N]
    return (
        (_h(a.norm.weight, dev), _h(a.norm.bias, dev)),
        (_w4(a.qkv.weight, dev), _h(a.qkv.bias, dev)),
        _h(bias, dev),
        (_w4(a.proj.weight, dev), _h(a.proj.bias, dev)),
        _conv(blk.local_conv, dev),
        (_h(m.norm.weight, dev), _h(m.norm.bias, dev)),
        (_w4(m.fc1.weight, dev), _h(m.fc1.bias, dev)),
        (_w4(m.fc2.weight, dev), _h(m.fc2.bias, dev)),
    )


class Kernels(Runner):
    """fvm_conv kernels on fp16 channels-last activations, the head in float32.

    Patch embedding, MBConv and patch-merging convolutions as fused CUTLASS convolutions (bias + GELU, and the MBConv
    residual, in the epilogue) and fvm depthwise convolutions; in the attention blocks the qkv / projection / MLP
    GEMMs as fused GEMMs (bias + GELU or residual in the epilogue; cuBLAS for GEMMs of at most 600 rows,
    `graph_sized`), attention as `scaled_dot_product_attention` with the gathered bias table, the local depthwise
    convolution on the token map without a layout copy. Calls of up to `_GRAPH_PIXELS` replay a CUDA graph per batch
    size, captured on its second call.
    """

    batch = None
    Params = libnet.Params

    def __init__(self, ctx: Context) -> None:
        """Build timm's network from the checkpoint and keep its weights as fp16 kernel operands."""
        super().__init__(ctx)
        from fastforks._optimize import conv_ops  # noqa: PLC0415 - fvm_conv only (not `ops`: all fvm_* libraries)

        self.ops = conv_ops
        dev = ctx.device
        net = cast("Any", libnet.build(ctx))
        with torch.no_grad():
            self.embed = (_conv(net.patch_embed.conv1, dev), _conv(net.patch_embed.conv2, dev))
            self.mbconvs = [tuple(_conv(c, dev) for c in (b.conv1, b.conv2, b.conv3)) for b in net.stages[0].blocks]
            self.stages = []
            for st in net.stages[1:]:
                d = st.downsample
                down = tuple(_conv(c, dev) for c in (d.conv1, d.conv2, d.conv3))
                b0 = st.blocks[0]
                self.eps = float(b0.attn.norm.eps)
                self.stages.append((down, b0.window_size, b0.attn.num_heads, [_block(b, dev) for b in st.blocks]))
            head = net.head
            self.head_eps = float(head.norm.eps)
            self.head = tuple(
                t.detach().float().to(dev) for t in (head.norm.weight, head.norm.bias, head.fc.weight, head.fc.bias)
            )
        self._graphs = SmallShapeGraphs(self._forward, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def _cv(self, x: torch.Tensor, conv: _Conv, act: int, res: torch.Tensor | None = None) -> torch.Tensor:
        w, b, stride, padding = conv
        if w.dim() == 3:  # noqa: PLR2004 - depthwise `[R, S, C]`
            return self.ops.dwconv(x, w, b, stride, padding, act)
        return self.ops.conv_fused(x, w, b, res, padding, stride, act)

    def _attend(self, x: torch.Tensor, heads: int, blk: tuple[Any, ...], res: torch.Tensor | None) -> torch.Tensor:
        """Attention of timm's `Attention` module on `[B', N, C]` (+ `res` in the projection's epilogue)."""
        norm, qkv, bias, proj = blk[:4]
        b, n, c = x.shape
        t = self.ops.lin(F.layer_norm(x, (c,), norm[0], norm[1], self.eps), qkv[0], qkv[1])
        q, k, v = t.view(b, n, heads, -1).permute(0, 2, 1, 3).chunk(3, dim=3)  # key dim = value dim (attn_ratio 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.ops.lin(o.transpose(1, 2).reshape(b, n, c), proj[0], proj[1], res=res)

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised `[B, 3, H, W]` float32 -> logits `[B, classes]` float32."""
        ops = self.ops
        x = self._cv(self._cv(ops.input_nhwc8(image), self.embed[0], _GELU), self.embed[1], 0)
        for c1, c2, c3 in self.mbconvs:
            x = self._cv(self._cv(self._cv(x, c1, _GELU), c2, _GELU), c3, _GELU, res=x)
        for (d1, d2, d3), ws, heads, blocks in self.stages:
            x = self._cv(self._cv(self._cv(x, d1, _GELU), d2, _GELU), d3, 0)
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]: a view of the channels-last map
            b, h, w, c = x.shape
            if h % ws or w % ws:
                msg = f"TinyViT: a {h}x{w} token map is not a multiple of the window {ws} (the card's crop is fixed)"
                raise ValueError(msg)
            nh, nw = h // ws, w // ws
            for blk in blocks:
                if nh == nw == 1:
                    x = self._attend(x.reshape(b, h * w, c), heads, blk, x).view(b, h, w, c)
                else:
                    t = x.view(b, nh, ws, nw, ws, c).transpose(2, 3).reshape(b * nh * nw, ws * ws, c)
                    t = self._attend(t, heads, blk, None)
                    x = x + t.view(b, nh, nw, ws, ws, c).transpose(2, 3).reshape(b, h, w, c)
                x = self._cv(x.permute(0, 3, 1, 2), blk[4], 0).permute(0, 2, 3, 1)
                nm, fc1, fc2 = blk[5:]
                t = ops.lin(F.layer_norm(x, (c,), nm[0], nm[1], self.eps), fc1[0], fc1[1], act=_GELU)
                x = ops.lin(t, fc2[0], fc2[1], res=x)
            x = x.permute(0, 3, 1, 2)
        nw_, nb_, fw, fb = self.head
        pooled = x.permute(0, 2, 3, 1).float().mean((1, 2))
        return F.linear(F.layer_norm(pooled, (pooled.shape[-1],), nw_, nb_, self.head_eps), fw, fb)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, classes]` of a transformed batch (a CUDA graph replay for small repeated batches)."""
        # `graph_sized` for every call, eager or captured, so the two agree
        with torch.autocast("cuda", enabled=False), fp16_accumulate(), graph_sized():
            return {"logits": self._graphs(image)}


RUNNERS = {"portable": libnet.Portable, "kernels": Kernels}
