"""timm's ConvNeXt (V1) trunk as fvm_conv / fvm_seg kernel calls (fastforks; shared by the ConvNeXt families).

`Trunk(net, device)` reads the weights out of a timm `ConvNeXt` and runs stem, stages and downsampling on fp16
channels-last activations: stem and downsampling convolutions and the MLP GEMMs as fused CUTLASS convolutions (bias +
GELU, and LayerScale (folded) + residual in the epilogue), LayerNorms as fvm_seg `ln_gather_h` (fp32 statistics),
depthwise 7x7 in fp32 accumulation. Per-shape dispatch: GEMMs of at most 600 rows run cuBLAS under `graph_sized` (in
a replayed graph the extra launches are free), depthwise convolutions of at most `_DW_CUDNN_ROWS` rows run cuDNN.
The caller adds its head and the CUDA graphs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_GELU = 5  # fixed: fvm activation code
# depthwise 7x7 of a small map (batch x H x W rows) runs cuDNN: 224 at batch 1, stages 3 and 4 (196 and 49 rows),
# 1.86 -> 1.79 ms (Base), 3.20 -> 3.07 (Large); from 576 rows on fvm_conv's is faster (Base 384: 2.80 vs 3.02 ms)
_DW_CUDNN_ROWS = 256  # fixed: measured switch point (64 / 256 / 600 rows tried)
_LN_ROWS = 8  # fixed: rows (warps) per block of `ln_gather_h`, as measured for DINOv3 ConvNeXt (same kernel, shapes)
_LN_MAX_BLOCKS = 65535 * 8  # fixed: grid cap, the kernel loops over the rest
_IN_CHANNELS = 8  # fixed: `input_nhwc8` pads the image's channels to 8


def _h(t: torch.Tensor | None, dev: torch.device) -> torch.Tensor:
    assert t is not None  # noqa: S101 - every conv / norm / linear of timm's ConvNeXt has a bias
    return t.detach().half().to(dev)


def _w4(w: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """Linear `[N, K]` or conv weight as a channels-last fp16 conv weight `[N, K, R, S]`."""
    w = w.detach().half()
    return (w[:, :, None, None] if w.dim() == 2 else w).contiguous(memory_format=torch.channels_last).to(dev)  # noqa: PLR2004


def _block(blk: Any, dev: torch.device) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - a timm ConvNeXtBlock
    """Depthwise weight as `[7, 7, C]` (fvm) and `[C, 1, 7, 7]` (cuDNN) + bias, LayerNorm, fc1, fc2 * LayerScale."""
    dw, fc1, fc2 = blk.conv_dw, blk.mlp.fc1, blk.mlp.fc2
    w2, b2 = fc2.weight.detach().float(), fc2.bias.detach().float()
    if blk.gamma is not None:  # folded in fp32, then fp16
        g = blk.gamma.detach().float()
        w2, b2 = w2 * g[:, None], b2 * g
    return (
        _h(dw.weight[:, 0].permute(1, 2, 0).contiguous(), dev),
        _w4(dw.weight, dev),
        _h(dw.bias, dev),
        _h(blk.norm.weight, dev),
        _h(blk.norm.bias, dev),
        _w4(fc1.weight, dev),
        _h(fc1.bias, dev),
        _w4(w2, dev),
        _h(b2, dev),
    )


class Trunk:
    """The stages of a timm ConvNeXt: normalised `[B, 3, H, W]` float32 -> stage-4 map `[B, C, H/32, W/32]` fp16."""

    def __init__(self, net: Any, dev: torch.device) -> None:  # noqa: ANN401 - a timm ConvNeXt
        """Keep the weights of `net` as fp16 channels-last kernel operands on `dev`."""
        from fastforks._optimize import conv_ops  # noqa: PLC0415 - fvm_conv only (not `ops`: all fvm_* libraries)
        from fastforks.kernels import fvm_seg  # noqa: PLC0415

        self.ops = conv_ops
        self.seg = fvm_seg.load(dev)
        self._idx = torch.arange(0, dtype=torch.int32, device=dev)  # identity row index of `ln_gather_h`
        self._old: list[torch.Tensor] = []  # outgrown index buffers a captured graph may still read
        self.eps = float(net.stem[1].eps)
        self.stages = []
        with torch.no_grad():
            for i, st in enumerate(net.stages):
                norm, conv = (net.stem[1], net.stem[0]) if i == 0 else (st.downsample[0], st.downsample[1])
                w = conv.weight if i else F.pad(conv.weight, (0, 0, 0, 0, 0, _IN_CHANNELS - conv.in_channels))
                down = (_w4(w, dev), _h(conv.bias, dev), list(conv.stride))
                self.stages.append(
                    (down, (_h(norm.weight, dev), _h(norm.bias, dev)), [_block(b, dev) for b in st.blocks])
                )

    def _ln(self, x: torch.Tensor, wb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """LayerNorm over the last dim of a contiguous fp16 tensor (fp32 statistics)."""
        c = x.shape[-1]
        m = x.numel() // c
        if self._idx.numel() < m:  # grows on an eager first call of a larger shape, never while capturing
            self._old = [*self._old, self._idx]  # rebind, not append: atomic vs `release_graphs` from another thread
            self._idx = torch.arange(max(m, 2 * self._idx.numel()), dtype=torch.int32, device=x.device)
        y = torch.empty_like(x)
        grid = (min(-(-m // _LN_ROWS), _LN_MAX_BLOCKS), 1, 1)
        self.seg.launch("ln_gather_h", grid, (32 * _LN_ROWS, 1, 1), x, self._idx, wb[0], wb[1], y, m, c, self.eps)
        return y

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """The stage-4 map of a normalised batch (call under `fp16_accumulate()`, `graph_sized()`, no autocast)."""
        ops = self.ops
        x = ops.input_nhwc8(image)
        for i, ((wd, bd, sd), norm, blocks) in enumerate(self.stages):
            if i == 0:
                x = ops.conv_fused(x, wd, bd, None, [0, 0], sd, 0)
                x = self._ln(x.permute(0, 2, 3, 1), norm).permute(0, 3, 1, 2)
            else:
                x = self._ln(x.permute(0, 2, 3, 1), norm).permute(0, 3, 1, 2)
                x = ops.conv_fused(x, wd, bd, None, [0, 0], sd, 0)
            cudnn = x.shape[0] * x.shape[2] * x.shape[3] <= _DW_CUDNN_ROWS
            for dw, dw4, dwb, nw, nb, w1, b1, w2, b2 in blocks:
                if cudnn:
                    h = F.conv2d(x, dw4, dwb, padding=3, groups=x.shape[1])
                else:
                    h = ops.dwconv(x, dw, dwb, [1, 1], [3, 3], 0)
                h = ops.lin(self._ln(h.permute(0, 2, 3, 1), (nw, nb)), w1, b1, act=_GELU)
                x = ops.lin(h, w2, b2, res=x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def release(self) -> None:
        """Drop the outgrown index buffers (after the graphs that read them are released)."""
        self._old = []
