"""`fast(backbone)`: inference forwards of the Swin v1 layers and blocks with fewer memory passes.

Upstream's block pads the normalised tokens, rolls them (two passes, one per dim), partitions them into windows (a
copy) and reverses all three after the attention (three more copies and a crop). Here one gather takes the tokens
straight into the shifted, padded window order (the padding rows are zeroed in place, as upstream's zero padding)
and one gather takes the attention output back to image order. The shifted-window mask, the gather indices and the
relative position bias are built once per size. The mask is added to the bias once per call for one image's windows,
not for the whole batch: a batch of images runs one attention call per image, each against that `[nW, nH, N, N]`
mask (upstream materialises it `[B * nW, nH, N, N]`, 100 MB per stage-1 block of BEN2's five 512 x 512 crops).

The token stream also stays contiguous: upstream's `flatten(2).transpose(1, 2)` patch tokens made every stage-1
residual add and LayerNorm read a transposed tensor.

The values are upstream's up to the order of the floating-point sums (the same SDPA call on the same windows).
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch
import torch.nn.functional as F

from fastforks._core.patch import override
from fastforks.shared import sdpa

__all__ = ["fast"]

_INFERENCE_ONLY = "swin_v1.fast is inference only (run it under torch.inference_mode or torch.no_grad)"


def fast(backbone: Any, *, sdpa_auto: str | None = None) -> None:  # noqa: ANN401 - the vendored SwinTransformer
    """Swap the layer and block forwards of `backbone` (in place) for the gather-based ones.

    Args:
        backbone: A `SwinTransformer`.
        sdpa_auto: The window attention's own `fastforks.shared.sdpa` priority list under `FASTFORKS_SDPA=auto`
            (`Runner.sdpa_auto`, for a model whose other attention keeps the runner's order); None: the runner's.
    """
    for layer in backbone.layers:
        override(layer, _layer_forward)
        for blk in layer.blocks:
            override(blk, _block_forward)
            blk._fast_sdpa = sdpa_auto  # noqa: SLF001
            blk.attn._fast_bias = None  # noqa: SLF001 - its cached [nH, N, N] bias


class _Geometry:
    """Window order of an `h` x `w` map: gather indices both ways, padding rows and the shifted-window mask."""

    def __init__(self, h: int, w: int, window: int, shift: int, device: torch.device, dtype: torch.dtype) -> None:  # noqa: PLR0913, PLR0917
        # integer bookkeeping on the CPU, then one copy each: on the GPU its tiny kernels and the syncs of
        # `torch.where(cond)` cost ~15 ms per geometry at each new image size (birefnet: 10 geometries per size)
        cpu = torch.device("cpu")
        hp, wp = -(-h // window) * window, -(-w // window) * window
        y = torch.arange(hp, device=cpu)[:, None].expand(hp, wp)
        x = torch.arange(wp, device=cpu)[None, :].expand(hp, wp)
        if shift:  # the token at rolled position (y, x) came from (y + shift, x + shift)
            y, x = (y + shift) % hp, (x + shift) % wp
        part = lambda t: t.reshape(hp // window, window, wp // window, window).permute(0, 2, 1, 3).reshape(-1)  # noqa: E731
        y, x = part(y), part(x)
        real = (y < h) & (x < w)
        to_windows = torch.where(real, y * w + x, 0)  # [nW * N] -> source token (0 for padding)
        to_image = torch.empty(h * w, dtype=torch.long, device=cpu)
        to_image[to_windows[real]] = torch.where(real)[0]
        self.to_windows = to_windows.to(device)
        self.padding = torch.where(~real)[0].to(device)  # window rows that upstream pads with zeros
        self.to_image = to_image.to(device)
        self.windows = (hp // window) * (wp // window)
        self.mask = None
        if shift:  # upstream's -inf mask of BasicLayer.forward, by region label
            ry = (torch.arange(hp, device=cpu) >= hp - window).long() + (torch.arange(hp, device=cpu) >= hp - shift)
            rx = (torch.arange(wp, device=cpu) >= wp - window).long() + (torch.arange(wp, device=cpu) >= wp - shift)
            label = part(3 * ry[:, None].expand(hp, wp) + rx[None, :].expand(hp, wp)).view(self.windows, -1).to(device)
            self.mask = torch.zeros(label.shape + label.shape[-1:], device=device, dtype=dtype).masked_fill_(
                label[:, None, :] != label[:, :, None], float("-inf")
            )[:, None]  # [nW, 1, N, N]


@functools.lru_cache(maxsize=32)
def _geometry(h: int, w: int, window: int, shift: int, device: torch.device, dtype: torch.dtype) -> _Geometry:  # noqa: PLR0913, PLR0917
    return _Geometry(h, w, window, shift, device, dtype)


def _layer_forward(self: Any, x: torch.Tensor, H: int, W: int) -> tuple:  # noqa: ANN401, N803 - a BasicLayer; upstream's names
    x = x.contiguous()
    for blk in self.blocks:
        blk.H, blk.W = H, W
        x = blk(x, None)
    if self.downsample is not None:
        return x, H, W, self.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2
    return x, H, W, x, H, W


def _bias(attn: Any, dtype: torch.dtype) -> torch.Tensor:  # noqa: ANN401 - a WindowAttention
    b = attn._fast_bias  # noqa: SLF001
    if b is None or b.dtype != dtype:
        n = attn.relative_position_index.shape[0]
        b = attn.relative_position_bias_table[attn.relative_position_index.view(-1)].view(n, n, -1)
        b = b.permute(2, 0, 1).to(dtype).contiguous()
        attn._fast_bias = b  # noqa: SLF001
    return b


def _block_forward(self: Any, x: torch.Tensor, mask_matrix: object = None) -> torch.Tensor:  # noqa: ANN401 - a SwinTransformerBlock
    del mask_matrix  # the geometry's
    if torch.is_grad_enabled():
        raise RuntimeError(_INFERENCE_ONLY)
    b, _, c = x.shape
    g = _geometry(self.H, self.W, self.window_size, self.shift_size, x.device, x.dtype)
    attn, heads, n = self.attn, self.attn.num_heads, self.window_size**2
    t = self.norm1(x).index_select(1, g.to_windows)  # [B, nW * N, C]
    if g.padding.numel():
        t.index_fill_(1, g.padding, 0)
    q, k, v = attn.qkv(t).view(b, g.windows, n, 3, heads, c // heads).permute(3, 0, 1, 4, 2, 5).unbind(0)
    del t
    bias = _bias(attn, q.dtype)[None]  # [1, nH, N, N]
    with contextlib.nullcontext() if self._fast_sdpa is None else sdpa.policy(x.device, self._fast_sdpa):
        o = _attention(q, k, v, bias, g.mask)
    del q, k, v
    o = attn.proj(o.reshape(b, g.windows * n, c))  # o: [.., N, nH, Dh], the fused kernels' output layout
    x = x + o.index_select(1, g.to_image)
    del o
    return x + self.mlp(self.norm2(x))


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """`[B, nW, N, nH, Dh]` (a view) of the window attention of `[B, nW, nH, N, Dh]` q, k, v."""
    b = q.shape[0]
    if mask is None or b == 1:  # one call: the bias (+ one image's mask) broadcast over the windows
        add = bias if mask is None else bias + mask
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=add.expand(q.shape[0], -1, -1, -1))
        return o.transpose(1, 2).unflatten(0, (b, -1))
    add = bias + mask  # one call per image against one image's [nW, nH, N, N] mask
    o = q.new_empty((*q.shape[:2], q.shape[3], q.shape[2], q.shape[4]))
    for i in range(b):
        o[i] = F.scaled_dot_product_attention(q[i], k[i], v[i], attn_mask=add).transpose(1, 2)
    return o
