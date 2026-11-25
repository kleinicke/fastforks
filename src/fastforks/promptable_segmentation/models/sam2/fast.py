"""SAM 2 input handling and the portable image-encoder forward (`sam2/fast.py`, "friendly" path).

- input: upstream `SAM2Transforms` (ToTensor, `Resize((S, S))` = antialiased bilinear on the float image, ImageNet
  normalisation); the same torch ops on every device, so the CPU is exact
- encoder: upstream `Hiera.forward` on the upstream blocks (channels-last stage outputs), the position embedding
  (bicubic resize of the background embedding + the tiled window embedding) computed once, since the input size is fixed
- neck (image path of `FpnNeck` + `SAM2Base.forward_image`): only the three levels the mask decoder reads; the top
  level's lateral 1 x 1 conv, nearest 2x upsampling and the sum into level 2 (+ `no_mem_embed`, upstream
  `directly_add_no_mem_embed`); for levels 0 / 1 the lateral 1 x 1 conv and the mask decoder's `conv_s0` / `conv_s1`
  (also 1 x 1) are folded into one linear map each (`fold`). The position encodings only feed video memory attention
  and are not computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.mlx_cast import float32_call, to_float32, to_run_dtype
from fastforks._core.tracing import exporting_for_mlx
from fastforks.promptable_segmentation._runner import scale

from .net.backbone_utils import window_partition, window_unpartition
from .net.hieradet import do_pool

if TYPE_CHECKING:
    from fastforks._core.cards import Io

Lin = tuple[torch.Tensor, torch.Tensor]


def prepare(image: torch.Tensor, io: Io, size: int) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the normalised float32 `[1, 3, size, size]` encoder input."""
    x = F.interpolate(image, (size, size), mode="bilinear", align_corners=False, antialias=True)
    return scale(x, io)


def pos_embed(trunk: Any, tokens: int) -> torch.Tensor:  # noqa: ANN401 - upstream Hiera
    """Upstream `Hiera._get_pos_embed` for a `tokens x tokens` patch grid: `[1, tokens, tokens, C]`."""
    with torch.no_grad():
        return trunk._get_pos_embed((tokens, tokens))  # noqa: SLF001 - upstream helper, called once


def _lin(conv: torch.nn.Conv2d) -> Lin:
    if conv.bias is None:  # ValueError, not BackendUnavailableError: both backends build this neck
        msg = "SAM 2 neck: expected 1 x 1 convs with a bias"
        raise ValueError(msg)
    return conv.weight.detach().flatten(1), conv.bias.detach()


def fold(lateral: torch.nn.Conv2d, proj: torch.nn.Conv2d) -> Lin:
    """Two 1 x 1 convs `proj(lateral(x))` as one linear map `(W, b)` (computed in float64)."""
    (w1, b1), (w2, b2) = _lin(lateral), _lin(proj)
    w = w2.double() @ w1.double()
    return w.to(w1.dtype), (w2.double() @ b1.double() + b2.double()).to(b1.dtype)


@dataclass
class Neck:
    """The neck maps the mask decoder reads (float32 weights; `to` converts)."""

    top: Lin  # level 3 lateral (C * 8 -> 256)
    mid: Lin  # level 2 lateral (C * 4 -> 256)
    s1: Lin  # conv_s1 . level 1 lateral (C * 2 -> 64)
    s0: Lin  # conv_s0 . level 0 lateral (C -> 32)
    no_mem: torch.Tensor  # [256]

    @classmethod
    def build(cls, image_encoder: Any, decoder: Any, no_mem_embed: torch.Tensor) -> Neck:  # noqa: ANN401 - upstream modules
        """From upstream `ImageEncoder` (FpnNeck: `convs[3 - level]`), `MaskDecoder` and `no_mem_embed`."""
        neck = image_encoder.neck
        if list(neck.fpn_top_down_levels) != [2, 3] or neck.fpn_interp_model != "nearest" or image_encoder.scalp != 1:
            # ValueError, not BackendUnavailableError: portable builds this neck too, so an auto fallback fails the same
            msg = "SAM 2 neck: expected top-down levels [2, 3], nearest interpolation and scalp 1"
            raise ValueError(msg)
        c = neck.convs
        return cls(
            _lin(c[0].conv),
            _lin(c[1].conv),
            fold(c[2].conv, decoder.conv_s1),
            fold(c[3].conv, decoder.conv_s0),
            no_mem_embed.detach().flatten(),
        )

    def to(self, device: torch.device, dtype: torch.dtype) -> Neck:
        """A copy on `device`, weights in `dtype` (`no_mem` stays float32)."""

        def cv(t: Lin) -> Lin:
            return t[0].to(device, dtype).contiguous(), t[1].to(device, dtype)

        return Neck(cv(self.top), cv(self.mid), cv(self.s1), cv(self.s0), self.no_mem.to(device))


def up2(x: torch.Tensor) -> torch.Tensor:
    """Nearest 2x upsampling of a channels-last `[B, H, W, C]` map."""
    b, h, w, c = x.shape
    return x[:, :, None, :, None].expand(b, h, 2, w, 2, c).reshape(b, 2 * h, 2 * w, c)


def neck(n: Neck, feats: list[torch.Tensor], linear: Any = F.linear) -> dict[str, torch.Tensor]:  # noqa: ANN401 - F.linear-like
    """Channels-last Hiera stage outputs -> float32 NCHW `image_embed` `[1, 256, S/16, S/16]`, `feat_s1`, `feat_s0`."""
    f0, f1, f2, f3 = feats
    top = to_float32(linear(f3, *n.top))
    mid = to_float32(linear(f2, *n.mid)) + up2(top) + n.no_mem
    return {
        "image_embed": mid.permute(0, 3, 1, 2).contiguous(),
        "feat_s1": to_float32(linear(f1, *n.s1)).permute(0, 3, 1, 2).contiguous(),
        "feat_s0": to_float32(linear(f0, *n.s0)).permute(0, 3, 1, 2).contiguous(),
    }


def _block_f32_stream(blk: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - upstream MultiScaleBlock
    """`MultiScaleBlock.forward` with a float32 residual stream in a float16 MLX replay (autocast's layout).

    The norms read the float32 stream; attention and MLP run in the replay's dtype; the residual adds are float32.
    A stage transition's projection *is* the stream there (its pooled output is the shortcut), so it stays float32
    with float32 weights: rounded to float16 it put base-plus over its mask limit (mean 0.0028 -> 0.0012).
    """
    shortcut = x
    h = blk.norm1(x)
    if blk.dim != blk.dim_out:
        shortcut = do_pool(float32_call(blk.proj, h), blk.pool)
    h = to_run_dtype(h, torch.float32)
    size = h.shape[1:3]
    pad_hw = size
    if blk.window_size > 0:
        h, pad_hw = window_partition(h, blk.window_size)
    h = blk.attn(h)
    window = blk.window_size
    if blk.q_stride:
        window = blk.window_size // blk.q_stride[0]
        size = shortcut.shape[1:3]
        pad_hw = (size[0] + (window - size[0] % window) % window, size[1] + (window - size[1] % window) % window)
    if blk.window_size > 0:
        h = window_unpartition(h, window, pad_hw, size)
    x = shortcut + to_float32(h)
    return x + to_float32(blk.mlp(to_run_dtype(blk.norm2(x), torch.float32)))


def trunk(t: Any, x: torch.Tensor, pos: torch.Tensor) -> list[torch.Tensor]:  # noqa: ANN401 - upstream Hiera
    """Upstream `Hiera.forward` with a precomputed position embedding: the channels-last output of every stage."""
    x = t.patch_embed(x.to(pos.dtype)) + pos
    f32_stream = exporting_for_mlx()  # the float32 twin of an MLX replay: no-op casts in a float32 one
    if f32_stream:
        x = to_float32(x)
    out = []
    for i, blk in enumerate(t.blocks):
        x = _block_f32_stream(blk, x) if f32_stream else blk(x)
        if i in t.stage_ends:
            out.append(to_run_dtype(x, torch.float32) if f32_stream else x)
    return out
