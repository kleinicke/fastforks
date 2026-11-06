"""Sapiens segmentation runners: the ViT + `VitHead` network (`net.py`) with the Sapiens-Lite TorchScript weights.

Test-time protocol of the Sapiens-Lite demo (`lite/demo/vis_seg.py` + `adhoc_image_dataset.py`): resize to 1024 x 768
(bilinear, half-pixel centres, no antialiasing: OpenCV `INTER_LINEAR`; done on the float image, not on uint8), RGB
normalised with mean (123.5, 116.5, 103.5) and std (58.5, 57.0, 57.5) on the 0-255 scale, one forward pass, logits
(1/2 resolution) resized bilinearly to the image size. The archives' TorchScript code is never run: only their
tensors are read (`fastforks._core.checkpoint.torchscript_state_dict`).

Both runners run the network as functional ops on its weights (`fast_sapiens.py`): the decoder
channels-last, the InstanceNorm statistics in fp32 and the normalisation fused with the SiLU in place (the batch-8
decoder maps are 2.4 GB each in fp16). `portable` (friendly) is plain PyTorch: SDPA attention, cuDNN
convolutions, fp16 on CUDA. `kernels` (kernels) adds fp16-accumulate cuBLAS GEMMs, the `fastbb` flash attention
kernel (head_dim 64, fp32 `q k^T`) and fp16-accumulate cuDNN convolutions / transposed convolutions (`cudnn_hacc`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core import mps
from fastforks._core.errors import BackendUnavailableError
from fastforks._core.half import fp16_accumulate
from fastforks.kernels import cudnn_hacc, fastbb
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.segmentation.models.sapiens.net import IN_EPS, LN_EPS, PATCH, Arch, Layer, Sapiens
from fastforks.shared.seginfer import upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

FLASH_HEAD_DIM = 64  # fixed: head_dim of the fastbb flash kernel
# The archives' normalisation buffers (the runner uses the card's mean / std); any other key hits the strict load.
_NOT_THE_NETWORK = frozenset({"data_preprocessor.mean", "data_preprocessor.std"})


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` (the network shape; the input size is the position table's grid x 16)."""


def _prepare(image: torch.Tensor, ctx: Context) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> the normalised network input at the table's size (float32)."""
    gh, gw = ctx.params.grid
    x = F.interpolate(image, (gh * PATCH, gw * PATCH), mode="bilinear", align_corners=False)
    return scale(x, ctx.io)


def _load(ctx: Context) -> Sapiens:
    p: Params = ctx.params
    net = Sapiens(p)
    sd = ctx.state_dict(strip="")
    net.load_state_dict({k: v for k, v in sd.items() if k not in _NOT_THE_NETWORK})
    return net.eval().requires_grad_(requires_grad=False)


def _inorm_silu_(x: torch.Tensor) -> torch.Tensor:
    """InstanceNorm (no affine) + SiLU in place on a fresh map (memory format kept); statistics in float32."""
    v, m = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=False)
    a = (v.float() + IN_EPS).rsqrt()
    b = (-m.float() * a).to(x.dtype)
    return F.silu(torch.addcmul(b, x, a.to(x.dtype), out=x), inplace=True)


class _Functional(Runner):
    """The network as functional ops on its weights: decoder channels-last, InstanceNorm + SiLU in place."""

    batch = None
    Params = Params
    fused = False  # `seg_up` resizing (kernels)

    def __init__(self, ctx: Context, dtype: torch.dtype) -> None:
        """Weights on the device in `dtype` (convolutions channels-last)."""
        super().__init__(ctx)
        net = _load(ctx).to(ctx.device, dtype)
        cl = torch.channels_last
        head = net.decode_head
        self.patch_w = net.backbone.patch_embed.projection.weight.contiguous(memory_format=cl)
        self.deconv = [m.weight.contiguous(memory_format=cl) for m in head.deconvs]
        self.convs = [(m.weight.contiguous(memory_format=cl), m.bias) for m in head.convs]
        self.seg_w = head.conv_seg.weight.contiguous(memory_format=cl)
        self.net = net
        self.dtype = dtype

    def _conv(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, stride: int) -> torch.Tensor:
        return F.conv2d(x, w, b, stride)

    def _deconv(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if x.device.type == "mps":  # torch's op is 3x slower there
            return mps.conv_transpose(x, w, None, 2, 1)
        return F.conv_transpose2d(x, w, None, 2, 1)

    def _attn(self, layer: Layer, x: torch.Tensor) -> torch.Tensor:
        """SDPA on the packed qkv."""
        b, n, c = x.shape
        q, k, v = (
            F.linear(x, layer.attn.qkv.weight, layer.attn.qkv.bias)
            .view(b, n, 3, layer.attn.heads, c // layer.attn.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return F.linear(out, layer.attn.proj.weight, layer.attn.proj.bias)

    def _layer(self, layer: Layer, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1]
        x = x + self._attn(layer, F.layer_norm(x, (c,), layer.ln1.weight, layer.ln1.bias, LN_EPS))
        fc1, fc2 = layer.ffn.fc
        y = F.gelu(F.linear(F.layer_norm(x, (c,), layer.ln2.weight, layer.ln2.bias, LN_EPS), fc1.weight, fc1.bias))
        return x.add_(F.linear(y, fc2.weight, fc2.bias))

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised float32 input -> logits at 1/2 size (compute dtype)."""
        bb = self.net.backbone
        x = x.to(self.dtype).contiguous(memory_format=torch.channels_last)
        t = self._conv(x, self.patch_w, bb.patch_embed.projection.bias, PATCH)
        b, c, gh, gw = t.shape
        t = t.flatten(2).transpose(1, 2) + bb.pos_embed
        for layer in bb.blocks:
            t = self._layer(layer, t)
        f = F.layer_norm(t, (c,), bb.ln1.weight, bb.ln1.bias, LN_EPS).view(b, gh, gw, c).permute(0, 3, 1, 2)
        for w in self.deconv:
            f = _inorm_silu_(self._deconv(f, w))
        for w, bias in self.convs:
            f = _inorm_silu_(self._conv(f, w, bias, 1))
        return self._conv(f, self.seg_w, self.net.decode_head.conv_seg.bias, 1)

    def _scores(self, image: torch.Tensor) -> torch.Tensor:
        """Logits at 1/2 size for the resize to the image size, in the compute dtype (fp16 on CUDA).

        Kernels: `seg_up` reads the channels-last logits as they are (float32 math in one pass); portable: NCHW
        (1.6x faster bilinear resizing).
        """
        s = self._logits(_prepare(image, self.ctx))
        return s if self.fused else s.contiguous()

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return upsample(self._scores(image), (tuple(image.shape[-2:]),), fused=self.fused)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 28, H, W]` of a batch."""
        return {"logits": self._run(image)}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no full-size logits)."""
        return upsample_argmax(self._scores(image), (tuple(image.shape[-2:]),), fused=self.fused)


class Portable(_Functional):
    """Plain PyTorch (friendly): SDPA attention, float32, or fp16 on CUDA."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network on the device in the compute dtype."""
        super().__init__(ctx, ctx.torch_dtype)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 28, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image))


_fp16_accumulate = fp16_accumulate


class Kernels(_Functional):
    """fast path: fp16, fastbb flash attention, fp16-accumulate GEMMs and cuDNN convolutions, `seg_up` resizing."""

    fused = True

    def __init__(self, ctx: Context) -> None:
        """fp16 weights on the device (decoder channels-last), load the kernel libraries."""
        p: Params = ctx.params
        if p.embed_dim != p.num_heads * FLASH_HEAD_DIM:
            msg = f"{ctx.variant.ref}: the flash kernel needs head_dim {FLASH_HEAD_DIM}"
            raise BackendUnavailableError(msg)  # mode="auto" falls back to portable
        super().__init__(ctx, torch.float16)
        fastbb.load(ctx.device)
        self.flash = fastbb.Flash64("fh")
        self.cv = cudnn_hacc.load(ctx.device)

    def _conv(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, stride: int) -> torch.Tensor:
        return self.cv.conv(x, w, b, (0, 0), (stride, stride))

    def _deconv(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.cv.deconv(x, w, (1, 1), (2, 2))

    def _attn(self, layer: Layer, x: torch.Tensor) -> torch.Tensor:
        """Flash attention on the packed qkv `[B, N, 3, H, 64]` -> `[B, N, C]` (no transposes)."""
        b, n, c = x.shape
        h = c // FLASH_HEAD_DIM
        qkv = F.linear(x, layer.attn.qkv.weight, layer.attn.qkv.bias).view(b, n, 3, h, FLASH_HEAD_DIM)
        out = self.flash(*qkv.unbind(2), FLASH_HEAD_DIM**-0.5)
        return F.linear(out.view(b, n, c), layer.attn.proj.weight, layer.attn.proj.bias)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 28, H, W]` of a batch."""
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            return super().__call__(image)

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch (see `_Functional.labels`)."""
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            return super().labels(image)


RUNNERS = {"portable": Portable, "kernels": Kernels}
