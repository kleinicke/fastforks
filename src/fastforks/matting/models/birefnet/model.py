"""BiRefNet runners: weights -> network and the portable / kernels forward passes.

Inference as the HF model card: the image resized to `size x size` (bilinear, antialiased, as torchvision `Resize`),
ImageNet normalisation, the network's last prediction through a sigmoid, resized back to the image size (bicubic,
clamped to [0, 1]; upstream resizes the 8-bit mask with PIL's default bicubic filter).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16 weights and activations (the authors' `half()`), SDPA
window attention in the Swin encoder (BiRefNet's `SDPA_enabled`). `kernels`: fp16 channels-last; each deformable
ASPP branch (offset and modulator convs as one conv, then the deformable conv with its BatchNorm and ReLU) runs the
fused `deform_conv` kernel (fp32 sampling and accumulation, no column buffer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.patch import override
from fastforks.kernels import deform_conv as deform_kernels
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared.cnn_kernels import CL, Patched, bn_affine

from .net.birefnet import BiRefNet
from .net.modules.aspp import _ASPPModuleDeformable

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        backbone: Swin size, `"swin_v1_l"` (BiRefNet) or `"swin_v1_t"` (BiRefNet_lite).
        size: Network input side (square).
    """

    backbone: str
    size: int


def _cpu_net(ctx: Context) -> BiRefNet:
    net = BiRefNet(ctx.params.backbone)
    net.load_state_dict(ctx.state_dict())
    return net.eval()


class _Matting(Runner):
    """Resize, normalise, network, sigmoid, resize back."""

    Params = Params
    compiled_methods = ("_net",)
    net: BiRefNet
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[-1]

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False, antialias=True)
        y = self._net(scale(x, self.ctx.io).to(self.dtype)).float().sigmoid()
        y = F.interpolate(y, image.shape[-2:], mode="bicubic", align_corners=False)
        return {"alpha": y[:, 0].clamp(0, 1)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["alpha"])


class Portable(_Matting):
    """Upstream's modules in plain PyTorch, float32 or float16 (SDPA window attention)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _cpu_net(ctx).to(ctx.device, self.dtype)


def _prepare_aspp(m: Patched) -> None:
    """One conv for offsets + modulator logits, packed weights, the BatchNorm as epilogue (call while fp32)."""
    dc = m.atrous_conv
    c = dc.regular_conv.weight.shape[1]
    if reason := deform_kernels.fit_reason(c, 1):
        msg = f"BiRefNet kernels: deformable conv {tuple(dc.regular_conv.weight.shape)}: {reason}"
        raise BackendUnavailableError(msg)
    oc, mc = dc.offset_conv, dc.modulator_conv
    om = torch.nn.Conv2d(c, oc.out_channels + mc.out_channels, oc.kernel_size, oc.stride, oc.padding)
    om.weight.data = torch.cat((oc.weight.data, mc.weight.data))
    om.bias = torch.nn.Parameter(torch.cat((oc.bias.data, mc.bias.data)))
    m._om = om  # noqa: SLF001 - fastforks attributes on upstream modules (moved to the device in `Kernels`)
    m._w = deform_kernels.pack_weight(dc.regular_conv.weight)  # noqa: SLF001
    m._s, m._t = bn_affine(m.bn, dc.regular_conv)  # noqa: SLF001 - fp32 epilogue
    override(m, _aspp_forward)


def _aspp_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    """`relu(bn(deform_conv(x)))` with offsets and modulator from one conv, as one kernel."""
    x = x.contiguous(memory_format=CL)
    om = self._om(x).contiguous(memory_format=CL)
    w = self.atrous_conv.regular_conv
    k = w.kernel_size
    return deform_kernels.deform_conv(
        x, om, self._w, w.out_channels, k, w.stride, (k[0] // 2, k[1] // 2), (1, 1), 1, 2.0, self._s, self._t,
        deform_kernels.ACT_RELU,
    )  # fmt: skip


class Kernels(_Matting):
    """fp16 channels-last network; the deformable ASPP branches run the fused `deform_conv` kernel."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert the deformable branches."""
        super().__init__(ctx)
        deform_kernels.load(ctx.device)
        net = _cpu_net(ctx)
        for m in net.modules():
            if isinstance(m, _ASPPModuleDeformable):
                _prepare_aspp(m)
        self.dtype = torch.float16
        self.net = net.to(ctx.device, torch.float16).to(memory_format=CL)
        for m in self.net.modules():
            if isinstance(m, _ASPPModuleDeformable):
                m._w, m._s, m._t = (v.to(ctx.device) for v in (m._w, m._s, m._t))  # noqa: SLF001


RUNNERS = {"portable": Portable, "kernels": Kernels}
