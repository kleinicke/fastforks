"""SegNeXt runner: the vendored network (`net.py`) with the mmsegmentation ADE20K checkpoints.

Test-time protocol of the configs (mmseg `test_pipeline`, `SegDataPreProcessor` with `size_divisor=32`,
`EncoderDecoder.whole_inference`, `postprocess_result`): resize keeping the aspect ratio to fit `2048 x 512`
(bilinear), ImageNet normalisation, zero padding on the right and bottom to multiples of 32, one forward pass, logits
(1/8 resolution) resized bilinearly to the padded size, the padding cut off, then resized to the image size.

`portable` runs the folded network (`net.fold`) as plain PyTorch, fp16 channels-last on CUDA with the depthwise strip
convs in NCHW (PyTorch's native kernel). `kernels` runs every depthwise conv as one direct NHWC kernel (`seg_conv`;
the MLP's 3x3 with its GELU in `fvm_conv`), the 1x1 convs as fp16-accumulate cuBLAS GEMMs with the bias fused, the
logits resize as one `seg_up` pass, and replays a CUDA graph per repeated small input shape.

One deviation from upstream: the head's matrix factorisation starts from fixed bases (`net.Hamburger`), so the same
image always gives the same logits; upstream draws random ones on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks._optimize import ops as fvm
from fastforks.kernels import seg_conv
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.segmentation.models.segnext import net as sn
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.seginfer import resize, target_size, upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

_DIVISOR = 32  # fixed: the configs' `test_cfg=dict(size_divisor=32)`
_CL = torch.channels_last
_GELU = 5  # fixed: fvm activation code of exact (erf) GELU, the MLP activation
# CUDA graphs (kernels): the eager forward is host-bound below the ADE20K test size (~15 launches per block)
_GRAPH_PIXELS = 512 * 512  # fixed: larger inputs are GPU-bound
_GRAPHS = 2  # fixed: graph cache policy


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: Network (`net.ARCHS`): `mscan-t`, `mscan-s`, `mscan-b`, `mscan-l`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), `["mmseg", 2048, 512]`.
    """

    arch: str
    resize: tuple[str, int, int]


def network(ctx: Context) -> sn.EncoderDecoder:
    """The variant's network with its checkpoint (CPU, float32, eval), BatchNorms and layer scales folded.

    Raises:
        CheckpointMismatchError: Any missing, unexpected or reshaped tensor.
    """
    model = sn.build(ctx.params.arch)
    load_strict(model, ctx.state_dict(0, key="state_dict", strip=""), f"{ctx.variant.ref} checkpoint")
    sn.fold(model)
    return model.eval().requires_grad_(requires_grad=False)


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 channels-last on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    fused = False  # kernels: logits resize and argmax by `seg_up`

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = network(ctx).to(ctx.device, ctx.torch_dtype)
        if ctx.device.type == "cuda":
            self.net = self.net.to(memory_format=torch.channels_last)
            for m in self.net.modules():  # strip-conv weights `[C, 1, 1, k]` / `[C, 1, k, 1]` back to NCHW strides:
                if isinstance(m, torch.nn.Conv2d) and m.groups > 1 and m.kernel_size[0] != m.kernel_size[1]:
                    m.weight.data = m.weight.data.clone(memory_format=torch.contiguous_format)  # (ambiguous for 1-sized
                    # dims; a channels-last tag would route the NCHW strip convs back to cuDNN's grouped conv)

    def _scores(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
        """Logits at 1/8 of the padded size (NCHW) and their remaining resize steps (padded size, image size).

        Padding on the right / bottom (sides not divisible by 32) is cut off after the first step here, which leaves
        one step; otherwise both steps are returned (`seg_up` runs them as one pass).
        """
        size = target_size(*image.shape[-2:], self.ctx.params.resize)
        x = scale(resize(image, size), self.ctx.io)
        x = F.pad(x, (0, -size[1] % _DIVISOR, 0, -size[0] % _DIVISOR))
        if x.is_cuda:
            x = x.contiguous(memory_format=_CL)
        s = self._forward(x.to(next(self.net.parameters()).dtype)).contiguous()  # NCHW: PyTorch's bilinear resize
        padded = (x.shape[-2], x.shape[-1])  # runs 2-3x faster than channels-last on 150 classes
        if padded == size:
            return s, (padded, (image.shape[-2], image.shape[-1]))
        return resize(s, padded)[..., : size[0], : size[1]], ((image.shape[-2], image.shape[-1]),)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return upsample(*self._scores(image), fused=self.fused)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 150, H, W]` of a batch."""
        return {"logits": self._run(image)}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no full-size logits)."""
        return upsample_argmax(*self._scores(image), fused=self.fused)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


# ------------------------------------------------------------------------------------------------ kernels forwards
_Mod = Any  # modules with the derived buffers `_dw` / `_bias32` (set by `_prepare_kernels`)


def _dw(conv: _Mod, x: torch.Tensor, *, act: int = 0) -> torch.Tensor:
    """`act(dwconv(x))`, bias included, as one `seg_conv` kernel (fp16 NHWC, fp32 accumulation)."""
    x = x.contiguous(memory_format=_CL)
    n, c, h, w = x.shape
    (kh, kw), (ph, pw) = conv.kernel_size, conv.padding
    y = torch.empty((n, c, h + 2 * ph - kh + 1, w + 2 * pw - kw + 1), device=x.device, dtype=x.dtype, memory_format=_CL)
    geo = {"padding": (ph, pw), "stride": (1, 1), "dilation": (1, 1), "act": act}
    if seg_conv.load(x.device).dwconv(x, conv._dw, y, None, conv._bias32, **geo):  # noqa: SLF001
        return y
    return conv(x)  # C % 8 != 0: none in the zoo


def _pw(conv: _Mod, x: torch.Tensor) -> torch.Tensor:
    """A 1 x 1 conv as a cuBLAS GEMM over the NHWC view: the bias fused, fp16 accumulation inside `fp16_accumulate`."""
    return F.linear(x.permute(0, 2, 3, 1), conv._w2, conv.bias).permute(0, 3, 1, 2)  # noqa: SLF001


def _attention(self: sn.MSCAAttention, x: torch.Tensor) -> torch.Tensor:
    a = _dw(self.conv0, x)
    a0 = _dw(self.conv0_2, _dw(self.conv0_1, a))
    a1 = _dw(self.conv1_2, _dw(self.conv1_1, a))
    a2 = _dw(self.conv2_2, _dw(self.conv2_1, a))
    return _pw(self.conv3, a.add_(a0).add_(a1).add_(a2)) * x


def _spatial(self: sn.MSCASpatialAttention, x: torch.Tensor) -> torch.Tensor:
    y = _pw(self.proj_2, self.spatial_gating_unit(F.gelu(_pw(self.proj_1, x))))
    return y + x if self.res_scale is None else torch.addcmul(y, x, self.res_scale)


def _mlp(self: sn.Mlp, x: torch.Tensor) -> torch.Tensor:
    dw: _Mod = self.dwconv
    return _pw(self.fc2, fvm.dwconv(_pw(self.fc1, x), dw._dw, dw.bias, [1, 1], [1, 1], _GELU))  # noqa: SLF001


def _prepare_kernels(net: sn.EncoderDecoder, device: torch.device) -> None:
    """Route the blocks' depthwise convs through `seg_conv` / `fvm_conv` and their 1 x 1 convs through cuBLAS.

    Raises:
        KernelUnavailableError: No `seg_conv` / `fvm_conv` binary for this GPU / driver.
    """
    seg_conv.load(device)
    for m in net.backbone.modules():
        if isinstance(m, torch.nn.Conv2d) and m.groups == m.in_channels and m.in_channels % 8 == 0:
            conv: _Mod = m
            conv._dw = m.weight.detach()[:, 0].permute(1, 2, 0).contiguous()  # noqa: SLF001 - [R, S, C]
            conv._bias32 = None if m.bias is None else m.bias.detach().float().contiguous()  # noqa: SLF001
        elif isinstance(m, torch.nn.Conv2d) and m.groups == 1 and m.kernel_size == (1, 1):
            conv = m
            conv._w2 = m.weight.detach().flatten(1).contiguous()  # noqa: SLF001 - [K, C]
        if isinstance(m, sn.MSCAAttention):
            override(m, _attention)
        elif isinstance(m, sn.MSCASpatialAttention):
            override(m, _spatial)
        elif isinstance(m, sn.Mlp):
            override(m, _mlp)


class Kernels(Portable):
    """Depthwise convs as direct NHWC kernels, 1 x 1 convs as fp16-accumulate GEMMs, fused upsampling, CUDA graphs."""

    fused = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107 - see Portable
        Runner.__init__(self, ctx)
        self.net = network(ctx).to(ctx.device, torch.float16).to(memory_format=_CL)
        _prepare_kernels(self.net, ctx.device)
        self._graphs = SmallShapeGraphs(self._eager, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self.net(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._graphs(x)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        self._graphs.release()

    module = Runner.module


RUNNERS = {"portable": Portable, "kernels": Kernels}
