"""mmsegmentation CNN classics runners: the vendored networks (`net.py`) with the authors' ADE20K checkpoints.

Test-time protocol of the ADE20K configs (mmseg `test_pipeline` + `EncoderDecoder.whole_inference` +
`postprocess_result`): resize keeping the aspect ratio to fit `2048 x 512` (bilinear), ImageNet normalisation, one
forward pass of the whole image, logits (1/8 or 1/4 resolution) resized bilinearly to the resized input's size and
then to the image size.

`portable` runs the modules with every BatchNorm folded into its conv (fp16 channels-last on CUDA; dilated depthwise
convs in NCHW, where cuDNN is 4-5x faster). `kernels` keeps every BatchNorm as an fp32 per-channel affine applied in the
conv epilogue, with the activation and the bottleneck residual in the same pass: every dense conv with channel counts
divisible by 8 (dilated, strided and residual ones included) is one `seg_conv` CUTLASS kernel with that epilogue, as in
our fork, each depthwise conv one direct `seg_conv` kernel, HRNet's fuse layers sum in the conv epilogues and
one upsample + add kernel per term; the RGB stem and the classifier run on `fastforks.shared.cnn_kernels` (cuDNN
fp16-accumulate graphs or CUTLASS convs plus the epilogue pass).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_weights

from fastforks._core import kernels
from fastforks._core.checkpoint import load_strict
from fastforks._core.patch import base_type, is_base, override
from fastforks.kernels import cudnn_hacc, seg_conv, seg_up
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.segmentation.models.mmseg import net as mm
from fastforks.shared import cnn_kernels as cm
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.seginfer import resize, target_size, upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

_Mod = Any  # a module patched with our attributes (`_s`, `_t`) and forward
# CUDA graphs (kernels): up to a 512 x 512 network input the eager forward is host-bound (the Python conv epilogue
# hooks and plan launches take longer than the GPU runs them); a repeated input shape replays a graph
_GRAPH_PIXELS = 512 * 512  # fixed: the ADE20K test size of square images; larger inputs are GPU-bound
_GRAPHS = 2  # fixed: graph cache policy
# kernels: every dense conv with C, K % 8 == 0 (dilated, strided and residual ones included) as one `seg_conv` CUTLASS
# kernel with the BN affine, residual and ReLU in its epilogue, as in our fork; False: cnn_kernels only (A/B)
_SEG_CONV = True  # fixed: implementation switch
_SEG_ACT = {0: 0, cm.ACT["relu"]: 1}  # cnn_kernels activation code -> seg_conv's
# kernels: HRNet fuse layers with the running sum in the conv epilogues and one upsample + add (+ ReLU) kernel per
# upsampled term, as our fork; False: the upstream forward (A/B)
_HR_FUSE = True  # fixed: implementation switch
# portable: BatchNorm folded into the conv weights (float32, before the fp16 cast), as our fork; False: separate BNs
_FOLD_BN = True  # fixed: implementation switch
# portable fp16: the logits resize in fp16 NCHW (as our fork; half the bytes of float32), False: float32 (A/B)
_RESIZE_FP16 = True  # fixed: implementation switch


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: Network (`net.build`): `fcn-r101-d8`, `psp-r101-d8`, `aspp-r101-d8`, `sepaspp-r101-d8`, `fcn-hr48`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), `["mmseg", 2048, 512]`.
        hacc: Kernels tier: fp16-accumulate cuDNN convs. Off for HRNet: its ~300 small convs re-plan per input shape and
            run slower than plain cuDNN fp16.
    """

    arch: str
    resize: tuple[str, int, int]
    hacc: bool = True


def network(ctx: Context) -> mm.EncoderDecoder:
    """The variant's network with its checkpoint (CPU, float32, eval); the training-only `auxiliary_head.*` is dropped.

    Raises:
        CheckpointMismatchError: Any other missing, unexpected or reshaped tensor.
    """
    model = mm.build(ctx.params.arch)
    sd = ctx.state_dict(0, key="state_dict", strip="")
    sd = {k: v for k, v in sd.items() if not k.startswith("auxiliary_head.")}
    load_strict(model, sd, f"{ctx.variant.ref} checkpoint")
    return model.eval().requires_grad_(requires_grad=False)


# ------------------------------------------------------------------------------------------------ kernels forwards
def _conv(  # noqa: PLR0913, PLR0917 - conv module, input and the epilogue terms
    conv: _Mod,
    x: torch.Tensor,
    s: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    act: int = 0,
    r: torch.Tensor | None = None,
) -> torch.Tensor:
    """`act(conv(x) * s + t + r)`: one `seg_conv` kernel for a marked conv (`_seg`), else `cnn_kernels.conv_ep`."""
    if getattr(conv, "_segdw", None) is not None and r is None and act in _SEG_ACT and x.dtype == torch.float16:
        x = x.contiguous(memory_format=cm.CL)
        n, c, h, w = x.shape
        (kh, kw), (ph, pw), (sh, sw), (dh, dw) = conv.kernel_size, conv.padding, conv.stride, conv.dilation
        ho, wo = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        y = torch.empty((n, c, ho, wo), device=x.device, dtype=x.dtype, memory_format=cm.CL)
        if s is None and t is None:
            t = conv._bias32  # noqa: SLF001 - set by prepare_convs
        lib = seg_conv.load(x.device)
        geo = {"padding": (ph, pw), "stride": (sh, sw), "dilation": (dh, dw), "act": _SEG_ACT[act]}
        if lib.dwconv(x, conv._segdw, y, s, t, **geo):  # noqa: SLF001
            return y
    if getattr(conv, "_seg", False) and act in _SEG_ACT and x.dtype == torch.float16 and x.dim() == 4:  # noqa: PLR2004
        x = x.contiguous(memory_format=cm.CL)
        n, _, h, w = x.shape
        k = conv.weight.shape[0]
        (kh, kw), (ph, pw), (sh, sw), (dh, dw) = conv.kernel_size, conv.padding, conv.stride, conv.dilation
        ho, wo = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        if r is not None:
            r = r.contiguous(memory_format=cm.CL)
        if r is None or (r.shape == (n, k, ho, wo) and r.dtype == torch.float16):
            y = torch.empty((n, k, ho, wo), device=x.device, dtype=x.dtype, memory_format=cm.CL)
            if s is None and t is None:
                t = conv._bias32  # noqa: SLF001 - set by prepare_convs
            pads, strides, dils = (ph, pw), (sh, sw), (dh, dw)
            lib = seg_conv.load(x.device)
            if lib.conv(x, conv.weight, y, s, t, r, padding=pads, stride=strides, dilation=dils, act=_SEG_ACT[act]):
                return y if k == conv.out_channels else y[:, : conv.out_channels]
    return cm.conv_ep(conv, x, s, t, act, r=r)


def _plain_conv(self: _Mod, x: torch.Tensor) -> torch.Tensor:
    return _conv(self, x)


def _fused_conv_bn(self: _Mod, x: torch.Tensor) -> torch.Tensor:
    return _conv(self.conv, x, self.s, self.t, self.act)


def _conv_module(self: _Mod, x: torch.Tensor) -> torch.Tensor:
    return _conv(self.conv, x, self.conv._s, self.conv._t, cm.ACT["relu"] if self.act else 0)  # noqa: SLF001


def _basic(self: _Mod, x: torch.Tensor) -> torch.Tensor:
    idt = x if self.downsample is None else self.downsample(x)
    y = _conv(self.conv1, x, self.conv1._s, self.conv1._t, cm.ACT["relu"])  # noqa: SLF001
    return _conv(self.conv2, y, self.conv2._s, self.conv2._t, cm.ACT["relu"], r=idt)  # noqa: SLF001


def _bottleneck(self: _Mod, x: torch.Tensor) -> torch.Tensor:
    idt = x if self.downsample is None else self.downsample(x)
    y = _conv(self.conv1, x, self.conv1._s, self.conv1._t, cm.ACT["relu"])  # noqa: SLF001
    y = _conv(self.conv2, y, self.conv2._s, self.conv2._t, cm.ACT["relu"])  # noqa: SLF001
    return _conv(self.conv3, y, self.conv3._s, self.conv3._t, cm.ACT["relu"], r=idt)  # noqa: SLF001


def _up_add(y: torch.Tensor, z: torch.Tensor, mid: list[int], *, relu: bool, inplace: bool) -> torch.Tensor:
    """`relu?(y + resize(resize(z, mid), y size))`, an HRNet fuse term: one `seg_up` kernel for fp16 NHWC."""
    c = y.shape[1]
    h, w = z.shape[-2:]
    if (
        y.dtype == z.dtype == torch.float16
        and z.shape[:2] == y.shape[:2]
        and c % 2 == 0
        and y.is_contiguous(memory_format=cm.CL)
        and z.is_contiguous(memory_format=cm.CL)
        and h <= mid[0]
        and w <= mid[1]
    ):
        out = y if inplace else torch.empty_like(y, memory_format=cm.CL)
        # NHWC views of the channels-last tensors
        seg_up.up_add(
            z.permute(0, 2, 3, 1), y.permute(0, 2, 3, 1), out.permute(0, 2, 3, 1), (mid[0], mid[1]), relu=relu
        )
        return out
    kernels.note_fallback("seg_up", f"up_add {z.dtype} {tuple(z.shape)} -> {tuple(y.shape)}")
    t = mm.resize(mm.resize(z, mid), y.shape[2:])
    y = y.add_(t) if inplace else y + t
    return y.relu_() if relu else y


def _hrmodule(self: _Mod, x: list[torch.Tensor]) -> list[torch.Tensor]:
    """`HRModule.forward` with fused fuse terms, as our fork.

    Output `i` starts from `x[i]`; each `j < i` chain ends in a conv whose epilogue adds the running sum, each `j > i`
    1x1 conv is followed by one upsample + add kernel; the final ReLU goes into the last term. Same terms as upstream,
    summed in another order (fp16 rounding differs).
    """
    x = [br(xi) for br, xi in zip(self.branches, x, strict=True)]
    if self.fuse_layers is None:
        return x
    nb, out = len(x), []
    for i, row in enumerate(self.fuse_layers):
        y, fresh = x[i].contiguous(memory_format=cm.CL), False
        for j in range(i):
            h, chain = x[j], list(row[j])
            for k, sub in enumerate(chain):
                f = sub[0]  # FusedConvBN (fuse_sequentials); its BN / ReLU slots are Identity
                if k < len(chain) - 1:
                    h = _conv(f.conv, h, f.s, f.t, f.act)
                else:
                    h = _conv(f.conv, h, f.s, f.t, cm.ACT["relu"] if j == i - 1 and i == nb - 1 else 0, r=y)
            y, fresh = h, True
        for j in range(i + 1, nb):
            f, up = row[j][0], row[j][2]
            z = _conv(f.conv, x[j], f.s, f.t, 0)
            mid = [int(t * float(up.scale)) for t in z.shape[-2:]]
            y, fresh = _up_add(y, z, mid, relu=j == nb - 1, inplace=fresh), True
        out.append(y)
    return out


def _prepare_kernels(model: nn.Module, device: torch.device, *, hacc: bool = True) -> None:
    """Patch `model` (fp32, eval, on `device`) for the kernels path.

    Raises:
        KernelUnavailableError: See `cnn_kernels.load`.
    """
    cm.load(device)
    cm.fuse_sequentials(model)  # stems, shortcuts, HRNet transitions and fuse layers
    for m in model.modules():
        pairs, fwd = [], None
        if isinstance(m, mm.ConvModule):
            pairs, fwd = [(m.conv, m.bn)], _conv_module
        elif isinstance(m, mm.BasicBlock):
            pairs, fwd = [(m.conv1, m.bn1), (m.conv2, m.bn2)], _basic
        elif isinstance(m, mm.Bottleneck):
            pairs, fwd = [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)], _bottleneck
        for conv, bn in pairs:
            conv._s, conv._t = cm.bn_affine(bn, conv)  # noqa: SLF001
        if fwd is not None:
            override(m, fwd)
    cm.patch_norms(model)  # the rest (HRNet's stem) as standalone affine kernels
    cm.prepare_convs(model)
    _mark_hacc(model, hacc=hacc)
    if _SEG_CONV:
        _mark_seg(model, device)
    if _HR_FUSE:
        _mark_hr(model)


def _mark_hacc(model: nn.Module, *, hacc: bool) -> None:
    """cnn_kernels marks dilation-1 convs only; cudnn_hacc runs dilated ones too (convs `seg_conv` does not take)."""
    for m in model.modules():
        if is_base(m, nn.Conv2d):
            conv: _Mod = m
            dilated = (
                m.groups == 1 and m.dilation != (1, 1) and m.padding_mode == "zeros" and m.weight.shape[0] % 8 == 0
            )
            conv._hacc = hacc and (dilated or getattr(conv, "_hacc", False))  # noqa: SLF001


def _mark_hr(model: nn.Module) -> None:
    """Run every multi-branch `HRModule` with fusable fuse layers through `_hrmodule`."""
    for m in model.modules():
        if isinstance(m, mm.HRModule) and m.fuse_layers is not None and _hr_fusable(m):
            override(m, _hrmodule)


def _hr_fusable(m: mm.HRModule) -> bool:
    """The fuse layers are `fuse_sequentials`' conv + BN runs (+ upsample), as `_hrmodule` reads them."""
    assert m.fuse_layers is not None  # noqa: S101 - checked by the caller
    for i, row in enumerate(m.fuse_layers):
        for j, f in enumerate(row):  # ty: ignore[invalid-argument-type] - rows are ModuleLists
            if j > i and not (isinstance(f[0], cm.FusedConvBN) and isinstance(f[2], mm._Upsample)):  # noqa: SLF001
                return False
            if j < i and not all(isinstance(sub[0], cm.FusedConvBN) for sub in f):
                return False
    return True


def _mark_seg(model: nn.Module, device: torch.device) -> None:
    """Route the FusedConvBN runs, every dense conv with `C, K % 8 == 0` and the depthwise ones through `seg_conv`.

    Raises:
        KernelUnavailableError: No `seg_conv` binary for this GPU / driver.
    """
    seg_conv.load(device)
    for m in model.modules():
        if isinstance(m, cm.FusedConvBN):
            override(m, _fused_conv_bn)
        elif (
            is_base(m, nn.Conv2d)
            and m.groups == 1
            and m.padding_mode == "zeros"
            and isinstance(m.padding, tuple)
            and m.weight.shape[0] % 8 == 0
            and m.weight.shape[1] % 8 == 0
        ):
            conv: _Mod = m
            conv._seg = True  # noqa: SLF001
            override(conv, _plain_conv)
        elif (
            is_base(m, nn.Conv2d)
            and m.groups == m.in_channels == m.out_channels
            and m.in_channels % 8 == 0
            and m.padding_mode == "zeros"
            and isinstance(m.padding, tuple)
        ):  # depthwise (DeepLabV3+ separable ASPP): the direct kernel; cuDNN's channels-last fp16 ones are 4-5x slower
            conv: _Mod = m
            conv._segdw = m.weight.detach()[:, 0].permute(1, 2, 0).contiguous().half()  # noqa: SLF001 - [R, S, C]
            override(conv, _plain_conv)


# ------------------------------------------------------------------------------------------------ portable forwards
def _bn_pairs(model: nn.Module) -> list[tuple[nn.Module, str, str]]:
    """`(owner, conv attribute, BatchNorm attribute)` of every conv directly followed by its BatchNorm."""
    named = {
        mm.ConvModule: [("conv", "bn")],
        mm.BasicBlock: [("conv1", "bn1"), ("conv2", "bn2")],
        mm.Bottleneck: [("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")],
        mm.HRNet: [("conv1", "bn1"), ("conv2", "bn2")],
    }
    pairs = []
    for m in model.modules():
        pairs += [(m, c, b) for c, b in named.get(base_type(m), [])]
        if isinstance(m, nn.Sequential):
            keys = list(m._modules)  # noqa: SLF001
            for c, b in itertools.pairwise(keys):
                if base_type(m._modules[c]) is nn.Conv2d and base_type(m._modules[b]) is nn.BatchNorm2d:  # noqa: SLF001
                    pairs.append((m, c, b))
    return pairs


def fold_batchnorms(model: nn.Module) -> int:
    """Fold every eval BatchNorm into the conv before it (float32, before the fp16 cast), the BN slot an `Identity`."""
    pairs = _bn_pairs(model)
    for owner, c, b in pairs:
        conv, bn = getattr(owner, c), getattr(owner, b)
        w, bias = fuse_conv_bn_weights(
            conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
        )
        conv.weight, conv.bias = w, bias
        setattr(owner, b, nn.Identity())
    return len(pairs)


def _dw_nchw(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Dilated depthwise conv in NCHW, the output back to channels-last.

    cuDNN's channels-last fp16 depthwise kernels are 4-5x slower on sm_120 (measured in our fork).
    """
    y = F.conv2d(x.contiguous(), self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return y.contiguous(memory_format=torch.channels_last)


def _aspp_nchw(self: mm.ASPPHead, x: torch.Tensor) -> torch.Tensor:
    """`ASPPHead.aspp` with one NCHW copy of `x` for all dilated depthwise branches (`_dw_nchw` takes it as it is)."""
    xn = x.contiguous()
    outs = [mm.resize(self.image_pool(x), x.shape[2:])]
    outs += [m(xn if isinstance(m, mm.DepthwiseSeparableConvModule) else x) for m in self.aspp_modules]
    return self.bottleneck(mm.cat(outs))


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 channels-last on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    fused = False  # kernels: logits resize and argmax by `seg_up`

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = network(ctx)
        if _FOLD_BN:
            fold_batchnorms(net)
        self.net = net.to(ctx.device, ctx.torch_dtype)
        if ctx.device.type == "cuda":
            self.net = self.net.to(memory_format=torch.channels_last)
            for m in self.net.modules():
                if is_base(m, nn.Conv2d) and m.groups == m.in_channels > 1 and m.dilation != (1, 1):
                    # NCHW strides: `[C, 1, k, k]` channels-last strides also pass as contiguous but pick cuDNN's
                    # channels-last grouped kernel; these pick PyTorch's depthwise kernel, as the fork
                    m.weight.data = m.weight.data.flatten().view(m.weight.shape)
                    override(m, _dw_nchw)
            for m in self.net.modules():
                if isinstance(m, mm.DepthwiseSeparableASPPHead):
                    override(m, _aspp_nchw, "aspp")

    def _scores(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[tuple[int, int], tuple[int, int]]]:
        """Low-resolution logits and their resize steps: to the network input size, then the image size."""
        h, w = image.shape[-2:]
        size = target_size(h, w, self.ctx.params.resize)
        x = scale(resize(image, size), self.ctx.io)
        dtype = next(self.net.parameters()).dtype
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        s = self._forward(x.to(dtype))
        if self.fused:  # seg_up reads the fp16 channels-last scores as they are
            return s, (size, (h, w))
        if not (_RESIZE_FP16 and s.is_cuda):
            s = s.float()
        return s.contiguous(), (size, (h, w))  # NCHW: 1.6x faster bilinear upsampling

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        s, sizes = self._scores(image)
        return upsample(s, sizes, fused=self.fused)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 150, H, W]` of a batch."""
        return {"logits": self._run(image)}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no full-size logits)."""
        s, sizes = self._scores(image)
        return upsample_argmax(s, sizes, fused=self.fused)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


class Kernels(Portable):
    """BatchNorm as fp32 conv epilogues of `seg_conv` CUTLASS convs (dilated included); fused upsampling."""

    fused = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107 - see Portable
        Runner.__init__(self, ctx)
        net = network(ctx).to(ctx.device)
        _prepare_kernels(net, ctx.device, hacc=ctx.params.hacc)
        self.net = net
        self._graphs = SmallShapeGraphs(
            self._eager, pixels=_GRAPH_PIXELS, graphs=_GRAPHS, capture_after=cudnn_hacc.capture_after()
        )

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._graphs(x)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        self._graphs.release()

    module = Runner.module


RUNNERS = {"portable": Portable, "kernels": Kernels}
