"""Kernels-mode building blocks for CNNs: fp16 channels-last convolutions with an fp32 per-channel epilogue.

Shared by the kernels backends of the CNN stereo families (IGEV-Stereo and the cost-volume nets). The rules, measured in
our forks (v8, IGEV `cuda-kernels`, the stereo zoo `zoo_fastgen`):

- weights pure fp16, channels-last (NHWC / NDHWC); activations stay channels-last through every patched module
- BatchNorm is kept as an fp32 per-channel affine `(s, t)` applied in the conv epilogue: folding it into fp16 weights
  loses precision
- convs run as fp16-accumulate cuDNN graphs (`cudnn_hacc`, ConvTranspose as conv_dgrad); output channels are padded
  to a multiple of 8 and dropped again by `conv_ep`; bias / affine / residual / activation are one `igev` epilogue
- conv + epilogue as one CUTLASS kernel (`cutlass_conv`) for >= 128 output channels (64/96-channel convs are as fast
  or faster on cuDNN + a separate pass); a geometry CUTLASS does not support falls back to cuDNN for that module

Use: `load(device)` (raises if a library is unusable), compute BN affines with `bn_affine` while the model is fp32,
then `prepare_convs(model)`; or `prepare_generic(model, device)` for plain `Sequential(conv, BN, act)` nets.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, is_base, override
from fastforks.kernels import cudnn_hacc, cutlass_conv
from fastforks.kernels import igev as igev_kernels
from fastforks.kernels.cudnn_hacc import conv_half_acc, deconv_half_acc
from fastforks.kernels.cutlass_conv import ConvEpiArgs, narrow_cfg

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

# A module this file patches: it carries our attributes (`_s`, `_bias32`, `_fuse`, ...) and a replaced `forward`;
# torch types every attribute of an `nn.Module` as `Tensor | Module`, which cannot describe that.
Patched: TypeAlias = Any
CL = torch.channels_last
# activation codes of the epilogue kernels (affine_act / cutlass_conv)
ACT = {"none": 0, "relu": 1, "leaky": 2, "tanh": 3, "sigmoid": 4, "relu6": 5, "silu": 6, "gelu": 7}
CONVS = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
BATCH_NORMS = (nn.BatchNorm2d, nn.BatchNorm3d)
FUSE_MIN_CHANNELS = 128
_CUTLASS_ACT = {ACT["none"]: 0, ACT["relu"]: 1, ACT["gelu"]: 4}  # ACT -> the cutlass_conv epilogue's act code
_ACT_MODULES = {nn.ReLU: ACT["relu"], nn.ReLU6: ACT["relu6"], nn.Sigmoid: ACT["sigmoid"], nn.Tanh: ACT["tanh"]}
_LEAKY_SLOPE = 0.01  # the only LeakyReLU slope the kernels implement
_KERNELS: dict[int | None, CubinModule] = {}  # device index -> igev kernels (filled by load)
_CUTLASS: dict[int | None, cutlass_conv.FusedConvs] = {}  # device index -> cutlass_conv library


def load(device: torch.device) -> None:
    """Load the kernel libraries on `device`.

    Raises:
        KernelUnavailableError: A library has no usable binary, or cuDNN is unusable: there is no silent
            fallback to PyTorch convs.
    """
    k, fc = igev_kernels.load(device), cutlass_conv.load(device)
    cudnn_hacc.load(device)
    _KERNELS[k.device.index], _CUTLASS[k.device.index] = k, fc


def kernels(device: torch.device) -> CubinModule:
    """The `igev` kernel library loaded on `device` by `load`."""
    return _KERNELS[device.index]


def memory_format(dim: int) -> torch.memory_format:
    """Channels-last format for a 4-D or 5-D tensor."""
    return CL if dim == 4 else torch.channels_last_3d  # noqa: PLR2004 - 4-D: NCHW


def fused_cfg(pixels: int) -> int:
    """CUTLASS tile config, fixed size rule (RTX 5060 Ti): 128x128x32/3 stages, 64x128x64/4 for small problems."""
    return 0 if pixels >= 7000 else 1  # noqa: PLR2004 - measured crossover


def _rows(x: torch.Tensor) -> tuple[int, int, int] | None:
    """(pixels, channels, row stride) of a dense channels-last fp16 tensor, else None."""
    if x.dim() not in {4, 5} or not x.is_contiguous(memory_format=memory_format(x.dim())) or x.dtype != torch.float16:
        return None
    c = x.shape[1]
    return x.numel() // c, c, c


def epilogue(
    y: torch.Tensor,
    s: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    act: int = 0,
    r: torch.Tensor | None = None,
) -> torch.Tensor:
    """In place `y = act(y * s + t + r)`; `y` channels-last fp16, `s` / `t` fp32 `[C]` or None, `r` like `y` or None.

    Other layouts / dtypes run the same math in PyTorch (fp32, then cast back).
    """
    ry = _rows(y)
    if ry is None or (r is not None and (_rows(r) is None or r.shape != y.shape)):
        return _epilogue_torch(y, s, t, act, r)
    p, c, stride = ry
    vec = c % 8 == 0
    n = p * (c // 8) if vec else p * c
    rc = 0 if r is None else c
    name = ("affine_act8" if vec else "affine_act") + (
        "x" if act > ACT["relu6"] else ""
    )  # silu / gelu: extended kernels
    kernels(y.device).launch(name, *linear(n), y, stride, s, t, r, rc, p, c, act)
    return y


def _epilogue_torch(
    y: torch.Tensor, s: torch.Tensor | None, t: torch.Tensor | None, act: int, r: torch.Tensor | None
) -> torch.Tensor:
    shape = (1, -1) + (1,) * (y.dim() - 2)
    yf = y.float()
    if s is not None:
        yf = yf * s.view(shape)
    if t is not None:
        yf = yf + t.view(shape)
    if r is not None:
        yf = yf + r.float()
    acts = (
        lambda v: v,
        F.relu,
        lambda v: F.leaky_relu(v, _LEAKY_SLOPE),
        torch.tanh,
        torch.sigmoid,
        F.relu6,
        F.silu,
        F.gelu,
    )
    return y.copy_(acts[act](yf))


def _fusable(m: Patched, x: torch.Tensor, act: int, r: torch.Tensor | None) -> bool:
    return (
        getattr(m, "_fuse", False)
        and r is None
        and (act in {ACT["none"], ACT["relu"]} or (act in _CUTLASS_ACT and getattr(m, "_wave", False)))
        and x.dim() == 4  # noqa: PLR2004 - 2-D convs only
        and x.dtype == torch.float16
        and x.shape[0] == 1
        and x.is_contiguous(memory_format=CL)
    )


def conv_ep(  # noqa: PLR0913 - conv module, input and the epilogue terms
    m: Patched,
    x: torch.Tensor,
    s: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    act: int = 0,
    *,
    r: torch.Tensor | None = None,
    keep_pad: bool = False,
) -> torch.Tensor:
    """`act(conv(x) * s + t + r)` for a prepared conv module (Conv / ConvTranspose 2d / 3d, any groups).

    `t` defaults to the conv bias; with `s` (BatchNorm) pass `t = shift + bias * s` (`bn_affine` does). `keep_pad`
    keeps the zero output channels the weights were padded with.
    """
    if t is None and s is None:
        t = m._bias32  # noqa: SLF001 - set by prepare_convs
    if _fusable(m, x, act, r):
        y = conv_fused(m, x, s, t, _CUTLASS_ACT[act])
        if y is not None:
            return y if y.shape[1] == m.out_channels or keep_pad else y[:, : m.out_channels]
    y = conv_raw(m, x)
    if s is not None or t is not None or act or r is not None:
        y = epilogue(y, s, t, act, r)
    if y.shape[1] != m.out_channels and not keep_pad:
        y = y[:, : m.out_channels]
    return y


def conv_raw(m: Patched, x: torch.Tensor) -> torch.Tensor:
    """The convolution of a prepared conv module without bias, channels-last (padded output channels included)."""
    if getattr(m, "_hacc", False):
        return conv_half_acc(x, m.weight, None, list(m.padding), list(m.stride), list(m.dilation))
    if getattr(m, "_hacc_t", False):
        return deconv_half_acc(x, m.weight, list(m.padding), list(m.stride))
    mf = memory_format(x.dim())
    if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        f = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d  # noqa: PLR2004
        return f(x, m.weight, None, m.stride, m.padding, m.output_padding, m.groups, m.dilation).contiguous(
            memory_format=mf
        )
    f = F.conv2d if x.dim() == 4 else F.conv3d  # noqa: PLR2004
    return f(x, m.weight, None, m.stride, m.padding, m.dilation, m.groups).contiguous(memory_format=mf)


def cutlass(  # noqa: PLR0913, PLR0917 - the kernel's argument list
    cfg: int,
    x: torch.Tensor,
    w: torch.Tensor,
    hw: tuple[int, int],
    e: ConvEpiArgs,
    padding: tuple[int, int] | None = None,
    stride: int = 1,
) -> bool:
    """One CUTLASS launch; `w` is the channels-last `[K, C, R, S]` weight. False: geometry not supported."""
    return _CUTLASS[x.device.index].conv(cfg, x, w.permute(0, 2, 3, 1), hw, e, stride=stride, padding=padding)


def conv_fused(  # noqa: PLR0913, PLR0917 - conv module, input, epilogue terms and destinations
    m: Patched,
    x: torch.Tensor,
    s: torch.Tensor | None,
    t: torch.Tensor | None,
    act: int,
    out: tuple[torch.Tensor, int] | None = None,
    out2: tuple[torch.Tensor, int] | None = None,
    res: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """2-D conv + epilogue as one CUTLASS kernel; None if the geometry is unsupported (the module then uses cuDNN).

    The decision is per input size: a size the kernel rejected is remembered on the module and goes straight to
    cuDNN; other sizes keep the kernel (`fuse_ok`).

    `res` (with the kernel's `act` 3, not `ACT`): the ResidualBlock tail `relu(res + relu(conv * s + t))`, `res` a
    dense channels-last fp16 tensor of the output's shape.

    `out` / `out2`: `(buffer, channel offset)` destinations inside wider channels-last fp16 buffers (`out2`: a second
    copy with the same shape); the offset is a multiple of 8 and the output's channels fit behind it.

    Raises:
        RuntimeError: `out` / `out2` / `res` do not fit the output, or `res` is combined with `out`.
    """
    _, _, h, w = x.shape
    bad = m.__dict__.get("_fuse_bad")
    if bad is not None and (h, w) in bad:
        return None
    k = m.weight.shape[0]
    (kh, kw), (ph, pw), (st, _) = m.kernel_size, m.padding, m.stride
    ho, wo = (h + 2 * ph - kh) // st + 1, (w + 2 * pw - kw) // st + 1
    if out is None:
        y = torch.empty((1, k, ho, wo), device=x.device, dtype=x.dtype, memory_format=CL)
        yp, ys = y.data_ptr(), k
    else:
        y, c0 = out
        _check_dest(out, k, (ho, wo), "out")
        if res is not None:  # the kernel reads the residual at the output's row stride
            msg = "conv_fused: `res` with `out` is not supported"
            raise RuntimeError(msg)
        yp, ys = y.data_ptr() + 2 * c0, y.shape[1]
    y2p = None
    if out2 is not None:
        _check_dest(out2, k, (ho, wo), "out2")
        if out2[0].shape != y.shape:
            msg = f"conv_fused: out2 {tuple(out2[0].shape)} must have the shape of out {tuple(y.shape)}"
            raise RuntimeError(msg)
        y2p = out2[0].data_ptr() + 2 * out2[1]
    if res is not None and (res.shape != (1, k, ho, wo) or _rows(res) is None or res.data_ptr() % 16):
        msg = f"conv_fused: res must be a dense channels-last fp16 (1, {k}, {ho}, {wo}) tensor, got {tuple(res.shape)}"
        raise RuntimeError(msg)
    sp, tp = 0 if s is None else s.data_ptr(), 0 if t is None else t.data_ptr()
    rp = 0 if res is None else res.data_ptr()
    e = ConvEpiArgs(2, act, rp, ys, yp, ys, None, ys, y2p, None, sp, tp)
    wave = k < FUSE_MIN_CHANNELS or getattr(m, "_wave", False)
    cfg = narrow_cfg(ho * wo, k, x.device) if wave else fused_cfg(ho * wo)
    if not cutlass(cfg, x, m.weight, (h, w), e, padding=(ph, pw), stride=st):
        m.__dict__.setdefault("_fuse_bad", set()).add((h, w))  # this input size runs on cuDNN from now on
        return None
    return y


def fuse_ok(m: Patched, h: int, w: int) -> bool:
    """`m` is marked for the CUTLASS kernel and an `h` x `w` input was not rejected by it before."""
    return getattr(m, "_fuse", False) and (h, w) not in m.__dict__.get("_fuse_bad", ())


def _check_dest(dest: tuple[torch.Tensor, int], k: int, hw: tuple[int, int], name: str) -> None:
    """`dest = (y, c0)` can take a `k`-channel output of size `hw` at channel offset `c0` (16-byte stores)."""
    (y, c0), (ho, wo) = dest, hw
    if not (
        y.dim() == 4  # noqa: PLR2004 - NCHW
        and y.dtype == torch.float16
        and y.is_contiguous(memory_format=CL)
        and tuple(y.shape[2:]) == (ho, wo)
        and y.shape[0] == 1
        and y.data_ptr() % 16 == 0
    ):
        msg = f"conv_fused: {name} must be a channels-last fp16 (1, C, {ho}, {wo}) buffer, got {tuple(y.shape)}"
        raise RuntimeError(msg)
    if c0 < 0 or c0 % 8 or c0 + k > y.shape[1]:
        msg = f"conv_fused: {name} channel offset {c0} must be a multiple of 8 with {c0} + {k} <= {y.shape[1]}"
        raise RuntimeError(msg)


# ------------------------------------------------------------------------------------------------ norms
def bn_affine(bn: Patched, conv: Patched = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Eval BatchNorm as an fp32 affine `(s, t)`; with `conv`, its bias goes into `t` (call before the fp16 cast)."""
    s = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
    t = bn.bias.float() - bn.running_mean.float() * s
    if conv is not None and conv.bias is not None:
        t = t + conv.bias.float() * s
    return s.contiguous(), t.contiguous()


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fold an eval BatchNorm into the conv before it, in place (call before the fp16 cast; `bn` is then a no-op).

    The conv then outputs the BatchNorm's scale: an unnormalised conv output can exceed float16 where the normalised
    one does not, also when `conv_ep` applies the BatchNorm in its epilogue (the raw conv is stored as float16 first).
    """
    s, t = bn_affine(bn, conv)
    with torch.no_grad():
        conv.weight.mul_(s.to(conv.weight.dtype).view(-1, *[1] * (conv.weight.dim() - 1)))
    conv.bias = nn.Parameter(t.to(conv.weight.dtype))
    for name, v in (("running_mean", 0.0), ("running_var", 1.0 - bn.eps), ("weight", 1.0), ("bias", 0.0)):
        getattr(bn, name).data.fill_(v)


def instance_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample, per-channel mean and (biased) variance over the spatial dims, fp32."""
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    return mean, var


def instance_norm(x: torch.Tensor, eps: float, act: int = 0) -> torch.Tensor:
    """`act(InstanceNorm(x))` (no affine) in place on a channels-last `x`, one epilogue per sample."""
    mean, var = instance_stats(x)
    rstd = torch.rsqrt(var + eps)
    shift = -mean * rstd
    if x.shape[0] == 1:
        return epilogue(x, rstd.view(-1).contiguous(), shift.view(-1).contiguous(), act)
    for i in range(x.shape[0]):  # the epilogue's scale / shift are per channel
        epilogue(x[i : i + 1], rstd[i].view(-1).contiguous(), shift[i].view(-1).contiguous(), act)
    return x


def _instance_norm_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    return instance_norm(x.contiguous(memory_format=CL), self.eps)


def _bn_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    y = epilogue(x, self._s, self._t, self._act)
    return y if self._post is None else self._post(y)


def act_code(m: Patched) -> int | None:
    """Epilogue code of an activation module (None: not implemented by the kernels)."""
    if m is None or isinstance(m, nn.Identity):
        return 0
    if isinstance(m, nn.LeakyReLU):
        return ACT["leaky"] if abs(m.negative_slope - _LEAKY_SLOPE) < 1e-12 else None  # noqa: PLR2004
    return _ACT_MODULES.get(base_type(m))


def patch_norms(model: Patched) -> None:
    """Standalone BatchNorm (incl. timm's BatchNormAct2d) and InstanceNorm2d (no affine) as layout-preserving kernels.

    A BatchNorm whose `_s` is already set keeps it (the caller fused it). Call while the model is fp32.
    """
    for m in model.modules():
        if isinstance(m, BATCH_NORMS):
            if not hasattr(m, "_s"):
                m._s, m._t = bn_affine(m)  # noqa: SLF001
            a = getattr(m, "act", None)  # BatchNormAct2d: the activation lives in the BN module
            code = act_code(a)
            m._act, m._post = (0, a) if code is None else (code, None)  # noqa: SLF001
            override(m, _bn_forward)
        elif is_base(m, nn.InstanceNorm2d) and not m.affine and not m.track_running_stats:
            override(m, _instance_norm_forward)  # torch's returns NCHW for an NHWC input


# ------------------------------------------------------------------------------------------------ convs
def _conv_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    return conv_ep(self, x)


def _mark_hacc(model: Patched) -> None:
    """Mark the fp16 convs cuDNN's fp16-accumulate graphs support.

    Groups 1, dilation 1, zero padding, no output padding; conv output channels are zero-padded to a multiple of 8
    (`conv_ep` drops them).
    """
    for m in model.modules():
        if (
            base_type(m) in {nn.ConvTranspose2d, nn.ConvTranspose3d}
            and m.groups == 1
            and all(d == 1 for d in m.dilation)
            and not any(m.output_padding)
        ):
            m.weight.data = m.weight.data.contiguous(memory_format=memory_format(m.weight.dim()))
            m._hacc_t = True  # noqa: SLF001
        if (
            base_type(m) in {nn.Conv2d, nn.Conv3d}
            and m.groups == 1
            and all(d == 1 for d in m.dilation)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
        ):
            w = m.weight.data
            if w.shape[0] % 8:
                pad = torch.zeros((8 - w.shape[0] % 8, *w.shape[1:]), dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], 0)
                if m._bias32 is not None:  # noqa: SLF001
                    m._bias32 = torch.cat([m._bias32, m._bias32.new_zeros(pad.shape[0])]).contiguous()  # noqa: SLF001
            m.weight.data = w.contiguous(memory_format=memory_format(w.dim()))
            m._hacc = True  # noqa: SLF001


def _fuse_rule(m: Patched, min_channels: int = FUSE_MIN_CHANNELS) -> bool:
    return (
        is_base(m, nn.Conv2d)
        and m.groups == 1
        and tuple(m.dilation) == (1, 1)
        and m.stride[0] == m.stride[1]
        and m.weight.shape[0] >= min_channels
        and m.weight.shape[0] % 8 == 0
        and m.weight.shape[1] % 8 == 0
        and isinstance(m.padding, tuple)
        and m.padding_mode == "zeros"
    )


def prepare_convs(model: Patched, *, fuse: bool = True) -> None:
    """Cast `model` to fp16 and route every conv through `conv_ep`.

    Keeps an fp32 copy of each conv bias (`_bias32`), makes weights channels-last, marks the fp16-accumulate cuDNN
    convs and (with `fuse`) the CUTLASS-fused ones. Convs with a padding mode other than zeros keep PyTorch.
    """
    model.half()
    for m in model.modules():
        if isinstance(m, CONVS):
            m._bias32 = None if m.bias is None else m.bias.float().contiguous()  # noqa: SLF001
            m.weight.data = m.weight.data.contiguous(memory_format=memory_format(m.weight.dim()))
    _mark_hacc(model)
    for m in model.modules():
        if base_type(m) in CONVS and m.padding_mode == "zeros":
            override(m, _conv_forward)
            m._fuse = fuse and _fuse_rule(m)  # noqa: SLF001


def fuse_narrow(model: Patched, min_channels: int = 64) -> int:
    """Also route the convs with `min_channels <= out < 128` through the CUTLASS kernel (narrow tiles).

    Call after `prepare_convs`; returns how many. For the BatchNorm context encoders, where it beats cuDNN + a separate
    pass.
    """
    n = 0
    for m in model.modules():
        if (
            base_type(m) in CONVS
            and m.padding_mode == "zeros"
            and not getattr(m, "_fuse", False)
            and _fuse_rule(m, min_channels)
        ):
            m._fuse = True  # noqa: SLF001
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ generic nets
class FusedConvBN(nn.Module):
    """`act(bn(conv(x)))` as `conv_ep(conv, x, s, t, act)`: same math, the BN affine in fp32 in the epilogue.

    Build it on the target device: the affine is not a buffer, so `.to()` does not move it.
    """

    def __init__(self, conv: Patched, bn: Patched, act: int) -> None:
        """Fuse `conv`, eval-mode `bn` and the activation code `act` (call while the modules are fp32)."""
        super().__init__()
        self.conv = conv
        self.act = act
        self.s, self.t = bn_affine(bn, conv)  # plain attributes: the fp16 cast must not reach the fp32 affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused conv -> BN -> activation."""
        return conv_ep(self.conv, x, self.s, self.t, self.act)


class FirstChannels(nn.Module):
    """`x[:, :n]`."""

    def __init__(self, n: int) -> None:
        """Keep the first `n` channels."""
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The first `n` channels of `x` (a view)."""
        return x[:, : self.n]


def pad_deconv_out(m: nn.Module) -> nn.Module:
    """`m` with its output channels zero-padded to a multiple of 8, the padding sliced off after it (call while fp32).

    cuDNN's fp16 NHWC transposed-conv kernels need 8 | output channels; with 9 channels (the superpixel upsampling
    weights of LAS2 / CoEx) a layout transform of the full-resolution output runs around the conv.

    Raises:
        TypeError: `m` is not a `ConvTranspose2d` with groups 1.
    """
    if not isinstance(m, nn.ConvTranspose2d) or m.groups != 1:
        msg = f"pad_deconv_out: needs a ConvTranspose2d with groups 1, got {m}"
        raise TypeError(msg)
    k = -(-m.out_channels // 8) * 8
    if k == m.out_channels:
        return m
    d = copy.deepcopy(m)
    w = m.weight.detach()
    d.weight = nn.Parameter(torch.cat((w, w.new_zeros(w.shape[0], k - m.out_channels, *w.shape[2:])), 1))
    if m.bias is not None:
        b = m.bias.detach()
        d.bias = nn.Parameter(torch.cat((b, b.new_zeros(k - m.out_channels))))
    d.out_channels = k
    return nn.Sequential(d, FirstChannels(m.out_channels))


def fuse_sequentials(model: Patched) -> int:
    """Replace `conv -> BatchNorm [-> activation]` runs inside every `nn.Sequential` by `FusedConvBN`.

    The BN and activation slots become `nn.Identity` (indices and state-dict keys of the conv are unchanged). Only
    plain BatchNorm modules fuse (BatchNormAct has its own activation). Returns the number of fused runs.
    """
    n = 0
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        children = list(seq._modules.items())  # noqa: SLF001 - in-place slot replacement
        i = 0
        while i < len(children):
            name, m = children[i]
            if not (
                isinstance(m, CONVS)
                and m.padding_mode == "zeros"
                and i + 1 < len(children)
                and base_type(children[i + 1][1]) in BATCH_NORMS
            ):
                i += 1
                continue
            act, j = 0, i + 2
            if j < len(children) and (code := act_code(children[j][1])):
                act, j = code, j + 1
            seq._modules[name] = FusedConvBN(m, children[i + 1][1], act)  # noqa: SLF001
            for k in range(i + 1, j):
                seq._modules[children[k][0]] = nn.Identity()  # noqa: SLF001
            n += 1
            i = j
    return n


def fuse_selective(model: Patched, ratio: int = 4) -> int:
    """Keep the CUTLASS conv + epilogue only for stride > 1 convs and 1x1 expansions (out >= `ratio` x in channels).

    Call after `prepare_convs` / `prepare_generic`. The kept convs choose tiles by wave count (`narrow_cfg`, also for
    >= 128 channels) and may fuse GELU; every other conv runs on cuDNN fp16-accumulate + the epilogue pass. Rule from
    per-conv GPU times (CUDA graphs) on LAS2 at 384x1248 and 994x1482: the 128-wide tile configs of `fused_cfg` were up
    to 2x slower than the wave rule, and cuDNN beat every config on 1x1 projections and stride-1 3x3 convs by up to
    1.5x. Returns the number of convs kept on CUTLASS.
    """
    n = 0
    for m in model.modules():
        if getattr(m, "_fuse", False):
            k, c, kh, kw = m.weight.shape
            keep = m.stride[0] > 1 or (kh == kw == 1 and k >= ratio * c)
            m._fuse = m._wave = keep  # noqa: SLF001
            n += keep
    return n


def prepare_generic(model: Patched, device: torch.device, *, fuse: bool = True) -> int:
    """Kernels path for a plain CNN on `device` (fp32, eval): fused conv-BN-act runs, norm kernels, fp16 convs.

    The model code must then feed fp16 channels-last inputs and keep softmax / regression in fp32 itself. Returns the
    number of fused `Sequential` runs.

    Raises:
        KernelUnavailableError: See `load`.
    """
    load(device)
    model.eval()
    n = fuse_sequentials(model)
    patch_norms(model)
    prepare_convs(model, fuse=fuse)
    return n
