"""fp16 channels-last fnet / cnet for the DEFOM kernels path (port of `fastbb.raft.optimize_encoder`, spec "fp16-hacc").

What changes in `BasicEncoder` / `MultiBasicEncoder`:
  - eval BatchNorm folded into the conv in front of it (the cnet; `conv1/norm1`, `conv2/norm2`, `conv/norm1` pairs and
    the `Sequential(conv, norm)` shortcut of a ResidualBlock)
  - InstanceNorm2d (the fnet): the fastbb kernels, fp32 statistics in one read and one apply pass on the NHWC
    activation (no layout copies); in a ResidualBlock the ReLUs and the residual add are fused into the apply passes
  - weights fp16 channels-last, dense convs with at least `HACC_MIN_MACS` multiply-adds as fp16-accumulate cuDNN graphs
    (`cudnn_hacc`); the encoder casts its image input to fp16 channels-last and returns fp16 channels-last
  - the convs on the depth features (`convd` of the fnet, `conv08 / conv16 / conv32` of the cnet) stay bf16: the DPT
    features reach |x| = 2.5e5 there, beyond the fp16 range (their output is normalised, so it is handed on as fp16)

Anything that does not fit (an unexpected norm, a channel count % 8 != 0, ...) raises: there is no PyTorch fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, is_base, override
from fastforks.kernels import cudnn_hacc, fastbb
from fastforks.shared.cnn_kernels import CL

if TYPE_CHECKING:
    from collections.abc import Sequence

HACC_MIN_MACS = 2e8  # per conv, batch x pixels x MACs per pixel: smaller convs stay on PyTorch (memory bound)
_PAIRS = (("conv1", "norm1"), ("conv2", "norm2"), ("conv", "norm1"))
_RELU, _RELU_RES = 1, 2  # in_apply modes (0: plain norm)


def _fold(conv: nn.Module, bn: nn.Module) -> None:
    with torch.no_grad():
        s = (bn.weight.float() if bn.affine else 1) / torch.sqrt(bn.running_var.float() + bn.eps)
        b = (bn.bias.float() if bn.affine else 0) - bn.running_mean.float() * s
        cb = conv.bias.float() * s + b if conv.bias is not None else b
        conv.weight.data = (conv.weight.float() * s.reshape(-1, 1, 1, 1)).to(conv.weight.dtype)
        conv.bias = nn.Parameter(cb.to(conv.weight.dtype))


def _foldable(bn: nn.Module) -> bool:
    return is_base(bn, nn.BatchNorm2d) and bn.track_running_stats and not bn.training


def fold_bn(module: nn.Module) -> int:
    """Fold the eval BatchNorm2d after a conv (RAFT block layouts) into that conv; returns how many."""
    n = 0
    for m in list(module.modules()):
        for c, b in _PAIRS:
            conv, bn = getattr(m, c, None), getattr(m, b, None)
            if is_base(conv, nn.Conv2d) and _foldable(bn):
                _fold(conv, bn)
                setattr(m, b, nn.Identity())
                n += 1
        if isinstance(m, nn.Sequential):
            ch = list(m.children())
            for i in range(len(ch) - 1):
                if is_base(ch[i], nn.Conv2d) and _foldable(ch[i + 1]):
                    _fold(ch[i], ch[i + 1])
                    m[i + 1] = nn.Identity()
                    n += 1
    return n


# ------------------------------------------------------------------------------------------------ instance norm
def _instance_norm(self: nn.Module, x: torch.Tensor, mode: int = 0, res: torch.Tensor | None = None) -> torch.Tensor:
    """InstanceNorm2d (no affine, no running stats) of an NHWC fp16 / bf16 activation with the fastbb kernels.
    mode 0: norm, 1: relu(norm), 2: relu(res + relu(norm)) (the tail of a ResidualBlock)."""
    n, c, h, w = x.shape
    ok = (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and c % 8 == 0
        and c <= 2048
        and x.is_contiguous(memory_format=CL)
        and x.data_ptr() % 16 == 0
        and not self.affine
        and (res is None or (res.dtype == x.dtype and res.shape == x.shape and res.is_contiguous(memory_format=CL)))
    )
    if not ok:
        msg = f"fastbb instance norm needs NHWC fp16/bf16, C % 8 == 0, C <= 2048, no affine (got {tuple(x.shape)} {x.dtype})"
        raise RuntimeError(msg)
    k = fastbb.load(x.device)
    sfx = "h" if x.dtype == torch.float16 else "b"
    hw = h * w
    rows = max(1, 256 // (c // 8))
    nbx = max(1, min((hw + rows - 1) // rows, max(1, 288 // n)))  # blocks per image
    rpb = (hw + nbx - 1) // nbx
    buf = torch.empty(2 * n * c * (nbx + 1), device=x.device, dtype=torch.float32)
    partial, st = buf[: 2 * n * c * nbx], buf[2 * n * c * nbx :]  # per block (mean, M2); result (mean, rstd)
    k.launch(f"in_stats_{sfx}", (nbx, n, 1), (256, 1, 1), x, partial, hw, c, rpb, smem=fastbb.IN_STATS_SMEM)
    k.launch(
        "in_stats_final",
        ((n * c + 7) // 8, 1, 1),
        (256, 1, 1),
        partial,
        st,
        n,
        nbx,
        hw,
        c,
        rpb,
        float(self.eps),
        None,
        None,
    )
    y = torch.empty_like(x)
    k.launch(
        f"in_apply_{sfx}", *linear(n * hw * c // 8), x, y, st, None, None, res, n, hw, c, float(self.eps), mode
    )  # fmt: skip
    return y


def _resblock_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ResidualBlock with instance norms: the ReLUs and the residual add run inside the norm passes."""
    y = _instance_norm(self.norm1, self.conv1(x), _RELU)
    r = x if self.downsample is None else _instance_norm(self.norm3, self.downsample[0](x), 0)
    return _instance_norm(self.norm2, self.conv2(y), _RELU_RES, r.contiguous(memory_format=CL))


def _is_in_resblock(m: nn.Module) -> bool:
    """A RAFT ResidualBlock with instance norms (the layout `_resblock_forward` fuses)."""
    if base_type(m).__name__ != "ResidualBlock":
        return False
    norms = ("norm1", "norm2")
    if not all(base_type(getattr(m, n, None)) is nn.InstanceNorm2d for n in norms):
        return False
    d = getattr(m, "downsample", "missing")
    if d is None:
        return True
    return (
        isinstance(d, nn.Sequential)
        and len(d) == 2
        and is_base(d[0], nn.Conv2d)
        and d[1] is getattr(m, "norm3", None)
        and is_base(d[1], nn.InstanceNorm2d)
    )


# ------------------------------------------------------------------------------------------------ convs
def _conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    n, _, h, w = x.shape
    if self._macs_per_px * n * h * w >= HACC_MIN_MACS:
        return cudnn_hacc.conv_half_acc(
            x, self.weight, self.bias, list(self.padding), list(self.stride), list(self.dilation)
        )
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _cast(v: object, dt: torch.dtype) -> object:
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        return v.to(dt).contiguous(memory_format=CL) if v.dim() == 4 else v.to(dt)
    if isinstance(v, (list, tuple)):
        return type(v)(_cast(t, dt) for t in v)
    return v


def _bf16_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return base_type(self).forward(self, _cast(x, torch.bfloat16)).to(self._dt_out)


def _keep_bf16(module: nn.Module, dt_out: torch.dtype) -> None:
    """`module` runs in bf16: its input is cast to bf16, its output to `dt_out`."""
    module.to(torch.bfloat16)
    module._dt_out = dt_out
    override(module, _bf16_forward)


def _fp16_encoder_forward(self: nn.Module, x: torch.Tensor, *a: object, **kw: object) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        return base_type(self).forward(self, _cast(x, torch.float16), *a, **kw)


def optimize_encoder(enc: nn.Module, keep_bf16: Sequence[str] = ()) -> dict[str, int]:
    """Convert a DEFOM fnet / cnet in place (see the module docstring); `keep_bf16`: direct submodules that stay bf16.
    The first positional argument (the image, or a list of them) is cast to fp16 channels-last; others pass through.
    Returns counts of what was converted."""
    enc.eval()
    mods: list[Any] = list(enc.modules())
    kept = [getattr(enc, n) for n in keep_bf16]
    kept_ids = {id(m) for k in kept for m in k.modules()}
    info = {"folded_bn": fold_bn(enc), "instance_norm": 0, "fused_blocks": 0, "hacc_convs": 0}
    for m in mods:
        if is_base(m, nn.InstanceNorm2d):
            if m.affine or m.track_running_stats:
                msg = "DEFOM kernels encoder: instance norms without affine / running statistics only"
                raise NotImplementedError(msg)
            override(m, _instance_norm)
            info["instance_norm"] += 1
        elif isinstance(m, nn.GroupNorm):
            msg = "DEFOM kernels encoder: GroupNorm encoders are not supported (norm_fn must be batch or instance)"
            raise NotImplementedError(msg)
    for m in mods:
        if _is_in_resblock(m) and id(m) not in kept_ids:
            override(m, _resblock_forward)
            info["fused_blocks"] += 1
    enc.to(torch.float16)
    for k in kept:
        _keep_bf16(k, torch.float16)
    for m in mods:
        if isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=CL)
            if id(m) in kept_ids:
                continue
            kh, kw = m.kernel_size
            narrow = min(m.in_channels, m.out_channels) < 64
            m._macs_per_px = (
                0 if narrow or m.groups != 1 else m.in_channels * m.out_channels * kh * kw / (m.stride[0] * m.stride[1])
            )
            override(m, _conv_forward)
            info["hacc_convs"] += m._macs_per_px > 0
    override(enc, _fp16_encoder_forward)
    return info


def prepare_encoders(model: nn.Module) -> None:
    """fnet and cnet of a DEFOMStereo `model` (eval, on its device)."""
    device = next(model.fnet.parameters()).device
    cudnn_hacc.load(device)
    fastbb.load(device)
    optimize_encoder(model.fnet, keep_bf16=("convd",))
    optimize_encoder(model.cnet, keep_bf16=("conv08", "conv16", "conv32"))


__all__ = ["optimize_encoder", "prepare_encoders"]
