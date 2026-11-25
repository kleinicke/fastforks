"""Unaffined InstanceNorm2d of NHWC fp16 / bf16 activations on the `fastbb` kernels, and RAFT's ResidualBlock on them.

fp32 statistics in one read pass and one apply pass over the channels-last activation (no layout copies); the apply
pass can add a ReLU (`RELU`) or a ResidualBlock's tail `relu(res + relu(norm))` (`RELU_RES`). Used by the kernels
encoders of DEFOM-Stereo (fnet) and NMRF / BridgeDepth (ResNet backbone and feature heads: `fuse_nmrf_encoder`).
There is no PyTorch fallback: an activation the kernels do not take raises.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, is_base, override
from fastforks.kernels import fastbb
from fastforks.shared.cnn_kernels import CL

RELU, RELU_RES = 1, 2  # in_apply modes (0: plain norm)
_MAX_C = 2048  # channels the stats kernel holds
_V8 = 8
_HEAD_LEN = 4  # NMRF `concatconv` / `gw`: Sequential(conv, InstanceNorm2d, ReLU, conv)


def instance_norm(  # noqa: PLR0913 - an nn.Module method (override) with the fused-pass options
    self: nn.Module,
    x: torch.Tensor,
    mode: int = 0,
    res: torch.Tensor | None = None,
    *,
    inplace: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """InstanceNorm2d (no affine, no running stats) of an NHWC fp16 / bf16 activation with the fastbb kernels.

    mode 0: norm, 1: relu(norm), 2: relu(res + relu(norm)) (the tail of a ResidualBlock). `inplace`: the result
    overwrites `x`; `out`: it is written there (`res` itself may be `out`: the apply pass is elementwise).

    Raises:
        RuntimeError: `x`, `res` or the norm is not what the kernels take.
    """
    n, c, h, w = x.shape
    ok = (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and c % _V8 == 0
        and c <= _MAX_C
        and x.is_contiguous(memory_format=CL)
        and x.data_ptr() % 16 == 0
        and not self.affine
        and (res is None or (res.dtype == x.dtype and res.shape == x.shape and res.is_contiguous(memory_format=CL)))
    )
    if not ok:
        msg = (
            f"fastbb instance norm needs NHWC fp16/bf16, C % 8 == 0, C <= {_MAX_C}, no affine"
            f" (got {tuple(x.shape)} {x.dtype})"
        )
        raise RuntimeError(msg)
    k = fastbb.load(x.device)
    eps = float(cast("float", self.eps))
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
        eps,
        None,
        None,
    )
    y = x if inplace else torch.empty_like(x) if out is None else out
    k.launch(f"in_apply_{sfx}", *linear(n * hw * c // 8), x, y, st, None, None, res, n, hw, c, eps, mode)
    return y


def resblock_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ResidualBlock with instance norms: the ReLUs and the residual add run inside the norm passes, in place.

    The norms overwrite their conv's output, and the block's result overwrites its residual: the downsample's output,
    or the block input `x` itself. Use it only where no block input is used again (the stem output and each block's
    result are only the next block's input: DEFOM's fnet, NMRF's backbone); it frees the activation the caller still
    holds during the next block (DEFOM 994 x 1482: peak 2771 -> 2437 MiB, reserved 4212 -> 3836).
    """
    b: Any = self  # a ResidualBlock (`is_in_resblock`)
    y = b.conv2(instance_norm(b.norm1, b.conv1(x), RELU, inplace=True))
    r = x if b.downsample is None else instance_norm(b.norm3, b.downsample[0](x), 0, inplace=True)
    r = r.contiguous(memory_format=CL)
    instance_norm(b.norm2, y, RELU_RES, r, out=r)
    return r


def is_in_resblock(m: nn.Module) -> bool:
    """A RAFT ResidualBlock with instance norms (the layout `resblock_forward` fuses)."""
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
        and len(d) == 2  # noqa: PLR2004 - Sequential(conv, norm)
        and is_base(d[0], nn.Conv2d)
        and d[1] is getattr(m, "norm3", None)
        and is_base(d[1], nn.InstanceNorm2d)
    )


def _backbone_forward(self: Any, x: torch.Tensor) -> list[torch.Tensor]:  # noqa: ANN401 - a torch module, typed dynamically
    """`Backbone.forward` in fp16 channels-last from the scaled input on, the stem ReLU inside its norm pass."""
    x = (2 * (x / 255.0) - 1.0).half().contiguous(memory_format=CL)
    x = instance_norm(self.norm1, self.conv1(x), RELU, inplace=True)
    x = self.conv2(self.layer3(self.layer2(self.layer1(x))))
    return [x, F.avg_pool2d(x, kernel_size=2, stride=2)]  # high to low res


def _norm_relu(self: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - a torch module, typed dynamically
    return instance_norm(self, x, RELU, inplace=True)


def fuse_nmrf_encoder(encoder: Any) -> int:  # noqa: ANN401 - a torch module, typed dynamically
    """Run an NMRF / BridgeDepth encoder's instance norms on `fastbb` (fp16, in place); the modules changed.

    The ResNet `Backbone` (if it is one: BridgeDepth's ConvNeXt variant has none) and the `concatconv` / `gw` heads,
    `Sequential(conv, InstanceNorm2d, ReLU, conv)`, whose ReLU moves into the norm's apply pass.

    Raises:
        RuntimeError: A block or head does not have the layout the fused forwards assume.
    """
    count = 0
    bb = encoder.backbone
    if base_type(bb).__name__ == "Backbone" and is_base(getattr(bb, "norm1", None), nn.InstanceNorm2d):
        blocks = [m for m in bb.modules() if base_type(m).__name__ == "ResidualBlock"]
        if not blocks or not all(is_in_resblock(b) for b in blocks):
            msg = "NMRF kernels: unexpected ResidualBlock layout in the backbone"
            raise RuntimeError(msg)
        for b in blocks:
            override(b, resblock_forward)
        override(bb, _backbone_forward)
        bb.to(memory_format=CL)
        count += len(blocks) + 1
    for seq in (encoder.concatconv, encoder.gw):
        if not (len(seq) == _HEAD_LEN and is_base(seq[1], nn.InstanceNorm2d) and isinstance(seq[2], nn.ReLU)):
            msg = "NMRF kernels: unexpected concatconv / gw layout"
            raise RuntimeError(msg)
        override(seq[1], _norm_relu)
        seq[2] = nn.Identity()
        seq.to(memory_format=CL)
        count += 1
    return count
