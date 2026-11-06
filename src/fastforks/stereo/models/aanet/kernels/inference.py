"""Kernels inference path for AANet / AANet+ (general: any input size, no per-size compilation / tuning).

Port of our AANet fork's `cuda-kernels` branch (`zoo_fast.py` + `zoo_fastgen/wrap.py` @ f774b57): the generic fp16
path of the CNN stereo zoo (`fastforks.shared.cnn_kernels.prepare_generic`: conv -> BatchNorm -> activation runs as one
conv with an fp32 affine epilogue, fp16-accumulate cuDNN convs, CUTLASS fused convs for >= 128 output channels) plus
the AANet specifics:

- the grouped offset convolution of every deformable convolution (groups = deformable groups) is one dense conv with
  block-diagonal weights: twice the MACs, but tensor-core friendly (the grouped fp16 conv at full resolution hit a
  slow cuDNN layout-transform path)
- every deformable convolution is one `deform_conv` kernel: bilinear sampling (float32) + sigmoid mask + conv in one
  pass, no column buffer
- the aggregation's bottlenecks run every BatchNorm, ReLU and the residual add in a conv epilogue (the deformable
  conv's BatchNorm + ReLU in the `deform_conv` kernel)
- the probability volume / disparity regression runs in float32, and so does the refinement's image warp (an fp16
  sampling grid cannot address pixels beyond x = 1024); upstream's fork fed fp16 there
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.patch import is_base, override
from fastforks.kernels import deform_conv as deform_kernels
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import CL

from ..net.aanet import AANet
from ..net.deform import DeformConv2d, DeformSimpleBottleneck, SimpleBottleneck
from ..net.estimation import DisparityEstimation
from ..net.refinement import HourglassRefinement, StereoDRNetRefinement
from ..net.warp import disp_warp


def _dense(conv: nn.Conv2d) -> nn.Conv2d:
    """A grouped conv as the same conv with block-diagonal dense weights (zeros elsewhere)."""
    g = conv.groups
    w = conv.weight.data
    o, i = w.shape[0] // g, w.shape[1]
    geometry: dict[str, Any] = {
        "kernel_size": conv.kernel_size,
        "stride": conv.stride,
        "padding": conv.padding,
        "dilation": conv.dilation,
    }
    dense = nn.Conv2d(i * g, w.shape[0], bias=conv.bias is not None, **geometry).to(w.device, w.dtype)
    dense.weight.data.zero_()
    for k in range(g):
        dense.weight.data[k * o : (k + 1) * o, k * i : (k + 1) * i] = w[k * o : (k + 1) * o]
    if conv.bias is not None and dense.bias is not None:
        dense.bias.data.copy_(conv.bias.data)
    return dense


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _prepare_deform(m: cnn_kernels.Patched) -> None:
    """Pack the weights of `m` for the `deform_conv` kernel (call while the model is fp32).

    Raises:
        BackendUnavailableError: The kernel does not support the layer's channels / groups.
    """
    dc = m.deform_conv
    c = dc.weight.shape[1]
    reason = "groups > 1" if dc.groups != 1 else deform_kernels.fit_reason(c, m.deformable_groups)
    if reason:
        msg = f"AANet kernels: deformable conv {tuple(dc.weight.shape)}: {reason}"
        raise BackendUnavailableError(msg)
    m._w = deform_kernels.pack_weight(dc.weight)
    m._s, m._t, m._act = None, None, deform_kernels.ACT_NONE
    if getattr(dc, "bias", None) is not None:
        m._t = dc.bias.float().contiguous()
    m._mscale = (2.0 if m.double_mask else 1.0) if m.modulation else None
    override(m, _deform_forward)


def _deform_forward(self: cnn_kernels.Patched, x: torch.Tensor) -> torch.Tensor:
    """Offset conv, then sampling + mask + conv (+ the fused epilogue) as one kernel."""
    x = x.contiguous(memory_format=CL)
    om = self.offset_conv(x).contiguous(memory_format=CL)
    dc = self.deform_conv
    return deform_kernels.deform_conv(
        x, om, self._w, dc.weight.shape[0], _pair(dc.kernel_size), _pair(dc.stride), _pair(dc.padding),
        _pair(dc.dilation), self.deformable_groups, self._mscale, self._s, self._t, self._act,
    )  # fmt: skip


def _prepare_bottleneck(m: cnn_kernels.Patched) -> None:
    """Fold the BatchNorms of a (deformable) bottleneck into conv epilogues (call while the model is fp32)."""
    m._a1 = cnn_kernels.bn_affine(m.bn1, m.conv1)
    m._a3 = cnn_kernels.bn_affine(m.bn3, m.conv3)
    if isinstance(m, DeformSimpleBottleneck):  # bn2 + ReLU in the deform_conv epilogue
        m._a2 = None
        m.conv2._s, m.conv2._t = cnn_kernels.bn_affine(m.bn2, m.conv2.deform_conv)
        m.conv2._act = deform_kernels.ACT_RELU
    else:
        m._a2 = cnn_kernels.bn_affine(m.bn2, m.conv2)
    override(m, _bottleneck_forward)


def _bottleneck_forward(self: cnn_kernels.Patched, x: torch.Tensor) -> torch.Tensor:
    """Upstream's forward with every BatchNorm, ReLU and the residual add in a conv epilogue (fp32)."""
    relu = cnn_kernels.ACT["relu"]
    out = cnn_kernels.conv_ep(self.conv1, x, *self._a1, relu)
    out = self.conv2(out) if self._a2 is None else cnn_kernels.conv_ep(self.conv2, out, *self._a2, relu)
    identity = x if self.downsample is None else self.downsample(x)
    return cnn_kernels.conv_ep(self.conv3, out, *self._a3, relu, r=identity)


def _estimation_forward(self: DisparityEstimation, cost_volume: torch.Tensor) -> torch.Tensor:
    """Upstream's soft-argmin in float32 (fp16 would quantize disparities to 0.06-0.125 px at 64-192 px)."""
    cost = cost_volume.float()
    prob = F.softmax(cost if self.match_similarity else -cost, dim=1)
    candidates = torch.arange(prob.size(1), device=prob.device, dtype=prob.dtype).view(1, -1, 1, 1)
    return torch.sum(prob * candidates, 1)


def _refine(
    self: StereoDRNetRefinement | HourglassRefinement, low_disp: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared head of both refinements: upsampled disparity and the (error, left) / disparity feature input in fp16.

    `left` / `right` are the float32 images; returns the float32 disparity `[B, 1, H, W]` and the first features.
    """
    disp = low_disp.unsqueeze(1)
    scale = left.size(-1) / disp.size(-1)
    if scale != 1.0:
        disp = F.interpolate(disp, size=left.size()[-2:], mode="bilinear", align_corners=False) * scale
    warped = disp_warp(right, disp)[0]
    concat = torch.cat((warped - left, left), dim=1).half().contiguous(memory_format=CL)
    feat = torch.cat((self.conv1(concat), self.conv2(disp.half().contiguous(memory_format=CL))), dim=1)
    return disp, feat


def _drnet_forward(
    self: StereoDRNetRefinement, low_disp: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    disp, feat = _refine(self, low_disp, left, right)
    residual = self.final_conv(self.dilated_blocks(feat))
    return F.relu(disp + residual.float()).squeeze(1)


def _hourglass_forward(
    self: HourglassRefinement, low_disp: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    disp, x = _refine(self, low_disp, left, right)
    x = self.conv_start(x)
    rem0 = x
    x = self.conv1a(x)
    rem1 = x
    x = self.conv2a(x)
    rem2 = x
    x = self.conv3a(x)
    rem3 = x
    x = self.conv4a(x)
    rem4 = x
    x = self.deconv4a(x, rem3)
    rem3 = x
    x = self.deconv3a(x, rem2)
    rem2 = x
    x = self.deconv2a(x, rem1)
    rem1 = x
    x = self.deconv1a(x, rem0)
    rem0 = x
    x = self.conv1b(x, rem1)
    rem1 = x
    x = self.conv2b(x, rem2)
    rem2 = x
    x = self.conv3b(x, rem3)
    rem3 = x
    x = self.conv4b(x, rem4)
    x = self.deconv4b(x, rem3)
    x = self.deconv3b(x, rem2)
    x = self.deconv2b(x, rem1)
    x = self.deconv1b(x, rem0)
    return F.relu(disp + self.final_conv(x).float()).squeeze(1)


def prepare_kernels(net: AANet, device: torch.device) -> None:
    """Turn a float32 eval `net` on `device` into the kernels network.

    Raises:
        KernelUnavailableError: A kernel library has no usable binary (no silent fallback).
        BackendUnavailableError: A deformable convolution does not fit the `deform_conv` kernel.
    """
    deform_kernels.load(device)
    for m in net.modules():
        if isinstance(m, DeformConv2d):
            if m.offset_conv.groups > 1:
                m.offset_conv = _dense(m.offset_conv)
            _prepare_deform(m)
    for m in net.modules():
        if isinstance(m, (SimpleBottleneck, DeformSimpleBottleneck)) and is_base(m.relu, nn.ReLU):
            _prepare_bottleneck(m)
        elif isinstance(m, DisparityEstimation):
            override(m, _estimation_forward)
        elif isinstance(m, StereoDRNetRefinement):
            override(m, _drnet_forward)
        elif isinstance(m, HourglassRefinement):
            override(m, _hourglass_forward)
    cnn_kernels.prepare_generic(net, device)


def forward_kernels(net: AANet, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Disparity `[B, H, W]` float32 of a padded, normalized pair."""
    lh, rh = (x.half().contiguous(memory_format=CL) for x in (left, right))
    cost = net.cost_volume_construction(net.feature_extraction(lh), net.feature_extraction(rh))
    pyramid = net.disparity_computation(net.aggregation(cost))
    pyramid += net.disparity_refinement(left, right, pyramid[-1])
    disp = pyramid[-1]
    if disp.shape[-1] < left.shape[-1]:
        disp = F.interpolate(disp[:, None], left.shape[-2:], mode="bilinear")[:, 0] * (left.shape[-1] / disp.shape[-1])
    return disp
