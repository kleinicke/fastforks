"""Deformable convolution (v1 and modulated v2) in plain PyTorch ops.

Replaces upstream's `nets/deform_conv/` (the mmdetection-v1 CUDA extension, GPU only, does not build on current
torch). Same module classes, parameters and state-dict keys as upstream. Same math as the extension:

- `offset`: `[B, deformable_groups * 2 * kh * kw, Ho, Wo]`, (dy, dx) pairs per deformable group and kernel tap
  (tap index `i * kw + j`); `mask`: `[B, deformable_groups * kh * kw, Ho, Wo]`
- the input is sampled bilinearly at `(ho * stride - padding + i * dilation + dy, ...)`; samples outside the image
  contribute zero (a sample in (-1, 0) interpolates with a zero neighbour)
- the sampled values are scaled by the mask and contracted with the weight

The sampling is `grid_sample(align_corners=True, padding_mode="zeros")`, the contraction a matmul per row chunk.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from fastforks.shared.deform_conv import modulated_deform_conv  # fastforks: the plain-PyTorch op, shared with ALIKED (lightglue)


class DeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.modulated_deform_conv = modulated_deform_conv  # the kernels runner on Apple MPS swaps in a Metal kernel

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return self.modulated_deform_conv(x, offset, None, self.weight, None, self.stride, self.padding,
                                          self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.modulated_deform_conv = modulated_deform_conv  # the kernels runner on Apple MPS swaps in a Metal kernel

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return self.modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                          self.stride, self.padding, self.dilation,
                                          self.groups, self.deformable_groups)
