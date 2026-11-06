"""Kernels inference path for DLNR (general: any input size /32, batch 1, no per-size compilation / tuning).

The portable forward (`net/inference.py`) on the shared kernel libraries:

prepare_kernels(model, device), then forward_kernels(model, k, left, right, iters) (images [1, 3, H, W], 0..255, /32):
  - fp16 weights; the convs outside the Restormer blocks as fp16-accumulate cuDNN graphs (`cudnn_hacc`) or CUTLASS
    conv + epilogue (`cutlass_conv`), bias / InstanceNorm / ReLU in the `igev` epilogue kernels
    (`fastforks.shared.cnn_kernels`); the Restormer blocks on the `fastbb` LayerNorm / depthwise / GELU-gate kernels
    (`kernels.restormer`)
  - NormalizationRefinement: warp, normalisation and the final residual sum in fp32 as upstream, its hourglass of
    conv + BN + ReLU in fp16 channels-last with BN in the conv epilogue (portable runs it all in fp32: 75 ms of the
    400 ms at Middlebury-H); the 1-channel disparity conv stays fp32
  - LSTM gates: one merged gate conv per level whose epilogue adds the context (+ gate biases) and applies the sigmoid
    to all four gates; the candidate gate's weights and context are pre-scaled by 2 so that tanh(c) = 2 sigmoid(2c) - 1
    (as our MoCha-Stereo kernels path)
  - correlation: the `raft` kernels of our RAFT-Stereo kernels engine (fp16 all-pairs volume `corr_gemm`, 1-D average
    pyramid `corr_pool`, 4 levels x 9 linearly interpolated taps `corr_lookup`, 40 channels with 4 zero channels: the
    motion encoder's 1x1 correlation conv gets 4 zero input channels); portable keeps the volume in fp32
"""

import torch
from torch import nn

from fastforks._core.kernels import linear
from fastforks.kernels import fastbb, raft
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep, instance_norm

from ..net.extractor import TransformerBlock
from ..net.inference import _upsample
from ..net.refinement import disp_warp
from ..net.update import interp, pool2x
from .restormer import swap_blocks

SIG = ACT["sigmoid"]
RELU = ACT["relu"]
LEAKY = 0.2  # refinement.conv2d's LeakyReLU (the epilogue kernels have 0.01 only)
LOOK = 40  # corr_lookup output channels (4 levels x 9 taps + 4 zero)


def _lstm(g: torch.Tensor, h: int, ctx: torch.Tensor, *x: torch.Tensor) -> torch.Tensor:
    """Upstream LSTM with c = h: o * tanh(h * f + i * c~); sigmoid on all gates in the conv epilogue (ctx: context +
    gate biases, candidate rows x 2)."""
    s = conv_ep(g.conv4, torch.cat([h, *x], 1), act=SIG, r=ctx)
    i, f, c2, o = s.chunk(4, 1)
    ct = torch.addcmul(h * f, i, c2.mul(2).sub_(1))  # tanh(c) = 2 sigmoid(2c) - 1
    return o * torch.tanh(ct)


def _motion(e: nn.Module, flow2: torch.Tensor, look: torch.Tensor) -> torch.Tensor:
    """BasicMotionEncoder.forward with the ReLUs in the conv epilogues; [out | flow] without a concatenation copy."""
    cor = conv_ep(e.convc2, conv_ep(e.convc1, look, act=RELU), act=RELU)
    flo = conv_ep(e.convf2, conv_ep(e.convf1, flow2, act=RELU), act=RELU)
    out = conv_ep(e.conv, torch.cat([cor, flo], 1), act=RELU, keep_pad=True)  # 126 outputs padded to 128
    if out.shape[1] == e.conv.out_channels + 2:
        out[:, -2:].copy_(flow2)
        return out
    return torch.cat([out[:, : e.conv.out_channels], flow2], 1).contiguous(memory_format=CL)


def _volume(vc: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """volume_conv (ResidualBlock with InstanceNorm, conv 128 -> 256) on one image (the norm kernels need batch 1)."""
    rb = vc[0]
    y = instance_norm(conv_ep(rb.conv1, x), rb.norm1.eps, RELU)
    y = instance_norm(conv_ep(rb.conv2, y), rb.norm2.eps, RELU)
    return conv_ep(vc[1], torch.relu_(y.add_(x))).contiguous(memory_format=CL)


@torch.no_grad()
def forward_kernels(
    model: nn.Module, k: torch.Tensor, left: torch.Tensor, right: torch.Tensor, iters: int
) -> torch.Tensor:
    """`net.inference.forward` on the kernels (batch 1); returns the disparity [1, H, W] fp32."""
    if left.shape[0] != 1:
        msg = f"DLNR kernels runs batch 1, got {left.shape[0]}"
        raise ValueError(msg)
    ub = model.update_block
    i1 = 2 * (left / 255.0) - 1.0
    i2 = 2 * (right / 255.0) - 1.0
    *cnet_list, x = model.extractor(torch.cat((i1, i2), dim=0).half().contiguous(memory_format=CL))
    x = x.contiguous(memory_format=CL)
    fmaps = [_volume(model.volume_conv, x[j : j + 1]) for j in range(2)]
    del x
    net = [torch.tanh(c[0]).contiguous(memory_format=CL) for c in cnet_list]
    lstms = (ub.lstm08, ub.lstm16, ub.lstm32)
    ctx = [
        conv_ep(conv, torch.relu(c[1]).contiguous(memory_format=CL), g._sc, g._tc)
        for c, conv, g in zip(cnet_list, model.bias_convs, lstms, strict=True)
    ]
    del cnet_list

    # fp16 correlation pyramid (raft kernels): volume [h, w, w] scaled by 1 / sqrt(256), 3 halvings of the last axis
    _, d, h, w = fmaps[0].shape
    p = h * w
    widths = [w >> level for level in range(4)]
    corr = [torch.empty(p * cw, device=left.device, dtype=torch.half) for cw in widths]
    t = (w + 127) // 128
    k.launch("corr_gemm", (t, t, h), (256, 1, 1), fmaps[0], fmaps[1], corr[0], w, d**-0.5)
    del fmaps
    for level in range(3):
        k.launch(
            "corr_pool", *linear(p * widths[level + 1]), corr[level], corr[level + 1], p, *widths[level : level + 2]
        )

    x0 = torch.arange(w, device=left.device, dtype=torch.float32).view(1, 1, 1, w)
    flow = torch.zeros(1, 1, h, w, device=left.device, dtype=torch.float32)  # x-flow (upstream's y-flow stays 0)
    look = torch.empty((1, LOOK, h, w), device=left.device, dtype=torch.half, memory_format=CL)
    flow2 = torch.zeros((1, 2, h, w), device=left.device, dtype=torch.half).contiguous(memory_format=CL)
    for _ in range(iters):
        coords = (x0 + flow).reshape(-1)
        k.launch("corr_lookup", *linear(p), *corr, *widths, coords, look, p)
        flow2[:, :1].copy_(flow)
        net[2] = _lstm(ub.lstm32, net[2], ctx[2], pool2x(net[1]))
        net[1] = _lstm(ub.lstm16, net[1], ctx[1], pool2x(net[0]), interp(net[2], net[1]))
        motion = _motion(ub.encoder, flow2, look)
        net[0] = _lstm(ub.lstm08, net[0], ctx[0], motion, interp(net[1], net[0]))
        del motion
        fh = ub.flow_head
        flow = flow + conv_ep(fh.conv2, conv_ep(fh.conv1, net[0], act=RELU))[:, :1].float()

    del corr, look
    mask = 0.25 * conv_ep(ub.mask[2], conv_ep(ub.mask[0], net[0], act=RELU)).float()
    flow = _upsample(flow, mask, 2**model.args.n_downsample)
    return -_refine(model, flow, i1, i2)[:, 0]


def _bc(b: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """refinement BasicConv: relu(bn(conv(x))) with the BN affine in the epilogue."""
    return conv_ep(b.conv, x, b._s, b._t, RELU if b.relu else 0)


def _c2x(c: nn.Module, x: torch.Tensor, rem: int) -> torch.Tensor:
    """refinement Conv2x (concat = True in every NormalizationRefinement block)."""
    return _bc(c.conv2, torch.cat((_bc(c.conv1, x), rem), 1))


def _refinement(r: nn.Module, low_disp: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """NormalizationRefinement.forward(low_disp, left, right) (full-resolution flow, fp32 images); returns -disp."""
    disp = -low_disp
    w = disp.shape[3]
    warped = disp_warp(right, disp)
    disp = disp / w * 1024
    c1 = r.conv1
    x1 = torch.nn.functional.leaky_relu_(
        conv_ep(c1[0], torch.cat((warped - left, left), 1).half().contiguous(memory_format=CL), c1._s, c1._t), LEAKY
    )
    c2 = r.conv2  # fp32: the normalised disparity reaches ~1000, fp16 steps there are 0.5
    x2 = torch.nn.functional.conv2d(disp, c2[0].weight, None, 1, 1)
    x2 = torch.nn.functional.leaky_relu_(x2 * c2._s.view(1, -1, 1, 1) + c2._t.view(1, -1, 1, 1), LEAKY)
    x = _bc(r.conv_start, torch.cat((x1, x2.half()), 1).contiguous(memory_format=CL))
    rem0 = x
    x = rem1 = _bc(r.conv1a, x)
    x = rem2 = _bc(r.conv2a, x)
    x = rem3 = _bc(r.conv3a, x)
    x = rem4 = _bc(r.conv4a, x)
    x = rem3 = _c2x(r.deconv4a, x, rem3)
    x = rem2 = _c2x(r.deconv3a, x, rem2)
    x = rem1 = _c2x(r.deconv2a, x, rem1)
    x = rem0 = _c2x(r.deconv1a, x, rem0)
    x = rem1 = _c2x(r.conv1b, x, rem1)
    x = rem2 = _c2x(r.conv2b, x, rem2)
    x = rem3 = _c2x(r.conv3b, x, rem3)
    x = _c2x(r.conv4b, x, rem4)
    x = _c2x(r.deconv4b, x, rem3)
    x = _c2x(r.deconv3b, x, rem2)
    x = _c2x(r.deconv2b, x, rem1)
    x = _c2x(r.deconv1b, x, rem0)
    residual = conv_ep(r.final_conv, x).float()
    new = torch.nn.functional.leaky_relu(disp + residual)
    return -(new / 1024 * w)


def _refine(model: nn.Module, flow: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    """net.inference._refine on _refinement: flow + refinement only if every flow value is negative."""
    if bool((flow < 0).all()):
        return flow + _refinement(model.normalizationRefinement, flow, i1, i2)
    return flow


def _prepare_refinement(r: nn.Module, device: torch.device) -> None:
    """BN affines (fp32) of the refinement's conv blocks; its convs on the kernels except the fp32 disparity conv."""
    for m in r.modules():
        if hasattr(m, "use_bn"):  # BasicConv
            m._s, m._t = cnn_kernels.bn_affine(m.bn, m.conv)
    for seq in (r.conv1, r.conv2):
        seq._s, seq._t = cnn_kernels.bn_affine(seq[1], seq[0])
    for name, child in r.named_children():
        if name != "conv2":
            cnn_kernels.prepare_convs(child)


@torch.no_grad()
def prepare_kernels(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Load the kernel libraries for `device` and patch `model` (fp32, on `device`) for forward_kernels.

    Returns:
        The loaded `raft` kernel library.

    Raises:
        BackendUnavailableError: A Restormer block does not fit the fastbb kernels (the reason says which).
        RuntimeError: No conv was routed through the kernels or no Restormer block was found (the module tree changed).
    """
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    k = raft.load(device)
    fastbb.load(device)
    model.eval()
    ub = model.update_block
    for g, conv in zip((ub.lstm08, ub.lstm16, ub.lstm32), model.bias_convs, strict=True):
        hd = g.conv_it.out_channels
        convs = (g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot)
        c4 = nn.Conv2d(g.conv_it.in_channels, 4 * hd, 3, padding=1, bias=False)
        c4.weight.data = torch.cat([c.weight.data for c in convs], 0)
        c4.weight.data[2 * hd : 3 * hd] *= 2  # candidate gate: tanh(c) = 2 sigmoid(2c) - 1
        g.conv4 = c4.to(device)
        sc = torch.ones(4 * hd, device=device)
        sc[2 * hd : 3 * hd] = 2
        # context: conv * sc + (conv bias + gate biases) * sc, fp32 epilogue terms
        g._sc = sc
        g._tc = ((conv.bias.float() + torch.cat([c.bias.float() for c in convs])) * sc).contiguous()
        del g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot
    e = ub.encoder
    c1 = nn.Conv2d(LOOK, e.convc1.out_channels, 1).to(device)  # the lookup's 4 zero channels get zero weights
    w = torch.zeros_like(c1.weight)
    w[:, : e.convc1.in_channels] = e.convc1.weight
    c1.weight.data, c1.bias = w, e.convc1.bias
    e.convc1 = c1

    in_blocks = {id(m) for b in model.extractor.modules() if isinstance(b, TransformerBlock) for m in b.modules()}
    convs = [
        m for m in model.extractor.modules() if isinstance(m, nn.Conv2d) and id(m) not in in_blocks
    ]  # the Restormer blocks use their weights as GEMM matrices: keep them unpadded
    for m in convs:
        cnn_kernels.prepare_convs(m)
    for mod in (ub, model.bias_convs, model.volume_conv):
        cnn_kernels.prepare_convs(mod)
    _prepare_refinement(model.normalizationRefinement, device)
    model.half()
    model.normalizationRefinement.conv2.float()
    if not swap_blocks(model.extractor):  # after half(): the blocks keep fp32 copies of the norm / gate weights
        msg = "DLNR kernels: no Restormer TransformerBlock found"
        raise RuntimeError(msg)
    if not any(getattr(m, "_hacc", False) for m in model.modules()):
        msg = "DLNR kernels: no conv was routed through the fp16-accumulate kernels"
        raise RuntimeError(msg)
    return k
