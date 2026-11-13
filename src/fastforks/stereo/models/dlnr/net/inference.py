"""Portable inference path for DLNR (fastforks; plain PyTorch ops only, exports to ONNX / TensorRT).

Port of our fork's general path (`core/dlnr_fast.py`, branch cuda-kernels) without its custom kernels:

prepare(model, half): the 4 gate convs of each LSTM merged into one conv (order i, f, c~, o = the order of the
  context biases from `bias_convs`), their biases folded into the bias convs (the per-level context is computed once);
  half: fp16 weights except the NormalizationRefinement (fp32: upstream runs it outside autocast); channels-last
  weights on CUDA.
forward(model, left, right, iters): == upstream `DLNR.forward(test_mode=True)`: fp32 all-pairs correlation pyramid
  with the gather lookup (`corr.CorrBlockGather1D`), fp32 flow state, the Restormer extractor's friendly NHWC blocks
  (`extractor.TransformerBlock`), LSTM cell input = the previous hidden state (upstream's `netC` and `net_h` are the
  same list, so every cell state is overwritten by the hidden state), convex upsampling once after the last
  iteration, upstream's refinement rule (refine only if all upsampled flow is < 0; here per sample, upstream checks
  the whole batch). Returns the disparity [B, H, W] (= -flow).
"""
import torch
from fastforks._core.tracing import in_onnx_export
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlockGather1D
from .update import interp, pool2x


def _tracing():
    return torch.compiler.is_exporting() or torch.compiler.is_compiling() or in_onnx_export()


@torch.no_grad()
def prepare(model, half):
    model.eval()
    ub = model.update_block
    for g, bias_conv in zip((ub.lstm08, ub.lstm16, ub.lstm32), model.bias_convs):
        convs = (g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot)
        gates = nn.Conv2d(g.conv_it.in_channels, 4 * g.conv_it.out_channels, 3, padding=1, bias=False)
        gates.weight.copy_(torch.cat([c.weight for c in convs]))
        bias_conv.bias.add_(torch.cat([c.bias for c in convs]))
        g.gates = gates.to(g.conv_it.weight.device)
        del g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot
    if half:
        model.half()
        model.normalizationRefinement.float()
    if next(model.parameters()).is_cuda:
        model.to(memory_format=torch.channels_last)
    return model


def _lstm(g, h, ctx, *x):
    """Upstream LSTM with c = h: o * tanh(h * f + i * c~), gates from one conv over [h | x] + context."""
    i, f, c, o = (g.gates(torch.cat([h, *x], dim=1)) + ctx).chunk(4, dim=1)
    return torch.sigmoid(o) * torch.tanh(h * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(c))


def _upsample(flow, mask, factor):
    """Upstream `DLNR.upsample_flow` (convex combination of 3 x 3 neighbours) for the x-flow."""
    N, _, H, W = flow.shape
    mask = torch.softmax(mask.view(N, 1, 9, factor, factor, H, W), dim=2)
    up = F.unfold(factor * flow, [3, 3], padding=1).view(N, 1, 9, 1, 1, H, W)
    up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
    return up.reshape(N, 1, factor * H, factor * W)


def _refine(model, flow, i1, i2):
    """Upstream: flow + NormalizationRefinement(flow, ...) if every flow value is negative (fp32)."""
    ref = model.normalizationRefinement
    neg = flow.amax(dim=(1, 2, 3)) < 0

    def refined(flow, i1, i2, neg):
        return torch.where(neg.view(-1, 1, 1, 1), flow + ref(flow, i1, i2), flow)

    def unchanged(flow, i1, i2, neg):
        return flow.clone()

    if _tracing():
        return torch.cond(neg.any(), refined, unchanged, (flow, i1, i2, neg))
    return refined(flow, i1, i2, neg) if bool(neg.any()) else flow


def forward(model, left, right, iters):
    a = model.args
    ub = model.update_block
    dt = ub.lstm08.gates.weight.dtype
    mf = torch.channels_last if left.is_cuda else torch.contiguous_format
    i1 = 2 * (left / 255.0) - 1.0
    i2 = 2 * (right / 255.0) - 1.0
    *cnet_list, x = model.extractor(torch.cat((i1, i2), dim=0).to(dt).contiguous(memory_format=mf))
    fmap1, fmap2 = model.volume_conv(x).float().split(dim=0, split_size=x.shape[0] // 2)
    net = [torch.tanh(c[0]) for c in cnet_list]
    ctx = [conv(torch.relu(c[1])) for c, conv in zip(cnet_list, model.bias_convs)]
    corr_fn = CorrBlockGather1D(fmap1, fmap2, radius=a.corr_radius, num_levels=a.corr_levels)

    b, _, h, w = net[0].shape
    x0 = torch.arange(w, device=left.device, dtype=torch.float32).view(1, 1, 1, w)
    flow = torch.zeros(b, 1, h, w, device=left.device, dtype=torch.float32)  # x-flow (upstream's y-flow stays 0)
    for _ in range(iters):
        corr = corr_fn(x0 + flow).to(dt)
        flow2 = torch.cat([flow, torch.zeros_like(flow)], dim=1).to(dt).contiguous(memory_format=mf)
        net[2] = _lstm(ub.lstm32, net[2], ctx[2], pool2x(net[1]))
        net[1] = _lstm(ub.lstm16, net[1], ctx[1], pool2x(net[0]), interp(net[2], net[1]))
        motion = ub.encoder(flow2, corr)
        net[0] = _lstm(ub.lstm08, net[0], ctx[0], motion, interp(net[1], net[0]))
        flow = flow + ub.flow_head(net[0])[:, :1].float()

    mask = 0.25 * ub.mask(net[0]).float()
    flow = _upsample(flow, mask, 2**a.n_downsample)
    return -_refine(model, flow, i1, i2)[:, 0]
