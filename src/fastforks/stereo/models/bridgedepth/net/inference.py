"""Portable inference path for BridgeDepth (fastforks; plain PyTorch ops only, exports to ONNX / TensorRT).

Port of our BridgeDepth fork's upstream-friendly fp16 path (`bridgedepth/fp16.py` `prepare_fp16`, a74ff35 / ac7df15):

prepare(model, half):
  - the Depth Anything V2 ViT attention as fused `scaled_dot_product_attention` (same maths, no N x N matrix)
  - half: every affine LayerNorm (also ConvNeXt's NCHW `LayerNorm2d`) in the activation dtype under float16 autocast
    (fp16 in / out, fp32 statistics inside the kernel) instead of autocast's fp32 output, which costs an fp32 copy of
    every token stream and a cast back at the next Linear
  - the window-attention masks are cached in the modules (`net.nmp.cached_mask`, always on)
forward(model, left, right, half): the test-mode forward, under float16 autocast if `half` (upstream runs its mono
  branch under float16 autocast already); disparity arithmetic (proposals, sorting / gathering) stays fp32 as under
  plain autocast.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastforks.shared.depth_anything_v2 import use_sdpa_attention

from .convnext import LayerNorm2d


class LN16(nn.Module):
    def __init__(self, ln, chw=False):
        super().__init__()
        self.normalized_shape, self.eps = ln.normalized_shape, ln.eps
        self.weight, self.bias = ln.weight, ln.bias
        self.chw = chw  # ConvNeXt LayerNorm2d (stem / downsampling): NCHW in, normalizes over C

    def forward(self, x):
        if self.chw:
            x = x.permute(0, 2, 3, 1)
        if x.dtype == torch.float32:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            with torch.autocast(x.device.type, enabled=False):
                y = F.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)
        return y.permute(0, 3, 1, 2) if self.chw else y


def _swap_ln(mod):
    for n, c in mod.named_children():
        if isinstance(c, nn.LayerNorm) and c.elementwise_affine:
            setattr(mod, n, LN16(c, chw=isinstance(c, LayerNorm2d)))
        else:
            _swap_ln(c)


def prepare(model, half):
    model.eval()
    use_sdpa_attention(model.mono.pretrained)
    if half:
        _swap_ln(model)
    return model


def forward(model, left, right, half):
    with torch.autocast(left.device.type, dtype=torch.float16, enabled=half):
        return model(left, right).float()
