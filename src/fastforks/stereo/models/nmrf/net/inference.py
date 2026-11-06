"""Portable inference path for NMRF (fastforks; plain PyTorch ops only, exports to ONNX / TensorRT).

Port of our NMRF fork's general path (`nmrf/models/fast.py` `prepare_fast`, branch cuda-kernels) without its custom
kernel (the fused window attention `win_rpe_attn`):

prepare(model, half):
  - the CSWin stripe attention of the label-seed propagation as fused `scaled_dot_product_attention` with the
    self-edge mask as additive bias (in `net/nmp.py`, always on)
  - the window-attention masks are cached in the modules (`net.nmp.cached_mask`, always on)
  - half: every LayerNorm in the activation dtype under float16 autocast (fp16 in / out, fp32 statistics inside the
    kernel) instead of autocast's fp32 output, which costs an fp32 copy of every token stream and a cast back
forward(model, left, right, half): the test-mode forward, under float16 autocast if `half` (our fork's 'amp' path);
  disparity arithmetic (proposals, gathering, sorting) stays fp32 as under plain autocast.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LN16(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.normalized_shape, self.eps = ln.normalized_shape, ln.eps
        self.weight, self.bias = ln.weight, ln.bias

    def forward(self, x):
        if x.dtype == torch.float32:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        with torch.autocast(x.device.type, enabled=False):
            return F.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)


def _swap_ln(mod):
    for n, c in mod.named_children():
        if isinstance(c, nn.LayerNorm) and c.elementwise_affine:
            setattr(mod, n, LN16(c))
        else:
            _swap_ln(c)


def prepare(model, half):
    model.eval()
    if half:
        _swap_ln(model)
    return model


def forward(model, left, right, half):
    with torch.autocast(left.device.type, dtype=torch.float16, enabled=half):
        return model(left, right).float()
