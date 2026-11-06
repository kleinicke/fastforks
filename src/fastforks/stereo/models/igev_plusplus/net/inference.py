"""Inference-only patches for IGEV++ (general: no per-size compilation / tuning).

fold_for_inference(model, half=True, channels_last=True):
  - pure fp16 weights once (no autocast re-casts); BatchNorm statistics (context encoder, MobileNetV2) folded into an
    fp32 per-channel affine that is applied to the fp16 conv output (folding it into the fp16 weights loses precision)
  - InstanceNorm2d / 3d (feature decoder, the whole cost aggregation) with fp32 statistics, layout-preserving
    (PyTorch's norm kernels return NCHW for NHWC / NDHWC input)
  - in-place norm / activation / residual add / feature attention on freshly computed conv outputs (memory; same math)
  - plain PyTorch ops only (exports to ONNX / TensorRT)
"""
import torch
from fastforks._core.patch import base_type, override
import torch.nn as nn
import torch.nn.functional as F

from fastforks.shared.igev_blocks.net import submodule as shared_submodule, extractor
from . import submodule


def _bn_affine(m):
    scale = m.weight.float() * torch.rsqrt(m.running_var.float() + m.eps)
    shift = m.bias.float() - m.running_mean.float() * scale
    return scale, shift


def _bn_forward(self, x):
    shp = (1, -1) + (1,) * (x.dim() - 2)
    return x.mul_(self._s.to(x.dtype).view(shp)).add_(self._t.to(x.dtype).view(shp))


def _bn_act_forward(self, x):  # timm BatchNormAct2d (mobilenet): bn -> drop (identity in eval) -> act
    x = _bn_forward(self, x)
    return self.act(x)


def _in_forward(self, x):
    # InstanceNorm2d / 3d (affine=False): fp32 statistics; fp32 applied in place (keeps NHWC / NDHWC), fp16 computed
    # per element in fp32 (as F.instance_norm does) and stored fp16
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + self.eps)
    if x.dtype == torch.float32:
        return x.sub_(mean).mul_(rstd)
    return ((x.float() - mean) * rstd).to(x.dtype)


def _basic_conv_forward(self, x):  # BasicConv (BN / IN): conv -> norm -> LeakyReLU(0.01), in place
    x = self.conv(x)
    if getattr(self, "use_bn", False):
        x = self.bn(x)
    if getattr(self, "use_in", False):
        x = self.IN(x)
    if self.relu:
        x = F.leaky_relu_(x)
    return x


def _resblock_forward(self, x):
    y = self.relu(self.norm1(self.conv1(x)))
    y = self.relu(self.norm2(self.conv2(y)))
    if self.downsample is not None:
        x = self.downsample(x)
    return y.add_(x).relu_()


def _feature_att_forward(self, cv, feat):  # cost volumes are fresh conv outputs: re-weight in place
    return cv.mul_(torch.sigmoid(self.feat_att(feat)).unsqueeze(2))


@torch.no_grad()  # the folded affines are constants, not part of an autograd graph
def fold_for_inference(model, half=True, channels_last=True):
    model.eval()
    dt = torch.float16 if half else torch.float32
    for m in model.modules():
        t = base_type(m)
        if t in (nn.BatchNorm2d, nn.BatchNorm3d) or t.__name__ == "BatchNormAct2d":
            m._s, m._t = _bn_affine(m)
            override(m, _bn_act_forward if t.__name__ == "BatchNormAct2d" else _bn_forward)
        elif t in (nn.InstanceNorm2d, nn.InstanceNorm3d) and not m.affine and not m.track_running_stats:
            override(m, _in_forward)
        elif t in (submodule.BasicConv, shared_submodule.BasicConv, shared_submodule.BasicConv_IN):
            override(m, _basic_conv_forward)
        elif t is extractor.ResidualBlock:
            override(m, _resblock_forward)
        elif t is submodule.FeatureAtt:
            override(m, _feature_att_forward)
    model.to(dt)
    for m in model.modules():  # BN affine stays fp32 (cast to the activation dtype at use, tiny)
        if hasattr(m, "_s"):
            m._s, m._t = m._s.float(), m._t.float()
    if channels_last:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    model.inf_dtype, model.inf_cl = dt, channels_last
    return model
