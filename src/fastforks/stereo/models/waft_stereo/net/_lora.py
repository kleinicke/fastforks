"""fastforks addition (not upstream code; from our WAFT-Stereo fork, `model/peft_lite.py`, BSD-3 like upstream).

Minimal drop-in for the part of `peft` WAFT uses (LoraConfig, get_peft_model, PeftModel.merge_and_unload).

Why: `import peft` imports `transformers` (peft.auto), ~2.3 s of every cold start, although WAFT only wraps timm /
DINOv2 Linear and Conv2d layers with plain LoRA. This module builds the same module tree and parameter names as
peft 0.21 (`<prefix>.base_model.model.<path>.base_layer.weight`, `.lora_A.default.weight`, `.lora_B.default.weight`),
so the published checkpoints load unchanged, and merges with peft's arithmetic (LoRA delta in the weight's dtype on the
weight's device: Linear B @ A * s; Conv 1x1 (B @ A) * s; Conv kxk conv2d(A^T, B)^T * s; then W += delta).
Unmerged forward = base(x) + B(A(x)) * s (lora_dropout 0 at eval), like peft's vanilla LoRA layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraConfig:
    def __init__(self, r=8, lora_alpha=8, target_modules=None, lora_dropout=0.0, **kw):
        assert not kw.get("use_rslora") and not kw.get("use_dora") and not kw.get("lora_bias"), kw
        assert lora_dropout == 0.0, "dropout only matters for training"
        self.r, self.lora_alpha, self.target_modules = r, lora_alpha, list(target_modules or [])


class _LoraLayer(nn.Module):
    def __init__(self, base_layer, r, alpha):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / r
        if isinstance(base_layer, nn.Linear):
            a = nn.Linear(base_layer.in_features, r, bias=False)
            b = nn.Linear(r, base_layer.out_features, bias=False)
        elif isinstance(base_layer, nn.Conv2d):
            assert base_layer.groups == 1
            a = nn.Conv2d(base_layer.in_channels, r, base_layer.kernel_size, base_layer.stride, base_layer.padding,
                          base_layer.dilation, bias=False)
            b = nn.Conv2d(r, base_layer.out_channels, 1, 1, bias=False)
        else:
            raise TypeError(f"LoRA target must be Linear or Conv2d, got {type(base_layer).__name__}")
        nn.init.zeros_(b.weight)  # adapter starts as identity (peft: A kaiming, B zeros); checkpoints overwrite both
        self.lora_A = nn.ModuleDict({"default": a})
        self.lora_B = nn.ModuleDict({"default": b})

    def forward(self, x, *args, **kwargs):
        return self.base_layer(x, *args, **kwargs) + self.lora_B["default"](self.lora_A["default"](x)) * self.scaling

    def delta_weight(self):
        wa, wb = self.lora_A["default"].weight, self.lora_B["default"].weight
        w = self.base_layer.weight
        cast = w.device.type == "cpu" and w.dtype in (torch.float16, torch.bfloat16)
        if cast:
            wa, wb = wa.float(), wb.float()
        if isinstance(self.base_layer, nn.Linear):
            d = (wb @ wa) * self.scaling
        elif w.shape[2:4] == (1, 1):
            d = (wb.squeeze(3).squeeze(2) @ wa.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * self.scaling
        else:
            d = F.conv2d(wa.transpose(0, 1), wb).transpose(0, 1) * self.scaling
        return d.to(w.dtype) if cast else d

    def merged(self):
        w = self.base_layer.weight
        w.data += self.delta_weight().to(w.dtype)
        return self.base_layer


def _matches(name, targets):
    return any(name == t or name.endswith("." + t) for t in targets)


class LoraModel(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        for name, mod in list(model.named_modules()):
            if name and _matches(name, cfg.target_modules) and isinstance(mod, (nn.Linear, nn.Conv2d)):
                parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
                setattr(parent, name.rpartition(".")[2], _LoraLayer(mod, cfg.r, cfg.lora_alpha))

    def forward(self, *a, **k):
        return self.model(*a, **k)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)


class PeftModel(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.base_model = LoraModel(model, cfg)

    def forward(self, *a, **k):
        return self.base_model(*a, **k)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base_model"), name)

    def merge_and_unload(self):
        m = self.base_model.model
        for name, mod in list(m.named_modules()):
            if isinstance(mod, _LoraLayer):
                parent = m.get_submodule(name.rpartition(".")[0]) if "." in name else m
                setattr(parent, name.rpartition(".")[2], mod.merged())
        return m


def get_peft_model(model, cfg):
    return PeftModel(model, cfg)
