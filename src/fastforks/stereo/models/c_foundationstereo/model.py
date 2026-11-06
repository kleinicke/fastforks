"""C-FoundationStereo runners (NVIDIA TAO code): weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks._core.checkpoint import Inert, latin1_encode, numeric_scalar
from fastforks.stereo._runner import Forward, Runner

from .net.fast_foundation_stereo.ckpt_utils import remap_state_dict, tao_edgenext_keys
from .net.fast_foundation_stereo.fast_foundation_stereo import FastFoundationStereo
from .net.foundation_stereo.foundation_stereo import FoundationStereo
from .portable import prepare_portable

if TYPE_CHECKING:
    from torch import nn

    from fastforks.stereo._runner import Context


class _Args(dict):
    """The config the network reads: attribute access and `.get`, like TAO's OmegaConf model config."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]` + variant `params`) and load options (card `[options]`)."""

    arch: str  # "fs" (C-FoundationStereo) or "ffs" (C-Fast-FoundationStereo)
    encoder: str  # DINOv2 size ("ffs": only sets the width of the feature slot the ViT filled in FoundationStereo)
    max_disp: int
    valid_iters: int  # the variant's GRU iterations (TAO `valid_iters`)
    iters: int  # option; 0 = `valid_iters`


def _args(p: Params) -> _Args:
    """TAO's model config: `DepthNetModelConfig` defaults (C-FS-S), `experiment_fast_foundation_stereo.yaml` (C-FFS)."""
    common = {
        "encoder": p.encoder,
        "max_disparity": p.max_disp,
        "corr_radius": 4,
        "corr_levels": 2,
        "n_downsample": 2,
        "cv_group": 8,
        "concat_channel": 24,
        "mixed_precision": False,  # TAO's AMP switch; the portable runner sets it on CUDA float16
        "low_memory": False,
        "gwc_feature_normalize": True,
        "stereo_backbone": _Args(use_bn=False, use_clstoken=False),
    }
    if p.arch == "fs":
        return _Args(common, hidden_dims=[128] * 3, n_gru_layers=3, volume_dim=32)
    return _Args(
        common,
        hidden_dims=[128],
        n_gru_layers=1,
        volume_dim=28,
        motion_encoder_widths=[56, 96, 16, 12],
        motion_encoder_final=48,
        gru_hidden=60,
        gru_gating_conv_widths=[100, 168],
        disp_head_input_dim=60,
        disp_head_intermediate=36,
        disp_head_pwconv1_widths=[212, 244],
        mask_widths=[32, 16],
        stem_2_widths=[12, 16],
        spx_2_gru_widths=[16, 12, 16, 24],
        spx_gru_out=9,
        classifier_mid=14,
        cnet_conv04_widths=[60, 48],
        cam_mid_channels=8,
        cost_agg_conv_patch_padding=[0, 0, 0],
    )


class _Tree(dict):
    """Stand-in for a pickled `nn.Module`: keeps only its parameters and persistent buffers, as a flat state dict."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        skip = state.get("_non_persistent_buffers_set", set())
        for k, v in {**state.get("_parameters", {}), **state.get("_buffers", {})}.items():
            if v is not None and k not in skip:
                self[k] = v
        for name, child in state.get("_modules", {}).items():
            if child is not None:
                self.update({f"{name}.{k}": v for k, v in child.items()})


def _parameter(data: torch.Tensor, *_: object) -> torch.Tensor:
    """`torch._utils._rebuild_parameter` without the `nn.Parameter` wrapper (the tensor is all a state dict needs)."""
    return data


# The classes the C-FFS and the research Fast-FoundationStereo checkpoints (whole pickled research `nn.Module`s)
# reference, by module. The research checkpoint names the shared blocks `core.*`, C-FFS `foundation_stereo_ori.*`.
_FFS_MODULES = {
    "core.distill_block": "ForwardHelper PostForwardHelper",
    "core.extractor": "Feature ContextNetSharedBackbone",
    "core.foundation_stereo": "FastFoundationStereo hourglass",
    "core.submodule": "BasicConv BasicConv_IN ChannelAttentionEnhancement Conv2x Conv2x_IN Conv3dNormActReduced "
    "CostVolumeDisparityAttention EdgeNextConvEncoder FeatureAtt FlashAttentionTransformerEncoderLayer "
    "FlashMultiheadAttention PositionalEmbedding ResnetBasicBlock ResnetBasicBlock3D SpatialAttentionExtractor",
    "core.update": "BasicMotionEncoder BasicSelectiveMultiUpdateBlock DispHead RaftConvGRU SelectiveConvGRU",
    "foundation_stereo_ori.extractor": "ContextNetSharedBackbone",
    "foundation_stereo_ori.foundation_stereo": "hourglass",
    "foundation_stereo_ori.submodule": "BasicConv BasicConv_IN ChannelAttentionEnhancement Conv2x "
    "Conv3dNormActReduced CostVolumeDisparityAttention EdgeNextConvEncoder FeatureAtt "
    "FlashAttentionTransformerEncoderLayer FlashMultiheadAttention PositionalEmbedding ResnetBasicBlock3D "
    "SpatialAttentionExtractor",
    "foundation_stereo_ori.update": "BasicMotionEncoder BasicSelectiveMultiUpdateBlock DispHead RaftConvGRU "
    "SelectiveConvGRU",
    "timm.layers.mlp": "Mlp",
    "timm.layers.norm": "LayerNorm2d",
    "timm.models.edgenext": "ConvBlock CrossCovarianceAttn EdgeNeXtStage PositionalEncodingFourier SplitTransposeBlock",
    "torch.nn.modules.activation": "GELU LeakyReLU ReLU Sigmoid",
    "torch.nn.modules.batchnorm": "BatchNorm3d SyncBatchNorm",
    "torch.nn.modules.container": "ModuleDict ModuleList Sequential",
    "torch.nn.modules.conv": "Conv2d Conv3d ConvTranspose2d ConvTranspose3d",
    "torch.nn.modules.dropout": "Dropout",
    "torch.nn.modules.instancenorm": "InstanceNorm2d",
    "torch.nn.modules.linear": "Identity Linear",
    "torch.nn.modules.normalization": "LayerNorm",
    "torch.nn.modules.pooling": "AdaptiveAvgPool2d AdaptiveMaxPool2d",
    "torch.nn.modules.upsampling": "Upsample",
}


def _ffs_extra() -> dict[tuple[str, str], object]:
    """Globals of the C-FFS pickle: every module class -> `_Tree`, its OmegaConf `args` -> `Inert`, plus scalars."""
    scalar = numeric_scalar
    extra: dict[tuple[str, str], object] = {
        (mod, name): _Tree for mod, names in _FFS_MODULES.items() for name in names.split()
    }
    extra.update({("omegaconf.base", "ContainerMetadata"): Inert, ("omegaconf.base", "Metadata"): Inert})
    extra.update({("omegaconf.dictconfig", "DictConfig"): Inert, ("omegaconf.listconfig", "ListConfig"): Inert})
    extra.update({("omegaconf.nodes", "AnyNode"): Inert, ("typing", "Any"): Any})
    extra.update({("collections", "defaultdict"): defaultdict, ("_codecs", "encode"): latin1_encode})
    extra.update({("numpy", "dtype"): np.dtype, ("numpy._core.multiarray", "scalar"): scalar})
    extra[("numpy.core.multiarray", "scalar")] = scalar  # numpy 1 spelling, in the research checkpoint
    extra.update({("torch", "float32"): torch.float32, ("torch._utils", "_rebuild_parameter"): _parameter})
    extra.update({("torch", "CharStorage"): torch.CharStorage, ("torch", "HalfStorage"): torch.HalfStorage})
    return extra


def build(ctx: Context) -> nn.Module:
    """TAO's network with the checkpoint loaded, in eval mode on the CPU."""
    p: Params = ctx.params
    args = _args(p)
    if p.arch == "fs":
        net = FoundationStereo(args)
        # Lightning checkpoint with TAO EdgeNeXt keys.
        net.load_state_dict(tao_edgenext_keys(ctx.state_dict(key="state_dict", strip="model.")))
    else:
        net = FastFoundationStereo(args)
        sd = remap_state_dict(ctx.state_dict(strip="", extra=_ffs_extra()))
        net.load_state_dict(sd, strict=True)
    return net.eval()


def iters(p: Params) -> int:
    """GRU iterations of this load."""
    return p.iters or p.valid_iters


class Portable(Runner):
    """Plain PyTorch: TAO's forward (its fp16 AMP on CUDA), the disparity transformer chunked over its sequences."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = build(ctx).to(ctx.device)
        # TAO's own AMP switch: fp16 autocast inside the network's AUTOCAST blocks, the GRU lookups in fp32
        cast("_Args", self.net.args)["mixed_precision"] = ctx.device.type == "cuda" and ctx.dtype == "float16"
        prepare_portable(self.net, channels_last=ctx.device.type == "cuda")

    def _run(self, net: nn.Module, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return net(left, right, iters=iters(self.ctx.params), test_mode=True)[:, 0].float()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels (see `kernels/`)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import KernelsForward  # noqa: PLC0415 - needs the kernel libraries

        self._forward = KernelsForward(build(ctx), ctx.device)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._forward(left, right, iters(self.ctx.params))}


RUNNERS = {"portable": Portable, "kernels": Kernels}
