"""The max fusion passes convert as many modules as before `_core.patch.override`: an exact `type(m) is X` check on a
module that an earlier pass already overrode is False (`type(m)` is a per-instance subclass, see `base_type`), so a
later pass silently skips it and kernels runs slower with the same output (the goldens cannot see it).
`EXPECTED` was recorded on main f5927bc and eaac007 (bound-method overrides, no class swap; AANet, CoEx, FS-S and
LAS2-S re-recorded on main e7cf4f8, where the 10-29..10-31 kernel rounds only added overrides; AANet, BridgeDepth,
FS-S, HITNet and NMRF on main 8f373685, likewise only added overrides): the smallest variant of
each family with a kernels backend plus `_VARIANTS`, loaded on CUDA and called once at its minimum input size (some
passes
run on the first call)."""

from __future__ import annotations

import collections
from typing import Any

import pytest
import torch
from torch import nn

from synth import image, pair, size
from test_golden import API, GOLDEN, SMALLEST, TASK

# ref -> (modules, nn.Identity modules, overridden methods as "BaseClass.method": count); recorded on main f5927bc, the
# D1 depth refs (Depth Anything 3, Depth Pro, MoGe, UniK3D) on main eaac007. RAFT-Stereo kernels builds its engine from
# the state dict (no nn.Module), so it has no entry; neither have DA3 metric and mono, whose kernels engine
# (`_FastBackbone`, `_FusedHead`) holds its modules deeper than `held_modules` looks (the other DA3 entries count only
# the camera decoder).
EXPECTED: dict[str, tuple[int, int, dict[str, int]]] = {
    "aanet-sceneflow@1": (  # deform overrides: fused deform_conv kernels (kernel round 10-29)
        668,
        68,
        {
            "BatchNorm2d.forward": 117,
            "Conv2d.forward": 171,
            "DeformConv2d.forward": 15,
            "DeformSimpleBottleneck.forward": 9,
            "DisparityEstimation.forward": 1,
            "ModulatedDeformConv.modulated_deform_conv": 15,
            "SimpleBottleneck.forward": 9,
            "StereoDRNetRefinement.forward": 2,
        },
    ),
    "bridgedepth-sf@1": (
        1229,
        104,
        {
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "KernelsWindowAttention.attn": 10,
            "MemEffAttention.forward": 24,
        },
    ),
    "c-fast-foundationstereo@1": (
        594,
        79,
        {
            "Conv.forward": 32,
            "Conv2d.forward": 76,
            "Conv3d.forward": 40,
            "ConvTranspose2d.forward": 5,
            "ConvTranspose3d.forward": 5,
            "CostVolumeDisparityAttention.forward": 2,
            "ResnetBasicBlock.forward": 3,
            "ResnetBasicBlock3D.forward": 1,
        },
    ),
    "cas-gwcnet-c-sceneflow@1": (692, 124, {"Conv2d.forward": 64, "Conv3d.forward": 60, "ConvTranspose3d.forward": 12}),
    "cfnet-sceneflow@1": (666, 111, {"Conv2d.forward": 50, "Conv3d.forward": 70, "ConvTranspose3d.forward": 12}),
    "coex-sceneflow@1": (  # CostVolume override: stereo_ops cost-volume kernel (kernel round 10-29)
        526,
        118,
        {
            "BasicConv.forward": 38,
            "BatchNorm2d.forward": 20,
            "BatchNorm3d.forward": 19,
            "BatchNormAct2d.forward": 48,
            "Conv2d.forward": 75,
            "Conv3d.forward": 16,
            "ConvTranspose2d.forward": 5,
            "ConvTranspose3d.forward": 3,
            "CostVolume.forward": 1,
        },
    ),
    "crestereo-eth3d@1": (126, 0, {"Conv2d.forward": 38, "LinearAttention.feature_map": 2}),
    "da3-giant-1.1@1": (11, 0, {}),
    "da3-large-1.1@1": (11, 0, {}),
    "da3nested-giant-large-1.1@1": (11, 0, {}),
    "defom-stereo-vitl-sceneflow@1": (
        838,
        113,
        {
            "BasicEncoder.forward": 1,
            "Conv2d.forward": 100,
            "ConvBlock.forward": 4,
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "InstanceNorm2d.forward": 18,
            "MultiBasicEncoder.forward": 1,
            "ResidualBlock.forward": 6,
        },
    ),
    "depth-anything-v2-s@1": (209, 26, {}),
    "depth-pro@1": (1249, 150, {}),
    "distill-any-depth-s@1": (209, 26, {}),  # the family smallest since 2026-11-10; same structure as DAv2-S
    "dinov2-vits14@1": (211, 26, {}),
    "dinov3-convnext-tiny@1": (
        0,
        0,
        {},
    ),  # fvm_conv / fvm_seg kernels: tensors only, no nn.Module (recorded 2026-11-01)
    "dinov3-vits16@1": (125, 0, {}),
    "dlnr-middlebury@1": (236, 0, {"Conv2d.forward": 53, "ConvTranspose2d.forward": 8}),
    "fast-acvnet-generalization@1": (
        500,
        121,
        {
            "BasicConv.forward": 40,
            "BatchNorm2d.forward": 22,
            "BatchNorm3d.forward": 19,
            "BatchNormAct2d.forward": 48,
            "Conv2d.forward": 79,
            "Conv3d.forward": 17,
            "ConvTranspose2d.forward": 5,
            "ConvTranspose3d.forward": 4,
        },
    ),
    "foundationstereo-vit-s@1": (
        999,
        125,
        {
            "BasicConv.forward": 20,
            "Conv2d.forward": 143,
            "Conv3d.forward": 34,
            "Conv3dNormActReduced.forward": 9,
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "FeatureAtt.forward": 5,
            "InstanceNorm2d.forward": 18,
            "ResidualBlock.forward": 14,
            "ResnetBasicBlock3D.forward": 3,
        },
    ),
    "gwcnet-g-sceneflow@1": (477, 87, {"Conv2d.forward": 55, "Conv3d.forward": 30, "ConvTranspose3d.forward": 6}),
    "great-igev-eth3d@1": (  # recorded on attempt/igevpp-max (new family overrides)
        955,
        113,
        {
            "BasicConv.forward": 27,
            "BasicConv_IN.forward": 15,
            "BatchNorm2d.forward": 40,
            "BatchNorm3d.forward": 20,
            "BatchNormAct2d.forward": 48,
            "Conv2d.forward": 286,
            "Conv3d.forward": 17,
            "ConvTranspose2d.forward": 7,
            "ConvTranspose3d.forward": 3,
            "ExcitiveAttentionHourglassEncoder.disparity": 1,
            "ExcitiveAttentionVolume.cost_volume": 1,
            "FeedForwardLayer.forward": 16,
            "InstanceNorm2d.forward": 19,
            "LayerNorm.forward": 62,
            "MultiBasicEncoder.forward": 1,
            "OutlookAttentionLayer.forward": 4,
            "ResidualBlock.forward": 14,
            "VolumeTransformerBlock.forward": 6,
        },
    ),
    "hitnet-eth3d@1": (166, 0, {"LevelInit.init_cost": 5, "LevelProp.tile_sad": 5, "Refinement.refine_sad": 3}),
    "igev-plusplus-eth3d@1": (  # recorded on attempt/igevpp-max
        788,
        113,
        {
            "BasicConv.forward": 78,
            "BasicConv_IN.forward": 7,
            "BatchNorm2d.forward": 33,
            "BatchNormAct2d.forward": 48,
            "Conv2d.forward": 171,
            "Conv3d.forward": 42,
            "ConvTranspose2d.forward": 7,
            "ConvTranspose3d.forward": 9,
            "FeatureAtt.forward": 18,
            "InstanceNorm2d.forward": 40,
            "MultiBasicEncoder.forward": 1,
            "ResidualBlock.forward": 14,
        },
    ),
    "igev-stereo-kitti12@1": (
        579,
        113,
        {
            "BasicConv.forward": 24,
            "BasicConv_IN.forward": 13,
            "BatchNorm2d.forward": 41,
            "BatchNorm3d.forward": 16,
            "BatchNormAct2d.forward": 48,
            "Conv2d.forward": 136,
            "Conv3d.forward": 14,
            "ConvTranspose2d.forward": 7,
            "ConvTranspose3d.forward": 3,
            "InstanceNorm2d.forward": 16,
            "MultiBasicEncoder.forward": 1,
            "ResidualBlock.forward": 14,
        },
    ),
    "las2-s@1": (  # AttentionModule + Conv2d overrides of the kernel round (10-29)
        372,
        68,
        {
            "AttentionModule.forward": 3,
            "BatchNorm2d.forward": 4,
            "Conv2d.forward": 105,
            "ConvTranspose2d.forward": 7,
            "InstanceNorm2d.forward": 3,
        },
    ),
    "lotus-normal-d-v1-1@1": (855, 48, {}),
    "marigold-depth-v1-1@1": (852, 48, {}),
    "marigold-normals-lcm-v0-1@1": (852, 48, {}),
    "metric3d-vit-large@1": (
        581,
        52,
        {"Conv2dLoRA.lora_dropout": 62, "ConvTranspose2dLoRA.lora_dropout": 1, "LoRALinear.lora_dropout": 8},
    ),
    "midas-dpt-beit-large-512@1": (
        330,
        1,
        {"Attention.forward": 24, "DPTDepthModel.forward_transformer": 1, "Interpolate.interp": 1},
    ),
    "mocha-v1-sceneflow@1": (  # BasicConv 24 -> 53: feature.BasicConv (2-D) fused too (attempt/stereo-depth-speed)
        1342,
        68,
        {
            "BasicConv.forward": 53,
            "BasicConv_IN.forward": 16,
            "BatchNorm2d.forward": 72,
            "BatchNorm3d.forward": 16,
            "BatchNormAct2d.forward": 195,
            "Conv2d.forward": 427,
            "Conv3d.forward": 17,
            "ConvTranspose2d.forward": 15,
            "ConvTranspose3d.forward": 3,
            "InstanceNorm2d.forward": 16,
            "MultiBasicEncoder.forward": 1,
            "ResidualBlock.forward": 14,
        },
    ),
    "moge-2-vitl-normal@1": (419, 50, {}),
    "moge-2-vits-normal@1": (215, 26, {}),
    "moge-3-vitl@1": (497, 50, {}),
    "monster-mix-all@1": (
        981,
        77,
        {
            "BasicConv.forward": 24,
            "BasicConv_IN.forward": 8,
            "BatchNorm2d.forward": 41,
            "BatchNorm3d.forward": 16,
            "Conv2d.forward": 202,
            "Conv3d.forward": 14,
            "ConvTranspose2d.forward": 19,
            "ConvTranspose3d.forward": 3,
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "InstanceNorm2d.forward": 17,
        },
    ),
    "monster-plusplus-mix-all@1": (
        981,
        77,
        {
            "BasicConv.forward": 24,
            "BasicConv_IN.forward": 8,
            "BatchNorm2d.forward": 41,
            "BatchNorm3d.forward": 16,
            "Conv2d.forward": 202,
            "Conv3d.forward": 14,
            "ConvTranspose2d.forward": 19,
            "ConvTranspose3d.forward": 3,
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "InstanceNorm2d.forward": 17,
        },
    ),
    "nmrf-sceneflow@1": (388, 0, {"KernelsWindowAttention.attn": 10}),
    "psmnet-sceneflow@1": (475, 85, {"Conv2d.forward": 61, "Conv3d.forward": 22, "ConvTranspose3d.forward": 6}),
    "rt-monster-plusplus-mix-all@1": (
        759,
        53,
        {
            "BatchNorm2d.forward": 9,
            "BatchNorm3d.forward": 4,
            "Conv2d.forward": 262,
            "Conv3d.forward": 5,
            "ConvTranspose2d.forward": 12,
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "InstanceNorm2d.forward": 20,
            "MemEffAttention.forward": 12,
            "Monster.normalize": 1,
        },
    ),
    "s2m2-s@1": (
        879,
        0,
        {
            "AvgPool2d.forward": 12,
            "BasicAttnBlock.forward": 6,
            "CNNEncoder.forward": 1,
            "Conv2d.forward": 199,
            "ConvBlock2D.forward": 18,
            "ConvGRU.forward": 1,
            "DispInit.forward": 1,
            "FeatureFusion.forward": 16,
            "GlobalAttnBlock.forward": 12,
            "GlobalRefiner.forward": 1,
            "LocalRefiner.forward": 1,
            "MRT.forward": 1,
            "S2M2.forward_fast": 1,
            "StackedMRT.forward": 1,
            "Unet.forward": 3,
            "UpsampleMask1x.forward": 1,
            "UpsampleMask4x.forward": 1,
        },
    ),
    "selective-igev-eth3d-finetune@1": (
        604,
        113,
        {
            "BasicConv.forward": 24,
            "BasicConv_IN.forward": 13,
            "BatchNorm2d.forward": 41,
            "BatchNorm3d.forward": 16,
            "BatchNormAct2d.forward": 48,
            "Conv2d.forward": 148,
            "Conv3d.forward": 14,
            "ConvTranspose2d.forward": 7,
            "ConvTranspose3d.forward": 3,
            "InstanceNorm2d.forward": 16,
            "MultiBasicEncoder.forward": 1,
            "ResidualBlock.forward": 14,
        },
    ),
    "unidepth-v2-vitl14@1": (614, 80, {"UniDepthV2.stacking_fn": 1}),
    "unik3d-vitl@1": (482, 50, {}),
    "waft-stereo-s@1": (
        1376,
        285,
        {
            "Conv2d.forward": 173,
            "ConvTranspose2d.forward": 12,
            "DinoVisionTransformer.get_intermediate_layers": 1,
            "InstanceNorm2d.forward": 92,
            "Mlp.forward": 11,
            "resconv.forward": 46,
        },
    ),
    "zoedepth-nk@1": (
        486,
        1,
        {
            "Attention.forward": 24,
            "DPTDepthModel.forward_transformer": 1,
            "Interpolate.interp": 1,
            "LogBinomial.act": 2,
            "MidasCore.prep": 1,
            "TransformerEncoderLayer.activation": 4,
        },
    ),
}


def held_modules(obj: Any, depth: int = 3, seen: set[int] | None = None) -> list[nn.Module]:
    """The modules an object holds, through attributes, lists, tuples and dicts."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth < 0:
        return []
    seen.add(id(obj))
    if isinstance(obj, nn.Module):
        return [obj]
    if isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        items = list(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, torch.Tensor)):
        items = list(vars(obj).values())
    else:
        return []
    return [m for v in items for m in held_modules(v, depth - 1, seen)]


def _methods(m: nn.Module) -> list[str]:
    """Methods set on this instance: `__dict__` callables (bound methods, closures) and `override` methods."""
    own = [k for k, v in vars(m).items() if callable(v) and not isinstance(v, (nn.Module, torch.Tensor, type))]
    return sorted({*own, *(k for k, _ in getattr(type(m), "_fastforks_methods", ()))})


def structure(model: Any) -> tuple[int, int, dict[str, int]]:
    """(modules, nn.Identity modules, overridden methods) of everything the loaded model's runner holds."""
    mods = {id(s): s for r in held_modules(model._runner) for s in r.modules()}
    base = {k: getattr(type(m), "_fastforks_base", type(m)) for k, m in mods.items()}
    over = collections.Counter(f"{base[k].__name__}.{n}" for k, m in mods.items() for n in _methods(m))
    return len(mods), sum(b is nn.Identity for b in base.values()), dict(sorted(over.items()))


def load_and_call(ref: str) -> Any:
    task = TASK[ref]
    m = API[task].load(ref, mode="kernels", device="cuda")
    hw = size(m.variant.io.min_size)
    m(*pair(*hw)) if task in {"stereo", "flow", "matching"} else m(image(*hw))
    return m


# Variants whose kernels passes see other modules than their family's smallest (other backbone, head or nesting).
_VARIANTS = {
    "da3-giant-1.1@1",
    "da3-large-1.1@1",
    "da3mono-large@1",
    "da3nested-giant-large-1.1@1",
    "moge-2-vitl-normal@1",
    "moge-3-vitl@1",
}
MAX_REFS = sorted(r for r in {*SMALLEST, *_VARIANTS} if r in GOLDEN and "kernels" in API[TASK[r]].model_card(r).modes)


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("ref", MAX_REFS)
def test_kernels_passes_convert_as_before(ref):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if ref not in EXPECTED:
        pytest.skip(f"{ref}: no recorded structure")
    assert structure(load_and_call(ref)) == EXPECTED[ref]
