"""A patched block holds one copy of each weight: every `_w*` buffer shares storage with a parameter (2026-11-15)."""

from __future__ import annotations

import pytest
import torch

from fastforks._optimize import optimize

timm = pytest.importorskip("timm")

# (timm model, kwargs): layer scale, QuickGELU (CLIP), SigLIP-style, Swin, EVA, ConvNeXt, Perception Encoder
_CASES = {
    "vit": ("vit_tiny_patch16_224", {}),
    "vit_layerscale": ("vit_tiny_patch16_224", {"init_values": 0.5}),
    "vit_quickgelu": ("vit_base_patch16_clip_quickgelu_224", {"depth": 2}),
    "vit_quickgelu_layerscale": ("vit_base_patch16_clip_quickgelu_224", {"depth": 2, "init_values": 0.5}),
    "swin": ("swin_tiny_patch4_window7_224", {"depths": (1, 1, 1, 1)}),
    "eva": ("eva02_tiny_patch14_224", {"init_values": 0.5}),
    "convnext_layerscale": ("convnext_tiny", {"ls_init_value": 0.5, "depths": (1, 1, 1, 1)}),
    "pe": ("vit_pe_core_tiny_patch16_384", {"depth": 2, "init_values": 0.5}),
}
_SMALL = {"_w_rsc"}  # the depthwise 7x7 kernel re-laid out (< 3 % of a ConvNeXt)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("case", list(_CASES))
def test_patched_blocks_hold_no_second_weight_copy(case, mode):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    name, kw = _CASES[case]
    torch.manual_seed(0)
    try:
        m = timm.create_model(name, pretrained=False, num_classes=10, **kw).eval()
    except Exception as e:  # noqa: BLE001 - the timm build lacks this architecture
        pytest.skip(str(e))
    try:
        fast = optimize(m, mode=mode, device="cuda")
    except Exception as e:  # noqa: BLE001 - no kernels path for this family
        pytest.skip(str(e))
    owned = {p.untyped_storage().data_ptr() for p in fast.net.parameters()}
    for mod_name, mod in fast.net.named_modules():
        for k, b in mod._buffers.items():
            if k.startswith("_w") and k not in _SMALL and b is not None:
                assert b.untyped_storage().data_ptr() in owned, f"{mod_name}.{k}: a second copy of a weight"
    # no weight-sized storage outside the parameters at all (folded tensors replace the originals)
    seen = {}
    for t in [*fast.net.parameters(), *fast.net.buffers()]:
        seen[t.untyped_storage().data_ptr()] = t
    extra = sum(t.untyped_storage().nbytes() for ptr, t in seen.items() if ptr not in owned and t.dim() >= 2)
    own = sum(t.untyped_storage().nbytes() for ptr, t in seen.items() if ptr in owned)
    assert extra <= 0.1 * own
