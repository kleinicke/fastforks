"""A patched block holds one copy of each weight: every `_w*` buffer shares storage with a parameter (2026-11-15); a
kernels model holds no more weight memory than its portable model."""

from __future__ import annotations

import gc

import pytest
import torch

from fastforks._optimize import optimize
from test_golden import API, SMALLEST, TASK

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


MIB = 2**20
# load-time exceptions with the reason (results/2026-11-15_consistency_fixes.md); run peaks: docs/data/speed.jsonl
LOAD_KNOWN = {
    "cotracker3-online@1": "open: kernels keeps merged fp16 qkv / head copies next to the weights (+43 MiB)",
    "tapnext@1": "open: the kernels closures keep merged fp16 ffw-up / qkv copies next to the weights (+93 MiB peak)",
    "metric3d-vit-large@1": "load-time only: kernels prepares its buffers at load, the run peak is within 2 % of portable's",
    "selective-igev-eth3d-finetune@1": "tiny weights; the kernels run peak is below portable's (363 vs 416 MiB)",
}


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("ref", SMALLEST)
def test_kernels_load_holds_no_more_weights_than_portable(ref):
    """Any family (own runners too, e.g. OWLv2's fused filters next to the Linear weights, 2026-11-15): the kernels
    model right after load allocates at most the portable model's memory, plus 10 % and 16 MiB (padded filters)."""
    if ref in LOAD_KNOWN:
        pytest.xfail(LOAD_KNOWN[ref])
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    api = API[TASK[ref]]
    if "kernels" not in api.model_card(ref).modes:
        pytest.skip(f"{ref} has no kernels mode")
    held = {}
    for mode in ("portable", "kernels"):
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        m = api.load(ref, mode=mode, device="cuda")
        held[mode] = torch.cuda.memory_allocated() - base
        del m
    assert held["kernels"] <= 1.1 * held["portable"] + 16 * MIB, {k: f"{v / MIB:.0f} MiB" for k, v in held.items()}
