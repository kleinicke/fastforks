"""BiRefNet: the card and the vendored network's fastforks changes (einops-free patches, the merged offset conv)."""

from __future__ import annotations

import sys
import weakref

import pytest
import torch
import torch.nn.functional as F

from fastforks import matting
from fastforks._core import registry
from fastforks._core.upload import capture_uploads
from fastforks.matting.models.birefnet import model
from fastforks.matting.models.birefnet.net.birefnet import image2patches
from fastforks.matting.models.birefnet.net.modules.aspp import _ASPPModuleDeformable
from fastforks.shared.swin_v1 import SwinTransformer, fast
from fastforks.shared.swin_v1.fast import _geometry, geometry_stamp


def test_card() -> None:
    for name, backbone in (("birefnet", "swin_v1_l"), ("birefnet-lite", "swin_v1_t")):
        card, variant = registry.cards("matting")[name]
        assert variant.outputs == ("alpha",)
        assert variant.params["backbone"] == backbone
        assert card.licence.code == card.licence.weights == "MIT"
        assert name in matting.list_models()
    card, variant = registry.cards("matting")["rmbg-2.0"]  # BRIA's BiRefNet Swin-L weights, non-commercial
    assert variant.params["backbone"] == "swin_v1_l"
    assert card.variant_licence(variant).weights == "CC-BY-NC-4.0"
    assert card.status(variant).commercial == "no"


def test_image2patches() -> None:
    """`b c (hg h) (wg w) -> b (c hg wg) h w` as einops: channel-major, then grid row, then grid column."""
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    got = image2patches(x, 2, 3, transformation="b c (hg h) (wg w) -> b (c hg wg) h w")
    want = torch.stack(
        [x[:, c, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] for c in range(3) for i in range(2) for j in range(3)], 1
    )
    assert torch.equal(got, want)


def test_merged_offset_conv() -> None:
    """The kernels runner's one conv = upstream's offset conv, then its modulator conv (channel order of `om`)."""
    torch.manual_seed(0)
    m = _ASPPModuleDeformable(16, 16, 3, padding=1).eval()
    dc = m.atrous_conv
    x = torch.randn(1, 16, 8, 8)
    want = torch.cat((dc.offset_conv(x), dc.modulator_conv(x)), 1)
    model._prepare_aspp(m)
    torch.testing.assert_close(m._om(x), want)  # ty: ignore[call-non-callable]
    assert m._w.shape == (9 * 16, 16)


def test_swin_sdpa_mask_stride(monkeypatch) -> None:
    """Shared swin_v1 (BiRefNet, RMBG-2.0, BEN2, InSPyReNet): the bias mask has stride 1 in its last dim.

    CUDA's fused SDPA kernels reject any other mask layout and fall back to the math path (2-10x slower).
    """
    seen = []
    sdpa = F.scaled_dot_product_attention

    def spy(*args, attn_mask=None, **kwargs):
        seen.append(attn_mask)
        return sdpa(*args, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    net = SwinTransformer(embed_dim=16, depths=[2, 2], num_heads=[2, 4], window_size=4, out_indices=(0, 1))
    net.eval()  # upstream's `train` returns None
    with torch.no_grad():
        net(torch.randn(1, 3, 64, 64))
    assert len(seen) == 4  # a plain and a shifted-window block per stage
    assert all(m is not None and m.stride(-1) == 1 for m in seen), [m.stride() for m in seen if m is not None]


def test_swin_capture_holds_evicted_geometry() -> None:
    """A recording retains exactly the cached indices and mask it read."""
    net = SwinTransformer(embed_dim=16, depths=[2, 2], num_heads=[2, 4], window_size=4, out_indices=(0, 1))
    net.eval()
    fast(net)
    x = torch.randn(1, 3, 44, 52)
    with torch.inference_mode(), capture_uploads() as inputs:
        net(x)
    block = net.layers[0].blocks[1]
    geometry = _geometry(block.H, block.W, block.window_size, block.shift_size, x.device, x.dtype)
    assert geometry.padding.numel()
    assert geometry.mask is not None
    assert any(v is block.attn._fast_bias for v in inputs)
    refs = [weakref.ref(t) for t in (geometry.to_windows, geometry.to_image, geometry.padding, geometry.mask)]
    del geometry
    _geometry.cache_clear()
    assert all(ref() is not None and any(ref() is v for v in inputs) for ref in refs)
    inputs.clear()
    assert all(ref() is None for ref in refs)


def test_swin_capture_cannot_build_a_missing_cuda_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A late cache miss can retry with a checked eager forward before recording."""
    monkeypatch.setattr(sys, "platform", "linux")
    _geometry.cache_clear()
    before = geometry_stamp()
    with capture_uploads(), pytest.raises(RuntimeError, match="Swin geometry was evicted"):
        _geometry(9, 13, 7, 3, torch.device("cuda:0"), torch.float16)
    assert geometry_stamp() == before
