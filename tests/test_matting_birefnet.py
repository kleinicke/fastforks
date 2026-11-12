"""BiRefNet: the card and the vendored network's fastforks changes (einops-free patches, the merged offset conv)."""

from __future__ import annotations

import torch

from fastforks import matting
from fastforks._core import registry
from fastforks.matting.models.birefnet import model
from fastforks.matting.models.birefnet.net.birefnet import image2patches
from fastforks.matting.models.birefnet.net.modules.aspp import _ASPPModuleDeformable


def test_card() -> None:
    for name, backbone in (("birefnet", "swin_v1_l"), ("birefnet-lite", "swin_v1_t")):
        card, variant = registry.cards("matting")[name]
        assert variant.outputs == ("alpha",)
        assert variant.params["backbone"] == backbone
        assert card.licence.code == card.licence.weights == "MIT"
        assert name in matting.list_models()


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
