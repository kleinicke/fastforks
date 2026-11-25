"""SAM 3 mask pyramid kernels match the fp16 stages used by the image detector."""

import pytest
import torch
import torch.nn.functional as F

from fastforks.kernels import sam3_fpn
from fastforks.promptable_segmentation.models.sam3 import model as sam3_model
from fastforks.promptable_segmentation.models.sam3.fast import pixel_decode
from fastforks.promptable_segmentation.models.sam3.net.sam3 import PixelDecoder

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("layout", ["packed", "nchw", "strided", "unaligned", "mixed"])
def test_nearest_add_matches_pyramid(layout: str) -> None:
    b, c, h, w = 2, 16, 5, 9
    coarse = torch.randn((b, c, h, w), device="cuda", dtype=torch.float16)
    skip = torch.randn((b, c, h * 2, w * 2), device="cuda", dtype=torch.float16)
    if layout in ("packed", "unaligned", "mixed"):
        coarse = coarse.contiguous(memory_format=torch.channels_last)
    if layout in ("packed", "unaligned"):
        skip = skip.contiguous(memory_format=torch.channels_last)
    if layout == "unaligned":
        storage = torch.empty(coarse.numel() + 1, device=coarse.device, dtype=coarse.dtype)
        coarse = storage[1:].as_strided(coarse.shape, coarse.stride()).copy_(coarse)
    if layout == "strided":
        coarse = coarse.transpose(0, 1).contiguous().transpose(0, 1)
        skip = skip.transpose(2, 3).contiguous().transpose(2, 3)
    expected = F.interpolate(coarse, scale_factor=2, mode="nearest") + skip
    got = sam3_fpn.nearest_add(coarse, skip)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", [torch.contiguous_format, torch.channels_last])
def test_affine_relu_uses_conv_storage(fmt: torch.memory_format) -> None:
    x = torch.randn((2, 16, 7, 11), device="cuda", dtype=torch.float16).contiguous(memory_format=fmt)
    scale, shift = torch.randn(2, 2, 16, device=x.device, dtype=x.dtype).unbind()
    expected = torch.addcmul(shift[..., None, None], x, scale[..., None, None]).relu_()
    got = sam3_fpn.affine_relu_(x, scale, shift)
    assert got.data_ptr() == x.data_ptr()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_two_stage_decoder_matches_upstream() -> None:
    torch.manual_seed(17)
    model = PixelDecoder(2).cuda().half().eval()
    for norm in model.norms:
        norm.weight.data.uniform_(0.1, 2)
        norm.bias.data.uniform_(-1, 1)
    feats = [
        torch.randn(2, 256, h, w, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        for h, w in ((20, 36), (10, 18), (5, 9))
    ]
    with torch.inference_mode():
        expected = model(feats)
        got = pixel_decode(model, feats)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("h", "w", "oh", "ow"),
    [(288, 288, 720, 1280), (19, 37, 8, 13), (1, 31, 13, 1), (2, 2, 3, 5), (9, 11, 9, 11), (13, 11, 17, 23)],
)
@pytest.mark.parametrize("pattern", ["strided", "zero_boundary"])
def test_bilinear_mask_matches_float_threshold(h, w, oh, ow, pattern) -> None:
    torch.manual_seed(28)
    if pattern == "strided":
        src = torch.randn((3, w, 2, h), device="cuda").permute(0, 2, 3, 1)
    else:
        # Interpolation can land on or very near a zero crossing, including at the edges.
        y = torch.arange(h, device="cuda")[:, None]
        x = torch.arange(w, device="cuda")[None, :]
        src = ((1 - 2 * ((x + y) % 2)).float())[None, None]
    want = F.interpolate(src, (oh, ow), mode="bilinear", align_corners=False) > 0
    got = sam3_fpn.bilinear_mask(src, (oh, ow))
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_portable_binary_masks_keep_group_boundaries_exact(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """Several groups of noncontiguous mask logits match the ungrouped threshold."""
    size = 11, 19
    src = torch.randint(-1, 2, (5, 3, 7, 11), device="cuda").to(dtype).transpose(-1, -2)
    monkeypatch.setattr(sam3_model, "_MASK_RESIZE_PIXELS", 2 * 3 * size[0] * size[1])
    runner = object.__new__(sam3_model.Portable)
    expected = F.interpolate(src, size, mode="bilinear", align_corners=False) > 0
    assert torch.equal(runner.binary_masks(src, size), expected)
