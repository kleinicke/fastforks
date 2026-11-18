"""SAM 3 accepts byte RGB directly while preserving the original 8-bit quantization of float and uint16 inputs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fastforks._core import registry
from fastforks._core.image import to_rgb_batch
from fastforks._core.preprocess import scale
from fastforks.promptable_segmentation.models.sam3 import model


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.float32])
def test_sam3_prepare_retains_input_pixels(device, dtype, monkeypatch) -> None:
    card, _ = registry.resolve("promptable_segmentation", "sam3")
    ctx = SimpleNamespace(io=card.io)
    dev = torch.device(device)
    torch.manual_seed(8)
    x = torch.rand(3, 37, 19)
    if dtype != torch.float32:
        x = (x * torch.iinfo(dtype).max).round().to(dtype)
    side = 32
    monkeypatch.setattr(model, "SIZE", side)
    old = to_rgb_batch(x, dev)
    levels = (old * 255).round()
    exact = device == "cpu"
    if exact:
        resized = F.interpolate(levels.byte(), (side, side), mode="bilinear", align_corners=False, antialias=True)
    else:
        resized = (
            F.interpolate(levels, (side, side), mode="bilinear", align_corners=False, antialias=True)
            .round()
            .clamp(0, 255)
        )
    expected = scale(resized.float() / 255, card.io)
    kept = to_rgb_batch(x, dev, keep_uint8=True)
    before = kept.clone()
    got = model.prepare(kept, ctx, exact=exact)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(kept, before, rtol=0, atol=0)
