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


@pytest.mark.gpu
@pytest.mark.parametrize(("h", "w", "side"), [(72, 118, 1008), (2160, 3840, 1008), (17, 17, 17), (1, 23, 19)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_cuda_antialias_writes_the_same_encoder_input(h, w, side, dtype) -> None:
    from fastforks.kernels.sam3_resize import normalized_rgb  # noqa: PLC0415

    torch.manual_seed(19)
    # Non-contiguous CHW views of RGB frames, including unquantized float inputs.
    if dtype == torch.uint8:
        image = torch.randint(0, 256, (1, h, w, 3), device="cuda", dtype=dtype).permute(0, 3, 1, 2)
        levels = image.float()
    else:
        image = torch.rand(1, h, w, 3, device="cuda").permute(0, 3, 1, 2)
        levels = (image * 255).round()
    resized = F.interpolate(levels, (side, side), mode="bilinear", antialias=True, align_corners=False)
    expected = ((resized.round().clamp(0, 255) / 255 - 0.5) / 0.5).half()
    got = normalized_rgb(image, side)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
