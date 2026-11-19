"""Direct CUDA cost-volume checks at strip boundaries and non-integral disparities."""

import pytest
import torch

from fastforks.stereo.models.cfnet import model as cfnet_model
from fastforks.stereo.models.cfnet.model import _cuda_coarse, _cuda_stage, _volume, stage_volume
from fastforks.stereo.models.cfnet.net.cfnet import cfnet


def _features() -> dict[str, torch.Tensor]:
    return {
        key: torch.randn(2, channels, 7, 11, device="cuda", dtype=torch.float16).contiguous(
            memory_format=torch.channels_last
        )
        for key, channels in (("gw4", 16), ("concat_feature4", 3), ("gw5", 16), ("concat_feature5", 3))
    }


@pytest.mark.gpu
def test_cfnet_volumes() -> None:
    left, right = _features(), _features()
    for level in (4, 5):
        result = _cuda_coarse(left, right, level, 7, 4)
        expected = _volume(left, right, level, 7, 4).to(result.dtype)
        torch.testing.assert_close(result, expected)
    net = cfnet(64, use_concat_volume=True)
    samples = torch.rand(2, 5, 3, 11, device="cuda") * 19 - 4
    got = _cuda_stage(net, (left, right), samples, level=4, a=2, groups=4)
    expected = stage_volume(net, (left, right), samples, level=4, a=2, groups=4)
    torch.testing.assert_close(got, expected)


@pytest.mark.gpu
def test_cfnet_portable_stage_matches_export_path(monkeypatch: pytest.MonkeyPatch) -> None:
    left, right = _features(), _features()
    net = cfnet(64, use_concat_volume=True)
    samples = torch.linspace(-4, 15, 2 * 5 * 3 * 11, device="cuda").view(2, 5, 3, 11)
    portable = stage_volume(net, (left, right), samples, level=4, a=2, groups=4)
    monkeypatch.setattr(cfnet_model, "in_onnx_export", lambda: True)
    expected = stage_volume(net, (left, right), samples, level=4, a=2, groups=4)
    torch.testing.assert_close(portable, expected, rtol=0, atol=0)


@pytest.mark.gpu
@torch.inference_mode()
def test_cfnet_cuda_halo_preserves_refinement_at_strip_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    height, width = 384, 32

    def features() -> dict[str, torch.Tensor]:
        return {
            name: torch.rand(1, channels, height, width, device="cuda", dtype=torch.float16, generator=generator)
            for name, channels in (("gw2", 40), ("concat_feature2", 6))
        }

    net = cfnet(64, use_concat_volume=True).eval().cuda().half()
    data = features(), features()
    lower = torch.linspace(0, 8, height, device="cuda").view(1, 1, height, 1).expand(1, 1, height, width)
    upper = lower + 6
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    actual, _ = cfnet_model._stage(net, "s2", data, lower, upper, with_std=False, volume=stage_volume)
    monkeypatch.setattr(cfnet_model, "_CUDA_HALO", cfnet_model.HALO)
    expected, _ = cfnet_model._stage(net, "s2", data, lower, upper, with_std=False, volume=stage_volume)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
