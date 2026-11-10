"""The traced (ONNX / TensorRT) cost volumes of (Fast-)FoundationStereo match the per-disparity loops they replace."""

from unittest import mock

import pytest
import torch

from fastforks.stereo.models.c_foundationstereo.net.foundation_stereo import utils


def _loops(a: torch.Tensor, b: torch.Tensor, d: int, groups: int, *, normalize: bool) -> tuple[torch.Tensor, ...]:
    """The upstream loops: disparity i pairs ref column w with target column w - i, zero where w < i."""
    n, c, h, w = a.shape
    gwc = torch.zeros(n, groups, d, h, w, dtype=a.dtype)
    cat = torch.zeros(n, 2 * c, d, h, w, dtype=a.dtype)
    for i in range(d):
        gwc[:, :, i, :, i:] = utils.groupwise_correlation(a[..., i:], b[..., : w - i], groups, normalize=normalize)
        cat[:, :c, i] = a
        cat[:, c:, i, :, i:] = b[..., : w - i]
    return gwc, cat


@pytest.mark.parametrize(
    ("shape", "d", "groups", "normalize", "dtype"),
    [
        ((1, 32, 37, 50), 12, 8, True, torch.float32),  # H not a multiple of the row block
        ((2, 16, 70, 41), 40, 4, False, torch.float32),  # batch 2, disparities close to W
        ((1, 32, 20, 30), 29, 8, True, torch.float16),  # the largest disparity count the traced path takes
    ],
)
def test_traced_volumes_match_loops(shape, d, groups, normalize, dtype) -> None:
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    gwc_ref, cat_ref = _loops(a, b, d, groups, normalize=normalize)
    with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        gwc = utils.build_gwc_volume(a, b, d, groups, normalize=normalize)
        cat = utils.build_concat_volume(a, b, d)
    assert gwc.dtype == dtype
    assert gwc.shape == gwc_ref.shape
    # fp16: one unit in the last place below 1 (4.9e-4); the sum order differs between the two forms and torch versions
    torch.testing.assert_close(gwc.float(), gwc_ref.float(), atol=5e-4 if dtype == torch.float16 else 1e-5, rtol=1e-5)
    assert torch.equal(cat, cat_ref)
