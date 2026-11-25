"""MVSplat camera geometry."""

import pytest
import torch

from fastforks._core import registry
from fastforks._core.cards import require_available
from fastforks.multiview._runner import project_harmonics
from fastforks.multiview.models.mvsplat.model import _poses


def test_onnx_and_tensorrt_are_available() -> None:
    _, variant = registry.resolve("multiview", "mvsplat-re10k@1")
    require_available(variant, "onnx")
    require_available(variant, "tensorrt")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_relative_poses_match_matrix_inverse(dtype: torch.dtype) -> None:
    c2w = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    c2w[:, :3, :3] = torch.tensor([[[1.0, 0.1, 0.0], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]]] * 3, dtype=dtype)
    c2w[:, :3, 3] = torch.tensor([[0, 0, 0], [1, 2, 3], [-2, 4, 1]], dtype=dtype)
    actual = _poses(c2w)
    assert len(actual) == 2
    for offset, pose in enumerate(actual, 1):
        expected = torch.linalg.inv(c2w).roll(-offset, 0) @ c2w
        torch.testing.assert_close(pose, expected)


@pytest.mark.parametrize("channels", [3, 5, 7, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_harmonic_projection_matches_matrix_product(channels: int, dtype: torch.dtype) -> None:
    gen = torch.Generator().manual_seed(0)
    harmonics = torch.randn(2, 64, channels, generator=gen, dtype=dtype)
    inverse = torch.randn(64, channels, generator=gen, dtype=dtype)
    expected = harmonics.transpose(1, 2) @ inverse
    torch.testing.assert_close(project_harmonics(harmonics, inverse), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("views", [2, 3])
def test_fused_plane_sweep(views: int) -> None:
    from fastforks.multiview.models.mvsplat.model import correlation, cuda_accumulate  # noqa: PLC0415

    torch.manual_seed(25)
    feat = torch.randn(views, 37, 9, 13, device="cuda", dtype=torch.float16)
    intrinsics = torch.eye(3, device="cuda").repeat(views, 1, 1)
    intrinsics[:, 0, 0] = 10.5
    intrinsics[:, 1, 1] = 8.75
    camera = torch.eye(4, device="cuda").repeat(views, 1, 1)
    camera[:, 0, 3] = torch.arange(views, device="cuda") * 0.17
    disp = torch.linspace(0.05, 2.3, 15, device="cuda").expand(views, -1)
    with torch.autocast("cuda", dtype=torch.float16):
        expected = correlation(feat, intrinsics, camera, disp, dtype=torch.float16)
        actual = correlation(feat, intrinsics, camera, disp, dtype=torch.float16, fused=cuda_accumulate)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)


@pytest.mark.parametrize("views", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_metal_plane_sweep(views: int, dtype: torch.dtype) -> None:
    """`mvsplat_metal.accumulate` against the portable `grid_sample` sweep (in float64 on the CPU)."""
    if not torch.backends.mps.is_available():
        pytest.skip("needs PyTorch MPS")
    import functools  # noqa: PLC0415

    from fastforks.kernels import mvsplat_metal  # noqa: PLC0415 - device-specific library
    from fastforks.multiview.models.mvsplat.model import correlation  # noqa: PLC0415

    torch.manual_seed(25)
    feat = torch.randn(views, 36, 9, 13).to(dtype)
    intrinsics = torch.eye(3).repeat(views, 1, 1)
    intrinsics[:, 0, 0] = 10.5
    intrinsics[:, 1, 1] = 8.75
    camera = torch.eye(4).repeat(views, 1, 1)
    camera[:, 0, 3] = torch.arange(views) * 0.17
    disp = torch.linspace(0.05, 2.3, 15).expand(views, -1)
    want = correlation(feat.double(), intrinsics, camera, disp, dtype=torch.float64).float()
    fused = functools.partial(mvsplat_metal.accumulate, mvsplat_metal.load(torch.device("mps")))
    mps = [t.to("mps") for t in (feat, intrinsics, camera, disp)]
    got = correlation(*mps, dtype=dtype, fused=fused).cpu()
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
