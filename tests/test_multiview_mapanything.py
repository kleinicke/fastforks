"""MapAnything stages its two transformer stacks with their original, distinct LayerScale precisions."""

import math
from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks._core.kernels import linear
from fastforks.kernels import map_geometry
from fastforks.multiview.models.mapanything import model


def test_stream_requires_cuda_before_checkpoint_load() -> None:
    ctx = SimpleNamespace(params=model.Params(long_side=518, stream=True), device=torch.device("cpu"))
    with pytest.raises(OptionsError, match="stream=True requires CUDA"):
        model.Portable(ctx)  # ty: ignore[invalid-argument-type]


@pytest.mark.gpu
def test_staged_block_order_and_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "DIM", 8)
    enc = torch.nn.ModuleList([model._Block(), model._Block()])
    info = torch.nn.ModuleList([model._Block()])
    dev = torch.device("cuda")
    for i, block in enumerate([*enc, *info]):
        # Values distinguish blocks after buffer reuse; retain sub-half precision in info-sharing LayerScale.
        block.ls1.gamma.data.fill_(i + 0.1234567)
    host = model._host_blocks(enc, info, torch.float16)
    resident = model._resident(enc, torch.float16, dev, info=False) + model._resident(
        info, torch.float16, dev, info=True
    )
    for _ in range(2):
        for expected, state in zip(resident, host.run(dev), strict=True):
            actual = model._Blk({k.partition(".")[2]: v for k, v in state.items()})
            for got, want in ((actual.g1, expected.g1), (actual.w12, expected.w12), (actual.n2[0], expected.n2[0])):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert resident[0].g1.dtype == torch.float16
    assert resident[-1].g1.dtype == torch.float32
    assert len(host.dev) == 2


@pytest.mark.gpu
def test_dense_projection_preserves_all_channels_and_pixel_tails() -> None:
    # Exactly representable products and sums make every pixel comparable without rounding tolerances.
    rng = torch.Generator(device="cuda").manual_seed(5)
    features = torch.randint(-64, 64, (3, 128, 19, 29), device="cuda", generator=rng).half().div_(16)
    features = features.contiguous(memory_format=torch.channels_last)
    weight = torch.randint(-64, 64, (6, 128), device="cuda", generator=rng).float().div_(32)
    bias = torch.arange(6, device="cuda").float().div_(16)
    result = map_geometry.project(features, weight, bias)
    want = torch.nn.functional.linear(features.permute(0, 2, 3, 1).float(), weight, bias)
    torch.testing.assert_close(result, want, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("planar", [False, True])
def test_geometry_windows_match_torch_with_invalid_points(*, planar: bool) -> None:
    rng = torch.Generator(device="cuda").manual_seed(9)
    points = torch.randn((2, 71, 105, 3), device="cuda", generator=rng)
    # The second view is piecewise planar with a depth step and invalid neighbors on its boundary.
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 71, device="cuda"), torch.linspace(-1, 1, 105, device="cuda"), indexing="ij"
    )
    z = 2 + 0.01 * x + (x > 0.15) * 2
    points[1] = torch.stack((x * z, y * z, z), -1)
    points[:, 10:12, 10, :] = float("nan")
    if planar:
        points = points.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
    depth = points[..., 2]
    mask = torch.rand(depth.shape, device="cuda", generator=rng) > 0.15
    want = model.edge_mask(points, depth, mask)
    actual = map_geometry.edge_mask(points, depth, mask, math.radians(model.EDGE_NORMAL_DEG), model.EDGE_DEPTH_RTOL)
    torch.testing.assert_close(actual, want, atol=0, rtol=0)


@pytest.mark.gpu
def test_angular_cutoff_preserves_float32_arccos_rounding() -> None:
    angle = math.radians(model.EDGE_NORMAL_DEG)
    cosine = math.cos(angle)
    center_bits = torch.tensor(cosine, dtype=torch.float32).view(torch.int32).item()
    dot = (torch.arange(-32, 33, dtype=torch.int32, device="cuda") + int(center_bits)).view(torch.float32)
    normals = torch.zeros((dot.numel(), 1, 3, 4), device="cuda")
    normals[..., 0] = normals[..., 3] = 1
    normals[:, 0, 0, 0] = dot
    normals[:, 0, 0, 1] = (1 - dot.square()).sqrt()
    edges = torch.empty(normals.shape[:-1], dtype=torch.bool, device="cuda")
    map_geometry.load(edges.device).launch(
        "angle_edge", *linear(edges.numel()), normals, edges, 1, 3, edges.numel(), angle, cosine, 1
    )
    torch.testing.assert_close(edges[:, 0, 1], dot.acos() > angle, atol=0, rtol=0)
