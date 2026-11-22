"""DepthSplat: the card, the SH rotation and quaternion helpers, and the intrinsics normalisation."""

from __future__ import annotations

import math
import types

import pytest
import torch

import fastforks
from fastforks._core import registry
from fastforks._core.bench import BenchSize, bench_inputs
from fastforks._core.errors import WeightsError
from fastforks.multiview._gaussians import pop_gaussians
from fastforks.multiview.models.depthsplat import model
from fastforks.multiview.models.depthsplat.net.matching import reference_cost_volume


def _rotations(n: int, seed: int = 0) -> torch.Tensor:
    q = torch.linalg.qr(torch.randn(n, 3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))[0]
    return q * torch.linalg.det(q).sign()[:, None, None]


def test_card() -> None:
    card, variant = registry.cards("multiview")["depthsplat-small-re10k"]
    assert "gaussians" in variant.outputs
    assert card.licence.weights == "MIT"
    assert card.options["long_side"] == 256


def test_network_size() -> None:
    assert model.network_size(120, 160, 256) == (192, 256)
    assert model.network_size(256, 256, 256) == (256, 256)
    assert model.network_size(4, 1000, 256) == (16, 256)


def test_sh_rotation_is_a_representation() -> None:
    """`D(R1 R2) = D(R1) D(R2)`, `D` orthogonal, `D_1 = R` and the degree-2 block maps `Y_2(s)` to `Y_2(R s)`."""
    r = _rotations(6)
    d = model.sh_rotation(r)
    torch.testing.assert_close(model.sh_rotation(r[:3] @ r[3:]), d[:3] @ d[3:])
    torch.testing.assert_close(d @ d.transpose(1, 2), torch.eye(model.SH, dtype=torch.float64).expand(6, -1, -1))
    torch.testing.assert_close(d[:, 1:4, 1:4], r)
    s = torch.nn.functional.normalize(torch.randn(6, 3, dtype=torch.float64), dim=1)
    torch.testing.assert_close(d[:, 4:, 4:] @ model._y2(s)[..., None], model._y2((r @ s[..., None])[..., 0])[..., None])
    expected = model._y2((r @ model._S).transpose(1, 2)).transpose(1, 2) @ model._P
    torch.testing.assert_close(d[:, 4:, 4:], expected, rtol=1e-12, atol=1e-12)


def _matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        -1,
    ).unflatten(-1, (3, 3))  # fmt: skip


def test_quaternions() -> None:
    """`matrix_quaternion` inverts the rotation matrix of a quaternion; the product composes rotations."""
    r = torch.cat((_rotations(8), torch.diag_embed(torch.tensor([[1.0, -1, -1], [-1, 1, -1], [-1, -1, 1]]).double())))
    q = model.matrix_quaternion(r)
    torch.testing.assert_close(_matrix(q), r)
    torch.testing.assert_close(_matrix(model.quaternion_product(q[:4], q[4:8])), r[:4] @ r[4:8])


def test_normalized_intrinsics_are_resize_invariant() -> None:
    k = torch.tensor([[[100.0, 0, 79.5], [0, 90, 59.5], [0, 0, 1]]])
    kn = model.normalized_intrinsics(k, 120, 160)
    torch.testing.assert_close(kn[0, :2, 2], torch.tensor([0.5, 0.5]))
    k2 = k.clone()
    k2[:, :2] *= 2
    k2[:, :2, 2] += 0.5  # the same camera at twice the size (pixel centres at integers)
    torch.testing.assert_close(model.normalized_intrinsics(k2, 240, 320), kn)


def test_gaussians_fields() -> None:
    v, h, w = 2, 4, 6
    raw = torch.randn(v, model.RAW_DIM, h, w)
    k = model.normalized_intrinsics(torch.tensor([[50.0, 0, 2.5], [0, 50, 1.5], [0, 0, 1]]).expand(v, 3, 3), h, w)
    c2w = torch.eye(4).repeat(v, 1, 1)
    c2w[:, :3, :3] = _rotations(v).float()
    depth = torch.full((v, h, w), 2.0)
    out = model.gaussians(raw, depth, torch.rand(v, 3, h, w), k, c2w)
    g = pop_gaussians(out)
    assert g is not None
    assert len(g) == v * h * w
    assert g.sh_degree == model.SH_DEGREE
    torch.testing.assert_close(g.rotations.norm(dim=1), torch.ones(v * h * w))
    assert bool((g.rotations[:, 0] >= 0).all())
    # every centre lies at depth 2 along its camera's optical axis
    z = ((g.means.view(v, -1, 3) - c2w[:, None, :3, 3]) @ c2w[:, :3, :3])[..., 2]
    torch.testing.assert_close(z, torch.full_like(z, 2.0))


@pytest.mark.parametrize("deg", [1, 2])
def test_sh_mask(deg: int) -> None:
    """Upstream `sh_mask`: 1 for degree 0, `0.1 * 0.25 ** l` for degree `l` (applied to the raw coefficients)."""
    raw = torch.zeros(1, model.RAW_DIM, 1, 1)
    raw[0, 10 + deg**2] = 1.0  # red, first coefficient of degree `deg`
    k = model.normalized_intrinsics(torch.tensor([[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]]), 1, 1)
    out = model.gaussians(raw, torch.ones(1, 1, 1), torch.full((1, 3, 1, 1), 0.5), k, torch.eye(4)[None])
    assert out["gaussians/harmonics"][0, 0, deg**2].item() == pytest.approx(0.1 * 0.25**deg)
    assert math.isclose(out["gaussians/harmonics"][0, 0, 0].item(), 0.0, abs_tol=1e-7)


class _StubNet(torch.nn.Module):
    """Depth from the scene's scale (near and the baseline), so a missing normalisation shows."""

    def forward(
        self, x: torch.Tensor, _k: torch.Tensor, c2w: torch.Tensor, near: torch.Tensor, far: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, v, _, h, w = x.shape
        base = (c2w[0, :, :3, 3] - c2w[0, :1, :3, 3]).norm(dim=1)[:, None, None]
        depth = (near[0, :, None, None] + far[0, :, None, None] + base) * x[0, :, 0] + 0.1
        raw = torch.zeros(v, model.RAW_DIM, h, w)
        raw[:, 3:6] = x[0, :, :1]
        return depth, torch.ones(v, 1, h, w), raw


def test_scene_scale_normalised() -> None:
    """The scene is scaled to near = `NEAR`: scaling cameras and depth range scales depth, centres and scales."""
    runner = object.__new__(model.Portable)
    runner.ctx = types.SimpleNamespace(params=types.SimpleNamespace(long_side=16), device=torch.device("cpu"))
    runner.net, runner.half = _StubNet(), False
    imgs = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    k = torch.tensor([[20.0, 0, 7.5], [0, 20, 7.5], [0, 0, 1]]).expand(2, 3, 3)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = 0.3
    a = runner(imgs, k, c2w, torch.tensor([2.0, 20.0]))
    c2w[:, :3, 3] *= 500
    b = runner(imgs, k, c2w, torch.tensor([1000.0, 10000.0]))
    for key in ("depth", "gaussians/means", "gaussians/scales"):
        torch.testing.assert_close(b[key], a[key] * 500, rtol=1e-5, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "neighbors", "per_pixel"),
    [(torch.float32, 1, False), (torch.float32, 2, True), (torch.float16, 2, False)],
)
def test_plane_sweep_matches_grid_sample(dtype: torch.dtype, neighbors: int, *, per_pixel: bool) -> None:
    """Compare the full reprojection, zero-padded bilinear sampling and cost over independently chosen neighbours."""
    torch.manual_seed(34)
    b, c, h, w, d = 3, 67, 11, 13, 9
    ref = torch.randn(b, c, h, w, device="cuda", dtype=dtype)
    targets = torch.randn(b, neighbors, c, h, w, device="cuda", dtype=dtype)
    poses = torch.eye(4, device="cuda").repeat(b, neighbors, 1, 1)
    poses[:, :, :3, 3] = torch.randn(b, neighbors, 3, device="cuda") * 0.5
    k = torch.tensor([[8.0, 0, 5.25], [0, 7.0, 4.75], [0, 0, 1]], device="cuda").expand(b, 3, 3)
    depth = torch.linspace(1, 8, d, device="cuda").view(1, d, 1, 1).expand(b, -1, -1, -1)
    if per_pixel:
        depth = depth + torch.rand(b, d, h, w, device="cuda") * 0.01
    with torch.autocast("cuda", dtype=torch.float16):
        want = reference_cost_volume(ref, targets, k, poses, depth, grid_sample_disable_cudnn=True)
        got = model._fused_cost_volume(None, ref, targets, k, poses, depth, _grid_sample_disable_cudnn=True)  # ty: ignore[invalid-argument-type]
    torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-5)


def test_chunked_plane_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial final depth slice and spatially varying candidates produce the same portable cost volume."""
    torch.manual_seed(8)
    b, neighbors, c, h, w, d = 2, 2, 7, 5, 8, 7
    targets = torch.randn(b, neighbors, c, h, w)
    ref = torch.randn(b, c, h, w)
    k = torch.eye(3).expand(b, -1, -1)
    poses = torch.eye(4).repeat(b, neighbors, 1, 1)
    poses[..., 0, 3] = torch.randn(b, neighbors)
    depth = torch.rand(b, d, h, w) + 1
    monkeypatch.setattr(model, "_PORTABLE_WARP_BYTES", 3 * targets.numel() * targets.element_size())
    want = reference_cost_volume(ref, targets, k, poses, depth, grid_sample_disable_cudnn=False)
    got = model._chunked_cost_volume(None, ref, targets, k, poses, depth, grid_sample_disable_cudnn=False)  # ty: ignore[invalid-argument-type]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def _arrays(result: object) -> dict[str, torch.Tensor]:
    out = {k: v for k, v in vars(result).items() if isinstance(v, torch.Tensor)}
    return out | {f"gaussians/{k}": v for k, v in vars(result.gaussians).items() if isinstance(v, torch.Tensor)}


@pytest.mark.gpu
def test_kernels_graphs_match_eager() -> None:
    """The graphed kernels forward: bit-equal to eager for two input sizes; release frees the graphs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        graphed = fastforks.load("depthsplat-small-re10k", mode="kernels", device="cuda")
        eager = fastforks.load("depthsplat-small-re10k", mode="kernels", device="cuda", cuda_graphs=False)
    except WeightsError as e:
        pytest.skip(f"depthsplat weights not available: {e}")
    assert eager._runner.graphs is None
    sizes = [bench_inputs(graphed, BenchSize(h, w, 8)) for h, w in ((120, 160), (96, 200))]
    with torch.inference_mode():
        for args, kw in [sizes[0]] * 5 + [sizes[1]] * 5 + [sizes[0]]:
            got, want = _arrays(graphed(*args, **kw)), _arrays(eager(*args, **kw))
            assert got.keys() == want.keys()
            for k, v in want.items():
                torch.testing.assert_close(got[k], v, rtol=0, atol=0, msg=k)
    graphs = graphed._runner.graphs
    assert len(graphs.graphs) == 2, f"want one graph per input size (failed: {list(graphs._failed)})"
    graphed.release_graphs()
    assert not graphs.graphs
