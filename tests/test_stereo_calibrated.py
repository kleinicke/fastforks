"""Calibrated stereo in one call: `model(left, right, rig=..., views=...)` and `run_many(..., views=...)`.

The runner is a stand-in (constant disparity, or the left-minus-right red channel), so the tests check the geometry
and the plumbing, not a network.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from PIL import Image
from tests.test_api import Dummy

import fastforks
import fastforks.stereo.geometry as g
from fastforks import InputError, OptionsError, stereo
from fastforks._core import registry
from fastforks.stereo._runner import Runner
from fastforks.stereo.calibrated import make_view
from fastforks.stereo.geometry.rig import side_of

pytestmark = pytest.mark.usefixtures("card_min_size")

W, H, F, B = 64, 48, 60.0, 0.1


class Plane(Runner):
    """Disparity `D0` everywhere (a fronto-parallel plane at depth F * B / D0), whatever the images show."""

    Params = Dummy.Params
    D0 = 3.0

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        return {"disparity": torch.full_like(left[:, 0], self.D0)}


@pytest.fixture
def plane(monkeypatch):
    monkeypatch.setattr(
        registry, "import_model", lambda _c: types.SimpleNamespace(RUNNERS={"portable": Plane, "kernels": Plane})
    )
    return stereo.load("s2m2-s", device="cpu")


@pytest.fixture
def dummy(monkeypatch):
    monkeypatch.setattr(
        registry, "import_model", lambda _c: types.SimpleNamespace(RUNNERS={"portable": Dummy, "kernels": Dummy})
    )
    return stereo.load("s2m2-s", device="cpu")


def _rig(dist=(0.0, 0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    cam = g.camera("pinhole-opencv", W, H, F, F, (W - 1) / 2, (H - 1) / 2, *dist)
    vec = np.asarray(rot, float)
    angle = np.linalg.norm(vec)
    k = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    r = np.eye(3) if angle == 0 else np.eye(3) + np.sin(angle) / angle * k + (1 - np.cos(angle)) / angle**2 * k @ k
    return g.StereoRig(cam, cam, r, np.array([-B, 0.0, 0.0]))


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8), rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def test_plain_call_is_unchanged(dummy):
    left, right = _pair()
    assert type(dummy(left, right)).__name__ == "StereoResult"
    with pytest.raises(InputError, match="need a calibration"):
        dummy(left, right, focal=50.0)


def test_right_view_is_the_swapped_flipped_call(dummy):
    """views='right' equals model(flip(right), flip(left)) flipped back, and 'left' stays the plain call."""
    left, right = _pair()
    res = dummy(left, right, views="both")
    swapped = dummy(right[:, ::-1].copy(), left[:, ::-1].copy())
    assert torch.equal(res.right.disparity, swapped.disparity.flip(-1))
    assert torch.equal(res.disparity, dummy(left, right).disparity)
    assert torch.equal(res.right.confidence, swapped.confidence.flip(-1))
    assert res.right.extras.keys() == swapped.extras.keys()
    only = dummy(left, right, views="right")
    assert only.view == "right"
    assert torch.equal(only.disparity, res.right.disparity)
    with pytest.raises(InputError, match="view 'left' was not run"):
        only.left  # noqa: B018


def _same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("rig", [_rig(), _rig((-0.2, 0.05, 0.001, 0.0), (0.01, -0.02, 0.005))])
def test_both_views_equal_two_single_calls_bit_for_bit(dummy, rig):
    left, right = _pair(1)
    both = dummy(left, right, rig=rig, views="both")
    one_left = dummy(left, right, rig=rig)
    one_right = dummy(left, right, rig=rig, views="right")
    _same(both.depth(), one_left.depth())
    assert torch.equal(both.disparity, one_left.disparity)
    _same(both.right.depth(), one_right.depth())
    assert torch.equal(both.right.disparity, one_right.disparity)


def test_one_call_equals_the_manual_pipeline(dummy):
    left, right = _pair(2)
    rig = _rig((-0.2, 0.05, 0.001, 0.0), (0.01, -0.02, 0.005))
    res = dummy(left, right, rig=rig)
    rect = g.rectify(rig)  # the same default rectification as the call
    lt, rt = rect.remap(left, right)
    disparity = dummy(lt, rt).disparity
    manual = rect.unrectify(rect.depth(disparity))
    _same(res.depth(), torch.as_tensor(manual))


def test_defaults_size_focal_and_cache(plane):
    left, right = _pair()
    rig = _rig()
    res = plane(left, right, rig=rig)
    rect = res.rectification
    assert rect.size == (H, W)
    assert rect.f == pytest.approx(rig.left.fx)
    assert plane(left, right, rig=rig).rectification is rect  # maps cached per (calibration, size)
    big = plane(left, right, rig=rig, size=(2 * H, 2 * W), focal=100.0).rectification
    assert big.size == (2 * H, 2 * W)
    assert big.f == 100.0
    np.testing.assert_allclose(big.P1[:2, 2], [(2 * W - 1) / 2, (2 * H - 1) / 2], atol=1e-9)
    assert torch.isnan(res.depth()).float().mean() < 0.2
    assert res.depth().shape == (H, W)


def test_plane_depth_and_point_clouds_in_each_cameras_frame(plane):
    left, right = _pair()
    res = plane(left, right, rig=_rig(), views="both")
    z = F * B / Plane.D0
    for depth in (res.depth(), res.right.depth()):
        d = depth.numpy()
        assert np.isfinite(d).mean() > 0.8
        np.testing.assert_allclose(d[np.isfinite(d)], z, rtol=1e-5)
    for view in ("left", "right"):
        cloud = getattr(res, view).points()
        valid = np.asarray(cloud.valid)
        assert valid.mean() > 0.8
        pts = np.asarray(cloud.points)
        np.testing.assert_allclose(pts[..., 2][valid], z, rtol=1e-5)
        # the optical axis pixel looks straight ahead: x = y = 0 at the centre (pixel centres at integer coordinates)
        mid = valid[H // 2]
        xs = pts[H // 2, :, 0][mid]
        np.testing.assert_allclose(xs, (np.arange(W)[mid] - (W - 1) / 2) * z / F, rtol=1e-4, atol=1e-6)
        assert cloud.colors.shape == (H, W, 3)
    # a rotated rig: 'camera' is the original camera (R^T), 'rectified' keeps the plane at constant z
    rotated = plane(left, right, rig=_rig(rot=(0.0, 0.1, 0.0)), views="both")
    rect = rotated.rectification
    for view, rot in (("left", rect.R1), ("right", rect.R2)):
        rectified = np.asarray(getattr(rotated, view).points(frame="rectified").points)
        camera = np.asarray(getattr(rotated, view).points(frame="camera").points)
        ok = np.isfinite(rectified).all(-1)
        np.testing.assert_allclose(rectified[..., 2][ok], z, rtol=1e-5)
        np.testing.assert_allclose(
            camera.reshape(-1, 3)[ok.ravel()], (rectified.reshape(-1, 3) @ np.asarray(rot))[ok.ravel()], atol=1e-5
        )


def test_unrun_view_raises(plane):
    left, right = _pair()
    res = plane(left, right, rig=_rig())
    assert res.view == "left"
    with pytest.raises(InputError, match="run with views='both' or 'right'"):
        res.right  # noqa: B018
    with pytest.raises(InputError, match="views must be"):
        plane(left, right, rig=_rig(), views="middle")
    with pytest.raises(InputError, match="view must be"):
        side_of("up")


def test_holes_are_nan_with_a_valid_mask(plane):
    """A wide-angle lens at a zoomed-out focal length leaves rectified pixels without a source pixel."""
    cam = g.camera("kb4", W, H, 30.0, 30.0, (W - 1) / 2, (H - 1) / 2, 0.0, 0.0, 0.0, 0.0)
    rig = g.StereoRig(cam, cam, np.eye(3), np.array([-B, 0.0, 0.0]))
    left, right = _pair()
    res = plane(left, right, rig=rig, views="both", focal=12.0)
    for depth, valid in ((res.depth(), res.valid()), (res.right.depth(), res.right.valid())):
        assert not valid.all()
        assert valid.any()
        assert torch.equal(valid, ~torch.isnan(depth))
        assert (depth[valid] > 0).all()
    cloud = res.left.points()
    assert np.isnan(np.asarray(cloud.points)[~np.asarray(cloud.valid)]).all()


def test_euclidean_range(plane):
    left, right = _pair()
    res = plane(left, right, rig=_rig())
    z, rng = res.depth(), res.depth(euclidean=True)
    ys, xs = np.mgrid[0:H, 0:W]
    ray = np.sqrt(1 + ((xs - (W - 1) / 2) / F) ** 2 + ((ys - (H - 1) / 2) / F) ** 2)
    np.testing.assert_allclose(rng.numpy(), z.numpy() * ray, rtol=1e-5)


def test_ply_roundtrip(plane, tmp_path):
    left, right = _pair()
    res = plane(left, right, rig=_rig(), views="both")
    fastforks.save(tmp_path / "left.ply", res)
    n = W * H
    raw = (tmp_path / "left.ply").read_bytes()
    head, _, body = raw.partition(b"end_header\n")
    assert b"element vertex %d" % n in head
    data = np.frombuffer(body, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    np.testing.assert_allclose(data["xyz"].reshape(H, W, 3), res.left.points().points.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(data["rgb"].reshape(H, W, 3), left)  # identity rectification keeps the colours


def test_rectified_rig_and_unsupported_arguments(plane):
    left, right = _pair()
    rig = g.RectifiedRig(F, B, size=(H, W))
    res = plane(left, right, rig=rig, views="both")
    np.testing.assert_allclose(res.depth().numpy(), F * B / Plane.D0, rtol=1e-6)
    np.testing.assert_allclose(res.right.points().points[..., 2], F * B / Plane.D0, rtol=1e-6)
    with pytest.raises(InputError, match="StereoRig"):
        plane(left, right, rig=rig, focal=10.0)


def test_image_of_another_size_rescales_the_calibration(plane):
    left, right = _pair()
    big_l, big_r = np.kron(left, np.ones((2, 2, 1), np.uint8)), np.kron(right, np.ones((2, 2, 1), np.uint8))
    res = plane(big_l, big_r, rig=_rig())
    rect = res.rectification
    assert rect.size == (2 * H, 2 * W)
    assert rect.f == pytest.approx(2 * F)
    np.testing.assert_allclose(rect.P1[:2, 2], [(2 * W - 1) / 2, (2 * H - 1) / 2], atol=1e-9)


def test_run_many_rig_and_views(dummy, tmp_path):
    pairs = [_pair(s) for s in range(3)]
    lefts, rights = [p[0] for p in pairs], [p[1] for p in pairs]
    rig = _rig((-0.2, 0.05, 0.001, 0.0), (0.01, -0.02, 0.005))
    results = list(dummy.run_many(lefts, rights, rig=rig, views="both"))
    for (left, right), r in zip(pairs, results, strict=True):
        single = dummy(left, right, rig=rig, views="both")
        np.testing.assert_array_equal(r.depth(), single.depth().numpy())
        np.testing.assert_array_equal(r.right.depth(), single.right.depth().numpy())
        np.testing.assert_array_equal(r.right.confidence, single.right.confidence.numpy())
    only_right = list(dummy.run_many(lefts, rights, views="right"))
    with pytest.raises(InputError, match="view 'left' was not run"):
        only_right[0].left  # noqa: B018
    with pytest.raises(InputError, match="pass rig="):
        only_right[0].depth()
    assert "PairResult(" in repr(only_right[0])
    np.testing.assert_array_equal(only_right[0].disparity, dummy(*pairs[0], views="right").disparity.numpy())
    # size= and focal= reach the rectification, as in a single call
    sized = next(iter(dummy.run_many(lefts[:1], rights[:1], rig=rig, size=(2 * H, 2 * W), focal=90.0)))
    single = dummy(lefts[0], rights[0], rig=rig, size=(2 * H, 2 * W), focal=90.0)
    np.testing.assert_array_equal(sized.depth(), single.depth().numpy())
    with pytest.raises(InputError, match="need a calibration"):
        list(dummy.run_many(lefts, rights, focal=90.0))
    # files
    list(dummy.run_many(lefts, rights, rig=rig, views="both", save_to=tmp_path, save_format=".npy"))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [f"{i:06d}{s}.npy" for i in range(3) for s in ("", "_right")]
    np.testing.assert_array_equal(np.load(tmp_path / "000001_right.npy"), results[1].right.depth())


def test_batches_with_both_views_and_a_rig(dummy):
    """`[B, 3, H, W]` inputs give `[B, H, W]` maps per view, each item equal to its single call."""
    pairs = [_pair(s) for s in (3, 4)]
    rig = _rig((-0.2, 0.05, 0.001, 0.0), (0.01, -0.02, 0.005))
    lefts = torch.from_numpy(np.stack([p[0] for p in pairs])).permute(0, 3, 1, 2)
    rights = torch.from_numpy(np.stack([p[1] for p in pairs])).permute(0, 3, 1, 2)
    for kw in ({"rig": rig}, {}):
        batch = dummy(lefts, rights, views="both", **kw)
        assert batch.disparity.shape == batch.right.disparity.shape == (2, H, W)
        for i, (left, right) in enumerate(pairs):
            single = dummy(left, right, views="both", **kw)
            assert torch.equal(batch.right.disparity[i], single.right.disparity)
            if kw:
                _same(batch.depth()[i], single.depth())
                _same(batch.right.depth()[i], single.right.depth())
    one = dummy(lefts[:1], rights[:1], rig=rig)
    assert one.depth().shape == (1, H, W)
    _same(one.depth()[0], dummy(*pairs[0], rig=rig).depth())


def test_save_by_suffix_and_accessor_errors(plane, tmp_path):
    left, right = _pair()
    res = plane(left, right, rig=_rig(), views="both")
    fastforks.save(tmp_path / "d.npy", res)  # the result's default map: depth of the first view
    np.testing.assert_array_equal(np.load(tmp_path / "d.npy"), res.depth().numpy())
    res.right.save(tmp_path / "r.npy", "depth")
    np.testing.assert_array_equal(np.load(tmp_path / "r.npy"), res.right.depth().numpy())
    res.right.save(tmp_path / "r.ply")
    assert (tmp_path / "r.ply").stat().st_size > 0
    fastforks.save(tmp_path / "cloud.ply", res.left.points())
    fastforks.save(tmp_path / "plain.npy", res.depth())
    with pytest.raises(InputError, match="no output"):
        res.save(tmp_path / "x.npy", "nope")
    res.save(tmp_path / "d.png")  # KITTI disparity, as the plain result
    res.right.save(tmp_path / "r.png", "disparity")
    png = np.asarray(Image.open(tmp_path / "d.png"))
    assert png.dtype == np.uint16
    np.testing.assert_allclose(png / 256, np.clip(np.nan_to_num(res.disparity.numpy()), 0, 255.996), atol=1 / 256)
    with pytest.raises(OptionsError, match="only holds key='disparity'"):
        res.save(tmp_path / "x.png", "depth")
    plain = plane(left, right, views="both")  # no rig: no depth, no cloud
    with pytest.raises(InputError, match="pass rig="):
        plain.depth()
    fresh = plane(left, right, rig=_rig(), views="both")
    fresh.save(tmp_path / "disp.npy", "disparity")  # no depth is computed for a disparity
    assert not fresh.__dict__.get("_depths")
    with pytest.raises(InputError, match="not found"):
        plane(left, right, rig=tmp_path / "missing.yaml")


@pytest.mark.gpu
def test_both_views_equal_two_single_calls_on_a_max_model():
    """The real kernels: views="both" is bit-identical to the two single calls (B7; kernels are deterministic)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        model = stereo.load("fast-acvnet-plus-generalization", mode="kernels", device="cuda")
    except fastforks.FastforksError as err:
        pytest.skip(f"kernels build not available: {err}")
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    right = np.roll(left, -4, axis=1)
    rig = _rig((-0.1, 0.02, 0.0, 0.0), (0.01, -0.02, 0.005))  # rescaled to the image size by the call
    both = model(left, right, rig=rig, views="both")
    one_left = model(left, right, rig=rig)
    one_right = model(left, right, rig=rig, views="right")
    _same(both.depth(), one_left.depth())
    _same(both.right.depth(), one_right.depth())
    assert torch.equal(both.disparity, one_left.disparity)
    assert torch.equal(both.right.disparity, one_right.disparity)


def test_result_saves_laz(plane, tmp_path):
    laspy = pytest.importorskip("laspy")
    pytest.importorskip("lazrs")
    left, right = _pair()
    res = plane(left, right, rig=_rig(), views="both")
    res.save(tmp_path / "left.laz")
    fastforks.save(tmp_path / "right.las", res.right)
    assert len(laspy.read(tmp_path / "left.laz").x) == W * H
    z = res.left.points().points[..., 2].numpy().ravel()
    assert np.abs(laspy.read(tmp_path / "left.laz").z - z).max() < 1e-4


def test_float64_numpy_matches_float32_on_every_path(dummy):
    """One input conversion: float64 numpy gives the float32 numbers in model(), views='both' and run_many."""
    rig = _rig((-0.2, 0.05, 0.001, 0.0), (0.01, -0.02, 0.005))
    rng = np.random.default_rng(3)
    l64, r64 = rng.random((H, W, 3)), rng.random((H, W, 3))
    l32, r32 = l64.astype(np.float32), r64.astype(np.float32)
    assert torch.equal(dummy(l64, r64).disparity, dummy(l32, r32).disparity)
    for kw in ({"views": "both"}, {"views": "both", "rig": rig}, {"rig": rig}):
        a, b = dummy(l64, r64, **kw), dummy(l32, r32, **kw)
        assert torch.equal(a.disparity, b.disparity)
        if "rig" in kw:
            _same(a.depth(), b.depth())
        for many in (*dummy.run_many([l64], [r64], **kw), *dummy.run_many([l32], [r32], **kw)):
            np.testing.assert_array_equal(many.disparity, b.disparity.numpy())


def test_cached_maps_do_not_go_stale(dummy):
    """`numpy()` and run_many maps are read-only; an in-place edit of a tensor invalidates the cached depth."""
    left, right = _pair()
    rig = _rig()
    view = dummy(left, right, rig=rig).left
    z = view.depth()
    for a in view.numpy().values():
        assert not a.flags.writeable
    assert view.depth() is z  # cached
    z.fill_(1.0)  # an edit of the returned tensor does not stick in the cache
    assert not torch.equal(view.depth(), z)
    disparity = view.disparity.clone()  # model outputs are inference tensors: in-place edits raise outside of it
    edited = make_view("left", {"disparity": disparity}, view.rectification)
    before = edited.depth().clone()
    disparity.mul_(2)
    _same(edited.depth(), before / 2)  # recomputed from the edited disparity
    many = next(iter(dummy.run_many([left], [right], rig=rig)))
    for a in (many.disparity, many.depth(), many.depth(euclidean=True), many.left.confidence):
        assert isinstance(a, np.ndarray)
        assert not a.flags.writeable
    np.testing.assert_array_equal(many.points().points, view.points(colors=False).points.numpy())
    np.testing.assert_array_equal(g.rectify(rig).depth(many.disparity), view.rectification.depth(view.disparity))


def test_valid_on_each_grid(dummy):
    """`valid()` masks depth() (original grid), `valid(grid='rectified')` masks disparity and points()."""
    left, right = _pair()
    for kw in ({"rig": _rig(), "size": (32, 96)}, {"rig": _rig()}, {"views": "both"}):
        res = dummy(left, right, **kw)
        cloud = res.points() if "rig" in kw else None
        rectified = res.valid(grid="rectified")
        assert rectified.shape == res.disparity.shape
        if cloud is not None:
            assert torch.equal(rectified, cloud.valid)
            assert cloud.points[rectified].shape[-1] == 3
            assert res.valid().shape == res.depth().shape == (H, W)
        else:
            assert torch.equal(rectified, res.valid())
    many = next(iter(dummy.run_many([left], [right], rig=_rig(), size=(32, 96))))
    np.testing.assert_array_equal(many.valid(grid="rectified"), many.points().valid)
    assert many.valid().shape == (H, W)
    with pytest.raises(InputError, match="grid must be 'original'"):
        res.valid(grid="camera")


def _np(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _assert_view_equal(a, b):
    """Every output of two views (or results) is the same."""
    for kw in ({}, {"euclidean": True}):
        np.testing.assert_array_equal(_np(a.depth(**kw)), _np(b.depth(**kw)))
    for grid in ("original", "rectified"):
        np.testing.assert_array_equal(_np(a.valid(grid=grid)), _np(b.valid(grid=grid)))
    pa, pb = a.points(), b.points()
    np.testing.assert_array_equal(_np(pa.points), _np(pb.points))
    np.testing.assert_array_equal(_np(pa.valid), _np(pb.valid))
    assert (pa.colors is None) == (pb.colors is None)
    if pa.colors is not None:
        np.testing.assert_array_equal(_np(pa.colors), _np(pb.colors))
    na, nb = a.numpy(), b.numpy()
    assert na.keys() == nb.keys()
    for k in na:
        np.testing.assert_array_equal(na[k], nb[k])


def _assert_saves_equal(a, b, tmp_path):
    for suffix in (".tif", ".ply"):
        pa, pb = tmp_path / f"a{suffix}", tmp_path / f"b{suffix}"
        a.save(pa)
        b.save(pb)
        assert pa.read_bytes() == pb.read_bytes()


def test_views_survive_to_and_cpu(dummy, tmp_path):
    """`.to()` / `.cpu()` of a both-views result and a run_many (numpy) result keep colours, depth and every method."""
    left, right = _pair()
    rig = _rig()
    res = dummy(left, right, rig=rig, views="both")
    res.left.depth(euclidean=True)  # warm the cache
    for moved in (res.to("cpu"), res.cpu(), res.to(torch.device("cpu"))):
        assert type(moved) is type(res)
        assert moved.points().colors is not None
        for name in ("left", "right"):
            _assert_view_equal(getattr(res, name), getattr(moved, name))
        _assert_saves_equal(res, moved, tmp_path)
    view = res.left.to("cpu")
    assert view.depth() is view.depth()  # cache carried and valid
    many = next(iter(dummy.run_many([left], [right], rig=rig, views="both")))
    assert isinstance(many.disparity, np.ndarray)
    for moved in (many.to("cpu"), many.cpu()):
        assert type(moved) is type(many)
        assert isinstance(moved.disparity, np.ndarray)
        assert (moved.index, moved.views, moved.calibrated) == (many.index, many.views, many.calibrated)
        for name in ("left", "right"):
            _assert_view_equal(getattr(many, name), getattr(moved, name))
        _assert_saves_equal(many, moved, tmp_path)


def test_result_without_maps_moves(dummy, tmp_path):
    left, right = _pair()
    Image.fromarray(left).save(tmp_path / "l.png")
    Image.fromarray(right).save(tmp_path / "r.png")
    saved = next(iter(dummy.run_many([tmp_path / "l.png"], [tmp_path / "r.png"], save_to=tmp_path / "o")))
    assert saved.cpu().disparity is None


@pytest.mark.gpu
def test_cuda_view_cpu_equals_original(dummy, tmp_path):
    left, right = _pair()
    res = dummy(left, right, rig=_rig(), views="both")
    res.left.depth()
    cuda = res.to("cuda")
    assert cuda.left.disparity.device.type == "cuda"
    assert cuda.left.points().points.device.type == "cuda"
    host = cuda.cpu()
    assert host.left.disparity.device.type == "cpu"
    for name in ("left", "right"):
        _assert_view_equal(getattr(res, name), getattr(host, name))
    _assert_saves_equal(res, host, tmp_path)
