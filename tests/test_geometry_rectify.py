"""Rectification, remap, depth and point clouds of `fastforks.stereo.geometry` against OpenCV and closed forms."""

import numpy as np
import pytest
import torch

from fastforks import InputError
from fastforks.stereo import geometry as g
from fastforks.stereo.geometry.rectify import _ALIVE, _rotation_from_vector, _scale


@pytest.fixture
def cv2():
    """OpenCV, the oracle of the comparison tests: only they skip without it."""
    return pytest.importorskip("cv2")


W, H = 640, 480
K1 = np.array([[500.0, 0, 320.5], [0, 505, 241.2], [0, 0, 1]])
K2 = np.array([[510.0, 0, 315.0], [0, 512, 236.0], [0, 0, 1]])
STRONG = (np.array([-0.28, 0.09, 0.001, -0.0005, -0.01]), np.array([-0.27, 0.08, -0.0007, 0.0004, -0.012]))
MILD = (np.array([-0.04, 0.01, 0.0005, -0.0002, 0.0]), np.array([-0.03, 0.008, -0.0003, 0.0002, 0.0]))
R = _rotation_from_vector(np.array([0.01, -0.02, 0.005]))
T = np.array([-0.12, 0.003, -0.002])


def _rig(dist, rot=R, t=T):
    left = g.camera("pinhole-opencv", W, H, K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2], *dist[0])
    right = g.camera("pinhole-opencv", W, H, K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2], *dist[1])
    return g.StereoRig(left, right, rot, t)


def _smooth_image(channels=None, dtype=np.uint8, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w]
    img = 0.5 + 0.4 * np.sin(xx / 13.0) * np.cos(yy / 17.0)
    if channels:
        img = np.stack([img, img[::-1], 1 - img][:channels], -1)
    scale = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    return (img * scale).astype(dtype)


@pytest.mark.parametrize(
    ("dist", "alpha", "size", "p_tol"),
    [
        # OpenCV stops undistorting the image corners after 5 iterations; with strong barrel distortion that moves
        # its principal point / focal length by up to ~0.3 px against the converged values used here.
        (STRONG, -1, None, 0.05),
        (STRONG, 0.0, None, 0.05),
        (STRONG, 0.5, None, 0.25),
        (STRONG, 1.0, None, 0.4),
        (STRONG, 0.0, (600, 800), 0.05),
        (MILD, 0.0, None, 1e-6),
        (MILD, 1.0, None, 1e-6),
    ],
)
def test_matches_cv2_stereo_rectify(cv2, dist, alpha, size, p_tol):
    rect = g.rectify(_rig(dist), alpha=alpha, size=size)
    # the oracle is cv2 on the cameras resized the same way: f s, (c + 0.5) s - 0.5 (cv2's own newImageSize rescales
    # the result of the source grid, which is a different, inconsistent choice)
    nh, nw = size or (H, W)
    s = np.array([nw / W, nh / H])
    ks = []
    for k0 in (K1, K2):
        k = k0.copy()
        k[:2] *= s[:, None]
        k[:2, 2] += 0.5 * (s - 1)
        ks.append(k)
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        ks[0], dist[0], ks[1], dist[1], (nw, nh), R, T.reshape(3, 1), alpha=alpha
    )
    np.testing.assert_allclose(rect.R1, r1, atol=1e-12)
    np.testing.assert_allclose(rect.R2, r2, atol=1e-12)
    np.testing.assert_allclose(rect.P1, p1, atol=p_tol)
    np.testing.assert_allclose(rect.P2, p2, atol=p_tol * abs(T[0]) * 2 + p_tol)
    np.testing.assert_allclose(rect.Q, q, atol=p_tol)
    for ours, theirs in ((rect.roi_left, roi1), (rect.roi_right, roi2)):
        assert np.abs(np.subtract(ours, theirs)).max() <= 1
    assert rect.size == (size or (H, W))


@pytest.mark.parametrize("size", [(600, 800), (240, 320), (480, 1000)])
@pytest.mark.parametrize(
    ("alpha", "focal"), [(None, None), (-1, None), (0.0, None), (0.5, None), (1.0, None), (None, 450.0)]
)
def test_rectify_at_a_size_equals_rectify_of_the_resized_rig(alpha, size, focal):
    """The scale is applied once: rectifying to `size` equals rectifying the rig that was resized to `size`."""
    rig = _rig(STRONG)
    a = g.rectify(rig, alpha=alpha, size=size, focal=focal)
    b = g.rectify(rig.resized(size), alpha=alpha, focal=focal)
    np.testing.assert_allclose(a.P1, b.P1, atol=0.05)
    np.testing.assert_allclose(a.Q, b.Q, atol=0.05)
    for ours, theirs in ((a.roi_left, b.roi_left), (a.roi_right, b.roi_right)):
        assert np.abs(np.subtract(ours, theirs)).max() <= 1


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_maps_match_cv2(cv2, alpha):
    rect = g.rectify(_rig(STRONG), alpha=alpha)
    maps, valid = rect.maps()
    for i, (k, d, rot, p) in enumerate(((K1, STRONG[0], rect.R1, rect.P1), (K2, STRONG[1], rect.R2, rect.P2))):
        mx, my = cv2.initUndistortRectifyMap(k, d, rot, p, (W, H), cv2.CV_32FC1)
        v = valid[i].numpy()
        assert v.mean() > (0.99 if alpha == 0 else 0.7)
        np.testing.assert_allclose(maps[i, ..., 0].numpy()[v], mx[v], atol=1e-3)
        np.testing.assert_allclose(maps[i, ..., 1].numpy()[v], my[v], atol=1e-3)
        assert (maps[i][~valid[i]] == -1).all()
    assert rect.maps()[0] is maps  # built once per device


@pytest.mark.parametrize(("channels", "dtype"), [(None, np.uint8), (3, np.uint8), (None, np.uint16)])
def test_remap_matches_cv2_remap(cv2, channels, dtype):
    rect = g.rectify(_rig(STRONG), alpha=0.5)
    left, right = _smooth_image(channels, dtype), _smooth_image(channels, dtype)[:, ::-1].copy()
    out_left, out_right, valid_left, valid_right = rect.remap(left, right, return_valid=True)
    maps, _ = rect.maps()
    for i, (src, out, valid) in enumerate(((left, out_left, valid_left), (right, out_right, valid_right))):
        expected = cv2.remap(src, maps[i, ..., 0].numpy(), maps[i, ..., 1].numpy(), cv2.INTER_LINEAR)
        assert out.dtype == dtype
        assert out.shape == src.shape
        assert out.flags.c_contiguous
        assert np.abs(out.astype(np.int64) - expected.astype(np.int64)).max() <= 1  # cv2 quantizes the maps
        assert not out[~valid].any()


def test_remap_layouts_and_kinds():
    rect = g.rectify(_rig(MILD))
    img = _smooth_image(3, np.float32)
    ref, _ = rect.remap(img, img)
    chw = torch.from_numpy(img).permute(2, 0, 1)
    out, _ = rect.remap(chw, chw)
    assert isinstance(out, torch.Tensor)
    assert out.shape == chw.shape
    np.testing.assert_allclose(out.permute(1, 2, 0).numpy(), ref, atol=1e-6)
    batch = torch.stack([chw, chw.flip(-1)])
    out_b, _ = rect.remap(batch, batch)
    assert out_b.shape == batch.shape
    np.testing.assert_allclose(out_b[0].numpy(), out.numpy(), atol=1e-6)
    bhwc = np.stack([img, img[:, ::-1]])  # numpy batches are [B, H, W, C], as StereoModel takes them
    out_bhwc, _, valid_l, _ = rect.remap(bhwc, bhwc, return_valid=True)
    assert out_bhwc.shape == bhwc.shape
    assert isinstance(valid_l, np.ndarray)
    np.testing.assert_allclose(out_bhwc[0], ref, atol=1e-6)
    np.testing.assert_allclose(out_bhwc[1], out_b[1].permute(1, 2, 0).numpy(), atol=1e-6)
    gray, _ = rect.remap(torch.from_numpy(img[..., 0]), torch.from_numpy(img[..., 0]))
    assert gray.shape == (H, W)
    with pytest.raises(InputError, match=r"left image is 480x320 \(height x width\)"):
        rect.remap(img[:, :320], img)
    with pytest.raises(InputError, match="left and right images differ"):
        rect.remap(img, img[..., :2])


def test_fisheye_matches_cv2_fisheye(cv2):
    d1, d2 = np.array([0.05, -0.01, 0.002, -0.0003]), np.array([0.04, -0.008, 0.001, -0.0002])
    k1, k2 = (
        np.array([[300.0, 0, 319], [0, 300, 239], [0, 0, 1]]),
        np.array([[302.0, 0, 321], [0, 301, 241], [0, 0, 1]]),
    )
    left = g.KB4(W, H, 300.0, 300.0, 319.0, 239.0, k1=d1[0], k2=d1[1], k3=d1[2], k4=d1[3])
    right = g.KB4(W, H, 302.0, 301.0, 321.0, 241.0, k1=d2[0], k2=d2[1], k3=d2[2], k4=d2[3])
    rect = g.rectify(g.StereoRig(left, right, R, T), alpha=0.0)
    r1, r2, *_ = cv2.fisheye.stereoRectify(k1, d1, k2, d2, (W, H), R, T.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY)
    np.testing.assert_allclose(rect.R1, r1, atol=1e-12)  # K_new differs by design (OpenCV uses `balance`)
    np.testing.assert_allclose(rect.R2, r2, atol=1e-12)
    maps, valid = rect.maps()
    for i, (k, d, rot, p) in enumerate(((k1, d1, rect.R1, rect.P1), (k2, d2, rect.R2, rect.P2))):
        mx, my = cv2.fisheye.initUndistortRectifyMap(k, d, rot, p, (W, H), cv2.CV_32FC1)
        v = valid[i].numpy()
        assert v[2:-2, 2:-2].all()  # alpha = 0 fits the valid region as cv2.stereoRectify does, to about a pixel
        np.testing.assert_allclose(maps[i, ..., 0].numpy()[v], mx[v], atol=1e-3)
        np.testing.assert_allclose(maps[i, ..., 1].numpy()[v], my[v], atol=1e-3)


@pytest.mark.parametrize(
    "cam", [g.KB4(W, H, 110.0, 110.0, 319.5, 239.5), g.DoubleSphere(W, H, 160, 160, 319.5, 239.5, xi=-0.2, alpha=0.6)]
)
def test_wide_fov_rectifies_with_the_fov_guard(cam):
    rig = g.StereoRig(cam, cam, _rotation_from_vector(np.array([0.0, 0.05, 0.0])), np.array([-0.1, 0.0, 0.0]))
    narrow = g.rectify(rig, alpha=0.0, fov_max=90.0)
    wide = g.rectify(rig, alpha=0.0, fov_max=150.0)
    for rect in (narrow, wide):
        assert np.isfinite(rect.P1).all()
        assert rect.roi_left[2] > W // 2
        assert rect.maps()[1][0].float().mean() > 0.9
    assert narrow.f > wide.f  # a narrower view gets a longer focal length
    full = g.rectify(rig, alpha=1.0, fov_max=150.0)
    assert full.f < wide.f


@pytest.mark.parametrize(
    ("rig", "kwargs", "match"),
    [
        (_rig(MILD, np.eye(3), np.array([0.0, -0.1, 0.01])), {}, "vertical stereo rigs"),
        (_rig(MILD, np.eye(3), np.array([0.0, -0.1, 0.01])), {}, r"direction \(x, y, z\) = \(0.00, -1.00"),
        (_rig(MILD, np.eye(3), np.array([0.12, 0.0, 0.0])), {}, "swap the two cameras"),
        (_rig(MILD, np.eye(3), np.array([-0.01, 0.0, -0.1])), {}, "forward motion"),
        (_rig(MILD, np.eye(3), np.array([-0.05, 0.0, -0.1])), {}, "forward motion"),
        (_rig(MILD), {"alpha": 1.5}, "alpha must be in"),
        (_rig(MILD), {"fov_max": 180}, "fov_max"),
        (_rig(MILD), {"size": (100, 1)}, "size must be"),
        (g.RectifiedRig(500.0, 0.1, 320.0, 240.0), {}, "already rectified"),
    ],
)
def test_rectify_errors(rig, kwargs, match):
    with pytest.raises(InputError, match=match):
        g.rectify(rig, **kwargs)


def test_scale_rejects_a_valid_area_that_only_touches_the_edges():
    cc, size, outer = np.array([49.5, 29.5]), (100, 60), (-10.0, -10.0, 120.0, 80.0)
    assert _scale([((0.0, 0.0, 100.0, 60.0), outer)], cc, cc, size, 0.5) > 0
    for inner in [(49.5, 0.0, 50.0, 60.0), (0.0, 0.0, 100.0, 29.5), (60.0, 0.0, 40.0, 60.0)]:
        with pytest.raises(InputError, match="only touches the image edges"):
            _scale([(inner, outer)], cc, cc, size, 0.0)


def test_stereo_rig_validation():
    cam = g.Pinhole(W, H, 500.0, 500.0, 320.0, 240.0)
    rounded = np.round(R, 6)  # as stored in calibration files
    rig = g.StereoRig(cam, cam, rounded, T)
    np.testing.assert_allclose(rig.R @ rig.R.T, np.eye(3), atol=1e-12)
    for rot, t, match in (
        (-np.eye(3), T, "rotation matrix"),
        (np.round(R, 2), T, r"orthonormal to 1e-4, det \+1\), got \|R R\^T - I\| = 0.00"),  # 0.6 degrees off: a bug
        (np.eye(3), np.zeros(3), "baseline"),
        (np.eye(3), [1, 2], "R"),
    ):
        with pytest.raises(InputError, match=match):
            g.StereoRig(cam, cam, rot, np.asarray(t))
    with pytest.raises(InputError, match="camera model"):
        g.StereoRig("pinhole", cam, np.eye(3), T)


def test_rectified_rig_depth_with_doffs():
    rig = g.RectifiedRig(f=4000.0, cx=1400.0, cy=980.0, baseline=0.18, doffs=200.0, size=(2, 4))
    disparity = np.array([[100.0, -200.0, np.nan, 0.0], [300.0, -250.0, 50.0, 1e6]], np.float32)
    depth = rig.depth(disparity)
    with np.errstate(divide="ignore"):
        expected = 4000 * 0.18 / (disparity + 200)
    expected[(disparity + 200 <= 0) | ~np.isfinite(disparity)] = np.nan
    np.testing.assert_allclose(depth, expected, rtol=1e-6, equal_nan=True)
    assert depth.dtype == np.float32
    assert isinstance(rig.depth(torch.from_numpy(disparity)), torch.Tensor)
    with pytest.raises(InputError, match="disparity is 2x3"):
        rig.depth(disparity[:, :3])
    with pytest.raises(InputError, match="baseline > 0"):
        g.RectifiedRig(500.0, 0.0, 1.0, 1.0)


def test_point_cloud_frames_colors_and_valid():
    rect = g.rectify(_rig(MILD))
    rig = rect.rectified
    assert rig.size == (H, W)
    assert rig.baseline == pytest.approx(rect.baseline)
    xx = np.mgrid[0:H, 0:W][1]
    disparity = (20 + xx / W * 10).astype(np.float32)
    disparity[0, 0] = -1
    colors = _smooth_image(3)
    cloud = rect.point_cloud(disparity, colors=colors)
    assert cloud.points.shape == (H, W, 3)
    assert cloud.colors is colors
    assert not cloud.valid[0, 0]
    assert np.isnan(cloud.points[0, 0]).all()
    z = rect.f * rect.baseline / disparity[5, 7]
    np.testing.assert_allclose(cloud.points[5, 7], [(7 - rig.cx) * z / rig.f, (5 - rig.cy) * z / rig.f, z], rtol=1e-5)
    np.testing.assert_allclose(rect.depth(disparity)[5, 7], z, rtol=1e-6)
    left = rect.point_cloud(torch.from_numpy(disparity), frame="camera").points
    np.testing.assert_allclose(left[5, 7].numpy(), rect.R1.T @ cloud.points[5, 7], rtol=1e-5)
    with pytest.raises(InputError, match="frame must be"):
        rect.point_cloud(disparity, frame="right")
    with pytest.raises(InputError, match="colors must have shape"):
        rect.point_cloud(disparity, colors=colors[:10])


def test_depth_and_point_cloud_drop_pixels_without_a_source():
    """With alpha = 1 the rectified border is black; a network still predicts disparities there."""
    rect = g.rectify(_rig(STRONG), alpha=1.0)
    _, valid = rect.maps()
    assert not valid[0].all()
    disparity = np.full((H, W), 25.0, np.float32)
    cloud = rect.point_cloud(disparity)
    np.testing.assert_array_equal(cloud.valid, valid[0].numpy())
    assert np.isnan(cloud.points[~cloud.valid]).all()
    depth = rect.depth(torch.from_numpy(disparity))
    assert isinstance(depth, torch.Tensor)
    assert depth[~valid[0]].isnan().all()
    assert (depth[valid[0]] > 0).all()
    assert np.nanmax(rect.unrectify(rect.depth(disparity))) > 0
    plain = rect.rectified.depth(disparity)  # RectifiedRig knows no border: every positive disparity counts
    assert (plain > 0).all()
    border = rect.unrectify(np.where(valid[0].numpy(), 0, 5.0).astype(np.float32))
    assert np.isnan(border).all()  # unrectify ignores depth in the border, however it got there


def test_half_precision_disparity():
    rig = g.RectifiedRig(f=4000.0, cx=1400.0, cy=980.0, baseline=176.0, size=(4, 1500))  # mm: f * b > fp16 max
    disparity = torch.full((4, 1500), 200.0)
    ref = rig.point_cloud(disparity).points
    for dtype in (torch.float16, torch.bfloat16):
        assert rig.depth(disparity.to(dtype)).dtype == torch.float32
        np.testing.assert_allclose(rig.depth(disparity.to(dtype)).numpy(), 3520.0)
        points = rig.point_cloud(disparity.to(dtype)).points
        assert points.dtype == torch.float32
        torch.testing.assert_close(points, ref)  # bf16 pixel indices would round u = 1001 to 1000


def test_unrectify_a_plane():
    """A plane z = 2 m in front of the original left camera keeps depth 2 after rectify -> unrectify."""
    rect = g.rectify(_rig(MILD), alpha=1.0)  # every source pixel stays in the rectified image
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([(xx - rect.P1[0, 2]) / rect.f, (yy - rect.P1[1, 2]) / rect.f, np.ones_like(xx)], -1)
    normal = rect.R1 @ np.array([0.0, 0.0, 1.0])  # the plane's normal in the rectified frame
    depth_rect = (2.0 / (rays @ normal)).astype(np.float32)
    depth = rect.unrectify(depth_rect)
    assert depth.shape == (H, W)
    covered = depth > 0
    assert covered.mean() > 0.95
    np.testing.assert_allclose(depth[covered], 2.0, rtol=2e-3)  # nearest-neighbour sampling of a tilted plane
    batch = rect.unrectify(torch.from_numpy(np.stack([depth_rect, depth_rect])))
    assert batch.shape == (2, H, W)


def test_float64_work_leaves_mps():
    """MPS has no float64: the coarse inverse and the slow pixels' float64 solve run on the CPU there."""
    from fastforks.stereo.geometry.rectify import _f64_device  # noqa: PLC0415

    assert _f64_device(torch.device("mps")) == torch.device("cpu")
    assert _f64_device(torch.device("cpu")) == torch.device("cpu")
    assert _f64_device(torch.device("cuda", 0)) == torch.device("cuda", 0)
    rect = g.rectify(_rig(MILD), alpha=1.0)
    coarse = rect._unrectify_coarse(torch.device("cpu"))
    assert coarse.dtype == torch.float32


@pytest.mark.gpu
def test_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rect = g.rectify(_rig(STRONG), alpha=0.5)
    cpu_maps, cpu_valid = rect.maps()
    gpu_maps, gpu_valid = rect.maps("cuda")
    assert (gpu_valid.cpu() == cpu_valid).float().mean() > 0.9999
    both = gpu_valid.cpu() & cpu_valid
    torch.testing.assert_close(gpu_maps.cpu()[both], cpu_maps[both], atol=1e-4, rtol=0)
    img = torch.from_numpy(_smooth_image(3)).permute(2, 0, 1)
    out_cpu, _ = rect.remap(img, img)
    out_gpu, _ = rect.remap(img.cuda(), img.cuda())
    assert out_gpu.device.type == "cuda"
    assert (out_gpu.cpu().int() - out_cpu.int()).abs().max() <= 1
    keys = len(rect._cache)
    rect.remap(img.cuda(), img.cuda(), device="cuda")  # "cuda" and cuda:0 share one cache entry
    assert len(rect._cache) == keys
    out_np, _ = rect.remap(_smooth_image(3), _smooth_image(3), device="cuda")  # numpy in and out, computed on the GPU
    assert isinstance(out_np, np.ndarray)
    assert np.abs(out_np.astype(int) - out_cpu.permute(1, 2, 0).numpy().astype(int)).max() <= 1


def test_rectify_reuses_a_live_result_for_the_same_calibration():
    """Equal calibrations share one Rectification (and its maps) while it is alive; different ones do not."""
    import gc  # noqa: PLC0415

    rect = g.rectify(_rig(MILD))
    assert g.rectify(_rig(MILD)) is rect
    assert g.rectify(_rig(MILD), alpha=0.5) is not rect
    assert g.rectify(_rig(STRONG)) is not rect
    assert g.rectify(_rig(MILD, t=T * 2)) is not rect
    fisheye = g.StereoRig(
        g.KB4(W, H, 300.0, 300.0, 319.0, 239.0, k1=0.05), g.KB4(W, H, 300.0, 300.0, 321.0, 241.0), R, T
    )
    assert g.rectify(fisheye) is g.rectify(fisheye)
    ident = id(rect)
    del rect
    gc.collect()
    assert len([r for r in _ALIVE.values() if id(r) == ident]) == 0  # not kept alive


def test_prepare_builds_the_maps_once_from_any_thread():
    import concurrent.futures as cf  # noqa: PLC0415

    rect = g.rectify(_rig(STRONG), alpha=0.25)
    rect.prepare("cpu", background=True)
    img = _smooth_image(3)
    with cf.ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: rect.remap(img, img)[0], range(8)))
    assert all(np.array_equal(o, outs[0]) for o in outs)
    keys = dict(rect._cache)
    rect.prepare("cpu")
    assert all(rect._cache[k] is v for k, v in keys.items())  # nothing rebuilt
    depth = np.full((H, W), 2.0, np.float32)
    np.testing.assert_array_equal(rect.unrectify(depth), g.rectify(_rig(STRONG), alpha=0.25).unrectify(depth))


def _unrectify_reference(rect):
    """Float64 inverse of every original left pixel (the map before the float32 refinement)."""
    h, w = rect.rig.left.size
    nh, nw = rect.size
    u, v = torch.arange(w, dtype=torch.float64), torch.arange(h, dtype=torch.float64)
    uv, factor, ok = rect._to_rectified(torch.stack(torch.broadcast_tensors(u, v[:, None]), -1))
    ok &= (uv[..., 0] >= 0) & (uv[..., 0] <= nw - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= nh - 1)
    return uv, factor, ok


@pytest.mark.parametrize("kind", ["mild", "strong", "kb4", "double-sphere"])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_unrectify_map_matches_float64(kind, alpha):
    """The float32-refined map picks the same nearest rectified pixel as the float64 inverse (>= 99.9 %)."""
    if kind == "kb4":
        cams = (
            g.KB4(W, H, 300.0, 300.0, 319.0, 239.0, k1=0.02, k2=-0.01, k3=0.003),
            g.KB4(W, H, 302.0, 301.0, 321.0, 241.0),
        )
        rig = g.StereoRig(*cams, R, T)
    elif kind == "double-sphere":
        cam = g.DoubleSphere(W, H, 160, 160, 319.5, 239.5, xi=-0.2, alpha=0.6)
        rig = g.StereoRig(cam, cam, R, T)
    else:
        rig = _rig(MILD if kind == "mild" else STRONG)
    rect = g.rectify(rig, alpha=alpha)
    grid, factor = rect._unrectify_map(torch.device("cpu"))
    uv, factor64, ok = _unrectify_reference(rect)
    mine = grid[0, ..., 0] > -1.5
    assert (mine == ok).float().mean() > 0.99999
    nh, nw = rect.size
    q = ((grid[0].double() + 1) / torch.tensor([2 / (nw - 1), 2 / (nh - 1)], dtype=torch.float64))[mine & ok]
    assert (q - uv[mine & ok]).abs().max() < 1e-2
    assert (q.round() == uv[mine & ok].round()).all(-1).float().mean() > 0.999
    torch.testing.assert_close(factor.double()[mine & ok], factor64[mine & ok], rtol=1e-6, atol=0)


def test_rectification_pickles_and_copies():
    import copy  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    rect = g.rectify(_rig(MILD))
    maps = rect.maps()[0]
    for other in (pickle.loads(pickle.dumps(rect)), copy.deepcopy(rect)):  # noqa: S301 - our own bytes
        assert other is not rect
        np.testing.assert_array_equal(other.P1, rect.P1)
        torch.testing.assert_close(other.maps()[0], maps, rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [None, 0.0])
def test_point_cloud_matches_cv2_reproject_image_to_3d(cv2, alpha):
    """Our rectified-frame cloud equals `cv2.reprojectImageTo3D(disparity, Q)` (OpenCV's Q, pixel-centre shifted)."""
    rect = g.rectify(_rig(MILD), alpha=alpha)
    rng = np.random.default_rng(0)
    disp = rng.uniform(4, 60, (H, W)).astype(np.float32)
    cloud = rect.point_cloud(torch.from_numpy(disp)).points.numpy()
    ref = cv2.reprojectImageTo3D(disp, rect.Q)
    ok = np.isfinite(cloud).all(-1)
    assert ok.mean() > 0.5
    np.testing.assert_allclose(cloud[ok], ref[ok], rtol=1e-4, atol=1e-4)


def test_unrectified_plane_depth_matches_the_ground_truth_plane():
    """A fronto-parallel plane seen by a rotated, distorted rig: depth in the original left image is the plane's Z."""
    rig = _rig(MILD)
    rect = g.rectify(rig, alpha=0.0)
    d0 = 3.0
    z_rect = d0  # the rectified left camera looks along +z of R1 (X): plane Z_rect = d0
    disp = torch.full((H, W), float(rect.P1[0, 0] * rig.baseline / z_rect))
    depth = rect.unrectify(rect.depth(disp), view="left")
    # a rectified plane Z_rect = d0 has, in the original camera, Z = d0 / (r3 . ray): check at the principal point
    cy, cx = round(rig.left.cy), round(rig.left.cx)
    ray = rect.R1 @ np.array([0.0, 0.0, 1.0])  # original-frame direction of the rectified axis
    assert np.isfinite(depth[cy, cx].item())
    np.testing.assert_allclose(depth[cy, cx].item(), d0 / ray[2], rtol=2e-3)


@pytest.mark.parametrize(
    ("shape", "out_shape"), [((H, W), (24, 480)), ((H, W, 3), (24, 480, 3)), ((2, H, W, 3), (2, 24, 480, 3))]
)
def test_remap_at_another_size_keeps_the_numpy_layout(shape, out_shape):
    """A `size=` rectification returns numpy at the output's H, W in the input's layout (was a reshape scramble)."""
    rect = g.rectify(_rig(MILD), size=(24, 480))
    img = np.random.default_rng(0).random(shape).astype(np.float32)
    out, _ = rect.remap(img, img)
    assert out.shape == out_shape
    chw = torch.from_numpy(img) if len(shape) == 2 else torch.from_numpy(img).movedim(-1, -3)
    ref, _ = rect.remap(chw, chw)  # the tensor path was right
    np.testing.assert_array_equal(out, (ref if len(shape) == 2 else ref.movedim(-3, -1)).numpy())


def test_rectification_matrices_are_read_only():
    import pickle  # noqa: PLC0415

    rect = g.rectify(_rig(MILD))
    for other in (rect, pickle.loads(pickle.dumps(rect))):  # noqa: S301 - our own bytes
        for name in ("R1", "R2", "P1", "P2", "Q"):
            with pytest.raises(ValueError, match="read-only"):
                getattr(other, name)[0, 0] = 0.0
    assert g.rectify(_rig(MILD)) is rect  # the shared object other callers get stays intact
    cloud = rect.point_cloud(torch.full((H, W), 5.0), frame="camera")  # read-only R1 needs no writable tensor
    assert bool(cloud.valid.any())


def test_a_tilted_baseline_still_rectifies():
    rect = g.rectify(_rig(MILD, np.eye(3), np.array([-0.1, 0.0, -0.1])))  # 45 degrees off the optical axis
    assert rect.f > 0


def test_alpha_1_keeps_the_border_of_a_lens_whose_corners_pass_the_fold():
    """KITTI raw camera 03: its corners lie past the distortion fold; the valid border must still fit (alpha=1)."""
    cam = g.camera("pinhole-opencv", 1392, 512, 903.7596, 901.9653, 695.7519, 224.2509,
                   -0.3639558, 0.1788651, 6.029694e-4, -3.922424e-4, -0.0538246)  # fmt: skip
    rot = _rotation_from_vector(np.array([0.002, -0.02, 0.004]))
    rect = g.rectify(g.StereoRig(cam, cam, rot, np.array([-0.54, 0.005, -0.008])), alpha=1.0)
    t = np.linspace(0, 1, 400)
    x0, x1, y0, y1 = t * 0, t * 0 + 1391, t * 0, t * 0 + 511
    pts = np.concatenate([np.c_[t * 1391, y0], np.c_[t * 1391, y1], np.c_[x0, t * 511], np.c_[x1, t * 511]])
    rays, ok = cam.unproject(torch.from_numpy(pts))
    assert 0.9 < float(ok.float().mean()) < 1
    for rot, p in ((rect.R1, rect.P1), (rect.R2, rect.P2)):
        r = rays.numpy()[ok.numpy()] @ rot.T
        x, y = p[0, 0] * r[:, 0] / r[:, 2] + p[0, 2], p[1, 1] * r[:, 1] / r[:, 2] + p[1, 2]
        assert ((x >= -0.5) & (x <= 1391.5) & (y >= -0.5) & (y <= 511.5)).mean() > 0.99
