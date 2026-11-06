"""Pixel-centre conventions: OpenCV (centres at integer coordinates, image centre ((W - 1) / 2, (H - 1) / 2)).

Resizing an image by `s` maps a pixel coordinate `c` to `(c + 0.5) s - 0.5`; COLMAP/DSO put the first pixel centre at
0.5 (`cx_opencv = cx_colmap - 0.5`).
"""

import importlib

import numpy as np
import pytest
import torch

import fastforks.stereo.geometry as g
from fastforks._core.errors import InputError
from fastforks.depth.models.depth_anything_3.model import _fov_intrinsics
from fastforks.multiview.models.mapanything.model import intrinsics_from_rays
from fastforks.multiview.models.mvsanywhere.model import MATCH, geometry, network_intrinsics
from fastforks.multiview.models.patchmatchnet.model import rescale_intrinsics


def test_camera_resized_keeps_the_image_centre_the_centre():
    cam = g.camera("pinhole-opencv", 640, 480, 500.0, 500.0, 319.5, 239.5, -0.1, 0.01, 0.0, 0.0)
    for size in [(240, 320), (960, 1280), (480, 800)]:  # (height, width)
        small = cam.resized(size)
        assert small.size == size
        assert (small.width, small.height) == size[::-1]
        assert small.cx == pytest.approx((size[1] - 1) / 2)
        assert small.cy == pytest.approx((size[0] - 1) / 2)
        assert small.fx == pytest.approx(500.0 * size[1] / 640)
        assert small.k1 == cam.k1  # coefficients act on normalized coordinates


def test_a_size_that_swaps_portrait_and_landscape_raises():
    cam = g.camera("pinhole", 640, 480, 500.0, 500.0, 319.5, 239.5)
    rig = g.StereoRig(cam, cam, np.eye(3), np.array([-0.1, 0.0, 0.0]))
    hint = r"sizes are \(height, width\), did you mean \(480, 640\)"
    for call in (
        lambda: cam.resized((640, 480)),
        lambda: rig.resized((640, 480)),
        lambda: g.rectify(rig, size=(640, 480)),
    ):
        with pytest.raises(InputError, match=hint):
            call()
    with pytest.raises(InputError, match="portrait and landscape"):
        g.RectifiedRig(f=500.0, baseline=0.1, size=(480, 640)).resized((1280, 960))
    assert g.rectify(rig, size=(480, 480)).size == (480, 480)  # square is neither


def test_resized_camera_sees_the_same_rays():
    """A pixel centre of the big image and the matching (c + 0.5) s - 0.5 position of the small one share a ray."""
    cam = g.camera("pinhole", 640, 480, 500.0, 520.0, 300.0, 250.0)
    small = cam.resized((240, 320))
    pix = np.array([[10.0, 20.0], [300.0, 250.0]])
    ray, _ = cam.unproject(pix)
    ray_small, _ = small.unproject((pix + 0.5) * 0.5 - 0.5)
    np.testing.assert_allclose(ray, ray_small, atol=1e-9)


def test_rectified_rig_default_centre_and_resized():
    rig = g.RectifiedRig(f=500.0, baseline=0.1, size=(481, 641))
    assert (rig.cx, rig.cy) == (320.0, 240.0)
    half = g.RectifiedRig(f=500.0, baseline=0.1, size=(480, 640)).resized((240, 320))
    assert (half.cx, half.cy, half.f, half.size) == (159.5, 119.5, 250.0, (240, 320))
    with pytest.raises(InputError, match="size"):
        g.RectifiedRig(f=500.0, baseline=0.1)
    rounded = g.RectifiedRig(f=500.0, baseline=0.1, size=(481, 641)).resized((241, 320))  # one pixel of rounding
    assert rounded.size == (241, 320)
    with pytest.raises(InputError, match=r"scales x by 0\.5 and y by 1: a rectified rig has one focal length"):
        g.RectifiedRig(f=500.0, baseline=0.1, size=(240, 640)).resized((240, 320))


def test_rectify_default_principal_point_is_the_centre_and_resize_is_exact():
    cam = g.camera("pinhole", 640, 480, 500.0, 500.0, 319.5, 239.5)
    rig = g.StereoRig(cam, cam, np.eye(3), np.array([-0.1, 0.0, 0.0]))
    same = g.rectify(rig)
    np.testing.assert_allclose(same.P1[:2, 2], [319.5, 239.5], atol=1e-9)
    assert same.f == 500.0
    bigger = g.rectify(rig, size=(960, 1280))
    np.testing.assert_allclose(bigger.P1[:2, 2], [639.5, 479.5], atol=1e-9)  # c * s would give 639.0 / 479.0
    assert bigger.f == 1000.0
    focal = g.rectify(rig, size=(960, 1280), focal=900.0)
    assert focal.f == 900.0


def test_patchmatchnet_intrinsics_rescale():
    k = torch.tensor([[[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]]])
    out = rescale_intrinsics(k, 1.5, 1.0)
    assert out[0, 0, 2].item() == pytest.approx(479.5)  # (319.5 + 0.5) * 1.5 - 0.5, not 319.5 * 1.5
    assert out[0, 0, 0].item() == 750.0
    assert out[0, 1, 2].item() == 239.5


def test_rectify_with_focal_centres_the_principal_point_for_an_asymmetric_rig():
    left = g.camera("pinhole", 160, 120, 100.0, 100.0, 88.0, 70.0)  # principal points far from the image centre
    right = g.camera("pinhole", 160, 120, 100.0, 100.0, 70.0, 50.0)
    rig = g.StereoRig(left, right, np.eye(3), np.array([-0.1, 0.0, 0.0]))
    for size in [(120, 160), (200, 320)]:
        rect = g.rectify(rig, size=size, focal=120.0)
        np.testing.assert_allclose(rect.P1[:2, 2], [(size[1] - 1) / 2, (size[0] - 1) / 2], atol=1e-9)
        np.testing.assert_allclose(rect.P2[:2, 2], rect.P1[:2, 2], atol=1e-9)


def test_rectified_rig_right_view_uses_doffs_for_the_principal_point():
    """Middlebury: doffs = cx_right - cx_left, so the right camera's principal point is cx + doffs."""
    rig = g.RectifiedRig(f=500.0, baseline=0.1, cx=100.0, cy=50.0, doffs=8.0, size=(101, 201))
    d = np.full((101, 201), 22.0, np.float32)
    z = 500.0 * 0.1 / (22.0 + 8.0)
    u = np.arange(201)
    left = rig.point_cloud(d, view="left").points
    right = rig.point_cloud(d, view="right").points
    np.testing.assert_allclose(left[50, :, 0], (u - 100.0) * z / 500.0, rtol=1e-6)
    np.testing.assert_allclose(right[50, :, 0], (u - 108.0) * z / 500.0, rtol=1e-6)
    np.testing.assert_allclose(right[..., 2], z, rtol=1e-6)
    ray = np.sqrt(1 + ((u - 108.0) / 500.0) ** 2)
    np.testing.assert_allclose(rig.depth(d, euclidean=True, view="right")[50], z * ray, rtol=1e-6)
    ray_left = np.sqrt(1 + ((u - 100.0) / 500.0) ** 2)
    np.testing.assert_allclose(rig.depth(d, euclidean=True)[50], z * ray_left, rtol=1e-6)


def test_patchmatchnet_stage_intrinsics_follow_the_strided_convolutions():
    """The stage projection scales c by the stage scale (`c s`, as upstream), which is exact for its features.

    The feature net halves with 5x5 stride-2 padding-2 convolutions: output pixel i is centred on input pixel 2 i (not
    2 i + 0.5), so a coarse pixel i sits at c / 2 in integer-centre coordinates and `(c + 0.5) s - 0.5` would be wrong.
    """
    from fastforks.multiview.models.patchmatchnet.model import _proj  # noqa: PLC0415

    x = torch.zeros(1, 1, 1, 33)
    x[..., 10] = 1.0
    conv = torch.nn.Conv2d(1, 1, 5, 2, 2, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 2, 2] = 1.0  # delta kernel: output i copies input 2 i
    assert conv(x).argmax().item() == 5  # input pixel 10 -> output pixel 5 = 10 / 2
    k = torch.tensor([[[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]]]).expand(2, 3, 3)
    c2w = torch.eye(4).expand(2, 4, 4).clone()
    c2w[1, 0, 3] = 0.1
    proj = _proj(k, c2w, 0.5)
    ref = _proj(torch.stack([torch.tensor([[250.0, 0, 319.5 / 2], [0, 250.0, 239.5 / 2], [0, 0, 1]])] * 2), c2w, 1.0)
    torch.testing.assert_close(proj, ref)


def test_patchmatchnet_projection_does_not_depend_on_the_world_origin():  # GEO-1
    """Relative poses are composed before the intrinsics: a common offset of the cameras changes nothing."""
    from fastforks.multiview.models.patchmatchnet.model import _proj  # noqa: PLC0415

    k = torch.tensor([[[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]]]).expand(2, 3, 3)
    c2w = torch.eye(4).expand(2, 4, 4).clone()
    c2w[1, 0, 3] = 0.125
    base = _proj(k, c2w, 1.0)
    assert base[0, 9].item() == pytest.approx(-62.5)
    c2w[:, 0, 3] += 1e6
    torch.testing.assert_close(_proj(k, c2w, 1.0), base)


def test_moge_points_and_intrinsics_use_opencv_pixel_centres():
    """MoGe's network works in corner-origin pixels (centre w / 2); the returned intrinsics are OpenCV ((w - 1) / 2)
    and project the returned points onto the integer pixel centres: no half-pixel shift."""
    from fastforks.depth.models.moge.model import finish, view_plane_uv  # noqa: PLC0415

    h, w, z, focal = (
        24,
        32,
        3.0,
        1.2,
    )  # a tilted plane: constant depth would not fix the focal length  # focal in view-plane units (half diagonal = 1)
    uv = view_plane_uv(w, h, w / h, "cpu") / focal  # x / z, y / z of a pinhole camera with that focal length
    coords = torch.cat([uv.permute(2, 0, 1), torch.log(z + 1.5 * uv[..., 0] - 1.0 * uv[..., 1])[None]])[None]
    out = finish(coords, torch.full((1, 1, h, w), 5.0), None, torch.ones(1), (h, w), None)
    k = out["intrinsics"][0]
    assert (k[0, 2].item(), k[1, 2].item()) == ((w - 1) / 2, (h - 1) / 2)
    fx = focal / 2 * (1 + (w / h) ** 2) ** 0.5 / (w / h) * w
    assert k[0, 0].item() == pytest.approx(fx, rel=1e-3)
    pts = out["points"][0]
    xs = pts[..., 0] / pts[..., 2] * k[0, 0] + k[0, 2]  # projection back to OpenCV pixel coordinates
    ys = pts[..., 1] / pts[..., 2] * k[1, 1] + k[1, 2]
    torch.testing.assert_close(xs, torch.arange(w).float()[None].expand(h, w), atol=1e-3, rtol=0)
    torch.testing.assert_close(ys, torch.arange(h).float()[:, None].expand(h, w), atol=1e-3, rtol=0)


def test_depth_pro_uses_only_fx_of_given_intrinsics():
    """Depth Pro scales the inverse depth by w / fx; the principal point plays no role (nothing to rescale)."""
    from fastforks.depth.models.depth_pro.model import _finish  # noqa: PLC0415

    cid = torch.full((1, 1, 16, 16), 0.5)
    k1 = torch.tensor([[[100.0, 0, 8.0], [0, 100.0, 8.0], [0, 0, 1]]])
    k2 = torch.tensor([[[100.0, 0, 3.0], [0, 100.0, 11.0], [0, 0, 1]]])
    a, b = _finish(cid, None, k1, 12, 20), _finish(cid, None, k2, 12, 20)
    torch.testing.assert_close(a["depth"], b["depth"])
    assert a["depth"].mean().item() == pytest.approx(100.0 / (0.5 * 20))
    est = _finish(cid, torch.tensor([60.0]), None, 12, 20)["intrinsics"][0]
    assert (est[0, 2].item(), est[1, 2].item()) == (9.5, 5.5)  # OpenCV centre ((W - 1) / 2, (H - 1) / 2)


def _opencv_k(f, cx, cy):
    return torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("family", ["vggt", "vggt_omega"])
def test_vggt_intrinsics_use_opencv_pixel_centres(family):
    outputs = importlib.import_module(f"fastforks.multiview.models.{family}.model").outputs
    hn, wn, h0, w0, s = 28, 42, 60, 90, 2
    enc = torch.zeros(1, s, 9)
    enc[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity rotation (xyzw)
    enc[..., 7], enc[..., 8] = 0.9, 1.2  # fov_h, fov_w
    pred = {"pose_enc": enc, "depth": torch.ones(1, s, hn, wn, 1), "depth_conf": torch.ones(1, s, hn, wn),
            "world_points": torch.ones(1, s, hn, wn, 3), "world_points_conf": torch.ones(1, s, hn, wn)}  # fmt: skip
    k = outputs(pred, hn, wn, h0, w0)["intrinsics"]
    torch.testing.assert_close(k[:, 0, 2], torch.full((s,), (w0 - 1) / 2))
    torch.testing.assert_close(k[:, 1, 2], torch.full((s,), (h0 - 1) / 2))


def test_mapanything_intrinsics_from_rays_round_trip_a_known_k():
    h, w, k = 48, 64, _opencv_k(55.0, 30.25, 22.75)
    v, u = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    rays = torch.nn.functional.normalize(torch.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1],
                                                      torch.ones_like(u)], -1), dim=-1)[None]  # fmt: skip
    torch.testing.assert_close(intrinsics_from_rays(rays)[0], k, atol=1e-3, rtol=1e-4)


def test_mvsanywhere_rays_match_the_opencv_unprojection_of_the_input_pixel():
    (h0, w0), (hn, wn) = (90, 120), (64, 96)
    k = _opencv_k(100.0, 58.5, 44.0)[None].repeat(2, 1, 1)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = 0.1
    hm, wm = hn // MATCH, wn // MATCH
    ray = geometry(network_intrinsics(k, (h0, w0), (hn, wn)), c2w, hm, wm).ray.view(3, hm, wm)
    j, i = 7, 5  # match pixel: its centre is input pixel (j + 0.5) * MATCH * w0 / wn - 0.5 (OpenCV)
    u, v = (j + 0.5) * MATCH * w0 / wn - 0.5, (i + 0.5) * MATCH * h0 / hn - 0.5
    want = torch.tensor([(u - k[0, 0, 2]) / k[0, 0, 0], (v - k[0, 1, 2]) / k[0, 1, 1], 1.0])
    torch.testing.assert_close(ray[:, i, j] / ray[2, i, j], want)


def test_depth_anything_3_fov_intrinsics_centre():
    k = _fov_intrinsics(torch.tensor([[0.8, 1.1]]), 97, 131)
    assert (float(k[0, 0, 2]), float(k[0, 1, 2])) == (65.0, 48.0)


def _patchmatch_rel(k, c2w):
    from fastforks.multiview.models.patchmatchnet.model import _proj  # noqa: PLC0415

    return _proj(k, c2w, 1.0)


def _mvsanywhere_rel(k, c2w):
    g = geometry(k, c2w, 8, 8)
    return torch.cat([g.proj.flatten(), g.cams.flatten(), g.rng])


def _unprojected(k, c2w):
    from fastforks.multiview._api import unproject  # noqa: PLC0415

    return unproject(torch.full((2, 4, 4), 2.0), k, c2w).double().flatten()


@pytest.mark.parametrize("family", [_patchmatch_rel, _mvsanywhere_rel, _unprojected])
def test_float64_poses_with_a_large_world_offset_keep_their_precision(family):
    """Every multi-view family / helper that takes `cam_to_world`: float64 poses survive the input check and a common
    world offset (1e6) changes nothing, because relative poses are formed in float64 before any float32 cast."""
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks.multiview._api import MultiViewModel  # noqa: PLC0415

    self = SimpleNamespace(variant=SimpleNamespace(ref="x"), device=torch.device("cpu"))
    k = torch.tensor([[[500.0, 0, 3.5], [0, 500.0, 3.5], [0, 0, 1]]]).expand(2, 3, 3)
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[1, 0, 3] = 0.1

    def cam(x):
        return MultiViewModel._camera(self, "cam_to_world", "required", x, 2, (4, 4))

    base = family(k, cam(c2w))
    c2w[:, 0, 3] += 1e6 + 0.3
    shifted = cam(c2w)
    assert shifted.dtype == torch.float64
    got = family(k, shifted)
    if family is _unprojected:  # world points move with the offset: compare in the reference's frame
        got = got.view(2, -1, 3) - torch.tensor([1e6 + 0.3, 0, 0], dtype=torch.float64)
        base = base.view(2, -1, 3)
        torch.testing.assert_close(got, base, atol=1e-6, rtol=0)
    else:
        torch.testing.assert_close(got, base)
