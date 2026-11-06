"""Review R1 input / result fixes (CPU, fake families): aliasing, iteration, path-likes, error messages, camera checks,
saving, features options, empty SAM prompts, calibrated stereo sizes."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from tests.test_stereo_calibrated import H, Plane, W, _pair, _rig
from tests.test_tasks import FEAT_IO, Calibrated, Depth, Feat, Prompt, _views, card_data

import fastforks.stereo.geometry as g
from fastforks import InputError, depth, features, multiview, promptable_segmentation, save, stereo
from fastforks._core import registry
from fastforks._core.cards import parse_card
from fastforks._core.image import decode_image, to_rgb_batch
from fastforks._core.instances import Instances
from fastforks.multiview import MultiViewResult

pytestmark = pytest.mark.usefixtures("card_min_size")
CPU = torch.device("cpu")


@pytest.fixture
def fake(monkeypatch):
    """`fake(task, runner, outputs, output, io)`: make `<task>.load("fake")` build `runner` (as in test_tasks)."""

    def install(task, runner, outputs, output, io=None):
        card = parse_card(card_data(outputs, output, io), family="fake", task=task, where="fake card")
        monkeypatch.setattr(registry, "resolve", lambda t, _m: (card, card.variants[0]) if t == task else None)
        monkeypatch.setattr(registry, "import_model", lambda _c: types.SimpleNamespace(RUNNERS={"portable": runner}))

    return install


@pytest.fixture
def plane(monkeypatch):
    runners = {"portable": Plane, "kernels": Plane}
    monkeypatch.setattr(registry, "import_model", lambda _c: types.SimpleNamespace(RUNNERS=runners))
    return stereo.load("s2m2-s", device="cpu")


def test_instances_iterate_and_out_of_range_raises():  # R1-25
    r = Instances(torch.rand(3, 4), torch.tensor([0.2, 0.9, 0.5]), torch.arange(3), None, (), "m@1", None)
    items = list(r)
    assert [len(i) for i in items] == [1, 1, 1]
    assert torch.equal(torch.cat([i.scores for i in items]), r.scores)
    assert torch.equal(r[-1].scores, r.scores[2:])
    for bad in (3, -4):
        with pytest.raises(IndexError, match="out of range"):
            r[bad]


def test_path_like_images(tmp_path):  # R1-24
    a = np.random.default_rng(0).integers(0, 256, (5, 6, 3), dtype=np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    assert torch.equal(to_rgb_batch(tmp_path / "a.png", CPU), to_rgb_batch(str(tmp_path / "a.png"), CPU))
    assert decode_image(Path(tmp_path / "a.png")) is not None


@pytest.mark.parametrize(
    ("img", "match"),
    [
        (np.zeros((2, 2, 2), np.uint8), r"got 2 \(shape \(2, 2, 2\)"),
        (np.zeros((3, 5, 6), np.uint8), r"shape \(3, 5, 6\).*gray batch BHW"),
        (np.zeros((3, 4, 5), np.uint8), "gray batch BHW"),
    ],
)
def test_image_errors_name_the_callers_shape(img, match):  # R1-35
    with pytest.raises(InputError, match=match):
        to_rgb_batch(img, CPU)


def test_depth_intrinsics_are_copied_and_checked(fake):  # R1-1, R1-46
    fake("depth", Depth, ["depth", "mask", "intrinsics"], {"kind": "metric"})
    m = depth.load("fake", device="cpu")
    img = np.zeros((6, 7, 3), np.uint8)
    for k in (torch.tensor([[100.0, 0, 3], [0, 100, 3], [0, 0, 1]]), np.diag([100.0, 100.0, 1.0]).astype(np.float32)):
        r = m(img, intrinsics=k)
        assert r.intrinsics is not None
        k[0, 0] = 1.0  # the caller's matrix changes later: the result does not
        assert float(r.intrinsics[0, 0]) == 100.0
    bad = {
        "fx": [[0.0, 0, 3], [0, 100, 3], [0, 0, 1]],
        "fy": [[100.0, 0, 3], [0, -1, 3], [0, 0, 1]],
        "last row": [[100.0, 0, 3], [0, 100, 3], [0, 1, 1]],
        "non-finite": [[float("inf"), 0, 3], [0, 100, 3], [0, 0, 1]],
    }
    for match, k in bad.items():
        with pytest.raises(InputError, match=match):
            m(img, intrinsics=k)


def test_multiview_camera_checks(fake):  # R1-46
    fake("multiview", Calibrated, ["depth", "mask", "range"], {"kind": "metric"})
    m = multiview.load("fake", device="cpu")
    k = np.diag([5.0, 5.0, 1.0])
    c2w = np.stack([np.eye(4)] * 3)
    c2w[:, 0, 3] = [1.0, 2.0, 3.0]
    rot = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # a rigid rotation passes
    ok = c2w.copy()
    ok[1, :3, :3] = rot
    assert m(_views(), intrinsics=k, cam_to_world=ok).depth is not None
    cases = {"last row must be \\[0, 0, 0, 1\\]": (3, 1, 1.0), "orthonormal": (0, 0, 2.0), "det": None}
    for match, change in cases.items():
        bad = c2w.copy()
        if change is None:
            bad[2, 2, 2] = -1.0  # a reflection
        else:
            bad[0, change[0], change[1]] = change[2]
        with pytest.raises(InputError, match=match):
            m(_views(), intrinsics=k, cam_to_world=bad)
    with pytest.raises(InputError, match="fx"):
        m(_views(), intrinsics=np.diag([0.0, 5.0, 1.0]), cam_to_world=c2w)


def _mv(depth_, views):
    fields = dict.fromkeys(("world_points", "confidence", "mask", "intrinsics", "cam_to_world"), None)
    return MultiViewResult(depth=depth_, **fields, extras={}, views=views, kind="metric", model="m", backend="portable")


def test_multiview_save_one_view(tmp_path):  # R1-31
    d = torch.rand(2, 4, 5)
    r = _mv(d, (3, 1))
    r.save(tmp_path / "v1.npy", view=1)
    np.testing.assert_array_equal(np.load(tmp_path / "v1.npy"), d[1].numpy())
    with pytest.raises(InputError, match="pass view="):
        r.save(tmp_path / "x.npy")
    with pytest.raises(InputError, match="not in this result"):
        r.save(tmp_path / "x.npy", view=0)
    _mv(d[:1], (0,)).save(tmp_path / "one.npy")  # one view: no view= needed
    np.testing.assert_array_equal(np.load(tmp_path / "one.npy"), d[0].numpy())


def test_save_keeps_integer_npy_and_names_the_remedy(tmp_path):  # R1-36
    labels = np.arange(12, dtype=np.int64).reshape(3, 4)
    save(tmp_path / "l.npy", labels)
    assert np.load(tmp_path / "l.npy").dtype == np.int64
    save(tmp_path / "m.npy", labels > 4)
    assert np.load(tmp_path / "m.npy").dtype == np.bool_
    save(tmp_path / "f.npy", labels.astype(np.float64))
    assert np.load(tmp_path / "f.npy").dtype == np.float32
    inst = Instances(torch.rand(1, 4), torch.ones(1), torch.zeros(1, dtype=torch.long), None, (), "m@1", None)
    with pytest.raises(InputError, match=r"np\.save"):
        save(tmp_path / "i.npy", inst)
    with pytest.raises(InputError, match="pass a result with save"):
        save(tmp_path / "s.npy", "text")


def test_features_strict_size_and_layers(fake):  # R1-34
    fake("features", Feat, ["last_hidden_state", "pooler_output", "hidden_states"], {}, FEAT_IO)
    m = features.load("fake", device="cpu")
    img = np.zeros((6, 10, 3), np.uint8)
    assert m(img, size=np.int64(4)).grid == (2, 3)
    assert m(img, layers=np.int64(1)).layers == (1,)
    for size in (True, 4.5, (4,), (4, 6, 8), "4", (4.0, 6.0)):
        with pytest.raises(InputError, match="size"):
            m(img, size=size)
    for layers in ("last", True, 1.0, (0, True), None):
        with pytest.raises(InputError, match="layers"):
            m(img, layers=layers)


def test_sam_empty_prompts(fake):  # R1-45
    fake("promptable_segmentation", Prompt, ["pred_masks", "iou_scores"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    img = np.zeros((8, 12, 3), np.uint8)
    for boxes in ([], np.zeros((0, 4)), torch.zeros(0, 4)):
        r = m(img, input_boxes=boxes)
        assert r.masks.shape == (0, 3, 8, 12)
        assert r.masks.dtype == torch.bool
        assert r.iou_scores.shape == (0, 3)
        assert r.pred_masks.shape == (0, 3, 4, 4)
    assert m(img, input_boxes=[], multimask_output=False).masks.shape == (0, 1, 8, 12)
    with pytest.raises(InputError, match="no points"):
        m(img, input_points=np.zeros((2, 0, 2)), input_labels=np.zeros((2, 0)))
    r = m(img, input_points=np.zeros((1, 0, 2)), input_labels=np.zeros((1, 0)), input_boxes=[[0, 0, 3, 8]])
    assert r.masks.shape == (1, 3, 8, 12)  # empty points dropped, the box stays
    assert r.masks[0, :, :, :3].all()
    with pytest.raises(InputError, match=r"must end in \(4,\), got shape \(3,\)"):
        m(img, input_boxes=[0, 0, 1])  # the caller's shape, not the batched one


def test_calibrated_stereo_sizes(plane):  # R1-6/7/8
    left, right = _pair()
    with pytest.raises(InputError, match="portrait and landscape are swapped"):
        plane(left.transpose(1, 0, 2).copy(), right.transpose(1, 0, 2).copy(), rig=_rig())
    with pytest.raises(InputError, match=r"aspect ratio differs.*shift cx / cy"):
        plane(left[:, : W - 16], right[:, : W - 16], rig=_rig())  # a crop
    res = plane(left[:, : W - 16], right[:, : W - 16], rig=_rig().resized((H, W - 16)))  # explicit: allowed
    assert res.rectification is not None
    assert res.rectification.size == (H, W - 16)
    rig = g.RectifiedRig(60.0, 0.1, size=(H, W))
    auto = plane(left[: H // 2, : W // 2], right[: H // 2, : W // 2], rig=rig)  # same aspect: rescaled (TA-5)
    explicit = plane(left[: H // 2, : W // 2], right[: H // 2, : W // 2], rig=rig.resized((H // 2, W // 2)))
    assert torch.equal(auto.depth(), explicit.depth())
    with pytest.raises(InputError, match=r"RectifiedRig is 48x64.*keeping the aspect ratio.*StereoRig"):
        plane(left[: H // 2, : W // 3], right[: H // 2, : W // 3], rig=rig)  # another aspect ratio still raises


def test_thin_images_get_a_valid_network_size():  # R1-27: sides upstream makes 0 / 1 / < 8 are clamped, others stay
    from fastforks.depth.models.depth_anything_3.model import network_size as da3  # noqa: PLC0415
    from fastforks.depth.models.marigold.model import network_size as marigold  # noqa: PLC0415
    from fastforks.depth.models.midas.model import network_size as midas  # noqa: PLC0415
    from fastforks.depth.models.zoedepth.net.midas_core import Resize  # noqa: PLC0415

    assert midas(8, 2000, 512) == (32, 512)
    assert midas(480, 640, 512) == (512, 672)
    assert da3(8, 2000, 504)[1] == (14, 504)
    assert da3(480, 640, 504)[1] == (378, 504)
    assert marigold(8, 2000, 768) == (8, 768)
    assert marigold(480, 640, 768) == (576, 768)
    zoe = Resize(512, 384, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal")
    assert zoe.get_size(2000, 8) == (512, 32)  # (width, height)
    assert zoe.get_size(640, 480) == (512, 384)
