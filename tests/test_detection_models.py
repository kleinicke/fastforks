"""Detection family helpers (no weights) and behaviour with weights (skipped without them)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError, WeightsError, detection
from fastforks._core import registry
from fastforks.detection._resize import pil_bilinear
from synth import image

if TYPE_CHECKING:
    from fastforks._core.cards import Mode

SIZES = [  # (h, w) -> (nh, nw): both pass orders, up / down scaling, the shortest-edge-800 resize of a 1297 x 840 image
    ((840, 1297), (800, 1235)),
    ((560, 168), (83, 142)),
    ((1465, 564), (1126, 617)),
    ((75, 42), (509, 372)),
    ((693, 142), (209, 70)),
    ((222, 571), (80, 165)),
    ((31, 40), (31, 97)),
    ((97, 40), (12, 40)),
]
RCNN = ["mask-rcnn-r50-fpn-3x-coco", "faster-rcnn-r50-fpn-3x-coco"]
RTDETR = ["rtdetrv2-s-coco", "rtdetrv2-m-coco", "rtdetrv2-ms-coco", "rtdetrv2-l-coco", "rtdetrv2-x-coco"]
DFINE = ["dfine-n-coco", "dfine-s-coco", "dfine-m-coco", "dfine-l-coco", "dfine-x-coco"]
PORTABLE_ONLY = ["detr-r50-coco", "detr-r101-coco", *RTDETR, *DFINE]  # no max backend
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def _cuda(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("size", "out"), SIZES)
def test_pil_bilinear_is_bit_exact(size: tuple[int, int], out: tuple[int, int], device: str) -> None:
    _cuda(device)
    rgb = np.random.default_rng(sum(size + out)).integers(0, 256, (*size, 3), dtype=np.uint8)
    want = np.asarray(Image.fromarray(rgb).resize(out[::-1], Image.Resampling.BILINEAR))
    got = pil_bilinear(torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device), out)
    assert got.dtype == torch.float32
    np.testing.assert_array_equal(got[0].permute(1, 2, 0).cpu().numpy(), want)


def _load(ref: str, device: str, mode: Mode, **options: object) -> detection.DetectionModel:
    card, variant = registry.resolve("detection", ref)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")
    return detection.load(ref, mode=mode, device=device, options=options)


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize(
    ("device", "mode"),
    [
        ("cpu", "portable"),
        pytest.param("cuda", "portable", marks=pytest.mark.gpu),
        pytest.param("cuda", "kernels", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("ref", [*RCNN, *PORTABLE_ONLY])
def test_no_detections(ref: str, device: str, mode: Mode) -> None:
    """No box above the threshold (a softmax score never exceeds 1): empty outputs of the right shapes, no crash."""
    _cuda(device)
    if mode == "kernels" and ref in PORTABLE_ONLY:
        pytest.skip("portable only")
    m = _load(ref, device, mode, score_threshold=1.0)
    for x in (image(), np.zeros((64, 96, 3), np.uint8)):
        r = m(x)
        assert isinstance(r, detection.DetectionResult)
        h, w = x.shape[:2]
        assert r.boxes.shape == (0, 4)
        assert r.scores.shape == (0,)
        assert r.labels is not None
        assert r.labels.shape == (0,)
        if "masks" in m.variant.outputs:
            assert r.masks is not None
            assert r.masks.shape == (0, h, w)
            assert r.masks.dtype == torch.bool
        else:
            assert r.masks is None


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", ["faster-rcnn-r50-fpn-3x-coco", "detr-r50-coco", "rtdetrv2-s-coco"])
def test_per_call_score_threshold_equals_the_load_option(ref: str) -> None:
    """`model(x, score_threshold=t)` gives the result of a model loaded with `score_threshold=t`."""
    m = _load(ref, "cpu", "portable")
    at = _load(ref, "cpu", "portable", score_threshold=0.5)
    x = image()
    default, high, loaded = m(x), m(x, score_threshold=0.5), at(x)
    assert isinstance(default, detection.DetectionResult)
    assert isinstance(high, detection.DetectionResult)
    assert isinstance(loaded, detection.DetectionResult)
    assert len(high) == len(loaded) <= len(default)
    torch.testing.assert_close(high.boxes, loaded.boxes)
    torch.testing.assert_close(high.scores, loaded.scores)
    assert bool((high.scores >= 0.5).all())
    assert high.options == loaded.options == {"score_threshold": 0.5}
    assert default.options == m.options
    for bad in (-0.1, float("nan"), "0.5", True):
        with pytest.raises(InputError, match="score_threshold"):
            m(x, score_threshold=bad)


def test_fold_frozen_bn_rejects_stray_norm_keys():
    from fastforks._core.errors import CheckpointMismatchError  # noqa: PLC0415
    from fastforks.detection.models.rcnn.net import fold_frozen_bn  # noqa: PLC0415

    sd = {f"c.norm.{n}": torch.ones(2) for n in ("weight", "bias", "running_mean", "running_var")}
    sd["c.weight"] = torch.ones(2, 1, 1, 1)
    assert set(fold_frozen_bn(sd)) == {"c.weight", "c.bias"}
    with pytest.raises(CheckpointMismatchError, match=r"c\.norm\.extra"):
        fold_frozen_bn({**sd, "c.norm.extra": torch.ones(2)})


def test_detr_resize_truncates_like_the_official_transform():
    from fastforks.detection.models.detr.model import resized_size  # noqa: PLC0415

    assert resized_size(480, 640, 800, 1333) == (800, 1066)  # int(800 * 640 / 480), not rounded
    assert resized_size(400, 1000, 800, 1333) == (533, 1332)  # the long side capped
    assert resized_size(800, 1200, 800, 1333) == (800, 1200)  # already the right size


def test_detr_class_map_is_the_80_coco_classes():
    from fastforks.detection.models.detr.model import class_map  # noqa: PLC0415

    ids = class_map(91)
    assert ids.shape == (91,)
    assert ids[ids >= 0].tolist() == list(range(80))
    assert ids[0] == ids[12] == ids[83] == -1
    assert ids[1] == 0
    assert ids[90] == 79  # toothbrush


def test_detr_fold_bn_merges_every_norm():
    from fastforks.detection.models.detr.net import Arch, Detr, fold_frozen_bn  # noqa: PLC0415

    net = Detr(Arch([1, 1, 1, 1], 32, 2, 1, 1, 64, 5, 91))
    sd = {}
    for n, c in net.backbone.named_modules():
        if isinstance(c, torch.nn.Conv2d):
            bn = f"{n[:-1]}1" if n.endswith("downsample.0") else n.replace("conv", "bn")
            sd[f"backbone.0.body.{n}.weight"] = c.weight.detach().clone()
            for k in ("weight", "bias", "running_mean", "running_var"):
                sd[f"backbone.0.body.{bn}.{k}"] = torch.rand(c.out_channels) + 0.5
    sd |= {n: p.detach() for n, p in net.named_parameters() if not n.startswith("backbone.")}
    out = fold_frozen_bn(sd)
    assert set(out) == {n for n, _ in net.named_parameters()}
    net.load_state_dict(out)


def test_rtdetrv2_fold_bn_merges_norms_and_repvgg_branches():
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks.detection.models.rtdetrv2.net import fold_bn  # noqa: PLC0415

    g = torch.Generator().manual_seed(0)
    c = 4
    sd = {"blk.bottlenecks.0.conv1.conv.weight": torch.randn(c, c, 3, 3, generator=g)}
    sd["blk.bottlenecks.0.conv2.conv.weight"] = torch.randn(c, c, 1, 1, generator=g)
    for n in ("conv1", "conv2"):
        for k, lo in (("weight", 0.5), ("bias", -1.0), ("running_mean", -1.0), ("running_var", 0.5)):
            sd[f"blk.bottlenecks.0.{n}.norm.{k}"] = torch.rand(c, generator=g) + lo
        sd[f"blk.bottlenecks.0.{n}.norm.num_batches_tracked"] = torch.tensor(7)
    sd["decoder.anchors"] = torch.zeros(1, 4, 4)
    out = fold_bn(sd)
    assert set(out) == {"blk.bottlenecks.0.conv.weight", "blk.bottlenecks.0.conv.bias"}
    x = torch.randn(1, c, 6, 5, generator=g)

    def branch(n, pad):
        y = F.conv2d(x, sd[f"blk.bottlenecks.0.{n}.conv.weight"], padding=pad)
        m, v = sd[f"blk.bottlenecks.0.{n}.norm.running_mean"], sd[f"blk.bottlenecks.0.{n}.norm.running_var"]
        w, b = sd[f"blk.bottlenecks.0.{n}.norm.weight"], sd[f"blk.bottlenecks.0.{n}.norm.bias"]
        return F.batch_norm(y, m, v, w, b, eps=1e-5)

    ref = branch("conv1", 1) + branch("conv2", 0)
    got = F.conv2d(x, out["blk.bottlenecks.0.conv.weight"], out["blk.bottlenecks.0.conv.bias"], padding=1)
    assert torch.allclose(got, ref, atol=1e-5)


def test_rtdetrv2_anchors_mark_the_border_cells_invalid():
    from fastforks.detection.models.rtdetrv2.net import anchors  # noqa: PLC0415

    a, valid = anchors([(8, 8), (4, 4), (2, 2)], torch.device("cpu"))
    assert a.shape == (1, 84, 4)
    assert valid.shape == (1, 84, 1)
    assert valid[0, 0:8].sum() < 8 or valid.any()  # fixed anchor size 0.05 * 2**level stays inside (0.01, 0.99)
    assert torch.isinf(a[~valid.expand_as(a)]).all()


def test_dfine_bins_and_distances():
    from fastforks.detection.models.dfine.net import BINS, UP, distance2bbox, weighting_function  # noqa: PLC0415

    for scale in (4.0, 8.0):
        w = weighting_function(scale)
        assert w.shape == (BINS + 1,)
        assert w[0] == -2 * UP * scale
        assert w[-1] == 2 * UP * scale
        assert w[BINS // 2] == 0
        assert torch.allclose(w, -w.flip(0))
        assert (w.diff() > 0).all()
        ref = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        assert torch.allclose(distance2bbox(ref, torch.zeros(1, 4), scale), ref)  # no distance: the reference box
