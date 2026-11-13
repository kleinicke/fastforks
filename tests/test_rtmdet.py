"""RTMDet: the helpers without weights, and the runners away from the golden's single call (skipped without weights).

The goldens make one call per variant on one photo with one score threshold, so they cannot see: the OpenCV resize
emulation at other size pairs, the per-level `nms_pre` cut, an empty result, a per-call threshold, and whether the
kernels runner's CUDA graph is replayed with new images.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

from fastforks import WeightsError, detection
from fastforks._core import registry
from fastforks.detection._cv2_resize import cv2_linear
from fastforks.detection.models.rtmdet.model import SPLIT, classwise_nms, fit
from fastforks.detection.models.rtmdet.net import STRIDES, Arch, RTMDet, convert, decode, points
from fastforks.shared.cspnext import BN_EPS
from synth import image

if TYPE_CHECKING:
    from fastforks._core.cards import Mode

RESIZES = [  # (h, w) -> (h, w): same size, up, the exact 2x shrink (OpenCV's area path), down, thin, 3x, mixed
    ((480, 640), (480, 640)),
    ((427, 640), (640, 640)),
    ((600, 400), (300, 200)),
    ((333, 500), (640, 427)),
    ((31, 40), (97, 320)),
    ((1080, 1920), (360, 640)),
    ((75, 42), (509, 372)),
    ((640, 480), (213, 160)),
]
RESIZED_SHA256 = "987925bf433561f377fe64ed908e2d50d85d761cf32eb5da5e04ccf6f7baa7b7"
"""sha256 over `cv2.resize(noise, INTER_LINEAR)` of `RESIZES` (HWC uint8, in order), opencv-python-headless 4.10.0.84
on arm64 (recorded; `test_cv2_linear_matches_opencv_4` recomputes it where OpenCV 4 is installed)."""
REFS = ["rtmdet-tiny-coco", "rtmdet-nano-person"]  # shared head convs / own ones, depthwise, 640 / 320
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
TINY = Arch(deepen=0.167, widen=0.375, neck_blocks=1, num_classes=80, exp_on_reg=False, depthwise=False)
NANO = Arch(deepen=0.33, widen=0.25, neck_blocks=1, num_classes=1, exp_on_reg=False, depthwise=True)


def _cuda(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")


def _noise(size: tuple[int, int], out: tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(sum(size + out)).integers(0, 256, (*size, 3), dtype=np.uint8)


@pytest.mark.parametrize("device", DEVICES)
def test_cv2_linear_equals_the_recorded_opencv_result(device: str) -> None:
    _cuda(device)
    digest = hashlib.sha256()
    for size, out in RESIZES:
        got = cv2_linear(torch.from_numpy(_noise(size, out)).permute(2, 0, 1)[None].float().to(device), *out)
        assert got.dtype == torch.float32
        assert got.shape == (1, 3, *out)
        digest.update(got[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy().tobytes())
    assert digest.hexdigest() == RESIZED_SHA256


@pytest.mark.parametrize("device", DEVICES)
def test_cv2_linear_takes_uint8_and_unit_floats_alike(device: str) -> None:
    _cuda(device)
    for size, out in RESIZES:
        u8 = torch.from_numpy(_noise(size, out)).permute(2, 0, 1)[None].to(device)
        want = cv2_linear(u8.float(), *out)
        np.testing.assert_array_equal(cv2_linear(u8, *out).cpu(), want.cpu())
        np.testing.assert_array_equal(cv2_linear(u8.float() / 255, *out, 255.0).cpu(), want.cpu())


def test_cv2_linear_matches_opencv_4() -> None:
    cv2 = pytest.importorskip("cv2")
    if not cv2.__version__.startswith("4."):
        pytest.skip(f"OpenCV {cv2.__version__}: the 8-bit INTER_LINEAR path emulated is OpenCV 4's")
    for size, out in RESIZES:
        rgb = _noise(size, out)
        want = cv2.resize(rgb, out[::-1], interpolation=cv2.INTER_LINEAR)
        got = cv2_linear(torch.from_numpy(rgb).permute(2, 0, 1)[None].float(), *out)
        np.testing.assert_array_equal(got[0].permute(1, 2, 0).numpy(), want)


def test_fit_is_mmcv_rescale_size() -> None:
    assert fit(480, 640, 640) == (480, 640)
    assert fit(427, 640, 640) == (427, 640)
    assert fit(400, 600, 640) == (427, 640)  # 426.67 rounds up
    assert fit(1080, 1920, 640) == (360, 640)
    assert fit(333, 500, 320) == (213, 320)  # 213.12
    assert fit(2000, 3, 640) == (640, 1)  # 0.96 -> at least one pixel


def test_classwise_nms_in_several_calls_keeps_what_one_call_per_class_keeps() -> None:
    pytest.importorskip("torchvision")
    from torchvision.ops import nms  # noqa: PLC0415

    g = torch.Generator().manual_seed(0)
    n = 2 * SPLIT + 500
    xy = torch.rand(n, 2, generator=g) * 600
    boxes = torch.cat([xy, xy + 5 + torch.rand(n, 2, generator=g) * 60], 1)
    labels = torch.randint(0, 7, (n,), generator=g)
    labels[: SPLIT + 300] = 3  # one class larger than a call's limit
    scores = torch.rand(n, generator=g)
    kept = classwise_nms(boxes, labels, scores, 0.65)
    want = torch.cat([m[nms(boxes[m], scores[m], 0.65)] for c in range(7) for m in [(labels == c).nonzero()[:, 0]]])
    assert sorted(kept.tolist()) == sorted(want.tolist())
    assert (scores[kept].diff() <= 0).all()
    few = classwise_nms(boxes[:500], labels[:500], scores[:500], 0.65)
    one = torch.cat(
        [m[nms(boxes[m], scores[m], 0.65)] for c in range(7) for m in [(labels[:500] == c).nonzero()[:, 0]]]
    )
    assert sorted(few.tolist()) == sorted(one.tolist())


def test_points_are_the_cell_corners_level_by_level() -> None:
    pts, strides = points(64, 96, torch.device("cpu"))
    cells = [(64 // s) * (96 // s) for s in STRIDES]
    assert pts.shape == (sum(cells), 2)
    assert strides.tolist() == [float(s) for s, n in zip(STRIDES, cells, strict=True) for _ in range(n)]
    assert pts[:3].tolist() == [[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]]  # x first, rows first
    assert pts[96 // 8].tolist() == [0.0, 8.0]
    assert pts[-1].tolist() == [64.0, 32.0]


@pytest.mark.parametrize("exp_on_reg", [False, True])
def test_decode_is_point_minus_and_plus_the_distances(*, exp_on_reg: bool) -> None:
    pts, strides = points(32, 32, torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    maps = []
    for s in STRIDES:
        maps += [torch.randn(1, 3, 32 // s, 32 // s, generator=g), torch.rand(1, 4, 32 // s, 32 // s, generator=g)]
    logits, boxes = decode(tuple(maps), pts, strides, exp_on_reg=exp_on_reg)
    assert logits.shape == (1, 21, 3)
    assert boxes.shape == (1, 21, 4)
    assert torch.equal(logits[0, 1], maps[0][0, :, 0, 1])
    d = maps[1][0, :, 0, 1]
    d = (d.exp() if exp_on_reg else d) * 8
    want = torch.tensor([8.0 - d[0], 0.0 - d[1], 8.0 + d[2], 0.0 + d[3]])
    torch.testing.assert_close(boxes[0, 1], want)


def test_convert_folds_every_batchnorm_and_flips_the_first_conv() -> None:
    g = torch.Generator().manual_seed(0)
    conv, bn = nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4, eps=BN_EPS).eval()
    with torch.no_grad():
        for p in (conv.weight, bn.weight, bn.bias, bn.running_mean):
            p.copy_(torch.randn(p.shape, generator=g))
        bn.running_var.copy_(torch.rand(4, generator=g) + 0.5)
    sd = {"backbone.stem.0.conv.weight": conv.weight.detach()}
    sd |= {f"backbone.stem.0.bn.{k}": v for k, v in bn.state_dict().items()}
    sd |= {"bbox_head.rtm_cls.0.weight": torch.ones(1), "data_preprocessor.mean": torch.zeros(3)}
    out = convert(sd)
    assert sorted(out) == ["backbone.stem.0.conv.bias", "backbone.stem.0.conv.weight", "bbox_head.rtm_cls.0.weight"]
    x = torch.randn(1, 3, 8, 8, generator=g)
    with torch.no_grad():
        want = bn(conv(x))
        got = torch.nn.functional.conv2d(
            x.flip(1), out["backbone.stem.0.conv.weight"], out["backbone.stem.0.conv.bias"], padding=1
        )
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arch", [TINY, NANO], ids=["tiny", "nano"])
def test_network_output_shapes(arch: Arch) -> None:
    net = RTMDet(arch).eval()
    with torch.no_grad():
        maps = net(torch.zeros(1, 3, 64, 96))
    assert len(maps) == 2 * len(STRIDES)
    for i, s in enumerate(STRIDES):
        assert maps[2 * i].shape == (1, arch.num_classes, 64 // s, 96 // s)
        assert maps[2 * i + 1].shape == (1, 4, 64 // s, 96 // s)


def test_every_variant_builds_and_takes_its_checkpoint_keys() -> None:
    """Each variant's `Params` builds a network whose keys are `convert`'s output of an mmdet-shaped state dict."""
    for card, variant in registry.cards("detection").values():
        if card.family != "rtmdet":
            continue
        p = {**card.params, **variant.params}
        arch = Arch(p["deepen"], p["widen"], p["neck_blocks"], p["num_classes"], p["exp_on_reg"], p["depthwise"])
        net = RTMDet(arch)
        sd = {}
        for k, v in net.state_dict().items():
            if k.endswith(".conv.bias"):  # a folded BatchNorm
                stem = k[: -len(".conv.bias")]
                sd |= {f"{stem}.bn.{n}": torch.ones_like(v) for n in ("weight", "bias", "running_mean", "running_var")}
            else:
                sd[k] = v
        net.load_state_dict(convert(sd))


def _load(ref: str, device: str, mode: Mode, **options: object) -> detection.DetectionModel:
    _cuda(device)
    card, variant = registry.resolve("detection", ref)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")
    return detection.load(ref, mode=mode, device=device, options=options)


RUNS = [
    ("cpu", "portable"),
    pytest.param("cuda", "portable", marks=pytest.mark.gpu),
    pytest.param("cuda", "kernels", marks=pytest.mark.gpu),
]


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize(("device", "mode"), RUNS)
@pytest.mark.parametrize("ref", REFS)
def test_no_detections(ref: str, device: str, mode: Mode) -> None:
    """No sigmoid score reaches 1 on a flat grey image: empty outputs of the right shapes and dtypes."""
    model = _load(ref, device, mode, score_threshold=1.0)
    out = model(torch.full((3, 96, 128), 0.5, device=device))
    assert out.boxes.shape == (0, 4)
    assert out.scores.shape == (0,)
    assert out.labels.shape == (0,)
    assert out.boxes.dtype == out.scores.dtype == torch.float32
    assert out.labels.dtype == torch.int64


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize(("device", "mode"), RUNS)
@pytest.mark.parametrize("ref", REFS)
def test_threshold_zero_takes_the_nms_pre_cut_and_keeps_the_contract(ref: str, device: str, mode: Mode) -> None:
    """Threshold 0 makes every cell and class a candidate: the per-level top-k path, sorted scores, valid boxes."""
    model = _load(ref, device, mode, score_threshold=0.0, max_detections=50)
    out = model(image(240, 320, seed=3))
    assert out.boxes.shape == (50, 4)
    assert bool((out.scores[:-1] >= out.scores[1:]).all())
    assert bool((out.boxes[:, 2:] > out.boxes[:, :2]).all())
    assert int(out.labels.min()) >= 0
    assert int(out.labels.max()) < model._runner.ctx.params.num_classes


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", REFS)
def test_per_call_score_threshold_equals_the_load_option(ref: str) -> None:
    img = image(240, 320, seed=5)
    low = _load(ref, "cpu", "portable", score_threshold=0.001)
    want = _load(ref, "cpu", "portable", score_threshold=0.01)(img)
    got = low(img, score_threshold=0.01)
    assert len(want.scores) > 0
    assert torch.equal(got.boxes, want.boxes)
    assert torch.equal(got.scores, want.scores)
    assert torch.equal(got.labels, want.labels)
    assert len(low(img).scores) >= len(got.scores)


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("ref", REFS)
def test_kernels_graph_replays_new_images_like_the_eager_network(ref: str) -> None:
    """The graph is captured on the first call; later images of other sizes replay it and match `cuda_graphs=False`."""
    graph = _load(ref, "cuda", "kernels", score_threshold=0.01)
    eager = _load(ref, "cuda", "kernels", score_threshold=0.01, cuda_graphs=False)
    runner = graph._runner
    assert runner.graphs is not None
    assert eager._runner.graphs is None
    for seed, (h, w) in enumerate([(240, 320), (480, 300), (96, 640), (240, 320)]):
        img = image(h, w, seed=seed)
        got, want = graph(img), eager(img)
        assert len(want.scores) > 0
        assert torch.equal(got.labels, want.labels)
        assert torch.equal(got.scores, want.scores)
        assert torch.equal(got.boxes, want.boxes)
    assert len(runner.graphs.graphs) == 1  # one size x size graph for every image size
