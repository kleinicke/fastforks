"""YOLOX: the helpers without weights, and the runners away from the golden's single call (skipped without weights).

The goldens make one call per variant on one photo with one score threshold, so they cannot see: the truncated resize
at other sizes, the stem's BGR -> RGB flip on its own, an empty result, and whether the kernels runner's CUDA graph is
replayed with new images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fastforks import WeightsError, detection
from fastforks._core import registry
from fastforks.detection.models.yolox.model import fit
from fastforks.detection.models.yolox.net import BN_EPS, STRIDES, YOLOX, Arch, Focus, cells, convert, decode
from synth import image

if TYPE_CHECKING:
    from fastforks._core.cards import Mode

REF = "yolox-tiny"
NANO = Arch(depth=0.33, width=0.125, num_classes=3)


def test_fit_truncates_as_the_reference_preproc() -> None:
    assert fit(480, 640, 640) == (480, 640)
    assert fit(1000, 1500, 640) == (426, 640)  # 1000 * 0.42666 = 426.67, truncated
    assert fit(300, 200, 416) == (416, 277)  # enlarged
    assert fit(1, 4000, 640) == (1, 640)  # at least one pixel


def _bn(c: int, g: torch.Generator) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(c, eps=BN_EPS).eval()
    with torch.no_grad():
        for p in (bn.weight, bn.bias, bn.running_mean):
            p.copy_(torch.randn(p.shape, generator=g))
        bn.running_var.copy_(torch.rand(c, generator=g) + 0.5)
    return bn


def test_convert_folds_batchnorm_and_turns_the_bgr_focus_stem_into_rgb() -> None:
    """The reference Focus (BGR, phases top-left, bottom-left, top-right, bottom-right) equals ours on RGB."""
    g = torch.Generator().manual_seed(0)
    net = YOLOX(NANO).eval()
    sd: dict[str, torch.Tensor] = {}
    for k, v in net.state_dict().items():  # an upstream-shaped state dict: conv + BatchNorm per BaseConv
        if k.startswith("head.reg_obj_preds."):
            i, n = k.split(".")[2:]
            sd[f"head.reg_preds.{i}.{n}"], sd[f"head.obj_preds.{i}.{n}"] = torch.randn(v.shape, generator=g).split(4)
        elif k.endswith(".conv.bias"):
            stem = k[: -len(".conv.bias")]
            sd |= {
                f"{stem}.bn.{n}": t for n, t in _bn(v.shape[0], g).state_dict().items() if n != "num_batches_tracked"
            }
        else:
            sd[k] = torch.randn(v.shape, generator=g)
    net.load_state_dict(convert(sd))
    pre = "backbone.backbone.stem.conv"
    bn = nn.BatchNorm2d(sd[f"{pre}.bn.weight"].shape[0], eps=BN_EPS).eval()
    bn.load_state_dict(
        {n: sd[f"{pre}.bn.{n}"] for n in ("weight", "bias", "running_mean", "running_var")}, strict=False
    )
    bgr = torch.randn(1, 3, 16, 12, generator=g)
    phases = (bgr[..., ::2, ::2], bgr[..., 1::2, ::2], bgr[..., ::2, 1::2], bgr[..., 1::2, 1::2])
    with torch.no_grad():
        want = F.silu(bn(F.conv2d(torch.cat(phases, 1), sd[f"{pre}.conv.weight"], padding=1)))
        stem = net.backbone.backbone.stem
        assert isinstance(stem, Focus)
        torch.testing.assert_close(stem(bgr.flip(1)), want, atol=1e-5, rtol=1e-5)
    reg_obj = net.head.reg_obj_preds[1].weight
    torch.testing.assert_close(reg_obj[:4], sd["head.reg_preds.1.weight"])
    torch.testing.assert_close(reg_obj[4:], sd["head.obj_preds.1.weight"])


def test_decode_is_the_reference_decode_outputs() -> None:
    """Cell (x, y) of each level, centre `(offset + cell) * stride`, size `exp(log size) * stride`."""
    h, w = 64, 96
    grid, strides = cells(h, w, torch.device("cpu"))
    assert grid.shape[0] == sum((h // s) * (w // s) for s in STRIDES)
    g = torch.Generator().manual_seed(1)
    maps = []
    for s in STRIDES:
        maps += [torch.randn(2, 5, h // s, w // s, generator=g), torch.randn(2, 3, h // s, w // s, generator=g)]
    boxes, obj, cls = decode(tuple(maps), grid, strides)
    # the last cell of level 1 (stride 16): column 5, row 3
    n0, n1 = (h // 8) * (w // 8), (h // 16) * (w // 16)
    o = maps[2][1, :, 3, 5]
    cx, cy = (o[0] + 5) * 16, (o[1] + 3) * 16
    bw, bh = o[2].exp() * 16, o[3].exp() * 16
    want = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    torch.testing.assert_close(boxes[1, n0 + n1 - 1], want)
    torch.testing.assert_close(obj[1, n0 + n1 - 1], o[4].sigmoid())
    torch.testing.assert_close(cls[1, n0 + n1 - 1], maps[3][1, :, 3, 5].sigmoid())


@pytest.mark.slow
def test_every_variant_builds_and_takes_its_checkpoint_keys() -> None:
    for card, variant in registry.cards("detection").values():
        if card.family != "yolox":
            continue
        p = {**card.params, **variant.params}
        net = YOLOX(Arch(p["depth"], p["width"], p["num_classes"]))
        sd: dict[str, torch.Tensor] = {}
        for k, v in net.state_dict().items():
            if k.startswith("head.reg_obj_preds."):
                i, n = k.split(".")[2:]
                sd[f"head.reg_preds.{i}.{n}"], sd[f"head.obj_preds.{i}.{n}"] = v[:4], v[4:]
            elif k.endswith(".conv.bias"):
                stem = k[: -len(".conv.bias")]
                sd |= {f"{stem}.bn.{n}": torch.ones_like(v) for n in ("weight", "bias", "running_mean", "running_var")}
            else:
                sd[k] = v
        net.load_state_dict(convert(sd))


def _load(device: str, mode: Mode, **options: object) -> detection.DetectionModel:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    card, variant = registry.resolve("detection", REF)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {REF} not available locally: {e}")
    return detection.load(REF, mode=mode, device=device, options=options)


RUNS = [
    ("cpu", "portable"),
    pytest.param("cuda", "portable", marks=pytest.mark.gpu),
    pytest.param("cuda", "kernels", marks=pytest.mark.gpu),
]


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize(("device", "mode"), RUNS)
def test_no_detections(device: str, mode: Mode) -> None:
    """No score reaches 1: empty outputs of the right shapes and dtypes."""
    out = _load(device, mode, score_threshold=1.0)(torch.full((3, 96, 128), 0.5, device=device))
    assert out.boxes.shape == (0, 4)
    assert out.scores.shape == out.labels.shape == (0,)
    assert out.boxes.dtype == out.scores.dtype == torch.float32
    assert out.labels.dtype == torch.int64


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.gpu
def test_kernels_graph_replays_new_images_like_the_eager_network() -> None:
    """The graph is captured on the first call; later images of other sizes replay it and match `cuda_graphs=False`."""
    graph = _load("cuda", "kernels", score_threshold=0.0)  # synthetic images: every cell a candidate
    eager = _load("cuda", "kernels", score_threshold=0.0, cuda_graphs=False)
    runner = graph._runner
    assert runner.graphs is not None
    for seed, (h, w) in enumerate([(240, 320), (480, 300), (96, 640), (240, 320)]):
        img = image(h, w, seed=seed)
        got, want = graph(img), eager(img)
        assert len(want.scores) > 0
        assert torch.equal(got.labels, want.labels)
        assert torch.equal(got.scores, want.scores)
        assert torch.equal(got.boxes, want.boxes)
    assert len(runner.graphs.graphs) == 1  # one size x size graph for every image size
