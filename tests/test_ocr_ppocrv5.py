"""PP-OCRv5 and the shared OCR steps: card, components, box lines, crops, CTC, float16 neck scaling, bench page."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastforks._core import registry
from fastforks.ocr import _crop, _db, _runner
from fastforks.ocr._api import text_page
from fastforks.ocr.models.ppocrv5.model import fold_affine
from fastforks.ocr.models.ppocrv5.net import Detector, Recognizer
from fastforks.ocr.models.ppocrv5.net.heads import DBHead
from fastforks.ocr.models.ppocrv5.net.layers import LearnableAffineBlock, up


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("ocr").values() if c.family == "ppocrv5"}
    assert sorted(cards) == ["ppocrv5-mobile", "ppocrv5-server"]
    assert cards["ppocrv5-server"].licence.code == "Apache-2.0"
    assert cards["ppocrv5-server"].options["box_threshold"] == 0.6


def test_components_label_order_and_empty_map() -> None:
    mask = torch.zeros(6, 8, dtype=torch.bool)
    mask[1:3, 5:7] = True  # first in raster order
    mask[2, 1] = mask[3, 2] = mask[4, 1:4] = True  # 8-connected (diagonal step)
    rows = [tuple(map(list, c)) for c in _runner.components(mask)]
    assert rows == [([1, 2], [5, 5], [6, 6]), ([2, 3, 4], [1, 2, 1], [1, 2, 3])]
    assert _runner.components(torch.zeros(4, 4, dtype=torch.bool)) == []


def _flood_rows(mask: np.ndarray) -> list[tuple[list[int], list[int], list[int]]]:
    """Reference: 8-connected flood fill in raster order of the first pixel, rows as `components` gives them."""
    seen, out = np.zeros_like(mask), []
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        stack, pix = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for yy in range(max(y - 1, 0), min(y + 2, h)):
                for xx in range(max(x - 1, 0), min(x + 2, w)):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        rows = sorted({y for y, _ in pix})
        lo = [min(x for y, x in pix if y == r) for r in rows]
        hi = [max(x for y, x in pix if y == r) for r in rows]
        out.append((rows, lo, hi))
    return out


def test_components_match_flood_fill_on_random_and_spiral_maps() -> None:
    rng = np.random.default_rng(0)
    spiral = np.zeros((21, 21), bool)  # one long winding component: many rounds for plain label propagation
    y0, x0, y1, x1 = 0, 0, 20, 20
    while y0 <= y1 and x0 <= x1:
        spiral[y0, x0 : x1 + 1] = spiral[y0 : y1 + 1, x1] = spiral[y1, x0 : x1 + 1] = True
        spiral[y0 + 2 : y1 + 1, x0] = True
        y0, x0, y1, x1 = y0 + 2, x0 + 2, y1 - 2, x1 - 2
    for mask in [spiral, *(rng.random((24, 40)) < p for p in (0.3, 0.5, 0.6))]:
        rows = [tuple(map(list, c)) for c in _runner.components(torch.from_numpy(mask))]
        assert rows == _flood_rows(mask)


def test_cubic_table_rows_sum_to_one() -> None:
    tab = _crop._cubic_table().astype(np.int64)
    assert (tab.sum(1) == 1 << 15).all()
    # integer positions: OpenCV saturates 2^15 to 32767 and puts the missing 1 on tap (2, 2)
    assert tab[0].tolist() == [0] * 5 + [(1 << 15) - 1] + [0] * 4 + [1] + [0] * 5


def test_axis_aligned_crop_is_the_pixels() -> None:
    img = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (3, 20, 30), dtype=np.uint8))
    src = np.array([[[4, 3], [21, 3], [21, 12], [4, 12]]], np.float32)
    (crop,) = _crop.crop_quads(img, src, [(9, 17)])
    torch.testing.assert_close(crop, img[:, 3:12, 4:21], rtol=0, atol=0)


def test_ctc_decode_merges_repeats_and_drops_blanks() -> None:
    k = torch.tensor([[1, 1, 0, 1, 2, 2, 0, 0]])
    probs = torch.nn.functional.one_hot(k, 3).float() * 0.5 + 0.1
    texts, scores = _runner.ctc_decode(probs, ["", "a", "b"])
    assert texts == ["aab"]
    np.testing.assert_allclose(scores, [0.6])
    assert _runner.ctc_decode(torch.zeros(1, 3, 3).index_fill_(2, torch.tensor([0]), 1), ["", "a", "b"])[0] == [""]


@pytest.mark.slow
def test_scale_neck_keeps_the_map() -> None:
    """The float16 rescaling of the server neck is the same function (float32, random weights and BN statistics)."""
    torch.manual_seed(0)
    net = Detector(server=True).eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2)
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        want = net(x)
        net.scale_neck(256.0)
        got = net(x)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


@pytest.mark.parametrize("kind", ["det", "rec"])
def test_fold_affine_keeps_the_output(kind: str) -> None:
    """The mobile networks with their affine blocks folded into the convs: the same function (float32)."""
    torch.manual_seed(0)
    net = (Detector(server=False) if kind == "det" else Recognizer(server=False, classes=11)).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2)
            for name in ("scale", "bias"):  # the learnable affine blocks: 1 and 0 at init
                t = getattr(m, name, None)
                if isinstance(t, torch.nn.Parameter) and t.numel() == 1:
                    t.uniform_(0.5, 1.5) if name == "scale" else t.uniform_(-0.5, 0.5)
        x = torch.randn(2, 3, 48, 96) if kind == "rec" else torch.randn(1, 3, 64, 96)
        want = net(x)
        fold_affine(net)
        got = net(x)
    folded = [m.consts == (1.0, 0.0) for m in net.modules() if isinstance(m, LearnableAffineBlock)]
    assert sum(folded) > len(folded) / 2  # every `lab`, and the depthwise `act_lab`s before a 1 x 1 conv
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_bench_page_has_text_lines() -> None:
    page = text_page(108, 192)
    assert page.shape == (108, 192, 3)
    assert page.dtype == np.uint8
    assert (page < 128).any(axis=(1, 2)).sum() > 10  # dark rows: several lines of words
    np.testing.assert_array_equal(page, text_page(108, 192))  # fixed seed


def test_local_detector_strips_preserve_halos() -> None:
    head = DBHead(16, local=True).eval()
    rng = torch.Generator().manual_seed(541)
    half_res = torch.randn(2, 4, 131, 9, generator=rng)
    shrink = torch.rand(2, 1, 262, 18, generator=rng)
    with torch.inference_mode():
        full = head.last_1(head.last_3(torch.cat((shrink, up(half_res, 2)), dim=1))).sigmoid()
        strips = head._local_strips(shrink, half_res)
    torch.testing.assert_close(strips, full, rtol=0.0, atol=1e-6)


def _reference_line(mask: np.ndarray, p: tuple[int, int], q: tuple[int, int]) -> None:
    """OpenCV's 8-connected `Line` stepped pixel by pixel (the loop `_db._line` replaces with its closed form)."""
    h, w = mask.shape
    a, b = list(p), list(q)
    if _db._outside(w, h, a, b) and not _db._clip_line(w, h, a, b):
        return
    (x0, y0), (x1, y1) = a, b
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    dx, dy = x1 - x0, y1 - y0
    sy = 1 if dy >= 0 else -1
    dy = abs(dy)
    vert = dy > dx
    if vert:
        dx, dy = dy, dx
    err = dx - 2 * dy
    x, y = x0, y0
    for _ in range(dx + 1):
        mask[y, x] = 1
        diag = err < 0
        err += -2 * dy + (2 * dx if diag else 0)
        if vert:
            y += sy
            x += 1 if diag else 0
        else:
            x += 1
            y += sy if diag else 0


def test_line_matches_the_stepped_line() -> None:
    """`_db._line`'s closed form draws OpenCV's stepped line exactly: every direction, long lines, clipped ends."""
    rng = np.random.default_rng(0)
    h, w = 61, 3000
    ends = [((5, 5), (5, 5)), ((0, 0), (2999, 60)), ((2999, 0), (0, 60)), ((10, 60), (11, 0)), ((-40, -7), (3100, 70))]
    ends += [
        tuple(tuple(int(v) for v in rng.integers((-50, -20), (w + 50, h + 20))) for _ in range(2)) for _ in range(400)
    ]
    for p, q in ends:
        got, want = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        _db._line(got, p, q)
        _reference_line(want, p, q)
        assert np.array_equal(got, want), (p, q)
