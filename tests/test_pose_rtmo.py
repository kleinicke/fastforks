"""Weight-free checks of RTMO's pre- and post-processing: letterbox, the folded Focus stem, priors and NMS."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks.pose.models.rtmo import model as rtmo
from fastforks.pose.models.rtmo.net import _unfocus


def _image(h: int, w: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (1, 3, h, w), generator=g, dtype=torch.uint8).float() / 255


def test_letterbox_of_a_square_image_at_the_network_size_is_the_image():
    image = _image(64, 64)
    got, r = rtmo.letterbox(image, 64)
    assert r == 1.0
    assert torch.equal(got, (image * 255).round())


def test_letterbox_centres_the_image_and_fills_with_grey():
    image = _image(64, 128)  # r = 0.5: output (u, v) reads source pixel (2 u, 2 v - 32) exactly
    got, r = rtmo.letterbox(image, 64)
    assert r == 0.5
    levels = (image * 255).round()
    assert torch.equal(got[0, :, 16:48], levels[0, :, ::2, ::2])
    assert torch.equal(got[0, :, :16], torch.full((3, 16, 64), rtmo.PAD))
    assert torch.equal(got[0, :, 48:], torch.full((3, 16, 64), rtmo.PAD))


def test_folded_stem_is_the_focus_conv():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(4, 12, 3, 3, generator=g, dtype=torch.float64)
    bgr = torch.randn(1, 3, 16, 20, generator=g, dtype=torch.float64)
    phases = [bgr[..., 0::2, 0::2], bgr[..., 1::2, 0::2], bgr[..., 0::2, 1::2], bgr[..., 1::2, 1::2]]
    want = F.conv2d(torch.cat(phases, 1), w, padding=1)
    got = F.conv2d(bgr.flip(1), _unfocus(w), stride=2, padding=2)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)


def test_priors_are_cell_centres_row_by_row():
    points, strides = rtmo.priors(64, torch.device("cpu"))
    assert points.shape == (16 + 4, 2)
    assert points[:5].tolist() == [[7.5, 7.5], [23.5, 7.5], [39.5, 7.5], [55.5, 7.5], [7.5, 23.5]]
    assert points[16:18].tolist() == [[15.5, 15.5], [47.5, 15.5]]
    assert strides.tolist() == [16.0] * 16 + [32.0] * 4


def test_nms_drops_boxes_a_kept_one_overlaps():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 0.0, 11.0, 10.0],  # IoU 9 / 11 with the first: goes
            [8.0, 0.0, 18.0, 10.0],  # IoU 2 / 18 with the first, 3 / 17 with the dropped second: stays
            [8.0, 1.0, 18.0, 11.0],  # IoU 9 / 11 with the third: goes
        ]
    )
    assert rtmo.nms(boxes, 0.65).tolist() == [0, 2]
    assert rtmo.nms(boxes[:0], 0.65).tolist() == []


@pytest.mark.gpu
@pytest.mark.parametrize(("h", "w", "size"), [(63, 199, 64), (199, 63, 128), (64, 64, 64), (1, 13, 8)])
def test_cuda_letterbox_matches_portable(h: int, w: int, size: int) -> None:
    image = _image(w, h).cuda().transpose(2, 3)
    expected, scale = rtmo.letterbox(image, size)
    got, actual_scale = rtmo.Kernels.resize(image, size)
    assert actual_scale == scale
    assert torch.equal(got, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("people", [1, 3, rtmo.DCC_ROWS, rtmo.DCC_ROWS + 6])  # graph (padded), graph, eager
def test_graphed_dcc_matches_eager(people: int) -> None:
    """Kernels' `_GraphedDCC` pads the people to `DCC_ROWS` copies: the real rows keep the eager values."""
    from fastforks.pose.models.rtmo.net import DCC, Arch  # noqa: PLC0415

    torch.manual_seed(0)
    dcc = DCC(Arch(deepen=0.33, widen=0.5, keypoints=17, size=640)).cuda().eval()
    for p in dcc.parameters():
        torch.nn.init.normal_(p, std=0.05)
    dcc.x_bins.copy_(torch.linspace(0, 640, dcc.x_bins.numel()))
    dcc.y_bins.copy_(torch.linspace(0, 640, dcc.y_bins.numel()))
    dcc.dim_t.copy_(torch.logspace(0, 4, dcc.dim_t.numel()))
    graphed = rtmo._GraphedDCC(dcc, enabled=True, capture_after=1)
    with torch.inference_mode():
        for _ in range(3):  # the first call runs eagerly, the second captures, the third replays
            pos = torch.rand(people, 2, device="cuda") * 640
            inputs = (torch.randn(people, 256, device="cuda"), pos, pos * 0.3 + 10, pos.flip(1))
            got, want = graphed(*inputs), dcc(*inputs)
            assert got.shape == want.shape == (people, 17, 2)
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-3)
    assert (graphed.graphs.cache is not None and len(graphed.graphs.cache.graphs) == 1) == (people <= rtmo.DCC_ROWS)
