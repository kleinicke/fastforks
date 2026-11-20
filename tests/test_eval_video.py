"""The fast DAVIS scorer uses the same disk and zero-padded boundaries."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import eval_video


def test_davis_boundary_scores_match_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    ndimage = pytest.importorskip("scipy.ndimage")
    pytest.importorskip("cv2")
    gt = np.zeros((480, 854), dtype=bool)
    gt[0:20, 830:] = True
    gt[120:410, 45:230] = True
    pred = np.roll(gt, (-2, 3), axis=(0, 1))
    radius = int(np.ceil(eval_video.BOUND_TH * np.linalg.norm(gt.shape)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx**2 + yy**2 <= radius**2

    for mask in (gt, pred):
        boundary = eval_video.seg2bmap(mask)
        np.testing.assert_array_equal(
            eval_video._dilate_boundary(boundary, eval_video._disk(radius)), ndimage.binary_dilation(boundary, disk)
        )

    score = eval_video.f_score(gt, pred)
    assert eval_video.f_score(gt, gt) == 1
    assert eval_video.f_score(gt, np.zeros_like(gt)) == 0
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert eval_video.f_score(gt, pred) == score
