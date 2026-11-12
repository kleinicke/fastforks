"""SAM 2 video tracking: hole filling and frame resizing against the references, `track` input errors."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage
import torch
from PIL import Image

from fastforks import InputError, WeightsError, promptable_segmentation
from fastforks._core import registry
from fastforks.promptable_segmentation.models.sam2.video import fill_holes, resize_frame
from synth import clip


def _fill_holes_scipy(low: np.ndarray, max_area: int = 8) -> np.ndarray:
    """Upstream `fill_holes_in_mask_scores` with 8-connected components (its CUDA extension's labelling)."""
    out = low.copy()
    for i in range(low.shape[0]):
        lab, _ = scipy.ndimage.label(low[i, 0] <= 0, structure=np.ones((3, 3)))
        area = np.bincount(lab.ravel())
        hole = (lab > 0) & (area[lab] <= max_area)
        out[i, 0][hole] = 0.1
    return out


@pytest.mark.parametrize("p_bg", [0.2, 0.5, 0.8])  # few / many / almost only background pixels
def test_fill_holes_matches_connected_components(p_bg: float) -> None:
    g = np.random.default_rng(int(p_bg * 10))
    low = np.where(g.random((3, 1, 40, 56)) < p_bg, -g.random((3, 1, 40, 56)), g.random((3, 1, 40, 56)) + 0.01)
    low[0, 0, 5:8, 5:8] = -1.0  # a 9-pixel hole: kept
    low = low.astype(np.float32)
    np.testing.assert_array_equal(fill_holes(torch.from_numpy(low)).numpy(), _fill_holes_scipy(low))


def test_resize_frame_matches_pil_bicubic() -> None:  # CPU: exact (the GPU path: the video goldens)
    f = clip(1, 60, 90)[0]
    want = np.asarray(Image.fromarray(f).resize((128, 128)), np.float32) / 255
    got = resize_frame(torch.from_numpy(f).permute(2, 0, 1)[None].float() / 255, 128)[0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(got, want)


@pytest.mark.slow
@pytest.mark.model
def test_track_input_errors() -> None:
    card, variant = registry.resolve("promptable_segmentation", "sam2.1-hiera-tiny")
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"sam2.1-hiera-tiny: weights not available locally: {e}")
    m = promptable_segmentation.load("sam2.1-hiera-tiny", mode="portable", device="cpu")
    frames = clip(2)
    point = {"input_points": [[50, 45]], "input_labels": [1]}
    with pytest.raises(InputError, match="no frames"):
        m.track([], **point)
    with pytest.raises(InputError, match="sequence of frames"):
        m.track("video.mp4", **point)
    with pytest.raises(InputError, match="one size"):
        m.track([frames[0], frames[1][:50]], **point)
    with pytest.raises(InputError, match="not both"):
        m.track(frames, input_masks=np.ones((100, 150)), **point)
    with pytest.raises(InputError, match="frame size"):
        m.track(frames, input_masks=np.ones((50, 150)))
