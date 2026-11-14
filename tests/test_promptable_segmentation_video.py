"""SAM 2 video tracking: hole filling and frame resizing against the references, SAMURAI motion, `track` errors."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage
import torch
from PIL import Image

from fastforks import InputError, WeightsError, promptable_segmentation
from fastforks._core import registry
from fastforks.promptable_segmentation.models.sam2.video import STABLE_FRAMES, Motion, fill_holes, resize_frame
from synth import clip, clip_prompts


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


def _masks(*boxes: tuple[int, int, int, int]) -> torch.Tensor:
    """`[1, M, 64, 64]` logits, +1 inside each `(x0, y0, x1, y1)` box (inclusive, low-res pixels), -1 outside."""
    low = -torch.ones(1, len(boxes), 64, 64)
    for m, (x0, y0, x1, y1) in enumerate(boxes):
        low[0, m, y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return low


def test_samurai_motion_selects_by_box_once_stable() -> None:
    motion = Motion(1, 0.25)
    near, far = (10, 10, 20, 20), (40, 40, 50, 50)
    for _ in range(STABLE_FRAMES):  # unstable: the best predicted IoU (the filter follows the near box)
        assert motion.select(_masks(near, far), torch.tensor([[0.9, 0.5]]), 64).item() == 0
    assert motion.stable.item() == STABLE_FRAMES
    # stable: the box the filter predicts wins over a higher predicted IoU (0.25 * 1 + 0.75 * 0.8 > 0.75 * 0.9)
    assert motion.select(_masks(far, near), torch.tensor([[0.9, 0.8]]), 64).item() == 1
    assert motion.select(_masks(far, near), torch.tensor([[0.2, 0.1]]), 64).item() == 1  # chosen IoU < 0.3: reset
    assert motion.stable.item() == 0


def test_samurai_memory_keeps_confident_frames() -> None:
    motion = Motion(2, 0.25)
    for t in range(1, 12):
        motion.best_iou = torch.tensor([0.9, 0.9 if t % 3 else 0.1])  # object 1: weak every third frame
        kept = motion.kept(t, torch.tensor([[1.0], [1.0 if t != 4 else -1.0]]))  # object 1 absent on frame 4
    groups = motion.memory(12)
    assert groups == {(6, 7, 8, 9, 10, 11): [0], (2, 5, 7, 8, 10, 11): [1]}
    assert kept == {2, 5, 6, 7, 8, 9, 10}  # frame 11's flags are read one frame later (frame 11 is never pruned at 11)
    assert motion.memory(1) == {(): [0, 1]}


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


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
def test_track_device_memory_is_bounded(mode: str) -> None:
    """`track` results live on the CPU: device memory does not grow with the video length (only the memory bank)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    card, variant = registry.resolve("promptable_segmentation", "sam2.1-hiera-tiny")
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"sam2.1-hiera-tiny: weights not available locally: {e}")
    m = promptable_segmentation.load("sam2.1-hiera-tiny", mode=mode, device="cuda")
    box = clip_prompts()["video_box_"]
    peaks = []
    for n in (12, 36):  # the bank is full after 7 frames; 24 more 720p frames were ~30 MB of on-device outputs
        frames = clip(n, 720, 1280)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        r = m.track(frames, **box)
        peaks.append(torch.cuda.max_memory_allocated())
        assert {t.device.type for t in (r.masks, r.pred_masks, r.object_score_logits)} == {"cpu"}
        assert r.masks.shape == (n, 720, 1280)
        del r
    assert peaks[1] - peaks[0] < 4 * 2**20, peaks
