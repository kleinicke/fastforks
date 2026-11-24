"""`shared.cv2_resize` against OpenCV: uint8 levels (DA3) and float32 images (Prompt Depth Anything)."""

import numpy as np
import pytest
import torch

from fastforks.shared.cv2_resize import _cubic, _separable, cv2_resize

cv2 = pytest.importorskip("cv2")

SIZES = [((480, 640), (476, 630)), ((97, 131), (40, 61)), ((64, 64), (32, 32))]


@pytest.mark.parametrize(("src", "dst"), SIZES)
@pytest.mark.parametrize("levels", [True, False])
def test_area_matches_opencv(src: tuple[int, int], dst: tuple[int, int], levels: bool) -> None:  # noqa: FBT001
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (*src, 3), dtype=np.uint8)
    if not levels:
        img = img.astype(np.float32) / 255
    want = cv2.resize(img, dst[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
    got = cv2_resize(x, *dst, "area", levels=levels)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, want, atol=1.0 if levels else 1e-5)
    if levels:
        assert np.mean(got != want) < 0.01  # a few rounding ties only


CUBIC_SIZES = [((97, 131), (240, 161)), ((64, 80), (37, 53)), ((48, 48), (60, 60))]


@pytest.mark.parametrize(("src", "dst"), CUBIC_SIZES)
def test_cubic_levels_match_opencv(src: tuple[int, int], dst: tuple[int, int]) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (*src, 3), dtype=np.uint8)
    want = cv2.resize(img, dst[::-1], interpolation=cv2.INTER_CUBIC).astype(np.float32)
    got = cv2_resize(torch.from_numpy(img).permute(2, 0, 1)[None], *dst, "cubic")[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, want, atol=1.0, rtol=0)
    # The uint8 oracle can use fixed-point coefficients depending on the OpenCV build.
    rounded = cv2.resize(img.astype(np.float64), dst[::-1], interpolation=cv2.INTER_CUBIC).round().clip(0, 255)
    assert np.mean(got != rounded) < 0.01  # a few rounding ties only


@pytest.mark.parametrize(("src", "dst"), CUBIC_SIZES)
def test_cubic_interpolate_is_the_tap_sums(src: tuple[int, int], dst: tuple[int, int]) -> None:
    """The bicubic `interpolate` path gives the levels of OpenCV's taps (`_separable`) on every pixel."""
    x = torch.from_numpy(np.random.default_rng(1).integers(0, 256, (1, 3, *src))).double()
    taps = _separable(x, _cubic(src[1], dst[1]), _cubic(src[0], dst[0]))
    assert torch.equal(cv2_resize(x, *dst, "cubic"), taps.round().clamp(0, 255).float())
    torch.testing.assert_close(cv2_resize(x, *dst, "cubic", levels=False), taps.float())
