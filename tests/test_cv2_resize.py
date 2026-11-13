"""`shared.cv2_resize` against OpenCV: uint8 levels (DA3) and float32 images (Prompt Depth Anything)."""

import numpy as np
import pytest
import torch

from fastforks.shared.cv2_resize import cv2_resize

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
