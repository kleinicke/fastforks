"""InSPyReNet: the card's variants, the OpenCV shim and the runner's pre- and post-processing around the network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastforks import matting
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.matting.models.inspyrenet.net import InSPyReNet_SwinB, _cv2

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matting").values() if c.family == "inspyrenet"}
    assert sorted(cards) == ["inspyrenet", "inspyrenet-fast"]
    for vid, size in {"inspyrenet": 1024, "inspyrenet-fast": 384}.items():
        card, variant = cards[vid]
        assert card.licence.code == card.licence.weights == "MIT"
        assert {**card.params, **variant.params}["size"] == size


@pytest.mark.parametrize("k", [5, 7, 9, 17])
def test_cv2_shim(k: int) -> None:
    np.testing.assert_allclose(_cv2.getGaussianKernel(k, 1), cv2.getGaussianKernel(k, 1), rtol=1e-12, atol=0)
    want = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    np.testing.assert_array_equal(_cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k)), want)


@pytest.mark.slow
def test_matches_remover(tmp_path: Path) -> None:
    """`Remover.process` on float tensors: resize to 384, ImageNet, network (sigmoid, min-max), bilinear back."""
    torch.manual_seed(0)
    net = InSPyReNet_SwinB(depth=64, pretrained=False, base_size=[384, 384], threshold=None).eval()
    torch.save(net.state_dict(), tmp_path / "w.pth")
    m = matting.load("inspyrenet-fast", weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    img = synthetic_image(45, 67, torch.device("cpu"))
    x = F.interpolate(img, (384, 384), mode="bilinear", antialias=True)
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    with torch.inference_mode():
        pred = net((x - mean[:, None, None]) / std[:, None, None])
        want = F.interpolate(pred, (45, 67), mode="bilinear", align_corners=True)[0, 0]
        got = m(img).alpha[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)


def test_sica_float32_context() -> None:
    """The fp16 runner's `SICA` override (context in float32) is upstream's forward, here both in float32."""
    from fastforks.matting.models.inspyrenet.model import _sica  # noqa: PLC0415
    from fastforks.matting.models.inspyrenet.net.modules.attention_module import SICA  # noqa: PLC0415

    torch.manual_seed(0)
    for lmap_in in (False, True):
        m = SICA(16, depth=8, base_size=[32, 32], stage=2, lmap_in=lmap_in).eval()
        x, smap, lmap = torch.randn(2, 16, 8, 8), torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
        args = (x, smap, lmap if lmap_in else None)
        with torch.inference_mode():
            for got, want in zip(_sica(m, *args), m(*args), strict=True):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
