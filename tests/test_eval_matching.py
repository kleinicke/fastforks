"""The accuracy adapter uses the same detector geometry for both sparse runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image

from eval_matching import SPARSE_OPTIONS, pair_images, read, sparse_options
from fastforks._core import registry

if TYPE_CHECKING:
    from pathlib import Path


def test_sparse_accuracy_detector_geometry() -> None:
    """LoMa and FlexGlue's original adapters use the 1024 px detector settings in their cards."""
    for ref in ("loma-g", "aliked-flexglue"):
        card, _ = registry.resolve("matching", ref)
        assert card.options["long_side"] == 1024
        assert card.options["max_keypoints"] == 2048
        assert sparse_options(ref) == {}
    for ref in ("aliked-lightglue", "xfeat"):
        assert sparse_options(ref) == SPARSE_OPTIONS


def test_pair_decoding_preserves_layout_and_order(tmp_path: Path) -> None:
    """Prefetch keeps nonconsecutive pair IDs, asymmetric RGB inputs and their original geometry."""
    Image.new("RGB", (8, 4), (1, 127, 255)).save(tmp_path / "wide.png")
    Image.new("L", (3, 7), 24).save(tmp_path / "tall.png")
    wide, tall = (read(tmp_path / name, device="cpu") for name in ("wide.png", "tall.png"))
    pairs = [(20, "wide.png", "tall.png"), (5, "tall.png", "wide.png")]
    out = list(pair_images(tmp_path, pairs))
    assert [i for i, _, _ in out] == [20, 5]
    for (_, left, right), (first, second) in zip(out, ((wide, tall), (tall, wide)), strict=True):
        torch.testing.assert_close(left, first, check_stride=True)
        torch.testing.assert_close(right, second, check_stride=True)
    assert list(pair_images(tmp_path, [])) == []
