"""RealPLKSR: the card's variant, `load_file` detection and the network against upstream's forward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks._core.errors import WeightsError
from fastforks.restoration.models.realplksr import model
from fastforks.restoration.models.realplksr.net import realplksr

if TYPE_CHECKING:
    from pathlib import Path


def test_card_variant() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "realplksr"}
    assert sorted(cards) == ["realplksr-nomoswebphoto-x4"]


@pytest.mark.parametrize(("scale", "use_ea"), [(4, True), (2, False)])
def test_load_file_matches_upstream(tmp_path: Path, scale: int, use_ea: bool) -> None:  # noqa: FBT001
    torch.manual_seed(0)
    net = realplksr(dim=16, n_blocks=2, upscaling_factor=scale, kernel_size=7, split_ratio=0.25, use_ea=use_ea).eval()
    save_file(net.state_dict(), tmp_path / "w.safetensors")
    m = restoration.load_file(tmp_path / "w.safetensors", device="cpu", dtype="float32")
    assert m.variant.id == "realplksr-nomoswebphoto-x4"
    img = synthetic_image(17, 23, torch.device("cpu"))
    with torch.inference_mode():
        want = net(img.clone())
        raw = m._runner(img)["image"]
        exported = m.module()(img)
    torch.testing.assert_close(raw, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(exported, raw.clamp(0, 1), rtol=0, atol=1e-6)
    assert m(img).image.shape == (1, 3, 17 * scale, 23 * scale)


def test_from_checkpoint_rejects_dysample() -> None:
    sd = realplksr(dim=8, n_blocks=1, kernel_size=3).state_dict()
    assert model.from_checkpoint(sd) is not None
    with pytest.raises(WeightsError, match="DySample"):
        model.from_checkpoint({**sd, "to_img.init_pos": torch.zeros(1)})
