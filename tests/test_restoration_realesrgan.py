"""Real-ESRGAN: the card's variants, the network on random weights, x2 padding, ComfyUI tiling, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks._core.errors import InputError
from fastforks.restoration._api import _ramp, _starts
from fastforks.restoration.models.realesrgan.net import RRDBNet

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {"realesrgan-x4plus": (4, 23), "realesrgan-x2plus": (2, 23), "realesrgan-x4plus-anime-6b": (4, 6)}


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "realesrgan"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (scale, blocks) in VARIANTS.items():
        card, variant = cards[vid]
        p = {**card.params, **variant.params}
        assert (p["scale"], p["num_block"]) == (scale, blocks), vid


def _load(tmp: Path, ref: str) -> restoration.RestorationModel:
    """`ref` with random weights in the releases' format (`params_ema`)."""
    scale, blocks = VARIANTS[ref]
    torch.manual_seed(0)
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64, num_block=blocks, num_grow_ch=32)
    path = tmp / f"{ref}.pth"
    torch.save({"params_ema": net.state_dict()}, path)
    return restoration.load(ref, weights=path, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def anime(tmp_path_factory: pytest.TempPathFactory) -> restoration.RestorationModel:
    return _load(tmp_path_factory.mktemp("realesrgan"), "realesrgan-x4plus-anime-6b")


@pytest.mark.slow
def test_x2_odd_size_is_padded_and_exports(tmp_path: Path) -> None:
    x2 = _load(tmp_path, "realesrgan-x2plus")
    img = synthetic_image(23, 31, torch.device("cpu"))  # reflect-padded to 24 x 32 (pixel unshuffle), cropped again
    out = x2(img).image
    assert out.shape == (1, 3, 46, 62)
    assert 0 <= out.min() <= out.max() <= 1
    with torch.inference_mode():
        exported = x2.module()(img)
    torch.testing.assert_close(exported, out, rtol=0, atol=1e-5)


def test_tiles_blend_to_the_one_pass_size(anime: restoration.RestorationModel) -> None:
    img = synthetic_image(41, 70, torch.device("cpu"))
    whole = anime(img).image
    tiled = anime(img, tile=40).image  # 2 x 5 tiles of at most 40 px, overlapping by 32
    assert tiled.shape == whole.shape == (1, 3, 164, 280)
    assert torch.equal(anime(img, tile=70).image, whole)  # fits in one tile: one pass
    seams = (tiled - whole).abs()
    assert seams.mean() < 0.05  # the same image away from the tile borders


def test_tile_starts_and_ramps_match_comfyui() -> None:
    # comfy.utils.tiled_scale_multidim: range(0, n - overlap, tile - overlap), clamped to n - overlap; feather ramps
    assert _starts(100, 512) == [0]
    assert _starts(1000, 512) == [0, 480, 960]
    assert _starts(600, 512) == [0, 480]
    r = _ramp(8, 3, torch.device("cpu"))
    torch.testing.assert_close(r, torch.tensor([1 / 3, 2 / 3, 1, 1, 1, 1, 2 / 3, 1 / 3]))


@pytest.mark.parametrize("tile", [0, 32, 40.0, True])
def test_bad_tile_is_rejected(anime: restoration.RestorationModel, tile: object) -> None:
    with pytest.raises(InputError, match="tile"):
        anime(synthetic_image(8, 8, torch.device("cpu")), tile=tile)
