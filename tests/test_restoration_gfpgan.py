"""GFPGAN: the card, the inference forward against the vendored upstream network, the size check, `load_file`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.errors import InputError, WeightsError
from fastforks.restoration.models.gfpgan.model import from_checkpoint
from fastforks.restoration.models.gfpgan.net import GFPGANv1Clean

if TYPE_CHECKING:
    from pathlib import Path


def _net() -> GFPGANv1Clean:
    torch.manual_seed(0)
    net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, fix_decoder=False, num_mlp=8,
                        input_is_latent=True, different_w=True, narrow=1, sft_half=True)  # fmt: skip
    with torch.no_grad():  # the training init leaves the noise weights and biases at zero; exercise them
        for name, p in net.named_parameters():
            if name.endswith((".weight", ".bias")) and p.dim() <= 4 and "noises" not in name and p.abs().max() == 0:
                p.normal_(0, 0.05)
    return net.eval()


def test_card_variants() -> None:
    (card, variant) = next(cv for cv in registry.cards("restoration").values() if cv[0].family == "gfpgan")
    assert variant.id == "gfpgan-v1.4"
    assert card.params["size"] == 512


@pytest.mark.slow
def test_matches_vendored_upstream(tmp_path: Path) -> None:
    net = _net()
    torch.save({"params_ema": net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load("gfpgan-v1.4", weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    img = torch.rand(1, 3, 512, 512)
    with torch.inference_mode():
        want = ((net(img * 2 - 1, return_rgb=False, randomize_noise=False)[0] + 1) / 2).clamp(0, 1)
        got = m(img).image
        exported = m.module()(img)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-4)
    torch.testing.assert_close(exported.clamp(0, 1), got, rtol=0, atol=1e-6)
    with pytest.raises(InputError, match="512 x 512"):
        m(torch.rand(1, 3, 512, 640))
    m2 = restoration.load_file(tmp_path / "w.pth", device="cpu", dtype="float32")
    assert m2.variant.id == "gfpgan-v1.4"


def test_from_checkpoint_rejects_other_configs() -> None:
    sd = dict(_net().state_dict().items())
    assert from_checkpoint({"conv_first.weight": torch.zeros(1)}) is None
    sd["final_linear.weight"] = torch.zeros(512, 4096)  # one style for all layers (different_w False)
    with pytest.raises(WeightsError, match="final_linear"):
        from_checkpoint(sd)
