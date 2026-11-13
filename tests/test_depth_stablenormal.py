"""StableNormal turbo: the card, the input sizes and padding, the prompt and the output frame."""

from __future__ import annotations

import pytest
import torch

from fastforks import depth
from fastforks._core import registry
from fastforks.depth.models.stablenormal.model import finish, network_size, prepare
from fastforks.shared.sd2 import EMPTY_PROMPT, EMPTY_PROMPT_77


def test_card() -> None:
    card, variant = registry.cards("depth")["stablenormal-turbo"]
    assert card.outputs == ("normals",)
    assert len(variant.weights) == 4
    assert "stablenormal-turbo" in depth.list_models()


def test_empty_prompt_padded() -> None:
    """diffusers `encode_prompt` pads the empty prompt to 77 tokens with the SD2 pad token (0)."""
    assert len(EMPTY_PROMPT_77) == 77
    assert EMPTY_PROMPT_77[:2] == EMPTY_PROMPT
    assert set(EMPTY_PROMPT_77[2:]) == {0}


def test_sizes_and_padding() -> None:
    """diffusers `resize_to_max_edge` truncates; the bottom / right are padded to a multiple of 8 (replicate)."""
    assert network_size(480, 641, 768) == (575, 768)
    x, pad = prepare(torch.rand(1, 3, 480, 641), 768)
    assert tuple(x.shape[-2:]) == (576, 768)
    assert pad == (1, 0)
    torch.testing.assert_close(x[..., -1, :], x[..., -2, :])
    x, pad = prepare(torch.rand(1, 3, 30, 41), 0)
    assert tuple(x.shape[-2:]) == (32, 48)
    assert pad == (2, 7)
    with pytest.raises(ValueError, match="aspect ratio"):
        network_size(1, 2000, 768)


def test_finish_unpads_and_camera_frame() -> None:
    """Unit, unpad, bilinear resize, unit again, StableNormal (x left, y up, z out) -> (-x, -y, -z)."""
    n = torch.tensor([0.3, 0.4, 0.5]).view(1, 3, 1, 1).expand(1, 3, 6, 8).clone()
    n[:, :, -2:] = 100.0  # the padding rows are removed
    out = finish(n, (2, 0), (8, 16))
    assert out.shape == (1, 8, 16, 3)
    torch.testing.assert_close(out[0, 7, 15], torch.tensor([-0.3, -0.4, -0.5]) / 0.5**0.5)
