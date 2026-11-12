"""Marigold normals LCM: the card, the LCM schedule and update, the input sizes and the output frame."""

from __future__ import annotations

import pytest
import torch

from fastforks import depth
from fastforks._core import registry
from fastforks.depth.models.marigold_normals.model import finish, network_size, prepare
from fastforks.depth.models.marigold_normals.net.lcm import LCM


def test_card() -> None:
    card, variant = registry.cards("depth")["marigold-normals-lcm-v0-1"]
    assert card.outputs == ("normals",)
    assert variant.output_info["kind"] == "none"
    assert card.options["steps"] == 1
    assert "marigold-normals-lcm-v0-1" in depth.list_models()


def test_lcm_timesteps() -> None:
    """diffusers `LCMScheduler.set_timesteps` with 50 original steps: evenly spaced indices into 999, 979, ..., 19."""
    assert LCM(1).timesteps == [999]
    assert LCM(4).timesteps == [999, 759, 499, 259]
    assert LCM(50).timesteps == list(range(999, 0, -20))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lcm_step(dtype: torch.dtype) -> None:
    """The cached coefficients give diffusers' update (v-prediction, boundary terms, noise but in the last step)."""
    gen = torch.Generator().manual_seed(0)
    v, x, n = (torch.randn(1, 4, 8, 8, generator=gen).to(dtype) for _ in range(3))
    lcm = LCM(4)
    for i, t in enumerate(lcm.timesteps):
        last = i == len(lcm.timesteps) - 1
        a_t = lcm.alphas_cumprod[t]
        a_prev = a_t if last else lcm.alphas_cumprod[lcm.timesteps[i + 1]]
        s = torch.tensor(t) * 10.0
        c_skip, c_out = 0.25 / (s**2 + 0.25), s / (s**2 + 0.25) ** 0.5
        denoised = c_out * (a_t.sqrt() * x - (1 - a_t).sqrt() * v) + c_skip * x
        want = denoised if last else a_prev.sqrt() * denoised + (1 - a_prev).sqrt() * n
        torch.testing.assert_close(lcm.step(v, t, x, None if last else n), want, rtol=0, atol=0)


def test_sizes_and_padding() -> None:
    assert network_size(100, 150, 768) == (512, 768)
    assert network_size(480, 641, 768) == (575, 768)  # truncated, then padded to 576
    x, pad = prepare(torch.rand(1, 3, 480, 641), 768)
    assert (tuple(x.shape[-2:]), pad) == ((576, 768), (1, 0))
    with pytest.raises(ValueError, match="aspect ratio"):
        network_size(1, 2000, 768)


def test_finish_unpads_and_returns_camera_frame() -> None:
    """Marigold's (x right, y up, z to the viewer) -> unit normals (x, -y, -z), the padding removed."""
    n = torch.tensor([0.3, 0.4, 0.5]).view(1, 3, 1, 1).expand(1, 3, 10, 12).clone()
    out = finish(n, (2, 4), (8, 8))
    assert out.shape == (1, 8, 8, 3)
    torch.testing.assert_close(out[0, 0, 0], torch.tensor([0.3, -0.4, -0.5]) / 0.5**0.5)
