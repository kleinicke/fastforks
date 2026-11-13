"""Prompt Depth Anything: the `prompt_depth` argument (checks run on the weights if available), network size."""

from __future__ import annotations

import numpy as np
import pytest

from fastforks import InputError, WeightsError, depth
from fastforks._core import registry
from fastforks.depth.models.prompt_depth_anything.model import network_size
from synth import image, prompt_depth

REF = "prompt-depth-anything-s"


def test_network_size() -> None:
    assert network_size(1440, 1920, 1008) == (756, 1008)  # upstream example image
    assert network_size(100, 140, 1008) == (98, 140)  # never upscaled, rounded down to 14
    assert network_size(10, 10, 1008) == (14, 14)


@pytest.fixture(scope="module")
def model() -> depth.DepthModel:
    card, variant = registry.resolve("depth", REF)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {REF} not available locally: {e}")
    return depth.load(REF, device="cpu", options={"max_size": 56})


def test_prompt_errors(model: depth.DepthModel) -> None:
    img = image(56, 84)
    p = prompt_depth()
    with pytest.raises(InputError, match="needs a depth prompt"):
        model(img)
    for bad in (
        np.zeros((1, 2, 3, 4)),
        np.full((4, 4), np.nan),
        np.zeros((4, 4)),
        -np.ones((4, 4)),
        np.ones((3, 4, 4)),
    ):
        with pytest.raises(InputError):
            model(img, prompt_depth=bad)
    assert model(img, prompt_depth=p).depth.shape[-2:] == (56, 84)


def test_prompt_unused_rejected() -> None:
    card, variant = registry.resolve("depth", "depth-anything-v2-s")
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights not available locally: {e}")
    m = depth.load("depth-anything-v2-s", device="cpu")
    with pytest.raises(InputError, match="does not use a depth prompt"):
        m(image(56, 84), prompt_depth=prompt_depth())
