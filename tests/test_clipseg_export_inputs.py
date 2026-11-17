"""CLIPSeg's exported module has a separate projected text input."""

from types import SimpleNamespace

import pytest
import torch

from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks._core.export import _check_clipseg_bounds, _input_names
from fastforks._core.sizes import size_range
from fastforks.promptable_segmentation._api import ConceptSegmentationModel, PromptableSegmentationModel


def test_clipseg_example_condition_and_fixed_size(monkeypatch):
    model = object.__new__(ConceptSegmentationModel)
    model.card = SimpleNamespace(family="clipseg")
    model._runner = SimpleNamespace(pixels=lambda x: x + 2, condition=lambda _: torch.ones(1, 512))
    monkeypatch.setattr(
        PromptableSegmentationModel,
        "example_inputs",
        lambda _self, height, width, batch: (torch.zeros(batch, 3, height or 352, width or 352),),
    )
    image, condition = model.example_inputs(batch=2)
    assert image.shape == (2, 3, 352, 352)
    assert image.min() == 2
    assert condition.shape == (2, 512)
    assert condition.sum() == 1024
    assert _input_names(model, (image, condition)) == ["image", "condition"]
    with pytest.raises(InputError, match="352x352"):
        model.example_inputs(224, 224)
    with pytest.raises(InputError, match="fixed size"):
        _check_clipseg_bounds(model, ((256, 256), (512, 512)))
    card, variant = registry.resolve_any("clipseg")
    assert size_range(variant, card).export_size == (352, 352)
