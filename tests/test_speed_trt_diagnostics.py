"""TensorRT speed rows must retain numerical diagnostics for every output."""

import pytest
import torch

from fastforks._core import export
from speed_check import trt_output_diagnostics


def test_trt_diagnostics(monkeypatch):
    monkeypatch.setattr(export, "output_names", lambda _model, _module, out: list(out))

    class Model:
        def module(self):
            return lambda x: {"value": x, "indices": torch.tensor([1, 2])}

    model = Model()
    x = torch.tensor([1.0, 2.0])
    actual = {"value": torch.tensor([1.5, 2.0]), "indices": torch.tensor([1, 3])}
    out = trt_output_diagnostics(model, (x,), actual)
    assert out["outputs"]["value"]["max_abs"] == 0.5
    assert out["outputs"]["indices"]["unequal"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        trt_output_diagnostics(model, (x,), {**actual, "value": torch.tensor([float("nan"), 2.0])})
    with pytest.raises(ValueError, match="names"):
        trt_output_diagnostics(model, (x,), {"wrong": x})
