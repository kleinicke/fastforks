"""Exercise the optional SDK boundaries without requiring a gated checkpoint."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from fastforks import sam3d


def test_body_uses_local_snapshot_and_forwards_estimators(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/mhr_model.pt").touch()
    (tmp_path / "model.ckpt").touch()
    sdk = ModuleType("sam_3d_body")
    loader = Mock(return_value=("model", "config"))
    estimator = Mock()
    monkeypatch.setattr(sdk, "load_sam_3d_body", loader, raising=False)
    monkeypatch.setattr(sdk, "SAM3DBodyEstimator", estimator, raising=False)
    monkeypatch.setitem(sys.modules, "sam_3d_body", sdk)
    assert sam3d.load_body(tmp_path, human_detector="boxes") is estimator.return_value
    loader.assert_called_once_with(
        str(tmp_path / "model.ckpt"), device="cuda", mhr_path=str(tmp_path / "assets/mhr_model.pt")
    )
    estimator.assert_called_once_with(
        sam_3d_body_model="model", model_cfg="config", human_detector="boxes", human_segmentor=None, fov_estimator=None
    )


def test_objects_loads_the_inference_module_from_the_requested_checkout(tmp_path):
    config = tmp_path / "pipeline.yaml"
    config.touch()
    notebook = tmp_path / "notebook"
    notebook.mkdir()
    (notebook / "inference.py").write_text(
        "def Inference(config, *, compile):\n    return lambda *_: (config, compile)\n", encoding="utf-8"
    )
    predict = sam3d.load_objects(config, checkout=tmp_path, compile_model=True)
    assert predict(None, None) == (str(config), True)


def test_missing_snapshot_names_the_required_file(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        sam3d.load_body(tmp_path)
    assert error.value.args == (tmp_path / "model.ckpt",)
