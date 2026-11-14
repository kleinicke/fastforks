"""Review R2 fixes (CPU, no weights): extras names, `describe_models` strings, the MPS call lock, multi-view clouds."""

from __future__ import annotations

import dataclasses
import threading
import types

import numpy as np
import pytest
import torch

import fastforks
from fastforks import InputError
from fastforks._core import model as core_model
from fastforks._core import registry
from fastforks._core.tasks import spec
from fastforks.calibration import CalibrationResult
from fastforks.classification import ClassificationResult
from fastforks.depth import DepthResult
from fastforks.detection import DetectionResult
from fastforks.features import FeaturesResult
from fastforks.flow import FlowResult
from fastforks.interpolation import InterpolationResult
from fastforks.matching import MatchingResult, SparseMatchingResult
from fastforks.matting import MattingResult
from fastforks.multiview import MultiViewResult
from fastforks.pose import PoseResult
from fastforks.promptable_segmentation._api import ConceptSegmentationResult, PromptableSegmentationResult
from fastforks.restoration import RestorationResult
from fastforks.retrieval import RetrievalResult
from fastforks.segmentation import SegmentationResult
from fastforks.stereo import StereoResult
from fastforks.stereo.calibrated import PairViews, StereoPairResult, StereoView
from fastforks.stereo.pipeline import PairResult
from fastforks.volume import VolumeResult
from fastforks.tracking import TrackingResult

RESULTS = {
    "stereo": (StereoResult, StereoView, PairViews, StereoPairResult, PairResult),
    "depth": (DepthResult,),
    "multiview": (MultiViewResult,),
    "detection": (DetectionResult,),
    "pose": (PoseResult,),
    "flow": (FlowResult,),
    "matching": (MatchingResult, SparseMatchingResult),
    "matting": (MattingResult,),
    "interpolation": (InterpolationResult,),
    "retrieval": (RetrievalResult,),
    "calibration": (CalibrationResult,),
    "restoration": (RestorationResult,),
    "features": (FeaturesResult,),
    "classification": (ClassificationResult,),
    "promptable_segmentation": (PromptableSegmentationResult, ConceptSegmentationResult),
    "segmentation": (SegmentationResult,),
    "tracking": (TrackingResult,),
    "volume": (VolumeResult,),
}


def _names(cls: type) -> set[str]:
    """Dataclass fields and public attributes / methods of a result class."""
    fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    return fields | {n for n in dir(cls) if not n.startswith("_")}


def test_extras_names_do_not_shadow_result_fields():
    """R2-11: `numpy()` merges `extras` after the fields: an extras name like a field (or stereo `depth`) hides it."""
    assert set(RESULTS) == set(registry.TASKS)
    for task, classes in RESULTS.items():
        taken = set().union(*map(_names, classes)) | ({"depth"} if task == "stereo" else set())
        for mid, (card, variant) in registry.cards(task).items():
            extras = set(variant.outputs or card.outputs) - set(spec(task).standard)
            assert not extras & taken, f"{task} {mid}: extras {sorted(extras & taken)} shadow result fields"


def test_describe_models_rejects_a_string():
    """R2-3: a str is an id or a pattern; the error says which argument takes which."""
    with pytest.raises(TypeError, match=r"filter="):
        fastforks.describe_models("igev*")


def _stub(device: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device=torch.device(device), _runner=types.SimpleNamespace(deterministic_cudnn=False), _lock=threading.RLock()
    )


def test_mps_calls_take_the_process_lock():
    """R2-5: MPS calls run one at a time per process, CPU calls do not take the lock."""
    with core_model.Model._session(_stub("mps")):
        assert core_model._GPU_CALLS._is_owned()
    with core_model.Model._session(_stub("cpu")):
        assert not core_model._GPU_CALLS._is_owned()


def _mv(views, world_points=None, mask=None):
    v = len(views)
    fields = dict.fromkeys(("confidence", "intrinsics", "cam_to_world"), None)
    return MultiViewResult(
        depth=torch.rand(v, 2, 3),
        world_points=world_points,
        mask=mask,
        **fields,
        extras={},
        views=views,
        kind="metric",
        model="m",
        backend="portable",
    )


def _read_ply(path) -> np.ndarray:
    data = path.read_bytes()
    head, body = data.split(b"end_header\n", 1)
    n = int(next(ln for ln in head.decode().splitlines() if ln.startswith("element vertex")).split()[-1])
    return np.frombuffer(body, "<f4").reshape(n, 3)


def test_multiview_save_cloud_of_one_view(tmp_path):
    """R2-13: `.ply` writes one view's world points outside the mask; a bool view is refused."""
    pts = torch.arange(2 * 2 * 3 * 3, dtype=torch.float32).reshape(2, 2, 3, 3)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)
    mask[1, 0, 0] = False
    r = _mv((0, 4), pts, mask)
    r.save(tmp_path / "v4.ply", view=4)
    np.testing.assert_array_equal(_read_ply(tmp_path / "v4.ply"), pts[1].reshape(-1, 3)[1:].numpy())
    with pytest.raises(InputError, match="pass view="):
        r.save(tmp_path / "x.npy")
    r.save(tmp_path / "all.ply")  # no view=: the merged cloud of all views
    np.testing.assert_array_equal(_read_ply(tmp_path / "all.ply"), pts.reshape(-1, 3)[mask.reshape(-1)].numpy())
    with pytest.raises(InputError, match="view index"):
        r.save(tmp_path / "x.npy", view=True)
    with pytest.raises(InputError, match="key= and bits= are for maps"):
        r.save(tmp_path / "x.ply", "confidence", view=0)
    with pytest.raises(InputError, match="no world_points"):
        _mv((0,)).save(tmp_path / "x.ply")
    _mv((0,), pts[:1]).save(tmp_path / "one.ply")  # one view, no mask: every point
    assert len(_read_ply(tmp_path / "one.ply")) == 6
