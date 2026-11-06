"""Test-user findings TC-1..TC-5, TC-7, TA-5, TA-6 (CPU, no weights)."""

from __future__ import annotations

import sys
import types

import pytest
import torch

import fastforks
import fastforks.stereo.geometry as g
from fastforks import BackendUnavailableError, InputError, MissingDependencyError
from fastforks._core import files, model
from fastforks._core.errors import ArgumentError
from fastforks.classification._api import ClassificationResult
from fastforks.promptable_segmentation import _api as sam
from fastforks.stereo._api import StereoResult
from fastforks.stereo.calibrated import resolve_geometry


def test_missing_dependency_error(monkeypatch):
    from fastforks.shared import libnet  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "torchvision", None)  # `import torchvision` raises ImportError
    with pytest.raises(MissingDependencyError, match="torchvision") as e:
        libnet.torchvision()
    assert isinstance(e.value, BackendUnavailableError)
    assert fastforks.MissingDependencyError is MissingDependencyError


class _Toy(model.Model):
    def __init__(self):
        pass

    def __call__(self, image, *, score_threshold: float = 0.5, max_detections: int = 10):  # noqa: ARG002
        if image == "boom":
            msg = "internal"
            raise TypeError(msg)
        return image, score_threshold


def test_keyword_typo_did_you_mean():
    m = _Toy()
    assert m("x", score_threshold=0.1) == ("x", 0.1)
    with pytest.raises(InputError, match="did you mean 'score_threshold'") as e:
        m("x", score_treshold=0.1)  # ty: ignore[unknown-argument]
    assert isinstance(e.value, TypeError)
    assert isinstance(e.value, ArgumentError)
    with pytest.raises(InputError, match="missing"):
        m()  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="internal") as e2:  # arguments bind: the model's own TypeError stays
        m("boom")
    assert not isinstance(e2.value, InputError)


def test_top_rejects_non_integer_k():
    res = ClassificationResult(
        logits=torch.tensor([[1.0, 2.0]]),
        class_names=("a", "b"),
        model="m",
        backend=model.Backend("portable", "cpu", "fp32"),
    )
    for bad in ("a", 1.5, None, True):
        with pytest.raises(InputError):
            res.top(bad)
    assert res.top(2)[0][0][0] == "b"


def test_flipped_sam_box_raises():
    fake = types.SimpleNamespace(
        prompts={"points", "boxes", "masks"}, variant=types.SimpleNamespace(ref="sam"), device=torch.device("cpu")
    )
    with pytest.raises(InputError, match="x1 >= x0"):
        sam.PromptableSegmentationModel._prompts(fake, {"input_boxes": [[50, 10, 20, 40]]})
    with pytest.raises(InputError, match="y1 >= y0"):
        sam._check_flipped(torch.tensor([[0.0, 9.0, 5.0, 3.0]]))
    sam._check_flipped(torch.tensor([[0.0, 1.0, 0.0, 1.0]]))  # a degenerate box is accepted


def test_kernel_downloads_announce_once_per_burst():
    files._burst = None
    assert files._announce(None)
    assert files._announce(3)  # first file of the burst
    assert not files._announce(3)
    assert not files._announce(3)
    assert files._announce(4)  # the next call


def test_stereo_result_depth_needs_rig():
    res = StereoResult(torch.zeros(4, 4), None, {}, "m", model.Backend("portable", "cpu", "fp32"))
    with pytest.raises(InputError, match="rig="):
        res.depth()


def test_rectified_rig_follows_aspect_preserving_resize():
    rig = g.RectifiedRig(700.0, 0.54, size=(375, 1242))
    got = resolve_geometry(types.SimpleNamespace(), rig, (188, 621), size=None, focal=None)
    assert got.size == (188, 621)
    assert got.f == pytest.approx(350.0)
    with pytest.raises(InputError, match="resized"):
        resolve_geometry(types.SimpleNamespace(), rig, (375, 1000), size=None, focal=None)
