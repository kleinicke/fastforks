"""Every renamed 0.1.1 name still works and warns (`FutureWarning` naming the replacement and the removal date)."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
from tests.test_tasks import Concept, fake  # noqa: F401 - the fixture

import fastforks
from fastforks import InputError, OptionsError, classification, promptable_segmentation, stereo
from fastforks._core import registry
from fastforks._core.arguments import REMOVAL
from fastforks.stereo._runner import Runner

pytestmark = pytest.mark.usefixtures("card_min_size")  # tiny images through fake runners


@dataclasses.dataclass(frozen=True)
class _P:
    iters: int


class _Iters(Runner):
    """Disparity = the `iters` option."""

    Params = _P

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        return {"disparity": left[:, 0] * 0 + float(self.ctx.params.iters)}


@pytest.fixture
def iters_runner(monkeypatch):
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": _Iters}))


def _warns(new: str):
    return pytest.warns(FutureWarning, match=rf"renamed to {new}.*removed after {REMOVAL}")


def test_load_options_dict(iters_runner):
    img = np.zeros((4, 5, 3), np.uint8)
    for load in (stereo.load, fastforks.load):
        with _warns(r"load\(\.\.\., \*\*options\)"):
            m = load("raft-stereo-sceneflow", device="cpu", options={"iters": 7})
        assert m.options == {"iters": 7}
        assert float(m(img, img).disparity[0, 0]) == 7
    with _warns(r"load\(\.\.\., \*\*options\)"), pytest.raises(OptionsError, match="load\\(\\) argument"):
        stereo.load("raft-stereo-sceneflow", device="cpu", options={"mode": "portable"})
    with _warns(r"load\(\.\.\., \*\*options\)"), pytest.raises(OptionsError, match="also given"):
        stereo.load("raft-stereo-sceneflow", device="cpu", iters=3, options={"iters": 7})


def test_stereo_result_types():
    from fastforks.stereo import calibrated, pipeline  # noqa: PLC0415

    for module, old in ((stereo, "PairResult"), (stereo, "StereoPairResult"), (pipeline, "PairResult")):
        with _warns(r"fastforks\.stereo\.StereoResult"):
            assert getattr(module, old) is stereo.StereoResult
    with _warns(r"fastforks\.stereo\.StereoResult"):
        assert calibrated.StereoPairResult is stereo.StereoResult
    with pytest.raises(AttributeError):
        _ = stereo.NoSuchName


def test_stereo_result_keeps_maps():
    """`replace` / `copy` keep the maps; the 0.1.1 constructor (positional or keyword) still builds a result."""
    import copy  # noqa: PLC0415

    import torch  # noqa: PLC0415

    d = torch.ones(4, 5)
    with _warns(r"a result from a model call"):
        old = stereo.StereoResult(d, None, {"occlusion": d}, "m@1", "portable")
    with _warns(r"a result from a model call"):
        kw = stereo.StereoResult(disparity=d, confidence=None, extras={}, model="m@1", backend="portable")  # ty: ignore[unknown-argument]
    assert old.disparity is not None
    assert old.extras["occlusion"].shape == (4, 5)
    assert kw.model == "m@1"
    for new in (dataclasses.replace(old, options={"a": 1}), copy.copy(old), old.cpu()):
        assert new.disparity is not None
        assert new.disparity.shape == (4, 5)
    assert dataclasses.replace(old, options={"a": 1}).options == {"a": 1}


def test_from_import_warns_once():
    with pytest.warns(FutureWarning) as seen:
        exec("from fastforks.stereo import StereoPairResult")  # noqa: S102 - the import statement is the subject
    assert len(seen) == 1


def test_stereo_pipeline_pairviews():
    from fastforks.stereo import calibrated, pipeline  # noqa: PLC0415

    with _warns(r"fastforks\.stereo\.calibrated\.PairViews"):
        assert pipeline.PairViews is calibrated.PairViews


def test_write_disparity(tmp_path):
    with _warns(r"StereoResult\.save"):
        write = stereo.write_disparity
    write(tmp_path / "d.png", np.ones((3, 4), np.float32))
    assert (tmp_path / "d.png").stat().st_size > 0


def test_classification_labels(monkeypatch):
    from fastforks._core import model as core  # noqa: PLC0415

    seen = {}
    monkeypatch.setattr(core, "load_model", lambda *_a, **_k: types.SimpleNamespace(_class_names=None))
    names = [f"c{i}" for i in range(1000)]
    for load in (classification.load, fastforks.load):
        with _warns(r"load\(\.\.\., class_names=\)"):
            m = load("resnet50", device="cpu", labels=names)
        seen[load] = m._class_names
    assert all(v == tuple(names) for v in seen.values())
    with _warns(r"load\(\.\.\., class_names=\)"), pytest.raises(OptionsError, match="pass only class_names"):
        classification.load("resnet50", device="cpu", labels=names, class_names=names)


def test_sam_threshold(fake):  # noqa: F811 - the fixture
    fake("promptable_segmentation", Concept, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    img = np.zeros((8, 12, 3), np.uint8)
    with _warns("score_threshold="):
        r = m(img, text="coin", threshold=0.1)
    assert len(r) == 3
    assert r.options["score_threshold"] == 0.1
    with _warns("score_threshold="), pytest.raises(InputError, match="pass only score_threshold"):
        m(img, text="coin", threshold=0.1, score_threshold=0.2)


@pytest.mark.parametrize("task", registry.TASKS)
def test_shared_types_in_tasks(task):
    module = getattr(fastforks, task)
    for name in ("Backend", "ImageLike", "ModelCard", "SizeRange", "TensorRTEngine"):
        assert name not in module.__all__
        with _warns(rf"fastforks\.{name}"):
            assert getattr(module, name) is getattr(fastforks, name)


def test_export_size(fake):  # noqa: F811 - the fixture
    fake("promptable_segmentation", Concept, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    with _warns(r"model\.size_range\(\)\.export_size"):
        assert m.export_size() == m.size_range().export_size is not None


def test_load_errors_name_the_path_rule(tmp_path, monkeypatch):
    from fastforks import ArgumentError, ModelNotFoundError  # noqa: PLC0415
    from fastforks._core.errors import UnknownOptionError  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights_dir").mkdir()
    with pytest.raises(ModelNotFoundError, match=r"pass './weights_dir'"):
        fastforks.load("weights_dir")
    with pytest.raises(ModelNotFoundError, match=r"no such file or directory: '/no/such/dir'.*is a path"):
        fastforks.load("/no/such/dir")
    # one error class for a typo among the load arguments or the options
    for load in (stereo.load, fastforks.load):
        with pytest.raises(UnknownOptionError, match="unknown option"):
            load("raft-stereo-sceneflow", device="cpu", iter=7)
    with pytest.raises(ArgumentError):
        stereo.load("raft-stereo-sceneflow", device="cpu", mdoe="portable")
    with pytest.raises(OptionsError):
        stereo.load("raft-stereo-sceneflow", device="cpu", mdoe="portable")
