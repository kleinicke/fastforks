"""Small review fixes: checkpoint identity, point labels, rig copies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import fastforks.stereo.geometry as g
from fastforks import InputError, OptionsError
from fastforks._core import registry
from fastforks._core.model import Backend, _checkpoint_id, check_options
from fastforks.promptable_segmentation._api import _check_point_labels

BK = Backend("portable", "cpu", "fp32")
BK2 = Backend("portable", "cpu", "fp32", (), ("f:1:2",))


def test_checkpoint_identity_follows_the_file(tmp_path):
    f = tmp_path / "w.pt"
    f.write_bytes(b"abc")
    a = _checkpoint_id((f,))
    assert a == _checkpoint_id((f,))
    f.write_bytes(b"abcd")
    b = _checkpoint_id((f,))
    assert a != b
    assert _checkpoint_id(None) == ()
    base = Backend("portable", "cpu", "fp32")
    assert base == Backend("portable", "cpu", "fp32", (), ())
    assert base != Backend("portable", "cpu", "fp32", (), a)
    assert Backend("portable", "cpu", "fp32", (), a) != Backend("portable", "cpu", "fp32", (), b)


@pytest.mark.parametrize("ok", [[1, 0, -1], np.array([1.0, 0.0]), torch.tensor([[1, 1]])])
def test_point_labels_accept_1_0_minus1(ok):
    _check_point_labels(ok)


@pytest.mark.parametrize("bad", [[0.7], [2], [1, 3], np.array([0.5])])
def test_point_labels_reject_others(bad):
    with pytest.raises(InputError, match="input_labels"):
        _check_point_labels(bad)


def test_rig_copies_and_freezes_its_arrays():
    cam = g.camera("pinhole-opencv", 64, 48, 60.0, 60.0, 31.5, 23.5, 0.0, 0.0, 0.0, 0.0)
    r, t = np.eye(3), np.array([-0.1, 0.0, 0.0])
    rig = g.StereoRig(cam, cam, r, t)
    r[0, 0] = 5.0
    t[0] = 9.0
    assert rig.R[0, 0] == 1.0
    assert rig.T[0] == -0.1
    with pytest.raises(ValueError, match="read-only"):
        rig.T[0] = 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_float_options_must_be_finite(bad):
    card = next(c for c, _ in registry.cards("detection").values() if "score_threshold" in c.options)
    with pytest.raises(OptionsError, match="finite"):
        check_options(card, {"score_threshold": bad})
    assert check_options(card, {"score_threshold": 0.5}) == {"score_threshold": 0.5}


def test_top_rejects_k_below_one():
    from fastforks.classification._api import ClassificationResult  # noqa: PLC0415

    res = ClassificationResult(
        logits=torch.tensor([[1.0, 2.0, 0.5]]),
        class_names=("a", "b", "c"),
        model="m",
        backend=Backend("portable", "cpu", "fp32"),
    )
    assert res.top(1)[0][0][0] == "b"
    with pytest.raises(InputError, match="integer >= 1"):
        res.top(0)


def test_save_rejects_unknown_kwargs(tmp_path):
    from fastforks import save  # noqa: PLC0415

    with pytest.raises(InputError, match="'colour'"):
        save(tmp_path / "a.tif", np.zeros((4, 4), np.float32), colour=1)
    with pytest.raises(InputError, match="'bits'"):
        save(tmp_path / "a.ply", np.zeros((4, 3), np.float32), bits=8)
    assert not (tmp_path / "a.tif").exists()
    assert not (tmp_path / "a.ply").exists()


def test_errors_and_results_pickle():
    import pickle  # noqa: PLC0415

    from fastforks._core.errors import CheckpointMismatchError, KernelUnavailableError  # noqa: PLC0415
    from fastforks.classification._api import ClassificationResult  # noqa: PLC0415

    e = pickle.loads(pickle.dumps(KernelUnavailableError("lib", "no gpu", model="m")))  # noqa: S301
    assert (e.library, e.reason, e.model) == ("lib", "no gpu", "m")
    assert "kernel library 'lib' unavailable" in str(e)
    c = pickle.loads(pickle.dumps(CheckpointMismatchError("w", "missing x")))  # noqa: S301
    assert (c.what, c.details) == ("w", "missing x")
    res = ClassificationResult(
        logits=torch.zeros(1, 2),
        class_names=("a", "b"),
        model="m",
        backend=Backend("portable", "cpu", "fp32", (), ("f:1:2",)),
    )
    back = pickle.loads(pickle.dumps(res))  # noqa: S301
    assert back.backend == res.backend
    assert back.class_names == ("a", "b")


def test_onnx_output_names_for_tuple_outputs():
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks._core.export import output_names  # noqa: PLC0415
    from fastforks._core.runner import Forward  # noqa: PLC0415

    model = SimpleNamespace(card=SimpleNamespace(task="detection"), variant=SimpleNamespace(outputs=("boxes",)))
    t = torch.zeros(1)
    named = Forward(torch.nn.Identity(), lambda _n, x: x, output_names=("logits", "boxes"))
    assert output_names(model, named, (t, t)) == ["logits", "boxes"]
    plain = Forward(torch.nn.Identity(), lambda _n, x: x)
    assert output_names(model, plain, (t, t, t)) == ["boxes_0", "boxes_1", "boxes_2"]
    assert output_names(model, plain, {"a": t, "b": t}) == ["a", "b"]
    assert output_names(model, plain, t) == ["boxes"]
    one = Forward(torch.nn.Identity(), lambda _n, x: x, output_names=("image_embeddings",))  # R2-2: SAM's encoder
    assert output_names(model, one, t) == ["image_embeddings"]
    same = SimpleNamespace(card=SimpleNamespace(task="restoration"), variant=SimpleNamespace(outputs=("image",)))
    assert output_names(same, plain, t) == ["image_out"]  # input `image`: the same name is a cycle for TensorRT


def test_engine_call_returns_the_first_named_output(monkeypatch):
    from fastforks._core.export import TensorRTEngine  # noqa: PLC0415

    engine = object.__new__(TensorRTEngine)
    a, b = torch.zeros(1), torch.ones(1)
    monkeypatch.setattr(TensorRTEngine, "run", lambda *_a, **_k: {"logits": a, "boxes": b})
    assert engine(a) is a
    assert engine.run(a) == {"logits": a, "boxes": b}


def test_fold_layer_scale_keeps_eval_mode():
    from fastforks.shared.depth_anything_v2 import fold_layer_scale  # noqa: PLC0415

    net = torch.nn.Module()
    from fastforks.shared.depth_anything_v2 import Block  # noqa: PLC0415

    blk = Block(dim=16, num_heads=2, init_values=1.0)
    net.blk = blk
    net.eval()
    assert fold_layer_scale(net) == 2
    assert not any(m.training for m in net.modules())
