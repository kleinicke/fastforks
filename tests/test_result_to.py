"""`.to(device)` / `.cpu()` on every result type: same type, all tensors moved, methods keep working."""

import copy
import dataclasses
import pickle

import numpy as np
import pytest
import torch

from fastforks._core.instances import Instances
from fastforks._core.model import Backend, ReadOnlyDict
from fastforks.classification import ClassificationResult
from fastforks.depth import DepthResult
from fastforks.features import FeaturesResult
from fastforks.multiview import MultiViewResult
from fastforks.promptable_segmentation import PromptableSegmentationResult
from fastforks.segmentation import SegmentationResult
from fastforks.stereo import StereoResult

B = None  # backend is not touched by the methods


def _results():
    t = torch.rand
    depth = DepthResult(
        t(4, 5), t(4, 5), None, torch.eye(3), None, t(4, 5) > 0.5, None, {"sky": t(4, 5)}, "metric", "m@1", B
    )
    mv = MultiViewResult(
        t(1, 4, 5), None, None, None, torch.eye(3).expand(1, 3, 3), None, {"r": t(1, 4, 5)}, (0,), "metric", "m@1", B
    )
    st = StereoResult(t(4, 5), t(4, 5), {"occlusion": t(4, 5)}, "m@1", B)
    feats = FeaturesResult(t(7, 3), t(3), (None, t(7, 3)), (1,), (2, 3), 1, 0, "m@1", B)
    cls = ClassificationResult(t(10), tuple(f"c{i}" for i in range(10)), "m@1", B)
    seg = SegmentationResult(torch.zeros(4, 5, dtype=torch.long), t(3, 4, 5), ("a", "b", "c"), "m@1", B)
    prom = PromptableSegmentationResult(t(1, 3, 4, 5) > 0.5, t(1, 3), t(1, 3, 2, 2), "m@1", B)
    inst = Instances(t(3, 4), t(3), torch.arange(3), t(3, 4, 5) > 0.5, ("a", "b", "c"), "m@1", B)
    return {r.__class__.__name__: r for r in (depth, mv, st, feats, cls, seg, prom, inst)}


def _tensors(r):
    out = []
    for f in dataclasses.fields(r):
        v = getattr(r, f.name)
        if f.name == "extras":
            out += list(v.values())
        elif isinstance(v, tuple):
            out += [x for x in v if isinstance(x, torch.Tensor)]
        elif isinstance(v, torch.Tensor):
            out.append(v)
    return out


@pytest.mark.parametrize("name", list(_results()))
def test_to_moves_every_tensor_and_keeps_other_fields(name):
    r = _results()[name]
    m = r.to("meta")
    assert type(m) is type(r)
    assert all(t.device.type == "meta" for t in _tensors(m))
    assert len(_tensors(m)) == len(_tensors(r))
    for f in dataclasses.fields(r):
        v = getattr(r, f.name)
        if not isinstance(v, torch.Tensor) and f.name != "extras" and not (isinstance(v, tuple) and v and None in v):
            assert getattr(m, f.name) == v
    assert r.to(torch.device("cpu")).numpy().keys() == r.numpy().keys()


@pytest.mark.parametrize("name", list(_results()))
def test_cpu_same_device_is_not_copied(name):
    r = _results()[name]
    for moved in (r.cpu(), r.to("cpu")):
        assert type(moved) is type(r)
        for a, b in zip(_tensors(r), _tensors(moved), strict=True):
            if name != "Instances":  # Instances.cpu() keeps its pinned staging for device tensors only
                assert a is b


def test_none_entries_and_extras_survive():
    r = _results()
    m = r["FeaturesResult"].to("meta")
    assert m.hidden_states[0] is None
    assert m.hidden_states[1].device.type == "meta"
    assert r["DepthResult"].to("meta").extras["sky"].device.type == "meta"
    assert r["DepthResult"].to("meta").points is None


def test_methods_work_on_the_moved_result(tmp_path):
    r = _results()
    cpu = {k: v.cpu() for k, v in r.items()}
    assert cpu["ClassificationResult"].top(3) == r["ClassificationResult"].top(3)
    assert len(cpu["Instances"]) == 3
    assert cpu["Instances"][:2].labels.tolist() == [0, 1]
    assert cpu["Instances"].to_list()[0]["class_name"] == "a"
    assert cpu["FeaturesResult"].cls_token.shape == (3,)
    assert cpu["FeaturesResult"].feature_map(1).shape == (3, 2, 3)
    assert cpu["SegmentationResult"].labels.shape == (4, 5)
    assert cpu["PromptableSegmentationResult"].masks.dtype is torch.bool
    for k in ("DepthResult", "StereoResult", "MultiViewResult"):
        a, b = tmp_path / f"{k}_a.npy", tmp_path / f"{k}_b.npy"
        r[k].save(a)
        cpu[k].save(b)
        assert a.read_bytes() == b.read_bytes()
        for key, arr in cpu[k].numpy().items():
            np.testing.assert_array_equal(arr, r[k].numpy()[key])


@pytest.mark.gpu
def test_cuda_result_cpu_equals_original_and_saves_identically(tmp_path):
    cuda = {k: v.to("cuda") for k, v in _results().items()}
    for r in cuda.values():
        c = r.cpu()
        assert type(c) is type(r)
        for a, b in zip(_tensors(r), _tensors(c), strict=True):
            assert b.device.type == "cpu"
            assert torch.equal(a.cpu(), b)
    for k in ("DepthResult", "StereoResult", "MultiViewResult"):
        a, b = tmp_path / f"{k}_a.npy", tmp_path / f"{k}_b.npy"
        cuda[k].save(a)
        cuda[k].cpu().save(b)
        assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("name", list(_results()))
def test_results_pickle_copy_and_save(name, tmp_path):
    """Results leave worker processes, get cached and `torch.save`d: `options` is a read-only mapping that pickles."""
    backend = Backend("portable", "cpu", "float32", (), ("w.pt:1:2",))
    r = dataclasses.replace(_results()[name], backend=backend, options=ReadOnlyDict({"iters": 7}))
    torch.save(r, tmp_path / "r.pt")
    copies = (
        pickle.loads(pickle.dumps(r)),  # noqa: S301 - our own bytes
        copy.deepcopy(r),
        torch.load(tmp_path / "r.pt", weights_only=False),
    )
    for c in copies:
        assert type(c) is type(r)
        assert c.backend == backend
        assert isinstance(c.options, ReadOnlyDict)
        assert c.options == {"iters": 7}
        for a, b in zip(_tensors(c), _tensors(r), strict=True):
            torch.testing.assert_close(a, b)


def test_read_only_options():
    o = ReadOnlyDict({"iters": 7}, threshold=0.5)
    assert o == {"iters": 7, "threshold": 0.5}
    assert repr(o) == "{'iters': 7, 'threshold': 0.5}"
    with pytest.raises(TypeError):
        o["iters"] = 8  # type: ignore[index]
    with pytest.raises(AttributeError):
        o.x = 1  # type: ignore[attr-defined]
