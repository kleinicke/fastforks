"""The restricted checkpoint reader: reads real checkpoint shapes, refuses everything else."""

from __future__ import annotations

import collections
import pickle
import sys
import types
import zipfile

import numpy as np
import pytest
import torch

from fastforks._core.checkpoint import NUMPY_ARRAYS, Inert, load_state_dict, restricted_load
from fastforks._core.errors import WeightsError


class _Bad:
    """Pickles as `OrderedDict(3)`: an allowed global with arguments it cannot take."""

    def __reduce__(self):
        return (collections.OrderedDict, (3,))


class _Payload:
    """Pickles as `exec(<code>)`: what a hostile checkpoint would do."""

    def __init__(self, code: str) -> None:
        self.code = code

    def __reduce__(self):
        return (exec, (self.code,))


# Both readers: torch's `weights_only` one (no extra globals) and ours (any extra global).
_EXTRAS = [None, {("omegaconf.dictconfig", "DictConfig"): Inert}]


@pytest.mark.parametrize("extra", _EXTRAS)
@pytest.mark.parametrize("zip_format", [True, False])
def test_exec_payload_is_refused_and_does_nothing(tmp_path, zip_format, extra):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.pth"
    torch.save(
        {"model": {}, "x": _Payload(f"open({str(marker)!r}, 'w').write('x')")},
        path,
        _use_new_zipfile_serialization=zip_format,
    )
    with pytest.raises(WeightsError, match=r"not on the allow-list|cannot read the checkpoint safely"):
        restricted_load(path, extra)
    assert not marker.exists()


@pytest.mark.parametrize("extra", _EXTRAS)
def test_other_builtins_and_unknown_globals_are_refused(tmp_path, extra):
    # torch's `weights_only` reader allows the harmless `Counter` / `slice`; our table allows neither.
    harmless = (collections.Counter(a=1), slice(1, 2)) if extra else ()
    for obj in (_Payload("1"), *harmless):
        torch.save({"o": obj}, tmp_path / "x.pth")
        with pytest.raises(WeightsError):
            restricted_load(tmp_path / "x.pth", extra)


@pytest.mark.parametrize("zip_format", [True, False])
def test_state_dict_key_and_strip(tmp_path, zip_format):
    w = torch.arange(4.0)
    path = tmp_path / "ckpt.pth"
    torch.save(
        {"model": {"module.a": w, "module.b": w + 1}, "epoch": 3}, path, _use_new_zipfile_serialization=zip_format
    )
    sd = load_state_dict(path, key="model")
    assert list(sd) == ["a", "b"]
    assert torch.equal(sd["b"], w + 1)
    assert list(load_state_dict(path, key="model", strip="")) == ["module.a", "module.b"]
    with pytest.raises(WeightsError, match=r"no entry.*torch\.save"):
        load_state_dict(path, key="state_dict")
    with pytest.raises(WeightsError, match="expected a state dict"):
        load_state_dict(path, key="epoch")


@pytest.mark.parametrize(("zip_format", "mmap"), [(True, True), (False, False)])
def test_zip_files_are_mmapped_legacy_files_are_read(tmp_path, monkeypatch, zip_format, mmap):
    path = tmp_path / "ckpt.pth"
    torch.save({"a": torch.zeros(4)}, path, _use_new_zipfile_serialization=zip_format)
    seen = {}
    real = torch.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert restricted_load(path)["a"].shape == (4,)
    assert seen["mmap"] is mmap


def test_extra_globals_and_inert_stubs(tmp_path, monkeypatch):
    class Config:
        """Stands for a saved config class the checkpoint carries."""

    mod = types.ModuleType("some_config")
    mod.__dict__["Config"] = Config
    Config.__module__, Config.__qualname__ = "some_config", "Config"
    monkeypatch.setitem(sys.modules, "some_config", mod)
    path = tmp_path / "ckpt.pth"
    torch.save({"model": {"w": torch.ones(2)}, "cfg": Config()}, path)
    monkeypatch.delitem(sys.modules, "some_config")
    with pytest.raises(WeightsError, match=r"some_config\.Config"):
        restricted_load(path)
    out = restricted_load(path, {("some_config", "Config"): Inert})
    assert isinstance(out["cfg"], Inert)
    assert torch.equal(out["model"]["w"], torch.ones(2))


def test_plain_pickle_of_numpy_arrays(tmp_path):
    """MegEngine `.mge` files: a pickled dict of numpy arrays, read without torch's reader; arrays become tensors."""
    path = tmp_path / "ckpt.mge"
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    path.write_bytes(pickle.dumps({"state_dict": collections.OrderedDict(a=w)}, protocol=4))
    with pytest.raises(WeightsError, match="plain pickle"):
        restricted_load(path)
    sd = load_state_dict(path, key="state_dict", extra=NUMPY_ARRAYS)
    assert torch.equal(sd["a"], torch.from_numpy(w))


def test_plain_pickle_exec_payload_is_refused(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.mge"
    path.write_bytes(pickle.dumps({"x": _Payload(f"open({str(marker)!r}, 'w').write('x')")}, protocol=4))
    with pytest.raises(WeightsError, match="not on the allow-list"):
        restricted_load(path, NUMPY_ARRAYS)
    assert not marker.exists()


def test_plain_pickle_arrays_are_checked(tmp_path):
    """Big-endian arrays are byte-swapped; object arrays and malformed states raise `WeightsError`."""
    path = tmp_path / "ckpt.mge"
    w = np.arange(6, dtype=">f4")
    path.write_bytes(pickle.dumps({"a": w}, protocol=4))
    sd = load_state_dict(path, extra=NUMPY_ARRAYS)
    assert torch.equal(sd["a"], torch.arange(6, dtype=torch.float32))
    path.write_bytes(pickle.dumps({"a": np.array([1, "x"], dtype=object)}, protocol=4))
    with pytest.raises(WeightsError, match=r"ckpt\.mge: a has dtype object"):
        load_state_dict(path, extra=NUMPY_ARRAYS)

    path.write_bytes(pickle.dumps({"a": _Bad()}, protocol=4))  # an allowed global called with bad arguments
    with pytest.raises(WeightsError, match="malformed checkpoint"):
        restricted_load(path, NUMPY_ARRAYS)


@pytest.mark.parametrize("name", ["model.safetensors", "ba73d3de"])
def test_safetensors_detected_by_content(tmp_path, name):
    """The download cache names weight files by hash, so safetensors files are recognized without their suffix."""
    from safetensors.torch import save_file  # noqa: PLC0415 - optional for the other tests

    path = tmp_path / name
    save_file({"a": torch.arange(4.0)}, path)
    assert torch.equal(load_state_dict(path)["a"], torch.arange(4.0))


def test_npz_detected_by_content_without_pickles(tmp_path):
    """JAX checkpoints come as `.npz` (TAPNext): arrays become tensors; object arrays (pickles) are refused."""
    path = tmp_path / "ba73d3de"  # the download cache's name
    with path.open("wb") as f:
        np.savez(f, **{"a/kernel": np.arange(6, dtype=np.float32).reshape(2, 3)})
    assert torch.equal(load_state_dict(path, strip="")["a/kernel"], torch.arange(6.0).reshape(2, 3))
    with path.open("wb") as f:
        np.savez(f, a=np.array([1, "x"], dtype=object))
    with pytest.raises(WeightsError, match="ba73d3de: Object arrays cannot be loaded"):
        load_state_dict(path)


class _Scripted(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.head = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2))
        self.scale16 = torch.nn.Parameter(torch.randn(5, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).mean((2, 3))) + self.scale16.sum()


@pytest.mark.filterwarnings("ignore:`torch.jit.[a-z]*` is deprecated:FutureWarning")  # only to write the fixture
@pytest.mark.filterwarnings("ignore:`torch.jit.[a-z]*` is not supported in Python 3.14:FutureWarning")  # same
def test_torchscript_archive_tensors_without_running_code(tmp_path, monkeypatch):
    torch.manual_seed(0)
    path = tmp_path / "net.pt2"
    module = torch.jit.script(_Scripted().eval())
    torch.jit.save(module, path)
    monkeypatch.setattr(torch.jit, "load", lambda *_a, **_k: pytest.fail("TorchScript code was loaded"))
    sd = load_state_dict(path)
    ref = module.state_dict()
    assert sorted(sd) == sorted(ref)
    for k, v in ref.items():
        assert sd[k].dtype == v.dtype
        assert torch.equal(sd[k], v), k


def test_torchscript_archive_with_foreign_globals_is_refused(tmp_path):
    path = tmp_path / "bad.pt2"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("bad/data.pkl", pickle.dumps(_Payload("raise SystemExit")))
        z.writestr("bad/code/__torch__/m.py", "")
    with pytest.raises(WeightsError, match="allow-list"):
        load_state_dict(path)
