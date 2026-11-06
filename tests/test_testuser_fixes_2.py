"""Test-user findings TH-8, TG-1..TG-3, TG-5, TG-6, TG-9, TM-6, TM-8, TM-9 (CPU, no weights)."""

from __future__ import annotations

import copy
import hashlib
import logging
import types

import pytest
import torch
from torch import nn

import fastforks
from fastforks import OptionsError, WeightsHashError
from fastforks._core import cli, files
from fastforks._core.files import FileSpec, locate
from fastforks._core.model import Model
from fastforks._core.runner import Runner
from fastforks._optimize import FastModel, patch
from fastforks.stereo import Backend


def test_version_is_only_a_string():  # TH-8
    assert isinstance(fastforks.__version__, str)
    assert not hasattr(fastforks, "version")


def _fast(mode: str, **attrs: object) -> FastModel:
    net = nn.Identity()
    for k, v in attrs.items():
        setattr(net, k, v)
    return FastModel(net, nn.Identity(), mode, {"device": "cpu"})


def test_rewritten_stem_checks_the_input_channels():  # TG-1
    fast = _fast("kernels", stem_in_chans=3)
    with pytest.raises(RuntimeError, match=r"expected input \[B, 3, H, W\], got \[1, 1, 8, 8\]"):
        fast(torch.zeros(1, 1, 8, 8))
    assert fast(torch.zeros(1, 3, 8, 8)).shape == (1, 3, 8, 8)


def test_kernels_refuse_an_input_that_needs_gradients():  # TG-3
    fast = _fast("kernels")
    x = torch.zeros(1, 3, 4, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match=r"no backward.*mode='portable'"):
        fast(x)
    with torch.no_grad():
        fast(x)
    fast(x.detach())
    _fast("portable")(x)


def test_dinov2_block_backward():  # TG-2
    from fastforks.features.models.dinov2.model import _block  # noqa: PLC0415
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    torch.manual_seed(0)
    vit = DinoVisionTransformer(
        img_size=28, patch_size=14, embed_dim=32, depth=1, num_heads=2, init_values=1.0, block_chunks=0
    ).eval()
    blk = vit.blocks[0]
    x = torch.randn(2, 5, 32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    _block(blk, x).square().sum().backward()
    blk(xr).square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=1e-5)
    with torch.no_grad():  # the inference path (in place) gives the same values
        torch.testing.assert_close(_block(blk, x.detach().clone()), blk(x.detach()), rtol=1e-4, atol=1e-5)


def test_timm_vit_portable_backward(caplog):  # TG-2 (shared timm ViT pass), TG-5 (no warning when covered)
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    m = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, img_size=32, embed_dim=32, depth=1, num_heads=2, init_values=1.0
    ).eval()
    ref = copy.deepcopy(m)
    with caplog.at_level(logging.WARNING, logger="fastforks"):
        fast = patch(m, mode="portable", device="cpu")
    assert "no covered block" not in caplog.text
    assert fast.report["vit_blocks"] == 1
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    fast(x).square().sum().backward()
    ref(xr).square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-3, atol=1e-5)


def test_patch_says_when_nothing_was_replaced(caplog):  # TG-5
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
    with caplog.at_level(logging.WARNING, logger="fastforks"):
        fast = patch(m, device="cpu")
    assert fast.report["family"] == "generic"
    assert "Sequential: no covered block found" in caplog.text


def test_module_state_dict_loads_as_weights(tmp_path):  # TG-6
    from fastforks import features  # noqa: PLC0415
    from fastforks.features.models.dinov2.model import Params, _vit  # noqa: PLC0415

    torch.manual_seed(0)
    torch.save(_vit(Params("vits", 0, False)).state_dict(), tmp_path / "hub.pth")  # noqa: FBT003
    m = features.load("dinov2-vits14", weights=tmp_path / "hub.pth", device="cpu", mode="portable")
    state = m.module().state_dict()
    assert next(iter(state)).startswith("net.net.")
    with torch.no_grad():
        next(iter(m.module().parameters())).add_(1.0)  # a "fine-tune"
    state = {k: v.clone() for k, v in m.module().state_dict().items()}
    torch.save(state, tmp_path / "tuned.pth")
    tuned = features.load("dinov2-vits14", weights=tmp_path / "tuned.pth", device="cpu", mode="portable")
    got = tuned.module().state_dict()
    assert got.keys() == state.keys()
    for k, v in state.items():
        torch.testing.assert_close(got[k], v, rtol=0, atol=0)


class _Probe(Runner):
    compiled_methods = ("__call__",)

    def __call__(self, x):
        self.compiling = torch.compiler.is_compiling()
        return x * 2


def test_kernels_call_is_a_graph_break_in_a_user_compile():  # TG-9
    ctx = types.SimpleNamespace(card=None, variant=None, device=torch.device("cpu"), user_weights=None)
    m = Model(_Probe(ctx), Backend("kernels", "cpu", "float32", ()), {})
    r = m._runner
    assert isinstance(r, _Probe)
    assert type(r).__name__ == "_Probe"
    f = torch.compile(lambda x: r(x) + 1, backend="eager")
    torch.testing.assert_close(f(torch.ones(3)), torch.full((3,), 3.0))
    assert r.compiling is False  # dynamo did not trace into the runner
    r(torch.ones(1))
    assert r.compiling is False
    plain = _Probe(ctx)
    _ = Model(plain, Backend("portable", "cpu", "float32", ()), {})
    assert type(plain) is _Probe  # portable runners stay traceable (`Model.compile`)


def _spec(data: bytes, url: str) -> FileSpec:
    return FileSpec("w-1234.pth", hashlib.sha256(data).hexdigest(), len(data), url)


def test_corrupt_cache_blob_falls_through_to_torch_hub(tmp_path, monkeypatch, caplog):  # TM-6
    monkeypatch.setenv(files.OFFLINE_ENV, "1")
    spec = _spec(b"payload", "https://dl.fbaipublicfiles.com/models/w-1234.pth")
    blob = files.blob_path(spec.sha256)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"paylaod")  # same size, wrong hash
    hub = tmp_path / "torch_home" / "hub" / "checkpoints" / "w-1234.pth"
    hub.parent.mkdir(parents=True)
    hub.write_bytes(b"payload")
    with caplog.at_level(logging.WARNING, logger="fastforks"):
        assert locate(spec, group="fam", env_var="FF_TEST_PATH") == hub
    assert "does not match its SHA-256" in caplog.text
    hub.unlink()
    with pytest.raises(WeightsHashError):  # no other copy: the corrupt blob's error, as before
        locate(spec, group="fam", env_var="FF_TEST_PATH")


def test_archive_download_announces_the_archive_size(tmp_path, monkeypatch, caplog):  # TM-8
    monkeypatch.setattr(files, "_from_archive", lambda *_a: None)
    spec = FileSpec("w.bin", "0" * 64, 3 << 20, None, member="w.bin", archive_size=420 << 20)
    with caplog.at_level(logging.WARNING, logger="fastforks"):
        files.download("https://example.com/a.zip", tmp_path / "d" / "w.bin", spec)
    assert "420 MB" in caplog.text
    assert "3 MB" not in caplog.text


def test_cache_variable_naming_a_file(tmp_path, monkeypatch, capsys):  # TM-9
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    for env in (f, f / "sub"):
        monkeypatch.setenv("FASTFORKS_CACHE", str(env))
        with pytest.raises(OptionsError, match=r"FASTFORKS_CACHE=.*is not a directory"):
            files.cache_root()
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "new" / "cache"))  # missing: fine, created on use
    assert files.cache_root() == tmp_path / "new" / "cache"
    monkeypatch.setenv("FASTFORKS_CACHE", str(f))
    monkeypatch.delenv("FASTFORKS_WEIGHTS_PATH", raising=False)
    assert cli.main(["download", "hitnet-eth3d", "--check"]) == 1
    err = capsys.readouterr().err
    assert "FASTFORKS_CACHE=" in err
    assert "Traceback" not in err
