"""Error paths a user hits: wrong argument types, unusable local weights, missing kernel libraries."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import logging
import socket
import types

import numpy as np
import pytest
import torch

from fastforks import InputError, ModelNotFoundError, OptionsError, WeightsDownloadError, stereo
from fastforks._core import files, kernels, registry
from fastforks._core import model as core
from fastforks._core.errors import KernelUnavailableError, WeightsError
from fastforks._core.files import FileSpec, locate
from fastforks.kernels import cudnn_hacc

ENV = "FF_TEST_PATH"


@pytest.mark.parametrize("model", [3, None, ["las2-s"]])
def test_model_id_must_be_a_string(model):
    with pytest.raises(ModelNotFoundError, match="model must be an id string"):
        stereo.load(model)


@pytest.mark.parametrize("options", [[("iters", 3)], "iters=3"])
def test_options_must_be_a_mapping(options):
    with pytest.raises(OptionsError, match="options must be a dict"):
        stereo.load("crestereo-eth3d", options=options)


def test_unusable_local_copy_is_named(tmp_path, monkeypatch, caplog):
    """A same-named copy in the search path that does not verify is reported (log + error), not skipped silently."""
    good = b"weights"
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "w.bin").write_bytes(b"truncated")
    monkeypatch.setenv(ENV, str(tmp_path / "w"))
    spec = FileSpec("w.bin", hashlib.sha256(good).hexdigest(), len(good), (tmp_path / "offline").as_uri())
    with caplog.at_level(logging.WARNING), pytest.raises(WeightsDownloadError, match=r"do not verify: .*w\.bin"):
        locate(spec, group="fam", env_var=ENV)
    assert "not using" in caplog.text
    assert "w.bin" in caplog.text
    (tmp_path / "src").write_bytes(good)  # the download still wins when it works
    assert locate(dataclasses.replace(spec, url=(tmp_path / "src").as_uri()), group="fam", env_var=ENV).is_file()


@pytest.mark.parametrize(("size", "want"), [(46320, "45 KB"), (3374674, "3 MB"), (107453000, "102 MB")])
def test_sizes_in_messages(size, want):
    assert files._human_size(size) == want


@pytest.mark.gpu
def test_kernels_without_kernel_libraries(tmp_path, monkeypatch, caplog):
    """Empty kernel search path and cache, no network: kernels raises naming the remedy, auto says why it falls back."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    card, variant = registry.resolve_any("hitnet-eth3d")
    try:  # the portable fallback loads the weights, and the network is off below
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"hitnet-eth3d weights not available locally: {e}")
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv(kernels.KERNEL_PATH_ENV, str(tmp_path / "empty"))
    for v in ("FASTFORKS_OFFLINE", "HF_HUB_OFFLINE"):  # offline mode words the error differently
        monkeypatch.delenv(v, raising=False)

    def offline(*_a, **_k):
        raise OSError(101, "Network is unreachable")

    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(core, "_KERNELS_DEVICE", [])
    with pytest.raises(KernelUnavailableError, match=r"mode='kernels'.*cannot download.*fastforks download --kernels"):
        stereo.load("hitnet-eth3d", mode="kernels")
    with caplog.at_level(logging.INFO, logger="fastforks"):
        m = stereo.load("hitnet-eth3d", mode="auto")
    assert m.backend.mode == "portable"
    assert "kernels unavailable" in caplog.text
    x = np.zeros((64, 64, 3), np.uint8)
    assert m(x, x).disparity.shape == (64, 64)


def test_cudnn_hacc_errors_name_the_model(monkeypatch):
    """cudnn_hacc raises `KernelUnavailableError`, so `load` prefixes the model like for the binary libraries."""
    card = stereo.model_card("las2-s")
    assert "cudnn_hacc" in card.kernel_libs
    for name in card.kernel_libs:  # every other library loads, so cudnn_hacc decides
        if name != "cudnn_hacc":
            lib = importlib.import_module(f"fastforks.kernels.{name}")
            monkeypatch.setattr(lib, "load", lambda _d: types.SimpleNamespace(tag="stub"))
    monkeypatch.setattr(core, "_KERNELS_DEVICE", [])
    with pytest.raises(
        KernelUnavailableError, match=r"^las2-s@1 \(mode='kernels'\): kernel library 'cudnn_hacc'.*CUDA"
    ) as e:
        stereo.load("las2-s", mode="kernels", device="cpu")
    assert e.value.library == "cudnn_hacc"

    def no_cudnn() -> None:
        msg = "libcudnn.so.9: cannot open shared object file"
        raise OSError(msg)

    monkeypatch.setattr(cudnn_hacc._backend, "version", lambda: cudnn_hacc._backend.MIN_VERSION)  # no cuDNN on macOS
    monkeypatch.setattr(cudnn_hacc._backend, "create_handle", no_cudnn)
    cudnn_hacc._instance.cache_clear()
    try:
        with pytest.raises(KernelUnavailableError, match=r"backend API is not usable: libcudnn\.so\.9"):
            cudnn_hacc.CudnnHalfConv()
    finally:
        cudnn_hacc._instance.cache_clear()
    monkeypatch.setattr(cudnn_hacc._backend, "version", lambda: 90100)  # cuDNN 9.1.0: older than MIN_VERSION
    with pytest.raises(KernelUnavailableError, match=r"needs cuDNN >= 9\.10\.0, torch loaded cuDNN 9\.1\.0"):
        cudnn_hacc.CudnnHalfConv()


@pytest.mark.parametrize("model", ["s2m2-s", "hitnet-eth3d"])
def test_input_below_the_documented_minimum_raises(model):
    """The probed minimum (`size_range().reject_below`) is enforced, not only the card's `io.min_size`."""
    m = stereo.load(model, device="cpu")
    n = m.size_range().reject_below
    assert n > 1
    x = torch.rand(3, n - 1, n - 1)
    with pytest.raises(InputError, match=rf"at least {n} px"):
        m(x, x)
