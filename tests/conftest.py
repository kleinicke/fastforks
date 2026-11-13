from __future__ import annotations

import gc
import shutil
import sys

import pytest

from fastforks._core import registry
from fastforks._core.errors import BackendUnavailableError


@pytest.fixture(autouse=True)
def _isolated_cache(request, tmp_path, monkeypatch, _weight_stamps):
    """Every test gets an empty fastforks cache and empty Hugging Face / torch hub caches; real search paths stay as
    configured by the caller.

    `model` tests share the SHA-256 stamps of the real weights they verified (only `stamps/`, nothing else of the
    cache): each test otherwise hashed every weight file again (125 GB for the CPU goldens alone). The caches
    are removed after the test."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTFORKS_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_hub"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch_home"))
    monkeypatch.delenv("FASTFORKS_READ_CACHES", raising=False)
    registry.cards.cache_clear()
    shared = request.node.get_closest_marker("model") is not None
    if shared and _weight_stamps.is_dir():
        shutil.copytree(_weight_stamps, cache / "stamps")
    yield
    if shared and (cache / "stamps").is_dir():
        shutil.copytree(cache / "stamps", _weight_stamps, dirs_exist_ok=True)
    for d in (cache, tmp_path / "hf_hub", tmp_path / "torch_home"):  # downloaded weights: pytest keeps every tmp_path
        shutil.rmtree(d, ignore_errors=True)  # until the session ends, all models' weights overflow a CI runner's disk


@pytest.fixture(scope="session")
def _weight_stamps(tmp_path_factory):
    return tmp_path_factory.mktemp("weight_stamps") / "stamps"


@pytest.fixture(autouse=True)
def _noncommercial_weights(monkeypatch):
    """Tests may download non-commercial weights (goldens, parity); the opt-in tests remove the variable."""
    monkeypatch.setenv(registry.NONCOMMERCIAL_ENV, "1")


@pytest.fixture(autouse=True)
def _no_real_hf_login(tmp_path, monkeypatch):
    """No test sees the machine's Hugging Face login: no token in the environment, none in the token file."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "hf_home" / "token"))
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError:
        return
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)


def _no_network(*_a, **_k):
    msg = "hf_hub_download must be stubbed by the test"
    raise AssertionError(msg)


@pytest.fixture
def card_min_size(monkeypatch):
    """Check only the card's `io.min_size`, not the probed minimum of `size_range()`: for tests that push a few-pixel
    image through a fake runner. (`test_input_below_the_documented_minimum_raises` covers the real check.)"""
    from fastforks._core import model  # noqa: PLC0415

    def check(self, x):
        if min(x.shape[-2:]) < self.variant.io.min_size:
            msg = "below the card's minimum"
            raise model.InputError(msg)

    monkeypatch.setattr(model.Model, "_check_size", check)


def pytest_addoption(parser):
    parser.addoption(
        "--record",
        action="store_true",
        help="GPU goldens: record the kernel builds of passing kernels runs as [validated] in the family golden.toml",
    )
    parser.addoption(
        "--slow", action="store_true", help="also run the tests marked `slow` (> 1 s; release suite, changed code)"
    )


_OPTIONAL = ("timm", "torchvision", "transformers", "scipy")  # the opt-in `vision` group / extras


def _skip_missing_optional():
    """A test whose model needs an optional library that is not installed is skipped, not failed."""
    try:
        return (yield)
    except BackendUnavailableError as e:
        cause = e.__cause__
        if isinstance(cause, ImportError) and (cause.name or "").split(".")[0] in _OPTIONAL:
            pytest.skip(f"optional dependency missing: {cause.name}")
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call():
    return (yield from _skip_missing_optional())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup():  # module-scoped fixtures load their models here
    return (yield from _skip_missing_optional())


def _sweep_ref(item: pytest.Item) -> str | None:
    """The `ref` parameter of a test in a `LOAD_ONCE` module (it loads its model with `test_golden.sweep_model`)."""
    callspec = getattr(item, "callspec", None)
    if callspec is None or not getattr(getattr(item, "module", None), "LOAD_ONCE", False):
        return None
    return callspec.params.get("ref")


def pytest_collection_modifyitems(config, items):
    """Tests marked `slow` are skipped without `--slow`. The ref-parametrized tests of `LOAD_ONCE` modules run as one
    block, grouped by ref (in collection order within a ref), at the place of the first of them: `sweep_model` then
    loads each model once. Tests do not depend on order."""
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="slow (> 1 s): --slow runs it")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
    block = [i for i in items if _sweep_ref(i) is not None]
    if block:
        first = items.index(block[0])
        rest = [i for i in items if _sweep_ref(i) is None]
        items[:] = rest[:first] + sorted(block, key=_sweep_ref) + rest[first:]


def pytest_runtest_teardown(item, nextitem):
    """Drop the sweep model when the next test does not use it (RAM for the next model)."""
    sweep = getattr(sys.modules.get("test_golden"), "SWEEP", None)
    if sweep and (nextitem is None or _sweep_ref(nextitem) not in sweep):
        sweep.clear()
        gc.collect()


@pytest.fixture(autouse=True)
def _local_urls(monkeypatch):
    """The suite's local-server and `file://` fixtures need the insecure-URL override (off by default in the library;
    `test_security.py` turns it back off)."""
    from fastforks._core import files  # noqa: PLC0415

    monkeypatch.setattr(files, "_INSECURE_OK", True)
