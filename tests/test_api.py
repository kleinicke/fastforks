from __future__ import annotations

import dataclasses
import logging
import types

import numpy as np
import pytest
import torch

from fastforks import BackendUnavailableError, CardError, InputError, KernelUnavailableError, OptionsError, stereo
from fastforks._core import model as core_model
from fastforks._core import registry
from fastforks.stereo import Backend
from fastforks.stereo._runner import Runner

pytestmark = pytest.mark.usefixtures("card_min_size")  # tiny images through fake runners


@dataclasses.dataclass(frozen=True)
class Params:
    feature_channels: int
    num_transformer: int
    dim_expansion: int
    use_positivity: bool
    refine_iter: int


class Dummy(Runner):
    """Disparity = 10 x (left - right) on the red channel; returns padded maps like a real runner."""

    Params = Params
    calls: list[str] = []  # noqa: RUF012 - test log

    def __init__(self, ctx):
        super().__init__(ctx)
        Dummy.calls.append(ctx.mode)

    def __call__(self, left, right):
        assert left.dtype == torch.float32
        assert left.shape[-1] % 32 == 0  # the core padded it
        return {
            "disparity": (left - right)[:, 0] / 25.5,
            "confidence": torch.ones_like(left[:, 0]),
            "occlusion": left[:, 1],
        }


class Narrow(Dummy):
    batch = None

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        return {"disparity": left[:, 0, :2]}  # too few rows


@pytest.fixture
def fake_runner(monkeypatch):
    runners = {"portable": Dummy, "kernels": Dummy}
    Dummy.calls = []
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS=runners))
    return runners


def test_single_and_batch(fake_runner):
    m = stereo.load("s2m2-s", device="cpu")
    left = np.full((4, 5, 3), 200, np.uint8)
    right = np.full((4, 5, 3), 100, np.uint8)
    r = m(left, right)
    assert r.disparity.shape == (4, 5)
    torch.testing.assert_close(r.disparity, torch.full((4, 5), 1000 / 255))
    assert r.confidence is not None
    assert r.confidence.shape == (4, 5)
    assert set(r.extras) == {"occlusion"}
    assert r.model == "s2m2-s@1"
    assert r.backend == Backend("portable", "cpu", "float32", ())
    assert r.options == {}
    rb = m(np.stack([left, right]), np.stack([right, left]))
    assert rb.disparity.shape == (2, 4, 5)
    torch.testing.assert_close(rb.disparity[1], -r.disparity)


def test_shape_mismatch(fake_runner):
    m = stereo.load("s2m2-s", device="cpu")
    with pytest.raises(InputError, match="differ in shape"):
        m(np.zeros((4, 5, 3), np.uint8), np.zeros((4, 6, 3), np.uint8))


def test_output_shape_is_checked(fake_runner):
    fake_runner["portable"] = Narrow
    m = stereo.load("s2m2-s", device="cpu")
    with pytest.raises(CardError, match="expected"):
        m(np.zeros((4, 5, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))


def test_kernels_fails_fast_on_cpu(fake_runner):
    with pytest.raises(KernelUnavailableError, match=r"needs a CUDA device.*use mode='auto'"):
        stereo.load("s2m2-s", mode="kernels", device="cpu")
    assert Dummy.calls == []  # nothing built, no weights touched


def test_auto_logs_choice(fake_runner, caplog):
    with caplog.at_level(logging.INFO, logger="fastforks.stereo"):
        m = stereo.load("s2m2-s", mode="auto", device="cpu")
    assert m.backend.mode == "portable"
    records = [r for r in caplog.records if "using portable" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.INFO]  # once per load; WARNING only for a kernels failing on CUDA
    assert "kernels unavailable" in caplog.text
    assert "needs a CUDA device" in caplog.text


def test_unknown_mode(fake_runner):
    with pytest.raises(OptionsError, match="mode must be 'portable', 'kernels' or 'auto', got 'turbo'"):
        stereo.load("s2m2-s", mode="turbo", device="cpu")  # runtime check


def test_batch_of_one_keeps_batch_dim(fake_runner):
    m = stereo.load("s2m2-s", device="cpu")
    one = np.full((1, 4, 5, 3), 200, np.uint8)
    assert m(one, one).disparity.shape == (1, 4, 5)
    assert m(one[0], one[0]).disparity.shape == (4, 5)


class Mobile(Dummy):
    """A MobileNet-like block (depthwise + pointwise conv, ReLU6) on the left image; records the autocast state."""

    seen: list[bool] = []  # noqa: RUF012 - test log

    def __init__(self, ctx):
        super().__init__(ctx)
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3, padding=1, groups=3),
            torch.nn.Conv2d(3, 8, 1),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(8, 1, 1),
        ).eval()

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        Mobile.seen.append(torch.is_autocast_enabled("cpu"))
        return {"disparity": self.net(left * 0.37)[:, 0]}


def test_inherited_autocast_does_not_change_results(monkeypatch):
    """A caller's CPU bf16 autocast is off inside the call: the float32 result equals the one without it."""
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Mobile}))
    m = stereo.load("s2m2-s", device="cpu")
    left = np.random.default_rng(0).integers(0, 256, (32, 48, 3), np.uint8)
    Mobile.seen = []
    plain = m(left, left).disparity
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = m(left, left).disparity
        assert torch.is_autocast_enabled("cpu")  # the caller's state is back after the call
    assert Mobile.seen == [False, False]
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(mixed, plain, rtol=0, atol=0)


def test_options(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class P:
        iters: int

    class Iters(Runner):
        Params = P

        def __call__(self, left, right):  # noqa: ARG002 - test stub
            return {"disparity": torch.full_like(left[:, 0], float(self.ctx.params.iters))}

    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Iters}))
    img = np.zeros((4, 5, 3), np.uint8)
    r = stereo.load("raft-stereo-sceneflow", device="cpu")(img, img)
    assert r.options == {}
    assert float(r.disparity[0, 0]) == 32
    r = stereo.load("raft-stereo-sceneflow", device="cpu", options={"iters": 7})(img, img)
    assert r.options == {"iters": 7}
    assert float(r.disparity[0, 0]) == 7
    assert stereo.load("raft-stereo-sceneflow", device="cpu", options={"iters": 32}).options == {}
    with pytest.raises(OptionsError, match="unknown option"):
        stereo.load("raft-stereo-sceneflow", device="cpu", options={"iter": 7})
    with pytest.raises(OptionsError) as e:
        stereo.load("raft-stereo-sceneflow", device="cpu", options={"iter": 7})
    assert "raft-stereo-sceneflow" in str(e.value)
    assert "did you mean 'iters'" in str(e.value)
    assert "valid options: iters" in str(e.value)
    with pytest.raises(OptionsError, match="must be int"):
        stereo.load("raft-stereo-sceneflow", device="cpu", options={"iters": "7"})
    with pytest.raises(OptionsError, match="must be >= 1, got -3"):
        stereo.load("raft-stereo-sceneflow", device="cpu", options={"iters": -3})


def test_params_must_match_card(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class Few:
        feature_channels: int

    class Bad(Dummy):
        Params = Few

    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Bad}))
    with pytest.raises(CardError, match="do not match"):
        stereo.load("s2m2-s", device="cpu")


@pytest.mark.parametrize("task", ["stereo"])
def test_every_card_builds_its_params(task):
    """Card params and options fit each family's `Params` (imports model code, builds no network)."""
    for card, v in registry.cards(task).values():
        for mode in card.modes:
            cls = core_model.runner_class(card, mode)
            core_model.make_params(cls.Params, card, v, {})


@pytest.mark.parametrize("task", registry.TASKS)
def test_mps_dtypes_are_implemented(task):
    """A card's MPS dtypes are dtypes its portable runner implements."""
    for card, _ in registry.cards(task).values():
        if card.portable_mps_dtypes:
            runner = core_model.runner_class(card, "portable")
            assert set(card.portable_mps_dtypes) - {"float32"} <= set(runner.dtypes), card.family


@pytest.mark.parametrize("task", registry.TASKS)
def test_metal_kernels_have_an_mps_runner(task):
    """A family with Metal kernels has a `kernels_mps` runner in its kernels dtype; `runner_class` picks it on MPS."""
    mps = torch.device("mps")
    for card, _ in registry.cards(task).values():
        if card.mps_kernel_libs:
            runner = core_model.runner_class(card, "kernels", mps)
            assert runner is registry.import_model(card).RUNNERS["kernels_mps"], card.family
            assert card.kernels_dtype_on(mps) in runner.dtypes, card.family
            assert core_model.runner_class(card, "kernels", torch.device("cuda")) is not runner, card.family


def test_metal_library_needs_mps():
    from fastforks.kernels import stereo_metal  # noqa: PLC0415

    with pytest.raises(KernelUnavailableError, match="Apple MPS only"):
        stereo_metal.load(torch.device("cpu"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_kernels_on_mps_without_metal_kernels():
    card = stereo.model_card("s2m2-s")
    assert "kernels" in card.modes
    assert not card.mps_kernel_libs
    with pytest.raises(BackendUnavailableError, match="no Metal kernels"):
        stereo.load("s2m2-s", mode="kernels", device="mps")


def test_kernels_one_device_per_process(monkeypatch):
    monkeypatch.setattr(core_model, "_KERNELS_DEVICE", [])
    core_model.claim_kernels_device(torch.device("cuda:0"))
    core_model.claim_kernels_device(torch.device("cuda:0"))
    with pytest.raises(BackendUnavailableError, match="one process per GPU"):
        core_model.claim_kernels_device(torch.device("cuda:1"))


@pytest.mark.parametrize(
    ("device", "match"),
    [
        ("gpu", "invalid device 'gpu'"),
        ("cuda", "has no usable CUDA GPU .*device='cpu'"),
        ("cuda:3", "has no usable CUDA GPU"),
    ],
)
def test_device_errors(monkeypatch, device, match):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(OptionsError, match=match):
        core_model.resolve_device(device)
    assert core_model.resolve_device(None) == torch.device("cpu")


def test_device_index_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(OptionsError, match="has 1 CUDA device"):
        core_model.resolve_device("cuda:1")


def test_arguments_are_checked_before_user_weights(fake_runner, tmp_path, caplog):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"0")
    for bad in ({"mode": "bogus"}, {"dtype": "int3"}, {"options": {"nope": 1}}, {"options": []}, {"options": 0}):
        with caplog.at_level(logging.WARNING), pytest.raises(OptionsError):
            stereo.load("s2m2-s", device="cpu", weights=weights, **bad)
    with caplog.at_level(logging.WARNING), pytest.raises(OptionsError):
        stereo.load("s2m2-s", device="bogus", weights=weights)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # no "unverified weights" warning


def test_top_level_load_labels(monkeypatch):
    import fastforks  # noqa: PLC0415
    from fastforks import classification  # noqa: PLC0415

    with pytest.raises(OptionsError, match="labels= is for classifiers; 's2m2-s' is a stereo model"):
        fastforks.load("s2m2-s", labels=["a"])
    seen = {}
    monkeypatch.setattr(classification, "load", lambda model, **kw: seen.update(kw, model=model))
    fastforks.load("alexnet.tv_in1k", labels=("a", "b"))
    assert seen["model"] == "alexnet.tv_in1k"
    assert seen["labels"] == ("a", "b")


def test_class_names_are_read_only():
    from fastforks.classification import ClassificationModel  # noqa: PLC0415

    prop = ClassificationModel.class_names
    assert isinstance(prop, property)
    assert prop.fset is None
