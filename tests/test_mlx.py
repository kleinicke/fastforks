"""Apple MLX backend (`load(..., device="mlx")`): card fields, load errors, tunables; no weights, no GPU work."""

from __future__ import annotations

import copy
import dataclasses
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from fastforks import (
    BackendUnavailableError,
    CardError,
    MissingDependencyError,
    OptionsError,
    features,
    matching,
    stereo,
)
from fastforks._core import model as core_model
from fastforks._core import registry
from fastforks._core.cards import parse_card
from fastforks._core.image import to_rgb_batch
from fastforks._core.outputs import NumpyOutputs

MODELS = Path(registry.__file__).parents[1] / "stereo" / "models"


@pytest.fixture
def cre():
    return tomllib.loads((MODELS / "crestereo/card.toml").read_text(encoding="utf-8"))


def parse(d):
    return parse_card(d, family="crestereo", task="stereo", where="card.toml")


def test_card_declares_the_port_per_mode(cre):
    card = parse(cre)
    assert card.portable_mlx_dtypes == ("float32", "float16")
    assert card.mlx_kernel_libs == ("crestereo_mlx",)
    assert card.mlx_dtypes("kernels") == ("float32", "float16")
    assert "  mlx       portable (float32, float16), kernels (float32, float16; crestereo_mlx)" in str(card)
    none = registry.resolve("matching", "aliked-lightglue")[0]
    assert (none.portable_mlx_dtypes, none.mlx_kernel_libs, none.mlx_dtypes("kernels")) == ((), (), ())


@pytest.mark.parametrize(
    ("task", "ref", "libs"),
    [
        ("classification", "efficientnet_b0.tv_in1k", ("cnn_mlx",)),
        ("classification", "efficientnet_v2_s.tv_in1k", ("cnn_mlx",)),
        ("classification", "mobilenet_v3_large.tv2_in1k", ("cnn_mlx",)),
        ("classification", "regnety_160.tv2_in1k", ("cnn_mlx",)),
        ("classification", "convnext_tiny.fb_in1k", ("cnn_mlx",)),
        ("classification", "convnextv2_base.fcmae_ft_in22k_in1k", ("cnn_mlx",)),
        ("classification", "resnet50", ("cnn_mlx",)),
        ("classification", "mobilenet_v2", ("cnn_mlx",)),
        ("features", "convnext_tiny.fb_in22k", ("cnn_mlx",)),
        ("features", "dinov2-vits14", ()),
        ("features", "dinov3-vits16", ()),
        ("features", "dinov3-convnext-large", ("cnn_mlx",)),
        ("features", "tipsv2-l14", ()),
        ("features", "c-radio-v3-g", ()),
        ("classification", "dinov2_vitl14_lc", ()),
        ("segmentation", "dinov2-l-linear-ade20k", ()),
        ("depth", "depth-anything-v2-s", ("dpt_mlx",)),
        ("depth", "distill-any-depth-s", ("dpt_mlx",)),
    ],
)
def test_ports_declare_both_dtypes(task, ref, libs):
    card = registry.resolve(task, ref)[0]
    both = ("float32", "float16")
    assert card.portable_mlx_dtypes == both
    assert (card.mlx_kernel_libs, card.mlx_dtypes("kernels")) == (libs, both if libs else ())


def test_romav2_declares_float32_mlx_matching():
    card = registry.resolve("matching", "romav2")[0]
    assert card.portable_mlx_dtypes == card.mlx_dtypes("kernels") == ("float32",)
    assert card.mlx_kernel_libs == ("roma_mlx", "dpt_mlx")


def test_seedvr2_declares_its_reference_dtype_first():
    """A port whose float32 weights would not fit starts with the family's `cuda_dtype` (bf16) instead."""
    card = registry.resolve("restoration", "seedvr2-3b")[0]
    assert card.portable_mlx_dtypes == ("bfloat16", "float16")
    assert card.portable_cuda_dtype == "bfloat16"


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d["modes"]["kernels"].pop("mlx_dtypes"), "come together"),
        (lambda d: d["modes"]["kernels"].pop("mlx_kernels"), "come together"),
        (lambda d: d["modes"]["portable"].pop("mlx_dtypes"), "come together"),
        (lambda d: d["modes"]["portable"].update(mlx_dtypes=["float16", "float32"]), "float32"),
        (lambda d: d["modes"]["portable"].update(mlx_dtypes=["float32", "int8"]), "int8"),
        (lambda d: d["modes"]["portable"].update(mlx_dtypes=["bfloat16"]), "float32"),  # not the cuda_dtype
    ],
)
def test_card_rejects_inconsistent_mlx_fields(cre, edit, match):
    d = copy.deepcopy(cre)
    edit(d)
    with pytest.raises(CardError, match=match):
        parse(d)


def test_family_without_a_port_raises():
    with pytest.raises(BackendUnavailableError, match="no Apple MLX port"):
        matching.load("aliked-lightglue", device="mlx")


def test_port_without_kernels_raises_for_kernels():
    pytest.importorskip("mlx")
    with pytest.raises(BackendUnavailableError, match="no kernels"):
        features.load("dinov2-vits14", device="mlx", mode="kernels")


def test_unlisted_dtype_raises():
    with pytest.raises(OptionsError, match="bfloat16"):
        stereo.load("crestereo-eth3d", device="mlx", mode="portable", dtype="bfloat16")


def test_without_the_mlx_package(monkeypatch):
    for name in [n for n in sys.modules if n == "mlx" or n.startswith("mlx.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.delitem(sys.modules, "fastforks._core.mlx", raising=False)
    with pytest.raises(MissingDependencyError, match=r"fastforks-core\[mlx\]"):
        stereo.load("crestereo-eth3d", device="mlx", mode="portable")


def test_mx_array_images_are_laid_out_as_numpy():
    mx = pytest.importorskip("mlx.core")
    img = np.random.default_rng(0).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    cpu = torch.device("cpu")
    assert torch.equal(to_rgb_batch(mx.array(img), cpu), to_rgb_batch(img, cpu))


class TestTuning:
    @pytest.fixture
    def rt(self, monkeypatch):
        pytest.importorskip("mlx")
        from fastforks._core import mlx as runtime  # noqa: PLC0415 - optional (the mlx extra)

        monkeypatch.delenv("FASTFORKS_MLX_TUNING", raising=False)
        return runtime

    def test_default_row_is_the_m1(self, rt):
        assert rt.TUNING["default"] == rt.Tuning()
        assert rt.tuning_row() in rt.TUNING
        assert rt.tuning() == rt.TUNING[rt.tuning_row()]

    def test_most_specific_row_wins(self, rt, monkeypatch):
        g = rt.gpu()
        family = g.architecture.rstrip("abcdefghijklmnopqrstuvwxyz")
        monkeypatch.setitem(rt.TUNING, family, dataclasses.replace(rt.Tuning(), threadgroup=512))
        assert (rt.tuning_row(), rt.tuning().threadgroup) == (family, 512)
        monkeypatch.setitem(rt.TUNING, g.architecture, dataclasses.replace(rt.Tuning(), threadgroup=64))
        assert (rt.tuning_row(), rt.tuning().threadgroup) == (g.architecture, 64)

    def test_env_and_block_overrides(self, rt, monkeypatch):
        monkeypatch.setenv("FASTFORKS_MLX_TUNING", "winograd=0, threadgroup=128,cache_fraction=0.1")
        t = rt.tuning()
        assert (t.winograd, t.threadgroup, t.cache_fraction) == (False, 128, 0.1)
        with rt.tuned(threadgroup=64, batch_pair=True):
            assert (rt.tuning().threadgroup, rt.tuning().batch_pair, rt.tuning().winograd) == (64, True, False)
        assert rt.tuning().threadgroup == 128

    @pytest.mark.parametrize("text", ["tile=4", "threadgroup=big"])
    def test_bad_env_raises(self, rt, monkeypatch, text):
        monkeypatch.setenv("FASTFORKS_MLX_TUNING", text)
        with pytest.raises(OptionsError, match="FASTFORKS_MLX_TUNING"):
            rt.tuning()

    def test_bad_block_field_raises(self, rt):
        with pytest.raises(OptionsError, match="tile"), rt.tuned(tile=4):
            pass

    def test_cache_limit_scales_with_memory_between_its_bounds(self, rt):
        t, gb = rt.Tuning(), 1 << 30
        assert [t.cache_limit(n * gb) >> 20 for n in (8, 16, 48, 64, 128)] == [128, 128, 384, 512, 1024]


class TestMpsRouting:
    """`load(device="mps")` takes the MLX port where `apple_mlx.json` says it is faster (`_mlx_for_mps`)."""

    @pytest.fixture(autouse=True)
    def mlx_on(self, monkeypatch):
        monkeypatch.setenv("FASTFORKS_MLX", "1")
        monkeypatch.setattr(core_model.importlib.util, "find_spec", lambda name: object() if name == "mlx" else None)

    @staticmethod
    def route(ref, mode="auto", dtype=None, task="stereo"):
        card, variant = registry.resolve(task, ref)
        return core_model._mlx_for_mps(card, variant, mode, dtype)

    def test_auto_takes_the_fastest_measured_option(self):
        assert self.route("crestereo-eth3d") == ("kernels", "float16")
        assert self.route("realesr-general-x4v3", task="restoration") == ("kernels", "float16")  # faster and leaner
        assert self.route("dinov2-vits14", task="features") is None  # faster at float16, but more memory than MPS

    def test_auto_with_a_dtype_takes_mlx_only_where_it_measured_faster(self):
        assert self.route("dinov2-vits14", dtype="float32", task="features") == ("portable", "float32")
        assert self.route("dinov3-vith16plus", dtype="float32", task="features") == (
            "portable",
            "float32",
        )  # re-surveyed: 1.08x, leaner

    def test_portable_stays_on_pytorch(self):
        assert self.route("crestereo-eth3d", mode="portable") is None

    def test_kernels_takes_mlx_kernels(self):
        assert self.route("crestereo-eth3d", mode="kernels") == ("kernels", "float16")
        assert self.route("crestereo-eth3d", mode="kernels", dtype="float32") == ("kernels", "float32")  # no fp32 Metal
        assert self.route("depth-anything-v2-b", mode="kernels", task="depth") == ("kernels", "float16")  # dpt_mlx
        assert self.route("dinov2-vits14", mode="kernels", task="features") == (
            "portable",
            "float16",
        )  # kernels: more memory

    def test_explicit_kernels_prefers_native_metal_without_timing_data(self, monkeypatch):
        monkeypatch.setattr(core_model, "_apple_mlx", dict)
        assert self.route("crestereo-eth3d", mode="kernels") is None
        assert self.route("crestereo-eth3d", mode="kernels", dtype="float32") == ("kernels", "float32")

    def test_measured_kernel_and_explicit_dtype(self, monkeypatch):
        monkeypatch.setattr(core_model, "_apple_mlx", lambda: {"romav2@1": {"kernels": {"float32": "portable"}}})
        assert self.route("romav2", mode="kernels", dtype="float32", task="matching") == ("portable", "float32")
        monkeypatch.setattr(core_model.importlib.util, "find_spec", lambda _name: None)
        assert self.route("romav2", mode="kernels", dtype="float32", task="matching") is None

    def test_port_without_kernels_and_family_without_port(self):
        assert self.route("dinov2-vits14", mode="kernels", task="features") == ("portable", "float16")
        assert self.route("raft-stereo-middlebury") is None

    def test_off_by_env_and_without_mlx(self, monkeypatch):
        monkeypatch.setenv("FASTFORKS_MLX", "0")
        assert self.route("crestereo-eth3d") is None
        monkeypatch.setenv("FASTFORKS_MLX", "1")
        monkeypatch.setattr(core_model.importlib.util, "find_spec", lambda _name: None)
        assert self.route("crestereo-eth3d") is None

    def test_table_lists_ports_only(self):
        for model_id, entry in core_model._apple_mlx().items():
            card, _ = registry.resolve_any(model_id)
            if set(entry) == {"mps_auto"}:
                assert card.mps_kernel_libs, model_id
                continue
            assert card.portable_mlx_dtypes, model_id
            modes = {m for m, _ in [entry.get("auto", ("portable", "float32"))]} | set(entry.get("dtypes", {}).values())
            assert modes <= ({"portable", "kernels"} if card.mlx_kernel_libs else {"portable"}), model_id


def test_results_move_to_the_requested_device():
    @dataclasses.dataclass(frozen=True)
    class R(NumpyOutputs):
        x: torch.Tensor

    meta = torch.device("meta")
    one = R(torch.zeros(2))
    assert core_model._results_to(one, meta).x.device == meta
    moved = core_model._results_to([one, one], meta)
    assert isinstance(moved, list)
    assert [r.x.device for r in moved] == [meta, meta]
    assert core_model._results_to({"x": 1}, meta) == {"x": 1}


@pytest.mark.parametrize("size", [(37, 53), (74, 106), (1, 9)])
def test_dpt_resize_matches_torch(size):
    """The DPT head's bilinear resize, portable and Metal kernel, against `F.interpolate(align_corners=True)`."""
    mx = pytest.importorskip("mlx.core")
    from fastforks.kernels.dpt_mlx import ops  # noqa: PLC0415
    from fastforks.shared.depth_anything_v2 import mlx_net  # noqa: PLC0415

    x = torch.randn(2, 64, 19, 27)
    want = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
    xm = mx.array(x.permute(0, 2, 3, 1).numpy())
    for resize in (mlx_net.resize, ops.resize):
        got = torch.from_numpy(np.array(resize(xm, *size)))
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
