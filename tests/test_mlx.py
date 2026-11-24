"""Apple MLX backend (`load(..., device="mlx")`): card fields, load errors, tunables; no weights, no GPU work."""

from __future__ import annotations

import copy
import dataclasses
import functools
import gc
import sys
import threading
import tomllib
import types
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
    matting,
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


@pytest.fixture
def portless(monkeypatch):
    """MatAnyone as a family without an MLX port (every family has one)."""
    lookup = registry._lookup

    def without(task, model_id):
        found = lookup(task, model_id)
        if found is None or found[0].family != "matanyone":
            return found
        return dataclasses.replace(found[0], portable_mlx_dtypes=()), found[1]

    monkeypatch.setattr(registry, "_lookup", without)


def test_card_declares_the_port_per_mode(cre):
    card = parse(cre)
    assert card.portable_mlx_dtypes == ("float32", "float16")
    assert card.mlx_kernel_libs == ("crestereo_mlx",)
    assert card.mlx_dtypes("kernels") == ("float32", "float16")
    assert "  mlx       portable (float32, float16), kernels (float32, float16; crestereo_mlx)" in str(card)
    for mode in cre["modes"].values():
        mode.pop("mlx_dtypes", None)
        mode.pop("mlx_kernels", None)
    none = parse(cre)
    assert (none.portable_mlx_dtypes, none.mlx_kernel_libs, none.mlx_dtypes("kernels")) == ((), (), ())
    assert "  mlx " not in str(none)


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


def test_romav2_declares_both_mlx_dtypes():
    card = registry.resolve("matching", "romav2")[0]
    assert card.portable_mlx_dtypes == card.mlx_dtypes("kernels") == ("float32", "float16")
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


@pytest.mark.usefixtures("portless")
def test_family_without_a_port_raises():
    with pytest.raises(BackendUnavailableError, match="no Apple MLX port"):
        matting.load("matanyone", device="mlx")


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

    def test_cache_request_is_capped_by_the_memory(self, rt):
        gb = 1 << 30
        assert [rt.cache_request_limit(2048, n * gb) >> 20 for n in (8, 16, 64)] == [1024, 2048, 2048]
        assert rt.cache_request_limit(256, 8 * gb) >> 20 == 256

    def test_cache_limit_is_the_largest_request_of_the_live_models(self, rt, monkeypatch):
        monkeypatch.delenv("FASTFORKS_MLX_CACHE_MB", raising=False)

        class Runner:
            pass

        memory = rt.gpu().memory
        base = rt.request_cache(None, 0)
        want = {mb: max(base, rt.cache_request_limit(mb, memory)) for mb in (256, 512)}
        big, small = Runner(), Runner()
        assert rt.request_cache(big, 512) == want[512]
        assert rt.request_cache(small, 256) == want[512]  # a later, smaller request does not lower it
        rt.setup()  # nor does the next MLX load
        assert rt.mx.set_cache_limit(want[512]) == want[512]  # returns the limit that was in effect
        del big
        gc.collect()
        assert rt.request_cache(None, 0) == want[256]  # dropped back when its model was freed
        del small
        gc.collect()
        assert rt.mx.set_cache_limit(base) == base


class TestMpsRouting:
    """`load(device="mps", mode="auto")` takes the MLX port where `apple_mlx.json` says so (`_mlx_for_mps`)."""

    @pytest.fixture(autouse=True)
    def mlx_on(self, monkeypatch):
        monkeypatch.setenv("FASTFORKS_MLX", "1")
        monkeypatch.setattr(core_model.importlib.util, "find_spec", lambda name: object() if name == "mlx" else None)

    @staticmethod
    def route(ref, mode="auto", dtype=None, task="stereo"):
        card, variant = registry.resolve(task, ref)
        return core_model._mlx_for_mps(card, variant, mode, dtype)

    def test_auto_takes_the_fastest_measured_option(self):
        assert self.route("sam3", task="promptable_segmentation") == ("portable", "float16")
        assert self.route("crestereo-eth3d") == ("kernels", "float16")
        # hand-written, so nothing is exported: the first call (4.3 s) is within MPS's first and two warm ones
        assert self.route("depth-pro", task="depth") == ("portable", "float16")
        assert self.route("realesr-general-x4v3", task="restoration") is None

    def test_auto_with_a_dtype_takes_mlx_only_where_it_measured_faster(self):
        assert self.route("dinov2-vits14", dtype="float32", task="features") == ("portable", "float32")
        assert self.route("dinov3-vith16plus", dtype="float32", task="features") == (
            "portable",
            "float32",
        )  # re-surveyed: 1.08x, leaner

    def test_portable_stays_on_pytorch(self):
        assert self.route("crestereo-eth3d", mode="portable") is None

    def test_explicit_kernels_never_crosses_to_mlx(self, monkeypatch):
        """Metal or an error: with or without a Metal runner, an MLX port, MLX kernels or a dtype Metal lacks."""
        assert self.route("crestereo-eth3d", mode="kernels") is None  # Metal runner
        assert self.route("crestereo-eth3d", mode="kernels", dtype="float32") is None  # no fp32 Metal: an error
        assert self.route("depth-anything-v2-b", mode="kernels", task="depth") is None  # MLX kernels (dpt_mlx) only
        assert self.route("dinov2-vits14", mode="kernels", task="features") is None  # a port without kernels
        # a table that still lists explicit-kernels routes (older generator) is not read
        monkeypatch.setattr(
            core_model,
            "_apple_mlx",
            lambda: {"romav2@1": {"kernels": {"float32": "portable"}, "kernels_default": ["portable", "float32"]}},
        )
        assert self.route("romav2", mode="kernels", task="matching") is None
        assert self.route("romav2", mode="kernels", dtype="float32", task="matching") is None

    @pytest.mark.usefixtures("portless")
    def test_family_without_port(self):
        assert self.route("matanyone", task="matting") is None

    @pytest.mark.usefixtures("portless")
    def test_kernels_on_mps_without_metal_raises_and_names_the_alternatives(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs Apple MPS")
        loaded = []
        monkeypatch.setattr(core_model, "_load_mlx", lambda *_a, **_k: loaded.append(1))
        for ref, task in (("dinov2-vits14", "features"), ("depth-anything-v2-b", "depth")):
            card, variant = registry.resolve(task, ref)
            assert card.portable_mlx_dtypes
            assert not card.mps_kernel_libs
            with pytest.raises(BackendUnavailableError, match="mode='auto' or 'portable' or device='mlx'"):
                core_model.load_variant(task, card, variant, object, mode="kernels", device="mps")
        assert not loaded
        card, _ = registry.resolve("matting", "matanyone")  # no port: `device='mlx'` is not offered
        assert core_model._mps_alternatives(card) == "mode='auto' or 'portable'"

    def test_off_by_env_and_without_mlx(self, monkeypatch):
        monkeypatch.setenv("FASTFORKS_MLX", "0")
        assert self.route("crestereo-eth3d") is None
        monkeypatch.setenv("FASTFORKS_MLX", "1")
        monkeypatch.setattr(core_model.importlib.util, "find_spec", lambda _name: None)
        assert self.route("crestereo-eth3d") is None

    def test_table_lists_ports_only(self):
        for model_id, entry in core_model._apple_mlx().items():
            card, _ = registry.resolve_any(model_id)
            assert "mps_dtype" not in entry or entry["mps_dtype"] in card.portable_mps_dtypes, model_id
            assert card.mps_kernel_libs or "mps_auto" not in entry, model_id
            if set(entry) <= {"mps_auto", "mps_dtype"}:  # PyTorch MPS choices only: no MLX port needed
                continue
            assert card.portable_mlx_dtypes, model_id
            modes = {m for m, _ in [entry.get("auto", ("portable", "float32"))]} | set(entry.get("dtypes", {}).values())
            assert modes <= ({"portable", "kernels"} if card.mlx_kernel_libs else {"portable"}), model_id


class TestMlxDevice:
    """`device="mlx"` without a dtype takes the fastest passing survey run (`mlx_default`), where it fits `mode`."""

    @staticmethod
    def built(monkeypatch, ref, task, mode="auto", dtype=None):
        card, variant = registry.resolve(task, ref)
        monkeypatch.setattr(core_model, "_build_mlx", lambda *_a, mode, dtype, **_k: (mode, dtype))
        log = core_model.logging.getLogger("test")
        return core_model._load_mlx(
            object, card, variant, mode=mode, dtype=dtype, options={}, weights=None, state=None, log=log
        )

    def test_default_is_the_fastest_measured(self, monkeypatch):
        assert self.built(monkeypatch, "sam3", "promptable_segmentation") == ("portable", "float16")
        assert self.built(monkeypatch, "depth-anything-v2-s", "depth") == ("kernels", "float16")

    def test_explicit_dtype_or_other_mode_keeps_the_card_order(self, monkeypatch):
        assert self.built(monkeypatch, "sam3", "promptable_segmentation", dtype="float32") == ("portable", "float32")
        assert self.built(monkeypatch, "depth-anything-v2-s", "depth", mode="portable") == ("portable", "float32")

    def test_without_measurements(self, monkeypatch):
        monkeypatch.setattr(core_model, "_apple_mlx", dict)
        assert self.built(monkeypatch, "sam3", "promptable_segmentation") == ("portable", "float32")


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
    nested = core_model._results_to({"a": torch.zeros(1), "b": (one, 2)}, meta)
    assert nested["a"].device == meta
    assert nested["b"][0].x.device == meta
    assert nested["b"][1] == 2


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


@pytest.mark.parametrize(("kernel", "bias"), [((2, 2), True), ((4, 4), False), ((2, 3), True)])
def test_conv_transpose_kernel_equals_stride_matches_torch(kernel, bias):
    """A transposed conv whose kernel equals its stride (the GEMM + depth-to-space lowering) against torch's."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    torch.manual_seed(0)
    m = torch.nn.ConvTranspose2d(12, 8, kernel, kernel, bias=bias)
    x = torch.randn(2, 12, 5, 7)
    torch.testing.assert_close(MlxModule(m, "float32")(x), m(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("keep", [False, True])
def test_logsumexp_runs_natively_and_matches_torch(keep):
    """`torch.logsumexp` (kept out of the decomposition, `mx.logsumexp`) over one and two dims against eager."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Lse(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.logsumexp(x, dim=2, keepdim=keep) + torch.logsumexp(x, dim=(1, 3), keepdim=True).sum()

    torch.manual_seed(0)
    x = torch.randn(2, 5, 7, 9) * 20
    torch.testing.assert_close(MlxModule(Lse(), "float32")(x), Lse()(x), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("chunk_mb", [128, 4])
def test_conv_padded_into_winograd_matches_torch(monkeypatch, chunk_mb):
    """A float32 3x3 with 163 -> 168 channels (padded to 192 -> 192: MLX's Winograd), whole and per-sample chunked."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_graph  # noqa: PLC0415 - optional (the mlx extra)

    monkeypatch.setattr(mlx_graph, "WINO_CHUNK_BYTES", chunk_mb << 20)
    torch.manual_seed(0)
    m = torch.nn.Conv2d(163, 168, 3, padding=1)
    x = torch.randn(3, 163, 40, 40)
    torch.testing.assert_close(mlx_graph.MlxModule(m, "float32")(x), m(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cout", [80, 96])
def test_narrow_conv_padded_into_winograd_matches_torch(cout):
    """A float32 3x3 with 128 -> 80 / 96 channels (under MLX's Winograd width; padded to 128 outputs, sliced)."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_graph  # noqa: PLC0415 - optional (the mlx extra)

    torch.manual_seed(0)
    m = torch.nn.Conv2d(128, cout, 3, padding=1)
    x = torch.randn(1, 128, 64, 72)
    torch.testing.assert_close(mlx_graph.MlxModule(m, "float32")(x), m(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("cout", "bias", "gemm1x1", "taken"),
    [
        (48, True, True, True),
        (48, False, True, True),
        (1, True, True, True),
        (8, True, True, False),
        (48, True, False, False),
    ],
)
def test_conv1x1_as_gemm_matches_torch(monkeypatch, cout, bias, gemm1x1, taken):
    """A 1x1 stride-1 conv is one GEMM from 33 outputs on (and for 1), `mx.conv2d` for 2-16 or with `gemm1x1=False`."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_graph  # noqa: PLC0415 - optional (the mlx extra)

    calls = []
    real = mlx_graph.conv1x1
    monkeypatch.setattr(mlx_graph, "conv1x1", lambda *a: calls.append(1) or real(*a))
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(96, cout, 1, bias=bias), torch.nn.ReLU())
    x = torch.randn(2, 96, 20, 24)
    torch.testing.assert_close(mlx_graph.MlxModule(m, "float32", gemm1x1=gemm1x1)(x), m(x), atol=1e-4, rtol=1e-4)
    assert bool(calls) == taken


def test_conv1x1_rule_keeps_other_convs():
    """Strided, padded, grouped, larger or very wide (a million weight elements) convolutions stay convolutions."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import conv1x1_ok  # noqa: PLC0415 - optional (the mlx extra)

    assert conv1x1_ok(96, 288, 0, (1, 1), (1, 1), (0, 0), 1)
    assert not conv1x1_ok(96, 288, 0, (1, 1), (2, 2), (0, 0), 1)
    assert not conv1x1_ok(96, 288, 0, (1, 1), (1, 1), (1, 1), 1)
    assert not conv1x1_ok(96, 288, 0, (1, 1), (1, 1), (0, 0), 2)
    assert not conv1x1_ok(96, 288, 0, (3, 3), (1, 1), (0, 0), 1)
    assert not conv1x1_ok(1232, 1232, 196, (1, 1), (1, 1), (0, 0), 1)
    assert not conv1x1_ok(64, 24, 64 * 64, (1, 1), (1, 1), (0, 0), 1)
    assert conv1x1_ok(64, 24, 128 * 128, (1, 1), (1, 1), (0, 0), 1)


@pytest.mark.parametrize("wino", [False, True])
def test_half_ragged_conv_matches_torch(wino):
    """A float16 3x3 with 196 -> 196 channels: output-only padding by default, MLX's Winograd with `half_wino_pad`."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_graph  # noqa: PLC0415 - optional (the mlx extra)

    torch.manual_seed(0)
    m = torch.nn.Conv2d(196, 196, 3, padding=1)
    x = torch.randn(2, 196, 64, 72)
    want = m(x)
    got = mlx_graph.MlxModule(m, "float16", half_wino_pad=wino)(x).float()
    assert got.shape == want.shape
    err = ((got - want).abs().mean() / want.abs().mean()).item()
    assert err < (2e-2 if wino else 2e-3)  # measured 5e-3 (half Winograd) and 2e-4 (direct)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_float32_call_keeps_float32_weights(dtype):
    """`float32_call`: the stage's weights stay float32 in a half replay (batch norms folded in float32)."""
    pytest.importorskip("mlx.core")
    from fastforks._core.mlx_cast import float32_call  # noqa: PLC0415
    from fastforks.shared import mlx_graph  # noqa: PLC0415 - optional (the mlx extra)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            conv, bn = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False), torch.nn.BatchNorm2d(16)
            self.stage = torch.nn.Sequential(conv, bn, torch.nn.ReLU(), torch.nn.Conv2d(16, 4, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return float32_call(self.stage, x)

    torch.manual_seed(0)
    net = Net().eval()
    net.stage[1].running_mean.normal_()
    net.stage[1].running_var.uniform_(0.5, 2.0)
    x = torch.randn(1, 8, 20, 24).half().float()  # exact in half: the input is cast to the run dtype at entry
    want = net(x)
    got = mlx_graph.MlxModule(net, dtype)(x)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)  # half-rounded weights: 1e-3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linear_sample_kernel_matches_torch(dtype):
    """The IGEV-family geometry / correlation lookup (`fastforks::linear_sample`, one Metal kernel) against eager."""
    pytest.importorskip("mlx.core")
    from fastforks._core.volume_ops import linear_sample  # noqa: PLC0415
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Lookup(torch.nn.Module):
        def forward(self, geo: torch.Tensor, corr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([linear_sample(geo, x), linear_sample(corr, x * 3.0)], 1)

    torch.manual_seed(0)
    geo, corr = torch.randn(500, 8, 24), torch.randn(500, 40).to(dtype)
    x = torch.rand(500, 9) * 30 - 3  # some taps outside both ends
    want = Lookup()(geo, corr, x)
    torch.testing.assert_close(MlxModule(Lookup(), "float32")(geo, corr, x), want, atol=1e-5, rtol=1e-5)


def test_gwc_warp_kernel_matches_torch():
    """RT-MonSter++'s selective group-wise correlation (`fastforks::gwc_warp`, one Metal kernel) against eager."""
    pytest.importorskip("mlx.core")
    from fastforks._core.volume_ops import gwc_warp  # noqa: PLC0415
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Volume(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
            return gwc_warp(left, right, disp, 8)

    torch.manual_seed(0)
    left, right = torch.randn(1, 96, 20, 37), torch.randn(1, 96, 20, 37)
    disp = torch.rand(1, 13, 20, 37) * 12 - 1  # negative and fractional samples, some with x < disp
    want = Volume()(left, right, disp)
    torch.testing.assert_close(MlxModule(Volume(), "float32")(left, right, disp), want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("count", [300, 1])
@pytest.mark.parametrize("side", [7, 14])
def test_rcnn_roi_align_kernel_matches_torchvision(side, count):
    """Mask / Faster R-CNN's ROIAlign over the FPN levels (one Metal kernel) against torchvision's, level by level."""
    mx = pytest.importorskip("mlx.core")
    tv = pytest.importorskip("torchvision.ops")
    from fastforks.detection.models.rcnn import mlx_net  # noqa: PLC0415
    from fastforks.detection.models.rcnn.model import box_levels, pool  # noqa: PLC0415

    g = torch.Generator().manual_seed(0)
    feats = [torch.randn(1, 16, 96 >> i, 160 >> i, generator=g) for i in range(4)]
    centre = torch.rand(count, 2, generator=g) * torch.tensor([640.0, 384.0])
    extent = torch.rand(count, 2, generator=g) ** 3 * torch.tensor([600.0, 360.0]) + 0.5  # most small, some huge
    boxes = torch.cat([centre - extent / 2, centre + extent / 2], 1)  # some reach outside the image
    if count > 1:
        boxes[0] = torch.tensor([-20.0, 10.0, 640.0, 384.0])  # the whole image: the coarsest level
    lvl = box_levels(boxes, len(feats))
    assert count == 1 or set(lvl.tolist()) == {0, 1, 2, 3}
    want = pool(feats, boxes, side, tv).permute(0, 2, 3, 1)
    got = mlx_net.roi_align([mx.array(f.permute(0, 2, 3, 1).numpy()) for f in feats], boxes, lvl, side)
    torch.testing.assert_close(torch.from_numpy(np.array(got)), want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("align", [True, False])
def test_channels_last_instance_norm_and_bilinear_match_torch(align):
    """Instance norm (N > 1, affine) and bilinear resize of conv outputs (the channels-last lowerings) against torch."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.norm = torch.nn.InstanceNorm2d(16, affine=True)
            self.b = torch.nn.Conv2d(16, 24, 3, stride=2, padding=1)
            self.out = torch.nn.Conv2d(40, 8, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = torch.relu(self.norm(self.a(x)))
            b = torch.nn.functional.instance_norm(self.b(a))
            up = torch.nn.functional.interpolate(b, size=a.shape[2:], mode="bilinear", align_corners=align)
            return self.out(torch.cat([a, up], 1))

    torch.manual_seed(0)
    m = Net().eval()
    with torch.no_grad():
        m.norm.weight.uniform_(0.5, 1.5)
        m.norm.bias.normal_()
    x = torch.randn(3, 3, 21, 30)
    torch.testing.assert_close(MlxModule(m, "float32")(x), m(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", ["replicate", "reflect"])
@pytest.mark.parametrize("pad", [(1, 1, 1, 1), (2, 0, 1, 3), (0, 2, 0, 0)])
def test_channels_last_edge_pads_match_torch(mode, pad):
    """Replicate and reflect padding of a conv output (padded on the channels-last view) and of a plain tensor."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Conv2d(3, 8, 3)
            self.b = torch.nn.Conv2d(8, 4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.b(torch.nn.functional.pad(self.a(torch.nn.functional.pad(x, pad, mode=mode)), pad, mode=mode))
            return torch.nn.functional.pad(y, pad, mode=mode).flatten(1)

    torch.manual_seed(0)
    m = Net().eval()
    x = torch.randn(2, 3, 11, 14)
    torch.testing.assert_close(MlxModule(m, "float32")(x), m(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "resize",
    [
        {"scale_factor": 1.3},
        {"scale_factor": 2.4},
        {"scale_factor": (2.0, 0.6)},
        {"scale_factor": 3},
        {"scale_factor": 1.3, "recompute_scale_factor": True},
        {"size": (13, 4)},
    ],
)
@pytest.mark.parametrize("after_conv", [False, True])
def test_nearest_resize_picks_torchs_source_pixels(resize, after_conv):
    """Nearest resize steps by `1 / scale_factor` when a scale is given (not by input / output), plain and NHWC."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1) if after_conv else torch.nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(self.conv(x), mode="nearest", **resize) * 1.0

    torch.manual_seed(0)
    m = Net().eval()
    for hw in ((5, 5), (2, 7), (10, 3)):
        x = torch.randn(1, 2, *hw)
        torch.testing.assert_close(MlxModule(m, "float32")(x), m(x), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("size", [(7, 9), (1, 16), (20, 5)])
def test_antialiased_resize_matches_torch(mode, align, size):
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Net(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(x, size=size, mode=mode, align_corners=align, antialias=True)

    torch.manual_seed(0)
    x = torch.randn(1, 2, 16, 12)
    m = Net().eval()
    torch.testing.assert_close(MlxModule(m, "float32")(x), m(x), atol=1e-5, rtol=1e-5)


def test_instance_norm_is_not_banded_with_the_tail(monkeypatch):
    """A banded output tail stops at an instance norm: its statistics are over the whole map, not a band of rows."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_band  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415

    monkeypatch.setattr(mlx_band, "BAND_MIN_MB", 0)  # band a small map: one band is a few rows
    monkeypatch.setattr(mlx_band, "BAND_MB", 0)
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.InstanceNorm2d(8),
        torch.nn.GELU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    ).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 40) * torch.linspace(0.2, 3, 32)[:, None]  # rows of different scale
    torch.testing.assert_close(MlxModule(m, "float32")(x), m(x), atol=1e-4, rtol=1e-4)


def test_fp32_inputs_keep_float32_in_a_float16_replay():
    """`MlxModule(fp32_inputs=...)`: the named argument enters a float16 replay unrounded, the others are cast."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    class Diff(torch.nn.Module):
        def forward(self, x: torch.Tensor, cam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x - 1, cam - 1

    v = torch.full((4,), 1 + 2**-14)  # rounds to 1 in float16
    x_out, cam_out = MlxModule(Diff(), "float16", fp32_inputs=("cam",))(v, cam=v.clone())
    assert float(x_out[0]) == 0
    assert float(cam_out[0]) == pytest.approx(2**-14)
    with pytest.raises(TypeError, match="no argument"):
        MlxModule(Diff(), "float16", fp32_inputs=("k",))(v, v)


def test_released_twin_exports_a_second_shape_after_inference_mode_calls():
    """`release=True` stand-ins made in an inference-mode call are normal tensors: the next shape's export (outside
    inference mode) lifts them (an inference stand-in failed on `requires_grad`; waft-a1's survey second shape)."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - optional (the mlx extra)

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 4, 1)).eval()
    ref = copy.deepcopy(net)
    m = MlxModule(net, "float32", release=True)
    for hw in ((16, 16), (24, 32)):
        x = torch.randn(1, 3, *hw)
        with torch.inference_mode():
            got = m(x)
            want = ref(x)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert not any(t.is_inference() for t in (*net.parameters(), *net.buffers()))
    # the biases stay real: a stride-0 bias turns the next export's `addmm` into `mm` + `add` (tapnext float16)
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), ref.parameters(), strict=True) if a.ndim == 1)


class TestLazyTwin:
    """`mlx_wrap._lazy_twin` patches `torch.nn.init` and `load_state_dict` for its own thread only, refcounted."""

    @staticmethod
    def _originals():
        return (torch.nn.init.uniform_, torch.nn.init.kaiming_uniform_, torch.nn.Module.load_state_dict)

    def test_assigns_the_checkpoint_and_rejects_an_unloaded_parameter(self):
        from fastforks.shared.mlx_wrap import LazyTwinError, _lazy_twin  # noqa: PLC0415 - private helper under test

        sd = torch.nn.Linear(4, 3).state_dict()
        before = self._originals()
        with _lazy_twin():
            net = torch.nn.Linear(4, 3)
            net.load_state_dict(sd)
        assert net.weight.data_ptr() == sd["weight"].data_ptr()  # assigned, not copied
        with pytest.raises(LazyTwinError), _lazy_twin():
            kept = torch.nn.Linear(4, 3)  # never loaded: its parameters hold whatever the allocator returned
        del kept
        assert self._originals() == before

    def test_overlapping_twins_restore_the_real_functions(self):
        from fastforks.shared.mlx_wrap import _lazy_twin  # noqa: PLC0415 - private helper under test

        before = self._originals()
        a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

        def first():
            with _lazy_twin():
                a_in.set()
                assert b_in.wait(10)
            a_out.set()

        def second():
            assert a_in.wait(10)
            with _lazy_twin():  # entered while the first is open, left after the first has left
                b_in.set()
                assert a_out.wait(10)

        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert self._originals() == before
        torch.manual_seed(0)
        ref = torch.nn.Linear(4, 3).weight.clone()
        torch.manual_seed(0)
        assert torch.equal(torch.nn.Linear(4, 3).weight, ref)
        assert ref.abs().sum() > 0

    def test_other_threads_build_and_load_as_usual(self):
        from fastforks.shared.mlx_wrap import _lazy_twin  # noqa: PLC0415 - private helper under test

        torch.manual_seed(0)
        ref = torch.nn.Linear(64, 64)
        sd = {k: v.clone() for k, v in ref.state_dict().items()}
        opened, done = threading.Event(), threading.Event()

        def twin():
            with _lazy_twin():
                opened.set()
                assert done.wait(10)

        t = threading.Thread(target=twin)
        t.start()
        try:
            assert opened.wait(10)
            torch.manual_seed(0)
            other = torch.nn.Linear(64, 64)  # built on this thread while the twin's patch is installed
            assert torch.equal(other.weight, ref.weight)
            assert torch.equal(other.bias, ref.bias)
            other.load_state_dict(sd)
            assert other.weight.data_ptr() != sd["weight"].data_ptr()  # copied in place: no `assign=True` here
        finally:
            done.set()
            t.join(20)


@pytest.mark.parametrize("fused", [False, True])
def test_float_attention_mask_stays_finite_in_float16(fused):
    """A float32 additive mask of `finfo.min` must not turn into -inf under float16 scores: a fully masked row is NaN.

    This is the mask a float32 island or an explicit `dtype=torch.float32` hands a float16 attention; a mask that is
    float16 already (a cast input or buffer) is -inf in PyTorch too and stays so.
    """
    pytest.importorskip("mlx.core")
    import mlx.core as mx  # noqa: PLC0415 - optional (the mlx extra)

    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415
    from fastforks.shared import mlx_graph  # noqa: PLC0415

    torch.manual_seed(0)
    d = mlx_graph.SDPA_FUSED_DIMS[0] if fused else 8
    mask = torch.zeros(1, 1, 5, 5)
    mask[..., 1, :] = torch.finfo(torch.float32).min  # one query sees nothing
    mask[..., 3, :2] = -1e9
    mask[..., 4, 0] = -torch.inf  # a mask's own -inf stays
    x = torch.randn(1, 2, 5, d)
    want = torch.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=mask)
    q = to_mx(x).astype(mx.float16)
    got = to_torch(mlx_graph._sdpa(q, q, q, to_mx(mask)))
    assert torch.isfinite(got).all()  # the fully masked row too (a weighting of the values, not always the uniform one)
    rows = [0, 2, 3, 4]
    torch.testing.assert_close(got[..., rows, :], want[..., rows, :], atol=2e-2, rtol=2e-2)
    clamped = mlx_graph._additive_mask(to_mx(mask), mx.float16)
    assert clamped.dtype == mx.float16
    assert to_torch(clamped)[0, 0, 4, 0] == -torch.inf
    assert torch.isfinite(to_torch(clamped)[0, 0, :4]).all()


def test_signature_caches_keep_the_most_recent_shapes(caplog):
    """`MlxModule` holds one exported graph per input signature, at most `SIGNATURES`; an evicted one exports again."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_graph  # noqa: PLC0415 - optional (the mlx extra)

    torch.manual_seed(0)
    net = torch.nn.Conv2d(3, 4, 3, padding=1).eval()
    module = mlx_graph.MlxModule(net, "float32")
    sizes = range(4, 4 + mlx_graph.SIGNATURES + 2)
    with caplog.at_level("WARNING", logger="fastforks.mlx"):
        for n in (*sizes, sizes[0]):
            x = torch.randn(1, 3, n, 5)
            torch.testing.assert_close(module(x), net(x), atol=1e-5, rtol=1e-5)
    assert len(module._nets) == mlx_graph.SIGNATURES
    assert sum("signature" in r.getMessage() for r in caplog.records) == 1  # said once, not per eviction


class TestAutoFallback:
    """An MLX port that `mode="auto"` chose for Apple MPS falls back to PyTorch MPS when a call fails (`auto` only)."""

    @staticmethod
    def model(calls, fallback=None):
        class Fake:
            card = types.SimpleNamespace(task="stereo")
            variant = types.SimpleNamespace(ref="fake@1")

            def __init__(self, name, error=None):
                self.name, self.error = name, error

            @core_model._typo_checked
            def __call__(self, x):
                calls.append(self.name)
                if self.error is not None:
                    raise self.error
                return x + 1

        m = Fake("mlx", RuntimeError("[metal::malloc] Attempting to allocate 99 bytes which is greater than ..."))
        if fallback:
            m._mps_fallback = lambda: Fake("mps")
        return m

    def test_auto_route_falls_back_once_and_says_so(self, caplog):
        pytest.importorskip("mlx.core")
        calls = []
        m = self.model(calls, fallback=True)
        with caplog.at_level("WARNING", logger="fastforks.stereo"):
            assert m(1) == 2
            assert m(2) == 3
        assert calls == ["mlx", "mps", "mps"]
        assert m.name == "mps"
        assert m.error is None
        assert [r.getMessage().split(":")[0] for r in caplog.records] == ["fake@1"]
        assert "PyTorch MPS" in caplog.records[0].getMessage()

    def test_unsupported_op_falls_back_too(self):
        pytest.importorskip("mlx.core")
        from fastforks.shared.mlx_graph import UnsupportedOp  # noqa: PLC0415 - optional (the mlx extra)

        calls = []
        m = self.model(calls, fallback=True)
        m.error = UnsupportedOp("aten::nope")
        assert m(1) == 2
        assert calls == ["mlx", "mps"]

    def test_explicit_routes_and_other_errors_raise(self):
        calls = []
        with pytest.raises(RuntimeError, match="metal::malloc"):
            self.model(calls)(1)  # `device="mlx"`: no fallback attached
        m = self.model(calls, fallback=True)
        m.error = ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            m(1)
        assert calls == ["mlx", "mlx"]

    def test_a_failing_fallback_raises_the_first_error(self):
        pytest.importorskip("mlx.core")
        m = self.model([], fallback=True)

        def broken():
            msg = "no PyTorch MPS either"
            raise OSError(msg)

        m._mps_fallback = broken
        with pytest.raises(RuntimeError, match="metal::malloc") as e:
            m(1)
        assert isinstance(e.value.__cause__, OSError)
        with pytest.raises(RuntimeError, match="metal::malloc"):  # one attempt
            m(1)

    def test_only_auto_loads_take_the_port_and_carry_the_fallback(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs Apple MPS")
        card, variant = registry.resolve("stereo", "crestereo-eth3d")
        seen = []

        def load_mlx(*_a, **_k):
            port = core_model.Model.__new__(core_model.Model)
            port.device, port._result_device = torch.device("cpu"), None
            port.backend = core_model.Backend("portable", "mlx", "float16")
            return port

        def build(*_a, **k):
            seen.append(k["mode"])
            return "torch"

        monkeypatch.setenv("FASTFORKS_MLX", "1")
        monkeypatch.setattr(core_model.importlib.util, "find_spec", lambda name: object() if name == "mlx" else None)
        monkeypatch.setattr(core_model, "_load_mlx", load_mlx)
        monkeypatch.setattr(core_model, "_build", build)
        load = functools.partial(core_model.load_variant, "stereo", card, variant, object, device="mps")
        assert load(mode="kernels") == "torch"  # explicit: PyTorch MPS, no port
        assert load(mode="portable") == "torch"
        assert seen == ["kernels", "portable"]
        m = load(mode="auto")
        assert m.backend.device == "mlx"
        assert m._mps_fallback() == "torch"  # the PyTorch MPS load, not the MLX route again


def test_a_model_routed_from_mps_reports_the_device_of_its_results(monkeypatch):
    """`model.device` is where results land (MPS), so `x.to(model.device)` combines with them; inputs stay on CPU."""
    port = core_model.Model.__new__(core_model.Model)
    port.device, port._result_device = torch.device("cpu"), None
    port.backend = core_model.Backend("portable", "mlx", "float16")
    assert port._device == port.device == torch.device("cpu")  # device="mlx": inputs and results on the CPU
    monkeypatch.setattr(core_model, "_mlx_for_mps", lambda *_a: ("portable", "float16"))
    monkeypatch.setattr(core_model, "_load_mlx", lambda *_a, **_k: port)
    card, variant = registry.resolve("stereo", "crestereo-eth3d")
    mps = torch.device("mps")
    log = core_model.logging.getLogger("test")
    m = core_model._mps_via_mlx(
        object, card, variant, mode="auto", dtype=None, dev=mps, opts={}, files=None, state=None, log=log
    )
    assert m is port
    assert (m.device, m._device, m._result_device) == (mps, torch.device("cpu"), mps)
    # what runs, not a relabel: the port's own mode and kernels
    assert (m.backend.device, m.backend.mode, m.backend.kernels) == ("mlx", "portable", ())
