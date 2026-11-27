"""Apple MLX backend (`load(..., device="mlx")`): card fields, load errors, tunables; no weights, no GPU work."""

from __future__ import annotations

import copy
import dataclasses
import functools
import gc
import importlib
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


@pytest.mark.slow
def test_mlx_export_context_is_only_read_outside_dynamo():
    from fastforks._core.tracing import EXPORTING_FOR_MLX, exporting_for_mlx  # noqa: PLC0415 - compile test only

    class Probe(torch.nn.Module):
        def forward(self, x):
            return x + 2 if exporting_for_mlx() else x - 1

    module = Probe()
    value = torch.tensor(4)
    assert torch.compile(module, backend="eager", fullgraph=True)(value).item() == 3
    assert torch.export.export(module, (value,), strict=True).module()(value).item() == 3
    token = EXPORTING_FOR_MLX.set(True)
    try:
        assert torch.export.export(module, (value,), strict=False).module()(value).item() == 6
    finally:
        EXPORTING_FOR_MLX.reset(token)
    assert torch.export.export(module, (value,), strict=False).module()(value).item() == 3


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


def test_stereo_results_move_all_views_to_the_requested_device():
    from fastforks.stereo.calibrated import StereoResult, make_view  # noqa: PLC0415 - this result type only

    meta = torch.device("meta")
    maps = {"disparity": torch.zeros(2, 3)}
    result = StereoResult(
        "test@1",
        core_model.Backend("portable", "mlx", "float16"),
        views="both",
        _by_view={view: make_view(view, maps, None) for view in ("left", "right")},
    )
    moved = core_model._results_to(result, meta)
    assert moved.left.disparity.device == moved.right.disparity.device == meta
    assert result.left.disparity.device == torch.device("cpu")


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


@pytest.mark.parametrize(
    ("d", "lq", "blocks"), [(32, 300, False), (60, 300, False), (32, 50, False), (160, 300, False), (160, 300, True)]
)
def test_shared_attention_pads_and_blocks_exactly(monkeypatch, d, lq, blocks):
    """`mlx.attention`: a head size MLX does not fuse is zero-padded (smaller; 300 x 260 scores, 50 x 260 unpadded)
    or run in query blocks (larger)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx  # noqa: PLC0415 - optional (the mlx extra)

    from fastforks._core import mlx as mlx_core  # noqa: PLC0415
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415

    if blocks:  # one query block per 7 rows of scores
        monkeypatch.setattr(mlx_core, "SDPA_BLOCK_BYTES", 2 * 3 * 260 * 4 * 7)
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, n, d) for n in (lq, 260, 260))
    mask = torch.randn(1, 1, lq, 260)
    for m in (None, mask):
        want = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=0.1)
        got = mlx_core.attention(*map(to_mx, (q, k, v)), scale=0.1, mask=None if m is None else to_mx(m))
        torch.testing.assert_close(to_torch(got), want, atol=1e-5, rtol=1e-5)
    half = mlx_core.attention(*(to_mx(t).astype(mx.float16) for t in (q, k, v)), scale=0.1)
    assert half.dtype == mx.float16


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_seg_resize_matches_torch_in_its_dtype(dtype):
    """`mlx_seg.resize` (the fused NHWC kernel) keeps the map's dtype and gives torch's half-pixel bilinear values."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx  # noqa: PLC0415 - optional (the mlx extra)

    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415
    from fastforks.shared import mlx_seg  # noqa: PLC0415

    torch.manual_seed(0)
    x = torch.randn(1, 5, 7, 3)
    want = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), (12, 9), mode="bilinear").permute(0, 2, 3, 1)
    a = to_mx(x).astype(getattr(mx, dtype))
    for fn in (mlx_seg.resize, mlx_seg._resize):
        got = fn(a, (12, 9))
        assert got.dtype == a.dtype
        tol = 2e-3 if dtype == "float16" else 1e-5
        torch.testing.assert_close(to_torch(got.astype(mx.float32)), want, atol=tol, rtol=tol)


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

            @staticmethod
            def _move_result(out):
                return out

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


@pytest.mark.parametrize(("features", "patch"), [((4, 8), 16), ((16, 32, 32), 32)])
def test_mlx_unet_matches_torch(features, patch):
    """`volume.mlx_net.MlxUNet` (direct and banded Winograd 3x3x3 convs) against the PyTorch `UNet` in float32."""
    pytest.importorskip("mlx.core")
    from fastforks.volume import _nnunet, mlx_net  # noqa: PLC0415 - optional (the mlx extra)

    n = len(features)
    arch = {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {
            "n_stages": n, "features_per_stage": list(features), "kernel_sizes": [[3, 3, 3]] * n,
            "strides": [[1, 1, 1]] + [[2, 2, 2]] * (n - 1), "n_conv_per_stage": [2] * n,
            "n_conv_per_stage_decoder": [2] * (n - 1), "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d", "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
        },
    }  # fmt: skip
    conf = {"patch_size": [patch] * 3, "spacing": [1.5] * 3, "normalization_schemes": ["CTNormalization"],
            "use_mask_for_norm": [False], "architecture": arch}  # fmt: skip
    props = {"0": {"mean": 50.0, "std": 200.0, "percentile_00_5": -900.0, "percentile_99_5": 1200.0}}
    plans = {"configurations": {"3d_fullres": conf}, "foreground_intensity_properties_per_channel": props}
    plan = _nnunet.read_plan(plans, {"channel_names": {"0": "CT"}, "labels": {"background": 0, "a": 1, "b": 2}})
    torch.manual_seed(0)
    net = _nnunet.UNet(plan, 1).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.InstanceNorm3d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.3)
    assert mlx_net.supported(plan)
    mnet = mlx_net.MlxUNet(mlx_net.params(net.state_dict(), plan), plan, "float32")
    x = torch.randn(1, 1, patch, patch, patch)
    with torch.no_grad():
        want = net(x)
    torch.testing.assert_close(mnet(x), want, atol=2e-3, rtol=2e-3)


def test_mlx_rfdetr_keypoint_matches_torch():
    """`shared.rfdetr.mlx_net.MlxRFDETR` with the keypoint branch against the PyTorch LWDETR (random weights, fp32)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    pytest.importorskip("transformers")
    from fastforks.pose.models.rfdetr_keypoint import model as kp  # noqa: PLC0415
    from fastforks.shared.rfdetr import mlx_net  # noqa: PLC0415 - optional (the mlx extra)

    _, v = registry.resolve("pose", "rf-detr-keypoint-preview")
    params = core_model.make_params(kp.Params, core_model.model_card("pose", v.id), v, {})
    torch.manual_seed(0)
    net = kp.network(params).eval()
    mnet = mlx_net.MlxRFDETR(net, "float32", kp.MEAN, kp.STD)
    r = params.resolution
    image = torch.rand(1, 3, r, r, generator=torch.Generator().manual_seed(1))
    mean, std = torch.tensor(kp.MEAN).view(1, 3, 1, 1), torch.tensor(kp.STD).view(1, 3, 1, 1)
    with torch.no_grad():
        want = net((image - mean) / std)
    got = mnet(image)
    for k in ("pred_logits", "pred_boxes", "pred_keypoints"):
        assert got[k].shape == want[k].shape, k
        torch.testing.assert_close(got[k], want[k].float(), atol=2e-3, rtol=2e-3, msg=k)


@pytest.mark.parametrize(("scales", "refine"), [(1, 0), (2, 0), (2, 2)])
def test_mlx_gmflow_matches_torch(scales, refine):
    """`gmflow.mlx_net.MlxGMFlow` against `model.forward` (random weights, float32, a batch of two pairs)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.flow.models.gmflow import mlx_net  # noqa: PLC0415
    from fastforks.flow.models.gmflow import model as gm  # noqa: PLC0415
    from fastforks.flow.models.gmflow.net.unimatch import UniMatch  # noqa: PLC0415

    torch.manual_seed(0)
    net = UniMatch(num_scales=scales, upsample_factor=8 // scales, reg_refine=refine > 0, task="flow").eval()
    with torch.no_grad():  # random weights give sharp softmaxes that amplify rounding; soften the features
        net.backbone.conv2.weight.mul_(0.3)
        net.backbone.conv2.bias.mul_(0.3)
    params = gm.Params(scales=scales, refine=refine, cuda_graphs=False)
    g = torch.Generator().manual_seed(1)
    image1 = torch.randn(2, 3, 64, 96, generator=g)
    image2 = torch.roll(image1, (2, 3), (2, 3)) + 0.1 * torch.randn(2, 3, 64, 96, generator=g)
    with torch.no_grad():
        want = gm.forward(net, image1, image2, params, fp16=False)
    got = to_torch(mlx_net.MlxGMFlow(net, "float32", scales, refine)(to_mx(image1), to_mx(image2)))
    assert got.shape == want.shape
    diff = (got - want).abs()
    bad = (diff > 2e-3 + 2e-3 * want.abs()).float().mean().item()  # the second scale's softmaxes amplify rounding
    assert bad < 0.01, f"{bad:.2%} outside tolerance, max {diff.max().item():.4g}"
    assert diff.median().item() < 1e-3  # flows of ~7 px: 4e-4 at two scales


@pytest.mark.parametrize("plus", [False, True])
def test_mlx_fast_acvnet_matches_torch(plus):
    """`fast_acvnet.mlx_net.MlxFastACV` against `model.forward` (random weights and batch norms, float32)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.stereo.models.fast_acvnet import mlx_net  # noqa: PLC0415
    from fastforks.stereo.models.fast_acvnet import model as fa  # noqa: PLC0415

    torch.manual_seed(0)
    net = (fa.Fast_ACVNet_plus if plus else fa.Fast_ACVNet)(192, att_weights_only=False).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.weight.uniform_(0.5, 1.5, generator=g)
                m.bias.normal_(0, 0.1, generator=g)
                m.running_mean.normal_(0, 0.1, generator=g)
                m.running_var.uniform_(0.5, 1.5, generator=g)
        if not plus:
            net.gamma.fill_(0.3)
    net._gb32 = (net.gamma.detach().clone(), net.beta.detach().clone()) if not plus else None
    left = torch.randn(1, 3, 128, 192, generator=g)
    right = torch.roll(left, -5, 3) + 0.05 * torch.randn(1, 3, 128, 192, generator=g)
    with torch.no_grad():
        want = fa.forward(net, left, right)
    got = to_torch(mlx_net.MlxFastACV(net, "float32")(to_mx(left), to_mx(right)))
    assert got.shape == want.shape
    # random weights leave the final costs of many candidates within 1e-4 of each other (the regression's top-2 pick
    # is then a tie break); the attention's top-k before it matches to 1e-8 there
    bad = ((got - want).abs() > 2e-3 + 2e-3 * want.abs()).float().mean().item()
    assert bad < 0.02, f"{bad:.2%} of the pixels differ"
    torch.testing.assert_close(got.median(), want.median(), atol=2e-3, rtol=2e-3)


def _random_batch_norms(net: torch.nn.Module, g: torch.Generator) -> None:
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.weight.uniform_(0.5, 1.5, generator=g)
                m.bias.normal_(0, 0.1, generator=g)
                m.running_mean.normal_(0, 0.1, generator=g)
                m.running_var.uniform_(0.5, 1.5, generator=g)


def test_mlx_igev_stereo_matches_torch():
    """`igev_stereo.mlx_net.MlxIGEV` against the folded `forward_fast` (random weights and batch norms, float32)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    import copy  # noqa: PLC0415
    import types  # noqa: PLC0415

    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.stereo.models.igev_stereo import mlx_net  # noqa: PLC0415
    from fastforks.stereo.models.igev_stereo.net.igev_stereo import IGEVStereo  # noqa: PLC0415
    from fastforks.stereo.models.igev_stereo.net.inference import fold_for_inference  # noqa: PLC0415

    torch.manual_seed(0)
    args = types.SimpleNamespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=192,
        mixed_precision=False, precision_dtype="float32",
    )  # fmt: skip
    net = IGEVStereo(args).eval()
    g = torch.Generator().manual_seed(1)
    _random_batch_norms(net, g)
    left = torch.rand(1, 3, 128, 192, generator=g) * 255
    right = (torch.roll(left, -5, 3) + 5 * torch.randn(1, 3, 128, 192, generator=g)).clamp(0, 255)
    ref = copy.deepcopy(net)
    fold_for_inference(ref, half=False, channels_last=False)
    with torch.no_grad():
        want = ref.forward_fast(left, right, iters=3)[:, 0]
    got = to_torch(mlx_net.MlxIGEV(net, "float32", 3)(to_mx(left), to_mx(right)))
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_mlx_selective_igev_matches_torch():
    """`selective_igev.mlx_net.MlxSelectiveIGEV` against the folded `forward_fast` (random weights and batch norms)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    import copy  # noqa: PLC0415
    import types  # noqa: PLC0415

    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.stereo.models.selective_igev import mlx_net, portable  # noqa: PLC0415
    from fastforks.stereo.models.selective_igev.net.igev_stereo import IGEVStereo  # noqa: PLC0415
    from fastforks.stereo.models.selective_igev.net.inference import fold_for_inference  # noqa: PLC0415

    torch.manual_seed(0)
    args = types.SimpleNamespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=192,
        mixed_precision=False, precision_dtype="float32",
    )  # fmt: skip
    net = IGEVStereo(args).eval()
    g = torch.Generator().manual_seed(1)
    _random_batch_norms(net, g)
    left = torch.rand(1, 3, 128, 192, generator=g) * 255
    right = (torch.roll(left, -5, 3) + 5 * torch.randn(1, 3, 128, 192, generator=g)).clamp(0, 255)
    ref = copy.deepcopy(net)
    fold_for_inference(ref, half=False, channels_last=False)
    with torch.no_grad():
        want = portable.forward_fast(ref, left, right, 3)[:, 0]
    got = to_torch(mlx_net.MlxSelectiveIGEV(net, "float32", 3)(to_mx(left), to_mx(right)))
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_mlx_igev_plusplus_matches_torch():
    """`igev_plusplus.mlx_net.MlxIGEVPlusPlus` against the folded `forward_fast` (random weights and batch norms).

    max_disp 384: three ranges of 24 bins (1/16 at 1, 1/8 at 2, 1/4 at 4), each halving evenly in the hourglass.
    """
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    import copy  # noqa: PLC0415
    import types  # noqa: PLC0415

    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.stereo.models.igev_plusplus import mlx_net, portable  # noqa: PLC0415
    from fastforks.stereo.models.igev_plusplus.net.igev_stereo import IGEVStereo  # noqa: PLC0415
    from fastforks.stereo.models.igev_plusplus.net.inference import fold_for_inference  # noqa: PLC0415

    torch.manual_seed(0)
    md = 384
    args = types.SimpleNamespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=md,
        s_disp_range=md // 16, m_disp_range=md // 8, l_disp_range=md // 4,
        s_disp_interval=1, m_disp_interval=2, l_disp_interval=4, mixed_precision=False, precision_dtype="float32",
    )  # fmt: skip
    net = IGEVStereo(args).eval()
    g = torch.Generator().manual_seed(1)
    _random_batch_norms(net, g)
    left = torch.rand(1, 3, 128, 192, generator=g) * 255
    right = (torch.roll(left, -5, 3) + 5 * torch.randn(1, 3, 128, 192, generator=g)).clamp(0, 255)
    ref = copy.deepcopy(net)
    fold_for_inference(ref, half=False, channels_last=False)
    with torch.no_grad():
        want = portable.forward_fast(ref, left, right, 3)[:, 0]
    got = to_torch(mlx_net.MlxIGEVPlusPlus(net, "float32", 3)(to_mx(left), to_mx(right)))
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(("arch", "size"), [("l0", 512), ("xl0", 512)])
def test_mlx_efficientvit_sam_matches_torch(arch, size):
    """`efficientvit_sam.mlx_net.MlxEncoder` against `fast.Trunk` and the final norm (random weights and batch norms).

    L0 covers the MBConv / ResBlock stages and the 5 x 5 attention scales, XL0 the fused MBConvs, the 3 x 3 scales and
    the four-times-expanded neck.
    """
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.promptable_segmentation.models.efficientvit_sam import mlx_net  # noqa: PLC0415
    from fastforks.promptable_segmentation.models.efficientvit_sam.fast import Trunk  # noqa: PLC0415
    from fastforks.promptable_segmentation.models.efficientvit_sam.model import Params, Sam  # noqa: PLC0415

    torch.manual_seed(0)
    enc = Sam(Params(image_size=1024, arch=arch, input_size=size, cuda_graphs=False)).image_encoder.eval()
    g = torch.Generator().manual_seed(1)
    _random_batch_norms(enc, g)
    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5, generator=g)
                m.bias.normal_(0, 0.1, generator=g)
    trunk = Trunk(enc, size).eval()
    x = torch.randn(1, 3, size, size, generator=g)
    with torch.no_grad():
        want = enc.norm(trunk(x))
    got = mlx_net.MlxEncoder(trunk, enc.norm, "float32")(to_mx(x.permute(0, 2, 3, 1)))
    got = to_torch(got).permute(0, 3, 1, 2)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_mlx_loma_matches_torch():
    """`loma.mlx_net.MlxLoMa` against the PyTorch `Matcher` (random weights and batch norms, float32).

    DaD's detector logits and keypoints, the small DeDoDe-B descriptor sampled at the same keypoints and the matcher's
    mutual matches on the same keypoints and descriptions (DeDoDe-G's DINOv2 is `mlx_vit`'s, tested there).
    """
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.matching.models.loma import mlx_net  # noqa: PLC0415
    from fastforks.matching.models.loma.model import SUBPIXEL_TEMP, Matcher, Params  # noqa: PLC0415
    from fastforks.matching.models.loma.net.dad import sample_keypoints  # noqa: PLC0415
    from fastforks.matching.models.loma.net.loma import filter_matches  # noqa: PLC0415
    from fastforks.shared.dedode import describe_keypoints  # noqa: PLC0415

    torch.manual_seed(0)
    params = Params(
        dim=64, input_dim=32, descriptor="dedode_b_vgg11", normalize_descriptor=True, heads=2, layers=4,
        threshold=0.0, long_side=192, size=128, max_keypoints=256,
    )  # fmt: skip
    net = Matcher(params).eval()
    g = torch.Generator().manual_seed(1)
    _random_batch_norms(net, g)
    det = torch.randn(1, 3, 128, 192, generator=g)
    res = torch.randn(1, 3, 128, 128, generator=g)
    with torch.no_grad():
        logits = net.detector(det)
        kp = sample_keypoints(logits, params.max_keypoints, 3, SUBPIXEL_TEMP)
        desc = describe_keypoints(net.descriptor(res), kp)
        kp1 = torch.roll(kp, 3, 0) * 0.9
        desc1 = torch.roll(desc, 3, 0) + 0.1 * torch.randn(desc.shape, generator=g)
        want_m, want_s = filter_matches(net.matcher(kp[None], kp1[None], desc[None], desc1[None]), 0.0)
    m = mlx_net.MlxLoMa(net, "float32")
    nhwc = to_mx(det).transpose(0, 2, 3, 1)
    torch.testing.assert_close(to_torch(m.detector(nhwc)), logits[0, 0], atol=1e-3, rtol=1e-3)
    got_kp = to_torch(m.keypoints(nhwc, params.max_keypoints, SUBPIXEL_TEMP))
    torch.testing.assert_close(got_kp, kp, atol=1e-4, rtol=0)
    got_desc = to_torch(m.descriptions(to_mx(res).transpose(0, 2, 3, 1), to_mx(kp)))
    torch.testing.assert_close(got_desc, desc, atol=1e-3, rtol=1e-3)
    got_m, got_s = (to_torch(t) for t in m.match(*(to_mx(t) for t in (kp, kp1, desc, desc1)), 0.0))
    torch.testing.assert_close(got_s, want_s, atol=1e-5, rtol=1e-3)
    assert (got_m == want_m).float().mean() > 0.99


@pytest.mark.parametrize(("scales", "hidden"), [((1.0,), 64), ((2.0, 0.5), 64), ((2.0, 0.5), 576)])
def test_mlx_lwdetr_matches_torch(scales, hidden):
    """`lwdetr.mlx_net.MlxLWDETR` against the folded PyTorch `model.Network` (random weights, float32).

    The cases cover the one-level projector (tiny / small / medium), the two-level one (large) and the pre-conv
    upsampling branch of encoders wider than 512 (xlarge).
    """
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    transformers = pytest.importorskip("transformers")
    from fastforks.detection.models.lwdetr import mlx_net  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.detection.models.lwdetr import model as lw  # noqa: PLC0415

    size = 256
    bb = {"hidden_size": hidden, "num_hidden_layers": 4, "num_attention_heads": 4, "image_size": size,
          "window_block_indices": [0, 2], "out_indices": [2, 4], "model_type": "lw_detr_vit"}  # fmt: skip
    config = transformers.LwDetrConfig(
        backbone_config=bb, projector_scale_factors=list(scales), d_model=64, decoder_ffn_dim=128,
        decoder_cross_attention_heads=4, decoder_self_attention_heads=2, decoder_n_points=2, num_queries=30,
        num_labels=91,
    )  # fmt: skip
    torch.manual_seed(0)
    net = transformers.LwDetrForObjectDetection(config).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d | torch.nn.BatchNorm1d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.1)
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 2.0)
    assert lw.fold_batch_norms(net)
    mnet = mlx_net.MlxLWDETR(net.state_dict(), config, "float32", size)
    x = torch.randint(0, 256, (1, 3, size, size), generator=torch.Generator().manual_seed(1)).float()
    mean, std = torch.tensor(lw.MEAN).view(1, 3, 1, 1), torch.tensor(lw.STD).view(1, 3, 1, 1)
    with torch.no_grad():
        want = lw.Network(net)((x / 255 - mean) / std)
    got = mnet(x)
    for name, g, w in zip(("logits", "boxes"), got, want, strict=True):
        assert g.shape == w.shape, name
        torch.testing.assert_close(g, w, atol=2e-3, rtol=2e-3, msg=name)


_RTDETR_ARCHS = {
    # RT-DETRv2-S's basic blocks; a bottleneck PResNet that stops before the last decoder layer
    "rtdetrv2-basic": {"blocks": [1, 1, 1, 1], "bottleneck": False, "enc_dim": 64, "enc_ffn": 128, "expansion": 0.5,
                       "dec_layers": 3, "eval_layers": 3},
    "rtdetrv2-bottleneck": {"blocks": [1, 1, 1, 1], "bottleneck": True, "enc_dim": 64, "enc_ffn": 128,
                            "expansion": 1.0, "dec_layers": 3, "eval_layers": 2},
}  # fmt: skip
_DFINE_BASE = {"stem": [16, 16], "mid": [16, 32, 64, 128], "out": [64, 128, 256, 512], "blocks": [1, 1, 2, 1],
               "layers": 3, "lab": True, "levels": 3, "enc_dim": 64, "dec_dim": 64, "enc_ffn": 128, "dec_ffn": 128,
               "csp_blocks": 1, "expansion": 0.5, "dec_layers": 3, "points": [3, 6, 3], "reg_scale": 4.0,
               "act": "relu"}  # fmt: skip
_DFINE_ARCHS = {
    "dfine-ragged": _DFINE_BASE,  # D-FINE-S..X: three levels, 3 / 6 / 3 points
    # D-FINE-N: two levels, a narrower decoder (the decoder input projection); DEIM's SiLU heads
    "dfine-nano-silu": {**_DFINE_BASE, "levels": 2, "dec_dim": 32, "points": [6, 6], "act": "silu", "lab": False},
}


@pytest.mark.parametrize("arch", [*_RTDETR_ARCHS, *_DFINE_ARCHS])
def test_mlx_rtdetr_matches_torch(arch):
    """`detection._rtdetr_mlx` against the PyTorch `RtDetr` / `DFine` (random weights, float32; float16 run)."""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks.detection import _rtdetr_mlx  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.detection._attention import Attention  # noqa: PLC0415
    from fastforks.detection.models.dfine import net as dfine  # noqa: PLC0415
    from fastforks.detection.models.rtdetrv2 import net as rtdetr  # noqa: PLC0415

    torch.manual_seed(0)
    common = {"num_queries": 50, "num_classes": 7}
    if arch in _RTDETR_ARCHS:
        net = rtdetr.RtDetr(rtdetr.Arch(**_RTDETR_ARCHS[arch], **common)).eval()
    else:
        net = dfine.DFine(dfine.Arch(**_DFINE_ARCHS[arch], **common)).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, Attention):  # its projection is `torch.empty` (filled by the checkpoint)
                torch.nn.init.xavier_uniform_(m.in_proj_weight)
                m.in_proj_bias.normal_(0, 0.02)
            elif arch in _DFINE_ARCHS and isinstance(m, dfine.Lab):
                m.scale.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.1)
    x = torch.rand(1, 3, 192, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
    got = _rtdetr_mlx.build(net, "float32")(x)
    assert [t.shape for t in got] == [t.shape for t in want]
    torch.testing.assert_close(got[0], want[0], atol=2e-3, rtol=2e-3)
    # random weights score neighbouring anchors almost alike: the top-k may swap them (same box, shifted along a
    # row or a column) or swap the last selected one for the next, so each wanted box looks for its match anywhere
    near = (got[1][:, :, None] - want[1][:, None]).abs().amax(-1).amin(1)
    assert (near < 2e-3).float().mean() >= 0.95, near.max()
    # float16: the query ranking may swap near ties, so compare the sets of the best queries' boxes
    half = _rtdetr_mlx.build(net, "float16")(x)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in half)
    torch.testing.assert_close(half[0].amax(1), want[0].amax(1), atol=0.05, rtol=0.05)


def _randomize_timm(net):
    """Random batch-norm statistics, attention biases and layer scales (their inits are trivial)."""
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d | torch.nn.BatchNorm1d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.1)
            elif hasattr(m, "attention_biases"):
                m.attention_biases.normal_(0, 0.5)
            elif isinstance(getattr(m, "gamma", None), torch.nn.Parameter):  # layer scales (init 1e-5)
                m.gamma.uniform_(0.2, 1.0)


@pytest.mark.parametrize(
    ("family", "arch", "size"),
    [
        ("tinyvit", "tiny_vit_5m_224", 224),
        ("tinyvit", "tiny_vit_21m_224", 200),  # pads the windows
        ("mobilenetv4", "mobilenetv4_conv_medium", 224),
        ("mobilenetv4", "mobilenetv4_conv_small", 160),
        ("inception", "inception_v3", 299),
        ("efficientvit", "efficientvit_b2", 288),
        ("efficientvit", "efficientvit_b0", 160),
        ("shvit", "shvit_s1", 224),
        ("shvit", "shvit_s4", 256),
        ("efficientformer", "efficientformer_l1", 224),
        ("fastvit", "fastvit_t8", 256),
        ("fastvit", "fastvit_sa12", 224),  # the attention stage
    ],
)
def test_mlx_timm_port_matches_timm(family, arch, size):
    """The `libnet` hand ports against the timm network (random weights, batch-norm statistics, attention biases)."""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Apple Metal")
    import importlib  # noqa: PLC0415

    import timm  # noqa: PLC0415

    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415

    port = getattr(importlib.import_module(f"fastforks.shared.libnet.mlx_{family}"), family)
    torch.manual_seed(0)
    if family == "inception":  # torchvision's, with the input remap of its ImageNet weights
        from torchvision.models import inception_v3  # noqa: PLC0415

        net = inception_v3(aux_logits=False, transform_input=True, num_classes=11, init_weights=False).eval()
    else:
        net = timm.create_model(arch, pretrained=False, num_classes=11).eval()
    _randomize_timm(net)
    x = torch.randn(2, 3, size, size, generator=torch.Generator().manual_seed(1))
    if family == "mobilenetv4":  # random statistics blow its activations past float16 (1e8): the batch's own instead
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        with torch.no_grad():
            net.train()(x)
        net.eval()
    with torch.no_grad():
        want = net(x)
    if family == "fastvit":  # the port takes the re-parameterized network, as the family's runner builds it
        from fastforks._optimize.fold import reparam  # noqa: PLC0415

        reparam(net)
    for dtype, tol in ((mx.float32, 2e-3), (mx.float16, 5e-2)):
        features, head = port(net, dtype)
        y = features(to_mx(x.permute(0, 2, 3, 1)).astype(dtype))
        got = to_torch(head(mx.mean(y.astype(mx.float32), axis=(1, 2)).astype(dtype)).astype(mx.float32))
        torch.testing.assert_close(got, want, atol=tol * want.abs().max().item(), rtol=tol, msg=str(dtype))


@pytest.mark.parametrize(
    ("arch", "kernels"),
    [
        ("fcn-r101-d8", False),
        ("psp-r101-d8", False),
        ("aspp-r101-d8", False),
        ("sepaspp-r101-d8", False),
        ("sepaspp-r101-d8", True),
        ("fcn-hr48", False),
        ("fcn-hr48", True),
    ],
)
def test_mlx_mmseg_matches_torch(arch, kernels):
    """`mmseg.mlx_net.MlxMMSeg` against the folded PyTorch network (random weights and BN statistics, float32)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.segmentation.models.mmseg import mlx_net  # noqa: PLC0415
    from fastforks.segmentation.models.mmseg import model as mmodel  # noqa: PLC0415
    from fastforks.segmentation.models.mmseg import net as mm  # noqa: PLC0415

    torch.manual_seed(0)
    net = mm.build(arch).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d | torch.nn.BatchNorm1d):
                m.weight.uniform_(0.3, 0.8)
                m.bias.normal_(0, 0.1)
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    mmodel.fold_batchnorms(net)
    mnet = mlx_net.MlxMMSeg(net, "float32", kernels=kernels)
    x = torch.randn(1, 3, 224, 256, generator=torch.Generator().manual_seed(1))  # Winograd on the stride-4 maps
    with torch.no_grad():
        want = net(x)
    got = to_torch(mnet(to_mx(x)))
    assert got.shape == want.shape
    scale = want.abs().max().item()
    torch.testing.assert_close(got, want, atol=2e-4 * scale, rtol=1e-3)


@pytest.mark.parametrize("norm", ["WithBias", "BiasFree"])
def test_mlx_restormer_matches_torch(norm):
    """The hand-written Restormer (`restormer/mlx_net.py`) against the vendored upstream network, float32."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.restormer import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.restormer.net import Restormer  # noqa: PLC0415

    torch.manual_seed(0)
    net = Restormer(dim=8, num_blocks=[1, 1, 2, 2], num_refinement_blocks=1, heads=[1, 2, 2, 4], LayerNorm_type=norm)
    for name, p in net.named_parameters():  # init: unit norms and temperatures would hide folding errors
        if ".norm" in name or name.endswith("temperature"):
            p.data.add_(torch.randn_like(p) * 0.5)
    net.eval()
    x = torch.rand(2, 3, 48, 40)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.MlxRestormer(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def _randomize_norms(net):
    """Norm weights and biases off their unit / zero init (which would hide a missing or misplaced norm)."""
    for name, p in net.named_parameters():
        if "norm" in name:
            p.data.add_(torch.randn_like(p) * 0.5)


@pytest.mark.parametrize(
    ("scale", "upsampler", "resi"),
    [(4, "pixelshuffle", "1conv"), (2, "pixelshuffle", "3conv"), (4, "nearest+conv", "3conv")],
)
@pytest.mark.filterwarnings("ignore:torch.meshgrid")
def test_mlx_swinir_matches_torch(scale, upsampler, resi, monkeypatch):
    """The hand-written SwinIR (`swinir/mlx_net.py`, `shared/swin_sr/mlx_net.py`) against the vendored network,
    float32: both upsamplers (the tail in row bands), both residual connections, shifted windows (the blocks in bands
    of window rows), a padded map."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.swinir import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.swinir.net import SwinIR  # noqa: PLC0415
    from fastforks.shared.swin_sr import mlx_net as swin  # noqa: PLC0415

    torch.manual_seed(0)
    net = SwinIR(
        upscale=scale, in_chans=3, img_size=16, window_size=4, img_range=1.0, depths=[2, 2], embed_dim=24,
        num_heads=[2, 3], mlp_ratio=2, upsampler=upsampler, resi_connection=resi,
    )  # fmt: skip
    _randomize_norms(net)
    net.eval()
    x = torch.rand(1, 3, 20, 12)  # a multiple of the window, not of the training size
    with torch.no_grad():
        want = net(x)
    monkeypatch.setattr(swin, "TAIL_BAND_BYTES", 64 * 12 * scale * scale * 4 * 6)  # 6-row bands
    monkeypatch.setattr(swin, "BLOCK_BAND_BYTES", 3 * 16 * 16 * 4 * 3 * 2)  # blocks in bands of 2-3 window rows
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("scale", "upsampler"), [(2, "pixelshuffle"), (4, "nearest+conv")])
@pytest.mark.filterwarnings("ignore:torch.meshgrid")
def test_mlx_swin2sr_matches_torch(scale, upsampler):
    """The hand-written Swin2SR (`swin2sr/mlx_net.py`) against the vendored network, float32: post-norm blocks,
    cosine attention with the continuous position bias, the groups' 1 x 1 patch-embedding convs, both upsamplers."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.swin2sr import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.swin2sr.net import Swin2SR  # noqa: PLC0415

    torch.manual_seed(0)
    net = Swin2SR(
        upscale=scale, in_chans=3, img_size=16, window_size=4, img_range=1.0, depths=[2, 2], embed_dim=24,
        num_heads=[2, 3], mlp_ratio=2, upsampler=upsampler, resi_connection="1conv",
    )  # fmt: skip
    _randomize_norms(net)
    with torch.no_grad():
        for name, t in net.named_parameters():  # the zero-initialised q / v biases would hide their layout
            if name.endswith(("q_bias", "v_bias")):
                t.add_(torch.randn_like(t) * 0.5)
    net.eval()
    x = torch.rand(1, 3, 20, 12)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.filterwarnings("ignore:torch.meshgrid")
def test_mlx_hat_matches_torch():
    """The hand-written HAT (`hat/mlx_net.py`) against the vendored network, float32: hybrid attention blocks with the
    channel-attention branch (weighted 1 here, not 0.01, so it shows), overlapping cross-attention, shifted windows."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.hat import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.hat.net import HAT  # noqa: PLC0415

    torch.manual_seed(0)
    net = HAT(
        upscale=4, in_chans=3, img_size=16, window_size=4, compress_ratio=3, squeeze_factor=4, conv_scale=1.0,
        overlap_ratio=0.5, img_range=1.0, depths=[2, 2], embed_dim=24, num_heads=[2, 3], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    )  # fmt: skip
    _randomize_norms(net)
    net.eval()
    x = torch.rand(1, 3, 20, 12)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.filterwarnings("ignore:torch.meshgrid")
def test_mlx_drct_matches_torch():
    """The hand-written DRCT (`drct/mlx_net.py`) against the vendored network, float32: the dense groups (growing
    widths, their head counts and 1 x 1 adjust convs), shifted windows."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.drct import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.drct.net import DRCT  # noqa: PLC0415

    torch.manual_seed(0)
    net = DRCT(
        upscale=4, in_chans=3, img_size=16, window_size=4, img_range=1.0, depths=[6, 6], embed_dim=24,
        num_heads=[2, 3], mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv", gc=8,
    )  # fmt: skip
    _randomize_norms(net)
    net.eval()
    x = torch.rand(1, 3, 20, 12)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("scale", "kernels", "per_rrdb"), [(4, False, False), (2, False, False), (4, True, True), (1, True, False)]
)
def test_mlx_realesrgan_matches_torch(scale, kernels, per_rrdb, monkeypatch):
    """The hand-written RRDBNet (`realesrgan/mlx_net.py`, `shared/swin_sr/mlx_cnn.py`) against the vendored network,
    float32: x4 and the pixel-unshuffled x2 / x1 inputs, the tail in row bands, portable and the Winograd kernels
    (their fused LeakyReLU and `0.2 * conv5 + x`; channels and maps large enough for `conv_mlx.supported`), one step per
    RRDB or per dense block."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.realesrgan import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.realesrgan.net import RRDBNet  # noqa: PLC0415

    torch.manual_seed(0)
    net = RRDBNet(3, 3, scale=scale, num_feat=32, num_block=2, num_grow_ch=16).eval()
    size = {4: (60, 52), 2: (120, 104), 1: (240, 208)}[scale]  # 60 x 52 after the unshuffle
    x = torch.rand(1, 3, *size)
    with torch.no_grad():
        want = net(x)
    monkeypatch.setattr(mlx_net, "TAIL_BAND_BYTES", 32 * 16 * 52 * 4 * 16)  # 16-row bands
    monkeypatch.setattr(mlx_net, "RRDB_STEP_BYTES", 0 if per_rrdb else 1 << 30)
    got = to_torch(mlx_net.network(net, "float32", kernels=kernels)(to_mx(x)))
    tol = 1e-3 if kernels else 1e-4  # Winograd F(4,3) rounds differently from the direct sum
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


@pytest.mark.parametrize(("scale", "kernels"), [(4, False), (2, True)])
def test_mlx_span_matches_torch(scale, kernels):
    """The hand-written SPAN (`span/mlx_net.py`) against the vendored network (its `Conv3XC`s fused), float32: the
    SPABs' attention, `conv_cat`'s taps (block 6's SiLU'd first conv), the shuffle; portable and Winograd convs."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.span import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.span.net import SPAN  # noqa: PLC0415
    from fastforks.restoration.models.span.net.span_arch import Conv3XC  # noqa: PLC0415

    torch.manual_seed(0)
    net = SPAN(3, 3, feature_channels=32, upscale=scale).eval()
    for m in net.modules():
        if isinstance(m, Conv3XC):
            m.update_params()
    x = torch.rand(1, 3, 60, 52)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32", kernels=kernels)(to_mx(x)))
    tol = 1e-3 if kernels else 1e-4  # Winograd F(4,3) rounds differently from the direct sum
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


@pytest.mark.parametrize(("scale", "kernels"), [(4, False), (2, True)])
def test_mlx_compact_matches_torch(scale, kernels):
    """The hand-written SRVGGNetCompact (`compact/mlx_net.py`) against the vendored network, float32: per-channel
    PReLUs (off their init), the nearest skip added before the shuffle; portable and Winograd convs (fused epilogue)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.compact import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.compact.net import SRVGGNetCompact  # noqa: PLC0415

    torch.manual_seed(0)
    net = SRVGGNetCompact(3, 3, num_feat=32, num_conv=4, upscale=scale, act_type="prelu").eval()
    for m in net.modules():
        if isinstance(m, torch.nn.PReLU):
            m.weight.data.uniform_(-0.5, 0.5)
    x = torch.rand(1, 3, 60, 52)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32", kernels=kernels)(to_mx(x)))
    tol = 1e-3 if kernels else 1e-4  # Winograd F(4,3) rounds differently from the direct sum
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


@pytest.mark.parametrize(("upsampler", "resi"), [("pixelshuffle", "1conv"), ("pixelshuffledirect", "3conv")])
@pytest.mark.filterwarnings("ignore:torch.meshgrid")
def test_mlx_dat_matches_torch(upsampler, resi):
    """The hand-written DAT (`dat/mlx_net.py`) against the vendored network, float32: spatial blocks unshifted and
    shifted in both groups' patterns, channel blocks, a size the windows pad, folded BatchNorms."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.dat import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.dat.net import DAT  # noqa: PLC0415

    torch.manual_seed(0)
    net = DAT(
        upscale=4, in_chans=3, img_size=16, img_range=1.0, split_size=[2, 4], depth=[4, 2], embed_dim=32,
        num_heads=[4, 4], expansion_factor=2.0, resi_connection=resi, upsampler=upsampler,
    )  # fmt: skip
    _randomize_norms(net)
    for m in net.modules():  # eval BatchNorms off their identity init, temperatures off one
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.3)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.normal_(1, 0.3)
            m.bias.data.normal_(0, 0.3)
        if hasattr(m, "temperature"):
            m.temperature.data.uniform_(0.5, 2.0)
    net.eval()
    x = torch.rand(1, 3, 18, 14)  # padded to 20 x 16 in the spatial attention
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("width", "tlc"), [(16, 32), (3, 0)])
def test_mlx_nafnet_matches_torch(width, tlc):
    """The hand-written NAFNet (`nafnet/mlx_net.py`) against the prepared network, float32: TLC windows / global."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.nafnet import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.nafnet import model as nm  # noqa: PLC0415

    torch.manual_seed(0)
    net = nm.NAFNet(img_channel=3, width=width, middle_blk_num=1, enc_blk_nums=[1, 2], dec_blk_nums=[1, 1])
    for name, p in net.named_parameters():  # zero beta / gamma and unit norms would hide folding errors
        if name.endswith(("beta", "gamma")) or ".norm" in name:
            p.data.add_(torch.randn_like(p) * 0.5)
    net.eval()
    with torch.no_grad():
        nm._prepare(net, tlc)
        x = torch.rand(2, 3, 38, 54)  # padded to 40 x 56; TLC: box windows at both levels
        want = net(x)
    got = to_torch(mlx_net.MlxNAFNet(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_mlx_segformer_matches_torch():
    """The hand-written SegFormer (`segformer/mlx_net.py`) against `_Net` (the folded transformers network), float32."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    transformers = pytest.importorskip("transformers")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.segmentation.models.segformer import mlx_net  # noqa: PLC0415
    from fastforks.segmentation.models.segformer.model import _Net  # noqa: PLC0415

    torch.manual_seed(0)
    config = transformers.SegformerConfig(
        hidden_sizes=[16, 32, 48, 64], depths=[1, 2, 1, 1], num_attention_heads=[1, 2, 2, 4],
        decoder_hidden_size=32, num_labels=5,
    )  # fmt: skip
    model = transformers.SegformerForSemanticSegmentation(config).eval()
    for name, p in model.named_parameters():  # unit norms would hide layout errors
        if "norm" in name:
            p.data.add_(torch.randn_like(p) * 0.3)
    bn = model.decode_head.batch_norm
    bn.running_mean.uniform_(-0.5, 0.5)
    bn.running_var.uniform_(0.5, 1.5)
    net = _Net(model, torch.float32, kernels=False).eval()
    x = torch.randn(2, 3, 70, 90)  # odd stage sizes: the spatial reductions crop
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.MlxSegFormer(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("scale", "use_ea", "size"), [(4, True, (50, 62)), (2, False, (17, 23))])
def test_mlx_realplksr_matches_torch(scale, use_ea, size):
    """The hand-written RealPLKSR (`realplksr/mlx_net.py`) against the vendored network, float32.

    50 x 62 takes the Winograd path of the 16 -> 32 channel mixer (`conv_mlx.supported`), 17 x 23 `mx.conv2d`.
    """
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.restoration.models.realplksr import mlx_net  # noqa: PLC0415
    from fastforks.restoration.models.realplksr.net import realplksr  # noqa: PLC0415

    torch.manual_seed(0)
    net = realplksr(dim=16, n_blocks=2, upscaling_factor=scale, kernel_size=7, use_ea=use_ea)
    for name, p in net.named_parameters():  # unit norms and zero biases would hide layout errors
        if "norm" in name:
            p.data.add_(torch.randn_like(p) * 0.3)
    net.eval()
    x = torch.rand(2, 3, *size)
    with torch.no_grad():
        want = net(x.clone())
    got = to_torch(mlx_net.MlxRealPLKSR(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("given", [False, True])
def test_mlx_unik3d_matches_torch(monkeypatch, given):
    """The hand-written UniK3D (`unik3d/mlx_net.py`) against the network, float32: four ViT-L blocks, random decoder."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks.depth.models.unik3d import mlx_net  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.depth.models.unik3d import net as un  # noqa: PLC0415

    monkeypatch.setattr(un, "LAYERS", (0, 1, 2, 3))  # read at run time by both: a 4-block encoder
    torch.manual_seed(0)
    net = un.UniK3DNet()
    net.pixel_encoder.blocks = net.pixel_encoder.blocks[:4]
    for name, p in net.named_parameters():  # unit norms / scales and zero positions would hide folding errors
        if "norm" in name or "gamma" in name or "latents_pos" in name or name.endswith(("pos_embed", "cls_token")):
            p.data.add_(torch.randn_like(p) * 0.3)
    net.eval()
    x = torch.randn(2, 3, 112, 154)
    with torch.no_grad():
        rays = net(x, None)[2] if given else None
        want = net(x, rays)
    got = mlx_net.MlxUniK3D(net, "float32")(x, rays)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "params",
    [
        {"deepen": 0.167, "widen": 0.375, "neck_blocks": 1, "depthwise": False},  # tiny
        {"deepen": 0.33, "widen": 0.25, "neck_blocks": 1, "depthwise": True},  # nano (separable convs)
    ],
)
def test_mlx_rtmdet_matches_torch(params):
    """The hand-written RTMDet (`rtmdet/mlx_net.py`, the shared CSPNeXt layers) against the network, float32."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.detection.models.rtmdet import mlx_net  # noqa: PLC0415
    from fastforks.detection.models.rtmdet.net import Arch, RTMDet  # noqa: PLC0415

    torch.manual_seed(0)
    net = RTMDet(Arch(num_classes=3, exp_on_reg=False, **params)).eval()
    for p in net.parameters():  # the default init is small: larger weights and biases exercise SiLU and attention
        p.data.mul_(1.5)
    x = torch.randn(2, 3, 128, 160, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
    got = mlx_net.MlxRTMDet(net, "float32")(to_mx(x.permute(0, 2, 3, 1)))
    assert len(got) == len(want) == 6
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(to_torch(g).permute(0, 3, 1, 2), w, atol=1e-4 * w.abs().max().item(), rtol=1e-3)


@pytest.mark.parametrize("graph", ["portable", "kernels"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mlx_yolox_matches_torch(graph, dtype):
    """The hand-written YOLOX (`yolox/mlx_net.py`, the shared CSPNeXt layers) against the network (tiny's widths)."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.detection.models.yolox import mlx_net  # noqa: PLC0415
    from fastforks.detection.models.yolox.net import YOLOX, Arch  # noqa: PLC0415
    from fastforks.shared.cspnext import mlx_net as csp  # noqa: PLC0415

    torch.manual_seed(0)
    net = YOLOX(Arch(depth=0.33, width=0.375, num_classes=3)).eval()
    x = 255 * torch.rand(2, 3, 128, 160, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
    csp.options(graph, gemm1x1=False)
    got = mlx_net.MlxYOLOX(net, dtype, graph)(to_mx(x.permute(0, 2, 3, 1)))
    assert len(got) == len(want) == 6
    tol = 1e-4 if dtype == "float32" else 2e-2
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(to_torch(g).permute(0, 3, 1, 2), w, atol=tol * w.abs().max().item(), rtol=tol * 10)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mlx_rtmo_matches_torch(dtype):
    """The hand-written RTMO network (`rtmo/mlx_net.py`, the shared CSPNeXt layers) against `net.RTMO`."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.pose.models.rtmo import mlx_net  # noqa: PLC0415
    from fastforks.pose.models.rtmo.net import RTMO, Arch  # noqa: PLC0415
    from fastforks.shared.cspnext import mlx_net as csp  # noqa: PLC0415

    torch.manual_seed(0)
    net = RTMO(Arch(deepen=0.33, widen=0.375, keypoints=17, size=128)).eval()
    torch.nn.init.xavier_uniform_(net.neck.encoder.in_proj_weight)  # `torch.empty` parameters (the checkpoint's)
    torch.nn.init.normal_(net.neck.encoder.in_proj_bias, std=0.02)
    x = 255 * torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
    csp.options(mlx_net.GRAPH, half_wino_f32=True)
    got = mlx_net.MlxRTMO(net, dtype, 128)(to_mx(x.permute(0, 2, 3, 1)))
    assert len(got) == len(want) == 4
    tol = 1e-4 if dtype == "float32" else 2e-2
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(to_torch(g), w, atol=tol * w.abs().max().item(), rtol=tol * 10)


def _randomise(net: torch.nn.Module) -> torch.nn.Module:
    """Random parameters that exercise every term: unit-gain maps, norms and layer scales away from identity."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, p[0].numel() ** -0.5)
            elif name.endswith(("gamma", "lambda1")) or ("norm" in name and name.endswith("weight")):
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0, 0.1)
    return net.eval()


@pytest.mark.parametrize(("registers", "ffn"), [(0, "mlp"), (4, "swiglufused")])
def test_mlx_dinov2_engine_matches_torch(registers, ffn):
    """`mlx_vit.DinoFeatureEngine` (the shared blocks, evaluated one by one) against the vendored DINOv2 ViT."""
    pytest.importorskip("mlx.core")
    from fastforks.features.models.dinov2 import model  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.shared import mlx_vit  # noqa: PLC0415
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    vit = DinoVisionTransformer(
        img_size=56, patch_size=14, embed_dim=48, depth=3, num_heads=4, mlp_ratio=4, init_values=1.0, ffn_layer=ffn,
        block_chunks=0, num_register_tokens=registers, interpolate_antialias=bool(registers),
        interpolate_offset=0.0 if registers else 0.1,
    )  # fmt: skip
    vit.__class__ = model._Vit
    _randomise(vit)
    params = mlx_vit.dinov2_params(dict(vit.state_dict()), 14)
    engine = mlx_vit.DinoFeatureEngine(params, "float32", 4, functools.partial(model._pos_embed, vit))
    x = torch.randn(2, 3, 42, 70)  # not the table's grid: the positions are interpolated
    with torch.no_grad():
        t = vit.prepare_tokens_with_masks(x)
        want = []
        for blk in vit.blocks:  # ty: ignore[not-iterable]
            t = blk(t)
            want.append(t)
        last = vit.norm(t)
    got_last, got = engine(x, (1, 3))
    assert (engine.depth, engine.registers, engine.hidden_size) == (3, registers, 48)
    torch.testing.assert_close(got_last, last, atol=2e-3, rtol=2e-3)
    for g, w in zip(got, (want[0], want[2]), strict=True):
        torch.testing.assert_close(g, w, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("image_logits", [True, False])
def test_mlx_dino_linear_probe_matches_torch(image_logits):
    """`mlx_vit.DinoLinearProbe`: the image head reads `[cls, mean of patches]`, the patch head every patch token."""
    pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_vit  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    vit = _randomise(
        DinoVisionTransformer(
            img_size=56, patch_size=14, embed_dim=48, depth=2, num_heads=4, mlp_ratio=4, init_values=1.0, block_chunks=0
        )
    )
    weight, bias = torch.randn(5, 96 if image_logits else 48) / 7, torch.randn(5)
    probe = mlx_vit.DinoLinearProbe(vit, weight, bias, "float32", image_logits=image_logits)
    x = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        t = vit.prepare_tokens_with_masks(x)
        for blk in vit.blocks:  # ty: ignore[not-iterable]
            t = blk(t)
        t = vit.norm(t)
        tokens = torch.cat([t[:, 0], t[:, 1:].mean(1)], -1) if image_logits else t[:, 1:]
        want = tokens @ weight.T + bias
    torch.testing.assert_close(probe(x), want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("gated", [False, True])
def test_mlx_dinov3_matches_torch(gated):
    """`dinov3.mlx_net` (packed q / k / v, folded LayerScale, RoPE with identity prefix rows) against `DINOv3ViT`."""
    pytest.importorskip("mlx.core")
    from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.features.models.dinov3 import mlx_net, net  # noqa: PLC0415
    from fastforks.shared.mlx_vit import rope_tables  # noqa: PLC0415

    cfg = net.Config(48, 4, 3, 96, gated=gated)
    vit = _randomise(net.DINOv3ViT(cfg))
    params = mlx_net.prepare(dict(vit.state_dict()))
    x = torch.randn(2, 3, 48, 80)
    cos, sin = net.rope_table(3, 5, cfg.hidden_size // cfg.num_heads, cfg.rope_theta, torch.device("cpu"))
    cos, sin = rope_tables(to_mx(cos), to_mx(sin), 1 + net.REGISTERS)
    out = mlx_net.forward(params, cfg, (0, 2, 3), 1.0, to_mx(x), cos, sin)
    mx.eval(*out)
    with torch.no_grad():
        want, hidden = vit.run(x, (0, 2, 3))
    torch.testing.assert_close(torch.from_numpy(np.array(out[0])), want, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        torch.from_numpy(np.stack([np.array(a) for a in out[1:]], 1)), hidden, atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("layer_scale", [False, True])
def test_mlx_c_radio_engine_matches_torch(layer_scale):
    """`c_radio.mlx_net.Engine` from `params` against the timm ViT with the checkpoint's position resampling."""
    pytest.importorskip("mlx.core")
    from fastforks.features.models.c_radio import mlx_net, model  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.shared.libnet import timm  # noqa: PLC0415

    dim, heads, patch, grid = 48, 4, model._PATCH, model._GRID
    vit = _randomise(
        timm().models.vision_transformer.VisionTransformer(
            img_size=grid * patch, patch_size=patch, num_classes=0, global_pool="", embed_dim=dim, depth=3,
            num_heads=heads, mlp_ratio=4, init_values=1.0 if layer_scale else None, class_token=False,
            pos_embed="none", no_embed_class=True, final_norm=False, dynamic_img_size=True,
        )
    )  # fmt: skip
    with torch.no_grad():
        vit.patch_embed.proj.bias.zero_()  # the checkpoint's embedder has none
    pos, tok, summary = torch.randn(1, grid * grid, dim) / 4, torch.randn(5, dim) / 4, [0, 2]
    state = dict(vit.state_dict())
    del state["patch_embed.proj.bias"]
    params = mlx_net.params(state, pos, tok, summary)
    assert not state
    engine = mlx_net.Engine(params, heads, "float32", align=False)
    x = torch.randn(2, 3, 3 * patch, 5 * patch)
    with torch.no_grad():
        t = vit.patch_embed.proj(x).flatten(2).transpose(1, 2) + model._resample(pos, 3, 5, align_corners=False)
        t = torch.cat([tok.expand(2, -1, -1), t], 1)
        want = []
        for blk in vit.blocks:  # ty: ignore[not-iterable]
            t = blk(t)
            want.append(t)
    got = engine(x, (1, 2))
    assert (engine.prefix_tokens, engine.summaries) == (5, 2)
    torch.testing.assert_close(got["last_hidden_state"], t, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got["pooler_output"], t[:, summary].flatten(1), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got["hidden_states"], torch.stack(want[:2], 1), atol=2e-3, rtol=2e-3)
    assert set(engine(x, ())) == {"last_hidden_state", "pooler_output"}


def _features_engine(monkeypatch, net: torch.nn.Module, dtype: str = "float32"):
    """`mlx_features.Engine` of a random timm network (user weights: converted, not cached)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks.shared.libnet import mlx_features  # noqa: PLC0415 - optional (the mlx extra)

    monkeypatch.setattr(mlx_features, "build", lambda *_, **__: net)
    ctx = SimpleNamespace(dtype=dtype, user_weights=object(), variant=SimpleNamespace(ref="test"))
    return mlx_features.Engine(ctx)  # ty: ignore[invalid-argument-type]


def _timm_features(net: torch.nn.Module, x: torch.Tensor, layers: tuple[int, ...]):
    """timm's `(last_hidden_state, pooler_output, [block outputs at layers])` (as `features._Walk`)."""
    outs = []
    hooks = [b.register_forward_hook(lambda *a: outs.append(a[-1])) for b in net.blocks]  # ty: ignore[not-iterable]
    with torch.no_grad():
        last = net.forward_features(x)
        pooled = net.forward_head(last, pre_logits=True)
    for h in hooks:
        h.remove()
    return last, pooled, [outs[i - 1] for i in layers]


@pytest.mark.parametrize("kind", ["core", "spatial", "spatial-noclass"])
def test_mlx_features_eva_matches_torch(monkeypatch, kind):
    """The Perception Encoder `Eva` of `mlx_features`: de-interleaved RoPE, cls rows, attention pool / patch mean.

    Off its table's grid (the positions resampled, the RoPE tables of the new grid), with LayerScale.
    """
    pytest.importorskip("mlx.core")
    from fastforks.shared.libnet import timm  # noqa: PLC0415

    core = kind == "core"
    net = _randomise(
        timm().models.eva.Eva(
            img_size=64, patch_size=16, embed_dim=64, depth=3, num_heads=4, mlp_ratio=4.0, attn_type="rope",
            use_pre_transformer_norm=True, use_post_transformer_norm=core, use_fc_norm=False, use_rot_pos_emb=True,
            ref_feat_shape=(4, 4), rope_grid_offset=1.0 if core else 0.0, rope_grid_indexing="xy",
            global_pool="map" if core else "avg", attn_pool_num_heads=4, num_classes=0, init_values=1.0,
            class_token=kind != "spatial-noclass", dynamic_img_size=True,
        )
    )  # fmt: skip
    engine = _features_engine(monkeypatch, net)
    assert engine.cfg.rope
    assert engine.cfg.avg_pool is not core
    assert engine.cfg.final_norm is core
    x = torch.randn(2, 3, 48, 80)
    last, pooled, hs = _timm_features(net, x, (1, 3))
    got_last, got_pooled, *got = engine(x, (1, 3))
    torch.testing.assert_close(got_last, last, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got_pooled, pooled, atol=2e-3, rtol=2e-3)
    for g, w in zip(got, hs, strict=True):
        torch.testing.assert_close(g, w, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("kind", ["clip", "siglip"])
def test_mlx_features_vit_matches_torch(monkeypatch, kind):
    """The CLIP / SigLIP ViT of `mlx_features`: `norm_pre`, QuickGELU, cls token / attention pool, resampled table."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.libnet import timm  # noqa: PLC0415

    clip = kind == "clip"
    net = _randomise(
        timm().models.vision_transformer.VisionTransformer(
            img_size=64, patch_size=16, embed_dim=64, depth=3, num_heads=4, num_classes=0,
            global_pool="token" if clip else "map", class_token=clip, pre_norm=clip,
            act_layer="quick_gelu" if clip else None, dynamic_img_size=True,
        )
    )  # fmt: skip
    engine = _features_engine(monkeypatch, net)
    x = torch.randn(2, 3, 48, 80)
    last, pooled, hs = _timm_features(net, x, (2,))
    got_last, got_pooled, *got = engine(x, (2,))
    torch.testing.assert_close(got_last, last, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got_pooled, pooled, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got[0], hs[0], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("kind", ["token", "avg-fc-norm", "map", "eva-swiglu", "eva-fused"])
def test_mlx_vit_classifier_matches_torch(kind):
    """`mlx_vit_cls.Network` against timm: ViT (cls token / patch mean with `fc_norm` / attention pool, LayerScale) and
    EVA-02 (separate q / k / v with k unbiased, interleaved RoPE, SwiGLU with its LayerNorm; fused qkv, plain MLP)."""
    pytest.importorskip("mlx.core")
    from fastforks.shared.libnet import mlx_vit_cls, timm  # noqa: PLC0415 - optional (the mlx extra)

    models = timm().models
    if kind.startswith("eva"):
        swiglu = kind == "eva-swiglu"
        net = models.eva.Eva(
            img_size=64, patch_size=16, embed_dim=64, depth=3, num_heads=4, num_classes=7, qkv_fused=not swiglu,
            swiglu_mlp=swiglu, scale_mlp=swiglu, mlp_ratio=4 * 2 / 3 if swiglu else 4.0, use_rot_pos_emb=True,
            ref_feat_shape=(4, 4), global_pool="avg", init_values=1.0 if not swiglu else None,
        )  # fmt: skip
    else:
        net = models.vision_transformer.VisionTransformer(
            img_size=64, patch_size=16, embed_dim=64, depth=3, num_heads=4, num_classes=7,
            global_pool="token" if kind == "token" else "avg" if kind.startswith("avg") else "map",
            class_token=kind != "map", fc_norm=kind.startswith("avg") or None, init_values=1.0, pre_norm=kind == "map",
        )  # fmt: skip
    net = _randomise(net)
    engine = mlx_vit_cls.Network(net, "float32")
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        want = net(x)
    torch.testing.assert_close(engine(x), want, atol=2e-3, rtol=2e-3)
    half = mlx_vit_cls.Network(net, "float16")
    torch.testing.assert_close(half(x), want, atol=5e-2, rtol=5e-2)
    with pytest.raises(ValueError, match="takes 64 x 64"):
        engine(torch.randn(1, 3, 48, 64))


@pytest.mark.parametrize("family", ["swin", "davit"])
def test_mlx_windowed_classifier_matches_torch(family):
    """`mlx_windowed` against timm: Swin (shifted windows, padded last stage), DaViT (spatial / channel blocks)."""
    pytest.importorskip("mlx.core")
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks.shared.libnet import mlx_cnn, timm  # noqa: PLC0415 - optional (the mlx extra)

    models = timm().models
    if family == "swin":
        net = models.swin_transformer.SwinTransformer(
            img_size=112, patch_size=4, window_size=4, embed_dim=32, depths=(2, 2, 2), num_heads=(2, 4, 4),
            num_classes=7,
        )  # fmt: skip  # maps 28 -> 14 -> 7: the last stage padded to 8, 7 > 4 keeps its shift
        size = 112
    elif family == "davit":
        net = models.davit.DaVit(depths=(1, 1, 1), embed_dims=(32, 64, 96), num_heads=(2, 4, 4), window_size=4,
                                 num_classes=7)  # fmt: skip
        size = 96  # maps 24 -> 12 -> 6 (6: padded windows)
    net = _randomise(net)
    for m in net.modules():  # BatchNorm statistics away from the identity too
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        want = net(x)
    for dtype, tol in (("float32", 2e-3), ("float16", 5e-2)):
        ctx = SimpleNamespace(dtype=dtype, mode="portable", card=SimpleNamespace(family=family))
        engine = mlx_cnn.Network(ctx, net)  # ty: ignore[invalid-argument-type]
        torch.testing.assert_close(engine(x), want, atol=tol, rtol=tol)


@pytest.mark.parametrize("pad_to", [{}, {16: 64, 32: 64}])
@pytest.mark.parametrize(("dtype", "tol"), [("float32", 2e-3), ("float16", 3e-2)])
def test_mlx_sam2_encoder_matches_torch(dtype, tol, pad_to):
    """`sam2.mlx_net.MlxEncoder` against upstream Hiera and the folded neck.

    The small trunk has every block kind: windows that divide the map and windows that need padding, a global block,
    and the three stage transitions with pooled queries (the last one from padded windows). `pad_to` runs the
    stages with head dim 16 and 32 through the fused attention kernel, as Base Plus's 56 is.
    """
    pytest.importorskip("mlx.core")
    from fastforks.promptable_segmentation.models.sam2 import fast, mlx_net  # noqa: PLC0415 - optional (mlx extra)
    from fastforks.promptable_segmentation.models.sam2.net.hieradet import Hiera  # noqa: PLC0415

    trunk = _randomise(
        Hiera(
            embed_dim=16, num_heads=1, stages=(1, 2, 3, 2), global_att_blocks=(4,),
            window_pos_embed_bkg_spatial_size=(7, 7), window_spec=(8, 4, 6, 3),
        )
    )  # fmt: skip

    def lin(cout: int, cin: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(cout, cin) * cin**-0.5, torch.randn(cout) / 10

    neck = fast.Neck(lin(24, 128), lin(24, 64), lin(8, 32), lin(4, 16), torch.randn(24) / 10)
    pos = fast.pos_embed(trunk, 32)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        want = fast.neck(neck, fast.trunk(trunk, x, pos))
    got = mlx_net.MlxEncoder(trunk, neck, pos, dtype, pad_to=pad_to)(x)
    assert set(got) == set(want)
    for k, w in want.items():
        torch.testing.assert_close(got[k], w, atol=tol, rtol=tol)


def _sam2_prompt_side(tokens: int):
    """SAM 2's prompt encoder and mask decoder (the released configuration) for a `tokens` x `tokens` embedding."""
    from fastforks.promptable_segmentation.models.sam2.net.mask_decoder import MaskDecoder  # noqa: PLC0415
    from fastforks.promptable_segmentation.models.sam2.net.prompt_encoder import PromptEncoder  # noqa: PLC0415
    from fastforks.promptable_segmentation.models.sam2.net.transformer import TwoWayTransformer  # noqa: PLC0415

    pe = PromptEncoder(
        embed_dim=256, image_embedding_size=(tokens, tokens), input_image_size=(16 * tokens, 16 * tokens),
        mask_in_chans=16,
    )  # fmt: skip
    dec = MaskDecoder(
        num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=True,
        iou_prediction_use_sigmoid=True, pred_obj_scores=True, pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True, dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98,
    )  # fmt: skip
    return _randomise(pe), _randomise(dec)


@pytest.mark.parametrize("mask_prompt", [False, True])
@pytest.mark.parametrize("prompts", [1, 3])
def test_mlx_sam2_decoder_matches_torch(prompts, mask_prompt):
    """`sam2.mlx_heads.MlxDecoder` against upstream `MaskDecoder.predict_masks`: one image for `prompts` prompts.

    With and without a mask prompt (the dense embedding a map, or one vector expanded over the map), the feature maps
    as PyTorch tensors of the encoder's MLX arrays (`Twins`, not converted again) and as plain tensors.
    """
    pytest.importorskip("mlx.core")
    from fastforks.promptable_segmentation.models.sam2 import mlx_heads, mlx_net  # noqa: PLC0415 - optional (mlx extra)

    t = 12
    pe, dec = _sam2_prompt_side(t)
    torch.manual_seed(1)
    embed, s1, s0 = torch.randn(1, 256, t, t), torch.randn(1, 64, 2 * t, 2 * t), torch.randn(1, 32, 4 * t, 4 * t)
    pts = (torch.rand(prompts, 2, 2) * 16 * t, torch.tensor([[1, 0]]).expand(prompts, 2))
    masks = torch.randn(prompts, 1, 4 * t, 4 * t) if mask_prompt else None
    with torch.no_grad():
        sparse, dense = pe(points=pts, boxes=None, masks=masks)
    kw = {
        "image_pe": pe.get_dense_pe(), "sparse_prompt_embeddings": sparse, "dense_prompt_embeddings": dense,
        "repeat_image": prompts > 1,
    }  # fmt: skip
    with torch.no_grad():
        want = dec.predict_masks(image_embeddings=embed, high_res_features=[s0, s1], **kw)
    twins = mlx_net.Twins()
    run = mlx_heads.MlxDecoder(dec, twins)
    plain = run(image_embeddings=embed, high_res_features=[s0, s1], **kw)
    kept = [twins.tensor(mlx_net.to_mx(x.permute(0, 2, 3, 1))) for x in (embed, s0, s1)]
    seen = len(twins.items)
    found = run(image_embeddings=kept[0], high_res_features=kept[1:], **kw)
    assert len(twins.items) == seen  # all three found again
    for got in (plain, found):
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(("dtype", "tol"), [("float32", 2e-3), ("float16", 3e-2)])
def test_mlx_sam2_memory_matches_torch(dtype, tol):
    """`sam2.mlx_heads` memory attention and memory encoder against the upstream modules, two objects.

    The memory is two frames and three object pointers (12 tokens the rotary encoding skips); the mask is 16x the
    features' side, as the image-size mask is.
    """
    pytest.importorskip("mlx.core")
    import mlx.core as mx  # noqa: PLC0415

    from fastforks.promptable_segmentation.models.sam2 import mlx_heads, mlx_net, video  # noqa: PLC0415 - optional

    t, p, plain = 8, 2, 12
    mods = _randomise(video.VideoModules(t))
    torch.manual_seed(2)
    n, m = t * t, 2 * t * t + plain
    curr, pos = torch.randn(n, p, 256), torch.randn(n, 1, 256)
    memory, memory_pos = torch.randn(m, p, 64), torch.randn(m, 1, 64)
    pix, mask = torch.randn(p, 256, t, t), torch.randn(p, 1, 16 * t, 16 * t) * 10
    with torch.no_grad():
        want = mods.memory_attention(
            curr=curr, curr_pos=pos.expand(-1, p, -1), memory=memory, memory_pos=memory_pos.expand(-1, p, -1),
            num_obj_ptr_tokens=plain,
        )  # fmt: skip
        want_mem = mods.memory_encoder(pix, mask, skip_mask_sigmoid=True)["vision_features"]

    def batch_first(x: torch.Tensor) -> mx.array:
        return mlx_net.to_mx(x.permute(1, 0, 2).contiguous())

    att = mlx_heads.MlxMemoryAttention(mods.memory_attention, getattr(mx, dtype))
    got = att(batch_first(curr), mlx_net.to_mx(pos[:, 0]), batch_first(memory), mlx_net.to_mx(memory_pos[:, 0]), plain)
    torch.testing.assert_close(mlx_net.to_torch(got).permute(1, 0, 2), want, atol=tol, rtol=tol)
    enc = mlx_heads.MlxMemoryEncoder(mods.memory_encoder, getattr(mx, dtype))
    got = enc(mlx_net.to_mx(pix.permute(0, 2, 3, 1)), mlx_net.to_mx(mask.permute(0, 2, 3, 1)))
    torch.testing.assert_close(
        mlx_net.to_torch(got.astype(mx.float32)).permute(0, 3, 1, 2), want_mem, atol=tol, rtol=tol
    )


@pytest.mark.parametrize("p_bg", [0.02, 0.3, 0.6, 0.97])
def test_mlx_sam2_fill_holes_matches_torch(p_bg):
    """`sam2.mlx_heads.fill_holes` gives exactly `video.fill_holes`, from sparse to mostly-background masks."""
    pytest.importorskip("mlx.core")
    from fastforks.promptable_segmentation.models.sam2 import mlx_heads, mlx_net, video  # noqa: PLC0415 - optional

    torch.manual_seed(4)
    low = torch.where(torch.rand(3, 1, 48, 40) < p_bg, -1.0, 1.0) * (torch.rand(3, 1, 48, 40) + 0.5)
    got = mlx_net.to_torch(mlx_heads.fill_holes(mlx_net.to_mx(low[:, 0])))[:, None]
    assert torch.equal(got, video.fill_holes(low))
    assert (got != low).any() or p_bg > 0.9


def test_mlx_sam2_tracker_matches_torch():
    """`sam2.mlx_heads.MlxTracker` against `video.Tracker` over 10 frames, two objects (float32).

    Ten frames fill the six-frame memory window and release the oldest maps; the heads run the MLX mask decoder on
    the conditioned maps handed back as PyTorch tensors.
    """
    pytest.importorskip("mlx.core")
    import copy  # noqa: PLC0415

    from fastforks.promptable_segmentation.models.sam2 import mlx_heads, mlx_net, video  # noqa: PLC0415 - optional

    t, frames = 8, 10
    pe, dec = _sam2_prompt_side(t)
    mods = _randomise(video.VideoModules(t))
    torch.manual_seed(3)
    no_mem = torch.randn(256) / 10
    feats = [
        {"image_embed": torch.randn(1, 256, t, t), "feat_s0": torch.randn(1, 32, 4 * t, 4 * t),
         "feat_s1": torch.randn(1, 64, 2 * t, 2 * t)}
        for _ in range(frames)
    ]  # fmt: skip
    coords, labels = torch.tensor([[[40.0, 50.0]], [[90.0, 30.0]]]), torch.ones(2, 1, dtype=torch.int32)

    def track(tracker: video.Tracker) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        with torch.no_grad():
            tracker.run(iter(feats), frames, coords=coords, labels=labels, mask=None, emit=lambda *a: out.append(a[1:]))
        return out

    want = track(video.Tracker(copy.deepcopy(mods), pe, copy.deepcopy(dec), no_mem, torch.float32, 16 * t))
    twins = mlx_net.Twins()
    dec.predict_masks = mlx_heads.MlxDecoder(dec, twins)
    got = track(mlx_heads.MlxTracker(mods, pe, dec, no_mem, "float32", 16 * t, twins))
    assert len(got) == frames
    for (low, obj), (want_low, want_obj) in zip(got, want, strict=True):
        torch.testing.assert_close(obj, want_obj, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(low, want_low, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_mlx_sam3_geometry_matches_torch(count):
    """`sam3.net.mlx_net.Net.geometry` against `sam3.GeometryEncoder`: the exemplar boxes' RoI pooling (two matmuls
    with the `roi_weights`), the detector's attention at head dim 32 and the float32 residual, on a 12 x 9 map.
    """
    pytest.importorskip("mlx.core")
    import mlx.core as mx  # noqa: PLC0415

    from fastforks.promptable_segmentation.models.sam3.net import mlx_net, sam3  # noqa: PLC0415 - optional (mlx extra)

    torch.manual_seed(0)
    h, w = 12, 9
    enc = _randomise(sam3.GeometryEncoder(mlx_net.GEOMETRY_LAYERS))
    vision, pos = torch.randn(1, h * w, sam3.DIM), torch.randn(1, h * w, sam3.DIM)
    ex = None
    if count:
        boxes = torch.rand(count, 4) * torch.tensor([0.6, 0.6, 0.35, 0.35]) + torch.tensor([0.2, 0.2, 0.02, 0.02])
        pool_y, pool_x = sam3.roi_weights(boxes, h, w)
        ex = sam3.Exemplars(boxes[None], torch.arange(count)[None] % 2, pool_y[None], pool_x[None])
    with torch.no_grad():
        want = enc(vision, pos, ex)

    net = object.__new__(mlx_net.Net)
    net.dt = mx.float32
    net.pos = mx.array(pos.numpy())
    net.w = {}
    for k, v in enc.state_dict().items():
        name = f"geometry_encoder.{k}"
        net.w[name] = mlx_net._bias(name, mlx_net._layout(name, mx.array(v.numpy())))
    fields = None if ex is None else [mlx_net.to_mx(t) for t in (ex.boxes, ex.labels, ex.pool_y, ex.pool_x)]
    got = mlx_net.to_torch(net.geometry(mx.array(vision.numpy()), fields))
    torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(("value", "on"), [(None, False), ("", False), ("0", False), ("true", False), ("1", True)])
def test_graph_switch_is_only_on_for_1(monkeypatch, value, on):
    from fastforks._core import mlx as runtime  # noqa: PLC0415 - optional (the mlx extra)

    if value is None:
        monkeypatch.delenv(runtime.GRAPH_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime.GRAPH_ENV, value)
    assert runtime.graph_switch() is on


@pytest.mark.slow
def test_graph_runners_are_mlx_runners_of_the_family():
    """Each `GRAPH_RUNNERS` (the graph switch) names only MLX modes the family's `RUNNERS` has, as runner classes."""
    src = Path(registry.__file__).parents[1]
    found = 0
    for path in sorted(src.glob("*/models/*/model.py")):
        if "GRAPH_RUNNERS" not in path.read_text(encoding="utf-8"):
            continue
        mod = importlib.import_module(".".join(("fastforks", *path.relative_to(src).with_suffix("").parts)))
        graph, runners = mod.GRAPH_RUNNERS, mod.RUNNERS
        assert graph, path
        assert set(graph) <= {k for k in runners if k.endswith("_mlx")}, path
        assert all(isinstance(r, type) and r is not runners[k] for k, r in graph.items()), path
        found += 1
    assert found > 40


@pytest.mark.parametrize("decoder", ["classic", "simple"])
def test_mlx_vitpose_matches_torch(decoder):
    """The hand-written ViTPose (`vitpose/mlx_net.py`) against the network with the runner's normalisation, float32:
    the padded patch embedding, the shared blocks, both decoders."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD  # noqa: PLC0415
    from fastforks.pose.models.vitpose import mlx_net  # noqa: PLC0415
    from fastforks.pose.models.vitpose.net import Arch, ViTPose  # noqa: PLC0415

    a = Arch(dim=64, depth=2, heads=2, part=0, experts=0, decoder=decoder, keypoints=5, height=64, width=48)
    net = _randomise(ViTPose(a))
    x = torch.rand(2, 3, 64, 48)
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        want = net((x - mean) / std)
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_mlx_hrnet_matches_torch():
    """The hand-written HRNet (`hrnet/mlx_net.py`) against the network, float32: the bottleneck stem, the
    transitions, every module's branches and exchange (nearest upsampling, stride-2 chains), the head."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.pose.models.hrnet import mlx_net  # noqa: PLC0415
    from fastforks.pose.models.hrnet.net import Arch, HRNet  # noqa: PLC0415

    net = _randomise(HRNet(Arch(channels=8, keypoints=5, height=128, width=96)))
    x = torch.randn(2, 3, 128, 96)
    with torch.no_grad():
        want = net(x)
    got = to_torch(mlx_net.network(net, "float32")(to_mx(x)))
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)


def test_mlx_detr_matches_torch():
    """The hand-written DETR (`detr/mlx_net.py`) against the network, float32: the ResNet with its max pool, the
    sine positions, the encoder and decoder layers, both heads."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.detection.models.detr import mlx_net  # noqa: PLC0415
    from fastforks.detection.models.detr.net import Arch, Detr  # noqa: PLC0415

    a = Arch(blocks=[1, 1, 2, 1], d_model=32, heads=4, enc_layers=2, dec_layers=2, ffn=64, num_queries=7, num_classes=9)
    net = _randomise(Detr(a))
    x = torch.randn(1, 3, 97, 130)
    with torch.no_grad():
        want = net(x)
    got = mlx_net.network(net, "float32")(to_mx(x))
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(to_torch(g), w, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("backbone", ["mobilenet0.25", "resnet50"])
def test_mlx_retinaface_matches_torch(backbone):
    """The hand-written RetinaFace (`retinaface/mlx_net.py`) against the network with the runner's input transform,
    float32: BatchNorms folded at load, the FPN's cropped upsampling on odd maps, SSH, the merged heads."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.pose.models.retinaface import mlx_net  # noqa: PLC0415
    from fastforks.pose.models.retinaface.net import Arch, RetinaFace  # noqa: PLC0415

    net = _randomise(RetinaFace(Arch(backbone=backbone, out_channels=64 if backbone == "mobilenet0.25" else 256)))
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    x = torch.rand(1, 3, 75, 102)
    mean = torch.tensor(mlx_net.MEAN_BGR).view(1, 3, 1, 1)
    with torch.no_grad():
        want = net(x.flip(1) * 255 - mean)
    got = mlx_net.network(net, "float32")(to_mx(x))
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(to_torch(g), w, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize(("stage1", "shared"), [(64, 1), (32, 0)])
def test_mlx_yunet_matches_torch(stage1, shared):
    """The hand-written YuNet (`yunet/mlx_net.py`) against the network, float32: the units with their BatchNorms
    folded, the pools, the neck, the shared head units of yunet-n."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.pose.models.yunet import mlx_net  # noqa: PLC0415
    from fastforks.pose.models.yunet.net import Arch, YuNet  # noqa: PLC0415

    net = _randomise(YuNet(Arch(stage1=stage1, shared_convs=shared)))
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    x = torch.rand(1, 3, 96, 128) * 255
    with torch.no_grad():
        want = net(x)
    got = mlx_net.network(net, "float32")(to_mx(x))
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(to_torch(g), w, atol=1e-2, rtol=1e-3)


def test_mlx_owlv2_matches_torch():
    """The hand-written OWLv2 (`owlv2/mlx_net.py`) against transformers' network as the portable runner calls it,
    float32: the processor's blur and resize, the image tower and heads, the causal text tower and its pooling."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    transformers = pytest.importorskip("transformers")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.detection.models.owlv2 import mlx_net  # noqa: PLC0415
    from fastforks.detection.models.owlv2 import model as owl  # noqa: PLC0415

    layers = {"intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 32}
    config = {
        "text_config": {**layers, "max_position_embeddings": owl.CONTEXT, "vocab_size": 100},
        "vision_config": {**layers, "image_size": 64, "patch_size": 16},
        "projection_dim": 32,
    }
    net = _randomise(transformers.Owlv2ForObjectDetection(transformers.Owlv2Config(**config))).eval()
    image = torch.rand(1, 3, 50, 70)
    mean, std = torch.tensor(owl.MEAN).view(1, 3, 1, 1), torch.tensor(owl.STD).view(1, 3, 1, 1)
    ids = torch.zeros(3, owl.CONTEXT, dtype=torch.int64)
    for i, n in enumerate((3, 5, owl.CONTEXT)):
        ids[i, :n] = torch.randint(1, 98, (n,))
        ids[i, n - 1] = 99  # the end-of-text token: the highest id
    with torch.no_grad():
        x = (owl.gaussian_resize(torch.nn.functional.pad(image, (0, 0, 0, 20)), 64) - mean) / std
        t = net.owlv2.vision_model.post_layernorm(net.owlv2.vision_model(pixel_values=x).last_hidden_state)
        f = net.layer_norm(t[:, 1:] * t[:, :1])
        emb = net.class_head.dense0(f)
        want = (
            emb / (torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-6),
            net.class_head.logit_shift(f),
            torch.nn.functional.elu(net.class_head.logit_scale(f)) + 1,
            torch.sigmoid(net.box_head(f) + net.box_bias),
        )
        q = net.owlv2.text_projection(net.owlv2.text_model(input_ids=ids).pooler_output)
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
    m = mlx_net.network(net.state_dict(), config, "float32")
    pix = m.pixels(to_mx(image))
    torch.testing.assert_close(to_torch(pix).permute(0, 3, 1, 2), x, atol=1e-4, rtol=1e-4)
    for g, w in zip(m.image(pix), want, strict=True):
        torch.testing.assert_close(to_torch(g), w, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(to_torch(m.text(ids.numpy())), q, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(("arch", "res_scale"), [("dinov2", 1.0), ("dinov3", 4.0)])
def test_mlx_eomt_matches_torch(arch, res_scale):
    """The hand-written EoMT (`eomt/mlx_net.py`) against the portable runner's folded network, float32: the blocks
    (the queries joining for the last ones, DINOv3's RoPE), the upscale blocks and heads, and the semantic scores."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    transformers = pytest.importorskip("transformers")
    from fastforks._core.mlx import mx, to_mx, to_torch  # noqa: PLC0415 - optional (the mlx extra)
    from fastforks.segmentation.models.eomt import mlx_net  # noqa: PLC0415
    from fastforks.segmentation.models.eomt import model as eomt  # noqa: PLC0415
    from fastforks.shared.seginfer import mask_semantic  # noqa: PLC0415

    config = {
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_blocks": 2,
        "num_queries": 6,
        "num_register_tokens": 4,
        "num_upscale_blocks": 2,
        "image_size": 64,
        "patch_size": 16,
        "num_labels": 5,
    }
    if arch == "dinov2":
        hf = transformers.EomtForUniversalSegmentation(
            transformers.EomtConfig(**config, mlp_ratio=2, layer_norm_eps=1e-6)
        )
    else:
        hf = transformers.EomtDinov3ForUniversalSegmentation(
            transformers.EomtDinov3Config(**config, intermediate_size=128, layer_norm_eps=1e-5, rope_theta=100.0)
        )
    hf = _randomise(hf)
    hf.attn_mask_probs.zero_()
    full = {**hf.config.to_dict(), **config}
    sd = {k: v.clone() for k, v in hf.state_dict().items()}
    net = eomt._Net(hf, torch.float32, res_scale=res_scale, kernels=False).eval()  # folds `hf`
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        masks, classes = net(x)
        want = mask_semantic(eomt.resize(masks, (64, 64)), classes)
    m = mlx_net.network(sd, full, res_scale, "float32")
    got_masks, got_classes = m.forward(to_mx(x).transpose(0, 2, 3, 1))
    torch.testing.assert_close(to_torch(got_masks).permute(0, 3, 1, 2), masks, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(to_torch(got_classes), classes, atol=2e-3, rtol=2e-3)
    scores = mlx_net.scores(m, to_mx(x[:1]).transpose(0, 2, 3, 1), [(0, 0)], 64)
    torch.testing.assert_close(to_torch(scores).permute(0, 3, 1, 2), want[:1], atol=1e-4, rtol=1e-3)
    # two windows of a 64 x 96 image: their scores averaged where they overlap
    wide = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        mk, cl = net(torch.cat([wide[..., :64], wide[..., 32:]]))
        lg = mask_semantic(eomt.resize(mk, (64, 64)), cl)
    ref = torch.zeros(1, lg.shape[1], 64, 96)
    ref[..., :64] += lg[0]
    ref[..., 32:] += lg[1]
    ref[..., 32:64] /= 2
    got = mlx_net.scores(m, to_mx(wide).transpose(0, 2, 3, 1), [(0, 0), (0, 32)], 64)
    mx.eval(got)
    torch.testing.assert_close(to_torch(got).permute(0, 3, 1, 2), ref, atol=1e-4, rtol=1e-3)
