"""Size ranges: `size_range`, the required-size error of the exports and the tested-size data."""

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks._core.export import _bounds, _dynamic_spec, check_exportable, export_onnx, resolve_size
from fastforks._core.model import Model
from fastforks._core.sizes import CAP, SizeRange, card_size_lines, size_range


def _variant(ref):
    return registry.resolve_any(ref)[1]


def test_range_from_card_io():
    v = _variant("s2m2-s")
    r = size_range(v)
    assert (r.multiple, r.native) == (v.io.pad_multiple, None)
    assert r.min_size >= v.io.min_size
    assert f"padded to a multiple of {v.io.pad_multiple}" in r.describe()


def test_range_without_core_padding_and_fixed_crop():
    assert size_range(_variant("depth-anything-v2-s")).multiple == 1  # pad = "none": the net resizes inside
    assert "inside the model" in size_range(_variant("depth-anything-v2-s")).describe()
    fixed = SizeRange(min_size=1, multiple=32, native=(224, 224), resize=256)
    assert "fixed 224x224" in fixed.describe(tested=False)


def test_tested_max_and_not_measured():
    r = SizeRange(min_size=32, multiple=32, tested_max=(1080, 1920), tested_alloc_mib=2048, tested_reserved_mib=3072)
    assert "tested up to 1080x1920 (largest size that ran; 2.0 GiB peak allocated, 3.0 reserved)" in r.describe()
    assert "no limit found" in SizeRange(min_size=1, multiple=1, tested_max=CAP).describe()
    assert "not measured yet" in SizeRange(min_size=32, multiple=32).describe()
    assert "tested" not in r.describe(tested=False)
    measured = [v for t in registry.TASKS for _, v in registry.cards(t).values() if size_range(v).tested_max]
    assert measured  # sizes.json is packaged and matches refs


# refs without a measured largest size, with the reason (cards never say "not measured" for anything else)
_NO_DISK = "weights not downloaded: the probe machine was at its 40 GB free-disk floor (2026-11-12)"
UNMEASURED = {
    "pe-core-g14-448": "weights 7.0 GB: over the probe machine's free-disk floor (2026-11-12)",
} | dict.fromkeys(
    [
        "vitpose-plus-huge",
    ],
    _NO_DISK,
)


def test_every_ref_has_a_tested_size():
    """A new ref needs its sizes.json entry: tools/bench_sizes.py on a GPU, then tools/extract_sizes.py."""
    refs = {v.ref.split("@")[0]: size_range(v, card) for t in registry.TASKS for card, v in registry.cards(t).values()}
    missing = sorted(r for r, s in refs.items() if not s.tested_max and r not in UNMEASURED)
    assert not missing, f"no tested size (run tools/bench_sizes.py + tools/extract_sizes.py): {missing}"
    assert not [r for r in UNMEASURED if r in refs and refs[r].tested_max], "measured now: drop it from UNMEASURED"


def test_card_lines_cover_every_variant():
    card, _ = registry.resolve_any("s2m2-s")
    lines = card_size_lines(card)
    assert lines[0].startswith("- Input size:")
    assert all(f"`{v.id}`" in lines[-1] for v in card.variants)


def _stub(  # noqa: PLR0913 - a stub with the fields the tests vary
    task,
    default,
    *,
    min_size=64,
    multiple=32,
    native=None,
    family="raft_stereo",  # a dynamic-capable family
):
    model: Any = object.__new__(Model)
    model.card = SimpleNamespace(task=task, family=family)
    model.variant = SimpleNamespace(ref="x@1", outputs=("depth",))
    model.backend = SimpleNamespace(mode="portable")
    model.export_size = lambda: default
    model.size_range = lambda: SizeRange(min_size=min_size, multiple=multiple, native=native)
    return model


def test_required_size_error_names_range_and_example():
    with pytest.raises(
        InputError, match=r"give an export size.*any H x W >= 64, padded to a multiple of 32.*--size 704x1280"
    ):
        resolve_size(_stub("stereo", None), None, None)
    assert resolve_size(_stub("stereo", None), 480, 640) == (480, 640)
    assert resolve_size(_stub("classification", (224, 224)), None, None) == (224, 224)
    with pytest.raises(InputError, match="both height and width"):
        resolve_size(_stub("stereo", None), 480, None)


def test_example_size_satisfies_every_family():
    seen = set()
    for t in registry.TASKS:
        for card, v in registry.cards(t).values():
            r = size_range(v, card)
            if r.native or r.encoder:
                continue
            h, w = r.example()
            assert h % r.multiple == 0
            assert w % r.multiple == 0
            assert min(h, w) >= r.min_size
            seen.add((h, w))
    assert (704, 1280) in seen  # the common case (multiples of 32 and 64 and 1)
    assert SizeRange(min_size=1, multiple=48).example() == (720, 1296)
    assert SizeRange(min_size=1000, multiple=32).example() == (1024, 1280)


def test_exact_min_passes_and_one_below_fails():
    model = _stub("stereo", None)
    check_exportable(model, 64, 64, 1)
    check_exportable(model, 64, 640, 1)
    with pytest.raises(InputError, match=r"63x64.*outside the supported range: any H x W >= 64"):
        check_exportable(model, 63, 64, 1)
    with pytest.raises(InputError, match=r"64x8.*any H x W >= 64"):
        check_exportable(model, 64, 8, 1)
    with pytest.raises(InputError, match="batch 0"):
        check_exportable(model, 64, 64, 0)


def test_model_size_range_of_a_loaded_model():
    from fastforks import stereo  # noqa: PLC0415

    try:
        model = stereo.load("s2m2-s", device="cpu")
    except Exception as e:  # noqa: BLE001 - weights are not local
        pytest.skip(f"s2m2-s not available: {e}")
    r = model.size_range()
    assert r == size_range(model.variant, model.card)
    assert r.multiple == model.variant.io.pad_multiple
    assert r.min_size >= model.variant.io.min_size


def test_dynamic_bounds_checks():
    model = _stub("stereo", None)
    _bounds(model, 128, 192, ((64, 64), (384, 384)), 1)  # multiples of 32, inside
    _bounds(model, 64, 64, ((64, 64), (384, 384)), 1)  # min and max themselves are in range
    with pytest.raises(InputError, match="must lie in the size range"):
        _bounds(model, 512, 192, ((64, 64), (384, 384)), 1)
    with pytest.raises(InputError, match="multiples of 32"):
        _bounds(model, 130, 192, ((64, 64), (384, 384)), 1)
    with pytest.raises(InputError, match=r"63x64.*outside the supported range"):
        _bounds(model, 128, 192, ((63, 64), (384, 384)), 1)  # one below the model's minimum of 64
    with pytest.raises(InputError, match="max >= min"):
        _bounds(model, 64, 64, ((64, 64), (64, 64)), 1)
    with pytest.raises(InputError, match="max >= min"):
        _bounds(model, 128, 128, ((128, 64), (64, 128)), 1)
    with pytest.raises(InputError, match="min < max on both axes"):
        _bounds(model, 64, 128, ((64, 64), (64, 256)), 1)  # one axis with equal bounds
    with pytest.raises(InputError, match="at least 64 px"):
        _bounds(_stub("stereo", None, min_size=32), 32, 64, ((32, 64), (128, 128)), 1)  # lo // 32 == 1: Dim min 1
    _bounds(model, 128, 192, None, 1)


def test_dynamic_bounds_rejected_where_meaningless():
    with pytest.raises(InputError, match="fixed 224x224 input"):
        _bounds(_stub("classification", (224, 224), native=(224, 224)), 224, 224, ((224, 224), (448, 448)), 1)
    with pytest.raises(InputError, match=r"dynamic export is not supported \(the model rounds"):
        _bounds(_stub("depth", None, family="depth_anything_v2"), 128, 128, ((64, 64), (256, 256)), 1)


class _Two(torch.nn.Module):
    def forward(self, a, b):
        return a + b


class _Varargs(torch.nn.Module):
    def forward(self, *inputs):
        return inputs[0] + inputs[1]


def test_dynamic_spec_follows_the_forward_signature():
    axes = {2: "h", 3: "w"}
    assert _dynamic_spec(_Two(), 2, axes) == ({2: "h", 3: "w"}, {2: "h", 3: "w"})
    assert _dynamic_spec(_Varargs(), 2, axes) == (({2: "h", 3: "w"}, {2: "h", 3: "w"}),)
    assert _dynamic_spec(torch.nn.Identity(), 1, axes) == ({2: "h", 3: "w"},)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, image):
        return self.conv(image).squeeze(1)


@pytest.mark.skipif(not importlib.util.find_spec("onnxscript"), reason="needs onnxscript (dependency group export)")
def test_dynamic_onnx_export_of_a_toy_model_on_the_cpu(tmp_path):
    model = _stub("depth", None, min_size=32, multiple=1)
    model.weights = None
    model.device = torch.device("cpu")
    toy = _Toy().eval()  # one network: the export's value check compares it with the file
    model.module = lambda: toy
    model.example_inputs = lambda h, w, *, batch=1: (torch.rand(batch, 3, h, w),)
    path = export_onnx(model, tmp_path / "t.onnx", 64, 96, size_range=((32, 32), (128, 160)))
    import onnx  # noqa: PLC0415

    dims = [
        [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in onnx.load(str(path)).graph.input
    ]
    assert dims[0][2] != 64, dims  # height and width are symbolic
    assert dims[0][3] != 96, dims
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert meta == {"fastforks.model": "x@1", "fastforks.weights": "official"}
    fixed = export_onnx(model, tmp_path / "f.onnx", 64, 96)
    fdims = [d.dim_value for d in onnx.load(str(fixed)).graph.input[0].type.tensor_type.shape.dim]
    assert fdims == [1, 3, 64, 96]


def test_patch_multiple_is_shown_where_the_family_has_one():
    card = SimpleNamespace(task="depth", family="depth_anything_v2")
    v = SimpleNamespace(ref="x@1", io=SimpleNamespace(crop=0, resize=0, pad_mode="none", pad_multiple=1, min_size=14))
    assert "ViT patch 14" in size_range(v, card).describe(tested=False)
    assert "ViT patch" not in size_range(v, SimpleNamespace(task="depth", family="midas")).describe(tested=False)


def test_thin_input_between_probe_minimum_and_pad_multiple_is_not_rejected():
    model: Any = object.__new__(Model)
    model.variant = SimpleNamespace(ref="x@1")
    model._min_size = 0
    model.size_range = lambda: SizeRange(min_size=64, multiple=64, reject_below=8)
    model._check_size(torch.zeros(3, 48, 640))  # padded to 64 inside, ran in the CPU check of 2026-10-31
    with pytest.raises(InputError, match="at least 8 px"):
        model._check_size(torch.zeros(3, 7, 640))
