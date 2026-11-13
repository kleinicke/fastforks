"""`module()` of every family's portable backend on the CPU: it equals the model call, compiles without graph breaks,
and `torch.export` reproduces it; for stereo also ONNX (onnxruntime).

Uses the smallest variant of each family whose weights are available locally (never downloaded here) and few
iterations: the traces are the same code at any iteration count.
"""

from __future__ import annotations

import gc
import importlib.util
import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from fastforks import stereo
from fastforks._core import model as core_model
from fastforks._core import registry
from fastforks._core.errors import BackendUnavailableError, InputError
from fastforks._core.export import flat_outputs, output_names
from fastforks._core.preprocess import resize_crop, scale
from fastforks._core.tasks import spec
from fastforks.stereo import _module  # noqa: F401 - registers the kernels custom op
from fastforks.stereo._export import _cuda_device, _fix_folded_weight_casts
from synth import pair, size
from test_golden import FAMILIES, GOLDEN, SMALLEST, TASK, sweep_model

LOAD_ONCE = True  # conftest runs the tests of one ref together: `sweep_model` loads it once

SHAPE = (1, 3, 96, 128)  # not a multiple of every pad multiple: padding and cropping are in the traced graph


_ONNX_NEEDS_NEWER_TORCH = {"c-fast-foundationstereo@1", "foundationstereo-vit-s@1", "selective-igev-eth3d-finetune@1"}
_ONNX_KNOWN_MISMATCH = {  # open: the export runs, ONNX Runtime disagrees (or rejects the graph)
    "bridgedepth-sf@1": "top-k ties on the noise test pair resolve differently in ORT (synthetic textured pair: max"
    " 0.010 px, mean 0.001)",
    "monster-mix-all@1": "7 of 12288 pixels differ by up to 0.3 px",
    "nmrf-sceneflow@1": "top-k ties on the noise test pair resolve differently in ORT (synthetic textured pair: max"
    " 0.56 px, mean 0.006)",
}

_NOT_TRACEABLE = {  # module() runs, but neither dynamo nor torch.export captures it: documented
    "aliked-lightglue@1": "sparse matching: the keypoint and match counts are data-dependent (card limitation)",
    "xfeat@1": "sparse matching: the keypoint and match counts are data-dependent (card limitation)",
}
_GRAPH_BREAKS = {  # torch.compile graph-breaks (fine without fullgraph); torch.export captures the graph
    "rf-detr-keypoint-preview@1": "pinned upstream disables Dynamo for positional interpolation to preserve"
    " antialiased bicubic gradients; non-strict torch.export still captures this fixed-shape inference",
    "rf-detr-seg-n-coco@1": "pinned upstream disables Dynamo for positional interpolation (as RF-DETR keypoint);"
    " non-strict torch.export still captures this fixed-shape inference",
    "zoedepth-nk@1": "the NK model routes each image to the NYU or KITTI head by the argmax of its domain logits"
    " (.item(); the export runs both heads and selects)",
}


STEREO = [r for r in SMALLEST if TASK[r] == "stereo"]
DEPTH = [r for r in SMALLEST if TASK[r] == "depth"]
FEATURES = [r for r in SMALLEST if TASK[r] == "features"]
MULTIVIEW = [r for r in SMALLEST if TASK[r] == "multiview"]
_CAMERAS = ("intrinsics", "cam_to_world", "depth_range")  # the multiview inputs after the images, in module() order
_CPU_RAM_HEAVY = {  # compile + torch.export on the CPU need more RAM than the suite's 20 GB cgroup
    "marigold-depth-v1-1@1": "CPU torch.export of the Stable Diffusion UNet + VAE is OOM-killed at 20 GB RAM (also"
    " before export-all); run it alone in a larger cgroup with FASTFORKS_HEAVY_CPU_TESTS=1",
    "marigold-normals-lcm-v0-1@1": "the same Stable Diffusion UNet + VAE as Marigold depth; run it alone in a larger"
    " cgroup with FASTFORKS_HEAVY_CPU_TESTS=1",
    "lotus-normal-d-v1-1@1": "the same Stable Diffusion UNet + VAE as Marigold depth; run it alone in a larger cgroup"
    " with FASTFORKS_HEAVY_CPU_TESTS=1",
    "stablenormal-turbo@1": "the Stable Diffusion UNet + VAE of Marigold depth plus a ControlNet encoder; run it alone"
    " in a larger cgroup with FASTFORKS_HEAVY_CPU_TESTS=1",
    "depth-pro@1": "CPU dynamo explain + torch.export of the two DINOv2-L encoders at 1536 px is OOM-killed at 20 GB"
    " RAM (after a 7.5 GB load); run it alone in a larger cgroup with FASTFORKS_HEAVY_CPU_TESTS=1",
    "sam3@1": "CPU dynamo explain + torch.export peak at 13.2 GB RSS, above the suite's 13 GB cap; run it alone in a"
    " larger cgroup with FASTFORKS_HEAVY_CPU_TESTS=1",
}
_ONNX_SKIP_MULTIVIEW: dict[str, str] = {}
_ONNX_SKIP_DEPTH = {  # exports, but too big for the CPU suite
    "marigold-depth-v1-1@1": "Stable Diffusion UNet + VAE (> 2 GB graph, minutes on the CPU); checked on the GPU",
    "marigold-normals-lcm-v0-1@1": "Stable Diffusion UNet + VAE (> 2 GB graph, minutes on the CPU); checked on the GPU",
    "lotus-normal-d-v1-1@1": "Stable Diffusion UNet + VAE (> 2 GB graph, minutes on the CPU); checked on the GPU",
    "stablenormal-turbo@1": "Stable Diffusion UNet + ControlNet + VAE (> 2 GB graph); checked on the GPU",
}
_HAS_ONNX = bool(importlib.util.find_spec("onnxscript") and importlib.util.find_spec("onnxruntime"))
ONNX_TESTS = "FASTFORKS_ONNX_TESTS"  # 1: the ONNX export + ONNX Runtime checks (release only: ~27 min of the CPU suite)
_RUN_ONNX = _HAS_ONNX and bool(os.environ.get(ONNX_TESTS))
_needs_onnx = pytest.mark.skipif(
    not _RUN_ONNX, reason=f"release only ({ONNX_TESTS}=1); needs onnxscript + onnxruntime (dependency group export)"
)


def _onnx_checks_the_export(ref: str) -> bool:
    """The ONNX test of `ref` passes: it captures `module()` with the same `torch.export(strict=False)` (torch.onnx's
    first strategy) and compares ONNX Runtime with `module()`, so test_module_compile_export can skip the export."""
    if not _RUN_ONNX or TASK[ref] not in {"stereo", "depth", "features", "multiview"}:
        return False
    if ref in _ONNX_NEEDS_NEWER_TORCH and torch.__version__ < "2.14":
        return False
    return ref not in _ONNX_KNOWN_MISMATCH | _ONNX_SKIP_DEPTH | _ONNX_SKIP_MULTIVIEW | _NOT_TRACEABLE


def _load(ref: str) -> Any:
    return sweep_model(ref)


def _shape(model: Any) -> tuple[int, ...]:
    return (*SHAPE[:2], *size(model.size_range().min_size, *SHAPE[2:]))


def _pair(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return torch.rand(shape, generator=gen), torch.rand(shape, generator=gen)


def _inputs(ref: str, model: Any) -> tuple[torch.Tensor, ...]:
    pair = _pair(_shape(model))
    if TASK[ref] == "classification":  # the module takes transformed images (the model call's [io] transform)
        io = model.variant.io
        return (scale(resize_crop(pair[0], io), io),)
    if TASK[ref] in {"multiview", "tracking"}:  # one scene / video: views or frames (+ cameras / queries)
        return model.example_inputs(*_shape(model)[2:], batch=2)
    return pair if TASK[ref] in {"stereo", "flow", "matching", "interpolation"} else pair[:1]


def _multiview_call(model: Any, inputs: tuple[torch.Tensor, ...]) -> Any:
    """The model call on `module()` inputs (depth_range as a `(near, far)` pair)."""
    kw = {n: tuple(x.tolist()) if n == "depth_range" else x for n, x in zip(_CAMERAS, inputs[1:], strict=False)}
    return model(inputs[0], **kw)


def _main_map(result: Any) -> torch.Tensor:
    """The first standard map of a result (what a non-classification `module()` returns)."""
    return next(getattr(result, k) for k in spec(TASK[result.model]).standard if getattr(result, k) is not None)


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", SMALLEST)
def test_module_in_eval_mode(ref):
    """`module()` is in eval mode, submodules included: export would trace training-mode branches."""
    module = _load(ref).module()
    training = [name or type(module).__name__ for name, m in module.named_modules() if m.training]
    assert not training, f"{ref}: module() in training mode: {training[:5]}"


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """A failed trace must not leave dynamo / dispatcher state behind: every later test would fail with it."""
    yield
    torch._dynamo.reset()


def _check_export_names(ref: str, model: Any, module: torch.nn.Module, out: Any) -> None:
    """The ONNX / TensorRT output names (R2-2) name what `module()` returns: one name per tensor, no duplicates.

    One tensor of a map task is the variant's main output; a tuple, and a promptable encoder (its embedding is not a
    card output), carries the names its module declares; a dict its keys.
    """
    names = output_names(model, module, out)
    assert len(names) == len(flat_outputs(out)) == len(set(names)), f"{ref}: {names}"
    if isinstance(out, dict):
        return
    if isinstance(out, torch.Tensor) and TASK[ref] != "promptable_segmentation":
        assert names == [_main_map_name(ref)], f"{ref}: {names}"
        return
    assert tuple(names) == tuple(getattr(module, "output_names", ())), f"{ref}: {names} are not declared by module()"
    if TASK[ref] == "promptable_segmentation":
        assert not set(names) & set(model.variant.outputs), f"{ref}: {names} reuse card outputs"


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.filterwarnings("ignore:The .grad attribute of a Tensor that is not a leaf")  # dynamo reads .grad
@pytest.mark.parametrize("ref", SMALLEST)
def test_module_compile_export(ref):  # noqa: C901 - one branch per task
    if ref in _NOT_TRACEABLE:
        pytest.xfail(_NOT_TRACEABLE[ref])
    if ref in _CPU_RAM_HEAVY and not os.environ.get("FASTFORKS_HEAVY_CPU_TESTS"):
        pytest.skip(_CPU_RAM_HEAVY[ref])
    model = _load(ref)
    module = model.module()
    inputs = _inputs(ref, model)
    with torch.inference_mode():
        want = module(*inputs)
    _check_export_names(ref, model, module, want)
    if TASK[ref] in {"stereo", "depth"}:  # [B, 3, H, W] in [0, 1] -> the main map, as the model call
        channels = (3,) if _main_map_name(ref) == "normals" else ()  # normals-only models: [B, H, W, 3]
        assert want.shape == (1, *inputs[0].shape[2:], *channels)
        assert torch.equal(want[0], _main_map(model(*(x[0] for x in inputs))))
    elif TASK[ref] == "flow":  # two [B, 3, H, W] in [0, 1] -> [B, 2, H, W], as the model call
        assert want.shape == (1, 2, *inputs[0].shape[2:])
        assert torch.equal(want[0], _main_map(model(*(x[0] for x in inputs))))
    elif TASK[ref] == "interpolation":  # two [B, 3, H, W] in [0, 1] -> the middle frame [B, 3, H, W]
        assert want.shape == (1, 3, *inputs[0].shape[2:])
        assert torch.equal(want[0], model(*(x[0] for x in inputs)).frame)
    elif TASK[ref] == "features":  # -> last_hidden_state [B, N, C], as the model call with the default size
        assert torch.equal(want[0], model(inputs[0][0]).last_hidden_state)
    elif TASK[ref] == "classification":  # transformed images -> logits, as the model call on the raw image
        assert want.shape == (1, len(model.class_names))
        assert torch.equal(want[0].float(), model(_pair(_shape(model))[0][0]).logits)
    elif TASK[ref] == "multiview":  # one scene -> a dict of maps, as the model call
        assert torch.equal(want["depth"].float(), _multiview_call(model, inputs).depth)
    elif TASK[ref] == "tracking":  # one video + queries -> {"tracks", "visibility"}, as the model call
        assert torch.equal(want["tracks"], model(inputs[0], inputs[1]).tracks)

    if ref not in _GRAPH_BREAKS:
        explain = torch._dynamo.explain(module)(*inputs)
        assert explain.graph_break_count == 0, explain.break_reasons  # torch.compile(fullgraph=True) works

    if _onnx_checks_the_export(
        ref
    ):  # the same capture, checked end to end there (CPU: export + run ~45 % of this test)
        return
    ep = torch.export.export(module, inputs, strict=False)
    with torch.inference_mode():
        got = ep.module()(*inputs)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)  # fp32 reordering noise grows with the value


@pytest.mark.model
@_needs_onnx
@pytest.mark.parametrize("ref", STEREO)
def test_onnx(ref, tmp_path, request):
    import onnxruntime as ort  # noqa: PLC0415 - optional

    if ref in _ONNX_KNOWN_MISMATCH:
        request.applymarker(pytest.mark.xfail(reason=_ONNX_KNOWN_MISMATCH[ref]))

    if ref in _ONNX_NEEDS_NEWER_TORCH and torch.__version__ < "2.14":
        pytest.xfail("torch 2.8's ONNX exporter has no conversion of aten.adaptive_max_pool2d (2.14 has)")
    model = _load(ref)
    shape = _shape(model)
    left, right = _pair(shape)
    path = model.export_onnx(tmp_path / "m.onnx", *shape[2:])
    with torch.inference_mode():
        want = model.module()(left, right)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (got,) = session.run(["disparity"], {"left": left.numpy(), "right": right.numpy()})
    torch.testing.assert_close(torch.from_numpy(got), want, rtol=1e-3, atol=0.05)  # fp32 reordering in ORT kernels


@pytest.mark.model
@_needs_onnx
@pytest.mark.parametrize("ref", ["c-fast-foundationstereo@1"])
def test_onnx_ffs_variants(ref, tmp_path):
    """Variants of a family beyond the smallest: the eager-only GEMM group-wise volume must not reach the export."""
    if ref in _ONNX_NEEDS_NEWER_TORCH and torch.__version__ < "2.14":
        pytest.xfail("torch 2.8's ONNX exporter has no conversion of aten.adaptive_max_pool2d (2.14 has)")
    test_onnx(ref, tmp_path, SimpleNamespace(applymarker=lambda _m: None))


@pytest.mark.model
@_needs_onnx
@pytest.mark.parametrize("ref", DEPTH)
def test_onnx_depth(ref, tmp_path):
    import onnxruntime as ort  # noqa: PLC0415 - optional

    if ref in _NOT_TRACEABLE:
        pytest.xfail(_NOT_TRACEABLE[ref])
    if ref in _ONNX_SKIP_DEPTH:
        pytest.skip(_ONNX_SKIP_DEPTH[ref])
    model = _load(ref)
    (image,) = model.example_inputs(*SHAPE[2:])
    path = model.export_onnx(tmp_path / "m.onnx", *SHAPE[2:])
    with torch.inference_mode():
        want = model.module()(image)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == [_main_map_name(ref)]
    (got,) = session.run(None, {"image": image.numpy()})
    torch.testing.assert_close(torch.from_numpy(got), want, rtol=1e-3, atol=1e-3 * want.abs().max().item())


@pytest.mark.model
@_needs_onnx
@pytest.mark.parametrize("ref", FEATURES)
def test_onnx_features(ref, tmp_path):
    import onnxruntime as ort  # noqa: PLC0415 - optional

    model = _load(ref)
    size = model.export_size()
    assert all(s % model.stride == 0 for s in size)  # the exported graph runs at export_size, not a snapped size
    (image,) = model.example_inputs(*size)
    path = model.export_onnx(tmp_path / "m.onnx")
    with torch.inference_mode():
        want = model.module()(image)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == ["last_hidden_state"]
    (got,) = session.run(None, {"image": image.numpy()})
    got = torch.from_numpy(got)  # ORT types the outputs as a union (arrays, sparse tensors, sequences, maps)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3 * want.abs().max().item())


@pytest.mark.model
@_needs_onnx
@pytest.mark.parametrize("ref", MULTIVIEW)
def test_onnx_multiview(ref, tmp_path):
    """One scene at a fixed size and view count: the required inputs, every map of `module()` as an output."""
    import onnxruntime as ort  # noqa: PLC0415 - optional

    if ref in _ONNX_SKIP_MULTIVIEW:
        pytest.skip(_ONNX_SKIP_MULTIVIEW[ref])
    model = _load(ref)
    inputs = _inputs(ref, model)
    v, _, h, w = inputs[0].shape
    with pytest.raises(InputError, match="fixed size"):
        model.export_onnx(tmp_path / "m.onnx", h, w, batch=v, size_range=((h, w), (2 * h, 2 * w)))
    path = model.export_onnx(tmp_path / "m.onnx", h, w, batch=v)
    with torch.inference_mode():
        want = model.module()(*inputs)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == list(spec("multiview").inputs[: len(inputs)])
    assert [o.name for o in session.get_outputs()] == list(want)
    got = session.run(None, {i.name: x.numpy() for i, x in zip(session.get_inputs(), inputs, strict=True)})
    for name, out in zip(want, got, strict=True):
        w, g = want[name], torch.from_numpy(out)
        if w.dtype == torch.bool:  # a threshold of a map: pixels at the threshold may flip
            assert (g != w).float().mean() < 1e-3, f"{ref}: {name}"
        else:
            tol = 1e-3 * max(w.float().abs().max().item(), 1.0)
            torch.testing.assert_close(g, w, rtol=1e-3, atol=tol, msg=f"{ref}: {name}")


def _main_map_name(ref: str) -> str:
    card, variant = registry.resolve(TASK[ref], ref)
    return next(k for k in spec(card.task).standard if k in variant.outputs)


@pytest.mark.model
@_needs_onnx
def test_onnx_depth_batch2_distinct_images(tmp_path):
    import onnxruntime as ort  # noqa: PLC0415 - optional

    model = _load("depth-anything-v2-s@1")  # the smallest depth graph (CPU export ~4 s; da3metric-large took ~30 s)
    (image,) = model.example_inputs(*SHAPE[2:])
    batch = torch.cat([image, image.flip(-1).roll(7, -2)])  # two different images
    path = model.export_onnx(tmp_path / "m.onnx", *SHAPE[2:], batch=2)
    with torch.inference_mode():
        want = model.module()(batch)
    (got,) = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": batch.numpy()})
    torch.testing.assert_close(torch.from_numpy(got), want, rtol=1e-3, atol=1e-3 * want.abs().max().item())
    assert not torch.allclose(want[0], want[1])  # the second image is not a copy of the first


def test_classification_export_overrides():
    from fastforks.classification._api import ClassificationModel  # noqa: PLC0415

    model: Any = object.__new__(ClassificationModel)  # a stub: only what the overrides read
    model.card = SimpleNamespace(task="classification")
    model.device = "cpu"
    io = SimpleNamespace(crop=0, resize=0, input="imagenet", mean=(), std=())
    model.variant = SimpleNamespace(io=io)
    assert model.export_size() == spec("classification").export_size == (224, 224)  # no crop / resize: the task default
    io.resize = 256
    assert model.export_size() == (256, 256)
    io.crop = 224
    assert model.export_size() == (224, 224)
    (x,) = model.example_inputs(32, 48, batch=2)
    assert x.shape == (2, 3, 32, 48)
    assert x.min() < 0  # ImageNet-normalised, not [0, 1]


def test_engine_input_names_checked_before_running():
    from fastforks._core.export import TensorRTEngine  # noqa: PLC0415

    engine = object.__new__(TensorRTEngine)
    engine.inputs = ("left", "right")
    t = torch.zeros(1)
    for args, kwargs in [((t,), {}), ((t, t, t), {}), ((t,), {"left": t}), ((), {"left": t, "top": t})]:
        with pytest.raises(InputError, match="the engine takes the inputs"):
            engine(*args, **kwargs)


def test_trt_dtype_unknown_is_a_clear_error():
    from fastforks._core.export import _torch_dtype  # noqa: PLC0415

    trt = SimpleNamespace(float32=1, float16=2, bfloat16=3, int32=4, int64=5, int8=6, uint8=7, bool=8)
    assert _torch_dtype(trt, 2) is torch.float16
    with pytest.raises(RuntimeError, match="does not support engine tensors of type 99"):
        _torch_dtype(trt, 99)


def test_export_defaults_per_task():
    assert spec("stereo").inputs == ("left", "right")
    assert all(spec(t).inputs == ("image",) for t in ("depth", "classification", "segmentation"))
    assert [spec(t).export_size for t in ("stereo", "depth", "segmentation")] == [None] * 3  # no legacy default size
    assert spec("features").export_size == (224, 224)


@pytest.mark.slow
@pytest.mark.model
def test_depth_export_rejects_kernels(tmp_path, monkeypatch):
    model = _load(DEPTH[0])
    monkeypatch.setattr(model, "backend", stereo.Backend("kernels", "cuda:0", "float16"))
    with pytest.raises(BackendUnavailableError, match="cannot run"):
        model.export_onnx(tmp_path / "m.onnx", 64, 64)
    with pytest.raises(InputError, match="both height and width"):
        model.export_onnx(tmp_path / "m.onnx", 64)


def test_every_family_is_exported():
    assert set(FAMILIES) == {card.family for task in registry.TASKS for card, _ in registry.cards(task).values()}
    assert GOLDEN


@pytest.mark.parametrize("half", ["FLOAT16", "BFLOAT16"])  # bf16: DEFOM's autocast
def test_fix_folded_weight_casts(half):
    ir = pytest.importorskip("onnx_ir")
    dt = getattr(ir.DataType, half)
    x = ir.Value(name="x", shape=ir.Shape([1, 1, 4, 4]), type=ir.TensorType(dt))
    w = ir.Value(
        name="w",
        shape=ir.Shape([1, 1, 3, 3]),
        type=ir.TensorType(ir.DataType.FLOAT),
        const_value=ir.tensor(torch.full((1, 1, 3, 3), 0.1).numpy()),
    )
    convs = [ir.Node("", "Conv", [x, w]) for _ in range(2)]  # one weight shared by two convolutions
    graph = ir.Graph([x], [c.outputs[0] for c in convs], nodes=convs, initializers=[w], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=10)
    assert _fix_folded_weight_casts(model) == 2
    assert all(c.inputs[1].dtype == dt for c in convs)
    want = torch.full((1, 1, 3, 3), 0.1).to(torch.float16 if half == "FLOAT16" else torch.bfloat16)
    assert torch.equal(torch.from_numpy(convs[0].inputs[1].const_value.numpy().astype("float32")), want.float())
    assert "w" not in graph.initializers
    assert _fix_folded_weight_casts(model) == 0


@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("half", ["FLOAT16", "BFLOAT16"])
def test_bf16_deconvolutions_as_convs(half, stride):
    """bf16 upsampling ConvTranspose -> 1x1 Conv + DepthToSpace (TensorRT builds no bf16 deconvolution), same sum."""
    ir = pytest.importorskip("onnx_ir")
    from fastforks._core.export import _bf16_deconvolutions_as_convs  # noqa: PLC0415

    dt = getattr(ir.DataType, half)
    g = torch.Generator().manual_seed(0)
    weight, bias = torch.randn(3, 5, stride, stride, generator=g), torch.randn(5, generator=g)
    x = ir.Value(name="x", shape=ir.Shape([1, 3, 4, 4]), type=ir.TensorType(dt))
    w = ir.Value(name="w", shape=ir.Shape(list(weight.shape)), type=ir.TensorType(ir.DataType.FLOAT))
    w.const_value = ir.tensor(weight.numpy())
    b = ir.Value(
        name="b", shape=ir.Shape([5]), type=ir.TensorType(dt), const_value=ir.tensor(bias.numpy().astype(dt.numpy()))
    )
    cast = ir.Node("", "Cast", [w], attributes=[ir.AttrInt64("to", dt.value)])
    cast.outputs[0].dtype = dt
    deconv = ir.Node("", "ConvTranspose", [x, cast.outputs[0], b], attributes=[ir.AttrInt64s("strides", [stride] * 2)])
    deconv.outputs[0].dtype = dt
    graph = ir.Graph([x], deconv.outputs, nodes=[cast, deconv], initializers=[w, b], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=10)
    if half == "FLOAT16":
        assert _bf16_deconvolutions_as_convs(model) == 0
        return
    assert _bf16_deconvolutions_as_convs(model) == 1
    assert [n.op_type for n in graph] == ["Conv", "DepthToSpace"]
    conv, shuffle = graph.node(0), graph.node(1)
    assert shuffle.attributes["mode"].value == "CRD"
    assert graph.outputs[0].producer() is shuffle
    assert set(graph.initializers) == {v.name for v in conv.inputs[1:]}
    cw, cb = (torch.from_numpy(v.const_value.numpy().astype("float32")) for v in conv.inputs[1:])
    xs = torch.randn(1, 3, 4, 4, generator=g)
    want = torch.nn.functional.conv_transpose2d(xs, weight.bfloat16().float(), bias.bfloat16().float(), stride=stride)
    got = torch.nn.functional.pixel_shuffle(torch.nn.functional.conv2d(xs, cw, cb), stride)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_kernels_op_fake_shape():
    with FakeTensorMode():
        x = torch.empty(2, 3, 50, 70)
        out = torch.ops.fastforks.stereo_disparity(0, x, x)
    assert out.shape == (2, 50, 70)
    assert out.dtype == torch.float32


def test_kernels_op_unknown_model():
    with pytest.raises(RuntimeError, match="no kernels model"):
        torch.ops.fastforks.stereo_disparity(-1, torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))


@pytest.mark.slow
@pytest.mark.model
def test_export_rejects_kernels_and_small_sizes(tmp_path, monkeypatch):
    model = _load("hitnet-eth3d@1")
    low = model.size_range().min_size
    with pytest.raises(
        InputError, match=rf"export size {low - 1}x64 .*outside the supported range: any H x W >= {low}"
    ):
        model.export_onnx(tmp_path / "m.onnx", low - 1, 64)
    monkeypatch.setattr(model, "backend", stereo.Backend("kernels", "cuda:0", "float16"))
    with pytest.raises(BackendUnavailableError, match="cannot run"):
        model.export_onnx(tmp_path / "m.onnx", 64, 64)


def test_float32_compute_dtype():
    card, _ = registry.resolve("stereo", "s2m2-s")
    runner = core_model.runner_class(card, "portable")
    cuda = torch.device("cuda", 0)
    assert core_model.compute_dtype(card, runner, "portable", cuda, fp32=True) == "float32"
    assert core_model.compute_dtype(card, runner, "portable", cuda) == card.portable_cuda_dtype
    assert core_model.compute_dtype(card, runner, "portable", torch.device("cpu")) == "float32"


def test_cuda_device_gets_an_index(monkeypatch):
    # A bare "cuda" never equals a tensor's device (`cuda:0`): TensorRTEngine rejected every input before this.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert _cuda_device("cuda") == torch.device("cuda", 1)
    assert _cuda_device("cuda:0") == torch.device("cuda", 0)
    assert _cuda_device(torch.device("cpu")) == torch.device("cpu")


def _first_per_task() -> list[str]:
    """The smallest-listed family of each task whose model call is benchable (one small model per task)."""
    seen: dict[str, str] = {}
    for ref in SMALLEST:
        # promptable segmentation: SAM's CPU embed resizes with PIL / NumPy (bit-exact with upstream), which dynamo's
        # guards reject on the CPU; the GPU embed is plain torch
        if (
            ref not in _CPU_RAM_HEAVY
            and ref not in _NOT_TRACEABLE | _GRAPH_BREAKS
            and TASK[ref] != "promptable_segmentation"
        ):
            seen.setdefault(TASK[ref], ref)
    return sorted(seen.values())


def _tensors(result: Any) -> dict[str, torch.Tensor]:
    return {k: v for k, v in vars(result).items() if isinstance(v, torch.Tensor)}


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.filterwarnings("ignore:Dynamo detected")  # detection's resize taps are lru_cached: traced, fine
@pytest.mark.parametrize("ref", _first_per_task())
def test_model_compile_equals_eager(ref):
    """`model.compile()` keeps the model call: same results as eager within tolerance, the runner runs compiled."""
    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    model = _load(ref)
    args, kwargs = bench_inputs(model, BenchSize(*_shape(model)[2:], 2))
    want = model(*args, **kwargs)
    runner_type = type(model._runner)
    try:
        assert model.compile(backend="aot_eager") is model
        assert type(model._runner) is not runner_type
        got = model(*args, **kwargs)
        for name, w in _tensors(want).items():
            torch.testing.assert_close(_tensors(got)[name].float(), w.float(), rtol=1e-3, atol=2e-3)
        assert model.compile(backend="eager") is model  # a repeat compile replaces the first
        assert type(model._runner).__bases__ == (runner_type,)
    finally:
        model._runner.__class__ = runner_type  # the sweep model is shared with later tests


@pytest.mark.slow
def test_compile_rejects_kernels(monkeypatch):
    model = _load("hitnet-eth3d@1")
    monkeypatch.setattr(model, "backend", stereo.Backend("kernels", "cuda:0", "float16"))
    with pytest.raises(BackendUnavailableError, match="only the portable backend"):
        model.compile()


@pytest.mark.slow
def test_failed_compiled_call_keeps_eager_torch_working(monkeypatch):
    """A compiled call that fails in AOT autograd must not break torch's dispatcher for the rest of the process.

    The error's traceback holds AOT autograd frames with python dispatcher guards; freed after the call (pytest's
    excinfo, a notebook's last exception), they used to set the PythonDispatcher key again: every later op of the
    process raised "PythonDispatcherTLS was not set" (one failed multiview compile failed ~1200 later tests).
    """
    from torch._dynamo.exc import BackendCompilerFailed  # noqa: PLC0415

    model = _load("hitnet-eth3d@1")
    images = pair(*size(model.size_range().min_size, 64, 96))
    want = model(*images).disparity
    leaf = torch.nn.Parameter(torch.zeros(1, 1, 4))

    def fails_in_aot(self: Any, *_a: Any, **_k: Any) -> Any:
        with torch.inference_mode():  # a view of a parameter in inference mode: aot_eager raises
            return leaf.expand(2, -1, -1)

    runner_type = type(model._runner)
    monkeypatch.setattr(runner_type, "__call__", fails_in_aot)
    try:
        model.compile(backend="aot_eager")
        with pytest.raises(BackendCompilerFailed, match="inference tensor") as err:
            model(*images)
    finally:
        model._runner.__class__ = runner_type  # the sweep model is shared with later tests
        monkeypatch.undo()
    del err  # the traceback (and the frames it holds) is freed outside the call
    gc.collect()
    assert not torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.PythonDispatcher)
    torch.nn.Embedding(8, 4)  # the first eager op that failed after the multiview compile
    torch.testing.assert_close(model(*images).disparity, want)


def _tiny_vit_stub(side: int) -> Any:
    """A Model stand-in around a random timm ViT with a strict `side` input (like siglip / eva02 at 378 / 448 px)."""
    timm = pytest.importorskip("timm")

    from fastforks._core.sizes import SizeRange  # noqa: PLC0415

    torch.manual_seed(0)
    net = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, img_size=side, embed_dim=32, depth=1, num_heads=2, num_classes=5
    ).eval()
    model: Any = SimpleNamespace(
        backend=SimpleNamespace(mode="portable"),
        variant=SimpleNamespace(
            ref="tiny-vit", outputs=("logits",), io=SimpleNamespace(crop=side, resize=side, input="imagenet")
        ),
        card=SimpleNamespace(task="classification"),
        weights=None,
        device="cpu",
        module=lambda: net,
        size_range=lambda: SizeRange(min_size=1, multiple=1, native=(side, side)),
    )
    model.example_inputs = lambda h, w, *, batch=1: (torch.randn(batch, 3, h, w),)
    return model


@pytest.mark.skipif(
    not (importlib.util.find_spec("onnxscript") and importlib.util.find_spec("onnxruntime")),
    reason="needs onnxscript + onnxruntime (dependency group export)",
)
def test_fixed_size_vit_export(tmp_path):
    """A timm ViT with a strict input size exports at its crop (ONNX Runtime == module); another size says why not."""
    import onnxruntime as ort  # noqa: PLC0415

    from fastforks._core.export import export_onnx  # noqa: PLC0415

    model = _tiny_vit_stub(64)
    with pytest.raises(InputError, match=r"takes exactly 64x64.*not 32x32"):
        export_onnx(model, tmp_path / "bad.onnx", 32, 32)
    path = export_onnx(model, tmp_path / "ok.onnx", 64, 64)
    (x,) = model.example_inputs(64, 64)
    with torch.inference_mode():
        want = model.module()(x)
    (got,) = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    torch.testing.assert_close(torch.from_numpy(got), want, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("nan", [False, True])
def test_tensorrt_engine_all_nan_is_removed(tmp_path, monkeypatch, nan):
    """An engine whose output is NaN everywhere (a TensorRT fusion bug, MVSAnywhere) is removed with a clear error."""
    from fastforks._core import export  # noqa: PLC0415

    class Engine:
        def __init__(self, _path, device):
            assert device == torch.device("cpu")

        def run(self, x):
            return {"depth": torch.full_like(x, torch.nan if nan else 1.0), "mask": torch.ones(1, dtype=torch.bool)}

    monkeypatch.setattr(export, "TensorRTEngine", Engine)
    path = tmp_path / "m.engine"
    path.write_bytes(b"plan")
    model = SimpleNamespace(
        example_inputs=lambda h, w, *, batch: (torch.zeros(batch, 3, h, w),), variant=SimpleNamespace(ref="m@1")
    )
    if not nan:
        export._check_engine(model, path, 8, 8, 1)
        assert path.exists()
        return
    with pytest.raises(BackendUnavailableError, match=r"m@1: the TensorRT engine outputs NaN everywhere \('depth'\)"):
        export._check_engine(model, path, 8, 8, 1)
    assert not path.exists()


@pytest.mark.skipif(importlib.util.find_spec("tensorrt") is None, reason="needs tensorrt")
def test_tensorrt_logger_keeps_errors(capsys):
    """The shared logger prints warnings and errors (not info) and keeps the errors for a build failure's message."""
    import tensorrt as trt  # noqa: PLC0415

    from fastforks._core.export import _trt_logger  # noqa: PLC0415

    logger = _trt_logger()
    assert logger is _trt_logger()
    logger.errors.clear()
    sev = trt.ILogger.Severity
    for severity, text in [(sev.INFO, "info"), (sev.WARNING, "warn"), (sev.ERROR, "err"), (sev.INTERNAL_ERROR, "ie")]:
        logger.log(severity, text)
    assert logger.errors == ["err", "ie"]
    assert capsys.readouterr().err.split() == ["[TRT]", "[W]", "warn", "[TRT]", "[E]", "err", "[TRT]", "[F]", "ie"]
    logger.errors.clear()


@pytest.mark.gpu
def test_tensorrt_engine_concurrent_threads(tmp_path):
    """One engine, calls from 4 threads (each on its own stream, distinct inputs) = the sequential results.

    The engine has one execution context: unlocked, the threads overwrote each other's tensor addresses."""
    import threading  # noqa: PLC0415

    if not torch.cuda.is_available() or importlib.util.find_spec("tensorrt") is None:
        pytest.skip("needs CUDA and tensorrt")
    from fastforks._core.export import TensorRTEngine, build_tensorrt  # noqa: PLC0415

    torch.manual_seed(0)
    net = torch.nn.Sequential(*(m for _ in range(6) for m in (torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.ReLU())))
    x = torch.randn(1, 16, 256, 256)
    program = torch.onnx.export(
        net.eval(), (x,), input_names=["image"], output_names=["out"], dynamo=True, verbose=False
    )
    assert program is not None
    program.save(str(tmp_path / "m.onnx"))
    engine = TensorRTEngine(build_tensorrt(tmp_path / "m.onnx", tmp_path / "m.engine"))
    inputs = [torch.randn(1, 16, 256, 256, device=engine.device) * (i + 1) for i in range(4)]
    want = [engine(t).clone() for t in inputs]
    torch.cuda.synchronize()
    got: dict[int, list[torch.Tensor]] = {i: [] for i in range(4)}
    start = threading.Barrier(4)

    def work(i: int) -> None:
        stream = torch.cuda.Stream(engine.device)
        start.wait()
        with torch.cuda.stream(stream):
            for _ in range(20):
                got[i].append(engine(inputs[i]))
        stream.synchronize()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    for i in range(4):
        assert len(got[i]) == 20
        for out in got[i]:
            torch.testing.assert_close(out, want[i], rtol=0, atol=0)


@pytest.mark.parametrize("s", [2, 4])
@pytest.mark.parametrize("autocast", [False, True])
def test_defom_exported_deconv_is_a_matmul(s, autocast):
    """DEFOM's kernel == stride ConvTranspose2d (TensorRT cannot build it in bf16) exports as a matmul, same output."""
    from fastforks.stereo.models.defom_stereo.net.dpt import _resize  # noqa: PLC0415

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = torch.nn.ConvTranspose2d(6, 5, kernel_size=s, stride=s)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cpu", torch.bfloat16, enabled=autocast):
                return _resize(self.up, x)

    m = M().eval()
    x = torch.randn(2, 6, 7, 9, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = m(x)
        ep = torch.export.export(m, (x,))
        got = ep.module()(x)
    graphs = [g.graph for g in ep.graph_module.modules() if isinstance(g, torch.fx.GraphModule)]  # + autocast bodies
    ops = {str(n.target) for g in graphs for n in g.nodes if n.op == "call_function"}
    assert not any("conv" in o for o in ops), ops
    assert got.dtype == want.dtype
    tol = {"rtol": 2e-2, "atol": 2e-2} if autocast else {"rtol": 1e-5, "atol": 1e-5}
    torch.testing.assert_close(got.float(), want.float(), **tol)


@pytest.mark.parametrize(
    "module",
    [
        "multiview.models.vggt.net.heads.utils",
        "multiview.models.vggt_omega.net.models.heads.utils",
        "depth.models.depth_anything_3.net.utils.head_utils",
    ],
)
def test_sincos_einsum_has_one_input_dtype(module):
    """The VGGT / Omega / DA3 position embedding (fp16 positions, fp32 frequencies) exports a one-dtype einsum (TRT)."""
    import importlib  # noqa: PLC0415

    utils = importlib.import_module(f"fastforks.{module}")

    class M(torch.nn.Module):
        def forward(self, pos: torch.Tensor) -> torch.Tensor:
            return utils.make_sincos_pos_embed(16, pos)

    pos = torch.linspace(-1, 1, 12).half()
    ep = torch.export.export(M(), (pos,), strict=False)
    eins = [n for n in ep.graph.nodes if "einsum" in str(n.target)]
    assert eins
    for n in eins:
        assert len({a.meta["val"].dtype for a in n.args[1]}) == 1
    torch.testing.assert_close(ep.module()(pos), M()(pos), rtol=1e-6, atol=1e-6)


def test_failed_tensorrt_build_is_one_clear_error(monkeypatch, tmp_path):
    """A failing engine build raises BackendUnavailableError naming TensorRT's first error and a way out."""
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks._core import export  # noqa: PLC0415
    from fastforks._core.errors import BackendUnavailableError  # noqa: PLC0415

    logger = SimpleNamespace(errors=[])

    def build(*_):
        logger.errors.append("Inconsistent strides In compileGraph\nmore detail")

    network = SimpleNamespace(num_inputs=0)
    builder = SimpleNamespace(
        create_network=lambda _: network,
        create_builder_config=lambda: SimpleNamespace(set_memory_pool_limit=lambda *_: None),
        build_serialized_network=build,
    )
    trt = SimpleNamespace(
        Builder=lambda _: builder,
        OnnxParser=lambda *_: SimpleNamespace(parse_from_file=lambda _: True),
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
    )
    monkeypatch.setattr(export, "_tensorrt", lambda: trt)
    monkeypatch.setattr(export, "_trt_logger", lambda: logger)
    with pytest.raises(
        BackendUnavailableError, match=r"build failed: Inconsistent strides In compileGraph\. Try float32"
    ):
        export.build_tensorrt(tmp_path / "m.onnx", tmp_path / "m.engine")
