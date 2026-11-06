"""Review R2 pass 2 fixes (CPU, no weights): dynamic export checks, RT-DETR sizes and query parity, runnable, pickle."""

from __future__ import annotations

import pickle
import types
from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import BackendUnavailableError, InputError
from fastforks._core import cli, export, registry
from fastforks._core.model import Model
from fastforks._core.sizes import size_range
from fastforks.detection._rtdetr_runner import _quantise

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("model_id", ["moge-2-vits-normal", "las2-s", "rtdetrv2-s-coco"])
def test_dynamic_export_refused_up_front(model_id: str) -> None:
    """R2-16 / R2-17: families whose traced graph fixes the size or fails inside the range refuse a size range."""
    card, v = registry.resolve_any(model_id)
    assert "dynamic export is not supported" in export.fixed_size_reason(card, size_range(v, card))


def test_probe_sizes() -> None:
    assert export.probe_sizes(((64, 96), (128, 192))) == [(64, 96), (128, 192), (64, 192), (128, 96)]
    assert export.probe_sizes(((64, 64), (64, 64))) == [(64, 64)]


def _onnx_file(path: Path, dims: list[int | str], *, reshape: bool = False) -> Path:
    """An `image -> out` graph: Identity, or a Reshape to a fixed shape (fails at other sizes)."""
    onnx = pytest.importorskip("onnx")
    h = onnx.helper
    image = h.make_tensor_value_info("image", onnx.TensorProto.FLOAT, dims)
    out = h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, None)
    if reshape:
        shape = h.make_tensor("shape", onnx.TensorProto.INT64, [4], [1, 3, 32, 32])
        graph = h.make_graph([h.make_node("Reshape", ["image", "shape"], ["out"])], "g", [image], [out], [shape])
    else:
        graph = h.make_graph([h.make_node("Identity", ["image"], ["out"])], "g", [image], [out])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10), str(path))
    return path


def _stub_model() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        variant=types.SimpleNamespace(ref="stub"),
        device=torch.device("cpu"),
        example_inputs=lambda h, w, batch=1: (torch.rand(batch, 3, h, w),),
        module=torch.nn.Identity,  # the eager reference of the Identity graphs
    )


@pytest.mark.parametrize("run", [True, False])
def test_dynamic_onnx_fixed_graph_is_refused_and_removed(tmp_path: Path, *, run: bool) -> None:
    """R2-17: a graph whose image input has fixed H / W is not dynamic (also checked for the TensorRT path)."""
    path = _onnx_file(tmp_path / "m.onnx", [1, 3, 32, 32])
    with pytest.raises(InputError, match="not dynamic"):
        export._check_dynamic_onnx(_stub_model(), path, ((32, 32), (64, 64)), 1, run=run)
    assert not path.exists()


def test_dynamic_onnx_run_at_the_corners(tmp_path: Path) -> None:
    """R2-16: a dynamic graph runs at the corners of its range; one that fails there is removed."""
    pytest.importorskip("onnxruntime")
    good = _onnx_file(tmp_path / "good.onnx", ["b", 3, "h", "w"])
    export._check_dynamic_onnx(_stub_model(), good, ((32, 32), (64, 96)), 1, run=True)
    assert good.exists()
    bad = _onnx_file(tmp_path / "bad.onnx", ["b", 3, "h", "w"], reshape=True)
    export._check_dynamic_onnx(_stub_model(), bad, ((32, 32), (64, 96)), 1, run=False)
    assert bad.exists()
    with pytest.raises(InputError, match="fails at 64x96"):
        export._check_dynamic_onnx(_stub_model(), bad, ((32, 32), (64, 96)), 1, run=True)
    assert not bad.exists()


def test_export_tensorrt_needs_cuda_before_the_onnx_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2-27: no CUDA device -> `BackendUnavailableError` before any ONNX work."""
    monkeypatch.setattr(export, "check_exportable", lambda *_: None)
    monkeypatch.setattr(export, "_tensorrt", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def no_export(*_: object, **__: object) -> None:
        pytest.fail("the ONNX export ran")

    monkeypatch.setattr(export, "_export_onnx", no_export)
    with pytest.raises(BackendUnavailableError, match="needs a CUDA GPU"):
        export.export_tensorrt(_stub_model(), "x.engine", 64, 64)


class _Context:
    """A TensorRT execution context whose shape graph rejects the size (as at 100x150 in a /32 engine)."""

    def __init__(self, *, error: bool) -> None:
        self.error = error

    def set_input_shape(self, *_: object) -> None:
        pass

    def get_tensor_shape(self, _: str) -> tuple[int, ...]:
        if self.error:
            msg = "Could not get shape for tensor 'disparity'"
            raise RuntimeError(msg)
        return (1, -1, -1)


@pytest.mark.parametrize("error", [True, False])
def test_dynamic_engine_off_multiple_raises_input_error(monkeypatch: pytest.MonkeyPatch, *, error: bool) -> None:
    """R2-20: a dynamic engine at a size its shape graph rejects raises `InputError` naming the multiple."""
    monkeypatch.setattr(torch.cuda, "device", lambda _: __import__("contextlib").nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: types.SimpleNamespace(cuda_stream=0))
    engine = object.__new__(export.TensorRTEngine)
    engine.__dict__.update(
        _lock=__import__("threading").Lock(),
        device=torch.device("cpu"),
        _stream=None,
        shapes={"left": (1, 3, -1, -1), "disparity": (1, -1, -1)},
        outputs=("disparity",),
        _context=_Context(error=error),
    )
    with pytest.raises(InputError, match="multiples of the model's padding multiple"):
        engine._execute({"left": torch.zeros(1, 3, 100, 150)})


def test_rtdetr_module_takes_multiples_of_32() -> None:
    """R2-18: RT-DETRv2 `module()` (and its export) refuses sizes it cannot run, with the multiple."""
    x = torch.rand(1, 3, 64, 96)
    assert torch.equal(_quantise(x, "RT-DETRv2"), x.mul(255).round().div(255))
    with pytest.raises(InputError, match="multiples of 32"):
        _quantise(torch.rand(1, 3, 136, 136), "RT-DETRv2")


def _queries(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """RT-DETR-like outputs: logits whose sigmoid max is `scores`, distinct boxes."""
    q = scores.numel()
    logits = torch.full((1, q, 3), -20.0)
    logits[0, :, 1] = torch.logit(scores)
    boxes = torch.stack([torch.linspace(0.1, 0.9, q)] * 4, -1)[None]
    return {"logits": logits, "boxes": boxes}


def test_parity_matches_queries_by_box() -> None:
    """RT-DETR TRT diagnosis: permuted / near-tied queries are matched by box, not compared per index."""
    scores = lambda lg: lg.sigmoid().amax(-1)  # noqa: E731
    want = _queries(torch.tensor([0.9, 0.5, 0.1, 0.05]))
    perm = torch.tensor([3, 1, 0, 2])
    got = {n: t[:, perm] for n, t in want.items()}
    line = cli._parity(want, got, stereo=False, scores=scores)
    assert line.startswith("2 / 2 queries score > 0.3"), line
    assert "max |d boxes| 0," in line
    assert "max |d score| 0" in line
    assert "max 0.8" in cli._parity(want, got, stereo=False)
    none = _queries(torch.tensor([0.1, 0.05]))
    assert cli._parity(none, none, stereo=False, scores=scores).startswith("no query scores > 0.3")


def test_detection_modules_declare_query_scores() -> None:
    """The CLI finds the scorer on `module()` (DETR: softmax without no-object; RT-DETR: sigmoid)."""
    from fastforks._core.runner import Forward  # noqa: PLC0415

    m = Forward(torch.nn.Identity(), lambda net, x: net(x), query_scores=lambda lg: lg.amax(-1))
    assert getattr(m, "query_scores", None) is not None
    assert getattr(Forward(torch.nn.Identity(), lambda _, x: x), "query_scores", 0) is None


def test_runnable_counts_local_noncommercial_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R2-22: a variant with blocked non-commercial weights is runnable once its files are local."""
    for env in (registry.NONCOMMERCIAL_ENV, registry.WEIGHTS_PATH_ENV):
        monkeypatch.delenv(env, raising=False)
    for env in ("FASTFORKS_CACHE", "HF_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.setenv(env, str(tmp_path / env))
    found = next(
        (card, v)
        for task in registry.TASKS
        for card, v in registry.cards(task).values()
        if registry.nc_weights_blocked(card, v)
        and not registry.missing(card)
        and v.weights
        and all(w.size and w.member is None for w in v.weights)
    )
    card, v = found
    assert registry.needs_nc_download(card, v)
    assert v.id not in registry.list_models([card.task], runnable=True)
    root = tmp_path / "weights"
    (root / card.family).mkdir(parents=True)
    for w in v.weights:
        with (root / card.family / w.name).open("wb") as f:
            f.truncate(w.size)  # sparse: size only (runnable does not hash)
    monkeypatch.setenv(registry.WEIGHTS_PATH_ENV, str(root))
    assert not registry.needs_nc_download(card, v)
    assert v.id in registry.list_models([card.task], runnable=True)


def test_model_pickle_says_load_per_worker() -> None:
    """R2-26: pickling a model (spawned workers) raises a `TypeError` that says what to do."""
    m = object.__new__(Model)
    m.__dict__.update(card=types.SimpleNamespace(task="classification"), variant=types.SimpleNamespace(id="resnet18"))
    with pytest.raises(
        TypeError, match=r"load the model in each worker: fastforks\.classification\.load\('resnet18'\)"
    ):
        pickle.dumps(m)
