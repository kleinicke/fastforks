"""CPU checks of the input layouts and request boundary for measured Linux CUDA graphs."""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from fastforks._core import linux_graph_defaults
from fastforks._core import model as model_api
from fastforks.interpolation import _api as interpolation_api
from fastforks.shared import cuda_graphs
from fastforks.shared.cuda_graphs import GraphCache, SmallShapeGraphs, graph_replay_allowed, graph_request

if TYPE_CHECKING:
    from fastforks._core.model import Model


def test_public_layouts_distinguish_rgb_views() -> None:
    chw = torch.empty((3, 16, 32), dtype=torch.uint8)
    interleaved = torch.empty((16, 32, 3), dtype=torch.uint8).permute(2, 0, 1)
    offset = torch.empty((17, 32, 3), dtype=torch.uint8)[1:].permute(2, 0, 1)
    bchw = torch.empty((1, 3, 16, 32), dtype=torch.float32)
    assert linux_graph_defaults._image_layout(interleaved) == "u8-chw-rgb-interleaved"
    assert linux_graph_defaults._image_layout(bchw) == "f32-bchw-contiguous"
    assert linux_graph_defaults._image_layout(chw) is None
    assert linux_graph_defaults._image_layout(offset) is None
    assert linux_graph_defaults._image_layout(bchw.expand(2, -1, -1, -1)) is None


def test_unmeasured_linux_input_stays_eager_and_windows_keeps_its_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = cast(
        "Model",
        SimpleNamespace(
            _device=torch.device("cuda:0"),
            backend=SimpleNamespace(mode="kernels", dtype="float16"),
            card=SimpleNamespace(task="pose", family="yunet"),
            variant=SimpleNamespace(ref="yunet-n@1"),
            weights=None,
            options={"cuda_graphs": True},
        ),
    )
    monkeypatch.setattr(linux_graph_defaults.sys, "platform", "linux")
    assert not linux_graph_defaults.allows(model)
    monkeypatch.setattr(linux_graph_defaults.sys, "platform", "win32")
    assert linux_graph_defaults.allows(model)


def test_nested_request_cannot_reenable_denied_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_graphs.sys, "platform", "linux")
    cache = GraphCache(torch.neg)
    assert not graph_replay_allowed()
    with graph_request():
        assert not graph_replay_allowed()
    with graph_request(allow_replay=False), graph_request(allow_replay=True):
        assert not graph_replay_allowed()
        assert cache.defer_first_request("shape")
        assert cache.run("shape", torch.ones(1)) is None
        assert cache.static_inputs("shape") is None
    assert not graph_replay_allowed()
    assert not cache._seen

    with graph_request(allow_replay=True):
        assert graph_replay_allowed()
        with graph_request():
            assert graph_replay_allowed()
        with graph_request(allow_replay=False):
            assert not graph_replay_allowed()
        assert graph_replay_allowed()
    assert not graph_replay_allowed()


class _CudaInput(torch.Tensor):
    @property
    def device(self) -> torch.device:  # ty: ignore[invalid-property-type-override]
        return torch.device("cuda:0")

    @property
    def is_cuda(self) -> bool:  # ty: ignore[invalid-property-type-override]
        return True


def test_interpolation_admits_only_the_measured_two_frame_storage_layout() -> None:
    device = torch.device("cuda:0")
    pair = linux_graph_defaults.ImagePair
    frames = torch.empty((2, 16, 32, 3)).permute(0, 3, 1, 2).as_subclass(_CudaInput)
    first, second = frames[:1], frames[1:]
    signature = ("f32-bchw-adjacent-rgb-pair", (1, 3, 16, 32))
    model = cast(
        "Model",
        SimpleNamespace(
            variant=SimpleNamespace(ref="rife-4.26@1"),
            backend=SimpleNamespace(mode="kernels", dtype="float16"),
            card=SimpleNamespace(task="interpolation"),
        ),
    )
    assert linux_graph_defaults._workload_key(model, pair(first, second), device) == (
        "rife-4.26@1",
        "kernels",
        "float16",
        "interpolation-midpoint-pair",
        (signature, signature),
    )
    extra = torch.empty((3, 16, 32, 3)).permute(0, 3, 1, 2).as_subclass(_CudaInput)
    for operands in (pair(second, first), pair(first, frames.clone()[1:]), pair(extra[:1], extra[1:2])):
        assert linux_graph_defaults._workload_key(model, operands, device) is None


def test_top_down_pose_graph_key_includes_raw_box_count_and_layout() -> None:
    device = torch.device("cuda:0")
    image = torch.empty((3, 16, 32)).as_subclass(_CudaInput)
    boxes = torch.empty((2, 4)).as_subclass(_CudaInput)
    model = cast(
        "Model",
        SimpleNamespace(
            variant=SimpleNamespace(ref="rtmpose-t@1"),
            backend=SimpleNamespace(mode="kernels", dtype="float16"),
            card=SimpleNamespace(task="pose"),
        ),
    )
    pose_boxes = linux_graph_defaults.PoseBoxes
    key = linux_graph_defaults._workload_key(model, pose_boxes(image, boxes), device)
    assert key == (
        "rtmpose-t@1",
        "kernels",
        "float16",
        "pose-boxes",
        (("f32-chw-contiguous", (3, 16, 32)), ("f32-p4-cuda-contiguous", (2, 4))),
    )
    assert linux_graph_defaults._workload_key(model, pose_boxes(image, boxes[:1]), device) != key
    assert linux_graph_defaults._workload_key(model, pose_boxes(image, boxes[1:]), device) is None
    assert linux_graph_defaults._workload_key(model, pose_boxes(image, boxes.half()), device) is None
    assert linux_graph_defaults._workload_key(model, pose_boxes(image.unsqueeze(0), boxes), device) is None


def _profile_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Model, torch.Tensor]:
    mock = SimpleNamespace(
        _device=torch.device("cuda:0"),
        _runner=SimpleNamespace(deterministic_cudnn=False, sdpa_auto=False),
        _lock=threading.RLock(),
        backend=SimpleNamespace(mode="portable", dtype="float16", device="cuda:0", kernels=(), weights=("weight",)),
        card=SimpleNamespace(task="ocr"),
        variant=SimpleNamespace(ref="parseq-test@1"),
        weights=None,
        options={},
    )
    monkeypatch.setattr(cuda_graphs.sys, "platform", "linux")
    monkeypatch.setattr(linux_graph_defaults.os, "getenv", lambda _key, default=None: default)
    monkeypatch.setattr(linux_graph_defaults, "_gpu_name", lambda _dev: "test GPU")
    monkeypatch.setattr(linux_graph_defaults, "_gpu_memory", lambda _dev: 16_616_521_728)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "device", lambda _dev: contextlib.nullcontext())
    monkeypatch.setattr(model_api.warmup, "wait", lambda _dev: None)
    monkeypatch.setattr(model_api.sdpa, "policy", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(model_api, "begin_call", lambda: None)
    monkeypatch.setattr(
        linux_graph_defaults,
        "APPROVED",
        {
            ("parseq-test@1", "portable", "float16", "f32-bchw-contiguous", (16, 24)): (
                (
                    str(torch.__version__),
                    "test GPU",
                    "native",
                    None,
                    None,
                    None,
                    None,
                    torch.get_num_threads(),
                    (),
                    ("weight",),
                ),
            ),
        },
    )
    return cast("Model", mock), torch.ones((1, 3, 16, 24)).as_subclass(_CudaInput)


def test_measured_model_session_is_mode_specific_and_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    model, image = _profile_model(monkeypatch)
    assert not graph_replay_allowed()
    with model_api.Model._session(model, graph_input=image):
        assert graph_replay_allowed()
        with model_api.Model._session(model):
            assert not graph_replay_allowed()
        assert graph_replay_allowed()
    assert not graph_replay_allowed()
    with (
        model_api.Model._without_graph_admission(model),
        model_api.Model._session(model, graph_input=image),
    ):
        assert not graph_replay_allowed()
    with monkeypatch.context() as different:
        different.setattr(model.backend, "mode", "kernels")
        with model_api.Model._session(model, graph_input=image):
            assert not graph_replay_allowed()
    with monkeypatch.context() as compiled:
        compiled.setattr(model._runner, "_ff_eager", object(), raising=False)
        with model_api.Model._session(model, graph_input=image):
            assert not graph_replay_allowed()
    assert linux_graph_defaults.allows(model, image)


def test_options_cannot_force_unmeasured_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    model, image = _profile_model(monkeypatch)
    for options, measured_eligible in [
        ({"cuda_graphs": True}, True),
        ({"cuda_graphs": False}, False),
        ({"cuda_graphs": True, "iters": 2}, False),
        ({"cuda_graphs_encoder": True}, False),
    ]:
        monkeypatch.setattr(model, "options", options)
        assert linux_graph_defaults.allows(model, image) is measured_eligible
        assert not linux_graph_defaults.allows(model, image[..., :16])
        assert not linux_graph_defaults.allows(model, image.clone(memory_format=torch.channels_last))
    monkeypatch.setattr(model, "options", {"cuda_graphs": True})
    monkeypatch.setattr(model, "weights", ("custom",))
    assert not linux_graph_defaults.allows(model, image)


def test_measured_graph_uses_the_standard_cuda_backend_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    model, image = _profile_model(monkeypatch)
    assert linux_graph_defaults.allows(model, image)
    for obj, attr, changed in (
        (torch, "are_deterministic_algorithms_enabled", lambda: True),
        (torch.backends.cudnn, "benchmark", True),
        (torch.backends.cudnn, "deterministic", True),
        (torch.backends.cuda, "flash_sdp_enabled", lambda: False),
        (torch.backends.cuda, "matmul", SimpleNamespace(allow_fp16_accumulation=True)),
        (torch, "get_float32_matmul_precision", lambda: "high"),
    ):
        with monkeypatch.context() as override:
            override.setattr(obj, attr, changed)
            assert not linux_graph_defaults.allows(model, image)
    with monkeypatch.context() as override:
        override.setattr(torch.backends.cudnn, "deterministic", True)
        override.setattr(model._runner, "deterministic_cudnn", True)
        assert linux_graph_defaults.allows(model, image)

    def custom_attention(key: str, default: str | None = None) -> str | None:
        return "math" if key == "FASTFORKS_SDPA" else default

    with monkeypatch.context() as override:
        override.setattr(linux_graph_defaults.os, "getenv", custom_attention)
        assert not linux_graph_defaults.allows(model, image)
    assert linux_graph_defaults.allows(model, image)


def test_lower_vram_board_with_matching_name_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    model, image = _profile_model(monkeypatch)
    assert linux_graph_defaults.allows(model, image)
    monkeypatch.setattr(linux_graph_defaults, "_gpu_memory", lambda _dev: 8 << 30)
    assert not linux_graph_defaults.allows(model, image)


def test_windows_no_scope_keeps_legacy_and_explicit_deny_is_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_graphs.sys, "platform", "win32")
    assert graph_replay_allowed()
    with graph_request(), graph_request(allow_replay=True):
        assert graph_replay_allowed()
        with graph_request(allow_replay=False), graph_request(allow_replay=True):
            assert not graph_replay_allowed()
    assert graph_replay_allowed()


def test_standalone_unmeasured_cache_touches_no_cuda_or_pending_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_graphs.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a: pytest.fail("denied cache touched the driver"))
    cache = GraphCache(torch.neg)
    for _ in range(3):
        assert cache.run("shape", torch.ones(1)) is None
    assert cache.defer_first_request("shape")
    assert cache.static_inputs("shape") is None
    assert not cache._seen
    assert not cache.graphs
    assert cache._calls == 0


def test_denied_larger_input_can_reclaim_previous_graph_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_graphs.sys, "platform", "linux")
    changes: list[tuple[torch.device, bool]] = []
    monkeypatch.setattr(
        cuda_graphs, "empty_graph_cache", lambda device, *, released: changes.append((device, released))
    )
    device = torch.device("cuda:0")
    small = torch.ones((1, 3, 8, 8)).as_subclass(_CudaInput)
    large = torch.ones((1, 3, 24, 24)).as_subclass(_CudaInput)
    graph = cast("torch.cuda.CUDAGraph", object())
    cache = GraphCache(torch.neg)
    cache.graphs["small"] = graph, (small,), small
    cache._pool_device = device
    assert cache.run("small", small) is None
    assert "small" in cache.graphs
    assert cache.static_inputs("small") is None
    assert cache.run("large", large) is None
    assert not cache.graphs
    assert not cache._seen
    assert cache._calls == 0
    assert changes == [(device, True)]

    wrapper = SmallShapeGraphs(torch.neg, pixels=8**2, one_shape=True)
    assert wrapper.cache is not None
    wrapper.cache.graphs["small"] = graph, (small,), small
    wrapper.cache._pool_device = device
    wrapper._last_key = "small"
    torch.testing.assert_close(wrapper(large), -large)
    assert not wrapper.cache.graphs
    assert wrapper._last_key is None
    assert changes == [(device, True), (device, True)]

    wrapper = SmallShapeGraphs(torch.neg, pixels=32**2)
    assert wrapper.cache is not None
    wrapper.cache.graphs["small"] = graph, (small,), small
    wrapper.cache._pool_device = device
    torch.testing.assert_close(wrapper(large), -large)
    assert not wrapper.cache.graphs
    assert changes == [(device, True)] * 3


def test_unmeasured_sam3_concepts_skip_graph_only_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastforks.promptable_segmentation.models.sam3 import model as sam3  # noqa: PLC0415

    monkeypatch.setattr(cuda_graphs.sys, "platform", "linux")
    result = {"scores": torch.ones(1)}
    monkeypatch.setattr(sam3._Sam3, "segment_concept", lambda *_: result)
    runner = sam3.Kernels.__new__(sam3.Kernels)
    runner.ctx = SimpleNamespace(params=SimpleNamespace(cuda_graphs=True))  # type: ignore[assignment]
    runner.lock = threading.Lock()
    runner.fp16_accumulate = contextlib.nullcontext  # type: ignore[method-assign]
    levels = {f"fpn{i}": torch.ones(1) for i in range(3)}
    assert runner.segment_concept(levels, "coins") is result


def test_unmeasured_dinov2_crop_skips_graph_position_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastforks.segmentation.models.dinov2_linear import model as dino  # noqa: PLC0415

    monkeypatch.setattr(cuda_graphs.sys, "platform", "linux")
    result = torch.ones(1)
    monkeypatch.setattr(dino._Net, "_patch_logits", lambda *_: result)
    net = dino._SmallGraphNet.__new__(dino._SmallGraphNet)
    torch.nn.Module.__init__(net)
    net._graph_position = None
    image = torch.empty(dino._GRAPH_CROP, dtype=torch.float16).as_subclass(_CudaInput)
    with torch.inference_mode():
        assert net._patch_logits(image) is result
    assert net._graph_position is None


def test_interpolation_midpoint_uses_original_ordered_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    mock, first = _profile_model(monkeypatch)
    model = interpolation_api.InterpolationModel.__new__(interpolation_api.InterpolationModel)
    model.__dict__.update(vars(mock))
    model.device = first.device
    model.card.task = "interpolation"
    model.variant.ref = "rife-test@1"
    monkeypatch.setattr(model._runner, "batch", 1, raising=False)
    monkeypatch.setattr(model._runner, "batch_pixels", None, raising=False)
    monkeypatch.setattr(model, "_check_size", lambda _x: None)
    converted = torch.ones(first.shape)
    monkeypatch.setattr(interpolation_api, "to_rgb_pair", lambda *_: (converted, converted))

    def run(x: torch.Tensor, _y: torch.Tensor, *, timestep: float) -> dict[str, torch.Tensor]:
        seen.append((timestep, graph_replay_allowed()))
        return {"frame": x}

    seen: list[tuple[float, bool]] = []
    monkeypatch.setattr(model, "_run", run)
    _, _, h, w = first.shape
    second = torch.zeros((h, w, 3), dtype=torch.uint8).permute(2, 0, 1).as_subclass(_CudaInput)
    pair = linux_graph_defaults.ImagePair(first, second)
    key = (
        model.variant.ref,
        "portable",
        "float16",
        "interpolation-midpoint-pair",
        (("f32-bchw-contiguous", (1, 3, h, w)), ("u8-chw-rgb-interleaved", (3, h, w))),
    )
    assert linux_graph_defaults._workload_key(model, pair, model.device) == key
    assert linux_graph_defaults._workload_key(model, first, model.device) is None

    model(first, second)
    assert seen.pop() == (0.5, False)
    monkeypatch.setitem(linux_graph_defaults.APPROVED, key, next(iter(linux_graph_defaults.APPROVED.values())))
    model(first, second)
    model(first, second, timestep=0.5)
    model(first, second, timestep=0.25)
    model(second, first)
    model(first, second[:, :, : w - 1])
    model(first, torch.empty((3, h, w), dtype=torch.uint8).as_subclass(_CudaInput))
    model(first.as_subclass(torch.Tensor), second)
    model([first], [second])
    batch = first.expand(2, -1, -1, -1)
    model(batch, batch)
    offset = torch.empty((2, 3, h, w))[1:].as_subclass(_CudaInput)
    model(offset, second)
    assert seen[:3] == [(0.5, True), (0.5, True), (0.25, False)]
    assert seen[3:] == [(0.5, False)] * 7
    with pytest.raises(model_api.InputError, match="timestep"):
        model(first, second, timestep=float("nan"))
    assert not graph_replay_allowed()


def test_windows_interpolation_retains_legacy_graph_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    mock, _ = _profile_model(monkeypatch)
    monkeypatch.setattr(cuda_graphs.sys, "platform", "win32")
    mock.card.task = "interpolation"
    with model_api.Model._session(mock):
        assert graph_replay_allowed()
