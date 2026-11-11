"""RF-DETR person filtering, uncertainty fusion and empty-output contracts without model weights."""

from types import SimpleNamespace

import pytest
import torch

from fastforks.pose.models.rfdetr_keypoint.model import Portable
from fastforks.shared.rfdetr.net.models.postprocess import PostProcess


def _runner(monkeypatch, threshold):
    runner = object.__new__(Portable)
    runner.ctx = SimpleNamespace(params=SimpleNamespace(score_threshold=threshold))
    runner.postprocess = PostProcess(num_select=2, num_keypoints_per_class=[0, 17], trace_alpha=0.2)
    # One background and one person query. Unit precision gives covariance trace 2.
    points = torch.zeros(1, 2, 34, 7)
    points[..., 0] = 0.25
    points[..., 1] = 0.75
    raw = {
        "pred_logits": torch.tensor([[[8.0, -8.0], [-8.0, 4.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]]]),
        "pred_keypoints": points,
    }
    monkeypatch.setattr(runner, "_raw", lambda _image: raw)
    return runner


def test_person_schema_scaling_and_uncertainty_score(monkeypatch):
    runner = _runner(monkeypatch, 0.0)
    got = runner(torch.zeros(1, 3, 100, 200), None)
    assert got["keypoints"].shape == (1, 17, 2)
    torch.testing.assert_close(got["keypoints"], torch.tensor([50.0, 75.0]).expand(1, 17, 2))
    torch.testing.assert_close(got["boxes"], torch.tensor([[50.0, 25.0, 150.0, 75.0]]))
    torch.testing.assert_close(got["keypoint_scores"], torch.full((1, 17), 0.5))
    expected_score = (torch.tensor(4.0).sigmoid().log() - 0.2 * torch.tensor(2.0).log()).sigmoid()
    torch.testing.assert_close(got["scores"], expected_score[None])


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_threshold_applies_after_uncertainty_fusion_and_keeps_empty_shapes(monkeypatch, threshold):
    # The raw class score is above 0.98, but uncertainty-fused score is below 0.5.
    got = _runner(monkeypatch, threshold)(torch.zeros(1, 3, 100, 200), None)
    assert got["keypoints"].shape == (0, 17, 2)
    assert got["keypoint_scores"].shape == (0, 17)
    assert got["boxes"].shape == (0, 4)
    assert got["scores"].shape == (0,)


def test_uncertainty_fusion_promotes_half_and_handles_sharp_precision():
    points = torch.zeros(2, 17, 7, dtype=torch.float16)
    points[1, :, 4] = points[1, :, 6] = 20
    got = PostProcess._keypoint_log_mean_trace(points)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, torch.tensor([2.0, 2.0]).log() - torch.tensor([0.0, 40.0]))


@pytest.mark.gpu
@pytest.mark.model
def test_graph_replay_reuses_fixed_network_shape_and_preserves_outputs():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks import pose  # noqa: PLC0415 - GPU-only model load
    from fastforks._core.errors import WeightsError  # noqa: PLC0415 - GPU-only model load

    try:
        model = pose.load("rf-detr-keypoint-preview", mode="kernels", device="cuda")
    except WeightsError as exc:
        pytest.skip(f"RF-DETR weights not available: {exc}")
    runner = model._runner
    x = torch.zeros(1, 3, 96, 128, device="cuda")
    y = torch.ones(1, 3, 128, 96, device="cuda")
    with torch.inference_mode():
        runner._raw(x)  # first call warms the fixed 576 x 576 network input
        first = runner._raw(x)
        retained = {key: value.clone() for key, value in first.items()}
        assert len(runner.graphs.graphs) == 1, f"no capture: {list(runner.graphs._failed)}"
        second = runner._raw(y)
        eager = runner.net(runner._prepare(y))
        for key, want in zip(("pred_boxes", "pred_logits", "pred_keypoints"), eager, strict=True):
            torch.testing.assert_close(second[key], want)
            torch.testing.assert_close(first[key], retained[key], rtol=0, atol=0)
        assert len(runner.graphs.graphs) == 1  # source image size does not create another graph
    model.release_graphs()
    assert not runner.graphs.graphs


def test_fused_paths_do_not_bypass_class_level_overrides():
    from fastforks._core.patch import override  # noqa: PLC0415 - targeted override guard
    from fastforks.shared.rfdetr.net.models.transformer import (  # noqa: PLC0415 - optional network internals
        _module_call_is_plain,
    )

    module = torch.nn.Linear(2, 2)
    assert _module_call_is_plain(module)

    def replaced(_module, value):
        return value + 1

    override(module, replaced)
    assert not _module_call_is_plain(module)


def test_export_adapter_does_not_keep_network_alive():
    import weakref  # noqa: PLC0415 - lifetime regression

    from fastforks.shared.rfdetr.kernels import prepare_graph  # noqa: PLC0415 - adapter-only, no CUDA load

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(enable_cuda_graph_capture=lambda: None)

        def forward(self, value):
            return value

        def forward_export(self, value):
            return value + 1

        def export(self):
            self._forward_origin = self.forward
            self.forward = self.forward_export

    net = Network()
    prepare_graph(net)
    assert net(torch.tensor(1)).item() == 2
    ref = weakref.ref(net)
    del net
    assert ref() is None  # no cyclic garbage collection needed to free the network
