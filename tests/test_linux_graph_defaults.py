"""CPU checks of the input layouts and request boundary for measured Linux CUDA graphs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch

from fastforks._core import linux_graph_defaults
from fastforks.shared.cuda_graphs import GraphCache, graph_replay_allowed, graph_request

if TYPE_CHECKING:
    import pytest

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
            backend=SimpleNamespace(mode="kernels"),
            card=SimpleNamespace(task="pose", family="yunet"),
            weights=None,
            options={"cuda_graphs": True},
        ),
    )
    monkeypatch.setattr(linux_graph_defaults.sys, "platform", "linux")
    assert not linux_graph_defaults.allows(model)
    monkeypatch.setattr(linux_graph_defaults.sys, "platform", "win32")
    assert linux_graph_defaults.allows(model)


def test_nested_request_cannot_reenable_denied_graph() -> None:
    cache = GraphCache(torch.neg)
    assert graph_replay_allowed()
    with graph_request(allow_replay=False), graph_request():
        assert not graph_replay_allowed()
        assert cache.defer_first_request("shape")
        assert cache.run("shape", torch.ones(1)) is None
        assert cache.static_inputs("shape") is None
    assert graph_replay_allowed()
