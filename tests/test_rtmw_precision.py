"""The RTMW-X384 precision recipe preserves residual/activation and selects explicit graph ops."""

import torch

from fastforks._optimize import ops
from fastforks.pose.models.rtmpose.model import _precise_backbone, _precise_conv


def test_precise_conv_residual_precedes_activation() -> None:
    x = torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float16)
    w = torch.ones((1, 1, 1, 1), dtype=torch.float16)
    residual = torch.tensor([[[[2.0, -2.0]]]], dtype=torch.float16)
    result = _precise_conv(x, w, None, residual, (0, 0), (1, 1), 1, channels=1)
    torch.testing.assert_close(result, torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16))
    assert result.is_contiguous(memory_format=torch.channels_last)


def test_precision_recipe_selects_accumulators() -> None:
    graph = torch.fx.Graph()
    x, w = graph.placeholder("x"), graph.placeholder("w")
    conv = graph.call_function(ops.conv_fused, (x, w, None, None, 0, 1, 3))
    depthwise = graph.call_function(ops.dwconv2, (conv, w, None, 0, 1, 3))
    graph.output(depthwise)
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    _precise_backbone(module, 64)
    assert conv.target is _precise_conv
    assert conv.kwargs == {"channels": 64}
    assert depthwise.target is ops.dwconv
    graph.lint()
