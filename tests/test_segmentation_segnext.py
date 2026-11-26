"""SegNeXt: the BatchNorm / layer-scale fold leaves plain constants behind; kernels graphs at the ADE20K test size."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from fastforks import segmentation
from fastforks._core.errors import WeightsError
from fastforks.segmentation.models.segnext.net import build, fold


def test_fold_leaves_no_buffer_that_requires_grad() -> None:
    """`res_scale` is computed from Parameters: kept attached it requires grad, and torch.export warns about (and
    detaches) such an attribute, which failed test_module_compile_export (warnings are errors there)."""
    net = build("mscan-t").eval()
    fold(net)
    grads = [name for name, b in net.named_buffers() if b.requires_grad]
    assert not grads, grads[:3]
    scales = [m.res_scale for m in net.modules() if getattr(m, "res_scale", None) is not None]
    assert scales
    assert not any(s.requires_grad for s in scales)
    with torch.inference_mode():
        assert net(torch.rand(1, 3, 64, 64)).isfinite().all()


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.usefixtures("graph_lifecycle")
def test_kernels_graph_at_the_test_size() -> None:
    """A 1024 x 2048 image (network input 512 x 1024, launch-bound) replays a graph: 1/8 logits == eager."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        m = segmentation.load("segnext-mscan-t-ade20k", mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"segnext-mscan-t-ade20k weights not available: {e}")
    r: Any = m._runner
    g = r._graphs
    gen = torch.Generator("cuda").manual_seed(0)
    x, y = (torch.rand(1, 3, 1024, 2048, device="cuda", generator=gen) for _ in range(2))
    with torch.inference_mode():
        pixels, g.pixels = g.pixels, 0
        eager = [r._scores(x)[0], r._scores(y)[0]]
        g.pixels = pixels
        r._scores(x)  # first call: eager
        assert not g.cache.graphs
        first = r._scores(x)[0]  # second: captures
        assert len(g.cache.graphs) == 1
        second = r._scores(y)[0]  # replay
    torch.testing.assert_close(first, eager[0], rtol=0, atol=0)
    torch.testing.assert_close(second, eager[1], rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()
    m.release_graphs()
    assert not g.cache.graphs
