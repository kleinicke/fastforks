"""SegNeXt: the BatchNorm / layer-scale fold leaves plain constants behind."""

from __future__ import annotations

import torch

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
