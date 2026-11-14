"""No float16 model keeps a BatchNorm whose running statistics overflow float16.

Some checkpoints hold BatchNorm running variances far beyond the float16 range (CFNet, CAS-GwcNet, DEFOM-Stereo, LAS2,
InSPyReNet, EfficientViT-SAM XL: up to 3e10). Cast to float16 unfolded, they become inf and the layer outputs zeros
or NaN; folded into the convolution before the cast (in float32) they are harmless. Every golden ref with a float16
CUDA path is loaded there (portable, and kernels where the card has it) and each BatchNorm left in the runner's modules
must have finite statistics and affine parameters.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks._core.errors import WeightsError
from test_golden import API, GOLDEN, TASK
from test_kernels_structure import held_modules

pytestmark = [pytest.mark.model, pytest.mark.gpu, pytest.mark.slow]


def _bad_batchnorms(model: object) -> list[str]:
    """`name: tensor` of every half-precision BatchNorm tensor with a non-finite value."""
    bad = []
    for root in held_modules(model._runner):
        for name, m in root.named_modules():
            if not isinstance(m, nn.modules.batchnorm._BatchNorm):  # every BatchNorm1d / 2d / 3d
                continue
            for key, t in (*m.named_buffers(recurse=False), *m.named_parameters(recurse=False)):
                if t.is_floating_point() and t.dtype != torch.float32 and not torch.isfinite(t).all():
                    bad.append(f"{name}.{key} ({int((~torch.isfinite(t)).sum())} non-finite)")
    return bad


@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("ref", sorted(GOLDEN))
def test_fp16_batchnorm_stats_are_finite(ref, mode):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    card = API[TASK[ref]].model_card(ref)
    if mode not in card.modes:
        pytest.skip(f"{ref} has no {mode}")
    try:
        model = API[TASK[ref]].load(ref, mode=mode, device="cuda")
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")
    try:
        bad = _bad_batchnorms(model)
        assert not bad, f"{ref} {mode}: fold these BatchNorms before the float16 cast: {bad[:5]}"
    finally:
        del model
        torch.cuda.empty_cache()
