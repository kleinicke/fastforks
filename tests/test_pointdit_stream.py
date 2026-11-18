"""PointDiT folds per-step gates on the target device before staging the transformer blocks."""

from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.depth.models.pointdit.model import Portable
from fastforks.depth.models.pointdit.net import Config, PointDiT, checkpoint_shapes, sample
from fastforks.shared.host_blocks import StreamedModuleList


def test_stream_requires_cuda_before_weights() -> None:
    ctx = SimpleNamespace(device=torch.device("cpu"), params=SimpleNamespace(stream=True))
    with pytest.raises(OptionsError, match="requires CUDA"):
        Portable(ctx)  # ty: ignore[invalid-argument-type] - minimal context


@pytest.mark.gpu
def test_folded_streamed_sampler_matches_across_steps() -> None:
    torch.manual_seed(24)
    cfg = Config(depth=5, hidden=128, heads=2, enc_depth=4, enc_hidden=64)
    sd = {k: torch.randn(shape) * 0.05 for k, shape in checkpoint_shapes(cfg).items()}
    dev = torch.device("cuda")
    ln = (torch.ones(cfg.enc_hidden, device=dev), torch.zeros(cfg.enc_hidden, device=dev))
    with torch.inference_mode():
        resident = PointDiT({k: v.to(dev) for k, v in sd.items()}, ln, cfg, 3).cast(torch.float16)
        streamed = PointDiT(sd, ln, cfg, 3, device=dev, dtype=torch.float16, stream=True)
        streamed.blocks = StreamedModuleList(streamed.layers, dev)  # ty: ignore[invalid-assignment] - callable blocks
        for h, w in [(3, 5), (5, 4), (3, 5)]:
            hidden = torch.randn(2, 4, h * w + 5, cfg.enc_hidden, device=dev)
            got = sample(streamed, streamed.condition(hidden, h, w), h, w)
            expected = sample(resident, resident.condition(hidden, h, w), h, w)
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
