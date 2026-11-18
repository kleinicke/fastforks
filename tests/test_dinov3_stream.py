"""Streamed ViT blocks use the same packed tensors and hidden-state indices as resident DINOv3."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.features.models.dinov3 import model
from fastforks.features.models.dinov3.net import CONFIGS, Config, DINOv3ViT


def test_cpu_stream_rejected_before_loading() -> None:
    ctx = SimpleNamespace(device=torch.device("cpu"), params=model.Params("vits16", fp16_accumulate=False, stream=True))
    with pytest.raises(OptionsError, match="requires CUDA"):
        model._state(ctx)  # ty: ignore[invalid-argument-type] - minimal context


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("gated", [True, False])
def test_staged_block_sequence(mode: str, gated: bool, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: FBT001
    arch = "vith16plus" if gated else "vits16"
    cfg = Config(128, 2, 5, 256, gated=gated)
    monkeypatch.setitem(CONFIGS, arch, cfg)
    torch.manual_seed(37)
    state = DINOv3ViT(cfg).state_dict()
    for name, value in state.items():
        if value.ndim == 1 and name.endswith("weight"):
            value.fill_(1.0)
        else:
            value.normal_(0, 0.05)

    def make(*, stream: bool) -> Any:
        ctx = SimpleNamespace(
            device=torch.device("cuda"),
            dtype="float16",
            torch_dtype=torch.float16,
            mode=mode,
            params=model.Params(arch, fp16_accumulate=False, stream=stream),
            variant=SimpleNamespace(ref=arch),
            state_dict=lambda: dict(state),
        )
        return model.RUNNERS[mode](ctx)  # ty: ignore[invalid-argument-type] - test checkpoint

    with torch.inference_mode():
        resident, streamed = make(stream=False), make(stream=True)
        assert all(p.is_meta for blk in streamed.net.layer.shells for p in blk.parameters())
        for h, layers in ((64, ()), (48, (0, 2, 5)), (80, ())):
            x = torch.randn(2, 3, h, 64, device="cuda")
            expected = resident(x, layers)
            actual = streamed(x, layers)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
