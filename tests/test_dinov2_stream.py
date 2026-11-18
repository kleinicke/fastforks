"""DINOv2 block streaming covers both feed-forward layouts and fused and unfused token counts."""

import copy
from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.features.models.dinov2 import model


def test_stream_requires_cuda() -> None:
    ctx = SimpleNamespace(
        device=torch.device("cpu"), params=model.Params("vits", 0, fp16_accumulate=False, stream=True)
    )
    with pytest.raises(OptionsError, match="requires CUDA"):
        model._network(ctx)  # ty: ignore[invalid-argument-type]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("ffn", ["mlp", "swiglufused"])
def test_streamed_blocks_match_across_sizes(mode: str, ffn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(21)
    vit = model._Vit(
        img_size=56,
        patch_size=14,
        embed_dim=128,
        num_heads=2,
        depth=3,
        init_values=0.7,
        block_chunks=0,
        ffn_layer=ffn,
        num_register_tokens=4,
    ).eval()
    monkeypatch.setattr(model, "_network", lambda _ctx: copy.deepcopy(vit))
    runners = [
        model.RUNNERS[mode](
            SimpleNamespace(  # ty: ignore[invalid-argument-type]
                device=torch.device("cuda"),
                dtype="float16",
                params=model.Params("vits", 4, fp16_accumulate=False, stream=enabled),
            )
        )
        for enabled in (False, True)
    ]
    with torch.inference_mode():
        for side in (56, 322, 56):
            image = torch.randn(1, 3, side, side, device="cuda")
            expected, streamed = (runner(image, (0, 2, 3)) for runner in runners)
            for name in expected:
                torch.testing.assert_close(streamed[name], expected[name], atol=0, rtol=0)
