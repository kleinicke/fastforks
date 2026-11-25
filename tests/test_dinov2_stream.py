"""DINOv2 and TIPSv2 streaming preserves intermediate features across token counts."""

import copy
from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.features.models.dinov2 import model as dinov2
from fastforks.features.models.tipsv2 import model as tipsv2


@pytest.mark.parametrize("family", ["dinov2", "tipsv2"])
def test_stream_requires_cuda(family: str) -> None:
    model = tipsv2 if family == "tipsv2" else dinov2
    ctx = SimpleNamespace(device=torch.device("cpu"), params=SimpleNamespace(stream=True))
    with pytest.raises(OptionsError, match="requires CUDA"):
        model._network(ctx)  # ty: ignore[invalid-argument-type]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize(("family", "ffn"), [("dinov2", "mlp"), ("dinov2", "swiglufused"), ("tipsv2", "mlp")])
def test_streamed_blocks_match_across_sizes(mode: str, family: str, ffn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tipsv2 if family == "tipsv2" else dinov2
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
        num_register_tokens=1 if family == "tipsv2" else 4,
    ).eval()
    monkeypatch.setattr(model, "_network", lambda _ctx: copy.deepcopy(vit))
    runners = [
        model.RUNNERS[mode](
            SimpleNamespace(  # ty: ignore[invalid-argument-type]
                device=torch.device("cuda"),
                dtype="float16",
                params=SimpleNamespace(
                    arch="vitb" if family == "tipsv2" else "vits", fp16_accumulate=False, stream=enabled
                ),
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
