"""MegaFlow rotary positions preserve their eager result when compiled."""

import pytest
import torch

from fastforks.flow.models.megaflow.net.layers.rope import RotaryPositionEmbedding2D


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_positions_compile_without_scalar_extraction(dtype: torch.dtype) -> None:
    model = RotaryPositionEmbedding2D().eval()
    tokens = torch.randn(2, 3, 12, 64).to(dtype)
    positions = torch.tensor([[[i // 4, i % 4] for i in range(12)]]).expand(2, -1, -1)
    eager = model(tokens, positions)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(tokens, positions), eager, rtol=0, atol=0)
