"""The shared VGGT rotary helper remains traceable."""

import pytest
import torch

from fastforks.multiview.models.vggt.model import _swap


@pytest.mark.slow
def test_swap_compiles_without_cached_function_warning() -> None:
    x = torch.randn(2, 3, 64)
    compiled = torch.compile(_swap, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), _swap(x), rtol=0, atol=0)
