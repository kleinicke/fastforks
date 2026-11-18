"""The shared VGGT rotary helper remains traceable."""

import pytest
import torch

from fastforks.multiview.models.vggt.model import _swap


@pytest.mark.slow
def test_swap_compiles_without_cached_function_warning() -> None:
    x = torch.randn(2, 3, 64)
    compiled = torch.compile(_swap, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), _swap(x), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(("batch", "tokens", "heads"), [(1, 13, 16), (3, 5, 4)])
def test_rotary_matches_half_operations(batch: int, tokens: int, heads: int) -> None:
    from fastforks.kernels import vggt_rotary  # noqa: PLC0415
    from fastforks.multiview.models.vggt.model import _Fast  # noqa: PLC0415

    def values(*shape: int) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float16, device="cuda")

    norm = values(batch, tokens, 2, heads, 64)
    weight, bias = values(2, 2, 1, 64)
    cos, sin = values(2, batch, tokens, 1, 1, 64)
    expected = _Fast.rotate(norm, weight, bias, cos, sin)
    assert torch.equal(vggt_rotary.apply(norm, weight, bias, cos, sin), expected)


def test_vggt_stream_requires_cuda_before_checkpoint_load() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks._core.errors import OptionsError  # noqa: PLC0415
    from fastforks.multiview.models.vggt.model import Params, Portable  # noqa: PLC0415

    ctx = SimpleNamespace(params=Params(long_side=518, stream=True), device=torch.device("cpu"))
    with pytest.raises(OptionsError, match="requires CUDA"):
        Portable(ctx)  # ty: ignore[invalid-argument-type] - the check runs before loading weights
