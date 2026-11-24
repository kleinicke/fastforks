"""Omni-DC solver: operator agreement at borders and at both pooled resolutions, the exported solve."""

import pytest
import torch

from fastforks.depth.models.omni_dc.net import CGState, normal_op, solve
from fastforks.kernels import omni_cg


@pytest.mark.gpu
@pytest.mark.parametrize(("batch", "height", "width"), [(1, 8, 12), (2, 36, 60), (1, 120, 208)])
def test_fused_normal_operator(batch: int, height: int, width: int) -> None:
    direction = torch.randn((batch, 1, height, 2 * width), device="cuda")[..., ::2]
    diagonal = torch.rand_like(direction)
    actual = omni_cg.apply(direction, diagonal)
    expected = normal_op(direction, diagonal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _Solve(torch.nn.Module):
    def forward(self, d: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return solve(d, b)


def test_exported_solve_keeps_the_data_dependent_step_count() -> None:
    """`torch.export` captures the solve as one `torch.while_loop`: the eager iterate, whatever the step count."""
    gen = torch.Generator().manual_seed(0)

    def problem(share: float) -> tuple[torch.Tensor, torch.Tensor]:
        d = torch.rand((2, 1, 16, 24), generator=gen) * (torch.rand((2, 1, 16, 24), generator=gen) < share)
        return d, torch.randn((2, 1, 16, 24), generator=gen)

    easy, hard = problem(1.0), problem(0.05)
    steps = []
    for d, b in (easy, hard):
        state = CGState(d, b)
        while not bool(state.done):
            state.step()
        steps.append(int(state.k))
    assert steps[0] != steps[1]
    exported = torch.export.export(_Solve(), easy, strict=False).module()
    for d, b in (easy, hard):
        torch.testing.assert_close(exported(d, b), solve(d, b), rtol=1e-5, atol=1e-6)
