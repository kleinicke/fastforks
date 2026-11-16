"""`shared.arena.compact`: values kept, one allocation, no freed holes left in the reserved memory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks.shared.arena import compact


def _net(device: str) -> nn.Module:
    net = nn.Sequential(nn.Linear(40, 24), nn.LayerNorm(24), nn.Linear(24, 8, bias=False)).to(device).half()
    net.register_buffer("scale", torch.arange(5, dtype=torch.float32, device=device), persistent=False)
    return net


def test_compact_keeps_values_and_shares_one_arena_cpu_untouched() -> None:
    net = _net("cpu")
    before = {k: v.clone() for k, v in net.state_dict().items()}
    ptrs = [p.data_ptr() for p in net.parameters()]
    compact(net)  # CPU tensors are left alone
    assert [p.data_ptr() for p in net.parameters()] == ptrs
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())


@pytest.mark.gpu
@pytest.mark.slow
def test_compact_gpu_one_storage_per_dtype_aligned_ties_kept() -> None:
    net = _net("cuda")
    want = {k: v.clone() for k, v in net.state_dict().items()}
    net.alias = nn.Parameter(net[2].weight.data, requires_grad=False)  # shares storage with 2.weight: stays shared
    compact(net)
    assert all(torch.equal(want[k], net.state_dict()[k]) for k in want)
    arena = [p for n, p in net.named_parameters() if n in {"0.weight", "0.bias", "1.weight", "1.bias"}]
    assert len({p.untyped_storage().data_ptr() for p in arena}) == 1
    assert all(p.data_ptr() % 512 == 0 for p in arena)
    assert net.alias.data_ptr() == net[2].weight.data_ptr()


@pytest.mark.gpu
@pytest.mark.slow
def test_compact_leaves_no_holes_in_reserved_memory() -> None:
    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved()
    net = nn.ModuleList()
    for _ in range(
        40
    ):  # sizes the allocator rounds up (2 MB pieces in 20 MB segments), with freed fp32 transients between
        tmp = torch.randn(1024, 1024, device="cuda")
        net.append(nn.Linear(1024, 1024).half().cuda())
        del tmp
    live = torch.cuda.memory_allocated()
    compact(net)
    torch.cuda.empty_cache()
    assert torch.cuda.memory_reserved() - base <= live + (4 << 20)
