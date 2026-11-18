"""`shared.arena.compact`: values kept, one allocation, no freed holes left in the reserved memory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks.shared import arena
from fastforks.shared.arena import compact, compact_runner


@pytest.fixture(autouse=True)
def _small_nets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arena, "_MIN_BYTES", 0)  # the unit nets are far below the size threshold


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
def test_compact_gpu_one_storage_aligned_ties_and_views_kept() -> None:
    net = _net("cuda")
    want = {k: v.clone() for k, v in net.state_dict().items()}
    net.alias = nn.Parameter(net[2].weight.data, requires_grad=False)  # shares storage with 2.weight: stays shared
    compact(net)
    assert all(torch.equal(want[k], net.state_dict()[k]) for k in want)
    arena = [p for n, p in net.named_parameters() if n in {"0.weight", "0.bias", "1.weight", "1.bias"}]
    assert len({p.untyped_storage().data_ptr() for p in arena}) == 1
    assert all(p.data_ptr() % 512 == 0 for p in arena)
    assert net.alias.data_ptr() == net[2].weight.data_ptr()
    assert net[2].weight.untyped_storage().data_ptr() == net.alias.untyped_storage().data_ptr()


@pytest.mark.gpu
@pytest.mark.slow
def test_compact_keeps_views_of_one_storage_and_buffers() -> None:
    net = nn.Module()
    packed = torch.arange(24, dtype=torch.float16, device="cuda")
    net.a = nn.Parameter(packed[:8].view(2, 4), requires_grad=False)  # views of one packed storage
    net.b = nn.Parameter(packed[8:].view(4, 4)[:, :2], requires_grad=False)  # not contiguous
    net.register_buffer("c", torch.ones(3, device="cuda"))
    want = [t.clone() for t in (net.a, net.b, net.c)]
    del packed
    compact(net)
    assert all(torch.equal(w, t) for w, t in zip(want, (net.a, net.b, net.c), strict=True))
    assert net.a.untyped_storage().data_ptr() == net.b.untyped_storage().data_ptr()
    assert net.b.stride() == (4, 1)


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


def test_threshold_leaves_small_nets_alone() -> None:
    net = _net("cuda" if torch.cuda.is_available() else "cpu")
    ptrs = [p.data_ptr() for p in net.parameters()]
    compact(net, min_bytes=128 << 20)
    assert [p.data_ptr() for p in net.parameters()] == ptrs


@pytest.mark.gpu
@pytest.mark.slow
def test_tied_weights_stay_tied_and_cuda_graph_captures_after_compaction() -> None:
    net = nn.Linear(16, 16).cuda().half()
    net.tied = nn.Linear(16, 16, bias=False).cuda().half()
    net.tied.weight = net.weight  # one Parameter in two modules
    compact(net)
    assert net.tied.weight is net.weight
    x = torch.randn(4, 16, device="cuda", dtype=torch.half)
    net(x)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        y = net.tied(net(x))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, net.tied(net(x)))


@pytest.mark.gpu
@pytest.mark.slow
def test_compact_runner_handles_attribute_modules() -> None:
    class R:
        def networks(self) -> list[nn.Module]:
            raise NotImplementedError

    r = R()
    r.net = _net("cuda")  # type: ignore[attr-defined]
    want = {k: v.clone() for k, v in r.net.state_dict().items()}  # type: ignore[attr-defined]
    compact_runner(r)
    assert all(torch.equal(want[k], v) for k, v in r.net.state_dict().items())  # type: ignore[attr-defined]


@pytest.mark.gpu
def test_upload_preserves_cpu_storage_aliases_and_offsets() -> None:
    from fastforks.shared.arena import upload  # noqa: PLC0415

    module = nn.Module()
    original = torch.arange(40, dtype=torch.float32).view(5, 8)
    module.register_parameter("weight", nn.Parameter(original))
    module.register_parameter("tied", module.get_parameter("weight"))
    module.register_buffer("stride_view", original[1::2, 2::2])
    module.register_buffer("integer", torch.arange(7, dtype=torch.int16))
    upload(module, torch.device("cuda"))
    weight = module.get_parameter("weight")
    stride_view = module.get_buffer("stride_view")
    integer = module.get_buffer("integer")
    assert module.get_parameter("tied") is weight
    assert stride_view.stride() == original[1::2, 2::2].stride()
    assert len({t.untyped_storage().data_ptr() for t in (weight, stride_view, integer)}) == 1
    torch.testing.assert_close(stride_view.cpu(), original[1::2, 2::2])
    torch.testing.assert_close(integer.cpu(), torch.arange(7, dtype=torch.int16))
