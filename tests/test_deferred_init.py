"""`deferred_init`: loaded tensors skip the random fill; tensors the checkpoint lacks still get it."""

from __future__ import annotations

import threading

import pytest
import torch

from fastforks._core.deferred_init import deferred_init


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        self.lin = torch.nn.Linear(8, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.extra = torch.nn.Parameter(torch.empty(5))
        torch.nn.init.normal_(self.extra, std=0.5)


def _reference() -> _Net:
    torch.manual_seed(0)
    return _Net()


def test_loaded_tensors_skip_the_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = torch.nn.init.kaiming_uniform_
    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", lambda t, *a, **k: calls.append(t) or original(t, *a, **k))
    sd = _reference().state_dict()
    calls.clear()
    with deferred_init():
        net = _Net()
        net.load_state_dict(sd)
    assert not calls
    for k, v in net.state_dict().items():
        assert torch.equal(v, sd[k]), k
    assert torch.nn.init.kaiming_uniform_ is not original  # the test's patch is restored, not torch's
    monkeypatch.undo()
    assert torch.nn.init.kaiming_uniform_ is original


def test_unloaded_tensors_are_initialised() -> None:
    sd = {k: v for k, v in _reference().state_dict().items() if k not in {"extra", "lin.bias"}}
    with deferred_init():
        net = _Net()
        net.load_state_dict(sd, strict=False)
    assert torch.isfinite(net.extra).all()
    assert 0 < net.extra.std() < 2
    bound = 1 / 8**0.5  # Linear's bias fill: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    assert net.lin.bias.abs().max() <= bound
    assert net.lin.bias.abs().sum() > 0
    assert torch.equal(net.norm.weight, torch.ones(4))  # ones_ ran: the checkpoint has it, and it is ones there too


def test_constant_inits_still_apply_without_a_checkpoint() -> None:
    with deferred_init():
        norm = torch.nn.LayerNorm(6)
    assert torch.equal(norm.weight, torch.ones(6))
    assert torch.equal(norm.bias, torch.zeros(6))


def test_assigned_tensors_are_not_touched() -> None:
    sd = _reference().state_dict()
    with deferred_init():
        net = _Net()
        net.load_state_dict(sd, assign=True)
    for k, v in net.state_dict().items():
        assert torch.equal(v, sd[k]), k


def test_a_failing_build_initialises_nothing_and_restores_torch() -> None:
    original = torch.nn.init.uniform_
    patched = []

    def build() -> None:
        with deferred_init():
            patched.append(torch.nn.init.uniform_ is not original)
            lin = torch.nn.Linear(2, 2)
            patched.append(lin)
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        build()
    assert patched[0]
    assert torch.equal(patched[1].weight, torch.zeros(2, 2))  # only the record-time memset ran
    assert torch.nn.init.uniform_ is original


def test_other_threads_initialise_normally() -> None:
    seen: list[torch.Tensor] = []
    entered, done = threading.Event(), threading.Event()

    def other() -> None:
        entered.wait()
        seen.append(torch.nn.init.uniform_(torch.zeros(3), 1.0, 1.0))  # a deferred initialiser
        done.set()

    th = threading.Thread(target=other)
    th.start()
    with deferred_init():
        entered.set()
        done.wait()
    th.join()
    assert torch.equal(seen[0], torch.ones(3))


def test_nested_blocks_share_the_outer_record() -> None:
    with deferred_init():
        with deferred_init():
            lin = torch.nn.Linear(4, 4)
        assert not lin.weight.any()  # the inner block's end initialised nothing
    assert lin.weight.any()  # the outer block's end did


def test_data_assignment_keeps_its_values() -> None:
    fused = torch.full((4, 8), 0.25)
    with deferred_init():
        net = _Net()
        net.lin.weight.data = fused  # a reparameterisation: invisible to the version counter
    assert torch.equal(net.lin.weight, fused)


def test_data_copy_keeps_its_values() -> None:
    fused = torch.full((4, 8), 0.25)
    with deferred_init():
        net = _Net()
        net.lin.weight.data.copy_(fused)  # timm's `fuse()` (RepViT): an in-place write that bumps no version counter
    assert torch.equal(net.lin.weight, fused)


def test_moved_unloaded_tensors_hold_zeros() -> None:
    with deferred_init():
        net = _Net().to(torch.float64)
    assert torch.equal(net.lin.weight, torch.zeros_like(net.lin.weight))


def test_a_reused_address_does_not_count_as_unwritten() -> None:
    lin = None
    with deferred_init():
        lin = torch.nn.Linear(64, 64)
        lin.weight.data = torch.full((64, 64), 0.5)  # frees the first storage ...
        lin.weight.data = lin.weight.data.clone()  # ... whose address the allocator may hand to this copy
    assert torch.equal(lin.weight, torch.full((64, 64), 0.5))


class _Scaled(torch.nn.Module):
    """A LayerScale-like weight and a non-persistent buffer: neither is in a checkpoint's state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.gamma = torch.nn.Parameter(torch.empty(4))
        torch.nn.init.constant_(self.gamma, 0.1)
        self.register_buffer("scale", torch.nn.init.ones_(torch.empty(4)), persistent=False)


def _cast_keeps_deterministic_inits(device: str, dtype: torch.dtype) -> None:
    sd = {k: v for k, v in _Scaled().state_dict().items() if k.startswith("lin.")}
    with deferred_init():
        net = _Scaled().to(device, dtype)  # new storages before the block ends
        net.load_state_dict(sd, strict=False)
    assert torch.equal(net.scale, torch.ones(4, device=device, dtype=dtype))
    assert torch.equal(net.gamma, torch.full((4,), 0.1).to(device, dtype))
    assert torch.equal(net.lin.weight, sd["lin.weight"].to(device, dtype))


def test_deterministic_inits_survive_a_cast() -> None:
    _cast_keeps_deterministic_inits("cpu", torch.float64)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_deterministic_inits_survive_a_move_to_the_gpu() -> None:
    _cast_keeps_deterministic_inits("cuda", torch.half)
