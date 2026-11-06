"""`_core.patch.override`: per-instance method overrides without a module -> bound method -> module cycle."""

from __future__ import annotations

import copy
import gc
import re
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import fastforks
from fastforks._core.patch import base_type, override


class Net(nn.Module):
    """A toy net."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _double(self: Net, x: torch.Tensor) -> torch.Tensor:
    return 2 * base_type(self).forward(self, x)


def _shift(self: Net, x: torch.Tensor) -> torch.Tensor:
    return self.forward(x) + 1


def _conv_half(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return nn.Conv2d.forward(self, x) / 2


@pytest.fixture
def gc_off():
    gc.collect()
    gc.disable()
    yield
    gc.enable()


def test_override_runs_fn_on_this_instance_only():
    a, b = Net(), Net()
    b.load_state_dict(a.state_dict())
    x = torch.rand(1, 3, 8, 8)
    override(a, _double)
    torch.testing.assert_close(a(x), 2 * b(x))
    assert type(b) is Net


def test_no_reference_cycle(gc_off):
    m = Net()
    override(m, _double)
    override(m.conv, _conv_half)
    ref, ref_conv = weakref.ref(m), weakref.ref(m.conv)
    del m
    assert ref() is None  # freed by reference counting, without gc.collect()
    assert ref_conv() is None


def test_bound_method_assignment_is_a_cycle(gc_off):
    """The pattern `override` replaces: documents why it exists."""
    m: Any = Net()
    m.forward = _double.__get__(m)
    ref = weakref.ref(m)
    del m
    assert ref() is not None
    gc.collect()
    assert ref() is None


def test_the_module_looks_unchanged():
    m = Net()
    before = (repr(m), list(m.state_dict()), Net.__name__)
    override(m, _double)
    override(m.conv, _conv_half)
    assert isinstance(m, Net)
    assert isinstance(m.conv, nn.Conv2d)
    assert base_type(m) is Net
    assert base_type(m.conv) is nn.Conv2d
    assert base_type(3) is int
    assert type(m).__name__ == "Net"
    assert type(m.conv).__qualname__ == "Conv2d"
    assert (repr(m), list(m.state_dict()), Net.__name__) == before
    m.load_state_dict(Net().state_dict())


def test_classes_are_cached_and_overrides_combine():
    a: Any = Net()
    b = Net()
    override(a, _double)
    override(b, _double)
    assert type(a) is type(b)
    override(a, _shift, "shifted")  # a second method keeps the first override
    x = torch.rand(1, 3, 8, 8)
    torch.testing.assert_close(a.shifted(x), a(x) + 1)
    assert type(a) is not type(b)
    assert base_type(a) is Net
    override(b, _shift, "shifted")
    assert type(a) is type(b)


def test_replaces_an_instance_attribute():
    m: Any = Net()
    m.forward = lambda x: x  # an earlier instance-level override would shadow the class method
    override(m, _double)
    assert "forward" not in m.__dict__
    x = torch.rand(1, 3, 8, 8)
    torch.testing.assert_close(m(x), 2 * m.conv(x))


def test_compile_and_export():
    m = Net().eval()
    override(m, _double)
    override(m.conv, _conv_half)
    x = torch.rand(1, 3, 8, 8)
    want = m(x)
    explain = torch._dynamo.explain(m)(x)
    assert explain.graph_break_count == 0, explain.break_reasons
    ep = torch.export.export(m, (x,), strict=False)
    torch.testing.assert_close(ep.module()(x), want)


def test_fast_vit_no_reference_cycle(gc_off):
    """`use_fast_vit` keeps its ViT weakly: the ViT holds the FastViT through its `get_intermediate_layers`."""
    from fastforks.shared.depth_anything_v2 import use_fast_vit  # noqa: PLC0415 - needs the DINOv2 net
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    vit = DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=64, depth=1, num_heads=1, block_chunks=0)
    fast = use_fast_vit(vit)
    assert fast.vit is vit
    ref = weakref.ref(vit)
    del vit
    assert ref() is None
    with pytest.raises(ReferenceError):
        _ = fast.vit


def test_fast_vit_subclasses_hold_the_vit():
    """Used directly (Metric3D, UniDepth, DAv2 fused), `FastViT` holds its ViT: nothing points back to it."""
    from fastforks.shared.depth_anything_v2 import FastViT  # noqa: PLC0415 - needs the DINOv2 net
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    def tiny() -> DinoVisionTransformer:
        return DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=64, depth=1, num_heads=1, block_chunks=0)

    vit = tiny()
    fast = FastViT(vit, "hh", None)
    ref = weakref.ref(vit)
    del vit
    assert fast.vit is ref()
    other = tiny()
    fast.vit = other  # subclasses reassign it
    assert fast.vit is other


def test_defom_max_encoder_fuses_all_resblocks():
    """The instance norms are overridden first; the ResidualBlock check after that must still see them."""
    from fastforks.stereo.models.defom_stereo.kernels.encoder import optimize_encoder  # noqa: PLC0415 - DEFOM only
    from fastforks.stereo.models.defom_stereo.net.extractor import BasicEncoder  # noqa: PLC0415

    info = optimize_encoder(BasicEncoder(64, output_dim=256, norm_fn="instance", downsample=2), keep_bf16=("convd",))
    assert info["fused_blocks"] == 6, info
    assert info["instance_norm"] > 0, info


_EXACT_TYPE = re.compile(
    r"(?<![\w.`])type\([^()]*(?:\([^()]*\))?[^()]*\)\s*(?:is|in|==|!=|not in)\b|\.get\(type\(|\[type\("
)


def test_no_exact_type_checks_on_modules():
    """`type(m) is X` is False for an overridden module: use `base_type(m)`. Lines that compare non-modules say so with
    `# exact type: not a module`."""
    src = Path(fastforks.__file__).parent
    bad = [
        f"{p.relative_to(src)}:{i}: {line.strip()}"
        for p in sorted(src.rglob("*.py"))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if _EXACT_TYPE.search(line) and "# exact type: not a module" not in line
    ]
    assert not bad, "\n".join(bad)


def test_no_bound_method_overrides():
    """`m.forward = fn.__get__(m)` / `MethodType(fn, m)` keep `del model` from freeing the weights: use `override`."""
    src = Path(fastforks.__file__).parent
    pattern = re.compile(r"\.__get__\(|\bMethodType\(")
    bad = [
        f"{p.relative_to(src)}:{i}: {line.strip()}"
        for p in sorted(src.rglob("*.py"))
        if p.relative_to(src).as_posix() != "_core/patch.py"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not bad, "\n".join(bad)


def test_pickle_and_deepcopy_keep_the_override(tmp_path):
    """`torch.save(model.module())` pickles by class name: the subclass pickles as its base class plus the overrides."""
    m = Net()
    override(m, _double)
    override(m.conv, _conv_half)
    x = torch.rand(1, 3, 8, 8)
    torch.save(m, tmp_path / "m.pt")
    for c in (torch.load(tmp_path / "m.pt", weights_only=False), copy.deepcopy(m)):
        assert type(c) is type(m)
        assert type(c.conv) is type(m.conv)
        torch.testing.assert_close(c(x), m(x))
