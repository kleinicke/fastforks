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
from fastforks.depth.models.depth_anything_3.net.dinov2.layers.rope import RotaryPositionEmbedding2D
from fastforks.shared.rope2d import rope_forward


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


@pytest.mark.slow
def test_no_reference_cycle(gc_off):
    m = Net()
    override(m, _double)
    override(m.conv, _conv_half)
    ref, ref_conv = weakref.ref(m), weakref.ref(m.conv)
    del m
    assert ref() is None  # freed by reference counting, without gc.collect()
    assert ref_conv() is None


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.gpu
@pytest.mark.parametrize(("ffn", "ratio"), [("mlp", 4), ("swiglu", 4), ("mlp", 64)])  # 64: FFN output with DEEP_K
def test_fast_vit_fused_blocks_match_cublas(ffn, ratio):
    """`FastViT(fused=...)`: calls of `FUSED_ROWS` tokens and more run the fused GEMMs, to the cuBLAS blocks' values."""
    import copy  # noqa: PLC0415

    from fastforks.kernels.dpt_fused import DEEP_K, DptFused  # noqa: PLC0415 - needs the GPU library
    from fastforks.shared.depth_anything_v2 import FastViT  # noqa: PLC0415 - needs the DINOv2 net
    from fastforks.shared.depth_anything_v2.fast_vit import FUSED_ROWS  # noqa: PLC0415
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    torch.manual_seed(0)
    vit = DinoVisionTransformer(
        img_size=28, patch_size=14, embed_dim=128, depth=3, num_heads=2, mlp_ratio=ratio, ffn_layer=ffn,
        init_values=0.5, block_chunks=0,
    )  # fmt: skip
    dev = torch.device("cuda")
    plain = FastViT(copy.deepcopy(vit), "ff", None, swiglu=True)
    kern, calls = DptFused(dev), []
    gemm = kern.gemm
    kern.gemm = lambda *a, **k: calls.append(1) or gemm(*a, **k)
    fused = FastViT(vit, "ff", None, swiglu=True, fused=kern)
    for f in (plain, fused):
        f.vit, f.pos_fp32 = f.vit.to(dev), f.pos_fp32.to(dev)
    with torch.inference_mode():
        small = torch.randn(1, 3, 14 * 8, 14 * 12, device=dev).half()  # 97 tokens: the cuBLAS blocks in both
        assert torch.equal(fused.get_intermediate_layers(small)[0], plain.get_intermediate_layers(small)[0])
        assert not calls
        big = torch.randn(1, 3, 14 * 23, 14 * 23, device=dev).half()
        assert FUSED_ROWS <= 23 * 23 + 1
        got, want = fused.get_intermediate_layers(big)[0].float(), plain.get_intermediate_layers(big)[0].float()
    assert len(calls) == 3 * (3 if ratio * 128 >= DEEP_K else 4)  # FFN outputs with DEEP_K inputs stay in cuBLAS
    # measured: bit-equal with the GELU FFN, 1.8e-5 with SwiGLU
    assert ((got - want).abs().mean() / want.abs().mean()).item() < 2.7e-5


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
    """`m.forward = fn.__get__(m)` / `MethodType(fn, m)` / `partial(fn, m)` keep `del model` from freeing the weights:
    use `override`."""
    src = Path(fastforks.__file__).parent
    pattern = re.compile(r"\.__get__\(|\bMethodType\(|\.forward = (functools\.)?partial\(")
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


def test_rope_forward_matches_upstream() -> None:
    rope = RotaryPositionEmbedding2D(frequency=100.0)
    tokens = torch.randn(2, 3, 12, 64, dtype=torch.float64)
    positions = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij"), -1).view(1, 12, 2) + 1
    positions = torch.cat([positions, torch.zeros_like(positions)])  # a patch grid, and all tokens at (0, 0)
    assert torch.allclose(rope_forward(rope, tokens, positions), rope(tokens, positions), rtol=0, atol=1e-12)
