"""Correctness of the generic patch() path (external review 2026-11-04, batch A): kernel eligibility, thread safety
of shared state, stale transformed-weight caches, structural BN folding, exact ViT matching, dtype handling."""

from __future__ import annotations

import copy
import gc
import threading

import pytest
import torch
from torch import nn

from fastforks._core.deferred_init import float32_default
from fastforks._optimize import conv_ops, ops, patch

timm = pytest.importorskip("timm")
CL = torch.channels_last


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).abs().max() / b.abs().max())


def _convnext():
    return timm.models.convnext.ConvNeXt(
        depths=(1, 1, 1, 1), dims=(32, 64, 128, 256), num_classes=10, ls_init_value=0.5
    )


def _vit(**kw):
    kw = {"img_size": 32, "patch_size": 8, "embed_dim": 64, "depth": 2, "num_heads": 4, "num_classes": 5} | kw
    return timm.models.vision_transformer.VisionTransformer(**kw)


def _two(mk):
    """Two differently seeded instances (eval) of one architecture."""
    torch.manual_seed(0)
    a = mk().eval()
    torch.manual_seed(1)
    return a, mk().eval()


# --- finding 3: the stem_c8 packed-weight cache --------------------------------------------------------------------


@pytest.mark.slow
def test_stem_c8_cache_hits_follows_writes_and_drops_dead_weights():
    w = torch.randn(16, 8, 3, 3)
    a = conv_ops._stem_c8_weight(w)
    assert conv_ops._stem_c8_weight(w) is a  # a hit (WeakKeyDictionary raised "Boolean value of Tensor" here)
    w.mul_(2)  # in-place write: bumps the version
    b = conv_ops._stem_c8_weight(w)
    assert b is not a
    assert torch.equal(b, 2 * a)
    w.data = torch.randn(16, 8, 3, 3)  # new storage
    assert not torch.equal(conv_ops._stem_c8_weight(w), b)
    n = len(conv_ops._STEM_C8_W)
    del w
    gc.collect()
    assert len(conv_ops._STEM_C8_W) == n - 1


# --- finding 4: transformed-weight caches follow load_state_dict ------------------------------------------------


def test_convnext_portable_follows_load_state_dict():
    m, other = _two(_convnext)
    x = torch.randn(2, 3, 64, 64)
    fast = patch(m, device="cpu")
    assert fast.report["convnext_blocks"] > 0
    m.load_state_dict(other.state_dict())
    assert _rel(fast(x), other(x)) < 1e-5


# --- finding 5: BN folding only for structurally proven conv -> bn pairs -----------------------------------------


class _Branchy(nn.Module):
    """conv -> (input-size dependent affine) -> bn: one probe at the small size would see conv -> bn."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        h = self.conv(x)
        if x.shape[-1] > 64:
            h = h * 2.0 + 1.0
        return self.fc(self.bn(h).relu().mean((2, 3)))


def _random_bn(m: nn.Module) -> nn.Module:
    with torch.no_grad():
        for b in m.modules():
            if isinstance(b, nn.BatchNorm2d):
                b.running_mean.uniform_(-1, 1)
                b.running_var.uniform_(0.5, 2)
                if b.affine:
                    b.weight.uniform_(0.5, 1.5)
    return m.eval()


def test_bn_not_folded_across_a_data_dependent_branch():
    m = _random_bn(_Branchy())
    ref = copy.deepcopy(m)
    fast = patch(m, device="cpu")
    assert fast.report["bn_folded"] == 0
    for w in (64, 96):
        x = torch.randn(1, 3, w, w)
        assert _rel(fast(x), ref(x)) < 1e-5


def test_bn_folded_in_a_sequential_and_a_traced_graph():
    seq = _random_bn(
        nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8))
    )
    res = _random_bn(timm.create_model("resnet10t", pretrained=False, num_classes=5))
    for m in (seq, res):
        ref = copy.deepcopy(m)
        fast = patch(m, device="cpu")
        n_bn = sum(isinstance(b, nn.BatchNorm2d) for b in ref.modules())
        assert fast.report["bn_folded"] == n_bn
        x = torch.randn(1, 3, 64, 64)
        assert _rel(fast(x), ref(x)) < 1e-4


# --- finding 6: only exactly equivalent ViT blocks match ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "blocks"),
    [
        ({}, 2),
        ({"mlp_layer": "SwiGLUPacked", "mlp_ratio": 8.0, "act_layer": nn.SiLU}, 0),  # crashed (packed fc1)
        ({"attn_layer": "DiffAttention"}, 0),  # silently replaced by plain attention
        ({"qk_norm": True}, 0),
    ],
)
def test_vit_matches_only_exact_blocks(kw, blocks):
    for k in ("mlp_layer", "attn_layer"):
        if k in kw:
            kw = kw | {k: getattr(timm.layers, kw[k])}
    m = _vit(**kw).eval()
    ref = copy.deepcopy(m)
    fast = patch(m, device="cpu")
    assert fast.report.get("vit_blocks", 0) == blocks
    x = torch.randn(2, 3, 32, 32)
    assert _rel(fast(x), ref(x)) < 1e-4


def test_vit_with_an_instance_forward_is_left_alone():
    m = _vit().eval()
    blk = m.blocks[0]
    blk.forward = lambda x, **_: x + blk.attn(blk.norm1(x))  # an attention-only block
    ref_y = m(torch.zeros(1, 3, 32, 32))
    fast = patch(m, device="cpu")
    assert fast.report.get("vit_blocks", 0) == 1
    assert _rel(fast(torch.zeros(1, 3, 32, 32)), ref_y) < 1e-4


# --- findings 8 + 9: dtypes ----------------------------------------------------------------------------------------


def test_patch_float32_converts_half_and_double_params():
    for dt in (torch.float64, torch.float16):
        m = timm.create_model("resnet10t", pretrained=False, num_classes=5).eval().to(dt)
        fast = patch(m, device="cpu", dtype="float32")
        assert {p.dtype for p in fast.net.parameters()} == {torch.float32}
        assert fast(torch.randn(1, 3, 64, 64)).dtype == torch.float32


def test_fastmodel_rejects_conversions_that_break_its_backend():
    fast = patch(timm.create_model("resnet10t", pretrained=False, num_classes=5), device="cpu")
    for f in (
        fast.half,
        fast.double,
        lambda: fast.to(torch.float64),
        fast.train,
        lambda: fast.load_state_dict(fast.state_dict(), assign=True),
    ):
        with pytest.raises(RuntimeError, match="FastModel"):
            f()
    fast.float()
    fast.cpu()
    fast.eval()
    fast.train(mode=False)
    fast.load_state_dict(fast.state_dict())
    assert {p.dtype for p in fast.net.parameters()} == {torch.float32}
    assert not fast.training


def test_patch_ignores_a_float64_default_dtype():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        fast = patch(timm.create_model("resnet10t", pretrained=False, num_classes=5).eval(), device="cpu")
        assert torch.get_default_dtype() == torch.float64  # restored
        assert {p.dtype for p in fast.net.parameters()} == {torch.float32}
        assert fast(torch.randn(1, 3, 64, 64, dtype=torch.float32)).dtype == torch.float32
    finally:
        torch.set_default_dtype(old)


def test_float32_default_is_refcounted_across_threads():
    old = torch.get_default_dtype()
    entered, a_out = threading.Barrier(2), threading.Event()
    seen: dict[str, torch.dtype] = {}

    def a() -> None:
        with float32_default():
            entered.wait()
        a_out.set()

    def b() -> None:
        with float32_default():
            entered.wait()
            a_out.wait()
            seen["b"] = torch.get_default_dtype()

    torch.set_default_dtype(torch.float64)
    try:
        threads = [threading.Thread(target=f) for f in (a, b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == {"b": torch.float32}
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(old)


# --- GPU: findings 1-4 on the kernels ------------------------------------------------------------------------------


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


@pytest.mark.gpu
def test_xdw_rejects_more_than_8_taps():
    dev = _cuda()
    x = torch.randn(1, 16, 32, 32, device=dev).half().contiguous(memory_format=CL)
    w1, b1 = torch.randn(32, 16, device=dev).half(), torch.zeros(32, device=dev).half()
    for k, ok in ((7, True), (9, False)):
        wd, bd = torch.randn(k, k, 32, device=dev).half() * 0.2, torch.zeros(32, device=dev).half()
        if ok:
            ops.xdw(x, w1, b1, wd, bd, 32, (1, 1), (k // 2, k // 2), (0, 0))
        else:
            with pytest.raises(ValueError, match="xdw"):  # the kernel's tap loops stop at 8
                ops.xdw(x, w1, b1, wd, bd, 32, (1, 1), (k // 2, k // 2), (0, 0))


@pytest.mark.gpu
def test_mobilenet_with_9x9_depthwise_is_not_fused_into_xdw():
    dev = _cuda()
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    m = torchvision.models.mobilenet_v2(num_classes=10).eval()
    for i in (2, 3):
        dw = m.features[i].conv[1][0]
        m.features[i].conv[1][0] = nn.Conv2d(
            dw.in_channels, dw.out_channels, 9, dw.stride, 4, groups=dw.groups, bias=False
        )
    with torch.no_grad():  # non-trivial BN statistics: the default init gives near-zero logits (fp16 noise only)
        for b in m.modules():
            if isinstance(b, nn.BatchNorm2d):
                b.running_mean.uniform_(-0.1, 0.1)
                b.running_var.uniform_(0.5, 1.5)
    x = torch.randn(2, 3, 96, 96, device=dev)
    ref = copy.deepcopy(m).to(dev)(x)
    fast = patch(m.to(dev), mode="kernels")
    assert fast.report["xdw"] == 8  # 10 expand -> depthwise pairs, the two 9x9 ones stay unfused
    assert _rel(fast(x), ref) < 2e-2


@pytest.mark.gpu
def test_pw_bias_is_per_call_under_threads():
    dev = _cuda()
    torch.manual_seed(0)
    nt, it = 4, 2000
    x = torch.randn(1, 64, 16, 16, device=dev).half().contiguous(memory_format=CL)
    w = (torch.randn(256, 64, 1, 1, device=dev) * 0.1).half().contiguous(memory_format=CL)
    biases = [torch.full((256,), float(i * 100), device=dev).half() for i in range(nt)]
    refs = [ops.pw(x, w, b, None, 0, 1)[0, 0, 0, 0].item() for b in biases]
    outs: list[list[torch.Tensor]] = [[] for _ in range(nt)]
    start = threading.Barrier(nt)

    def worker(i: int) -> None:
        start.wait()
        for _ in range(it):
            outs[i].append(ops.pw(x, w, biases[i], None, 0, 1)[0, 0, 0, 0])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(nt)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    bad = [int((torch.stack(outs[i]).float() != refs[i]).sum()) for i in range(nt)]
    assert bad == [0] * nt  # unlocked: another thread's bias pointer on the shared cuBLASLt plan


@pytest.mark.gpu
def test_stem_c8_conv_repeated_and_after_a_weight_write():
    dev = _cuda()
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32, 32, device=dev).half().contiguous(memory_format=CL)
    w = (torch.randn(16, 8, 3, 3, device=dev) * 0.1).half().contiguous(memory_format=CL)
    b = torch.zeros(16, device=dev).half()
    ref = torch.nn.functional.conv2d(x.float(), w.float(), b.float(), padding=1)
    for scale in (1, 1, 2):
        if scale == 2:
            with torch.no_grad():
                w.mul_(2)
        y = ops.conv_fused(x, w, b, None, (1, 1), (1, 1), 0)
        assert _rel(y, scale * ref) < 1e-2


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mk",
    [
        _convnext,
        lambda: _vit(img_size=64, patch_size=16, embed_dim=128, num_heads=2, num_classes=10, init_values=0.5),
    ],
    ids=["convnext", "vit"],
)
def test_kernels_follow_load_state_dict(mk):
    dev = _cuda()
    m, other = _two(mk)
    x = torch.randn(2, 3, 64, 64, device=dev)
    ref = other.to(dev)(x)
    fast = patch(m.to(dev), mode="kernels")
    fast.load_state_dict({f"net.{k}": v for k, v in other.state_dict().items()})
    assert _rel(fast(x), ref) < 1e-2


# --- Mac verify round 2 (B17): patch of an independent copy, portable, against the original -------------------------


class _DirectConv(nn.Module):
    """Untraceable: a large input calls features[0] (the conv) without its BN, a small one the Sequential (B17-1)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))  # random stats: `_random_bn`

    def forward(self, x):
        f = self.features[0] if x.shape[-1] > 100 else self.features
        return f(x).mean((2, 3))


class _ViewHead(nn.Module):
    """conv -> bn -> `view` (B17-2: channels-last activations)."""

    def __init__(self) -> None:
        super().__init__()
        self.c, self.bn, self.fc = nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Linear(4 * 16 * 16, 2)

    def forward(self, x):
        return self.fc(self.bn(self.c(x)).view(x.shape[0], -1))


def _hooked_root():
    m = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    m.register_forward_hook(lambda _m, _a, o: o + 3)
    return m


def _hooked_conv():
    m = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    m[0].register_forward_pre_hook(lambda _m, a: (a[0] * 2,))
    m[0].register_forward_hook(lambda _m, _a, o: o - 1)
    return m


def _hooked_block():
    m = _convnext()
    m.stages[0].blocks[0].mlp.fc2.register_forward_hook(lambda _m, _a, o: o * 2)
    return m


@pytest.mark.parametrize(
    ("mk", "sizes"),
    [
        (_DirectConv, (64, 128)),  # B17-1
        (_hooked_root, (16,)),  # B17-4
        (_hooked_conv, (16,)),
        (_hooked_block, (64,)),
        (_ViewHead, (16,)),  # B17-2
        (lambda: nn.Sequential(nn.BatchNorm2d(3, affine=False), nn.AdaptiveAvgPool2d(1), nn.Flatten()), (16,)),  # B17-3
    ],
    ids=["conv-reuse-untraced", "hook-root", "hook-conv", "hook-in-block", "view", "buffers-only"],
)
def test_patch_matches_the_original(mk, sizes):
    m = _random_bn(mk())
    ref = copy.deepcopy(m)
    fast = patch(m, mode="portable", device="cpu")
    for s in sizes:
        x = torch.randn(1, 3, s, s)
        assert _rel(fast(x), ref(x)) < 1e-5


def test_convnext_portable_weight_grads():  # B17-5: grads through the parameters, not the folded copies
    torch.manual_seed(0)
    blk = timm.models.convnext.ConvNeXtBlock(8, 8, ls_init_value=0.5).eval()
    ref = copy.deepcopy(blk)
    fast = patch(blk, mode="portable", device="cpu")
    assert fast.report["convnext_blocks"] == 1
    x = torch.randn(2, 8, 6, 6)
    fast(x).square().sum().backward()
    ref(x).square().sum().backward()
    for (k, p), q in zip(fast.net.named_parameters(), ref.parameters(), strict=True):
        assert p.grad is not None, k
        assert _rel(p.grad, q.grad) < 1e-4, k


def test_transformers_capture_hooks_keep_the_rewrites():  # recorder-only hooks, inactive unless outputs are captured
    oc = pytest.importorskip("transformers.utils.output_capturing")
    torch.manual_seed(0)
    blk = timm.models.convnext.ConvNeXtBlock(8, 8, ls_init_value=0.5).eval()
    oc.install_output_capuring_hook(blk.mlp.fc2, "hidden_states", 0)
    ref = copy.deepcopy(blk)
    fast = patch(blk, mode="portable", device="cpu")
    assert fast.report["convnext_blocks"] == 1
    x = torch.randn(2, 8, 6, 6)
    assert _rel(fast(x), ref(x)) < 1e-5


# --- review 2026-11-05: optimizer findings OPT-1..5 -----------------------------------------------------------------


def _same_out(m: nn.Module, x: torch.Tensor) -> None:
    m = m.eval()
    ref = copy.deepcopy(m)
    fast = patch(m, mode="portable")
    a, b = fast(x), ref(x)
    for u, v in zip(a, b, strict=True) if isinstance(b, tuple) else [(a, b)]:
        torch.testing.assert_close(u, v, rtol=1e-4, atol=1e-4)


def test_bn_not_folded_into_a_conv_with_a_shared_weight():  # OPT-1
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = (nn.Conv2d(3, 3, 1, bias=False) for _ in range(2))
            self.b.weight = self.a.weight
            self.bn = nn.BatchNorm2d(3)
            self.a.weight.data.fill_(1)
            self.bn.weight.data.fill_(2.0)

        def forward(self, x):
            return self.bn(self.a(x)) + self.b(x) * (x.shape[-1] - 64)

    _same_out(M(), torch.ones(1, 3, 32, 32))


def test_linear_then_batchnorm1d_on_a_sequence_is_not_fused():  # OPT-3
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.fc, self.bn = nn.BatchNorm2d(4), nn.Linear(4, 4, bias=False), nn.BatchNorm1d(4)
            self.fc.weight.data.copy_(torch.eye(4))
            self.bn.weight.data.copy_(torch.tensor([1.0, 2, 3, 4]))

        def forward(self, x):
            return self.bn(self.fc(self.pre(x).flatten(start_dim=2)))

    _same_out(M(), torch.ones(2, 4, 2, 2))


@pytest.mark.parametrize("kind", ["forward", "pre", "backward"])
def test_hooked_identity_is_kept(kind):  # OPT-2
    m = nn.Sequential(nn.BatchNorm2d(3), nn.Identity()).eval()
    if kind == "forward":
        m[1].register_forward_hook(lambda _m, _a, out: out * 2)
    elif kind == "pre":
        m[1].register_forward_pre_hook(lambda _m, a: (a[0] * 2,))
    else:
        m[1].register_full_backward_hook(lambda _m, gi, _go: (gi[0] * 2,))
    ref = copy.deepcopy(m)
    fast = patch(m, mode="portable")
    x = torch.ones(1, 3, 4, 4, requires_grad=True)
    ga, gb = (torch.autograd.grad(n(x).sum(), x)[0] for n in (fast, ref))
    torch.testing.assert_close(fast(x), ref(x))
    torch.testing.assert_close(ga, gb)


def test_view_with_a_dtype_keyword_is_kept():  # OPT-4
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(x).contiguous().view(dtype=torch.int32)

    _same_out(M(), torch.ones(1, 3, 4, 4))


def test_torch_return_types_survive_patch():  # OPT-5
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(x).max(dim=1)

    m = M().eval()
    y = patch(m, mode="portable")(torch.randn(1, 3, 4, 4))
    assert type(y) is type(m(torch.randn(1, 3, 4, 4)))


H = torch.float16


@pytest.mark.parametrize(
    ("op", "bad"),
    [
        ("lin_res", "residual"),
        ("lin_bias", "bias"),
        ("conv_x", "x"),
        ("dw_bias", "bias"),
        ("dw_weight", "weight"),
    ],
)
def test_fp16_kernels_reject_other_dtypes(op, bad):
    """The fvm_conv ops read raw fp16 pointers: an fp32 operand must raise, not run on reinterpreted bits (a float32
    residual into `lin` gave NaN tracks in the TAPIR kernels runner)."""
    x, f = torch.zeros(1, 8, 4, 1, dtype=H), torch.float32
    w4, b, res = torch.zeros(16, 8, 1, 1, dtype=H), torch.zeros(16, dtype=H), torch.zeros(4, 16, dtype=H)
    wd, bd = torch.zeros(3, 1, 8, dtype=H), torch.zeros(8, dtype=H)
    calls = {
        "lin_res": lambda: conv_ops.lin(x[0, :, :, 0].T, w4, b, res=res.to(f)),
        "lin_bias": lambda: conv_ops.lin(x[0, :, :, 0].T, w4, b.to(f)),
        "conv_x": lambda: conv_ops.conv_fused(x.to(f), w4, b, None, [0, 0], [1, 1], 0),
        "dw_bias": lambda: conv_ops.dwconv(x, wd, bd.to(f), [1, 1], [1, 0], 0),
        "dw_weight": lambda: conv_ops.dwconv(x, wd.to(f), bd, [1, 1], [1, 0], 0),
    }
    with pytest.raises(ValueError, match=f"{bad} must be float16"):
        calls[op]()
