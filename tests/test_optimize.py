"""CPU tests of `fastforks._optimize.optimize`: mode validation, family detection, portable on CPU, max refusals."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import fx, nn

from fastforks._core.deferred_init import deferred_init
from fastforks._core.errors import BackendUnavailableError, OptionsError
from fastforks._optimize import COVERED, FastModel, _family, cnn, fold, optimize, patch


class _Tiny(nn.Module):
    """Conv + BN + ReLU + pooled linear head (an unknown architecture)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.fc = nn.Linear(8, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.act(self.bn(self.conv(x))).mean((2, 3)))


class _TinyHF(_Tiny):
    """HF-style signature: the image batch is the `pixel_values` keyword."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return super().forward(pixel_values)


def _tiny(cls: type[_Tiny] = _Tiny) -> _Tiny:
    torch.manual_seed(0)
    m = cls().eval()
    with torch.no_grad():  # non-trivial running stats, so folding changes the weights
        m.bn.running_mean = torch.linspace(-0.5, 0.5, 8)
        m.bn.running_var = torch.linspace(0.5, 2.0, 8)
        m.bn.weight.uniform_(0.5, 1.5)
        m.bn.bias.uniform_(-0.2, 0.2)
    return m


def test_mode_validated():
    with pytest.raises(OptionsError, match="mode"):
        optimize(_tiny(), mode="fast")


def test_portable_cpu_folds_bn_and_matches():
    m = _tiny()
    ref = copy.deepcopy(m)
    x = torch.randn(2, 3, 16, 16)
    fast = optimize(m, mode="portable")
    assert isinstance(fast, FastModel)
    assert fast.report["family"] == "generic"
    assert fast.report["bn_folded"] == 1
    assert fast.report["fp16"] is False
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), rtol=1e-4, atol=1e-5)
    assert not any(isinstance(mod, nn.BatchNorm2d) for mod in fast.net.modules())


def test_pixel_values_keyword_and_attribute_fallback():
    m = _tiny(_TinyHF)
    m.config = {"name": "tiny"}  # HF-style attribute on the user's model
    fast = optimize(m, mode="portable")
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(fast(pixel_values=x), fast(x))
    assert fast.config == {"name": "tiny"}
    assert fast.original is m


def test_kernels_needs_cuda():
    with pytest.raises(BackendUnavailableError, match="CUDA"):
        optimize(_tiny(), mode="kernels", device="cpu")


def test_refused_max_leaves_the_model_alone():
    m = _tiny().train()
    with pytest.raises(BackendUnavailableError, match="CUDA"):
        optimize(m, mode="kernels", device="cpu")
    assert m.training


def test_forward_without_input():
    fast = optimize(_tiny(), mode="portable", device="cpu")
    with pytest.raises(TypeError, match="image batch"):
        fast()


def test_cudnn_hacc_geometry():
    from fastforks.kernels import cudnn_hacc  # noqa: PLC0415

    x, w = torch.zeros(2, 8, 13, 17), torch.zeros(16, 8, 3, 3)
    for pad, stride, dil in (((1, 1), (1, 1), (1, 1)), ((2, 4), (1, 2), (2, 4)), ((0, 0), (2, 1), (3, 2))):
        want = torch.nn.functional.conv2d(x, w, None, stride, pad, dil).shape
        assert cudnn_hacc._out_shape(x, w, pad, stride, dgrad=False, dilation=dil) == list(want)
    with pytest.raises(TypeError, match="fp16 only"):
        cudnn_hacc._check_inputs(x, w.half())


@pytest.mark.gpu
def test_kernels_rejects_unknown_architecture():
    if not torch.cuda.is_available():
        pytest.skip("the architecture check runs after the device check")
    with pytest.raises(BackendUnavailableError, match="does not cover"):
        optimize(_tiny(), mode="kernels", device="cuda")


@pytest.mark.parametrize(
    ("names", "family"),
    [
        ({"ResNet", "Bottleneck"}, "cnn"),
        ({"ConvNeXt", "ConvNeXtBlock"}, "convnext"),
        ({"SwinTransformer", "SwinTransformerStage"}, "swin"),
        ({"VisionTransformer", "Block"}, "vit"),
        ({"Eva", "EvaBlock"}, "vit"),
        ({"SiglipVisionTransformer", "SiglipEncoderLayer"}, "vit"),
        ({"Sequential", "Linear"}, None),
    ],
)
def test_family(names, family):
    assert _family(names) == family


def test_covered_lists_every_family():
    assert set(COVERED) == {"cnn", "convnext", "swin", "vit"}


def test_torchvision_resnet_portable_cpu():
    tv = pytest.importorskip("torchvision")
    m = tv.models.resnet18(weights=None, num_classes=10).eval()
    ref = copy.deepcopy(m)
    x = torch.randn(1, 3, 64, 64)
    fast = optimize(m, mode="portable")
    assert fast.report["family"] == "cnn"
    assert fast.report["bn_folded"] == 20  # every BN of ResNet-18 follows a conv
    assert fast.report["fx"] == 1
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), rtol=1e-3, atol=1e-4)


def test_timm_vit_portable_cpu():
    timm = pytest.importorskip("timm")
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10, img_size=32).eval()
    ref = copy.deepcopy(m)
    x = torch.randn(2, 3, 32, 32)
    fast = optimize(m, mode="portable")
    assert fast.report["family"] == "vit"
    assert fast.report["vit_blocks"] == len(m.blocks)
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), rtol=1e-3, atol=1e-4)


@pytest.mark.slow  # ~1.5 s (timm build + fx trace)
def test_timm_repvit_fused_inside_deferred_init():
    timm = pytest.importorskip("timm")

    def build() -> torch.nn.Module:
        return timm.models.repvit.RepVit(embed_dim=(8, 16), depth=(1, 1), num_classes=10).eval()

    ref = build()
    for m in ref.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.weight.data.uniform_(0.5, 2)  # timm starts some at 0; trained ones are not
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2)
    with deferred_init():  # as `load` builds a network; timm's `fuse()` writes the fused weights through `.data`
        net = build()
        net.load_state_dict(ref.state_dict())
        fast = optimize(net, mode="portable")
    assert fast.report["family"] == "cnn"
    assert fast.report["reparam"] == 1
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in fast.modules())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), rtol=1e-3, atol=1e-4)


def test_patch_public_and_auto_falls_back_to_portable(caplog: pytest.LogCaptureFixture):
    import fastforks  # noqa: PLC0415

    m = _tiny()
    ref = copy.deepcopy(m)
    with caplog.at_level("INFO", logger="fastforks"):
        fast = fastforks.patch(m, device="cpu")
    assert fastforks.patch is patch
    assert isinstance(fast, fastforks.FastModel)
    assert fast.mode == fast.backend.mode == "portable"
    assert fast.backend.device == "cpu"
    assert fast.backend.dtype == "float32"
    assert fast.backend.kernels == ()
    assert "using portable" in caplog.text
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), rtol=1e-4, atol=1e-5)


def test_patch_explicit_max_raises_and_options_validated():
    with pytest.raises(BackendUnavailableError, match="CUDA"):
        patch(_tiny(), mode="kernels", device="cpu")
    with pytest.raises(OptionsError, match="dtype must be None or 'float32'"):
        patch(_tiny(), dtype="float16")
    with pytest.raises(OptionsError, match="mode"):
        patch(_tiny(), mode="fast")
    with pytest.raises(OptionsError, match="dtype must be None"):
        patch(_tiny(), mode="kernels", dtype="float32")
    fast = patch(_tiny(), mode="portable", device="cpu", dtype="float32")
    assert fast.backend.mode == "portable"


@pytest.mark.gpu
def test_patch_auto_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    tv = pytest.importorskip("torchvision")
    fast = patch(_tiny(), device="cuda")  # not covered: portable, no raise
    assert fast.backend.mode == "portable"
    assert fast.backend.dtype == "float16"
    m = tv.models.resnet18(weights=None, num_classes=10).eval()
    ref = copy.deepcopy(m).cuda()
    fast = patch(m, device="cuda")
    assert fast.backend.mode == "kernels"
    assert fast.backend.kernels  # the kernel libraries it loaded
    x = torch.randn(2, 3, 64, 64, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(fast(x).float(), ref(x), rtol=2e-2, atol=2e-2)


def test_cpu_conv_net_runs_channels_last_with_identical_output():
    """CPU: a convolution net gets channels_last weights once at load (oneDNN is faster); outputs match NCHW."""
    m = _tiny()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        want = copy.deepcopy(m)(x)
        fast = optimize(m, mode="portable", device="cpu")
        got = fast(x)
    assert fast.report["channels_last"] is True
    convs = [c for c in fast.net.modules() if isinstance(c, nn.Conv2d)]
    assert convs
    assert all(c.weight.is_contiguous(memory_format=torch.channels_last) for c in convs)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_cpu_net_without_batchnorm_keeps_contiguous_weights():
    """A net without the conv-like family (no BatchNorm2d) is not converted."""
    m = nn.Sequential(nn.Flatten(), nn.Linear(12, 4)).eval()
    fast = optimize(m, mode="portable", device="cpu")
    assert fast.report["channels_last"] is False


def test_cpu_dense_output_is_contiguous_nchw():
    """A conv net with a 4-D output (segmentation style) returns a plain NCHW tensor, so `.view(-1)` works."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Conv2d(4, 2, 1)).eval()
    fast = optimize(m, mode="portable", device="cpu")
    y = fast(torch.randn(1, 3, 8, 8))
    assert y.is_contiguous()
    assert y.view(-1).numel() == 2 * 8 * 8


def test_bn_without_running_stats_is_not_folded():
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4, track_running_stats=False)).eval()
    ref = copy.deepcopy(m)
    x = torch.randn(2, 3, 8, 8)
    fast = patch(m, mode="portable")
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x))


def test_patch_of_a_fast_model_is_itself_or_refused():
    """R2-7: a `FastModel` is returned as it is when the settings agree, never wrapped twice."""
    fast = patch(_tiny(), device="cpu")
    assert patch(fast) is fast
    assert patch(fast, mode="portable", device="cpu", dtype="float32") is fast
    with pytest.raises(OptionsError, match="already patched"):
        patch(fast, mode="kernels")
    with pytest.raises(OptionsError, match="already patched"):
        patch(fast, dtype="float32", device="meta")


def test_patch_device_resolved_like_load(monkeypatch: pytest.MonkeyPatch):
    """R2-8: `device=` goes through `resolve_device`: unusable or invalid devices are an `OptionsError`."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(OptionsError, match="no usable CUDA GPU"):
        patch(_tiny(), device="cuda")
    with pytest.raises(OptionsError, match="invalid device"):
        patch(_tiny(), device="gpu0")


# --- MIRA-R3-01: no fold into a conv that has a second use --------------------------------------------------------


class _Tied(nn.Module):
    """conv -> bn -> relu, plus `skip`: the same conv module (shared) or a copy of it, or its weight read directly."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.skip = self.conv if kind == "shared" else copy.deepcopy(self.conv)
        self.fc = nn.Linear(4, 3)
        with torch.no_grad():
            self.bn.running_mean.fill_(0.7)
            self.bn.running_var.fill_(1.8)
            self.bn.weight.fill_(1.3)

    def forward(self, x):
        if self.kind == "siamese":  # the same conv -> bn twice: one fold per call would fold twice
            return self.fc((self.bn(self.conv(x)) + self.bn(self.conv(x.flip(3)))).mean((2, 3)))
        s = F.conv2d(x, self.conv.weight, padding=1) if self.kind == "weight_read" else self.skip(x)
        return self.fc((self.bn(self.conv(x)).relu() + s).mean((2, 3)))


@pytest.mark.parametrize("kind", ["untied", "shared", "siamese", "weight_read"])
def test_bn_not_fused_into_a_conv_with_a_second_use(kind):
    torch.manual_seed(309)
    m = _Tied(kind).eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.inference_mode():
        ref = m(x)
        fast = patch(copy.deepcopy(m), mode="portable", device="cpu")
        assert (fast(x) - ref).abs().max() < 1e-5
        gm = fold._fuse(fx.symbolic_trace(copy.deepcopy(m)))  # the fx step alone: fuses exactly the single-use pair
        assert any(isinstance(b, nn.BatchNorm2d) for b in gm.modules()) == (kind != "untied")
        assert (gm(x) - ref).abs().max() < 1e-5


class _ScaledConv(nn.Module):
    """conv * gamma (LayerScale) [+ a direct read of the conv weight]."""

    def __init__(self, *, second_use: bool) -> None:
        super().__init__()
        self.read = second_use
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.gamma = nn.Parameter(torch.rand(8, 1, 1) + 0.5)

    def forward(self, x):
        h = self.conv(x) * self.gamma
        return h + F.conv2d(x, self.conv.weight, padding=1) if self.read else h


class _ConvPair(nn.Module):
    """1x1 conv (4 channels out) -> relu -> conv: a second conv module, or the same one again."""

    def __init__(self, *, second_use: bool) -> None:
        super().__init__()
        self.a = nn.Conv2d(4, 4, 1)
        self.b = self.a if second_use else nn.Conv2d(4, 4, 1)

    def forward(self, x):
        return self.b(self.a(x).relu())


class _Stem(nn.Module):
    """3-channel stem conv, run a second time on 3 channels of its own output when shared."""

    def __init__(self, *, second_use: bool) -> None:
        super().__init__()
        self.shared = second_use
        self.stem = nn.Conv2d(3, 8, 3, padding=1).half()

    def forward(self, x):
        h = self.stem(x)
        return h + self.stem(h[:, :3]) if self.shared else h


@pytest.mark.parametrize("second_use", [False, True])
def test_kernel_passes_leave_a_conv_with_a_second_use_alone(second_use):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 8, 8)
    for mk, run in (
        (_ScaledConv, cnn.fold_layer_scale),
        (_ConvPair, cnn.pad_conv_pairs),
    ):
        m = mk(second_use=second_use).eval()
        gm = fx.symbolic_trace(copy.deepcopy(m))
        with torch.inference_mode():
            assert run(gm) == int(not second_use)
            gm.recompile()
            xi = x[:, : 8 if mk is _ScaledConv else 4]
            assert (gm(xi) - m(xi)).abs().max() < 1e-5
    gm = fx.symbolic_trace(_Stem(second_use=second_use))
    assert cnn.pad_stem(gm) is not second_use
    assert gm.stem.weight.shape[1] == (3 if second_use else 8)


class _DeviceCache(nn.Module):
    """Conv + BN whose forward caches a derived tensor per device, as timm's attention bias tables do."""

    def __init__(self) -> None:
        super().__init__()
        self.conv, self.bn = nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)
        self.bias = nn.Parameter(torch.randn(4, 1, 1))
        self.cache: dict[str, torch.Tensor] = {}

    def forward(self, x):
        key = str(x.device)
        if key not in self.cache:
            self.cache[key] = self.bias * 2
        return self.bn(self.conv(x)) + self.cache[key]


def test_fold_leaves_no_proxy_in_a_forward_cache():
    """Each fold pass traces the model; a Proxy cached by one trace must not reach the next one (EfficientFormer)."""
    torch.manual_seed(0)
    m = _DeviceCache().eval()
    x = torch.randn(1, 3, 4, 4)
    with torch.inference_mode():
        want = copy.deepcopy(m)(x)
        net, _ = fold.fold_bn(m)
        assert (net(x) - want).abs().max() < 1e-5
    assert not any(isinstance(v, fx.Proxy) for mod in m.modules() for v in getattr(mod, "cache", {}).values())
