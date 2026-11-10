# ruff: noqa: SLF001 - patches library modules in place
"""`fastforks.patch(model, mode=...)`: fast inference for classifiers built with timm, torchvision or HF transformers.

The model stays the user's model: the passes fold, cast and patch it in place (BatchNorm folded into convs, fp16
channels-last weights, block forwards replaced) and the returned `FastModel` wraps it: fp32 NCHW images in, fp32
outputs out, any batch and image size, nothing compiled or tuned per shape.

Modes (`mode="auto"`, the default: kernels where it applies, else portable; the choice is logged at INFO and recorded in
`FastModel.backend`):
    portable: plain PyTorch only (exportable to ONNX / TensorRT): BatchNorm folding (timm re-parameterization
        first), torch.fx flattening, fp16 + channels-last, SDPA attention, LayerScale folding, fp16-accumulate
        cuBLAS GEMMs (`torch.backends.cuda.matmul.allow_fp16_accumulation`). Any model; the passes that do not apply
        are skipped and `FastModel.report` lists what was applied. On CPU the model stays fp32.
    kernels: everything in portable plus the prebuilt fvm_* CUDA kernels (fused conv + bias + residual + activation,
        depthwise / grouped / stem / pooling kernels, fused transformer GEMMs) and fp16-accumulate cuDNN convs for
        large inputs. Only for the covered architectures (`COVERED`); anything else raises BackendUnavailableError.

Covered architectures (kernels; validated on the fast-vision-models zoo checkpoints):
    torchvision: AlexNet, VGG, Inception3, ResNet, DenseNet, SqueezeNet, MobileNetV2 / V3, ShuffleNetV2, EfficientNet
        (V1 / V2), RegNet.
    timm: ResNet, EfficientNet / MobileNetV3-class nets (MobileNet V1 / V4, EfficientNet), FastVit, EfficientVit (MIT),
        ConvNeXt (V1 / V2), Swin, VisionTransformer-style `Block`s (ViT, DeiT, DINOv2, SigLIP), Eva (EVA-02).
    transformers: SigLIP / SigLIP 2 vision towers (`SiglipEncoderLayer`).
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import fx, nn

from fastforks._core.deferred_init import float32_default
from fastforks._core.errors import BackendUnavailableError, OptionsError
from fastforks._core.half import fp16_accumulate  # re-exported: `fastforks._optimize.fp16_accumulate`
from fastforks._core.kernels import cuda_device
from fastforks._core.model import Backend, dtype_name, resolve_device

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["COVERED", "FastModel", "optimize", "patch"]

log = logging.getLogger(__name__)

MODES = ("portable", "kernels")
# big-input switch: from this many output pixels (batch * H * W) the kernels CNN path uses the fp16-accumulate cuDNN
# copy (no fused kernels) / the ShuffleNet standard-kernel net; smaller inputs are launch bound.
HACC_PIX = 8 * 224 * 224

_CNN = frozenset({
    # torchvision
    "AlexNet", "VGG", "Inception3", "ResNet", "DenseNet", "SqueezeNet", "MobileNetV2", "MobileNetV3", "ShuffleNetV2",
    "EfficientNet", "RegNet",
    # timm (ResNet / EfficientNet / MobileNetV3 shared above)
    "FastVit", "EfficientVit", "MobileNetV4",
})  # fmt: skip
COVERED: dict[str, tuple[str, ...]] = {
    "cnn": tuple(sorted(_CNN)),
    "convnext": ("ConvNeXtBlock",),
    "swin": ("SwinTransformerStage",),
    "vit": ("Block (timm, packed qkv)", "EvaBlock", "SiglipEncoderLayer"),
}


def _to_float(y: object) -> object:
    if isinstance(y, torch.Tensor):
        y = y.float() if y.is_floating_point() else y
        # CPU nets run channels_last; hand back plain NCHW maps (CUDA keeps its strides, no copy there)
        return y.contiguous() if y.dim() == 4 and not y.is_cuda else y  # noqa: PLR2004
    if isinstance(y, dict):  # also HF ModelOutput (an OrderedDict)
        for k in list(y.keys()):
            y[k] = _to_float(y[k])
        return y
    if isinstance(y, tuple):
        vals = [_to_float(v) for v in y]
        if hasattr(y, "_fields"):
            return y._make(vals)  # ty: ignore[unresolved-attribute] - namedtuple
        return tuple(vals) if type(y) is tuple else type(y)(vals)  # exact type: not a module (structseq)
    if isinstance(y, list):
        return [_to_float(v) for v in y]
    return y


class FastModel(nn.Module):
    """The optimized model: fp32 NCHW in, fp32 out (same output structure as the wrapped model).

    Attributes:
        net: The rewritten network (fp16 channels-last on CUDA).
        mode: `"portable"` or `"kernels"` (what ran; `backend.mode`).
        backend: What runs: mode, device, compute dtype and the kernel libraries (`name@build:target`) of `kernels`.
        report: The passes applied and their counts.
        original: The user's module (folded / cast in place; attribute lookups that miss on `FastModel` go there,
            e.g. `.config`, `.pretrained_cfg`).
    """

    def __init__(
        self, net: nn.Module, original: nn.Module, mode: str, report: dict[str, Any], big: nn.Module | None = None
    ) -> None:
        """Wrap `net`; `big` (sharing the weights) runs inputs with batch * H * W >= HACC_PIX."""
        super().__init__()
        self.net = net
        self.mode = mode
        self.report = report
        self._big = [big]  # a list: not registered (parameters are shared with net)
        self._orig = [original]
        self._half = bool(report.get("fp16"))
        self._acc = bool(report.get("fp16_accumulate"))
        self.eval()  # inference only: `module()` and export must not see training-mode branches
        if big is not None:
            big.eval()
        hacc = (report["cudnn_hacc"],) if "cudnn_hacc" in report else ()
        dtype = "float16" if self._half else "float32"
        self.backend = Backend(mode, report["device"], dtype, (*report.get("kernels", ()), *hacc))  # ty: ignore[invalid-argument-type]

    @property
    def original(self) -> nn.Module:
        """The user's module."""
        return self._orig[0]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - forwards to the wrapped model
        """Fall back to the user's module for attributes the wrapper does not have."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            orig = self.__dict__.get("_orig")
            if orig is None:
                raise
            return getattr(orig[0], name)

    def traceable(self) -> FastModel:
        """The same model (shared weights) without the fp16-accumulate switch, for `torch.compile` / `torch.export`.

        `allow_fp16_accumulation` is a process-global cuBLAS flag, which tracing cannot capture: set
        `torch.backends.cuda.matmul.allow_fp16_accumulation = True` around the compiled / exported call to keep it.
        """
        m = copy.copy(self)
        m.__dict__["_acc"] = False
        return m

    def _input(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        c = getattr(net, "stem_in_chans", None)  # a rewritten stem (kernels) reads any channel count: check as conv2d
        if c is not None and (x.dim() != 4 or x.shape[1] != c):  # noqa: PLR2004
            msg = f"{type(self.original).__name__}: expected input [B, {c}, H, W], got {list(x.shape)}"
            raise RuntimeError(msg)
        if getattr(net, "raw_input", False):  # the stem kernel reads the fp32 NCHW input itself
            return x.float().contiguous()
        if getattr(net, "pad8", False):
            from . import ops  # noqa: PLC0415

            return ops.input_nhwc8(x.float().contiguous())
        if x.device.type == "cpu":  # CPU stays NCHW float32
            return x.float()
        if not x.is_cuda:  # MPS: NCHW (channels-last is not faster there)
            return x.half() if self._half else x.float()
        mf = torch.channels_last if x.dim() == 4 else torch.contiguous_format  # noqa: PLR2004
        return x.to(torch.float16 if self._half else torch.float32, memory_format=mf)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> FastModel:  # noqa: FBT001, FBT002 - torch's signature
        """`.half()` / `.float()` / `.to()` / `.cuda()` / `.cpu()`: only what keeps the built dtype, device and layout.

        The passes fixed them (folded fp16 weights, kernels for one device, `backend` / `report`): another one would
        run wrong or report wrong, so it raises; patch the original model again for it.
        """
        dt = torch.float16 if self._half else torch.float32
        probe = torch.empty((1, 2, 2, 2), dtype=dt, device=torch.device(self.report["device"]))
        if self.report.get("channels_last"):
            probe = probe.contiguous(memory_format=torch.channels_last)
        out = fn(probe)
        if not isinstance(out, torch.Tensor) or (out.dtype, out.device, out.stride()) != (
            dt,
            probe.device,
            probe.stride(),
        ):
            msg = (
                f"FastModel runs {self.backend.dtype} on {self.backend.device} as built; it cannot change dtype, "
                "device or memory format. Patch the original model again for another one (`patch(model, "
                "device=..., dtype=...)`)."
            )
            raise RuntimeError(msg)
        return super()._apply(fn, recurse)

    def train(self, mode: bool = True) -> FastModel:  # noqa: FBT001, FBT002 - torch's signature
        """`eval()` / `train(False)` only: the model is inference only (BatchNorm folded, fused and fp16 weights).

        Args:
            mode: Must be False; True raises RuntimeError (train the original model and patch it again).
        """
        if mode:
            msg = "FastModel is inference only (BatchNorm folded, fused weights); train the original model and patch it"
            raise RuntimeError(msg)
        return super().train(mode=False)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:  # noqa: ANN401, FBT001, FBT002 - torch's signature
        """Load weights in place (cast to the built dtype; the derived fused copies follow).

        Args:
            state_dict: Weights with the FastModel's keys (`net.` prefix).
            strict: As in `torch.nn.Module.load_state_dict`.
            assign: Must be False; True raises RuntimeError (it would replace the built, cast tensors).
        """
        if assign:
            msg = (
                "FastModel.load_state_dict: assign=True would replace the built (cast, fused) tensors; use assign=False"
            )
            raise RuntimeError(msg)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor | None = None, *args: object, **kwargs: object) -> object:
        """Run the model; the image batch is `x` or the `pixel_values` keyword (HF).

        Args:
            x: Image batch `[B, 3, H, W]` (transformed as the model expects).
            *args: Passed on to the model's forward.
            **kwargs: Passed on to the model's forward (`pixel_values=` instead of `x`).
        """
        key = None
        if x is None:
            key = "pixel_values"
            x = kwargs.pop(key, None)  # ty: ignore[invalid-assignment]
        if not isinstance(x, torch.Tensor):
            msg = f"{type(self.original).__name__}: pass the image batch as a tensor (`x` or `pixel_values=`)"
            raise TypeError(msg)
        if self.mode == "kernels" and x.requires_grad and torch.is_grad_enabled():  # kernels have no backward (TG-3)
            msg = (
                f"{type(self.original).__name__}: the kernels have no backward, so gradients to the input would be "
                "wrong; patch a fresh copy with mode='portable' (differentiable) or run under torch.no_grad()"
            )
            raise RuntimeError(msg)
        net = self.net
        big = self._big[0]
        # per-shape dispatch inside kernels, not a fallback: launch-bound inputs (< HACC_PIX) run the same optimized
        # network on plain fp16 cuDNN convs (faster than the fp16-accumulate ones there); documented on the cards
        if big is not None and x.dim() == 4 and x.shape[0] * x.shape[2] * x.shape[3] >= HACC_PIX:  # noqa: PLR2004
            net = big
        xi = self._input(net, x)
        if not self._acc:  # see `traceable()`
            return _to_float(net(*args, **{key: xi}, **kwargs) if key else net(xi, *args, **kwargs))
        with fp16_accumulate():
            y = net(*args, **{key: xi}, **kwargs) if key else net(xi, *args, **kwargs)
        return _to_float(y)


def _family(names: set[str]) -> str | None:
    if "ConvNeXtBlock" in names:
        return "convnext"
    if "SwinTransformerStage" in names:
        return "swin"
    if names & {"Block", "EvaBlock", "SiglipEncoderLayer"}:
        return "vit"
    if names & _CNN:
        return "cnn"
    return None


def _device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    """`device` as `load` resolves it (CUDA with an index; `OptionsError` if unusable), else the parameters' device."""
    if device is not None:
        return resolve_device(device)
    p = next(model.parameters(), None)
    return cuda_device(p.device) if p is not None else torch.device("cpu")


def _repatch(fast: FastModel, mode: str, device: torch.device | str | None, dtype: str | None) -> FastModel:
    """`patch` of a `FastModel`: itself if `mode`, `device` and `dtype` agree with what it runs, else `OptionsError`."""
    b = fast.backend
    if mode in {"auto", b.mode} and _device(fast, device) == torch.device(b.device) and dtype in {None, b.dtype}:
        return fast
    msg = (
        f"patch: the model is already patched ({b.mode}, {b.device}, {b.dtype}); for other settings patch a fresh"
        f" copy of the original model (mode={mode!r}, device={device!r}, dtype={dtype!r} were asked)"
    )
    raise OptionsError(msg)


def patch(
    model: nn.Module,
    *,
    mode: str = "auto",
    device: torch.device | str | None = None,
    dtype: str | torch.dtype | None = None,
) -> FastModel:
    """Patch a timm / torchvision / HF transformers classifier for fast inference, in place (the `load` vocabulary).

    Args:
        model: The model (weights loaded). It is modified in place: eval mode, BatchNorm folded, cast to fp16 on CUDA,
            block forwards replaced. Keep a copy if you need the original.
        mode: `"auto"` (kernels where it applies, else portable; logged at INFO), `"portable"` (plain PyTorch, any
            model) or `"kernels"` (custom kernels, covered architectures only; raises where it cannot run).
        device: Device to run on; default: where the model's parameters are.
        dtype: `None` (fp16 on CUDA, float32 on the CPU and Apple MPS) or `"float32"` (portable only; `"auto"` mode
            then runs portable).

    Returns:
        The wrapped model; `.backend` records what runs, `.report` the applied passes.

    Raises:
        OptionsError: Unknown mode or dtype, `dtype="float32"` with `mode="kernels"`, an invalid or unavailable
            `device`, or a `FastModel` with other settings (a `FastModel` whose mode, device and dtype match is
            returned as it is).
        BackendUnavailableError: `mode="kernels"` on CPU, on an architecture outside `COVERED`, on a model with
            module hooks (portable keeps them: modules with hooks are not rewritten), or where no fast path applies;
            KernelUnavailableError (a subclass) when the kernel libraries cannot load. Never for `"auto"`.
            The CPU, architecture and hook checks run before the model is touched; when no fast path applies or the
            `cudnn_hacc` library fails to load, the model is already modified (eval, moved, BatchNorm folded, cast).
    """
    want = dtype_name(dtype)
    if want not in {None, "float32"}:
        msg = f"patch: dtype must be None or 'float32' (torch.float32), got {dtype!r}"
        raise OptionsError(msg)
    if mode not in {*MODES, "auto"}:
        msg = f"patch: mode must be 'auto', 'portable' or 'kernels', got {mode!r}"
        raise OptionsError(msg)
    if mode == "kernels" and want is not None:
        msg = "patch(mode='kernels') runs its own mixed precision: dtype must be None"
        raise OptionsError(msg)
    if isinstance(model, FastModel):  # wrapping it again would convert inputs and outputs twice
        return _repatch(model, mode, device, want)
    fp16 = want is None and _device(model, device).type != "mps"  # MPS: float32 by default, as for `load`
    name = type(model).__name__
    if mode != "auto":
        return _replaced(optimize(model, mode=mode, device=device, fp16=fp16), name)
    if not fp16:
        log.info("%s: dtype='float32'; using portable", name)
        return _replaced(optimize(model, mode="portable", device=device, fp16=False), name)
    try:  # the kernels checks run before the model is touched; a covered model without a kernels pass returns portable
        fast = optimize(model, mode="kernels", device=device, auto=True)
    except BackendUnavailableError as e:
        log.info("%s: %s; using portable", name, e)
        return _replaced(optimize(model, mode="portable", device=device), name)
    log.info("%s: using %s", name, fast.mode)
    return _replaced(fast, name)


def _replaced(fast: FastModel, name: str) -> FastModel:
    """Say once when no block of `fast` was rewritten: an architecture outside `COVERED` only gets cast (TG-5)."""
    if fast.report["family"] == "generic":
        log.warning(
            "%s: no covered block found, so no forward was replaced: only BatchNorm folding, the dtype cast and "
            "channels-last apply (`.report` lists the passes; covered classes: docs/usage.md, 'Patching your own')",
            name,
        )
    return fast


def optimize(  # noqa: C901, PLR0913 - the one entry point: argument checks, then the pass pipeline
    model: nn.Module,
    *,
    mode: str = "portable",
    device: torch.device | str | None = None,
    fp16: bool = True,
    auto: bool = False,
    family: str | None = None,
) -> FastModel:
    """Optimize a timm / torchvision / HF transformers classifier in place (the engine behind `patch`).

    `fp16=False` keeps portable CUDA / MPS weights float32 (no fp16 GEMMs; the registry runners' `dtype="float32"`);
    `kernels` is always fp16. `auto`: a covered model on which no kernels pass applies comes back portable instead of
    raising (`patch(mode="auto")`). `family` (a `COVERED` key): the architecture family of a registry network that its
    class names do not reveal (e.g. `"cnn"`: a plain conv / SiLU / concat / pool net of our own).
    """
    if mode not in MODES:
        msg = f"optimize: mode must be one of {MODES}, got {mode!r}"
        raise OptionsError(msg)
    if mode == "kernels" and not fp16:
        msg = "optimize(mode='kernels') runs in float16 only"
        raise OptionsError(msg)
    dev = _device(model, device)
    names = {type(m).__name__ for m in model.modules()}
    if family is not None and family not in COVERED:
        msg = f"optimize: family must be one of {sorted(COVERED)}, got {family!r}"
        raise OptionsError(msg)
    family = family or _family(names)
    if mode == "kernels":
        if dev.type != "cuda":
            msg = f"optimize(mode='kernels') needs a CUDA device, got {dev}"
            raise BackendUnavailableError(msg)
        if family is None:
            msg = (
                f"optimize(mode='kernels') does not cover {type(model).__name__}; covered: "
                f"{', '.join(sorted({n for v in COVERED.values() for n in v}))}. Use mode='portable'."
            )
            raise BackendUnavailableError(msg)
        from . import fold  # noqa: PLC0415

        if any(fold.hooked(m) for m in model.modules()):
            msg = f"optimize(mode='kernels'): {type(model).__name__} has module hooks, which the kernel rewrites drop"
            raise BackendUnavailableError(msg)
        from . import ops  # noqa: PLC0415 - loads the kernel libraries (raises KernelUnavailableError)

        libs = ops.libraries(dev)
    else:
        libs = []
    # after the kernels checks: a refusal up to here leaves the user's model as it was; the refusals after the passes
    # (no fast path applies, cudnn_hacc fails to load) leave it modified (see `patch`)
    model = model.eval().to(dev)
    if any(
        t.is_floating_point() and t.dtype != torch.float32 for t in itertools.chain(model.parameters(), model.buffers())
    ):
        model = model.float()  # the passes (BatchNorm fold, probes) run in float32; fp16 after them where requested
    report: dict[str, Any] = {"mode": mode, "family": family or "generic", "device": str(dev)}
    if libs:
        report["kernels"] = libs
    with torch.no_grad(), float32_default():  # tensors made by the passes: float32 whatever the caller's default
        if mode == "kernels" and type(model).__name__ == "ShuffleNetV2" and hasattr(model, "stage2"):
            from .blocks import build_shufflenet  # noqa: PLC0415

            small, big = build_shufflenet(model, dev)
            report.update(fp16=True, fp16_accumulate=True, shufflenet_units=len(small.units))
            fast = FastModel(small, model, mode, report, big=big)
            fast._shuffle_big = big
            return fast
        return _optimize(model, mode, family, names, dev, report, half=fp16 and dev.type in {"cuda", "mps"}, auto=auto)


def _optimize(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 - one pass pipeline
    model: nn.Module,
    mode: str,
    family: str | None,
    names: set[str],
    dev: torch.device,
    report: dict[str, Any],
    *,
    half: bool,
    auto: bool = False,
) -> FastModel:
    from . import blocks, fold, transformer  # noqa: PLC0415

    is_kernels = mode == "kernels"
    cuda = dev.type == "cuda"
    conv_like = family == "cnn" or (family is None and "BatchNorm2d" in names)
    if is_kernels and "_DenseBlock" in names:
        report["densenet_blocks"] = blocks.patch_densenet(model)
    net: nn.Module = model
    if conv_like:
        net, rep = fold.fold_bn(model)
        report.update(rep)
    if half:
        net = net.half()
    cpu_cl = dev.type == "cpu" and conv_like  # oneDNN: channels_last convolutions run 12-30 % faster (cpu_speed.md)
    if cuda or cpu_cl:
        net = net.to(memory_format=torch.channels_last)
    report["fp16"] = half
    report["channels_last"] = cuda or cpu_cl
    report["fp16_accumulate"] = half and cuda
    nmax = 0
    if family == "convnext":
        n = transformer.patch_convnext(net, mode)
        report["convnext_blocks"] = n
        nmax += n if is_kernels else 0
    elif family == "swin":
        report["swin_sdpa"] = transformer.patch_swin_portable(net)
        if is_kernels:
            n = transformer.patch_swin_kernels(net)
            report["swin_stages"] = n
            nmax += n
    if is_kernels and "SiglipEncoderLayer" in names:
        n = transformer.patch_hf_siglip(net)
        report["siglip_layers"] = n
        nmax += n
    if is_kernels and "EvaBlock" in names:
        n = transformer.patch_eva(net)
        report["eva_blocks"] = n
        nmax += n
        n = transformer.patch_pe(net, mode)
        report["pe_blocks"] = n
        nmax += n
    elif "EvaBlock" in names:
        report["pe_blocks"] = transformer.patch_pe(net, mode)
    if "Block" in names:
        n = transformer.patch_vit_blocks(net, mode)
        report["vit_blocks"] = n
        nmax += n if is_kernels else 0
    big = None
    if is_kernels and family == "cnn":
        nfused = 0
        if isinstance(net, fx.GraphModule):
            nfused = _fuse_cnn(net, names, report)
        report["fused_convs"] = nfused
        if not nfused:  # untraceable / nothing fused: fp16-accumulate cuDNN convs for big inputs instead
            from fastforks.kernels import cudnn_hacc  # noqa: PLC0415

            try:
                report["cudnn_hacc"] = cudnn_hacc.load(dev).tag
            except BackendUnavailableError as e:
                if not auto:
                    raise
                log.info("%s: cudnn_hacc: %s", type(model).__name__, e)
            else:
                report["cudnn_hacc_min_pixels"] = (
                    HACC_PIX  # smaller inputs: plain fp16 convs (documented per-shape dispatch)
                )
                big = fold.hacc_copy(net)
                nmax += 1
        else:
            nmax += 1
    if is_kernels and not nmax:
        msg = (
            f"optimize(mode='kernels'): no fast path applies to {type(model).__name__} ({family}); use mode='portable'"
        )
        if not auto:
            raise BackendUnavailableError(msg)
        # every pass above that ran in kernels mode without a kernel left the portable form: the net IS the portable
        # result
        log.info("%s; using portable", msg.removeprefix("optimize(mode='kernels'): "))
        mode = report["mode"] = "portable"
        report.pop("kernels", None)
    log.info("optimize(%s): %s", mode, report)
    return FastModel(net, model, mode, report, big=big)


def _fuse_cnn(gm: fx.GraphModule, names: set[str], report: dict[str, Any]) -> int:
    from . import blocks, cnn  # noqa: PLC0415

    n = cnn.fuse_all(gm)
    for k in ("nls", "npad", "raw_input", "pad8", "nbnact", "ninto", "nmergecat", "ncatinto", "npool", "nstempool"):
        report[k] = int(getattr(gm, k, 0))
    report["xdw"] = cnn.fuse_xdw(gm)
    if "LiteMLA" in names:
        report["litemla"] = blocks.patch_litemla(gm)
    report["gconv"] = cnn.fuse_gconv(gm)
    report["dwmult"] = cnn.fuse_dwmult(gm)
    # half2 depthwise not for FastViT: re-parameterized RepMixer activations are too large for fp16 accumulation
    report["dw2"] = 0 if "RepMixer" in names else cnn.to_dw2(gm)
    report["gconv2"], report["se_scale"] = cnn.fuse_regnet(gm)
    report["conv_tc"] = cnn.to_conv_tc(gm)
    report["pw"] = cnn.to_pw(gm)
    return n
