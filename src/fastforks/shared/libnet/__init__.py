"""Networks built by an installed model library (timm, torchvision) from a variant's verified weights.

We extend timm and torchvision rather than re-implement them: a family's card names the library and the architecture
(`[params] library`, `arch`), the weight files come from the authors' hosts (hash-checked by the core), and the
network is the library's own module with those weights. `Portable` and `Kernels` run it through `fastforks._optimize`
(portable: plain-PyTorch passes, exportable; kernels: our kernels) and return `{"logits": net(image)}`, the runner
contract of the classification task. A family subclasses them only to change how the network is built (`network`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.checkpoint import load_strict, mismatch_details, torch_mismatch
from fastforks._core.deferred_init import deferred_library_inits
from fastforks._core.errors import (
    CardError,
    CheckpointMismatchError,
    MissingDependencyError,
    install_hint,
)
from fastforks._core.runner import Runner
from fastforks._optimize import HACC_PIX, optimize
from fastforks.shared.cuda_graphs import SmallShapeGraphs

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from fastforks._core.runner import Context
    from fastforks._optimize import FastModel

__all__ = ["Kernels", "Mlx", "Params", "Portable", "build", "timm", "torchvision"]

# torchvision's DenseNet checkpoints predate its module names (`norm.1` -> `norm1`); torchvision renames on load too.
_TIMM_INITS = ("trunc_normal_", "trunc_normal_tf_", "lecun_normal_", "variance_scaling_")  # timm.layers
_MAPPED_TIMM = {
    "perception_encoder": "eva",
    "siglip2": "vision_transformer",
    "clip": "vision_transformer",
}
_MAPPED_WEIGHT_BYTES = 1 << 30
_DENSENET_KEY = re.compile(
    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
)


@dataclass(frozen=True)
class Params:
    """Card `[params]` of a family built by a model library.

    Attributes:
        library: `"timm"` or `"torchvision"`.
        arch: `timm.create_model` name (with the pretrained tag, which selects timm's config of those weights) or
            `torchvision.models.get_model` name.
    """

    library: str
    arch: str


def timm() -> ModuleType:
    """The installed `timm`.

    Raises:
        MissingDependencyError: timm is not installed.
    """
    try:
        import timm  # noqa: PLC0415 - optional dependency (extra `timm`)
    except ImportError as e:
        msg = f"this model is built by timm, which is not installed or cannot be imported ({e}): {install_hint('timm')}"
        raise MissingDependencyError(msg) from e
    return timm


_shared_lock = threading.Lock()
_shared_sds: dict[int, int] = {}  # id(state dict) -> open `_shared_state_dict` blocks passing it
_shared_saved: list[Any] = []  # timm's own `PretrainedCfg.to_dict`, while patched


@contextlib.contextmanager
def _shared_state_dict(lib: ModuleType, state_dict: dict[str, torch.Tensor]) -> Generator[None]:
    """Make timm's model config keep `state_dict` itself instead of a deep copy.

    `build_model_with_cfg` turns the pretrained config into a dict with `dataclasses.asdict`, which deep-copies every
    tensor of an in-memory state dict, and keeps that dict on the model (`.pretrained_cfg`): one more copy of all the
    weights, in anonymous memory (the read checkpoint is file-backed), for as long as the model lives (SigLIP so400m:
    1.6 GB). Only a config carrying `state_dict` (by identity) changes; others, also on other threads, convert as
    before.
    """
    cls = lib.models._pretrained.PretrainedCfg  # noqa: SLF001 - the dataclass timm converts
    with _shared_lock:
        if not _shared_sds:
            orig = cls.to_dict
            _shared_saved.append(orig)

            def to_dict(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - timm's signature
                sd = self.state_dict
                if id(sd) not in _shared_sds:
                    return orig(self, *args, **kwargs)
                out = orig(dataclasses.replace(self, state_dict=None), *args, **kwargs)
                out["state_dict"] = sd
                return out

            cls.to_dict = to_dict
        _shared_sds[id(state_dict)] = _shared_sds.get(id(state_dict), 0) + 1
    try:
        yield
    finally:
        with _shared_lock:
            if (n := _shared_sds.pop(id(state_dict)) - 1) > 0:
                _shared_sds[id(state_dict)] = n
            if not _shared_sds:
                cls.to_dict = _shared_saved.pop()


def torchvision() -> ModuleType:
    """The installed `torchvision`.

    Raises:
        MissingDependencyError: torchvision is not installed.
    """
    try:
        import torchvision  # noqa: PLC0415 - optional dependency (extra `torchvision`)
    except ImportError as e:
        msg = (
            f"this model is built by torchvision, which is not installed or cannot be imported ({e})"
            f": {install_hint('torchvision')}"
        )
        raise MissingDependencyError(msg) from e
    return torchvision


def _mapped_timm(
    ctx: Context, lib: ModuleType, implementation: str, checkpoint: dict[str, torch.Tensor], options: dict[str, Any]
) -> torch.nn.Module:
    """Assign a verified native timm checkpoint without a duplicate CPU weight allocation."""
    net = lib.create_model(ctx.params.arch, pretrained=False, device="meta", **options)
    if implementation == "eva":
        # RoPE bands are derived by timm and deliberately absent from its state dict.
        rope = net.get_submodule("rope")
        rope.to_empty(device="cpu")
        rope.reset_parameters()
    filter_checkpoint = getattr(lib.models, implementation).checkpoint_filter_fn
    load_strict(net, filter_checkpoint(checkpoint, net), f"{ctx.variant.ref} checkpoint", assign=True)
    return net.eval()


def build(ctx: Context, index: int = 0, **kwargs: Any) -> torch.nn.Module:  # noqa: ANN401 - library keyword args
    """The library's network for `ctx.params` with the variant's weight file `index`, in eval mode on the CPU.

    The core reads the file and timm loads that state dict (`pretrained_cfg_overlay={"state_dict": ...}`), so its
    checkpoint filter (e.g. position embedding resampling for another `img_size`) applies exactly as for
    `timm.create_model(pretrained=True)`.

    Raises:
        MissingDependencyError: The library is not installed.
        CardError: Unknown library.
    """
    p = ctx.params
    if p.library == "timm":
        overlay = {"state_dict": ctx.state_dict(index, strip="")}  # read (and hash-checked) by the core, loaded by timm
        if "num_classes" in kwargs:  # else timm drops a head whose size differs from its config's and loads non-strict
            overlay["num_classes"] = kwargs["num_classes"]
        lib = timm()
        if (
            getattr(ctx, "mode", None) == "kernels"
            and ctx.device.type == "cuda"
            and (implementation := _MAPPED_TIMM.get(ctx.card.family)) is not None
            and ctx.user_weights is None
            and ctx.user_state is None
            and (ctx.variant.weights[index].size or 0) >= _MAPPED_WEIGHT_BYTES
            and not getattr(p, "stream", False)
            and "num_classes" not in kwargs
        ):
            return _mapped_timm(ctx, lib, implementation, overlay["state_dict"], kwargs)
        # timm's ViT-style initialisers bypass torch.nn.init: deferred as well, every tensor the checkpoint loads skips
        # its random fill (wd-vit-tagger-v3, a ViT-B, on the CPU: about 0.55 s per load)
        timm_modules = [m for n, m in tuple(sys.modules.items()) if n == "timm" or n.startswith("timm.")]
        try:
            with deferred_library_inits(timm_modules, _TIMM_INITS), _shared_state_dict(lib, overlay["state_dict"]):
                net = lib.create_model(p.arch, pretrained=True, pretrained_cfg_overlay=overlay, **kwargs)
            if isinstance(cfg := getattr(net, "pretrained_cfg", None), dict):
                cfg.pop("state_dict", None)  # loaded: the model need not keep the checkpoint alive
            return net.eval()
        except RuntimeError as e:  # timm's strict load of the state dict
            if (keys := torch_mismatch(e)) is None:
                raise
            what = f"{ctx.variant.ref} checkpoint"
            raise CheckpointMismatchError(what, mismatch_details(*keys)) from e
    if p.library == "torchvision":
        net = torchvision().models.get_model(p.arch, weights=None, **kwargs)
        sd = ctx.state_dict(index, strip="")
        if p.arch.startswith("densenet"):
            sd = {_DENSENET_KEY.sub(lambda m: m[1] + m[2], k): v for k, v in sd.items()}
        # torchvision's checkpoints predate BatchNorm's `num_batches_tracked` (torch fills it in on load too)
        load_strict(net, sd, f"{ctx.variant.ref} checkpoint", not_in_checkpoint=("num_batches_tracked",))
        return net.eval()
    msg = f"{ctx.card.family}: unknown library {p.library!r} (timm, torchvision)"
    raise CardError(msg)


class _LibRunner(Runner):
    """Shared part of `Portable` and `Kernels`: build `network(ctx)` and optimise it for `ctx.mode` on `ctx.device`."""

    family: str | None = None
    fuse_convs = True
    batch = None  # any batch size
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and optimise it."""
        super().__init__(ctx)
        net = self.network(ctx)
        self.net: FastModel = optimize(
            net,
            mode=ctx.mode,
            device=ctx.device,
            fp16=ctx.dtype == "float16",
            family=self.family,
            fuse_convs=self.fuse_convs,
            cast_cpu_weights=True,
        )
        if ctx.device.type == "cuda":  # the fp32 weights and the BatchNorm-fold probe stay cached otherwise
            torch.cuda.empty_cache()

    def network(self, ctx: Context) -> torch.nn.Module:
        """The upstream network with its weights (CPU, eval); families override this to wrap or adapt it."""
        return build(ctx)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, classes]` of a transformed batch."""
        return {"logits": self.net(image)}


class Portable(_LibRunner):
    """Plain PyTorch through `fastforks._optimize` (BatchNorm folding, fused attention, fp16: NHWC CUDA, NCHW MPS)."""

    dtypes = ("float32", "float16")

    def module(self) -> torch.nn.Module:
        """The optimised network (shared weights): transformed images `[B, 3, h, w]` -> logits.

        On CUDA the model call runs its fp16 GEMMs with cuBLAS fp16 accumulation (a process-global flag that tracing
        cannot capture); this module does not, so its logits differ from the call within the GPU golden tolerance
        (`tests/test_golden.py::test_gpu_module`). Set `torch.backends.cuda.matmul.allow_fp16_accumulation = True`
        around a compiled / exported call to match the call. On the CPU both are the same float32 computation.
        """
        return self.net.traceable()


# CUDA graphs (kernels): below the big-input switch (batch x crop x crop < `HACC_PIX`, batch <= 7 at 224) the eager
# forward is launch-bound (the host issues the ~100-300 kernels slower than the GPU runs them: MobileNetV2 1.0 ms eager
# vs 0.3 GPU); a repeated batch shape replays a CUDA graph, captured on its second call
_GRAPH_PIXELS = HACC_PIX - 1  # fixed: the launch-bound range ends where the big-input (cudnn_hacc) path starts
_GRAPHS = 2  # fixed: graph cache policy (the crop is fixed, so a key is a batch size)


class Kernels(_LibRunner):
    """Our kernels through `fastforks._optimize` (fp16-accumulate convolutions, fused epilogues and attention).

    A repeated small batch (`_GRAPH_PIXELS`) replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager forward is
    launch-bound there. The output is a fresh tensor per call; `release_graphs()` frees the graphs and their pool.
    """

    _graph_capture_warmup = True

    def __init__(self, ctx: Context) -> None:
        """Build and optimise the network; set up the graph cache."""
        super().__init__(ctx)
        self._graphs = SmallShapeGraphs(
            self.net, pixels=_GRAPH_PIXELS, graphs=_GRAPHS, capture_warmup=self._graph_capture_warmup
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, classes]` of a transformed batch (a CUDA graph replay for small repeated batches)."""
        return {"logits": self._graphs(image)}


class CNNKernels(Kernels):
    """Apply convolution rewrites to a registry network with a validated CNN kernels mode."""

    family = "cnn"


class PoolKernels(CNNKernels):
    """Fuse pooling while preserving native convolutions for sensitive classifiers."""

    fuse_convs = False


class Mlx(Runner):
    """Apple MLX runner for the supported timm and torchvision classifiers."""

    Params = Params
    batch = None

    def __init__(self, ctx: Context) -> None:
        """Translate the verified network to a channels-last MLX forward."""
        super().__init__(ctx)
        from fastforks.shared.mlx_wrap import lazy_build  # noqa: PLC0415 - with the MLX runner only

        from . import mlx_cnn  # noqa: PLC0415 - optional MLX dependency

        if ctx.card.family in mlx_cnn.FAMILIES and not os.environ.get("FASTFORKS_MLX_GRAPH"):
            self.net = mlx_cnn.Network(ctx)
        else:  # every other timm / torchvision network: the generic exported-graph translation
            from fastforks.shared.mlx_graph import GraphNet  # noqa: PLC0415 - optional MLX dependency

            # the twin's weights stay the memory-mapped checkpoint (`lazy_build`); the graph keeps the twin
            self.net = GraphNet(lazy_build(lambda: self.network(ctx)), ctx.dtype, kernels=ctx.mode == "kernels")

    def network(self, ctx: Context) -> torch.nn.Module:
        """The upstream network with its weights (CPU, eval); families override this to wrap or adapt it."""
        return build(ctx)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Class logits for a transformed image batch."""
        out = self.net(image)
        return {"logits": out[0] if isinstance(out, tuple) else out}
