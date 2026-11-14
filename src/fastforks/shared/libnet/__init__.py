"""Networks built by an installed model library (timm, torchvision) from a variant's verified weights.

We extend timm and torchvision rather than re-implement them: a family's card names the library and the architecture
(`[params] library`, `arch`), the weight files come from the authors' hosts (hash-checked by the core), and the
network is the library's own module with those weights. `Portable` and `Kernels` run it through `fastforks._optimize`
(portable: plain-PyTorch passes, exportable; kernels: our kernels) and return `{"logits": net(image)}`, the runner
contract of the classification task. A family subclasses them only to change how the network is built (`network`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.checkpoint import load_strict, mismatch_details, torch_mismatch
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
    from types import ModuleType

    from fastforks._core.runner import Context
    from fastforks._optimize import FastModel

__all__ = ["Kernels", "Params", "Portable", "build", "timm", "torchvision"]

# torchvision's DenseNet checkpoints predate its module names (`norm.1` -> `norm1`); torchvision renames on load too.
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
        try:
            return timm().create_model(p.arch, pretrained=True, pretrained_cfg_overlay=overlay, **kwargs).eval()
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

    batch = None  # any batch size
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and optimise it."""
        super().__init__(ctx)
        net = self.network(ctx)
        self.net: FastModel = optimize(net, mode=ctx.mode, device=ctx.device, fp16=ctx.dtype == "float16")
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

    def __init__(self, ctx: Context) -> None:
        """Build and optimise the network; set up the graph cache."""
        super().__init__(ctx)
        self._graphs = SmallShapeGraphs(self.net, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, classes]` of a transformed batch (a CUDA graph replay for small repeated batches)."""
        return {"logits": self._graphs(image)}
