"""WAFT-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.patch import base_type, override
from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.depth_anything_v2.net.dinov2_layers.attention import Attention
from fastforks.stereo._runner import Forward, Runner

from .net.waft import WAFT

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    max_disp: int
    arch: str  # Depth Anything V2 feature encoder: "vits" or "vitb"
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _cfg(arch: str, max_disp: int) -> types.SimpleNamespace:
    """The `configs/SynLarge/DAv2<S|B>-4.yaml` values the network reads."""
    ns = types.SimpleNamespace
    iterative = ns(TYPE="vit", ARCH="vits", PATCH_SIZE=8, LORA_RANK=None, LORA_ALPHA=None)
    return ns(
        WAFT=ns(
            MAX_DISP=max_disp,
            LOSS=["mixlap_40"],
            FEATURE_ENCODER=ns(TYPE="dav2", ARCH=arch, LORA_RANK=8, LORA_ALPHA=16),
            ITERATIVE_MODULE=ns(TASK=["delta"] * 3, DELTA_ITER=iterative, PROP_ITER=iterative),
        )
    )


class _CfgNode(dict):
    """Stands for upstream's yacs `CfgNode` (the checkpoint's `model_config`, unused here)."""


_EXTRA: dict[tuple[str, str], object] = {  # what the checkpoints' pickles reference besides tensors
    ("bridgedepth.config.config", "CfgNode"): _CfgNode,
}
_ATTENTION_BYTES = 256 * 1024**2


def _bounded_attention(m: Attention, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Keep the original DINOv2 attention math with a bounded CUDA head workspace."""
    if attn_bias is not None:
        return base_type(m).forward(m, x, attn_bias)
    b, n, c = x.shape
    capture = torch.compiler.is_compiling() or torch.jit.is_tracing()
    if (
        not x.is_cuda
        or capture
        or torch.is_grad_enabled()
        or m.training
        or b * m.num_heads * n * n * x.element_size() <= _ATTENTION_BYTES
    ):
        return Attention.forward(m, x)
    qkv = m.qkv(x).reshape(b, n, 3, m.num_heads, c // m.num_heads).permute(2, 0, 3, 1, 4)
    q, key, value = qkv.unbind(0)
    q = q * m.scale
    key = key.transpose(-2, -1)
    step = max(1, _ATTENTION_BYTES // (b * n * n * q.element_size()))
    groups = zip(q.split(step, 1), key.split(step, 1), value.split(step, 1), strict=True)
    parts = [m.attn_drop((query @ key_part).softmax(dim=-1)) @ value_part for query, key_part, value_part in groups]
    out = torch.cat(parts, dim=1).transpose(1, 2).reshape(b, n, c)
    return m.proj_drop(m.proj(out))


def _net(ctx: Context) -> WAFT:
    net = WAFT(_cfg(ctx.params.arch, ctx.params.max_disp))
    net.load_state_dict(ctx.state_dict(key="model", strip="", extra=_EXTRA))
    return net.eval().to(ctx.device)


def _disparity(net: WAFT, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    capture = torch.compiler.is_compiling() or torch.jit.is_tracing()
    final_only = not capture and not torch.is_grad_enabled()
    return net({"img1": left, "img2": right}, final_disparity_only=final_only)["disp_pred"]


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 autocast on CUDA (upstream forward).

    NCHW like upstream: channels-last makes the convolutions faster but costs more in the layout copies between the
    convolutions and the ViT / attention parts (6 % slower in total at Middlebury-H).
    """

    batch = None
    # pixels (pairs x H x W) per network call: the eager call is launch bound at small sizes (RTX 5060 Ti,
    # Windows, 8 pairs 480 x 640: 98.5 -> 78.9 ms per pair)
    batch_pixels = 1_300_000

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.half = ctx.dtype == "float16"
        for module in self.net.modules():
            if isinstance(module, Attention):
                override(module, _bounded_attention)

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if not self.half:
            return _disparity(self.net, left, right)
        with torch.autocast(left.device.type, dtype=torch.float16):
            return _disparity(self.net, left, right)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a batch (0..255 images; the network pads and crops itself)."""
        return {"disparity": self._run(left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, lambda _net, left, right: self._run(left, right))


_GRAPH_PIXELS = 600_000  # fixed: batch x H x W; launch-bound up to KITTI 375x1242 and 544x960, measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: merged LoRA, fp16 NHWC convs with fused InstanceNorm / SiLU / GELU / residual epilogues.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and prepare it for the kernels."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels
        self._graphs = SmallShapeGraphs(
            self._run,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._forward(self.net, left, right)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a batch (0..255 images)."""
        return {"disparity": self._graphs(left, right)}


class _OnePair(Portable):
    """Portable for the MLX port: one pair per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OnePair, whole="disparity"),
}
