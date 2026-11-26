"""DINOv2 linear runners: the DINOv2 ViT (`fastforks.shared.depth_anything_v2`) with the authors' ADE20K linear heads.

Test-time protocol of `dinov2_vit*14_ade20k_linear_config.py` (mmseg, as the authors' segmentation notebook builds
it): keep-ratio resize of the shorter side to 512 (bilinear), ImageNet normalisation, mmseg `slide_inference` with
512 x 512 crops at stride 341; each crop is zero-padded to a multiple of 14 (centred, the notebook's `CenterPadding`
pre-hook of the backbone), the last block's normalised patch tokens go through the head's batch norm and 1x1 conv,
and the logits are resized bilinearly to the (unpadded) crop size. The averaged windows are resized to the image size.

The head's batch norm is folded into the classifier (one `[150, C]` matrix applied to the tokens). `portable` runs the
ViT with SDPA attention (fp16 on CUDA); `kernels` with the fastbb fp16 path (flash attention with fp32 softmax,
fp16-accumulate GEMMs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import mismatch_details
from fastforks._core.errors import CheckpointMismatchError
from fastforks.kernels import dpt_fused
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.shared.cuda_graphs import GraphCache, graph_replay_allowed
from fastforks.shared.depth_anything_v2 import DINOv2, use_fast_vit, use_sdpa_attention
from fastforks.shared.seginfer import resize, slide, target_size, upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks._core.mlx import mx
    from fastforks.segmentation._runner import Context

_GRAPH_CROP = (1, 3, 518, 518)  # fixed: one official 512² crop padded to 37 patches per side


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: DINOv2 size, `"vits"`, `"vitb"` or `"vitl"`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), `["mmseg", 99999999, 512]`.
        crop: Sliding-window crop side.
        stride: Sliding-window stride.
        cuda_graphs: Replay the ViT-S kernels patch logits for a repeated single full crop.
    """

    arch: str
    resize: tuple[str, int, int]
    crop: int
    stride: int
    cuda_graphs: bool = False


_HEAD_KEYS = frozenset(
    f"decode_head.{k}"
    for k in (
        "conv_seg.weight",
        "conv_seg.bias",
        "bn.weight",
        "bn.bias",
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
    )
)


def _check_head(sd: dict[str, torch.Tensor], what: str) -> dict[str, torch.Tensor]:
    """`sd` if it holds exactly the linear head (1x1 conv + BatchNorm) this family folds, else a clear error.

    Raises:
        CheckpointMismatchError: Missing or unexpected keys (another head type, extra layers), or tensors whose shapes
            do not fit one linear head.
    """
    missing = sorted(_HEAD_KEYS - sd.keys())
    unexpected = sorted(sd.keys() - _HEAD_KEYS)
    shapes = []
    if not missing:
        w = sd["decode_head.conv_seg.weight"]
        for k, want in (
            ("decode_head.conv_seg.weight", (w.shape[0], w.shape[1], 1, 1)),
            ("decode_head.conv_seg.bias", (w.shape[0],)),
            *((f"decode_head.bn.{n}", (w.shape[1],)) for n in ("weight", "bias", "running_mean", "running_var")),
        ):
            if tuple(sd[k].shape) != want:
                shapes.append((k, str(list(sd[k].shape)), str(list(want))))
    if missing or unexpected or shapes:
        raise CheckpointMismatchError(what, mismatch_details(missing, unexpected, shapes))
    return sd


def _head(ctx: Context) -> tuple[torch.Tensor, torch.Tensor]:
    sd = _check_head(ctx.state_dict(1, key="state_dict", strip=""), f"{ctx.variant.ref} head checkpoint")
    bn = {k: sd[f"decode_head.bn.{k}"].double() for k in ("weight", "bias", "running_mean", "running_var")}
    w = sd["decode_head.conv_seg.weight"].double().flatten(1)
    s = bn["weight"] / torch.sqrt(bn["running_var"] + 1e-5)  # SyncBN default eps
    b = sd["decode_head.conv_seg.bias"].double() + w @ (bn["bias"] - bn["running_mean"] * s)
    return (w * s).float(), b.float()


class _Net(torch.nn.Module):
    """ViT + folded linear head: a normalised crop `[B, 3, h, w]` -> logits `[B, 150, h, w]`."""

    cls_w: torch.Tensor
    cls_b: torch.Tensor

    def __init__(self, ctx: Context, *, kernels: bool) -> None:
        super().__init__()
        vit = DINOv2(ctx.params.arch)
        vit.load_state_dict(ctx.state_dict(0))
        vit.eval().requires_grad_(requires_grad=False)
        weight, bias = _head(ctx)
        self.register_buffer("cls_w", weight, persistent=False)
        self.register_buffer("cls_b", bias, persistent=False)
        self.patch_size = vit.patch_size
        self.vit = vit
        self.dtype = ctx.torch_dtype
        self.kernels = kernels
        self.to(ctx.device)
        if kernels:
            self.fast = use_fast_vit(vit, attn="fh", fused=dpt_fused.load(ctx.device))
        else:
            use_sdpa_attention(vit)
            vit.to(self.dtype)
        self.cls_w, self.cls_b = self.cls_w.to(self.dtype), self.cls_b.to(self.dtype)

    def _patch_logits(self, y: torch.Tensor) -> torch.Tensor:
        (t,) = self.vit.get_intermediate_layers(y, n=1)
        return F.linear(t.to(self.dtype), self.cls_w, self.cls_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised crop -> logits at the crop size (in the network dtype)."""
        h, w = x.shape[-2:]
        p = self.patch_size
        ph, pw = math.ceil(h / p) * p - h, math.ceil(w / p) * p - w
        y = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)).to(self.dtype)
        s = self._patch_logits(y)
        # NCHW before the resize: a channels-last result makes `slide`'s accumulation into NCHW scores ~6x slower
        s = s.unflatten(1, ((h + ph) // p, (w + pw) // p)).permute(0, 3, 1, 2).contiguous()
        return F.interpolate(s, (h, w), mode="bilinear", align_corners=False)


class _SmallGraphNet(_Net):
    """Only the fixed ViT-S patch logits are replayed; crop upsampling stays eager."""

    def __init__(self, ctx: Context) -> None:
        """Set up a one-entry graph and hold the exact position buffer it reads."""
        super().__init__(ctx, kernels=True)
        self.graphs = GraphCache(super()._patch_logits, graphs=1)
        self._graph_position: torch.Tensor | None = None

    def _patch_logits(self, y: torch.Tensor) -> torch.Tensor:
        if (
            not y.is_cuda
            or y.dtype != torch.float16
            or y.shape != _GRAPH_CROP
            or not y.is_contiguous()
            or not torch.is_inference_mode_enabled()
            or torch.compiler.is_compiling()
            or torch.jit.is_tracing()
            or not graph_replay_allowed()
        ):
            return super()._patch_logits(y)
        side = _GRAPH_CROP[2] // self.patch_size
        position = self.fast.pos_embed(side, side)
        if self._graph_position is not None and self._graph_position is not position:
            self.release_graphs()  # a different eager grid can evict this table
        self._graph_position = position
        out = self.graphs.run(y.device, y)
        return super()._patch_logits(y) if out is None else out

    def release_graphs(self) -> None:
        """Drop the graph before dropping its position buffer."""
        self.graphs.release()
        self._graph_position = None


class Portable(Runner):
    """Plain PyTorch: SDPA attention, float32, or fp16 on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    fused = False  # kernels: the image-size resize and argmax by `seg_up`

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _Net(ctx, kernels=False)

    def _scores(self, image: torch.Tensor) -> torch.Tensor:
        """Logits `[B, 150, h, w]` at the network input size (sliding window)."""
        p = self.ctx.params
        h, w = image.shape[-2:]
        x = scale(resize(image, target_size(h, w, p.resize)), self.ctx.io)
        # windows resized and averaged in the network dtype (fp16 on CUDA: half the bytes, as the fork's fp16 sums)
        return slide(self.net, x, (p.crop, p.crop), (p.stride, p.stride), self.net.dtype)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return upsample(self._scores(image), (tuple(image.shape[-2:]),), fused=self.fused).float()

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 150, H, W]` of a batch."""
        return {"logits": self._run(image)}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no image-size logits)."""
        return upsample_argmax(self._scores(image), (tuple(image.shape[-2:]),), fused=self.fused)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


class Kernels(Portable):
    """fastbb fp16 ViT: flash attention (fp32 softmax) and fp16-accumulate GEMMs; fused upsampling."""

    fused = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107 - see Portable
        Runner.__init__(self, ctx)
        if ctx.params.cuda_graphs and ctx.params.arch == "vits" and ctx.device.type == "cuda":
            self.net = _SmallGraphNet(ctx)
        else:
            self.net = _Net(ctx, kernels=True)

    def release_graphs(self) -> None:
        """Release the optional ViT-S patch graph and its retained buffers."""
        if isinstance(self.net, _SmallGraphNet):
            self.net.release_graphs()

    module = Runner.module


class _MlxNet(_Net):
    def __init__(self, ctx: Context) -> None:
        torch.nn.Module.__init__(self)
        from fastforks.shared.mlx_vit import DinoLinearProbe  # noqa: PLC0415 - optional MLX dependency

        vit = DINOv2(ctx.params.arch)
        vit.load_state_dict(ctx.state_dict(0), assign=True)
        weight, bias = _head(ctx)
        self.probe = DinoLinearProbe(vit, weight, bias, ctx.dtype, image_logits=False)
        self.patch_size, self.dtype = vit.patch_size, ctx.torch_dtype

    def _patch_logits(self, y: torch.Tensor) -> torch.Tensor:
        return self.probe(y).to(self.dtype)

    def low(self, x: torch.Tensor) -> mx.array:
        """Normalised crops `[B, 3, h, w]` -> patch logits `[B, gh, gw, 150]` on MLX (`forward` before its resize)."""
        h, w = x.shape[-2:]
        p = self.patch_size
        ph, pw = math.ceil(h / p) * p - h, math.ceil(w / p) * p - w
        y = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return self.probe.logits(y).reshape(x.shape[0], (h + ph) // p, (w + pw) // p, -1)


class Mlx(Portable):
    """MLX patch logits; the windows' resize and average and the image-size argmax are Metal kernels (`mlx_seg`).

    In PyTorch on the CPU that post-processing took 4/5 of a call (dinov2-s, 1024x2048: 660 ms, the ViT 90 ms).
    """

    def __init__(self, ctx: Context) -> None:
        """Convert the verified DINOv2 backbone and ADE20K head."""
        Runner.__init__(self, ctx)
        self.net = _MlxNet(ctx)

    def _slide(self, image: torch.Tensor) -> mx.array | None:
        """Scores `[B, h, w, 150]` (float32, MLX) at the network input size; None for an image below the crop."""
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - optional MLX dependency

        p = self.ctx.params
        h, w = image.shape[-2:]
        x = scale(resize(image, target_size(h, w, p.resize)), self.ctx.io)
        return mlx_seg.slide_scores(cast("_MlxNet", self.net).low, x, (p.crop, p.crop), (p.stride, p.stride))

    def _scores(self, image: torch.Tensor) -> torch.Tensor:
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - optional MLX dependency

        scores = self._slide(image)
        if scores is None:
            return super()._scores(image)
        return to_torch(scores.transpose(0, 3, 1, 2)).to(self.net.dtype)

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the scores stay on MLX up to the argmax."""
        from fastforks.shared import mlx_seg, mlx_wrap  # noqa: PLC0415 - optional MLX dependency

        scores = self._slide(image)
        if scores is not None:
            return mlx_seg.resize_argmax(scores, image.shape[-2:])
        token = mlx_wrap.POST_ON_MLX.set(True)
        try:
            return super().labels(image)
        finally:
            mlx_wrap.POST_ON_MLX.reset(token)

    module = Runner.module


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": Mlx}
