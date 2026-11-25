"""OMNI-DC runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `OGNIDC.forward` with the released demo configuration: the ImageNet-normalised image and the
sparse depth are zero-padded below and on the right to at least 32 and a multiple of 16 (the image in normalised space,
i.e. with the ImageNet mean) and the output is cropped back. Missing depth (0, negative, NaN, inf) enters as 0.
The conjugate-gradient solve runs in float32 in chunks of steps with one host sync each (upstream syncs every step); it
returns the same iterate as upstream's loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.half import pure_half
from fastforks._core.patch import override
from fastforks._core.preprocess import IMAGENET_MEAN
from fastforks._core.tracing import in_onnx_export
from fastforks._core.upload import device_tensor
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.depth.models.depth_anything_v2.fused import FusedHead, FusedViT
from fastforks.kernels import omni_cg
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import CAPTURE_LOCK, capture_stream, free_captured
from fastforks.shared.depth_anything_v2 import fold_layer_scale, sdpa_attention

from .net import (
    MULTIPLE,
    CBAMBlock,
    CGState,
    OmniDCNet,
    mono_size,
    quantile_normalise,
    solve,
    sparse_inputs,
    system,
)

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

_LAYERS = (4, 11, 17, 23)  # fixed: Depth Anything V2 Large's intermediate layers
_CBAM_BAND_ROWS = 192  # fixed: CBAM convolution band height for large CUDA inputs (memory bound, not numerics)
_CBAM_MIN_ROWS, _CBAM_MIN_PIXELS = 800, 900_000


@dataclass(frozen=True)
class Params:
    """Options for the depth completion runner."""

    cuda_graphs: bool


def _graph_solve(cg: GraphCG, d: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cg.d.copy_(d)
    cg.b.copy_(b)
    return cg.run().clone()


def _decoder_cbam(block: CBAMBlock, x: torch.Tensor) -> torch.Tensor:
    """Bound the two full-size 3 x 3 convolution buffers, then attend over the whole feature map."""
    if (
        not x.is_cuda
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or in_onnx_export()
    ):
        return CBAMBlock.forward(block, x)
    h, w = x.shape[-2:]
    if h < _CBAM_MIN_ROWS or h * w < _CBAM_MIN_PIXELS:
        return CBAMBlock.forward(block, x)

    out = torch.empty_like(x)
    for top in range(0, h, _CBAM_BAND_ROWS):
        bottom = min(top + _CBAM_BAND_ROWS, h)
        lo, hi = max(0, top - 2), min(h, bottom + 2)
        tile = block.bn2(block.conv2(F.relu(block.bn1(block.conv1(x[..., lo:hi, :])))))
        out[..., top:bottom, :].copy_(tile[..., top - lo : bottom - lo, :])
        del tile
    out = block.ca(out) * out
    out = block.sa(out) * out
    return F.relu(out + x)


def _net(ctx: Context) -> OmniDCNet:
    net = OmniDCNet()
    net.load_state_dict(net.checkpoint_keys(ctx.state_dict()))
    net.eval()
    net.fold_bn()
    override(net.backbone.dec2[1], _decoder_cbam)
    return net


def prepare(image: torch.Tensor, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Image `[B, 3, H, W]` and depth `[B, 1, h, w]` -> both padded to at least 32 and a multiple of 16."""
    h, w = image.shape[-2:]
    d = prompt.float()
    if tuple(d.shape[-2:]) != (h, w):
        d = F.interpolate(d, (h, w), mode="nearest")
    d = torch.where(torch.isfinite(d) & (d > 0), d, 0)
    # The first PVT attention stage applies an 8x8 reduction after a 4x downsample.
    # Pad small inputs to 32 as well as to the network's normal alignment.
    pads = (
        0,
        max(32, (w + MULTIPLE - 1) // MULTIPLE * MULTIPLE) - w,
        0,
        max(32, (h + MULTIPLE - 1) // MULTIPLE * MULTIPLE) - h,
    )
    if any(pads):
        mean = device_tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = F.pad(image - mean, pads) + mean
        d = F.pad(d, pads)
    return image, d


def _forward(
    net: OmniDCNet,
    x: torch.Tensor,
    dep: torch.Tensor,
    dav2: torch.Tensor,
) -> torch.Tensor:
    """Normalised padded image, padded sparse depth, Depth Anything V2 map -> depth `[B, 1, H, W]` (float32)."""
    net_in, small, mask = sparse_inputs(dep)
    guide, conf_spn, context, conf_in = net.backbone(x, torch.cat([net_in, dav2], 1))
    grad, up = net.update_block(context)
    d, b = system(grad.float(), small, mask, conf_in.float())
    log_depth = solve(d, b)
    depth = torch.exp(net.upsample(log_depth, up))
    return net.spn(depth, guide, dep, conf_spn)


class Portable(Runner):
    """Plain PyTorch with fused SDPA attention; BatchNorms folded. float32, or on CUDA fp16 networks without autocast.

    In fp16 the RGB-D backbone (its PVT features exceed the fp16 range; upstream's fp16 autocast overflows there),
    the convex upsampling, the DySPN propagation and the conjugate-gradient solve stay float32, as does the Depth
    Anything V2 head's last conv.
    """

    dtypes = ("float32", "float16")
    Params = Params
    batch = None
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = _net(ctx).to(ctx.device)
        sdpa_attention(net.depth_module.pretrained)
        fold_layer_scale(net.depth_module.pretrained)
        if ctx.dtype == "float16":
            head = net.depth_module.depth_head
            pure_half(net.depth_module, fp32=[head.scratch.output_conv2[2]])
            pure_half(net.update_block)  # the RGB-D backbone stays fp32: its PVT features reach ~1e6
            net.prop_layer.float()
        self.net = net

    def _run(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        x, dep = prepare(image, prompt)
        x = scale(x, self.ctx.io)
        return self._graph(x, dep)[:, 0, :h, :w]

    def _graph(self, x: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        """The network on the prepared image and sparse depth."""
        return _forward(self.net, x, dep, self.net.mono(x))

    def _pre(self, x: torch.Tensor, dep: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The network before the linear solve: the normal equations `(d, b)` and what the final steps need."""
        guide, conf_spn, context, conf_in = self._backbone(x, dep, self._mono(x))
        d, b, up = self._update(context, dep, conf_in)
        return d, b, up, guide, conf_spn

    def _mono(self, x: torch.Tensor) -> torch.Tensor:
        """The Depth Anything V2 map."""
        return self.net.mono(x)

    def _backbone(self, x: torch.Tensor, dep: torch.Tensor, dav2: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The RGB-D backbone (float32 in a float16 run: its PVT features reach ~1e6)."""
        net_in, _, _ = sparse_inputs(dep)
        return self.net.backbone(x, torch.cat([net_in, dav2], 1))

    def _update(self, context: torch.Tensor, dep: torch.Tensor, conf_in: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The update block and the normal equations `(d, b)` of the linear solve."""
        _, small, mask = sparse_inputs(dep)
        grad, up = self.net.update_block(context)
        d, b = system(grad.float(), small, mask, conf_in.float())
        return d.float(), b.float(), up

    def _post(
        self, log_depth: torch.Tensor, up: torch.Tensor, guide: torch.Tensor, dep: torch.Tensor, conf_spn: torch.Tensor
    ) -> torch.Tensor:
        """Convex upsampling and the propagation after the linear solve."""
        depth = torch.exp(self.net.upsample(log_depth, up))
        return self.net.spn(depth, guide, dep, conf_spn)

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its sparse depth maps `[B, 1, H, W]` (0 / NaN: missing)."""
        return {"depth": self._run(image, prompt_depth)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `([B, 3, H, W] in [0, 1], depth [B, 1, H, W])` -> depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image, prompt: self._run(image, prompt))


class _KernelCG(CGState):
    """The kernel-based CG solver, with one host check per chunk."""

    @staticmethod
    def normal(p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Apply the same three-scale FP32 operator in eager execution and graph replay."""
        return omni_cg.apply(p, d)

    CHUNK = 32  # fixed: steps per graph replay (one host sync each)

    def _advance(self) -> None:
        for _ in range(self.CHUNK):
            self.step()

    def run(self) -> torch.Tensor:
        """Solve for the current `d` and `b`, returning the state buffer."""
        self.reset()
        while True:
            self._advance()
            if bool(self.done):
                return self.x


class GraphCG(_KernelCG):
    """Replay CG chunks from a CUDA graph of the repeated solve shape."""

    def __init__(self, d: torch.Tensor, b: torch.Tensor) -> None:
        """Allocate the state for `b`'s shape and capture the graph."""
        super().__init__(d.clone(), b.clone())
        with torch.cuda.device(d.device), CAPTURE_LOCK:
            main = torch.cuda.current_stream()
            side = capture_stream(d.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.step()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=side):
                    try:
                        super()._advance()
                    except BaseException as e:
                        free_captured(e)  # before the capture ends (cudaMallocAsync would abort later)
                        raise
            main.wait_stream(side)

    def _advance(self) -> None:
        self.graph.replay()


class Kernels(Runner):
    """dpt_fused kernels for Depth Anything V2 Large (fused-epilogue GEMMs, flash attention, NHWC fp16 DPT head).

    The GRU step is the portable fp16 module, the RGB-D backbone float32 (its PVT features exceed the fp16 range); the
    conjugate-gradient solve is float32 replayed as CUDA graphs of 32 steps. The input resize for Depth Anything V2 is
    PyTorch's (bicubic, align_corners, as upstream), so the kernels' patch resize is the identity.
    """

    batch = None
    Params = Params
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        dev = ctx.device
        omni_cg.load(dev)
        k = load_dpt_fused(dev)  # before the weights: a kernel build / load failure costs no weight load
        net = _net(ctx)
        self.vit = FusedViT(net.depth_module.pretrained, dev, k, "hh")
        self.head = FusedHead(net.depth_module.depth_head, dev, k)
        del net.depth_module
        net.to(dev)
        pure_half(net.update_block)  # the RGB-D backbone stays fp32: its PVT features reach ~1e6
        self.net = net
        self.cg: GraphCG | None = None
        self._cg_seen: torch.Size | None = None
        self._cg_lock = threading.Lock()
        self._cg_done = torch.cuda.Event()

    def _solve(self, d: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Keep one shape's graph and order its static buffers across CUDA streams."""
        if not self.ctx.params.cuda_graphs:
            return _KernelCG(d, b).run()
        with self._cg_lock:
            if self.cg is not None:
                if self.cg.b.shape == b.shape:
                    torch.cuda.current_stream(b.device).wait_event(self._cg_done)
                else:
                    self._cg_done.synchronize()
                    self.cg = None
            if self.cg is None:
                if self._cg_seen != b.shape:
                    self._cg_seen = b.shape
                    return _KernelCG(d, b).run()
                self.cg = GraphCG(d, b)
            try:
                return _graph_solve(self.cg, d, b)
            finally:
                self._cg_done.record(torch.cuda.current_stream(b.device))

    def release_graphs(self) -> None:
        """Release the captured solver and its static tensors."""
        with self._cg_lock:
            self._cg_seen = None
            if self.cg is not None:
                self._cg_done.synchronize()
                self.cg = None

    def _mono(self, image01: torch.Tensor) -> torch.Tensor:
        h, w = image01.shape[-2:]
        size = mono_size(h, w)
        x = F.interpolate(image01, size, mode="bicubic", align_corners=True)
        feats = self.vit.features(x, size, _LAYERS)
        d = self.head(feats, size[0] // 14, size[1] // 14)[:, None]
        return quantile_normalise(F.interpolate(d, (h, w), mode="bilinear", align_corners=True))

    @torch.inference_mode()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its sparse depth maps `[B, 1, H, W]` (0 / NaN: missing)."""
        h, w = image.shape[-2:]
        with torch.autocast("cuda", enabled=False):
            x01, dep = prepare(image.float(), prompt_depth)
            dav2 = self._mono(x01)
            x = scale(x01, self.ctx.io)
            net = self.net
            net_in, small, mask = sparse_inputs(dep)
            guide, conf_spn, context, conf_in = net.backbone(x, torch.cat([net_in, dav2], 1))
            grad, up = net.update_block(context)
            d, b = system(grad.float(), small, mask, conf_in.float())
            log_depth = self._solve(d, b)
            depth = torch.exp(net.upsample(log_depth, up))
            out = net.spn(depth, guide, dep, conf_spn)
        return {"depth": out[:, 0, :h, :w]}


class _Mlx(
    mlx_wrap.port(  # ty: ignore[unsupported-base]
        Portable, attrs=(), methods=("_mono", "_backbone", "_update", "_post"), fp32_methods=("_backbone",), prune=True
    )
):
    """Apple MLX: the network before and after the linear solve; the conjugate-gradient solve stays PyTorch (CPU).

    As in the portable float16 run, the RGB-D backbone stays float32 (its PVT features overflow float16).
    """

    def _graph(self, x: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        d, b, up, guide, conf_spn = self._pre(x, dep)
        return self._post(solve(d, b), up, guide, dep, conf_spn)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _Mlx,
    "kernels_mlx": _Mlx,
}
