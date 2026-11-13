"""GeoCalib runners: the vendored network and its Levenberg-Marquardt camera fit.

The network (`net/geocalib.py`): MSCAN-L encoder, light Hamburger up / latitude heads; the fit: `net/lm_optimizer.py`.

Pre-processing as upstream `GeoCalib.calibrate`: the shorter side resized to `resize` (kornia: Gaussian antialiasing
blur, then bilinear), the result centre-cropped to multiples of `edge`; the fitted camera is mapped back to the input
image (`undo_scale_crop`).

Differences from upstream: the Hamburger heads' NMF starts from fixed bases (upstream draws new random bases per call:
results differ from run to run; the fixed ones are its draws after `torch.manual_seed(0)`, the same for every image of
a batch); the LM linear solve runs on the device (upstream copies each step's 3x3 systems to the CPU); the uncertainty
estimate (covariance) is not computed.

`portable`: plain PyTorch, the network in float32 or float16, the LM fit always in float32. `kernels`: the network in
float16 with fp16-accumulate convolutions (`cudnn_hacc`) and GEMMs, replayed from a CUDA graph per input shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.calibration._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net.geocalib import GeoCalib
from .net.modules import NMF2D

if TYPE_CHECKING:
    from fastforks.calibration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card / variant `params`.

    Attributes:
        resize: Shorter side of the network input.
        edge: The network input is cropped to multiples of this.
        camera_model: The LM fit's camera model (`pinhole`, `simple_radial`).
        cuda_graphs: Option: kernels replays the network from a CUDA graph.
    """

    resize: int
    edge: int
    camera_model: str
    cuda_graphs: bool


def _gauss(sigma: float, device: torch.device) -> torch.Tensor:
    """The 1-D Gaussian of kornia's of `resize(antialias=True)`: odd size `max(4 sigma, 3)`, normalised."""
    k = int(max(4.0 * sigma, 3))
    k += 1 - k % 2
    x = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    g = torch.exp(-(x**2) / (2.0 * sigma**2))
    return g / g.sum()


def _blur(x: torch.Tensor, gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """Separable blur with reflect borders (kornia `gaussian_blur2d`), x then y."""
    c = x.shape[1]
    ky, kx = gy.numel(), gx.numel()
    if kx > 1:
        x = F.pad(x, (kx // 2, kx // 2, 0, 0), mode="reflect")
        x = F.conv2d(x, gx.view(1, 1, 1, kx).expand(c, 1, 1, kx), groups=c)
    if ky > 1:
        x = F.pad(x, (0, 0, ky // 2, ky // 2), mode="reflect")
        x = F.conv2d(x, gy.view(1, 1, ky, 1).expand(c, 1, ky, 1), groups=c)
    return x


def preprocess(image: torch.Tensor, side: int, edge: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream `ImagePreprocessor({"resize": side, "edge_divisible_by": edge})` on `[B, 3, H, W]`.

    Returns:
        `(network input, scales [2] (x, y), crop_pad [2] (x, y, <= 0))`, the last two on the image's device.
    """
    h, w = image.shape[-2:]
    aspect = w / h
    nh, nw = (side, int(side * aspect)) if aspect >= 1 else (int(side / aspect), side)
    x = image
    fy, fx = h / nh, w / nw
    if max(fy, fx) > 1:  # kornia antialiasing: a Gaussian blur before the bilinear resize, only when shrinking
        sy, sx = max((fy - 1.0) / 2.0, 0.001), max((fx - 1.0) / 2.0, 0.001)
        x = _blur(x, _gauss(sy, x.device) if h > 1 else x.new_ones(1), _gauss(sx, x.device) if w > 1 else x.new_ones(1))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=None)
    pw, ph = (nw // edge) * edge - nw, (nh // edge) * edge - nh  # fit_to_multiple(crop=True): pads <= 0
    left, top = pw // 2, ph // 2
    x = x[..., -top : nh + ph - top, -left : nw + pw - left]
    dev = image.device
    scales = torch.tensor([nw / w, nh / h], dtype=torch.float32, device=dev)
    crop = torch.tensor([float(pw), float(ph)], dtype=torch.float32, device=dev)
    return x, scales, crop


def _fixed_bases(self: NMF2D, batch: int, s: int, *_: object, **__: object) -> torch.Tensor:
    """The fixed NMF start bases (`ff_bases`, `[1, D, R]`) for every image of the batch (upstream: random)."""
    bases: torch.Tensor = self.ff_bases  # ty: ignore[invalid-assignment]
    return bases.expand(batch * s, -1, -1)


def _no_uncertainty(self: torch.nn.Module, *_: object, **__: object) -> dict[str, torch.Tensor]:
    """No covariance / uncertainty estimate (upstream: inverse Hessian, eigenvalues on the CPU)."""
    del self
    return {}


def build(ctx: Context) -> GeoCalib:
    """The upstream network on the CPU in float32 with the variant's weights and fixed NMF bases (seed-0 draws)."""
    net = GeoCalib(camera_model=ctx.params.camera_model)
    load_strict(net, ctx.state_dict(key="model"), f"{ctx.variant.ref} checkpoint")
    gen = torch.Generator().manual_seed(0)
    for head in (net.perspective_decoder.up_head, net.perspective_decoder.latitude_head):  # upstream call order
        nmf = head.decoder.hamburger.ham
        bases = F.normalize(torch.rand((1, nmf.D, nmf.R), generator=gen), dim=1)
        nmf.register_buffer("ff_bases", bases, persistent=False)
        override(nmf, _fixed_bases, "_build_bases")
    net.eval().requires_grad_(requires_grad=False)
    override(net.optimizer, _no_uncertainty, "estimate_uncertainty")
    return net


class _GeoCalib(Runner):
    """Pre-processing, network, LM fit, camera of the input image."""

    Params = Params
    batch = None
    net: GeoCalib

    def _fields(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Up / latitude fields and confidences (float32) of the network input `x`."""
        raise NotImplementedError

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Camera and gravity of `[B, 3, H, W]` RGB in `[0, 1]`."""
        p = self.ctx.params
        x, scales, crop = preprocess(image, p.resize, p.edge)
        data = {k: v.float() for k, v in self._fields(x).items()} | {"scales": scales}
        with torch.autocast(image.device.type, enabled=False):
            out = self.net.optimizer(data)
        camera = out["camera"].undo_scale_crop({"scales": scales, "crop_pad": crop})
        gravity = out["gravity"]
        result = {
            "intrinsics": camera.K,
            "gravity": gravity.vec3d,
            "roll": gravity.roll,
            "pitch": gravity.pitch,
            "vfov": camera.vfov,
        }
        if "k1" in self.ctx.variant.outputs:
            result["k1"] = camera.k1
        return result

    def module(self) -> torch.nn.Module:
        """The network with its pre-processing and fit: `[B, 3, H, W]` in `[0, 1]` -> intrinsics `[B, 3, 3]`."""
        return Forward(self.net, lambda _net, image: self(image)["intrinsics"])


def _features(net: GeoCalib, x: torch.Tensor) -> dict[str, torch.Tensor]:
    data = {"image": x}
    features = {"hl": net.backbone(data)["features"], "ll": net.ll_enc(data)["features"]}
    return net.perspective_decoder({"features": features})


class Portable(_GeoCalib):
    """Plain PyTorch: the network in float32 or float16, the LM fit in float32."""

    dtypes = ("float32", "float16")
    compiled_methods = ("_fields",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = build(ctx)
        heads = (net.perspective_decoder.up_head, net.perspective_decoder.latitude_head)
        hams = [h.decoder.hamburger.ham for h in heads]
        bases = [h.ff_bases for h in hams]
        self.net = net.to(ctx.device, self.dtype)
        for ham, b in zip(hams, bases, strict=True):  # the NMF's multiplicative updates in float32 (sums over pixels)
            ham.ff_bases = b.to(ctx.device)
            override(ham, _nmf_fp32)

    def _fields(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _features(self.net, x.to(self.dtype))


def _nmf_fp32(self: NMF2D, x: torch.Tensor) -> torch.Tensor:
    """The NMF in float32 (returned in the input dtype)."""
    return NMF2D.forward(self, x.float()).to(x.dtype)


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16 network with fp16-accumulate convolutions (`cudnn_hacc`) and GEMMs, a CUDA graph per input shape."""

    module = Runner.module  # fp16-accumulate GEMMs are a cuBLAS mode, not an exportable operator

    def __init__(self, ctx: Context) -> None:
        """Build the fp16 network, route its dense convolutions through `cudnn_hacc`, set up the graph cache."""
        super().__init__(ctx)  # the card's kernels dtype is float16
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if is_base(m, torch.nn.Conv2d) and m.groups == 1 and m.padding_mode == "zeros" and m.padding != "same":
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with fp16_accumulate():
            return _features(self.net, x.half().contiguous(memory_format=torch.channels_last))

    def _fields(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = None if self.graphs is None else self.graphs.run((x.device, x.dtype, tuple(x.shape)), x)
        return self._eager(x) if out is None else out

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
