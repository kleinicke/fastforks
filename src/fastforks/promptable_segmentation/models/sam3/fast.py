"""Kernels image encoder of SAM 3: ViT trunk and neck in fp16 with fp16 accumulation (port of the FastViT).

- trunk: q / k / v packed into one GEMM, the residual stream kept in window order (9 windows of 24 x 24 tokens)
  between the global layers, RoPE as one in-place complex multiplication (torch), fastbb flash attention (`hh`: fp16
  scores and accumulation), dpt_fused GEMMs with fp16 accumulation, fused residuals and a bias + GELU epilogue
  (the GELU implementation follows the existing dpt_fused device policy)
- neck: the transposed and plain convolutions channels-last on `cudnn_hacc` (fp16 accumulation)

Input: the normalised `[1, 3, 1008, 1008]` image; output: the three neck levels in float16 (channels-last).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.half import fp16_accumulate
from fastforks.kernels import cudnn_hacc, fastbb, sam3_fpn, sam3_tracker
from fastforks.kernels.dpt_fused import DptFused
from fastforks.shared.host_blocks import HostBlocks

from .net.sam3 import GRID, HEAD, rope_tables

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .net.sam3 import FpnLayer, PixelDecoder, VisionEncoder, ViTLayer
    from .net.tracker import RoPEAttention


class _Rope:
    """Pairwise RoPE as one complex multiplication: `(x0 + i x1) * (cos + i sin)`, in place on q / k (complex32)."""

    def __init__(self, side: int, scale: float, device: torch.device) -> None:
        """Rotations `[1, N, 1, 1, 32]` for a `side x side` grid, positions times `scale`."""
        cos, sin = rope_tables(side, scale, device)
        n = side * side
        rot = torch.stack([cos[:, 0::2], sin[:, 0::2]], dim=-1).half()  # the tables repeat each angle twice
        self.rot = torch.view_as_complex(rot).view(1, n, 1, 1, HEAD // 2)

    def __call__(self, qkv: torch.Tensor) -> None:
        """Rotate q and k of the packed `[B, N, 3, H, 64]` in place."""
        p = torch.view_as_complex(qkv[:, :, :2].unflatten(-1, (HEAD // 2, 2)))
        p.mul_(self.rot)


def _bias(conv: torch.nn.Conv2d | torch.nn.ConvTranspose2d) -> torch.Tensor:
    if conv.bias is None:
        msg = "SAM 3 kernels encoder expects biased neck convolutions"
        raise BackendUnavailableError(msg)
    return conv.bias.detach()


class FastEncoder:
    """`VisionEncoder` (trunk + neck) in fp16 with fp16 accumulation, for the 1008 x 1008 input only."""

    def __init__(
        self, enc: VisionEncoder, device: torch.device, *, stream: bool = False, consume: bool = False
    ) -> None:
        """Convert the CPU float32 encoder to fp16 (packed q / k / v, channels-last convs) on `device`.

        Raises:
            BackendUnavailableError: An unexpected structure (head size, window grid), or a kernel library is unusable.
        """
        vit = enc.backbone
        layers = deque(cast("list[ViTLayer]", list(vit.layers)))
        dim, heads = int(layers[0].layer_norm1.normalized_shape[0]), int(layers[0].attention.heads)
        window = int(next(layer.window for layer in layers if layer.window))
        if dim // heads != HEAD or GRID % window:
            msg = f"SAM 3 kernels encoder needs head size {HEAD} and a {GRID}-token grid of whole windows"
            raise BackendUnavailableError(msg)
        fastbb.load(device)
        cudnn_hacc.load(device)
        self.device, self.dim, self.heads, self.window = device, dim, heads, window
        self.fused = DptFused(device)
        self.flash = fastbb.Flash64("hh")
        self.layout = [(layer.window, layer.layer_norm1.eps, layer.layer_norm2.eps) for layer in layers]
        storage = torch.device("cpu") if stream else device
        if consume:
            del vit.layers  # each independently mapped layer can die as soon as it has been converted
            self.layers = [self._layer(layers.popleft(), storage) for _ in range(len(layers))]
        else:
            self.layers = [self._layer(layer, storage) for layer in layers]
        self.host: HostBlocks | None = None
        if stream:
            # The packed projections feed the native GEMMs directly in both modes.
            tensors = {f"{i}.{k}": v for i, layer in enumerate(self.layers) for k, v in layer.items()}
            specs = [{f"{i}.{k}": (v.dtype, v.shape) for k, v in layer.items()} for i, layer in enumerate(self.layers)]

            def fill(name: str, dst: torch.Tensor) -> None:
                dst.copy_(tensors[name])

            self.host = HostBlocks(specs, fill)
            self.layers.clear()
        self.patch = vit.embeddings.patch_embeddings.projection.weight.detach().half().to(device)
        self.stride = int(vit.patch)
        self.pos = vit.embeddings.tiled(GRID, GRID).detach().half().to(device)  # [1, 72, 72, C]
        ln = vit.layer_norm
        self.ln_pre = (ln.weight.detach().half().to(device), ln.bias.detach().half().to(device), ln.eps)
        self.ropes = {window: _Rope(window, 1.0, device), 0: _Rope(GRID, window / GRID, device)}
        self.neck = [self._fpn(f, device) for f in cast("list[FpnLayer]", list(enc.neck.fpn_layers))]

    @staticmethod
    def _layer(layer: ViTLayer, device: torch.device) -> dict[str, torch.Tensor]:
        a, m = layer.attention, layer.mlp

        def h(t: torch.Tensor) -> torch.Tensor:
            return t.detach().half().to(device)

        return {
            "ln1_w": h(layer.layer_norm1.weight),
            "ln1_b": h(layer.layer_norm1.bias),
            "ln2_w": h(layer.layer_norm2.weight),
            "ln2_b": h(layer.layer_norm2.bias),
            "qkv_w": h(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])),
            "qkv_b": h(torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])),
            "o_w": h(a.o_proj.weight),
            "o_b": h(a.o_proj.bias),
            "fc1_w": h(m.fc1.weight),
            "fc1_b": h(m.fc1.bias),
            "fc2_w": h(m.fc2.weight),
            "fc2_b": h(m.fc2.bias),
        }

    @staticmethod
    def _fpn(f: FpnLayer, device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        cl = torch.channels_last
        ops = []
        for layer in f.scale_layers:
            if isinstance(layer, torch.nn.ConvTranspose2d):
                w = layer.weight.detach().half().to(device, memory_format=cl)
                ops.append(("deconv", w, _bias(layer).half().to(device).view(1, -1, 1, 1)))
            else:
                ops.append(("gelu", torch.empty(0), torch.empty(0)))
        for conv in (f.proj1, f.proj2):
            w = conv.weight.detach().half().to(device, memory_format=cl)
            ops.append(("conv", w, _bias(conv).half().to(device)))
        return ops

    def _weights(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.host is None:
            yield from self.layers
        else:
            for block in self.host.run(self.device):
                yield {key.split(".", 1)[1]: value for key, value in block.items()}

    def _block(self, p: dict[str, torch.Tensor], x: torch.Tensor, layout: tuple[int, float, float]) -> torch.Tensor:
        """One layer on `[B, N, C]` fp16 (windows: `B` = 9, `N` = 576; global: `B` = 1, `N` = 5184)."""
        b, n, c = x.shape
        window, eps1, eps2 = layout
        y = F.layer_norm(x, (c,), p["ln1_w"], p["ln1_b"], eps1)
        qkv = self.fused.gemm(y, p["qkv_w"], p["qkv_b"]).view(b, n, 3, self.heads, HEAD)
        del y
        self.ropes[window](qkv)
        o = self.flash(*qkv.unbind(2), HEAD**-0.5).view(b, n, c)
        del qkv
        x = self.fused.gemm(o, p["o_w"], p["o_b"], res=x, out=x)
        del o
        y = F.layer_norm(x, (c,), p["ln2_w"], p["ln2_b"], eps2).view(b * n, c)
        hid = self.fused.gemm(y, p["fc1_w"], p["fc1_b"], act=self.fused.gelu)
        return self.fused.gemm(hid.view(b, n, -1), p["fc2_w"], p["fc2_b"], res=x, out=x)

    def trunk(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised `[1, 3, 1008, 1008]` -> `[1, 72, 72, C]` fp16."""
        x = F.conv2d(image.half(), self.patch, None, stride=self.stride).permute(0, 2, 3, 1)
        _, h, w, c = x.shape
        if (h, w) != (GRID, GRID):
            msg = f"SAM 3 kernels encoder takes a {GRID * self.stride}-pixel square input, got {tuple(image.shape)}"
            raise BackendUnavailableError(msg)
        x = F.layer_norm(x + self.pos, (c,), *self.ln_pre)
        s, k = self.window, GRID // self.window

        def part(z: torch.Tensor) -> torch.Tensor:  # [1, 72, 72, C] -> [9, 576, C]
            return z.view(k, s, k, s, c).permute(0, 2, 1, 3, 4).reshape(k * k, s * s, c)

        def unpart(z: torch.Tensor) -> torch.Tensor:  # [9, 576, C] -> [1, 5184, C]
            return z.view(k, k, s, s, c).permute(0, 2, 1, 3, 4).reshape(1, h * w, c)

        windowed = False
        x = x.reshape(1, h * w, c)
        for p, layout in zip(self._weights(), self.layout, strict=True):
            if bool(layout[0]) != windowed:
                x = part(x.view(1, h, w, c)) if layout[0] else unpart(x)
                windowed = not windowed
            x = self._block(p, x, layout)
        if windowed:
            x = unpart(x)
        return x.view(1, h, w, c)

    @staticmethod
    def _level(ops: list[tuple[str, torch.Tensor, torch.Tensor]], x: torch.Tensor) -> torch.Tensor:
        for kind, w, b in ops:
            if kind == "deconv":
                x = cudnn_hacc.deconv_half_acc(x, w, (0, 0), (2, 2)).add_(b)
            elif kind == "gelu":
                x = F.gelu(x)
            else:
                pad = w.shape[-1] // 2
                x = cudnn_hacc.conv_half_acc(x, w, b, (pad, pad), (1, 1))
        return x

    def __call__(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Normalised `[1, 3, 1008, 1008]` -> the neck levels `[1, 256, 288 | 144 | 72, ...]` float16."""
        with fp16_accumulate():
            t = self.trunk(image).permute(0, 3, 1, 2)  # NHWC storage = channels-last NCHW
            return [self._level(ops, t) for ops in self.neck]


def pixel_decode(decoder: PixelDecoder, feats: list[torch.Tensor]) -> torch.Tensor:
    """Run the two fine mask stages while keeping each convolution result in its original storage."""
    x = feats[-1]
    for conv, norm, skip in zip(decoder.conv_layers, decoder.norms, reversed(feats[:-1]), strict=True):
        x = conv(sam3_fpn.nearest_add(x, skip))
        # Retain the reduction layout and fp16 rounding points of group_norm_relu.
        b, c, h, w = x.shape
        norm = cast("torch.nn.GroupNorm", norm)
        g = norm.num_groups
        if x.is_contiguous():
            v, m = torch.var_mean(x.view(b, g, -1), dim=2, correction=0)
        else:
            v, m = torch.var_mean(x.permute(0, 2, 3, 1).reshape(b, h * w, g, c // g), dim=(1, 3), correction=0)
        r = torch.rsqrt(v.float() + norm.eps)
        scale = (norm.weight.float().view(1, g, -1) * r[..., None]).view(b, c)
        shift = norm.bias.float() - (m.float()[..., None] * scale.view(b, g, -1)).view(b, c)
        sam3_fpn.affine_relu_(x, scale.half(), shift.half())
    return x


def tracker_attention(  # noqa: PLR0913, PLR0917 - the upstream attention signature
    self: RoPEAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    num_k_exclude_rope: int = 0,
) -> torch.Tensor:
    """Rotate the fresh fp16 projections in place, omitting each object's trailing pointer keys."""
    q = sam3_tracker.rotate_(self.q_proj(q), rope, q.shape[1])
    k = sam3_tracker.rotate_(self.k_proj(k), rope, k.shape[1] - num_k_exclude_rope)
    v = self.v_proj(v)
    return self.o_proj(F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0])
