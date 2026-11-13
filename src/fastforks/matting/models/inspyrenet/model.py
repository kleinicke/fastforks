"""InSPyReNet runner: weights -> network and the portable forward pass (no kernels runner: see the card).

Inference as transparent-background's `Remover.process` (what ComfyUI's InSPyReNet nodes call), static resize: the
image resized to `size x size` (PIL bilinear, antialiased; here `F.interpolate` with `antialias=True` on floats),
ImageNet-normalised, the network's finest saliency map `d0` (`threshold=None`: one pyramid, no LR/HR blending),
sigmoid, min-max normalised, resized back to the image (bilinear, `align_corners=True`). Upstream normalises min-max
over the whole batch (it runs one image); here per image.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16. In fp16 the attention modules (`SICA`) compute their
context vectors (a sum over all positions of a stage: up to 156 049 on the golden image) and the key / value convs
on them in float32; those convs are never cast to float16 (their batch-norm running stats exceed its range).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.patch import override
from fastforks.matting._runner import Forward, Runner, scale

from .net import InSPyReNet_SwinB
from .net.modules.attention_module import SICA

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        size: Network input side (square; transparent-background's `base_size`).
    """

    size: int


def _sica(m: SICA, x: torch.Tensor, smap: torch.Tensor, lmap: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """`SICA.forward` with the context vector and the key / value convs in float32."""
    b, _, h, w = x.shape
    p = torch.sigmoid(F.interpolate(smap, size=x.shape[-2:], mode="bilinear", align_corners=False)) - m.threshold
    prob = [torch.clip(p, 0, 1), torch.clip(-p, 0, 1), m.threshold - torch.abs(p)]
    if lmap is not None:  # SICA asserts that `lmap_in` modules get one
        lp = torch.sigmoid(F.interpolate(lmap, size=x.shape[-2:], mode="bilinear", align_corners=False)) - m.lthreshold
        prob += [torch.clip(lp, 0, 1), torch.clip(-lp, 0, 1)]
    shape = m.stage_size if m.stage_size is not None else (h, w)
    n = shape[0] * shape[1]
    f = F.interpolate(x, size=shape, mode="bilinear", align_corners=False).view(b, n, -1)  # upstream's view
    pr = F.interpolate(torch.cat(prob, dim=1), size=shape, mode="bilinear", align_corners=False).view(b, m.ctx, n)
    context = torch.bmm(pr.float(), f.float()).permute(0, 2, 1).unsqueeze(3)
    query = m.conv_query(x).view(b, m.depth, -1).permute(0, 2, 1)
    key = m.conv_key(context).to(x.dtype).view(b, m.depth, -1)
    value = m.conv_value(context).to(x.dtype).view(b, m.depth, -1).permute(0, 2, 1)
    sim = F.softmax((m.depth**-0.5) * torch.bmm(query, key), dim=-1)
    context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
    x = m.conv_out3(m.conv_out2(torch.cat([x, m.conv_out1(context)], dim=1)))
    return x, m.conv_out4(x)


def _net(ctx: Context, dtype: torch.dtype) -> torch.nn.Module:
    s = ctx.params.size
    net = InSPyReNet_SwinB(depth=64, pretrained=False, base_size=[s, s], threshold=None)
    net.load_state_dict(ctx.state_dict())
    torch.nn.Module.to(net, ctx.device)  # upstream's `to(device)` takes no dtype
    sica = [m for m in net.modules() if isinstance(m, SICA)] if dtype == torch.float16 else []
    keep = [(m, m.conv_key, m.conv_value) for m in sica]
    for m in sica:  # their batch-norm running stats overflow float16 (|mean| > 65504): converted, they turn NaN
        del m.conv_key, m.conv_value
    torch.nn.Module.to(net, dtype)
    for m, key, value in keep:
        m.conv_key, m.conv_value = key, value
        override(m, _sica)
    net.image_pyramid.kernel = net.image_pyramid.kernel.to(ctx.device, dtype)  # a plain tensor, not a buffer
    return net.eval()


class Portable(Runner):
    """Upstream's modules in plain PyTorch, float32 or float16."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("_net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _net(ctx, self.dtype)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_inspyre(x)["saliency"][3]  # ty: ignore[call-non-callable] - the upstream class

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False, antialias=True)
        y = torch.sigmoid(self._net(scale(x, self.ctx.io).to(self.dtype)).float())
        lo, hi = y.amin((1, 2, 3), keepdim=True), y.amax((1, 2, 3), keepdim=True)
        y = (y - lo) / (hi - lo + 1e-8)
        return {"alpha": F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=True)[:, 0]}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image, None)["alpha"])


RUNNERS = {"portable": Portable}
