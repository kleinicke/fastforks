"""DINOv2 linear: timm's DINOv2 ViT with the official linear ImageNet head, run through `fastforks._optimize`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from fastforks.shared import libnet

if TYPE_CHECKING:
    from fastforks.classification._runner import Context


class LinearClassifier(torch.nn.Module):
    """`torch.hub` `dinov2_vitl14_lc` (layers=1): logits = head([class token; mean of the patch tokens])."""

    def __init__(self, vit: Any, head: torch.nn.Linear) -> None:  # noqa: ANN401 - timm VisionTransformer
        """Wrap the backbone (timm `VisionTransformer`, no classifier) and the linear head."""
        super().__init__()
        self.vit: Any = vit
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits `[B, 1000]` of normalised images."""
        t = self.vit.forward_features(x)  # [B, 1 + N, C] after the final norm (no register tokens)
        return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=1))


def _network(_self: libnet.Portable | libnet.Kernels, ctx: Context) -> torch.nn.Module:
    """Backbone at the crop size (timm resamples the 518 px position embedding) + the head (weight file 1)."""
    vit = libnet.build(ctx, num_classes=0, img_size=ctx.io.crop)
    sd = ctx.state_dict(1, strip="")
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head.load_state_dict(sd)
    return LinearClassifier(vit, head).eval()


class Portable(libnet.Portable):
    """Plain PyTorch through `fastforks._optimize`."""

    network = _network


class Kernels(libnet.Kernels):
    """Our kernels through `fastforks._optimize`."""

    network = _network


RUNNERS = {"portable": Portable, "kernels": Kernels}
