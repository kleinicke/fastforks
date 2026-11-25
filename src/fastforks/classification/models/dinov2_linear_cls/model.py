"""DINOv2 linear: timm's DINOv2 ViT with the official linear ImageNet head, run through `fastforks._optimize`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from fastforks.classification._runner import Runner
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


def _network(ctx: Context) -> LinearClassifier:
    """Backbone at the crop size (timm resamples the 518 px position embedding) + the head (weight file 1)."""
    vit = libnet.build(ctx, num_classes=0, img_size=ctx.io.crop)
    sd = ctx.state_dict(1, strip="")
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head.load_state_dict(sd)
    return LinearClassifier(vit, head).eval()


class Portable(libnet.Portable):
    """Plain PyTorch through `fastforks._optimize`."""

    network = staticmethod(_network)


class Kernels(libnet.Kernels):
    """Our kernels through `fastforks._optimize`."""

    network = staticmethod(_network)


class Mlx(Runner):
    """MLX DINOv2-L ImageNet linear probe."""

    Params = libnet.Params
    batch = None

    def __init__(self, ctx: Context) -> None:
        """Convert the same verified backbone and head as the portable model."""
        from fastforks.shared.mlx_vit import DinoLinearProbe  # noqa: PLC0415 - optional MLX dependency
        from fastforks.shared.mlx_wrap import lazy_build  # noqa: PLC0415 - with the MLX runner only

        super().__init__(ctx)
        net = lazy_build(lambda: _network(ctx))  # the weights stay the memory-mapped checkpoint until converted
        self.probe = DinoLinearProbe(net.vit, net.head.weight, net.head.bias, ctx.dtype, image_logits=True)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return image logits."""
        return {"logits": self.probe(image)}


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": Mlx}
