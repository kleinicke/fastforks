"""Prompt Depth Anything network: the Depth Anything V2 ViT + DPT head with a depth prompt fused into every refinement.

Same architecture as upstream `promptda/model/dpt.py` (Apache-2.0, (c) Haotong Lin et al.): each of the four fusion
blocks adds `resConfUnit_depth(prompt resized to the block's resolution)` after its second residual conv unit, and the
output activation is a sigmoid (the prompt is normalised to [0, 1] by its own min / max, the output undone with them).
The ViT and the unchanged DPT blocks come from `fastforks.shared.depth_anything_v2`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.depth_anything_v2 import DPTHead, FeatureFusionBlock
from fastforks.shared.depth_anything_v2.net.dinov2 import DINOv2

LAYERS = {"vits": (2, 5, 8, 11), "vitb": (2, 5, 8, 11), "vitl": (4, 11, 17, 23)}  # upstream model_configs layer_idxs


class PromptFusionBlock(FeatureFusionBlock):
    """`FeatureFusionBlock` plus the prompt branch (`resConfUnit_depth`: conv, ReLU, conv, ReLU, conv; 1 -> F)."""

    def __init__(self, features: int) -> None:
        """Build a block of `features` channels (ReLU units, bilinear `align_corners=True`, as the DPT head)."""
        super().__init__(features, nn.ReLU(inplace=False), deconv=False, bn=False, expand=False, align_corners=True)
        self.resConfUnit_depth = nn.Sequential(
            nn.Conv2d(1, features, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, 3, 1, 1),
        )

    def forward(
        self, *xs: torch.Tensor, size: tuple[int, int] | None = None, prompt: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`xs[0] (+ rcu1(xs[1]))`, rcu2, `+ rcu_depth(prompt)`, bilinear to `size` (default x2), 1x1 conv."""
        assert prompt is not None  # noqa: S101 - the head always passes it
        out = xs[0]
        if len(xs) == 2:  # noqa: PLR2004 - (path, skip connection)
            out = out + self.resConfUnit1(xs[1])
        out = self.resConfUnit2(out)
        p = F.interpolate(prompt, out.shape[2:], mode="bilinear", align_corners=False)
        out = out + self.resConfUnit_depth(p)
        out = F.interpolate(out, size=size, scale_factor=None if size else 2, mode="bilinear", align_corners=True)
        return self.out_conv(out)


class PromptHead(DPTHead):
    """The DPT head with prompt-fed fusion blocks and a sigmoid output (`[B, 1, 14 ph, 14 pw]`)."""

    def __init__(self, in_channels: int, features: int, out_channels: list[int]) -> None:
        """Build the head; the weights are those of the DPT head plus the four `resConfUnit_depth`."""
        super().__init__(in_channels, features, use_bn=False, out_channels=out_channels, use_clstoken=False)
        for i in range(1, 5):
            setattr(self.scratch, f"refinenet{i}", PromptFusionBlock(features))
        self.scratch.output_conv2[3] = nn.Sigmoid()

    def _project(self, feat: torch.Tensor, i: int, patch_h: int, patch_w: int) -> torch.Tensor:
        x = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-1], patch_h, patch_w)
        return self.resize_layers[i](self.projects[i](x))

    def forward(self, out_features: list, patch_h: int, patch_w: int, prompt: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        """Features (`get_intermediate_layers` tuples) of 4 layers and the normalised prompt -> `[B, 1, H, W]`."""
        s = self.scratch
        # Prepare each skip near its refinement so only the current skip and the next one stay live.
        l4 = s.layer4_rn(self._project(out_features[3][0], 3, patch_h, patch_w))
        l3 = s.layer3_rn(self._project(out_features[2][0], 2, patch_h, patch_w))
        p = s.refinenet4(l4, size=l3.shape[2:], prompt=prompt)
        del l4
        l2 = s.layer2_rn(self._project(out_features[1][0], 1, patch_h, patch_w))
        p = s.refinenet3(p, l3, size=l2.shape[2:], prompt=prompt)
        del l3
        l1 = s.layer1_rn(self._project(out_features[0][0], 0, patch_h, patch_w))
        p = s.refinenet2(p, l2, size=l1.shape[2:], prompt=prompt)
        del l2
        p = s.refinenet1(p, l1, prompt=prompt)
        del l1
        p = s.output_conv1(p)
        o = F.interpolate(p, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        return s.output_conv2(o)


class PromptDepthAnything(nn.Module):
    """Upstream `PromptDA` without the `_mean` / `_std` buffers (the runners normalise): ViT + prompt head."""

    def __init__(self, encoder: str, features: int, out_channels: list[int]) -> None:
        """Build the network for `encoder` (`"vits"` / `"vitb"` / `"vitl"`)."""
        super().__init__()
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = PromptHead(self.pretrained.embed_dim, features, out_channels)
        self.layers = LAYERS[encoder]

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised `[B, 3, 14 ph, 14 pw]` and the prompt scaled to [0, 1] `[B, 1, h, w]` -> `[B, H, W]`."""
        feats = self.pretrained.get_intermediate_layers(x, self.layers, return_class_token=True)
        return self.depth_head(feats, x.shape[-2] // 14, x.shape[-1] // 14, prompt)[:, 0]
