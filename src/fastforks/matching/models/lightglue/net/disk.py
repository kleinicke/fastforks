# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# fastforks: kornia 0.8.3 kornia/feature/disk/{disk,detector,_unets/blocks,_unets/unet}.py in one file, with
# kornia.geometry.subpix.nms2d (5x5 path) inlined; DISKFeatures / Keypoints structs and from_pretrained removed
# (forward returns (keypoints, scores, descriptors) lists), the U-Net's size checks kept.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- _unets/blocks.py


class TrivialUpsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class TrivialDownsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2)


class Conv(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int, skip_norm_and_gate: bool = False) -> None:
        norm: nn.Module
        nonl: nn.Module

        if skip_norm_and_gate:
            norm = nn.Sequential()
            nonl = nn.Sequential()
        else:
            norm = nn.InstanceNorm2d(in_)
            nonl = nn.PReLU(in_)

        dropout = nn.Sequential()
        conv = nn.Conv2d(in_, out_, size, padding="same", bias=True)

        super().__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int = 5, is_first: bool = False) -> None:
        self.in_ = in_
        self.out_ = out_

        downsample: nn.Module
        if is_first:
            downsample = nn.Sequential()
            conv = Conv(in_, out_, size, skip_norm_and_gate=True)
        else:
            downsample = TrivialDownsample()
            conv = Conv(in_, out_, size)

        super().__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    def __init__(self, bottom_: int, horizontal_: int, out_: int, size: int = 5) -> None:
        super().__init__()

        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = TrivialUpsample()
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: torch.Tensor, hor: torch.Tensor) -> torch.Tensor:
        bot_big = self.upsample(bot)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)


# ---------------------------------------------------------------- _unets/unet.py


class Unet(nn.Module):
    def __init__(
        self, in_features: int = 1, up: Optional[list[int]] = None, down: Optional[list[int]] = None, size: int = 5
    ) -> None:
        super().__init__()
        if up is None:
            up = []
        self.up = up
        if down is None:
            down = []
        self.down = down
        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.in_features = in_features

        down_dims = [in_features, *down]
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            down_block = ThinUnetDownBlock(d_in, d_out, size=size, is_first=i == 0)
            self.path_down.append(down_block)

        bot_dims = [down[-1], *up]
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for _, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            up_block = ThinUnetUpBlock(d_bot, d_hor, d_out, size=size)
            self.path_up.append(up_block)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.size(1) != self.in_features:
            fmt = "Expected {} feature channels in input, got {}"
            msg = fmt.format(self.in_features, inp.size(1))
            raise ValueError(msg)

        input_size_divisor = 2 ** len(self.up)
        if (inp.size(2) % input_size_divisor != 0) or (inp.size(3) % input_size_divisor != 0):
            raise ValueError(
                f"Input image shape must be divisible by {input_size_divisor} (got {inp.size()}). "
                "This is not inherent to DISK, but to the U-Net architecture used in pretrained models. "
                "Please F.pad if necessary."
            )

        features = [inp]
        for layer in self.path_down:
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        return f_bot


# ---------------------------------------------------------------- detector.py


def nms2d_mask(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """fastforks: kornia nms2d(x, (k, k), mask_only=True) for odd k: strictly above all neighbours, borders False."""
    r = window_size // 2
    h, w = x.shape[-2:]
    ct = x[..., r : h - r, r : w - r]
    keep = torch.ones_like(ct, dtype=torch.bool)
    for dy in range(window_size):
        for dx in range(window_size):
            if dy != r or dx != r:
                keep &= ct > x[..., dy : h - 2 * r + dy, dx : w - 2 * r + dx]
    mask = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
    mask[..., r : h - r, r : w - r] = keep
    return mask


def heatmap_to_keypoints(
    heatmap: torch.Tensor, n: int | None = None, window_size: int = 5, score_threshold: float = 0.0
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Inference-time nms-based detection protocol."""
    heatmap = heatmap.squeeze(1)
    nms_mask = nms2d_mask(heatmap.unsqueeze(1), window_size).squeeze(1)  # fastforks: inlined nms2d

    nmsed = nms_mask & (heatmap > score_threshold)

    keypoints = []
    for b in range(heatmap.shape[0]):
        yx = nmsed[b].nonzero(as_tuple=False)
        detection_logp = heatmap[b][nmsed[b]]
        xy = yx.flip((1,))

        if n is not None:
            n_ = min(n + 1, detection_logp.numel())
            # torch.kthvalue picks in ascending order and we want to pick in
            # descending order, so we pick n-th smallest among -logp to get
            # -threshold
            minus_threshold, _indices = torch.kthvalue(-detection_logp, n_)
            mask = detection_logp > -minus_threshold

            xy = xy[mask]
            detection_logp = detection_logp[mask]

            # it may be that due to numerical saturation on the threshold we have
            # more than n keypoints, so we need to clip them
            xy = xy[:n]
            detection_logp = detection_logp[:n]

        keypoints.append((xy, detection_logp))  # fastforks: a tuple instead of the Keypoints struct

    return keypoints


# ---------------------------------------------------------------- disk.py


class DISK(nn.Module):
    def __init__(self, desc_dim: int = 128, unet: nn.Module | None = None) -> None:
        super().__init__()

        self.desc_dim = desc_dim

        if unet is None:
            unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, desc_dim + 1])
        self.unet = unet

    def heatmap_and_dense_descriptors(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unet_output = self.unet(images)

        if unet_output.shape[1] != self.desc_dim + 1:
            raise ValueError(
                f"U-Net output has {unet_output.shape[1]} channels, but expected self.desc_dim={self.desc_dim} + 1."
            )

        descriptors = unet_output[:, : self.desc_dim]
        heatmaps = unet_output[:, self.desc_dim :]

        return heatmaps, descriptors

    def forward(
        self,
        images: torch.Tensor,
        n: int | None = None,
        window_size: int = 5,
        score_threshold: float = 0.0,
        pad_if_not_divisible: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B = images.shape[0]
        if pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = F.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descriptors = self.heatmap_and_dense_descriptors(images)
        if pad_if_not_divisible:
            heatmaps = heatmaps[..., :h, :w]
            descriptors = descriptors[..., :h, :w]

        keypoints = heatmap_to_keypoints(heatmaps, n=n, window_size=window_size, score_threshold=score_threshold)

        # fastforks: Keypoints.merge_with_descriptors inlined; (keypoints, detection scores, descriptors) per image
        features = []
        for i in range(B):
            xys, logp = keypoints[i]
            dtype = descriptors.dtype
            x, y = xys.T
            desc = F.normalize(descriptors[i][:, y, x].T, dim=-1)
            features.append((xys.to(dtype), logp.to(dtype), desc))

        return features
