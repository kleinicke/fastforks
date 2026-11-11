"""HRNet top-down pose network, condensed from mmpose 1.3.2 (`HRNet` backbone + `HeatmapHead`, Apache-2.0).

Four branches at 1/4 .. 1/32 of the crop exchange features after every module (`HRModule`): each branch is four
residual blocks, then every output is the sum of all branches resampled to its size (1 x 1 conv + nearest upsampling
upwards, stride-2 3 x 3 convs downwards). The head is one 1 x 1 conv on the 1/4 branch: a heatmap per keypoint.

BatchNorm is folded into the convolutions (`convert`); parameter names are the reference's conv names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

BLOCKS = 4  # fixed: residual blocks per branch and module (reference config)
MODULES = (1, 4, 3)  # fixed: modules of the stages with 2, 3 and 4 branches (reference config)
STEM = 64  # fixed: stem and first-stage width (reference)
BN_EPS = 1e-5  # fixed: reference BatchNorm eps
STRIDE = 4  # fixed: crop pixels per heatmap pixel


@dataclass(frozen=True)
class Arch:
    """Network size."""

    channels: int  # width of the 1/4 branch (branch b: channels * 2^b)
    keypoints: int
    height: int  # crop size
    width: int


def conv(cin: int, cout: int, k: int, stride: int = 1) -> nn.Conv2d:
    """A `k x k` conv with its folded BatchNorm as the bias."""
    return nn.Conv2d(cin, cout, k, stride, k // 2)


class BasicBlock(nn.Module):
    """Two 3 x 3 convs with an identity shortcut."""

    def __init__(self, c: int) -> None:
        """Width `c`."""
        super().__init__()
        self.conv1 = conv(c, c, 3)
        self.conv2 = conv(c, c, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`relu(x + conv2(relu(conv1(x))))`."""
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class Bottleneck(nn.Module):
    """1 x 1, 3 x 3, 1 x 1 convs to four times the width; the first block projects its shortcut."""

    def __init__(self, cin: int, c: int) -> None:
        """`cin` -> `4 c` channels."""
        super().__init__()
        self.conv1 = conv(cin, c, 1)
        self.conv2 = conv(c, c, 3)
        self.conv3 = conv(c, 4 * c, 1)
        self.downsample = nn.Sequential(conv(cin, 4 * c, 1)) if cin != 4 * c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The residual block."""
        y = self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(x)))))
        return torch.relu(y + self.downsample(x))


def _fuse(cj: int, ci: int, steps: int) -> nn.Module:
    """Branch j -> branch i: `steps` > 0 levels down (stride-2 convs), < 0 up (1 x 1 conv + nearest), 0 identity."""
    if steps == 0:
        return nn.Identity()
    if steps < 0:
        return nn.Sequential(conv(cj, ci, 1), nn.Upsample(scale_factor=2**-steps, mode="nearest"))
    down = [nn.Sequential(conv(cj, cj, 3, 2), nn.ReLU()) for _ in range(steps - 1)]
    return nn.Sequential(*down, nn.Sequential(conv(cj, ci, 3, 2)))


class HRModule(nn.Module):
    """`BLOCKS` residual blocks per branch, then each of the `outputs` first branches as the sum of all of them."""

    def __init__(self, channels: tuple[int, ...], outputs: int) -> None:
        """Branch widths `channels`."""
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(*(BasicBlock(c) for _ in range(BLOCKS))) for c in channels)
        self.fuse_layers = nn.ModuleList(
            nn.ModuleList(_fuse(cj, channels[i], i - j) for j, cj in enumerate(channels)) for i in range(outputs)
        )

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """The branches' maps, finest first."""
        xs = [b(x) for b, x in zip(self.branches, xs, strict=True)]
        out = []
        for fuse in self.fuse_layers:
            parts = [f(x) for f, x in zip(cast("nn.ModuleList", fuse), xs, strict=True)]
            y = parts[0]
            for part in parts[1:]:
                y = y + part
            out.append(torch.relu(y))
        return out


class HRNet(nn.Module):
    """Crops `[P, 3, h, w]` (normalised) -> heatmaps `[P, K, h / 4, w / 4]`."""

    def __init__(self, a: Arch) -> None:
        """Build the network for `a`."""
        super().__init__()
        self.conv1 = conv(3, STEM, 3, 2)
        self.conv2 = conv(STEM, STEM, 3, 2)
        self.layer1 = nn.Sequential(Bottleneck(STEM, STEM), *(Bottleneck(4 * STEM, STEM) for _ in range(BLOCKS - 1)))
        before = (4 * STEM,)
        for n, modules in enumerate(MODULES, start=1):
            channels = tuple(a.channels << b for b in range(n + 1))
            trans = nn.ModuleList()
            for b, c in enumerate(channels):
                if b == len(before):  # the new branch: half the size of the coarsest one
                    trans.append(nn.Sequential(nn.Sequential(conv(before[-1], c, 3, 2), nn.ReLU())))
                elif before[b] != c:
                    trans.append(nn.Sequential(conv(before[b], c, 3), nn.ReLU()))
                else:
                    trans.append(nn.Identity())
            setattr(self, f"transition{n}", trans)
            last = n == len(MODULES)  # the head reads the finest branch only
            stage = (HRModule(channels, 1 if last and m == modules - 1 else len(channels)) for m in range(modules))
            setattr(self, f"stage{n + 1}", nn.Sequential(*stage))
            before = channels
        self.head = nn.Conv2d(a.channels, a.keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Heatmaps of the crops."""
        ys = [self.layer1(torch.relu(self.conv2(torch.relu(self.conv1(x)))))]
        for n in range(1, len(MODULES) + 1):
            trans = getattr(self, f"transition{n}")
            # A branch that exists passes through (stage 2: the stem's map for both); the new one reads the coarsest
            ys = getattr(self, f"stage{n + 1}")([t(ys[min(b, len(ys) - 1)]) for b, t in enumerate(trans)])
        return self.head(ys[0])


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The reference checkpoint as `HRNet` weights: every BatchNorm folded into the conv before it.

    The BatchNorm of `convN` is `bnN`, of a conv at index 0 of a `Sequential` the entry at index 1.
    """
    sd = {
        k: v for k, v in sd.items() if not k.endswith("num_batches_tracked") and not k.startswith("data_preprocessor.")
    }
    out = {"head.weight": sd.pop("head.final_layer.weight"), "head.bias": sd.pop("head.final_layer.bias")}
    for key in [k for k in sd if k.endswith(".weight") and sd[k].ndim == 4]:  # noqa: PLR2004 - conv weights
        name = key.removeprefix("backbone.").removesuffix(".weight")
        stem, _, leaf = key.removesuffix(".weight").rpartition(".")
        bn = f"{stem}.{leaf.replace('conv', 'bn') if leaf.startswith('conv') else int(leaf) + 1}"
        w = sd.pop(key).double()
        gamma, beta, mean, var = (
            sd.pop(f"{bn}.{p}").double() for p in ("weight", "bias", "running_mean", "running_var")
        )
        scale = gamma / (var + BN_EPS).sqrt()
        out[f"{name}.weight"] = (w * scale[:, None, None, None]).float()
        out[f"{name}.bias"] = (beta - mean * scale).float()
    if sd:
        msg = f"unexpected checkpoint entries: {sorted(sd)[:5]}"
        raise KeyError(msg)
    return out
