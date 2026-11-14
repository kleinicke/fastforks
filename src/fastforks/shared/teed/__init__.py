"""TEED (Soria et al., ICCV 2023 workshops): the network and the soft-edge map of its four outputs.

`edge_map` is controlnet_aux's `TEEDdetector` / `AnylineDetector` post-processing in float before `safe_step`: the
outputs resized to the input size (bilinear) and the sigmoid of their mean.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .net import TED

__all__ = ["TED", "edge_map"]


def edge_map(outputs: list[torch.Tensor]) -> torch.Tensor:
    """TED's four outputs `[B, 1, h, w]` (the first three at the input size) -> the map `[B, h, w]` in `[0, 1]`."""
    size = outputs[0].shape[-2:]
    acc = outputs[0][:, 0].float()
    for out in outputs[1:]:
        e = out.float()
        acc = acc + (e if e.shape[-2:] == size else F.interpolate(e, size, mode="bilinear", align_corners=False))[:, 0]
    return torch.sigmoid(acc / len(outputs))
