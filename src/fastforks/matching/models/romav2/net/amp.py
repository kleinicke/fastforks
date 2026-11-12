# fastforks: upstream runs the multi-view ViT, the DPT head, the VGG features and the refiner blocks under bf16
# autocast on every device. Here each of these modules has an `amp` attribute (default `None`: no autocast, float32
# as upstream with `enable_amp=False`) that the runner sets to the autocast dtype.
import contextlib

import torch


def autocast(x: torch.Tensor, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    """`torch.autocast(x.device.type, dtype)`, or nothing for `dtype=None`."""
    return contextlib.nullcontext() if dtype is None else torch.autocast(x.device.type, dtype)
