"""`torch.nn.functional` for the vendored network, with the deprecated `upsample` as `interpolate` (same arguments).

`F.upsample(x, size, mode="bilinear")` warns on every call; `interpolate` with the same arguments (`align_corners`
None = False) gives the same result.
"""

from torch.nn.functional import interpolate, sigmoid

upsample = interpolate

__all__ = ["interpolate", "sigmoid", "upsample"]
