"""`conv(nearest_2x(x))` as one stride-2 transposed conv of `x` (fastforks).

Nearest 2x upsampling then a 3x3 conv (padding 1), along one axis: output row 2m reads input rows m - 1 (tap w0) and
m (w1 + w2), row 2m + 1 reads m (w0 + w1) and m + 1 (w2). That is a stride-2, padding-1 transposed conv with 4 taps,
each the sum of the conv taps marked in `_UP_TAPS`: 4 instead of 9 multiply-adds per output pixel and input channel,
and no upsampled input map.
"""

from __future__ import annotations

import torch

_UP_TAPS = ((0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0))  # fixed: deconv tap k = sum of the conv taps marked 1


def up_deconv_weight(w: torch.Tensor) -> torch.Tensor:
    """The `[C, K, 4, 4]` stride-2 padding-1 transposed-conv weight of `conv(nearest_2x(x))` for a 3x3 conv weight `w`.

    The taps are summed in float64 and rounded once to `w`'s dtype.
    """
    a = torch.tensor(_UP_TAPS, dtype=torch.float64, device=w.device)
    return torch.einsum("ya,xb,oiab->ioyx", a, a, w.double()).to(w.dtype)
