"""TAPNext's one-frame convolution history and recurrent state on CUDA."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"history_update": "ppppppli", "history_eager": "ppppppli", "recurrent_update": "pppppppppli"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def history_update(
    x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, order: torch.Tensor, history: torch.Tensor
) -> torch.Tensor:
    """Convolve one half projection with its ring history, replacing the oldest entry in place."""
    conv = torch.empty_like(x, dtype=torch.float32)
    load(x.device).launch(
        "history_update",
        *linear(x.numel()),
        x,
        weights,
        bias,
        order,
        history,
        conv,
        x.numel(),
        x.shape[-1],
    )
    return conv


def history_eager(
    x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, history: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve one half projection and return the three latest entries in chronological order."""
    conv = torch.empty_like(x, dtype=torch.float32)
    next_history = x.new_empty((3, *x.shape[1:]))
    load(x.device).launch(
        "history_eager",
        *linear(x.numel()),
        x,
        weights,
        bias,
        history,
        next_history,
        conv,
        x.numel(),
        x.shape[-1],
    )
    return conv, next_history


def recurrent_update(  # noqa: PLR0913, PLR0917
    y: torch.Tensor,
    conv: torch.Tensor,
    gate: torch.Tensor,
    decay: torch.Tensor,
    a_param: torch.Tensor,
    first: torch.Tensor | None,
    h: torch.Tensor | None,
    *,
    inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the float32 hidden state and apply the half output projection gate."""
    out = torch.empty_like(y)
    state = h if h is not None and inplace else y.new_empty(y.shape[1:], dtype=torch.float32)
    load(y.device).launch(
        "recurrent_update",
        *linear(y.numel()),
        y,
        conv,
        gate.contiguous(),
        decay.contiguous(),
        a_param,
        first,
        h,
        state,
        out,
        y.numel(),
        y.shape[-1],
    )
    return out, state
