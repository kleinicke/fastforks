"""WD Tagger v3: the timm network of the tagger's `config.json`, run through `fastforks._optimize`.

The reference script feeds BGR images (`inputs[:, [2, 1, 0]]`); the first convolution's input channels are swapped at
load instead (exact: the mean / std are the same for every channel), so the runner takes RGB like every classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.labels import label_set
from fastforks.shared import libnet

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks._core.runner import Context


@dataclass(frozen=True)
class Params(libnet.Params):
    """Card `[params]`: `arch` is the bare timm architecture, `model_args` the tagger's `config.json` `model_args`."""

    model_args: Mapping[str, Any]


def network(ctx: Context) -> torch.nn.Module:
    """The tagger network with its weights, its first convolution taking RGB."""
    classes = len(label_set(ctx.variant.output_info["label_set"]))
    net = libnet.build(ctx, num_classes=classes, **ctx.params.model_args)
    stem = next(m for m in net.modules() if isinstance(m, torch.nn.Conv2d))
    with torch.no_grad():
        stem.weight.copy_(stem.weight.flip(1))  # BGR -> RGB
    return net


class Portable(libnet.Portable):
    """Plain PyTorch through `fastforks._optimize`."""

    Params = Params
    network = staticmethod(network)


class Kernels(libnet.Kernels):
    """Our kernels through `fastforks._optimize`."""

    Params = Params
    network = staticmethod(network)


RUNNERS = {"portable": Portable, "kernels": Kernels}
