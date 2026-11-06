"""Inception-v3: torchvision's network with the torchvision weights, run through `fastforks._optimize`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks.shared import libnet

if TYPE_CHECKING:
    import torch

    from fastforks.classification._runner import Context


def _network(_self: libnet.Portable | libnet.Kernels, ctx: Context) -> torch.nn.Module:
    """As `torchvision.models.inception_v3(weights=...)`: `transform_input`, auxiliary head dropped after loading.

    The weights expect `transform_input` and hold the auxiliary head; eval returns the main logits only.
    """
    net = libnet.build(ctx, transform_input=True, aux_logits=True, init_weights=False)
    net.aux_logits = False  # ty: ignore[invalid-assignment] - torchvision Inception3 attributes
    net.AuxLogits = None  # ty: ignore[invalid-assignment]
    return net


class Portable(libnet.Portable):
    """Plain PyTorch through `fastforks._optimize`."""

    network = _network


class Kernels(libnet.Kernels):
    """Our kernels through `fastforks._optimize`."""

    network = _network


RUNNERS = {"portable": Portable, "kernels": Kernels}
