"""RepViT: the timm network with the variant's weights, run through `fastforks._optimize`."""

from __future__ import annotations

from fastforks.shared import libnet

RUNNERS = {"portable": libnet.Portable, "kernels": libnet.Kernels}
