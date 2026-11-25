"""Prepare a timm vision transformer with CUDA transformer weights staged from host RAM."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch

from fastforks._optimize import optimize, transformer
from fastforks.shared.host_blocks import StreamedModuleList

if TYPE_CHECKING:
    from fastforks._optimize import FastModel
    from fastforks.features._runner import Context


def streamed_vit(net: Any, ctx: Context) -> FastModel:  # noqa: ANN401 - timm ViT or Eva
    """Optimize embeddings and head with one representative block before installing the host-backed blocks."""
    blocks = net.blocks
    net.blocks = torch.nn.ModuleList([copy.deepcopy(blocks[0])])
    blocks.to(dtype=ctx.torch_dtype)
    if type(net).__name__ == "Eva":
        transformer.patch_pe(blocks, ctx.mode)
    else:
        transformer.patch_vit_blocks(blocks, ctx.mode)
    staged = StreamedModuleList(blocks, ctx.device)
    fast = optimize(net, mode=ctx.mode, device=ctx.device, fp16=ctx.dtype == "float16")
    net.blocks = staged
    return fast
