"""timm building blocks with timm's module tree and parameter names, so timm-built checkpoints load unchanged.

`Mlp`, the pre-norm ViT `Block` (fused scaled-dot-product attention) and `create_vit_blocks(name)` for the
`vit_{tiny,small,base,large}_patch16_224` block stacks; `resnet_layers()` for the BasicBlock stages of timm's
ResNet-18 / -34. Randomly initialised: the weights come from the model's own checkpoint. See `meta.toml`.
"""

from __future__ import annotations

from .net.resnet import BasicBlock, make_layer, resnet_layers
from .net.vit import Attention, Block, Mlp, ViTBlocks, create_vit_blocks

__all__ = ["Attention", "BasicBlock", "Block", "Mlp", "ViTBlocks", "create_vit_blocks", "make_layer", "resnet_layers"]
