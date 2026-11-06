"""MVSAnywhere network modules, vendored from upstream nianticlabs/mvsanywhere at the card's commit.

`modules.py` holds the parts inference needs, from `src/mvsanywhere/modules/`: `layers.BasicBlock`,
`networks.MLP` / `ResnetMatchingEncoder`, `vit_modules.CostVolumePatchEmbed` and the `depth_anything_blocks.DPTHead`
(its fusion blocks are the shared Depth Anything V2 ones, which upstream copied). Changes: no loguru / timm /
torchvision; `ResnetMatchingEncoder` builds the antialiased ResNet-18 stem + layer1 itself (from `antialiased_cnns`,
Adobe, CC-BY-NC-SA-4.0: `BlurPool` and the stem layout); `DPTHead` keeps the scale-0 depth and sky heads only (the
others are not used at inference) and returns the logits. The DINOv2 ViTs are the shared vendored ones.
"""
