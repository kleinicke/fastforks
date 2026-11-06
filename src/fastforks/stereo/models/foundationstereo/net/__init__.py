"""FoundationStereo network, vendored from NVlabs/FoundationStereo main (6e88068) with import fixes only.

Changes to the vendored files: relative imports, `sys.path` hacks removed; `torch.autocast` instead of the deprecated
`torch.cuda.amp`; timm's EdgeNeXt replaced by the timm-free `fastforks.shared.edgenext` (same module names, so the
checkpoint's `feature.stem.*` / `feature.stages.*` keys load); the `torch.hub` DINOv2 + the Depth Anything DPT head
replaced by `fastforks.shared.depth_anything_v2` (same keys; fused-SDPA attention kept; the vendored `dinov2/` and
`depth_anything/` are removed); the `Utils` helpers inlined in `core/extractor.py`; unused huggingface / torchvision /
einops / scipy imports removed. `inference.py` is our fork's `core/inference.py` (folded BatchNorm, pure fp16, NHWC,
strip-wise cost volume; plain PyTorch ops only).
"""
