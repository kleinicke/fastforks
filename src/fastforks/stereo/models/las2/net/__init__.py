"""LiteAnyStereo V2 (LAS2) network, vendored from TomTomTommi/LiteAnyStereo `core/` at the pinned commit.

Changes to the vendored files: `fnet.py` builds the FasterNet-T0 feature pyramid with `fastforks.shared.fasternet`
instead of `timm.create_model(..., features_only=True)` (same module names, the checkpoints hold the backbone weights;
the unused LAS1 `FeatureNet` that needs timm's mobilenetv2 is removed). Nothing else differs.
"""
