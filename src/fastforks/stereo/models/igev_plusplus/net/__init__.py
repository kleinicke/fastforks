"""IGEV++ network, vendored from gangweix/IGEV-plusplus `core/` at the pinned commit.

`igev_stereo.py`, `update.py`, `geometry.py` and `submodule.py` are upstream's: relative imports, `torch.autocast`
instead of the deprecated `torch.cuda.amp.autocast`, the unused `time` / `numpy` imports removed, `bilinear_sampler`
copied in from `core/utils/utils.py`, and `context_upsample`'s `F.unfold` replaced by shifted slices (same values,
exports to ONNX / TensorRT). `extractor.py` is `fastforks.shared.igev_blocks` (IGEV++'s `MultiBasicEncoder` /
`Feature` are IGEV-Stereo's code), with the timm `mobilenetv2_100` features replaced by `fastforks.shared.mobilenetv2`
(same module names, the checkpoints hold the backbone weights). Added for inference: `build_gwc_volume_fast`
(`submodule.py`, from `fastforks.shared.igev_blocks`), `FastGeoEncodingVolume` (`geometry.py`) and `inference.py`
(plain-PyTorch `fold_for_inference`).
"""
