"""Selective-IGEV network, vendored from Windsrain/Selective-Stereo `Selective-IGEV/core/` at the pinned commit.

`igev_stereo.py` and `update.py` are upstream's: relative imports, `torch.autocast` instead of the deprecated
`torch.cuda.amp.autocast`, and the unused `opt_einsum` import removed. `geometry.py` is the IGEV-Stereo file and `extractor.py` / `submodule.py` come from
`fastforks.shared.igev_blocks` (Selective-IGEV's are the same code as IGEV-Stereo's), taken from our IGEV portable branch: relative imports, the
timm `mobilenetv2_100` features replaced by `fastforks.shared.mobilenetv2` (same module names, the checkpoints hold the
backbone weights), `bilinear_sampler` copied in, the missing `math` import added, plus the export-friendly
`build_gwc_volume_fast` / `FastGeoEncodingVolume`. `inference.py` is that branch's plain-PyTorch `fold_for_inference`.
"""
