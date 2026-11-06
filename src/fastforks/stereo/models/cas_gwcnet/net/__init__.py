"""CasStereoNet networks, vendored from upstream alibaba/cascade-stereo at the card's commit (`CasStereoNet/models/`).

`gwcnet.py` is Cas-GwcNet-gc (group-wise correlation + concatenation volume), `psmnet.py` Cas-GwcNet-c (PSMNet-style
features, concatenation volume). Changes to the vendored files: explicit relative imports; no random weight init (the
checkpoint overwrites it) and no constructor print; `F.interpolate` instead of the deprecated `F.upsample` (same
values); the warp's `grid_sample` passes `align_corners=True`, the PyTorch 1.2 default the code was written and trained
with (current PyTorch defaults to False and shifts the warp by half a pixel). Inference does not call the networks'
`forward` (full-resolution [B, D, H, W] hypotheses and [B, C, D, H, W] warped features); see `model.py`.
"""
