"""PSMNet network, vendored from upstream JiaRenChang/PSMNet at the card's commit (`models/`, stacked hourglass only).

Changes to the vendored files: explicit relative imports; no random weight init (the checkpoint overwrites it); the
feature network's pyramid pooling upsamples with `F.interpolate(align_corners=False)` (same values as the deprecated
`F.upsample`). Inference does not call `PSMNet.forward` (hard-coded `.cuda()`; its full-resolution trilinear
upsample needs 2 x 4.5 GB at Middlebury-F); see `model.py`.
"""
