"""GwcNet network, vendored from upstream xy-guo/GwcNet at the card's commit (`models/`).

Change to the vendored files: the relative import in `gwcnet.py`. Inference does not call `GwcNet.forward` (its
full-resolution trilinear upsample needs 2 x 4.5 GB at Middlebury-F); see `model.py`.
"""
