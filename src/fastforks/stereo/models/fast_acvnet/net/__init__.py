"""Fast-ACVNet networks, vendored from upstream gangweix/Fast-ACVNet at the card's commit (`models/`).

`fast_acv.py` is Fast-ACVNet (attention weights from a 1/8 group-wise correlation volume, propagated and re-weighted
at 1/4), `fast_acv_plus.py` Fast-ACVNet+ (attention weights from a 1/4 correlation volume). Changes to the vendored
files: line endings, the relative import of `submodule`, the MobileNetV2 encoder from `fastforks.shared.mobilenetv2`
instead of timm (same module names and math). Inference does not call the networks' `forward`; see `model.py`.
"""
