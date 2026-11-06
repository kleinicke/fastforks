"""AANet network, vendored from upstream haofeixu/aanet at the card's commit (`nets/`).

Changes to the vendored files: the absolute `nets.*` imports became relative ones, and `deform_conv.py` replaces
upstream's `nets/deform_conv/` (mmdetection-v1 CUDA extension) by plain PyTorch ops with the same parameters,
state-dict keys, offset layout and mask semantics. `warp.py` passes `align_corners=False` (the torch default, which
warned when implicit) to its two `grid_sample` calls and drops the data-dependent `assert disp.min() >= 0` of `disp_warp`
(not exportable); `disp_warp` samples in float32 (float16 cannot hold the pixel columns past 2048).
"""
