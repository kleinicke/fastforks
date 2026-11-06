"""NMRF network, vendored from upstream (aeolusguan/NMRF main a50bd8f, `nmrf/models/`) + our fork's exportable inference path.

Changes to the vendored files (inference only): relative imports; no einops (the rearranges / repeats / einsums as
reshape / permute / expand / matmul, each marked); no detectron2-style `configurable` / yacs config (`nmrf.build` takes
a plain namespace with upstream's config keys, built by `model.py`); no weight initialisation, no DropPath (identity
at inference), no `device_indicator_tensor` buffer (dropped from the checkpoint by `model.py`); training-only code
dropped (criterion, aux outputs, intermediate returns, the Swin backbone + MSDeformAttn adaptor of the Swin-T models);
upstream's forward_pre / forward_post pairs as one forward; window sizes and token grids passed as arguments instead of
module attributes (no state mutated in forward); `prob_[mask] = eps` and the grid-sample coordinates without in-place
writes; `fill_diagonal_` and `torch.median` as indexed write / sort (export to ONNX); the CSWin attention as fused
`scaled_dot_product_attention` with the self-edge mask as bias and the attention masks cached per module
(`nmp.cached_mask`, bypassed while compiling) — both from our fork's `fast.py`; `NMRF.forward(img1, img2)` takes padded
images and returns the disparity (upstream pads itself and returns a dict). `inference.py` is the fork's general fp16
path (`nmrf/models/fast.py` without the fused window-attention kernel).
"""
