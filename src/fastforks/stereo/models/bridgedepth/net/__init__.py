"""BridgeDepth network, vendored from upstream (aeolusguan/BridgeDepth main 9a23160, `bridgedepth/`) + our fork's exportable inference path.

Changes to the vendored files (inference only): relative imports; no einops (the rearranges / repeats as reshape /
permute / expand, each marked); no detectron2-style `configurable` / yacs `CfgNode` (the config is a plain namespace
built by `model.py`); no weight initialisation and no hub / ImageNet downloads (the checkpoint sets every parameter);
training-only code dropped (criterion, aux outputs, the unused `Inference` / `Refinement` / `SwinDPT` classes); the
Depth Anything V2 ViT and DPT head from `fastforks.shared.depth_anything_v2` (`mono.py`; upstream's copies are the
same modules); torchvision's ConvNeXt reduced to `convnext.py`; `FloatFunctional` adds as `+`; CSWin window sizes
passed as arguments instead of module attributes (no state mutated in forward); window-attention masks cached per
module (`nmp.cached_mask`, bypassed while compiling); `BridgeDepth.forward(img1, img2)` takes padded images and
returns the disparity (upstream pads itself and returns a dict). `inference.py` is the fork's upstream-friendly fp16
path (`bridgedepth/fp16.py`).
"""
