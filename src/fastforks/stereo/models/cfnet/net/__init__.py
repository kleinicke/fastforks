"""CFNet network, vendored from upstream gallenszl/CFNet at the card's commit (`models/`).

Changes to the vendored files: relative imports; no random weight init (the checkpoint overwrites it); `Mish` without
its load-time print; `pyramidPooling` computes its pool sizes in plain Python (same integers as `np.linspace(...,
dtype=int)`) and upsamples with `F.interpolate` (same values as the deprecated `F.upsample`); the disparity sampler and
the warp use `tensor.device` instead of `get_device()` (which is -1 on the CPU); the warp's out-of-range mask
is a logical or instead of a bool sum (same values; ONNX has no bool Add); the unused `warp` (hard-coded
`.cuda()`) and `build_corrleation_volume` are removed. Inference does not call `cfnet.forward`; see `model.py`.
"""
