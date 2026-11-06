"""RAFT-Stereo network, vendored from our fork's `portable` branch (princeton-vl/RAFT-Stereo + exportable ops).

Changes to the vendored files: relative imports; `torch.autocast` instead of the deprecated `torch.cuda.amp`;
unused imports (`opt_einsum`, `scipy`, `numpy`) and the training-only `forward_interpolate` removed;
`reg_cuda` (`CorrBlockFast1D`, needs upstream's CUDA extension) removed: `reg_gather` is its PyTorch equivalent.
An explicit fp32 cast before the upsampling in `update.interp` (what autocast does; `torch.export` needs it).
"""
