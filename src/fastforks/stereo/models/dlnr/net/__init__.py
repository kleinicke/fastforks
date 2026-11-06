"""DLNR network, vendored from upstream (David-Zhao-1997/High-frequency-Stereo-Matching-Network c10475f, `core/`,
`nets/`) + the plain-PyTorch part of our fork's fast path.

Changes to the vendored files (inference only): relative imports; no einops; `dlnr.py` keeps the module tree (the
test-mode forward is `inference.forward`); `update.py` without the unused GRU blocks, the LSTM / update-block forwards
in `inference.py` (merged gate convs); `extractor.py`: the Restormer `TransformerBlock` forward on the NHWC view (our
fork's friendly restructuring, same maths), its LayerNorm / Attention / FeedForward modules hold the parameters only;
`corr.py`: our RAFT-Stereo fork's gather lookup instead of reg / reg_cuda (CUDA extension); `refinement.py`:
`nets/refinement.py` with the parts of `nets/feature.py` and `nets/warp.py` it uses, `disp_warp` without its
data-dependent assert and in-place writes. `mogrifier.py` (unused) and the training code are not vendored.
"""
