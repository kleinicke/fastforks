"""RT-MonSter++ network, vendored from upstream (Junda24/MonSter-plusplus main 9167101, `RT-MonSter++/core/`) + our exportable inference path.

Changes to the vendored files: relative imports; `torch.autocast` instead of the deprecated `torch.cuda.amp`; the Depth
Anything V2 branch (DINOv2 + DPT head) from `fastforks.shared.depth_anything_v2` (identical to upstream's
`Depth-Anything-V2-list3`; the checkpoint holds its weights, so the hard-coded `/home/lwj/...` load at build time is
dropped), `DPTHead_decoder` in `dpt.py` (the same file as MonSter's); torchvision's `Normalize` replaced by the same
arithmetic; `bilinear_sampler` copied in from `core/utils/utils.py` (`utils.py`); unused `opt_einsum` / `matplotlib` /
`time` imports and the unused REMP import removed; submodule.py converted from CRLF to LF line endings, its
`get_warped_feats` passes the default meshgrid indexing / grid_sample align_corners explicitly (deprecation warnings).
`inference.py` is the port of our fork's general path (`core/rt_fast.py`, attempt/general-fused 3baec33).
"""
