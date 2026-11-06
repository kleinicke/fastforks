"""MonSter network, vendored from upstream (Junda24/MonSter main 49016c9, `core/`) + our fork's exportable inference path.

Changes to the vendored files: relative imports; `torch.autocast` instead of the deprecated `torch.cuda.amp`; the Depth Anything V2 branch (DINOv2 + DPT head) from
`fastforks.shared.depth_anything_v2` (identical to upstream's `Depth-Anything-V2-list3`; the checkpoints hold its
weights, so the `./pretrained` load at build time is dropped), MonSter's own `DPTHead_decoder` moved to `dpt.py`;
`bilinear_sampler` copied in from `core/utils/utils.py` (`utils.py`); unused `opt_einsum` / `matplotlib` / `numpy`
imports removed, PIL imported only in the debug image writer. Appended inference-only helpers from our IGEV / MonSter
forks: `build_gwc_volume_fast` (submodule.py) and `FastGeoEncodingVolume` (geometry.py). `inference.py` is the fork's
upstream-friendly tier on the IGEV-family `fold_for_inference`.
"""
