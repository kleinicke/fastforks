"""IGEV-Stereo network, vendored from our fork's `portable` branch (gangweix/IGEV main + exportable inference path).

Changes to the vendored files: relative imports; `torch.autocast` instead of the deprecated `torch.cuda.amp`;
timm's `mobilenetv2_100` features replaced by `fastforks.shared.mobilenetv2` (same module names, the checkpoints
hold the backbone weights); `bilinear_sampler` copied in from `core/utils/utils.py`; unused `time` / `numpy` imports
removed, the missing `math` import of `SubModule.weight_init` added. `inference.py` is the branch's
`core/inference.py` (plain-PyTorch `fold_for_inference`).
"""
