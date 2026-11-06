"""CoEx network, vendored from antabangun/coex (a98b37d, GPL-3.0-only).

Changes: the configuration (MobileNetV2-100 backbone, 192 max disparity, guided cost excitation, top-2 regression) is
fixed in the code and the card instead of a YAML file; the unused backbones, the training outputs and the no-correlation
branch are dropped; timm's `mobilenetv2_100` features come from `fastforks.shared.mobilenetv2` (same module names);
the spixel upsampling is copied in from `spixel_utils/spixel.py` (`upfeat`); the BatchNorm/conv initialisation and
debug imports are removed. Module names are unchanged, so the authors' checkpoint loads as is.
"""
