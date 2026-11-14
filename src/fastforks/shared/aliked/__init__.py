"""ALIKED (Shiaoming/ALIKED as cvg/LightGlue vendors it, in `net/`): the keypoint detector and descriptor.

`ALIKED(deform_op, model_name=..., max_num_keypoints=..., detection_threshold=..., nms_radius=...)` builds the
network (no weights; the caller loads the release state dict strictly). `deform_op` is the modulated deformable conv
(the families pass `fastforks.shared.deform_conv.modulated_deform_conv`; shared modules do not import each other).
Its call takes `{"image": [B, 3, H, W]}` in [0, 1] (the caller resizes) and returns `keypoints` (pixels, centres at
integers), `descriptors` and `keypoint_scores`. Used by the `lightglue` (ALIKED + LightGlue) and `flexglue` (ALIKED
keypoints + FlexGlue) matching families.
"""

from .net.aliked import ALIKED

__all__ = ["ALIKED"]
