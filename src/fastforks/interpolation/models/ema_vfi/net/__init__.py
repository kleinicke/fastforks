"""EMA-VFI `MultiScaleFlow` + `MotionFormer` (MCG-NJU/EMA-VFI `model/`), vendored with small patches listed in each file."""

from .feature_extractor import MotionFormer
from .flow_estimation import MultiScaleFlow

__all__ = ["MotionFormer", "MultiScaleFlow"]
