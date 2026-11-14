"""3D volume segmentation: `load(model_id)` returns a callable `VolumeModel` producing a `VolumeResult`.

```python
from fastforks import volume

ct, meta = volume.read_nifti("ct.nii.gz")  # [D, H, W] float32, RAS-canonical; needs fastforks-core[nifti]
model = volume.load("totalsegmentator-total")
result = model(ct, meta=meta)  # or spacing=(d, h, w) in mm
result.labels  # [D, H, W] uint8 labels on the input grid; result.class_names[k] names label k
result.save("labels.nii.gz")  # back in the file's orientation and affine (volume.write_nifti with meta)
```

`load_nnunet(path)` runs any trained nnU-Net v2 model folder or checkpoint (plain / residual U-Nets, 2D / 3D).
"""

from fastforks._core.cards import ModelCard
from fastforks.volume._api import (
    Backend,
    VolumeModel,
    VolumeResult,
    list_models,
    load,
    load_nnunet,
    model_card,
)
from fastforks.volume._nifti import VolumeMeta, read_nifti, write_nifti

__all__ = [
    "Backend",
    "ModelCard",
    "VolumeMeta",
    "VolumeModel",
    "VolumeResult",
    "list_models",
    "load",
    "load_nnunet",
    "model_card",
    "read_nifti",
    "write_nifti",
]
