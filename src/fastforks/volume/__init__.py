"""3D volume segmentation: `load(model_id)` returns a callable `VolumeModel` producing a `VolumeResult`.

```python
from fastforks import volume

ct, meta = volume.read_nifti("ct.nii.gz")  # [D, H, W] float32, RAS-canonical; needs fastforks-core[nifti]
model = volume.load("totalsegmentator-total")
result = model(ct, meta.spacing)
result.labels  # [D, H, W] uint8 labels on the input grid; result.class_names[k] names label k
volume.write_nifti("labels.nii.gz", result.labels, meta)  # back in the file's orientation and affine
```

`load_nnunet(folder)` runs any trained nnU-Net v2 model folder (PlainConvUNet, 3D) on the same engine.
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
