"""3D volume segmentation: `load(model_id)` returns a callable `VolumeModel` producing a `VolumeResult`.

```python
from fastforks import volume

ct, meta = volume.read_nifti("ct.nii.gz")  # [D, H, W] float32, RAS-canonical; needs fastforks-core[nifti]
# ct, meta = volume.read_dicom("series_folder")  # same convention from a DICOM series; needs fastforks-core[dicom]
model = volume.load("totalsegmentator-total")
result = model(ct, meta=meta)  # or spacing=(d, h, w) in mm
result.labels  # [D, H, W] uint8 labels on the input grid; result.class_names[k] names label k
result.save("labels.nii.gz")  # back in the file's orientation and affine (volume.write_nifti with meta)
```

`load(path)` runs any trained nnU-Net v2 model folder or checkpoint (plain / residual U-Nets, 2D / 3D):
`volume.load("nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres", folds=[0], step=0.5, mirror=True)`.
Encoder models (`nnfoundation-cnn`, `nnfoundation-vit`) load as a `VolumeEncoder`: `model(ct).features` are the
pretrained multi-scale feature maps. Promptable models (`nninteractive`) load as a `VolumeInteractive`:
`s = model.session(ct)`, then `s.add_point((z, y, x))`, `s.add_box(...)`, `s.add_scribble(...)` refine one mask.
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.volume._api import (
    VolumeEncoder,
    VolumeFeatures,
    VolumeInteractive,
    VolumeMask,
    VolumeModel,
    VolumeResult,
    VolumeSession,
    list_models,
    load,
    model_card,
)
from fastforks.volume._nifti import VolumeMeta, read_dicom, read_nifti, write_nifti

__all__ = [
    "VolumeEncoder",
    "VolumeFeatures",
    "VolumeInteractive",
    "VolumeMask",
    "VolumeMeta",
    "VolumeModel",
    "VolumeResult",
    "VolumeSession",
    "list_models",
    "load",
    "model_card",
    "read_dicom",
    "read_nifti",
    "write_nifti",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
