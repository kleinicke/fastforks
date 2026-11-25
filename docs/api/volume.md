# Volume segmentation API

_These pages are rendered by mkdocs (`uv run --group docs mkdocs serve`); in a plain viewer read the same text with `python -c "import fastforks; help(fastforks.volume)"`._

`load`, `list_models` and `model_card` have the signatures documented under [task functions](index.md#task-functions).

## nnU-Net v2

Load a trained model folder or a checkpoint with embedded plans. The plans and dataset metadata define the architecture, target spacing, input channels and output labels. The checkpoint provider determines the weight licence.

```python
from fastforks import volume

model = volume.load("./nnUNetTrainer__nnUNetPlans__3d_fullres", device="cuda")
ct, meta = volume.read_nifti("ct.nii.gz")
result = model(ct, meta=meta)
result.save("labels.nii.gz")
```

The loader supports 2D and 3D PlainConvUNet and basic-block ResidualEncoderUNet configurations. Cascade stages, BottleneckD blocks and custom preprocessors are outside its coverage. Folder loading defaults to all available folds, the checkpoint’s allowed mirroring axes and a sliding-window step of 0.5.

On an RTX 5060 Ti, two TotalSegmentator checkpoints ran single network patches 1.23× and 1.24× faster in FastForks portable than in upstream nnU-Net with FP16 autocast. Reusing the predictors on a CT already resampled to each checkpoint’s grid gave 14.0× for Dataset297 and 3.61× for Dataset291 with matching fold, step and mirroring settings. Other checkpoints and input grids need their own measurements. [Benchmark protocol and results](../plans/totalsegmentator-coverage.md#rtx-5060-ti-replication-november-17)

::: fastforks.volume
