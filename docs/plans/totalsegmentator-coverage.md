# TotalSegmentator coverage on dev/v0.2.1

FastForks commit `80345e4c` implements two configurations of the v2 CT `total` task. The [original port](https://github.com/kleinicke/fastforks/commit/c2f61430) explicitly scoped the work to this task. The [upstream August release](https://github.com/wasserth/TotalSegmentator/blob/184fa4765456492b2ab81a981cc387465906b003/CHANGELOG.md) and the [October 14 task registry](https://github.com/wasserth/TotalSegmentator/blob/a026545ece/totalsegmentator/registry.py) already included the three specialist aorta tasks. The October CLI listed 53 task names including legacy and test entries. At the August pin, the registry flagged 18 of the 53 as license-gated. Several omitted tasks, including whole-body MR and lung nodules, were available without that gate.

| Capability | Coverage |
| --- | --- |
| Standard v2 CT segmentation | All five 1.5 mm networks, fold zero, all 117 foreground classes, later parts taking precedence on overlap |
| Fast v2 CT segmentation | One 3 mm network, all 117 foreground classes, labels returned on the original grid |
| Volume I/O | Arrays and voxel spacing, NIfTI and DICOM series reading, multilabel NIfTI writing |
| Other configurations | The v2 6 mm fastest checkpoint, v3 checkpoints, MR tasks and organ-specific tasks have no model cards |
| Upstream workflow options | The runner omits ROI-specific cropping and network selection, large-volume three-way splitting, statistics, radiomics, previews and DICOM SEG/RTSTRUCT export. Users can extract individual classes from the returned labels |
| Generic nnU-Net loader | Supports user-supplied compatible model folders. A folder alone does not specify the TotalSegmentator task's cropping, inter-model dependencies, label remapping or reporting behavior |

## Aorta report

The report is an application built around several models. FastForks provides whole-body CT inference; upstream owns the measurement and reporting workflow.

The upstream changelog dates the report to July 21 (v2.16.0). The [November 16 commit](https://github.com/wasserth/TotalSegmentator/commit/24fd5476f087) added a reference to the [November 13 paper](https://arxiv.org/abs/2611.17968).

| Report stage | FastForks coverage |
| --- | --- |
| Fast localization and general segmentation | The CT models provide the aorta, heart, T12 and general branch and iliac masks. The report's crop, 0.8 mm resampling and sequence of calls require orchestration |
| Specialist masks | Missing `aortic_dissection` (true and false lumen), `aorta_annulus` (annulus and sinotubular junction), and `renal_arteries` (including celiac trunk and superior mesenteric artery). Upstream requires a license for all three |
| Analysis | Missing contrast-phase classification, centerline extraction, the 11 landmark planes, their diameters and areas, and summaries for the five aortic sections |
| Results | Missing report JSON, rendered RGB NIfTI, static CPR PNG and animated CPR NIfTI |

The upstream CLI options include precomputed ROI directories (`--rois_totalseg`, `--rois_details`), mask filename overrides, running models sequentially or in parallel, patient metadata, erosion, skipping dissection measurements, temporary files, test/debug settings and runtime logging. A precomputed-mask comparison can exercise measurement and rendering independently of inference. The authors include a [reference case, masks and JSON](https://github.com/wasserth/TotalSegmentator/tree/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/tests/reference_files/aorta_report). See their [report CLI](https://github.com/wasserth/TotalSegmentator/blob/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/totalsegmentator/bin/totalseg_aorta_report.py) and [report documentation](https://github.com/wasserth/TotalSegmentator/blob/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/resources/aorta_report.md).

### Using FastForks masks with the upstream report

The pinned upstream runner accepts local or Modal execution. Its local model path invokes the TotalSegmentator executable. To supply general-anatomy predictions from FastForks, use the report's precomputed-mask interface. Starting with the CT used for the report, export these seven labels as binary NIfTI files in patient coordinates.

```python
from pathlib import Path
from fastforks import volume

ct, meta = volume.read_nifti("ct.nii.gz")
result = volume.load("totalsegmentator-total")(ct, meta=meta)
labels = result.labels.cpu()
class_ids = {name: idx for idx, name in result.class_names.items()}
general = Path("general_rois")
general.mkdir(exist_ok=True)
for name in (
    "aorta",
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "common_carotid_artery_left",
    "iliac_artery_left",
    "iliac_artery_right",
    "vertebrae_T12",
):
    volume.write_nifti(general / f"{name}.nii.gz", labels == class_ids[name], meta)
```

Supply matching specialist masks for the same scan in `detail_rois`, then run the upstream executable in its own environment:

```sh
totalseg_aorta_report -i ct.nii.gz --rois_totalseg general_rois --rois_details detail_rois \
  -o report.nii.gz -j report.json -l report.log
```

The report still consumes a cached contrast-phase result or invokes its phase classifier. Using FastForks' v2 general masks changes the source predictions. The reference comparison below used the authors' masks and does not establish equivalence of the resulting measurements with substituted masks.

## What the published speedup measures

The [v2.18.0 prediction code](https://github.com/wasserth/TotalSegmentator/blob/184fa4765456492b2ab81a981cc387465906b003/totalsegmentator/nnunet.py) initializes a predictor for each part and stages intermediate NIfTI files. The public Python call repeats that work on every scan. FastForks loads all networks in `volume.load` and reuses them across calls. The original speed survey additionally supplies a device tensor to FastForks and measures upstream from an in-memory CPU NIfTI. The historical rows used different PyTorch versions (2.10.0 and 2.14.1).

| v2 CT task on a synthetic 200³ volume | Upstream default FP16 | FastForks portable | FastForks kernels | Portable / kernels speedup over upstream |
| --- | ---: | ---: | ---: | ---: |
| 3 mm | 7.270 s | 0.0671 s | 0.0635 s | 108× / 114× |
| 1.5 mm | 45.630 s | 1.927 s | 1.740 s | 24× / 26× |

Within FastForks, the fused-kernel change alone gives 1.06× and 1.11× respectively. The [checked-in accuracy results](../data/volume_eval.jsonl) compare segmentation against ground truth on eight cases at 3 mm and seven at 1.5 mm. Mean Dice for upstream and portable was 0.82848 and 0.82849 at 3 mm, and 0.93651 and 0.93652 at 1.5 mm. These experiments cover only the two segmentation configurations.

## Reuse upstream code for broader coverage

1. **Keep the upstream package intact in an optional integration.** Pin one upstream commit and install its real Python package, entry points, templates, images and dependencies in the report environment. Its code uses absolute package imports and reads resources by package name. Copying individual modules into a different namespace would require edits throughout the report. If a source copy is needed for distribution, import the package as a clean subtree in a separately built distribution and retain its license and resource layout. Keep FastForks changes in a short, reviewable patch series.
2. **Start with the aorta report on existing masks.** Run the authors’ CLI with the reference CT, detailed masks and cached contrast phase. Keep the upstream landmark placement, physical-coordinate calculations, missing-landmark behavior, erosion, section summaries, CPR and rendering. Compare the measurement values and endpoint coordinates in physical units, missing values, image grid and requested output files. Reuse the upstream regression and add tighter comparisons for any code that FastForks replaces.
3. **Connect the existing inference engine at the model call.** The report invokes the `TotalSegmentator` executable, sometimes in parallel. A Python-only hook in the parent process will miss those child calls. Introduce a narrow runner argument in upstream's report model dispatcher and keep one runner per worker. At the segmentation level, pass the resolved model folder, folds, checkpoint, task step, mirroring and input geometry to the FastForks predictor. Retain upstream’s per-task processing, including ROI selection, crop margins, cascades, label merging and output handling. Route explicitly requested upstream runs to upstream; require coverage and parity for every part of an explicitly requested FastForks run. Track the backend actually used for each stage.
4. **Add task families from the upstream registry.** Derive the task/configuration inventory and label maps from the pin. Maintain a small reviewed mapping of verified weight archives, supported plans and tested options. Checkpoint access for the three licensed tasks still follows the authors’ license mechanism. Exercise licensed end-to-end inference separately from cached-mask report tests. Never infer support from the mere presence of a task name.
5. **Make updates reviewable.** Advance the pin, reapply the small patches, diff task definitions, CLI flags, labels, assets and checkpoint URLs, then run the upstream tests and FastForks parity tests. Benchmark image-to-label, resident predictor and CT-to-final-report separately. Use the upstream revision under consideration for each comparison since the newer branch already avoids much of v2.18.0's intermediate disk I/O.

The relevant upstream modules are the [report dispatcher](https://github.com/wasserth/TotalSegmentator/blob/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/totalsegmentator/aorta_report/model_runner.py), [task configuration](https://github.com/wasserth/TotalSegmentator/blob/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/totalsegmentator/map_tasks_config.py), [nnU-Net adapter](https://github.com/wasserth/TotalSegmentator/blob/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/totalsegmentator/nnunet.py) and [report regression](https://github.com/wasserth/TotalSegmentator/blob/f3f68e2eaedaad338ef34465c4a60c786b3b6b6c/tests/test_aorta_report.py).

## RTX 5060 Ti replication, November 17

Both implementations ran on the same Vast.ai RTX 5060 Ti 16 GB, PyTorch 2.10.0+cu130, upstream TotalSegmentator 2.18.0, nnU-Net 2.8.1 and FastForks `80345e4c`. The public-API timings use the median of three synchronized calls after a first call, with CPU input and materialized CPU output for both. Upstream receives an in-memory multilabel NIfTI request, with default settings including its usual configuration and telemetry. FastForks portable keeps its loaded model. Downloads, imports and construction of the input are outside the timers. The example is the authors' public report CT (191 × 167 × 233 in ZYX, 1.5 mm). Voxel agreement measures equality of integer labels, including background.

| Input | Resolution | Upstream | FastForks | Speedup | Voxel agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic 200³ | 3 mm | 9.899 s | 0.0815 s | 121× | 99.8132% |
| Example CT | 3 mm | 9.787 s | 0.0807 s | 121× | 99.9868% |
| Synthetic 200³ | 1.5 mm | 53.771 s | 1.934 s | 27.8× | 99.9980% |
| Example CT | 1.5 mm | 52.993 s | 2.873 s | 18.4× | 99.9959% |

On this CT, FastForks load plus first inference took 2.27 s (3 mm) and 4.35 s (1.5 mm). Upstream's first calls took 15.05 s and 58.59 s. Imports and weight downloads are excluded.

| Matched operation | Upstream | FastForks portable | Speedup |
| --- | ---: | ---: | ---: |
| Reusable single-model predictor, 3 mm CT (Dataset297) | 962 ms | 68.5 ms | 14.0× |
| Reusable single-model predictor, 1.5 mm CT organs part (Dataset291) | 2052 ms | 568 ms | 3.61× |
| Single network patch, Dataset297 (112 × 112 × 128) | 40.68 ms | 33.18 ms | 1.23× |
| Single network patch, Dataset291 (128³) | 51.05 ms | 41.01 ms | 1.24× |

The reusable-predictor comparison uses upstream `nnUNetPredictor.predict_single_npy_array` and FastForks' generic nnU-Net folder loader on the same CT already resampled to each model's grid. Both use fold 0, step 0.8 and disabled mirroring. The 1.5 mm experiment here covers the organs network only. The direct patch runs use identical random inputs, upstream FP16 autocast, the FastForks FP16 channels-last layout, batch one, and the median of five forwards after three warmups. The timings support gains at all three levels, with the largest factor specific to the v2.18.0 public API and a preloaded FastForks model. The paper's report measurements have a different workload.

On the same instance the authors' unmodified report regression at `f3f68e2e` passed all three checks in 60.18 s. It used their sample CT, cached masks and phase result, basic test mode and erosion, with Xvfb and wkhtmltopdf installed. The tests check the log, compare report JSON to the authors' fixture, and verify that the rendered NIfTI exists. This establishes a reproducible baseline for adopting upstream code. Live licensed inference and the optional CPR outputs were outside this test.

### Newer upstream control

With the same PyTorch environment, example CT and v2 weights, upstream `f3f68e2e` using `task="total_v2", resampling_order=1` took 3.293 s at 3 mm and 14.467 s at 1.5 mm (three-call medians). Against that baseline, loaded FastForks portable was 40.8× and 5.0× faster. These measurements isolate the change in upstream software from the change in weights and default interpolation. They do not measure the new v3 checkpoints or the report pipeline.
