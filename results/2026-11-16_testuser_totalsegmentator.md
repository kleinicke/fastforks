# Test user: TotalSegmentator (+ stereo, OCR), 2026-11-16

Setup: fresh venv `scratch_tests/tu-venv-totalseg`, docs only. GPU RTX 5060 Ti, commit = this branch (work/ports). Data: totalsegmentator_subset/s0013 (301x175x231, 1.5 mm, RAS). Scripts: `~/scratch_tests/tu-ts/t1..t6.py`.

## Volume (radiology researcher)
Works: total-3mm GPU fp16 0.13 s warm, 1072/1514 MiB (alloc/reserved); `total` 3.9 s, 2417/3176 MiB; CPU total-3mm 7.8 s; 117 class names; liver 1096 mL, spleen 189 mL (plausible); anisotropic 4.5 mm slices liver 1097 mL; 2x2 tiled 301x350x462 volume runs in 3.2 GiB; LPS NIfTI in -> same array as RAS read, saved labels keep the input affine/shape/axcodes; overlay figure via matplotlib works; offline with empty cache gives a clear `WeightsDownloadError` with the fix; wrong variant name ("did you mean"), bad mode/device, 2D/5D input, wrong meta grid all clear.

| # | Tried | Expected | Got | Sev | Fix |
|---|-------|----------|-----|-----|-----|
| 1 | `print(result)` | short | full affine dump | low | short repr for result/meta objects (root, `NumpyOutputs`) |
| 2 | spacing in metres | clear error | raw `RuntimeError: sizes should be greater than 0` | med | `InputError` guard in `preprocess` |
| 3 | spacing 10x too large | clear error | 24.85 GiB alloc, CUDA OOM | med | same guard (cap 1e9 resampled voxels) |
| 4 | z-scored / uint8 / all-zero input | warning | silent garbage | med | warning when HU range <= 255; NaN/inf -> `InputError` (NaN gave 0 labels silently) |
| 5 | axes (W,H,D) | error | runs silently | low | documented (docs/usage.md) |
| 6 | `read_nifti(dicom_dir)` | hint | raw nibabel error; docs never mention DICOM | high | `InputError` pointing at pydicom + new docs "DICOM series" recipe |
| 7 | DICOM array as read (LPS) fed unflipped | right volumes | liver 331 mL instead of 1096 (silent) | high | docs: flip `[:, ::-1, ::-1]` first; verified flipped DICOM = RAS ref (99.99 % label agreement). Not auto-detectable from a bare array. |
| 8 | per-organ mL | helper | no recipe | med | docs snippet |
| 9 | API reference | readable | needs mkdocs build; `help()` works | low | not changed |

## Stereo (Middlebury Adirondack, igev-stereo-sceneflow)
Works: disparity 496x718 fp32 cuda, 0.09 s warm; numpy input; swapped / same image run; save png/npy. Clear errors: different sizes, depth without `rig=`, missing file, wrong model name. Nothing to fix.

## OCR (synthetic invoice page, ppocrv5-server/mobile)
Works: 17 lines correct on clean page, small and rotated+gray jpg (spaces lost in the rotated one, model quality), numpy input, blank page -> `[]`, wrong model and `lang=` give clear errors. Notes (low, not fixed): `OCRResult` has no `save` (docs do not claim it); `parseq` on a full page silently returns junk (it is a line recognizer).

## Part A (see commits): auto-mode dtype message, short reprs, `match_pairs` InputError, `matches(n)` cap, `verify()` ok semantics documented.
