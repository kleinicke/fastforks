# Checks for the CUDA and Windows machines

Mac-side changes that touch code shared with CUDA, ONNX/TensorRT or Windows, and that could only be checked on
Apple hardware and CPU. Tick an item once it has run on the named platform.

## Export gating (dev/v0.2.1, 2026-11-24)

MLX-only graph changes are now gated on `EXPORTING_FOR_MLX` instead of any export, so ONNX/TensorRT trace the
eager path as in v0.2.0. Mac coverage: 8 CPU ONNX exports and 16 MLX goldens.

Windows pre-check (2026-11-24, RTX 5060 Ti, dev/v0.2.1 b1ab72da; eager paths only, no ONNX/TensorRT there): the
GPU goldens, portable and kernels, pass for psmnet, gwcnet, cas-gwcnet, aanet, bridgedepth, great-igev, cfnet,
fast-acvnet, las2, rtmo (t/s/m/l), rife-4.25 / 4.25-lite / 4.26, gimm-vfi-r and the GMFlow refs with local weights
(76 tests). gimm-vfi-r-p and most GMFlow refs were not run: no weights on that machine.

- [ ] **Linux CUDA, A1 stereo families:** `export_onnx` and TensorRT engines for psmnet, gwcnet, cfnet,
  fast_acvnet and las2, including `--size-range`. The PSMNet/GwcNet regression block is the v0.2.0 one again.
- [ ] **Linux CUDA, other regated stereo:** ONNX/TRT build and parity for cas_gwcnet, aanet, bridgedepth and
  great_igev (they trace the in-place eager path now).
- [ ] **Linux CUDA, GMFlow:** fp16 TensorRT engine speed and memory against v0.2.0 (no fp32 residual in
  non-MLX exports).
- [ ] **Linux CUDA, GIMM-VFI and RIFE:** fp16 export as in v0.2.0; RIFE's export graph uses in-place writes again.
- [ ] **Linux CUDA, RTMO:** fp16 goldens and `[validated]` with the restored fp16 `out_cls` conv, plus its engine.
- [ ] **Linux CUDA, PSMNet/GwcNet kernels mode:** quick golden sanity (channels-last cost volume and `nchw` under
  the new gate).
- [x] **Windows:** `tests/test_kernels.py -k manifest_is_current` on a `core.autocrlf=true` checkout, and a fresh
  clone shows no modified files after the `.gitattributes` change. (2026-11-24, Windows 11, git autocrlf=true clone
  of dev/v0.2.1 85a7859b: `git status` clean, 88 passed; the only CRLF work-tree files are the 12 vendored `-text`
  ones.)

## Backlog fixes (maint/backlog)

- [ ] **Linux CUDA:** `tests/test_cudnn_hacc.py`, plus a streaming model that uses `shared/host_blocks.py` (for
  example `tests/test_vitpose_stream.py`), from a checkout whose path has a space. Both now split
  `/proc/self/maps` lines on at most five fields.
- [x] **Windows, ComfyUI:** load two models from parallel queue workers (the model cache is now behind a lock).
  (2026-11-24, RTX 5060 Ti: four threads started together, two Depth and two Classify runs on depth-anything-v2-s /
  convnext_tiny: no error, one cached model each, outputs equal to each other and to a serial run.)

## Test environment: OpenCV 5 (2026-11-24)

`tests/test_cv2_resize.py::test_cubic_levels_match_opencv` fails on the Mac venv, which has opencv-python-headless
5.0.0: 4.5-5.1 % of the uint8 pixels differ from `cv2.resize(INTER_CUBIC)` (limit 1 %, all within 1 level), on
origin as well. The test deps ask for `>=4.10`, so CI may pick up 5.x too. Check on Linux whether OpenCV 5 changed
its cubic rounding; if so, either pin `<5` for the oracle or match both (the models' preprocessing was written
against 4.x).
