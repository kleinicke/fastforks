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

## Follow-up to the two 57e09e83 reviews

Status against `d8099bff` and the release follow-up. The tables distinguish code changes from recorded runs.

| CUDA / Windows findings | What remains |
| --- | --- |
| A1-A3, MLX export gates and RTMO | Fixed in `0117df42`, `bf7e78b9` and `1509d3e0`. The Windows eager pre-check above passed. Use Ubuntu for fixed and dynamic ONNX, TensorRT, and the selected RTMO parity and speed checks. The GMFlow TensorRT card gate still applies. |
| A4, CRLF and CI | LF attributes and source hashing landed in `7a59fc43`. The Windows autocrlf clone passed 88 manifest checks. A focused Windows CPU job now covers kernel, cache, security and ComfyUI files. Its first hosted run is pending. |
| A5, driver floor | Windows DLLs and the MVSAnywhere Windows cubin require a driver supporting CUDA 13.0. The error and hardware guide now say this. Verify CUDA 12 compatibility on the relevant older driver and hardware if that support is needed. |
| A6, CUDA device | The public model call already guards its CUDA device and binds one kernels device per process. Cubin function lookup now stays inside that device context and direct launches reject a different current device. A real `cuda:1` run needs a two-GPU host. |
| A7, RoMa v1 | Restored checked Cholesky outside MLX export. The next RoMa v1 parity check can run on Ubuntu. |
| A8, DLL temp sweep | The loader retries once if its verified private copy disappears before Windows locks it. An ordinary load failure still raises. The native Windows run is pending. |
| A9, streamed blocks | Linux parses mapped paths with spaces. Verify a streamed model in a spaced path on Ubuntu. Windows wheel streaming has run. A CUDA runtime outside `torch/lib` needs a Windows conda environment to reproduce. |
| A10-A12, remaining hardening | Launcher guards and unchecked call-site layouts remain targeted work. The MSVC setup reads Unicode output for tool and environment paths, covered by a Windows-only test. The ComfyUI cache lock landed and was checked on Windows. The CUDA bicubic check is recorded below. |

| ComfyUI / API findings | What remains |
| --- | --- |
| 1, 4, package floor and release | Both packs and core are at 0.2.1, both pack licenses are present, and the changelog and planned-feature messages were updated. Registry PublisherId and Icon and a published-package install are release work. |
| 2, 3, 5-7, Apple routing and memory | Routed `Model.device`, output conversion (including video, matches and stereo views), strict explicit modes, MLX cache handling and the Apple-only dependency are in code. Whole-photo face restoration ran on Windows and Ubuntu. Confirm MLX interpolation, face paste-back, and eviction under pressure on macOS. The cache follow-up drops references that delayed reclaiming evicted models. |
| 8-10, example workflows and first start | w02 uses three universal backend choices and a commercial model. Input files and new workflows are documented, and the cache parent is created. Windows ran ten of fourteen workflows offline. Ubuntu ran all fourteen with their Fastforks branches, including all four checkpoints missing on Windows. See [Ubuntu ComfyUI results](comfyui-ubuntu-results.md). |
| 11, smaller points | The MLX load docstring, one-frame interpolation and duplicate detection conversion were addressed. Detector backend and cache sharing still follow the restorer selection. The transformers conflict depends on the installed packs. |

**Before switching from this Windows machine:** the existing kernel and ComfyUI runs cover the main native paths.
Keep any Windows-specific sign-off scoped to a release wheel or portable install, or to an issue that reproduces
only on Windows. Turing, Windows conda and multi-GPU coverage need the corresponding hardware or environment.

## Ubuntu CUDA measurements (2026-11-24)

PyTorch 2.14.1+cu130 on RTX 5060 Ti 16 GB with NVIDIA driver 595.99.02. Each measurement ran alone on the GPU.

- **NeuFlow v2:** all six CUDA goldens passed in portable and kernels mode. With the same FP16 weights and textured
  synthetic pair at 1920 × 1080, 15 timed calls after six warmups had median latency of 69.7 → 67.6 ms in portable
  and 57.2 → 55.1 ms in kernels. Kernels peak reserved memory was 824 → 680 MiB. Peak allocated memory was 507 MiB
  in each kernels run. The maximum difference was 0.00019 flow pixels. At 384 × 256, the CUDA graph replayed on the
  candidate with a maximum difference of 0.000002 pixels.
- **OpenCV-style bicubic:** on a 1024 × 1024 uint8 input resized to 1280 × 1280, the one-pass PyTorch float64 path
  produced the same pixels as the float64 tap gathers. Median latency was 4.45 → 1.57 ms. Peak allocation above the
  pre-call live tensors was 373 → 81 MiB.

## Test environment: OpenCV 5 (2026-11-24)

`tests/test_cv2_resize.py::test_cubic_levels_match_opencv` fails on the Mac venv, which has opencv-python-headless
5.0.0: 4.5-5.1 % of the uint8 pixels differ from `cv2.resize(INTER_CUBIC)` (limit 1 %, all within 1 level), on
origin as well. The test deps ask for `>=4.10`, so CI may pick up 5.x too. Check on Linux whether OpenCV 5 changed
its cubic rounding; if so, either pin `<5` for the oracle or match both (the models' preprocessing was written
against 4.x).
