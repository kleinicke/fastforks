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

- [x] **Linux CUDA, A1 stereo families:** fixed ONNX parity for PSMNet, GwcNet, CFNet, Fast-ACVNet and LAS2.
  The fixed FP16 engines passed. Dynamic GwcNet FP16 passed at min, opt and max, and dynamic PSMNet FP32 ONNX
  passed. TensorRT 11.4.0.106 fails to build the tested PSMNet FP16 profile on both 0.2.0 and 0.2.1.
- [ ] **Linux CUDA, other regated stereo:** ONNX/TRT build and parity for cas_gwcnet, aanet, bridgedepth and
  great_igev. AANet passed fixed and dynamic FP16 TensorRT. Dynamic FP16 ONNX and TensorRT used opt 336 × 384
  and a profile from 288 × 288 to 384 × 432. The engine's mean absolute difference from eager stayed below
  0.0018 pixels at min, opt and max. The 0.2.0 dynamic ONNX export failed on `aten.cudnn_grid_sampler`.
  CPU ONNX checks passed for CAS-GwcNet-C, CoEx and GREAT-IGEV. BridgeDepth hit the expected top-k tie in ONNX Runtime.
- [x] GMFlow-scale1-mixdata exported from CUDA in FP16 at 256 × 384. ONNX Runtime differed from CUDA eager by
  0.00053 flow pixels on average on the textured pair. The TensorRT card gate still applies, so engine speed and
  memory remain open.
- [x] GIMM-VFI-R exported from CUDA in FP16 at 128 × 128. ONNX Runtime differed from CUDA eager by an average
  of 0.000054 on the normalized pixels. RIFE 4.26 passed its fixed FP16 TensorRT smoke. The GIMM-VFI-R TensorRT
  quality gate still applies.
- [x] **Linux CUDA, RTMO:** all four variants passed portable and kernels goldens with the restored FP16
  `out_cls` convolution. The RTMO-T FP16 public TensorRT engine built and returned finite matching-shape outputs.
- [x] **Linux CUDA, PSMNet/GwcNet kernels mode:** portable and kernels goldens passed, including both GwcNet variants.
- [x] **Windows:** `tests/test_kernels.py -k manifest_is_current` on a `core.autocrlf=true` checkout, and a fresh
  clone shows no modified files after the `.gitattributes` change. (2026-11-24, Windows 11, git autocrlf=true clone
  of dev/v0.2.1 85a7859b: `git status` clean, 88 passed; the only CRLF work-tree files are the 12 vendored `-text`
  ones.)

## Backlog fixes (maint/backlog)

- [x] **Linux CUDA:** 41 cuDNN and streamed ViTPose checks passed from a checkout with spaces in its path. PyTorch's
  CUDA runtime was mapped from a separate environment whose path also contained spaces.
- [x] **Windows, ComfyUI:** load two models from parallel queue workers (the model cache is now behind a lock).
  (2026-11-24, RTX 5060 Ti: four threads started together, two Depth and two Classify runs on depth-anything-v2-s /
  convnext_tiny: no error, one cached model each, outputs equal to each other and to a serial run.)
- [x] Calibrated CUDA stereo point clouds and NumPy uploads inside a CUDA graph passed on Ubuntu. Dropping a
  Depth Anything V2 kernels model now releases its allocations immediately, including with cyclic GC disabled.

## Follow-up to the two 57e09e83 reviews

Status against `d8099bff` and the release follow-up. The tables distinguish code changes from recorded runs.

| CUDA / Windows findings | What remains |
| --- | --- |
| A1-A3, MLX export gates and RTMO | Fixed in `0117df42`, `bf7e78b9` and `1509d3e0`. Ubuntu export and golden evidence is above. AANet FP16 export also excludes its eager-only correlation GEMM, and the shared MLX gate stays out of Dynamo traces for PyTorch 2.8. The GMFlow and GIMM-VFI TensorRT card gates still apply. |
| A4, CRLF and CI | LF attributes and source hashing landed in `7a59fc43`. The Windows autocrlf clone passed 88 manifest checks. The first [hosted Windows CPU job](https://github.com/kleinicke/fastforks/actions/runs/45391237728) passed its kernel, cache, security and ComfyUI checks. CI runs on release development branches. |
| A5, driver floor | Windows DLLs and the MVSAnywhere Windows cubin require a driver supporting CUDA 13.0. The error and hardware guide now say this. Verify CUDA 12 compatibility on the relevant older driver and hardware if that support is needed. |
| A6, CUDA device | The public model call already guards its CUDA device and binds one kernels device per process. Cubin function lookup now stays inside that device context and direct launches reject a different current device. A real `cuda:1` run needs a two-GPU host. |
| A7, RoMa v1 | Restored checked Cholesky outside MLX export. Both RoMa v1 CUDA golden modes passed on Ubuntu. |
| A8, DLL temp sweep | The loader retries once if its verified private copy disappears before Windows locks it. An ordinary load failure still raises. The native Windows run is pending. |
| A9, streamed blocks | Linux parsed a loaded CUDA runtime under a path with spaces. The relocated environment passed 41 cuDNN and streamed ViTPose tests. Windows wheel streaming has run. The Windows loader now reuses the loaded runtime even outside `torch/lib`, covered by a mocked module lookup. A native conda test remains open. |
| A10-A12, launcher and layout hardening | cuDNN's `conv_into` checks its destination. The MSDA, volume norm, RoMa, LocoTrack, CFNet, image normalization and Smish launchers check their tensor contracts before dispatch. Existing CUDA binaries were used for the kernel checks. The MSVC setup reads Unicode paths and environment values, covered by a Windows-only test. The CUDA bicubic check is recorded below. |

| ComfyUI / API findings | What remains |
| --- | --- |
| 1, 4, package floor and release | Both packs and core are at 0.2.1, both pack licenses are present, and the changelog and planned-feature messages were updated. The local wheels passed an isolated Ubuntu install. Confirm the Comfy Registry publisher ID before publishing. The registry icon is optional. |
| 2, 3, 5-7, Apple routing and memory | Routed `Model.device`, output conversion (including video, matches and stereo views), strict explicit modes, MLX cache handling and the Apple-only dependency are in code. Whole-photo face restoration ran on Windows and Ubuntu. Confirm MLX interpolation, face paste-back, and eviction under pressure on macOS. The cache follow-ups release references and clear MLX buffers on eviction. |
| 8-10, example workflows and first start | w02 uses three universal backend choices and a commercial model. Input files and new workflows are documented, and the cache parent is created. Windows ran ten of fourteen workflows offline. Ubuntu ran all fourteen with their Fastforks branches, including all four checkpoints missing on Windows. See [Ubuntu ComfyUI results](comfyui-ubuntu-results.md). |
| 11, smaller points | The MLX load docstring, one-frame interpolation and duplicate detection conversion were addressed. Detector backend and cache sharing still follow the restorer selection. The transformers conflict depends on the installed packs. |

**Before switching from this Windows machine:** the existing kernel and ComfyUI runs cover the main native paths.
Keep any Windows-specific sign-off scoped to a release wheel or portable install, or to an issue that reproduces
only on Windows. Turing, Windows conda and multi-GPU coverage need the corresponding hardware or environment.

## Ubuntu CUDA measurements (2026-11-24)

PyTorch 2.14.1+cu130 on RTX 5060 Ti 16 GB with NVIDIA driver 595.99.02. Each measurement ran alone on the GPU.

An isolated PyTorch 2.15.0+cu132 environment passed 16 representative CUDA parity, export and stereo checks,
and all 101 portable classifier module comparisons. The CUDA 13.0 wheel environment remains the release default.

- **NeuFlow v2:** all six CUDA goldens passed in portable and kernels mode. With the same FP16 weights and textured
  synthetic pair at 1920 × 1080, 15 timed calls after six warmups had median latency of 69.7 → 67.6 ms in portable
  and 57.2 → 55.1 ms in kernels. Kernels peak reserved memory was 824 → 680 MiB. Peak allocated memory was 507 MiB
  in each kernels run. The maximum difference was 0.00019 flow pixels. At 384 × 256, the CUDA graph replayed on the
  candidate with a maximum difference of 0.000002 pixels.
- **OpenCV-style bicubic:** on a 1024 × 1024 uint8 input resized to 1280 × 1280, the one-pass PyTorch float64 path
  produced the same pixels as the float64 tap gathers. Median latency was 4.45 → 1.57 ms. Peak allocation above the
  pre-call live tensors was 373 → 81 MiB.
- **TensorRT comparison with 0.2.0:** on the textured synthetic pair at 256 × 256, CAS-GwcNet-GC's fixed FP16 engine
  differed from eager by a mean of 1.724 pixels on both release revisions. The AANet fixed FP16 engine passed after
  the export change, with a 0.0020 pixel mean difference at the same size. RIFE 4.26 also passed a 256 × 384 FP16
  fixed-engine smoke check.
- **Build reproducibility:** `group_silu` and `raft_flow` were each built twice with `nvcc` 13.0.88, GCC 15.2,
  glibc 2.43 and the local `rsqrt` header patch. Every binary and both generated manifests matched byte for byte.
  These checks use the Ubuntu workstation toolchain. The published binaries use their recorded toolchains.
- The RTMW channels-last change in `47915f2e` raised keypoint error past the CUDA golden limits for
  RTMW-L-384 portable and RTMW-L kernels. Keeping RTMW's initial weight layout contiguous restored all ten
  goldens. On a 1080p frame with 15 synthetic person boxes, portable RTMW-L went from 11.6 to 15.3 ms and
  RTMW-L-384 from 25.2 to 32.0 ms compared with the faster, out-of-tolerance layout. Kernels measured 9.3
  and 18.4 ms, respectively. DWPose and RTMPose retain their channels-last path.
- RTMO-T at 1080p measured 6.43 ms portable and 2.32 ms kernels. GMFlow-scale1-mixdata
  at 1080p measured 442.4 and 271.9 ms, with 2102 and 1040 MiB peak reserved. DepthFM at 994 × 1482 measured
  4.88 and 3.38 s, with 5850 and 4092 MiB peak reserved. These are warmed inference times.

## Test environment: OpenCV 5 (2026-11-24)

OpenCV 5.0's uint8 cubic result depends on whether its optimized IPP path is available. On Ubuntu, disabling IPP
reproduced the macOS comparison failure, with 4.5–5.9% of pixels one level apart. The test now checks every uint8
pixel within one level and checks agreement within the original 1% limit against OpenCV's rounded float64 output.
