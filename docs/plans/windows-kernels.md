# CUDA kernels on Windows

Status: prototype. The `"cubin"` kernel libraries run on Windows x64. Every `"shared"` library has a Windows DLL
build (`manifest.windows.toml`): `cutlass_conv`, `dpt_fused`, `fvm_conv`, `fvm_extra`, `fvm_mb`, `fvm_r3`,
`fs_cutlass`, `ffs_cutlass`, `s2m2_cutlass`, `hitnet_conv` and `seg_conv`. `mvsanywhere`, a cubin library with a kernel parameter over 4 KB, has a Windows cubin build. 565
references have a Windows
`[validated]` record (see [Validation coverage](#validation-coverage-on-windows)). The DLLs need no Visual Studio or
CUDA toolkit at run time.

## Why cubin libraries need no Windows build

A `"cubin"` library is a fatbin of SASS (plus PTX for `compute_*`): GPU code only. The driver API
(`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`) loads it the same way on every OS, and the launch
arguments are packed by `CubinModule` from the kernel signatures, not by a host compiler. So the Linux build
(same file, same SHA-256, same manifest entry) runs on Windows unchanged.

One exception: a kernel parameter over 4 KB. `mvsanywhere` passes its MLP weights by value, a 7.6 KB struct. nvcc
writes that parameter's `.nv.info` record (attribute `0x45`) in a host-OS form: 12 bytes when built on Linux, 16
bytes on Windows. The Windows driver rejects the Linux form with `CUDA_ERROR_INVALID_IMAGE`; the Linux driver (also in
WSL2) accepts it.
- Reproduced with a one-line kernel: a 7600-byte parameter built on Linux fails on Windows, built on Windows it loads.
  4000-byte ones load either way.
- The CUDA version does not matter: a 13.0 Linux rebuild fails the same way.

So such a library also gets a Windows cubin build, `manifest.windows.toml` with `kind = "cubin"` and
`os = "windows"`. `platform_manifest` picks it on Windows, and `platform_reason` refuses it on Linux.
`build_kernels.py` on Windows builds a cubin library only when named, or when it already has a Windows manifest. Only
`mvsanywhere@0c9bd88e6b2f` has one; its Linux `manifest.toml` (`7fdcb3aafdd8`) is unchanged. The alternative was to
change the kernel to read the weights from a device buffer, a source change for both OSes. It was not chosen.

Only the host side had to change:

| Linux | Windows | change |
| --- | --- | --- |
| `libcuda.so.1` (ldconfig dirs) | `%SYSTEMROOT%\System32\nvcuda.dll` (absolute path, no DLL search order) | `_libcuda_path`, `DRIVER_LIB` |
| gate: Linux x86-64 only | `platform_reason(manifest)`: a cubin library runs on Linux x86-64 and Windows AMD64, a `shared` one only on the OS of its manifest (`os`, default `linux`); everything else raises as before (macOS: Metal libraries are a separate path, untouched) | `binary` |
| `_private`: POSIX owner / mode bits (`pwd`, `grp`, `os.getuid`) | no `os.getuid`: never private, so every load hashes the bytes it loads (`verified_binary`) and loads those bytes (`cuModuleLoadData` takes the image from memory, so there is no check-to-load race) | `_private` |
| cuDNN: `libcudnn.so.9` via `/proc/self/maps` | the loaded `cudnn64_9.dll` (torch/lib) via `GetModuleHandleW` / `GetModuleFileNameW` | `cudnn_hacc._backend._cudnn_path` |

`mode="kernels"` with a `"shared"` library without a Windows build raises `KernelUnavailableError` ("this library is
a Linux .so and has no Windows DLL build yet (manifest.windows.toml); ..."); `mode="auto"` logs that and runs portable
(checked with coex). `fastforks env` lists such libraries as unusable; `fastforks download --kernels` for this machine
skips them with a note (with `--arch` / `--all-archs` it still fetches everything, for other machines).

`cudnn_hacc` is usable on Windows: its convs run through cuDNN's backend API by ctypes; its optional native plan
builder (`plan_build`, a `.so`) is unavailable there, which already falls back to the same plans built through ctypes
(that routing existed before, logged at DEBUG).

## Verified (RTX 5060 Ti sm_120, driver 610.88 / CUDA 13.3, torch 2.14.1+cu130, Windows 11)

Kernel binaries: the Linux cache's sm_120 files copied with `cp -L` into a `FASTFORKS_KERNEL_PATH` directory
(`<dir>/<lib>/<lib>-<build>-sm_120.fatbin`), hash-checked against the manifests on load.

```powershell
$env:FASTFORKS_CACHE = 'C:\Users\neuer\fastforks-cache'
$env:FASTFORKS_WEIGHTS_PATH = 'C:\Users\neuer\fastforks-cache'
$env:FASTFORKS_KERNEL_PATH = 'C:\Users\neuer\fastforks-cache\linux-kernels'
$env:FASTFORKS_GOLDEN_PATH = 'C:\Users\neuer\fastforks-cache\goldens'
$env:FASTFORKS_OFFLINE = '1'; $env:PYTHONUTF8 = '1'
uv run python -m pytest -m gpu tests/test_golden.py -k "nmrf or bridgedepth"   # 6 passed (3 refs x portable/kernels)
uv run fastforks bench nmrf-sceneflow --size 480x640 --runs 30 --mode kernels    # and --mode portable
```

Families: NMRF and BridgeDepth (the stereo families whose kernels mode needs only `fastbb`, a cubin library).

Kernels vs portable on the `tests/synth.py` pair (100x150), golden GPU tolerance in brackets:

| model | mean abs diff px | p99 px | tolerance mean / p99 |
| --- | --- | --- | --- |
| nmrf-sceneflow | 0.00087 | 0.0153 | 0.05 / 0.5 |
| bridgedepth-sf | 0.00081 | 0.0027 | 0.02 / 0.25 |
| bridgedepth-rvc-pretrain | 0.00043 | 0.0020 | 0.02 / 0.25 |

`test_gpu` (both modes against the CPU float32 golden, and the `[validated]` build `fastbb@7d058322634d:sm_120`)
passes for all three references.

Speed: see [Speed on Windows](#speed-on-windows).

Also verified: 25 of the 26 `tests/test_cudnn_hacc.py` GPU tests pass (5 more skip: no `plan_build` binary); the
failing one loads coex in `mode="kernels"`, which needs `cutlass_conv`. CPU suites `test_kernels.py`,
`test_kernels_structure.py`, `test_security.py`, `test_golden.py`, `test_cli.py`: only failures that also fail
without these changes (cp1252 decoding of TOML files in `test_kernels_validated`; POSIX mode bits in
`test_cache_dirs_are_private_and_stale_downloads_swept`).

Not verified: `fastforks download --kernels` from the Hugging Face repo on Windows (the manifest URLs are
OS-independent and `locate` is shared, but this machine ran offline from `FASTFORKS_KERNEL_PATH`); the PTX JIT path
(`compute_80`) on Windows; any other cubin-only family.

## What the remaining libraries need

By the cards (static count, not run): 61 families across tasks need only cubin libraries (+ `cudnn_hacc`) and should
now load on Windows, e.g. stereo nmrf / bridgedepth, multiview vggt, matching romav2, the restoration and
interpolation families. The rest are blocked by these `"shared"` libraries:

| library | families | notes |
| --- | --- | --- |
| `fvm_conv`, `fvm_extra`, `fvm_mb`, `fvm_r3` | 56 / 42 / 42 / 42 (classification, detection, pose, features) | done: Windows DLLs (`fvm_r3` via the `dlfcn.h` shim) |
| `cutlass_conv` | 35 (23 stereo, flow, matching, matting) | done: Windows DLL |
| `dpt_fused` | 31 (depth, features, SAM) | done: Windows DLL |
| `fs_cutlass`, `ffs_cutlass`, `s2m2_cutlass`, `hitnet_conv`, `seg_conv` | 1-2 each | done: Windows DLLs |
| `cudnn_hacc` (`plan_build`) | none blocked (optional) | |

### Shared libraries as DLLs

One build per library and target, like Linux:

1. Toolchain. No CUDA toolkit installer is needed: the pip `nvidia-cuda-nvcc` 13.0.88 of `uv sync --group kernels`
   ships the complete Windows toolchain under `.venv\Lib\site-packages\nvidia\cu13\` (`bin\nvcc.exe`, `ptxas.exe`,
   `fatbinary.exe`, `nvvm\bin\cicc.exe`, `nvvm\libdevice`, `lib\x64\cudart_static.lib`), the same toolkit the Linux
   main builds use. A 12.8 toolkit is only needed for `manifest.compat.toml`-style builds (older drivers), which are
   not planned for Windows. The host compiler must be MSVC (`cl.exe`; nvcc on Windows accepts nothing else):
   Visual Studio 2022 Build Tools, C++ workload (about 2.5-3 GB installed with `--includeRecommended`).
   CUTLASS: the 4.7.1 headers (`include/`, `version.h` = 4.7.1), passed with `--cutlass`.
2. `tools/build_kernels.py` on win32 (committed): `nvcc -shared -o <lib>-<build>-<target>.dll` with the toolkit
   passed explicitly (`--dont-use-profile`, `-ldir`, `-isystem cccl`, `-L lib\x64`), `-cudart static`,
   `-Xcompiler /MT,/EHsc,/Brepro,/Zc:__cplusplus`, `/d1trimfile:` for the stage and CUTLASS paths (the analogue of
   `-ffile-prefix-map`), `-Xlinker /DEF:<generated .def>` (exports = `c_exports`, no `__declspec` in the sources),
   `/Brepro /NOIMPLIB /NOEXP`. MSVC comes from vswhere + `vcvars64.bat`, handed to nvcc with `--use-local-env`. The
   `[toolchain]` table records the `cl` banner, `WindowsSDKVersion` and `c_runtime = "static (/MT)"` instead of
   glibc, so the build id differs from Linux. The fixed stage is `C:\fastforks-build` (owner checked by SID).
   Named cubin libraries are refused (they are OS independent, a Windows run would rewrite `manifest.toml`); with
   no names they are skipped with a note. Linux keeps the same nvcc command line and build ids.
3. Manifest (committed): `manifest.windows.toml` next to `manifest.toml`, with `os = "windows"` (allowed for
   `shared` only) and `.dll` files; its compat sibling would be `manifest.windows.compat.toml`.
   `platform_manifest` picks it on win32 when present (otherwise the Linux manifest, which the gate rejects with a
   reason). `[validated]` entries in `golden.toml` then name the Windows build ids, recorded by
   `pytest -m gpu --record` on Windows. The HF layout `{lib}/{file}` needs no change (`tools/stage_hf.py` globs
   `manifest*.toml`, so it stages the DLLs too).
4. Loader (committed): Windows has no `memfd_create`, and `_private` is never true there. On a local fixed drive
   `shared_library` opens the cache file with read sharing only (while that handle is open no process can write,
   rename or delete it through any hard link, and NTFS refuses to rename a directory above it), hashes the bytes
   read through that handle, and loads the handle's final path (`\\?\` form, reparse points resolved) with
   `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_SYSTEM32)` while holding it (the DLLs import only `nvcuda.dll` and
   system DLLs); the mapped file's name must then equal the handle's. Once mapped, Windows keeps the file
   write-locked. No new file is written, so Defender does not scan a fresh copy in every process (11 DLL libraries:
   340 -> 38 ms a process, RTX 5060 Ti). Off a local fixed drive (a network share, a removable drive, where other
   machines need not honour the share modes) it writes the verified bytes to a new file (exclusive create) in a
   per-process `fastforks-dll-*` directory under the user's temp directory and loads that copy; directories older
   than an hour are swept. Tested with all DLL libraries in the GPU sweep.
5. Sources: grep found no `long` in the ctypes C APIs (LLP64 would make it 32-bit on Windows) and no POSIX headers
   besides `fvm_r3`'s `<dlfcn.h>` (`dlopen` of cuBLASLt). That one is served by an include shim instead of a source
   change (see [Decisions to review](#decisions-to-review)), so the `.cu` and its Linux build id stay as they are.
   Inline PTX `asm` is device code and portable. All of cutlass_conv, dpt_fused and fvm_* compile unchanged.

### cutlass_conv on Windows

Built with Visual Studio 2022 Build Tools (MSVC `cl` 19.44.35230, Windows SDK 10.0.26100.0), the pip nvcc 13.0.88
and CUTLASS 4.7.1: `cutlass_conv@5f1e101619ed`, 7 targets, 8.4 MB in all (sm_120: 1.5 MB), 3.6 min with `--jobs 1`.
Each DLL imports only `KERNEL32.dll` (the CUDA runtime and the C runtime are static; `nvcuda.dll` is loaded at run
time) and exports `fused_conv_rs` (plus cudart's `NvOptimusEnablementCuda`). Two clean rebuilds gave the same bytes.
The `cl` banner is localized (German on this machine), so the toolchain table records `MSVC cl <version> x64`.

```powershell
uv sync --group kernels
uv run python tools/build_kernels.py cutlass_conv --cutlass C:\Users\neuer\cutlass-4.7.1
# put build\kernels\cutlass_conv\*-sm_120.dll on FASTFORKS_KERNEL_PATH (next to the cubin libraries), then:
uv run python -m pytest -m gpu tests/test_golden.py -k "test_gpu and (coex or psmnet or raft-stereo or igev-stereo or gwcnet)"
```

Verified (RTX 5060 Ti sm_120): the GPU golden tests pass in portable and kernels mode for 14 references of 6 families
(coex, psmnet, raft-stereo x3, igev-stereo x5, gwcnet-g/gc, cas-gwcnet-c/gc; 28 tests). Kernels mode reports
`cutlass_conv@5f1e101619ed:sm_120`. A `--record` run gives the same mean / p99 px as the committed Linux
`[validated]` entries for all 14 references; only the build id and symbol hash differ.

Kernels vs portable on the synth pair (100x150; psmnet 256x320, its minimum is 241 px):

| model | mean abs diff px | p99 px |
| --- | --- | --- |
| coex-sceneflow | 0.00100 | 0.0038 |
| psmnet-sceneflow | 0.00434 | 0.0152 |
| raft-stereo-sceneflow | 0.00251 | 0.0158 |
| igev-stereo-sceneflow | 0.00375 | 0.0612 |
| gwcnet-gc-sceneflow | 0.00524 | 0.0181 |

Speed: see [Speed on Windows](#speed-on-windows). The `[validated]` per-OS slot and the full stereo sweep followed; see below.

### dpt_fused and fvm_* on Windows

Same toolchain and recipe: `dpt_fused@2d01fadafb47` (13.6 MB), `fvm_conv@4ab8391e0d66`, `fvm_extra@89af88377928`,
`fvm_mb@f31d86f5b9ec` and `fvm_r3@3e4d13960677` (23 MB), 7 targets each, no source changes. `fvm_r3` builds through
the `dlfcn.h` shim. In the GPU goldens its cuBLASLt binding found PyTorch's loaded `cublasLt64_13.dll`, and the
families using it pass.

```powershell
uv run python tools/build_kernels.py dpt_fused fvm_conv fvm_extra fvm_mb fvm_r3 --cutlass C:\Users\neuer\cutlass-4.7.1
```

The sweep found one loader bug: a second load of the same DLL in one process (a second model using `dpt_fused`) failed
with `FileExistsError`, because the copy's name was fixed. Fixed: each load now gets its own exclusively created
file, as each Linux load gets its own memfd.

## Validation coverage on Windows

`pytest -m gpu --record` in kernels mode on all runnable references (RTX 5060 Ti sm_120, driver 610.88,
torch 2.14.1+cu130). Every GPU run held a machine-wide GPU lock (one job at a time). Weights were staged from a
read-only Linux cache in batches of at most 15 GB and deleted after each batch. Missing sm_120 cubins (the same
files as on Linux, hash-checked) came from the same cache. The optional extras `timm`, `transformers` and
`torchvision` were installed for the run.

- 578 references can run in kernels mode on Windows: every library has a Windows build and all binaries are
  present. 565 now have a Windows `[validated]` record. That includes 20 for the five CUTLASS DLLs built last:
  foundationstereo, c-foundationstereo, c-fast-foundationstereo, fast-foundationstereo, hitnet, s2m2, mmseg and
  segnext. mvsanywhere has one as well.
- 338 of the 565 records give the same mean / p99 as the Linux record. The others differ in the 4th or 5th decimal,
  all within the golden tolerance (largest: aliked-flexglue, mean 0.0033 vs 0.0012 px; cfnet and great-igev).
- The offline cache lacked the weights of 36 references, so they were handled separately. For 32 of them, the
  weights were downloaded from their recorded URLs into a temporary cache, recorded, and deleted afterwards. Those
  are the timm / torchvision classifiers, dinov2_vitl14_lc, eva02, siglip, omni-dc, segformer and sapiens-seg; all
  passed. mvsanywhere was recorded after its fix (below). dfine-n-coco, rtmo-t and sam-hq-vit-b fail in kernels
  mode (see below).
- On another Windows machine without these weights, `test_kernels_validated` skips a reference that has no Windows
  record and no local weights, worded like the repo's other weight-missing skips. It fails as before once the
  weights are present, or when a Windows record is stale. Linux checks exactly as before.

### Without Visual Studio or a CUDA toolkit

The DLLs need neither at run time. `dumpbin /dependents` lists only `KERNEL32.dll` for all 11 Windows DLLs: the C
runtime (`/MT`) and cudart are static. The only other DLL name in their bytes is `nvcuda.dll`, the driver, which
static cudart loads at run time. `fvm_r3`'s cuBLASLt comes from the `cublasLt64_13.dll` that PyTorch has already
loaded. Check: a process with `PATH=C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem` (no Build Tools, no pip
`nvidia-cuda-nvcc`, no `CUDA_PATH`) ran the kernels-mode goldens of aanet-plus, c-fast-foundationstereo,
defom-stereo, hitnet-eth3d and s2m2-l, and they passed. All 11 DLLs were loaded from the per-process temp
directory, and no module from Visual Studio or the nvcc package was loaded (only PyTorch's own `torch\lib`
nvrtc). dfine-n-coco and rtmo-t, run in the same process, fail as they do with the tools present.

Fixed after the sweep, each with a Windows record:

- `mvsanywhere@1` gave `CUDA_ERROR_INVALID_IMAGE`. The cause is the large-parameter cubin (see
  [Why cubin libraries need no Windows build](#why-cubin-libraries-need-no-windows-build)). Its Windows cubin build
  passes with mean 6e-05 / p99 0.00025 px. A Linux run in WSL2 on the same GPU gives the same numbers, and the Linux
  build and record are unchanged.
- `c-foundationstereo-s@1`: PyTorch's Windows wheels have no flash SDPA, and its kernels forward pinned
  `SDPBackend.FLASH_ATTENTION`. `sdpa_backend()` now takes flash where the build has it and `EFFICIENT_ATTENTION`
  where it does not. The rule depends only on the PyTorch build, is logged, and each OS's record validates its
  backend.
  - On Windows, efficient, cuDNN and math attention all pass the golden.
  - The record uses efficient: mean 0.00049 / p99 0.00161 px, against 0.00044 / 0.00146 with flash on Linux.
  - Linux still uses flash and passes, checked in WSL2.

Found by the sweep (causes in [Windows-only numerical differences](#windows-only-numerical-differences)), resolved
by the maintainer decisions of 2026-11-21 (decisions 6-9 below):

| reference | result on Windows before | now | cause |
| --- | --- | --- | --- |
| `seedvr2-7b@1` | passes (the sweep hit an access violation reading the 16 GB checkpoint) | passes | Not reproduced when run alone (see below). |
| `marigold-v2-depth@1` | depth mean 0.0188 (tolerance 0.01) | passes (Windows golden) | CPU `torch.polar` in the RoPE tables (MSVC vs glibc `cos` / `sin`). |
| `dfine-n-coco@1` | 8 detections, golden 7 | passes with cuDNN (default), flash, math | No flash SDPA; one query's score crosses the golden's threshold with memory-efficient. |
| `lw-detr-medium-coco@1` | boxes mean 0.29 px | passes | No flash SDPA (the `cudnn_hacc` ctypes plans are bit-equal to Linux). |
| `rtmo-t@1` | boxes p99 0.0016 | passes | No flash SDPA; just over the tolerance. |
| `sam2.1-hiera-tiny@1` | iou_scores mean 0.0107 | passes | No flash SDPA, plus the CPU bicubic `pos_embed` at load. |
| `sam-hq-vit-b@1` | pred_masks p99 0.0058 | passes | No flash SDPA. |
| `rf-detr-n-coco@1`, `rf-detr-seg-2xl-coco@1`, `rf-detr-keypoint-preview@1` | GPU portable fails | passes | No flash SDPA. |
| `grounding-dino-base@1` | passes | passes | Passes on a rerun with the `transformers` extra installed; the local npz matches the `golden.toml` hash. |

## Windows-only numerical differences

None of the open references above is a bug in fastforks code or in a Windows kernel build. A Linux clone of the same
commit in WSL2, on the same GPU with the same torch 2.14.1+cu130 and cuDNN 9.24, reproduces every Linux golden bit for
bit. It also reproduces every Windows failure bit for bit once the one platform difference is applied (SDPA forced to
memory-efficient, or the Windows-computed constant injected). `tools/op_trace.py` records a hash of every aten op output
of a golden forward on each OS and names the first op that differs.

### No flash SDPA in the Windows wheels

`torch.backends.cuda.is_flash_attention_available()` is False on Windows. The default SDPA order is flash,
memory-efficient, math, cuDNN, so fp16 / bf16 attention without a mask runs flash on Linux and memory-efficient on
Windows. Linux with `sdpa_kernel(EFFICIENT_ATTENTION)` gives the Windows numbers exactly:

| reference (mode) | Windows = Linux efficient | Linux flash (golden) | tolerance |
| --- | --- | --- | --- |
| `rf-detr-n-coco` (portable) | boxes mean 0.0692 / p99 0.2300 | 0.0524 / 0.1517 | p99 0.23 |
| `rf-detr-seg-2xl-coco` (portable) | boxes p99 2.156 | 1.135 | |
| `rf-detr-keypoint-preview` (portable) | keypoints p99 0.1745 (masked) | passes | |
| `dfine-n-coco` (kernels) | 8 detections | 7 | |
| `lw-detr-medium-coco` (kernels) | boxes mean 0.2928 | 0.1222 | |
| `rtmo-t` (kernels) | boxes p99 0.0016 | passes | |
| `sam-hq-vit-b` (kernels) | pred_masks p99 0.00579 | 0.00352 | |

Against a float64 reference, flash, efficient, cuDNN and math SDPA have the same error at these shapes. The difference
is rounding, not accuracy: the tolerances were fitted to the flash result with almost no margin (rf-detr-n's p99 is
0.23003 against 0.23). Some references pass with math or cuDNN on Windows, but that is luck in the rounding, so no
backend was pinned for them. Resolved by decision 6: their bounds now cover all four backends.

### CPU float32 ATen kernels (GCC vs MSVC builds)

Constants computed on the CPU at load time, and the float32 CPU goldens, depend on how PyTorch's own CPU kernels were
compiled. Bit-equal across the two OSes: MKL / oneDNN matmul, linear and convolution, sums and means, the vectorized
`exp` / `sin` / `cos` / `log` / `tanh` / `erf` / `sigmoid` / `gelu` / `silu` / `sqrt`, and `pow` with a scalar base.
The thread count makes no difference. In the last bits: bilinear / bicubic interpolation (antialiased or not),
`grid_sample`, `layer_norm`, `softmax`, CPU SDPA, tanh-`gelu`, `linspace`, `randn` on a CPU generator, and
`torch.polar` (scalar libm `cos` / `sin`: MSVC UCRT vs glibc).

- `marigold-v2-depth@1`: its DiT RoPE tables (`net/dit.py`, `_rope_axis`) come from CPU `torch.polar`, and 13452 of the
  524288 entries differ in the last bit. With the Linux tables injected, Windows matches the golden exactly; with the
  Linux noise injected instead, nothing changes. The model amplifies this to a depth mean of 0.0188 (tolerance 0.01).
  No formulation gives the Linux bits on both OSes: float64 trig rounded to float32 differs from the current Linux
  tables in 13778 entries, vectorized float32 in 51619. Any platform-independent table therefore changes the Linux
  output and needs a new golden and a revision bump. Resolved by decision 7 (a Windows golden).
- `sam2.1-hiera-tiny@1` (kernels): the bicubic `pos_embed` interpolation at load (`fast.py`) is the first op that
  differs. With it computed on the GPU, Linux passes both modes, but Windows kernels still fails (iou p99 0.0154)
  because of SDPA as well.
- The float32 CPU goldens below pass on Linux (WSL2) and fail on Windows by amounts just over their `cpu_atol`. In each
  case the first op that differs is a PyTorch CPU kernel:

  | reference | first differing op | example |
  | --- | --- | --- |
  | `wd-vit-tagger-v3`, `ben2`, `anycalib-pinhole`, `resplat-small-dl3dv` | `_upsample_bicubic2d_aa` | wd-tagger logits 2.1e-05, atol 1.5e-05 |
  | `megaloc` | `_upsample_bilinear2d_aa` | |
  | `rf-detr-seg-n-coco` | `upsample_bilinear2d` | |
  | `sam-vit-b`, `sam3`, `dat-light-x4` | `layer_norm` | sam3 mixed_boxes 0.00138, atol 0.001 |

  resplat also flips a data-dependent count (`repeat_interleave` gives 32946 elements vs 32945); the stored outputs
  keep their shapes, a few Gaussians move. Resolved by decision 8.

### seedvr2-7b

The access violation in the sweep did not come back when the reference ran alone: `test_gpu[seedvr2-7b@1-portable]` and
`[seedvr2-7b@1-kernels]` both pass on Windows. The checkpoint is already read through `safe_open` (memory-mapped), one tensor at a time, and NF4
quantization runs one tensor at a time on the GPU. The likely cause was memory pressure from the sweep's other staged
weights. Its weights (16.5 GB plus the 0.5 GB VAE) exceed the 15 GB staging limit on their own.

`fs_cutlass@b342a4231e8a`, `ffs_cutlass@c6b9e444a7c4`, `s2m2_cutlass@4ca15fd645fe`, `hitnet_conv@0bd638fd1a17` and
`seg_conv@d0f93d302f0e` were built with the same recipe and no source changes (5-17 MB each). They pass the GPU goldens
for all 20 references that use them; `c-foundationstereo-s` passes since the SDPA change above.

## Speed on Windows

ms per call, float16 (the validated dtype), median of 5 `fastforks bench --runs 30 --mode <mode>` runs (each itself a
median of 30 after warm-up; range in brackets). Each run is its own process under
a machine-wide GPU lock, one job on the GPU at a time, modes alternated per model and round (the
GPU was otherwise idle; an idle ComfyUI server held 0 % utilization). RTX 5060 Ti, driver 610.88, torch 2.14.1+cu130.

| model | size | kernels | portable | speedup | peak alloc kernels / portable |
| --- | --- | --- | --- | --- | --- |
| coex-sceneflow | 375x1242 | 6.9 (6.9-6.9) | 18.9 (18.8-22.4) | 2.74x | 131 / 346 MiB |
| psmnet-sceneflow | 375x1242 | 77.5 (77.4-77.6) | 110.1 (110.0-110.3) | 1.42x | 477 / 609 MiB |
| raft-stereo-sceneflow | 375x1242 | 89.6 (89.2-89.6) | 237.8 (237.5-237.9) | 2.65x | 239 / 470 MiB |
| igev-stereo-sceneflow | 375x1242 | 103.8 (103.5-124.8) | 284.2 (283.8-284.6) | 2.74x | 368 / 415 MiB |
| gwcnet-g-sceneflow | 375x1242 | 76.2 (76.2-76.3) | 107.3 (107.1-113.4) | 1.41x | 391 / 501 MiB |
| gwcnet-gc-sceneflow | 375x1242 | 81.9 (81.8-82.0) | 110.2 (110.1-110.3) | 1.35x | 481 / 568 MiB |
| cas-gwcnet-c-sceneflow | 375x1242 | 70.3 (70.2-70.5) | 99.1 (98.9-99.2) | 1.41x | 446 / 534 MiB |
| cas-gwcnet-gc-sceneflow | 375x1242 | 106.7 (106.5-108.3) | 140.5 (140.4-140.6) | 1.32x | 731 / 731 MiB |
| nmrf-sceneflow | 375x1242 | 55.7 (55.7-55.8) | 62.1 (62.0-62.7) | 1.11x | 289 / 445 MiB |
| bridgedepth-sf | 375x1242 | 139.2 (137.6-140.0) | 182.3 (181.5-183.7) | 1.31x | 1188 / 1773 MiB |
| sam2.1-hiera-tiny (dpt_fused) | 1024x1024 | 22.1 (22.1-24.6) | 24.9 (24.8-27.4) | 1.13x | 305 / 294 MiB |
| convnext_tiny.fb_in1k (fvm_conv) | 224x224 | 1.7 (1.7-1.7) | 11.2 (5.5-11.4) | 6.6x | 125 / 98 MiB |

Notes: the spread is now below 1 % for most rows; single outliers (igev-stereo kernels 124.8, sam2 24.6) are one run
of five. convnext portable is bimodal (5.5 ms in one run, 11.2-11.4 in the others): launch-bound at batch 1 under
WDDM, while kernels mode replays a CUDA graph. sam2.1-hiera-tiny with both modes in one process (`fastforks bench`
without `--mode`, kernels second) measured kernels 29.9 vs portable 24.8 ms once; in separate processes kernels is
always faster, so time one mode per process.

## Windows vs Linux

Same machine and GPU, same driver (610.88), same torch 2.14.1+cu130 / cuDNN 9.24 and the same Linux cubins / `.so`
files: Windows native against WSL2 (Ubuntu, a `git archive` of `work/windows` at `24bdc2c0` with its own `uv sync`).
WSL2 is not bare-metal Linux: CUDA there goes through the Windows driver (WDDM, via `dxgkrnl`). The last two columns
are the bare-metal Linux rows of `docs/data/speed.jsonl` for this GPU model (latest per row, 2026-11-03 to 11-20,
the speed-table harness: uint8 CUDA input, other warm-up and medians), so they compare only roughly. Method as above,
10 processes per cell, OS and mode alternated per round; median (best process).

| model | size | kernels Win | kernels WSL2 | portable Win | portable WSL2 | Linux kernels / portable |
| --- | --- | --- | --- | --- | --- | --- |
| coex-sceneflow | 375x1242 | 6.9 (6.9) | 7.3 (7.0) | 19.3 (18.7) | 18.2 (17.6) | 6.7 / 14.7 |
| igev-stereo-sceneflow | 375x1242 | 104.5 (103.9) | 105.3 (103.7) | 285.0 (250.0) | 246.2 (243.7) | 96.4 / 213.2 |
| raft-stereo-sceneflow | 375x1242 | 89.5 (89.4) | 90.8 (90.6) | 238.1 (237.2) | 240.4 (239.2) | 88.0 / 214.7 |
| psmnet-sceneflow | 375x1242 | fails | fails | 102.2 (101.9) | 102.1 (101.9) | 73.5 / 107.0 |
| nmrf-sceneflow | 375x1242 | 56.0 (55.8) | 56.3 (56.0) | 62.3 (62.0) | 70.8 (62.3) | 51.5 / 58.1 |
| bridgedepth-sf | 375x1242 | 141.1 (138.9) | 100.9 (100.7) | 183.1 (180.9) | 120.5 (120.2) | 92.1 / 111.0 |
| sam2.1-hiera-tiny | 1024x1024 | 22.4 (22.2) | 16.8 (16.5) | 25.6 (25.0) | 19.4 (19.2) | 13.9 / 16.6 |
| convnext_tiny.fb_in1k | 224x224 | 1.7 (1.3) | 1.6 (1.3) | 11.3 (5.5) | 5.0 (4.6) | 1.15 / 3.3 |
| depth-anything-v2-s | 518x770 | 5.2 (5.1) | 10.1 (5.0) | 11.1 (10.5) | 8.3 (8.3) | 4.45 / 7.7 |
| rf-detr-s-coco | 800x1344 | 4.0 (3.9) | 4.8 (3.7) | 23.4 (16.3) | 14.1 (13.7) | 3.3 / 10.0 |
| rife-4.26 | 1080x1920 | 23.8 (23.7) | 25.8 (23.6) | 27.8 (27.6) | 28.0 (27.5) | 22.0 / 25.0 |

- **Convolution and custom-kernel work runs at the same speed.** Best processes agree within 2 % in kernels mode
  for every model without attention, and in portable mode for coex, raft-stereo, psmnet, nmrf and rife.
- **Attention is slower on Windows.** The Windows torch wheel has no flash SDPA: the default falls back to
  memory-efficient attention, which takes 1.18 ms against flash's 0.62 ms for a [1, 6, 4096, 64] fp16 call. The
  cuDNN SDPA backend is in both wheels and takes 0.61 ms. That accounts for bridgedepth (+38 % kernels, +50 %
  portable), sam2.1-hiera-tiny (+30-35 %) and the ViT/DETR portable rows (depth-anything-v2-s +27 %, rf-detr-s +19 %).
  cuDNN attention is now the logged default where PyTorch has no flash (decision 9); the timings in this table
  were taken before that, with memory-efficient attention.
- **Some cells are bimodal between processes on both OSes**, not drift within a process (cuDNN benchmarking is
  off). Examples: igev-stereo portable 250 / 285 on Windows, depth-anything-v2-s kernels 5.0 / 11.9 and rife-4.26
  24 / 35 on WSL2, convnext portable 5 / 11 on both. Compare the best values there, not the medians.
- **WSL2 is still slower than bare-metal Linux.** Portable stereo is 10-20 % slower (coex 17.6 vs 14.7, igev 244
  vs 213, raft 239 vs 215). Heavy kernels-mode runs are within 3 % (raft 90.6 vs 88.0), and the attention models
  are 16-19 % slower (sam2 16.5 vs 13.9). WDDM is the likely common cause, so Windows native also trails bare
  metal. The harnesses differ, so these are estimates.
- **No drift on Windows.** It reproduces the table above (coex 6.9, raft 89.5, igev 104.5, nmrf 56.0, sam2 22.4,
  bridgedepth 141 vs 139). One exception: psmnet portable is now 102 ms, against 110 ms before BatchNorm folding
  at load (`bd881ef5`). That merge also breaks psmnet kernels mode on both OSes: `RuntimeError: kernels: no conv
  + BatchNorm run was fused (the module tree changed)`.

## Attention speed on Windows (work/attn-speed)

The remaining gap after decision 9 (cuDNN first without flash) was profiled with `torch.profiler` over one call
per reference, Windows native against WSL2 (same GPU, driver, torch 2.14.1+cu130 / cuDNN 9.24). Each cell is its own
process under the GPU lock, median of 30 calls after warm-up, fp16, `mode` as named. GPU time is the summed kernel
time of a profiled call; "attn" is the part spent in SDPA / flash kernels.

What was left, per reference (before this work):

| reference (size) | mode | Windows ms (GPU, attn) | Linux ms (GPU, attn) | cause |
| --- | --- | --- | --- | --- |
| sam-hq-vit-b (1024) | kernels | 67.4 (64.3, 45.0) | 36.9 (34.4, 15.2) | augmented head dim 192: no cuDNN, memory-efficient |
| sam-hq-vit-b (1024) | portable | 82.1 (80.0, 45.3) | 51.7 (45.7, 15.2) | same |
| c-foundationstereo-s (375x1242) | portable | 1706 (1682, 0 SDPA kernels) | 1170 (1116, 29) | head dim 7: math backend |
| c-foundationstereo-s (375x1242) | kernels | 459 (453, 52) | 442 (431, 34) | cuDNN at head dim 8 vs flash |
| sam2.1-hiera-tiny (1024) | kernels | 17.7 (13.3, 4.5) | 16.7 (12.9, 4.0) | cuDNN d96 global attention vs flash |
| depth-anything-v2-s (518x770) | kernels | 6.7, best 5.6 (4.3, 1.4) | 5.0 (4.5, 1.4) | none on the GPU (fastbb `flash64` on both): WDDM launches |
| bridgedepth-sf (375x1242) | kernels | 114, best 100 (99, 10.6) | 101 (92, 10.6) | not attention (equal); bimodal |

Fixed (pure PyTorch, so both modes gain and the portable path stays portable):

- **SAM / HQ-SAM / ViTMatte global attention** (`shared/sam_attention.py`, commit `Run SAM's global attention with a
  dense bias where SDPA has no flash`). The augmented form `q' = [q * scale, rel_h, rel_w]` has head dim 192 in
  the global blocks; cuDNN takes at most 128, so Windows ran memory-efficient (10.8 ms per `[1, 12, 4096, 192]`
  call; Linux flash 3.4 ms). Where the SDPA policy has no flash and the augmented head dim exceeds 128, the call now
  runs SDPA at head dim 64 with `rel_h + rel_w` as a dense mask (cuDNN: 3.1 ms), in chunks of at most 128 MiB
  (`MASK_BYTES`), logged once. Linux defaults keep the augmented form. Tried and rejected: q/k at 192 with v at 64
  (memory-efficient takes it, 7.9 ms; cuDNN still refuses), folding one table into the keys per query row (805 MB
  of keys at 1024).
- **C-FoundationStereo portable** (`c_foundationstereo/portable.py`, commit `Pad C-FoundationStereo's head dim 7 to
  8 in portable attention`): an override of `FlashMultiheadAttention.forward` zero-pads half-precision CUDA
  Q / K / V to head dim 8 (scale `7 ** -0.5`), which memory-efficient and cuDNN accept; flash did the same padding
  internally, and Linux output is bit-equal to before.

After (Windows "before" is the branch base run from a `git worktree` of it, same session):

| reference (size) | mode | Windows before | Windows after | Linux (WSL2) |
| --- | --- | --- | --- | --- |
| sam-hq-vit-b (1024) | kernels | 67.0 | 37.2 (best 35.6) | 37.2 |
| sam-hq-vit-b (1024) | portable | 88.1 | 53.8 | 54.0 |
| sam-vit-b (1024) | kernels | 69.3 | 42.2 (best 38.1) | 40.6 |
| sam-vit-b (1024) | portable | 83.9 | 49.7 | 50.4 |
| vitmatte-s (1080x1920) | kernels | 140.6 | 68.8 | 83.5 |
| vitmatte-s (1080x1920) | portable | 160.3 | 87.6 | 102.7 |
| c-foundationstereo-s (375x1242) | portable | 1706 | 1197 | 1170 |
| c-foundationstereo-s (375x1242) | kernels | 459 | 461 (unchanged) | 442 |

The SAM attention now takes 11.5 ms of GPU time per call on Windows against 15.2 ms for flash on Linux; ViTMatte
at 1080p (8160 tokens) is faster on Windows than on Linux. The bias form would also be faster on Linux, but Linux
keeps PyTorch's order (decision 9), so its numbers and goldens do not move. GPU goldens, both modes: Windows
sam-vit-b, sam-vit-l, sam-hq-vit-b, vitmatte-s, vitmatte-b, c-foundationstereo-s, c-fast-foundationstereo pass;
WSL2 sam-vit-b, sam-hq-vit-b, vitmatte-s, c-foundationstereo-s, c-fast-foundationstereo pass with the default
policy, and the three SAM / ViTMatte references also with `FASTFORKS_SDPA=cudnn,efficient,math` (the bias form). No
kernel library was rebuilt, so no `[validated]` record changed.

Not done:

- **A fastforks FA2 kernel for kernels mode.** A prototype cubin (fp16, no mask, any `Nq` / `Nk`, head dim a
  multiple of 8 up to 128 zero-padded in shared memory, fp32 accumulation, 64- or 128-row query tiles) matches
  cuDNN / flash accuracy. Per call against cuDNN on Windows: d32 `[1, 8, 4096]` 0.49 vs 0.87 ms (flash 0.48), SAM 2
  window `[25, 4, 196, 96]` 0.12 vs 0.17 ms, but SAM 2 global d96 `[1, 4, 4096]` 0.76 vs 0.90 ms (flash 0.67) and d128
  1.96 vs 1.71 ms. At the shapes the references run the gain is under 0.5 ms per call (sam2.1-hiera-tiny: about
  0.5 ms of 17.7), not worth a new library with Linux and Windows records. A kernel with the SAM bias fused in
  (the tables read per key tile instead of a dense mask) would save about another 1-2 ms per SAM global block.
- **C-FS-S kernels attention** at head dim 8: cuDNN 52 ms vs flash 34 ms of 459 ms (4 %).
- **WDDM launch overhead** (depth-anything-v2-s kernels 135 launches: 6.7 ms median vs 4.3 ms GPU): CUDA graphs
  in kernels mode where a family does not use them yet.
- `sam3` was not profiled on Windows (its weights are not in the Windows cache); on Linux it runs fastbb
  `flash64` plus PyTorch flash at d32 / d64 (about 4 ms), which cuDNN covers except d32.

## Decisions to review

Made without a maintainer to keep the work moving; each is in its own commit so it can be reverted on its own.

1. **`[validated]` per-OS slot** (commit `goldens: per-OS [validated] records (a windows sub-table)`). A Windows
   `pytest -m gpu --record` used to replace the reference's Linux entry. Now Windows records go to a sub-table,
   and the Linux entry stays where it was:

   ```toml
   [validated."coex-sceneflow@1"]          # Linux, unchanged (readers and writers as before)
   kernels = ["cudnn_hacc@cudnn92400", "cutlass_conv@988138e042cb:sm_120", ...]
   ...
   [validated."coex-sceneflow@1".windows]  # written only by --record on win32
   kernels = ["cudnn_hacc@cudnn92400", "cutlass_conv@5f1e101619ed:sm_120", ...]
   ...
   ```

   `tools/goldens.py`: `OS_SLOTS = ("windows",)`; `dump` writes each slot as its own sub-table after the entry, with
   LF line endings on every OS; `validated_records(entry)` returns the Linux record (if any) and the slot records.
   `tests/test_golden.py`: the recorder on win32 writes only `[...windows]` and keeps the Linux keys; on Linux it
   writes the entry and keeps any slot. `test_kernels_validated` accepts a library build when any one record
   validates it (its build id, or unchanged `[symbols]` hashes), and it checks against the manifest the platform
   actually loads (`platform_manifest`). A Linux-only checkout reads exactly as before. Two CPU tests cover the dump
   format and the either-record rule. Alternative: a separate `golden.windows.toml`. Rejected because it would
   double the files and the review diff for every family.
2. **`tools/win_include/dlfcn.h`** (commit `build_kernels: Windows dlfcn.h include shim for DLL builds`).
   `fvm_r3` binds to cuBLASLt with `dlopen("libcublasLt.so.13", RTLD_NOLOAD)` / `dlsym`. Rather than an `#ifdef _WIN32`
   branch in the `.cu` (a new Linux build id and a Linux rebuild), MSVC builds get a small header that maps
   `lib<x>.so.<n>` to `<x>64_<n>.dll`. It prefers an already loaded module (`GetModuleHandleA`: PyTorch's
   `cublasLt64_13.dll`). Otherwise it uses `LoadLibraryExA(..., LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`, never the
   current directory or `PATH`. The directory goes on the include path, and the header's bytes into the build id,
   only for Windows builds of libraries whose sources `#include` a shimmed header. So every other build id, and every
   Linux one, is unchanged. If upstream code ever needs more of `<dlfcn.h>`, the build fails at compile time, not
   silently.
3. **Toolchain record on Windows.** `host_compiler = "MSVC cl <version> <arch>"` instead of the full `cl` banner (it
   is localized), plus `windows_sdk` and `c_runtime`; CUTLASS from a release tarball records `4.7.1` without the
   commit hash (the Linux builds used a git checkout). The `[symbols]` hashes include the toolchain, so they differ
   from Linux by design.
4. **Windows cubin builds** (commit `kernels: allow a Windows cubin build for kernels with large parameters`). A
   cubin manifest may now carry `os = "windows"`, for a library whose Linux cubin the Windows driver rejects (a kernel
   parameter over 4 KB, see above). It is opt-in per library: name it to `build_kernels.py` on Windows. Alternative:
   pass mvsanywhere's MLP weights through a device buffer instead of by value. That is a source change, and needs a
   Linux rebuild and revalidation, with possibly a small speed cost. It was rejected for now.
5. **C-FS-S attention backend** (commit `c_foundationstereo: memory-efficient SDPA where PyTorch has no flash`).
   `sdpa_backend()` picks flash, or memory-efficient SDPA when the PyTorch build has no flash, and logs the latter. It
   is a property of the PyTorch build, not a recovery from a failed call, and each OS's `[validated]` record covers
   the backend that OS uses. It is still a choice inside kernels mode. If the policy reads "only `auto` chooses"
   strictly, the alternative is to keep raising `KernelUnavailableError` on Windows (commit `99d0b12c`).

Decided by the maintainer on 2026-11-21 (see [Windows-only numerical differences](#windows-only-numerical-differences)):

6. **Tolerances for PyTorch builds without flash SDPA: re-fitted, attention references only** (commit `goldens: GPU
   bounds of attention references that cover every SDPA backend`). rf-detr-n / s / seg-2xl / keypoint-preview,
   dfine-n, lw-detr-medium, rtmo-t, sam-hq-vit-b and sam2.1-hiera-tiny were measured in both modes with flash (WSL2)
   and memory-efficient, cuDNN and math (Windows 11), RTX 5060 Ti, torch 2.14.1+cu130. Each bound is 1.25 x the
   largest value, rounded up to two digits, where the old bound did not already cover it; primary outputs get a
   `gpu_mean_<mode>` / `gpu_p99_<mode>` key only for the mode that needed it, `gpu_extra` rows are per reference. The
   measured values are in a comment next to each bound. Changed (mean / p99, max measured -> new):

   | reference (mode) | output | old | max measured | new |
   | --- | --- | --- | --- | --- |
   | `rf-detr-n-coco` (portable) | boxes | 0.079 / 0.23 | 0.069 / 0.23 | 0.087 / 0.29 |
   | `rf-detr-s-coco` (portable) | boxes | 0.11 / 0.17 | 0.098 / 0.18 | 0.13 / 0.23 |
   | `rf-detr-seg-2xl-coco` (portable) | boxes | 0.49 / 1.8 | 0.40 / 2.16 | 0.51 / 2.7 |
   | `rf-detr-seg-2xl-coco` | scores | 0.03 / 0.076 | 0.028 / 0.059 | 0.036 / 0.076 |
   | `lw-detr-medium-coco` (kernels) | boxes | 0.19 / 0.64 | 0.38 / 1.83 | 0.47 / 2.3 |
   | `rtmo-t` (portable) | keypoints | 0.035 / 0.19 | 0.024 / 0.188 | 0.035 / 0.24 |
   | `rtmo-t` | boxes | 0.00029 / 0.0013 | 0.00024 / 0.0016 | 0.00031 / 0.0021 |
   | `rtmo-t` | scores | 0.0027 / 0.0047 | 0.0021 / 0.0038 | 0.0027 / 0.0048 |
   | `rf-detr-keypoint-preview` (kernels) | keypoints | 0.12 / 0.32 | 0.099 / 0.29 | 0.13 / 0.37 |
   | `rf-detr-keypoint-preview` (portable) | keypoints | 0.072 / 0.17 | 0.064 / 0.17 | 0.08 / 0.22 |
   | `rf-detr-keypoint-preview` | boxes | 0.00069 / 0.0016 | 0.00064 / 0.001 | 0.0008 / 0.0016 |
   | `rf-detr-keypoint-preview` | keypoint_scores | 0.00048 / 0.0039 | 0.00046 / 0.0053 | 0.00058 / 0.0066 |
   | `sam-hq-vit-b` (kernels) | pred_masks | 0.0019 / 0.0053 | 0.0029 / 0.0102 | 0.0036 / 0.013 |
   | `sam-hq-vit-b` | iou_scores | 0.0032 / 0.0032 | 0.0054 / 0.0054 | 0.0068 / 0.0068 |
   | `sam2.1-hiera-tiny` (kernels) | pred_masks | 0.0037 / 0.015 | 0.0039 / 0.0196 | 0.0049 / 0.025 |
   | `sam2.1-hiera-tiny` | iou_scores | 0.0081 / 0.013 | 0.015 / 0.026 | 0.02 / 0.034 |
   | `sam2.1-hiera-tiny` | video_object_score_logits | 0.0023 / 0.0038 | 0.0019 / 0.0028 | 0.0024 / 0.0038 |

   dfine-n's bounds already cover every backend (boxes max 0.099 / 0.53 against 0.2 / 1.0), but its kernels path
   with memory-efficient SDPA keeps 8 detections against the golden's 7: one query's score crosses the 0.45 cut the
   golden was made with. That is a different detection set, not a tolerance, and stays: it passes with flash, cuDNN
   (the Windows default since decision 9) and math. sam2.1-hiera-tiny's `pos_embed` stays a CPU interpolation at
   load: computed once, bit-equal to upstream's CPU reference on Linux, and a GPU version would change the Linux
   numbers for no speed gain.
7. **Marigold v2 RoPE tables: a Windows golden** (commits `goldens: per-OS golden files ([refs."<ref>".windows])`,
   `marigold_v2: per-OS golden for Windows`). The tables stay bit-equal to the authors' code. A `[refs]` entry can
   carry a per-OS golden as a sub-table, like `[validated."<ref>".windows]`:

   ```toml
   [refs."marigold-v2-depth@1".windows]   # file marigold-v2-depth@1.windows.npz
   sha256 = "525c0eae..."
   upstream = "..."
   settings = "..."
   ```

   The family's golden tool writes it on Windows with `FASTFORKS_GOLDEN_OS=windows` (or on Windows for a reference that
   already has one); `tests/test_golden.golden` and `tools/fetch_goldens.py` read this OS's file, `tools/stage_hf.py
   goldens` stages every file of an entry. Marigold's reference is the family's own net on the GPU with `exact_mask =
   True` (the authors' code needs more than 16 GB), run on Windows the same way as on Linux. Windows portable and
   kernels pass against it; Linux reads the unchanged golden. `marigold-v2-depth@1.windows.npz` still has to be
   uploaded with `tools/stage_hf.py goldens`.
8. **Float32 CPU goldens: widened per reference and output** (commit `goldens: CPU float32 bounds that cover
   PyTorch's Windows and Linux builds`). Each bound is 1.25 x the larger max |diff| of Windows and WSL2, with both in a
   comment; family bounds stay. resplat's count flip uses its outlier bound (share of elements beyond `cpu_atol`),
   which a `[refs]` entry can now override:

   | reference | output | Windows | Linux | old | new |
   | --- | --- | --- | --- | --- | --- |
   | `megaloc` | descriptor | 1.49e-07 | 5.84e-08 | 8.8e-08 | 1.9e-07 |
   | `anycalib-pinhole` | intrinsics | 3.66e-04 | 1.22e-04 | 1.9e-04 | 4.6e-04 |
   | `rf-detr-seg-n-coco` | boxes | 2.14e-04 | 0 | 1e-04 | 2.7e-04 |
   | `sam-vit-b` | pred_masks | 2.68e-03 | 9.52e-04 | 2e-03 | 3.4e-03 |
   | `ben2` | alpha | 7.15e-06 | 2.32e-06 | 3.5e-06 | 9e-06 |
   | `dat-light-x4` | image | 3.86e-06 | 2.86e-06 | 3.6e-06 | 4.9e-06 |
   | `wd-vit-tagger-v3` | logits | 2.10e-05 | 6.68e-06 | 1.5e-05 | 2.7e-05 |
   | `sam3` | mixed_boxes | 1.38e-03 | 6.49e-04 | 1e-03 | 1.8e-03 |
   | `resplat-small-dl3dv` | outlier share (rotations) | 0.0235 | 0.0150 | 0.02 | 0.03 |
9. **SDPA backend where PyTorch has no flash** (`fastforks.shared.sdpa`, `FASTFORKS_SDPA`, docs/usage.md). Every CUDA
   model call runs under one shared policy (`Model._session`); C-FS-S's kernels take theirs from it too. `auto`
   (default) changes nothing where the build has flash (Linux: no switch entered, C-FS-S pins flash). Without flash
   it now enables cuDNN, memory-efficient, math in that order and logs it (maintainer decision 2026-11-21, after
   decision 6 made the attention bounds cover every backend; commit `sdpa: cuDNN first where PyTorch has no flash`).
   `torch` leaves PyTorch alone; a list such as `efficient,math,cudnn` (PyTorch's order without flash) sets the
   priority. cuDNN first was measured on Windows (RTX 5060 Ti, torch 2.14.1+cu130, cuDNN 9.24):

   - Accuracy against float64 is the same for both (fp16 `[1, 6, 4096, 64]`: mean 5.23e-4, max 1.13e-2), and cuDNN
     is 1.4-2x faster per call (1.18 -> 0.61 ms at that shape).
   - `fastforks bench --runs 30 --dtype float16`, median (best) of 5 processes, efficient -> cuDNN first:
     sam2.1-hiera-tiny 1024x1024 portable 27.8 (25.4) -> 20.9 (20.4), kernels 22.3 (22.1) -> 17.6 (17.6);
     depth-anything-v2-s 518x770 portable 11.0 (10.5) -> 9.4 (9.3), kernels 5.2 -> 5.2; bridgedepth-sf 375x1242
     portable 175.5 (139.7) -> 172.0 (126.7), kernels 140.3 (135.0) -> 138.8 (102.7) (bimodal between processes,
     103 / 140 ms; not the attention: a profile of one kernels call has PyTorch SDPA at 0.77 ms efficient vs 0.68
     ms cuDNN, the rest of its attention is `flash64`, 5.7 ms either way); rf-detr-s-coco 800x1344 portable 16.2 (16.1) ->
     17.9 (16.4), kernels 3.9 (3.9) -> 4.4 (3.9) (slower).
   - Windows GPU goldens of 13 SDPA references (26 cases), efficient -> cuDNN first: rf-detr-n-coco portable and
     dfine-n-coco kernels start to pass, rf-detr-s-coco portable starts to fail (boxes p99 0.1823), and the others
     fail or pass alike (sam2.1-hiera-tiny, lw-detr-medium, rtmo-t and sam-hq-vit-b kernels with other numbers;
     rf-detr-seg-2xl and keypoint-preview portable). Rounding, not accuracy, as in decision 6; with the bounds of
     decision 6 all 26 cases pass with cuDNN first.

   Decided: cuDNN is the no-flash default (it is the faster backend for the SAM and ViT models; RF-DETR is about
   10 % slower with it).

   Per-family exception (`Runner.sdpa_auto`, only under `auto` without flash, logged): Swin v1 window attention
   (`shared.swin_v1`, 144 tokens with a bias mask) in BiRefNet and InSPyReNet runs memory-efficient first. cuDNN
   builds a graph of 35-55 ms for every new attention shape (each input size brings new ones) and is not faster at
   that shape (fp16 `[484, 6, 144, 32]`: 1.21 ms cuDNN, 0.98 ms efficient). RTX 5060 Ti, Windows, 1080p kernels,
   cuDNN first -> efficient first: birefnet first call 0.93 -> 0.34 s, steady 126.6 -> 124.2 ms; birefnet-lite 0.92
   -> 0.30 s, 53.6 -> 54.6 ms; inspyrenet 0.77 -> 0.48 s, 108.5 -> 107.2 ms; inspyrenet-fast 0.86 -> 0.58 s, 18.5 ->
   16.0 ms. BEN2 keeps cuDNN first: 0.82 -> 0.34 s on the first call but 102.5 -> 114 ms per later call (3 runs), a
   loss for video. The outputs differ by fp16 rounding (birefnet max 2.3e-7); the birefnet-lite and inspyrenet-fast
   GPU goldens pass in both modes.
   OneFormer (`shared.maskformer`, Swin-L windows with a bias mask) takes the same list: 1024 x 2048, cuDNN first ->
   efficient first, kernels first call 0.95 -> 0.45 s, steady 112.4 -> 111.0 ms; portable 0.95 -> 0.47 s, 152.1 ->
   150.6 ms (2 runs each). Labels identical; class probabilities differ by mean 3e-5, p99 9e-4 (tolerance 1e-3 /
   1e-2). Mask2Former shares the backbone but its weights were not available to measure, so it keeps cuDNN first.
10. **SAM relative-position attention formulation follows the SDPA policy** (`shared.sam_attention.use_biased`, see
    [Attention speed on Windows](#attention-speed-on-windows-workattn-speed)). Without flash in the policy, global
    blocks run SDPA with a dense bias mask instead of the augmented dot product: the same maths, a different SDPA
    call, chosen per call from the policy and logged once. It is not a fallback: nothing is retried, and a policy
    that names no backend for the call still raises. Linux defaults are unchanged; using the bias form there too
    (faster than flash on the augmented form) would move Linux numbers and is left to a maintainer.

## Platform notes

- WDDM (consumer GPUs on Windows) adds launch latency and batches submissions; launch-bound small inputs gain less
  than on Linux, CUDA graphs help (several stereo families already replay graphs). A kernel running > 2 s trips the
  TDR watchdog.
- `_MIN_DRIVER` names Linux driver branches (570 / 575 / 580); Windows driver numbers differ (e.g. 572.xx for CUDA
  12.8), so the remedy text in `binary`'s error is Linux-specific.
- `ty check` without `--python-platform linux` on Windows reports `os.getgid` / `grp` / `pwd` in `_own_group`
  (pre-existing, unreachable there now that `_private` returns early).
