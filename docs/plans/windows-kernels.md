# CUDA kernels on Windows

Status: prototype (branch `worktree-agent-a8a322db22ca4f8da`). The `"cubin"` kernel libraries run on Windows x64;
the `"shared"` ones (Linux `.so`) do not yet.

## Why cubin libraries need no Windows build

A `"cubin"` library is a fatbin of SASS (plus PTX for `compute_*`): GPU code only. The driver API
(`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`) loads it the same way on every OS, and the launch
arguments are packed by `CubinModule` from the kernel signatures, not by a host compiler. So the Linux build
(same file, same SHA-256, same manifest entry) runs on Windows unchanged. Only the host side had to change:

| Linux | Windows | change |
| --- | --- | --- |
| `libcuda.so.1` (ldconfig dirs) | `%SYSTEMROOT%\System32\nvcuda.dll` (absolute path, no DLL search order) | `_libcuda_path`, `DRIVER_LIB` |
| gate: Linux x86-64 only | `platform_reason(kind)`: Linux x86-64 runs both kinds, Windows AMD64 runs `cubin`; everything else raises as before (macOS: Metal libraries are a separate path, untouched) | `binary` |
| `_private`: POSIX owner / mode bits (`pwd`, `grp`, `os.getuid`) | no `os.getuid`: never private, so every load hashes the bytes it loads (`verified_binary`) and loads those bytes (`cuModuleLoadData` takes the image from memory, so there is no check-to-load race) | `_private` |
| cuDNN: `libcudnn.so.9` via `/proc/self/maps` | the loaded `cudnn64_9.dll` (torch/lib) via `GetModuleHandleW` / `GetModuleFileNameW` | `cudnn_hacc._backend._cudnn_path` |

`mode="kernels"` with a `"shared"` library on Windows raises `KernelUnavailableError` ("this library is a Linux .so
(no Windows DLL build yet); on Windows only cubin kernel libraries run"); `mode="auto"` logs that and runs portable
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

Speed, ms per pair, median of 3 `fastforks bench --runs 30` runs (each itself a median), modes interleaved. Taken while
another agent and background workers shared the GPU: the spread between runs is up to 1.5x, so read these as
indicative only. Peak memory is deterministic.

| model | size | kernels | portable | speedup | peak alloc kernels / portable |
| --- | --- | --- | --- | --- | --- |
| nmrf-sceneflow | 480x640 | 43.4 (41.7-44.1) | 58.3 (52.4-71.4) | 1.34x | 247 / 390 MiB |
| nmrf-sceneflow | 375x1242 | 60.4 (58.1-64.7) | 64.3 (63.9-65.6) | 1.06x | 356 / 540 MiB |
| bridgedepth-sf | 480x640 | 104.3 (98.2-106.7) | 112.2 (109.0-136.9) | 1.08x | 1096 / 1676 MiB |
| bridgedepth-sf | 375x1242 | 108.0 (105.6-149.8) | 160.0 (153.0-189.3) | 1.48x | 1270 / 1853 MiB |

The cards' Linux numbers (NMRF KITTI 51.8 vs 58.5 ms) are in the same range; a quiet-GPU rerun is needed for real
numbers.

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
| `fvm_conv`, `fvm_extra`, `fvm_mb`, `fvm_r3` | 56 / 42 / 42 / 42 (classification, detection, pose, features) | `fvm_r3` `dlopen`s cuBLASLt: port to `GetModuleHandleW` / `GetProcAddress` |
| `cutlass_conv` | 35 (23 stereo, flow, matching, matting) | CUTLASS 4.7.1 headers |
| `dpt_fused` | 31 (depth, features, SAM) | CUTLASS |
| `fs_cutlass`, `ffs_cutlass`, `s2m2_cutlass`, `hitnet_conv`, `seg_conv` | 1-2 each | CUTLASS |
| `cudnn_hacc` (`plan_build`) | none blocked (optional) | |

Plan (one build per library and target, like Linux):

1. Toolchain: CUDA toolkit for Windows (12.8 and 13.0, the manifests' toolkits; nvcc only, the CUDA runtime is linked
   statically) and MSVC v143 as the host compiler (nvcc on Windows only accepts `cl.exe`): Visual Studio 2022 Build
   Tools, workload "Desktop development with C++" (`winget install Microsoft.VisualStudio.2022.BuildTools` with
   `--override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`). Not installed here: not needed
   for the two families above.
2. `tools/build_kernels.py`: a Windows branch of `nvcc_cmd`: `-shared -o <lib>-<build>-<target>.dll`,
   `-cudart static`, `-Xcompiler /MT` (static CRT, the analogue of `-static-libstdc++`), exports through a generated
   `.def` file (`-Xlinker /DEF:...`, from `c_exports`, replacing the version script; no source change for
   `__declspec(dllexport)`), `-Xlinker /Brepro` and `-Xcompiler /d1trimfile:` (or `/pathmap`) for reproducible bytes
   (replacing `-ffile-prefix-map`, `--strip-all`), CUTLASS on MSVC usually wants `-Xcompiler /Zc:__cplusplus` and
   `/permissive-`; drop the glibc checks; the `STAGE` owner check uses `os.getuid` and needs a Windows variant.
3. Manifest: Windows DLLs have their own bytes, build id (host compiler is part of it) and symbol hashes, so a second
   manifest per library, e.g. `manifest.win.toml` (kind `shared`, `.dll` files), chosen by `read_manifest` on win32;
   `[validated]` entries in `golden.toml` then name the Windows build ids (recorded by `pytest -m gpu --record` on a
   Windows machine). The HF repo layout `{lib}/{file}` needs no change.
4. Loader: `shared_library` on Windows. `_load_image` uses `memfd_create`; Windows has no anonymous loadable file.
   Either implement `_private` for Windows (owner SID is the current user, no write ACE for other principals on the
   file and its parents up to the cache root; per-user `%LOCALAPPDATA%` is private by default) and `LoadLibraryExW`
   the verified path with `LOAD_LIBRARY_SEARCH_SYSTEM32` (its only dependency is `nvcuda.dll`), or copy the verified
   bytes into a fresh file in a private directory and load that. Until then a non-private DLL must raise, never load
   unhashed.
5. Sources: grep found no `long` in the ctypes C APIs (LLP64 would make it 32-bit on Windows) and no POSIX headers
   besides `fvm_r3`'s `<dlfcn.h>`; inline PTX `asm` is device code and portable. Still to check by compiling.

Order of value: `cutlass_conv` (23 stereo families, most of the stereo list) then `dpt_fused` (depth / SAM), then the
`fvm_*` group.

## Platform notes

- WDDM (consumer GPUs on Windows) adds launch latency and batches submissions; launch-bound small inputs gain less
  than on Linux, CUDA graphs help (several stereo families already replay graphs). A kernel running > 2 s trips the
  TDR watchdog.
- `_MIN_DRIVER` names Linux driver branches (570 / 575 / 580); Windows driver numbers differ (e.g. 572.xx for CUDA
  12.8), so the remedy text in `binary`'s error is Linux-specific.
- `ty check` without `--python-platform linux` on Windows reports `os.getgid` / `grp` / `pwd` in `_own_group`
  (pre-existing, unreachable there now that `_private` returns early).
