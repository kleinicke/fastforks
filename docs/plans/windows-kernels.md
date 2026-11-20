# CUDA kernels on Windows

Status: prototype (branch `worktree-agent-a8a322db22ca4f8da`). The `"cubin"` kernel libraries run on Windows x64.
For the `"shared"` ones the loader, the per-OS manifest and the DLL build recipe are in place, but no DLL is built
yet: the build needs MSVC (Visual Studio 2022 Build Tools), which is not installed on the dev machine (see
[Blocker](#blocker-msvc)).

## Why cubin libraries need no Windows build

A `"cubin"` library is a fatbin of SASS (plus PTX for `compute_*`): GPU code only. The driver API
(`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`) loads it the same way on every OS, and the launch
arguments are packed by `CubinModule` from the kernel signatures, not by a host compiler. So the Linux build
(same file, same SHA-256, same manifest entry) runs on Windows unchanged. Only the host side had to change:

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

### Shared libraries as DLLs (implemented, not yet built)

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
4. Loader (committed): Windows has no `memfd_create`, and `_private` is never true there, so `shared_library` writes
   the hash-verified bytes to a new file (exclusive create) in a per-process `fastforks-dll-*` directory under the
   user's temp directory and loads that copy with `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_SYSTEM32)` (the DLLs
   import only `nvcuda.dll` and system DLLs). A loaded DLL is locked by Windows, so the copy cannot be swapped while
   in use; directories older than a day are swept. Tested with a System32 DLL; not yet with a fastforks DLL.
5. Sources: grep found no `long` in the ctypes C APIs (LLP64 would make it 32-bit on Windows) and no POSIX headers
   besides `fvm_r3`'s `<dlfcn.h>` (`dlopen` of cuBLASLt: needs a `LoadLibraryExW` / `GetProcAddress` branch, which
   changes its sources and so its Linux build id: do it together with a Linux rebuild). Inline PTX `asm` is device
   code and portable. Still to check by compiling.

### Blocker: MSVC

The elevated `winget install Microsoft.VisualStudio.2022.BuildTools` was not run (the agent's permission policy
blocked an elevated system-wide install without the user's own approval). With the Build Tools installed:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
uv sync --group kernels
uv run python tools/build_kernels.py cutlass_conv --cutlass C:\path\to\cutlass-4.7.1   # writes manifest.windows.toml
$env:FASTFORKS_KERNEL_PATH = "$PWD\build\kernels"
uv run python -m pytest -m gpu tests/test_golden.py -k "coex or psmnet or raft_stereo or igev"         # then --record
```
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
