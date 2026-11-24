# ComfyUI nodes on Windows: results

2026-11-20. These are the results of [comfyui-windows-check.md](comfyui-windows-check.md).

## Setup

- Machine: Windows 11 Home 26200 with an RTX 5060 Ti (sm_120, driver 610.88).
- ComfyUI 4806189de runs in its own venv: torch 2.14.1+cu130, transformers 5.23.
- The pack is linked into `custom_nodes\comfyui-fastforks` with a directory junction.
- fastforks-core is installed editable from this branch.
- Every workflow was built in API format, sent with `POST /prompt` and polled through `/history`. Each output PNG was
  then opened and checked.
- The fastforks cache is the pack's default, `ComfyUI\models\fastforks`. The previous cache is read in place
  (`FASTFORKS_READ_CACHES`, blobs only), and `FASTFORKS_WEIGHTS_PATH` points at a directory laid out as
  `<family>/<file>`.
- Input images:
  - Stereo: Middlebury 2003 "cones" (im2 / im6, with ground truth disp2).
  - Other nodes: skimage `chelsea` (cat) and `astronaut`. The clips are four 384 px crops of `astronaut` panning
    across it.

Timings are the `execution_start` to `execution_success` span from `/history`, in seconds.

- Run 0 includes the model load, plus a download if one was needed.
- Runs 1 and 2 use the cached model.
- **The GPU was shared with other jobs during the whole run, so the timings are noisy.** Use them as orders of
  magnitude, not benchmarks.

## Checklist

| # | Step | Result |
|---|---|---|
| 1 | Install keeps ComfyUI's torch | pass. `uv pip install --dry-run "fastforks-core[timm,transformers]>=0.2"` into the ComfyUI venv keeps torch 2.14.1+cu130 (fastforks-core needs torch >= 2.8). ComfyUI portable was not run. |
| 2 | Start, 20 nodes, no import error | pass. All 20 `Fastforks*` nodes are in `/object_info`. The pack imports in 0.1 s. The log says `fastforks: downloads go to ...\models\fastforks; also reading [...]`. |
| 3 | Depth `depth-anything-v2-s`, `auto` | pass. There is one warning per model load: `kernels unavailable (... built for Linux x86-64, not win32 AMD64 ...); using portable`. Against a direct `fastforks` call on CPU in float32, the mean abs diff is 0.0017 and p99 is 0.004 on the 0-1 output. There was no Linux PNG to compare against. |
| 4 | Remove background, Segment, Interpolate; unload | pass. Outputs and the VRAM numbers are below. |
| 5 | Long paths | pass. The longest path is 125 characters. `LongPathsEnabled=0` on this machine. |
| 6 | `portable` + float16 / bfloat16 | pass. Both dtypes run on this GPU. A dtype that a family cannot compute raises an error naming the dtype. An RTX 20xx was not available. |
| 7 | Video depth, Depth list, Upscale `file:` | pass. |
| - | Stereo, small and big models | pass. Five models; numbers below. |
| - | Models stay cached; cancel | pass |
| - | `backend=kernels` gives the Windows error | pass. The message now says `backend=`, not `mode=` (e20c52b3). |
| - | Dropdowns match what runs | **fail, fixed** in b0dd3012 |
| - | Stereo `kernels` with the Windows kernel builds | pass. See [Stereo with Windows kernels](#stereo-with-windows-kernels). |

### Runs

| Node | Model | Run 0 / 1 / 2 (s) | Output |
|---|---|---|---|
| Depth | depth-anything-v2-s | 6.29 (incl. 95 MB download) / 0.09 / 0.08 | Correct relative depth of the cat. |
| Stereo | raft-stereo-middlebury | 2.72 / 0.59 / 0.72 | Through the library (CPU fp32): EPE 0.55 px, 8.3 % > 1 px against Middlebury GT. The node's output differs from that by a mean of 0.0017. |
| Stereo | igev-stereo-middlebury | 3.63 / 0.34 / 1.07 | Through the library (CPU fp32): EPE 0.98 px, 9.9 % > 1 px. |
| Stereo | fast-acvnet-plus-generalization | 1.34 / 0.37 / 0.34 | Plausible but dark (min / max scaling); fixed in f0a61f0e. |
| Stereo | s2m2-l | 4.80 / 0.54 / 0.45 | Clean. float32, float16 and bfloat16 all run. Against float32, float16 differs by a mean of 0.0004 and bfloat16 by 0.0019. |
| Stereo | foundationstereo-vit-l | 14.25 / 1.26 / 1.73 | Clean, apart from a noisy band on the left occlusion border. |
| Remove background | birefnet-lite | 1.62 / 1.36 / 0.24 | Clean cat cut-out. RGBA, with alpha covering the full range 0-255. |
| Segment | sam2.1-hiera-small (one positive and one negative point) | 1.20 / 0.09 / 0.09 | Mask of the cat. |
| Segment | sam2.1-hiera-large (box) | 3.8 | Speckled mask. The library gives the same mask on CPU in float32 (IoU 0.9999), so this is the model, not the node. |
| Interpolate | rife-4.26, x2 on 4 frames | 1.19 / 1.42 / 0.41 | 7 frames. The input frames are unchanged and the new frames are motion-compensated. |
| Interpolate | rife-4.25-lite, x3 | 0.93 | 10 frames. |
| Video depth | video-depth-anything-s, 4 frames | 1.63 / 0.29 / 0.32 | One scale for the whole clip: the per-frame max is 0.97-1.0. |
| Upscale | realesr-general-x4v3 (Compact) | 1.05 (incl. 4.9 MB download) / 0.21 / 0.22 | 4x, looks right. |
| Upscale | realesrgan-x4plus | 4.13 (incl. 67 MB download) / 0.43 / 1.98 | 4x (1804 x 1200), looks right. |
| Upscale | `file: RealESRGAN_x4plus.pth`, `file: realesr-general-x4v3.pth` | 4.4 / - | Byte-identical to the runs by id. |
| Video segment | sam2.1-hiera-tiny (points); SAMURAI with sam2.1-hiera-small (box) | 2.7; 1.43 | Tracks the face across the clip. |
| Optical flow + Warp | sea-raft-s | 3.76 (incl. 35.6 MB download) | The flow is uniform, as expected for a pure pan, and Warp aligns the frames. |

After the fixes and a server restart, Depth, Segment, Remove background and Upscale were run again:

| Node | Run 0 / 1 (s) |
|---|---|
| Depth | 1.46 / 0.15 |
| Segment | 0.81 / 0.10 |
| Upscale | 0.64 / 0.25 |
| Remove background | 1.21 / 0.18 |

`/object_info` takes 2.4 s with the pack. The model contracts are read without loading weights.

### Behaviour

- **Unload.** `POST /free` with `unload_models` logs `fastforks: unloading 2 cached model(s)`.
  - torch reserved memory drops from 864 to 96 MiB.
  - ComfyUI's `vram_free` rises by about 717 MiB.
  - Task Manager and `nvidia-smi` totals move too, but other jobs on the shared GPU make them noisy.
- **Caching.** Each model logs `fastforks: loaded` once. Later runs skip the load, which is why runs 1 and 2 above are
  fast. The pack used to keep at most two models. It now keeps up to eight and drops the least recently used ones only
  when ComfyUI needs the memory (see "Model cache" below).
- **Cancel.** I ran a 40-pair foundationstereo batch and sent `POST /interrupt` after 6.1 s.
  - The prompt ended 0.25 s later with `execution_interrupted`.
  - The model stayed cached: the next run took 0.54 s.
- **Backends.**
  - `portable` and `auto` work.
  - `kernels` raises `KernelUnavailableError: kernels are built for Linux x86-64, not win32 AMD64; use mode='auto'
    ...`. Since e20c52b3 the node says `backend='auto'` (see below).
  - `kernels` with `float32` raises the reduced-precision message, which names `mode='portable'` with
    `dtype='float32'` (now `backend='portable'`).
- **dtype.** `float16` works. `bfloat16` on depth_anything_v2 portable raises `OptionsError: depth_anything_v2
  portable cannot compute in bfloat16; dtype must be one of ['float32', 'float16']`. That is not a silent cast.
- **Long paths.**
  - The longest file is `models\fastforks\blobs\sha256\xx\<64 hex>`, which adds 81 characters to the ComfyUI folder.
    A download's temporary folder adds about 25 more.
  - Archives are streamed, never extracted.
  - So a ComfyUI folder up to about 140 characters needs no long-path setting.
- **Downloads made** (about 205 MB in total, all for fastforks):
  - depth-anything-v2-s: 95 MB, because the server had first been started without `FASTFORKS_WEIGHTS_PATH`.
  - realesrgan-x4plus: 67 MB.
  - sea-raft-s: 35.6 MB.
  - realesr-general-x4v3: 4.9 MB.
  - Sample images: about 2 MB.

## Stereo with Windows kernels

Second pass, same day. fastforks came from the `work/windows-portable` checkout (28f47f2b, Windows `cutlass_conv`
DLL and the cubin libraries); the pack from this branch.

- A second ComfyUI server on port 8190 ran while holding the machine's GPU lock, with `PYTHONPATH` set to that checkout's `src`. The
  ComfyUI venv still has this branch installed editable, so `PYTHONPATH` takes precedence.
- `FASTFORKS_CACHE` and `FASTFORKS_WEIGHTS_PATH` were `C:\Users\neuer\fastforks-cache`, and `FASTFORKS_KERNEL_PATH`
  was `...\fastforks-cache\linux-kernels` (as in that branch's `docs/plans/windows-kernels.md`). `FASTFORKS_OFFLINE=1`
  was set.
- The method was the same as above: API workflows, `POST /prompt`, `/history`, and the PNGs were checked. The input
  was cones.

| Model | Backend (kernels loaded) | Run 0 / 1 (s) | PNG vs `portable` mean / p99 |
|---|---|---|---|
| raft-stereo-middlebury | portable | 1.88 / 0.26 | - |
| raft-stereo-middlebury | kernels (`raft`, `cutlass_conv@5f1e101619ed:sm_120`, `cudnn_hacc`) | 0.30 / 0.12 | 0.0004 / 0.0039 |
| igev-stereo-middlebury | portable | 1.38 / 0.28 | - |
| igev-stereo-middlebury | kernels (`igev`, `cutlass_conv`, `cudnn_hacc`) | 0.74 / 0.20 | 0.0009 / 0.0118 |
| fast-acvnet-plus-generalization | portable | 0.96 / 0.11 | - |
| fast-acvnet-plus-generalization | kernels (`igev`, `cutlass_conv`, `cudnn_hacc`, `acv_volume`) | 0.52 / 0.12 | 0.0003 / 0.0039 |

- **`auto`** picks the kernels for raft-stereo: the log shows `Backend(mode='kernels', ...)`.
- **s2m2-l with `kernels`** fails as intended: `KernelUnavailableError: s2m2-l@1 (backend='kernels'): kernel library
  's2m2_cutlass' unavailable: this library is a Linux .so and has no Windows DLL build yet ...; use backend='auto' to
  fall back to portable`. With `auto` it runs portable and logs one warning.
- **Accuracy.** Raw disparity was checked through the library on CUDA under the GPU lock (float16, EPE against the
  Middlebury GT):

  | Model | portable EPE / > 1 px | kernels EPE / > 1 px | kernels vs portable mean / p99 |
  |---|---|---|---|
  | raft-stereo-middlebury | 0.555 px / 8.3 % | 0.555 px / 8.4 % | 0.013 / 0.109 px |
  | igev-stereo-middlebury | 0.987 px / 9.9 % | 0.994 px / 9.9 % | 0.026 / 0.670 px |
  | fast-acvnet-plus-generalization | 0.764 px / 14.5 % | 0.764 px / 14.5 % | 0.012 / 0.108 px |

  These match the CPU float32 numbers above (0.55 and 0.98 px).
- **Output scaling.** All of these PNGs use the new percentile scaling. fast-acvnet-plus on cones went from a median
  gray of 0.19 (min / max) to 0.43.

## Model cache (portable ComfyUI 0.46, `work/windows`)

ComfyUI portable 0.46.0 with `python_embeded`, the fastforks-core and fastforks-nc 0.2.0 wheels of this branch, and the
pack copied into `custom_nodes`. The server was started with `--cache-none`. Every node ran with `backend=auto`, which
picked the Windows kernels.

- **Before.** The pack kept two models. A workflow set cycling five nodes (Depth, Remove background, Segment, Upscale,
  Interpolate x8) reloaded a model on every run: Depth took 0.7 s instead of 0.1 s.
- **After.** `make_room` already drops our least recently used models when ComfyUI's `free_memory` is short, so the
  count cap is now only a bound (eight).
  - The same five-node cycle, eight rounds: each model logs `fastforks: loaded` once.
  - Warm runs: Depth 0.08 to 0.13 s, Remove background 0.15 to 0.17 s, Segment 0.11 s, Upscale 0.39 to 0.41 s,
    Interpolate x8 0.6 to 0.7 s.
  - VRAM stays flat at 576 MiB of torch memory across the rounds.
  - After `POST /free`, torch memory is 288 MiB.
- **Hog.** Another process held 12.1 GB of the 16 GB card. All eight workflows ran, including sam2.1-hiera-large, with
  no errors.
- **WDDM and other processes.** On Windows, `torch.cuda.mem_get_info` does not count VRAM that another process holds.
  - Measured: free memory stayed at 15173 MiB while a second process held 6 GiB. It dropped by 4 GiB when the same
    process allocated.
  - WDDM pages memory between processes instead of failing, so neither ComfyUI nor the pack can see another app's
    usage. Both only manage their own process.

## Bugs fixed (branch `work/comfyui-windows`, not pushed)

None of these is Windows-specific; the Windows run surfaced them.

- **e20c52b3: errors said `mode=`.** fastforks errors (`KernelUnavailableError`, `OptionsError`, ...) told the user
  to `use mode='auto'`, but the node input is `backend`.
  - The nodes now re-raise fastforks errors with `mode=` said as `backend=`. The type, attributes and traceback are
    kept.
  - This covers loading and every model call (`each`, `video`, tracking).
  - The library text is unchanged.
- **f0a61f0e: dark Stereo output.** The Stereo node now scales disparity between its 1st and 99th percentile; the
  outer 1 % saturate. Disparity is already an inverse depth, so near is still white.
  - Depth and Video depth keep min / max, the ControlNet preprocessor convention. `depth_image(..., clip=0)` is the
    default.
  - Unit test: 0.5 % outliers no longer squash the map, and the node output matches.
- **8ad74d59: version.** fastforks-core said 0.1.1 on this branch: `pyproject.toml`, `uv.lock`, `CITATION.cff`, and
  the 0.2.0 notes sat under "Unreleased". It now says 0.2.0, as tag v0.2.0 does.
  - The pack, never released, goes from 0.1.0 to 0.2.0, the fastforks-core minor it is written for.
  - The ComfyUI venv still has the old editable 0.1.1 metadata until it is reinstalled.
- **02312b5b: cache folders.** `setup_cache` now uses pathlib and compares resolved paths. A read cache spelled two
  ways is read once.

- **b0dd3012: node dropdowns listed models the node cannot run.** Picking any of these failed at run time:
  - Depth listed eight depth-prompted and depth-completion models: Prompt Depth Anything s/l, LingBot-Depth (+ DC),
    OMNI-DC, Any2Full s/b/l.
  - Remove background listed MatAnyone, which needs a first-frame mask.
  - Segment listed INSID3, which needs a reference image.
  - Upscale listed LaMa (needs a mask) and GFPGAN (needs aligned faces).

  Each node now declares the inputs and prompt kinds it passes. It lists an id only if
  `model_card(id).contract().inputs` requires nothing else.

  Restore faces listed only CodeFormer. CodeFormer's code is non-commercial and is not in a fastforks-core install
  without `[nc]`, so the list was empty there. It now lists GFPGAN v1.4 and CodeFormer.

  Tests in `tests/test_comfyui.py` check the excluded ids and that every listed id passes `node.runs(id)`.
- **f533fc32: packaging.**
  - `requirements.txt` and `pyproject.toml` named fastforks-core with no version. 0.1.x has no
    `fastforks.restoration` and no Video Depth Anything, which three nodes call, so both now pin `>=0.2`.
  - A missing library failed with a bare `No module named fastforks`. It now raises an `ImportError` that gives the
    pip command for ComfyUI's Python, including portable's `python_embeded`.
- **5555a521: docs.**
  - `docs/comfyui.md` said to clone into `custom_nodes`, but ComfyUI then reports the repository root as
    IMPORT FAILED. It also said to copy the folder on Windows.
  - It now says to clone elsewhere and link the folder: `ln -s` on Linux and macOS, `mklink /J` on Windows (no admin
    rights needed).
  - It covers pip with ComfyUI's Python, the `[nc]` extra, Windows backend behaviour and path lengths.
  - The stale "Compact / SRVGG files are not listed" is gone.

## Linux / macOS and packaging (code reading)

- The pack itself has no POSIX or Windows assumptions. Paths are pathlib (02312b5b), and lists of paths use
  `os.pathsep`. It runs on ComfyUI's device, including MPS.
  - The percentile scaling sorts rather than calling `torch.quantile`, so it runs on every device.
  - The only Windows path is the `python_embeded` hint in the missing-library message, which is for ComfyUI portable
    (Windows only).
  - Not run on Linux or macOS here: WSL has no torch.
- The pack finds fastforks only as an installed package (`import fastforks`). It does not add paths to `sys.path`, so a
  junction, a symlink or a copy behave the same.
- On Linux and macOS, `ln -s` works. On Windows, `mklink /D` needs admin rights or developer mode, while
  `mklink /J` and a copy do not.

## Open issues

1. ~~The kernels error names `mode=`~~: fixed in the pack (e20c52b3). The info log `fastforks: loaded X (...)`
   now prints the device, dtype and kernels in plain words (`describe_backend`).
2. ~~Stereo output scaling~~: fixed (f0a61f0e). Depth keeps min / max on purpose.
3. ~~Version mismatch on this branch~~: fixed (8ad74d59).
   - `docs/comfyui.md` and the `backend` tooltip now say the kernels run on Windows x86-64 too (true from the
     release that ships the Windows kernel builds; the pack's `fastforks>=0.2` floor moves with it).
4. **transformers pin.** The `[transformers]` extra needs transformers >= 5.19, which can conflict with other node
   packs that pin 4.x.
5. **Publishing.**
   - The pack folder has no LICENSE file, and the `[tool.comfy]` Icon is empty.
   - ComfyUI-Manager's "Install via Git URL" clones the repository root, which is not a pack.
   - Publishing to the Comfy registry from `integrations/comfyui` is still to be done.
6. **Old weights layout.** Weights stored in a directory laid out as `<family>/<file>` are found only through
   `FASTFORKS_WEIGHTS_PATH`. The pack's read cache reads only `blobs/`.
7. **Not run:**
   - bfloat16 on an RTX 20xx (no such card here).
   - A comparison against Linux output PNGs.
   - Done since (2026-11-23): ComfyUI portable with `python_embeded` (a clean install of fresh wheels, all 19 nodes ran
     one model each, weights and kernel DLLs downloaded on the first run), and Restore faces end to end on whole
     photos (codeformer and gfpgan, RetinaFace detection and paste-back, 27343c97).
