# ComfyUI nodes on macOS: results

2026-11-22. The macOS counterpart of [comfyui-windows-results.md](comfyui-windows-results.md), for the steps of
[comfyui-windows-check.md](comfyui-windows-check.md) that apply to a Mac.

## Setup

- Machine: Apple M1 Pro, 16 GB, macOS 27.0.1. Free internal disk 25 GB at start, 24 GB at the end (everything is on
  `/Volumes/Daten`); swap used 1.6 GB at start, 2.4 GB at the end.
- ComfyUI 0.46.0 (`8b25bbc`, shallow clone) in `/Volumes/Daten/comfyui-mac/ComfyUI`, venv `/Volumes/Daten/comfyui-mac/venv`
  (uv, Python 3.13): torch 2.15.0 (stable PyPI wheel, not a nightly), `mlx` 0.32.4,
  `pip install -e <repo>[libs]` plus ComfyUI's `requirements.txt`.
- The pack is symlinked into `custom_nodes/comfyui-fastforks`. `FASTFORKS_CACHE` and `FASTFORKS_WEIGHTS_PATH` point at
  `/Volumes/Daten/fastforks-cache` / `fastforks-weights`, `FASTFORKS_OFFLINE=1`. Nothing was downloaded.
- Server: `python main.py --port 8188` (log: "Device: mps", "Set vram state to: SHARED"). The survey on the Mac had
  ended before any model ran; no other model job ran during the checks.
- Workflows in API format through `POST /prompt`, polled through `/history`; input: a 640 x 533 crop of a repo test
  image (`test_data/depth_scene/l_00_img.png`), four 384 px pan crops for the clips, a 160 x 120 crop for Upscale.
  Run 0 = first call (includes model load), run 1 = a second call with a different (mirrored) input. A repeat with an
  identical prompt returns in 0.0 s because ComfyUI caches the node result, so it is no timing.

## Upstream ComfyUI on macOS

- README: Comfy Desktop is the recommended install on macOS. `docs/installation.md` "Apple Mac silicon": install the
  PyTorch nightly following Apple's guide, install the manual-install requirements, `python main.py`. Stable torch 2.15
  from PyPI also works (device `mps`, vram state SHARED); no nightly was needed.
- No MLX support: `grep -ri mlx` over the clone finds nothing. ComfyUI's own models run on PyTorch MPS only. The MLX
  route in this run comes entirely from fastforks (`backend=auto` takes the MLX port of a family when it has one and
  `mlx` is installed).

## Checklist

| # | Step | Result |
|---|---|---|
| 1 | Install keeps ComfyUI's torch | pass. One venv, torch 2.15.0. |
| 2 | Start, 20 nodes, no import error | pass. All 20 `Fastforks*` nodes in `/object_info`; the pack imports in 1.3 s; no error or warning of the pack. |
| 3 | Depth `depth-anything-v2-s`, `auto` | pass, **MLX** (see below). |
| 4 | Remove background, Segment, Interpolate; unload | pass (MPS). |
| 7 | Video depth, Upscale | pass (MPS). |
| - | `tests/test_comfyui.py` | 31 of 32 passed; 1 fixed in the test (below). 32 passed after. |
| 1-7 Windows / CUDA only steps (long paths, kernels, bfloat16 on RTX 20xx) | not applicable. |

### Runs

Backend = what the log line `fastforks: loaded ... Backend(...)` reports for `backend=auto`, `dtype=auto`.

| Node | Model | Backend / dtype | Run 0 / 1 (s) | Output |
|---|---|---|---|---|
| Depth | depth-anything-v2-s | `kernels`, device **mlx**, float16 (`dpt_mlx`) | 3.66 / 0.12 | Plausible relative depth (0-255 range, near = white). |
| Remove background | birefnet-lite | portable, mps, float32 | 1.94 / 0.49 | RGBA, alpha 0-255. |
| Segment | sam2.1-hiera-small, one positive point | portable, mps, float32 | 1.10 / 0.20 | Mask of the seat under the point (5.7 % of the image). |
| Interpolate | rife-4.26, x2, 4 frames | portable, mps, float32 | 0.88 / 0.15 | 7 frames. |
| Video depth | video-depth-anything-s, 4 frames | portable, mps, float32 | 2.02 / 0.74 | One scale for the clip: per-frame max 247 / 251 / 250 / 255. |
| Upscale | realesrgan-x4plus, 160 x 120, `tile=0` | portable, mps, float32 | 1.20 / 0.43 | 640 x 480. |

(Depth and Remove background run 0 are from the first call of a fresh server; their "run 1" here is a second image
after the model was cached. Timings are noisy: the Mac is shared and these are single runs.)

### Behaviour

- **Auto route.** Only depth-anything-v2-s has an MLX port among the models tried; it is the only one that took the
  MLX route. The others stay on MPS because no port exists or the `mps_auto` policy keeps MPS.
- **Unload.** `POST /free` (`unload_models`, `free_memory`) logs `fastforks: unloading ... cached model(s)`. The
  server's `footprint` dropped from 2423 MB to 1055 MB. `ps` RSS is useless on macOS (stayed at about 885 MB). Earlier
  models had already been dropped by ComfyUI's memory pressure hook ("unloading ... for ComfyUI (all memory needed)"),
  so only one model was left to unload. Depth after the unload reloaded and ran in 0.31 s.
- **Cache.** Each model loads once (`fastforks: loaded` once per model); later runs reuse it.

## Fixes

- `tests/test_comfyui.py::test_flow_image_and_warp` asserted `img[1, 0, 0] == 0` exactly; on Apple arm64 `atan2` puts
  the colour wheel stop 1 ulp off and the value is 1.19e-7. The assertion now allows 1e-6 (a rounding effect, not a
  node bug; no golden or model tolerance touched).

## Not checked

Segment with a box / text, Video segment, Stereo, Flow, forced `backend=mlx` on the other nodes, Comfy Desktop on macOS,
and the MLX path of nodes beyond Depth.
