# ComfyUI nodes on Windows: checklist

Status 2026-11-20: all steps run on Windows 11 (ComfyUI 4806189de in a venv with torch 2.14.1+cu130, RTX 5060 Ti)
through the ComfyUI API; results, timings and fixes in [comfyui-windows-results.md](comfyui-windows-results.md).
ComfyUI portable (`python_embeded`, 2026-11-23): fresh wheels installed as `docs/comfyui.md` says, with an empty
weights cache. All 19 nodes ran one model each, and weights and kernel DLLs downloaded on the first run. Not yet run:
bfloat16 on an RTX 20xx.

## What the code already handles
- The pack (`integrations/comfyui/`) uses torch, numpy and PIL only: no paths, no POSIX calls, no subprocesses.
- Kernels: since 0.2.1 the kernel libraries also run on Windows x64 ([windows-kernels.md](windows-kernels.md)). The
  cubin libraries load unchanged and the shared ones load as DLLs. The POSIX-only code (the cache-permission check
  and `os.memfd_create`) is skipped on Windows. On other platforms `resolve` raises `KernelUnavailableError`;
  `backend=auto` then loads the portable backend and logs one warning.
- Weights cache: `platformdirs.user_cache_dir("fastforks")` (`%LOCALAPPDATA%\fastforks\Cache`); `FASTFORKS_CACHE`
  moves it.

## To check on Windows (ComfyUI portable, its bundled torch + CUDA)
1. `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-fastforks\requirements.txt` installs
   fastforks-core without replacing the bundled torch (pip must keep the installed torch; if it tries to upgrade it,
   the `torch` pin of fastforks-core is too narrow).
2. ComfyUI starts, the 20 `fastforks` nodes appear, no import error in the console.
3. Depth with `depth-anything-v2-s`, `backend=auto`: kernels mode (0.2.0: portable, with one warning "kernels ...
   built for Linux x86-64"); the output matches the Linux result within the depth family tolerance.
4. Remove background (`birefnet-lite`), Segment (`sam2.1-hiera-small`), Interpolate (`rife-4.26`): run, outputs look
   right; "Unload models" frees the VRAM (Task Manager, dedicated GPU memory drops).
5. Long paths: the weights cache under `%LOCALAPPDATA%` stays below 260 characters (Hugging Face snapshot paths are
   the longest); if not, set `FASTFORKS_CACHE=C:\ff`.
6. `backend=portable` with `dtype=float16`, then `bfloat16` on an RTX 20xx (no bf16): the error names the dtype.
7. Video depth (`video-depth-anything-s`) on a 4-frame batch, then Depth: the Depth list holds no video model; the clip
   output has one 0-1 scale over all frames. Upscale with a `.pth` in `models/upscale_models` (`file: ...`).
