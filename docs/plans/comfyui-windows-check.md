# ComfyUI nodes on Windows: checklist

Status 2026-11-13: tested on Linux only (ComfyUI dc9ffe6a, RTX 5060 Ti). Expected to work on Windows with the
portable backend; to be confirmed on a Windows machine with these steps.

## What the code already handles
- The pack (`integrations/comfyui/`) uses torch, numpy and PIL only: no paths, no POSIX calls, no subprocesses.
- Kernels: `fastforks._core.kernels.resolve` raises `KernelUnavailableError` on anything but Linux x86-64 before it
  touches POSIX-only code (`grp`/`pwd`/`os.getuid` in the cache-permission check, `os.memfd_create`); `backend=auto`
  then loads the portable backend and logs one warning. `backend=kernels` fails with that message (wanted).
- Weights cache: `platformdirs.user_cache_dir("fastforks")` (`%LOCALAPPDATA%\fastforks\Cache`); `FASTFORKS_CACHE`
  moves it.

## To check on Windows (ComfyUI portable, its bundled torch + CUDA)
1. `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-fastforks\requirements.txt` installs
   fastforks-core without replacing the bundled torch (pip must keep the installed torch; if it tries to upgrade it,
   the `torch` pin of fastforks-core is too narrow).
2. ComfyUI starts, the 8 `fastforks` nodes appear, no import error in the console.
3. Depth with `depth-anything-v2-s`, `backend=auto`: one warning "kernels ... built for Linux x86-64", output
   matches the Linux result within the depth family tolerance.
4. Remove background (`birefnet-lite`), Segment (`sam2.1-hiera-small`), Interpolate (`rife-4.26`): run, outputs look
   right; "Unload models" frees the VRAM (Task Manager, dedicated GPU memory drops).
5. Long paths: the weights cache under `%LOCALAPPDATA%` stays below 260 characters (Hugging Face snapshot paths are
   the longest); if not, set `FASTFORKS_CACHE=C:\ff`.
6. `backend=portable` with `dtype=float16`, then `bfloat16` on an RTX 20xx (no bf16): the error names the dtype.
