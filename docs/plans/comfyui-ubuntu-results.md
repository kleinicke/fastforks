# ComfyUI on Ubuntu

2026-11-24. Ubuntu 26.04.1 x86-64, RTX 5060 Ti 16 GB, NVIDIA driver 595.99.02, Python 3.12, PyTorch 2.14.1+cu130.
Fastforks `dev/v0.2.1` at `d8099bff` with the cache and CUDA follow-up in `6bda5e54`. ComfyUI `4806189d`, the same
revision used in the Windows validation. Its dependencies installed alongside the existing CUDA torch and
Transformers 5.19.0. The test server listened on localhost with `--offline --cache-none`. Weights and kernels came
from the workstation's caches. One ComfyUI process used CUDA at a time.

## Workflow runs

The stock `example.png`, cropped and translated frames, a rectified synthetic pair, and person images from the
RTMO and RetinaFace test fixtures supplied the inputs. The Fastforks branches of the shipped JSONs ran via the
server's `/prompt` and `/history` endpoints. Optional comparison branches from other packs and ComfyUI's separate
upscaler were omitted.

| Workflow | Result |
| --- | --- |
| w01 depth | Four 768 × 768 maps. |
| w02 backend compare | All three branches ran. The `auto` and `kernels` previews matched byte for byte. |
| w03 background removal | All five outputs were RGBA, including IS-Net. |
| w04 prompted segmentation | All six masks returned, including the SAM 3 and CLIPSeg text branches. |
| w05 detection | Four model branches ran on the person image. |
| w06 pose | Four pose branches ran, including top-down models linked to the Detect node. |
| w07 upscale | Four outputs at 640 × 480 from a 160 × 120 crop. |
| w08 interpolation | Seven frames from four on each of the four branches at 2×. |
| w09 stereo | Four 640 × 448 maps, with CUDA kernels selected for all four models. |
| w10 flow and warp | Two outputs per branch from three input frames, including the warp. |
| w11 edges | Six edge or line maps. |
| w12 inpaint | The SAM 2 mask branch and both LaMa branches ran. |
| w13 segment like | Three masks from the guide image and mask. |
| w14 video matte | Four alpha maps. |
| Face Restore | GFPGAN v1.4 and RetinaFace ResNet-50 used CUDA kernels. Whole-photo detection and paste-back changed the output and preserved 512 × 512. |

Both node packs installed together registered 24 classes. The server logged that the full pack supplied the nodes.
The commercial pack alone registered 22 classes, and its model lists omitted non-commercial entries. The same
w02 workflow ran in both installations. In the full pack, w02, w03 and w09 were repeated after ComfyUI's `/free`
request with both `unload_models` and `free_memory`. The log recorded model eviction and reloading, and the
decoded previews matched byte for byte. The commercial pack's w02 reload also matched.

The source packs used local weights. A separate virtual environment installed the three local 0.2.1 wheels and
ComfyUI, with the library checkout off the import path. Its w02, w03, w08 and w14 runs passed. The standalone
commercial pack also passed w02 and reproduced the output after eviction with only `fastforks-core` installed.
Publishing the wheels and the Comfy Registry packs are release steps. See the [Windows results](comfyui-windows-results.md) and
[macOS results](comfyui-mac-results.md) for the other platform runs.
