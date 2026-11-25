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

## Registry archives

Comfy CLI `c1abf5a` validated and packed both integration folders at Fastforks `5b09289e`. In fresh ComfyUI
install folders, the template API served all 14 full-pack templates and all 13 commercial templates byte for byte.
The extracted full pack passed w02, w06 and whole-photo face restoration. The extracted commercial pack passed the
Fastforks branches of all thirteen templates. Both packs reproduced w02 after eviction.

Rebuilt 0.2.1 wheels at `9a85ce60` also passed dependency checks and archive workflows without an editable library
install. The full pack served all 14 templates and passed w02, w03, w08 and w09 on PyTorch 2.15.0+cu132.
The commercial pack served all 13 templates and passed w02, w06 and w09 on PyTorch 2.14.1+cu130, with only
`fastforks-core` installed. Both installations reproduced w02 after `/free`.

Separate prompts exercised the nine node types outside the shipped workflows on PyTorch 2.15.0+cu132. Normals,
Semantic Segment and Parse returned images and their auxiliary outputs. Classify, Tag and NSFW Check produced text
or scores. Video Depth, Video Segment and Point Track returned four outputs for four input frames. The tracking
JSON contained four finite point positions per frame. Together with the workflows and face restoration, this
covered all 24 node types.
In whole-photo face restoration, CodeFormer fidelity 0.2 and 0.9 produced different 512 × 512 outputs. Sending one
frame to Interpolate preserved the source preview byte for byte.

On a fresh install with empty caches, the pack wrote its input schema cache at startup. Cancelling Depth during the
public checkpoint download produced a ComfyUI interrupt event and emptied the download directory. Queueing again
verified and cached the checkpoint and rendered a 768 × 768 depth preview.

Both rebuilt archives recovered from a malformed on-disk input cache at startup and completed their smoke
workflows. In a separate test, six cycles of w02, w09 and `/free` reproduced all previews exactly under both
CUDA allocators. Warmed process memory after unloading was stable at 660 MiB with `cudaMallocAsync` and
722 MiB with the native allocator, measured with `nvidia-smi`.

Both archives also passed their smoke workflows against the wheels from `ca21523f`. The full archive repeated
the empty-cache public download, cancellation and retry check with those wheels. Two clean builds of all three
wheels produced identical SHA-256 digests.

Publishing the wheels and the Comfy Registry packs are release steps. The public Registry lookup returned
`Publisher not found` for the configured `kleinicke` ID on 2026-11-24. Confirm the publisher account and its
ID before publishing either pack. See the [Windows results](comfyui-windows-results.md) and
[macOS results](comfyui-mac-results.md) for the other platform runs.
