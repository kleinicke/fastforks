# ComfyUI nodes

Two ComfyUI node packs over the public API (`fastforks.load(...)`, then a call per image). They add no models of their
own: every dropdown lists fastforks ids of that task.

- **fastforks** (`integrations/comfyui`, the default): every fastforks model, those not cleared for commercial use
  tagged `[non-commercial]`. Some models are for non-commercial use only; using each within its licence is your
  responsibility (see [Licences](#licences)). It installs the `fastforks` package (`fastforks-core` with every model
  family).
- **fastforks-commercial** (`integrations/comfyui-commercial`): the same nodes with only the models cleared for
  commercial use (code and weights licence); it never downloads non-commercial weights. It installs
  `fastforks-core[timm,transformers]`.

Each pack's README (`integrations/comfyui/README.md`, `integrations/comfyui-commercial/README.md`) lists its nodes and
models, generated from the registry (`tools/gen_comfyui_readme.py`).

## Install

With ComfyUI-Manager or the Comfy registry: `comfy node install fastforks` (or `fastforks-commercial`), then
restart ComfyUI.

From a clone: a pack is a folder of `integrations/`, not the repository. Clone the repository outside `custom_nodes`
(ComfyUI imports every folder there and reports the repository root as IMPORT FAILED) and link the folder in.

```bash
git clone https://github.com/kleinicke/fastforks
ln -s "$PWD/fastforks/integrations/comfyui" ComfyUI/custom_nodes/comfyui-fastforks   # Linux / macOS
pip install -r ComfyUI/custom_nodes/comfyui-fastforks/requirements.txt              # with ComfyUI's Python
```

Windows (cmd): a directory junction needs no administrator rights or developer mode (a symlink, `mklink /D`, does);
copying the folder also works but does not follow `git pull`.

```bat
mklink /J ComfyUI\custom_nodes\comfyui-fastforks C:\path\to\fastforks\integrations\comfyui
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-fastforks\requirements.txt
```

`requirements.txt` installs `fastforks>=0.2` from PyPI and keeps ComfyUI's torch (it needs torch >= 2.8); use
`pip install -e "./fastforks[libs]"` instead to run the cloned source (it has every model family). For the commercial-only pack link
`integrations/comfyui-commercial` instead (it finds the nodes in `integrations/comfyui` next to it); its
`requirements.txt` installs `fastforks-core[timm,transformers]>=0.2`. Installing the repository URL with
ComfyUI-Manager's "Install via Git URL" clones the root, which is not a node pack.

Windows: the `kernels` backend works like on Linux (the DLLs download once, like the weights, and need no C++ toolchain). A
library without a Windows build makes `kernels` fail for that model and `auto` run portable, with one warning per model load. Paths stay short: the cache adds at most ~110 characters
to the ComfyUI folder (`models\fastforks\blobs\sha256\ab\<sha256>`, downloads in a `.download.*` folder next to it),
so a ComfyUI folder below ~140 characters needs no long-path setting.

## Nodes (category `fastforks`)

| Node | Inputs | Outputs |
|---|---|---|
| Depth | IMAGE | IMAGE: inverse depth scaled to 0-1 per image (near = white); image models only |
| Video depth | IMAGE batch (frames of one clip) | IMAGE batch: Video Depth Anything run as one clip, one 0-1 scale for all frames (no flicker) |
| Normals | IMAGE | IMAGE: normal map (red = right, green = up, blue = towards the camera) |
| Remove background | IMAGE, trimap MASK (ViTMatte only) | IMAGE (RGBA), MASK (alpha) |
| Segment (SAM) | IMAGE, points `x,y; x,y`, box `x0,y0,x1,y1`, text (SAM 3) | MASK |
| Detect objects | IMAGE, score threshold, classes (open-vocabulary query or class filter) | MASK, STRING (JSON boxes) |
| Pose | IMAGE, detections (JSON of a Detect node; top-down models) | IMAGE (skeletons on black), STRING (JSON keypoints) |
| Interpolate frames | IMAGE batch, multiplier | IMAGE batch with `multiplier - 1` new frames per pair |
| Classify | IMAGE, top k | STRING |
| Tag (WD tagger) | IMAGE, general / character thresholds (0.35 / 0.75) | STRING: Danbooru tags, most likely first |
| NSFW check | IMAGE | STRING (normal / nsfw per image), FLOAT (highest nsfw probability of the batch) |
| Semantic segmentation | IMAGE, classes for the mask | IMAGE (class colours), MASK (listed classes), STRING (classes found), MASK (one per listed class) |
| Parse clothes / face (not in fastforks-commercial) | as Semantic segmentation (SegFormer b2 clothes, b5 face parsing) | as Semantic segmentation |
| Stereo disparity | left IMAGE, right IMAGE (rectified) | IMAGE: disparity scaled to 0-1 per image (near = white) |
| Optical flow | IMAGE batch (frames) | IMAGE (colour wheel, one per pair), FASTFORKS_FLOW (pixels, for Warp) |
| Warp by flow | IMAGE, FASTFORKS_FLOW | IMAGE: image i sampled along flow i (`frames[1:]` -> `frames[:-1]`) |
| Video segment (SAM 2 / SAMURAI) | IMAGE batch, points / box on frame 0, samurai | MASK batch |
| Track points | IMAGE batch, grid size or points on frame 0 | IMAGE (tracks drawn on the frames), STRING (JSON tracks) |
| Upscale / restore | IMAGE, tile (input px, 32 px overlap; 0 = one pass; keep it on for HAT / DAT) | IMAGE |
| Restore aligned faces (GFPGAN, CodeFormer) | IMAGE: aligned 512 x 512 face crops only (no detection / alignment / paste-back in this pack) | IMAGE |

Every node has `model`, `backend` (`auto`: our CUDA kernels where they are faster, else portable PyTorch;
`portable`; `kernels`) and `dtype` (`auto`: the validated one). Both packs name a node `Fastforks<Name>` (shown as
"Depth (fastforks)" or "Depth (fastforks, commercial)"), so a workflow saved with one pack loads in the other. With both
installed in `custom_nodes`, fastforks-commercial registers nothing and logs that the fastforks pack's nodes are used
(two packs cannot register one class name; remove or disable one to choose). A pack has a node only if it lists a
model there: fastforks-commercial has no Parse (both of its models are non-commercial); Warp takes no model.

## Licences

**fastforks** lists every installed model; those not cleared for commercial use are tagged (`[non-commercial]`:
non-commercial / research use only, GPL code: its copyleft terms; `[licence unclear]`: not cleared). Installing the
pack is the explicit opt-in to their weights: its nodes download them for their own runs (each run is inside the
library's `allow_noncommercial()`); nothing is set in the environment. Each node's description and `model` tooltip say
which models are non-commercial and that using each within its licence is your responsibility; the startup log says
how many there are.

**fastforks-commercial** lists only ids whose code and weights licence are cleared for commercial use, and runs no
other: a saved or shared workflow naming another model fails validation (or, linked from another node, the run) with
one sentence, that the model is only in the fastforks pack and how to install it, never a download.

Installing a pack does not license anything: you are responsible for using each model within its code and weights
licences (`fastforks info MODEL` shows them, with the evidence). `FASTFORKS_COMMERCIAL_ONLY=1` makes the library refuse
every model not cleared for commercial use, in both packs.

A node lists only the ids it can run: those whose required inputs it passes (`model_card(id).contract().inputs`).
Depth-prompted and depth-completion models (Prompt Depth Anything, LingBot-Depth, OMNI-DC, Any2Full), video matting
from a first-frame mask (MatAnyone), in-context segmentation (INSID3) and inpainting (LaMa) need an input no node has
yet, so no node lists them.

Upscale also lists the files in `models/upscale_models` (`file: <name>`) whose architecture fastforks reads, detected
from the weights (`fastforks.restoration.load(path)`): ESRGAN / Real-ESRGAN (old or new key names), Compact (SRVGG),
SPAN, SwinIR, Swin2SR, HAT, DAT, DRCT, RealPLKSR, NAFNet, Restormer. Other files are not listed; `load(path)` names
the reason.

Weights and kernels already in the fastforks cache, the Hugging Face or torch hub cache are used in place; missing
ones download on first use into `models/fastforks` (another location: a `fastforks:` entry in
`extra_model_paths.yaml`). Models run on ComfyUI's device and
stay loaded between runs (the two most recently used); ComfyUI's "Unload models" frees them, and ComfyUI unloads its
own models first when a fastforks model needs the room. When ComfyUI needs room for one of its models and unloading its
own was not enough, the fastforks models on that device free their CUDA graphs, then unload, least recently used first. By default batches run one image at a time, so a cancel stops after the
current image. Depth, Normals, Remove background (without a trimap), Detect, Pose, Classify, Tag, NSFW check, Semantic
segment, Parse, Face restore and Stereo have an optional `batch_size` input: several images (pairs) per model call,
which is several times faster for small models at 224-512 px (convnext-tiny 8.6x, depth-anything-v2-s 1.8x, rf-detr-s
5.9x; 480 x 640: rtmo-s pose 2.3x, yunet-n faces 2.7x, stereo coex 1.8x, aanet 2.1x, on an RTX 5060 Ti under
Windows) and uses more memory; models that cannot
batch raise an error, stereo models that gain nothing from it run one pair at a time. Interpolate and Flow take `batch_size`
consecutive pairs per call, 8 by default (384 x 384: rife-4.26 5.6x; 480 x 640: rife-4.26 2.6x, raft-small 3.2x,
neuflow-v2 2.7x); models that gain nothing from it (sea-raft, gmflow, film, larger frames) still run them one pair at a
time. A cancel stops after the current `batch_size` pairs.

## Speed

A node's time is the model's time plus what ComfyUI does around every node, which any pack pays. On ComfyUI 0.46
(RTX 5060 Ti, Windows):

- IMAGE outputs live on the CPU, so each node copies its result off the GPU: ~6 ms for a 2048 x 2048 output.
- Dynamic VRAM (the default) synchronises after each node: 1-5 ms per node. `--disable-dynamic-vram` removes it
  (SPAN x4 on 512 x 512: 28.2 -> 24.4 ms per run, GFPGAN: 31.7 -> 26.8 ms).
- The executor adds ~8 ms per run.
- When the GPU is mostly idle between short node calls (small images, a few frames, a run every few hundred ms), the
  NVIDIA driver drops the memory clock to its lowest state (P5, 810 MHz) after a few seconds, even mid-run, and GPU work
  then takes about three times as long (rife-4.26 x2 on 4 frames of 384 x 384: 12 -> 35 ms per node). Calls back to back
  (the library's benchmarks) stay at full clock. The driver setting that keeps the clock up is NVIDIA Control Panel >
  Manage 3D settings > Power management mode > Prefer maximum performance (not measured here).

Models that take a few milliseconds (SPAN, Compact, the WD tagger, SegFormer) therefore take about twice their library
time in a workflow (SPAN x4: 13 ms model, 28 ms node); for slow ones (HAT, DAT, SwinIR) the overhead does not matter. The portable ComfyUI build and a
ComfyUI from git run the nodes at the same speed.

## Memory

ComfyUI runs PyTorch on the `cudaMallocAsync` allocator (unless started with `--disable-cuda-malloc`). The 70 kernels
models that fit the test machine's local weights, run that way at the [benchmarks](benchmarks.md)' inputs (RTX 5060
Ti, Windows), gave the same outputs at the same speed as under PyTorch's default allocator, with less memory reserved
in total (131 against 138 GiB summed over the models); the benchmarks' memory columns use the default allocator.
PyTorch's `torch.cuda.max_memory_allocated` reports about twice the peak under `cudaMallocAsync` for models that
capture CUDA graphs, but the device memory in use (`cudaMemGetInfo`) is not higher: ema-vfi at 1080 x 1920 peaks at
2350 MiB against 3376 MiB under the default allocator, aliked-lightglue at 768 x 1024 at 1456 against 1520 MiB.
