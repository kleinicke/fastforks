# ComfyUI nodes

`integrations/comfyui/` is a ComfyUI node pack over the public API (`fastforks.load(...)`, then a call per image).
It adds no models of its own: every dropdown lists the fastforks ids of that task.

## Install

The pack is the folder `integrations/comfyui`, not the repository: clone the repository outside `custom_nodes`
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

`requirements.txt` installs `fastforks-core[timm,transformers]>=0.2` from PyPI and keeps ComfyUI's torch (it needs
torch >= 2.8); use `pip install -e "./fastforks[timm,transformers]"` instead to run the cloned source. That is the
commercial-safe set; non-commercial and GPL models need an opt-in (see [Licences](#licences)). The Comfy registry and ComfyUI-Manager can install the pack once it is published from `integrations/comfyui` (installing
the repository URL with ComfyUI-Manager's "Install via Git URL" clones the root, which is not a node pack).

Windows: the `kernels` backend fails (the CUDA kernels are built for Linux x86-64 so far) and `auto` runs the
portable PyTorch backend, with one warning per model load. Paths stay short: the cache adds at most ~110 characters
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
| Parse clothes / face (non-commercial) | as Semantic segmentation (SegFormer b2 clothes, b5 face parsing) | as Semantic segmentation |
| Stereo disparity | left IMAGE, right IMAGE (rectified) | IMAGE: disparity scaled to 0-1 per image (near = white) |
| Optical flow | IMAGE batch (frames) | IMAGE (colour wheel, one per pair), FASTFORKS_FLOW (pixels, for Warp) |
| Warp by flow | IMAGE, FASTFORKS_FLOW | IMAGE: image i sampled along flow i (`frames[1:]` -> `frames[:-1]`) |
| Video segment (SAM 2 / SAMURAI) | IMAGE batch, points / box on frame 0, samurai | MASK batch |
| Track points | IMAGE batch, grid size or points on frame 0 | IMAGE (tracks drawn on the frames), STRING (JSON tracks) |
| Upscale / restore | IMAGE, tile (input px, 32 px overlap; 0 = one pass; keep it on for HAT / DAT) | IMAGE |
| Restore aligned faces (GFPGAN / CodeFormer) | IMAGE: aligned 512 x 512 face crops only (no detection / alignment / paste-back in this pack) | IMAGE |

Every node has `model`, `backend` (`auto`: our CUDA kernels where they are faster, else portable PyTorch;
`portable`; `kernels`) and `dtype` (`auto`: the validated one). Ids marked `[non-commercial]` or `[licence unclear]`
are not cleared for commercial use; non-commercial weights download only with `FASTFORKS_NONCOMMERCIAL=1` (see
[Licences](#licences)).

## Licences

The pack installs the commercial-safe default: `fastforks-core` ships the model families whose code is permissive,
and weights under a non-commercial licence do not download. Two separate steps open the rest; the pack takes neither
for you:

1. **Code**: families whose code is non-commercial or GPL (S2M2, FoundationStereo, MonSter, CoEx, CodeFormer,
   MatAnyone, Sapiens, ...) are in `fastforks-core[nc]` (or the `fastforks` metapackage, which adds every optional
   library too). Without it their ids are not listed. Install it into ComfyUI's own Python and restart ComfyUI:

   ```bat
   rem ComfyUI portable (Windows), in the ComfyUI_windows_portable folder
   python_embeded\python.exe -m pip install "fastforks-core[nc]"
   ```

   ```bash
   # a venv or conda install, with that environment active
   python -m pip install "fastforks-core[nc]"
   ```

2. **Weights**: models marked `[non-commercial]` (e.g. SegFormer b2 clothes and b5 face parsing of the Parse node,
   CodeFormer, S2M2) download their weights only with `FASTFORKS_NONCOMMERCIAL=1` in the environment ComfyUI starts
   from (copies already in the cache always load). ComfyUI portable: add the line `set FASTFORKS_NONCOMMERCIAL=1`
   above the `python_embeded\python.exe ...` line of `run_nvidia_gpu.bat` (or the `.bat` you start). Linux / macOS:
   `FASTFORKS_NONCOMMERCIAL=1 python main.py`. PowerShell: `$env:FASTFORKS_NONCOMMERCIAL = "1"`, then
   `python main.py`.

This is the library's own variable; the pack adds no switch of its own. At start the pack logs which state it runs in
(and how many published models are not installed), and the `model` tooltip of each node names the ones of its task
that are missing. A node whose model needs a step not taken fails with the library's error plus the ComfyUI command
for that step: a listed `[non-commercial]` model raises `LicenceError` when its weights would download; a shared
workflow with an id of a package that is not installed fails validation naming the package (instead of ComfyUI's
"Value not in list"). Dropdown entries keep the same text either way, so saved workflows stay valid.

Installing a package or setting the variable does not license anything: you are responsible for using each model
within its code and weights licences (`fastforks info MODEL` shows them, with the evidence). `[non-commercial]` means
research / non-commercial use only; GPL code carries its copyleft terms; `[licence unclear]` is not cleared for
commercial use. `FASTFORKS_COMMERCIAL_ONLY=1` makes the library refuse every model not cleared for commercial use.

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
own models first when a fastforks model needs the room. Batches run one image at a time, so a cancel stops after the
current image.
