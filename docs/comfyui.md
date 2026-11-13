# ComfyUI nodes

`integrations/comfyui/` is a ComfyUI node pack over the public API (`fastforks.load(...)`, then a call per image).
It adds no models of its own: every dropdown lists the fastforks ids of that task.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kleinicke/fastforks
ln -s fastforks/integrations/comfyui comfyui-fastforks    # Windows: copy the folder instead
pip install "./fastforks[timm,transformers]"               # with ComfyUI's Python (portable: python_embeded\python.exe -m pip)
```

Once fastforks is on PyPI, ComfyUI-Manager and the Comfy registry install the pack with its `requirements.txt`.

## Nodes (category `fastforks`)

| Node | Inputs | Outputs |
|---|---|---|
| Depth | IMAGE | IMAGE: inverse depth scaled to 0-1 per image (near = white) |
| Normals | IMAGE | IMAGE: normal map (red = right, green = up, blue = towards the camera) |
| Remove background | IMAGE, trimap MASK (ViTMatte only) | IMAGE (RGBA), MASK (alpha) |
| Segment (SAM) | IMAGE, points `x,y; x,y`, box `x0,y0,x1,y1`, text (SAM 3) | MASK |
| Detect objects | IMAGE, score threshold, classes (open-vocabulary query or class filter) | MASK, STRING (JSON boxes) |
| Pose | IMAGE, detections (JSON of a Detect node; top-down models) | IMAGE (skeletons on black), STRING (JSON keypoints) |
| Interpolate frames | IMAGE batch, multiplier | IMAGE batch with `multiplier - 1` new frames per pair |
| Classify | IMAGE, top k | STRING |
| Tag (WD tagger) | IMAGE, general / character thresholds (0.35 / 0.75) | STRING: Danbooru tags, most likely first |
| NSFW check | IMAGE | STRING (normal / nsfw per image), FLOAT (highest nsfw probability of the batch) |
| Semantic segmentation | IMAGE, classes for the mask | IMAGE (class colours), MASK (listed classes), STRING (classes found) |
| Stereo disparity | left IMAGE, right IMAGE (rectified) | IMAGE: disparity scaled to 0-1 per image (near = white) |
| Optical flow | IMAGE batch (frames) | IMAGE (colour wheel, one per pair), FASTFORKS_FLOW (pixels, for Warp) |
| Warp by flow | IMAGE, FASTFORKS_FLOW | IMAGE: image i sampled along flow i (`frames[1:]` -> `frames[:-1]`) |
| Video segment (SAM 2 / SAMURAI) | IMAGE batch, points / box on frame 0, samurai | MASK batch |
| Track points | IMAGE batch, grid size or points on frame 0 | IMAGE (tracks drawn on the frames), STRING (JSON tracks) |
| Upscale / restore | IMAGE, tile (input px, 32 px overlap; 0 = one pass; keep it on for HAT / DAT) | IMAGE |

Every node has `model`, `backend` (`auto`: our CUDA kernels where they are faster, else portable PyTorch;
`portable`; `kernels`) and `dtype` (`auto`: the validated one). Ids marked `[non-commercial]` or `[licence unclear]`
are not cleared for commercial use; non-commercial weights download only with `FASTFORKS_NONCOMMERCIAL=1`.

Upscale also lists the files in `models/upscale_models` (`file: <name>`) whose architecture fastforks runs, detected
from the weights: ESRGAN / Real-ESRGAN (RRDBNet, 64 features, x2 / x4, old or new key names), SPAN and SwinIR. Other
files (e.g. Compact / SRVGG) are not listed; `fastforks.restoration.load_file(path)` names the reason.

Weights and kernels already in the fastforks cache, the Hugging Face or torch hub cache are used in place; missing
ones download on first use into `models/fastforks` (another location: a `fastforks:` entry in
`extra_model_paths.yaml`). Models run on ComfyUI's device and
stay loaded between runs (the two most recently used); ComfyUI's "Unload models" frees them, and ComfyUI unloads its
own models first when a fastforks model needs the room. Batches run one image at a time, so a cancel stops after the
current image.
