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

Every node has `model`, `backend` (`auto`: our CUDA kernels where they are faster, else portable PyTorch;
`portable`; `kernels`) and `dtype` (`auto`: the validated one). Ids marked `[non-commercial]` or `[licence unclear]`
are not cleared for commercial use; non-commercial weights download only with `FASTFORKS_NONCOMMERCIAL=1`.

Weights download on first use (`fastforks download MODEL` fetches them ahead). Models run on ComfyUI's device and
stay loaded between runs (the two most recently used); ComfyUI's "Unload models" frees them, and ComfyUI unloads its
own models first when a fastforks model needs the room. Batches run one image at a time, so a cancel stops after the
current image.
