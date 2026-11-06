# fastforks

Faster versions of published vision models behind one API: stereo matching, monocular and multi-view depth, image
classification, semantic and promptable segmentation (SAM), object detection and feature backbones (DINOv2, DINOv3).
You load a model by its id and get the original network's output, faster.

Every model has two backends; kernels run only where the GPU and driver fit (otherwise `auto` uses portable):

- **portable**: the original network in plain PyTorch, checked against the authors' code. Validated on CPU, NVIDIA
  CUDA and Apple MPS; other PyTorch devices are untested.
- **kernels**: the same network with our CUDA kernels (NVIDIA sm_80 or newer, Linux x86-64) and, for 19 families, Metal
  kernels on Apple silicon. Across the models and sizes we measured, the median is 1.9x faster than portable, with a
  quarter at 2.5x or more and the best at 10x (small classifiers at batch 1). A few plain ViT classifiers gain nothing
  ([benchmarks](https://github.com/kleinicke/fastforks/blob/main/docs/benchmarks.md)).

By default (`mode="auto"`) fastforks uses kernels where they can run and the recorded measurements prefer them, and portable everywhere else.

| task | package | call → result | example models |
|---|---|---|---|
| [stereo matching](https://github.com/kleinicke/fastforks/blob/main/docs/models/stereo.md) | `fastforks.stereo` | `model(left, right)` → disparity | `igev-stereo-sceneflow`, `raft-stereo-middlebury`, `las2-s` |
| [monocular depth](https://github.com/kleinicke/fastforks/blob/main/docs/models/depth.md) | `fastforks.depth` | `model(image)` → depth or inverse depth | `depth-anything-v2-s`, `moge-2-vitl-normal`, `depth-pro` |
| [multi-view depth](https://github.com/kleinicke/fastforks/blob/main/docs/models/multiview.md) | `fastforks.multiview` | `model(images)` → depth, points, cameras per view; calibrated MVS (mvsanywhere, patchmatchnet) also take `intrinsics=`, `cam_to_world=` (patchmatchnet also needs `depth_range=(near, far)` and returns the reference view only); VGGT / MapAnything: images only | `mvsanywhere`, `vggt-1b`, `mapanything-apache` |
| [image classification](https://github.com/kleinicke/fastforks/blob/main/docs/models/classification.md) | `fastforks.classification` | `model(image)` → logits, `top(k)` | `resnet50.tv2_in1k`, `efficientnet_b0.tv_in1k`, `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| [semantic segmentation](https://github.com/kleinicke/fastforks/blob/main/docs/models/segmentation.md) | `fastforks.segmentation` | `model(image)` → label map | `segformer-b0-ade20k`, `mask2former-swinl-ade20k`, `eomt-l-dinov2-ade20k` |
| [promptable segmentation: SAM / SAM 2](https://github.com/kleinicke/fastforks/blob/main/docs/models/promptable_segmentation.md) | `fastforks.promptable_segmentation` | `model(image, input_points=[[x, y]], input_labels=[1])` or `input_boxes=` → `masks [3, H, W]` bool + `iou_scores [3]` (one point) | `sam-vit-b`, `sam2.1-hiera-large` |
| promptable segmentation: SAM 3 concepts | `fastforks.promptable_segmentation` | `model(image, text="cat")` and / or exemplar boxes `input_boxes=` → every instance of the concept | `sam3` |
| [object detection](https://github.com/kleinicke/fastforks/blob/main/docs/models/detection.md) | `fastforks.detection` | `model(image)` → boxes, labels, scores | `faster-rcnn-r50-fpn-3x-coco`, `mask-rcnn-r50-fpn-3x-coco` |
| [feature backbones](https://github.com/kleinicke/fastforks/blob/main/docs/models/features.md) | `fastforks.features` | `model(image)` → token features, image embedding | `dinov2-vitb14`, `dinov3-vitl16`, `dinov3-convnext-small` |

The task links go to the model pages, which list every model of the task with its licences, accuracy, speed and
memory. Not sure which model to use? The
[overview](https://github.com/kleinicke/fastforks/blob/main/docs/overview.md) compares them per task and names the
fastest, the most accurate and the best commercially usable ones.

```python
from fastforks import stereo

model = stereo.load("igev-stereo-sceneflow")  # kernels on a supported NVIDIA GPU or Apple silicon, else portable
result = model("left.png", "right.png")
result.disparity  # [H, W] float32 tensor, pixels
result.backend  # which backend ran, on which device

cpu_model = stereo.load("igev-stereo-sceneflow", device="cpu")  # also with a CUDA build of PyTorch
```

One line per task (families built on timm, torchvision or transformers need that extra; see the install table below):

```python
# needs: pip install "fastforks-core[torchvision]"
from fastforks import classification, depth, detection, segmentation

depth.load("depth-anything-v2-s")("room.jpg").inverse_depth  # [H, W] float32 relative inverse depth
detection.load("rtdetrv2-s-coco")("street.jpg", score_threshold=0.5).to_list()  # dicts: box, class_name, score
classification.load("resnet50.tv2_in1k")("cat.jpg").top(3)  # [(class name, probability), ...]
segmentation.load("dinov2-s-linear-ade20k")("street.jpg").labels  # [H, W] class index per pixel
```

`mode="kernels"` or `mode="portable"` forces a backend; kernels raises with the reason if it cannot run here instead of
silently falling back. `fastforks bench igev-stereo-sceneflow` times both on your machine. For deployment, TensorRT can
be faster than kernels for a few models (see [export](https://github.com/kleinicke/fastforks/blob/main/docs/export.md)).
The [usage guide](https://github.com/kleinicke/fastforks/blob/main/docs/usage.md) has a quickstart per task and covers
precision, batches, video, several GPUs, `torch.compile`, ONNX and TensorRT export, and loading your own weights.

## Install

Install PyTorch for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/)); fastforks uses the
build you have, CUDA or CPU-only. Then:

```
pip install fastforks                            # everything: all models + timm, torchvision, transformers, ONNX
pip install "fastforks[tensorrt]"                # everything + TensorRT export

pip install fastforks-core                       # lean: every model whose code licence is permissive, no extra libraries
pip install "fastforks-core[nc]"                 # + models with non-commercial or GPL code (S2M2, FoundationStereo, MonSter, CoEx, ...)
pip install "fastforks-core[timm]"               # libraries some model families are built on
pip install "fastforks-core[torchvision]"
pip install "fastforks-core[transformers]"
pip install "fastforks-core[onnx]"               # ONNX export
pip install "fastforks-core[tensorrt]"           # TensorRT export
pip install "fastforks-core[libs]"               # timm, torchvision, transformers and ONNX at once
```

Without an extra, every task works except these (`MissingDependencyError` names the extra when one is missing):

| Task | Needs |
| --- | --- |
| classification | `[timm]` (11 models) or `[torchvision]` (15 models) |
| segmentation | `[transformers]` (SegFormer, Mask2Former, ...); the other 10 models need none |
| detection | `[torchvision]` for the R-CNN models |
| stereo, depth, features, promptable segmentation, multi-view | none |
| ONNX / TensorRT export | `[onnx]` / `[tensorrt]` |

`pip install fastforks` pulls a default torch build, which may be CPU-only on your platform. For a CUDA build install torch
first from [pytorch.org](https://pytorch.org/get-started/locally/) (e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu130`); `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

From a git checkout (every model, like `fastforks`):

```
uv pip install -e '.[libs]'                      # or pip install -e '.[libs]'; `pip install .` (no -e) works too
```

`fastforks` is `fastforks-core[nc,libs]`; both install the same `import fastforks`. For commercial products start from
`fastforks-core`. Weights with a non-commercial licence (S2M2, FoundationStereo, but also `bridgedepth-sf` with
permissive code) are never downloaded unless you set `FASTFORKS_NONCOMMERCIAL=1`; `weights=PATH` uses a copy you have.
A model whose library is missing fails in `load` with the extra to install. Weights and kernels are downloaded on first use
into `~/.cache/fastforks` and checked against a recorded SHA-256; `fastforks download MODEL` fetches them ahead of
time, and [docs/docker.md](https://github.com/kleinicke/fastforks/blob/main/docs/docker.md) covers offline machines
and Docker images. Weights you already have are used in place: files in the Hugging Face hub cache (e.g. from
`hf download`) or in a directory on `FASTFORKS_WEIGHTS_PATH`, or one file passed as `load(..., weights=PATH)`.
Gated weights (e.g. DINOv3): accept the licence on the model's Hugging Face page, log in (`hf auth login`, or
`huggingface_hub.login()` in Python), and load the model; the download goes through `huggingface_hub`. fastforks never
handles your Hugging Face token.

Requirements: Python >= 3.11 and PyTorch >= 2.8. Metal kernels run on Apple silicon (see the usage guide). The CUDA kernels need Linux on x86-64, an NVIDIA GPU of compute
capability 8.0 or newer (RTX 30 series, A100 and later) and a driver >= 570. So far it is tested only on an
RTX 5060 Ti: `fastforks env` shows what your GPU would use, and
[Other GPUs](https://github.com/kleinicke/fastforks/blob/main/docs/OTHER_GPUS.md) explains how to measure it and send
us the numbers.

## Models and licences

Many models are licensed for non-commercial use only. Each model page lists the code licence, the weights licence,
the training data and a commercial-use flag, which is `yes` only if both the code and the weights licence allow
commercial use. Each model directory keeps the original licence file, and weights are downloaded from the authors'
own hosts (a few from a mirror of the same files).

```python
import fastforks
from fastforks import stereo

stereo.list_models(commercial="yes")  # stereo models whose code and weights licences allow commercial use
print(fastforks.describe_models(stereo.list_models("igev*")))  # licences, and what blocks a "yes"
```

Training-data terms (SceneFlow, KITTI, ImageNet, ...) are shown as a note next to the flag; the
[usage guide](https://github.com/kleinicke/fastforks/blob/main/docs/usage.md#licences-and-commercial-use) explains how
the flag is set. None of this is legal advice.

## Command line

```
fastforks list [PATTERN] [--task TASK] [--commercial yes|no|unclear] [--tier stable|experimental|deprecated] [--format text|json|csv]
fastforks info depth-anything-v2-s                                       # one model in detail
fastforks download [MODEL ...] [--kernels]                               # fetch weights and kernels ahead of time
fastforks stereo run LEFT RIGHT --model MODEL --out DIR                  # a stereo model over many pairs (folders or globs)
fastforks bench igev-stereo-sceneflow [--mode auto|portable|kernels] [--size HxW]  # portable and kernels side by side
fastforks export MODEL --format onnx|trt [--size HxW] [--fp16] [-o FILE]
fastforks doctor                                                         # which models can run here, and what blocks the others
fastforks env                                                            # versions, GPU, kernel binaries (for bug reports)
```

`fastforks COMMAND --help` lists all options.

## Licence

The fastforks core, the wrappers and our kernels are Apache-2.0 (LICENSE). Vendored model code keeps its original
licence (the `LICENSE*` file in each model directory), and so do our modifications of it.

## Contributing

See [CONTRIBUTING.md](https://github.com/kleinicke/fastforks/blob/main/CONTRIBUTING.md); the checklist for adding a model
is there. What changed between releases: [CHANGELOG.md](https://github.com/kleinicke/fastforks/blob/main/CHANGELOG.md).
