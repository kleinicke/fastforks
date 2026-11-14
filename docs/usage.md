# Usage guide

Not sure which model to use? The [overview](overview.md) compares them per task: commercial status, accuracy, speed against the original code and memory.

## Install

From a git checkout: `uv pip install -e '.[libs]'` (or `pip install -e '.[libs]'`; a plain `pip install .` works too)
installs every model plus timm, torchvision, transformers and ONNX (`[libs]`).

`pip install fastforks` installs everything: every model plus timm, torchvision, transformers and ONNX export.
`pip install fastforks-core` is the lean install: the API and every model whose code licence is permissive, no optional
libraries; its extras add the rest. `fastforks-core[nc]` adds the models with non-commercial or GPL code (S2M2,
FoundationStereo, MonSter, CoEx, ...; see [licences](#licences-and-commercial-use)), `fastforks-core[timm]` /
`[torchvision]` / `[transformers]` the libraries some families are built on, `fastforks-core[onnx]` ONNX export and
`fastforks-core[libs]` all of those libraries; `fastforks` is `fastforks-core[nc,libs]`. TensorRT is a separate extra
(`fastforks-core[tensorrt]`, or `fastforks[tensorrt]`). A model whose library is missing fails in
`load` with the pip command to run, before any weight is read.
`fastforks.model_card(id).requires` (or the task's `model_card`) shows what a model needs and
`list_models(runnable=True)` lists the models whose dependencies are installed and whose non-commercial opt-in is set; it does not check weight access (gated repos), downloads or the offline cache.

A release is the `fastforks-core` wheel with the permissive model code, the `fastforks-nc` wheel with the rest and the
`fastforks` metapackage (no code), so `fastforks-core` never pulls non-commercial or GPL code.

### Cache and environment variables

Weights and kernel binaries are downloaded on first use into the user cache (`~/.cache/fastforks` on Linux; the
download line names the folder; `fastforks env` prints it with its size). It grows with every model you use; to free
space delete the folder (`fastforks download MODEL` refetches). `XDG_CACHE_HOME` is honoured. Concurrent downloads of
the same file are safe but not deduplicated. Path variables expand `~`; relative paths resolve against the current
directory.

| variable | meaning |
|---|---|
| `FASTFORKS_CACHE` | cache directory (default: the platform user cache) |
| `FASTFORKS_WEIGHTS_PATH` | extra directories searched for weight files before downloading (`os.pathsep`-separated) |
| `FASTFORKS_READ_CACHES` | other fastforks cache directories whose files are used in place, never written (`os.pathsep`-separated; e.g. the user cache when an app moves `FASTFORKS_CACHE`) |
| `FASTFORKS_KERNEL_PATH` | extra directories searched for kernel binaries (offline machines, own builds) |
| `FASTFORKS_OFFLINE` | `1` = never touch the network (also `HF_HUB_OFFLINE=1`); missing files raise `WeightsDownloadError` |
| `FASTFORKS_NONCOMMERCIAL` | `1` = also download weights with non-commercial licences (off in every install, also `fastforks`) |
| `FASTFORKS_COMMERCIAL_ONLY` | `1` = refuse (and hide from `list`, `download`) every model not cleared for commercial use |
| `FASTFORKS_HF_ENDPOINT` | redirects only our own Hugging Face repos (kernels, mirrored weights) to a mirror or `file://` stand-in, not the authors' hosts; for a full mirror use `FASTFORKS_WEIGHTS_PATH` |

The three on/off variables take `1/true/yes/on` or `0/false/no/off/empty` (any case); any other value raises
`OptionsError`. `mode` and `device` are load arguments; there are no `FASTFORKS_MODE` / `FASTFORKS_DEVICE` variables.
| `HF_TOKEN` | sent only to our own Hugging Face repos (private or rate-limited access); other repos: `huggingface_hub`'s own login |

## Quickstart per task

Every task works the same way: `<task>.load(model_id)` once, then call the model; `<task>.list_models()` (or
`fastforks.list_models()` for all tasks, `fastforks list` on the command line) shows the installed ids.

Stereo matching ([models](models/stereo.md), [API](api/stereo.md)):

```python
from fastforks import stereo

model = stereo.load("igev-stereo-sceneflow")  # our kernels on a supported GPU, else portable PyTorch (GPU or CPU)
result = model("left.png", "right.png")
result.disparity  # [H, W] float32 tensor on the model's device, pixels at the input resolution (x_left - x_right)
result.backend  # what ran: Backend(mode='kernels', device='cuda:0', dtype='float16', kernels=(...))
```

Monocular depth ([models](models/depth.md), [API](api/depth.md)):

```python
from fastforks import depth

model = depth.load("depth-anything-v2-s")
result = model("room.jpg")  # metric models: model("room.jpg", intrinsics=K), K [3, 3] in pixels (see below)
result.inverse_depth  # [H, W] float32; result.kind says what the maps mean (here relative inverse depth)
result.backend
```

`result.kind` is `metric` (metres) or a relative kind (unknown scale or shift: `da3mono-large`, MiDaS, Depth Anything V2). Metric
models differ in the camera: `metric3d-vit-large` and `da3metric-large` need the real `intrinsics=` for correct metres
(`da3metric-large`: metres = output x focal / 300; without `intrinsics=` it assumes a 300 px focal length, which is
too small for a real camera), `unidepth-v2-vitl14`, `unik3d-vitl`, `depth-pro` and MoGe estimate the camera (given
`intrinsics=` replace `fx`) and return it as `result.intrinsics`, `zoedepth-nk` takes none. `fastforks info MODEL` shows `kind`.

Other tasks work the same way (families built on timm, torchvision or transformers need that extra):

```python
# needs: pip install "fastforks-core[torchvision]"
from fastforks import classification, features, segmentation

classification.load("resnet50")("cat.jpg").top(3)  # [(class name, probability), ...]
segmentation.load("dinov2-s-linear-ade20k")("street.jpg").labels  # [H, W] class index per pixel
features.load("dinov2-vitb14")("cat.jpg").pooler_output  # image embedding; .last_hidden_state has all tokens
```

```python
# needs: pip install "fastforks-core[torchvision]" "fastforks-core[nc]"
# MVSAnywhere's code and weights are non-commercial: its weights download only with FASTFORKS_NONCOMMERCIAL=1
import torch

from fastforks import calibration, detection, edges, flow, matting, multiview, pose, promptable_segmentation, retrieval

det = detection.load("faster-rcnn-r50-fpn-3x-coco")
det("street.jpg", score_threshold=0.5).to_list()  # dicts: box, class_name, score
# defaults of every detection family: score >= 0.05, at most 100 boxes (load options score_threshold, max_detections)
# a batch gives list[DetectionResult] (one per image; the only task that returns a list, the others a batched result)
sam = promptable_segmentation.load("sam-vit-b")
# one point (x, y) in pixels, label 1 = foreground: masks [3, H, W] bool (3 candidates), iou_scores [3]
r = sam("cat.jpg", input_points=[[320, 240]], input_labels=[1])
best = r.masks[r.iou_scores.argmax()]  # [H, W]; several prompts add a leading axis; also input_boxes=, text= (SAM 3)
# SAM 3's text prompt cuts scores with threshold= (a call keyword only), detection with score_threshold=
multiview.load("mvsanywhere")(
    ["a.jpg", "b.jpg"], intrinsics=K, cam_to_world=poses
).depth  # [1, H, W]: the reference view (reference= picks it)
people = pose.load("rtmpose-m")(  # top-down: one box (x0, y0, x1, y1 in pixels) per person
    "street.jpg", boxes=[[40, 30, 220, 400]]
)  # e.g. the person boxes of a detector
people.keypoints, people.keypoint_scores  # [P, 17, 2] pixels of the input, [P, 17]; names: people.keypoint_names
flow.load("raft-large")("frame1.png", "frame2.png").flow  # [2, H, W]: dx, dy in pixels, frame 1 -> frame 2
matting.load("birefnet-lite")("photo.jpg").alpha  # [H, W] in [0, 1]: 1 = foreground (background removal)
edges.load("lineart")("photo.jpg").edges  # [H, W] in [0, 1]: 1 = line (ControlNet preprocessor)
vpr = retrieval.load("megaloc")
d = [vpr(p).descriptor for p in paths]  # each [8448], unit norm; any image sizes
retrieval.pairs(torch.stack(d), k=20)  # [P, 2] image pairs to match (each image with its 20 nearest), e.g. for SfM

cam = calibration.load("geocalib-pinhole")(image)  # one photo: no checkerboard, no EXIF
cam.intrinsics, cam.roll, cam.pitch, cam.vfov  # [3, 3] (pixels of the input), radians
```

3D CT segmentation ([models](models/volume.md), [API](api/volume.md)); also any trained nnU-Net v2 folder via
`volume.load_nnunet(folder)`:

```python
# read_nifti / write_nifti need: pip install "fastforks-core[nifti]"
from fastforks import volume

ct, meta = volume.read_nifti("ct.nii.gz")  # [D, H, W] float32 (RAS-canonical), meta.spacing in mm
seg = volume.load("totalsegmentator-total-3mm")(ct, meta.spacing)
seg.labels  # [D, H, W] uint8 class index on the input grid; seg.class_names {1: "spleen", ...}
volume.write_nifti("seg.nii.gz", seg.labels.cpu().numpy(), meta)  # on the grid of ct.nii.gz
```

How a call runs: the model card (`card.toml`) is resolved to a variant (the id and revision); the variant and `mode` select a
runner (portable or kernels backend) with its weights; the task API (`<task>.load`) wraps the runner and turns the input into
tensors; calling it returns the typed result, whose `.backend` records what ran. The exceptions follow.

Each task has its own module: `fastforks.stereo`, `fastforks.flow`, `fastforks.depth`, `fastforks.classification`, `fastforks.segmentation`,
`fastforks.features`, `fastforks.pose`, `fastforks.detection`, `fastforks.promptable_segmentation`,
`fastforks.multiview`, `fastforks.matching`, `fastforks.matting`, `fastforks.tracking`, `fastforks.interpolation`,
`fastforks.retrieval`, `fastforks.calibration`, `fastforks.restoration`, `fastforks.volume` and `fastforks.edges`.
`multiview` models differ in what they need: PatchmatchNet also takes `depth_range=(near, far)`; MVSAnywhere and
PatchmatchNet return the reference view only. AnySplat also returns 3D Gaussians: `result.gaussians.save("scene.ply")`
writes a standard 3DGS PLY. See the [API pages](api/index.md) for every result's fields.

Every result has `.to(device)` and `.cpu()` (same type, so `save`, `top`, ... keep working; tensors already there are not copied), and `.numpy()` for a dict of arrays.

Load once, call many times: weights and (for `kernels`) kernels are loaded by `load`, so a call only runs the network.
The rest of this guide uses stereo in its examples; everything applies to every task.

## Calibrated stereo in one call

Raw (unrectified, possibly distorted) pairs with a calibration go in as they are; `rig=` takes a calibration file
(any format of [`read_calib`](stereo_geometry.md#calibration-files)) or a `StereoRig` you built:

```python
from fastforks import stereo

model = stereo.load("fast-acvnet-generalization")
result = model("left.png", "right.png", rig="calib.yaml")  # rectify -> match -> back to the left image
result.depth()  # [H, W] metres at the left image's pixels, NaN where no source pixel exists
result.valid()  # the mask of depth(): [H, W] at the left image's pixels
depth = result.depth().cpu().numpy()  # depth() / valid() are torch tensors on the model's device
result.save("scene.ply")  # point cloud in the left camera's frame, coloured by the left image
```

- `result.depth(euclidean=False)`, `.points(frame="camera")`, `.disparity`, `.confidence` and `.extras` belong to the
  main view; `euclidean=True` returns the distance along each pixel's ray instead of the depth `z`. `views="right"` or
  `"both"` also runs the right camera; `result.left` and `result.right` are the per-view results with the same
  methods (each in its own camera frame), and `result.view` names the main one (the left for `"both"`). The views run
  one after the other and share the decoded images, the rectification maps and the compiled plans. A view that was
  not run raises `InputError`. Batches (`[B, 3, H, W]`) give `[B, H, W]` maps.
- `fastforks.save(path, thing)` writes a result, a `PointCloud` or an `[H, W]` map; the suffix selects the format
  (`.tif`, `.npy`, `.pfm`, `.ply`, `.las` / `.laz` for clouds). `result.save(path, key=None)` and
  `result.right.save(...)` are the shortcuts for a result; `.png` (KITTI 16-bit disparity) only through a stereo
  result's `save`.
- `size=(height, width)` and `focal=` set the rectified image (default: the input size and the left camera's `fx`;
  the principal point is the image centre), the same as `g.rectify(rig)`.
- Holes (no source pixel, invalid disparity) are NaN everywhere.
- Two pixel grids: `depth()` and `valid()` are at the original (input) pixels; `disparity`, `confidence`, `extras` and
  `points()` are on the rectified grid (`size=`). Mask those with `valid(grid="rectified")` (or `cloud.valid`), not
  `valid()`. Without `rig` or with a `RectifiedRig` the grids are the same.
- `depth()` is cached; `numpy()` (and `run_many`'s numpy maps) are read-only arrays, so an edit cannot leave the cache
  stale: `.copy()` to edit. The shared `Rectification`'s matrices are read-only too.
- The maps are built once per (calibration, size, device) and cached, so a second call only runs the network; a plain
  `model(left, right)` is unchanged.
- `model.run_many(lefts, rights, rig=..., views=..., size=..., focal=...)` applies the calibration to every pair;
  with `save_to=` the right view is saved as `<name>_right<suffix>`.
- Directory inputs (`run_many`, `stereo run`) are not recursive (use a glob such as `'dir/**/*.jpg'`); an unreadable
  entry or broken symlink is a failed pair (`--skip-errors` reports it and the command exits 1).
- Several rigs: load the model once in Python and call `run_many` once per calibration.

Already rectified pairs (Middlebury, KITTI) work the same: `rig=read_calib("calib.txt")` (a `RectifiedRig`, see
[calibration files](stereo_geometry.md#calibration-files)), then `result.depth()`.

Principal points follow OpenCV: pixel centres at integer coordinates, image centre `((W - 1) / 2, (H - 1) / 2)`;
resizing uses `c' = (c + 0.5) s - 0.5` and COLMAP's `cx` is read as `cx - 0.5`.

Process-global state: CUDA and MPS calls of all models run one at a time per process (see [Threads](#threads)); `kernels` runs on
one CUDA device per process (a second device raises `BackendUnavailableError`: use one process per GPU); and `load`
temporarily sets `torch.get_default_dtype()` to float32 and patches the `torch.nn.init` initialisers while it builds
the network (restored afterwards, but visible to other threads running at that moment).

## Choosing a backend: `mode`

Four things differ and are reported separately: a model is *installed* when its package and card are present (`fastforks list`); its
dependencies are *available* when the optional libraries it needs (`timm`, `transformers`, `torchvision`, ... ; `runnable=True` in
`list_models`) are present, else `load` raises; a backend is *supported* when the card lists the mode and this device and
architecture fit it (kernels: see [platforms](#supported-platforms)); and a model is *validated* when golden parity against the
reference code is recorded (the "Parity" rows of the model pages).

| `mode` | runs on | what you get |
|---|---|---|
| `"portable"` | CPU, CUDA, Apple silicon ([MPS](#apple-silicon-mps)), other PyTorch devices where the network's ops exist | the original network in plain PyTorch, checked against the reference |
| `"kernels"` (CUDA) | NVIDIA GPU, compute capability >= 8.0, driver >= 570 | our CUDA kernels, faster than portable ([benchmarks](benchmarks.md) per model); memory is similar, sometimes higher. Raises `BackendUnavailableError` (its subclass `KernelUnavailableError` for a missing GPU or binary) with the reason if it cannot run here; never falls back silently |
| `"kernels"` (Metal) | Apple silicon, macOS, the families listed under [Apple silicon](#apple-silicon-mps) | Metal kernels, compiled on the device on first use; same error behaviour |
| `"auto"` (default) | anywhere | `kernels` if it can run and the recorded measurements prefer it for this model, else `portable` (a few plain ViT classifiers gain nothing from kernels); the choice is logged once per load (WARNING if kernels fails on a CUDA device, else INFO) and recorded in `result.backend` |

`mode` selects the backend; the result's `result.backend` (and `.mode`) reports which one ran.

`fastforks bench MODEL` times all of this on your machine (see below).

## Compute dtype: `dtype`

- `None` (default): the model's tested precision (`fastforks info MODEL`): float16 / bfloat16 on CUDA where the model supports it,
  float32 on the CPU and on Apple MPS, always reduced precision for `kernels`.
- `"float32"`: float32 everywhere, portable only (`kernels` raises `OptionsError`). This is the reference-accurate path:
  the closest to the original implementation, slower and with more memory. CUDA convolutions then use PyTorch's
  default TF32; `torch.backends.cudnn.allow_tf32 = False` makes them full float32 and the output CPU-identical
  (slower: HITNet 720x1280 70 -> 230 ms).
- `"float16"` / `"bfloat16"`: need CUDA and a dtype the model code implements (portable) or the kernels' own dtype
  (`kernels`; `mode="auto"` then falls back to portable), or MPS where the card lists float16 (see
  [below](#apple-silicon-mps)); anything else raises `OptionsError` naming the dtypes available.
  A dtype other than the tested one is logged and not covered by the reference checks.

Precision of `kernels` (CUDA): the card's `kernels` dtype (float16; bfloat16 for one family; see the
[table](OTHER_GPUS.md#what-each-family-runs-on)); some heads and box / mask steps stay float32 (noted on the card).
Families that list `cudnn_hacc` also accumulate convolutions in float16: faster on GeForce, a larger error than portable
float16. Each family's GPU golden tolerances ("Parity" on the model pages) bound the error. On MPS the Metal `kernels`
run float16 for stereo and float32 for MoGe-3 and VGGT.

Every model is checked against reference outputs: saved results of the original code at a pinned commit, which the
portable (CPU, float32) and CUDA backends must match. `fastforks info MODEL` shows the pinned commit.
`"<id>@N"` is an assertion against revision drift: it checks that the installed package has revision N of the model and
raises `ModelNotFoundError` otherwise (e.g. after an upgrade that changed the model's outputs); it does not fetch older code or weights (weights are pinned per revision by URL and SHA-256). To reproduce a run, pin the package and the
revision together: `pip install fastforks==<version>` and `stereo.load("las2-s@1")`; `fastforks info MODEL` shows the revision,
weight URLs and SHA-256.

Pin a deployment: (1) `pip install fastforks==<version>`; (2) load `model@rev`; (3) record `fastforks list --format json`
(`weights_sha256` per model) as the lock record; (4) `fastforks download MODEL --check` verifies the files offline;
(5) set `FASTFORKS_OFFLINE=1` in production.

## Options

`fastforks info MODEL` lists a model's load options with their validated defaults, e.g. the number of refinement
iterations:

```python
from fastforks import stereo

model = stereo.load("raft-stereo-middlebury", options={"iters": 16})
result = model("left.png", "right.png")
result.options  # {"iters": 16}: non-default options are recorded in every result
```

One name per concept across families; the old (upstream) names are accepted as aliases (giving both is an
`OptionsError`):

| Concept | Option | Old names |
|---|---|---|
| Longer side of the network input, px | `long_side` | `process_res` (Depth Anything 3), `processing_res` (Marigold), `resolution` (MapAnything) |
| Shorter side (lower bound) of the network input, px | `short_side` | `input_size` (Depth Anything V2) |
| Side of a square with the network input's area, px | `area_side` | `resolution` (VGGT-Omega) |
| Resolution level 0 to 9 (not pixels) | `resolution_level` | |
| Recurrent refinement iterations | `iters` | |
| Diffusion denoising steps | `steps` | |

Options are checked at load (`OptionsError`) against the bounds and choices on the card, the same ones
`fastforks info MODEL` and the model pages list (e.g. MoGe `resolution_level` 0 to 9). Options marked "speed only"
(`cuda_graphs`) give the same results up to floating-point rounding, "precision only" (`fp16_accumulate`,
`fp32_accumulate`) trade accuracy against speed; all others change the outputs.

What a model takes and returns, without loading weights: `depth.model_card("moge-2-vitl-normal").contract()` lists the
inputs (required, optional or unused: e.g. `intrinsics`, `depth_range`, prompts), the outputs with units and pixel /
camera conventions, the options with their bounds, the dtypes per mode and device (implemented and validated), and what
an export contains (stages inside and outside the graph, dynamic sizes); `.to_dict()` gives plain JSON-ready data.
Intrinsics of every task use OpenCV pixel centres (see [Pixel coordinates](#pixel-coordinates)); `contract().camera`
says so.

## Licences and commercial use

fastforks separates three things; none of this is legal advice.

- **Code** is installed per licence class: `pip install fastforks-core` has the models with permissive code,
  `fastforks-core[nc]` (and `fastforks`) adds non-commercial and copyleft (GPL) code. Our own code and kernels are Apache-2.0. Permissive code
  alone does not allow commercial use: some of these models have non-commercial weights (see below).
- **Weights are never shipped.** They are downloaded from the authors' hosts (for a few models a mirror of the same
  files) at pinned revisions and checked against a
  SHA-256, so the weights licence belongs to the authors, not to fastforks. Weights with a non-commercial licence
  are only downloaded with `FASTFORKS_NONCOMMERCIAL=1`, in every install (`LicenceError` otherwise); local copies
  (`weights=PATH`, `FASTFORKS_WEIGHTS_PATH`, the cache) always load.
- **Every variant has an effective commercial status** that needs the code AND the weights licence to allow
  commercial use: `yes`, `no` (the code or weights licence forbids it) or `unclear` (a custom licence, treat as `no`).
  The training data does not change the status, it is listed next to it: datasets with research-only, unclear or
  unstated terms are a caveat you should read (whether weights inherit them is legally unsettled). Weights without a
  licence of their own are assumed to carry the code licence. The model pages list the status and
  add a licence note where the reason is not obvious (inherited non-commercial terms, unclear dataset terms).

```python
# needs: pip install "fastforks-core[nc]"
import fastforks
from fastforks import stereo

fastforks.list_models(commercial="yes")  # all tasks
fastforks.stereo.list_models(commercial="unclear")  # one task
stereo.model_card("foundationstereo-vit-s").licence_notes
print(stereo.model_card("hitnet-eth3d"))  # readable summary: licences and commercial status per variant
print(fastforks.describe_models(stereo.list_models("hitnet*")))  # table: licences, commercial, what blocks a "yes"
```

The flag is decided by the code and weights licences only. GPL code counts as `yes` (with copyleft obligations: see `fastforks info MODEL`); training data is a note, not part of the flag.

Most variants are `yes`; the others are `no` or `unclear` (`fastforks list --commercial no` shows them). A `yes` can still depend on training data
with research-only or unstated terms, or on weights whose licence is only assumed from the code: check
`fastforks info MODEL` (licence notes) before shipping.

## Already have the weights?

```python
from fastforks import stereo

model = stereo.load("igev-stereo-sceneflow", weights="runs/igev_finetuned.pth")  # your own weights, e.g. a fine-tune
```

- `weights=PATH`: the file, or for a variant with several files (e.g. Marigold's diffusers layout) a directory that
  holds them as `<dir>/<name>`; for weights shipped inside an archive, the extracted file. It must have the same file
  format, keys and shapes as the original. It is not hash-checked, so the reference checks of the model id no longer
  apply (one WARNING log line says so; `model.weights` holds the files and the model's `repr` shows them), but it
  is loaded strictly: missing, unexpected or reshaped keys raise `CheckpointMismatchError` (a `WeightsError`) listing
  them, and a missing or unreadable file (another format, empty, truncated) a `WeightsError` naming it. Floating-point
  tensors in another precision than the original (e.g. a float32 fine-tune of Depth Pro's float16 checkpoint) are
  cast to the model's.
- `labels=` (classification): `classification.load("resnet50", weights="ft.pth", labels=my_1000_names)` names
  the classes of a fine-tuned head (a list of non-empty strings, in logit order, one per logit: 1000 for `resnet50`, else
  `OptionsError`, checked before loading; the head size is the family's, fixed).
  Only the names change (`result.class_names`, `top()`); the input handling and what the logits mean stay the
  family's. Fine-tunes with another number of classes do not load (the head shape is checked): build that model in
  timm / torchvision with your head and speed it up with [`fastforks.patch`](#patching-your-own-timm--torchvision--transformers-model).
- `FASTFORKS_WEIGHTS_PATH`: directories (`os.pathsep`-separated) holding the original files as
  `<dir>/<family>/<file>` or `<dir>/<file>` (e.g. a shared network drive); they are used in place and hash-checked,
  a copy that does not match is skipped.
- Hugging Face hub cache: weights hosted on Hugging Face (`huggingface.co/<repo>/resolve/<rev>/<file>`) are taken
  from `HF_HUB_CACHE` (else the older `HUGGINGFACE_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`) if a snapshot there matches the recorded
  SHA-256; the torch hub cache (`TORCH_HOME/hub/checkpoints`, else `~/.cache/torch/hub/checkpoints`) likewise by
  file name. Both are only read; a copy that does not match, or a cache directory that cannot be read, is skipped and
  the file is downloaded.
- `fastforks download MODEL` (`--kernels` for the kernel binaries) fetches everything into the user cache ahead of
  time, e.g. before going offline. A model that fails (e.g. gated without a token) does not stop the others: it is
  listed as `MISSING ...` at the end and the command exits 1. `--arch sm_89` fetches the binaries of a GPU you do not have, `--check` verifies the
  cache without network (a file whose size and mtime match its recorded stamp is not re-hashed), `--dry-run` lists
  the download size (the archive, once) and licence classes. `FASTFORKS_OFFLINE=1` (or `HF_HUB_OFFLINE=1`)
  forbids every download; a missing file then raises an error naming it and the command that fetches it. For
  containers see [Docker images](docker.md).
  `fastforks download --kernels-only --all-archs` fetches every kernel binary (~240 MiB; one GPU target ~34 MiB)
  without weights; the files land in `FASTFORKS_CACHE` (default: the platform user cache, `~/.cache/fastforks` on
  Linux). The cache is content-addressed (`blobs/sha256/<ab>/<sha256>`), so a copy of it is not a
  `FASTFORKS_KERNEL_PATH`: that wants the manifest's file names, `<dir>/<lib>/<lib>-<hash>-<arch>.fatbin` (or `.so`;
  `<dir>/<name>` also works), e.g. `kernels/fs/fs-f6f8de2cca1e-sm_80.fatbin` (names: `fastforks/kernels/<lib>/manifest.toml`).
  `fastforks download --kernels --arch sm_80` prints `<lib>-<hash>:<arch>: <cache path>` per file; copy each path to that
  name. Files are hash-checked on use. To move everything instead, copy the whole cache and set `FASTFORKS_CACHE` to
  the copy (with `FASTFORKS_OFFLINE=1` it may be read-only). `--check` verifies offline.

## Inputs and outputs

- Images: file paths, PIL images, numpy arrays (`HWC` / `BHWC`, RGB; convert OpenCV's BGR first) and torch tensors
  (`CHW` / `BCHW`). Integer images are scaled by their dtype's maximum; float images must be in `[0, 1]`.
- Paths are a convenience for standard images (PNG, JPEG, BMP, WebP, 8/16-bit grayscale TIFF / PNG), read with Pillow.
  Files Pillow would read lossily (16-bit colour PNG / TIFF, float or 32-bit integer images) raise `InputError`. For
  anything else (scientific TIFF, EXR, raw, `.npy`, OpenCV), load it yourself and pass a numpy array (`HWC`, RGB;
  e.g. `cv2.imread(path, cv2.IMREAD_UNCHANGED)` then BGR to RGB, uint16 or float in `[0, 1]`) or a torch tensor
  (`CHW`).
- Files are read as stored: an EXIF orientation tag (phone photos) is not applied, so the pixels (and SAM prompt
  coordinates) are the stored, possibly rotated ones; one WARNING per process names the first such file. For an upright
  photo pass `ImageOps.exif_transpose(Image.open(path))`. CMYK / YCbCr JPEGs: pass `Image.open(path).convert("RGB")`.
- Classifiers resize the shorter side (to `resize`, e.g. 256) and centre-crop (`crop`, e.g. 224), so content near the
  border is cut off; probabilities are a softmax over the model's fixed label set (not calibrated, they sum to 1 even
  for an image of no class). Steps, as `data_config` lists them: float image in `[0, 1]` resized with torch
  (antialiased, the card's interpolation), centre crop, mean / std normalisation. The original PIL uint8 resize (and so
  `timm.data.create_transform(**model.data_config)` on a PIL image) gives different pixels: logits can move by
  ~0.5 and close top-5 ranks can swap. Compare against timm with the same float tensor input, not a PIL image.
- MoGe `intrinsics=`: used only as the focal length `fx`; `fy` within 1 % of `fx` and a principal point within 5 % of the
  centre, else `InputError`. Use the result's `intrinsics` (the camera the points really use) afterwards.
- Segmentation labels: class 0 is a class like any other (ADE20K `wall`), there is no ignore / background mask.
- Grayscale (`HW`, `1HW`, mode `L`) and RGBA images are accepted and converted to RGB (grayscale is repeated to three
  channels, alpha is dropped).
- Stereo pairs are not checked for swapped left / right images: the result is finite but wrong (disparity is
  `x_left - x_right`, so pass the left camera's image first).
- Prompt points outside the image are accepted as given (as upstream SAM does); boxes `(x0, y0, x1, y1)` with
  `x1 < x0` or `y1 < y0` raise `InputError`.
- Embedding search (`features`): `pooler_output` is not unit-length and every image must have the same size, for the
  gallery and the queries alike (the size changes the embedding):
  ```python
  import torch.nn.functional as F

  g = F.normalize(model(gallery_images).pooler_output, dim=-1)  # [N, D]; same-size images, one batched call
  q = F.normalize(model(query_images).pooler_output, dim=-1)  # same size as the gallery
  scores = q @ g.T  # cosine similarity [Q, N]; scores.topk(5, dim=-1) for the best matches
  ```
- Lists are not batches: stack same-size images into one array or tensor (multi-view models take a list of views).
- Any size above the model's minimum: resizing or padding to what the network needs and back happen inside the call, so
  dense maps (disparity, depth, labels) have the input's resolution. Classifiers resize and crop as the model expects.
- Batches: pass `[B, 3, H, W]` tensors (or `BHWC` arrays) to get `[B, H, W]` maps (`[B, classes]` logits).
- Exceptions to batching: detection and pose return a `list` of results (one per image; pose takes a list of box
  arrays, one per image); promptable segmentation
  calls take one image (batch the image encoder with `get_image_embeddings`, then prompt per image); multi-view models
  take the `N` views of one scene as the leading dimension (not independent images). Stereo, depth, classification,
  segmentation and features return tensors with a leading `B`.
- Outputs are tensors on the model's device; `result.numpy()` returns them as a dict of NumPy arrays by name (`None`
  fields left out, `extras` flattened in, bfloat16 as float32). Some models also return
  `result.confidence` or `result.extras` (see the model page).
- Results are snapshots you own: frozen dataclasses with new tensors per call (never buffers a later call reuses, also
  with CUDA graphs). `.to(device)` / `.cpu()` return new results. `numpy()` arrays of CPU tensors share their memory;
  arrays of cached values (calibrated stereo `depth()`, `run_many`'s numpy maps, rig matrices) are read-only:
  `.copy()` to edit.

### Pixel coordinates

Intrinsics, cameras and rigs (depth, multi-view, stereo) use OpenCV pixel centres and axes: the top-left pixel's centre
is `(0, 0)`, the image centre `((W - 1) / 2, (H - 1) / 2)`, x right, y down, z forward, `cam_to_world` maps camera to
world. Models whose original code uses another grid are converted inside. From a corner-origin matrix (COLMAP, MoGe,
nerfstudio): `cx_colmap = cx_opencv + 0.5` (same for `cy`); `read_calib(..., "colmap")` does it for you.

### Saving maps

`save` exists on results that hold dense maps: stereo, depth, multi-view, segmentation (`labels`) and promptable
segmentation (`masks`); classification, features, pose and detection results have none (use `result.numpy()` or the
fields). `result.save(path)` writes the primary map (`disparity`, `depth` or `inverse_depth`, `labels`, `masks`;
`key="confidence"` or an `extras` name picks another) as one `[H, W]` file; the suffix selects the format, and an
unknown suffix raises with the list this result accepts. Batches are not split (pass one image per call; leading
dimensions of size 1 are dropped; for several masks write one, `fastforks.save("m.png", r.masks[i])`); a multi-view
result with several views takes `view=` (the input view index). `.npy` and `.png` keep integer and bool maps exact
(their dtype); `fastforks.save` of a result without `save` raises and names the arrays to `np.save`. A calibrated
stereo result's `.tif` / `.npy` / `.pfm` hold the depth (`key="disparity"` for the disparity) and its `.png` the KITTI
disparity. Maps are written as they are: invalid pixels keep their finite values, so mask them first
(`np.where(result.valid(), depth, np.nan)`; depth results: their `mask`). A `.tif` carries one ImageDescription tag (model, map, kind / unit).
`torch.save(result, path)` works; `torch.load(path)` in a fresh process needs the result's class imported first (e.g.
`from fastforks.classification import ClassificationResult`; importing `fastforks.<task>` is enough), and an
`ImageEmbeddings` also `torch.serialization.safe_globals([ImageEmbeddings])` (or `weights_only=False` for your own
files). Result tensors are inference tensors: `.clone()` one before an in-place edit.

| suffix | content |
|---|---|
| `.tif` / `.tiff` | float32 TIFF, deflate + floating-point predictor; opens in tifffile, OpenCV, Pillow, GDAL |
| `.npy` | raw float32; bool and integer maps (labels, masks) keep their dtype (`bool`, `int64`, ...) |
| `.pfm` | raw float32, bottom row first (Middlebury) |
| `.png` | integer and bool maps (labels, masks): 1-bit for bool, 8-bit for 0..255, 16-bit for 0..65535; float maps are refused, except a stereo disparity or a depth result (`depth`, `inverse_depth`): uint16, value x 256, 0 = invalid, clipped at 255.996 (KITTI convention; metric depth keeps 1/256 m) (run_many / `stereo run` with `rig=` write depth: not `.png`) |
| `.ply`, `.las` / `.laz` | point clouds: stereo with `rig=`, multi-view (all views merged; without `world_points`: depth unprojected with `intrinsics` and `cam_to_world`, header units: those of the given `cam_to_world`), depth results with `points` or metric `depth` + `intrinsics` (the finite points inside `mask`), coloured; see below |

Point clouds (`.ply`, `.las` / `.laz`) are metric in OpenCV axes (x right, y down, z forward) unless `axes=` says
otherwise:

- `axes=` (stereo, depth, multi-view results and `fastforks.save`): `"opencv"` (default, RDF), `"opengl"` / `"blender"`
  (RUB), `"ros"` (FLU), `"pytorch3d"` (LUF) or any 3-letter code with one of R/L, U/D, F/B (the directions of x, y, z),
  e.g. `"RFU"`; anything else raises `OptionsError` listing the presets. The PLY header comments (and the LAS
  VLR, system id `fastforks`) name the version, units, axes, frame and the pixel convention. A LAS / LAZ is a local
  frame without CRS.
- `max_depth=` (metres, z in OpenCV axes) drops points beyond it; the default keeps every finite point. There is no
  other masking: stereo points are the finite ones, depth and multi-view clouds keep the model's `mask`.
- Colours: stereo uses the rectified image; depth and multi-view results colour from the input image they were called
  with (the result keeps a reference, no copy). A metric depth result without `points` but with `intrinsics` (e.g.
  `da3metric`) writes its unprojected depth; without intrinsics call the model with `intrinsics=K`.
- A multi-view result saves the merged, masked cloud of all views in the world frame; `view=` writes one view;
  `save_scene` writes cameras, images and points as a COLMAP or nerfstudio scene (below).
- Unposed multi-view models (no `cam_to_world=` given) use the frame of view 0 as the world; `cam_to_world` is OpenCV
  camera-to-world. For MapAnything `world_points` is the primary output; `depth` + `intrinsics` + `cam_to_world` is its
  pinhole approximation (intrinsics are fitted to the predicted rays), so they differ slightly; its `metric_scale` is in
  `extras`.

To COLMAP / nerfstudio: `r.save_scene(folder, "colmap")` writes the views' input images losslessly
(`images/frame_00001.png`, numbered by input view; `image_names=` references your own files, relative to `folder`)
and a text model in `sparse/0` (PINHOLE, pixel centres at +0.5, world-to-camera poses, points without tracks);
`"nerfstudio"` writes `transforms.json` (camera-to-world in OpenGL axes, `cx` / `cy` +0.5) and `sparse_pc.ply`. The points
are the masked points of all views, at most `max_points=100_000` (fixed seed); both formats can share one folder.

```bash
python examples/simple_trainer.py default --data_dir scene --data_factor 1  # gsplat
ns-train splatfacto --data scene  # nerfstudio: reads transforms.json and sparse_pc.ply
```

Colored PNG for viewing (numpy only; `matplotlib.cm` works too if installed):

```python
import numpy as np
from PIL import Image

d = result.disparity.cpu().numpy()  # or result.depth() with rig=, inverse_depth for depth
ok = np.isfinite(d)
t = np.zeros_like(d)
t[ok] = (d[ok] - d[ok].min()) / max(np.ptp(d[ok]), 1e-9)
stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)  # viridis
rgb = np.stack([np.interp(t, np.linspace(0, 1, 5), stops[:, c]) for c in range(3)], -1).astype(np.uint8)
rgb[~ok] = 0
Image.fromarray(rgb).save("depth_color.png")
# label map (segmentation): result.save("labels.png") (8-bit up to 255 classes, else 16-bit) or result.save("labels.npy")
```

A TIFF is about 1.9x smaller than raw float32 and bit-exact. `bits=10` first rounds the mantissa to 10 bits (relative
error at most `2**-11`, 0.05 % for normal floats; values that would overflow or flush to 0 stay exact) and gives 7-10x; any `bits` in 1..23 works. Writing takes about 5-15 ms for a 1.4 MP
map (strips are compressed in parallel with `isal`; without it the standard `zlib` writes the same file more slowly).

```python
result.save("disparity.tif")  # lossless
result.save("disparity.tif", bits=10)  # 7-10x smaller than raw, error <= 0.05 %
```

## Video and repeated sizes

Keep the model and the frame size fixed. The first call at a new size is slower (memory planning, cuDNN plans), so
warm up at each size before timing: 4 calls for kernels models that replay CUDA graphs (Fast-ACVNet, CoEx, LAS2,
RT-MonSter++), 1-2 for the others. Stereo models also have `model.run_many(lefts, rights)` for a stream of pairs; it returns a pipeline that starts working at once, a bounded `prefetch` pairs ahead (with `save_to=`, files may appear before the first iteration). Iterate it (`for r in ...`, or `list(...)` with `save_to=`) to drain it; to stop early use `with model.run_many(...) as p: for r in p: ...` or `p.close()` (it finishes the writes in progress and drops the other queued pairs); a plain `break` does not close a pipeline that is still referenced. Outputs for the same input are
bitwise equal between calls and processes, except portable CoEx and Fast-ACVNet on CUDA (differences of 1-3 px on
single pixels; set `torch.backends.cudnn.deterministic = True` to remove them, at about 5 % speed). Batched calls can
differ from single calls in the last bits.

## Several GPUs

`portable` runs on any device: `stereo.load(..., device="cuda:1")`. `kernels` uses one CUDA device per process (the
kernels are bound to one context); a second device in the same process raises `BackendUnavailableError`. Use one
process per GPU.

## Threads

Deployment: CUDA and MPS calls take one process-wide lock and `kernels` binds one CUDA device per process, so run models in parallel
(or one per GPU) as separate processes.

Calls from several threads give the same outputs as single-threaded calls. Load every model before other threads
start calling models.

| case | behaviour |
|---|---|
| one model, several threads | calls run one at a time (a model is not reentrant) |
| CUDA calls of different models | one at a time per process (one process-wide lock), so a pipeline of models in threads does not overlap on the GPU: use one process per model |
| MPS calls of different models | one at a time per process (the same lock: PyTorch's MPS backend is not thread-safe) |
| CPU calls of different models | run in parallel |
| a model called inside another model's call | allowed (the lock is reentrant) |
| CUDA streams | calls run on the caller's current stream (`with torch.cuda.stream(s):`), also `TensorRTEngine`; synchronise that stream before reading outputs on another one |
| devices | `kernels`: one CUDA device per process (see above) |

## Running on the CPU

`device="cpu"` runs the portable backend in float32. On a Ryzen 7 5800X (8 threads) fastforks is 1.0-2.2x faster
than the original code. Set `torch.set_num_threads(n)` before the first call to the cores you really have (in a
container, `os.cpu_count()` shows the host's). Going from 1 to 8 threads gives about 4x on large networks and 2x on
small ones. Kernels need CUDA. Tested on x86-64 only; on a Mac, use the GPU (next section).

## Apple silicon (MPS)

On a Mac without CUDA, `load` puts a model on the Apple GPU (`mps`) when its family is validated there, else on the
CPU; `result.backend.device` says which. Every family is validated except those with gated weights (DINOv3,
DINOv3-ConvNeXt, SAM 3, VGGT-Omega). `device="mps"` also runs those, unvalidated: an op MPS lacks raises. On MPS
`kernels` exists for RAFT-Stereo, the IGEV family (IGEV-Stereo, Selective-IGEV, GREAT-IGEV, IGEV++, MoCha-Stereo,
MonSter, MonSter++), CFNet, LiteAnyStereo V2 (LaS2), HITNet, AANet, CREStereo, NMRF, BridgeDepth, DEFOM-Stereo and
FoundationStereo: Metal kernels for each family's hot spot (ConvGRUs, correlation and
geometry lookups, cost volumes, HITNet's warp costs, AANet's deformable sampling), float16, and for MoGe-3 (its
sparse convolutions) and VGGT (the aggregator's q / k LayerNorm and RoPE) in float32. Flow, pose and features have
them too: RAFT and SEA-RAFT (the correlation lookup without the all-pairs volume, float16), RTMPose (the person crops,
float32), ViTPose / ViTPose++ (patch embedding, decoder and keypoint decoding, float16) and Perception Encoder (the
blocks' RoPE, float16). They compile on the device in under a second on first use;
`mode="auto"` picks it. Every other family runs `portable`.

- dtype: float32 by default. `dtype="float16"` works where the card lists it (`fastforks info MODEL`, row
  `apple gpu`). It saves memory (up to half) and on an M1 Pro is also faster: median 1.15x over 60 models, up to
  2.9x on memory-bound stereo cost volumes (GREAT-IGEV, CoEx, CAS-GwcNet), rarely slower (HITNet, Mask2Former 0.94x).
- A few small steps run on the CPU or in float32 by design: float64 maths in MoGe and SAM, one adaptive pooling in
  S2M2, Depth Anything 3's bicubic resize.
- The first call at a new input size takes 0.5-4 s longer (PyTorch compiles the Metal kernels for that shape).
- MPS shares the host RAM (no separate GPU memory); the MPS memory in the benchmarks is the process's peak footprint
  (Activity Monitor). A model larger than the free RAM swaps (slower, but it runs); keep several GB free on
  the system disk, since with a full disk swapping can freeze the machine.

Time and memory per model on an M1 Pro: [Apple M1 Pro](devices/apple-m1-pro-16gb.md).

## torch.compile, torch.export, ONNX and TensorRT

Every model has `model.module()`, the traceable part of the portable model (what it covers per task: [API](api/index.md#what-modulemodule-covers); stereo and depth include scaling, padding, network and cropping) as an `nn.Module`
(inputs and outputs as the task's runner takes and returns them; see the API pages): for stereo RGB `[B, 3, H, W]`
float32 in `[0, 1]` for both views in, disparity `[B, H, W]` float32 out; depth one image in, its main map out. Call
it under `torch.inference_mode()`. ONNX and TensorRT export and `TensorRTEngine` work the same way for every
task.

```python
import torch

from fastforks import stereo

model = stereo.load("igev-stereo-sceneflow", mode="portable")
module = model.module()
compiled = torch.compile(module, fullgraph=True)
program = torch.export.export(module, model.example_inputs(384, 1248))
```

A `kernels` model called inside your own `torch.compile`d function is a graph break: the kernels run eagerly, the
code around them is compiled. TorchScript (`torch.jit.trace` / `script`) is not supported (deprecated upstream): use
`torch.export` of the portable `module()`; kernels-patched models (`load(mode="kernels")`, `patch`) do not export.

Or keep the high-level API and compile in place (`kernels` raises `BackendUnavailableError`), or export:

```python
model.compile()  # every task call now runs compiled; again with other arguments replaces it
model.export_onnx("m.onnx", 384, 1248)
model.export_tensorrt("m.engine", 384, 1248)  # on the target GPU; then stereo.TensorRTEngine("m.engine")(left, right)
```

```
# needs: pip install "fastforks-core[nc]" "fastforks-core[onnx]"
# S2M2's code and weights are non-commercial: its weights download only with FASTFORKS_NONCOMMERCIAL=1
fastforks export s2m2-s --format onnx --size 384x1248          # float32 ONNX of the portable backend
fastforks export s2m2-s --format trt --fp16 --size 704x1280    # stereo / depth / segmentation / detection need --size; engine + check
fastforks export depth-anything-v2-s --format trt --fp16 --size 518x770
```

From Python: `model.export_onnx(path[, height, width])` and `model.export_tensorrt(path[, height, width])`; run an
engine with `stereo.TensorRTEngine(path)(left, right)` (`depth.TensorRTEngine(path)(image)`). Exports have one fixed
size and batch by default (size ranges: see [export](export.md)); the size must lie in `model.size_range()` (stereo, depth, segmentation and detection have no default size;
classification and features default to `model.export_size()`). **A TensorRT engine only runs on the GPU model and
TensorRT version it was built with: build it on the target machine.** Install the exporters with
`pip install "fastforks-core[onnx]"` / `"fastforks-core[tensorrt]"`. See [export](export.md) for inputs and outputs per task,
what works per model and how TensorRT compares with `kernels`.

## Patching your own timm / torchvision / transformers model

`fastforks.patch(model, mode="auto")` speeds up a classifier you built yourself (weights loaded) in place and returns a
`FastModel` (fp32 NCHW in and out; `.backend`, `.report`, `.original`). `base_type(module)` returns the class a patched
module had before the override. Both are for models outside the fastforks registry; the registry models use `load`.

Rewritten: timm / torchvision CNNs (ResNet, RegNet, EfficientNet, MobileNetV2-V4, DenseNet, VGG, ShuffleNetV2, ...),
ConvNeXt, Swin, timm ViT `Block` (packed qkv), `EvaBlock` and HF `SiglipEncoderLayer` (the full list:
`fastforks._optimize.COVERED`). Anything else (e.g. HF `ViTModel`, `Dinov2Model`) only gets BatchNorm folding, the
dtype cast and channels-last: `patch` logs a WARNING saying so, and `mode="kernels"` raises.
A patched model's device and dtype are fixed by `patch` (`.half()` / `.to()` raise, by design): patch a
`copy.deepcopy` of the original again for another device or dtype. Modules with forward or backward hooks are not
rewritten (their hooks keep running; `mode="kernels"` refuses such a model). Activations run channels-last (CUDA, and
CNNs on the CPU): in a forward that torch.fx cannot trace, a `.view()` of a conv output can raise; use `.reshape()` or
`.flatten()`.

## Training and gradients

fastforks is for inference. Task calls (`model(x)`) run without autograd. For gradients use the portable backend:
`load(..., mode="portable").module()` and a `patch(model, mode="portable")` model are plain PyTorch (eval mode,
BatchNorm folded) and backpropagate to the input and the weights. `kernels` have no backward: a `FastModel` in
`kernels` mode raises for an input that requires grad, and `module()` raises `BackendUnavailableError`. Train the
original model; `patch` modifies it in place, so patch a copy. A fine-tuned `module()` state dict loads back with
`load(..., weights=)` (the `module()` key prefix is accepted), except classification, whose `module()` has folded
BatchNorm: fine-tune the timm / torchvision model and pass its `state_dict` file as `weights=`.

## Serving (web service, batch worker)

- One model per process and GPU: calls hold a process-wide lock, so threads add no throughput; scale with
  processes (load the model inside each worker, see Processes below) or one process per GPU.
- Warm up before the first request, at the sizes you serve: one call with a real image or
  `torch.rand(1, 3, height, width)` (images are in [0, 1]; `example_inputs` are normalised, for export only); the first
  call at a new size runs eagerly (and with `kernels` captures graphs).
- Cap the input size yourself (e.g. resize to a few buckets): one 48 MP upload keeps the reserved GPU memory high
  (allocator cache); `torch.cuda.empty_cache()` after such a request gives it back.
- Map errors to HTTP: `InputError` / `ArgumentError` (bad image, bad keyword) to 4xx, other `FastforksError`
  subclasses (`WeightsError`, `BackendUnavailableError`, ...) to 5xx; one `except fastforks.FastforksError` covers all.
- Images as uploaded bytes: pass `bytes`, `bytearray` or `io.BytesIO` of the encoded file; corrupt data raises `InputError`.
- Image inputs also take an `http://` or `https://` URL (`model("https://example.com/cat.jpg")`; downloaded, 30 s timeout, 100 MB cap).

## Notes for deployments

Per-frame loop (mono depth, detection, classification): load once, warm up at the frame size, then call per frame.
There is no `run_many` and no folder command outside stereo; for a folder of images load once and loop:

```python
import pathlib
from fastforks import depth

model = depth.load("depth-anything-v2-s")
for p in sorted(pathlib.Path("images").glob("*.jpg")):
    model(p).save(f"out/{p.stem}.tif")
```

With a small model the JPEG decode (about 10 ms for 2 MP on one core) costs more than the network. `read_images`
decodes the next files in background threads while the model runs; same pixels, same results:

```python
import fastforks

paths = sorted(pathlib.Path("images").glob("*.jpg"))
for p, image in zip(paths, fastforks.read_images(paths)):
    model(image).save(f"out/{p.stem}.tif")
```

A pair or view list given as files in one call (matching, stereo, flow, multiview) is decoded in parallel.

```python
import time
from fastforks import detection

det = detection.load("rtdetrv2-s-coco", max_detections=30)  # limit set when loading
det(first_frame, score_threshold=0.5)  # warm-up at the real size; the first call also captures CUDA graphs (kernels)
for frame in frames:  # numpy HWC uint8, torch or PIL; keep the size fixed, call from one thread
    boxes = det(frame, score_threshold=0.5)  # score threshold can be overridden per call
```

Memory per input size: `kernels` keep a cuDNN plan and a CUDA graph per input size, so a stream of many different
sizes grows the reserved memory (bounded: 512 cuDNN plans, a per-model cap on graphs that evicts the stalest; the first
calls at a new size run eagerly). Fixed sizes, or resizing to a few buckets, avoids it; `del model; gc.collect();
torch.cuda.empty_cache()` releases a model's share. Host RSS of a GPU process is mostly the CUDA / cuDNN libraries
(about 1.1 GB for plain PyTorch with one conv and one GEMM, 2.3 GB for a detection model).

CPU-only install: take PyTorch's CPU wheel first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `pip install fastforks-core`: the lean
CPU install is about 0.9 GB on disk, mostly torch.
`fastforks env` shows the thread count and whether torch is a CPU build.

Processes: a model cannot be pickled (it holds device memory and locks), so with `multiprocessing` / a `Pool` load the
model inside each worker (`initializer=`), or start workers with `spawn`. Loading it before a `fork` and calling it in
the child can deadlock in the CPU thread pool (libgomp); fastforks logs a warning once when a forked child calls such a
model. Results pickle, and `torch.save(result, path)` / `torch.load(path)` work with the default
`weights_only=True`.

Provenance: `result.backend` also carries `library` (the fastforks version) and `weights` (SHA-256 of the recorded weight
files; with `load(weights=PATH)` see `backend.checkpoint`); the kernel libraries and builds are in `backend.kernels`.

Audits: `fastforks list --format json` (or `csv`) prints every model with its code and weights licence, commercial
status and reasons, data notes, revision, weights URLs and SHA-256. `FASTFORKS_COMMERCIAL_ONLY=1` makes `load` raise a
`LicenceError` for every model whose status is not `yes` (also with `weights=PATH` and for cached weights) and
`list` / `list_models` show only the `yes` models. `commercial=yes` includes copyleft licences (GPL): check the licence
column of `list` / `info`. Weight URLs on Hugging Face are pinned to a commit; the weights of
other hosts (Google Drive, Dropbox, GitHub releases) are pinned by SHA-256 only.

Kernel trust model: the kernel binaries come from our Hugging Face repo; their SHA-256 are in the manifest inside the
wheel (which pip installs from PyPI) and every download is checked against it before it is loaded. There are no
signatures or SBOM yet, so the wheel's integrity is the root of trust (pin it with `pip --require-hashes`).

## Command line helpers

`fastforks env` prints versions, GPU and kernel binaries (paste it into bug reports); `fastforks <command> --help`
lists every flag.

## Speed check

```
fastforks bench igev-stereo-sceneflow                            # portable and kernels side by side, median of 10 runs
fastforks bench igev-stereo-sceneflow --mode portable --dtype float32 --size 544x960
fastforks bench depth-anything-v2-s --device cpu --size 518x518
```

`bench` works for any model, with synthetic input of the right kind (an image; a pair for stereo). It prints time
and peak memory in MiB. On CUDA: "allocated" is what tensors use (weights included), "reserved" is what PyTorch's allocator holds
from the driver (roughly what `nvidia-smi` shows for the process, without the CUDA context). On MPS it prints the Metal driver's
allocation (GPU buffers only); on the CPU no memory.

## Tiers

`fastforks list` shows a tier per model: `stable` (tested and measured), `experimental` (works, less tested) or
`deprecated` (kept for old code, will be removed). `--tier` filters by it.

## Supported platforms

| | portable | kernels (CUDA) |
|---|---|---|
| OS | Linux, macOS (Apple silicon, [MPS](#apple-silicon-mps)), any PyTorch platform | Linux x86-64, glibc >= 2.34, driver >= 570 |
| GPU | CPU, CUDA, MPS (validated); other PyTorch devices untested | NVIDIA sm_80 or newer: RTX 30, 40, 50 series, A100, H100, B200 |
| RTX 20 / GTX 16 (sm_75) | works | no binary: `auto` uses portable, `mode="kernels"` raises |
| Python / PyTorch | 3.11-3.14, torch >= 2.8 | same |

The `kernels` column is the CUDA kernels; on Apple silicon the Metal kernels run for some families ([Apple silicon](#apple-silicon-mps)). Kernels were tuned and measured on an RTX 5060 Ti only; see [Other GPUs](OTHER_GPUS.md). Before Ampere, bfloat16 is
emulated, so models that use it compute in float32 there.

## Troubleshooting

- `BackendUnavailableError` / `KernelUnavailableError`: `mode="kernels"` cannot run here. The message names the
  reason (GPU too old, no binary, driver). `mode="auto"` falls back to portable instead. `fastforks env` shows the
  binaries your GPU would use.
- A kernel or weight download fails: check the network, or fetch ahead with `fastforks download MODEL --kernels` and
  copy the cache (see [Docker images](docker.md)). `FASTFORKS_OFFLINE=1` forbids downloads.
- Gated weights (for example VGGT-Omega, DINOv3): fastforks never handles your Hugging Face token; gated downloads go
  through `huggingface_hub` (`pip install huggingface_hub`). Accept the licence on the model's Hugging Face page, log in
  (`hf auth login`, or `huggingface_hub.login()` in Python), then load again. By hand:
  `hf download <repo> <file> --revision <rev>` (the error prints it) puts the file in the hub cache, or use a copy on
  `FASTFORKS_WEIGHTS_PATH` or `weights=PATH`.
- `CUDA out of memory`: use a smaller input, `mode="portable"` (often less memory), or `release_graphs()` on models
  that replay CUDA graphs. `fastforks info MODEL` shows the largest size tested.
- A model fails in `load` naming a missing package: install the extra it names, for example `fastforks-core[transformers]`.
