# Other GPUs

The reference times and parity checks in these docs come from one RTX 5060 Ti 16 GB (sm_120). The A100, A10, L4, L40S
and RTX PRO 6000 pages are speed surveys, each result checked against the golden outputs. This page explains what runs on
other hardware, how to check your setup, and how to measure your GPU and send us the numbers.

## What runs where

`portable` is plain PyTorch and runs wherever PyTorch runs (CPU, any CUDA GPU, MPS). `kernels` adds our CUDA kernels.
It needs all of the following:

- Linux on x86-64 with glibc 2.34 or newer. aarch64 hosts (Jetson, Grace, DGX Spark) run `portable` only.
- An NVIDIA GPU with compute capability 8.0 or newer.
- An NVIDIA driver for CUDA 12.8 or newer (driver 570 or newer). The kernels are built with CUDA 12.8, those of
  GREAT-IGEV, IGEV++, MoCha-Stereo, S2M2 and `fvm_conv` (all classification families, DINOv3-ConvNeXt, SegFormer)
  with CUDA 13.0 (driver 580 or newer). These libraries (`fvm_conv`, `igevpp`, `mocha`, `s2m2_cutlass`) also ship
  their CUDA 12.8 build, which drivers 570-579 load automatically (`fvm_conv`'s is slower: ResNet-50 core 0.76 instead of 0.64 ms).
- cuDNN 9 (the copy torch ships): most stereo families build fp16-accumulate cuDNN plans through its backend API.
- One CUDA device per process for `kernels`.

Each kernel library ships native code (SASS) for sm_80, 86, 89, 90, 100 and 120, plus PTX for compute_80. A GPU uses
the exact SASS if there is one. Otherwise it uses SASS with the same major version and the highest lower minor,
otherwise the PTX, which the driver compiles on the first load.

| compute capability | example GPUs | `kernels` uses | tested |
|---|---|---|---|
| CPU | any | not available, `portable` only | yes |
| MPS | Apple silicon | Metal kernels for the families in the table below, else `portable` | M1 Pro ([usage](usage.md#apple-silicon-mps), [speed](devices/apple-m1-pro-16gb.md)); recent Metal changes not yet rechecked |
| < 8.0 | T4, V100, RTX 20xx, GTX 10xx | not available, `portable` only | no |
| 8.0 | A100, A30 | sm_80 | survey ([A100](devices/a100-sxm4-40gb.md)) |
| 8.6 | RTX 30xx, A10, A40, RTX A6000 | sm_86 | survey ([A10](devices/a10-22gb.md)) |
| 8.7 | Jetson Orin | not available (aarch64) | no |
| 8.9 | RTX 40xx, L4, L40S, RTX 6000 Ada | sm_89 | survey ([L4](devices/l4-22gb.md), [L40S](devices/l40s-44gb.md)) |
| 9.0 | H100, H200 | sm_90 | no |
| 10.0 | B200, GB200 (x86-64 hosts only) | sm_100 | no |
| 10.3 | B300 | sm_100 | no |
| 12.0 | RTX 50xx, RTX PRO 6000 Blackwell | sm_120 | yes (RTX 5060 Ti); survey ([RTX PRO 6000](devices/rtx-pro-6000-96gb.md)) |
| 12.1 and newer minors | future 12.x GPUs | sm_120 | no |
| newer majors | future GPUs | compute_80 PTX, JIT on first load | no |

"Built but not tested" means the kernels compile for that target, but nobody has checked them against the reference outputs on
such a GPU. If `kernels` gives wrong results, or a different speed-up than [Benchmarks](benchmarks.md), please
[report it](#contributing-numbers). `portable` with fp16 is the same PyTorch code on every GPU.

Precision: on CUDA, the default `dtype=None` uses each model's tested reduced precision (float16, or bfloat16 for
DEFOM). GPUs before compute capability 8.0 have no fast bfloat16, so DEFOM computes in float32 there automatically.

`mode="auto"` (the default) picks `kernels` when it can load and falls back to `portable` otherwise. It logs the reason once
per load: at WARNING (visible with default logging) when kernels fails on a CUDA device, else at INFO. `mode="kernels"` never
falls back: it raises `BackendUnavailableError` / `KernelUnavailableError` with the reason and a remedy.

### What each family runs on

Per family (families with the same support share a row; from the model cards; variants of a family share it). CPU: `portable`, float32 only.
`kernels` runs only on CUDA and on MPS (where listed); all other pairs run `portable`. fp32 / fp16 / bf16 = float32 / float16 / bfloat16.
"MPS portable (validated)" lists the dtypes checked on Apple silicon, the first is the default; `-` = not validated
there (`device="mps"` still tries, and an op MPS lacks raises).

<!-- capabilities:begin (tools/gen_docs.py, from the model cards) -->
| families | CUDA portable | CUDA kernels | MPS portable (validated) | MPS kernels (libraries, dtype) |
|---|---|---|---|---|
| alltracker, anycalib, anysplat, ben2, birefnet, boq, c_foundationstereo, convnext, convnext_features, convnextv2, dat, deit, densenet, depth_anything_v2, depthsplat, detr, dfine, dinov2_linear, dinov2_linear_cls, efficientnet, efficientnetv2, efficientvit, efficientvit_sam, fast_acvnet, fastvit, geocalib, gwcnet, hat, inception_v3, isnet, lingbot_depth, locotrack, lwdetr, megaloc, memflow, memfof, metric3d, mobile_sam, mobilenetv1, mobilenetv2, mobilenetv3, mobilenetv4, mvsanywhere, mvsplat, neuflow_v2, nsfw_vit, owlv2, patchmatchnet, prompt_depth_anything, psmnet, rcnn, realesrgan, repvit, resnet, rife, romav1, rt_monster_plusplus, rtdetrv2, sam, sam2, sam_hq, segformer, shufflenetv2, span, starnet, swin, swinir, tapir, tapnext, tinyvit, track_on, u2net, unidepth_v2, unik3d, video_depth_anything, vit, vitmatte, waft_stereo, wd_tagger | fp16 | fp16 | fp32 | - |
| alexnet, c_radio, cas_gwcnet, clip, coex, depth_anything_3, depth_pro, dinov2, dinov3, dinov3_convnext, dlnr, eomt, eva02, hrnet, lotus, mapanything, marigold, marigold_normals, mask2former, midas, mmseg, oneformer, regnet, retinaface, rfdetr, rfdetr_keypoint, rtmdet, rtmo, s2m2, sam3, sapiens, segnext, siglip, siglip2, squeezenet, stablenormal, tipsv2, vgg, vggt_omega, yolox, zoedepth | fp16 | fp16 | fp32/fp16 | - |
| coatnet, codeformer, davit, dedode, efficientformer, efficientformerv2, flexglue, flowseek, focalnet, gmflow, grounding_dino, hgnetv2, inspyrenet, loftr, shvit | fp16 | - | fp32 | - |
| cfnet, crestereo, foundationstereo, great_igev, hitnet, igev_plusplus, mocha_stereo, monster, monster_plusplus, raft_stereo, selective_igev | fp16 | fp16 | fp32/fp16 | stereo_metal fp16 |
| rtmpose | fp16 | fp16 | fp32/fp16 | pose_metal fp32 |
| aanet, bridgedepth, igev_stereo, las2, nmrf | fp16 | fp16 | fp32 | stereo_metal fp16 |
| raft, sea_raft | fp16 | fp16 | fp32 | flow_metal fp16 |
| perception_encoder | fp16 | fp16 | fp32/fp16 | vit_metal fp16 |
| flowformer, lightglue, xfeat | fp32 | - | fp32 | - |
| vitpose | fp16 | fp16 | fp32/fp16 | pose_metal fp16 |
| moge | fp16 | fp16 | fp32 | sparse_metal fp32 |
| insid3, pi3 | fp16 | fp16 | - | - |
| loma | bf16 | - | fp32 | - |
| romav2 | bf16 | bf16 | fp32 | - |
| defom_stereo | bf16 | fp16 | fp32 | stereo_metal fp16 |
| vggt | fp16 | fp16 | fp32 | vggt_metal fp32 |
<!-- capabilities:end -->

On MPS (Apple silicon), `portable` works for every validated family and `mode="auto"` picks the Metal `kernels` where the card lists
`mps_kernels` (`card.kernel_libs_on(device)` returns the Metal libraries there, the CUDA ones elsewhere). The Metal kernels compile on
the device at first use. They were written and tested on an M1 Pro only; the changes of 2026-11-04 (`mps.adapt` on
all Metal runners, `win_rpe_attn` sentinel, kernel input guards, `rectify` in float64 on CPU) still need a check on a Mac.

The pose and Perception Encoder Metal paths (2026-11-09, M1 Pro) cover the memory-bound parts only and leave the
rest on PyTorch: `vit_metal` rotates PE's q / k in one pass; `pose_metal` runs ViTPose's patch embedding and
deconvolutions as GEMMs plus a Metal overlap-add, DARK decoding in one pass, and RTMPose's OpenCV-exact crops.
RTMPose's `kernels` stays fp32 on MPS (fp16 is not faster there and moves unsure keypoints more).

### Kernel libraries per family

Each family's kernels backend needs these libraries; `fastforks env` shows whether a binary exists for your GPU.

<!-- libraries:begin (tools/gen_docs.py, from the model cards) -->
| library | families |
|---|---|
| `cfs` | c_foundationstereo |
| `crestereo` | crestereo |
| `cudnn_hacc` | aanet, alexnet, anycalib, anysplat, ben2, c_foundationstereo, c_radio, cas_gwcnet, cfnet, clip, coex, convnextv2, crestereo, dat, defom_stereo, deit, densenet, depthsplat, detr, dfine, dinov2_linear_cls, dlnr, efficientnet, efficientnetv2, efficientvit, eva02, fast_acvnet, fastvit, foundationstereo, geocalib, great_igev, gwcnet, hat, hrnet, igev_plusplus, igev_stereo, inception_v3, isnet, las2, lotus, lwdetr, marigold, marigold_normals, midas, mmseg, mobilenetv1, mobilenetv2, mobilenetv3, mobilenetv4, mocha_stereo, monster, monster_plusplus, mvsanywhere, mvsplat, nsfw_vit, perception_encoder, psmnet, raft, raft_stereo, rcnn, realesrgan, regnet, repvit, resnet, retinaface, rife, rt_monster_plusplus, rtdetrv2, rtmdet, rtmo, rtmpose, s2m2, sam3, sapiens, sea_raft, segformer, segnext, selective_igev, shufflenetv2, siglip, siglip2, span, squeezenet, stablenormal, starnet, swin, swinir, u2net, unidepth_v2, vgg, vggt, vit, vitmatte, waft_stereo, wd_tagger, yolox, zoedepth |
| `cutlass_conv` | aanet, ben2, c_foundationstereo, cas_gwcnet, cfnet, coex, crestereo, defom_stereo, dlnr, fast_acvnet, great_igev, gwcnet, igev_plusplus, igev_stereo, isnet, las2, mmseg, mocha_stereo, monster, monster_plusplus, mvsanywhere, psmnet, raft, raft_stereo, rt_monster_plusplus, sea_raft, selective_igev, u2net, vitmatte, waft_stereo |
| `deform_conv` | aanet, birefnet |
| `dpt_fused` | anycalib, boq, defom_stereo, depth_anything_3, depth_anything_v2, depth_pro, dinov2, dinov2_linear, dinov3, insid3, lingbot_depth, megaloc, metric3d, moge, monster, monster_plusplus, mvsanywhere, prompt_depth_anything, sam, sam2, sam_hq, tipsv2, track_on, unidepth_v2, unik3d, vitmatte, waft_stereo |
| `fastbb` | anycalib, anysplat, boq, bridgedepth, c_foundationstereo, crestereo, defom_stereo, depth_anything_3, depth_anything_v2, depth_pro, depthsplat, dinov2, dinov2_linear, dinov3, dlnr, eomt, foundationstereo, insid3, lingbot_depth, mapanything, megaloc, metric3d, moge, monster, monster_plusplus, mvsanywhere, nmrf, pi3, prompt_depth_anything, rfdetr, rfdetr_keypoint, romav1, rt_monster_plusplus, sam3, sapiens, tipsv2, track_on, unidepth_v2, unik3d, vggt, vggt_omega, video_depth_anything, waft_stereo |
| `ffs_cutlass` | c_foundationstereo |
| `fs` | c_foundationstereo, foundationstereo |
| `fs_cutlass` | c_foundationstereo, foundationstereo |
| `fvm_conv` | alexnet, alltracker, c_radio, clip, convnext, convnext_features, convnextv2, deit, densenet, dfine, dinov2_linear_cls, dinov3_convnext, efficientnet, efficientnetv2, efficientvit, efficientvit_sam, eva02, fastvit, hrnet, inception_v3, locotrack, mobile_sam, mobilenetv1, mobilenetv2, mobilenetv3, mobilenetv4, nsfw_vit, owlv2, perception_encoder, regnet, repvit, resnet, retinaface, rfdetr, rtmdet, rtmo, rtmpose, segformer, segnext, shufflenetv2, siglip, siglip2, squeezenet, starnet, swin, tapir, tapnext, tinyvit, vgg, vit, vitpose, wd_tagger, yolox |
| `fvm_extra` | alexnet, c_radio, clip, convnextv2, deit, densenet, dinov2_linear_cls, efficientnet, efficientnetv2, efficientvit, efficientvit_sam, eva02, fastvit, hrnet, inception_v3, mobile_sam, mobilenetv1, mobilenetv2, mobilenetv3, mobilenetv4, nsfw_vit, perception_encoder, regnet, repvit, resnet, retinaface, rtmdet, rtmo, rtmpose, segformer, segnext, shufflenetv2, siglip, siglip2, squeezenet, starnet, swin, vgg, vit, wd_tagger, yolox |
| `fvm_mb` | alexnet, c_radio, clip, convnextv2, deit, densenet, dinov2_linear_cls, efficientnet, efficientnetv2, efficientvit, efficientvit_sam, eva02, fastvit, hrnet, inception_v3, mobile_sam, mobilenetv1, mobilenetv2, mobilenetv3, mobilenetv4, nsfw_vit, perception_encoder, regnet, repvit, resnet, retinaface, rtmdet, rtmo, rtmpose, segformer, segnext, shufflenetv2, siglip, siglip2, squeezenet, starnet, swin, vgg, vit, wd_tagger, yolox |
| `fvm_r3` | alexnet, c_radio, clip, convnextv2, deit, densenet, dinov2_linear_cls, efficientnet, efficientnetv2, efficientvit, efficientvit_sam, eva02, fastvit, hrnet, inception_v3, mobile_sam, mobilenetv1, mobilenetv2, mobilenetv3, mobilenetv4, nsfw_vit, perception_encoder, regnet, repvit, resnet, retinaface, rtmdet, rtmo, rtmpose, segformer, segnext, shufflenetv2, siglip, siglip2, squeezenet, starnet, swin, vgg, vit, wd_tagger, yolox |
| `fvm_seg` | convnext, convnext_features, dinov3_convnext, mask2former, oneformer |
| `hitnet` | hitnet |
| `hitnet_conv` | hitnet |
| `igev` | aanet, ben2, c_foundationstereo, cas_gwcnet, cfnet, coex, crestereo, defom_stereo, dlnr, fast_acvnet, great_igev, gwcnet, igev_plusplus, igev_stereo, isnet, las2, mmseg, mocha_stereo, monster, monster_plusplus, mvsanywhere, psmnet, rt_monster_plusplus, selective_igev, u2net, vitmatte, waft_stereo |
| `igevpp` | great_igev, igev_plusplus, monster_plusplus |
| `mocha` | mocha_stereo |
| `mvsanywhere` | mvsanywhere |
| `patchmatchnet` | patchmatchnet |
| `raft` | dlnr, raft_stereo |
| `raft_flow` | memflow, memfof, neuflow_v2, raft, sea_raft |
| `romav2` | romav2 |
| `s2m2` | s2m2 |
| `s2m2_cutlass` | s2m2 |
| `seg_conv` | mmseg, segnext |
| `seg_up` | dinov2_linear, eomt, mask2former, mmseg, oneformer, sapiens, segformer, segnext |
| `sparse_conv` | moge |
| `stereo_ops` | coex, las2 |
| `vggt_omega` | vggt_omega |
<!-- libraries:end -->

## Check your setup

```bash
fastforks env
```

This prints the fastforks, Python, torch, CUDA, cuDNN and driver versions, and the GPU with its compute capability.
For each kernel library it shows which binary your GPU would use and whether it is already downloaded. Example on the
tested machine (RTX 5060 Ti, 2026-10-27):

```text
fastforks              0.1.0
python                 3.12.15 (linux x86_64)
torch                  2.14.1+cu130 (CUDA 13.0, cuDNN 92400)
driver                 supports CUDA 13.2
gpu                    NVIDIA GeForce RTX 5060 Ti, sm_120
kernels cfs            sm_120, local
kernels crestereo      sm_120, local
kernels cudnn_hacc     no binaries: cuDNN plans via the cuDNN backend API (cuDNN 9.24.0)
kernels cutlass_conv   sm_120, local
kernels fastbb         sm_120, local
kernels fs             sm_120, local
kernels fs_cutlass     sm_120, local
kernels hitnet         sm_120, local
kernels hitnet_conv    sm_120, local
kernels igev           sm_120, local
kernels raft           sm_120, local
kernels s2m2           sm_120, local
kernels s2m2_cutlass   sm_120, local
```

Possible states of a kernel library:

| state | meaning |
|---|---|
| `sm_XY, local` | ready |
| `sm_XY, not downloaded yet (fastforks download --kernels)` | downloaded on the first `kernels` load, or now with `fastforks download MODEL --kernels` |
| `compute_80 (PTX, JIT-compiled on first load), local` | no native code for this GPU: the first load takes longer and speed is untested |
| `no binary for sm_XY (built: ...)` | GPU too old (< 8.0): `portable` only |
| `..., but needs a driver for CUDA >= 12.8 (have ...)` | update the NVIDIA driver |
| `built for ...; no CUDA GPU` | no GPU visible to torch |
| `cuDNN plans ... (unusable: ...)` | torch's cuDNN 9 (`libcudnn.so.9`) cannot be loaded: reinstall torch |

Which backend a loaded model uses:

```python
import logging

from fastforks import stereo

logging.basicConfig(level=logging.INFO)  # shows every backend choice, e.g. "kernels unavailable (...); using portable"
model = stereo.load("las2-s")
print(model.backend.mode)  # "kernels" or "portable"
stereo.load("las2-s", mode="kernels")  # raises with the reason instead of falling back
```

Offline machines: download weights and kernels on a machine with internet, copy the cache, and point
`FASTFORKS_CACHE` at the copy.

```bash
fastforks download las2-s depth-anything-v2-s --kernels --all-archs   # --all-archs: binaries for every target
# copy the cache folder (see FASTFORKS_CACHE) to the offline machine, then there:
export FASTFORKS_CACHE=/path/to/copy FASTFORKS_OFFLINE=1
```

### Common errors

| message contains | fix |
|---|---|
| `no binary for sm_75` | GPU older than compute capability 8.0: use `mode="portable"` (or `auto`) |
| `update the NVIDIA driver to >= 570` (or 580) | the kernels need a driver for their CUDA version (12.8, or 13.0) |
| `backend API is not usable: ...` | torch's cuDNN 9 (`libcudnn.so.9`) cannot be loaded: reinstall torch |
| `cannot download ... fastforks download --kernels` | no network: download on another machine, set `FASTFORKS_CACHE` to a copy of the cache (see Offline machines above) |
| `built for Linux x86-64` / `needs a CUDA device` / `ROCm/HIP` | `portable` only on this host |
| `kernels already runs on cuda:0 in this process` | the kernels and their buffers are per process: use one process per GPU |

## Measure your GPU

`fastforks bench` times one model on a synthetic image (a pair for stereo). It runs 4 warm-up calls, then prints the
median of `--runs` calls and the peak memory of the torch allocator:

```bash
fastforks bench las2-s --mode kernels --size 375x1242 --runs 20
fastforks bench las2-s --mode portable --size 375x1242 --runs 20
```

Output of the two commands on the tested machine (2026-11-05):

```text
las2-s@1 375x1242 kernels float16: 4.0 ms (median of 20), peak 115 MiB allocated / 326 MiB reserved
las2-s@1 375x1242 portable float16: 20.5 ms (median of 20), peak 302 MiB allocated / 330 MiB reserved
```

To measure the models of the [Benchmarks](benchmarks.md) tables in both modes and save everything to one file:

```bash
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
out=fastforks_bench_$gpu.txt
fastforks env > "$out"
for m in las2-s hitnet-eth3d coex-sceneflow s2m2-s igev-stereo-sceneflow crestereo-eth3d depth-anything-v2-s; do
  for mode in kernels portable; do
    fastforks bench "$m" --mode "$mode" --size 375x1242 --runs 20 2>&1 | tee -a "$out"
  done
done
```

Close other GPU programs first. Laptop GPUs and GPUs without a fixed clock vary by 5-10 % between runs, so run the
loop twice and report both runs.

Check that `kernels` and `portable` agree on your GPU:

```python
import numpy as np

from fastforks import stereo

rng = np.random.default_rng(0)
left = rng.integers(0, 256, (375, 1242, 3), dtype=np.uint8)
right = np.roll(left, -8, axis=1)
a = stereo.load("las2-s", mode="kernels")(left, right).disparity
b = stereo.load("las2-s", mode="portable")(left, right).disparity
print("max |kernels - portable| px:", (a - b).abs().max().item(), "mean:", (a - b).abs().mean().item())
```

On the tested machine this prints `max |kernels - portable| px: 4.794837951660156 mean: 0.1790103018283844` (2026-10-27).
Random images have no real structure, so single pixels can differ by several pixels of disparity: compare the mean. The GPU reference tests compare both modes with one stored reference output on a
synthetic pair (`tests/synth.py`), by the mean and the 99th percentile of the difference.

## Contributing numbers

Open an issue at [github.com/kleinicke/fastforks](https://github.com/kleinicke/fastforks/issues) titled
"Numbers: <GPU name>" and attach:

1. the `fastforks env` output,
2. the bench file from the loop above (both runs),
3. the output of the agreement snippet,
4. anything unusual: errors, a slow first load (PTX JIT), clocks locked or not, laptop or desktop.

Numbers from untested architectures (everything except sm_120) are the most useful. If a `kernels` load fails, the full
error message and `fastforks env` are enough.
