# MLX on newer Macs: what to re-tune and what to try

All MLX numbers so far are from the M1 Pro (16 GB, 16 GPU cores, 200 GB/s). Every port must keep running on M1, so the
changes below go behind runtime dispatch on `mx.metal.device_info()` (architecture string, memory size), with the M1
paths as the fallback. Context: the roadmap's MLX bullet, and `/Volumes/Daten/ff-mlx/v2/REPORT.md` for the M1
measurements behind each tuning choice.

## First run on a new Mac (planned: M4 Pro)

With the MLX backend (branch `work/mlx`, cut from `work/ports`):

1. Add both device entries to `docs/data/devices.toml`: `python tools/speed_survey.py detect` (MPS) and
   `detect --mlx` (MLX). Add the new device pages to `mkdocs.yml`.
2. `python tools/mlx_tune.py` runs all of it. The single steps:
   - `gates`: the golden gates of every MLX port, fp32 and fp16. Run these before any timing, since GPU
     generations can round differently in custom kernels.
   - `bench [NAME...]`: one micro-benchmark per tunable (`cache`, `threadgroup`, `lane`, `conv-pad`, `winograd`,
     `batch-pair`, `compile`). It prints the `TUNING` row to paste when a value should change.
   - `survey --mps`: the MLX and MPS survey rows on this machine.
   - `ratios`: MLX against MPS per ref and dtype. Compare ratios, not absolute ms, since PyTorch MPS also gets
     faster on newer chips.
   - Regenerate `src/fastforks/_core/apple_mlx.json` from the new rows (`tools/gen_auto_mode.py`, after setting
     `APPLE_DEVICE` / `APPLE_MLX_DEVICE` to the new slugs). It decides which ports `device="mps"` takes; today it
     holds M1 Pro measurements only, so one table per Mac generation may be needed if the choices differ.
3. A new chip's values go in one row of `fastforks._core.mlx.TUNING`, keyed by the exact GPU architecture
   (e.g. `applegpu_g16s`) or the generation (`applegpu_g16`). The M1 values are the `default` row.
   `FASTFORKS_MLX_TUNING="winograd=0,threadgroup=128"` tries values without editing code.

Decision rules used by `bench`:
- A tunable changes for more than 3 % speed.
- The cache limit changes for more than 5 % speed.
- A tunable that mainly decides memory changes for 5 % lower peak at equal speed.

Things to know:
- **Weight caches:** `conv_pad` and `winograd` change the converted weight layout, so each setting writes its own
  cache file under `<cache>/mlx/stereo/crestereo/`. Delete the extras after benchmarking.
- **Threadgroup maximum:** it differs per GPU (768 on the M1, so 1024 is rejected there). `bench` reports a size
  that isn't possible and carries on.
- **Cache limit:** 1/32 of RAM, with a 512 MB floor and a 2 GB cap. On the M1, limits above 1 GB bought at most
  3 %. `bench cache` re-decides it per Mac.
- **M5 hook:** `has_neural_accelerators()` exists for M5-specific paths, but nothing uses it yet.
- **Golden tests:** they use an empty cache, so the original checkpoints must be findable by file name on
  `FASTFORKS_WEIGHTS_PATH`.

## M1 tuning decisions to re-measure

Each of these was measured on the M1 and may flip on a newer GPU.

| decision (M1) | why on M1 | re-check on M4 / M5 |
|---|---|---|
| Pad 96-channel conv stages to 128 (CREStereo encoder), 126 to 128 (motion encoder) | MLX 128->96 3x3 conv 40 ms vs 7 ms for 128->128 | Measure the conv kernel selection per channel count; drop the padding if the pathology is gone (it costs memory) |
| Winograd F(4,5) as one fp32 matmul for the GRU 1x5 / 5x1 convs | 15 % whole-run win | Probably bigger with faster matmul; also try it for the 3x3 convs, where it lost on M1 |
| im2col + matmul for convs: rejected | slower than MLX conv on M1 | Re-check on M5 (matmul on neural accelerators) |
| Encoder runs one image at a time (`batch_pair` off) | same speed as batched, about 230 MB lower peak | With more bandwidth and memory, batching both images may win |
| Cache limit 1/32 of RAM, 512 MB floor, 2 GB cap | uncapped grew to 10.7 GB and swapped a 16 GB Mac; 1 GB was only 3 % faster than 512 MB | `bench cache` |
| fp16 storage, fp32 accumulation in kernels | bf16 not faster on M1 | Measure bf16 on M2+ (bf16 storage keeps the fp32 range); re-run the gates |
| Threadgroup sizes and tiles of the correlation / sampling / GRU kernels | tuned for 16 cores, M1 occupancy rules | M3/M4 have Dynamic Caching (registers and threadgroup memory allocated at run time), so larger tiles and more registers per thread may now pay off; sweep sizes |
| `mx.compile`: per iteration step only | 5-10 % at most | Re-check; the gain depends on per-kernel launch overhead |

## Ideas for the M4 Pro specifically

- **More cores and bandwidth** (up to 20 GPU cores, 273 GB/s vs 200 GB/s). Memory-bound parts gain the most: the
  portable correlation and the Winograd transform traffic. Check whether CREStereo stops being conv-bound.
- **More memory** (24 / 48 / 64 GB):
  - Run families that OOM on the 16 GB Mac (foundationstereo, flowformer, gmflow scale2 at 1080x1920).
  - Try larger working sets: keep both images' features resident, larger cache limit.
- **Dynamic Caching** (M3 / M4 GPUs): retune kernel occupancy (see the table above). Fused kernels that were
  register-limited on M1 (the GRU epilogue, the offset correlation with bilinear taps) are the first to sweep.
- No neural accelerators in the M4 GPU, so the matmul-heavy ViT families should scale roughly with cores and
  bandwidth.

## Ideas for the M5 (GPU neural accelerators)

- MLX routes matmul and `scaled_dot_product_attention` to the neural accelerators (macOS 26.2+), so the ViT families gain
  with no code change. There is no accelerated conv path, so moving conv work into matmul becomes the main lever:
  - Winograd-as-matmul for the 3x3 convs and the encoders.
  - im2col + matmul re-checked.
- Metal 4 tensor ops (`matmul2d` inside shaders): fuse input transform, multiply, output transform and the gate
  epilogue into one kernel. This removes the Winograd tile buffers (about 350 MB peak in fp16 for CREStereo) and the
  transform traffic. The same applies to the correlation dot products.
- If the accelerated path is half-precision only, use Winograd points with smaller constants and re-run the gates.
- bf16 for the ViT families where the hardware path prefers it.

## Keeping it maintainable

Done in the first backend version:
- All per-generation tunables are in `fastforks._core.mlx.TUNING`.
- Each tunable has a micro-benchmark.
- Each machine records its MLX rows as its own device slug (`apple-m1-pro-mlx-16gb` for the M1 Pro).

## The ViT and DPT ports (DINOv2, DINOv3, Depth Anything V2)

These ports share `fastforks.shared.mlx_vit` (blocks, attention, RoPE) and, for Depth Anything V2 and its
fine-tunes, `fastforks.shared.depth_anything_v2.mlx_net` with the `dpt_mlx` resize kernel. What the M1 Pro showed,
and what to re-measure:

| finding (M1) | consequence there | re-check on M4 Pro / M5 |
|---|---|---|
| matmul, `scaled_dot_product_attention` and conv are at parity with PyTorch MPS per op at 2036 tokens | the ViT gain is overhead (small models, register variants) and memory, not throughput: 1.6x for ViT-S, 1.05x for ViT-L / H+ | M4: expect the same ratios. M5: matmul and attention go to the neural accelerators in MLX, so the large backbones should move from parity to a clear win with no code change; re-run `ratios` first |
| Depth Anything V2 float32 is 0.9x to 1.0x of MPS (the DPT head's 3x3 convs at full resolution are slower in MLX) | only float16 pays there (MPS is validated in float32 only) | Measure the 3x3 conv at 64 to 256 channels again; with more memory try float32 as the default where the head reaches parity |
| MLX bilinear resize through `take` is 3 to 4 times slower than one Metal kernel | `dpt_mlx.resize`, one thread per output pixel and lane of channels (`warp_lane`) | Sweep `warp_lane` (`bench lane`); with Dynamic Caching a whole-pixel lane (all channels per thread) may win |
| float16 stream scale 8 for hidden sizes from 768 (DINOv3) | keeps the residual stream inside float16 without float32 blocks | Try bf16 storage on M2+ instead of the scale; re-run the gates |
| conv as matmul plus shifted adds for the head: rejected | no gain, four times the memory | M5 only: matmul is accelerated and conv is not, so this (or Winograd F(4x4, 3x3) as one matmul) is the lever for the head |

Ideas:

- **M4 Pro, memory:** the Depth Anything V2 pos-embedding table and the compiled graph are cached per input size;
  with 24 GB or more keep several sizes resident and allow float32 ViT-L at 1080p without touching swap.
- **M4 Pro, head fusion:** fuse the head's resize with the following 1x1 / residual add in one kernel (register
  limited on M1); it removes one full-resolution buffer per fusion stage.
- **M5:** GELU and layer norm stay on the ordinary GPU path while matmul moves to the accelerators, so they become
  the next cost; a fused `layer_norm + matmul` input or a tanh-free GELU kernel is worth a micro-benchmark there.
- **M5, half precision:** if the accelerated matmul is float16 only, the float16 stream scale already keeps DINOv3
  in range; check ViT-g / H+ with registers at the gates before making float16 the default.

## Not ported on M1, and what would change that

- **RAFT-Stereo, IGEV and the other correlation / GRU stereo families:** the per-iteration conv mix is at parity
  with MPS in plain MLX (107.5 ms against 104.4 ms float32, 92.9 ms against 96.3 ms float16 at 1/4 resolution), so
  a portable port cannot reach 5 %. The lever is a 2-D Winograd F(4x4, 3x3) as one float32 matmul for the 384 -> 128
  GRU convs (about half of an iteration; estimated 19 % whole-run gain) on top of the CREStereo correlation
  kernels. Worth building first on the M5, where the matmul is accelerated and the conv is not.
- **Semantic segmentation heads on DINOv2 (`dinov2_linear`):** the sliding window and the 150-channel logit resize
  would run on the CPU; needs an MLX slide and the `dpt_mlx` resize with accumulation.
- **SAM image encoders, Depth Anything 3, MoGe, Metric3D, UniDepth:** not tried. They reuse `mlx_vit`; expect the
  DINOv2 pattern (float16 about 1.2x where MPS runs float32 only, parity otherwise) on M1 / M4 and a real win on M5.
