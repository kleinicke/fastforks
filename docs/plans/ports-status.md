# Ports in progress: who does what, and where each network stands

Status (2026-11-10): working notes for the branch `work/ports`, which sits on top of 0.1.1. Two machines work on it:

- **CUDA machine (RTX 5060 Ti):** ports a network, tunes its CUDA kernels and records its rows. It owns the model
  code, the CUDA kernels and every file in `docs/data/` except the Mac speed file.
- **Mac (M1 Pro):** takes a network once the CUDA machine is done with it: MPS validation, Metal kernels where they
  pay, and the Mac rows in `docs/data/speed/apple-m1-pro-16gb.jsonl`. It does not change the portable model code
  beyond what MPS needs.

## Rules for the branch

- One commit per finished network, not before. Finished means: `ruff format --check`, `ruff check`, `ty check`,
  `tools/gen_docs.py --check` and `pytest -m "not gpu"` pass, the GPU goldens are recorded (`pytest -m gpu --record`),
  the size-table entries exist (`tools/bench_sizes.py run --modes portable --runs 3 --out new.jsonl REF ...`, then
  `tools/extract_sizes.py new.jsonl`; `tests/test_sizes.py` fails without them), and the rows below exist.
- `git pull --rebase` before every push. Never force-push.
- A licence gate comes first: code and weights licences are checked against the source before any work starts, and
  a non-commercial weight licence stops the port (see [model-landscape.md](model-landscape.md)).
- Rows every network needs (the statistics pages reuse them): original fp32 and fp16, `portable`, `kernels`,
  `torch.compile` and TensorRT, each with peak memory, and accuracy against the original. A failed export is a row
  too.
- Notes on how to tune a network differently on datacenter GPUs go to
  [datacenter-tuning.md](datacenter-tuning.md).
- When the CUDA machine finishes a network it adds a line to "Waiting for the Mac" below. The Mac removes the line
  when its part is committed.

## Open on the CUDA machine

In this order, unless something more useful turns up:

1. CLIP: `clip-l14-dfn2b`, `clip-h14-dfn5b` and `clip-h14-378-dfn5b` have no `torch.compile` or TensorRT rows in
   `docs/data/speed_compile.jsonl`.
2. Next networks, from [model-landscape.md](model-landscape.md): MaxViT, `moge-2-vitl` (no normal head), and more
   flow and pose models. ConvNeXt features, TIPSv2, RepViT and MobileSAM are already merged.

## Waiting for the Mac

- nnFoundation (`nnfoundation-cnn`, `nnfoundation-vit`, volume encoders; CC-BY-SA-4.0 weights): portable MPS goldens
  and Mac speed rows (CNN 200^3, ViT 192^3; the ViT's 13824-token attention may not fit an MPS fp32 path). Weights in
  `/home/florian/projects/data/ff-weights/nnfoundation/`.
- nnU-Net v2 loader (`fastforks.volume.load_nnunet`, volume; shared engine with TotalSegmentator): MPS check of
  `tests/test_volume.py` and the TotalSegmentator MPS goldens / speed rows again (the engine now accumulates in float16
  like nnU-Net, also on the CPU). Weights: any nnU-Net v2 results folder (TotalSegmentator's Dataset297 in
  `/home/florian/projects/data/ff-weights/totalsegmentator/`).
- PP-OCRv5 and PARSeq (`ppocrv5-server`, `ppocrv5-mobile`, `parseq`, `parseq-tiny`, ocr): portable MPS goldens
  and Mac speed rows (1080x1920 page, 32x128 crop). No kernels runner. Weights: the pinned Hugging Face (PP-OCRv5) and
  GitHub release (PARSeq) URLs of the cards (also in `/home/florian/projects/data/ff-weights/ppocrv5/` and
  `/home/florian/projects/data/ff-weights/parseq/`).
- **Since 2026-11-15 `stream` defaults to True (CUDA only; MPS / CPU stay resident) for SeedVR2 and Marigold V2: Mac re-measures Marigold V2 and SeedVR2 after this change (speed rows and goldens).**
- SeedVR2 (`seedvr2-3b`, `seedvr2-7b`, restoration): portable MPS goldens and Mac speed rows (512x512 -> 1024x1024;
  7B only with `stream=True` or `quant="nf4"` below 32 GB). No kernels runner (nf4 is CUDA only). Weights: converted
  bf16 files in `/home/florian/projects/data/ff-weights/seedvr2/` (7B on /mnt/hdd, symlinked); HF repos not created yet.
- CodeFormer (`codeformer`, restoration): portable MPS goldens and Mac speed rows (512x512 aligned face; MPS runs
  float32). No kernels runner. Weights: the pinned GitHub release URL of the card (also in
  `/home/florian/projects/data/ff-weights/codeformer/`).
- SegFormer fine-tunes (`segformer-b2-clothes`, `segformer-b5-face-parsing`, segmentation): portable MPS goldens
  and Mac speed rows (512x512). Weights: the pinned Hugging Face URLs of the card (also in
  `/home/florian/projects/data/ff-weights/segformer/`).
- Falconsai NSFW (`falconsai-nsfw-vit`, classification): portable MPS goldens and Mac speed rows (224x224). Weights:
  the pinned Hugging Face URL of the card (also in `/home/florian/projects/data/ff-weights/nsfw_vit/`).
- WD v3 taggers (`wd-vit-tagger-v3`, `wd-convnext-tagger-v3`, `wd-vit-large-tagger-v3`, `wd-eva02-large-tagger-v3`,
  classification): portable MPS goldens and Mac speed rows (448x448). Weights: the pinned Hugging Face URLs of the
  card (also in `/home/florian/projects/data/ff-weights/wd_tagger/`).
- DAT (`dat-x4`, `dat-2-x4`, `dat-light-x4`, restoration): portable MPS goldens and Mac speed rows (512x512; MPS runs
  float32). No Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in
  `/home/florian/projects/data/ff-weights/dat/`).
- HAT (`hat-x4`, `hat-s-x4`, `real-hat-gan-x4`, restoration): portable MPS goldens and Mac speed rows (512x512; MPS
  runs float32). No Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in
  `/home/florian/projects/data/ff-weights/hat/`).
- DRCT (`drct-x4`, `drct-l-x4`, restoration): portable MPS goldens and Mac speed rows (512x512; MPS runs float32). No
  Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in
  `/home/florian/projects/data/ff-weights/drct/`).
- Swin2SR (`swin2sr-x4`, `swin2sr-x2`, `swin2sr-x4-real`, restoration): portable MPS goldens and Mac speed rows
  (512x512; MPS runs float32). No Metal kernels runner. Weights: the pinned GitHub release URLs of the card (also in
  `/home/florian/projects/data/ff-weights/swin2sr/`).
- NAFNet (`nafnet-gopro`, `nafnet-sidd`, restoration): portable MPS goldens and Mac speed rows (512x512; MPS runs
  float32). No Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in
  `/home/florian/projects/data/ff-weights/nafnet/`).
- Restormer (`restormer-deblur`, `restormer-derain`, `restormer-sidd`, restoration): portable MPS goldens and Mac
  speed rows (512x512; MPS runs float32). No Metal kernels runner. Weights: the pinned GitHub release URLs of the card
  (also in `/home/florian/projects/data/ff-weights/restormer/`).
- YuNet (`yunet-n`, `yunet-s`, pose): portable MPS goldens and Mac speed rows (1080x1920; MPS runs float32). No Metal
  kernels runner. Weights: the pinned GitHub URLs of the card (also in `/home/florian/projects/data/ff-weights/yunet/`).
- MatAnyone (`matanyone`, matting): portable MPS goldens and Mac speed rows (1080x1920; MPS runs float32). No Metal
  kernels runner. Weights: the pinned GitHub release URL of the card (also in
  `/home/florian/projects/data/ff-weights/matanyone/`).
- InSPyReNet (`inspyrenet`, `inspyrenet-fast`, matting): portable MPS goldens and Mac speed rows (1080x1920; check
  float16 on MPS: the context sums need float32). Weights: the pinned GitHub release URLs of the card.
- IS-Net (`isnet-general-use`, `isnet`, matting): portable MPS goldens and Mac speed rows (1080x1920). No Metal
  kernels runner. Weights: the pinned Google Drive URLs of the card (also in `~/scratch_tests/port-p5/w/`).
- U²-Net (`u2net`, `u2netp`, `u2net-human-seg`, matting): portable MPS goldens and Mac speed rows (1080x1920). No
  Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in `~/scratch_tests/port-p5/w/`).
- SwinIR (`swinir-m-x4`, `swinir-m-x2`, `swinir-m-x4-real`, `swinir-l-x4-real`, restoration): portable MPS goldens
  and Mac speed rows (512x512; check the float16 last-group scaling on MPS). No Metal kernels runner. Weights: the
  pinned GitHub release URLs of the card.
- SPAN (`span-x4`, `span-x2`, restoration): portable MPS goldens and Mac speed rows (512x512). No Metal kernels
  runner. Weights: the pinned Google Drive URLs of the card (also in `~/scratch_tests/port-p5/w/` on the CUDA machine).
- Real-ESRGAN (`realesrgan-x4plus`, `realesrgan-x2plus`, `realesrgan-x4plus-anime-6b`, restoration): portable MPS
  goldens and Mac speed rows (512x512). No Metal kernels runner. Weights: the pinned GitHub release URLs of the card.
- RIFE (`rife-4.25`, `rife-4.26`, `rife-4.25-lite`, interpolation): portable MPS goldens and Mac speed rows
  (1080x1920 pair). No Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in
  `~/scratch_tests/port-p5/w/rife/` on the CUDA machine).
- GIMM-VFI (`gimm-vfi-r`, `gimm-vfi-r-p`, interpolation, nc weights): portable MPS goldens and Mac speed rows
  (480x854 pair; the splatting is plain torch `scatter_add`, check its speed on MPS). No Metal kernels runner. Weights:
  the pinned HF URLs of the card (also in `data/ff-weights/gimm_vfi/`).
- EMA-VFI (`ema-vfi`, `ema-vfi-small`, `ema-vfi-t`, `ema-vfi-small-t`, interpolation, Apache-2.0): portable MPS goldens
  and Mac speed rows (480x854 pair). No Metal kernels runner. Weights: Google Drive URLs of the card (also in
  `data/ff-weights/ema_vfi/`); mirror them to HF at release.
- FILM (`film`, interpolation, Apache-2.0): portable MPS goldens and Mac speed rows (480x854 pair). No Metal kernels
  runner. Weights: the pinned GitHub release URL of the card (also in `data/ff-weights/film/`).
- MegaLoc (`megaloc`, new retrieval task): portable float32 MPS goldens and Mac speed rows at 480x640. No Metal
  kernels runner. Weights: the pinned HF URL of the card (also in `data/ff-weights/megaloc/`).
- BoQ (`boq-dinov2`, retrieval): portable float32 MPS goldens and Mac speed rows at 480x640. No Metal kernels runner.
  Weights: the pinned GitHub release URL of the card (also in `data/ff-weights/boq/`).
- SAMURAI (`samurai` option of the `sam2.1-hiera-*` cards, promptable_segmentation `track`): portable float32 MPS
  goldens (`samurai_box_*` keys). The sam2 goldens were regenerated, so all sam2 MPS validation is needed again. Golden
  files in `golden_ports/`. No new weights.
- ByteTrack multi-object tracking (`detection.ByteTracker`): CPU numpy, nothing device-specific; optional Mac row of
  `tools/eval_mot.py track` (tracker ms per frame). No weights.
- Perception Encoder `g14`: the peak footprint (about 17.8 GB) is above the 16 GB of the machine, so the times
  include swapping.
- Metal kernels for the families that have none (the feature backbones except Perception Encoder, RTMDet,
  ConvNeXt, the SAM variants): not tried yet. Flow has them since 2026-11-11 (`flow_metal`, see the device page).

## Known gaps that wait for a release

- The `dpt_fused` build `48e0065bb7cf` for `sm_120` is not on Hugging Face yet; kernel builds are uploaded at
  release time.
- The `romav2` build `bb29839cee98` (fused depthwise conv) is not on Hugging Face yet; local copy in
  `wt-roma-dw/build/kernels` on the Ubuntu box.
- The `gsraster` build `be9d99ba7071` (ReSplat rasteriser + exact kNN) is not on Hugging Face yet; local copy in
  `wt-p39knn/build/kernels` on the Ubuntu box.
- Licence questions still open: the Depth Anything V2 Metric B / L weights and Distill-Any-Depth L.
