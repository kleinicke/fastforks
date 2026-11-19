# Post-release work

## Measurement and implementation coverage

The November 19 release review covers 578 variants.

- [ ] Default portable accuracy is measured for 575 variants. Of the 39 gaps found in the portable accuracy audit,
  36 have a score. PiDiNet and TEED are being scored on the same 75 BSDS500 images as the FP32 comparisons.
  FalconsAI NSFW ViT is waiting for access to the labeled UnsafeBench test split.
- [x] Original and portable FP32 accuracy are each measured for 574 variants. Marigold V2 and both SeedVR2 models
  use BF16 in their CUDA portable runners and have scores at that precision. FalconsAI is the remaining labeled
  evaluation to run in FP32.
- [x] Compile timings cover all 578 variants. The measured stage is recorded for partial compilation.
- [ ] TensorRT timings cover 575 variants, including 54 whose selected measurement is for a standalone stage. Three
  configurations remain gated on paired accuracy checks: DISK-LightGlue, XFeat-LighterGlue and RoMa v1.
- [x] Kernels timings cover all 578 variants in the local setup. Public CUDA binary availability is tracked separately
  in [the kernel inventory](missing-kernel-modes-0.2.1.md).
- [x] Each of the 20 task categories has a labeled comparison for kernels, compile and TensorRT.

Coverage counts use a successful row at any reported resolution or dataset. Keep each comparison tied to its
measured shape, dataset and backend scope.

## Website model examples

- [x] Generate or restore examples for all 578 variants. The coverage is 477 rendered outputs and 101 classifier
  predictions displayed as text.
- [x] Build the website review branch and verify all 501 distinct referenced assets against its source assets.
  The assets and generated metadata are available on the
  [website review branch](https://github.com/kleinicke/fastforks-website/tree/complete-model-examples).
- [ ] Deploy the review build and verify the production pages.

`tools/examples.py` writes example metadata to `docs/data/examples.jsonl`. It skips examples with successful metadata
even if the corresponding local file is missing. Restore the asset or explicitly regenerate that example in this case.
