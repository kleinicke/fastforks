# Post-release work

## Measurement and implementation coverage

The November 19 release review covers 578 variants.

- [x] Default portable accuracy is measured for all 578 variants. The 39 gaps from the November 19 audit now
  have scores, including FalconsAI NSFW ViT on the official UnsafeBench Sexual test subset.
- [x] Original and portable FP32 accuracy use the same benchmark and sample count for 575 variants. Marigold V2
  and both SeedVR2 models use BF16 in their CUDA portable runners and have paired scores at that precision.
- [x] Compile timings cover all 578 variants. The measured stage is recorded for partial compilation.
- [ ] TensorRT timings cover 575 variants, with 54 selected timings from standalone builds. Scope labels identify
  staged runs and restricted inputs. DISK-LightGlue, XFeat-LighterGlue and RoMa v1 remain gated on paired accuracy checks.
- [x] Portable and kernels have a same-size timing comparison for all 578 variants in the local setup, including
  545 at the task's standard size. [Public CUDA binary inventory](missing-kernel-modes-0.2.1.md)
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
