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
  and the rows below exist.
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

1. ConvNeXt: move the datacenter notes from the comments in its `card.toml` to `datacenter-tuning.md`. RTMDet and
   RTMPose each carry their own CSPNeXt; move it to `shared/`.
2. `docs/data/flow_eval.jsonl`: the RAFT rows still name the old model ids.
3. CLIP: `clip-l14-dfn2b`, `clip-h14-dfn5b` and `clip-h14-378-dfn5b` have no `torch.compile` or TensorRT rows in
   `docs/data/speed_compile.jsonl`.
4. Next networks, from [model-landscape.md](model-landscape.md): a ConvNeXt v1 feature backbone (nearly free),
   TIPSv2, MaxViT / RepViT, MobileSAM (licence not checked yet), the MoGe-2 extras, more flow and pose models.

## Waiting for the Mac

- RTMPose zoo (7 Halpe26, hand5, 3 face6, AP-10K): MPS golden checks and portable speed rows.
  COCO accuracy covers Halpe26 body joints; hand, face and animal validation datasets are not staged.
  Weights on CUDA in `~/scratch_tests/port-rtmo/w/`.

- RTMW whole-body (5 variants): MPS golden checks, especially the 384-pixel variants, and portable speed rows.
  Weights on CUDA in `~/scratch_tests/port-rtmo/w/` or its `cache/`, otherwise the pinned upstream URLs.

- RTMO (4 bottom-up pose variants): MPS goldens, person counts near the score threshold, and portable speed.
  Weights in `~/scratch_tests/port-rtmo/w/` or its `cache/` on CUDA, otherwise the pinned upstream URLs.
- TinyViT, five classifiers: run portable fp32 MPS goldens and speed at native 224/384/512 crops. CUDA GPU goldens
  passed for portable and kernels; weights are downloadable from the pinned timm Hugging Face URLs in the card.

Done on 2026-11-10 for every family on the branch: Mac speed rows (`portable`, and `kernels` where Metal kernels
exist) and the MPS goldens, all passing. Still open:

- TIPSv2 `b14`, `l14` (new family, features): MPS goldens (cpu_atol 6.9e-05, gpu mean 0.019 / p99 0.069) and
  portable fp32/fp16 speed rows. Weights from the pinned HF URLs.
- MobileSAM `mobile-sam` (new, promptable segmentation; new `shared/sam_prompt`): MPS golden and portable speed.
  Weights `~/scratch_tests/port-mobilesam/weights/mobile_sam.pt` on the CUDA machine, or the pinned URL.
- RepViT, 10 variants (new family, classification; changes shared `_optimize/fold.py` reparam and
  `_core/deferred_init.py`): portable fp32 MPS goldens of all 10, check that fused weights load through
  `deferred_init`. Weights `~/scratch_tests/port-repvit/weights/` or the pinned timm URLs.
- ConvNeXt features `convnext_{tiny,small,base,large}.fb_in22k` (new family; the kernels trunk moved to
  `shared/convnext_trunk`): portable fp32 MPS goldens of the 4 refs, `tests/test_convnext_features.py::test_layout_cpu`,
  and classification ConvNeXt on MPS after the move. Weights `~/scratch_tests/port-convnext1/w/` or the pinned URLs.
- Perception Encoder `g14`: the peak footprint (about 17.8 GB) is above the 16 GB of the machine, so the times
  include swapping.
- SEA-RAFT `spring` variants run about ten times faster than the others at the same labelled input; not looked into.
- Metal kernels for the families that have none (flow, the feature backbones except Perception Encoder, RTMDet,
  ConvNeXt, the SAM variants): not tried yet.

## Known gaps that wait for a release

- The `dpt_fused` build `48e0065bb7cf` for `sm_120` is not on Hugging Face yet; kernel builds are uploaded at
  release time.
- Licence questions still open: the Depth Anything V2 Metric B / L weights and Distill-Any-Depth L.
