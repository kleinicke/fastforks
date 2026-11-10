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

1. RAFT TensorRT at 1080 x 1920 runs out of memory (4 failure rows). Depth Anything 3 `large-1.1`, `giant-1.1` and
   nested do not export to TensorRT (a data-dependent guard in the RoPE code).
2. `docs/data/experiments_resolution.jsonl`: the two scripts its `command` fields name were lost. Write them again
   under `tools/`, and check that a model loaded with `long_side=756` gives the same result as one whose
   `long_side` was changed after loading.
3. ConvNeXt: move the datacenter notes from the comments in its `card.toml` to `datacenter-tuning.md`. RTMDet and
   RTMPose each carry their own CSPNeXt; move it to `shared/`.
4. `docs/data/flow_eval.jsonl`: the RAFT rows still name the old model ids.
5. CLIP: `clip-l14-dfn2b`, `clip-h14-dfn5b` and `clip-h14-378-dfn5b` have no `torch.compile` or TensorRT rows in
   `docs/data/speed_compile.jsonl`.
6. Next networks, from [model-landscape.md](model-landscape.md): a ConvNeXt v1 feature backbone (nearly free),
   TIPSv2, MaxViT / RepViT, MobileSAM (licence not checked yet), the MoGe-2 extras, more flow and pose models.

## Waiting for the Mac

- ViTPose, RTMPose, DWPose, Perception Encoder: Metal kernels exist, Mac speed rows do not.
- ViTPose, RTMPose, SEA-RAFT, SigLIP 2: MPS goldens to be run again after the last changes.
- RAFT, SEA-RAFT, CLIP, C-RADIOv3, Depth Anything V2 Metric, Distill-Any-Depth, RTMDet, ConvNeXt: Mac speed rows.
- EfficientViT-SAM: only `l0` was checked on MPS, `portable` only.
- HQ-SAM: CUDA side checked again on 2026-11-10 (CPU and GPU goldens pass with `dpt_fused@48e0065bb7cf`, checks
  clean); nothing run on the Mac yet. Weights on the CUDA machine: `~/scratch_tests/evsam/w/sam_hq_vit_{b,l,h}.pth`.

## Known gaps that wait for a release

- The `dpt_fused` build `48e0065bb7cf` for `sm_120` is not on Hugging Face yet; kernel builds are uploaded at
  release time.
- Licence questions still open: the Depth Anything V2 Metric B / L weights and Distill-Any-Depth L.
