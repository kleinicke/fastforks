# ComfyUI integration: usability audit

2026-11-23. All 20 `Fastforks*` nodes, run through the ComfyUI API on ComfyUI 0.46 (Mac, MPS, M1 Pro) with deliberate
mistakes; the message a user sees in the red node was recorded for each. Scope: `integrations/comfyui` (nodes.py,
convert.py, workflows), `docs/comfyui.md`, `tests/test_comfyui.py`. Nothing under `src/fastforks/stereo` or the speed
data was touched. The Windows / CUDA behaviour was not re-measured here.

## Fine as it was

- **Names, categories, widget order, defaults.** One category (`fastforks`), display names say the task and the pack,
  `model`, `backend`, `dtype` always in that order after the image inputs, prompts and thresholds after them, optional
  `batch_size` last. Defaults run on the example images without changes.
- **Outputs.** All outputs but Classify's were named. IMAGE outputs of one frame batch stay batches.
- **dtype / backend errors.** `bfloat16` on MPS: "depth_anything_v2 portable on mps cannot compute in bfloat16; dtype
  must be one of [...]". `backend=mlx` for a model without an MLX port names the fix (the only such published model,
  MatAnyone, is listed by no node).
- **Prompts.** No prompt at all, text sent to a points / box model (already reworded earlier), a box with x1 < x0 and
  points outside the image give clear library messages; so do PointTrack points outside the frame.
- **Frames.** Flow with one frame: "Flow needs a batch of at least 2 frames".
- **Faces.** FaceRestore on a non-aligned image: "GFPGAN restores an aligned 512 x 512 face crop ...". (Since 27343c97 the node detects, aligns and pastes back the faces of a whole photo, so this message no longer reaches a user.)
- **Memory in Upscale.** The library's "out of GPU memory for a 1600x1600 input in one pass: ... tile=512" is exact.
- **Weights.** Missing weights offline and gated repos: the library text already names the file, the licence page and
  `hf auth login`.
- **Odd sizes.** 8 x 8 and 3000 x 3000 inputs run (depth, segment).

## Changed

| # | Mistake | Before | Now |
|---|---------|--------|-----|
| 1 | A model of another node in `model` (saved workflow, linked input), e.g. SAM 2 in Depth | "... is not one of this node's N models (it needs inputs the node does not pass)" | "'sam2.1-hiera-small' is not a model of Depth: it is listed by Segment (SAM) and Video segment (...)"; a model no node runs says so |
| 2 | Unknown model id | the library's long "unknown model ... (no task has it; tasks: ...)" with ids of every task | "unknown model 'x' for Depth; did you mean depth-anything-v2-s, ...?" (this node's ids only) |
| 3 | Non-numeric points, malformed JSON points, bad box (Segment, Video segment, Track points) | "could not convert string to float: 'abc'", "Expecting ',' delimiter ..." | "positive_points: write points as 'x,y; x,y' in pixels (e.g. '120,80; 300,210'), got 'abc'" (names the input; also NaN / inf, JSON that is not a list of pairs) |
| 4 | Pose `detections` that is not a Detect node's JSON | raw JSON error, "0" or "'box'" | "detections: connect the 'detections' output of a Detect node (...), got '...'" |
| 5 | Detect, closed-vocabulary model, a class it does not have | silently an empty mask | "classes: unknown classes ['unicorn'] (this model has 80 classes), e.g. person, bicycle, ...; names are case-insensitive"; matching is now case-insensitive |
| 6 | Detect, Grounding DINO, empty `classes` | "text must be a non-empty list ..." although the tooltip promises the card's label set | runs with the card's label set (bug: `[]` was passed) |
| 7 | Semantic segmentation with an unknown class | all 150 ADE20K names in the message | "classes: unknown classes ['hatt'] (this model has 18 classes); did you mean Hat?" or the first 12 classes |
| 8 | Segment / Video segment with points or a box on CLIPSeg | "clipseg@1 takes text and / or exemplar input_boxes, not ['input_labels', 'input_points']" | "clipseg does not take points prompts (it takes text): clear positive_points and negative_points and write a text prompt" |
| 9 | Stereo, left and right of two sizes | "left and right must be batches of one size, got (1, 6, 8, 3) and (1, 6, 4, 3)" | "left is 8x6 but right is 4x6 pixels: a rectified pair has one size (...)"; batch counts have their own message |
| 10 | Allocation failure on MPS / MLX (`Invalid buffer size`, `[metal::malloc]`, `Insufficient Memory`) outside Upscale | the raw RuntimeError | `torch.OutOfMemoryError` with "use smaller images or fewer frames, a smaller batch_size or tile where the node has one, or try backend=mlx"; ComfyUI unloads its models for it |
| 11 | Offline mode with weights missing | library text only | plus "put the weights in ComfyUI/models/fastforks (or a `fastforks` folder of extra_model_paths.yaml), or unset FASTFORKS_OFFLINE / HF_HUB_OFFLINE and restart ComfyUI" |
| 12 | Gated Hugging Face repo (SAM 3, DINOv3-based models) | library text only | plus "run `hf auth login` with ComfyUI's Python environment active (or set HF_TOKEN before starting ComfyUI), restart ComfyUI" |
| 13 | `backend` tooltip said `kernels` fails off NVIDIA | wrong on Apple silicon (Metal kernels exist) | says CUDA or Metal where the model has them |
| 14 | Classify's output was called STRING | `STRING` | `classes` |
| 15 | `model` tooltip said nothing about downloads or gated access | licence text only | adds that the first run downloads weights (progress only in the console) and that some models (SAM 3) are gated |
| 16 | Segment `text` tooltip | "Concept models (SAM 3) only" | "Text models only (SAM 3, CLIPSeg)" |

The widget layout did not change (tooltips only; one output got a name), so the ten example workflows were not
regenerated. All ten ran to `success` on a second ComfyUI server with the edited pack (the branches of third-party packs
were left out: those packs are not installed on the Mac).

Every changed message has a test in `tests/test_comfyui.py` that needs no model weights.

## Behaviour that stays (documented in docs/comfyui.md)

- Interpolate, Video depth, Video segment and Track points run a single frame without complaint; Interpolate returns it
  unchanged. A clear error would be a nuisance in a workflow that sometimes gets one frame.
- Library errors that mention `device=` are not rewritten; the nodes have no `device` input and none of those reach a
  user on the default path.
- Mixed-size frames in a batch cannot reach a node (ComfyUI's batch nodes resize), so there is nothing to message.

## Needs a decision

1. **Keep loaded / unload control.** Models stay cached (two most recently used) and ComfyUI's global "Unload models"
   frees them; there is no per-node `keep_loaded`. Adding a BOOLEAN to 19 nodes changes every widget layout, so all ten
   workflows would be regenerated. Probably not worth it until someone asks.
2. **Device input.** Nodes use ComfyUI's device (`--cpu`, `--cuda-device`). A per-node CPU choice would be useful on
   small Macs but is also one more widget per node; ComfyUI's own flag already covers it.
3. **Seed.** There is no seed widget. Nearly all models are deterministic; the Marigold models (listed by Depth and
   Normals) are diffusion models with a `seed` load option (default 2026), so their output is fixed but cannot be
   varied from ComfyUI. A `seed` input would need to be shown only for those models, which ComfyUI's static widgets
   cannot do without a new widget on every Depth / Normals node.
4. **Gated models in the dropdown.** Only the licence is tagged (`[non-commercial]`). Tagging gated ones (`[gated]`)
   would warn before the run, but the gated state is not in the cards: it is only known from the download error. A
   `gated` flag on the card would give the tag for free.
5. **Download progress.** Large first-run downloads show progress only in the console; ComfyUI's progress bar would need
   the library to report bytes to a callback.
6. **Mac benchmarks in the tooltips.** `backend=mlx` is described qualitatively ("often leaner, first call slow").
   Per-model numbers live in `docs/plans/mlx-sweep.md` and the speed data, which this audit did not touch.
