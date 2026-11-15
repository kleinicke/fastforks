# Remaining gaps (2026-11-15)

Source: `results/raw/2026-11-15_gaps/gapmatrix.py` on docs/data/speed*.jsonl after the gapfill queue; matrix `gaps.csv`.

Totals: 573 refs. Missing: upstream 14, upstream-default 164 (only refs with an upstream-default tier), portable 16, kernels 96 (refs without custom kernels run the portable path), compile 67, tensorrt 72. Errors (documented `unsupported:` rows): compile 9+1 OOM, tensorrt 54+2 OOM.

## Missing, by reason

### portable: no row (see per-model notes below) (16)

aliked-flexglue, any2full-b, any2full-l, any2full-s, depthfm-v1, lingbot-depth, lingbot-depth-dc, lingbot-map, loma-b, memfof-sintel, memfof-spring, omni-dc, resplat-base-re10k, resplat-small-dl3dv, tapnext-pp, video-depth-anything-metric-s

### compile: no row (see per-model notes below) (55)

aliked-flexglue, any2full-b, any2full-l, any2full-s, anyline, big-lama, c-radio-v4-h, c-radio-v4-so400m, clipseg, cotracker3-offline, cotracker3-online, depthfm-v1, ema-vfi, ema-vfi-small, ema-vfi-small-t, ema-vfi-t, film, gimm-vfi-r, gimm-vfi-r-p, hed, insid3-vitl16, lineart, lineart-anime, lineart-coarse, lingbot-depth, lingbot-depth-dc, lingbot-map, loma-b, marigold-v2-depth, matanyone, megaflow, memfof-sintel, memfof-spring, mlsd, mvsplat-re10k, omni-dc, pi3, pidinet, pointdit-l-512, prompt-depth-anything-l, prompt-depth-anything-s, recover-cx, recover-mn, recover-rn, resplat-base-re10k, resplat-small-dl3dv, retinaface-mobilenet0.25, retinaface-resnet50, swin2sr-x4, tapnext-pp, teed, video-depth-anything-metric-s, waft-a1, waft-a1-kitti, waft-a1-sintel

### compile: weights not cached (downloads stopped at the 30 GB free-disk floor) (12)

bridgedepth-rvc-pretrain, clip-h14-378-dfn5b, clip-h14-dfn5b, clip-l14-dfn2b, dinov2-vitg14-reg, dinov2-vitl14-reg, dinov3-vith16plus, dinov3-vitl16, fast-acvnet-plus-kitti15, foundationstereo-vit-l, moge-3-vitl, s2m2-m

### tensorrt: no row (see per-model notes below) (61)

aliked-flexglue, any2full-b, any2full-l, any2full-s, anyline, big-lama, c-radio-v4-h, c-radio-v4-so400m, clipseg, cotracker3-offline, cotracker3-online, dat-2-x4, dat-light-x4, dat-x4, depthfm-v1, drct-l-x4, drct-x4, ema-vfi, ema-vfi-small, ema-vfi-small-t, ema-vfi-t, film, gimm-vfi-r, gimm-vfi-r-p, hat-s-x4, hat-x4, hed, insid3-vitl16, lineart, lineart-anime, lineart-coarse, lingbot-depth, lingbot-depth-dc, lingbot-map, loma-b, matanyone, megaflow, memfof-sintel, memfof-spring, mlsd, mvsplat-re10k, omni-dc, pi3, pidinet, pointdit-l-512, prompt-depth-anything-l, prompt-depth-anything-s, real-hat-gan-x4, recover-cx, recover-mn, recover-rn, resplat-base-re10k, resplat-small-dl3dv, retinaface-mobilenet0.25, retinaface-resnet50, tapnext-pp, teed, video-depth-anything-metric-s, waft-a1, waft-a1-kitti, waft-a1-sintel

### tensorrt: weights not cached (downloads stopped at the 30 GB free-disk floor) (11)

bridgedepth-rvc-pretrain, clip-h14-378-dfn5b, clip-h14-dfn5b, clip-l14-dfn2b, dinov2-vitg14-reg, dinov2-vitl14-reg, dinov3-vith16plus, dinov3-vitl16, foundationstereo-vit-l, moge-3-vitl, s2m2-m

## compile: errors (recorded as unsupported rows)

- error: worker rc=124: mocha-v1-sceneflow, romav1, track-on-r, track-on2
- error: timeout after 290 s: falconsai-nsfw-vit, seedvr2-3b
- error: no exportable module (multi-network sliding-window pipeline): totalsegmentator-total, totalsegmentator-total-3mm
- error: run: TypeError: cannot unpack non-iterable NoneType object: nninteractive
- OOM: seedvr2-7b

## tensorrt: errors (recorded as unsupported rows)

- error: BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E: dedode-v2-g, gmflow-scale2-mixdata, gmflow-scale2-regrefine6-kitti, gmflow-scale2-regrefine6-mixdata, gmflow-scale2-regrefine6-sintelft, gmflow-scale2-regrefine6-things, gmflow-scale2-things
- error: BackendUnavailableError: TensorRT engine build failed: [defaultAllocator.cpp::allocate::62: memflow, memflow-kitti, memflow-sintel, raft-large, raft-large-kitti, raft-large-things
- error: TorchExportError: Failed to export the model with torch.export. [96mThis is step 1/3[0m : alltracker, alltracker-tiny, disk-lightglue, loftr-indoor, xfeat-lighterglue
- error: ConversionError: Failed to convert the exported program to an ONNX model. [96mThis is ste: grounding-dino-base, mm-grounding-dino-base, mm-grounding-dino-tiny, romav1
- error: timeout after 290 s: seedvr2-7b, segformer-b5-face-parsing, wd-eva02-large-tagger-v3, wd-vit-tagger-v3
- error: unsupported: engine build exceeds 5 min lock rule (cold ONNX export + build still running : great-igev-middlebury, mocha-v1-sceneflow, tapnext
- error: ValueError: not enough values to unpack (expected 4, got 2): locotrack-b, locotrack-s, track-on-r
- error: unsupported: TensorRT 10 Myelin tactic check fails during the engine build: bridgedepth-sf, defom-stereo-vitl-sceneflow
- OOM: memfof-kitti, raft-small
- error: unsupported: fp16 engine output is all NaN (strongly-typed fp16 overflow); no fp32 island : mvsanywhere, waft-stereo-b
- error: no exportable module (multi-network sliding-window pipeline; onnx was also missing): totalsegmentator-total, totalsegmentator-total-3mm
- error: unsupported: torch.export: data-dependent guard (masks.sum() == 0 branch): aliked-lightglue
- error: ConversionError: Failed to decompose the FX graph for ONNX compatibility. [96mThis is ste: anycalib-pinhole
- error: unsupported: TensorRT ITopK supports K <= 3840 only; the keypoint top-k exceeds it: dedode-v2-b
- error: unsupported: torch.export: data-dependent guard ((norm_x == 0).any()): geocalib-pinhole
- error: unsupported: TensorRT 10 Myelin compiler error CHECK(is_const()) on the attention graph: gmflow-scale2-sintel
- error: unsupported: ONNX export: aten._is_all_true has no ONNX decomposition: grounding-dino-tiny
- error: unsupported: TensorRT 10 Myelin autotuner exception (all tactics skipped): las2-h
- error: unsupported: torch.export: data-dependent guard in loftr.py:190: loftr-outdoor
- error: unsupported: 20 B NF4 transformer, weights expanded per linear at each call; no ONNX / eng: marigold-v2-depth
- error: RuntimeError: required rank 5 tensor to use channels_last_3d format: nnfoundation-cnn
- error: RuntimeError: Calculated padded input size per channel: (3 x 192 x 192). Kernel size: (8 x: nnfoundation-vit
- error: TypeError: 'NoneType' object is not subscriptable: nninteractive
- error: BackendUnavailableError: onnx is not installed: pip install 'fastforks-core[onnx]' (or 'pi: pointdit-b-512
- error: not run: onnx is not installed in the test venv: seedvr2-3b
- error: unsupported: torch.export: data-dependent guard in model.py:107: track-on2
- error: unsupported: torch.export: data-dependent guard in the NMS / keypoint selection: xfeat

## Notes

- Cold TRT engine builds over the 5 min lock rule: great-igev-middlebury, mocha-v1-sceneflow, tapnext (timed out at 295 s, recorded).
- fp16 all-NaN engines: waft-stereo-b, mvsanywhere. defom-stereo-vitl-sceneflow: TensorRT Myelin tactic error.
- dfine-x-coco TRT fixed (fp32 grid_sample island under export, 5.033 ms). c-foundationstereo-s compile fixed (`math.lcm`; cold 626 s as 3 slices).
- timm classification refs without kernels row: kernels path = portable (no custom kernels), nothing to measure.
- tapnext-pp original: weights `tapnextpp_ckpt.pt` not cached (download stopped at the disk floor).
- VDA metric-s: no adapter; no upstream clone in `upstream/` and no downloads possible at the disk floor.
- memfof-kitti/-sintel originals measured; 1080x1920 fp32 OOM on 16 GB for kitti/sintel (recorded).
- Queue tier for the solve queue stayed verylow (a change to a higher tier was denied by permissions); starved steps were rerun at plain gpulock.
