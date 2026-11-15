# Remaining gaps (2026-11-15)

Source: `results/raw/2026-11-15_gaps/gapmatrix.py` on docs/data/speed*.jsonl after the gapfill queue; matrix `gaps.csv`.

Totals: 573 refs. Missing: upstream 14, upstream-default 164 (only refs with an upstream-default tier), portable 16, kernels 96 (refs without custom kernels run the portable path), compile 67, tensorrt 72. Errors (documented `unsupported:` rows): compile 9+1 OOM, tensorrt 54+2 OOM.

Round 2 final (gapmatrix rerun 2026-11-15 ~15:25, all queues finished): portable ok 570, OOM 2 (memfof-sintel at 1080x1920, seedvr2-7b), missing 1 (resplat-base-re10k: weights are public and cached, the earlier "gated" claim was wrong; its portable bootstrap only timed out at the 175 s lock limit at 7.1 s per call, measured 2026-11-15 with a 290 s limit: see results/2026-11-15_original_gaps.md). kernels missing 95 (refs without a kernels runner, incl. the six timm families below; not a measurement gap). compile missing 12, tensorrt missing 13 (see reasons below). New compared with the first matrix: 16 portable, ~45 compile, ~45 TensorRT rows, VDA metric-s video rows, tapnext-pp original.

### Still missing after round 2, reason per ref
- Weights not cached, would drop `/` below the 30 GB floor or no usable URL: bridgedepth-rvc-pretrain, clip-h14-378-dfn5b, clip-h14-dfn5b, clip-l14-dfn2b, foundationstereo-vit-l, megaflow, pi3, pointdit-l-512, prompt-depth-anything-l, s2m2-m (compile + tensorrt); pe-core-g14-448, pe-spatial-g14-448 (7 GB each, not downloaded). Gated: resplat-base-re10k.
- video-depth-anything-metric-s compile / tensorrt: video pipeline (4 sliding windows, temporal head), no exportable single module and no compile adapter; portable / kernels / upstream rows exist.
- tapnext-pp tensorrt: stateful online tracker, same limitation as tapnext (cold export + build over the 5 min lock rule); compile fails (InductorError, recorded).
- memfof-sintel portable 1080x1920: OOM on 16 GB (as memfof-kitti); 375x1242 row exists.

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
- timm classification refs without a kernels row (coatnet, davit, focalnet, hgnetv2, efficientformer, efficientformerv2, shvit): NOT by design. Their `model.py` registers only `RUNNERS = {"portable": libnet.Portable}`; the cards say `[modes.kernels] status = "none"` ("not implemented / not built yet, needs its own runner"), not "kernels == portable". The generic `libnet.Kernels` runner (fp16-accumulate convs, fused epilogues, attention, CUDA graph replay) is registered by the other classification families and is a candidate for these six (register `"kernels": libnet.Kernels`, then goldens + a kernels row). Candidates for the kernels agent: channels-last fp16 conv_ep for hgnetv2 / efficientformer / shvit (conv-BN-act stacks), window / relative-bias attention for coatnet, davit, focalnet. Not done here (kernel work, would change goldens and the speed floor).
- tapnext-pp original measured (round 2, weights cached): 1492.8 ms per online frame, fp32, 1527 MB peak (`rows_original2.jsonl`); portable 528.7 ms / 1220 MB. compile fails (InductorError, recorded), TensorRT not run.
- VDA metric-s: upstream clone `upstream/Video-Depth-Anything` at pin 4f5ae23; video protocol identical to the VDA-s rows (88 frames, 4 windows, ms per video; `solve/vda_speed.py`). portable 480x640 1047.9 ms / 2921 MB, 1080x1920 2004.6 ms / 8203 MB; kernels 952.3 ms / 2923 MB, 1080x1920 1847.6 ms / 8208 MB; upstream fp32 11337 ms / 10522 MB at 1080x1920, upstream16 4012 ms (480x640) / 10776 ms (1080x1920). The first upstream 480x640 row (13362 ms) is a contention outlier, re-measured in `rows_vda_rerun.jsonl`. The earlier single-image bootstrap rows for this ref are excluded (not comparable with the video rows).
- loma-b measured (portable 308.1 ms / 1767 MB, compile 285.4 ms / 2870 MB; TensorRT export error `negative dimension -1`, recorded).
- Disk: see `results/2026-11-15_disk_moves.md` (hard-link dedup, `/` 28 GB -> 107 GB; cache fills push it back to ~30 GB). Weights still not cached because they would drop `/` below the 30 GB floor: pe-core-g14-448 and pe-spatial-g14-448 (7.2 / 7.1 GB). Others in `weights_not_cached.txt` are gated or have no usable URL.
- Verylow starvation: other workstreams hold gpulock-low / -verylow queues; the round-2 queues (`gapfill-m2`, `gapfill-m3`, `gapfill-m4`) waited up to > 45 min for a single step. They keep running as capped units; re-run `python3 results/raw/2026-11-15_gaps/solve/finalize2.py` (idempotent, new keys only) + the targeted tests to publish what they produce.
- /tmp is a 16 GB tmpfs: stale `ff-trt-*` ONNX dirs from killed runs filled it (rows failed with `Disk quota exceeded`); stale dirs > 2 h removed, those steps re-queued in `master3.sh` with TMPDIR on the SSD. Step timeouts of 190 s in the bootstrap queue counted the lock wait: retried with --timeout 7000 in `master4.sh`.
- memfof-kitti/-sintel originals measured; 1080x1920 fp32 OOM on 16 GB for kitti/sintel (recorded).
- Queue tier for the solve queue stayed verylow (a change to a higher tier was denied by permissions); starved steps were rerun at plain gpulock.
