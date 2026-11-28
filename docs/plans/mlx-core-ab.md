# MLX core A/B: hand-written networks against the graph translation (2026-11-26)

Every hand-ported ref (the families with `GRAPH_RUNNERS`, and the libnet classifiers written out in `mlx_cnn` /
`mlx_vit_cls`), portable mode, float16 and float32, on the network alone: `tools/mlx_core_ab.py` (how-to in
[mlx-coverage-gaps.md](mlx-coverage-gaps.md#how-to-ab-a-hand-written-mlx-network-against-its-graph-translation-2026-11-26)).
M1 Pro 16 GB, MLX 0.32.4, buffer cache 2048 MB in both arms, a throwaway warm-up process before each
arm (warm conversion cache), 4 warm-up calls then the median of 5. Rows: [mlx-core-ab.jsonl](mlx-core-ab.jsonl).
First measured as a diagnostic sweep; the families touched by parity round 1 (below) re-measured after it.

**Bar**: the hand network at most 2 % slower (warm) and at most 2 % heavier (network peak) than the exported-graph
translation it replaced. 583 rows over 80 families; 579 measured in both arms.

- After parity round 2: 242 rows flagged (95 slower, 159 heavier, 10 `worse` in the new quality column; 236 on
  speed or peak alone). Median hand / graph: warm 0.90, peak 0.92. After round 1: 277 (118 slower, 189 heavier;
  the sweep: 325), medians 0.91 / 0.93.
- 75 of the flagged rows are accepted as a peak trade (clearly faster, see
  [Accepted: faster, peak trade](#accepted-faster-peak-trade)); 167 rows still miss the bar (round 1: 72 / 205).
- Beyond 10 % (same number of network calls in both arms): 64 rows (round 1: 122), listed first below.
- 42 flagged rows time a different number of network calls per model call in the two arms (the hand port
  splits or merges the network differently): their warm sums compare, their peaks (the largest single call) less so.

**Parity round 1 (2026-11-26).** Systematic peak causes in the shared helpers, each fixed once and re-measured over the
families that use it (those families' rows re-run after the fixes; the other families keep their sweep rows):

- *Unfused attention head sizes.* `mx.fast.scaled_dot_product_attention` fuses only head sizes 64, 72, 80, 96, 128;
  any other materialises the float32 scores (SegFormer b0 stage 1: 16384 x 256 per head). `_core.mlx.attention`,
  shared by the graph translation's `_sdpa` and the hand networks, zero-pads a smaller head to the next fused size
  where the scores outweigh the padded copies (`lq * lk > 2 * fused * (lq + lk)`: 12 x 12 Swin windows stay unpadded;
  padding them was heavier and in float32 slower) and runs a larger one in query blocks. Exact (the padded columns
  are zero); a no-op for head size 64 (DINOv2 / v3, SAM, C-RADIO in `mlx_vit`). `mlx_vit`'s float16 blocks keep
  MLX's own kernel: there the padded one is closer to float32 per call (LW-DETR-M's global blocks: max error 0.002
  against 0.004-0.02), yet LW-DETR-M float16 then lost its golden's 0.91 box. SegFormer b0 float16 346 -> 158 MB
  (32.9 -> 19.2 ms), OneFormer float16 1746 -> 1364 MB (float32 2270 -> 1894).
- *Float32 resize taps.* The segmentation heads' bilinear resize built float32 tap weights, promoting float16 maps to
  float32. `mlx_seg.resize` now runs the graph translation's one-pass NHWC kernel (`ops.bilinear_nhwc`) in the map's
  dtype: HRNet-W48 float16 534 -> 313 MB (180 -> 153 ms), DeepLabV3+ float16 860 -> 773 MB, part of the SegFormer /
  SegNeXt gains. SegNeXt also evaluates its weight casts once at load (a lazy `astype` kept the float32 source
  resident and recomputed it in every call).
- *One command buffer per network.* MLX returns a command buffer's transients only when it completes, so one compiled
  function over a whole classifier holds every block's intermediates to the end, where the graph translation
  evaluates every 256 MB. `mlx_cnn._segmented` compiles the top-level ops in 16 segments of about equal output bytes
  (at least 3 ops each) and evaluates them one after the other: the libnet classifiers' peaks fall 10-40 %
  (ConvNeXt-B float32 475 -> 378 MB, CoAtNet-2 float32 552 -> 424 MB, FocalNet-B float32 544 -> 419 MB,
  EfficientFormerV2-L float16 178 -> 69 MB) at equal warm time (1 against 16 segments on ShViT-S4, MobileNetV3-L,
  MobileNetV4-M, HGNetV2-B0: within noise).
- LoFTR: the dense pass staged and its dual softmax run in blocks.

**Measurement drift.** The machine ran 15-35 % slower in both arms than during the sweep on the large networks
(Mask2Former float16: today's tree and the round's base `07223ce4` both 585-595 ms in the same session, against 425 ms
in the sweep; FCN-R101, which this round did not touch: 277 -> 326 ms). The hand / graph ratios come from
back-to-back arms and hold; before / after absolute times of the large networks do not. FCN's float16 peak moved
638 -> 877 MB without a code change, so peak differences under about 15 % on the 300 ms+ networks are within noise.

**End to end** (the MLX survey of the 127 touched refs, run twice, the better of the two against the rows before the
round; 254 rows, every golden passing): median warm 0.99, peak 0.96, first call 1.05 (first calls of 0.02-0.1 s:
noise). No peak grew beyond 5 % (FocalNet-B 1222 -> 632 MB, HGNetV2-B6 1289 -> 791, LoFTR 2129 -> 1661,
OneFormer 2587 -> 2144). Warm times beyond 5 % slower (up to 1.1x, MobileNetV1 float32 2.5 -> 3.2 ms) are the drift
above: their network A/B rows are equal or faster than before, and untouched networks slowed alike.

**Still flagged, with the reason** (beyond the accepted peak trades):

- SegFormer, SegNeXt: still 1.1-1.8x the graph's peak and 5-10 % slower after the attention and resize fixes; the
  rest was not located this round.
- RF-DETR (float16 5-9 % slower, float32 1-5 %, as in the sweep) and rf-detr-keypoint (peak in a post-backbone stage): unchanged by the shared fixes.
- Selective-IGEV float32: 25 % faster, but the peak (the 3 x 3 conv workspace of the GRU stage, about 850 MB
  transient) puts the e2e peak at or just above the best MPS row, so not accepted. A graph fallback was rejected: the
  graph is slower warm and its first call takes 8.7 s.
- Sapiens float32, TotalSegmentator: faster, but the e2e peak is above the best MPS row.
- ConvNeXt tiny / small, DaViT, ShViT, DeiT / ViT, RF-DETR: 2-19 % slower, as in the sweep; the segments did not
  change it (ShViT measured unsegmented: the same).
- TinyViT 384 / 512, CoAtNet-RMLP-2 384 float32, ConvNeXt-L 384 float32, wd-taggers, Swin-B float32: heavier, not
  clearly faster; the ratio grows with the crop (TinyViT-21M float32 0.73 at 224, 1.12 at 384, 1.30 at 512).
- LoFTR, XFeat, the R-CNN heads: A/B artifacts in round 1, like-for-like since round 2 (below).

Not reached in round 1: TinyViT's large crops, rcnn, raft_stereo, nafnet, rtmpose's float32 growth, the classic small
CNNs (shufflenetv2, squeezenet), mobilenetv1, lw-detr large / xlarge float16 (all but rtmpose taken up in round 2).

## Parity round 2 (2026-11-27)

The round-1 flagged families plus rcnn, raft_stereo, nafnet, shufflenetv2, squeezenet and mobilenetv1; every
touched family's goldens (float16 and float32) and `tests/test_mlx.py` match tests re-run after each change, the
A/B rows of the touched refs re-measured (arms back to back, suspicious rows repeated).

**Like-for-like scope.** `tools/mlx_core_ab.py` now times, per family in `SCOPE`, only the hand network's part that
the graph translation covers: R-CNN `Net.dense` (backbone, FPN, RPN, without the box and mask heads), XFeat
`MlxDense.__call__` (without LightGlue), LoFTR `MlxLoFTR._dense` (without the fine refinement). XFeat +
LightGlue 4.07x / 2.77x (warm / peak) was the artifact: like-for-like it is 0.51 / 1.24.

**Fixes, per family** (hand / graph, warm / network peak, before -> after the round):

- *Evaluation boundaries in long compiled functions* (the round-1 cause, applied where it was missing):
  - R-CNN: the dense pass as compiled steps, peak 1.72 -> 0.88-0.97 at warm 0.82-0.85; the runner requests a
    512 MB (float16) / 1024 MB (float32) buffer cache (below).
  - `mlx_vit.Stepped` (the timm ViT classifiers, RF-DETR's backbone, its decoder layers one in flight):
    ViT-B / DeiT-B peak 1.04-1.08 -> 0.80-0.87; rf-detr-keypoint 1.57 -> 1.09 (fp16),
    1.71 -> 1.28 (fp32); wd-vit-tagger fp32 1.29 -> 1.01; rf-detr m / n and seg-l / m / xl fp32 under the bar.
  - RAFT-Stereo: five-part iteration, the trunk split: fp16 1.40 -> 1.20 at warm 0.78-0.80, fp32 1.57 -> 1.06-1.19.
  - Lineart: stepped parts and the cached load path (recorded under the run dtype): fp16 peak 1.00 at warm 0.70-0.75.
  - NAFNet: the graph translation's row layer norm, large maps run as two half blocks: sidd fp16
    1.06 / 1.39 -> 0.89 / 1.21, gopro fp16 0.87; all rows faster. The cached load path is not adopted here: the builder runs on the portable-prepared, folded network, not on the checkpoint.
- *Depthwise kernels*: `mlx_cnn` ConvNeXt blocks and ShuffleNetV2 on `_conv_k`, ConvNeXt's LayerNorm over rows
  (opt-in) and its GELU in float32: ConvNeXt 15 flagged rows -> 2, convnextv2 1 -> 0, ShuffleNetV2 fp16 1.39 ->
  1.02-1.08. DaViT's `_cpe` depthwise kernel.
- *TinyViT*: a 32 MB evaluation budget per segment (`SYNC_FAMILIES`, TinyViT only: elsewhere it cost time):
  512 fp32 1.30 -> 1.00; 384 fp32 1.12 heavier -> 0.87 peak but 5-6 % slower.
- *SegFormer, SegNeXt*: large steps wait for themselves, SegFormer's head sum drops its references, SegNeXt's head
  in three parts: SegFormer b2 1.42 -> 1.15 / 1.53 -> 1.09, b5-ade fp16 1.28 ->
  0.74, fp32 1.81 -> 1.03; SegNeXt-T 1.41 -> 0.67 / 1.61 -> 1.01, SegNeXt-S fp32 1.30 -> 0.75.

Tried and reverted (no gain): evaluating the stereo features before the matching, a per-block evaluation in
`_mobile`, TotalSegmentator's volume step.

**The A/B's buffer cache hides the cost of evaluation boundaries.** Both arms run with a 2 GB cache, so a stepped
network's freed transients stay cached between its parts. The survey runs each port with its own request (the 128 MB
base where it asks for none), and there the transients go back to the system and are allocated again. Three round-2
changes won the A/B but lost end to end, and were reverted after the survey (MLX survey, warm ms / footprint MB):

| change | before the round | stepped, own cache | stepped, 1 GB | stepped, 2 GB |
|---|---|---|---|---|
| IGEV frame, four-part update: igev-stereo-sceneflow fp16 | 3290 / 2568 | 4591 / 2269 | 3508 / 2705 | 3261 / 3696 |
| Selective-IGEV's four parts: selective-igev-sceneflow fp16 | 3751 / 2779 | 4954 / 2280 | 4025 / 2720 | 3772 / 3706 |
| LW-DETR's size-gated evaluations: lw-detr-large fp16 | 56.7 / 1000 | 65.0 / 924 | 58.9 / 1082 | 58.8 / 1083 |

With the larger cache the speed returns but the cached buffers add more to the footprint than the net peak saved, so
IGEV-Stereo / Selective-IGEV (net peak 1.67 / 1.70x the graph's again, at 0.78-0.90 warm) and LW-DETR large / xlarge
(1.67 / 1.57x fp16) are back to round 1's fused graphs. R-CNN keeps its steps with a cache request: at 512 MB float16
runs 124 ms in 1148 MB (before 126 / 1156, at the base cache 136 / 922); float32 needs 1024 MB (145 / 1823 against
156 / 1576; 175 ms at 512). Lineart, RAFT-Stereo (1536 MB request) and NAFNet (512 MB) stay faster than the graph at
their own cache (Lineart at 128 MB: 0.60 / 0.79 warm). SegNeXt float32 and TinyViT 384 float32 are the remaining
cases: at their own cache they run 7-16 % slower end to end than before the round, for a 8-26 % smaller footprint; a
512 MB cache brings the speed back above the old footprint (segnext-mscan-s fp32 86 ms / 1052 MB against 85 / 978).

**Still flagged, with the reason:**

- Stereo: IGEV-Stereo and Selective-IGEV 1.5-1.7x the graph's net peak at 0.76-0.90 warm (the fused update, see
  above); IGEV++ 1.10-1.13x (its cost volumes); RAFT-Stereo fp16 1.20x at about 0.79 warm.
- LoFTR fp32 (1.64x peak at warm 0.29-0.36), XFeat (1.21-1.24x at 0.51-0.55), NAFNet fp32 (1.32-1.39x, faster),
  Lineart fp32 (1.32x at 0.84-0.94), TotalSegmentator (1.15-1.28x at 0.72-0.76): faster, the peak a trade (accepted
  where the e2e peak allows, see the generated list).
- SegFormer, SegNeXt float16: 8-24 % slower everywhere, b0 1.39-1.48x the peak, SegNeXt-L fp16 1.38x: the cost is
  in the attention and MLP kernels at float16, not located this round.
- ViT / DeiT / falconsai, siglip, wd-vit-large float16, DaViT tiny / small, RF-DETR s and seg-* float16: 2-17 %
  slower (DaViT tiny fp16 1.17), leaner; these move 5-10 % between runs in both arms.
- TinyViT 512 fp16 1.11x, ConvNeXt-L 384 fp32 1.04x, RF-DETR l 1.03-1.16x, seg-2xl fp16 1.41x (fp32 1.33 ->
  1.11): heavier, not clearly faster. LW-DETR large / xlarge float16 1.67 / 1.57x (fused again, see above).
- MobileNetV1, ShuffleNetV2, SqueezeNet: networks of 2-4 ms whose rows are bimodal in both arms (MobileNetV1's code
  is unchanged); their flags are noise.

**Quality column.** Every row measured in both arms has a golden error in both (200 rows); 10 hand arms are further
from the golden than the graph's (`worse`): convnext_tiny in22k fp16, efficientformer l1 / l3 / l7 fp16, mobilenetv1
fp16, raft-stereo-middlebury fp16, rf-detr-seg-2xl / seg-m fp16, rf-detr-seg-xl fp32, shufflenet fp16. All are within
their golden bounds (the goldens pass).

**Drift.** As in round 1, absolute times move between sessions: IGEV++'s graph arm fp16 ran 6998, 8080 and 6233 ms in
three runs of the same tree; the 2-4 ms networks are bimodal. Ratios come from back-to-back arms.

**End to end.** The MLX survey (`tools/speed_survey.py`, each port at its own cache request) re-run over the 74
surveyed refs of the touched families against their rows from before the round, the better of two passes per ref:
148 rows, every golden passes, median after / before warm 0.99, footprint 0.99, first call 1.01. Footprints down by
up to 41 % (falconsai fp16 677 -> 398 MB; SegNeXt-s fp32 0.77, EVA02-L and SigLIP-so400m fp16 0.81). Slower by more
than 5 %: SegNeXt s / b / l fp32 1.12 / 1.10 / 1.06 and TinyViT 21m 384 fp32 1.09 (the trades above), LW-DETR large
fp16 1.06 (fused again; within the drift seen on unchanged code). Heavier by more than 5 %: Faster / Mask R-CNN fp32
1.11 (the 1024 MB cache request, for 0.93 / 0.92 warm; still below the MPS rows' 1930 MB), NAFNet-GoPro fp32 1.06,
Selective-IGEV middlebury-finetune fp16 1.06 (unchanged code: drift). Lineart and lineart-coarse are not in the survey
rerun: the edges standard input became 1080 x 1920 while every device file holds Lineart at 1024 x 1024, and a ref
with mixed inputs is rejected, so their end-to-end numbers are the cache figures above.

## Larger inputs of the accepted trades (round 1)

Larger input, network peak hand / graph at the smaller and the larger crop of the same network (the other
accepted families have one crop in the survey):

- rtmpose-l 256x192 -> 384x288: fp16 1.16 -> 1.09, fp32 1.31 -> 1.50 (565 -> 743 MB for 2.25x the pixels, the graph
  431 -> 496); rtmpose-l-halpe26 and dwpose-l the same (fp32 1.29-1.31 -> 1.46-1.48). Sublinear in the pixels, so
  accepted, but the float32 ratio grows with the crop: the next pose round should look at it.
- rtmw-l: fp16 1.12 -> 1.07, fp32 1.25 -> 1.35; rtmw-x: fp16 1.18 -> 0.86, fp32 1.17 -> 1.07.
- lw-detr large -> xlarge (a larger network, 640 both): fp16 1.67 -> 1.57.

## SAM 3 after its parity pass (2026-11-26)

The two sam3 rows were measured again after the pass (linear weights stored `[in, out]`, the cached load path,
evaluation boundaries at the 288 x 288 level, the neck levels handed to the prompt without a copy); the counts at the
top are those of the first run. Both rows stay flagged, and by construction: the graph arm translates only `embed`
(trunk and neck, one network call) and runs the text encoder and the detector heads in PyTorch on the CPU, outside the
network timers; the hand arm runs them in MLX as a second network call (`detect`) and keeps their weights resident.
`net_call_ms` / `net_call_peak_mb` (new in the rows) split the hand arm's calls:

| dtype | embed warm ms hand / graph | x | embed peak MB hand / graph | resident MB | embed peak over resident MB | detect (hand only) ms / peak MB | e2e warm ms | e2e footprint MB |
|---|---|---|---|---|---|---|---|---|
| fp16 | 1371.9 / 1439.5 | 0.95 | 1970 / 1385 | 1633 / 892 | 336 / 493 | 202.1 / 2006 | 1590.7 / 2235.5 | 3496 / 4180 |
| fp32 | 1637.2 / 1741.0 | 0.94 | 3785 / 2551 | 3205 / 1776 | 580 / 775 | 260.2 / 3972 | 1909.2 / 2695.2 | 5625 / 5843 |

- On the call both arms make, the hand network is 5 to 6 % faster and its activations are a quarter to a third
  leaner; before the pass the same split was about 1320 / 1375 ms (fp16).
- The network peak is higher only by the text encoder's and the detector's weights (741 MB fp16, 1429 MB fp32), which
  the graph arm holds in PyTorch instead: its process footprint is the larger one. They buy the detector in MLX:
  29 % faster end to end in both dtypes.
- Hand peak before / after the pass: 2263 -> 2006 MB (fp16), 4202 -> 3972 MB (fp32).

## Restoration ports (2026-11-27)

SwinIR, Swin2SR, HAT, DRCT, DAT, Real-ESRGAN, SPAN, Compact and Restormer are hand-written (no graph replay):
`fastforks.shared.swin_sr.mlx_net` holds the Swin layers, the groups and the row-banded tails, `mlx_cnn` the 3 x 3
convs of the CNNs (Winograd `conv_mlx` in kernels mode), each family's `mlx_net` only its own blocks. All load
through `mlx_port.recorded` (the conversion cache, no checkpoint on a hit); `GRAPH_RUNNERS` rebuilds the graph arm
of every family but Restormer (below). 54 A/B rows (the CNNs in both modes, the Swin nets in portable mode, the only
one they have); the full table has no mode column: of a CNN ref's two rows per dtype, the kernels row comes first.

| family | net warm x fp16 | net peak x fp16 | net warm x fp32 | net peak x fp32 |
|---|---|---|---|---|
| swinir | 1.02-1.07 | 0.46-0.80 | 0.78-0.99 | 0.33-0.85 |
| swin2sr | 1.03-1.07 | 0.46-0.73 | 0.78-0.80 | 0.36-0.76 |
| hat | 0.66-0.78 | 0.73-0.91 | 0.42-0.48 | 0.82-1.00 |
| drct | 0.69-0.72 | 0.42-0.43 | 0.55-0.57 | 0.37-0.38 |
| dat | 0.31-0.53 | 0.62-0.74 | 0.48-0.75 | 0.95-1.26 |
| realesrgan kernels | 0.99-1.04 | 1.02-1.20 | 0.92-0.94 | 0.57-0.73 |
| realesrgan portable | 0.94-0.96 | 0.68-0.92 | 0.91-0.94 | 0.83-0.90 |
| span (both modes) | 0.81-0.84 | 0.78-1.00 | 0.75-0.76 | 1.08 |
| compact kernels / portable | 0.92 / 0.77 | 0.82 / 0.71 | 0.93 / 0.71 | 1.00 / 0.82 |

- *The buffer cache the mode asks for.* The graph ports' kernels runners asked `request_cache` for a larger MLX
  buffer cache (Real-ESRGAN x4plus 768 MB, anime 512, Compact and SPAN 512); the first hand runners did not, so the
  survey ran them at the 128 MB default: x4plus kernels float16 3907 ms against 2090 at 1024 MB (anime 1431 / 742,
  Compact float32 282 / 151). The hand runners now ask per variant (`CACHE_MB`: x4plus 1024, anime 512, x2plus 128,
  SPAN and Compact 512, Restormer 1024 as before).
- *Load.* A torch op on a meta tensor (the RDB's `0.2 * conv5` scale, DAT's BatchNorm fold, Swin2SR's bias MLP)
  imports `torch._dynamo` and `torch.distributed`, about 0.7 s: the scale is now applied in the conversion and the
  folds skip a replay's meta tensors. The Swin nets' skeletons still pay it once (the vendored modules' `torch.arange`
  on the meta device goes through the decomposition that imports it): their cached load is about 1.1-1.3 s in a
  fresh process against 0.2-0.4 s for the CNNs.
- *Restormer* was already hand-written; it now loads its folded, padded arrays from the conversion cache too
  (`mlx_port.tensor` by final shape): float32 load 0.77 -> 0.33 s. Its graph path stays gone (below), so it has no
  A/B rows; its e2e rows are below.

Still flagged, with the reason:

- SwinIR and Swin2SR float16: 2-7 % slower warm, 20-54 % leaner (float32 1-22 % faster). The slower part was not
  located this round.
- Real-ESRGAN x2plus kernels float16: peak 1.20 (1018 / 847 MB) at equal speed. One step per dense block against one
  per conv (cache 2048 MB): 479 / 485 ms at +885 / +885-970 MB, so the step granularity is not the cause; the rest
  was not located. x4plus and anime kernels float16: 1.00-1.04 slower and 1.02-1.03 heavier, within the drift above.
- DAT-light float32 (warm 0.75, peak 1.26) and SPAN float32 (0.75-0.76, 1.08): accepted peak trades (their e2e peaks
  are their own float32 rows, below the best MPS row). At 1024 x 1024 DAT-light float32 stays below the graph arm
  (MLX peak 6854 / 7428 MB, 12.2 / 17.5 s), SPAN-x4 float32 reaches 2382 MB (float16 1389).

End to end (`tools/speed_survey.py`, two runs, 512 x 512, the second run's rows kept), against the graph port's
MLX rows of 2026-11-19 to 25 (the old row) and the best MPS row:

| family | e2e warm new / old | e2e peak new / old | below the best MPS peak |
|---|---|---|---|
| swinir | 0.64-0.83 | 0.34-0.70 | all |
| swin2sr | 0.61-0.81 | 0.37-0.79 | all |
| hat | 0.46-0.66 | 0.60-0.94 | all |
| drct | 0.39-0.44 | 0.44-0.50 | all |
| dat | 0.33-0.70 | 0.88-1.17 | all |
| realesrgan | 0.66-1.00 | 0.57-0.95 | all |
| span | 0.68-0.82 | 0.76-0.92 | all |
| compact kernels / portable | 0.93 / 0.71-0.79 | 0.91-1.09 / 0.70-0.78 | all but compact kernels float32 |
| restormer (no graph arm) | 0.98-1.00 | 0.92-0.98 | all |

- Peaks above the old row: DAT float16 (dat-x4 3572 / 3305 MB, dat-2-x4 3643 / 3512, dat-light 2751 / 2425),
  DAT-light float32 (3919 / 3346) and Compact kernels float32 (1461 / 1335, best MPS 1289). The old rows do not
  reproduce: the graph arm (`FASTFORKS_MLX_GRAPH=1`) through the same survey worker on the same day peaks higher
  than the hand port in float16 (dat-light 3389 against 2784 MB, dat-x4 4265 / 3723, dat-2-x4 4332 / 3626) and in
  Compact kernels float32 (1477-1514 against 1444-1460, at 233-245 / 162-173 ms); only DAT-light float32 is
  heavier than it (3845 / 3324, the accepted trade). A smaller buffer cache does not lower Compact's float32 peak
  without the speed (512 / 256 / 128 MB: 159 / 183 / 268 ms at 1358 / 1346 / 1204 MB in one process).
- Load (`load_s`, torch already imported): the CNNs 0.16-0.35 s (old 0.10-0.27), the float16 kernels runners
  up to 0.54 (x2plus 0.51-0.54 against the graph arm's 0.30): their cache file holds the Winograd-transformed
  weights, 4x the plain ones (x2plus 167 / 33 MB), read from the cache disk. The Swin nets 0.76-1.15 s (old
  0.29-1.16): the skeleton's dynamo import above. Restormer float32 0.32-0.34 (old 0.76-0.78). The first call is
  far shorter everywhere but Restormer (no compile of a replayed graph: swinir-m-x4 float16 3.5 / 8.0 s,
  drct-l-x4 8.5 / 68.9 s).

## Detection / pose ports (2026-11-27)

ViTPose, HRNet, DETR, OWLv2, EoMT, RetinaFace and YuNet are hand-written (no graph replay). Each family's `mlx_net`
holds only its own blocks: the CNN backbones run through `mlx_cnn` and are stepped with `cspnext.mlx_net.stepped`
(HRNet, DETR's ResNet, RetinaFace, YuNet), and the ViTs use `mlx_vit`'s block maths (ViTPose, OWLv2, EoMT on the
DINOv2 / v3 backbones). They load through `mlx_port.recorded` / `tensor` (the conversion cache, no checkpoint on a
hit), do their pre- and post-processing in numpy / MLX, and ask `request_cache` for a cache sized per variant.
`GRAPH_RUNNERS` rebuilds the graph arm of every family. 40 A/B rows (portable, both dtypes). The goldens pass in
float16 and float32, and float16 is auto's pick.

| family | net warm x fp16 | net peak x fp16 | net warm x fp32 | net peak x fp32 |
|---|---|---|---|---|
| vitpose | 0.99-1.03 | 0.74-1.03 | 0.97-1.02 | 0.48-1.15 |
| hrnet | 0.87-0.92 | 0.86-1.08 | 0.83-0.87 | 0.96-1.08 |
| detr | 0.88-0.91 | 0.80-0.85 | 0.90-0.92 | 1.09-1.16 |
| owlv2 | 1.01 | 1.33-1.49 | 0.99-1.01 | 1.34-1.58 |
| eomt | 0.99-1.02 | 1.02-1.03 | 0.99-1.02 | 1.01-1.03 |
| retinaface | 0.89-0.90 | 0.52-1.26 | 0.88 | 0.76-1.29 |
| yunet | 0.83-0.84 | 0.83-0.93 | 0.80-0.81 | 1.12-1.35 |

- *Half blocks.* ViTPose steps its blocks in halves (attention, then MLP), and each half is `async_eval`'d while the
  previous one is awaited. That keeps the speed with about half the transients of `mlx_vit.blocks(bounded=True)`
  (ViTPose-B float32 141.0 ms at +262 MB -> 141.0 ms at +152 MB; one block per synchronous eval: 143.4 ms at +143 MB).
- *Post-processing outside the timed call.* EoMT's window averaging is a module function (`mlx_net.scores`), not a
  method, so the A/B times the same network in both arms.
- *Load.* No torch import on a cache hit. The A/B `load_s` is 0.04-0.11 s against the graph's 0.04-2.4 s. Three
  rows are higher: vitpose-plus-huge float32 1.17 / 0.69 s, owlv2-l14 float32 1.02 s, eomt-l-dinov3 float32 0.82 s.
  Those are large cache files read from the cache disk. The first network call takes 0.04-2.07 s against the
  graph's 0.95-4.2 s (no compile of a replayed graph).

Accepted peak trades (in the table above): detr-r50 float32, hrnet-w32 float32, hrnet-w48 both dtypes,
retinaface-resnet50 both dtypes, yunet-n and yunet-s float32.

Still flagged, with the reason:

- detr-r101 float32 (warm 0.92, peak 1.09) and hrnet-w48-384 float16 (0.92, 1.08) are faster but not by 10 %. The
  float32 peaks come from the stepped backbone's overlap (two stages in flight), which buys the speed.
- OWLv2, all four rows (peak 1.33-1.58 at equal speed): accounting. The hand port keeps the text tower in MLX
  (resident +121 / +242 MB at b16 float16 / float32, +236 / +472 MB at l14), while the graph arm ports only the image
  tower and leaves the text tower in torch. The hand arm also times three network calls per model call against the
  graph's one. Transients above resident are +45-74 MB. The e2e peak is lower than the old row everywhere but
  b16 float32 (1286 / 1253 MB).
- vitpose-base and vitpose-plus-base float32, peak 1.14-1.15 at 0.97-0.98 warm: the token preparation and the head
  on top of the half-block transients. Stepping the head (about -17 MB) and the tokens gave no net gain. Calling
  `mx.fast.scaled_dot_product_attention` directly gave nothing either, since head size 64 is already fused.
- ViTPose float16 base-simple, plus-small and plus-huge are 1.02-1.03 slower warm, and plus-large float16 is 1.03
  heavier. The cause was not located (GELU is already the fused one).
- EoMT, 1.01-1.03: drift level. Sharing the half-block stepper with `mlx_vit` would need an `mlx_vit` edit, which
  was left for another round (another port is editing it).
- grounding_dino was not ported: Swin + BERT + deformable attention + the fusion layers are too large for this
  round. It keeps its graph runner.

End to end (`tools/speed_survey.py`, two runs, the second run's rows kept; every row ok with the golden passing),
against the graph port's MLX row (the old row) and the best MPS row:

| family | e2e warm new / old | e2e peak new / old | below the best MPS peak |
|---|---|---|---|
| vitpose | 0.84-1.03 | 0.59-1.02 | all |
| hrnet | 0.77-0.93 | 0.80-0.90 | all |
| detr | 0.82-0.94 | 0.68-0.94 | all |
| owlv2 | 0.98-1.00 | 0.82-1.03 | all |
| eomt | 0.76-0.86 | 0.63-0.68 | all |
| retinaface | 0.66-0.95 | 0.76-0.83 | all |
| yunet | 0.62-0.79 | 0.78-0.96 | all |

- Above the old row: e2e warm for vitpose-plus-small float16 (1.03: 74.4 / 72.3 ms), plus-large and plus-huge
  float16 (1.02), base and plus-base float16 (1.01), and plus-huge and plus-large float32 (1.01). Peaks above it:
  owlv2-b16 float32 (1286 / 1253 MB) and vitpose-base float32 (1083 / 1066).
- MPS is faster warm on vitpose-plus-small (66.5 / 74.4 ms), hrnet-w48-384 (369 / 416 ms in float16) and
  retinaface-mobilenet0.25 float16 (15.3 / 15.8 ms). On vitpose-base and plus-base float32 the two are about equal
  (172-174 / 176 ms).
- Load (`load_s`, survey): the CNNs 0.15-0.51 s (old 0.16-0.92). EoMT 0.91-1.67 s (old 2.22-4.2). OWLv2 0.55-2.27 s
  (old 2.18-4.97). ViTPose takes longer than the old row: plus-huge 1.69 / 0.29 s in float16 and 3.21 / 2.66 in
  float32, plus-large 0.90 / 0.21 and 1.63 / 1.46. The old row loaded the replayed graph's weights lazily; the hand
  port reads its full cache file at load. The first call is 0.05-2.0 s against the old 1.3-5.9 s.

## SAM 2.1 prompt path and video, SAM 3 tracker (2026-11-27)

What still ran in PyTorch on the CPU in MLX mode is in `mx` now: SAM 2.1's prompt encoder and mask decoder
(`sam2/mlx_heads.py`, float32 as before), its video memory path (memory attention, memory encoder, object pointers:
`mlx_heads.py` in the run dtype, the loop in numpy / MLX in `sam2/mlx_video.py`) and SAM 3's tracker heads
(`sam3/net/mlx_tracker.py`). All weights come through `mlx_port.recorded`: a cache hit reads no checkpoint and builds
no float32 CPU twin. The graph arm is unchanged (`embed` alone, the prompt in PyTorch), so the SAM 2.1 rows now
count six network calls in the hand arm (embed, the prompt encoder's parts, the decoder: 5.5 ms and at most 98 MB
over the resident weights) against one, and its resident weights are 56-67 MB higher (the decoder and the dense
position embedding): that is most of the `heavier` flag. The embed call's peak over the resident weights is
236-348 MB against 276-320 MB in float16 (large alone is higher, by 48 MB) and 328-462 MB against 300-462 MB in
float32 (0-56 MB higher); the process footprint is 125-430 MB lower in every row. The sam3 rows are as in the
section above (image path untouched).

Image, end to end (`tools/speed_survey.py`, two runs, the second kept; before -> now):

| ref | dtype | warm ms | load s | peak footprint MB | load footprint MB |
|---|---|---|---|---|---|
| sam2.1-hiera-base-plus | fp16 | 233.8 -> 229.0 | 0.43 -> 0.38 | 1072 -> 911 | 624 -> 378 |
| sam2.1-hiera-base-plus | fp32 | 273.2 -> 271.8 | 0.97 -> 0.57 | 1311 -> 1153 | 713 -> 529 |
| sam2.1-hiera-large | fp16 | 522.8 -> 516.2 | 0.76 -> 0.77 | 1474 -> 1243 | 985 -> 654 |
| sam2.1-hiera-large | fp32 | 600.1 -> 597.1 | 2.28 -> 1.27 | 1986 -> 1737 | 1325 -> 1075 |
| sam2.1-hiera-small | fp16 | 136.6 -> 133.3 | 0.33 -> 0.29 | 931 -> 798 | 498 -> 304 |
| sam2.1-hiera-small | fp32 | 169.0 -> 168.5 | 0.67 -> 0.38 | 1075 -> 936 | 528 -> 380 |
| sam2.1-hiera-tiny | fp16 | 116.7 -> 114.7 | 0.31 -> 0.26 | 919 -> 785 | 486 -> 289 |
| sam2.1-hiera-tiny | fp32 | 146.2 -> 145.0 | 0.59 -> 0.34 | 1043 -> 905 | 501 -> 352 |
| sam3 | fp16 | 1543.8 -> 1546.6 | 2.24 -> 2.23 | 2590 -> 2582 | 1858 -> 1858 |
| sam3 | fp32 | 1901.6 -> 1893.9 | 4.21 -> 4.27 | 4501 -> 4472 | 3429 -> 3430 |

Per prompt on a cached embedding and per video frame (8 frames, one box prompt, warm second pass; a bench script, not
the survey; before -> now):

| ref | dtype | prompt ms (CPU ms) | 4 boxes ms | video ms / frame (CPU ms) | video MLX peak MB | video process footprint MB |
|---|---|---|---|---|---|---|
| sam2.1-hiera-tiny | fp16 | 10.8 (16.2) -> 8.1 (8.4) | 23.7 -> 18.8 | 348.6 (150) -> 330.4 (145) | 937 -> 1034 | 1527 -> 1384 |
| sam2.1-hiera-base-plus | fp16 | 10.1 (15.7) -> 8.4 (8.4) | 24.0 -> 19.0 | 467.8 (179) -> 446.5 (176) | 1025 -> 1136 | 1639 -> 1484 |
| sam2.1-hiera-base-plus | fp32 | 9.6 (15.3) -> 8.0 (8.3) | 22.1 -> 19.1 | 608.2 (337) -> 590.1 (336) | 1932 -> 2065 | 2339 -> 2242 |
| sam3 | fp16 | - | - | 2352.7 (4901) -> 1760.8 (316) | 2096 -> 2628 | 4205 -> 2977 |
| sam3 | fp32 | - | - | 2657.5 (5362) -> 2285.1 (666) | 4010 -> 4637 | 5418 -> 4904 |

- *Memory.* The MLX allocator's peak rises by the head weights that used to sit in PyTorch (about 100 MB for
  SAM 2.1, 530-630 MB for SAM 3); the process footprint, which counts both, falls in every row.
- *Load.* The replay skeleton of a cache hit ran torch ops that have no meta kernel (`torch.randn` in the prompt
  encoder's random position embedding, the neck fold's matmul, the position embedding's interpolate, `torch.arange`
  in the RoPE table, `scale * torch.ones` in the memory encoder's blocks): each goes through a Python decomposition
  that imports `torch._dynamo`, about 1 s once per process. The skeleton now gets the folded shapes directly
  (`_replay_shapes`) and builds the prompt encoder and the video modules unfilled on the CPU: cached load 0.63 ->
  0.23 s (tiny float16), 0.97 -> 0.53 s (base-plus float32).
- *Video is GPU-bound.* Of tiny's 330 ms / frame, 137 ms is the image encoder and about 200 ms the memory attention
  (four layers over up to seven frames of memory tokens, evaluated per layer); the host side (frame preparation in the
  prefetch thread, the tracker's numpy bookkeeping) is under 30 ms and not on the critical path.
- *Results.* Goldens pass in both dtypes (image for sam3, image and video for sam2.1); `test_mlx` has match tests
  against the PyTorch prompt path, the tracker and the motion model. Against the old CPU tracker the video mask
  logits move by at most 0.13 (float16) and 0.0023 (float32); at most 2.5e-6 of a frame's mask pixels flip.
- *PyTorch is still imported* in an MLX run: the task API takes and returns torch tensors, `model.py` / `video.py` /
  `net/*` import it at module level, the replay skeleton is a (weightless) torch module, and the image `prepare` /
  `upsample` are torch. No torch op runs per prompt or per frame in the heads or the tracker.

## Families whose graph path could not be rebuilt

These have no `GRAPH_RUNNERS` (restoring the pre-port runner would change the torch path or need far more than a small
wrapper), so they are not in the A/B:

- fast_acvnet: the hand port removed the `exporting_for_mlx` / float32-island logic from the torch forward.
- gmflow: the export branches were removed from the torch path (large diff).
- raft: needs the deleted 148-line `raft/mlx.py`.
- sea_raft: needs the deleted `sea_raft/mlx.py` and its Winograd arguments.
- rife: the export branches were removed from `_interpolate`.
- ema_vfi: the torch-side `_loaded` / `_net` were rewritten.
- realplksr, restormer: the custom ops and their lowerings were stripped (over 60 lines, another file).
- moge: needs the deleted `_below_top`, `_rep_pad32`, the old `_trunk` and `to_float32` in the torch `_FoldedTop`.
- resplat: `GRAPH_RUNNERS` builds, but today's torch network calls the custom op `fastforks.resplat_knn`, which the
  graph translator has no mapping for (`UnsupportedOp`): no graph arm.

Hand-written from the start (never had a graph path): crestereo, seedvr2, dinov2, dinov3, c_radio, tipsv2, romav2,
depth_anything_v2, marigold_v2, dinov2_linear, dinov2_linear_cls.

## Optical-flow ports (2026-11-27)

MEMFOF, MemFlow, FlowSeek, NeuFlow v2 and ReCoVEr (cx / rn / mn) are hand-written (no graph replay). Each family's
`mlx_net` holds its own encoders, correlation and update blocks on top of `mlx_flow`; FlowSeek's DINOv2 runs through
`mlx_vit`'s block maths. They load through `mlx_port.recorded` (the conversion cache, no checkpoint on a hit), do the
padding and upsampling in MLX, and ask `request_cache` for a cache sized per variant (memfof 512 MB, memfof-kitti and
-sintel 256, recover-cx and recover-rn float16 256, the rest the base limit). `GRAPH_RUNNERS` rebuilds the graph arm
of every family. 32 A/B rows (portable, both dtypes, 1080 x 1920). The goldens pass in float16 and float32, and
float16 is auto's pick.

| family | net warm x fp16 | net peak x fp16 | net warm x fp32 | net peak x fp32 |
|---|---|---|---|---|
| flowseek | 0.70-0.71 | 0.64 | 0.53-0.55 | 0.37-0.38 |
| memflow | 0.81-0.83 | 0.35 | 0.82-0.83 | 0.30 |
| memfof | 0.89-0.92 | 0.68-0.96 | 0.89-0.94 | 0.82-0.85 |
| neuflow_v2 | 0.43-0.48 | 0.19-0.37 | 0.32-0.58 | 0.19 |
| recover | 0.84-1.06 | 0.56-0.70 | 0.70-0.97 | 0.75-1.03 |

- *Transient and resident.* On the transient (call peak minus resident) the ratios are about the peak ratios above:
  flowseek 0.64 / 0.36-0.37, memflow 0.35 / 0.30, memfof 0.66-0.96, neuflow 0.19-0.36, recover 0.54-0.67 float16 and
  0.73-1.03 float32. The resident weights are lower or equal: flowseek 82 / 146 MB float16 and 164 / 228 float32,
  memflow 12 / 97 and 24 / 112, memfof 140 / 154 and 280 / 307, neuflow 26 / 36 and 35 / 45, recover about equal.
  float16 resident is about half of float32 everywhere.
- *The cache is part of the e2e peak.* The process footprint carries MLX's buffer cache, and the graph runners ran at
  the base limit (128 MB). At 1024 MB the hand ports were 15-59 % heavier end to end than the old rows. The sizes
  above are the largest that keep every e2e row below the graph port's (sweeps 128-1024 MB at 1080 x 1920).
- *ReCoVEr float16 keeps half-precision Winograd.* The graph translation runs float16 Winograd convs with float32
  transforms (`mlx_flow.options(half_wino_f32=True)`), which made rn's full-resolution convs allocation-bound at a
  small cache (e2e 495 ms at 256 MB). Without them: 415 ms, golden mean 0.0024 against the graph's 0.0021 (budget
  0.008). cx and mn give the same outputs either way.
- *MemFlow's checkpoint.* Its downsample batch norm lives at `downsample.1`, as torch's `load_state_dict` keeps it;
  the hand port reads it from there.
- *Odd size and batch 2.* 437 x 645, batch of two pairs: float32 hand against graph mean difference 0.0000 in every
  family, float16 0.0003-0.0013 (flowseek 0.023; its goldens show the hand arm closer). Row 0 of the batch equals the
  single call (max difference 0) everywhere.
- *Load.* No torch import on a cache hit, except FlowSeek: building its skeleton under `torch.device("meta")` runs the
  vendored DINOv2 LayerScale's `init_values * torch.ones(dim)`, which imports `torch._dynamo` (0.6-0.77 s). Its
  checkpoints are `.pth`, so `header_skeleton` does not apply. A/B `load_s` 0.04-0.05 s (memfof up to 0.41, flowseek
  0.50-0.62) against the graph's 0.03-0.16 s. The first network call takes 0.28-17.2 s against the graph's 1.6-22.5 s
  (flowseek 1.3-1.5 / 4.7-6.9, neuflow 0.4-0.5 / 2.6-4.2, recover 0.3-0.9 / 1.6-2.4).

Accepted peak trades (in the table above): recover-mn float32 (warm 0.70, peak 1.03; e2e 1091 MB against MPS 2562;
1024² -> 4x: 1294 MB / 646 ms against the graph's 1059 / 1087, no blow-up).

Still flagged, with the reason:

- recover-cx float16, warm 1.06 at the A/B's forced 2048 MB cache: the graph arm gains more from a large cache. Piece
  by piece the hand port matches it, and end to end at its own 256 MB it is faster than the old row (720.6 / 757.7
  ms).
- FlowSeek's load (survey 0.84-0.97 s against the old 0.17-0.25): the `torch._dynamo` import above. Avoiding it needs
  a vendored-code or a shared `mlx_port` change, left for another round.
- WAFT, FlowFormer and MegaFlow were not ported: WAFT's warping transformer, FlowFormer's Twins encoder and cost
  memory, and MegaFlow's ViT-L with the VGGT aggregator are too large for this round. They keep their graph runners.

End to end (`tools/speed_survey.py`, two runs, the second run's rows kept; every row ok with the golden passing),
against the graph port's MLX row (the old row) and the best MPS row:

| family | e2e warm new / old | e2e peak new / old | below the best MPS peak |
|---|---|---|---|
| flowseek | 0.46-0.73 | 0.41-0.65 | all |
| memflow | 0.80-0.82 | 0.36-0.52 | all |
| memfof | 0.90-0.96 | 0.88-1.00 | all with an MPS row (kitti and sintel have none) |
| neuflow_v2 | 0.43-0.57 | 0.21 | all |
| recover | 0.71-0.99 | 0.73-0.95 | all |

- No e2e warm or peak row is above the old one (the closest: memfof-spring float32 peak 2241 / 2244 MB, warm 0.96).
- MPS is faster warm only on recover-rn float16 (405 / 411 ms).
- Load (`load_s`, survey): memflow, neuflow and recover 0.18-0.32 s (old 0.13-0.32), memfof 0.33-0.57 (old
  0.14-0.19), flowseek 0.84-0.97 (old 0.17-0.25). The hand ports read their full cache file at load. The first call
  is much lower: flowseek 1.6-2.2 s (old 4.7-17), recover 0.6-1.0 (old 1.7-4.3), memflow 8.8-10.8 (old 11.7-14.5),
  memfof 3.0-19.0 (old 5.4-22.1).

## Method (from 2026-11-27)

What `tools/mlx_core_ab.py run` measures now, and how a row gets flagged:

- **Peak.** The network's transient peak is recorded as the peak over the active memory before the call, and
  `heavier` flags on that transient peak. The resident memory (the weights and caches held between calls) is a separate
  column. The end-to-end process peak `e2e_peak_mb` is a column too, and the accepted-trade bar uses it. The md checks the
  resident memory by dtype: float16 should be about 0.5x float32, and families above 0.6 are listed as float32 islands.
  Old rows have no transient peak, so their value is estimated from the peak minus the resident memory.
- **As shipped.** A `shipped` arm runs the hand network at its own `request_cache` (what a user gets) and flags
  `shipped-slower`. The 2048 MB cache arm is still measured, but only as a diagnostic.
- **Symmetric host copies.** In both arms, `to_mx` / `to_torch` inside a network call are left out of the timed span.
  Before this change the graph arm timed its input and output copies and the hand arm did not.
- **Repeats and CI.** The arms run in interleaved processes, ABAB (`--repeats`, default 2). `slower` flags only when the
  bootstrap 95 % interval of the warm hand / graph ratio lies above 1.02. The flagged-beyond-10 % list still uses the
  point ratio. Networks under 10 ms per call are timed over a loop of model calls.
- **Call counts.** A row whose arms make a different number of network calls (`net_calls`) is counted and marked. Its
  ratio compares whole model calls.
- **Old rows.** The schema only gains fields. Rows measured before this change keep their meaning: a point ratio and
  an estimated transient peak. `run --flagged` re-measures the flagged rows of the given refs in place.

Effect on the counts (709 rows):

| | flagged | slower | heavier | worse | accepted |
|---|---:|---:|---:|---:|---:|
| old rules, old rows | 280 | 105 | 187 | 10 | 92 |
| new rules, old rows | 284 | 105 | 189 | 10 | 96 |
| new rules, flagged rows re-measured (733 rows) | 226 | 36 | 184 | 10 | 90 |

With the old rows, the new rules leave the slower count unchanged, because a point ratio has no interval. The estimated
transient peak flags 2 more rows heavier. 96 rows show different call counts.

The re-measure covers 259 flagged rows. Both arms of each row ran in one session, interleaved ABAB. Of those 259 rows,
193 stay flagged: 35 slower, 151 heavier, 10 worse and 56 shipped-slower, with 70 accepted. Their earlier
measurements came from other sessions, with the arms in separate runs. Those flagged 104 slower, 164 heavier and 10
worse, with 81 accepted. They are context only, not a same-session comparison. The igev-plusplus rows measured while
the Mac was loaded (kitti12 in both dtypes, kitti15 in float16) were re-measured after the load passed. Skipped, so
still old-rule rows: igev-stereo-kitti15, -middlebury and -sceneflow, and the seven selective-igev refs. Each stereo
ref takes about 13 minutes. The table grew to 733 rows because ports landed upstream in the meantime.

Tools for a port (see each tool's docstring):

- `tools/mlx_core_ab.py quick REF...` is the check while iterating. It runs in one process, and only the hand arm is
  timed, against a graph baseline cached once per session. Each sample is normalized by a fixed reference net timed
  around it. Inputs are float16 at half size. It stops once the interval is clearly beyond or inside ±10 %.
  The reference net counts its fastest of 3 calls: one call right after a large network's release read about 2x slow,
  so IGEV++ looked x0.64 instead of x0.94 raw. Wall times, quick with a cached baseline vs the full A/B (f16 only):
  resnet50 4-5 s vs 21 s (x0.85-0.88 vs x0.88), mobilenet_v3_large 3-4 s vs 20 s (x0.77 vs x0.76), IGEV++ kitti12
  47 s vs 625 s (83 s with a new baseline). At half size a ratio can differ from the full-size one: IGEV++ is x0.93
  `open` quick vs x0.81 full, so the full A/B settles an `open` result.
- `tools/mlx_gate.py REF...` is run once per port. It runs the goldens, one survey run (the rows the website and auto
  use) and the `gen_auto_mode` / `gen_docs` checks of the port's refs, and prints them as one pass/fail table. The
  full A/B (`--ab`) is only for a borderline quick result and the final audit; an accepted heavier row passes.
- `tools/mlx_layer_diff.py REF --dtype f16 --size HxW [--time]` compares per tap against the torch reference. It
  prints max / mean / relative diff and the first diverging tap (`--atol` for small residuals). The taps come from
  `shared/mlx_port.tap`.
- `tests/test_mlx_hand_shapes.py --slow` checks each port at one odd size and at batch 2 against the graph arm or torch.
- `FASTFORKS_ROWS_AGENT=<name>` sends rows to `docs/data/speed/_pending/<name>/`, and `tools/merge_rows.py` merges them
  into the shared files.

<!-- generated by `tools/mlx_core_ab.py md`: everything below this line -->

## Counts

733 rows over 106 families; 729 measured in both arms.

- 226 rows flagged (36 slower, 184 heavier, 10 worse, 56 shipped-slower).
- 259 rows measured with the transient peak and the copies left out, 259 with a bootstrap interval; the other 470 flag `slower` on the point ratio and `heavier` on the estimated transient peak (net peak less resident).
- 94 rows whose arms make a different number of network calls.
- 90 of them accepted as a peak trade; 136 miss the bar (85 beyond 10 %).
- Median hand / graph: warm 0.89, peak 0.90, transient 0.78.
- Quality: 256 rows with a golden error in both arms, 0 without a golden, 10 hand arms `worse` than the graph's.

## Flagged beyond 10 %

Not accepted, hand / graph warm or transient peak above 1.10, worst first.

| ref | dtype | net warm ms hand / graph | x | 95 % CI | transient MB hand / graph | x | flag |
|---|---|---|---|---|---|---|---|
| lw-detr-large-coco | fp16 | 53.1 / 58.0 | 0.92 | 0.91-0.93 | 431 / 238 | 1.80 | heavier |
| rf-detr-keypoint-preview | fp32 | 73.3 / 75.2 | 0.97 | 0.97-0.98 | 178 / 99 | 1.80 | heavier |
| pe-core-b16-224 | fp32 | 17.1 / 17.6 | 0.97 | 0.95-0.98 | 144 / 82 | 1.75 | heavier |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.1 | 0.98 | 0.97-0.98 | 144 / 82 | 1.75 | heavier |
| selective-igev-eth3d-finetune | fp32 | 4788.7 / 6385.9 | 0.75 | - | 3448 / 2005 | 1.72 | heavier |
| selective-igev-eth3d-train | fp32 | 4778.2 / 6294.8 | 0.76 | - | 3448 / 2005 | 1.72 | heavier |
| selective-igev-kitti12 | fp32 | 4733.6 / 6386.6 | 0.74 | - | 3448 / 2005 | 1.72 | heavier |
| selective-igev-kitti15 | fp32 | 4725.9 / 6386.9 | 0.74 | - | 3448 / 2005 | 1.72 | heavier |
| selective-igev-middlebury-finetune | fp32 | 4760.9 / 6439.7 | 0.74 | - | 3448 / 2005 | 1.72 | heavier |
| selective-igev-middlebury-train | fp32 | 4728.9 / 6368.5 | 0.74 | - | 3448 / 2005 | 1.72 | heavier |
| selective-igev-sceneflow | fp32 | 4694.9 / 6337.5 | 0.74 | - | 3448 / 2005 | 1.72 | heavier |
| unik3d-vitl | fp16 | 1009.3 / 1021.2 | 0.99 | 0.98-1.00 | 808 / 470 | 1.72 | heavier |
| pe-core-g14-448 | fp32 | 1180.9 / 725.9 | 1.63 | 0.04-11.10 | 143 / 127 | 1.13 | heavier |
| efficientvit-sam-xl1 | fp32 | 347.6 / 344.8 | 1.01 | 0.82-1.02 | 1412 / 908 | 1.55 | heavier |
| igev-stereo-middlebury | fp32 | 4180.7 / 4608.6 | 0.91 | - | 3074 / 2005 | 1.53 | heavier |
| igev-stereo-sceneflow | fp32 | 4184.8 / 4628.8 | 0.90 | - | 3074 / 2005 | 1.53 | heavier |
| segnext-mscan-l-ade20k | fp16 | 139.1 / 127.7 | 1.09 | 1.07-1.12 | 320 / 211 | 1.52 | slower heavier shipped-slower |
| igev-stereo-kitti12 | fp32 | 4197.6 / 4663.5 | 0.90 | 0.89-0.93 | 3039 / 2005 | 1.52 | heavier shipped-slower |
| igev-stereo-eth3d | fp32 | 4171.1 / 4690.8 | 0.89 | 0.88-0.90 | 3039 / 2005 | 1.52 | heavier shipped-slower |
| segformer-b0-ade20k | fp32 | 24.2 / 22.2 | 1.09 | 1.08-1.14 | 202 / 136 | 1.49 | slower heavier shipped-slower |
| vitpose-base | fp32 | 160.5 / 163.6 | 0.98 | 0.97-0.98 | 183 / 126 | 1.45 | heavier |
| vitpose-plus-base | fp32 | 160.2 / 164.0 | 0.98 | 0.97-0.98 | 183 / 126 | 1.45 | heavier |
| efficientvit-sam-xl0 | fp32 | 191.0 / 203.6 | 0.94 | 0.93-0.97 | 856 / 596 | 1.44 | heavier |
| siglip2-g16-256 | fp16 | 158.8 / 157.8 | 1.01 | 0.97-1.04 | 68 / 47 | 1.43 | heavier |
| siglip2-g16-384 | fp16 | 154.5 / 149.2 | 1.04 | 1.03-1.04 | 68 / 47 | 1.43 | slower heavier shipped-slower |
| owlv2-b16 | fp32 | 364.9 / 366.5 | 1.00 | 0.99-1.00 | 285 / 200 | 1.42 | heavier |
| pspnet-r101-d8-ade20k | fp16 | 308.3 / 315.7 | 0.98 | 0.93-1.02 | 910 / 642 | 1.42 | heavier |
| rtmw-l | fp32 | 111.3 / 123.5 | 0.90 | 0.89-0.91 | 451 / 326 | 1.38 | heavier |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp32 | 23.3 / 25.7 | 0.91 | 0.84-0.94 | 113 / 83 | 1.36 | heavier |
| segformer-b0-ade20k | fp16 | 20.4 / 16.8 | 1.22 | 1.16-1.31 | 145 / 107 | 1.36 | slower heavier shipped-slower |
| raft-stereo-sceneflow | fp32 | 4928.5 / 5432.3 | 0.91 | 0.88-0.95 | 2256 / 1693 | 1.33 | heavier |
| lineart-coarse | fp32 | 97.6 / 117.2 | 0.83 | 0.82-0.84 | 1167 / 887 | 1.32 | heavier shipped-slower |
| sam-vit-h | fp16 | 1533.7 / 1698.2 | 0.90 | 0.90-0.91 | 512 / 388 | 1.32 | heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 24.8 / 25.3 | 0.98 | 0.95-1.04 | 142 / 110 | 1.30 | heavier |
| raft-stereo-eth3d | fp32 | 4931.3 / 5316.3 | 0.93 | 0.88-0.97 | 2256 / 1757 | 1.28 | heavier |
| raft-stereo-middlebury | fp32 | 4926.0 / 5140.2 | 0.96 | 0.94-0.98 | 2256 / 1757 | 1.28 | heavier |
| shvit_s4.in1k | fp16 | 6.1 / 4.8 | 1.28 | 1.14-1.36 | 8 / 20 | 0.45 | slower shipped-slower |
| totalsegmentator-total-3mm | fp16 | 367.5 / 544.6 | 0.67 | 0.60-0.74 | 2007 / 1574 | 1.28 | heavier |
| segnext-mscan-s-ade20k | fp16 | 48.4 / 38.3 | 1.26 | 1.12-1.37 | 204 / 170 | 1.20 | slower heavier shipped-slower |
| realesrgan-x2plus | fp16 | 499.2 / 506.6 | 0.99 | - | 855 / 682 | 1.25 | heavier |
| segnext-mscan-t-ade20k | fp16 | 32.6 / 26.1 | 1.25 | 1.10-1.27 | 112 / 168 | 0.67 | slower shipped-slower |
| deeplabv3-r101-d8-ade20k | fp16 | 367.1 / 379.9 | 0.97 | 0.94-0.97 | 514 / 413 | 1.25 | heavier |
| segnext-mscan-l-ade20k | fp32 | 157.1 / 164.6 | 0.95 | 0.94-0.97 | 325 / 261 | 1.24 | heavier shipped-slower |
| retinaface-resnet50 | fp16 | 120.2 / 133.0 | 0.90 | 0.90-0.91 | 397 / 320 | 1.24 | heavier |
| rf-detr-l-coco | fp32 | 53.8 / 52.7 | 1.02 | 1.01-1.02 | 132 / 107 | 1.23 | heavier shipped-slower |
| raft-stereo-middlebury | fp16 | 3204.2 / 4174.9 | 0.77 | 0.76-0.78 | 1176 / 960 | 1.22 | heavier worse |
| da3-large-1.1 | fp16 | 215.3 / 211.0 | 1.02 | 1.01-1.04 | 417 / 346 | 1.20 | heavier shipped-slower |
| owlv2-b16 | fp16 | 283.1 / 278.7 | 1.02 | 1.01-1.02 | 197 / 164 | 1.20 | heavier shipped-slower |
| rf-detr-seg-2xl-coco | fp16 | 128.0 / 124.8 | 1.03 | 0.95-1.05 | 268 / 223 | 1.20 | heavier worse |
| dat-light-x4 | fp32 | 1332.4 / 1712.8 | 0.78 | 0.76-0.79 | 2326 / 1943 | 1.20 | heavier shipped-slower |
| deeplabv3plus-r101-d8-ade20k | fp32 | 371.3 / 404.3 | 0.92 | 0.85-0.95 | 1169 / 980 | 1.19 | heavier |
| loma-b64 | fp32 | 1055.7 / 976.6 | 1.08 | 0.94-1.11 | 1059 / 892 | 1.19 | heavier |
| davit_tiny.msft_in1k | fp16 | 9.1 / 7.7 | 1.18 | 1.15-1.21 | 40 / 69 | 0.59 | slower shipped-slower |
| segformer-b2-clothes | fp16 | 29.3 / 25.6 | 1.14 | 1.09-1.18 | 130 / 110 | 1.18 | slower heavier shipped-slower |
| segnext-mscan-b-ade20k | fp16 | 82.8 / 70.6 | 1.17 | 1.10-1.24 | 204 / 210 | 0.97 | slower shipped-slower |
| sapiens-seg-1b-goliath | fp32 | 3309.6 / 4238.3 | 0.78 | 0.62-0.97 | 2333 / 1993 | 1.17 | heavier |
| sapiens-seg-0.3b-goliath | fp32 | 1434.3 / 1799.8 | 0.80 | 0.78-0.82 | 2327 / 1993 | 1.17 | heavier |
| sam2.1-hiera-base-plus | fp32 | 255.9 / 276.5 | 0.93 | 0.81-0.99 | 392 / 336 | 1.17 | heavier |
| deeplabv3plus-r101-d8-ade20k | fp16 | 270.8 / 289.8 | 0.93 | 0.92-1.01 | 652 / 559 | 1.17 | heavier |
| loma-b128 | fp32 | 1264.1 / 1270.7 | 0.99 | 0.98-1.02 | 1097 / 942 | 1.16 | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp16 | 200.6 / 204.3 | 0.98 | 0.98-0.99 | 97 / 83 | 1.16 | heavier |
| wd-eva02-large-tagger-v3 | fp16 | 195.6 / 201.7 | 0.97 | 0.94-0.97 | 97 / 83 | 1.16 | heavier |
| convnext_large.fb_in22k_ft_in1k_384 | fp32 | 67.3 / 71.0 | 0.95 | 0.94-0.95 | 159 / 137 | 1.16 | heavier |
| totalsegmentator-total-3mm | fp32 | 473.8 / 604.4 | 0.78 | 0.75-0.81 | 1960 / 1703 | 1.15 | heavier |
| rtmw-l | fp16 | 100.7 / 106.6 | 0.94 | 0.94-0.95 | 243 / 212 | 1.15 | heavier |
| lw-detr-large-coco | fp32 | 57.5 / 60.4 | 0.95 | 0.95-0.95 | 414 / 363 | 1.14 | heavier |
| sam2.1-hiera-small | fp32 | 156.1 / 168.7 | 0.93 | 0.78-1.02 | 328 / 288 | 1.14 | heavier |
| sam2.1-hiera-tiny | fp32 | 148.4 / 146.6 | 1.01 | 0.93-1.10 | 328 / 288 | 1.14 | heavier |
| hrnet-w48 | fp16 | 187.5 / 205.9 | 0.91 | 0.89-0.93 | 164 / 144 | 1.13 | heavier |
| vitpose-plus-large | fp16 | 445.3 / 453.4 | 0.98 | 0.96-1.05 | 118 / 104 | 1.13 | heavier |
| pe-spatial-g14-448 | fp32 | 292.0 / 290.7 | 1.00 | 0.12-1.01 | 143 / 127 | 1.13 | heavier |
| segformer-b5-face-parsing | fp16 | 68.2 / 64.4 | 1.06 | 1.05-1.07 | 130 / 116 | 1.13 | slower heavier shipped-slower |
| segnext-mscan-t-ade20k | fp32 | 36.8 / 32.7 | 1.13 | 1.09-1.13 | 208 / 208 | 1.00 | slower shipped-slower |
| mvsanywhere | fp16 | 408.0 / 363.1 | 1.12 | 1.12-1.14 | 382 / 382 | 1.00 | slower shipped-slower |
| rtdetrv2-l-coco | fp32 | 61.5 / 66.8 | 0.92 | 0.81-1.03 | 296 / 264 | 1.12 | heavier |
| disk-lightglue | fp16 | 519.0 / 464.9 | 1.12 | 1.09-1.20 | 940 / 837 | 1.12 | slower heavier shipped-slower |
| detr-r101-coco | fp32 | 133.0 / 144.5 | 0.92 | 0.84-0.92 | 352 / 315 | 1.12 | heavier |
| igev-plusplus-eth3d | fp32 | 7446.8 / 8025.1 | 0.93 | 0.87-1.06 | 4170 / 3729 | 1.12 | heavier |
| igev-plusplus-kitti12 | fp32 | 15223.8 / 15031.8 | 1.01 | 0.69-1.56 | 4170 / 3729 | 1.12 | heavier |
| igev-plusplus-kitti15 | fp32 | 7159.7 / 8034.9 | 0.89 | 0.87-0.92 | 4170 / 3729 | 1.12 | heavier shipped-slower |
| igev-plusplus-middlebury | fp32 | 7244.0 / 7900.0 | 0.92 | 0.85-0.95 | 4170 / 3729 | 1.12 | heavier shipped-slower |
| igev-plusplus-sceneflow | fp32 | 7113.1 / 7667.9 | 0.93 | 0.91-0.95 | 4170 / 3729 | 1.12 | heavier shipped-slower |
| segformer-b2-clothes | fp32 | 33.4 / 37.0 | 0.90 | 0.84-1.00 | 175 / 156 | 1.12 | heavier shipped-slower |
| owlv2-l14 | fp32 | 1869.2 / 1858.2 | 1.01 | 1.00-1.01 | 417 / 376 | 1.11 | heavier |
| hrnet-w48-384 | fp16 | 398.0 / 432.0 | 0.92 | 0.92-0.93 | 273 / 247 | 1.11 | heavier |

## Accepted: faster, peak trade

`heavier` only, warm at most 0.90 of the graph's, the end-to-end MLX peak (survey row) below the best MPS row and at most the ref's MLX float32 row (`accepted`).

| ref | dtype | net warm ms hand / graph | x | transient MB hand / graph | x | e2e peak MB | best MPS MB | MLX fp32 MB |
|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | fp16 | 681.1 / 1356.2 | 0.50 | 2141 / 1881 | 1.14 | 2259 | 3754 | 3104 |
| aanet-sceneflow | fp16 | 449.4 / 1244.0 | 0.36 | 1012 / 888 | 1.14 | 1362 | 2594 | 2140 |
| aanet-sceneflow | fp32 | 650.8 / 1286.9 | 0.51 | 2022 / 888 | 2.28 | 2140 | 2594 | 2140 |
| bisenet-face-parsing | fp16 | 10.3 / 16.2 | 0.64 | 123 / 87 | 1.41 | 371 | 440 | 500 |
| davit_base.msft_in1k | fp32 | 21.0 / 24.5 | 0.86 | 98 / 81 | 1.21 | 852 | 1967 | 852 |
| deim-l-coco | fp32 | 47.5 / 55.7 | 0.85 | 365 / 275 | 1.33 | 972 | 1806 | 972 |
| deim-x-coco | fp32 | 81.2 / 93.5 | 0.87 | 503 / 474 | 1.06 | 1288 | 2207 | 1288 |
| densenet121.tv_in1k | fp16 | 8.6 / 11.0 | 0.78 | 65 / 57 | 1.14 | 416 | 704 | 476 |
| densenet121.tv_in1k | fp32 | 10.0 / 13.1 | 0.76 | 118 / 112 | 1.05 | 476 | 704 | 476 |
| detr-r50-coco | fp32 | 84.9 / 94.9 | 0.89 | 352 / 290 | 1.21 | 1603 | 1954 | 1603 |
| dfine-l-coco | fp32 | 47.9 / 54.7 | 0.88 | 365 / 275 | 1.33 | 1024 | 1810 | 1024 |
| dfine-x-coco | fp32 | 79.6 / 91.7 | 0.87 | 503 / 474 | 1.06 | 1422 | 2211 | 1422 |
| dlnr-middlebury | fp16 | 1672.0 / 1985.6 | 0.84 | 1470 / 1027 | 1.43 | 2841 | 4149 | 3856 |
| dlnr-middlebury | fp32 | 2214.7 / 2801.1 | 0.79 | 2637 / 1947 | 1.35 | 3856 | 4149 | 3856 |
| dlnr-sceneflow | fp16 | 1659.1 / 1970.3 | 0.84 | 1470 / 1027 | 1.43 | 2844 | 4146 | 3849 |
| dlnr-sceneflow | fp32 | 2069.6 / 2832.2 | 0.73 | 2637 / 1947 | 1.35 | 3849 | 4146 | 3849 |
| dwpose-l | fp16 | 57.5 / 72.0 | 0.80 | 243 / 212 | 1.15 | 1144 | 1866 | 1397 |
| dwpose-l | fp32 | 63.8 / 81.4 | 0.78 | 451 / 320 | 1.41 | 1397 | 1866 | 1397 |
| dwpose-l-384 | fp16 | 112.8 / 137.1 | 0.82 | 309 / 298 | 1.04 | 1232 | 1872 | 1340 |
| dwpose-l-384 | fp32 | 128.4 / 166.4 | 0.77 | 618 / 390 | 1.58 | 1340 | 1872 | 1340 |
| dwpose-s | fp32 | 18.9 / 24.4 | 0.77 | 246 / 224 | 1.10 | 727 | 1543 | 727 |
| hat-x4 | fp32 | 10448.9 / 20066.3 | 0.52 | 4353 / 4095 | 1.06 | 5181 | 6650 | 5181 |
| igev-plusplus-eth3d | fp16 | 5810.7 / 7016.1 | 0.83 | 2350 / 2170 | 1.08 | 2953 | 4971 | 4217 |
| igev-plusplus-kitti12 | fp16 | 8964.0 / 13448.5 | 0.67 | 2350 / 2170 | 1.08 | 3016 | 4967 | 3926 |
| igev-plusplus-middlebury | fp16 | 5758.9 / 7138.5 | 0.81 | 2350 / 2170 | 1.08 | 2920 | 4966 | 3936 |
| igev-plusplus-sceneflow | fp16 | 5824.1 / 7542.0 | 0.77 | 2350 / 2160 | 1.09 | 2949 | 4972 | 3981 |
| igev-stereo-eth3d | fp16 | 3216.8 / 3667.7 | 0.88 | 1626 / 1031 | 1.58 | 2551 | 3789 | 3531 |
| igev-stereo-kitti12 | fp16 | 3205.7 / 3695.7 | 0.87 | 1626 / 1054 | 1.54 | 2559 | 3791 | 3445 |
| igev-stereo-kitti15 | fp16 | 3184.4 / 3614.1 | 0.88 | 1661 / 984 | 1.69 | 2570 | 3791 | 3426 |
| igev-stereo-kitti15 | fp32 | 4165.1 / 4787.5 | 0.87 | 3074 / 2005 | 1.53 | 3426 | 3791 | 3426 |
| igev-stereo-middlebury | fp16 | 3192.1 / 3633.4 | 0.88 | 1661 / 1054 | 1.57 | 2555 | 3792 | 3425 |
| igev-stereo-sceneflow | fp16 | 3190.5 / 3621.4 | 0.88 | 1661 / 984 | 1.69 | 2558 | 3793 | 3441 |
| isnet | fp16 | 111.0 / 124.9 | 0.89 | 484 / 410 | 1.18 | 1005 | 1745 | 1361 |
| isnet | fp32 | 139.8 / 162.0 | 0.86 | 891 / 700 | 1.27 | 1361 | 1745 | 1361 |
| isnet-general-use | fp16 | 111.0 / 126.9 | 0.87 | 484 / 408 | 1.19 | 1016 | 1746 | 1355 |
| isnet-general-use | fp32 | 147.9 / 165.7 | 0.89 | 891 / 700 | 1.27 | 1355 | 1746 | 1355 |
| lineart | fp32 | 97.1 / 117.3 | 0.83 | 1167 / 887 | 1.32 | 815 | 1361 | 815 |
| loftr-indoor | fp32 | 432.4 / 523.1 | 0.83 | 2079 / 1218 | 1.71 | 2293 | 3613 | 2293 |
| loftr-outdoor | fp32 | 442.1 / 528.8 | 0.84 | 2079 / 1234 | 1.68 | 2513 | 3614 | 2513 |
| lw-detr-xlarge-coco | fp16 | 96.8 / 108.0 | 0.90 | 433 / 226 | 1.91 | 1440 | 1765 | 1579 |
| nafnet-gopro | fp16 | 196.2 / 250.7 | 0.78 | 538 / 517 | 1.04 | 1295 | 1715 | 1686 |
| nafnet-gopro | fp32 | 257.3 / 293.7 | 0.88 | 907 / 475 | 1.91 | 1686 | 1715 | 1686 |
| nafnet-sidd | fp16 | 184.2 / 208.3 | 0.88 | 434 / 324 | 1.33 | 1252 | 1986 | 1735 |
| nafnet-sidd | fp32 | 233.5 / 269.5 | 0.87 | 739 / 454 | 1.63 | 1735 | 1986 | 1735 |
| oneformer-swinl-ade20k | fp16 | 565.1 / 690.3 | 0.82 | 832 / 507 | 1.64 | 2137 | 5559 | 2917 |
| raft-stereo-eth3d | fp16 | 3409.4 / 4391.4 | 0.78 | 1176 / 1084 | 1.08 | 2719 | 4714 | 3704 |
| retinaface-resnet50 | fp32 | 151.2 / 173.0 | 0.87 | 774 / 549 | 1.41 | 1291 | 1879 | 1291 |
| rtdetrv2-l-coco | fp16 | 48.6 / 55.6 | 0.87 | 187 / 176 | 1.06 | 831 | 1823 | 1027 |
| rtdetrv2-ms-coco | fp32 | 44.5 / 52.3 | 0.85 | 286 / 265 | 1.08 | 958 | 1821 | 958 |
| rtdetrv2-x-coco | fp32 | 92.0 / 104.1 | 0.88 | 441 / 393 | 1.12 | 1416 | 2017 | 1416 |
| rtmpose-l | fp16 | 48.9 / 62.6 | 0.78 | 243 / 212 | 1.15 | 1054 | 1841 | 1325 |
| rtmpose-l | fp32 | 56.7 / 78.2 | 0.72 | 451 / 320 | 1.41 | 1325 | 1841 | 1325 |
| rtmpose-l-384 | fp16 | 98.0 / 122.8 | 0.80 | 309 / 298 | 1.04 | 1159 | 1844 | 1313 |
| rtmpose-l-384 | fp32 | 111.4 / 156.1 | 0.71 | 618 / 402 | 1.53 | 1313 | 1844 | 1313 |
| rtmpose-l-halpe26 | fp16 | 51.0 / 62.6 | 0.81 | 243 / 212 | 1.15 | 1077 | 1847 | 1326 |
| rtmpose-l-halpe26 | fp32 | 56.0 / 76.3 | 0.73 | 451 / 320 | 1.41 | 1326 | 1847 | 1326 |
| rtmpose-l-halpe26-384 | fp16 | 96.9 / 120.3 | 0.81 | 309 / 298 | 1.04 | 1164 | 1851 | 1353 |
| rtmpose-l-halpe26-384 | fp32 | 109.6 / 152.5 | 0.72 | 618 / 390 | 1.58 | 1353 | 1851 | 1353 |
| rtmpose-s-face6 | fp16 | 20.8 / 24.3 | 0.86 | 170 / 158 | 1.08 | 709 | 1518 | 823 |
| rtmpose-s-face6 | fp32 | 22.5 / 28.7 | 0.78 | 288 / 218 | 1.32 | 823 | 1518 | 823 |
| rtmw-l-384 | fp32 | 224.7 / 268.7 | 0.84 | 618 / 415 | 1.49 | 1644 | 2076 | 1644 |
| rtmw-x | fp16 | 176.7 / 205.5 | 0.86 | 332 / 259 | 1.28 | 1605 | 2266 | 2033 |
| rtmw-x | fp32 | 203.3 / 234.5 | 0.87 | 608 / 479 | 1.27 | 2033 | 2266 | 2033 |
| rtmw-x-384 | fp32 | 422.1 / 484.6 | 0.87 | 804 / 741 | 1.08 | 2216 | 2268 | 2216 |
| sam-hq-vit-b | fp16 | 309.0 / 397.2 | 0.78 | 432 / 358 | 1.21 | 1374 | 4374 | 1712 |
| sam-hq-vit-h | fp16 | 1559.6 / 1733.7 | 0.90 | 523 / 398 | 1.31 | 2466 | 7860 | 3747 |
| sam-hq-vit-l | fp16 | 825.2 / 943.6 | 0.87 | 510 / 386 | 1.32 | 1884 | 6205 | 2571 |
| sam-vit-b | fp16 | 299.4 / 385.9 | 0.78 | 426 / 352 | 1.21 | 1069 | 3762 | 1412 |
| sam-vit-l | fp16 | 815.6 / 936.1 | 0.87 | 502 / 378 | 1.33 | 1639 | 5193 | 2342 |
| sam2.1-hiera-large | fp16 | 497.7 / 579.6 | 0.86 | 347 / 300 | 1.16 | 1243 | 2206 | 1737 |
| selective-igev-eth3d-finetune | fp16 | 3693.7 / 4643.9 | 0.80 | 1848 / 1075 | 1.72 | 2776 | 3854 | 3947 |
| selective-igev-eth3d-train | fp16 | 3652.2 / 4644.1 | 0.79 | 1848 / 1075 | 1.72 | 2777 | 3862 | 3935 |
| selective-igev-kitti12 | fp16 | 3643.3 / 4640.5 | 0.79 | 1848 / 1075 | 1.72 | 2764 | 3853 | 3938 |
| selective-igev-kitti15 | fp16 | 3672.6 / 4688.7 | 0.78 | 1848 / 1075 | 1.72 | 2774 | 3853 | 3934 |
| selective-igev-middlebury-finetune | fp16 | 3655.1 / 4663.3 | 0.78 | 1848 / 1075 | 1.72 | 2932 | 3852 | 3952 |
| selective-igev-middlebury-train | fp16 | 3651.5 / 4657.9 | 0.78 | 1848 / 1075 | 1.72 | 2779 | 3854 | 3941 |
| selective-igev-sceneflow | fp16 | 3643.8 / 4680.4 | 0.78 | 1848 / 1075 | 1.72 | 2781 | 3907 | 3938 |
| span-x2 | fp32 | 113.4 / 149.7 | 0.76 | 659 / 611 | 1.08 | 977 | 1296 | 977 |
| span-x2 | fp32 | 114.2 / 152.3 | 0.75 | 656 / 595 | 1.10 | 977 | 1296 | 977 |
| span-x4 | fp32 | 116.7 / 156.6 | 0.75 | 659 / 611 | 1.08 | 1068 | 1296 | 1068 |
| span-x4 | fp32 | 117.5 / 155.7 | 0.75 | 656 / 611 | 1.07 | 1068 | 1296 | 1068 |
| u2netp | fp16 | 21.5 / 26.5 | 0.81 | 122 / 116 | 1.05 | 504 | 1400 | 622 |
| u2netp | fp32 | 26.5 / 31.8 | 0.83 | 241 / 230 | 1.05 | 622 | 1400 | 622 |
| xfeat | fp32 | 17.5 / 30.7 | 0.57 | 182 / 162 | 1.12 | 642 | 1471 | 642 |
| xfeat-lighterglue | fp32 | 17.3 / 29.6 | 0.58 | 182 / 162 | 1.12 | 816 | 1499 | 816 |
| yolox-l | fp16 | 50.9 / 61.1 | 0.83 | 185 / 177 | 1.05 | 1021 | 1829 | 1307 |
| yolox-l | fp32 | 61.7 / 82.0 | 0.75 | 370 / 298 | 1.24 | 1307 | 1829 | 1307 |
| yolox-x | fp32 | 112.0 / 138.3 | 0.81 | 446 / 432 | 1.03 | 1675 | 2036 | 1675 |
| yunet-n | fp32 | 12.3 / 15.4 | 0.80 | 343 / 271 | 1.26 | 745 | 1377 | 745 |
| yunet-s | fp32 | 11.6 / 14.1 | 0.83 | 279 / 271 | 1.03 | 693 | 1380 | 693 |

## Resident: float16 against float32

The hand arm's resident memory (weights and what the network keeps) in float16 over float32: about 0.5 where the weights are all float16. Of the 299 refs keeping at least 32 MB in float32, 46 are above 0.6 (float32 islands, or buffers that do not scale with the dtype), largest first.

| ref | resident MB fp16 / fp32 | x | graph fp16 / fp32 |
|---|---|---|---|
| realesrgan-x2plus | 163 / 65 | 2.50 | 165 / 65 |
| realesrgan-x4plus | 163 / 65 | 2.50 | 165 / 65 |
| disk-lightglue | 50 / 50 | 1.01 | 5 / 4 |
| aanet-plus-sceneflow | 33 / 33 | 1.00 | 33 / 49 |
| aliked-lightglue | 49 / 49 | 1.00 | 2 / 2 |
| totalsegmentator-total | 567 / 595 | 0.95 | 298 / 595 |
| loma-b128 | 112 / 119 | 0.94 | 60 / 120 |
| lw-detr-tiny-coco | 30 / 36 | 0.83 | 19 / 37 |
| raft-stereo-eth3d | 35 / 42 | 0.83 | 82 / 116 |
| raft-stereo-middlebury | 35 / 42 | 0.83 | 82 / 116 |
| raft-stereo-sceneflow | 35 / 42 | 0.83 | 82 / 116 |
| loma-b64 | 79 / 96 | 0.83 | 48 / 96 |
| lw-detr-small-coco | 34 / 43 | 0.79 | 24 / 47 |
| lw-detr-large-coco | 118 / 154 | 0.77 | 81 / 160 |
| dwpose-s | 24 / 32 | 0.76 | 24 / 32 |
| neuflow-v2 | 26 / 35 | 0.74 | 36 / 45 |
| neuflow-v2-sintel | 26 / 35 | 0.74 | 36 / 45 |
| neuflow-v2-things | 26 / 35 | 0.74 | 36 / 45 |
| totalsegmentator-total-3mm | 45 / 63 | 0.71 | 32 / 63 |
| rf-detr-keypoint-preview | 95 / 138 | 0.69 | 71 / 142 |
| rtmw-m | 85 / 123 | 0.69 | 84 / 123 |
| dwpose-m | 47 / 69 | 0.68 | 45 / 69 |
| lw-detr-xlarge-coco | 288 / 427 | 0.68 | 219 / 434 |
| sam2.1-hiera-tiny | 129 / 192 | 0.67 | 73 / 136 |
| rtmpose-m-face6 | 43 / 65 | 0.66 | 42 / 65 |
| rf-detr-seg-xl-coco | 87 / 132 | 0.66 | 68 / 136 |
| rf-detr-seg-2xl-coco | 88 / 134 | 0.66 | 69 / 138 |
| sam2.1-hiera-small | 142 / 219 | 0.65 | 86 / 163 |
| rtmw-l-384 | 142 / 219 | 0.65 | 140 / 219 |
| rtmw-l | 142 / 218 | 0.65 | 139 / 218 |
| rf-detr-seg-m-coco | 80 / 124 | 0.65 | 63 / 126 |
| rf-detr-seg-l-coco | 80 / 124 | 0.65 | 64 / 127 |
| loftr-indoor | 40 / 64 | 0.64 | 21 / 43 |
| loftr-outdoor | 40 / 64 | 0.64 | 21 / 43 |
| rf-detr-m-coco | 73 / 115 | 0.63 | 59 / 119 |
| dwpose-l-384 | 81 / 128 | 0.63 | 78 / 128 |
| rf-detr-l-coco | 73 / 116 | 0.63 | 60 / 120 |
| dwpose-l | 81 / 128 | 0.63 | 78 / 128 |
| rf-detr-seg-n-coco | 74 / 117 | 0.63 | 59 / 118 |
| rf-detr-seg-s-coco | 74 / 117 | 0.63 | 59 / 119 |
| lw-detr-medium-coco | 60 / 96 | 0.63 | 51 / 100 |
| rtmw-x-384 | 220 / 352 | 0.62 | 216 / 352 |
| rtmw-x | 219 / 351 | 0.62 | 215 / 351 |
| dlnr-middlebury | 142 / 229 | 0.62 | 181 / 314 |
| dlnr-sceneflow | 142 / 229 | 0.62 | 181 / 314 |
| rf-detr-s-coco | 67 / 109 | 0.61 | 56 / 113 |

## Per family

Flagged rows / rows (accepted in brackets), the worst hand / graph ratio of the family (warm, peak).

| family | flagged (accepted) | slower | heavier | worse | worst warm x | worst transient x |
|---|---|---|---|---|---|---|
| aanet | 3 (3) / 4 | 0 | 3 | 0 | 0.54 | 2.28 |
| alexnet | 0 (0) / 2 | 0 | 0 | 0 | 0.99 | 0.42 |
| anyline | 0 (0) / 2 | 0 | 0 | 0 | 0.64 | 0.65 |
| ben2 | 0 (0) / 2 | 0 | 0 | 0 | 0.96 | 0.72 |
| birefnet | 0 (0) / 6 | 0 | 0 | 0 | 0.92 | 0.50 |
| bisenet | 2 (1) / 2 | 0 | 2 | 0 | 0.70 | 1.41 |
| clip | 1 (0) / 20 | 0 | 0 | 0 | 1.04 | 0.62 |
| coatnet | 0 (0) / 20 | 0 | 0 | 0 | 1.01 | 0.69 |
| coex | 0 (0) / 2 | 0 | 0 | 0 | 0.50 | 0.52 |
| compact | 0 (0) / 4 | 0 | 0 | 0 | 0.93 | 1.00 |
| convnext | 2 (0) / 24 | 0 | 1 | 1 | 1.01 | 1.16 |
| convnextv2 | 0 (0) / 2 | 0 | 0 | 0 | 0.98 | 0.66 |
| dat | 1 (0) / 6 | 0 | 1 | 0 | 0.78 | 1.20 |
| davit | 3 (1) / 6 | 1 | 1 | 0 | 1.18 | 1.21 |
| deit | 0 (0) / 2 | 0 | 0 | 0 | 1.07 | 0.39 |
| densenet | 2 (2) / 2 | 0 | 2 | 0 | 0.78 | 1.14 |
| depth_anything_3 | 6 (0) / 10 | 4 | 1 | 0 | 1.07 | 1.20 |
| depth_pro | 0 (0) / 2 | 0 | 0 | 0 | 0.96 | 0.97 |
| detr | 2 (1) / 4 | 0 | 2 | 0 | 0.92 | 1.21 |
| dfine | 4 (4) / 20 | 0 | 4 | 0 | 0.92 | 1.33 |
| dlnr | 4 (4) / 4 | 0 | 4 | 0 | 0.84 | 1.43 |
| drct | 0 (0) / 4 | 0 | 0 | 0 | 0.72 | 0.45 |
| efficientformer | 3 (0) / 6 | 0 | 0 | 3 | 0.92 | 0.45 |
| efficientformerv2 | 0 (0) / 8 | 0 | 0 | 0 | 0.96 | 0.52 |
| efficientnet | 0 (0) / 4 | 0 | 0 | 0 | 0.83 | 0.67 |
| efficientnetv2 | 0 (0) / 2 | 0 | 0 | 0 | 0.87 | 0.42 |
| efficientvit | 0 (0) / 2 | 0 | 0 | 0 | 0.79 | 0.51 |
| efficientvit_sam | 2 (0) / 5 | 0 | 2 | 0 | 1.01 | 1.55 |
| eomt | 3 (0) / 4 | 2 | 2 | 0 | 1.03 | 1.07 |
| eva02 | 1 (0) / 2 | 0 | 1 | 0 | 1.00 | 1.16 |
| fastvit | 0 (0) / 6 | 0 | 0 | 0 | 0.99 | 0.45 |
| flexglue | 0 (0) / 2 | 0 | 0 | 0 | 0.98 | 0.67 |
| flowseek | 0 (0) / 6 | 0 | 0 | 0 | 0.71 | 0.64 |
| focalnet | 0 (0) / 12 | 0 | 0 | 0 | 1.00 | 0.53 |
| hat | 1 (1) / 6 | 0 | 1 | 0 | 0.78 | 1.06 |
| hgnetv2 | 0 (0) / 14 | 0 | 0 | 0 | 0.95 | 0.60 |
| hrnet | 2 (0) / 8 | 0 | 2 | 0 | 0.92 | 1.13 |
| igev_plusplus | 10 (4) / 10 | 0 | 10 | 0 | 1.01 | 1.12 |
| igev_stereo | 10 (6) / 10 | 0 | 10 | 0 | 0.91 | 1.69 |
| inception_v3 | 0 (0) / 2 | 0 | 0 | 0 | 0.76 | 0.20 |
| isnet | 4 (4) / 4 | 0 | 4 | 0 | 0.89 | 1.27 |
| lightglue | 1 (0) / 4 | 1 | 1 | 0 | 1.12 | 1.12 |
| lineart | 2 (1) / 4 | 0 | 2 | 0 | 0.83 | 1.32 |
| loftr | 2 (2) / 4 | 0 | 2 | 0 | 0.84 | 1.71 |
| loma | 2 (0) / 12 | 0 | 2 | 0 | 1.08 | 1.19 |
| lwdetr | 3 (1) / 10 | 0 | 3 | 0 | 1.00 | 1.91 |
| mask2former | 0 (0) / 2 | 0 | 0 | 0 | 0.91 | 0.93 |
| memflow | 0 (0) / 6 | 0 | 0 | 0 | 0.83 | 0.35 |
| memfof | 0 (0) / 8 | 0 | 0 | 0 | 0.94 | 0.96 |
| mmseg | 6 (0) / 10 | 0 | 6 | 0 | 1.01 | 1.42 |
| mobilenetv1 | 1 (0) / 2 | 0 | 0 | 1 | 1.01 | 0.55 |
| mobilenetv2 | 0 (0) / 2 | 0 | 0 | 0 | 0.97 | 0.94 |
| mobilenetv3 | 0 (0) / 2 | 0 | 0 | 0 | 0.78 | 0.46 |
| mobilenetv4 | 0 (0) / 2 | 0 | 0 | 0 | 0.91 | 0.57 |
| mvsanywhere | 1 (0) / 2 | 1 | 0 | 0 | 1.12 | 1.00 |
| nafnet | 4 (4) / 4 | 0 | 4 | 0 | 0.88 | 1.91 |
| neuflow_v2 | 0 (0) / 6 | 0 | 0 | 0 | 0.58 | 0.36 |
| nsfw_vit | 0 (0) / 2 | 0 | 0 | 0 | 1.01 | 0.39 |
| oneformer | 1 (1) / 2 | 0 | 1 | 0 | 0.82 | 1.64 |
| owlv2 | 4 (0) / 4 | 0 | 4 | 0 | 1.02 | 1.42 |
| perception_encoder | 5 (0) / 20 | 0 | 5 | 0 | 1.63 | 1.75 |
| raft_stereo | 5 (1) / 6 | 0 | 5 | 1 | 0.96 | 1.33 |
| rcnn | 0 (0) / 4 | 0 | 0 | 0 | 0.85 | 0.88 |
| realesrgan | 3 (0) / 12 | 1 | 3 | 0 | 1.04 | 1.25 |
| recover | 1 (0) / 6 | 1 | 0 | 0 | 1.06 | 0.98 |
| regnet | 0 (0) / 2 | 0 | 0 | 0 | 0.94 | 0.33 |
| repvit | 0 (0) / 20 | 0 | 0 | 0 | 0.79 | 0.36 |
| resnet | 0 (0) / 6 | 0 | 0 | 0 | 0.91 | 0.69 |
| retinaface | 2 (1) / 4 | 0 | 2 | 0 | 0.90 | 1.41 |
| rfdetr | 9 (0) / 20 | 4 | 2 | 3 | 1.10 | 1.23 |
| rfdetr_keypoint | 2 (0) / 2 | 0 | 1 | 0 | 1.02 | 1.80 |
| rtdetrv2 | 4 (3) / 10 | 0 | 4 | 0 | 0.92 | 1.12 |
| rtmdet | 0 (0) / 14 | 0 | 0 | 0 | 0.92 | 0.95 |
| rtmo | 0 (0) / 8 | 0 | 0 | 0 | 0.84 | 1.01 |
| rtmpose | 23 (19) / 56 | 0 | 23 | 0 | 0.98 | 1.58 |
| sam | 3 (2) / 6 | 0 | 3 | 0 | 0.90 | 1.33 |
| sam2 | 4 (1) / 8 | 0 | 4 | 0 | 1.01 | 1.17 |
| sam3 | 0 (0) / 2 | 0 | 0 | 0 | 0.88 | 0.84 |
| sam_hq | 3 (3) / 6 | 0 | 3 | 0 | 0.90 | 1.32 |
| sapiens | 2 (0) / 4 | 0 | 2 | 0 | 0.88 | 1.17 |
| segformer | 7 (0) / 8 | 4 | 6 | 0 | 1.22 | 1.49 |
| segnext | 7 (0) / 8 | 5 | 3 | 0 | 1.26 | 1.52 |
| selective_igev | 14 (7) / 14 | 0 | 14 | 0 | 0.80 | 1.72 |
| shufflenetv2 | 1 (0) / 2 | 0 | 0 | 1 | 1.03 | 0.87 |
| shvit | 3 (0) / 8 | 2 | 0 | 0 | 1.28 | 0.55 |
| siglip | 0 (0) / 2 | 0 | 0 | 0 | 1.02 | 0.82 |
| siglip2 | 7 (0) / 20 | 6 | 2 | 0 | 1.06 | 1.43 |
| span | 4 (4) / 8 | 0 | 4 | 0 | 0.84 | 1.10 |
| squeezenet | 0 (0) / 2 | 0 | 0 | 0 | 0.86 | 0.84 |
| starnet | 0 (0) / 8 | 0 | 0 | 0 | 0.87 | 0.45 |
| swin | 1 (0) / 2 | 0 | 1 | 0 | 0.99 | 1.36 |
| swin2sr | 1 (0) / 6 | 1 | 0 | 0 | 1.06 | 0.78 |
| swinir | 1 (0) / 8 | 1 | 0 | 0 | 1.06 | 0.87 |
| tinyvit | 1 (0) / 10 | 0 | 1 | 0 | 1.11 | 1.30 |
| totalsegmentator | 2 (0) / 4 | 0 | 2 | 0 | 0.78 | 1.28 |
| u2net | 2 (2) / 6 | 0 | 2 | 0 | 0.96 | 1.05 |
| unik3d | 1 (0) / 2 | 0 | 1 | 0 | 1.02 | 1.72 |
| vgg | 0 (0) / 2 | 0 | 0 | 0 | 0.97 | 0.65 |
| vit | 0 (0) / 2 | 0 | 0 | 0 | 1.05 | 0.43 |
| vitmatte | 0 (0) / 4 | 0 | 0 | 0 | 0.70 | 0.92 |
| vitpose | 5 (0) / 12 | 2 | 3 | 0 | 1.03 | 1.45 |
| wd_tagger | 3 (0) / 8 | 0 | 3 | 0 | 1.05 | 1.16 |
| xfeat | 2 (2) / 2 | 0 | 2 | 0 | 0.58 | 1.12 |
| yolox | 3 (3) / 10 | 0 | 3 | 0 | 0.89 | 1.24 |
| yunet | 2 (2) / 4 | 0 | 2 | 0 | 0.84 | 1.26 |

## Quality: hand arms further from the golden

Rule (`_worse`): hand golden error (primary output, mean or p99) above 1.25 x the graph's and above it by more than 5% of the golden's bound; sparse matchers also recall / match agreement lower by more than 0.02.

| ref | dtype | output | mean hand / graph | p99 hand / graph | bound mean / p99 |
|---|---|---|---|---|---|
| convnext_tiny.fb_in22k_ft_in1k | fp16 | logits | 0.002296 / 0.001922 | 0.007701 / 0.006048 | 0.0033 / 0.012 |
| efficientformer_l1.snap_dist_in1k | fp16 | logits | 0.0034 / 0.002783 | 0.01271 / 0.009823 | 0.0046 / 0.019 |
| efficientformer_l3.snap_dist_in1k | fp16 | logits | 0.004146 / 0.001349 | 0.01336 / 0.004563 | 0.0046 / 0.019 |
| efficientformer_l7.snap_dist_in1k | fp16 | logits | 0.002436 / 0.00189 | 0.00978 / 0.006661 | 0.0046 / 0.019 |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp16 | logits | 0.04653 / 0.03247 | 0.1244 / 0.09309 | 0.08 / 0.3 |
| raft-stereo-middlebury | fp16 | disparity | 0.006122 / 0.004216 | 0.05389 / 0.0337 | 0.02 / 0.25 |
| rf-detr-seg-2xl-coco | fp16 | boxes | 0.2784 / 0.2438 | 1.008 / 0.7939 | 0.51 / 2.7 |
| rf-detr-seg-m-coco | fp16 | boxes | 0.2891 / 0.2143 | 1.198 / 0.7643 | 0.37 / 1.4 |
| rf-detr-seg-xl-coco | fp32 | boxes | 0.07297 / 0.03045 | 0.4197 / 0.157 | 1.1 / 4.2 |
| shufflenet_v2_x1_0.tv_in1k | fp16 | logits | 0.005165 / 0.003629 | 0.0185 / 0.01276 | 0.02 / 0.08 |

## Full table

hand / graph per cell; `x` = hand / graph; `net rel` = max |hand - graph| / max |graph| over the first network call's outputs (where both arms give the same number of arrays), `result rel` the same over the model's results (a sanity check); `golden mean` = each arm's mean |diff| of the primary output against the stored golden on the golden's input (tests/test_golden.py `gpu_errors`, the bound in brackets; `rN` the sparse matchers' keypoint recall).

| ref | dtype | net warm ms | x | 95 % CI | shipped ms | net first s | net peak MB | transient MB | x | resident MB | e2e peak MB | e2e warm ms | load s | e2e first s | net rel | result rel | golden mean (bound) | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | fp16 | 681.1 / 1356.2 | 0.50 | 0.45-0.55 | 753.4 | 1.01 / 7.96 | 2211 / 1914 | 2141 / 1881 | 1.14 | 33 / 33 | 4034 / 4052 | 697.5 / 1304.6 | 0.22 / 0.20 | 1.03 / 8.00 | 0.0057 | 0.012 (0.00014 % off) | - | heavier |
| aanet-plus-sceneflow | fp32 | 850.1 / 1572.5 | 0.54 | - | - | 0.99 / 6.25 | 3207 / 3767 | ~3174 / ~3719 | 0.85 | 33 / 49 | 5111 / 5612 | 860.7 / 1576.4 | 0.14 / 0.12 | 1.00 / 6.26 | 3.5e-05 | 7.2e-05 | - |  |
| aanet-sceneflow | fp16 | 449.4 / 1244.0 | 0.36 | 0.34-0.37 | 498.4 | 0.53 / 6.93 | 1062 / 911 | 1012 / 888 | 1.14 | 16 / 23 | 3062 / 3502 | 455.8 / 1223.9 | 0.19 / 0.12 | 0.55 / 6.95 | 0.019 (0.00027 % off) | 0.0079 | - | heavier |
| aanet-sceneflow | fp32 | 650.8 / 1286.9 | 0.51 | 0.44-0.56 | 699.7 | 0.84 / 6.07 | 2072 / 918 | 2022 / 888 | 2.28 | 16 / 30 | 4096 / 3558 | 652.2 / 1338.7 | 0.17 / 0.14 | 0.86 / 6.09 | 0.0002 | 7.3e-05 | - | heavier |
| alexnet.tv_in1k | fp16 | 2.5 / 2.5 | 0.99 | - | - | 0.02 / 0.38 | 119 / 125 | ~3 / ~8 | 0.42 | 117 / 117 | 861 / 779 | 2.8 / 2.9 | 0.64 / 0.49 | 0.02 / 0.38 | 0 | 0 | - |  |
| alexnet.tv_in1k | fp32 | 3.3 / 3.4 | 0.97 | - | - | 0.03 / 0.38 | 238 / 248 | ~4 / ~15 | 0.35 | 233 / 233 | 746 / 781 | 3.6 / 3.8 | 0.62 / 0.49 | 0.03 / 0.38 | 0 | 0 | - |  |
| aliked-flexglue | fp16 | 932.4 / 954.6 | 0.98 | - | - | 4.20 / 4.84 | 800 / 1063 | ~538 / ~802 | 0.67 | 263 / 261 | 1937 / 2550 | 969.2 / 1000.3 | 0.10 / 0.10 | 4.26 / 4.90 | - | empty (4 keys differ) | - | calls-differ |
| aliked-flexglue | fp32 | 861.4 / 896.2 | 0.96 | - | - | 4.08 / 4.73 | 1057 / 1319 | ~538 / ~801 | 0.67 | 520 / 518 | 2186 / 2732 | 897.7 / 942.5 | 0.10 / 0.10 | 4.14 / 4.78 | - | 2.2e-06 | - | calls-differ |
| aliked-lightglue | fp16 | 89.6 / 164.7 | 0.54 | - | - | 0.14 / 1.63 | 714 / 1324 | ~665 / ~1323 | 0.50 | 49 / 2 | 1968 / 3503 | 88.1 / 308.3 | 0.08 / 0.05 | 0.16 / 1.82 | - | 1.7e-05 | - | calls-differ |
| aliked-lightglue | fp32 | 88.3 / 167.4 | 0.53 | - | - | 0.13 / 1.63 | 714 / 1324 | ~665 / ~1323 | 0.50 | 49 / 2 | 1969 / 3460 | 88.4 / 309.8 | 0.08 / 0.05 | 0.16 / 1.79 | - | 1.7e-05 | - | calls-differ |
| anyline | fp16 | 49.9 / 77.9 | 0.64 | - | - | 0.08 / 1.15 | 394 / 861 | ~394 / ~861 | 0.46 | 0 / 0 | 1198 / 2593 | 65.4 / 223.5 | 0.02 / 0.02 | 0.11 / 1.34 | - | 0.45 (0.27 % off) | - |  |
| anyline | fp32 | 66.4 / 122.9 | 0.54 | - | - | 0.10 / 1.10 | 743 / 1148 | ~743 / ~1148 | 0.65 | 0 / 0 | 1800 / 3500 | 83.1 / 270.1 | 0.03 / 0.02 | 0.13 / 1.28 | - | 0.16 (0.00019 % off) | - |  |
| ben2 | fp16 | 782.8 / 812.6 | 0.96 | - | - | 0.96 / 4.98 | 1638 / 2202 | ~1428 / ~2018 | 0.71 | 210 / 184 | 4179 / 4711 | 789.8 / 819.2 | 0.79 / 0.07 | 1.02 / 4.99 | 0.009 | 0.069 (0.35 % off) | 1.11e-05 / 1.11e-05 (2.4e-05) |  |
| ben2 | fp32 | 964.5 / 1189.1 | 0.81 | - | - | 1.10 / 5.64 | 2852 / 3748 | ~2447 / ~3384 | 0.72 | 404 / 364 | 4928 / 6378 | 990.4 / 1200.4 | 0.81 / 0.07 | 1.17 / 5.65 | 1.6e-05 | 8.2e-05 | 2.42e-08 / 1.58e-08 (2.4e-05) |  |
| birefnet | fp16 | 949.2 / 1027.6 | 0.92 | - | - | 1.06 / 7.68 | 1062 / 1745 | ~588 / ~1246 | 0.47 | 474 / 499 | 3438 / 4308 | 933.4 / 1261.1 | 1.29 / 0.16 | 1.13 / 7.69 | 1 (1e+02 % off) | 0.0089 | 5.29e-06 / 6.09e-06 (6e-06) |  |
| birefnet | fp32 | 1016.7 / 1455.8 | 0.70 | - | - | 1.43 / 8.56 | 2129 / 3386 | ~1214 / ~2444 | 0.50 | 915 / 941 | 4399 / 5968 | 1010.2 / 1409.3 | 1.11 / 0.17 | 1.50 / 8.57 | 1 (1e+02 % off) | 2.8e-05 | 1.72e-07 / 1.37e-07 (6e-06) |  |
| birefnet-lite | fp16 | 252.9 / 373.1 | 0.68 | - | - | 0.31 / 5.88 | 556 / 1016 | ~444 / ~912 | 0.49 | 112 / 103 | 2228 / 3578 | 263.0 / 371.0 | 0.69 / 0.07 | 0.37 / 5.88 | 1 (1e+02 % off) | 0.0097 | 3.77e-06 / 5.29e-06 (6e-06) |  |
| birefnet-lite | fp32 | 387.6 / 645.2 | 0.60 | - | - | 0.38 / 6.28 | 845 / 1989 | ~648 / ~1800 | 0.36 | 197 / 189 | 3186 / 4635 | 393.1 / 646.9 | 0.69 / 0.06 | 0.45 / 6.29 | 1 (1e+02 % off) | 2.3e-05 | 1.64e-07 / 1.27e-07 (6e-06) |  |
| bisenet-face-parsing | fp16 | 10.3 / 16.2 | 0.64 | - | - | 0.04 / 1.55 | 148 / 112 | ~123 / ~87 | 1.41 | 25 / 25 | 405 / 624 | 12.2 / 22.3 | 0.04 / 0.05 | 0.05 / 1.56 | 0.00026 | 0 | 0.00346 / 0.00347 (0.0042) | heavier |
| bisenet-face-parsing | fp32 | 12.0 / 17.1 | 0.70 | - | - | 0.04 / 1.48 | 259 / 246 | ~209 / ~196 | 1.07 | 50 / 50 | 515 / 735 | 13.8 / 23.4 | 0.04 / 0.05 | 0.05 / 1.50 | 1.1e-06 | 0 | 2.36e-06 / 2.31e-06 (0.0042) | heavier |
| clip-b16-laion2b | fp16 | 14.2 / 14.0 | 1.02 | - | - | 0.04 / 0.67 | 186 / 231 | ~22 / ~67 | 0.34 | 164 / 164 | 830 / 599 | 14.5 / 14.2 | 0.85 / 0.71 | 0.04 / 0.68 | 0.0043 | 0.0043 | - |  |
| clip-b16-laion2b | fp32 | 16.3 / 16.1 | 1.01 | - | - | 0.06 / 0.66 | 362 / 417 | ~35 / ~90 | 0.39 | 327 / 327 | 680 / 780 | 16.6 / 16.5 | 0.78 / 0.71 | 0.06 / 0.66 | 6e-06 | 6e-06 | - |  |
| clip-b16-openai | fp16 | 14.2 / 14.0 | 1.01 | 1.00-1.09 | 14.2 | 0.08 / 0.88 | 182 / 231 | 19 / 67 | 0.29 | 164 / 164 | 835 / 606 | 15.5 / 14.4 | 1.30 / 0.84 | 0.08 / 0.89 | 0.019 (0.00066 % off) | 0.019 (0.00066 % off) | - |  |
| clip-b16-openai | fp32 | 16.4 / 16.1 | 1.02 | - | - | 0.06 / 0.68 | 362 / 404 | ~35 / ~77 | 0.46 | 327 / 327 | 677 / 764 | 16.6 / 16.4 | 0.79 / 0.72 | 0.06 / 0.68 | 2.4e-05 | 2.4e-05 | - |  |
| clip-h14-378-dfn5b | fp16 | 109.3 / 105.0 | 1.04 | 0.99-1.08 | 108.8 | 0.20 / 4.88 | 1237 / 1277 | 34 / 73 | 0.47 | 1203 / 1203 | 3757 / 1671 | 109.0 / 106.1 | 5.01 / 0.96 | 0.21 / 5.02 | 0.064 (0.025 % off) | 0.064 (0.025 % off) | - |  |
| clip-h14-378-dfn5b | fp32 | 127.4 / 125.0 | 1.02 | - | - | 0.69 / 1.66 | 2469 / 2524 | ~62 / ~117 | 0.53 | 2407 / 2407 | 2799 / 2938 | 127.6 / 125.4 | 2.11 / 0.74 | 0.69 / 1.66 | 1.8e-05 | 1.8e-05 | - |  |
| clip-h14-dfn5b | fp16 | 109.9 / 107.8 | 1.02 | 1.00-1.04 | 108.5 | 0.19 / 4.73 | 1241 / 1277 | 38 / 73 | 0.52 | 1203 / 1203 | 3752 / 1672 | 110.1 / 107.6 | 5.14 / 1.03 | 0.19 / 4.82 | 0.032 (0.034 % off) | 0.032 (0.034 % off) | - |  |
| clip-h14-dfn5b | fp32 | 127.4 / 125.0 | 1.02 | - | - | 0.25 / 1.66 | 2469 / 2524 | ~62 / ~117 | 0.53 | 2407 / 2407 | 2792 / 2924 | 127.8 / 125.3 | 1.59 / 0.73 | 0.25 / 1.66 | 3.6e-05 | 3.6e-05 | - |  |
| clip-h14-laion2b | fp16 | 110.9 / 109.1 | 1.02 | 0.98-1.11 | 110.9 | 0.19 / 4.29 | 1249 / 1277 | 46 / 73 | 0.62 | 1203 / 1203 | 3753 / 1675 | 111.3 / 110.6 | 5.13 / 0.93 | 0.19 / 4.38 | 0.017 (0.0021 % off) | 0.017 (0.0021 % off) | - |  |
| clip-h14-laion2b | fp32 | 131.9 / 128.5 | 1.03 | 1.00-1.04 | 129.9 | 0.67 / 5.04 | 2465 / 2524 | 59 / 117 | 0.50 | 2407 / 2407 | 2802 / 2900 | 135.2 / 128.4 | 4.38 / 0.96 | 0.68 / 5.40 | 8.6e-06 | 8.6e-06 | - |  |
| clip-l14-336-openai | fp16 | 54.9 / 54.4 | 1.01 | 1.00-1.01 | 56.8 | 0.15 / 1.76 | 608 / 658 | 30 / 79 | 0.38 | 578 / 578 | 2094 / 1058 | 55.2 / 54.8 | 2.06 / 0.90 | 0.15 / 1.80 | 0.022 (0.0023 % off) | 0.022 (0.0023 % off) | - |  |
| clip-l14-336-openai | fp32 | 64.2 / 63.2 | 1.02 | 1.01-1.02 | 64.5 | 0.23 / 1.50 | 1205 / 1283 | 48 / 127 | 0.38 | 1156 / 1157 | 1532 / 1670 | 65.1 / 63.6 | 1.40 / 0.87 | 0.23 / 1.56 | 1.5e-05 | 1.5e-05 | - |  |
| clip-l14-datacompxl | fp16 | 55.1 / 54.0 | 1.02 | - | - | 0.07 / 1.10 | 608 / 658 | ~30 / ~79 | 0.38 | 578 / 578 | 2084 / 1044 | 55.3 / 54.2 | 0.98 / 0.71 | 0.07 / 1.11 | 0.0063 | 0.0063 | - |  |
| clip-l14-datacompxl | fp32 | 64.3 / 65.8 | 0.98 | 0.98-1.02 | 64.3 | 0.22 / 1.70 | 1205 / 1283 | 48 / 127 | 0.38 | 1156 / 1157 | 1530 / 1663 | 64.8 / 66.6 | 1.72 / 0.86 | 0.23 / 1.77 | 1e-05 | 1e-05 | - |  |
| clip-l14-dfn2b | fp16 | 57.1 / 55.2 | 1.03 | 1.00-1.05 | 58.4 | 0.16 / 1.62 | 618 / 658 | 40 / 79 | 0.51 | 578 / 578 | 2098 / 1050 | 57.4 / 56.2 | 2.04 / 0.81 | 0.17 / 1.66 | 0.05 (0.18 % off) | 0.05 (0.18 % off) | - | shipped-slower |
| clip-l14-dfn2b | fp32 | 66.0 / 64.7 | 1.02 | 1.02-1.04 | 65.5 | 0.22 / 1.51 | 1205 / 1283 | 48 / 127 | 0.38 | 1156 / 1157 | 1529 / 1670 | 66.3 / 65.2 | 1.31 / 0.87 | 0.22 / 1.59 | 7.5e-05 | 7.5e-05 | - |  |
| clip-l14-laion2b | fp16 | 55.0 / 54.0 | 1.02 | - | - | 0.08 / 1.11 | 608 / 658 | ~30 / ~79 | 0.38 | 578 / 578 | 2085 / 1044 | 55.4 / 54.3 | 0.99 / 0.71 | 0.08 / 1.11 | 0.013 (0.00076 % off) | 0.013 (0.00076 % off) | - |  |
| clip-l14-laion2b | fp32 | 65.1 / 64.5 | 1.01 | 0.97-1.02 | 67.4 | 0.29 / 1.70 | 1205 / 1283 | 48 / 127 | 0.38 | 1156 / 1157 | 1532 / 1662 | 65.7 / 66.6 | 1.70 / 0.87 | 0.29 / 1.76 | 2.6e-05 | 2.6e-05 | - |  |
| clip-l14-openai | fp16 | 55.6 / 54.9 | 1.01 | 0.99-1.02 | 55.9 | 0.16 / 1.64 | 608 / 658 | 30 / 79 | 0.38 | 578 / 578 | 2090 / 1050 | 56.2 / 55.6 | 2.13 / 0.88 | 0.16 / 1.68 | 0.0046 | 0.0046 | - |  |
| clip-l14-openai | fp32 | 65.1 / 63.8 | 1.02 | 1.01-1.03 | 65.3 | 0.15 / 1.24 | 1205 / 1283 | 48 / 127 | 0.38 | 1156 / 1157 | 1530 / 1673 | 65.6 / 64.1 | 1.23 / 0.82 | 0.15 / 1.31 | 4.3e-06 | 4.3e-06 | - |  |
| coatnet_0_rw_224.sw_in1k | fp16 | 7.7 / 7.8 | 0.98 | - | - | 0.04 / 0.86 | 100 / 162 | ~42 / ~103 | 0.41 | 58 / 58 | 501 / 536 | 8.2 / 8.4 | 0.83 / 0.73 | 0.04 / 0.87 | 0.0012 | 0.0012 | 0.00104 / 0.0013 (0.0024) |  |
| coatnet_0_rw_224.sw_in1k | fp32 | 8.5 / 8.9 | 0.96 | - | - | 0.04 / 0.81 | 186 / 272 | ~70 / ~155 | 0.45 | 116 / 117 | 574 / 646 | 9.1 / 9.5 | 0.78 / 0.74 | 0.05 / 0.81 | 7.9e-07 | 7.9e-07 | 4.08e-06 / 4.06e-06 (0.0024) |  |
| coatnet_1_rw_224.sw_in1k | fp16 | 12.7 / 12.8 | 0.99 | - | - | 0.05 / 1.31 | 145 / 248 | ~54 / ~155 | 0.35 | 92 / 92 | 573 / 657 | 13.2 / 13.4 | 0.87 / 0.71 | 0.05 / 1.31 | 0.00052 | 0.00052 | 0.000584 / 0.000565 (0.0024) |  |
| coatnet_1_rw_224.sw_in1k | fp32 | 14.1 / 15.3 | 0.92 | - | - | 0.05 / 1.26 | 251 / 340 | ~68 / ~156 | 0.44 | 183 / 184 | 691 / 812 | 14.8 / 15.9 | 0.83 / 0.72 | 0.05 / 1.26 | 3.8e-07 | 3.8e-07 | 3.59e-06 / 3.63e-06 (0.0024) |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.1 / 19.6 | 0.87 | - | - | 0.05 / 1.42 | 230 / 343 | ~73 / ~186 | 0.40 | 157 / 157 | 719 / 762 | 17.8 / 20.5 | 0.92 / 0.76 | 0.06 / 1.43 | 0.0027 | 0.0027 | 0.00145 / 0.00164 (0.0024) |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.5 / 24.2 | 0.81 | - | - | 0.09 / 1.26 | 424 / 492 | ~110 / ~178 | 0.62 | 314 / 314 | 910 / 1032 | 20.3 / 24.7 | 0.96 / 0.74 | 0.09 / 1.26 | 2.9e-06 | 2.9e-06 | 7.66e-06 / 8.45e-06 (0.0024) |  |
| coatnet_nano_rw_224.sw_in1k | fp16 | 5.9 / 7.8 | 0.76 | - | - | 0.04 / 1.00 | 71 / 128 | ~39 / ~96 | 0.41 | 32 / 32 | 445 / 500 | 6.5 / 9.0 | 0.82 / 0.78 | 0.04 / 1.00 | 0.0041 | 0.0041 | 0.00111 / 0.000939 (0.0024) |  |
| coatnet_nano_rw_224.sw_in1k | fp32 | 8.2 / 9.2 | 0.90 | - | - | 0.11 / 0.95 | 118 / 205 | ~53 / ~140 | 0.38 | 65 / 65 | 498 / 574 | 8.8 / 10.1 | 0.85 / 0.78 | 0.11 / 0.96 | 5.6e-07 | 5.6e-07 | 2.08e-06 / 2.13e-06 (0.0024) |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp16 | 12.4 / 12.9 | 0.96 | - | - | 0.05 / 1.52 | 138 / 245 | ~46 / ~153 | 0.30 | 92 / 92 | 575 / 670 | 13.0 / 13.6 | 0.89 / 0.74 | 0.05 / 1.52 | 0.0051 | 0.0051 | 0.0012 / 0.00144 (0.0024) |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp32 | 13.7 / 15.4 | 0.89 | - | - | 0.06 / 1.44 | 275 / 340 | ~92 / ~156 | 0.59 | 183 / 184 | 690 / 837 | 14.4 / 16.0 | 0.87 / 0.73 | 0.07 / 1.45 | 3.6e-07 | 3.6e-07 | 3.01e-06 / 3.03e-06 (0.0024) |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp16 | 12.3 / 12.2 | 1.01 | - | - | 0.07 / 1.61 | 153 / 238 | ~61 / ~146 | 0.42 | 92 / 92 | 572 / 659 | 12.9 / 12.8 | 0.89 / 0.73 | 0.07 / 1.61 | 0.001 | 0.001 | 0.00145 / 0.00168 (0.0024) |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp32 | 13.7 / 14.8 | 0.93 | - | - | 0.05 / 1.51 | 244 / 350 | ~61 / ~167 | 0.37 | 183 / 184 | 679 / 808 | 14.2 / 15.5 | 0.89 / 0.76 | 0.06 / 1.51 | 3.9e-07 | 3.9e-07 | 3.39e-06 / 3.25e-06 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.7 / 19.9 | 0.89 | - | - | 0.05 / 1.69 | 230 / 343 | ~73 / ~186 | 0.40 | 157 / 157 | 742 / 775 | 18.0 / 20.6 | 0.98 / 0.76 | 0.06 / 1.70 | 0.0041 | 0.0041 | 0.00127 / 0.00102 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.4 / 24.3 | 0.80 | - | - | 0.06 / 1.54 | 424 / 491 | ~110 / ~177 | 0.62 | 314 / 314 | 932 / 1055 | 20.0 / 25.0 | 0.91 / 0.77 | 0.06 / 1.54 | 6.7e-07 | 6.7e-07 | 4.87e-06 / 4.81e-06 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp16 | 17.3 / 19.9 | 0.87 | - | - | 0.07 / 1.65 | 240 / 343 | ~83 / ~186 | 0.45 | 157 / 157 | 741 / 773 | 18.4 / 20.5 | 1.01 / 0.73 | 0.07 / 1.65 | 0.0015 | 0.0015 | 0.000714 / 0.00131 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp32 | 19.4 / 24.3 | 0.80 | - | - | 0.07 / 1.67 | 424 / 491 | ~110 / ~177 | 0.62 | 314 / 314 | 936 / 1059 | 20.1 / 25.0 | 0.92 / 0.79 | 0.07 / 1.68 | 4e-07 | 4e-07 | 2.7e-06 / 2.82e-06 (0.0024) |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp16 | 42.6 / 48.6 | 0.88 | - | - | 0.08 / 1.72 | 390 / 475 | ~111 / ~195 | 0.57 | 279 / 280 | 1091 / 1271 | 43.4 / 48.9 | 1.03 / 0.78 | 0.08 / 1.72 | 0.0017 | 0.0017 | 0.00116 / 0.00114 (0.0024) |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp32 | 49.3 / 61.4 | 0.80 | - | - | 0.10 / 1.75 | 740 / 823 | ~182 / ~264 | 0.69 | 558 / 559 | 1481 / 1871 | 49.8 / 61.2 | 0.99 / 0.76 | 0.10 / 1.76 | 4.8e-07 | 4.8e-07 | 6.73e-06 / 6.68e-06 (0.0024) |  |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp16 | 6.0 / 6.5 | 0.92 | - | - | 0.04 / 1.02 | 76 / 136 | ~44 / ~104 | 0.43 | 32 / 32 | 446 / 516 | 6.5 / 7.1 | 0.83 / 0.73 | 0.04 / 1.03 | 0.0014 | 0.0014 | 0.000958 / 0.00107 (0.0024) |  |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp32 | 6.4 / 7.4 | 0.87 | - | - | 0.04 / 0.98 | 130 / 227 | ~65 / ~162 | 0.41 | 65 / 65 | 503 / 606 | 6.9 / 8.2 | 0.79 / 0.75 | 0.04 / 0.98 | 2.7e-07 | 2.7e-07 | 2.41e-06 / 2.61e-06 (0.0024) |  |
| coex-sceneflow | fp16 | 197.2 / 409.9 | 0.48 | - | - | 0.24 / 2.21 | 550 / 1669 | ~539 / ~1663 | 0.32 | 11 / 6 | 1893 / 4319 | 202.2 / 423.0 | 0.08 / 0.07 | 0.25 / 2.22 | 0.31 (0.1 % off) | 0.034 (0.0055 % off) | - |  |
| coex-sceneflow | fp32 | 258.0 / 518.0 | 0.50 | - | - | 0.29 / 2.27 | 1020 / 1943 | ~1009 / ~1932 | 0.52 | 11 / 11 | 3135 / 4510 | 264.1 / 515.7 | 0.08 / 0.06 | 0.30 / 2.28 | 2.8e-06 | 3e-06 | - |  |
| convnext_base.fb_in1k | fp16 | 14.7 / 15.1 | 0.97 | - | - | 0.06 / 1.08 | 196 / 278 | ~27 / ~108 | 0.25 | 170 / 169 | 610 / 659 | 15.1 / 16.6 | 0.96 / 0.72 | 0.06 / 1.08 | 0.00068 | 0.00068 | 0.000653 / 0.0006 (0.0033) |  |
| convnext_base.fb_in1k | fp32 | 16.6 / 17.4 | 0.95 | - | - | 0.07 / 1.02 | 383 / 448 | ~45 / ~110 | 0.41 | 338 / 338 | 780 / 856 | 17.2 / 18.0 | 0.91 / 0.71 | 0.07 / 1.02 | 1.7e-07 | 1.7e-07 | 3.64e-06 / 3.59e-06 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k | fp16 | 14.6 / 15.0 | 0.97 | - | - | 0.05 / 1.05 | 196 / 278 | ~26 / ~108 | 0.25 | 170 / 169 | 611 / 659 | 15.3 / 15.8 | 0.91 / 0.72 | 0.05 / 1.05 | 0.0012 | 0.0012 | 0.000844 / 0.000932 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k | fp32 | 16.2 / 17.5 | 0.93 | - | - | 0.08 / 1.01 | 383 / 448 | ~45 / ~110 | 0.41 | 338 / 338 | 778 / 856 | 16.9 / 18.2 | 0.87 / 0.71 | 0.08 / 1.01 | 5.1e-07 | 5.1e-07 | 5.16e-06 / 5.15e-06 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k_384 | fp16 | 30.0 / 31.3 | 0.96 | - | - | 0.06 / 1.08 | 232 / 310 | ~62 / ~141 | 0.45 | 170 / 169 | 691 / 745 | 31.0 / 32.1 | 0.93 / 0.74 | 0.06 / 1.08 | 0.0031 | 0.0031 | 0.00106 / 0.000972 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k_384 | fp32 | 35.2 / 37.4 | 0.94 | - | - | 0.09 / 1.03 | 441 / 464 | ~102 / ~125 | 0.82 | 338 / 338 | 907 / 939 | 36.1 / 38.3 | 0.88 / 0.73 | 0.09 / 1.03 | 5e-07 | 5e-07 | 5.64e-06 / 5.63e-06 (0.0033) |  |
| convnext_large.fb_in1k | fp16 | 26.9 / 28.0 | 0.96 | - | - | 0.06 / 1.08 | 411 / 478 | ~33 / ~100 | 0.34 | 378 / 378 | 932 / 880 | 27.5 / 28.7 | 1.13 / 0.71 | 0.06 / 1.08 | 0.00062 | 0.00062 | 0.000625 / 0.000639 (0.0033) |  |
| convnext_large.fb_in1k | fp32 | 31.7 / 33.5 | 0.95 | - | - | 0.09 / 1.07 | 828 / 866 | ~73 / ~111 | 0.66 | 755 / 755 | 1299 / 1317 | 32.3 / 34.2 | 1.07 / 0.71 | 0.09 / 1.08 | 4.6e-07 | 4.6e-07 | 5.14e-06 / 5.17e-06 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k | fp16 | 26.9 / 28.0 | 0.96 | - | - | 0.07 / 1.08 | 414 / 478 | ~37 / ~100 | 0.37 | 378 / 378 | 933 / 879 | 27.5 / 28.7 | 1.14 / 0.74 | 0.07 / 1.08 | 0.0015 | 0.0015 | 0.000805 / 0.000907 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k | fp32 | 31.6 / 33.3 | 0.95 | - | - | 0.09 / 1.08 | 826 / 866 | ~71 / ~111 | 0.64 | 755 / 755 | 1297 / 1315 | 32.3 / 33.9 | 1.08 / 0.73 | 0.09 / 1.08 | 6.6e-07 | 6.6e-07 | 9.16e-06 / 9.12e-06 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp16 | 59.1 / 60.6 | 0.98 | - | - | 0.09 / 1.11 | 466 / 495 | ~89 / ~117 | 0.76 | 378 / 378 | 1046 / 989 | 59.9 / 61.6 | 1.11 / 0.71 | 0.09 / 1.12 | 0.00096 | 0.00096 | 0.00107 / 0.00114 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp32 | 67.3 / 71.0 | 0.95 | 0.94-0.95 | 68.6 | 0.12 / 1.09 | 913 / 892 | 159 / 137 | 1.16 | 755 / 755 | 1471 / 1452 | 68.3 / 71.9 | 2.35 / 0.75 | 0.12 / 1.16 | 4e-07 | 4e-07 | 9.03e-06 / 9.01e-06 (0.0033) | heavier |
| convnext_small.fb_in1k | fp16 | 10.5 / 10.7 | 0.98 | - | - | 0.05 / 1.05 | 118 / 185 | ~22 / ~89 | 0.26 | 96 / 96 | 498 / 558 | 11.0 / 11.3 | 0.86 / 0.72 | 0.05 / 1.05 | 0.00074 | 0.00074 | 0.000578 / 0.000627 (0.0033) |  |
| convnext_small.fb_in1k | fp32 | 11.8 / 12.5 | 0.94 | - | - | 0.06 / 1.00 | 225 / 302 | ~34 / ~110 | 0.31 | 192 / 192 | 604 / 678 | 12.4 / 13.1 | 0.81 / 0.71 | 0.06 / 1.00 | 3.6e-07 | 3.6e-07 | 3.01e-06 / 2.97e-06 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k | fp16 | 10.5 / 10.7 | 0.98 | - | - | 0.04 / 1.03 | 114 / 185 | ~18 / ~89 | 0.21 | 96 / 96 | 496 / 559 | 11.0 / 11.2 | 0.89 / 0.70 | 0.05 / 1.03 | 0.0014 | 0.0014 | 0.000852 / 0.000909 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k | fp32 | 11.8 / 12.4 | 0.95 | - | - | 0.06 / 0.99 | 225 / 302 | ~34 / ~110 | 0.31 | 192 / 192 | 600 / 678 | 12.4 / 13.1 | 0.82 / 0.70 | 0.06 / 1.00 | 5e-07 | 5e-07 | 5e-06 / 5.01e-06 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k_384 | fp16 | 21.0 / 21.2 | 0.99 | - | - | 0.05 / 1.05 | 143 / 233 | ~47 / ~137 | 0.35 | 96 / 96 | 559 / 656 | 21.6 / 22.0 | 0.91 / 0.71 | 0.06 / 1.05 | 0.00084 | 0.00084 | 0.00128 / 0.0014 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k_384 | fp32 | 23.8 / 24.9 | 0.95 | - | - | 0.07 / 1.01 | 293 / 308 | ~101 / ~116 | 0.87 | 192 / 192 | 703 / 784 | 24.5 / 25.6 | 0.81 / 0.75 | 0.07 / 1.02 | 3.4e-07 | 3.4e-07 | 5.74e-06 / 5.72e-06 (0.0033) |  |
| convnext_tiny.fb_in1k | fp16 | 6.0 / 6.0 | 1.00 | - | - | 0.04 / 0.68 | 71 / 108 | ~16 / ~53 | 0.31 | 55 / 55 | 443 / 471 | 6.5 / 6.6 | 0.84 / 0.73 | 0.04 / 0.69 | 0.00079 | 0.00079 | 0.000808 / 0.000935 (0.0033) |  |
| convnext_tiny.fb_in1k | fp32 | 6.7 / 6.8 | 1.00 | - | - | 0.06 / 0.66 | 144 / 206 | ~35 / ~97 | 0.36 | 109 / 109 | 506 / 567 | 7.2 / 7.4 | 0.78 / 0.70 | 0.06 / 0.66 | 6.3e-07 | 6.3e-07 | 4.47e-06 / 4.5e-06 (0.0033) |  |
| convnext_tiny.fb_in22k_ft_in1k | fp16 | 6.1 / 6.6 | 0.92 | 0.61-1.10 | 6.1 | 0.05 / 0.82 | 72 / 108 | 18 / 53 | 0.34 | 55 / 55 | 446 / 478 | 6.5 / 8.2 | 1.03 / 0.91 | 0.05 / 0.83 | 0.0012 | 0.0012 | 0.0023 / 0.00192 (0.0033) | worse |
| convnext_tiny.fb_in22k_ft_in1k | fp32 | 6.7 / 6.8 | 0.99 | - | - | 0.05 / 0.66 | 144 / 206 | ~35 / ~97 | 0.36 | 109 / 109 | 503 / 568 | 7.2 / 7.3 | 0.78 / 0.70 | 0.05 / 0.67 | 5.3e-07 | 5.3e-07 | 4.78e-06 / 4.79e-06 (0.0033) |  |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp16 | 12.1 / 12.0 | 1.01 | - | - | 0.04 / 0.69 | 97 / 161 | ~42 / ~107 | 0.40 | 55 / 55 | 500 / 534 | 13.0 / 12.8 | 0.85 / 0.69 | 0.04 / 0.69 | 0.0013 | 0.0013 | 0.00222 / 0.00224 (0.0033) |  |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp32 | 14.0 / 14.3 | 0.98 | - | - | 0.06 / 0.68 | 190 / 221 | ~81 / ~112 | 0.73 | 109 / 109 | 603 / 659 | 14.7 / 15.0 | 0.77 / 0.70 | 0.07 / 0.68 | 9.3e-07 | 9.3e-07 | 6.58e-06 / 6.75e-06 (0.0033) |  |
| convnextv2_base.fcmae_ft_in22k_in1k | fp16 | 17.9 / 18.1 | 0.98 | - | - | 0.20 / 1.69 | 244 / 318 | ~74 / ~148 | 0.50 | 170 / 170 | 680 / 700 | 18.4 / 19.1 | 1.37 / 0.84 | 0.20 / 1.69 | 0.0011 | 0.0011 | 0.000912 / 0.00137 (0.02) |  |
| convnextv2_base.fcmae_ft_in22k_in1k | fp32 | 20.9 / 21.4 | 0.98 | - | - | 0.42 / 1.66 | 415 / 456 | ~76 / ~116 | 0.66 | 339 / 339 | 849 / 913 | 21.3 / 26.5 | 1.73 / 0.84 | 0.43 / 1.66 | 4.1e-07 | 4.1e-07 | 4.9e-06 / 4.8e-06 (0.02) |  |
| da3-giant-1.1 | fp16 | 602.5 / 595.0 | 1.01 | 0.94-1.04 | 602.7 | 0.70 / 12.41 | 2729 / 2889 | 405 / 525 | 0.77 | 2321 / 2364 | 4791 / 4058 | 611.0 / 630.3 | 7.13 / 0.93 | 0.73 / 26.81 | 0.0057 | 0.0056 | - |  |
| da3-giant-1.1 | fp32 | 732.9 / 721.4 | 1.02 | - | - | 0.93 / 11.83 | 5376 / 5546 | ~805 / ~907 | 0.89 | 4571 / 4638 | 6757 / 7017 | 743.8 / 732.9 | 5.76 / 0.76 | 0.96 / 11.86 | 4.6e-06 | 4.2e-06 | - |  |
| da3-large-1.1 | fp16 | 215.3 / 211.0 | 1.02 | 1.01-1.04 | 218.8 | 0.33 / 4.10 | 1109 / 1096 | 417 / 346 | 1.20 | 690 / 750 | 2004 / 2266 | 222.3 / 218.7 | 1.74 / 0.85 | 0.35 / 4.19 | 0.0036 | 0.0032 | - | heavier shipped-slower |
| da3-large-1.1 | fp32 | 269.9 / 251.7 | 1.07 | 1.02-1.12 | 276.3 | 0.66 / 5.21 | 2148 / 2321 | 799 / 910 | 0.88 | 1346 / 1411 | 3372 / 3418 | 277.9 / 259.9 | 0.95 / 2.08 | 0.69 / 5.37 | 2.1e-06 | 2.1e-06 | - | slower shipped-slower |
| da3metric-large | fp16 | 206.9 / 198.5 | 1.04 | 1.03-1.05 | 210.0 | 0.29 / 2.93 | 1071 / 1102 | 433 / 463 | 0.93 | 636 / 639 | 1884 / 1814 | 225.2 / 208.6 | 1.88 / 0.77 | 0.33 / 3.26 | 0.018 (0.017 % off) | 0.017 (0.0054 % off) | - | slower shipped-slower |
| da3metric-large | fp32 | 250.0 / 237.9 | 1.05 | 1.04-1.05 | 272.3 | 0.55 / 2.88 | 2108 / 2195 | 834 / 917 | 0.91 | 1273 / 1278 | 3266 / 3128 | 269.4 / 255.8 | 1.17 / 0.76 | 0.61 / 2.98 | 2.7e-05 | 2.7e-05 | - | slower shipped-slower |
| da3mono-large | fp16 | 207.0 / 198.8 | 1.04 | 1.03-1.06 | 217.6 | 0.27 / 2.60 | 1071 / 1102 | 433 / 463 | 0.93 | 636 / 639 | 1881 / 1802 | 224.6 / 208.0 | 1.97 / 0.76 | 0.29 / 2.68 | 0.023 (0.0027 % off) | 1 (0.0013 % off) | - | slower shipped-slower |
| da3mono-large | fp32 | 250.8 / 239.9 | 1.05 | 1.02-1.06 | 565.6 | 0.64 / 3.84 | 2082 / 2195 | 807 / 917 | 0.88 | 1273 / 1278 | 3268 / 3126 | 271.1 / 260.5 | 1.28 / 0.80 | 0.70 / 4.09 | 1.3e-05 | 1.3e-05 | - | shipped-slower |
| da3nested-giant-large-1.1 | fp16 | 803.5 / 784.7 | 1.02 | 0.96-1.03 | 828.8 | 0.91 / 16.60 | 3392 / 3528 | 433 / 525 | 0.82 | 2958 / 3003 | 6070 / 4690 | 832.6 / 801.9 | 8.97 / 0.94 | 0.96 / 35.73 | 0.0029 | 0.017 (0.0041 % off) | - |  |
| da3nested-giant-large-1.1 | fp32 | 982.5 / 974.7 | 1.01 | - | - | 1.41 / 11.21 | 6652 / 6833 | ~809 / ~917 | 0.88 | 5844 / 5916 | 8238 / 8241 | 1010.0 / 1003.7 | 7.42 / 0.79 | 1.48 / 11.26 | 3.7e-06 | 2.7e-05 | - |  |
| dat-2-x4 | fp16 | 4832.4 / 9060.1 | 0.53 | - | - | 5.14 / 15.26 | 1974 / 2848 | ~1917 / ~2813 | 0.68 | 57 / 35 | 5275 / 6272 | 4790.4 / 8974.4 | 0.89 / 0.37 | 5.16 / 15.27 | 0.0089 | 0.0099 | - |  |
| dat-2-x4 | fp32 | 7227.0 / 15112.8 | 0.48 | - | - | 7.78 / 15.19 | 3943 / 4156 | ~3831 / ~4086 | 0.94 | 112 / 70 | 7024 / 6755 | 7499.0 / 15665.8 | 0.71 / 0.37 | 7.80 / 15.20 | 5.6e-06 | 6.3e-06 | - |  |
| dat-light-x4 | fp16 | 759.3 / 2449.4 | 0.31 | - | - | 1.11 / 6.19 | 1194 / 1921 | ~1166 / ~1913 | 0.61 | 28 / 8 | 3884 / 4987 | 763.3 / 2449.0 | 0.56 / 0.15 | 1.13 / 6.19 | 0.0031 | 0.0035 | - |  |
| dat-light-x4 | fp32 | 1332.4 / 1712.8 | 0.78 | 0.76-0.79 | 1915.7 | 2.67 / 6.40 | 2385 / 1959 | 2326 / 1943 | 1.20 | 55 / 16 | 5050 / 4352 | 1329.5 / 1743.7 | 0.80 / 0.25 | 2.69 / 6.43 | 8.1e-06 | 9.2e-06 | - | heavier shipped-slower |
| dat-x4 | fp16 | 5330.8 / 11403.9 | 0.47 | - | - | 5.60 / 17.86 | 2137 / 2869 | ~2073 / ~2827 | 0.73 | 64 / 42 | 5516 / 6385 | 5323.7 / 11352.6 | 0.70 / 0.42 | 5.61 / 17.87 | 0.0063 | 0.0071 | - |  |
| dat-x4 | fp32 | 8481.3 / 17063.6 | 0.50 | - | - | 9.08 / 17.50 | 3999 / 4170 | ~3873 / ~4086 | 0.95 | 126 / 84 | 7149 / 6650 | 8699.1 / 17565.1 | 0.70 / 0.48 | 9.10 / 17.51 | 5.5e-06 | 6.2e-06 | - |  |
| davit_base.msft_in1k | fp16 | 19.0 / 19.7 | 0.96 | 0.89-1.05 | 19.7 | 0.15 / 2.15 | 224 / 248 | 56 / 80 | 0.71 | 168 / 168 | 664 / 669 | 19.8 / 20.4 | 1.86 / 0.82 | 0.15 / 2.18 | 0.0008 | 0.0008 | 0.000779 / 0.000801 (0.0026) |  |
| davit_base.msft_in1k | fp32 | 21.0 / 24.5 | 0.86 | 0.78-0.94 | 23.4 | 0.15 / 2.22 | 434 / 417 | 98 / 81 | 1.21 | 336 / 336 | 854 / 890 | 22.2 / 24.5 | 0.98 / 0.84 | 0.15 / 2.26 | 3.3e-07 | 3.3e-07 | 6.87e-06 / 6.92e-06 (0.0026) | heavier |
| davit_small.msft_in1k | fp16 | 15.6 / 14.6 | 1.07 | 0.99-1.18 | 16.4 | 0.09 / 2.18 | 135 / 178 | 40 / 83 | 0.49 | 95 / 95 | 530 / 574 | 16.2 / 15.2 | 1.45 / 0.82 | 0.10 / 2.20 | 0.0015 | 0.0015 | 0.00115 / 0.00111 (0.0026) | shipped-slower |
| davit_small.msft_in1k | fp32 | 17.0 / 16.8 | 1.02 | 0.97-1.05 | 16.7 | 0.08 / 2.01 | 262 / 272 | 72 / 82 | 0.88 | 190 / 190 | 650 / 706 | 18.2 / 17.7 | 0.92 / 0.80 | 0.09 / 2.02 | 4e-07 | 4e-07 | 6.59e-06 / 6.61e-06 (0.0026) |  |
| davit_tiny.msft_in1k | fp16 | 9.1 / 7.7 | 1.18 | 1.15-1.21 | 9.1 | 0.04 / 1.27 | 95 / 123 | 40 / 69 | 0.59 | 54 / 54 | 484 / 497 | 9.9 / 8.5 | 1.13 / 0.79 | 0.04 / 1.29 | 0.00087 | 0.00087 | 0.0013 / 0.00136 (0.0026) | slower shipped-slower |
| davit_tiny.msft_in1k | fp32 | 10.3 / 10.3 | 1.00 | - | - | 0.15 / 1.29 | 180 / 179 | ~72 / ~71 | 1.01 | 108 / 108 | 558 / 585 | 11.0 / 10.6 | 0.86 / 0.82 | 0.15 / 1.29 | 5.3e-07 | 5.3e-07 | 3.74e-06 / 3.78e-06 (0.0026) |  |
| deeplabv3-r101-d8-ade20k | fp16 | 367.1 / 379.9 | 0.97 | 0.94-0.97 | 369.0 | 0.48 / 2.06 | 682 / 574 | 514 / 413 | 1.25 | 162 / 162 | 1908 / 1780 | 403.1 / 412.7 | 0.44 / 0.16 | 0.52 / 2.13 | 0.0007 | 0 | - | heavier |
| deeplabv3-r101-d8-ade20k | fp32 | 545.7 / 539.2 | 1.01 | 0.95-1.05 | 550.0 | 0.66 / 2.19 | 1224 / 1312 | 895 / 988 | 0.91 | 323 / 323 | 2330 / 2423 | 579.3 / 567.1 | 0.43 / 0.35 | 0.71 / 2.28 | 3.4e-07 | 0 | - |  |
| deeplabv3plus-r101-d8-ade20k | fp16 | 270.8 / 289.8 | 0.93 | 0.92-1.01 | 285.7 | 0.34 / 1.87 | 773 / 674 | 652 / 559 | 1.17 | 115 / 115 | 1860 / 1792 | 320.5 / 329.4 | 0.35 / 0.14 | 0.39 / 1.94 | 0.0011 | 0 | - | heavier |
| deeplabv3plus-r101-d8-ade20k | fp32 | 371.3 / 404.3 | 0.92 | 0.85-0.95 | 405.2 | 0.66 / 2.31 | 1404 / 1209 | 1169 / 980 | 1.19 | 230 / 230 | 2478 / 2418 | 396.3 / 453.2 | 0.35 / 0.15 | 0.72 / 2.41 | 4.2e-07 | 0 | - | heavier |
| deim-l-coco | fp16 | 40.8 / 46.2 | 0.88 | - | - | 0.07 / 2.72 | 249 / 274 | ~187 / ~212 | 0.88 | 63 / 62 | 941 / 1033 | 48.3 / 54.1 | 0.34 / 0.16 | 0.08 / 2.73 | 0.89 (48 % off) | 0.0013 | - |  |
| deim-l-coco | fp32 | 47.5 / 55.7 | 0.85 | 0.80-0.87 | 49.1 | 0.12 / 2.75 | 483 / 393 | 365 / 275 | 1.33 | 118 / 118 | 1314 / 1284 | 56.1 / 65.0 | 0.49 / 0.17 | 0.13 / 2.78 | 6.9e-06 | 1.1e-06 | - | heavier |
| deim-m-coco | fp16 | 27.4 / 33.6 | 0.82 | - | - | 0.09 / 2.44 | 197 / 232 | ~157 / ~193 | 0.82 | 40 / 40 | 727 / 897 | 35.5 / 42.5 | 0.27 / 0.13 | 0.10 / 2.45 | 0.89 (56 % off) | 0.0036 | - |  |
| deim-m-coco | fp32 | 31.7 / 40.2 | 0.79 | - | - | 0.09 / 2.26 | 381 / 376 | ~307 / ~301 | 1.02 | 74 / 74 | 979 / 1163 | 39.4 / 48.3 | 0.18 / 0.13 | 0.10 / 2.28 | 0.49 (0.95 % off) | 2.6e-06 | - |  |
| deim-n-coco | fp16 | 9.1 / 11.9 | 0.77 | - | - | 0.06 / 1.94 | 58 / 139 | ~49 / ~131 | 0.38 | 8 / 8 | 401 / 578 | 17.2 / 19.9 | 0.18 / 0.09 | 0.07 / 1.96 | 0.96 (59 % off) | 0.71 (2.5 % off) | - |  |
| deim-n-coco | fp32 | 10.3 / 13.5 | 0.76 | - | - | 0.07 / 1.83 | 106 / 175 | ~91 / ~161 | 0.57 | 15 / 15 | 468 / 660 | 18.4 / 20.8 | 0.11 / 0.09 | 0.08 / 1.85 | 1.7e-05 | 1.5e-06 | - |  |
| deim-s-coco | fp16 | 16.3 / 20.2 | 0.80 | - | - | 0.07 / 2.10 | 129 / 193 | ~107 / ~170 | 0.63 | 22 / 22 | 509 / 713 | 24.1 / 28.1 | 0.21 / 0.10 | 0.09 / 2.11 | 0.97 (49 % off) | empty (3 keys differ) | - |  |
| deim-s-coco | fp32 | 18.4 / 23.9 | 0.77 | - | - | 0.08 / 2.02 | 191 / 206 | ~151 / ~167 | 0.91 | 40 / 40 | 619 / 823 | 26.1 / 31.9 | 0.14 / 0.10 | 0.09 / 2.04 | 4.2e-06 | 1.7e-06 | - |  |
| deim-x-coco | fp16 | 69.8 / 75.9 | 0.92 | - | - | 0.10 / 3.00 | 376 / 380 | ~254 / ~258 | 0.98 | 122 / 122 | 1614 / 1557 | 78.2 / 84.2 | 0.49 / 0.22 | 0.11 / 3.01 | 0.84 (42 % off) | 0.0068 | - |  |
| deim-x-coco | fp32 | 81.2 / 93.5 | 0.87 | 0.85-0.89 | 85.1 | 0.30 / 3.19 | 740 / 710 | 503 / 474 | 1.06 | 236 / 236 | 2337 / 2157 | 91.2 / 102.0 | 0.66 / 0.24 | 0.31 / 3.23 | 0.37 (0.59 % off) | 7.7e-07 | - | heavier |
| deit_base_patch16_224.fb_in1k | fp16 | 14.2 / 14.0 | 1.01 | 0.96-1.07 | 14.5 | 0.14 / 0.89 | 186 / 232 | 21 / 67 | 0.33 | 165 / 165 | 400 / 596 | 15.4 / 14.7 | 0.38 / 1.68 | 0.15 / 0.91 | 0.00093 | 0.00093 | 0.00164 / 0.00122 (0.02) |  |
| deit_base_patch16_224.fb_in1k | fp32 | 18.2 / 17.1 | 1.07 | 1.01-1.14 | 17.4 | 0.08 / 0.79 | 365 / 420 | 35 / 90 | 0.39 | 330 / 330 | 580 / 782 | 19.1 / 18.0 | 0.27 / 0.84 | 0.09 / 0.82 | 6.8e-07 | 6.8e-07 | 7.18e-06 / 7.13e-06 (0.02) |  |
| densenet121.tv_in1k | fp16 | 8.6 / 11.0 | 0.78 | 0.74-0.81 | 8.5 | 0.03 / 1.61 | 81 / 73 | 65 / 57 | 1.14 | 16 / 16 | 418 / 466 | 9.2 / 11.6 | 0.84 / 0.67 | 0.04 / 1.62 | 0.0017 | 0.0017 | - | heavier |
| densenet121.tv_in1k | fp32 | 10.0 / 13.1 | 0.76 | 0.73-0.78 | 10.1 | 0.03 / 1.53 | 148 / 142 | 118 / 112 | 1.05 | 30 / 31 | 479 / 538 | 10.7 / 13.8 | 0.73 / 0.67 | 0.04 / 1.55 | 5.4e-07 | 5.4e-07 | - | heavier |
| depth-pro | fp16 | 4168.0 / 4341.7 | 0.96 | - | - | 4.27 / 9.37 | 4508 / 4609 | ~2699 / ~2794 | 0.97 | 1809 / 1815 | 6478 / 7153 | 4170.1 / 4666.0 | 3.88 / 3.42 | 4.28 / 9.38 | 0.0017 | 0.0011 | - |  |
| depth-pro | fp32 | 5497.0 / 6831.3 | 0.80 | - | - | 5.96 / 10.71 | 9014 / 9191 | ~5398 / ~5561 | 0.97 | 3616 / 3630 | 10220 / 11841 | 5552.0 / 6573.6 | 4.63 / 3.41 | 5.99 / 10.74 | 1.5e-06 | 1.3e-06 | - |  |
| detr-r101-coco | fp16 | 107.0 / 117.1 | 0.91 | - | - | 0.14 / 2.29 | 310 / 365 | ~194 / ~248 | 0.78 | 116 / 117 | 989 / 1400 | 119.0 / 127.5 | 0.07 / 0.15 | 0.16 / 2.31 | 0.0044 | empty | - |  |
| detr-r101-coco | fp32 | 133.0 / 144.5 | 0.92 | 0.84-0.92 | 138.4 | 0.18 / 2.22 | 596 / 547 | 352 / 315 | 1.12 | 232 / 232 | 1670 / 1804 | 146.2 / 159.7 | 0.22 / 0.28 | 0.21 / 2.26 | 2.4e-06 | empty | - | heavier |
| detr-r50-coco | fp16 | 69.1 / 78.2 | 0.88 | - | - | 0.10 / 1.86 | 274 / 343 | ~194 / ~262 | 0.74 | 80 / 81 | 953 / 1407 | 80.1 / 89.7 | 0.06 / 0.10 | 0.12 / 1.88 | 0.0078 | 0.00069 | - |  |
| detr-r50-coco | fp32 | 84.9 / 94.9 | 0.89 | 0.89-0.90 | 87.1 | 0.12 / 2.16 | 524 / 450 | 352 / 290 | 1.21 | 160 / 160 | 1597 / 1728 | 98.0 / 107.0 | 0.16 / 0.20 | 0.14 / 2.20 | 6.2e-06 | 3.7e-07 | - | heavier |
| dfine-l-coco | fp16 | 40.9 / 46.5 | 0.88 | - | - | 0.07 / 2.62 | 249 / 274 | ~186 / ~212 | 0.88 | 63 / 62 | 943 / 1056 | 48.1 / 54.1 | 0.29 / 0.10 | 0.08 / 2.63 | 0.97 (42 % off) | 0.97 (23 % off) | - |  |
| dfine-l-coco | fp32 | 47.9 / 54.7 | 0.88 | 0.84-0.89 | 49.5 | 0.10 / 2.80 | 483 / 393 | 365 / 275 | 1.33 | 118 / 118 | 1322 / 1302 | 55.5 / 64.1 | 0.35 / 0.11 | 0.11 / 2.83 | 1e-05 | 3.4e-06 | - | heavier |
| dfine-m-coco | fp16 | 27.2 / 34.0 | 0.80 | - | - | 0.08 / 2.39 | 197 / 232 | ~157 / ~193 | 0.82 | 40 / 40 | 729 / 910 | 34.7 / 41.8 | 0.22 / 0.08 | 0.09 / 2.40 | 0.95 (61 % off) | 0.96 (46 % off) | - |  |
| dfine-m-coco | fp32 | 31.8 / 40.2 | 0.79 | - | - | 0.09 / 2.25 | 381 / 376 | ~307 / ~301 | 1.02 | 74 / 74 | 983 / 1152 | 39.6 / 48.8 | 0.13 / 0.08 | 0.10 / 2.26 | 0.63 (0.6 % off) | 1.4e-05 | - |  |
| dfine-n-coco | fp16 | 9.0 / 11.8 | 0.76 | - | - | 0.06 / 1.93 | 58 / 139 | ~49 / ~131 | 0.38 | 8 / 8 | 402 / 583 | 16.9 / 19.6 | 0.14 / 0.06 | 0.08 / 1.95 | 0.95 (53 % off) | 0.97 (30 % off) | - |  |
| dfine-n-coco | fp32 | 10.3 / 13.7 | 0.75 | - | - | 0.07 / 1.85 | 106 / 175 | ~91 / ~161 | 0.57 | 15 / 15 | 473 / 664 | 17.6 / 22.1 | 0.08 / 0.06 | 0.08 / 1.86 | 6e-06 | 9e-07 | - |  |
| dfine-s-coco | fp16 | 16.2 / 20.2 | 0.80 | - | - | 0.07 / 2.10 | 129 / 193 | ~107 / ~170 | 0.63 | 22 / 22 | 511 / 701 | 24.1 / 28.4 | 0.17 / 0.07 | 0.09 / 2.12 | 0.96 (57 % off) | 1 (38 % off) | - |  |
| dfine-s-coco | fp32 | 18.4 / 23.7 | 0.77 | - | - | 0.08 / 1.99 | 191 / 206 | ~151 / ~167 | 0.91 | 40 / 40 | 624 / 814 | 26.4 / 31.7 | 0.10 / 0.06 | 0.09 / 2.00 | 8.8e-06 | 2e-05 | - |  |
| dfine-x-coco | fp16 | 69.8 / 76.0 | 0.92 | - | - | 0.10 / 2.92 | 376 / 380 | ~254 / ~258 | 0.98 | 122 / 122 | 1616 / 1564 | 77.8 / 83.7 | 0.39 / 0.15 | 0.11 / 2.94 | 1.1 (52 % off) | empty (3 keys differ) | - |  |
| dfine-x-coco | fp32 | 79.6 / 91.7 | 0.87 | 0.86-0.88 | 83.6 | 0.33 / 3.31 | 740 / 710 | 503 / 474 | 1.06 | 236 / 236 | 2348 / 2177 | 89.2 / 102.5 | 0.55 / 0.16 | 0.34 / 3.36 | 0.00025 | 4.9e-06 | - | heavier |
| disk-lightglue | fp16 | 519.0 / 464.9 | 1.12 | 1.09-1.20 | 635.7 | 1.73 / 2.34 | 990 / 842 | 940 / 837 | 1.12 | 50 / 5 | 3614 / 3758 | 665.7 / 683.7 | 0.20 / 0.07 | 2.00 / 3.13 | 0 | 7.4e-06 | - | slower heavier shipped-slower calls-differ |
| disk-lightglue | fp32 | 558.2 / 549.9 | 1.02 | - | - | 1.62 / 1.68 | 967 / 920 | ~917 / ~916 | 1.00 | 50 / 4 | 3642 / 3781 | 675.9 / 702.0 | 0.07 / 0.05 | 1.82 / 1.86 | 0 | 7.7e-06 | - | calls-differ |
| dlnr-middlebury | fp16 | 1672.0 / 1985.6 | 0.84 | 0.81-0.90 | 1685.3 | 2.01 / 8.18 | 1648 / 1208 | 1470 / 1027 | 1.43 | 142 / 181 | 3868 / 3518 | 1676.7 / 1983.9 | 0.72 / 0.25 | 2.02 / 8.21 | 0.33 (0.15 % off) | 0.33 (0.027 % off) | - | heavier |
| dlnr-middlebury | fp32 | 2214.7 / 2801.1 | 0.79 | 0.75-0.82 | 2297.2 | 2.41 / 8.66 | 2901 / 2261 | 2637 / 1947 | 1.35 | 229 / 314 | 4773 / 4698 | 2178.9 / 2855.3 | 1.00 / 0.24 | 2.42 / 8.70 | 0.02 (0.00091 % off) | 0.016 (0.00054 % off) | - | heavier |
| dlnr-sceneflow | fp16 | 1659.1 / 1970.3 | 0.84 | 0.81-0.86 | 1677.7 | 1.90 / 7.99 | 1647 / 1209 | 1470 / 1027 | 1.43 | 142 / 181 | 3866 / 3502 | 1668.3 / 1972.5 | 0.70 / 0.24 | 1.91 / 8.02 | 0.019 (0.0068 % off) | 0.017 (0.0013 % off) | - | heavier |
| dlnr-sceneflow | fp32 | 2069.6 / 2832.2 | 0.73 | 0.72-0.74 | 2299.3 | 2.27 / 8.83 | 2901 / 2261 | 2637 / 1947 | 1.35 | 229 / 314 | 4784 / 4687 | 2094.1 / 2820.4 | 1.16 / 0.24 | 2.28 / 8.87 | 1.2e-05 | 1.2e-05 | - | heavier |
| drct-l-x4 | fp16 | 8173.1 / 11297.8 | 0.72 | - | - | 8.45 / 20.83 | 1083 / 2518 | ~976 / ~2172 | 0.45 | 106 / 345 | 4171 / 5731 | 8295.1 / 11414.6 | 0.53 / 0.95 | 8.47 / 20.84 | 0.01 (1.6e-05 % off) | 0.011 (3.2e-05 % off) | - |  |
| drct-l-x4 | fp32 | 13143.6 / 23086.9 | 0.57 | - | - | 12.48 / 29.68 | 2130 / 5678 | ~1922 / ~5247 | 0.37 | 208 / 432 | 5014 / 8505 | 12237.2 / 23323.1 | 0.95 / 0.97 | 12.50 / 29.70 | 1e-05 | 1.1e-05 | - |  |
| drct-x4 | fp16 | 4250.7 / 6130.6 | 0.69 | - | - | 4.54 / 11.53 | 1040 / 2474 | ~976 / ~2173 | 0.45 | 64 / 301 | 4251 / 5606 | 4168.5 / 6226.9 | 0.54 / 0.32 | 4.56 / 11.54 | 0.025 (0.003 % off) | 0.029 (0.0047 % off) | - |  |
| drct-x4 | fp32 | 6124.1 / 11220.6 | 0.55 | - | - | 6.79 / 15.07 | 2046 / 5592 | ~1922 / ~5247 | 0.37 | 123 / 345 | 4861 / 8310 | 6147.2 / 11288.8 | 0.75 / 0.37 | 6.82 / 15.09 | 8.1e-06 | 9.3e-06 | - |  |
| dwpose-l | fp16 | 57.5 / 72.0 | 0.80 | 0.76-0.81 | 59.1 | 0.08 / 1.59 | 332 / 290 | 243 / 212 | 1.15 | 81 / 78 | 1215 / 1345 | 69.9 / 82.2 | 0.31 / 0.10 | 0.09 / 1.62 | - | 0.44 (0.099 % off) | - | heavier calls-differ |
| dwpose-l | fp32 | 63.8 / 81.4 | 0.78 | 0.77-0.80 | 66.0 | 0.10 / 1.49 | 587 / 448 | 451 / 320 | 1.41 | 128 / 128 | 1552 / 1782 | 73.6 / 91.5 | 0.17 / 0.09 | 0.11 / 1.52 | - | 3.3e-06 | - | heavier calls-differ |
| dwpose-l-384 | fp16 | 112.8 / 137.1 | 0.82 | 0.79-0.86 | 117.7 | 0.13 / 1.65 | 409 / 376 | 309 / 298 | 1.04 | 81 / 78 | 1733 / 1773 | 131.8 / 156.6 | 0.30 / 0.11 | 0.16 / 1.69 | - | 0.57 (0.21 % off) | - | heavier calls-differ |
| dwpose-l-384 | fp32 | 128.4 / 166.4 | 0.77 | 0.72-0.84 | 141.5 | 0.16 / 1.74 | 765 / 518 | 618 / 390 | 1.58 | 128 / 128 | 2586 / 2499 | 146.7 / 199.2 | 0.17 / 0.10 | 0.20 / 1.78 | - | 4e-06 | - | heavier calls-differ |
| dwpose-m | fp16 | 40.0 / 41.9 | 0.95 | 0.94-0.96 | 40.3 | 0.06 / 1.30 | 243 / 237 | 188 / 192 | 0.98 | 47 / 45 | 910 / 1130 | 49.3 / 50.7 | 0.19 / 0.12 | 0.07 / 1.32 | - | 0.42 (0.18 % off) | - | calls-differ |
| dwpose-m | fp32 | 44.2 / 50.3 | 0.88 | - | - | 0.07 / 1.21 | 400 / 429 | ~331 / ~360 | 0.92 | 69 / 69 | 1079 / 1506 | 53.0 / 58.3 | 0.10 / 0.06 | 0.08 / 1.22 | - | 2.1e-06 | - | calls-differ |
| dwpose-s | fp16 | 17.8 / 21.2 | 0.84 | - | - | 0.03 / 1.11 | 200 / 201 | ~176 / ~177 | 0.99 | 24 / 24 | 626 / 807 | 26.6 / 28.9 | 0.10 / 0.04 | 0.04 / 1.12 | - | 0.38 (0.017 % off) | - | calls-differ |
| dwpose-s | fp32 | 18.9 / 24.4 | 0.77 | 0.63-0.79 | 21.0 | 0.06 / 1.31 | 286 / 256 | 246 / 224 | 1.10 | 32 / 32 | 708 / 906 | 28.3 / 34.6 | 0.10 / 0.05 | 0.07 / 1.33 | - | 2.5e-06 | - | heavier calls-differ |
| dwpose-t | fp16 | 16.1 / 17.9 | 0.90 | - | - | 0.03 / 1.06 | 161 / 198 | ~144 / ~181 | 0.80 | 17 / 17 | 570 / 763 | 24.8 / 25.9 | 0.09 / 0.04 | 0.04 / 1.08 | - | 0.57 (0.23 % off) | - | calls-differ |
| dwpose-t | fp32 | 17.1 / 20.7 | 0.83 | - | - | 0.04 / 1.05 | 224 / 283 | ~202 / ~262 | 0.77 | 21 / 21 | 630 / 891 | 25.8 / 28.5 | 0.06 / 0.04 | 0.06 / 1.07 | - | 3.1e-06 | - | calls-differ |
| efficientformer_l1.snap_dist_in1k | fp16 | 3.9 / 4.8 | 0.80 | 0.65-0.94 | 4.6 | 0.05 / 0.83 | 34 / 59 | 11 / 35 | 0.34 | 23 / 24 | 400 / 446 | 5.6 / 5.3 | 0.96 / 0.80 | 0.05 / 0.84 | 0.0056 | 0.0056 | 0.0034 / 0.00278 (0.0046) | worse |
| efficientformer_l1.snap_dist_in1k | fp32 | 3.4 / 4.0 | 0.86 | - | - | 0.05 / 0.71 | 64 / 96 | ~18 / ~49 | 0.39 | 45 / 47 | 417 / 479 | 3.9 / 4.5 | 0.76 / 0.72 | 0.05 / 0.72 | 1.3e-06 | 1.3e-06 | 1.18e-05 / 1.22e-05 (0.0046) |  |
| efficientformer_l3.snap_dist_in1k | fp16 | 7.0 / 7.6 | 0.92 | 0.89-1.05 | 7.1 | 0.12 / 1.34 | 76 / 132 | 17 / 71 | 0.25 | 59 / 60 | 474 / 546 | 7.7 / 8.3 | 1.19 / 0.83 | 0.12 / 1.35 | 0.0021 | 0.0021 | 0.00415 / 0.00135 (0.0046) | worse |
| efficientformer_l3.snap_dist_in1k | fp32 | 6.6 / 8.0 | 0.82 | - | - | 0.06 / 1.18 | 145 / 218 | ~27 / ~98 | 0.29 | 118 / 120 | 546 / 625 | 7.2 / 8.8 | 0.79 / 0.72 | 0.06 / 1.18 | 7.9e-07 | 7.9e-07 | 6.83e-06 / 6.76e-06 (0.0046) |  |
| efficientformer_l7.snap_dist_in1k | fp16 | 12.2 / 14.3 | 0.85 | 0.83-0.89 | 12.1 | 0.10 / 1.82 | 185 / 290 | 29 / 133 | 0.23 | 156 / 157 | 641 / 741 | 13.0 / 15.2 | 2.01 / 0.83 | 0.10 / 1.84 | 0.006 | 0.006 | 0.00244 / 0.00189 (0.0046) | worse |
| efficientformer_l7.snap_dist_in1k | fp32 | 13.6 / 15.9 | 0.85 | - | - | 0.06 / 1.58 | 368 / 442 | ~57 / ~128 | 0.45 | 311 / 314 | 816 / 924 | 13.8 / 16.9 | 0.90 / 0.74 | 0.07 / 1.58 | 6.9e-07 | 6.9e-07 | 4.53e-06 / 4.66e-06 (0.0046) |  |
| efficientformerv2_l.snap_dist_in1k | fp16 | 8.6 / 9.2 | 0.93 | - | - | 0.05 / 2.33 | 69 / 100 | ~19 / ~49 | 0.40 | 50 / 51 | 446 / 530 | 9.0 / 9.8 | 0.92 / 0.75 | 0.05 / 2.33 | 0.0027 | 0.0027 | - |  |
| efficientformerv2_l.snap_dist_in1k | fp32 | 9.0 / 10.2 | 0.88 | - | - | 0.06 / 2.27 | 130 / 189 | ~31 / ~88 | 0.36 | 100 / 102 | 517 / 619 | 9.7 / 10.8 | 0.83 / 0.77 | 0.07 / 2.27 | 9.3e-07 | 9.3e-07 | - |  |
| efficientformerv2_s0.snap_dist_in1k | fp16 | 2.9 / 3.0 | 0.96 | - | - | 0.04 / 1.25 | 16 / 25 | ~9 / ~18 | 0.52 | 7 / 8 | 346 / 390 | 3.4 / 3.5 | 0.78 / 0.73 | 0.04 / 1.25 | 0.0043 | 0.0043 | - |  |
| efficientformerv2_s0.snap_dist_in1k | fp32 | 3.2 / 4.4 | 0.74 | - | - | 0.05 / 1.20 | 27 / 45 | ~14 / ~30 | 0.46 | 14 / 14 | 360 / 413 | 3.7 / 4.3 | 0.73 / 0.71 | 0.05 / 1.20 | 1.5e-06 | 1.5e-06 | - |  |
| efficientformerv2_s1.snap_dist_in1k | fp16 | 3.5 / 5.1 | 0.69 | - | - | 0.04 / 1.41 | 20 / 37 | ~8 / ~24 | 0.37 | 12 / 13 | 359 / 413 | 4.0 / 5.8 | 0.82 / 0.71 | 0.04 / 1.41 | 0.0056 | 0.0056 | - |  |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 4.1 / 4.7 | 0.87 | 0.67-0.94 | 4.5 | 0.06 / 1.56 | 39 / 66 | 16 / 41 | 0.39 | 24 / 24 | 385 / 446 | 5.0 / 6.0 | 0.95 / 0.82 | 0.06 / 1.57 | 1.8e-06 | 1.8e-06 | - |  |
| efficientformerv2_s2.snap_dist_in1k | fp16 | 5.6 / 6.1 | 0.93 | - | - | 0.04 / 1.78 | 40 / 56 | ~16 / ~31 | 0.52 | 25 / 25 | 387 / 460 | 6.1 / 6.7 | 0.84 / 0.71 | 0.04 / 1.78 | 0.0034 | 0.0034 | - |  |
| efficientformerv2_s2.snap_dist_in1k | fp32 | 6.0 / 8.4 | 0.71 | - | - | 0.06 / 1.68 | 73 / 112 | ~25 / ~63 | 0.40 | 48 / 50 | 425 / 503 | 6.5 / 8.7 | 0.77 / 0.71 | 0.06 / 1.68 | 1.3e-06 | 1.3e-06 | - |  |
| efficientnet_b0.tv_in1k | fp16 | 3.2 / 4.2 | 0.75 | - | - | 0.02 / 0.85 | 29 / 63 | ~18 / ~52 | 0.36 | 11 / 10 | 343 / 407 | 3.7 / 4.8 | 0.58 / 0.50 | 0.02 / 0.91 | 0.0013 | 0.0013 | - |  |
| efficientnet_b0.tv_in1k | fp32 | 3.3 / 4.5 | 0.75 | - | - | 0.04 / 0.83 | 45 / 94 | ~24 / ~73 | 0.34 | 21 / 21 | 354 / 433 | 3.8 / 4.9 | 0.53 / 0.53 | 0.04 / 0.89 | 9.5e-07 | 9.5e-07 | - |  |
| efficientnet_b4.tv_in1k | fp16 | 12.7 / 15.4 | 0.83 | - | - | 0.05 / 1.45 | 126 / 229 | ~88 / ~192 | 0.46 | 38 / 38 | 569 / 790 | 13.4 / 16.2 | 0.72 / 0.59 | 0.05 / 1.46 | 0.00084 | 0.00084 | - |  |
| efficientnet_b4.tv_in1k | fp32 | 13.9 / 18.0 | 0.77 | - | - | 0.05 / 1.41 | 196 / 256 | ~121 / ~181 | 0.67 | 75 / 75 | 653 / 906 | 14.5 / 18.9 | 0.63 / 0.59 | 0.05 / 1.41 | 5.1e-07 | 5.1e-07 | - |  |
| efficientnet_v2_s.tv_in1k | fp16 | 13.9 / 16.0 | 0.87 | - | - | 0.03 / 1.51 | 103 / 234 | ~62 / ~193 | 0.32 | 41 / 41 | 475 / 622 | 14.5 / 16.4 | 0.75 / 0.59 | 0.04 / 1.51 | 0.00064 | 0.00064 | - |  |
| efficientnet_v2_s.tv_in1k | fp32 | 15.0 / 17.8 | 0.84 | - | - | 0.05 / 1.46 | 157 / 260 | ~75 / ~177 | 0.42 | 82 / 82 | 563 / 757 | 15.6 / 18.3 | 0.64 / 0.60 | 0.05 / 1.46 | 3.9e-07 | 3.9e-07 | - |  |
| efficientvit-sam-l0 | fp32 | 28.7 / 52.3 | 0.55 | - | - | 0.06 / 1.23 | 458 / 472 | ~338 / ~355 | 0.95 | 120 / 117 | 1014 / 1230 | 57.8 / 81.1 | 0.15 / 1.17 | 0.10 / 1.27 | 4.4e-05 | 1 (0.00036 % off) | - |  |
| efficientvit-sam-l1 | fp32 | 38.7 / 62.4 | 0.62 | - | - | 0.07 / 1.38 | 615 / 624 | ~446 / ~457 | 0.98 | 169 / 166 | 1287 / 1594 | 67.8 / 91.6 | 0.18 / 1.40 | 0.10 / 1.42 | 0.00012 | 1 (0.0028 % off) | - |  |
| efficientvit-sam-l2 | fp32 | 51.8 / 76.7 | 0.68 | - | - | 0.08 / 1.50 | 732 / 778 | ~510 / ~560 | 0.91 | 222 / 219 | 1574 / 1779 | 80.7 / 105.8 | 0.23 / 1.66 | 0.12 / 1.54 | 0.0003 | 1 (0.005 % off) | - |  |
| efficientvit-sam-xl0 | fp32 | 191.0 / 203.6 | 0.94 | 0.93-0.97 | 194.2 | 0.23 / 1.59 | 1311 / 1026 | 856 / 596 | 1.44 | 443 / 431 | 2986 / 2792 | 216.9 / 223.4 | 0.86 / 1.49 | 0.31 / 1.66 | 3.5e-05 | 1 (0.0011 % off) | - | heavier |
| efficientvit-sam-xl1 | fp32 | 347.6 / 344.8 | 1.01 | 0.82-1.02 | 353.5 | 0.58 / 2.16 | 2197 / 1668 | 1412 / 908 | 1.55 | 772 / 760 | 4304 / 3891 | 372.5 / 366.2 | 1.45 / 1.67 | 0.66 / 2.27 | 3.7e-05 | 1 (0.003 % off) | - | heavier |
| efficientvit_b2.r288_in1k | fp16 | 6.5 / 8.2 | 0.79 | - | - | 0.04 / 1.35 | 90 / 137 | ~43 / ~90 | 0.48 | 47 / 47 | 528 / 553 | 7.1 / 8.9 | 0.82 / 0.69 | 0.04 / 1.36 | 0.0015 | 0.0015 | - |  |
| efficientvit_b2.r288_in1k | fp32 | 7.7 / 10.6 | 0.72 | - | - | 0.04 / 1.33 | 144 / 194 | ~51 / ~100 | 0.51 | 93 / 93 | 565 / 656 | 8.1 / 11.4 | 0.79 / 0.70 | 0.04 / 1.34 | 2.1e-06 | 2.1e-06 | - |  |
| eomt-l-dinov2-ade20k | fp16 | 393.1 / 382.6 | 1.03 | 1.02-1.03 | 393.7 | 0.42 / 2.33 | 909 / 878 | 294 / 276 | 1.07 | 603 / 603 | 3397 / 4160 | 444.2 / 524.9 | 2.54 / 3.58 | 0.90 / 3.38 | 0.0028 | 1 (0.06 % off) | - | slower heavier shipped-slower |
| eomt-l-dinov2-ade20k | fp32 | 479.3 / 468.0 | 1.02 | 1.02-1.03 | 478.4 | 0.66 / 2.29 | 1802 / 1785 | 584 / 579 | 1.01 | 1205 / 1206 | 4372 / 5396 | 534.6 / 615.7 | 3.30 / 3.31 | 1.10 / 3.02 | 4.1e-06 | 0 | - | slower shipped-slower |
| eomt-l-dinov3-ade20k | fp16 | 404.7 / 408.3 | 0.99 | 0.99-0.99 | 404.5 | 0.43 / 2.71 | 908 / 894 | 294 / 293 | 1.01 | 601 / 601 | 3423 / 4207 | 457.4 / 548.3 | 2.49 / 2.47 | 0.71 / 3.32 | 0.0039 | 1 (0.00033 % off) | - |  |
| eomt-l-dinov3-ade20k | fp32 | 503.8 / 510.9 | 0.99 | 0.98-0.99 | 503.7 | 0.56 / 2.97 | 1799 / 1749 | 584 / 547 | 1.07 | 1202 / 1202 | 4484 / 5366 | 560.4 / 649.8 | 2.40 / 4.08 | 0.92 / 3.77 | 3e-06 | 0 | - | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp16 | 200.6 / 204.3 | 0.98 | 0.98-0.99 | 200.6 | 0.24 / 2.46 | 680 / 666 | 97 / 83 | 1.16 | 583 / 583 | 924 / 1202 | 201.7 / 205.1 | 0.44 / 0.95 | 0.25 / 2.50 | 0.0012 | 0.0012 | 0.000779 / 0.000651 (0.02) | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp32 | 242.3 / 242.4 | 1.00 | - | - | 0.27 / 2.17 | 1354 / 1396 | ~189 / ~231 | 0.82 | 1165 / 1165 | 1617 / 1916 | 243.2 / 243.1 | 0.11 / 0.79 | 0.28 / 2.17 | 6e-07 | 6e-07 | 6.83e-07 / 6.35e-07 (0.02) |  |
| falconsai-nsfw-vit | fp16 | 14.2 / 14.1 | 1.01 | 0.97-1.09 | 14.1 | 0.08 / 0.84 | 186 / 231 | 22 / 67 | 0.34 | 164 / 164 | 400 / 672 | 14.5 / 14.1 | 0.16 / 0.97 | 0.09 / 0.86 | 0 | 0 | 0.000728 / 0.000728 (0.009) |  |
| falconsai-nsfw-vit | fp32 | 16.1 / 15.9 | 1.01 | - | - | 0.06 / 0.71 | 362 / 417 | ~35 / ~90 | 0.39 | 327 / 327 | 574 / 857 | 16.5 / 16.4 | 0.06 / 0.78 | 0.07 / 0.72 | 1.1e-07 | 1.1e-07 | 2.86e-06 / 2.38e-06 (0.009) |  |
| faster-rcnn-r50-fpn-3x-coco | fp16 | 90.7 / 108.4 | 0.84 | - | - | 0.11 / 0.94 | 412 / 432 | ~332 / ~380 | 0.87 | 80 / 52 | 1388 / 2048 | 125.1 / 381.6 | 0.60 / 0.68 | 0.17 / 1.34 | - | 0.00015 | 0.511 / 0.511 (1.6) |  |
| faster-rcnn-r50-fpn-3x-coco | fp32 | 114.9 / 136.7 | 0.84 | - | - | 0.41 / 0.84 | 757 / 857 | ~598 / ~753 | 0.79 | 159 / 105 | 1878 / 2320 | 149.7 / 424.2 | 0.66 / 0.63 | 0.55 / 1.17 | - | 1.6e-06 | 2.81e-05 / 2.91e-05 (1.6) |  |
| fastvit_sa24.apple_dist_in1k | fp16 | 6.1 / 6.8 | 0.90 | - | - | 0.04 / 0.87 | 60 / 104 | ~19 / ~63 | 0.31 | 41 / 41 | 508 / 557 | 6.6 / 7.3 | 1.02 / 0.90 | 0.04 / 0.87 | 0.003 | 0.003 | - |  |
| fastvit_sa24.apple_dist_in1k | fp32 | 6.8 / 7.7 | 0.88 | - | - | 0.06 / 0.81 | 122 / 197 | ~40 / ~115 | 0.35 | 82 / 82 | 562 / 645 | 7.4 / 8.3 | 0.95 / 0.90 | 0.06 / 0.81 | 3.7e-07 | 3.7e-07 | - |  |
| fastvit_t12.apple_dist_in1k | fp16 | 3.2 / 3.3 | 0.97 | - | - | 0.03 / 0.53 | 26 / 47 | ~12 / ~33 | 0.38 | 14 / 14 | 401 / 436 | 3.6 / 3.8 | 0.91 / 0.84 | 0.03 / 0.53 | 0.0051 | 0.0051 | - |  |
| fastvit_t12.apple_dist_in1k | fp32 | 3.6 / 3.6 | 0.99 | - | - | 0.05 / 0.49 | 54 / 87 | ~25 / ~58 | 0.44 | 29 / 29 | 420 / 471 | 4.1 / 4.2 | 0.86 / 0.86 | 0.05 / 0.50 | 7.1e-07 | 7.1e-07 | - |  |
| fastvit_t8.apple_dist_in1k | fp16 | 2.5 / 2.8 | 0.90 | - | - | 0.03 / 0.49 | 17 / 30 | ~9 / ~22 | 0.44 | 8 / 8 | 365 / 404 | 3.0 / 3.3 | 0.89 / 0.83 | 0.03 / 0.50 | 0.015 (0.7 % off) | 0.015 (0.7 % off) | - |  |
| fastvit_t8.apple_dist_in1k | fp32 | 2.9 / 3.5 | 0.81 | - | - | 0.05 / 0.47 | 33 / 55 | ~17 / ~40 | 0.45 | 15 / 15 | 375 / 425 | 3.4 / 3.6 | 0.85 / 0.83 | 0.05 / 0.47 | 1.3e-06 | 1.3e-06 | - |  |
| fcn-r101-d8-ade20k | fp16 | 283.1 / 293.0 | 0.97 | 0.94-1.01 | 296.8 | 0.42 / 2.04 | 638 / 670 | 505 / 543 | 0.93 | 126 / 126 | 1914 / 1889 | 319.5 / 325.2 | 0.36 / 0.15 | 0.46 / 2.12 | 0.001 | 0 | - |  |
| fcn-r101-d8-ade20k | fp32 | 371.4 / 397.2 | 0.93 | 0.90-0.95 | 419.8 | 0.60 / 2.21 | 1263 / 1236 | 1005 / 984 | 1.02 | 252 / 252 | 2711 / 2751 | 402.8 / 430.5 | 0.22 / 0.29 | 0.69 / 2.32 | 0 | 0 | - | heavier |
| flowseek-t | fp16 | 1269.7 / 1806.7 | 0.70 | - | - | 1.30 / 4.77 | 2734 / 4278 | ~2652 / ~4132 | 0.64 | 82 / 146 | 4725 / 6821 | 1272.6 / 1792.2 | 0.55 / 0.08 | 1.31 / 4.77 | 0.023 (0.026 % off) | 0.023 (0.026 % off) | 0.00335 / 0.00502 (0.0077) |  |
| flowseek-t | fp32 | 1441.8 / 2638.6 | 0.55 | - | - | 1.47 / 5.70 | 3192 / 8444 | ~3028 / ~8216 | 0.37 | 164 / 228 | 5358 / 10892 | 1445.7 / 2716.7 | 0.61 / 0.16 | 1.48 / 5.71 | 3.2e-06 | 3.2e-06 | 5.28e-06 / 5.3e-06 (0.0077) |  |
| flowseek-t-things | fp16 | 1264.5 / 1776.4 | 0.71 | - | - | 1.30 / 4.67 | 2734 / 4278 | ~2652 / ~4132 | 0.64 | 82 / 146 | 4730 / 6832 | 1273.7 / 1774.5 | 0.62 / 0.08 | 1.31 / 4.67 | 0.024 (0.064 % off) | 0.024 (0.064 % off) | 0.00248 / 0.00376 (0.0077) |  |
| flowseek-t-things | fp32 | 1436.2 / 2652.6 | 0.54 | - | - | 1.47 / 6.66 | 3128 / 8444 | ~2965 / ~8216 | 0.36 | 164 / 228 | 5445 / 10919 | 1448.6 / 2675.3 | 0.50 / 0.12 | 1.48 / 6.67 | 6.9e-06 | 6.9e-06 | 9.99e-06 / 9.95e-06 (0.0077) |  |
| flowseek-t-things-notartan | fp16 | 1268.0 / 1779.6 | 0.71 | - | - | 1.29 / 4.72 | 2734 / 4278 | ~2652 / ~4132 | 0.64 | 82 / 146 | 4704 / 6817 | 1276.3 / 1779.1 | 0.61 / 0.08 | 1.30 / 4.73 | 0.0015 | 0.0015 | 0.0016 / 0.00269 (0.0077) |  |
| flowseek-t-things-notartan | fp32 | 1433.8 / 2685.0 | 0.53 | - | - | 1.48 / 6.86 | 3128 / 8444 | ~2965 / ~8216 | 0.36 | 164 / 228 | 5389 / 10902 | 1450.0 / 2638.2 | 0.61 / 0.12 | 1.49 / 6.87 | 1.3e-06 | 1.3e-06 | 3e-06 / 3.05e-06 (0.0077) |  |
| focalnet_base_lrf.ms_in1k | fp16 | 19.0 / 21.6 | 0.88 | - | - | 0.06 / 1.46 | 217 / 289 | ~47 / ~118 | 0.40 | 171 / 170 | 626 / 714 | 19.6 / 22.7 | 0.92 / 0.72 | 0.06 / 1.47 | 0.0006 | 0.0006 | - |  |
| focalnet_base_lrf.ms_in1k | fp32 | 21.3 / 25.7 | 0.83 | - | - | 0.05 / 1.41 | 419 / 491 | ~80 / ~152 | 0.53 | 340 / 340 | 826 / 959 | 22.0 / 26.5 | 0.95 / 0.70 | 0.06 / 1.41 | 2.9e-07 | 2.9e-07 | - |  |
| focalnet_base_srf.ms_in1k | fp16 | 17.9 / 20.3 | 0.88 | - | - | 0.04 / 1.36 | 214 / 285 | ~44 / ~116 | 0.39 | 169 / 169 | 621 / 708 | 18.4 / 20.9 | 0.93 / 0.70 | 0.04 / 1.36 | 0.0008 | 0.0008 | - |  |
| focalnet_base_srf.ms_in1k | fp32 | 20.0 / 24.6 | 0.81 | - | - | 0.05 / 1.29 | 412 / 492 | ~75 / ~155 | 0.49 | 337 / 337 | 814 / 926 | 20.6 / 25.4 | 0.88 / 0.74 | 0.05 / 1.29 | 3.4e-07 | 3.4e-07 | - |  |
| focalnet_small_lrf.ms_in1k | fp16 | 14.6 / 14.9 | 0.98 | - | - | 0.04 / 1.46 | 131 / 195 | ~33 / ~98 | 0.35 | 97 / 97 | 506 / 604 | 15.1 / 15.6 | 0.89 / 0.71 | 0.04 / 1.47 | 0.00077 | 0.00077 | - |  |
| focalnet_small_lrf.ms_in1k | fp32 | 16.3 / 17.5 | 0.93 | - | - | 0.05 / 1.35 | 251 / 347 | ~58 / ~155 | 0.38 | 193 / 193 | 623 / 753 | 16.8 / 18.1 | 0.81 / 0.71 | 0.05 / 1.36 | 3.3e-07 | 3.3e-07 | - |  |
| focalnet_small_srf.ms_in1k | fp16 | 13.9 / 13.9 | 1.00 | - | - | 0.04 / 1.40 | 128 / 201 | ~32 / ~105 | 0.31 | 96 / 96 | 501 / 602 | 14.4 / 14.7 | 0.89 / 0.73 | 0.04 / 1.40 | 0.00069 | 0.00069 | - |  |
| focalnet_small_srf.ms_in1k | fp32 | 15.0 / 16.3 | 0.92 | - | - | 0.05 / 1.25 | 246 / 341 | ~55 / ~150 | 0.37 | 191 / 191 | 618 / 753 | 15.7 / 16.8 | 0.81 / 0.70 | 0.05 / 1.25 | 4e-07 | 4e-07 | - |  |
| focalnet_tiny_lrf.ms_in1k | fp16 | 8.5 / 8.5 | 1.00 | - | - | 0.03 / 0.85 | 89 / 143 | ~33 / ~87 | 0.39 | 55 / 55 | 459 / 515 | 9.1 / 9.1 | 0.82 / 0.71 | 0.03 / 0.85 | 0.00058 | 0.00058 | - |  |
| focalnet_tiny_lrf.ms_in1k | fp32 | 9.3 / 9.7 | 0.95 | - | - | 0.04 / 0.79 | 168 / 257 | ~58 / ~147 | 0.40 | 110 / 110 | 536 / 626 | 9.8 / 10.2 | 0.77 / 0.71 | 0.04 / 0.79 | 2.5e-07 | 2.5e-07 | - |  |
| focalnet_tiny_srf.ms_in1k | fp16 | 7.8 / 7.8 | 1.00 | 0.99-1.04 | 7.8 | 0.04 / 0.86 | 86 / 138 | 32 / 83 | 0.39 | 55 / 55 | 458 / 512 | 8.4 / 8.4 | 1.24 / 0.76 | 0.04 / 0.88 | 0.00088 | 0.00088 | - |  |
| focalnet_tiny_srf.ms_in1k | fp32 | 8.6 / 8.8 | 0.98 | - | - | 0.04 / 0.79 | 163 / 244 | ~54 / ~135 | 0.41 | 109 / 109 | 528 / 610 | 9.2 / 9.5 | 0.77 / 0.76 | 0.04 / 0.79 | 4.3e-07 | 4.3e-07 | - |  |
| hat-s-x4 | fp16 | 5957.5 / 9081.5 | 0.66 | - | - | 6.32 / 11.74 | 1958 / 2700 | ~1881 / ~2387 | 0.79 | 77 / 313 | 5071 / 5579 | 6019.4 / 9283.4 | 0.60 / 0.36 | 6.34 / 11.76 | 0.0089 | 0.01 (7.9e-06 % off) | - |  |
| hat-s-x4 | fp32 | 7460.5 / 17686.6 | 0.42 | - | - | 7.76 / 17.95 | 3907 / 4739 | ~3759 / ~4371 | 0.86 | 148 / 368 | 6650 / 7500 | 7705.2 / 17820.1 | 0.73 / 0.43 | 7.78 / 17.96 | 5.3e-06 | 6e-06 | - |  |
| hat-x4 | fp16 | 7017.5 / 8996.0 | 0.78 | - | - | 7.12 / 12.93 | 2276 / 2507 | ~2180 / ~2172 | 1.00 | 96 / 334 | 5415 / 5647 | 6905.1 / 9554.0 | 0.67 / 0.29 | 7.13 / 12.93 | 0.013 (0.00027 % off) | 0.014 (0.00058 % off) | - |  |
| hat-x4 | fp32 | 10448.9 / 20066.3 | 0.52 | 0.49-0.57 | 10852.7 | 11.47 / 20.16 | 4545 / 4506 | 4353 / 4095 | 1.06 | 189 / 411 | 7164 / 7970 | 10278.4 / 20740.5 | 0.88 / 0.47 | 11.51 / 20.21 | 6.9e-06 | 7.5e-06 | - | heavier |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp16 | 2.0 / 2.2 | 0.87 | - | - | 0.03 / 0.74 | 19 / 28 | ~8 / ~16 | 0.51 | 12 / 12 | 379 / 410 | 2.0 / 2.7 | 0.76 / 0.74 | 0.04 / 0.74 | 0.00081 | 0.00081 | - |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp32 | 2.5 / 3.5 | 0.72 | - | - | 0.04 / 0.69 | 36 / 52 | ~13 / ~29 | 0.47 | 23 / 23 | 378 / 427 | 2.4 / 3.2 | 0.72 / 0.69 | 0.05 / 0.69 | 8.1e-07 | 8.1e-07 | - |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp16 | 2.1 / 2.7 | 0.77 | - | - | 0.04 / 0.72 | 22 / 35 | ~10 / ~23 | 0.47 | 12 / 12 | 389 / 416 | 2.5 / 3.1 | 0.74 / 0.69 | 0.04 / 0.72 | 0.00063 | 0.00063 | - |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp32 | 2.4 / 3.0 | 0.80 | - | - | 0.04 / 0.71 | 42 / 66 | ~18 / ~41 | 0.45 | 24 / 24 | 386 / 439 | 2.9 / 3.5 | 0.70 / 0.71 | 0.05 / 0.71 | 7.8e-07 | 7.8e-07 | - |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp16 | 3.0 / 3.8 | 0.80 | - | - | 0.04 / 0.95 | 34 / 53 | ~12 / ~32 | 0.41 | 22 / 22 | 425 / 461 | 3.4 / 4.3 | 0.78 / 0.72 | 0.04 / 0.95 | 0.00053 | 0.00053 | - |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp32 | 3.5 / 4.3 | 0.82 | - | - | 0.05 / 0.91 | 65 / 100 | ~22 / ~58 | 0.39 | 43 / 43 | 423 / 504 | 4.2 / 4.8 | 0.75 / 0.70 | 0.06 / 0.91 | 9.1e-07 | 9.1e-07 | - |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp16 | 3.6 / 4.6 | 0.79 | - | - | 0.04 / 1.13 | 46 / 69 | ~14 / ~38 | 0.39 | 31 / 31 | 466 / 528 | 4.1 / 5.1 | 0.80 / 0.70 | 0.04 / 1.13 | 0.00052 | 0.00052 | - |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp32 | 4.2 / 5.1 | 0.83 | - | - | 0.06 / 1.08 | 88 / 131 | ~26 / ~69 | 0.38 | 62 / 62 | 471 / 587 | 4.8 / 5.8 | 0.74 / 0.70 | 0.06 / 1.08 | 5.1e-07 | 5.1e-07 | - |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp16 | 5.0 / 5.4 | 0.92 | - | - | 0.03 / 1.01 | 58 / 72 | ~20 / ~34 | 0.60 | 38 / 38 | 489 / 567 | 5.4 / 5.9 | 0.80 / 0.70 | 0.03 / 1.02 | 0.00064 | 0.00064 | - |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp32 | 5.7 / 6.3 | 0.90 | - | - | 0.04 / 1.01 | 111 / 137 | ~35 / ~62 | 0.57 | 76 / 76 | 500 / 612 | 6.2 / 6.7 | 0.74 / 0.70 | 0.04 / 1.01 | 5.4e-07 | 5.4e-07 | - |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp16 | 9.0 / 9.7 | 0.93 | - | - | 0.04 / 1.46 | 104 / 139 | ~28 / ~63 | 0.45 | 76 / 76 | 615 / 762 | 9.5 / 10.4 | 0.86 / 0.71 | 0.04 / 1.46 | 0.00064 | 0.00064 | - |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp32 | 10.4 / 11.3 | 0.92 | - | - | 0.04 / 1.43 | 196 / 256 | ~44 / ~105 | 0.43 | 151 / 151 | 665 / 877 | 10.9 / 11.9 | 0.80 / 0.71 | 0.05 / 1.43 | 3.9e-07 | 3.9e-07 | - |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp16 | 18.9 / 19.9 | 0.95 | - | - | 0.04 / 1.80 | 187 / 272 | ~43 / ~127 | 0.34 | 144 / 144 | 817 / 987 | 19.4 / 20.5 | 0.94 / 0.72 | 0.04 / 1.80 | 0.00088 | 0.00088 | - |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp32 | 21.7 / 23.5 | 0.92 | - | - | 0.06 / 1.77 | 368 / 493 | ~81 / ~205 | 0.40 | 288 / 288 | 916 / 1208 | 22.3 / 24.0 | 0.83 / 0.73 | 0.06 / 1.77 | 5.9e-07 | 5.9e-07 | - |  |
| hrnet-w32 | fp16 | 85.4 / 98.1 | 0.87 | - | - | 0.12 / 2.08 | 232 / 271 | ~176 / ~215 | 0.82 | 56 / 56 | 697 / 962 | 94.7 / 106.9 | 0.05 / 0.12 | 0.14 / 2.09 | 0.0036 | 0.29 (0.36 % off) | - |  |
| hrnet-w32 | fp32 | 102.4 / 122.8 | 0.83 | 0.82-0.84 | 111.0 | 0.91 / 2.37 | 450 / 465 | 332 / 355 | 0.93 | 110 / 110 | 1014 / 1143 | 110.8 / 132.1 | 0.15 / 0.30 | 0.93 / 2.40 | 0 | 0 | - |  |
| hrnet-w32-384 | fp16 | 177.4 / 201.0 | 0.88 | - | - | 0.21 / 2.15 | 439 / 471 | ~383 / ~415 | 0.92 | 56 / 56 | 1117 / 1274 | 193.7 / 217.2 | 0.05 / 0.12 | 0.23 / 2.17 | 0.011 (0.00023 % off) | 0.41 (0.83 % off) | - |  |
| hrnet-w32-384 | fp32 | 217.2 / 261.0 | 0.83 | - | - | 0.25 / 2.11 | 835 / 867 | ~725 / ~757 | 0.96 | 110 / 110 | 1847 / 1967 | 235.1 / 276.8 | 0.06 / 0.12 | 0.27 / 2.13 | 0 | 0 | - |  |
| hrnet-w48 | fp16 | 187.5 / 205.9 | 0.91 | 0.89-0.93 | 188.1 | 1.26 / 2.95 | 296 / 268 | 164 / 144 | 1.13 | 124 / 124 | 768 / 926 | 198.9 / 222.0 | 0.17 / 0.47 | 1.27 / 2.99 | 0.00028 | 0.0048 | - | heavier |
| hrnet-w48 | fp32 | 222.5 / 261.6 | 0.85 | 0.83-0.86 | 240.8 | 1.17 / 3.01 | 727 / 710 | 474 / 466 | 1.02 | 245 / 245 | 1334 / 1478 | 231.0 / 272.5 | 0.41 / 0.90 | 1.20 / 3.09 | 0 | 0 | - |  |
| hrnet-w48-384 | fp16 | 398.0 / 432.0 | 0.92 | 0.92-0.93 | 412.3 | 1.42 / 2.93 | 416 / 371 | 273 / 247 | 1.11 | 124 / 124 | 1144 / 1300 | 417.6 / 451.7 | 0.27 / 0.49 | 1.44 / 2.97 | 0.00065 | 0.00065 | - | heavier |
| hrnet-w48-384 | fp32 | 442.4 / 509.0 | 0.87 | - | - | 0.48 / 2.52 | 983 / 1012 | ~738 / ~767 | 0.96 | 245 / 245 | 2271 / 2371 | 455.7 / 525.2 | 0.07 / 0.16 | 0.50 / 2.54 | 0 | 0 | - |  |
| hrnet-w48-fcn-ade20k | fp16 | 157.6 / 162.5 | 0.97 | 0.93-1.05 | 154.6 | 0.20 / 2.52 | 312 / 292 | 178 / 163 | 1.09 | 129 / 128 | 1366 / 1194 | 196.9 / 202.9 | 0.43 / 0.21 | 0.25 / 2.58 | 0.00094 | 0.76 (0.056 % off) | - | heavier |
| hrnet-w48-fcn-ade20k | fp32 | 176.6 / 193.7 | 0.91 | - | - | 0.21 / 2.19 | 583 / 609 | ~329 / ~355 | 0.93 | 254 / 254 | 1628 / 1675 | 216.9 / 236.1 | 0.27 / 0.20 | 0.25 / 2.23 | 0 | 0 | - |  |
| igev-plusplus-eth3d | fp16 | 5810.7 / 7016.1 | 0.83 | 0.80-0.90 | 6136.1 | 6.34 / 15.13 | 2414 / 2199 | 2350 / 2170 | 1.08 | 28 / 29 | 4152 / 4450 | 5929.3 / 6689.2 | 0.47 / 0.20 | 6.36 / 15.36 | 0.45 (0.33 % off) | 0.29 (0.034 % off) | 0.00067 / 0.00256 (0.02) | heavier |
| igev-plusplus-eth3d | fp32 | 7446.8 / 8025.1 | 0.93 | 0.87-1.06 | 8239.9 | 8.72 / 16.48 | 4260 / 3786 | 4170 / 3729 | 1.12 | 55 / 56 | 5328 / 6022 | 7453.6 / 7966.0 | 0.52 / 0.20 | 8.74 / 16.68 | 0.0043 | 0.0043 | 1.36e-06 / 1.28e-06 (0.02) | heavier |
| igev-plusplus-kitti12 | fp16 | 8964.0 / 13448.5 | 0.67 | 0.61-0.82 | 9671.7 | 7.44 / 15.22 | 2414 / 2199 | 2350 / 2170 | 1.08 | 28 / 29 | 4190 / 4406 | 9560.3 / 10823.3 | 0.51 / 0.22 | 7.45 / 15.44 | 0.023 (0.0032 % off) | 0.026 (0.00027 % off) | 0.00114 / 0.000852 (0.02) | heavier |
| igev-plusplus-kitti12 | fp32 | 15223.8 / 15031.8 | 1.01 | 0.69-1.56 | 12864.3 | 13.82 / 15.68 | 4260 / 3786 | 4170 / 3729 | 1.12 | 55 / 56 | 5454 / 6038 | 13972.2 / 13931.0 | 0.49 / 0.20 | 13.85 / 15.89 | 0.00037 | 0.00053 | 9.87e-07 / 7.88e-07 (0.02) | heavier |
| igev-plusplus-kitti15 | fp16 | 8669.6 / 9429.7 | 0.92 | 0.44-1.92 | 12262.7 | 8.78 / 19.39 | 2414 / 2215 | 2350 / 2186 | 1.08 | 28 / 29 | 4177 / 4462 | 6752.1 / 9631.2 | 0.54 / 0.23 | 8.80 / 19.61 | 0.053 (0.003 % off) | 0.0086 | 0.00114 / 0.00103 (0.02) | heavier |
| igev-plusplus-kitti15 | fp32 | 7159.7 / 8034.9 | 0.89 | 0.87-0.92 | 8999.9 | 7.40 / 20.27 | 4260 / 3786 | 4170 / 3729 | 1.12 | 55 / 56 | 5330 / 5920 | 7410.4 / 7897.7 | 0.47 / 0.38 | 7.42 / 20.50 | 1.9e-05 | 2.1e-05 | 9.07e-07 / 8.77e-07 (0.02) | heavier shipped-slower |
| igev-plusplus-middlebury | fp16 | 5758.9 / 7138.5 | 0.81 | 0.78-0.89 | 6004.0 | 5.83 / 14.51 | 2414 / 2199 | 2350 / 2170 | 1.08 | 28 / 29 | 4175 / 4474 | 5783.6 / 7257.1 | 0.51 / 0.20 | 5.84 / 14.73 | 0.39 (1.2 % off) | 0.41 (0.25 % off) | 0.000571 / 0.00229 (0.02) | heavier |
| igev-plusplus-middlebury | fp32 | 7244.0 / 7900.0 | 0.92 | 0.85-0.95 | 8859.2 | 7.76 / 16.97 | 4260 / 3786 | 4170 / 3729 | 1.12 | 55 / 56 | 5320 / 5890 | 7409.0 / 7800.4 | 0.48 / 0.21 | 7.79 / 17.18 | 0.00055 | 0.00022 | 1.68e-06 / 1.75e-06 (0.02) | heavier shipped-slower |
| igev-plusplus-sceneflow | fp16 | 5824.1 / 7542.0 | 0.77 | 0.73-0.84 | 6220.6 | 5.90 / 15.42 | 2414 / 2189 | 2350 / 2160 | 1.09 | 28 / 29 | 4188 / 4464 | 5817.7 / 7497.1 | 0.50 / 0.21 | 5.92 / 15.63 | 0.5 (0.023 % off) | 0.44 (0.017 % off) | 0.00137 / 0.00153 (0.02) | heavier |
| igev-plusplus-sceneflow | fp32 | 7113.1 / 7667.9 | 0.93 | 0.91-0.95 | 8504.5 | 7.46 / 16.69 | 4260 / 3786 | 4170 / 3729 | 1.12 | 55 / 56 | 5322 / 5963 | 7237.6 / 7723.7 | 0.47 / 0.21 | 7.48 / 16.89 | 0.00045 | 0.00045 | 1.08e-06 / 1.09e-06 (0.02) | heavier shipped-slower |
| igev-stereo-eth3d | fp16 | 3216.8 / 3667.7 | 0.88 | 0.86-0.89 | 3315.2 | 3.45 / 9.63 | 1686 / 1057 | 1626 / 1031 | 1.58 | 24 / 26 | 3811 / 3451 | 3212.1 / 3665.7 | 0.43 / 0.17 | 3.46 / 9.81 | 0.49 (0.38 % off) | 0.49 (0.067 % off) | 0.00144 / 0.00166 (0.02) | heavier |
| igev-stereo-eth3d | fp32 | 4171.1 / 4690.8 | 0.89 | 0.88-0.90 | 5008.0 | 4.55 / 10.37 | 3122 / 2055 | 3039 / 2005 | 1.52 | 48 / 49 | 4878 / 4431 | 4192.1 / 4807.9 | 0.40 / 0.17 | 4.57 / 10.54 | 0.00084 | 0.00084 | 8.99e-06 / 8.31e-06 (0.02) | heavier shipped-slower |
| igev-stereo-kitti12 | fp16 | 3205.7 / 3695.7 | 0.87 | 0.83-0.89 | 3317.8 | 3.33 / 9.48 | 1686 / 1080 | 1626 / 1054 | 1.54 | 24 / 26 | 3840 / 3451 | 3198.5 / 3657.1 | 0.30 / 0.17 | 3.34 / 9.66 | 0.052 (0.016 % off) | 0.0042 | 0.000965 / 0.00117 (0.02) | heavier |
| igev-stereo-kitti12 | fp32 | 4197.6 / 4663.5 | 0.90 | 0.89-0.93 | 5218.6 | 4.51 / 10.21 | 3123 / 2055 | 3039 / 2005 | 1.52 | 48 / 49 | 4880 / 4362 | 4213.5 / 4730.8 | 0.40 / 0.17 | 4.53 / 10.38 | 2.7e-05 | 4.9e-06 | 1.03e-06 / 1.03e-06 (0.02) | heavier shipped-slower |
| igev-stereo-kitti15 | fp16 | 3184.4 / 3614.1 | 0.88 | - | - | 3.22 / 8.71 | 1686 / 1010 | ~1661 / ~984 | 1.69 | 24 / 26 | 3881 / 3459 | 3193.4 / 3614.6 | 0.15 / 0.09 | 3.23 / 8.71 | 0.018 (0.00084 % off) | 0.004 | 0.000928 / 0.00102 (0.02) | heavier |
| igev-stereo-kitti15 | fp32 | 4165.1 / 4787.5 | 0.87 | - | - | 4.17 / 9.99 | 3122 / 2055 | ~3074 / ~2005 | 1.53 | 48 / 49 | 4896 / 4258 | 4201.9 / 4750.9 | 0.12 / 0.09 | 4.18 / 9.99 | 9.1e-06 | 1.8e-06 | 7.73e-07 / 7.74e-07 (0.02) | heavier |
| igev-stereo-middlebury | fp16 | 3192.1 / 3633.4 | 0.88 | - | - | 3.22 / 9.05 | 1686 / 1080 | ~1661 / ~1054 | 1.57 | 24 / 26 | 3881 / 3477 | 3203.3 / 3624.6 | 0.15 / 0.10 | 3.23 / 9.06 | 5.2 (1.7 % off) | 5.2 (0.24 % off) | 0.001 / 0.00126 (0.02) | heavier |
| igev-stereo-middlebury | fp32 | 4180.7 / 4608.6 | 0.91 | - | - | 4.18 / 9.88 | 3122 / 2055 | ~3074 / ~2005 | 1.53 | 48 / 49 | 4894 / 4467 | 4191.3 / 4802.9 | 0.12 / 0.10 | 4.20 / 9.88 | 0.0013 | 0.0013 | 1.35e-06 / 1.4e-06 (0.02) | heavier |
| igev-stereo-sceneflow | fp16 | 3190.5 / 3621.4 | 0.88 | - | - | 3.23 / 8.76 | 1686 / 1010 | ~1661 / ~984 | 1.69 | 24 / 26 | 3882 / 3443 | 3196.6 / 3619.0 | 0.14 / 0.10 | 3.24 / 8.77 | 0.36 (0.16 % off) | 0.2 (0.03 % off) | 0.00188 / 0.00195 (0.02) | heavier |
| igev-stereo-sceneflow | fp32 | 4184.8 / 4628.8 | 0.90 | - | - | 4.20 / 9.75 | 3122 / 2055 | ~3074 / ~2005 | 1.53 | 48 / 49 | 4899 / 4321 | 4192.8 / 4640.4 | 0.12 / 0.09 | 4.21 / 9.76 | 3.7e-05 | 3.7e-05 | 1.65e-06 / 1.74e-06 (0.02) | heavier |
| inception_v3.tv_in1k | fp16 | 7.6 / 10.0 | 0.76 | - | - | 0.03 / 1.05 | 66 / 152 | ~20 / ~106 | 0.20 | 46 / 46 | 486 / 568 | 8.2 / 10.5 | 0.68 / 0.58 | 0.04 / 1.05 | 0.00084 | 0.00084 | - |  |
| inception_v3.tv_in1k | fp32 | 8.5 / 11.6 | 0.73 | - | - | 0.05 / 1.04 | 125 / 272 | ~34 / ~181 | 0.19 | 91 / 91 | 514 / 687 | 8.9 / 12.1 | 0.62 / 0.57 | 0.05 / 1.04 | 1e-06 | 1e-06 | - |  |
| isnet | fp16 | 111.0 / 124.9 | 0.89 | - | - | 0.12 / 2.68 | 569 / 494 | ~484 / ~410 | 1.18 | 84 / 85 | 1169 / 1552 | 114.5 / 129.2 | 0.07 / 0.09 | 0.15 / 2.69 | - | 0.00056 | 0.00021 / 0.00023 (0.02) | heavier |
| isnet | fp32 | 139.8 / 162.0 | 0.86 | - | - | 0.16 / 2.58 | 1059 / 868 | ~891 / ~700 | 1.27 | 168 / 168 | 1958 / 2386 | 148.5 / 175.8 | 0.07 / 0.09 | 0.20 / 2.59 | - | 0 | 4.24e-08 / 4.24e-08 (0.02) | heavier |
| isnet-general-use | fp16 | 111.0 / 126.9 | 0.87 | - | - | 0.12 / 2.60 | 569 / 492 | ~484 / ~408 | 1.19 | 84 / 85 | 1168 / 1550 | 114.5 / 130.1 | 0.06 / 0.08 | 0.16 / 2.60 | - | 0.00064 | 1.2e-05 / 6.03e-05 (0.02) | heavier |
| isnet-general-use | fp32 | 147.9 / 165.7 | 0.89 | - | - | 0.16 / 2.63 | 1059 / 868 | ~891 / ~700 | 1.27 | 168 / 168 | 1957 / 2383 | 148.5 / 170.9 | 0.06 / 0.08 | 0.20 / 2.63 | - | 7.3e-38 | 2.22e-08 / 2.22e-08 (0.02) | heavier |
| lineart | fp16 | 74.8 / 100.1 | 0.75 | - | - | 0.12 / 1.04 | 597 / 594 | ~589 / ~586 | 1.00 | 8 / 8 | 1658 / 1684 | 85.8 / 108.8 | 0.06 / 0.03 | 0.14 / 1.05 | - | 0.028 (0.047 % off) (1 keys differ) | 6.47e-05 / 0.000117 (0.02) |  |
| lineart | fp32 | 97.1 / 117.3 | 0.83 | 0.81-0.85 | 120.1 | 0.12 / 1.14 | 1188 / 903 | 1167 / 887 | 1.32 | 16 / 16 | 2880 / 2204 | 107.7 / 127.1 | 0.08 / 0.04 | 0.14 / 1.20 | - | 1.8e-05 (1 keys differ) | 5.61e-07 / 5.59e-07 (0.02) | heavier |
| lineart-coarse | fp16 | 70.1 / 100.1 | 0.70 | - | - | 0.08 / 1.02 | 597 / 594 | ~589 / ~586 | 1.00 | 8 / 8 | 1655 / 1687 | 79.4 / 109.0 | 0.06 / 0.03 | 0.10 / 1.04 | - | 0.022 (0.014 % off) (1 keys differ) | 0.000108 / 0.000162 (0.02) |  |
| lineart-coarse | fp32 | 97.6 / 117.2 | 0.83 | 0.82-0.84 | 121.1 | 0.12 / 1.04 | 1188 / 903 | 1167 / 887 | 1.32 | 16 / 16 | 2875 / 2190 | 107.9 / 128.1 | 0.07 / 0.03 | 0.13 / 1.11 | - | 1.6e-05 (1 keys differ) | 7.97e-07 / 7.98e-07 (0.02) | heavier shipped-slower |
| loftr-indoor | fp16 | 365.9 / 1226.4 | 0.30 | - | - | 1.06 / 5.90 | 1098 / 1237 | ~1058 / ~1216 | 0.87 | 40 / 21 | 3656 / 5015 | 465.9 / 5319.3 | 0.11 / 0.10 | 1.32 / 8.96 | - | empty (4 keys differ) | 0.00179 r0.998 / 0.00176 r0.998 (0.0025) | calls-differ |
| loftr-indoor | fp32 | 432.4 / 523.1 | 0.83 | 0.78-0.85 | 463.3 | 0.46 / 2.80 | 2161 / 1261 | 2079 / 1218 | 1.71 | 64 / 43 | 4806 / 5900 | 612.9 / 1676.0 | 0.17 / 0.06 | 0.67 / 4.10 | - | 2.4e-05 | 5.5e-06 r1.000 / 5.51e-06 r1.000 (0.0025) | heavier calls-differ |
| loftr-outdoor | fp16 | 345.8 / 1151.2 | 0.30 | - | - | 0.37 / 2.63 | 1098 / 1237 | ~1058 / ~1216 | 0.87 | 40 / 21 | 3647 / 5228 | 454.0 / 3979.9 | 0.10 / 0.04 | 0.55 / 3.96 | - | 0.0018 | 0.00219 r0.999 / 0.00233 r0.999 (0.0025) | calls-differ |
| loftr-outdoor | fp32 | 442.1 / 528.8 | 0.84 | 0.79-0.85 | 478.4 | 0.46 / 3.04 | 2161 / 1277 | 2079 / 1234 | 1.68 | 64 / 43 | 4655 / 5798 | 624.1 / 1770.7 | 0.18 / 0.08 | 0.68 / 4.47 | - | 7.6e-06 | 5.98e-06 r1.000 / 5.99e-06 r1.000 (0.0025) | heavier calls-differ |
| loma-b | fp16 | 2280.5 / 2547.8 | 0.90 | - | - | 2.30 / 7.88 | 1488 / 1856 | ~775 / ~1202 | 0.65 | 712 / 654 | 4586 / 6503 | 2301.5 / 2674.6 | 1.87 / 0.23 | 2.33 / 7.94 | - | 0.98 (79 % off) (3 keys differ) | - | calls-differ |
| loma-b | fp32 | 3166.5 / 3379.9 | 0.94 | - | - | 3.16 / 9.51 | 2751 / 3714 | ~1444 / ~2406 | 0.60 | 1308 / 1308 | 5651 / 8044 | 3184.8 / 3464.9 | 1.44 / 0.40 | 3.19 / 9.60 | - | 0.64 (0.084 % off) (1 keys differ) | - | calls-differ |
| loma-b128 | fp16 | 800.0 / 925.8 | 0.86 | - | - | 0.80 / 4.54 | 886 / 994 | ~774 / ~934 | 0.83 | 112 / 60 | 3426 / 3899 | 801.5 / 983.5 | 0.51 / 0.10 | 0.83 / 4.60 | - | 0.98 (79 % off) (3 keys differ) | - | calls-differ |
| loma-b128 | fp32 | 1264.1 / 1270.7 | 0.99 | 0.98-1.02 | 1279.2 | 1.36 / 5.33 | 1234 / 1062 | 1097 / 942 | 1.16 | 119 / 120 | 3578 / 4086 | 1271.6 / 1421.4 | 0.39 / 0.10 | 1.38 / 5.52 | - | 0.64 (0.084 % off) (1 keys differ) | - | heavier |
| loma-b64 | fp16 | 639.4 / 639.7 | 1.00 | - | - | 0.67 / 4.10 | 817 / 907 | ~737 / ~859 | 0.86 | 79 / 48 | 3048 / 3433 | 647.4 / 691.1 | 0.40 / 0.11 | 0.70 / 4.16 | - | 0.98 (83 % off) (3 keys differ) | - | calls-differ |
| loma-b64 | fp32 | 1055.7 / 976.6 | 1.08 | 0.94-1.11 | 1130.3 | 1.12 / 4.75 | 1164 / 988 | 1059 / 892 | 1.19 | 96 / 96 | 3640 / 3642 | 1059.8 / 1050.8 | 0.29 / 0.21 | 1.13 / 4.88 | - | 0.74 (0.23 % off) (1 keys differ) | - | heavier |
| loma-g | fp16 | 2751.7 / 3019.5 | 0.91 | - | - | 2.77 / 8.54 | 1813 / 1916 | ~776 / ~938 | 0.83 | 1037 / 978 | 4857 / 6820 | 2770.2 / 3080.3 | 2.30 / 0.23 | 2.81 / 8.61 | - | 0.98 (79 % off) (3 keys differ) | - | calls-differ |
| loma-g | fp32 | 3791.5 / 4200.1 | 0.90 | - | - | 3.92 / 9.92 | 3398 / 4360 | ~1444 / ~2406 | 0.60 | 1954 / 1954 | 6178 / 8689 | 3744.8 / 4117.5 | 1.89 / 0.40 | 3.95 / 10.00 | - | 0.64 (0.084 % off) (1 keys differ) | - | calls-differ |
| loma-l | fp16 | 2390.2 / 2626.9 | 0.91 | - | - | 2.40 / 7.66 | 1553 / 1929 | ~776 / ~1210 | 0.64 | 777 / 719 | 4680 / 6844 | 2407.0 / 2685.3 | 1.92 / 0.23 | 2.43 / 7.73 | - | 0.98 (79 % off) (3 keys differ) | - | calls-differ |
| loma-l | fp32 | 3299.5 / 3482.7 | 0.95 | - | - | 3.31 / 9.08 | 2881 / 3843 | ~1444 / ~2406 | 0.60 | 1437 / 1438 | 5805 / 8181 | 3321.7 / 3559.0 | 1.49 / 0.27 | 3.35 / 9.15 | - | 0.64 (0.084 % off) (1 keys differ) | - | calls-differ |
| loma-r | fp16 | 2272.8 / 2620.4 | 0.87 | - | - | 2.29 / 7.60 | 1488 / 1592 | ~775 / ~938 | 0.83 | 712 / 654 | 4548 / 6496 | 2290.7 / 2630.0 | 1.87 / 0.23 | 2.32 / 7.66 | - | 0.98 (79 % off) (3 keys differ) | - | calls-differ |
| loma-r | fp32 | 3145.3 / 3333.3 | 0.94 | - | - | 3.13 / 8.84 | 2751 / 3714 | ~1444 / ~2406 | 0.60 | 1308 / 1308 | 5670 / 8049 | 3148.5 / 3382.4 | 1.43 / 0.42 | 3.16 / 8.91 | - | 0.64 (0.084 % off) (1 keys differ) | - | calls-differ |
| lw-detr-large-coco | fp16 | 53.1 / 58.0 | 0.92 | 0.91-0.93 | 58.5 | 0.10 / 2.06 | 548 / 319 | 431 / 238 | 1.80 | 118 / 81 | 1255 / 1089 | 56.5 / 61.4 | 2.77 / 2.08 | 0.10 / 2.07 | 0.96 (66 % off) | 0.0072 | 0.0368 / 0.0657 (0.19) | heavier |
| lw-detr-large-coco | fp32 | 57.5 / 60.4 | 0.95 | 0.95-0.95 | 59.8 | 0.10 / 1.96 | 568 / 522 | 414 / 363 | 1.14 | 154 / 160 | 1210 / 1196 | 60.8 / 64.2 | 2.18 / 2.07 | 0.11 / 1.98 | 0.0015 | 2.3e-06 | 2.16e-05 / 3.12e-05 (0.19) | heavier |
| lw-detr-medium-coco | fp16 | 37.1 / 41.9 | 0.89 | - | - | 0.10 / 1.84 | 224 / 290 | ~164 / ~238 | 0.69 | 60 / 51 | 749 / 860 | 40.0 / 45.1 | 2.25 / 2.17 | 0.11 / 1.84 | 0.97 (76 % off) | empty (3 keys differ) | 0.0305 / 0.0343 (0.19) |  |
| lw-detr-medium-coco | fp32 | 41.5 / 41.6 | 1.00 | - | - | 0.09 / 1.76 | 228 / 231 | ~132 / ~130 | 1.01 | 96 / 100 | 744 / 774 | 44.5 / 44.9 | 2.15 / 2.01 | 0.09 / 1.77 | 0.00041 | 5.4e-06 | 2.73e-05 / 2.93e-05 (0.19) |  |
| lw-detr-small-coco | fp16 | 24.0 / 27.9 | 0.86 | - | - | 0.08 / 1.75 | 184 / 299 | ~150 / ~274 | 0.55 | 34 / 24 | 675 / 841 | 27.0 / 30.9 | 2.04 / 2.04 | 0.08 / 1.76 | 0.96 (66 % off) | empty (3 keys differ) | 0.0515 / 0.0868 (0.28) |  |
| lw-detr-small-coco | fp32 | 27.8 / 28.2 | 0.99 | - | - | 0.08 / 1.73 | 140 / 185 | ~97 / ~138 | 0.70 | 43 / 47 | 631 / 736 | 30.5 / 31.3 | 2.05 / 2.01 | 0.09 / 1.74 | 0.00023 | 6.2e-06 | 0.00017 / 0.000104 (0.28) |  |
| lw-detr-tiny-coco | fp16 | 17.1 / 19.6 | 0.87 | - | - | 0.08 / 1.49 | 138 / 258 | ~109 / ~238 | 0.46 | 30 / 19 | 590 / 748 | 20.0 / 22.8 | 2.11 / 2.00 | 0.08 / 1.50 | 0.96 (56 % off) | empty (3 keys differ) | 0.0445 / 0.0765 (0.23) |  |
| lw-detr-tiny-coco | fp32 | 19.8 / 20.0 | 0.99 | - | - | 0.06 / 1.46 | 125 / 176 | ~90 / ~138 | 0.65 | 36 / 37 | 583 / 705 | 22.5 / 23.1 | 2.04 / 2.02 | 0.07 / 1.46 | 0.00023 | 6.3e-06 | 3.5e-05 / 7.13e-05 (0.23) |  |
| lw-detr-xlarge-coco | fp16 | 96.8 / 108.0 | 0.90 | 0.89-0.90 | 101.9 | 0.17 / 2.10 | 722 / 445 | 433 / 226 | 1.91 | 288 / 219 | 1456 / 1322 | 100.3 / 111.6 | 2.78 / 2.12 | 0.18 / 2.12 | 0.97 (69 % off) | empty (3 keys differ) | 0.00829 / 0.0337 (0.14) | heavier |
| lw-detr-xlarge-coco | fp32 | 105.4 / 110.6 | 0.95 | - | - | 0.14 / 2.02 | 860 / 884 | ~433 / ~450 | 0.96 | 427 / 434 | 1602 / 1606 | 108.3 / 114.0 | 2.26 / 2.09 | 0.14 / 2.03 | 0.61 (0.78 % off) | 0.00068 | 3.66e-05 / 3.35e-05 (0.14) |  |
| mask-rcnn-r50-fpn-3x-coco | fp16 | 89.1 / 104.4 | 0.85 | - | - | 0.11 / 0.68 | 417 / 432 | ~332 / ~380 | 0.87 | 86 / 52 | 1450 / 2067 | 141.9 / 374.9 | 0.54 / 0.53 | 0.19 / 0.97 | - | 1 (0.0068 % off) | 0.0728 / 0.0717 (0.29) |  |
| mask-rcnn-r50-fpn-3x-coco | fp32 | 112.6 / 136.9 | 0.82 | - | - | 0.19 / 0.79 | 833 / 857 | ~664 / ~753 | 0.88 | 169 / 105 | 1952 / 2339 | 174.4 / 470.4 | 0.61 / 0.68 | 0.29 / 1.21 | - | 1 (7e-05 % off) | 4.77e-05 / 3.62e-05 (0.29) |  |
| mask2former-swinl-ade20k | fp16 | 590.2 / 651.5 | 0.91 | - | - | 1.22 / 6.25 | 849 / 929 | ~390 / ~474 | 0.82 | 458 / 455 | 3958 / 5379 | 647.9 / 835.7 | 4.67 / 3.13 | 1.65 / 7.21 | - | 0 (2 keys differ) | - |  |
| mask2former-swinl-ade20k | fp32 | 686.5 / 1000.8 | 0.69 | - | - | 0.72 / 6.28 | 1704 / 1760 | ~789 / ~853 | 0.93 | 915 / 907 | 4454 / 6088 | 755.4 / 1011.8 | 2.85 / 3.15 | 0.84 / 7.73 | - | 0 (2 keys differ) | - |  |
| memflow | fp16 | 5584.7 / 6862.6 | 0.81 | - | - | 5.59 / 10.81 | 2298 / 6650 | ~2286 / ~6553 | 0.35 | 12 / 97 | 4619 / 9191 | 5624.9 / 6888.3 | 0.04 / 0.07 | 5.61 / 10.82 | 0.0011 | 0.0011 | 0.000856 / 0.00105 (0.0018) |  |
| memflow | fp32 | 8012.9 / 9824.5 | 0.82 | - | - | 8.04 / 13.56 | 2476 / 8232 | ~2452 / ~8120 | 0.30 | 24 / 112 | 4877 / 11317 | 8009.5 / 9974.7 | 0.04 / 0.05 | 8.05 / 13.58 | 2.7e-05 | 2.7e-05 | 3.87e-06 / 2.26e-06 (0.0018) |  |
| memflow-kitti | fp16 | 5554.1 / 6742.1 | 0.82 | - | - | 5.58 / 10.28 | 2298 / 6650 | ~2286 / ~6553 | 0.35 | 12 / 97 | 4695 / 9177 | 5564.9 / 6882.6 | 0.04 / 0.05 | 5.59 / 10.29 | 0.0038 | 0.0038 | 0.000832 / 0.00102 (0.0018) |  |
| memflow-kitti | fp32 | 7996.1 / 9623.4 | 0.83 | - | - | 8.00 / 13.43 | 2476 / 8232 | ~2452 / ~8120 | 0.30 | 24 / 112 | 4874 / 11513 | 8077.2 / 9907.0 | 0.05 / 0.05 | 8.02 / 13.45 | 6.5e-05 | 6.5e-05 | 2.68e-06 / 1.9e-06 (0.0018) |  |
| memflow-sintel | fp16 | 5563.3 / 6734.9 | 0.83 | - | - | 5.69 / 10.39 | 2298 / 6650 | ~2286 / ~6553 | 0.35 | 12 / 97 | 4691 / 9179 | 5569.5 / 6779.9 | 0.05 / 0.05 | 5.70 / 10.40 | 0.001 | 0.001 | 0.000637 / 0.000858 (0.0018) |  |
| memflow-sintel | fp32 | 7991.1 / 9691.1 | 0.82 | - | - | 8.00 / 13.71 | 2476 / 8232 | ~2452 / ~8120 | 0.30 | 24 / 112 | 4875 / 11700 | 8006.2 / 9723.2 | 0.04 / 0.05 | 8.01 / 13.72 | 3.1e-05 | 3.1e-05 | 3.47e-06 / 1.53e-06 (0.0018) |  |
| memfof | fp16 | 2976.3 / 3252.5 | 0.92 | - | - | 3.01 / 5.77 | 1213 / 1408 | ~1073 / ~1254 | 0.86 | 140 / 154 | 3032 / 3616 | 3040.0 / 3287.7 | 0.05 / 0.04 | 3.04 / 5.78 | 0.00059 | 0.0006 | 0.00072 / 0.000669 (0.00099) |  |
| memfof | fp32 | 3514.6 / 3830.9 | 0.92 | - | - | 3.49 / 6.05 | 1388 / 1647 | ~1108 / ~1340 | 0.83 | 280 / 307 | 3399 / 3838 | 3674.6 / 3621.7 | 0.06 / 0.04 | 3.51 / 6.07 | 1.3e-06 | 1.3e-06 | 1.05e-06 / 1.33e-06 (0.00099) |  |
| memfof-kitti | fp16 | 14808.3 / 16249.6 | 0.91 | - | - | 14.11 / 17.77 | 2739 / 2865 | ~2599 / ~2711 | 0.96 | 140 / 154 | 4862 / 5019 | 14862.6 / 14873.6 | 0.23 / 0.08 | 14.23 / 17.81 | 0.00075 | 0.00073 | 0.00038 / 0.000453 (0.00099) |  |
| memfof-kitti | fp32 | 18090.9 / 19628.5 | 0.92 | - | - | 16.87 / 21.68 | 2648 / 3113 | ~2368 / ~2806 | 0.84 | 280 / 307 | 5169 / 5856 | 17512.2 / 19660.8 | 0.41 / 0.08 | 16.95 / 21.73 | 1.7e-06 | 1.7e-06 | 9.13e-07 / 9.22e-07 (0.00099) |  |
| memfof-sintel | fp16 | 13284.1 / 14969.7 | 0.89 | - | - | 13.29 / 19.42 | 2739 / 2865 | ~2599 / ~2711 | 0.96 | 140 / 154 | 4845 / 4982 | 14209.4 / 14876.4 | 0.24 / 0.09 | 13.39 / 19.47 | 0.00069 | 0.00067 | 0.000386 / 0.000429 (0.00099) |  |
| memfof-sintel | fp32 | 17538.2 / 19796.0 | 0.89 | - | - | 17.18 / 22.47 | 2648 / 3240 | ~2368 / ~2932 | 0.81 | 280 / 307 | 5197 / 6033 | 17760.5 / 18232.6 | 0.40 / 0.08 | 17.28 / 22.51 | 1.4e-06 | 1.3e-06 | 1.1e-06 / 1.09e-06 (0.00099) |  |
| memfof-spring | fp16 | 2856.5 / 3123.0 | 0.91 | - | - | 2.91 / 5.29 | 1213 / 1772 | ~1073 / ~1619 | 0.66 | 140 / 154 | 3189 / 3619 | 2878.2 / 3234.4 | 0.05 / 0.04 | 2.93 / 5.29 | 0.00066 | 0.00066 | 0.000589 / 0.000617 (0.00099) |  |
| memfof-spring | fp32 | 3234.1 / 3455.2 | 0.94 | - | - | 3.31 / 5.91 | 1388 / 1647 | ~1108 / ~1340 | 0.83 | 280 / 307 | 3426 / 3894 | 3277.9 / 3596.1 | 0.06 / 0.04 | 3.34 / 5.92 | 1.2e-06 | 1.2e-06 | 1.37e-06 / 1.47e-06 (0.00099) |  |
| mobilenet_v2.tv2_in1k | fp16 | 2.1 / 2.6 | 0.81 | - | - | 0.02 / 0.77 | 23 / 24 | ~16 / ~17 | 0.94 | 7 / 7 | 325 / 362 | 2.4 / 2.4 | 0.56 / 0.50 | 0.02 / 0.78 | 0.0012 | 0.0012 | - |  |
| mobilenet_v2.tv2_in1k | fp32 | 2.3 / 2.4 | 0.97 | - | - | 0.03 / 0.75 | 42 / 44 | ~28 / ~30 | 0.94 | 14 / 14 | 335 / 378 | 2.7 / 2.7 | 0.52 / 0.51 | 0.03 / 0.75 | 1.5e-06 | 1.5e-06 | - |  |
| mobilenet_v3_large.tv2_in1k | fp16 | 2.2 / 3.3 | 0.68 | - | - | 0.02 / 0.87 | 18 / 32 | ~8 / ~21 | 0.38 | 11 / 11 | 333 / 369 | 2.5 / 3.3 | 0.57 / 0.50 | 0.02 / 0.88 | 0.0025 | 0.0025 | - |  |
| mobilenet_v3_large.tv2_in1k | fp32 | 2.4 / 3.1 | 0.78 | 0.74-0.79 | 2.4 | 0.05 / 0.92 | 35 / 53 | 14 / 31 | 0.46 | 21 / 21 | 346 / 392 | 2.8 / 3.6 | 0.61 / 0.56 | 0.05 / 0.92 | 1.4e-06 | 1.4e-06 | - |  |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp16 | 1.8 / 1.8 | 1.01 | 0.92-1.09 | 1.8 | 0.02 / 0.55 | 16 / 23 | 8 / 15 | 0.55 | 8 / 8 | 356 / 404 | 2.3 / 2.2 | 0.80 / 0.70 | 0.02 / 0.56 | 0.0015 | 0.0015 | 0.0465 / 0.0325 (0.08) | worse |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp32 | 2.1 / 2.1 | 1.01 | - | - | 0.04 / 0.54 | 29 / 41 | ~13 / ~25 | 0.55 | 16 / 16 | 363 / 416 | 2.8 / 2.6 | 0.75 / 0.74 | 0.04 / 0.55 | 1.8e-06 | 1.8e-06 | 6.54e-06 / 1.06e-05 (0.08) |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp16 | 3.0 / 3.3 | 0.91 | - | - | 0.02 / 0.91 | 30 / 41 | ~11 / ~22 | 0.51 | 19 / 19 | 389 / 434 | 3.5 / 3.7 | 0.79 / 0.70 | 0.03 / 0.92 | 0.00074 | 0.00074 | - |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp32 | 3.2 / 3.5 | 0.91 | 0.87-0.95 | 4.0 | 0.05 / 0.93 | 58 / 74 | 20 / 37 | 0.57 | 37 / 37 | 414 / 472 | 3.7 / 4.2 | 0.85 / 0.74 | 0.05 / 0.94 | 5.4e-07 | 5.4e-07 | - |  |
| mvsanywhere | fp16 | 408.0 / 363.1 | 1.12 | 1.12-1.14 | 438.7 | 4.10 / 3.86 | 767 / 767 | 382 / 382 | 1.00 | 386 / 386 | 2399 / 2790 | 424.4 / 3512.5 | 0.27 / 0.11 | 4.17 / 7.22 | 0 | 0.045 (0.49 % off) | - | slower shipped-slower calls-differ |
| mvsanywhere | fp32 | 497.9 / 524.3 | 0.95 | 0.74-1.01 | 563.6 | 3.78 / 3.76 | 1407 / 1407 | 636 / 636 | 1.00 | 771 / 771 | 3803 / 4098 | 516.1 / 3601.8 | 0.19 / 0.12 | 3.87 / 6.94 | 0 | 0.00018 | - | calls-differ |
| nafnet-gopro | fp16 | 196.2 / 250.7 | 0.78 | 0.78-0.78 | 203.3 | 0.24 / 3.13 | 671 / 772 | 538 / 517 | 1.04 | 130 / 255 | 1357 / 2109 | 196.8 / 251.2 | 0.81 / 0.17 | 0.24 / 3.15 | 0.0019 | 1 (0.05 % off) | 0.0125 / 0.0142 (0.018) | heavier |
| nafnet-gopro | fp32 | 257.3 / 293.7 | 0.88 | 0.85-0.88 | 287.0 | 0.31 / 3.13 | 1169 / 859 | 907 / 475 | 1.91 | 259 / 384 | 2121 / 2678 | 258.1 / 294.3 | 0.27 / 0.17 | 0.31 / 3.16 | 2e-06 | 0.0098 | 6.55e-06 / 3.47e-06 (0.018) | heavier |
| nafnet-sidd | fp16 | 184.2 / 208.3 | 0.88 | 0.88-0.89 | 185.6 | 0.21 / 2.24 | 658 / 546 | 434 / 324 | 1.33 | 221 / 221 | 1216 / 1728 | 184.7 / 208.5 | 1.20 / 0.22 | 0.21 / 2.27 | 0.00088 | 0.0015 | 0.000206 / 0.000203 (0.0015) | heavier |
| nafnet-sidd | fp32 | 233.5 / 269.5 | 0.87 | 0.86-0.87 | 261.9 | 0.28 / 2.24 | 1184 / 896 | 739 / 454 | 1.63 | 442 / 442 | 1899 / 2294 | 234.3 / 270.3 | 0.41 / 0.22 | 0.28 / 2.28 | 7e-07 | 1.1e-06 | 9.81e-08 / 9.98e-08 (0.0015) | heavier |
| neuflow-v2 | fp16 | 415.5 / 956.5 | 0.43 | - | - | 0.46 / 3.60 | 1589 / 8232 | ~1563 / ~8196 | 0.19 | 26 / 36 | 3294 / 10645 | 423.2 / 787.2 | 0.04 / 0.04 | 0.48 / 3.62 | 0.51 (15 % off) | 0.51 (15 % off) | 0.000552 / 0.00067 (0.0014) |  |
| neuflow-v2 | fp32 | 464.3 / 807.3 | 0.58 | - | - | 0.45 / 2.72 | 1598 / 8229 | ~1563 / ~8184 | 0.19 | 35 / 45 | 3682 / 10779 | 476.4 / 878.0 | 0.04 / 0.03 | 0.47 / 2.72 | 0.091 (0.63 % off) | 0.091 (0.63 % off) | 2.71e-06 / 2.4e-06 (0.0014) |  |
| neuflow-v2-sintel | fp16 | 389.2 / 817.0 | 0.48 | - | - | 0.46 / 2.78 | 1589 / 8232 | ~1563 / ~8196 | 0.19 | 26 / 36 | 3314 / 10767 | 398.2 / 860.9 | 0.04 / 0.03 | 0.47 / 2.78 | 0.39 (10 % off) | 0.39 (10 % off) | 0.000581 / 0.000666 (0.0014) |  |
| neuflow-v2-sintel | fp32 | 460.0 / 800.1 | 0.57 | - | - | 0.50 / 2.63 | 1598 / 8229 | ~1563 / ~8184 | 0.19 | 35 / 45 | 3765 / 10781 | 476.0 / 1149.0 | 0.04 / 0.03 | 0.51 / 2.64 | 0.017 (0.0015 % off) | 0.017 (0.0015 % off) | 3.46e-06 / 2.57e-06 (0.0014) |  |
| neuflow-v2-things | fp16 | 387.5 / 856.2 | 0.45 | - | - | 0.41 / 4.17 | 1589 / 4341 | ~1563 / ~4305 | 0.36 | 26 / 36 | 3312 / 10771 | 394.7 / 843.6 | 0.04 / 0.03 | 0.42 / 4.19 | 0.41 (22 % off) | 0.41 (22 % off) | 0.000552 / 0.000774 (0.0014) |  |
| neuflow-v2-things | fp32 | 464.7 / 1468.0 | 0.32 | - | - | 0.46 / 3.30 | 1598 / 8229 | ~1563 / ~8184 | 0.19 | 35 / 45 | 3717 / 10689 | 453.3 / 796.4 | 0.04 / 0.03 | 0.47 / 3.31 | 0.14 (0.28 % off) | 0.14 (0.28 % off) | 3.17e-06 / 2.5e-06 (0.0014) |  |
| oneformer-swinl-ade20k | fp16 | 565.1 / 690.3 | 0.82 | 0.70-0.83 | 643.3 | 0.88 / 6.82 | 1364 / 1043 | 832 / 507 | 1.64 | 523 / 536 | 3884 / 3971 | 756.4 / 941.9 | 7.26 / 2.99 | 1.13 / 7.18 | - | 0 (2 keys differ) | - | heavier |
| oneformer-swinl-ade20k | fp32 | 735.6 / 948.3 | 0.78 | - | - | 1.18 / 6.84 | 1894 / 2040 | ~914 / ~970 | 0.94 | 980 / 1070 | 4557 / 4933 | 990.3 / 1177.7 | 2.71 / 2.76 | 1.44 / 7.11 | - | 0 (2 keys differ) | - |  |
| owlv2-b16 | fp16 | 283.1 / 278.7 | 1.02 | 1.01-1.02 | 287.9 | 0.55 / 1.25 | 505 / 338 | 197 / 164 | 1.20 | 296 / 175 | 1474 / 1074 | 284.5 / 289.1 | 1.06 / 2.36 | 0.56 / 1.39 | - | empty | - | heavier shipped-slower calls-differ |
| owlv2-b16 | fp32 | 364.9 / 366.5 | 1.00 | 0.99-1.00 | 370.3 | 0.54 / 1.34 | 881 / 544 | 285 / 200 | 1.42 | 585 / 343 | 1950 / 1260 | 366.4 / 377.1 | 0.68 / 2.35 | 0.56 / 1.49 | - | empty | - | heavier calls-differ |
| owlv2-l14 | fp16 | 1420.5 / 1408.0 | 1.01 | 1.01-1.01 | 1425.8 | 1.63 / 2.83 | 1055 / 793 | 208 / 194 | 1.07 | 835 / 599 | 2350 / 1896 | 1421.8 / 1420.5 | 3.58 / 4.14 | 1.65 / 3.08 | - | empty | - | heavier calls-differ |
| owlv2-l14 | fp32 | 1869.2 / 1858.2 | 1.01 | 1.00-1.01 | 1877.4 | 2.11 / 3.23 | 2087 / 1563 | 417 / 376 | 1.11 | 1659 / 1187 | 3337 / 2834 | 1871.0 / 1869.8 | 3.04 / 5.51 | 2.14 / 3.52 | - | empty | - | heavier calls-differ |
| pe-core-b16-224 | fp16 | 14.9 / 20.5 | 0.73 | - | - | 0.03 / 0.87 | 263 / 404 | ~87 / ~226 | 0.39 | 176 / 177 | 930 / 916 | 15.2 / 20.7 | 0.85 / 0.71 | 0.03 / 0.87 | 0.03 (0.024 % off) | 0.03 (0.024 % off) | - |  |
| pe-core-b16-224 | fp32 | 17.1 / 17.6 | 0.97 | 0.95-0.98 | 17.1 | 0.07 / 1.00 | 496 / 436 | 144 / 82 | 1.75 | 352 / 354 | 821 / 948 | 17.6 / 18.0 | 1.31 / 0.83 | 0.07 / 1.03 | 8.6e-05 | 8.6e-05 | - | heavier |
| pe-core-g14-448 | fp16 | 249.2 / 354.6 | 0.70 | - | - | 0.28 / 13.99 | 3664 / 3940 | ~84 / ~356 | 0.24 | 3580 / 3585 | 8522 / 5317 | 249.6 / 355.1 | 10.22 / 1.37 | 0.29 / 14.00 | 0.05 (0.014 % off) | 0.05 (0.014 % off) | - |  |
| pe-core-g14-448 | fp32 | 1180.9 / 725.9 | 1.63 | 0.04-11.10 | 292.6 | 4.54 / 37.55 | 7303 / 7296 | 143 / 127 | 1.13 | 7160 / 7169 | 7650 / 7742 | 293.3 / 293.0 | 10.77 / 1.31 | 4.56 / 70.09 | 4e-05 | 4e-05 | - | heavier |
| pe-core-l14-336 | fp16 | 56.7 / 77.5 | 0.73 | - | - | 0.08 / 1.58 | 680 / 881 | ~79 / ~278 | 0.29 | 601 / 602 | 2199 / 1562 | 57.0 / 77.8 | 0.99 / 0.76 | 0.08 / 1.58 | 0.0091 | 0.0091 | - |  |
| pe-core-l14-336 | fp32 | 66.2 / 66.3 | 1.00 | - | - | 0.10 / 1.59 | 1334 / 1372 | ~134 / ~167 | 0.80 | 1201 / 1205 | 1667 / 2067 | 66.7 / 66.5 | 0.84 / 0.77 | 0.11 / 1.60 | 3e-05 | 3e-05 | - |  |
| pe-core-s16-384 | fp16 | 6.4 / 8.0 | 0.79 | - | - | 0.02 / 0.88 | 94 / 200 | ~50 / ~155 | 0.32 | 45 / 45 | 500 / 597 | 6.6 / 8.3 | 0.83 / 0.76 | 0.02 / 0.88 | 0.0056 | 0.0056 | - |  |
| pe-core-s16-384 | fp32 | 7.2 / 7.3 | 1.00 | 0.98-1.00 | 7.2 | 0.05 / 0.91 | 170 / 166 | 81 / 77 | 1.05 | 89 / 90 | 496 / 566 | 7.5 / 7.6 | 0.88 / 0.75 | 0.05 / 0.92 | 1.3e-05 | 1.3e-05 | - | heavier |
| pe-core-t16-384 | fp16 | 3.7 / 4.5 | 0.82 | - | - | 0.02 / 0.83 | 34 / 67 | ~22 / ~56 | 0.41 | 12 / 12 | 376 / 431 | 3.8 / 4.7 | 0.82 / 0.71 | 0.02 / 0.83 | 0.011 (0.0053 % off) | 0.011 (0.0053 % off) | - |  |
| pe-core-t16-384 | fp32 | 3.9 / 4.2 | 0.92 | - | - | 0.04 / 0.81 | 62 / 59 | ~39 / ~36 | 1.06 | 23 / 23 | 381 / 424 | 4.1 / 4.3 | 0.77 / 0.71 | 0.04 / 0.81 | 4.6e-05 | 4.6e-05 | - |  |
| pe-spatial-b16-512 | fp16 | 14.5 / 19.7 | 0.73 | - | - | 0.03 / 0.84 | 251 / 390 | ~87 / ~226 | 0.39 | 164 / 164 | 906 / 903 | 14.7 / 20.0 | 0.84 / 0.71 | 0.03 / 0.84 | 0.0021 | 0.0021 | - |  |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.1 | 0.98 | 0.97-0.98 | 16.6 | 0.06 / 0.88 | 471 / 409 | 144 / 82 | 1.75 | 327 / 327 | 802 / 922 | 17.0 / 17.6 | 1.04 / 0.79 | 0.06 / 0.90 | 7e-06 | 7e-06 | - | heavier |
| pe-spatial-g14-448 | fp16 | 248.2 / 353.0 | 0.70 | - | - | 0.31 / 6.58 | 3614 / 3886 | ~84 / ~356 | 0.24 | 3531 / 3531 | 8452 / 5286 | 248.7 / 353.2 | 9.38 / 1.35 | 0.31 / 6.58 | 0.0069 | 0.0069 | - |  |
| pe-spatial-g14-448 | fp32 | 292.0 / 290.7 | 1.00 | 0.12-1.01 | 290.7 | 1.70 / 30.96 | 7204 / 7188 | 143 / 127 | 1.13 | 7061 / 7061 | 7556 / 7598 | 292.4 / 291.7 | 10.18 / 1.30 | 1.71 / 64.24 | 5.8e-06 | 5.8e-06 | - | heavier |
| pe-spatial-l14-448 | fp16 | 56.2 / 76.4 | 0.74 | - | - | 0.08 / 1.55 | 658 / 857 | ~79 / ~278 | 0.29 | 578 / 578 | 2148 / 1560 | 56.4 / 76.6 | 1.02 / 0.76 | 0.08 / 1.55 | 0.0024 | 0.0024 | - |  |
| pe-spatial-l14-448 | fp32 | 65.4 / 65.5 | 1.00 | - | - | 0.10 / 1.51 | 1290 / 1324 | ~134 / ~167 | 0.80 | 1157 / 1157 | 1619 / 2031 | 65.7 / 65.8 | 0.85 / 0.75 | 0.10 / 1.51 | 1.7e-06 | 1.7e-06 | - |  |
| pe-spatial-s16-512 | fp16 | 6.2 / 7.6 | 0.82 | - | - | 0.03 / 0.81 | 91 / 197 | ~49 / ~155 | 0.32 | 42 / 41 | 494 / 589 | 6.4 / 7.8 | 0.83 / 0.70 | 0.03 / 0.81 | 0.0055 | 0.0055 | - |  |
| pe-spatial-s16-512 | fp32 | 7.0 / 7.4 | 0.94 | 0.85-1.00 | 7.0 | 0.05 / 0.88 | 163 / 159 | 80 / 77 | 1.04 | 83 / 83 | 488 / 558 | 7.3 / 7.3 | 0.90 / 0.76 | 0.05 / 0.89 | 1.4e-05 | 1.4e-05 | - |  |
| pe-spatial-t16-512 | fp16 | 3.6 / 4.2 | 0.85 | - | - | 0.02 / 0.79 | 33 / 66 | ~22 / ~56 | 0.41 | 11 / 11 | 375 / 429 | 3.9 / 4.4 | 0.80 / 0.70 | 0.02 / 0.79 | 0.0077 | 0.0077 | - |  |
| pe-spatial-t16-512 | fp32 | 3.8 / 3.7 | 1.01 | - | - | 0.04 / 0.76 | 59 / 58 | ~38 / ~36 | 1.05 | 21 / 21 | 380 / 420 | 4.0 / 4.1 | 0.77 / 0.70 | 0.04 / 0.76 | 1.3e-05 | 1.3e-05 | - |  |
| pspnet-r101-d8-ade20k | fp16 | 308.3 / 315.7 | 0.98 | 0.93-1.02 | 331.7 | 0.58 / 2.20 | 1042 / 768 | 910 / 642 | 1.42 | 125 / 125 | 2151 / 2135 | 342.5 / 363.4 | 0.36 / 0.15 | 0.63 / 2.27 | 0.00078 | 0 | - | heavier |
| pspnet-r101-d8-ade20k | fp32 | 434.8 / 486.7 | 0.89 | - | - | 0.50 / 1.99 | 1547 / 1724 | ~1296 / ~1473 | 0.88 | 250 / 250 | 2654 / 2774 | 470.7 / 548.4 | 0.21 / 0.15 | 0.54 / 2.04 | 2.9e-07 | 0 | - |  |
| raft-stereo-eth3d | fp16 | 3409.4 / 4391.4 | 0.78 | 0.74-0.83 | 3253.8 | 3.62 / 10.09 | 1246 / 1166 | 1176 / 1084 | 1.08 | 35 / 82 | 3230 / 3756 | 3381.8 / 4388.9 | 0.30 / 0.15 | 3.64 / 10.18 | 0.058 (0.06 % off) | 0.049 (0.015 % off) | 0.00253 / 0.00243 (0.02) | heavier |
| raft-stereo-eth3d | fp32 | 4931.3 / 5316.3 | 0.93 | 0.88-0.97 | 5178.5 | 5.76 / 10.75 | 2334 / 1872 | 2256 / 1757 | 1.28 | 42 / 116 | 4216 / 4268 | 5000.8 / 5233.4 | 0.29 / 0.16 | 5.77 / 10.85 | 0.00012 | 0.00012 | 1.6e-06 / 1.7e-06 (0.02) | heavier |
| raft-stereo-middlebury | fp16 | 3204.2 / 4174.9 | 0.77 | 0.76-0.78 | 3264.2 | 3.25 / 9.38 | 1246 / 1042 | 1176 / 960 | 1.22 | 35 / 82 | 3244 / 3782 | 3244.0 / 4225.0 | 0.29 / 0.15 | 3.26 / 9.49 | 0.034 (0.15 % off) | 0.036 (0.13 % off) | 0.00612 / 0.00422 (0.02) | heavier worse |
| raft-stereo-middlebury | fp32 | 4926.0 / 5140.2 | 0.96 | 0.94-0.98 | 5150.2 | 5.07 / 10.29 | 2334 / 1872 | 2256 / 1757 | 1.28 | 42 / 116 | 4148 / 4223 | 4908.4 / 5155.2 | 0.28 / 0.14 | 5.08 / 10.39 | 0.00092 | 0.00098 | 3.37e-06 / 5.3e-06 (0.02) | heavier |
| raft-stereo-sceneflow | fp16 | 3365.1 / 4319.1 | 0.78 | 0.75-0.82 | 3256.0 | 3.34 / 9.56 | 1246 / 1354 | 1175 / 1272 | 0.92 | 35 / 82 | 3244 / 3746 | 3297.8 / 4312.2 | 0.20 / 0.15 | 3.35 / 9.66 | 0.27 (0.11 % off) | 0.28 (0.057 % off) | 0.00178 / 0.00198 (0.02) |  |
| raft-stereo-sceneflow | fp32 | 4928.5 / 5432.3 | 0.91 | 0.88-0.95 | 5376.1 | 5.49 / 11.57 | 2334 / 1808 | 2256 / 1693 | 1.33 | 42 / 116 | 4108 / 4161 | 4998.4 / 5488.5 | 0.30 / 0.16 | 5.51 / 11.67 | 0.00012 | 0.00012 | 1.46e-06 / 1.81e-06 (0.02) | heavier |
| real-hat-gan-x4 | fp16 | 7709.2 / 10941.2 | 0.70 | - | - | 7.76 / 14.16 | 2282 / 2938 | ~2185 / ~2603 | 0.84 | 96 / 335 | 5268 / 5502 | 7318.1 / 10434.9 | 0.76 / 0.60 | 7.78 / 14.17 | 0.032 (0.014 % off) | 0.033 (0.015 % off) | - |  |
| real-hat-gan-x4 | fp32 | 9875.1 / 21070.1 | 0.47 | 0.46-0.50 | 10985.3 | 10.31 / 20.73 | 4534 / 5166 | 4342 / 4755 | 0.91 | 189 / 411 | 7333 / 7907 | 9867.7 / 21056.9 | 0.90 / 0.60 | 10.33 / 20.79 | 0.0001 | 0.0001 | - |  |
| realesr-general-x4v3 | fp16 | 89.4 / 97.5 | 0.92 | - | - | 0.13 / 1.33 | 472 / 576 | ~460 / ~564 | 0.82 | 12 / 12 | 1023 / 1354 | 92.1 / 98.3 | 0.04 / 0.03 | 0.15 / 1.35 | 0.015 (0.0043 % off) | 0.015 (0.0043 % off) | - |  |
| realesr-general-x4v3 | fp32 | 146.6 / 158.5 | 0.93 | - | - | 0.20 / 1.39 | 938 / 938 | ~915 / ~915 | 1.00 | 23 / 23 | 1567 / 1759 | 149.2 / 158.7 | 0.04 / 0.03 | 0.22 / 1.40 | 0 | 0 | - |  |
| realesr-general-x4v3 | fp16 | 186.3 / 241.2 | 0.77 | - | - | 0.21 / 1.41 | 199 / 279 | ~197 / ~277 | 0.71 | 3 / 3 | 576 / 880 | 188.8 / 242.4 | 0.04 / 0.03 | 0.23 / 1.42 | 0 | 0 | - |  |
| realesr-general-x4v3 | fp32 | 227.9 / 322.7 | 0.71 | - | - | 0.26 / 1.43 | 296 / 360 | ~291 / ~355 | 0.82 | 5 / 5 | 673 / 991 | 230.4 / 324.3 | 0.04 / 0.03 | 0.28 / 1.43 | 0 | 0 | - |  |
| realesrgan-x2plus | fp16 | 499.2 / 506.6 | 0.99 | - | - | 0.56 / 3.99 | 1018 / 847 | ~855 / ~682 | 1.25 | 163 / 165 | 1797 / 2644 | 500.5 / 507.0 | 0.52 / 0.10 | 0.56 / 4.00 | 0.11 (2.6 % off) | 0.13 (3.8 % off) | - | heavier |
| realesrgan-x2plus | fp32 | 1276.3 / 1386.6 | 0.92 | - | - | 1.32 / 3.79 | 469 / 822 | ~404 / ~757 | 0.53 | 65 / 65 | 1407 / 2205 | 1275.0 / 1387.6 | 0.52 / 0.10 | 1.32 / 3.79 | 9.6e-05 | 0.00012 | - |  |
| realesrgan-x2plus | fp16 | 1006.8 / 1057.8 | 0.95 | - | - | 1.05 / 3.75 | 284 / 421 | ~249 / ~384 | 0.65 | 35 / 37 | 865 / 1753 | 1008.2 / 1056.9 | 0.52 / 0.10 | 1.05 / 3.75 | 0.14 (4 % off) | 0.18 (5.8 % off) | - |  |
| realesrgan-x2plus | fp32 | 1280.0 / 1366.0 | 0.94 | - | - | 1.32 / 3.83 | 469 / 566 | ~404 / ~501 | 0.81 | 65 / 65 | 1407 / 1629 | 1283.4 / 1363.2 | 0.62 / 0.10 | 1.33 / 3.83 | 9.6e-05 | 0.00012 | - |  |
| realesrgan-x4plus | fp16 | 1989.4 / 1987.8 | 1.00 | - | - | 2.02 / 5.38 | 1322 / 1292 | ~1159 / ~1127 | 1.03 | 163 / 165 | 3579 / 3247 | 1986.8 / 2004.0 | 0.53 / 0.10 | 2.04 / 5.39 | 0.014 (0.00023 % off) | 0.014 (0.00023 % off) | - | heavier |
| realesrgan-x4plus | fp32 | 5058.9 / 5356.9 | 0.94 | - | - | 5.18 / 7.80 | 1381 / 1883 | ~1316 / ~1818 | 0.72 | 65 / 65 | 3071 / 3045 | 5058.6 / 5572.1 | 0.62 / 0.10 | 5.21 / 7.81 | 1.9e-05 | 1.9e-05 | - |  |
| realesrgan-x4plus | fp16 | 4114.4 / 4302.8 | 0.96 | - | - | 4.14 / 7.24 | 801 / 867 | ~766 / ~830 | 0.92 | 35 / 37 | 2025 / 2360 | 4117.2 / 4304.2 | 0.61 / 0.10 | 4.16 / 7.25 | 0.017 (0.00052 % off) | 0.017 (0.00052 % off) | - |  |
| realesrgan-x4plus | fp32 | 5068.6 / 5405.7 | 0.94 | - | - | 5.22 / 7.88 | 1381 / 1531 | ~1316 / ~1466 | 0.90 | 65 / 65 | 3074 / 2993 | 5074.6 / 5579.2 | 0.64 / 0.10 | 5.24 / 7.89 | 1.9e-05 | 1.9e-05 | - |  |
| realesrgan-x4plus-anime-6b | fp16 | 685.8 / 658.9 | 1.04 | - | - | 0.76 / 2.22 | 1203 / 1171 | ~1159 / ~1127 | 1.03 | 44 / 44 | 3472 / 3324 | 665.0 / 668.5 | 0.46 / 0.05 | 0.78 / 2.23 | 0.18 (0.43 % off) | 0.21 (0.55 % off) | - | slower heavier |
| realesrgan-x4plus-anime-6b | fp32 | 1571.8 / 1717.7 | 0.92 | - | - | 1.60 / 2.99 | 1333 / 1836 | ~1316 / ~1818 | 0.72 | 18 / 18 | 3035 / 2958 | 1587.7 / 1726.8 | 0.48 / 0.05 | 1.62 / 3.00 | 0.00026 | 0.00031 | - |  |
| realesrgan-x4plus-anime-6b | fp16 | 1265.7 / 1349.1 | 0.94 | - | - | 1.29 / 2.72 | 775 / 840 | ~766 / ~830 | 0.92 | 9 / 10 | 2000 / 2266 | 1267.9 / 1351.3 | 0.60 / 0.05 | 1.31 / 2.74 | 0.23 (0.56 % off) | 0.27 (0.71 % off) | - |  |
| realesrgan-x4plus-anime-6b | fp32 | 1571.5 / 1733.4 | 0.91 | - | - | 1.62 / 3.13 | 1333 / 1483 | ~1316 / ~1466 | 0.90 | 18 / 18 | 2964 / 2925 | 1577.9 / 1731.8 | 0.59 / 0.04 | 1.64 / 3.14 | 0.00026 | 0.00031 | - |  |
| recover-cx | fp16 | 690.2 / 649.1 | 1.06 | 1.06-1.07 | 711.9 | 0.75 / 2.32 | 448 / 639 | 332 / 570 | 0.58 | 68 / 69 | 1602 / 2602 | 695.1 / 654.2 | 0.09 / 0.14 | 0.76 / 2.34 | 0.0013 | 0.0013 | 0.00219 / 0.00204 (0.008) | slower shipped-slower |
| recover-cx | fp32 | 825.4 / 850.7 | 0.97 | - | - | 0.89 / 2.38 | 831 / 850 | ~694 / ~712 | 0.98 | 137 / 138 | 2807 / 3157 | 833.2 / 865.3 | 0.05 / 0.07 | 0.91 / 2.39 | 6.4e-07 | 6.4e-07 | 1.26e-06 / 1.24e-06 (0.008) |  |
| recover-mn | fp16 | 215.3 / 255.7 | 0.84 | - | - | 0.28 / 1.80 | 392 / 622 | ~371 / ~601 | 0.62 | 20 / 21 | 1944 / 2929 | 220.5 / 256.8 | 0.04 / 0.07 | 0.30 / 1.81 | 0.0016 | 0.0016 | 0.00568 / 0.00509 (0.008) |  |
| recover-mn | fp32 | 282.6 / 389.4 | 0.73 | 0.71-0.74 | 325.0 | 0.35 / 1.80 | 734 / 721 | 647 / 680 | 0.95 | 40 / 41 | 2755 / 3066 | 291.7 / 394.3 | 0.09 / 0.10 | 0.36 / 1.85 | 9.4e-07 | 9.4e-07 | 2.18e-06 / 2.85e-06 (0.008) |  |
| recover-rn | fp16 | 374.3 / 393.3 | 0.95 | - | - | 0.43 / 1.63 | 401 / 719 | ~371 / ~689 | 0.54 | 30 / 30 | 1455 / 2870 | 378.9 / 395.6 | 0.04 / 0.06 | 0.45 / 1.64 | 0.0013 | 0.0013 | 0.00243 / 0.00212 (0.008) |  |
| recover-rn | fp32 | 454.4 / 528.5 | 0.86 | - | - | 0.54 / 1.74 | 753 / 1011 | ~694 / ~951 | 0.73 | 59 / 60 | 2204 / 3162 | 458.6 / 530.6 | 0.04 / 0.06 | 0.55 / 1.74 | 8.2e-07 | 8.2e-07 | 1.08e-06 / 1.27e-06 (0.008) |  |
| regnety_160.tv2_in1k | fp16 | 17.5 / 18.6 | 0.94 | - | - | 0.04 / 0.99 | 198 / 280 | ~38 / ~119 | 0.33 | 160 / 160 | 754 / 928 | 18.0 / 19.0 | 0.73 / 0.57 | 0.04 / 1.00 | 0.00029 | 0.00029 | - |  |
| regnety_160.tv2_in1k | fp32 | 18.4 / 23.6 | 0.78 | - | - | 0.05 / 0.96 | 377 / 574 | ~57 / ~255 | 0.23 | 319 / 319 | 856 / 1282 | 18.9 / 24.1 | 0.66 / 0.56 | 0.05 / 0.96 | 3.5e-07 | 3.5e-07 | - |  |
| repvit_m0_9.dist_300e_in1k | fp16 | 3.5 / 4.5 | 0.77 | - | - | 0.03 / 1.31 | 18 / 35 | ~8 / ~24 | 0.35 | 10 / 11 | 367 / 414 | 4.0 / 5.0 | 0.79 / 0.70 | 0.03 / 1.32 | 0.0078 | 0.0078 | - |  |
| repvit_m0_9.dist_300e_in1k | fp32 | 3.9 / 4.9 | 0.79 | - | - | 0.05 / 1.26 | 31 / 62 | ~11 / ~41 | 0.29 | 20 / 21 | 384 / 439 | 4.2 / 6.3 | 0.75 / 0.70 | 0.05 / 1.26 | 1.4e-06 | 1.4e-06 | - |  |
| repvit_m0_9.dist_450e_in1k | fp16 | 3.5 / 4.5 | 0.78 | - | - | 0.03 / 1.36 | 17 / 33 | ~7 / ~22 | 0.34 | 10 / 11 | 363 / 414 | 3.9 / 5.0 | 0.78 / 0.71 | 0.03 / 1.36 | 0.012 (0.2 % off) | 0.012 (0.2 % off) | - |  |
| repvit_m0_9.dist_450e_in1k | fp32 | 3.7 / 4.8 | 0.77 | - | - | 0.05 / 1.32 | 32 / 61 | ~12 / ~39 | 0.33 | 20 / 21 | 383 / 439 | 4.3 / 5.4 | 0.76 / 0.74 | 0.05 / 1.32 | 2.3e-06 | 2.3e-06 | - |  |
| repvit_m1_0.dist_300e_in1k | fp16 | 3.8 / 5.0 | 0.76 | - | - | 0.03 / 1.31 | 21 / 41 | ~8 / ~26 | 0.32 | 14 / 15 | 377 / 427 | 4.3 / 5.5 | 0.79 / 0.69 | 0.04 / 1.31 | 0.01 (0.1 % off) | 0.01 (0.1 % off) | - |  |
| repvit_m1_0.dist_300e_in1k | fp32 | 4.1 / 5.3 | 0.77 | - | - | 0.05 / 1.26 | 40 / 78 | ~13 / ~50 | 0.27 | 27 / 28 | 402 / 453 | 4.6 / 5.9 | 0.73 / 0.70 | 0.05 / 1.27 | 1.9e-06 | 1.9e-06 | - |  |
| repvit_m1_0.dist_450e_in1k | fp16 | 3.7 / 4.9 | 0.76 | - | - | 0.03 / 1.32 | 21 / 45 | ~8 / ~30 | 0.28 | 14 / 15 | 377 / 426 | 4.2 / 5.5 | 0.79 / 0.71 | 0.03 / 1.32 | 0.0065 | 0.0065 | - |  |
| repvit_m1_0.dist_450e_in1k | fp32 | 4.1 / 5.3 | 0.78 | - | - | 0.05 / 1.28 | 40 / 77 | ~13 / ~48 | 0.28 | 27 / 28 | 398 / 452 | 4.5 / 5.8 | 0.73 / 0.69 | 0.05 / 1.28 | 1.3e-06 | 1.3e-06 | - |  |
| repvit_m1_1.dist_300e_in1k | fp16 | 3.6 / 4.6 | 0.78 | - | - | 0.03 / 1.25 | 24 / 45 | ~9 / ~28 | 0.33 | 16 / 17 | 386 / 433 | 4.0 / 5.2 | 0.80 / 0.72 | 0.03 / 1.25 | 0.0061 | 0.0061 | - |  |
| repvit_m1_1.dist_300e_in1k | fp32 | 3.9 / 5.2 | 0.76 | - | - | 0.05 / 1.20 | 47 / 83 | ~16 / ~49 | 0.33 | 32 / 34 | 408 / 457 | 4.4 / 5.7 | 0.73 / 0.71 | 0.05 / 1.21 | 8.9e-07 | 8.9e-07 | - |  |
| repvit_m1_1.dist_450e_in1k | fp16 | 3.5 / 4.6 | 0.77 | - | - | 0.03 / 1.24 | 24 / 46 | ~9 / ~29 | 0.32 | 16 / 17 | 385 / 430 | 4.0 / 5.1 | 0.78 / 0.70 | 0.03 / 1.24 | 0.021 (5.2 % off) | 0.021 (5.2 % off) | - |  |
| repvit_m1_1.dist_450e_in1k | fp32 | 3.8 / 5.0 | 0.76 | - | - | 0.05 / 1.25 | 46 / 83 | ~15 / ~49 | 0.31 | 32 / 34 | 406 / 457 | 4.4 / 5.5 | 0.73 / 0.73 | 0.05 / 1.25 | 2.6e-06 | 2.6e-06 | - |  |
| repvit_m1_5.dist_300e_in1k | fp16 | 5.4 / 7.3 | 0.74 | - | - | 0.04 / 1.87 | 39 / 64 | ~12 / ~36 | 0.36 | 27 / 28 | 429 / 481 | 6.0 / 8.4 | 0.86 / 0.75 | 0.04 / 1.87 | 0.0073 | 0.0073 | - |  |
| repvit_m1_5.dist_300e_in1k | fp32 | 6.1 / 8.6 | 0.71 | - | - | 0.06 / 1.75 | 73 / 120 | ~20 / ~64 | 0.32 | 54 / 56 | 466 / 535 | 6.5 / 9.0 | 0.80 / 0.75 | 0.06 / 1.75 | 1.2e-06 | 1.2e-06 | - |  |
| repvit_m1_5.dist_450e_in1k | fp16 | 5.5 / 7.6 | 0.73 | - | - | 0.04 / 1.79 | 39 / 66 | ~12 / ~38 | 0.33 | 27 / 28 | 426 / 482 | 6.2 / 8.2 | 0.86 / 0.73 | 0.04 / 1.79 | 0.037 (12 % off) | 0.037 (12 % off) | - |  |
| repvit_m1_5.dist_450e_in1k | fp32 | 6.2 / 8.4 | 0.73 | - | - | 0.06 / 1.71 | 74 / 126 | ~20 / ~70 | 0.29 | 54 / 56 | 466 / 537 | 6.7 / 9.0 | 0.76 / 0.74 | 0.06 / 1.71 | 2e-06 | 2e-06 | - |  |
| repvit_m2_3.dist_300e_in1k | fp16 | 9.4 / 12.0 | 0.78 | - | - | 0.04 / 2.41 | 63 / 108 | ~19 / ~62 | 0.31 | 45 / 46 | 491 / 549 | 9.9 / 13.0 | 0.93 / 0.72 | 0.04 / 2.41 | 0.0098 | 0.0098 | - |  |
| repvit_m2_3.dist_300e_in1k | fp32 | 10.4 / 13.9 | 0.75 | - | - | 0.06 / 2.18 | 120 / 194 | ~31 / ~103 | 0.31 | 88 / 91 | 550 / 636 | 11.1 / 14.8 | 0.80 / 0.73 | 0.07 / 2.18 | 1.2e-06 | 1.2e-06 | - |  |
| repvit_m2_3.dist_450e_in1k | fp16 | 9.1 / 12.3 | 0.74 | - | - | 0.04 / 2.33 | 63 / 109 | ~18 / ~63 | 0.30 | 45 / 46 | 490 / 543 | 9.7 / 13.0 | 0.91 / 0.73 | 0.04 / 2.33 | 0.007 | 0.007 | - |  |
| repvit_m2_3.dist_450e_in1k | fp32 | 10.6 / 13.9 | 0.76 | - | - | 0.07 / 2.22 | 120 / 194 | ~32 / ~103 | 0.31 | 88 / 91 | 551 / 640 | 11.2 / 14.6 | 0.85 / 0.73 | 0.07 / 2.22 | 9.2e-07 | 9.2e-07 | - |  |
| resnet152.tv2_in1k | fp16 | 15.7 / 17.2 | 0.91 | - | - | 0.03 / 1.44 | 177 / 224 | ~62 / ~109 | 0.58 | 115 / 115 | 552 / 670 | 16.1 / 17.9 | 0.78 / 0.65 | 0.04 / 1.44 | 0.00062 | 0.00062 | - |  |
| resnet152.tv2_in1k | fp32 | 17.8 / 20.7 | 0.86 | - | - | 0.05 / 1.41 | 335 / 384 | ~106 / ~155 | 0.69 | 229 / 229 | 680 / 844 | 18.2 / 21.2 | 0.71 / 0.61 | 0.05 / 1.41 | 4.6e-07 | 4.6e-07 | - |  |
| resnet18.tv_in1k | fp16 | 3.4 / 3.9 | 0.88 | - | - | 0.02 / 0.49 | 30 / 52 | ~8 / ~30 | 0.29 | 22 / 22 | 357 / 436 | 3.7 / 4.2 | 0.54 / 0.49 | 0.02 / 0.49 | 0.00055 | 0.00055 | - |  |
| resnet18.tv_in1k | fp32 | 3.9 / 4.5 | 0.86 | - | - | 0.03 / 0.48 | 59 / 102 | ~14 / ~58 | 0.26 | 45 / 45 | 371 / 483 | 4.3 / 4.9 | 0.51 / 0.51 | 0.03 / 0.48 | 3.7e-07 | 3.7e-07 | - |  |
| resnet50.tv2_in1k | fp16 | 6.2 / 6.8 | 0.91 | - | - | 0.02 / 0.92 | 76 / 99 | ~28 / ~50 | 0.56 | 49 / 49 | 443 / 514 | 6.5 / 7.2 | 0.59 / 0.50 | 0.02 / 0.92 | 0.00082 | 0.00082 | - |  |
| resnet50.tv2_in1k | fp32 | 7.0 / 8.1 | 0.87 | - | - | 0.04 / 0.76 | 147 / 192 | ~50 / ~94 | 0.54 | 97 / 97 | 478 / 601 | 7.4 / 8.5 | 0.55 / 0.51 | 0.04 / 0.76 | 5.5e-07 | 5.5e-07 | - |  |
| retinaface-mobilenet0.25 | fp16 | 10.6 / 11.9 | 0.89 | - | - | 0.04 / 1.07 | 92 / 178 | ~91 / ~177 | 0.52 | 1 / 1 | 493 / 618 | 15.5 / 17.2 | 0.04 / 0.12 | 0.06 / 1.07 | 0.0077 | empty | - |  |
| retinaface-mobilenet0.25 | fp32 | 12.9 / 14.7 | 0.88 | - | - | 0.05 / 1.05 | 136 / 180 | ~134 / ~178 | 0.76 | 2 / 2 | 598 / 813 | 18.1 / 20.1 | 0.04 / 0.11 | 0.06 / 1.05 | 3.7e-06 | empty | - |  |
| retinaface-resnet50 | fp16 | 120.2 / 133.0 | 0.90 | 0.90-0.91 | 125.5 | 0.33 / 1.40 | 473 / 372 | 397 / 320 | 1.24 | 52 / 52 | 1508 / 1677 | 125.8 / 140.1 | 0.08 / 0.24 | 0.34 / 1.42 | 0.014 (0.0017 % off) | empty | - | heavier |
| retinaface-resnet50 | fp32 | 151.2 / 173.0 | 0.87 | 0.87-0.88 | 166.7 | 0.33 / 1.33 | 902 / 653 | 774 / 549 | 1.41 | 104 / 104 | 2558 / 2441 | 157.2 / 179.8 | 0.11 / 0.18 | 0.34 / 1.36 | 1.2e-05 | empty | - | heavier |
| rf-detr-keypoint-preview | fp16 | 63.3 / 62.1 | 1.02 | 1.01-1.03 | 64.5 | 0.11 / 3.32 | 244 / 225 | 150 / 154 | 0.97 | 95 / 71 | 894 / 956 | 64.6 / 63.6 | 2.75 / 2.20 | 0.11 / 3.33 | - | empty | 0.0079 / 0.0358 (0.08) | shipped-slower |
| rf-detr-keypoint-preview | fp32 | 73.3 / 75.2 | 0.97 | 0.97-0.98 | 75.0 | 0.11 / 3.65 | 316 / 241 | 178 / 99 | 1.80 | 138 / 142 | 998 / 1055 | 75.2 / 76.4 | 2.30 / 2.11 | 0.12 / 3.67 | - | empty | 2.36e-05 / 2.02e-05 (0.08) | heavier |
| rf-detr-l-coco | fp16 | 46.2 / 42.4 | 1.09 | 1.07-1.12 | 46.2 | 0.09 / 2.15 | 170 / 191 | 97 / 131 | 0.74 | 73 / 60 | 691 / 755 | 49.2 / 46.4 | 3.46 / 2.30 | 0.10 / 2.17 | 0.98 (69 % off) | empty | 0.124 / 0.121 (0.32) | slower shipped-slower |
| rf-detr-l-coco | fp32 | 53.8 / 52.7 | 1.02 | 1.01-1.02 | 54.8 | 0.09 / 2.05 | 248 / 226 | 132 / 107 | 1.23 | 116 / 120 | 784 / 852 | 56.8 / 56.6 | 2.42 / 2.27 | 0.10 / 2.07 | 0.42 (3.6 % off) | empty | 3.39e-05 / 1.81e-05 (0.32) | heavier shipped-slower |
| rf-detr-m-coco | fp16 | 32.7 / 32.4 | 1.01 | - | - | 0.25 / 2.35 | 173 / 179 | ~100 / ~119 | 0.84 | 73 / 59 | 683 / 729 | 36.2 / 34.5 | 2.46 / 2.56 | 0.25 / 2.36 | 0.9 (70 % off) | empty | 0.036 / 0.0519 (0.13) |  |
| rf-detr-m-coco | fp32 | 37.1 / 36.7 | 1.01 | 0.98-1.04 | 38.8 | 0.09 / 2.06 | 209 / 226 | 94 / 107 | 0.88 | 115 / 119 | 751 / 803 | 40.4 / 40.3 | 2.69 / 2.32 | 0.09 / 2.08 | 0.00048 | empty | 2.91e-05 / 4.29e-05 (0.13) |  |
| rf-detr-n-coco | fp16 | 16.3 / 16.1 | 1.02 | - | - | 0.23 / 2.14 | 126 / 146 | ~65 / ~93 | 0.71 | 61 / 53 | 591 / 641 | 19.7 / 20.3 | 2.78 / 2.42 | 0.23 / 2.15 | 0.93 (51 % off) | empty | 0.00925 / 0.0127 (0.087) |  |
| rf-detr-n-coco | fp32 | 18.8 / 19.3 | 0.97 | - | - | 0.28 / 2.06 | 175 / 190 | ~72 / ~84 | 0.87 | 102 / 106 | 658 / 703 | 22.0 / 22.3 | 2.53 / 2.42 | 0.29 / 2.06 | 0.75 (0.56 % off) | empty | 1.99e-05 / 3.36e-05 (0.087) |  |
| rf-detr-s-coco | fp16 | 26.2 / 24.5 | 1.07 | 1.03-1.10 | 26.6 | 0.07 / 1.98 | 129 / 165 | 63 / 109 | 0.58 | 67 / 56 | 642 / 698 | 29.8 / 27.8 | 2.97 / 2.29 | 0.08 / 1.99 | 0.95 (71 % off) | empty | 0.00598 / 0.0223 (0.13) | slower shipped-slower |
| rf-detr-s-coco | fp32 | 28.8 / 29.4 | 0.98 | - | - | 0.16 / 1.94 | 207 / 219 | ~98 / ~106 | 0.93 | 109 / 113 | 719 / 752 | 31.8 / 32.6 | 2.35 / 2.22 | 0.16 / 1.95 | 0.86 (1.4 % off) | empty | 7.63e-06 / 1.53e-05 (0.13) |  |
| rf-detr-seg-2xl-coco | fp16 | 128.0 / 124.8 | 1.03 | 0.95-1.05 | 134.0 | 0.17 / 2.61 | 355 / 292 | 268 / 223 | 1.20 | 88 / 69 | 1182 / 1212 | 132.2 / 128.9 | 3.00 / 2.31 | 0.18 / 2.63 | 1 (73 % off) | empty | 0.278 / 0.244 (0.51) | heavier worse |
| rf-detr-seg-2xl-coco | fp32 | 158.9 / 161.8 | 0.98 | 0.96-0.99 | 169.6 | 0.20 / 2.51 | 452 / 469 | 319 / 331 | 0.96 | 134 / 138 | 1420 / 1526 | 163.0 / 166.0 | 2.46 / 2.32 | 0.22 / 2.54 | 0.00069 | empty | 7.3e-05 / 5.33e-05 (0.51) |  |
| rf-detr-seg-l-coco | fp16 | 49.5 / 46.5 | 1.07 | 0.99-1.08 | 50.8 | 0.10 / 2.34 | 172 / 222 | 92 / 159 | 0.58 | 80 / 64 | 750 / 954 | 52.4 / 49.6 | 2.63 / 2.28 | 0.10 / 2.36 | 0.96 (84 % off) | empty | 0.187 / 0.497 (0.52) |  |
| rf-detr-seg-l-coco | fp32 | 58.8 / 60.1 | 0.98 | - | - | 0.34 / 2.78 | 267 / 301 | ~142 / ~174 | 0.82 | 124 / 127 | 879 / 931 | 60.1 / 62.8 | 2.44 / 2.51 | 0.34 / 2.79 | 0.00097 | empty | 6.07e-05 / 4.48e-05 (0.52) |  |
| rf-detr-seg-m-coco | fp16 | 39.0 / 35.6 | 1.10 | 1.06-1.13 | 39.9 | 0.09 / 2.33 | 154 / 211 | 74 / 148 | 0.51 | 80 / 63 | 732 / 904 | 41.2 / 38.6 | 2.62 / 2.25 | 0.10 / 2.35 | 0.93 (86 % off) | empty | 0.289 / 0.214 (0.37) | slower worse shipped-slower |
| rf-detr-seg-m-coco | fp32 | 45.1 / 44.2 | 1.02 | 1.01-1.04 | 46.9 | 0.09 / 2.25 | 243 / 289 | 119 / 162 | 0.74 | 124 / 126 | 857 / 1001 | 48.2 / 47.5 | 2.42 / 2.24 | 0.10 / 2.27 | 0.0042 | empty | 2.79e-05 / 4.51e-05 (0.37) | shipped-slower |
| rf-detr-seg-n-coco | fp16 | 22.8 / 21.9 | 1.04 | 1.01-1.07 | 22.9 | 0.08 / 2.22 | 118 / 150 | 44 / 91 | 0.49 | 74 / 59 | 622 / 784 | 24.8 / 23.6 | 2.81 / 2.29 | 0.09 / 2.23 | 0.94 (77 % off) | empty | 0.0535 / 0.223 (0.34) |  |
| rf-detr-seg-n-coco | fp32 | 27.0 / 26.4 | 1.02 | 0.97-1.10 | 26.8 | 0.08 / 2.11 | 186 / 249 | 69 / 131 | 0.53 | 117 / 118 | 705 / 832 | 28.0 / 27.4 | 2.45 / 2.28 | 0.09 / 2.13 | 0.00012 | empty | 5.99e-05 / 0.00012 (0.34) |  |
| rf-detr-seg-s-coco | fp16 | 29.7 / 28.4 | 1.05 | 1.03-1.11 | 30.3 | 0.08 / 2.21 | 136 / 176 | 62 / 117 | 0.53 | 74 / 59 | 662 / 810 | 31.5 / 30.0 | 2.60 / 2.25 | 0.09 / 2.22 | 0.93 (79 % off) | empty | 0.102 / 0.119 (0.24) | slower shipped-slower |
| rf-detr-seg-s-coco | fp32 | 34.4 / 33.6 | 1.03 | 0.99-1.06 | 34.7 | 0.08 / 2.18 | 216 / 266 | 99 / 148 | 0.67 | 117 / 119 | 764 / 906 | 36.6 / 34.5 | 2.50 / 2.27 | 0.09 / 2.20 | 0.001 | empty | 5.24e-05 / 1.62e-05 (0.24) |  |
| rf-detr-seg-xl-coco | fp16 | 82.3 / 78.1 | 1.05 | 1.02-1.16 | 82.2 | 0.13 / 2.58 | 213 / 263 | 126 / 195 | 0.65 | 87 / 68 | 931 / 1098 | 86.6 / 82.4 | 2.71 / 2.31 | 0.14 / 2.60 | 0.99 (79 % off) | empty | 0.462 / 0.682 (1.1) | shipped-slower |
| rf-detr-seg-xl-coco | fp32 | 98.5 / 99.9 | 0.99 | 0.95-1.01 | 101.1 | 0.14 / 2.46 | 346 / 354 | 214 / 219 | 0.98 | 132 / 136 | 1121 / 1218 | 102.8 / 103.2 | 2.53 / 2.37 | 0.15 / 2.49 | 0.00081 | empty | 0.073 / 0.0305 (1.1) | worse |
| rmbg-2.0 | fp16 | 882.0 / 1030.4 | 0.86 | - | - | 1.09 / 7.55 | 1062 / 1745 | ~588 / ~1246 | 0.47 | 474 / 499 | 3431 / 4458 | 894.8 / 1031.6 | 1.32 / 0.08 | 1.16 / 7.57 | 1 (1e+02 % off) | 0.0065 | 8.57e-05 / 8.41e-05 (0.00013) |  |
| rmbg-2.0 | fp32 | 1111.0 / 1399.2 | 0.79 | - | - | 1.62 / 8.56 | 2129 / 3386 | ~1214 / ~2444 | 0.50 | 915 / 941 | 4401 / 5999 | 1127.3 / 1404.3 | 1.57 / 0.17 | 1.71 / 8.58 | 1 (1e+02 % off) | 1.3e-05 | 8.49e-05 / 8.48e-05 (0.00013) |  |
| rtdetrv2-l-coco | fp16 | 48.6 / 55.6 | 0.87 | 0.85-0.90 | 53.7 | 0.10 / 2.81 | 270 / 259 | 187 / 176 | 1.06 | 83 / 83 | 976 / 1036 | 57.2 / 65.1 | 0.56 / 0.13 | 0.11 / 2.85 | 0.95 (61 % off) | 0.84 (3.3 % off) | - | heavier |
| rtdetrv2-l-coco | fp32 | 61.5 / 66.8 | 0.92 | 0.81-1.03 | 58.8 | 0.15 / 2.56 | 457 / 424 | 296 / 264 | 1.12 | 161 / 161 | 1317 / 1284 | 67.0 / 75.7 | 0.21 / 0.13 | 0.16 / 2.59 | 1.3e-05 | 1.5e-06 | - | heavier |
| rtdetrv2-m-coco | fp16 | 35.7 / 39.2 | 0.91 | - | - | 0.06 / 2.00 | 224 / 258 | ~162 / ~196 | 0.83 | 62 / 62 | 751 / 866 | 43.8 / 47.3 | 0.23 / 0.10 | 0.07 / 2.02 | 0.97 (45 % off) | empty (3 keys differ) | - |  |
| rtdetrv2-m-coco | fp32 | 42.0 / 47.0 | 0.89 | - | - | 0.08 / 1.89 | 418 / 453 | ~299 / ~334 | 0.90 | 119 / 119 | 985 / 1114 | 49.8 / 54.8 | 0.16 / 0.10 | 0.09 / 1.90 | 4.1e-05 | 1.8e-06 | - |  |
| rtdetrv2-ms-coco | fp16 | 37.4 / 42.2 | 0.89 | - | - | 0.06 / 1.98 | 246 / 242 | ~181 / ~177 | 1.02 | 65 / 65 | 846 / 907 | 45.3 / 50.1 | 0.25 / 0.12 | 0.07 / 2.00 | 0.98 (52 % off) | 0.63 (2.5 % off) | - |  |
| rtdetrv2-ms-coco | fp32 | 44.5 / 52.3 | 0.85 | 0.83-0.89 | 45.8 | 0.09 / 2.27 | 412 / 391 | 286 / 265 | 1.08 | 126 / 126 | 1078 / 1128 | 53.2 / 60.5 | 0.35 / 0.12 | 0.10 / 2.30 | 1.1e-05 | 1.3e-06 | - | heavier |
| rtdetrv2-s-coco | fp16 | 24.8 / 27.4 | 0.90 | - | - | 0.05 / 1.88 | 150 / 244 | ~110 / ~204 | 0.54 | 40 / 40 | 624 / 813 | 33.1 / 35.1 | 0.19 / 0.08 | 0.06 / 1.90 | 0.97 (29 % off) | empty (3 keys differ) | - |  |
| rtdetrv2-s-coco | fp32 | 28.7 / 33.3 | 0.86 | - | - | 0.07 / 1.72 | 271 / 362 | ~194 / ~285 | 0.68 | 77 / 77 | 780 / 912 | 40.2 / 41.5 | 0.13 / 0.08 | 0.08 / 1.73 | 1.6e-05 | 1.1e-05 | - |  |
| rtdetrv2-x-coco | fp16 | 79.4 / 86.6 | 0.92 | - | - | 0.17 / 2.54 | 374 / 389 | ~229 / ~244 | 0.94 | 146 / 146 | 1357 / 1192 | 88.5 / 94.9 | 0.38 / 0.17 | 0.18 / 2.55 | 0.98 (37 % off) | 0.42 (1.6 % off) | - |  |
| rtdetrv2-x-coco | fp32 | 92.0 / 104.1 | 0.88 | 0.88-0.89 | 94.2 | 0.26 / 2.64 | 727 / 679 | 441 / 393 | 1.12 | 286 / 286 | 1944 / 1610 | 102.9 / 114.2 | 0.63 / 0.18 | 0.27 / 2.69 | 2.4e-05 | 4.5e-07 | - | heavier |
| rtmdet-l-coco | fp16 | 52.2 / 63.1 | 0.83 | - | - | 0.08 / 1.60 | 292 / 307 | ~181 / ~198 | 0.91 | 112 / 109 | 841 / 1124 | 58.0 / 68.3 | 0.05 / 0.17 | 0.55 / 1.67 | 0.7 (68 % off) | empty (3 keys differ) | - |  |
| rtmdet-l-coco | fp32 | 61.2 / 77.5 | 0.79 | - | - | 0.09 / 1.54 | 554 / 573 | ~337 / ~356 | 0.95 | 218 / 218 | 1298 / 1579 | 67.2 / 82.8 | 0.06 / 0.17 | 0.56 / 1.61 | 0.7 (68 % off) | 4.3e-06 | - |  |
| rtmdet-m-coco | fp16 | 33.5 / 39.7 | 0.84 | - | - | 0.07 / 1.40 | 179 / 242 | ~125 / ~190 | 0.66 | 54 / 53 | 645 / 1000 | 39.3 / 45.2 | 0.04 / 0.12 | 0.54 / 1.47 | 0.81 (62 % off) | 0.0018 | - |  |
| rtmdet-m-coco | fp32 | 37.1 / 46.2 | 0.80 | - | - | 0.07 / 1.37 | 326 / 379 | ~221 / ~274 | 0.81 | 105 / 105 | 871 / 1303 | 43.0 / 51.7 | 0.04 / 0.12 | 0.54 / 1.45 | 0.81 (62 % off) | 1.1e-06 | - |  |
| rtmdet-m-person | fp16 | 33.4 / 39.3 | 0.85 | - | - | 0.07 / 1.41 | 179 / 240 | ~125 / ~187 | 0.67 | 54 / 53 | 557 / 944 | 37.9 / 43.8 | 0.04 / 0.08 | 0.08 / 1.42 | 0.046 (0.13 % off) | empty | - |  |
| rtmdet-m-person | fp32 | 37.0 / 46.2 | 0.80 | - | - | 0.08 / 1.39 | 318 / 379 | ~214 / ~274 | 0.78 | 104 / 104 | 784 / 1251 | 41.8 / 50.6 | 0.04 / 0.09 | 0.09 / 1.39 | 2e-05 | empty | - |  |
| rtmdet-nano-person | fp16 | 4.0 / 6.4 | 0.63 | - | - | 0.04 / 1.37 | 14 / 32 | ~12 / ~30 | 0.44 | 2 / 2 | 358 / 410 | 6.7 / 8.5 | 0.04 / 0.04 | 0.50 / 1.43 | 0.062 (1.3 % off) | 0.4 (14 % off) | - |  |
| rtmdet-nano-person | fp32 | 4.2 / 6.9 | 0.62 | - | - | 0.05 / 1.28 | 23 / 54 | ~19 / ~50 | 0.39 | 4 / 4 | 368 / 432 | 6.1 / 9.1 | 0.04 / 0.04 | 0.50 / 1.35 | 6.3e-06 | 1e-06 | - |  |
| rtmdet-s-coco | fp16 | 14.3 / 19.0 | 0.75 | - | - | 0.05 / 1.25 | 96 / 216 | ~76 / ~197 | 0.39 | 20 / 19 | 520 / 730 | 20.0 / 24.1 | 0.04 / 0.08 | 0.53 / 1.31 | 0.75 (67 % off) | 0.022 (17 % off) | - |  |
| rtmdet-s-coco | fp32 | 16.0 / 22.8 | 0.70 | - | - | 0.06 / 1.22 | 181 / 243 | ~142 / ~205 | 0.70 | 38 / 38 | 632 / 839 | 21.8 / 27.9 | 0.04 / 0.08 | 0.52 / 1.29 | 0.76 (67 % off) | 2.6e-06 | - |  |
| rtmdet-tiny-coco | fp16 | 12.9 / 16.1 | 0.80 | - | - | 0.05 / 1.21 | 66 / 180 | ~54 / ~168 | 0.33 | 12 / 11 | 444 / 651 | 18.7 / 21.3 | 0.04 / 0.07 | 0.51 / 1.28 | 0.72 (67 % off) | 0.55 (1.8 % off) | - |  |
| rtmdet-tiny-coco | fp32 | 15.1 / 18.9 | 0.80 | - | - | 0.05 / 1.18 | 105 / 262 | ~83 / ~241 | 0.35 | 22 / 22 | 487 / 755 | 21.2 / 24.5 | 0.04 / 0.07 | 0.51 / 1.25 | 0.73 (66 % off) | 1.6e-06 | - |  |
| rtmdet-x-coco | fp16 | 102.2 / 111.6 | 0.92 | - | - | 0.13 / 1.79 | 421 / 470 | ~221 / ~274 | 0.81 | 200 / 196 | 1091 / 1479 | 107.8 / 116.9 | 0.06 / 0.20 | 0.60 / 1.86 | 0.79 (65 % off) | empty (3 keys differ) | - |  |
| rtmdet-x-coco | fp32 | 116.5 / 131.7 | 0.88 | - | - | 0.17 / 1.74 | 808 / 913 | ~418 / ~523 | 0.80 | 390 / 390 | 1772 / 2340 | 122.6 / 137.4 | 0.06 / 0.21 | 0.63 / 1.81 | 0.79 (65 % off) | 1.9e-06 | - |  |
| rtmo-l | fp16 | 47.0 / 55.7 | 0.84 | - | - | 0.07 / 1.56 | 401 / 414 | ~319 / ~332 | 0.96 | 82 / 82 | 1371 / 1379 | 52.9 / 61.4 | 0.27 / 0.14 | 0.07 / 1.57 | 0.011 (0.00028 % off) | empty | - |  |
| rtmo-l | fp32 | 54.2 / 69.7 | 0.78 | 0.77-0.79 | 58.2 | 0.14 / 1.77 | 537 / 527 | 370 / 364 | 1.01 | 163 / 163 | 1522 / 1606 | 60.6 / 76.0 | 0.33 / 0.15 | 0.15 / 1.79 | 1e-05 | empty | - |  |
| rtmo-m | fp16 | 27.7 / 34.7 | 0.80 | - | - | 0.06 / 1.43 | 165 / 206 | ~124 / ~165 | 0.75 | 41 / 41 | 950 / 1061 | 33.6 / 40.0 | 0.20 / 0.09 | 0.06 / 1.44 | 0.0034 | empty | - |  |
| rtmo-m | fp32 | 30.1 / 39.1 | 0.77 | - | - | 0.07 / 1.39 | 278 / 313 | ~197 / ~232 | 0.85 | 82 / 82 | 1111 / 1331 | 35.8 / 44.2 | 0.13 / 0.09 | 0.08 / 1.40 | 1.2e-05 | empty | - |  |
| rtmo-s | fp16 | 13.3 / 17.3 | 0.77 | - | - | 0.05 / 1.30 | 82 / 199 | ~64 / ~182 | 0.36 | 18 / 18 | 599 / 780 | 18.8 / 22.5 | 0.14 / 0.06 | 0.05 / 1.31 | 0.017 (0.0018 % off) | empty | - |  |
| rtmo-s | fp32 | 14.7 / 20.2 | 0.73 | - | - | 0.06 / 1.28 | 128 / 226 | ~93 / ~191 | 0.49 | 35 / 35 | 666 / 841 | 20.3 / 25.3 | 0.09 / 0.06 | 0.07 / 1.29 | 1.4e-05 | empty | - |  |
| rtmo-t | fp16 | 7.2 / 9.6 | 0.75 | - | - | 0.04 / 1.29 | 35 / 88 | ~22 / ~75 | 0.31 | 12 / 12 | 483 / 605 | 11.6 / 14.0 | 0.13 / 0.06 | 0.05 / 1.30 | 0.015 (0.0011 % off) | empty | - |  |
| rtmo-t | fp32 | 7.7 / 10.6 | 0.73 | - | - | 0.05 / 1.26 | 65 / 155 | ~41 / ~131 | 0.31 | 24 / 24 | 522 / 679 | 12.3 / 14.8 | 0.08 / 0.06 | 0.06 / 1.26 | 1.9e-05 | empty | - |  |
| rtmpose-l | fp16 | 48.9 / 62.6 | 0.78 | 0.77-0.81 | 49.2 | 0.07 / 1.90 | 310 / 267 | 243 / 212 | 1.15 | 58 / 56 | 1118 / 1257 | 61.3 / 72.7 | 0.28 / 0.12 | 0.09 / 1.92 | - | 0.17 (0.36 % off) | - | heavier calls-differ |
| rtmpose-l | fp32 | 56.7 / 78.2 | 0.72 | 0.72-0.74 | 59.0 | 0.13 / 1.67 | 565 / 426 | 451 / 320 | 1.41 | 106 / 106 | 1452 / 1684 | 67.0 / 91.6 | 0.15 / 0.10 | 0.14 / 1.70 | - | 7.2e-06 | - | heavier calls-differ |
| rtmpose-l-384 | fp16 | 98.0 / 122.8 | 0.80 | 0.78-0.80 | 99.7 | 0.12 / 1.60 | 387 / 354 | 309 / 298 | 1.04 | 59 / 56 | 1634 / 1668 | 116.2 / 143.7 | 0.29 / 0.09 | 0.14 / 1.64 | - | 0.45 (12 % off) | - | heavier calls-differ |
| rtmpose-l-384 | fp32 | 111.4 / 156.1 | 0.71 | 0.70-0.73 | 129.3 | 0.15 / 1.72 | 743 / 508 | 618 / 402 | 1.53 | 106 / 106 | 2484 / 2402 | 129.9 / 174.1 | 0.15 / 0.09 | 0.17 / 1.75 | - | 6.3e-06 | - | heavier calls-differ |
| rtmpose-l-halpe26 | fp16 | 51.0 / 62.6 | 0.81 | 0.80-0.84 | 50.4 | 0.08 / 1.69 | 311 / 269 | 243 / 212 | 1.15 | 60 / 58 | 1126 / 1254 | 60.3 / 71.7 | 0.29 / 0.09 | 0.09 / 1.71 | - | 0.047 (0.08 % off) | - | heavier calls-differ |
| rtmpose-l-halpe26 | fp32 | 56.0 / 76.3 | 0.73 | 0.64-0.75 | 59.6 | 0.09 / 1.48 | 566 / 427 | 451 / 320 | 1.41 | 107 / 107 | 1468 / 1689 | 65.5 / 86.3 | 0.15 / 0.09 | 0.10 / 1.50 | - | 6.4e-06 | - | heavier calls-differ |
| rtmpose-l-halpe26-384 | fp16 | 96.9 / 120.3 | 0.81 | 0.80-0.82 | 98.2 | 0.12 / 1.52 | 389 / 356 | 309 / 298 | 1.04 | 61 / 58 | 1640 / 1669 | 116.2 / 139.4 | 0.26 / 0.09 | 0.14 / 1.56 | - | 0.28 (4.8 % off) | - | heavier calls-differ |
| rtmpose-l-halpe26-384 | fp32 | 109.6 / 152.5 | 0.72 | 0.71-0.72 | 115.9 | 0.14 / 1.50 | 745 / 498 | 618 / 390 | 1.58 | 108 / 108 | 2491 / 2392 | 127.8 / 171.2 | 0.15 / 0.09 | 0.17 / 1.53 | - | 9.5e-06 | - | heavier calls-differ |
| rtmpose-m | fp16 | 34.2 / 39.9 | 0.86 | - | - | 0.05 / 1.22 | 211 / 220 | ~181 / ~192 | 0.94 | 30 / 29 | 816 / 1050 | 42.5 / 47.3 | 0.13 / 0.06 | 0.06 / 1.23 | - | 0.18 (2.5 % off) | - | calls-differ |
| rtmpose-m | fp32 | 38.2 / 47.5 | 0.80 | - | - | 0.07 / 1.19 | 383 / 412 | ~331 / ~360 | 0.92 | 52 / 52 | 988 / 1432 | 46.5 / 55.3 | 0.09 / 0.06 | 0.08 / 1.20 | - | 9e-06 | - | calls-differ |
| rtmpose-m-384 | fp16 | 67.9 / 78.9 | 0.86 | - | - | 0.09 / 1.27 | 251 / 371 | ~220 / ~342 | 0.65 | 31 / 29 | 1013 / 1336 | 84.9 / 94.3 | 0.13 / 0.06 | 0.11 / 1.29 | - | 0.042 (1.5 % off) | - | calls-differ |
| rtmpose-m-384 | fp32 | 79.6 / 98.4 | 0.81 | - | - | 0.11 / 1.25 | 440 / 622 | ~387 / ~570 | 0.68 | 53 / 53 | 1435 / 2077 | 96.7 / 114.1 | 0.09 / 0.06 | 0.13 / 1.27 | - | 5.2e-06 | - | calls-differ |
| rtmpose-m-ap10k | fp16 | 42.6 / 49.9 | 0.85 | - | - | 0.06 / 1.22 | 228 / 232 | ~198 / ~203 | 0.97 | 30 / 29 | 892 / 1093 | 52.9 / 59.4 | 0.13 / 0.06 | 0.07 / 1.23 | - | 0.019 (0.12 % off) | - | calls-differ |
| rtmpose-m-ap10k | fp32 | 48.8 / 61.2 | 0.80 | - | - | 0.07 / 1.20 | 402 / 435 | ~349 / ~383 | 0.91 | 52 / 52 | 1130 / 1561 | 59.2 / 70.9 | 0.09 / 0.06 | 0.09 / 1.21 | - | 2.3e-06 | - | calls-differ |
| rtmpose-m-face6 | fp16 | 48.5 / 52.2 | 0.93 | - | - | 0.06 / 1.23 | 241 / 244 | ~198 / ~203 | 0.97 | 43 / 42 | 948 / 1149 | 58.9 / 61.8 | 0.14 / 0.06 | 0.08 / 1.24 | - | 0.2 (0.23 % off) | - | calls-differ |
| rtmpose-m-face6 | fp32 | 54.2 / 63.2 | 0.86 | - | - | 0.08 / 1.21 | 414 / 448 | ~349 / ~383 | 0.91 | 65 / 65 | 1186 / 1617 | 65.3 / 72.8 | 0.10 / 0.06 | 0.10 / 1.23 | - | 2.1e-06 | - | calls-differ |
| rtmpose-m-halpe26 | fp16 | 34.2 / 39.9 | 0.86 | - | - | 0.05 / 1.20 | 212 / 222 | ~181 / ~192 | 0.94 | 32 / 30 | 820 / 1055 | 42.8 / 47.5 | 0.15 / 0.06 | 0.06 / 1.21 | - | 0.45 (1.8 % off) | - | calls-differ |
| rtmpose-m-halpe26 | fp32 | 38.4 / 47.7 | 0.81 | - | - | 0.07 / 1.18 | 384 / 414 | ~331 / ~360 | 0.92 | 53 / 53 | 994 / 1438 | 46.7 / 55.3 | 0.09 / 0.06 | 0.08 / 1.19 | - | 9.1e-06 | - | calls-differ |
| rtmpose-m-halpe26-384 | fp16 | 68.1 / 78.6 | 0.87 | - | - | 0.09 / 1.26 | 252 / 372 | ~220 / ~342 | 0.65 | 32 / 30 | 1018 / 1341 | 85.5 / 94.1 | 0.13 / 0.06 | 0.10 / 1.28 | - | 0.43 (2.2 % off) | - | calls-differ |
| rtmpose-m-halpe26-384 | fp32 | 80.3 / 98.8 | 0.81 | - | - | 0.11 / 1.25 | 441 / 624 | ~387 / ~570 | 0.68 | 54 / 54 | 1440 / 2084 | 96.6 / 114.4 | 0.09 / 0.06 | 0.13 / 1.27 | - | 6e-06 | - | calls-differ |
| rtmpose-m-hand5 | fp16 | 42.8 / 50.3 | 0.85 | - | - | 0.06 / 1.23 | 228 / 232 | ~198 / ~203 | 0.97 | 31 / 29 | 896 / 1097 | 53.1 / 59.6 | 0.14 / 0.06 | 0.07 / 1.24 | - | 0.0025 | - | calls-differ |
| rtmpose-m-hand5 | fp32 | 49.1 / 61.4 | 0.80 | - | - | 0.07 / 1.24 | 402 / 436 | ~349 / ~383 | 0.91 | 53 / 53 | 1131 / 1558 | 59.4 / 71.2 | 0.09 / 0.06 | 0.09 / 1.25 | - | 3.1e-06 | - | calls-differ |
| rtmpose-s | fp16 | 13.7 / 19.0 | 0.72 | - | - | 0.03 / 1.11 | 162 / 190 | ~148 / ~177 | 0.84 | 13 / 13 | 575 / 740 | 21.9 / 26.6 | 0.09 / 0.04 | 0.04 / 1.12 | - | 0.16 (1.1 % off) | - | calls-differ |
| rtmpose-s | fp32 | 14.5 / 22.0 | 0.66 | - | - | 0.04 / 1.09 | 248 / 245 | ~227 / ~224 | 1.01 | 21 / 21 | 649 / 835 | 22.9 / 30.3 | 0.06 / 0.04 | 0.06 / 1.10 | - | 7.3e-06 | - | calls-differ |
| rtmpose-s-face6 | fp16 | 20.8 / 24.3 | 0.86 | 0.84-0.90 | 21.1 | 0.04 / 1.21 | 204 / 179 | 170 / 158 | 1.08 | 22 / 21 | 702 / 842 | 31.8 / 35.1 | 0.12 / 0.05 | 0.06 / 1.23 | - | 0.47 (0.21 % off) | - | heavier calls-differ |
| rtmpose-s-face6 | fp32 | 22.5 / 28.7 | 0.78 | 0.74-0.82 | 22.5 | 0.06 / 1.21 | 329 / 248 | 288 / 218 | 1.32 | 30 / 30 | 804 / 1008 | 36.1 / 41.1 | 0.07 / 0.05 | 0.07 / 1.23 | - | 2.7e-06 | - | heavier calls-differ |
| rtmpose-s-halpe26 | fp16 | 13.8 / 19.2 | 0.72 | - | - | 0.03 / 1.11 | 165 / 191 | ~151 / ~177 | 0.85 | 14 / 14 | 579 / 751 | 22.1 / 27.0 | 0.09 / 0.04 | 0.04 / 1.13 | - | 0.4 (6.3 % off) | - | calls-differ |
| rtmpose-s-halpe26 | fp32 | 14.7 / 21.2 | 0.69 | 0.68-0.71 | 14.7 | 0.05 / 1.29 | 251 / 246 | 221 / 224 | 0.99 | 22 / 22 | 661 / 841 | 24.5 / 30.1 | 0.09 / 0.05 | 0.07 / 1.31 | - | 0.001 | - | calls-differ |
| rtmpose-t | fp16 | 12.8 / 16.0 | 0.80 | - | - | 0.03 / 1.06 | 125 / 189 | ~117 / ~181 | 0.65 | 9 / 8 | 516 / 713 | 21.2 / 23.6 | 0.08 / 0.04 | 0.04 / 1.07 | - | 0.29 (2.7 % off) | - | calls-differ |
| rtmpose-t | fp32 | 13.8 / 18.8 | 0.73 | - | - | 0.04 / 1.07 | 188 / 275 | ~175 / ~262 | 0.67 | 13 / 13 | 571 / 840 | 22.0 / 26.4 | 0.05 / 0.04 | 0.05 / 1.08 | - | 8.9e-06 | - | calls-differ |
| rtmpose-t-face6 | fp16 | 18.8 / 21.0 | 0.90 | - | - | 0.04 / 1.06 | 188 / 195 | ~172 / ~180 | 0.96 | 15 / 15 | 595 / 798 | 29.2 / 30.3 | 0.09 / 0.04 | 0.05 / 1.07 | - | 0.24 (0.35 % off) | - | calls-differ |
| rtmpose-t-face6 | fp32 | 20.1 / 24.5 | 0.82 | - | - | 0.04 / 1.06 | 270 / 288 | ~250 / ~268 | 0.93 | 19 / 19 | 672 / 941 | 30.4 / 33.7 | 0.06 / 0.04 | 0.06 / 1.08 | - | 3.8e-06 | - | calls-differ |
| rtmpose-t-halpe26 | fp16 | 12.9 / 16.0 | 0.81 | - | - | 0.03 / 1.06 | 128 / 190 | ~119 / ~181 | 0.66 | 9 / 9 | 518 / 722 | 21.4 / 23.8 | 0.08 / 0.04 | 0.04 / 1.07 | - | 0.45 (2.9 % off) | - | calls-differ |
| rtmpose-t-halpe26 | fp32 | 13.8 / 18.8 | 0.73 | - | - | 0.04 / 1.07 | 191 / 276 | ~178 / ~262 | 0.68 | 14 / 14 | 576 / 847 | 22.2 / 26.7 | 0.05 / 0.04 | 0.05 / 1.08 | - | 6.9e-06 | - | calls-differ |
| rtmpose-x-halpe26-384 | fp16 | 198.3 / 214.5 | 0.92 | - | - | 0.21 / 1.59 | 553 / 658 | ~449 / ~557 | 0.81 | 105 / 100 | 2159 / 2579 | 216.4 / 231.0 | 0.26 / 0.12 | 0.23 / 1.61 | - | 0.19 (0.48 % off) | - | calls-differ |
| rtmpose-x-halpe26-384 | fp32 | 263.6 / 279.2 | 0.94 | 0.92-0.96 | 285.9 | 0.39 / 1.93 | 1006 / 932 | 796 / 741 | 1.07 | 191 / 191 | 3363 / 3482 | 284.3 / 300.6 | 0.33 / 0.12 | 0.41 / 1.98 | - | 0.00078 | - | heavier calls-differ |
| rtmw-l | fp16 | 100.7 / 106.6 | 0.94 | 0.94-0.95 | 101.3 | 0.12 / 1.74 | 393 / 350 | 243 / 212 | 1.15 | 142 / 139 | 1354 / 1550 | 109.7 / 116.3 | 0.50 / 0.11 | 0.13 / 1.78 | - | 0.21 (0.13 % off) (2 keys differ) | - | heavier calls-differ |
| rtmw-l | fp32 | 111.3 / 123.5 | 0.90 | 0.89-0.91 | 114.2 | 0.14 / 1.62 | 678 / 544 | 451 / 326 | 1.38 | 218 / 218 | 1737 / 2020 | 120.4 / 134.1 | 0.22 / 0.11 | 0.15 / 1.66 | - | 3.9e-06 (2 keys differ) | - | heavier calls-differ |
| rtmw-l-384 | fp16 | 206.5 / 228.4 | 0.90 | 0.89-0.91 | 207.9 | 0.23 / 1.86 | 470 / 438 | 309 / 298 | 1.04 | 142 / 140 | 1878 / 2018 | 223.9 / 248.6 | 0.50 / 0.11 | 0.25 / 1.91 | - | 0.43 (0.51 % off) (2 keys differ) | - | heavier calls-differ |
| rtmw-l-384 | fp32 | 224.7 / 268.7 | 0.84 | 0.83-0.84 | 230.4 | 0.25 / 1.74 | 856 / 634 | 618 / 415 | 1.49 | 219 / 219 | 2759 / 2792 | 242.8 / 290.5 | 0.21 / 0.11 | 0.28 / 1.79 | - | 0.00078 (2 keys differ) | - | heavier calls-differ |
| rtmw-m | fp16 | 71.2 / 72.4 | 0.98 | - | - | 0.09 / 1.49 | 278 / 275 | ~193 / ~192 | 1.01 | 85 / 84 | 1009 / 1233 | 80.4 / 80.4 | 0.20 / 0.08 | 0.10 / 1.50 | - | 0.45 (0.13 % off) (2 keys differ) | - | calls-differ |
| rtmw-m | fp32 | 76.8 / 82.4 | 0.93 | - | - | 0.10 / 1.54 | 468 / 484 | ~344 / ~360 | 0.96 | 123 / 123 | 1265 / 1640 | 86.3 / 90.6 | 0.14 / 0.09 | 0.12 / 1.56 | - | 2.3e-06 (2 keys differ) | - | calls-differ |
| rtmw-x | fp16 | 176.7 / 205.5 | 0.86 | 0.86-0.86 | 177.2 | 0.19 / 1.80 | 560 / 474 | 332 / 259 | 1.28 | 219 / 215 | 1723 / 2204 | 186.4 / 214.6 | 0.61 / 0.18 | 0.20 / 1.84 | - | 0.28 (0.21 % off) (2 keys differ) | - | heavier calls-differ |
| rtmw-x | fp32 | 203.3 / 234.5 | 0.87 | 0.86-0.87 | 209.3 | 0.23 / 1.82 | 968 / 830 | 608 / 479 | 1.27 | 351 / 351 | 2326 / 2982 | 212.4 / 245.8 | 0.34 / 0.18 | 0.24 / 1.87 | - | 5.9e-06 (2 keys differ) | - | heavier calls-differ |
| rtmw-x-384 | fp16 | 350.0 / 414.6 | 0.84 | - | - | 0.37 / 2.04 | 668 / 773 | ~449 / ~557 | 0.81 | 220 / 216 | 2522 / 3075 | 368.7 / 425.3 | 0.41 / 0.18 | 0.39 / 2.06 | - | 0.2 (5.8 % off) (2 keys differ) | - | calls-differ |
| rtmw-x-384 | fp32 | 422.1 / 484.6 | 0.87 | 0.86-0.88 | 435.1 | 0.46 / 2.11 | 1174 / 1093 | 804 / 741 | 1.08 | 352 / 352 | 3744 / 3938 | 442.4 / 507.4 | 0.54 / 0.18 | 0.48 / 2.17 | - | 1.3e-05 (2 keys differ) | - | heavier calls-differ |
| sam-hq-vit-b | fp16 | 309.0 / 397.2 | 0.78 | 0.77-0.79 | 309.9 | 0.33 / 2.63 | 622 / 539 | 432 / 358 | 1.21 | 179 / 181 | 1614 / 1770 | 365.5 / 449.3 | 0.54 / 0.06 | 0.50 / 2.78 | 0.0047 | 1 (0.39 % off) | - | heavier |
| sam-hq-vit-b | fp32 | 370.9 / 484.5 | 0.77 | - | - | 0.40 / 2.29 | 879 / 1066 | ~524 / ~704 | 0.75 | 354 / 362 | 2088 / 2561 | 418.1 / 536.4 | 0.14 / 0.05 | 0.47 / 2.35 | 6.7e-06 | 1 (0.00081 % off) | - |  |
| sam-hq-vit-h | fp16 | 1559.6 / 1733.7 | 0.90 | 0.85-0.93 | 1550.7 | 1.71 / 6.04 | 1762 / 1627 | 523 / 398 | 1.31 | 1227 / 1228 | 2846 / 3092 | 1638.3 / 1811.6 | 5.31 / 0.15 | 1.98 / 6.77 | 0.0054 | 1 (0.0083 % off) | - | heavier |
| sam-hq-vit-h | fp32 | 1881.1 / 2133.4 | 0.88 | - | - | 1.96 / 5.79 | 3059 / 3242 | ~611 / ~785 | 0.78 | 2449 / 2458 | 4438 / 5463 | 1947.3 / 2149.4 | 0.96 / 0.09 | 2.30 / 6.05 | 4.2e-06 | 2.8e-06 | - |  |
| sam-hq-vit-l | fp16 | 825.2 / 943.6 | 0.87 | 0.87-0.88 | 825.3 | 0.91 / 3.90 | 1119 / 985 | 510 / 386 | 1.32 | 597 / 599 | 2140 / 2219 | 875.1 / 993.8 | 1.40 / 0.09 | 1.10 / 4.07 | 0.0029 | 1 (0.0052 % off) | - | heavier |
| sam-hq-vit-l | fp32 | 976.4 / 1120.5 | 0.87 | - | - | 1.01 / 3.87 | 1778 / 1953 | ~588 / ~754 | 0.78 | 1190 / 1199 | 2952 / 3612 | 1027.3 / 1167.4 | 0.32 / 0.07 | 1.09 / 3.95 | 3.5e-06 | 4.2e-06 | - |  |
| sam-vit-b | fp16 | 299.4 / 385.9 | 0.78 | 0.77-0.78 | 299.0 | 0.32 / 2.45 | 613 / 531 | 426 / 352 | 1.21 | 175 / 179 | 1224 / 1334 | 324.3 / 410.1 | 0.51 / 0.06 | 0.39 / 2.51 | 0.0047 | 1 (0.015 % off) | - | heavier |
| sam-vit-b | fp32 | 365.7 / 480.0 | 0.76 | - | - | 0.40 / 2.21 | 865 / 1050 | ~514 / ~692 | 0.74 | 351 / 359 | 1707 / 2118 | 389.8 / 505.0 | 0.14 / 0.05 | 0.43 / 2.23 | 6.7e-06 | 1 (0.00057 % off) | - |  |
| sam-vit-h | fp16 | 1533.7 / 1698.2 | 0.90 | 0.90-0.91 | 1540.5 | 1.89 / 5.90 | 1745 / 1614 | 512 / 388 | 1.32 | 1222 / 1226 | 2546 / 2731 | 1558.6 / 1723.5 | 5.07 / 0.14 | 2.19 / 6.55 | 0.0054 | 1 (0.013 % off) | - | heavier |
| sam-vit-h | fp32 | 1873.2 / 2106.1 | 0.89 | - | - | 2.26 / 5.74 | 3034 / 3210 | ~591 / ~758 | 0.78 | 2443 / 2452 | 4095 / 5081 | 1905.9 / 2109.5 | 1.39 / 0.14 | 2.31 / 5.79 | 4.2e-06 | 1 (3e-05 % off) | - |  |
| sam-vit-l | fp16 | 815.6 / 936.1 | 0.87 | 0.87-0.87 | 816.4 | 0.91 / 4.04 | 1107 / 975 | 502 / 378 | 1.33 | 593 / 597 | 1822 / 1911 | 840.1 / 962.5 | 1.43 / 0.08 | 1.00 / 4.12 | 0.0029 | 1 (0.00033 % off) | - | heavier |
| sam-vit-l | fp32 | 972.1 / 1110.9 | 0.88 | - | - | 1.00 / 3.88 | 1758 / 1933 | ~572 / ~738 | 0.78 | 1186 / 1195 | 2632 / 3309 | 995.8 / 1131.8 | 0.32 / 0.07 | 1.04 / 3.91 | 3.5e-06 | 3.1e-06 | - |  |
| sam2.1-hiera-base-plus | fp16 | 213.5 / 244.4 | 0.87 | - | - | 0.26 / 2.27 | 506 / 475 | ~289 / ~320 | 0.90 | 217 / 155 | 1276 / 1467 | 216.9 / 262.8 | 0.09 / 0.08 | 0.26 / 2.30 | 0.008 | 1 (0.0028 % off) | - | calls-differ |
| sam2.1-hiera-base-plus | fp32 | 255.9 / 276.5 | 0.93 | 0.81-0.99 | 278.0 | 0.32 / 3.31 | 759 / 636 | 392 / 336 | 1.17 | 367 / 300 | 1900 / 2208 | 260.0 / 294.5 | 0.52 / 0.20 | 0.34 / 3.50 | 1.5e-05 | 1.6e-05 | - | heavier calls-differ |
| sam2.1-hiera-large | fp16 | 497.7 / 579.6 | 0.86 | 0.81-0.87 | 521.8 | 0.57 / 5.04 | 838 / 734 | 347 / 300 | 1.16 | 490 / 434 | 1536 / 1798 | 502.0 / 597.2 | 0.71 / 0.23 | 0.58 / 5.29 | 0.08 (0.0035 % off) | 1 (0.0023 % off) | - | heavier calls-differ |
| sam2.1-hiera-large | fp32 | 568.2 / 609.6 | 0.93 | - | - | 0.61 / 3.61 | 1373 / 1317 | ~462 / ~462 | 1.00 | 911 / 855 | 2361 / 2790 | 569.0 / 627.6 | 0.13 / 0.11 | 0.62 / 3.64 | 2.1e-05 | 9e-06 | - | calls-differ |
| sam2.1-hiera-small | fp16 | 123.1 / 138.2 | 0.89 | - | - | 0.16 / 1.80 | 378 / 362 | ~236 / ~276 | 0.85 | 142 / 86 | 1072 / 1225 | 124.5 / 157.4 | 0.08 / 0.07 | 0.17 / 1.82 | 0.012 (0.00014 % off) | 1 (0.012 % off) | - | calls-differ |
| sam2.1-hiera-small | fp32 | 156.1 / 168.7 | 0.93 | 0.78-1.02 | 168.0 | 0.26 / 3.28 | 547 / 451 | 328 / 288 | 1.14 | 219 / 163 | 1467 / 1608 | 166.2 / 216.3 | 0.31 / 0.15 | 0.27 / 3.52 | 5.1e-06 | 1 (3e-05 % off) | - | heavier calls-differ |
| sam2.1-hiera-tiny | fp16 | 103.1 / 115.6 | 0.89 | - | - | 0.15 / 1.61 | 378 / 349 | ~249 / ~276 | 0.90 | 129 / 73 | 1038 / 1210 | 105.5 / 134.8 | 0.08 / 0.07 | 0.15 / 1.64 | 0.0077 | 1 (0.024 % off) | - | calls-differ |
| sam2.1-hiera-tiny | fp32 | 148.4 / 146.6 | 1.01 | 0.93-1.10 | 141.5 | 1.19 / 2.98 | 520 / 424 | 328 / 288 | 1.14 | 192 / 136 | 1410 / 1600 | 140.5 / 187.4 | 0.42 / 0.16 | 1.29 / 3.23 | 5.6e-06 | 1 (6e-05 % off) | - | heavier calls-differ |
| sam3 | fp16 | 1705.6 / 1943.8 | 0.88 | 0.79-1.02 | 1607.2 | 2.02 / 8.17 | 2006 / 1385 | 325 / 493 | 0.66 | 1633 / 892 | 3509 / 4194 | 1715.9 / 3592.3 | 2.18 / 1.13 | 2.09 / 13.88 | 0.21 (0.84 % off) | empty | - | calls-differ |
| sam3 | fp32 | 2046.9 / 2438.2 | 0.84 | 0.79-1.30 | 2101.4 | 2.74 / 7.40 | 3931 / 2551 | 649 / 775 | 0.84 | 3205 / 1776 | 5644 / 5876 | 2205.4 / 2834.5 | 4.27 / 1.13 | 2.80 / 11.57 | 0.19 (0.7 % off) | empty | - | calls-differ |
| sapiens-seg-0.3b-goliath | fp16 | 1140.1 / 1509.1 | 0.76 | - | - | 1.18 / 3.15 | 1822 / 1808 | ~1175 / ~1161 | 1.01 | 648 / 648 | 3226 / 3616 | 1144.0 / 1829.0 | 0.32 / 0.10 | 1.20 / 3.47 | 0.003 | 0 | - |  |
| sapiens-seg-0.3b-goliath | fp32 | 1434.3 / 1799.8 | 0.80 | 0.78-0.82 | 1431.7 | 1.96 / 4.30 | 3631 / 3288 | 2327 / 1993 | 1.17 | 1295 / 1295 | 5570 / 5102 | 1424.5 / 2119.1 | 4.53 / 0.33 | 1.99 / 4.73 | 2.6e-06 | 0 | - | heavier |
| sapiens-seg-1b-goliath | fp16 | 2650.0 / 3004.7 | 0.88 | - | - | 2.81 / 7.35 | 3426 / 3730 | ~1178 / ~1481 | 0.80 | 2248 / 2248 | 4746 / 5455 | 2655.2 / 3316.0 | 23.74 / 0.34 | 2.84 / 7.72 | 0.002 | 0 | - |  |
| sapiens-seg-1b-goliath | fp32 | 3309.6 / 4238.3 | 0.78 | 0.62-0.97 | 3760.5 | 3.90 / 11.55 | 6839 / 6491 | 2333 / 1993 | 1.17 | 4497 / 4497 | 8710 / 8467 | 3883.3 / 4696.4 | 24.84 / 0.42 | 3.94 / 13.46 | 2.1e-06 | 0 | - | heavier |
| segformer-b0-ade20k | fp16 | 20.4 / 16.8 | 1.22 | 1.16-1.31 | 24.0 | 0.13 / 1.24 | 158 / 114 | 145 / 107 | 1.36 | 7 / 7 | 826 / 778 | 52.1 / 54.9 | 2.82 / 2.22 | 0.18 / 1.31 | 0.001 | 0.5 (0.067 % off) | 0.00608 / 0.00589 (0.02) | slower heavier shipped-slower |
| segformer-b0-ade20k | fp32 | 24.2 / 22.2 | 1.09 | 1.08-1.14 | 35.7 | 0.18 / 1.19 | 221 / 149 | 202 / 136 | 1.49 | 14 / 14 | 981 / 892 | 57.1 / 59.7 | 2.20 / 2.23 | 0.23 / 1.24 | 8.9e-07 | 0 | 5.81e-06 / 5.86e-06 (0.02) | slower heavier shipped-slower |
| segformer-b2-clothes | fp16 | 29.3 / 25.6 | 1.14 | 1.09-1.18 | 30.8 | 0.19 / 1.69 | 181 / 158 | 130 / 110 | 1.18 | 48 / 48 | 758 / 790 | 28.9 / 27.5 | 2.52 / 2.29 | 0.19 / 1.71 | 0.00081 | 0 | 0.00874 / 0.00866 (0.02) | slower heavier shipped-slower |
| segformer-b2-clothes | fp32 | 33.4 / 37.0 | 0.90 | 0.84-1.00 | 50.2 | 0.19 / 1.60 | 273 / 252 | 175 / 156 | 1.12 | 96 / 96 | 1045 / 1036 | 34.5 / 34.8 | 2.26 / 2.29 | 0.20 / 1.62 | 6.9e-07 | 0 | 0.0086 / 0.0086 (0.02) | heavier shipped-slower |
| segformer-b5-ade20k | fp16 | 226.2 / 220.3 | 1.03 | 0.90-1.08 | 272.7 | 0.33 / 4.25 | 409 / 480 | 242 / 322 | 0.75 | 158 / 158 | 1600 / 1613 | 258.9 / 257.2 | 3.16 / 3.71 | 0.37 / 4.38 | 0.00089 | 1 (0.015 % off) | 0.00712 / 0.0066 (0.02) | shipped-slower |
| segformer-b5-ade20k | fp32 | 282.7 / 281.8 | 1.00 | 0.76-1.14 | 340.2 | 0.51 / 5.00 | 807 / 780 | 484 / 466 | 1.04 | 314 / 314 | 2280 / 2244 | 305.7 / 334.0 | 2.89 / 2.98 | 0.56 / 5.17 | 7.1e-07 | 0 | 3.19e-05 / 3.11e-05 (0.02) | heavier |
| segformer-b5-face-parsing | fp16 | 68.2 / 64.4 | 1.06 | 1.05-1.07 | 72.1 | 0.12 / 3.60 | 291 / 274 | 130 / 116 | 1.13 | 158 / 158 | 974 / 1040 | 69.0 / 66.3 | 3.02 / 2.24 | 0.12 / 3.63 | 0.00088 | 0 | 0.0116 / 0.0116 (0.02) | slower heavier shipped-slower |
| segformer-b5-face-parsing | fp32 | 79.7 / 79.5 | 1.00 | - | - | 0.14 / 3.15 | 492 / 541 | ~178 / ~227 | 0.78 | 314 / 314 | 1344 / 1337 | 80.7 / 80.9 | 2.27 / 2.11 | 0.14 / 3.15 | 5.4e-07 | 0 | 0.0112 / 0.0112 (0.02) |  |
| segnext-mscan-b-ade20k | fp16 | 82.8 / 70.6 | 1.17 | 1.10-1.24 | 86.7 | 0.26 / 3.42 | 264 / 263 | 204 / 210 | 0.97 | 54 / 53 | 1090 / 1176 | 121.0 / 104.9 | 0.43 / 0.16 | 0.31 / 3.48 | 0.0076 | 0 | 0.0683 / 0.0697 (0.14) | slower shipped-slower |
| segnext-mscan-b-ade20k | fp32 | 88.0 / 87.7 | 1.00 | - | - | 0.13 / 2.17 | 324 / 366 | ~218 / ~260 | 0.84 | 106 / 106 | 1333 / 1541 | 121.6 / 121.1 | 0.18 / 0.12 | 0.17 / 2.21 | 4.2e-06 | 0 | 1.09e-05 / 2.01e-05 (0.14) |  |
| segnext-mscan-l-ade20k | fp16 | 139.1 / 127.7 | 1.09 | 1.07-1.12 | 148.2 | 0.28 / 3.56 | 421 / 305 | 320 / 211 | 1.52 | 95 / 94 | 1309 / 1254 | 174.3 / 161.3 | 0.62 / 0.20 | 0.33 / 3.63 | 0.004 | 0.5 (0.0021 % off) | 0.067 / 0.0643 (0.14) | slower heavier shipped-slower |
| segnext-mscan-l-ade20k | fp32 | 157.1 / 164.6 | 0.95 | 0.94-0.97 | 218.9 | 0.24 / 4.43 | 518 / 448 | 325 / 261 | 1.24 | 188 / 187 | 1546 / 1680 | 191.1 / 196.8 | 0.29 / 0.22 | 0.28 / 4.52 | 5.5e-06 | 0 | 1.42e-05 / 2.59e-05 (0.14) | heavier shipped-slower |
| segnext-mscan-s-ade20k | fp16 | 48.4 / 38.3 | 1.26 | 1.12-1.37 | 49.2 | 0.14 / 3.39 | 237 / 197 | 204 / 170 | 1.20 | 27 / 27 | 928 / 896 | 84.9 / 72.3 | 0.24 / 0.09 | 0.19 / 3.48 | 0.0022 | 0.12 (0.2 % off) | 0.0706 / 0.0755 (0.14) | slower heavier shipped-slower |
| segnext-mscan-s-ade20k | fp32 | 53.8 / 49.1 | 1.10 | 0.95-1.27 | 62.4 | 0.22 / 2.73 | 275 / 314 | 216 / 260 | 0.83 | 53 / 53 | 1064 / 1252 | 91.3 / 87.8 | 0.13 / 0.08 | 0.26 / 2.81 | 3e-06 | 0 | 7.33e-06 / 1.41e-05 (0.14) | shipped-slower |
| segnext-mscan-t-ade20k | fp16 | 32.6 / 26.1 | 1.25 | 1.10-1.27 | 36.9 | 0.08 / 2.25 | 126 / 176 | 112 / 168 | 0.67 | 8 / 9 | 682 / 894 | 66.5 / 59.4 | 0.17 / 0.08 | 0.13 / 2.32 | 0.0021 | 0.92 (0.00052 % off) | 0.0362 / 0.0347 (0.14) | slower shipped-slower |
| segnext-mscan-t-ade20k | fp32 | 36.8 / 32.7 | 1.13 | 1.09-1.13 | 43.4 | 0.10 / 1.88 | 231 / 225 | 208 / 208 | 1.00 | 17 / 17 | 920 / 1128 | 70.1 / 66.4 | 0.12 / 0.08 | 0.14 / 1.93 | 3.3e-06 | 0 | 1.07e-05 / 1.21e-05 (0.14) | slower shipped-slower |
| selective-igev-eth3d-finetune | fp16 | 3693.7 / 4643.9 | 0.80 | - | - | 3.95 / 11.08 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 4043 / 3557 | 3657.4 / 4665.0 | 0.17 / 0.10 | 3.96 / 11.09 | 0.5 (0.83 % off) | 0.5 (0.15 % off) | 0.00117 / 0.00125 (0.02) | heavier |
| selective-igev-eth3d-finetune | fp32 | 4788.7 / 6385.9 | 0.75 | - | - | 4.81 / 12.64 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5115 / 4631 | 4742.2 / 6321.1 | 0.12 / 0.10 | 4.82 / 12.65 | 0.00073 | 0.00074 | 1.95e-06 / 1.92e-06 (0.02) | heavier |
| selective-igev-eth3d-train | fp16 | 3652.2 / 4644.1 | 0.79 | - | - | 3.68 / 11.06 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 4073 / 3585 | 3662.4 / 4637.0 | 0.14 / 0.10 | 3.69 / 11.06 | 0.99 (0.49 % off) | 0.99 (0.14 % off) | 0.000911 / 0.00111 (0.02) | heavier |
| selective-igev-eth3d-train | fp32 | 4778.2 / 6294.8 | 0.76 | - | - | 4.80 / 12.88 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5079 / 4659 | 4793.4 / 6243.3 | 0.13 / 0.10 | 4.82 / 12.89 | 0.00073 | 0.00073 | 1.36e-06 / 1.38e-06 (0.02) | heavier |
| selective-igev-kitti12 | fp16 | 3643.3 / 4640.5 | 0.79 | - | - | 3.68 / 10.96 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 4067 / 3630 | 3655.1 / 4662.9 | 0.16 / 0.10 | 3.69 / 10.97 | 0.018 (0.00084 % off) | 0.0037 | 0.000678 / 0.000803 (0.02) | heavier |
| selective-igev-kitti12 | fp32 | 4733.6 / 6386.6 | 0.74 | - | - | 4.92 / 12.79 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5100 / 4687 | 4784.6 / 6359.4 | 0.14 / 0.10 | 4.95 / 12.80 | 6.3e-06 | 1.7e-06 | 9.41e-07 / 9.34e-07 (0.02) | heavier |
| selective-igev-kitti15 | fp16 | 3672.6 / 4688.7 | 0.78 | - | - | 3.70 / 11.12 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 3950 / 3636 | 3676.6 / 4683.3 | 0.15 / 0.10 | 3.71 / 11.12 | 0.015 (0.00071 % off) | 0.00088 | 0.00049 / 0.00073 (0.02) | heavier |
| selective-igev-kitti15 | fp32 | 4725.9 / 6386.9 | 0.74 | - | - | 4.87 / 13.19 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5066 / 4391 | 4759.5 / 6383.8 | 0.16 / 0.11 | 4.88 / 13.20 | 9.8e-06 | 1.4e-06 | 9.34e-07 / 9.13e-07 (0.02) | heavier |
| selective-igev-middlebury-finetune | fp16 | 3655.1 / 4663.3 | 0.78 | - | - | 3.69 / 11.01 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 3970 / 3580 | 3657.2 / 4671.3 | 0.15 / 0.10 | 3.70 / 11.02 | 0.84 (0.49 % off) | 0.84 (0.24 % off) | 0.000861 / 0.00102 (0.02) | heavier |
| selective-igev-middlebury-finetune | fp32 | 4760.9 / 6439.7 | 0.74 | - | - | 4.77 / 12.73 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5105 / 4362 | 4800.8 / 6409.6 | 0.12 / 0.10 | 4.78 / 12.74 | 0.001 | 0.001 | 1.21e-06 / 1.22e-06 (0.02) | heavier |
| selective-igev-middlebury-train | fp16 | 3651.5 / 4657.9 | 0.78 | - | - | 3.68 / 11.04 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 4001 / 3629 | 3661.5 / 4636.4 | 0.15 / 0.10 | 3.69 / 11.05 | 0.61 (0.14 % off) | 0.61 (0.1 % off) | 0.00174 / 0.00235 (0.02) | heavier |
| selective-igev-middlebury-train | fp32 | 4728.9 / 6368.5 | 0.74 | - | - | 4.78 / 13.14 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5058 / 4404 | 4759.6 / 6384.2 | 0.13 / 0.10 | 4.80 / 13.15 | 0.0012 | 0.0012 | 3.04e-06 / 2.87e-06 (0.02) | heavier |
| selective-igev-sceneflow | fp16 | 3643.8 / 4680.4 | 0.78 | - | - | 3.69 / 11.41 | 1873 / 1102 | ~1848 / ~1075 | 1.72 | 25 / 27 | 4069 / 3602 | 3665.5 / 4652.8 | 0.15 / 0.10 | 3.70 / 11.42 | 0.14 (0.099 % off) | 0.098 (0.064 % off) | 0.000883 / 0.00111 (0.02) | heavier |
| selective-igev-sceneflow | fp32 | 4694.9 / 6337.5 | 0.74 | - | - | 4.73 / 13.04 | 3498 / 2057 | ~3448 / ~2005 | 1.72 | 50 / 51 | 5093 / 4684 | 4727.2 / 6335.2 | 0.12 / 0.11 | 4.74 / 13.05 | 0.0026 | 0.0018 | 1.7e-06 / 4.11e-06 (0.02) | heavier |
| shufflenet_v2_x1_0.tv_in1k | fp16 | 2.7 / 2.8 | 0.95 | 0.67-1.04 | 2.9 | 0.19 / 0.96 | 15 / 17 | 10 / 12 | 0.87 | 4 / 4 | 325 / 364 | 3.4 / 3.8 | 0.80 / 0.61 | 0.20 / 0.97 | 0.0065 | 0.0065 | 0.00517 / 0.00363 (0.02) | worse |
| shufflenet_v2_x1_0.tv_in1k | fp32 | 3.1 / 3.0 | 1.03 | 0.97-1.12 | 3.1 | 0.03 / 0.95 | 27 / 30 | 18 / 21 | 0.85 | 9 / 9 | 330 / 374 | 3.6 / 3.5 | 0.66 / 0.89 | 0.03 / 0.95 | 1.2e-06 | 1.2e-06 | 8.09e-07 / 1.06e-06 (0.02) |  |
| shvit_s1.in1k | fp16 | 3.2 / 3.1 | 1.02 | 1.01-1.05 | 3.2 | 0.04 / 1.32 | 16 / 20 | 4 / 8 | 0.55 | 12 / 12 | 355 / 394 | 3.9 / 3.8 | 0.97 / 0.78 | 0.04 / 1.32 | 0.0016 | 0.0016 | - |  |
| shvit_s1.in1k | fp32 | 4.0 / 3.4 | 1.18 | 0.99-1.29 | 3.5 | 0.05 / 1.32 | 29 / 35 | 5 / 11 | 0.52 | 24 / 24 | 362 / 412 | 6.4 / 4.0 | 0.82 / 0.80 | 0.05 / 1.32 | 1.3e-06 | 1.3e-06 | - |  |
| shvit_s2.in1k | fp16 | 3.4 / 5.5 | 0.62 | 0.51-0.72 | 4.7 | 0.06 / 9.55 | 26 / 31 | 3 / 9 | 0.44 | 22 / 22 | 370 / 421 | 4.4 / 7.0 | 1.03 / 1.04 | 0.07 / 9.64 | 0.0023 | 0.0023 | - |  |
| shvit_s2.in1k | fp32 | 3.9 / 3.7 | 1.07 | 1.02-1.10 | 4.1 | 0.14 / 1.53 | 48 / 56 | 5 / 12 | 0.41 | 44 / 44 | 389 / 444 | 4.5 / 4.9 | 0.89 / 0.85 | 0.14 / 1.54 | 7.7e-07 | 7.7e-07 | - | slower shipped-slower |
| shvit_s3.in1k | fp16 | 4.4 / 4.0 | 1.09 | 1.01-1.26 | 5.1 | 0.06 / 1.46 | 33 / 41 | 6 / 13 | 0.48 | 28 / 28 | 383 / 446 | 6.4 / 4.7 | 1.01 / 0.80 | 0.06 / 1.47 | 0.0014 | 0.0014 | - |  |
| shvit_s3.in1k | fp32 | 4.6 / 4.6 | 0.99 | 0.85-1.07 | 4.6 | 0.06 / 1.35 | 62 / 74 | 8 / 19 | 0.44 | 54 / 54 | 401 / 468 | 5.2 / 5.0 | 0.81 / 0.76 | 0.06 / 1.36 | 7.2e-07 | 7.2e-07 | - |  |
| shvit_s4.in1k | fp16 | 6.1 / 4.8 | 1.28 | 1.14-1.36 | 6.8 | 0.08 / 1.86 | 40 / 52 | 8 / 20 | 0.45 | 32 / 32 | 389 / 454 | 8.5 / 5.5 | 1.04 / 0.79 | 0.08 / 1.87 | 0.002 | 0.002 | - | slower shipped-slower |
| shvit_s4.in1k | fp32 | 5.5 / 5.2 | 1.06 | 0.98-1.45 | 6.9 | 0.14 / 1.72 | 75 / 94 | 11 / 30 | 0.40 | 64 / 64 | 424 / 492 | 7.8 / 6.1 | 0.91 / 0.85 | 0.14 / 1.73 | 5e-07 | 5e-07 | - | shipped-slower |
| siglip2-b16-256 | fp16 | 14.5 / 14.4 | 1.01 | - | - | 0.04 / 0.70 | 197 / 244 | ~21 / ~67 | 0.33 | 176 / 177 | 863 / 616 | 14.8 / 14.6 | 0.84 / 0.70 | 0.05 / 0.70 | 0.0065 | 0.0065 | - |  |
| siglip2-b16-256 | fp32 | 16.8 / 16.7 | 1.00 | - | - | 0.06 / 0.70 | 387 / 444 | ~35 / ~89 | 0.39 | 352 / 354 | 705 / 810 | 17.0 / 17.0 | 0.78 / 0.71 | 0.06 / 0.70 | 4.8e-06 | 4.8e-06 | - |  |
| siglip2-b16-512 | fp16 | 14.5 / 14.3 | 1.02 | - | - | 0.04 / 0.72 | 197 / 244 | ~21 / ~67 | 0.33 | 176 / 177 | 868 / 620 | 14.8 / 14.5 | 0.85 / 0.71 | 0.04 / 0.72 | 0.0069 | 0.0069 | - |  |
| siglip2-b16-512 | fp32 | 16.8 / 16.7 | 1.01 | - | - | 0.07 / 0.70 | 387 / 444 | ~35 / ~89 | 0.39 | 352 / 354 | 712 / 818 | 17.0 / 17.1 | 0.77 / 0.70 | 0.07 / 0.70 | 3.6e-06 | 3.6e-06 | - |  |
| siglip2-g16-256 | fp16 | 158.8 / 157.8 | 1.01 | 0.97-1.04 | 159.7 | 0.24 / 10.29 | 2282 / 2266 | 68 / 47 | 1.43 | 2214 / 2218 | 5771 / 2692 | 156.2 / 155.6 | 9.23 / 1.10 | 0.25 / 21.43 | 0.076 (0.33 % off) | 0.076 (0.33 % off) | - | heavier |
| siglip2-g16-256 | fp32 | 185.8 / 186.7 | 1.00 | 0.96-1.01 | 188.3 | 1.20 / 10.53 | 4497 / 4515 | 68 / 77 | 0.89 | 4429 / 4438 | 4829 / 4894 | 187.2 / 188.4 | 7.31 / 0.98 | 1.20 / 23.45 | 1.3e-05 | 1.3e-05 | - |  |
| siglip2-g16-384 | fp16 | 154.5 / 149.2 | 1.04 | 1.03-1.04 | 157.7 | 0.19 / 9.55 | 2282 / 2266 | 68 / 47 | 1.43 | 2214 / 2218 | 5771 / 2696 | 155.1 / 149.8 | 8.21 / 1.00 | 0.19 / 16.86 | 0.029 (0.13 % off) | 0.029 (0.13 % off) | - | slower heavier shipped-slower |
| siglip2-g16-384 | fp32 | 184.3 / 182.1 | 1.01 | - | - | 0.48 / 2.84 | 4497 / 4515 | ~68 / ~77 | 0.89 | 4429 / 4438 | 4829 / 4942 | 184.5 / 181.3 | 5.29 / 1.02 | 0.48 / 2.84 | 3.4e-05 | 3.4e-05 | - |  |
| siglip2-l16-256 | fp16 | 44.7 / 44.1 | 1.01 | - | - | 0.06 / 1.16 | 623 / 663 | ~23 / ~60 | 0.38 | 601 / 602 | 2134 / 1052 | 45.1 / 44.2 | 1.02 / 0.72 | 0.07 / 1.16 | 0.089 (0.57 % off) | 0.089 (0.57 % off) | - |  |
| siglip2-l16-256 | fp32 | 52.8 / 51.6 | 1.02 | 1.02-1.06 | 52.7 | 0.20 / 1.81 | 1238 / 1301 | 37 / 96 | 0.39 | 1201 / 1205 | 1562 / 1684 | 53.5 / 52.4 | 1.78 / 0.85 | 0.20 / 1.90 | 4.5e-05 | 4.5e-05 | - |  |
| siglip2-l16-512 | fp16 | 46.6 / 44.6 | 1.05 | 1.02-1.08 | 46.9 | 0.19 / 1.87 | 646 / 663 | 46 / 60 | 0.76 | 601 / 602 | 2169 / 1058 | 46.6 / 46.7 | 2.50 / 0.92 | 0.19 / 2.08 | 0.011 (0.0005 % off) | 0.011 (0.0005 % off) | - | slower shipped-slower |
| siglip2-l16-512 | fp32 | 53.7 / 51.9 | 1.04 | 1.03-1.04 | 53.6 | 0.18 / 1.36 | 1247 / 1301 | 46 / 96 | 0.48 | 1201 / 1205 | 1580 / 1702 | 54.1 / 52.2 | 1.50 / 0.85 | 0.18 / 1.43 | 1.8e-05 | 1.8e-05 | - | slower shipped-slower |
| siglip2-so400m14-224 | fp16 | 63.9 / 63.2 | 1.01 | - | - | 0.08 / 1.36 | 846 / 883 | ~32 / ~67 | 0.49 | 814 / 816 | 2790 / 1262 | 64.2 / 63.4 | 1.63 / 0.70 | 0.08 / 1.37 | 0.046 (0.014 % off) | 0.046 (0.014 % off) | - |  |
| siglip2-so400m14-224 | fp32 | 76.0 / 73.7 | 1.03 | 0.95-1.08 | 76.8 | 0.48 / 3.23 | 1680 / 1740 | 53 / 108 | 0.49 | 1627 / 1632 | 2006 / 2125 | 75.0 / 72.8 | 3.39 / 0.93 | 0.49 / 3.34 | 2.8e-05 | 2.8e-05 | - |  |
| siglip2-so400m14-378 | fp16 | 64.0 / 62.9 | 1.02 | - | - | 0.09 / 1.38 | 846 / 883 | ~32 / ~67 | 0.49 | 814 / 816 | 2796 / 1269 | 64.2 / 63.4 | 1.78 / 0.71 | 0.09 / 1.38 | 0.02 (0.001 % off) | 0.02 (0.001 % off) | - |  |
| siglip2-so400m14-378 | fp32 | 73.7 / 73.8 | 1.00 | - | - | 0.13 / 1.52 | 1680 / 1740 | ~52 / ~108 | 0.49 | 1627 / 1632 | 2008 / 2143 | 74.5 / 74.2 | 0.97 / 0.74 | 0.13 / 1.52 | 1.7e-05 | 1.7e-05 | - |  |
| siglip2-so400m16-256 | fp16 | 65.0 / 61.6 | 1.06 | 1.02-1.06 | 63.1 | 0.17 / 3.64 | 863 / 884 | 49 / 67 | 0.74 | 814 / 817 | 2816 / 1268 | 64.9 / 61.9 | 3.95 / 0.91 | 0.18 / 3.72 | 0.037 (0.0071 % off) | 0.037 (0.0071 % off) | - | slower shipped-slower |
| siglip2-so400m16-256 | fp32 | 75.1 / 71.3 | 1.05 | 1.03-1.06 | 77.4 | 0.39 / 3.19 | 1678 / 1740 | 50 / 107 | 0.48 | 1628 / 1633 | 2006 / 2116 | 74.9 / 71.4 | 3.14 / 0.94 | 0.40 / 3.32 | 4.6e-05 | 4.6e-05 | - | slower shipped-slower |
| siglip2-so400m16-512 | fp16 | 64.1 / 61.7 | 1.04 | 0.87-1.06 | 63.3 | 0.20 / 3.38 | 863 / 884 | 49 / 67 | 0.74 | 814 / 817 | 2820 / 1270 | 64.7 / 61.9 | 3.93 / 0.93 | 0.21 / 3.45 | 0.0061 | 0.0061 | - |  |
| siglip2-so400m16-512 | fp32 | 74.9 / 70.7 | 1.06 | 1.04-1.10 | 75.7 | 0.47 / 3.44 | 1678 / 1740 | 50 / 107 | 0.48 | 1628 / 1633 | 2012 / 2129 | 75.7 / 71.1 | 2.72 / 0.91 | 0.47 / 3.63 | 9.6e-06 | 9.6e-06 | - | slower shipped-slower |
| span-x2 | fp16 | 59.3 / 72.1 | 0.82 | - | - | 0.11 / 1.08 | 391 / 499 | ~387 / ~495 | 0.78 | 4 / 4 | 830 / 1125 | 60.2 / 72.6 | 0.09 / 0.13 | 0.12 / 1.08 | 1 (0.14 % off) | 1 (0.46 % off) | - |  |
| span-x2 | fp32 | 113.4 / 149.7 | 0.76 | - | - | 0.15 / 1.14 | 661 / 613 | ~659 / ~611 | 1.08 | 2 / 2 | 1036 / 1270 | 114.3 / 150.0 | 0.09 / 0.13 | 0.15 / 1.14 | 0 | 0 | - | heavier |
| span-x2 | fp16 | 99.8 / 118.3 | 0.84 | - | - | 0.13 / 1.09 | 332 / 345 | ~331 / ~344 | 0.96 | 1 / 1 | 674 / 893 | 100.7 / 118.6 | 0.09 / 0.13 | 0.14 / 1.10 | 0.11 (0.021 % off) | 1 (0.15 % off) | - |  |
| span-x2 | fp32 | 114.2 / 152.3 | 0.75 | 0.67-0.76 | 118.8 | 0.54 / 1.54 | 661 / 597 | 656 / 595 | 1.10 | 2 / 2 | 1042 / 1278 | 114.9 / 154.5 | 0.11 / 0.18 | 0.55 / 1.55 | 0 | 0 | - | heavier |
| span-x4 | fp16 | 60.2 / 74.0 | 0.81 | - | - | 0.11 / 1.11 | 481 / 535 | ~477 / ~531 | 0.90 | 4 / 4 | 953 / 1262 | 62.9 / 75.4 | 0.09 / 0.13 | 0.13 / 1.11 | 0.03 (0.00061 % off) | 0.036 (0.001 % off) | - |  |
| span-x4 | fp32 | 116.7 / 156.6 | 0.75 | - | - | 0.15 / 1.14 | 661 / 613 | ~659 / ~611 | 1.08 | 2 / 2 | 1113 / 1359 | 119.2 / 154.7 | 0.09 / 0.13 | 0.17 / 1.14 | 0 | 0 | - | heavier |
| span-x4 | fp16 | 103.1 / 122.4 | 0.84 | - | - | 0.13 / 1.10 | 380 / 380 | ~379 / ~379 | 1.00 | 1 / 1 | 799 / 1027 | 105.6 / 124.0 | 0.09 / 0.13 | 0.15 / 1.11 | 0.0074 | 0.0088 | - |  |
| span-x4 | fp32 | 117.5 / 155.7 | 0.75 | 0.75-0.76 | 125.5 | 0.16 / 1.26 | 661 / 613 | 656 / 611 | 1.07 | 2 / 2 | 1117 / 1366 | 120.4 / 158.7 | 0.10 / 0.17 | 0.18 / 1.28 | 0 | 0 | - | heavier |
| squeezenet1_1.tv_in1k | fp16 | 1.8 / 3.0 | 0.61 | - | - | 0.02 / 0.45 | 16 / 18 | ~13 / ~16 | 0.84 | 2 / 2 | 307 / 336 | 2.1 / 3.3 | 0.56 / 0.54 | 0.02 / 0.45 | 0 | 0 | 0.00209 / 0.00209 (0.02) |  |
| squeezenet1_1.tv_in1k | fp32 | 1.9 / 2.2 | 0.86 | - | - | 0.04 / 0.44 | 28 / 32 | ~23 / ~27 | 0.84 | 5 / 5 | 314 / 346 | 2.0 / 2.3 | 0.51 / 0.53 | 0.05 / 0.45 | 9.2e-08 | 9.2e-08 | 4.19e-07 / 4.23e-07 (0.02) |  |
| starnet_s1.in1k | fp16 | 2.1 / 2.6 | 0.81 | - | - | 0.02 / 0.66 | 12 / 24 | ~6 / ~18 | 0.38 | 6 / 6 | 340 / 378 | 2.5 / 3.1 | 0.77 / 0.69 | 0.02 / 0.66 | 0.0015 | 0.0015 | - |  |
| starnet_s1.in1k | fp32 | 2.3 / 3.3 | 0.68 | - | - | 0.03 / 0.66 | 24 / 43 | ~12 / ~32 | 0.40 | 11 / 11 | 350 / 400 | 2.7 / 3.6 | 0.71 / 0.71 | 0.03 / 0.66 | 1.6e-06 | 1.6e-06 | - |  |
| starnet_s2.in1k | fp16 | 2.0 / 2.3 | 0.87 | - | - | 0.01 / 0.57 | 14 / 24 | ~7 / ~16 | 0.45 | 7 / 7 | 340 / 382 | 2.3 / 2.8 | 0.74 / 0.71 | 0.02 / 0.57 | 0.00093 | 0.00093 | - |  |
| starnet_s2.in1k | fp32 | 2.0 / 2.6 | 0.78 | - | - | 0.03 / 0.56 | 27 / 44 | ~13 / ~29 | 0.45 | 14 / 14 | 352 / 401 | 2.5 / 3.1 | 0.72 / 0.69 | 0.04 / 0.56 | 6.2e-07 | 6.2e-07 | - |  |
| starnet_s3.in1k | fp16 | 2.3 / 3.0 | 0.78 | - | - | 0.02 / 0.69 | 19 / 34 | ~8 / ~23 | 0.36 | 11 / 11 | 348 / 396 | 2.8 / 3.5 | 0.78 / 0.69 | 0.02 / 0.69 | 0.00078 | 0.00078 | - |  |
| starnet_s3.in1k | fp32 | 2.7 / 3.4 | 0.78 | - | - | 0.04 / 0.66 | 37 / 63 | ~15 / ~41 | 0.37 | 22 / 22 | 366 / 424 | 3.1 / 3.8 | 0.71 / 0.69 | 0.04 / 0.67 | 3.8e-07 | 3.8e-07 | - |  |
| starnet_s4.in1k | fp16 | 3.0 / 3.9 | 0.77 | - | - | 0.02 / 0.85 | 26 / 46 | ~12 / ~32 | 0.39 | 14 / 14 | 357 / 411 | 3.5 / 4.5 | 0.79 / 0.71 | 0.02 / 0.85 | 0.00086 | 0.00086 | - |  |
| starnet_s4.in1k | fp32 | 3.5 / 4.3 | 0.80 | - | - | 0.04 / 0.83 | 48 / 87 | ~19 / ~58 | 0.35 | 29 / 29 | 383 / 453 | 4.0 / 4.8 | 0.73 / 0.71 | 0.04 / 0.84 | 4.2e-07 | 4.2e-07 | - |  |
| swin2sr-x2 | fp16 | 3456.4 / 3265.6 | 1.06 | 1.05-1.07 | 3568.7 | 3.98 / 8.09 | 932 / 1277 | 896 / 1185 | 0.76 | 33 / 92 | 2455 / 3440 | 3516.9 / 3274.9 | 0.76 / 0.46 | 3.99 / 8.11 | 0.0059 | 0.0064 | - | slower shipped-slower |
| swin2sr-x2 | fp32 | 4362.9 / 5434.3 | 0.80 | - | - | 4.56 / 9.73 | 1575 / 2069 | ~1516 / ~1953 | 0.78 | 60 / 116 | 3591 / 4096 | 4400.9 / 5461.8 | 0.70 / 0.27 | 4.56 / 9.74 | 2.7e-06 | 2.7e-06 | - |  |
| swin2sr-x4 | fp16 | 3921.4 / 3728.4 | 1.05 | 0.98-1.27 | 3827.0 | 4.04 / 9.94 | 1022 / 2206 | 985 / 2114 | 0.47 | 33 / 92 | 3424 / 5133 | 3995.4 / 3769.2 | 0.85 / 0.50 | 4.07 / 10.03 | 0.012 (5.4e-05 % off) | 0.014 (0.00021 % off) | - |  |
| swin2sr-x4 | fp32 | 4605.0 / 5796.5 | 0.79 | - | - | 4.83 / 10.19 | 1576 / 4341 | ~1516 / ~4224 | 0.36 | 60 / 117 | 3940 / 7646 | 4627.1 / 6034.6 | 0.70 / 0.26 | 4.85 / 10.20 | 5.5e-06 | 6.4e-06 | - |  |
| swin2sr-x4-real | fp16 | 3814.4 / 3754.6 | 1.02 | 0.95-1.12 | 3850.7 | 3.97 / 8.92 | 792 / 1486 | 756 / 1395 | 0.54 | 33 / 92 | 2986 / 4075 | 3819.4 / 3754.6 | 0.86 / 0.47 | 4.04 / 8.95 | 0.0033 | 0.0035 | - |  |
| swin2sr-x4-real | fp32 | 4722.6 / 6065.2 | 0.78 | - | - | 4.80 / 9.83 | 1412 / 2496 | ~1352 / ~2380 | 0.57 | 60 / 116 | 3855 / 5037 | 4717.6 / 5897.9 | 0.68 / 0.26 | 4.82 / 9.84 | 2.1e-06 | 2.2e-06 | - |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp16 | 18.4 / 18.6 | 0.99 | - | - | 0.06 / 1.95 | 221 / 263 | ~48 / ~93 | 0.52 | 173 / 170 | 668 / 686 | 18.9 / 19.2 | 0.88 / 0.71 | 0.06 / 1.95 | 0.0029 | 0.0029 | - |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp32 | 23.3 / 25.7 | 0.91 | 0.84-0.94 | 24.0 | 0.15 / 2.26 | 459 / 422 | 113 / 83 | 1.36 | 346 / 340 | 888 / 907 | 24.9 / 28.2 | 2.08 / 0.86 | 0.16 / 2.29 | 1.6e-06 | 1.6e-06 | - | heavier |
| swinir-l-x4-real | fp16 | 6783.3 / 6479.0 | 1.05 | 0.98-1.07 | 6790.3 | 7.13 / 13.08 | 951 / 1978 | 880 / 1852 | 0.48 | 68 / 127 | 3108 / 4209 | 6503.2 / 6490.3 | 0.82 / 0.73 | 7.16 / 13.11 | 0.034 (0.02 % off) | 0.04 (0.042 % off) | - |  |
| swinir-l-x4-real | fp32 | 8269.7 / 10549.6 | 0.78 | - | - | 8.35 / 16.65 | 1889 / 2564 | ~1762 / ~2380 | 0.74 | 127 / 184 | 3993 / 4804 | 8344.2 / 10716.7 | 0.68 / 0.64 | 8.37 / 16.66 | 1.8e-05 | 2.1e-05 | - |  |
| swinir-m-x2 | fp16 | 2943.8 / 2769.0 | 1.06 | 1.05-1.07 | 3123.2 | 3.35 / 7.58 | 932 / 1169 | 896 / 1077 | 0.83 | 33 / 92 | 2432 / 2978 | 2942.4 / 2810.5 | 0.85 / 0.39 | 3.36 / 7.60 | 0.0065 | 0.006 | - | slower shipped-slower |
| swinir-m-x2 | fp32 | 3905.8 / 3942.6 | 0.99 | - | - | 4.09 / 8.09 | 1389 / 1642 | ~1330 / ~1526 | 0.87 | 59 / 116 | 3704 / 3975 | 3912.5 / 3993.5 | 0.51 / 0.23 | 4.10 / 8.09 | 2.6e-06 | 3e-06 | - |  |
| swinir-m-x4 | fp16 | 3081.4 / 3027.1 | 1.02 | - | - | 3.24 / 7.18 | 1021 / 2206 | ~988 / ~2114 | 0.47 | 33 / 92 | 3384 / 4870 | 3140.9 / 3108.6 | 0.62 / 0.23 | 3.26 / 7.19 | 0.01 (7.7e-06 % off) | 0.012 (1.6e-05 % off) | - |  |
| swinir-m-x4 | fp32 | 4146.5 / 4299.2 | 0.96 | - | - | 4.39 / 8.54 | 1417 / 4341 | ~1357 / ~4224 | 0.32 | 59 / 117 | 3919 / 7421 | 4154.9 / 4277.7 | 0.67 / 0.23 | 4.41 / 8.55 | 4.8e-06 | 5.5e-06 | - |  |
| swinir-m-x4-real | fp16 | 3196.2 / 3140.7 | 1.02 | - | - | 3.29 / 7.08 | 699 / 1415 | ~667 / ~1323 | 0.50 | 33 / 92 | 2951 / 4016 | 3192.9 / 3114.2 | 0.67 / 0.22 | 3.31 / 7.09 | 0.044 (0.011 % off) | 0.06 (0.036 % off) | - |  |
| swinir-m-x4-real | fp32 | 4243.3 / 4495.1 | 0.94 | - | - | 4.38 / 9.16 | 1389 / 2496 | ~1330 / ~2380 | 0.56 | 59 / 116 | 3805 / 4846 | 4283.3 / 4365.5 | 0.62 / 0.24 | 4.41 / 9.17 | 1.3e-05 | 1.8e-05 | - |  |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp16 | 4.5 / 5.0 | 0.90 | - | - | 0.04 / 1.07 | 43 / 64 | ~18 / ~40 | 0.46 | 25 / 25 | 402 / 437 | 5.0 / 5.4 | 0.85 / 0.81 | 0.04 / 1.07 | 0.0015 | 0.0015 | 0.00527 / 0.00519 (0.0095) |  |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp32 | 5.9 / 5.3 | 1.11 | 0.92-1.36 | 5.6 | 0.06 / 1.07 | 82 / 120 | 33 / 70 | 0.48 | 49 / 49 | 443 / 495 | 6.1 / 6.2 | 1.17 / 0.84 | 0.06 / 1.08 | 1.2e-06 | 1.2e-06 | 4.9e-06 / 4.94e-06 (0.0095) |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp16 | 6.6 / 6.8 | 0.98 | - | - | 0.04 / 0.98 | 72 / 104 | ~26 / ~58 | 0.46 | 46 / 46 | 451 / 480 | 7.2 / 7.3 | 0.84 / 0.73 | 0.04 / 0.98 | 0.0016 | 0.0016 | 0.00254 / 0.00258 (0.0095) |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp32 | 7.5 / 8.0 | 0.93 | - | - | 0.06 / 0.95 | 148 / 192 | ~57 / ~100 | 0.57 | 92 / 92 | 519 / 567 | 8.0 / 8.6 | 0.81 / 0.74 | 0.06 / 0.95 | 1e-06 | 1e-06 | 3.91e-06 / 3.86e-06 (0.0095) |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp16 | 14.3 / 14.5 | 0.99 | - | - | 0.05 / 1.09 | 162 / 184 | ~74 / ~96 | 0.78 | 88 / 88 | 652 / 652 | 14.7 / 14.8 | 1.19 / 1.09 | 0.05 / 1.09 | 0.001 | 0.001 | 0.00251 / 0.00229 (0.0095) |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp32 | 19.1 / 18.2 | 1.05 | 0.99-1.10 | 19.6 | 0.08 / 1.07 | 252 / 289 | 76 / 113 | 0.67 | 176 / 176 | 758 / 816 | 19.6 / 18.8 | 1.38 / 1.14 | 0.08 / 1.09 | 6.8e-07 | 6.8e-07 | 4.17e-06 / 3.95e-06 (0.0095) |  |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 24.8 / 25.3 | 0.98 | 0.95-1.04 | 25.0 | 0.06 / 1.12 | 333 / 300 | 142 / 110 | 1.30 | 191 / 191 | 958 / 892 | 25.7 / 25.9 | 2.17 / 1.82 | 0.06 / 1.14 | 0.0009 | 0.0009 | 0.00261 / 0.00268 (0.0095) | heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp32 | 32.3 / 31.8 | 1.01 | - | - | 0.12 / 0.99 | 537 / 537 | ~156 / ~156 | 1.00 | 381 / 381 | 1125 / 1171 | 33.1 / 32.1 | 1.87 / 1.70 | 0.12 / 0.99 | 6.3e-07 | 6.3e-07 | 4.43e-06 / 4.31e-06 (0.0095) |  |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp16 | 4.1 / 4.6 | 0.89 | - | - | 0.04 / 0.94 | 29 / 43 | ~16 / ~30 | 0.54 | 13 / 13 | 371 / 409 | 4.6 / 4.6 | 0.84 / 0.74 | 0.04 / 0.94 | 0.0014 | 0.0014 | 0.00446 / 0.00407 (0.0095) |  |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp32 | 4.4 / 4.5 | 0.97 | - | - | 0.05 / 0.92 | 56 / 81 | ~31 / ~55 | 0.57 | 25 / 25 | 403 / 448 | 4.9 / 5.2 | 0.85 / 0.76 | 0.05 / 0.93 | 8.5e-07 | 8.5e-07 | 3.9e-06 / 3.5e-06 (0.0095) |  |
| totalsegmentator-total | fp16 | 14612.3 / 26280.3 | 0.56 | - | - | 14.55 / 33.05 | 3464 / 3912 | ~2897 / ~3615 | 0.80 | 567 / 298 | 6117 / 6882 | 17992.2 / 30890.6 | 1.49 / 0.28 | 18.41 / 37.53 | - | 1 (0.0048 % off) (1 keys differ) | - | calls-differ |
| totalsegmentator-total | fp32 | 21167.3 / 38793.7 | 0.55 | - | - | 21.18 / 42.58 | 3456 / 3954 | ~2862 / ~3359 | 0.85 | 595 / 595 | 5493 / 6737 | 25054.6 / 42912.5 | 0.07 / 0.26 | 25.89 / 47.26 | - | 0 (1 keys differ) | - | calls-differ |
| totalsegmentator-total-3mm | fp16 | 367.5 / 544.6 | 0.67 | 0.60-0.74 | 362.1 | 0.79 / 2.09 | 2058 / 1609 | 2007 / 1574 | 1.28 | 45 / 32 | 4966 / 6240 | 547.6 / 787.1 | 0.31 / 0.88 | 1.34 / 3.00 | 0.045 (0.0023 % off) | 1 (0.29 % off) | 0.000534 / 0.00061 (0.003) | heavier |
| totalsegmentator-total-3mm | fp32 | 473.8 / 604.4 | 0.78 | 0.75-0.81 | 472.4 | 0.77 / 2.14 | 2033 / 1769 | 1960 / 1703 | 1.15 | 63 / 63 | 6064 / 6268 | 722.6 / 943.1 | 0.14 / 0.89 | 1.54 / 3.18 | 0.00012 | 0.74 (0.002 % off) | 6.61e-05 / 6.61e-05 (0.003) | heavier |
| u2net | fp16 | 51.2 / 55.9 | 0.92 | - | - | 0.06 / 2.48 | 276 / 326 | ~188 / ~241 | 0.78 | 88 / 84 | 744 / 1058 | 54.2 / 58.9 | 0.06 / 0.10 | 0.10 / 2.49 | - | 0.026 (0.082 % off) | 9.65e-05 / 0.0001 (0.02) |  |
| u2net | fp32 | 67.8 / 70.8 | 0.96 | - | - | 0.09 / 2.80 | 555 / 655 | ~383 / ~487 | 0.79 | 172 / 168 | 1107 / 1417 | 73.3 / 72.9 | 0.07 / 0.10 | 0.13 / 2.81 | - | 2e-05 | 1.11e-07 / 1.11e-07 (0.02) |  |
| u2net-human-seg | fp16 | 49.8 / 55.6 | 0.90 | - | - | 0.06 / 2.49 | 276 / 326 | ~188 / ~241 | 0.78 | 88 / 84 | 742 / 1058 | 53.2 / 59.7 | 0.06 / 0.11 | 0.10 / 2.50 | - | 0.01 (0.00019 % off) | 0.000356 / 0.000117 (0.02) |  |
| u2net-human-seg | fp32 | 61.8 / 71.1 | 0.87 | - | - | 0.08 / 2.44 | 555 / 655 | ~383 / ~487 | 0.79 | 172 / 168 | 1106 / 1388 | 66.0 / 74.9 | 0.07 / 0.10 | 0.12 / 2.44 | - | 1.9e-05 | 1.67e-07 / 1.57e-07 (0.02) |  |
| u2netp | fp16 | 21.5 / 26.5 | 0.81 | - | - | 0.04 / 2.29 | 128 / 119 | ~122 / ~116 | 1.05 | 6 / 2 | 508 / 590 | 26.0 / 29.6 | 0.06 / 0.08 | 0.07 / 2.30 | - | 0.041 (0.15 % off) | 0.000352 / 0.00037 (0.02) | heavier |
| u2netp | fp32 | 26.5 / 31.8 | 0.83 | - | - | 0.05 / 2.21 | 250 / 235 | ~241 / ~230 | 1.05 | 9 / 5 | 634 / 700 | 30.2 / 35.8 | 0.06 / 0.08 | 0.10 / 2.22 | - | 9.8e-06 | 2.78e-07 / 2.79e-07 (0.02) | heavier |
| unik3d-vitl | fp16 | 1009.3 / 1021.2 | 0.99 | 0.98-1.00 | 1027.5 | 1.07 / 4.45 | 1523 / 1166 | 808 / 470 | 1.72 | 715 / 696 | 3334 / 3153 | 1021.3 / 1030.8 | 2.92 / 0.10 | 1.09 / 4.52 | 0.009 | 0.016 (1.1 % off) | - | heavier |
| unik3d-vitl | fp32 | 1214.4 / 1191.7 | 1.02 | - | - | 1.36 / 3.91 | 3060 / 3117 | ~1685 / ~1731 | 0.97 | 1376 / 1386 | 4936 / 5024 | 1221.6 / 1197.9 | 1.36 / 0.07 | 1.37 / 3.92 | 5.1e-06 | 1.2e-05 | - |  |
| vgg16.tv_in1k | fp16 | 13.2 / 13.6 | 0.97 | - | - | 0.03 / 0.47 | 322 / 353 | ~58 / ~89 | 0.65 | 264 / 264 | 2102 / 1860 | 13.6 / 14.1 | 0.84 / 0.55 | 0.03 / 0.47 | 0 | 0 | - |  |
| vgg16.tv_in1k | fp32 | 16.7 / 17.5 | 0.96 | - | - | 0.04 / 0.50 | 643 / 705 | ~115 / ~177 | 0.65 | 528 / 528 | 1936 / 1932 | 17.3 / 18.0 | 0.79 / 0.57 | 0.04 / 0.50 | 0 | 0 | - |  |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp16 | 14.8 / 14.2 | 1.05 | 0.95-1.39 | 14.4 | 0.18 / 0.91 | 193 / 232 | 28 / 67 | 0.43 | 165 / 165 | 407 / 595 | 15.4 / 15.3 | 0.17 / 0.92 | 0.19 / 0.93 | 0.0011 | 0.0011 | 0.00117 / 0.00124 (0.02) |  |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp32 | 16.9 / 17.8 | 0.95 | - | - | 0.07 / 0.75 | 365 / 420 | ~35 / ~90 | 0.39 | 330 / 330 | 577 / 781 | 17.1 / 18.3 | 0.06 / 0.80 | 0.07 / 0.75 | 5.8e-07 | 5.8e-07 | 5.86e-06 / 5.9e-06 (0.02) |  |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp16 | 189.2 / 186.1 | 1.02 | 1.00-1.04 | 195.9 | 0.46 / 4.60 | 882 / 900 | 65 / 80 | 0.82 | 817 / 820 | 1114 / 1280 | 188.0 / 185.7 | 1.11 / 1.27 | 0.47 / 4.87 | 0.00081 | 0.00081 | 0.00249 / 0.00225 (0.02) |  |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp32 | 217.2 / 212.1 | 1.02 | 1.01-1.05 | 216.0 | 0.48 / 3.29 | 1740 / 1796 | 106 / 157 | 0.68 | 1634 / 1639 | 1982 / 2168 | 220.9 / 213.7 | 2.16 / 1.20 | 0.49 / 3.42 | 5.2e-07 | 5.2e-07 | 1.48e-05 / 1.45e-05 (0.02) |  |
| vitmatte-b | fp16 | 997.0 / 1576.2 | 0.63 | - | - | 1.08 / 4.02 | 734 / 952 | ~515 / ~742 | 0.69 | 219 / 209 | 3014 / 3323 | 1003.7 / 1582.6 | 0.06 / 0.05 | 1.10 / 4.03 | 0.016 (0.0021 % off) | 0.016 (0.0021 % off) | 1.89e-05 / 2.61e-05 (5.3e-05) |  |
| vitmatte-b | fp32 | 1276.4 / 2109.1 | 0.61 | - | - | 1.32 / 4.52 | 1481 / 1584 | ~1068 / ~1166 | 0.92 | 412 / 418 | 3824 / 4409 | 1285.6 / 2093.7 | 0.07 / 0.05 | 1.35 / 4.53 | 3.2e-05 | 3.2e-05 | 3.96e-08 / 3.99e-08 (5.3e-05) |  |
| vitmatte-s | fp16 | 518.8 / 742.8 | 0.70 | - | - | 0.58 / 3.21 | 646 / 898 | ~575 / ~830 | 0.69 | 71 / 68 | 2935 / 3025 | 541.4 / 750.3 | 0.06 / 0.04 | 0.60 / 3.22 | 0.045 (0.057 % off) | 0.045 (0.058 % off) | 1.66e-05 / 2.11e-05 (5.3e-05) |  |
| vitmatte-s | fp32 | 693.2 / 1063.3 | 0.65 | - | - | 0.78 / 3.58 | 1186 / 1381 | ~1056 / ~1245 | 0.85 | 130 / 136 | 3567 / 4289 | 717.1 / 1084.2 | 0.06 / 0.05 | 0.80 / 3.59 | 4.8e-05 | 4.8e-05 | 4.14e-08 / 4.13e-08 (5.3e-05) |  |
| vitpose-base | fp16 | 144.5 / 145.3 | 0.99 | - | - | 0.17 / 1.40 | 272 / 277 | ~100 / ~105 | 0.95 | 172 / 172 | 735 / 832 | 157.1 / 158.0 | 0.05 / 0.04 | 0.19 / 1.42 | 0.0051 | 0.025 (0.12 % off) | - |  |
| vitpose-base | fp32 | 160.5 / 163.6 | 0.98 | 0.97-0.98 | 163.2 | 0.20 / 1.43 | 535 / 469 | 183 / 126 | 1.45 | 343 / 343 | 1134 / 1044 | 173.7 / 177.4 | 0.70 / 0.04 | 0.22 / 1.47 | 4.8e-06 | 4.8e-06 | - | heavier |
| vitpose-base-simple | fp16 | 141.6 / 137.5 | 1.03 | 1.03-1.03 | 141.9 | 0.21 / 1.62 | 278 / 346 | 106 / 182 | 0.58 | 164 / 164 | 816 / 930 | 154.4 / 151.0 | 0.59 / 0.04 | 0.23 / 1.66 | 0.0038 | 0.15 (0.24 % off) | - | slower shipped-slower |
| vitpose-base-simple | fp32 | 156.4 / 158.1 | 0.99 | - | - | 0.19 / 1.40 | 556 / 1162 | ~228 / ~834 | 0.27 | 328 / 328 | 1040 / 1713 | 168.9 / 171.6 | 0.06 / 0.04 | 0.21 / 1.42 | 3.9e-06 | 2.9e-06 | - |  |
| vitpose-plus-base | fp16 | 144.6 / 144.6 | 1.00 | - | - | 0.17 / 1.40 | 272 / 277 | ~100 / ~105 | 0.95 | 172 / 172 | 735 / 876 | 157.2 / 156.8 | 0.05 / 0.06 | 0.19 / 1.42 | 0.0042 | 0.0085 | - |  |
| vitpose-plus-base | fp32 | 160.2 / 164.0 | 0.98 | 0.97-0.98 | 163.5 | 0.25 / 1.64 | 535 / 469 | 183 / 126 | 1.45 | 343 / 343 | 1184 / 1032 | 173.3 / 177.2 | 0.76 / 0.06 | 0.27 / 1.68 | 3e-06 | 5.2e-06 | - | heavier |
| vitpose-plus-huge | fp16 | 886.8 / 862.7 | 1.03 | 1.01-1.08 | 883.2 | 1.05 / 8.93 | 1372 / 1407 | 148 / 191 | 0.77 | 1215 / 1216 | 2552 / 3164 | 916.6 / 880.9 | 5.56 / 1.71 | 1.10 / 12.05 | 0.0056 | 0.11 (0.36 % off) | - |  |
| vitpose-plus-huge | fp32 | 1000.3 / 981.7 | 1.02 | - | - | 1.23 / 4.20 | 2693 / 2777 | ~262 / ~346 | 0.76 | 2431 / 2431 | 3293 / 4351 | 1012.5 / 994.6 | 1.17 / 0.69 | 1.26 / 4.23 | 3.2e-06 | 2.8e-06 | - |  |
| vitpose-plus-large | fp16 | 445.3 / 453.4 | 0.98 | 0.96-1.05 | 475.9 | 0.68 / 3.61 | 715 / 693 | 118 / 104 | 1.13 | 588 / 588 | 1456 / 1741 | 463.5 / 476.2 | 3.63 / 2.02 | 0.71 / 3.83 | 0.0026 | 0.21 (0.24 % off) | - | heavier |
| vitpose-plus-large | fp32 | 494.0 / 486.0 | 1.02 | - | - | 0.53 / 2.15 | 1391 / 1512 | ~214 / ~335 | 0.64 | 1177 / 1177 | 1909 / 2337 | 506.1 / 499.8 | 0.11 / 0.11 | 0.55 / 2.17 | 2.9e-06 | 2.7e-06 | - |  |
| vitpose-plus-small | fp16 | 61.8 / 59.8 | 1.03 | 1.03-1.04 | 62.3 | 0.14 / 1.78 | 129 / 176 | 74 / 129 | 0.58 | 46 / 46 | 622 / 685 | 74.7 / 73.3 | 0.28 / 0.06 | 0.16 / 1.81 | 0.0032 | 0.053 (0.48 % off) | - | slower shipped-slower |
| vitpose-plus-small | fp32 | 70.1 / 69.5 | 1.01 | - | - | 0.11 / 1.29 | 250 / 249 | ~158 / ~156 | 1.01 | 93 / 93 | 766 / 850 | 82.5 / 81.4 | 0.05 / 0.04 | 0.13 / 1.31 | 2.7e-06 | 2.5e-06 | - |  |
| wd-convnext-tagger-v3 | fp16 | 42.7 / 43.8 | 0.98 | - | - | 0.06 / 1.09 | 274 / 328 | ~85 / ~139 | 0.61 | 189 / 189 | 832 / 853 | 43.2 / 44.1 | 0.97 / 0.73 | 0.06 / 1.10 | 0.00096 | 0.00096 | 0.0032 / 0.00336 (0.015) |  |
| wd-convnext-tagger-v3 | fp32 | 49.6 / 53.5 | 0.93 | 0.86-0.95 | 51.2 | 0.18 / 1.12 | 516 / 512 | 139 / 135 | 1.03 | 377 / 377 | 1056 / 1160 | 50.2 / 53.4 | 1.94 / 0.80 | 0.19 / 1.17 | 7.6e-07 | 7.6e-07 | 1.76e-05 / 1.61e-05 (0.015) | heavier |
| wd-eva02-large-tagger-v3 | fp16 | 195.6 / 201.7 | 0.97 | 0.94-0.97 | 195.9 | 0.32 / 3.69 | 699 / 686 | 97 / 83 | 1.16 | 602 / 602 | 943 / 1184 | 196.4 / 202.0 | 0.83 / 1.15 | 0.33 / 3.80 | 0.001 | 0.001 | 0.0039 / 0.00337 (0.015) | heavier |
| wd-eva02-large-tagger-v3 | fp32 | 242.3 / 242.0 | 1.00 | - | - | 0.29 / 2.21 | 1392 / 1435 | ~189 / ~232 | 0.82 | 1203 / 1203 | 1643 / 2036 | 242.9 / 242.4 | 0.27 / 0.78 | 0.30 / 2.21 | 6.1e-07 | 6.1e-07 | 2.53e-06 / 2.39e-06 (0.015) |  |
| wd-vit-large-tagger-v3 | fp16 | 179.6 / 178.6 | 1.01 | 0.98-1.03 | 185.5 | 0.34 / 2.77 | 677 / 683 | 76 / 82 | 0.93 | 601 / 601 | 906 / 1104 | 180.2 / 181.8 | 0.80 / 1.14 | 0.35 / 3.10 | 0.0009 | 0.0009 | 0.00328 / 0.00337 (0.015) |  |
| wd-vit-large-tagger-v3 | fp32 | 214.1 / 204.3 | 1.05 | 0.94-1.11 | 205.4 | 0.43 / 2.80 | 1348 / 1401 | 146 / 199 | 0.74 | 1202 / 1202 | 1578 / 1816 | 215.7 / 208.7 | 1.60 / 1.18 | 0.43 / 2.90 | 3.9e-07 | 3.9e-07 | 2.86e-06 / 3.39e-06 (0.015) |  |
| wd-vit-tagger-v3 | fp16 | 46.1 / 44.4 | 1.04 | 0.96-1.12 | 47.8 | 0.12 / 0.89 | 225 / 251 | 45 / 70 | 0.65 | 180 / 180 | 446 / 650 | 46.6 / 49.1 | 0.18 / 1.14 | 0.12 / 0.92 | 0.00088 | 0.00088 | 0.00264 / 0.00244 (0.015) |  |
| wd-vit-tagger-v3 | fp32 | 53.7 / 54.8 | 0.98 | 0.94-0.99 | 56.5 | 0.20 / 0.87 | 448 / 444 | 87 / 83 | 1.05 | 361 / 361 | 666 / 848 | 54.5 / 55.1 | 0.29 / 1.10 | 0.20 / 0.91 | 2.7e-07 | 2.7e-07 | 2.01e-06 / 2.13e-06 (0.015) | heavier |
| xfeat | fp32 | 17.5 / 30.7 | 0.57 | 0.46-0.68 | 20.4 | 0.08 / 1.38 | 199 / 165 | 182 / 162 | 1.12 | 3 / 3 | 818 / 614 | 27.3 / 166.0 | 0.07 / 0.04 | 0.11 / 1.53 | - | 0.94 (0.33 % off) | 0 r1.000 / 0 r1.000 (0.0001) | heavier |
| xfeat-lighterglue | fp32 | 17.3 / 29.6 | 0.58 | 0.53-0.64 | 21.0 | 0.06 / 1.35 | 204 / 165 | 182 / 162 | 1.12 | 7 / 3 | 1621 / 1007 | 129.8 / 534.2 | 0.10 / 0.06 | 0.25 / 1.92 | - | 0.94 (0.36 % off) | 0 r1.000 / 0 r1.000 (0.0001) | heavier |
| yolox-l | fp16 | 50.9 / 61.1 | 0.83 | 0.82-0.84 | 52.1 | 0.08 / 1.62 | 293 / 280 | 185 / 177 | 1.05 | 103 / 103 | 936 / 954 | 56.2 / 66.6 | 0.52 / 0.15 | 0.70 / 1.72 | 0.7 (66 % off) | 0 | - | heavier |
| yolox-l | fp32 | 61.7 / 82.0 | 0.75 | 0.66-0.82 | 69.2 | 0.15 / 1.64 | 581 / 505 | 370 / 298 | 1.24 | 207 / 207 | 1336 / 1295 | 66.7 / 85.9 | 0.30 / 0.15 | 0.73 / 1.74 | 0.7 (66 % off) | 0 | - | heavier |
| yolox-m | fp16 | 30.0 / 37.5 | 0.80 | - | - | 0.06 / 1.35 | 170 / 207 | ~121 / ~158 | 0.77 | 49 / 49 | 533 / 773 | 34.8 / 41.8 | 0.04 / 0.11 | 0.07 / 1.35 | 0.74 (67 % off) | empty | - |  |
| yolox-m | fp32 | 33.1 / 42.5 | 0.78 | - | - | 0.06 / 1.38 | 368 / 441 | ~271 / ~344 | 0.79 | 97 / 97 | 806 / 1130 | 38.3 / 46.9 | 0.04 / 0.15 | 0.07 / 1.39 | 0.74 (67 % off) | empty | - |  |
| yolox-s | fp16 | 12.4 / 17.0 | 0.73 | - | - | 0.05 / 1.25 | 83 / 178 | ~66 / ~160 | 0.41 | 17 / 17 | 486 / 604 | 18.0 / 21.5 | 0.04 / 0.08 | 0.54 / 1.32 | 0.77 (66 % off) | 0 | - |  |
| yolox-s | fp32 | 14.9 / 20.7 | 0.72 | - | - | 0.05 / 1.20 | 159 / 214 | ~125 / ~180 | 0.70 | 34 / 34 | 594 / 697 | 20.1 / 25.8 | 0.04 / 0.08 | 0.55 / 1.27 | 0.77 (66 % off) | 0 | - |  |
| yolox-tiny | fp16 | 8.8 / 12.4 | 0.71 | - | - | 0.04 / 1.23 | 37 / 65 | ~27 / ~55 | 0.50 | 10 / 10 | 389 / 467 | 13.4 / 17.3 | 0.04 / 0.07 | 0.53 / 1.30 | 0 | 0 | - |  |
| yolox-tiny | fp32 | 8.9 / 13.7 | 0.65 | - | - | 0.05 / 1.23 | 72 / 134 | ~53 / ~114 | 0.46 | 20 / 20 | 429 / 533 | 13.2 / 16.0 | 0.04 / 0.08 | 0.55 / 1.30 | 0 | 0 | - |  |
| yolox-x | fp16 | 94.4 / 106.6 | 0.89 | - | - | 0.12 / 1.73 | 391 / 441 | ~201 / ~251 | 0.80 | 190 / 190 | 1080 / 1321 | 100.1 / 111.8 | 0.06 / 0.21 | 0.64 / 1.80 | 0.58 (65 % off) | 0 | - |  |
| yolox-x | fp32 | 112.0 / 138.3 | 0.81 | 0.79-0.84 | 117.3 | 0.35 / 1.88 | 829 / 810 | 446 / 432 | 1.03 | 378 / 378 | 2116 / 2050 | 119.0 / 141.5 | 0.89 / 0.22 | 0.37 / 1.93 | 0.58 (65 % off) | empty | - | heavier |
| yunet-n | fp16 | 9.3 / 11.1 | 0.84 | - | - | 0.04 / 1.00 | 207 / 223 | ~207 / ~223 | 0.93 | 0 / 0 | 579 / 713 | 14.0 / 15.1 | 0.04 / 0.10 | 0.05 / 1.01 | 0.0031 | empty | - |  |
| yunet-n | fp32 | 12.3 / 15.4 | 0.80 | 0.76-0.87 | 16.5 | 0.41 / 1.12 | 367 / 271 | 343 / 271 | 1.26 | 0 / 0 | 772 / 750 | 17.4 / 20.3 | 0.05 / 0.13 | 0.42 / 1.14 | 1.1e-06 | empty | - | heavier |
| yunet-s | fp16 | 8.5 / 10.1 | 0.83 | - | - | 0.04 / 1.00 | 175 / 211 | ~175 / ~211 | 0.83 | 0 / 0 | 546 / 703 | 13.2 / 14.2 | 0.05 / 0.09 | 0.05 / 1.01 | 0.0021 | empty | - |  |
| yunet-s | fp32 | 11.6 / 14.1 | 0.83 | 0.75-1.04 | 14.4 | 0.09 / 1.10 | 303 / 271 | 279 / 271 | 1.03 | 0 / 0 | 704 / 782 | 16.2 / 19.9 | 0.05 / 0.11 | 0.10 / 1.12 | 1.1e-06 | empty | - | heavier |

Not measured (one arm failed):

- resplat-base-re10k float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-base-re10k float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
