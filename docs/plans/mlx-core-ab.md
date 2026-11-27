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

<!-- generated by `tools/mlx_core_ab.py md`: everything below this line -->

## Counts

583 rows over 80 families; 579 measured in both arms.

- 242 rows flagged (95 slower, 159 heavier, 10 worse).
- 75 of them accepted as a peak trade; 167 miss the bar (64 beyond 10 %).
- Median hand / graph: warm 0.90, peak 0.92.
- Quality: 200 rows with a golden error in both arms, 0 without a golden, 10 hand arms `worse` than the graph's.

## Flagged beyond 10 %

Not accepted, hand / graph warm or peak above 1.10, worst first.

| ref | dtype | net warm ms hand / graph | x | net peak MB hand / graph | x | flag |
|---|---|---|---|---|---|---|
| lw-detr-large-coco | fp16 | 53.5 / 57.4 | 0.93 | 548 / 319 | 1.72 | heavier |
| selective-igev-eth3d-finetune | fp32 | 4788.7 / 6385.9 | 0.75 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-eth3d-train | fp32 | 4778.2 / 6294.8 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-kitti12 | fp32 | 4733.6 / 6386.6 | 0.74 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-kitti15 | fp32 | 4725.9 / 6386.9 | 0.74 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-middlebury-finetune | fp32 | 4760.9 / 6439.7 | 0.74 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-middlebury-train | fp32 | 4728.9 / 6368.5 | 0.74 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-sceneflow | fp32 | 4694.9 / 6337.5 | 0.74 | 3498 / 2057 | 1.70 | heavier |
| lw-detr-xlarge-coco | fp16 | 97.2 / 107.9 | 0.90 | 722 / 445 | 1.62 | heavier |
| sam3 | fp32 | 1896.0 / 1741.0 | 1.09 | 3972 / 2551 | 1.56 | slower heavier |
| igev-stereo-middlebury | fp32 | 4180.7 / 4608.6 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-eth3d | fp32 | 4165.2 / 4606.8 | 0.90 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-sceneflow | fp32 | 4184.8 / 4628.8 | 0.90 | 3122 / 2055 | 1.52 | heavier |
| segformer-b0-ade20k | fp32 | 23.9 / 22.3 | 1.07 | 221 / 149 | 1.48 | slower heavier |
| sam3 | fp16 | 1574.1 / 1439.5 | 1.09 | 2006 / 1385 | 1.45 | slower heavier |
| rf-detr-seg-2xl-coco | fp16 | 130.9 / 125.6 | 1.04 | 385 / 274 | 1.41 | slower heavier worse |
| segformer-b0-ade20k | fp16 | 20.0 / 17.3 | 1.16 | 158 / 114 | 1.39 | slower heavier |
| segnext-mscan-l-ade20k | fp16 | 138.7 / 126.9 | 1.09 | 421 / 305 | 1.38 | slower heavier |
| fcn-r101-d8-ade20k | fp16 | 326.3 / 354.0 | 0.92 | 877 / 638 | 1.38 | heavier |
| pspnet-r101-d8-ade20k | fp16 | 338.6 / 365.6 | 0.93 | 1042 / 768 | 1.36 | heavier |
| efficientvit-sam-xl1 | fp32 | 348.8 / 342.0 | 1.02 | 2197 / 1668 | 1.32 | heavier |
| lineart-coarse | fp32 | 110.1 / 117.4 | 0.94 | 1188 / 903 | 1.32 | heavier |
| sam2.1-hiera-tiny | fp32 | 130.7 / 127.7 | 1.02 | 572 / 436 | 1.31 | slower heavier |
| unik3d-vitl | fp16 | 889.6 / 897.5 | 0.99 | 1523 / 1166 | 1.31 | heavier |
| sam2.1-hiera-small | fp32 | 155.0 / 152.5 | 1.02 | 599 / 463 | 1.29 | heavier |
| totalsegmentator-total-3mm | fp16 | 310.8 / 433.0 | 0.72 | 2058 / 1609 | 1.28 | heavier |
| efficientvit-sam-xl0 | fp32 | 193.3 / 203.5 | 0.95 | 1311 / 1026 | 1.28 | heavier |
| rf-detr-keypoint-preview | fp32 | 73.7 / 74.9 | 0.98 | 316 / 248 | 1.28 | heavier |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp32 | 6.7 / 5.3 | 1.26 | 82 / 120 | 0.68 | slower |
| segnext-mscan-t-ade20k | fp16 | 32.2 / 26.0 | 1.24 | 118 / 176 | 0.67 | slower |
| segnext-mscan-s-ade20k | fp16 | 43.4 / 37.6 | 1.15 | 237 / 197 | 1.20 | slower heavier |
| raft-stereo-middlebury | fp16 | 3231.3 / 4128.3 | 0.78 | 1246 / 1042 | 1.20 | heavier worse |
| deeplabv3-r101-d8-ade20k | fp16 | 366.9 / 377.9 | 0.97 | 682 / 574 | 1.19 | heavier |
| raft-stereo-middlebury | fp32 | 4970.9 / 5321.8 | 0.93 | 2222 / 1872 | 1.19 | heavier |
| loma-b64 | fp32 | 1019.7 / 989.3 | 1.03 | 1164 / 988 | 1.18 | slower heavier |
| mobilenetv4_conv_medium.e500_r256_in1k | fp32 | 4.1 / 3.5 | 1.17 | 56 / 74 | 0.76 | slower |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 5.2 / 4.5 | 1.17 | 37 / 66 | 0.56 | slower |
| davit_tiny.msft_in1k | fp16 | 9.3 / 7.9 | 1.17 | 95 / 123 | 0.77 | slower |
| deeplabv3plus-r101-d8-ade20k | fp32 | 347.5 / 377.6 | 0.92 | 1404 / 1209 | 1.16 | heavier |
| rf-detr-l-coco | fp32 | 56.3 / 54.1 | 1.04 | 262 / 226 | 1.16 | slower heavier |
| loma-b128 | fp32 | 1195.5 / 1312.4 | 0.91 | 1234 / 1064 | 1.16 | heavier |
| sam2.1-hiera-large | fp16 | 555.8 / 562.2 | 0.99 | 849 / 734 | 1.16 | heavier |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.2 | 0.97 | 471 / 409 | 1.15 | heavier |
| totalsegmentator-total-3mm | fp32 | 415.4 / 547.4 | 0.76 | 2033 / 1769 | 1.15 | heavier |
| mobilenet_v3_large.tv2_in1k | fp32 | 3.7 / 3.2 | 1.15 | 34 / 53 | 0.64 | slower |
| deeplabv3plus-r101-d8-ade20k | fp16 | 268.6 / 290.9 | 0.92 | 773 / 674 | 1.15 | heavier |
| segformer-b2-clothes | fp16 | 27.4 / 24.5 | 1.12 | 181 / 158 | 1.15 | slower heavier |
| pe-core-b16-224 | fp32 | 17.3 / 17.6 | 0.98 | 496 / 436 | 1.14 | heavier |
| sam2.1-hiera-large | fp32 | 600.5 / 608.8 | 0.99 | 1495 / 1317 | 1.14 | heavier |
| igev-plusplus-eth3d | fp32 | 6918.7 / 7331.1 | 0.94 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-kitti12 | fp32 | 6907.6 / 7268.4 | 0.95 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-kitti15 | fp32 | 6913.8 / 7172.6 | 0.96 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-middlebury | fp32 | 6861.8 / 7266.2 | 0.94 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-sceneflow | fp32 | 6877.3 / 7236.8 | 0.95 | 4260 / 3786 | 1.13 | heavier |
| rtmw-l | fp16 | 101.4 / 108.8 | 0.93 | 393 / 350 | 1.12 | heavier |
| sam2.1-hiera-base-plus | fp16 | 232.8 / 245.5 | 0.95 | 535 / 478 | 1.12 | heavier |
| segnext-mscan-t-ade20k | fp32 | 36.5 / 32.6 | 1.12 | 227 / 225 | 1.01 | slower |
| mvsanywhere | fp16 | 414.7 / 372.0 | 1.11 | 767 / 767 | 1.00 | slower |
| rf-detr-seg-2xl-coco | fp32 | 163.5 / 164.4 | 0.99 | 482 / 433 | 1.11 | heavier |
| shvit_s1.in1k | fp16 | 3.3 / 3.0 | 1.11 | 16 / 20 | 0.80 | slower |
| segnext-mscan-b-ade20k | fp16 | 77.9 / 70.1 | 1.11 | 264 / 251 | 1.05 | slower heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 25.3 / 25.2 | 1.00 | 333 / 300 | 1.11 | heavier |
| davit_small.msft_in1k | fp16 | 15.5 / 14.1 | 1.11 | 135 / 178 | 0.76 | slower |
| sapiens-seg-0.3b-goliath | fp32 | 1399.4 / 1783.6 | 0.78 | 3631 / 3288 | 1.10 | heavier |

## Accepted: faster, peak trade

`heavier` only, warm at most 0.90 of the graph's, the end-to-end MLX peak (survey row) below the best MPS row and at most the ref's MLX float32 row (`accepted`).

| ref | dtype | net warm ms hand / graph | x | net peak MB hand / graph | x | e2e peak MB | best MPS MB | MLX fp32 MB |
|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | fp16 | 639.7 / 1224.1 | 0.52 | 2112 / 1914 | 1.10 | 2259 | 3754 | 3104 |
| aanet-sceneflow | fp16 | 443.1 / 1077.9 | 0.41 | 1062 / 900 | 1.18 | 1362 | 2594 | 2140 |
| aanet-sceneflow | fp32 | 636.4 / 1177.7 | 0.54 | 2072 / 855 | 2.42 | 2140 | 2594 | 2140 |
| deim-l-coco | fp32 | 46.7 / 54.4 | 0.86 | 483 / 393 | 1.23 | 972 | 1806 | 972 |
| deim-x-coco | fp32 | 80.3 / 90.3 | 0.89 | 740 / 710 | 1.04 | 1288 | 2207 | 1288 |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 81 / 74 | 1.11 | 416 | 704 | 476 |
| densenet121.tv_in1k | fp32 | 9.6 / 12.4 | 0.77 | 148 / 142 | 1.04 | 476 | 704 | 476 |
| dfine-l-coco | fp32 | 46.5 / 54.2 | 0.86 | 488 / 393 | 1.24 | 1024 | 1810 | 1024 |
| dfine-x-coco | fp32 | 80.4 / 90.3 | 0.89 | 740 / 710 | 1.04 | 1422 | 2211 | 1422 |
| dlnr-middlebury | fp16 | 1609.2 / 1927.6 | 0.83 | 1647 / 1209 | 1.36 | 2841 | 4149 | 3856 |
| dlnr-middlebury | fp32 | 2050.9 / 2774.3 | 0.74 | 2901 / 2261 | 1.28 | 3856 | 4149 | 3856 |
| dlnr-sceneflow | fp16 | 1614.9 / 1942.1 | 0.83 | 1647 / 1208 | 1.36 | 2844 | 4146 | 3849 |
| dlnr-sceneflow | fp32 | 2027.6 / 2769.5 | 0.73 | 2901 / 2261 | 1.28 | 3849 | 4146 | 3849 |
| dwpose-l | fp16 | 56.2 / 66.0 | 0.85 | 332 / 290 | 1.15 | 1144 | 1866 | 1397 |
| dwpose-l | fp32 | 62.2 / 79.0 | 0.79 | 587 / 454 | 1.29 | 1397 | 1866 | 1397 |
| dwpose-l-384 | fp16 | 109.3 / 132.1 | 0.83 | 409 / 376 | 1.09 | 1232 | 1872 | 1340 |
| dwpose-l-384 | fp32 | 122.6 / 163.0 | 0.75 | 765 / 518 | 1.48 | 1340 | 1872 | 1340 |
| dwpose-s | fp32 | 18.8 / 24.2 | 0.78 | 286 / 256 | 1.12 | 727 | 1543 | 727 |
| igev-plusplus-eth3d | fp16 | 5696.6 / 7121.3 | 0.80 | 2414 / 2199 | 1.10 | 2953 | 4971 | 4217 |
| igev-plusplus-kitti12 | fp16 | 5631.7 / 7096.5 | 0.79 | 2414 / 2189 | 1.10 | 3016 | 4967 | 3926 |
| igev-plusplus-kitti15 | fp16 | 5640.9 / 7267.1 | 0.78 | 2414 / 2199 | 1.10 | 2962 | 4967 | 3930 |
| igev-plusplus-middlebury | fp16 | 5695.1 / 7274.0 | 0.78 | 2414 / 2189 | 1.10 | 2920 | 4966 | 3936 |
| igev-plusplus-sceneflow | fp16 | 5654.4 / 7257.3 | 0.78 | 2414 / 2189 | 1.10 | 2949 | 4972 | 3981 |
| igev-stereo-eth3d | fp16 | 3183.2 / 3604.2 | 0.88 | 1686 / 1080 | 1.56 | 2551 | 3789 | 3531 |
| igev-stereo-kitti12 | fp16 | 3187.2 / 3651.0 | 0.87 | 1686 / 1080 | 1.56 | 2559 | 3791 | 3445 |
| igev-stereo-kitti12 | fp32 | 4138.5 / 4637.6 | 0.89 | 3122 / 2055 | 1.52 | 3445 | 3791 | 3445 |
| igev-stereo-kitti15 | fp16 | 3184.4 / 3614.1 | 0.88 | 1686 / 1010 | 1.67 | 2570 | 3791 | 3426 |
| igev-stereo-kitti15 | fp32 | 4165.1 / 4787.5 | 0.87 | 3122 / 2055 | 1.52 | 3426 | 3791 | 3426 |
| igev-stereo-middlebury | fp16 | 3192.1 / 3633.4 | 0.88 | 1686 / 1080 | 1.56 | 2555 | 3792 | 3425 |
| igev-stereo-sceneflow | fp16 | 3190.5 / 3621.4 | 0.88 | 1686 / 1010 | 1.67 | 2558 | 3793 | 3441 |
| lineart | fp32 | 99.0 / 117.6 | 0.84 | 1188 / 903 | 1.32 | 815 | 1361 | 815 |
| loftr-indoor | fp32 | 516.9 / 1759.6 | 0.29 | 2161 / 1314 | 1.64 | 2293 | 3613 | 2293 |
| loftr-outdoor | fp32 | 490.3 / 1366.8 | 0.36 | 2161 / 1314 | 1.64 | 2513 | 3614 | 2513 |
| nafnet-gopro | fp32 | 257.3 / 293.8 | 0.88 | 1169 / 839 | 1.39 | 1686 | 1715 | 1686 |
| nafnet-sidd | fp16 | 184.3 / 208.2 | 0.89 | 658 / 546 | 1.21 | 1252 | 1986 | 1735 |
| nafnet-sidd | fp32 | 233.4 / 269.8 | 0.87 | 1184 / 896 | 1.32 | 1735 | 1986 | 1735 |
| oneformer-swinl-ade20k | fp16 | 643.5 / 760.8 | 0.85 | 1364 / 1002 | 1.36 | 2137 | 5559 | 2917 |
| raft-stereo-eth3d | fp16 | 3267.0 / 4104.8 | 0.80 | 1246 / 1042 | 1.20 | 2719 | 4714 | 3704 |
| raft-stereo-sceneflow | fp16 | 3238.1 / 4084.1 | 0.79 | 1246 / 1042 | 1.20 | 2741 | 4695 | 3808 |
| rtdetrv2-l-coco | fp32 | 55.3 / 63.4 | 0.87 | 455 / 424 | 1.07 | 1027 | 1823 | 1027 |
| rtdetrv2-ms-coco | fp32 | 44.5 / 51.5 | 0.86 | 412 / 391 | 1.05 | 958 | 1821 | 958 |
| rtdetrv2-x-coco | fp32 | 92.5 / 103.2 | 0.90 | 736 / 679 | 1.08 | 1416 | 2017 | 1416 |
| rtmpose-l | fp16 | 48.6 / 60.8 | 0.80 | 310 / 267 | 1.16 | 1054 | 1841 | 1325 |
| rtmpose-l | fp32 | 54.5 / 73.8 | 0.74 | 565 / 431 | 1.31 | 1325 | 1841 | 1325 |
| rtmpose-l-384 | fp16 | 95.3 / 118.9 | 0.80 | 387 / 354 | 1.09 | 1159 | 1844 | 1313 |
| rtmpose-l-384 | fp32 | 107.9 / 149.1 | 0.72 | 743 / 496 | 1.50 | 1313 | 1844 | 1313 |
| rtmpose-l-halpe26 | fp16 | 48.6 / 61.0 | 0.80 | 311 / 269 | 1.16 | 1077 | 1847 | 1326 |
| rtmpose-l-halpe26 | fp32 | 54.6 / 73.9 | 0.74 | 566 / 433 | 1.31 | 1326 | 1847 | 1326 |
| rtmpose-l-halpe26-384 | fp16 | 95.6 / 118.8 | 0.80 | 389 / 356 | 1.09 | 1164 | 1851 | 1353 |
| rtmpose-l-halpe26-384 | fp32 | 108.4 / 149.3 | 0.73 | 745 / 510 | 1.46 | 1353 | 1851 | 1353 |
| rtmpose-s-face6 | fp16 | 20.6 / 24.8 | 0.83 | 204 / 179 | 1.14 | 709 | 1518 | 823 |
| rtmpose-s-face6 | fp32 | 21.9 / 29.0 | 0.76 | 329 / 248 | 1.33 | 823 | 1518 | 823 |
| rtmpose-s-halpe26 | fp32 | 14.6 / 22.1 | 0.66 | 251 / 246 | 1.02 | 673 | 1500 | 673 |
| rtmw-l | fp32 | 111.7 / 125.7 | 0.89 | 678 / 544 | 1.25 | 1550 | 2040 | 1550 |
| rtmw-l-384 | fp16 | 206.4 / 235.8 | 0.88 | 470 / 438 | 1.07 | 1343 | 2076 | 1644 |
| rtmw-l-384 | fp32 | 224.6 / 272.5 | 0.82 | 856 / 634 | 1.35 | 1644 | 2076 | 1644 |
| rtmw-x | fp16 | 176.2 / 206.2 | 0.85 | 560 / 474 | 1.18 | 1605 | 2266 | 2033 |
| rtmw-x | fp32 | 203.7 / 240.3 | 0.85 | 968 / 830 | 1.17 | 2033 | 2266 | 2033 |
| rtmw-x-384 | fp32 | 419.5 / 490.0 | 0.86 | 1174 / 1093 | 1.07 | 2216 | 2268 | 2216 |
| sam-hq-vit-b | fp16 | 303.6 / 393.4 | 0.77 | 622 / 539 | 1.15 | 1374 | 4374 | 1712 |
| sam-hq-vit-l | fp16 | 825.1 / 939.4 | 0.88 | 1119 / 982 | 1.14 | 1884 | 6205 | 2571 |
| sam-vit-b | fp16 | 299.6 / 383.8 | 0.78 | 613 / 531 | 1.15 | 1069 | 3762 | 1412 |
| sam-vit-l | fp16 | 815.8 / 931.0 | 0.88 | 1107 / 972 | 1.14 | 1639 | 5193 | 2342 |
| selective-igev-eth3d-finetune | fp16 | 3693.7 / 4643.9 | 0.80 | 1873 / 1102 | 1.70 | 2776 | 3854 | 3947 |
| selective-igev-eth3d-train | fp16 | 3652.2 / 4644.1 | 0.79 | 1873 / 1102 | 1.70 | 2777 | 3862 | 3935 |
| selective-igev-kitti12 | fp16 | 3643.3 / 4640.5 | 0.79 | 1873 / 1102 | 1.70 | 2764 | 3853 | 3938 |
| selective-igev-kitti15 | fp16 | 3672.6 / 4688.7 | 0.78 | 1873 / 1102 | 1.70 | 2774 | 3853 | 3934 |
| selective-igev-middlebury-finetune | fp16 | 3655.1 / 4663.3 | 0.78 | 1873 / 1102 | 1.70 | 2932 | 3852 | 3952 |
| selective-igev-middlebury-train | fp16 | 3651.5 / 4657.9 | 0.78 | 1873 / 1102 | 1.70 | 2779 | 3854 | 3941 |
| selective-igev-sceneflow | fp16 | 3643.8 / 4680.4 | 0.78 | 1873 / 1102 | 1.70 | 2781 | 3907 | 3938 |
| xfeat | fp32 | 17.5 / 32.0 | 0.55 | 199 / 165 | 1.21 | 642 | 1471 | 642 |
| xfeat-lighterglue | fp32 | 17.7 / 34.8 | 0.51 | 204 / 165 | 1.24 | 816 | 1499 | 816 |
| yolox-l | fp16 | 51.0 / 61.3 | 0.83 | 293 / 280 | 1.05 | 1021 | 1829 | 1307 |
| yolox-l | fp32 | 60.3 / 75.8 | 0.80 | 581 / 505 | 1.15 | 1307 | 1829 | 1307 |
| yolox-x | fp32 | 108.8 / 128.4 | 0.85 | 829 / 802 | 1.03 | 1675 | 2036 | 1675 |

## Per family

Flagged rows / rows (accepted in brackets), the worst hand / graph ratio of the family (warm, peak).

| family | flagged (accepted) | slower | heavier | worse | worst warm x | worst peak x |
|---|---|---|---|---|---|---|
| aanet | 3 (3) / 4 | 0 | 3 | 0 | 0.54 | 2.42 |
| alexnet | 0 (0) / 2 | 0 | 0 | 0 | 0.99 | 0.96 |
| anyline | 0 (0) / 2 | 0 | 0 | 0 | 0.64 | 0.65 |
| clip | 13 (0) / 20 | 13 | 0 | 0 | 1.04 | 0.98 |
| coatnet | 0 (0) / 20 | 0 | 0 | 0 | 1.01 | 0.90 |
| coex | 0 (0) / 2 | 0 | 0 | 0 | 0.50 | 0.53 |
| convnext | 2 (0) / 24 | 0 | 1 | 1 | 1.01 | 1.04 |
| convnextv2 | 0 (0) / 2 | 0 | 0 | 0 | 0.98 | 0.91 |
| davit | 5 (0) / 6 | 4 | 1 | 0 | 1.17 | 1.04 |
| deit | 2 (0) / 2 | 2 | 0 | 0 | 1.04 | 0.87 |
| densenet | 2 (2) / 2 | 0 | 2 | 0 | 0.77 | 1.11 |
| depth_anything_3 | 8 (0) / 10 | 8 | 0 | 0 | 1.06 | 1.00 |
| depth_pro | 0 (0) / 2 | 0 | 0 | 0 | 0.96 | 0.98 |
| dfine | 4 (4) / 20 | 0 | 4 | 0 | 0.92 | 1.24 |
| dlnr | 4 (4) / 4 | 0 | 4 | 0 | 0.83 | 1.36 |
| efficientformer | 3 (0) / 6 | 0 | 0 | 3 | 0.86 | 0.83 |
| efficientformerv2 | 1 (0) / 8 | 1 | 0 | 0 | 1.17 | 0.72 |
| efficientnet | 0 (0) / 4 | 0 | 0 | 0 | 0.83 | 0.77 |
| efficientnetv2 | 0 (0) / 2 | 0 | 0 | 0 | 0.87 | 0.60 |
| efficientvit | 0 (0) / 2 | 0 | 0 | 0 | 0.79 | 0.74 |
| efficientvit_sam | 2 (0) / 5 | 0 | 2 | 0 | 1.02 | 1.32 |
| eva02 | 1 (0) / 2 | 0 | 1 | 0 | 1.00 | 1.03 |
| fastvit | 0 (0) / 6 | 0 | 0 | 0 | 0.99 | 0.62 |
| flexglue | 0 (0) / 2 | 0 | 0 | 0 | 0.98 | 0.80 |
| focalnet | 1 (0) / 12 | 1 | 0 | 0 | 1.04 | 0.85 |
| hgnetv2 | 0 (0) / 14 | 0 | 0 | 0 | 0.95 | 0.81 |
| igev_plusplus | 10 (5) / 10 | 0 | 10 | 0 | 0.96 | 1.13 |
| igev_stereo | 10 (7) / 10 | 0 | 10 | 0 | 0.91 | 1.67 |
| inception_v3 | 0 (0) / 2 | 0 | 0 | 0 | 0.76 | 0.46 |
| lightglue | 2 (0) / 4 | 1 | 2 | 0 | 1.03 | 1.05 |
| lineart | 2 (1) / 4 | 0 | 2 | 0 | 0.94 | 1.32 |
| loftr | 2 (2) / 4 | 0 | 2 | 0 | 0.36 | 1.64 |
| loma | 2 (0) / 12 | 1 | 2 | 0 | 1.03 | 1.18 |
| lwdetr | 3 (0) / 10 | 0 | 3 | 0 | 1.00 | 1.72 |
| mask2former | 0 (0) / 2 | 0 | 0 | 0 | 0.91 | 0.97 |
| mmseg | 8 (0) / 10 | 1 | 7 | 0 | 1.03 | 1.38 |
| mobilenetv1 | 1 (0) / 2 | 0 | 0 | 1 | 1.01 | 0.72 |
| mobilenetv2 | 0 (0) / 2 | 0 | 0 | 0 | 0.97 | 0.96 |
| mobilenetv3 | 1 (0) / 2 | 1 | 0 | 0 | 1.15 | 0.64 |
| mobilenetv4 | 1 (0) / 2 | 1 | 0 | 0 | 1.17 | 0.76 |
| mvsanywhere | 2 (0) / 2 | 2 | 0 | 0 | 1.11 | 1.00 |
| nafnet | 3 (3) / 4 | 0 | 3 | 0 | 0.89 | 1.39 |
| nsfw_vit | 1 (0) / 2 | 1 | 0 | 0 | 1.04 | 0.87 |
| oneformer | 1 (1) / 2 | 0 | 1 | 0 | 0.85 | 1.36 |
| perception_encoder | 6 (0) / 20 | 1 | 6 | 0 | 1.03 | 1.15 |
| raft_stereo | 6 (2) / 6 | 0 | 6 | 1 | 0.94 | 1.20 |
| rcnn | 0 (0) / 4 | 0 | 0 | 0 | 0.85 | 0.97 |
| regnet | 0 (0) / 2 | 0 | 0 | 0 | 0.94 | 0.71 |
| repvit | 0 (0) / 20 | 0 | 0 | 0 | 0.79 | 0.62 |
| resnet | 0 (0) / 6 | 0 | 0 | 0 | 0.91 | 0.87 |
| rfdetr | 15 (0) / 20 | 12 | 5 | 3 | 1.08 | 1.41 |
| rfdetr_keypoint | 2 (0) / 2 | 1 | 2 | 0 | 1.03 | 1.28 |
| rtdetrv2 | 3 (3) / 10 | 0 | 3 | 0 | 0.92 | 1.08 |
| rtmdet | 0 (0) / 14 | 0 | 0 | 0 | 0.92 | 0.97 |
| rtmo | 0 (0) / 8 | 0 | 0 | 0 | 0.84 | 1.02 |
| rtmpose | 25 (22) / 56 | 0 | 25 | 0 | 0.98 | 1.50 |
| sam | 3 (2) / 6 | 0 | 3 | 0 | 0.90 | 1.15 |
| sam2 | 8 (0) / 8 | 1 | 8 | 0 | 1.02 | 1.31 |
| sam3 | 2 (0) / 2 | 2 | 2 | 0 | 1.09 | 1.56 |
| sam_hq | 3 (2) / 6 | 0 | 3 | 0 | 0.90 | 1.15 |
| sapiens | 2 (0) / 4 | 0 | 2 | 0 | 0.88 | 1.10 |
| segformer | 7 (0) / 8 | 6 | 6 | 0 | 1.16 | 1.48 |
| segnext | 7 (0) / 8 | 6 | 4 | 0 | 1.24 | 1.38 |
| selective_igev | 14 (7) / 14 | 0 | 14 | 0 | 0.80 | 1.70 |
| shufflenetv2 | 2 (0) / 2 | 2 | 0 | 1 | 1.08 | 0.90 |
| shvit | 8 (0) / 8 | 8 | 0 | 0 | 1.11 | 0.86 |
| siglip | 2 (0) / 2 | 2 | 0 | 0 | 1.02 | 0.98 |
| siglip2 | 11 (0) / 20 | 11 | 0 | 0 | 1.05 | 1.01 |
| squeezenet | 0 (0) / 2 | 0 | 0 | 0 | 0.86 | 0.86 |
| starnet | 0 (0) / 8 | 0 | 0 | 0 | 0.87 | 0.62 |
| swin | 1 (0) / 2 | 0 | 1 | 0 | 0.99 | 1.03 |
| tinyvit | 3 (0) / 10 | 2 | 1 | 0 | 1.26 | 1.11 |
| totalsegmentator | 2 (0) / 4 | 0 | 2 | 0 | 0.76 | 1.28 |
| unik3d | 1 (0) / 2 | 0 | 1 | 0 | 1.02 | 1.31 |
| vgg | 0 (0) / 2 | 0 | 0 | 0 | 0.97 | 0.91 |
| vit | 1 (0) / 2 | 1 | 0 | 0 | 1.05 | 0.87 |
| wd_tagger | 3 (0) / 8 | 3 | 0 | 0 | 1.03 | 1.02 |
| xfeat | 2 (2) / 2 | 0 | 2 | 0 | 0.55 | 1.24 |
| yolox | 3 (3) / 10 | 0 | 3 | 0 | 0.89 | 1.15 |

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

| ref | dtype | net warm ms | x | net first s | net peak MB | x | resident MB | e2e warm ms | load s | e2e first s | net rel | result rel | golden mean (bound) | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | fp16 | 639.7 / 1224.1 | 0.52 | 0.71 / 6.15 | 2112 / 1914 | 1.10 | 33 / 33 | 656.3 / 1183.7 | 0.16 / 0.18 | 0.73 / 6.16 | 0.0057 | 0.012 (0.00014 % off) | - | heavier |
| aanet-plus-sceneflow | fp32 | 850.1 / 1572.5 | 0.54 | 0.99 / 6.25 | 3207 / 3767 | 0.85 | 33 / 49 | 860.7 / 1576.4 | 0.14 / 0.12 | 1.00 / 6.26 | 3.5e-05 | 7.2e-05 | - |  |
| aanet-sceneflow | fp16 | 443.1 / 1077.9 | 0.41 | 0.48 / 5.17 | 1062 / 900 | 1.18 | 16 / 23 | 449.0 / 1090.5 | 0.13 / 0.11 | 0.50 / 5.18 | 0.019 (0.00027 % off) | 0.0079 | - | heavier |
| aanet-sceneflow | fp32 | 636.4 / 1177.7 | 0.54 | 0.66 / 5.16 | 2072 / 855 | 2.42 | 16 / 30 | 645.5 / 1183.5 | 0.12 / 0.11 | 0.67 / 5.17 | 0.0002 | 7.3e-05 | - | heavier |
| alexnet.tv_in1k | fp16 | 2.5 / 2.5 | 0.99 | 0.02 / 0.38 | 119 / 125 | 0.96 | 117 / 117 | 2.8 / 2.9 | 0.64 / 0.49 | 0.02 / 0.38 | 0 | 0 | - |  |
| alexnet.tv_in1k | fp32 | 3.3 / 3.4 | 0.97 | 0.03 / 0.38 | 238 / 248 | 0.96 | 233 / 233 | 3.6 / 3.8 | 0.62 / 0.49 | 0.03 / 0.38 | 0 | 0 | - |  |
| aliked-flexglue | fp16 | 932.4 / 954.6 | 0.98 | 4.20 / 4.84 | 800 / 1063 | 0.75 | 263 / 261 | 969.2 / 1000.3 | 0.10 / 0.10 | 4.26 / 4.90 | - | empty (4 keys differ) | - |  |
| aliked-flexglue | fp32 | 861.4 / 896.2 | 0.96 | 4.08 / 4.73 | 1057 / 1319 | 0.80 | 520 / 518 | 897.7 / 942.5 | 0.10 / 0.10 | 4.14 / 4.78 | - | 2.2e-06 | - |  |
| aliked-lightglue | fp16 | 89.6 / 164.7 | 0.54 | 0.14 / 1.63 | 714 / 1324 | 0.54 | 49 / 2 | 88.1 / 308.3 | 0.08 / 0.05 | 0.16 / 1.82 | - | 1.7e-05 | - |  |
| aliked-lightglue | fp32 | 88.3 / 167.4 | 0.53 | 0.13 / 1.63 | 714 / 1324 | 0.54 | 49 / 2 | 88.4 / 309.8 | 0.08 / 0.05 | 0.16 / 1.79 | - | 1.7e-05 | - |  |
| anyline | fp16 | 49.9 / 77.9 | 0.64 | 0.08 / 1.15 | 394 / 861 | 0.46 | 0 / 0 | 65.4 / 223.5 | 0.02 / 0.02 | 0.11 / 1.34 | - | 0.45 (0.27 % off) | - |  |
| anyline | fp32 | 66.4 / 122.9 | 0.54 | 0.10 / 1.10 | 743 / 1148 | 0.65 | 0 / 0 | 83.1 / 270.1 | 0.03 / 0.02 | 0.13 / 1.28 | - | 0.16 (0.00019 % off) | - |  |
| clip-b16-laion2b | fp16 | 14.2 / 14.0 | 1.02 | 0.04 / 0.67 | 186 / 231 | 0.80 | 164 / 164 | 14.5 / 14.2 | 0.85 / 0.71 | 0.04 / 0.68 | 0.0043 | 0.0043 | - |  |
| clip-b16-laion2b | fp32 | 16.3 / 16.1 | 1.01 | 0.06 / 0.66 | 362 / 417 | 0.87 | 327 / 327 | 16.6 / 16.5 | 0.78 / 0.71 | 0.06 / 0.66 | 6e-06 | 6e-06 | - |  |
| clip-b16-openai | fp16 | 14.2 / 13.9 | 1.03 | 0.03 / 0.67 | 185 / 231 | 0.80 | 164 / 164 | 14.4 / 14.2 | 0.86 / 0.71 | 0.04 / 0.68 | 0.019 (0.00066 % off) | 0.019 (0.00066 % off) | - | slower |
| clip-b16-openai | fp32 | 16.4 / 16.1 | 1.02 | 0.06 / 0.68 | 362 / 404 | 0.90 | 327 / 327 | 16.6 / 16.4 | 0.79 / 0.72 | 0.06 / 0.68 | 2.4e-05 | 2.4e-05 | - |  |
| clip-h14-378-dfn5b | fp16 | 107.4 / 104.8 | 1.03 | 0.13 / 1.59 | 1241 / 1277 | 0.97 | 1203 / 1203 | 107.4 / 104.9 | 2.21 / 0.73 | 0.13 / 1.59 | 0.064 (0.025 % off) | 0.064 (0.025 % off) | - | slower |
| clip-h14-378-dfn5b | fp32 | 127.4 / 125.0 | 1.02 | 0.69 / 1.66 | 2469 / 2524 | 0.98 | 2407 / 2407 | 127.6 / 125.4 | 2.11 / 0.74 | 0.69 / 1.66 | 1.8e-05 | 1.8e-05 | - |  |
| clip-h14-dfn5b | fp16 | 107.3 / 104.7 | 1.03 | 0.12 / 1.57 | 1241 / 1277 | 0.97 | 1203 / 1203 | 107.4 / 105.0 | 3.29 / 0.73 | 0.12 / 1.57 | 0.032 (0.034 % off) | 0.032 (0.034 % off) | - | slower |
| clip-h14-dfn5b | fp32 | 127.4 / 125.0 | 1.02 | 0.25 / 1.66 | 2469 / 2524 | 0.98 | 2407 / 2407 | 127.8 / 125.3 | 1.59 / 0.73 | 0.25 / 1.66 | 3.6e-05 | 3.6e-05 | - |  |
| clip-h14-laion2b | fp16 | 107.5 / 104.9 | 1.02 | 0.12 / 1.55 | 1241 / 1277 | 0.97 | 1203 / 1203 | 107.8 / 105.2 | 1.82 / 0.74 | 0.12 / 1.56 | 0.017 (0.0021 % off) | 0.017 (0.0021 % off) | - | slower |
| clip-h14-laion2b | fp32 | 127.4 / 124.8 | 1.02 | 0.80 / 1.61 | 2469 / 2524 | 0.98 | 2407 / 2407 | 127.8 / 125.1 | 1.61 / 0.73 | 0.81 / 1.62 | 8.6e-06 | 8.6e-06 | - | slower |
| clip-l14-336-openai | fp16 | 55.0 / 53.9 | 1.02 | 0.07 / 1.15 | 608 / 658 | 0.92 | 578 / 578 | 55.3 / 54.2 | 1.34 / 0.71 | 0.07 / 1.15 | 0.022 (0.0023 % off) | 0.022 (0.0023 % off) | - | slower |
| clip-l14-336-openai | fp32 | 63.9 / 62.4 | 1.02 | 0.10 / 1.15 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.6 | 0.86 / 0.70 | 0.10 / 1.15 | 1.5e-05 | 1.5e-05 | - | slower |
| clip-l14-datacompxl | fp16 | 55.1 / 54.0 | 1.02 | 0.07 / 1.10 | 608 / 658 | 0.92 | 578 / 578 | 55.3 / 54.2 | 0.98 / 0.71 | 0.07 / 1.11 | 0.0063 | 0.0063 | - |  |
| clip-l14-datacompxl | fp32 | 64.0 / 62.3 | 1.03 | 0.10 / 1.14 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.7 | 0.86 / 0.71 | 0.10 / 1.14 | 1e-05 | 1e-05 | - | slower |
| clip-l14-dfn2b | fp16 | 56.1 / 54.1 | 1.04 | 0.07 / 1.12 | 618 / 658 | 0.94 | 578 / 578 | 56.3 / 54.2 | 1.00 / 0.72 | 0.08 / 1.12 | 0.05 (0.18 % off) | 0.05 (0.18 % off) | - | slower |
| clip-l14-dfn2b | fp32 | 63.9 / 62.3 | 1.03 | 0.10 / 1.14 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.6 | 0.86 / 0.72 | 0.10 / 1.14 | 7.5e-05 | 7.5e-05 | - | slower |
| clip-l14-laion2b | fp16 | 55.0 / 54.0 | 1.02 | 0.08 / 1.11 | 608 / 658 | 0.92 | 578 / 578 | 55.4 / 54.3 | 0.99 / 0.71 | 0.08 / 1.11 | 0.013 (0.00076 % off) | 0.013 (0.00076 % off) | - |  |
| clip-l14-laion2b | fp32 | 64.0 / 62.3 | 1.03 | 0.10 / 1.11 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.6 | 0.88 / 0.73 | 0.10 / 1.11 | 2.6e-05 | 2.6e-05 | - | slower |
| clip-l14-openai | fp16 | 55.1 / 53.9 | 1.02 | 0.08 / 1.13 | 608 / 658 | 0.92 | 578 / 578 | 55.3 / 54.2 | 0.99 / 0.72 | 0.08 / 1.13 | 0.0046 | 0.0046 | - | slower |
| clip-l14-openai | fp32 | 63.9 / 62.1 | 1.03 | 0.10 / 1.13 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.3 / 62.4 | 0.85 / 0.73 | 0.11 / 1.14 | 4.3e-06 | 4.3e-06 | - | slower |
| coatnet_0_rw_224.sw_in1k | fp16 | 7.7 / 7.8 | 0.98 | 0.04 / 0.86 | 100 / 162 | 0.62 | 58 / 58 | 8.2 / 8.4 | 0.83 / 0.73 | 0.04 / 0.87 | 0.0012 | 0.0012 | 0.00104 / 0.0013 (0.0024) |  |
| coatnet_0_rw_224.sw_in1k | fp32 | 8.5 / 8.9 | 0.96 | 0.04 / 0.81 | 186 / 272 | 0.68 | 116 / 117 | 9.1 / 9.5 | 0.78 / 0.74 | 0.05 / 0.81 | 7.9e-07 | 7.9e-07 | 4.08e-06 / 4.06e-06 (0.0024) |  |
| coatnet_1_rw_224.sw_in1k | fp16 | 12.7 / 12.8 | 0.99 | 0.05 / 1.31 | 145 / 248 | 0.59 | 92 / 92 | 13.2 / 13.4 | 0.87 / 0.71 | 0.05 / 1.31 | 0.00052 | 0.00052 | 0.000584 / 0.000565 (0.0024) |  |
| coatnet_1_rw_224.sw_in1k | fp32 | 14.1 / 15.3 | 0.92 | 0.05 / 1.26 | 251 / 340 | 0.74 | 183 / 184 | 14.8 / 15.9 | 0.83 / 0.72 | 0.05 / 1.26 | 3.8e-07 | 3.8e-07 | 3.59e-06 / 3.63e-06 (0.0024) |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.1 / 19.6 | 0.87 | 0.05 / 1.42 | 230 / 343 | 0.67 | 157 / 157 | 17.8 / 20.5 | 0.92 / 0.76 | 0.06 / 1.43 | 0.0027 | 0.0027 | 0.00145 / 0.00164 (0.0024) |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.5 / 24.2 | 0.81 | 0.09 / 1.26 | 424 / 492 | 0.86 | 314 / 314 | 20.3 / 24.7 | 0.96 / 0.74 | 0.09 / 1.26 | 2.9e-06 | 2.9e-06 | 7.66e-06 / 8.45e-06 (0.0024) |  |
| coatnet_nano_rw_224.sw_in1k | fp16 | 5.9 / 7.8 | 0.76 | 0.04 / 1.00 | 71 / 128 | 0.56 | 32 / 32 | 6.5 / 9.0 | 0.82 / 0.78 | 0.04 / 1.00 | 0.0041 | 0.0041 | 0.00111 / 0.000939 (0.0024) |  |
| coatnet_nano_rw_224.sw_in1k | fp32 | 8.2 / 9.2 | 0.90 | 0.11 / 0.95 | 118 / 205 | 0.57 | 65 / 65 | 8.8 / 10.1 | 0.85 / 0.78 | 0.11 / 0.96 | 5.6e-07 | 5.6e-07 | 2.08e-06 / 2.13e-06 (0.0024) |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp16 | 12.4 / 12.9 | 0.96 | 0.05 / 1.52 | 138 / 245 | 0.56 | 92 / 92 | 13.0 / 13.6 | 0.89 / 0.74 | 0.05 / 1.52 | 0.0051 | 0.0051 | 0.0012 / 0.00144 (0.0024) |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp32 | 13.7 / 15.4 | 0.89 | 0.06 / 1.44 | 275 / 340 | 0.81 | 183 / 184 | 14.4 / 16.0 | 0.87 / 0.73 | 0.07 / 1.45 | 3.6e-07 | 3.6e-07 | 3.01e-06 / 3.03e-06 (0.0024) |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp16 | 12.3 / 12.2 | 1.01 | 0.07 / 1.61 | 153 / 238 | 0.64 | 92 / 92 | 12.9 / 12.8 | 0.89 / 0.73 | 0.07 / 1.61 | 0.001 | 0.001 | 0.00145 / 0.00168 (0.0024) |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp32 | 13.7 / 14.8 | 0.93 | 0.05 / 1.51 | 244 / 350 | 0.70 | 183 / 184 | 14.2 / 15.5 | 0.89 / 0.76 | 0.06 / 1.51 | 3.9e-07 | 3.9e-07 | 3.39e-06 / 3.25e-06 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.7 / 19.9 | 0.89 | 0.05 / 1.69 | 230 / 343 | 0.67 | 157 / 157 | 18.0 / 20.6 | 0.98 / 0.76 | 0.06 / 1.70 | 0.0041 | 0.0041 | 0.00127 / 0.00102 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.4 / 24.3 | 0.80 | 0.06 / 1.54 | 424 / 491 | 0.86 | 314 / 314 | 20.0 / 25.0 | 0.91 / 0.77 | 0.06 / 1.54 | 6.7e-07 | 6.7e-07 | 4.87e-06 / 4.81e-06 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp16 | 17.3 / 19.9 | 0.87 | 0.07 / 1.65 | 240 / 343 | 0.70 | 157 / 157 | 18.4 / 20.5 | 1.01 / 0.73 | 0.07 / 1.65 | 0.0015 | 0.0015 | 0.000714 / 0.00131 (0.0024) |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp32 | 19.4 / 24.3 | 0.80 | 0.07 / 1.67 | 424 / 491 | 0.86 | 314 / 314 | 20.1 / 25.0 | 0.92 / 0.79 | 0.07 / 1.68 | 4e-07 | 4e-07 | 2.7e-06 / 2.82e-06 (0.0024) |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp16 | 42.6 / 48.6 | 0.88 | 0.08 / 1.72 | 390 / 475 | 0.82 | 279 / 280 | 43.4 / 48.9 | 1.03 / 0.78 | 0.08 / 1.72 | 0.0017 | 0.0017 | 0.00116 / 0.00114 (0.0024) |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp32 | 49.3 / 61.4 | 0.80 | 0.10 / 1.75 | 740 / 823 | 0.90 | 558 / 559 | 49.8 / 61.2 | 0.99 / 0.76 | 0.10 / 1.76 | 4.8e-07 | 4.8e-07 | 6.73e-06 / 6.68e-06 (0.0024) |  |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp16 | 6.0 / 6.5 | 0.92 | 0.04 / 1.02 | 76 / 136 | 0.56 | 32 / 32 | 6.5 / 7.1 | 0.83 / 0.73 | 0.04 / 1.03 | 0.0014 | 0.0014 | 0.000958 / 0.00107 (0.0024) |  |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp32 | 6.4 / 7.4 | 0.87 | 0.04 / 0.98 | 130 / 227 | 0.57 | 65 / 65 | 6.9 / 8.2 | 0.79 / 0.75 | 0.04 / 0.98 | 2.7e-07 | 2.7e-07 | 2.41e-06 / 2.61e-06 (0.0024) |  |
| coex-sceneflow | fp16 | 197.2 / 409.9 | 0.48 | 0.24 / 2.21 | 550 / 1669 | 0.33 | 11 / 6 | 202.2 / 423.0 | 0.08 / 0.07 | 0.25 / 2.22 | 0.31 (0.1 % off) | 0.034 (0.0055 % off) | - |  |
| coex-sceneflow | fp32 | 258.0 / 518.0 | 0.50 | 0.29 / 2.27 | 1020 / 1943 | 0.53 | 11 / 11 | 264.1 / 515.7 | 0.08 / 0.06 | 0.30 / 2.28 | 2.8e-06 | 3e-06 | - |  |
| convnext_base.fb_in1k | fp16 | 14.7 / 15.1 | 0.97 | 0.06 / 1.08 | 196 / 278 | 0.71 | 170 / 169 | 15.1 / 16.6 | 0.96 / 0.72 | 0.06 / 1.08 | 0.00068 | 0.00068 | 0.000653 / 0.0006 (0.0033) |  |
| convnext_base.fb_in1k | fp32 | 16.6 / 17.4 | 0.95 | 0.07 / 1.02 | 383 / 448 | 0.85 | 338 / 338 | 17.2 / 18.0 | 0.91 / 0.71 | 0.07 / 1.02 | 1.7e-07 | 1.7e-07 | 3.64e-06 / 3.59e-06 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k | fp16 | 14.6 / 15.0 | 0.97 | 0.05 / 1.05 | 196 / 278 | 0.71 | 170 / 169 | 15.3 / 15.8 | 0.91 / 0.72 | 0.05 / 1.05 | 0.0012 | 0.0012 | 0.000844 / 0.000932 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k | fp32 | 16.2 / 17.5 | 0.93 | 0.08 / 1.01 | 383 / 448 | 0.85 | 338 / 338 | 16.9 / 18.2 | 0.87 / 0.71 | 0.08 / 1.01 | 5.1e-07 | 5.1e-07 | 5.16e-06 / 5.15e-06 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k_384 | fp16 | 30.0 / 31.3 | 0.96 | 0.06 / 1.08 | 232 / 310 | 0.75 | 170 / 169 | 31.0 / 32.1 | 0.93 / 0.74 | 0.06 / 1.08 | 0.0031 | 0.0031 | 0.00106 / 0.000972 (0.0033) |  |
| convnext_base.fb_in22k_ft_in1k_384 | fp32 | 35.2 / 37.4 | 0.94 | 0.09 / 1.03 | 441 / 464 | 0.95 | 338 / 338 | 36.1 / 38.3 | 0.88 / 0.73 | 0.09 / 1.03 | 5e-07 | 5e-07 | 5.64e-06 / 5.63e-06 (0.0033) |  |
| convnext_large.fb_in1k | fp16 | 26.9 / 28.0 | 0.96 | 0.06 / 1.08 | 411 / 478 | 0.86 | 378 / 378 | 27.5 / 28.7 | 1.13 / 0.71 | 0.06 / 1.08 | 0.00062 | 0.00062 | 0.000625 / 0.000639 (0.0033) |  |
| convnext_large.fb_in1k | fp32 | 31.7 / 33.5 | 0.95 | 0.09 / 1.07 | 828 / 866 | 0.96 | 755 / 755 | 32.3 / 34.2 | 1.07 / 0.71 | 0.09 / 1.08 | 4.6e-07 | 4.6e-07 | 5.14e-06 / 5.17e-06 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k | fp16 | 26.9 / 28.0 | 0.96 | 0.07 / 1.08 | 414 / 478 | 0.87 | 378 / 378 | 27.5 / 28.7 | 1.14 / 0.74 | 0.07 / 1.08 | 0.0015 | 0.0015 | 0.000805 / 0.000907 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k | fp32 | 31.6 / 33.3 | 0.95 | 0.09 / 1.08 | 826 / 866 | 0.95 | 755 / 755 | 32.3 / 33.9 | 1.08 / 0.73 | 0.09 / 1.08 | 6.6e-07 | 6.6e-07 | 9.16e-06 / 9.12e-06 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp16 | 59.1 / 60.6 | 0.98 | 0.09 / 1.11 | 466 / 495 | 0.94 | 378 / 378 | 59.9 / 61.6 | 1.11 / 0.71 | 0.09 / 1.12 | 0.00096 | 0.00096 | 0.00107 / 0.00114 (0.0033) |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp32 | 67.4 / 71.4 | 0.94 | 0.11 / 1.12 | 917 / 885 | 1.04 | 755 / 755 | 68.3 / 71.8 | 1.06 / 0.81 | 0.12 / 1.12 | 4e-07 | 4e-07 | 9.03e-06 / 9.01e-06 (0.0033) | heavier |
| convnext_small.fb_in1k | fp16 | 10.5 / 10.7 | 0.98 | 0.05 / 1.05 | 118 / 185 | 0.64 | 96 / 96 | 11.0 / 11.3 | 0.86 / 0.72 | 0.05 / 1.05 | 0.00074 | 0.00074 | 0.000578 / 0.000627 (0.0033) |  |
| convnext_small.fb_in1k | fp32 | 11.8 / 12.5 | 0.94 | 0.06 / 1.00 | 225 / 302 | 0.75 | 192 / 192 | 12.4 / 13.1 | 0.81 / 0.71 | 0.06 / 1.00 | 3.6e-07 | 3.6e-07 | 3.01e-06 / 2.97e-06 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k | fp16 | 10.5 / 10.7 | 0.98 | 0.04 / 1.03 | 114 / 185 | 0.62 | 96 / 96 | 11.0 / 11.2 | 0.89 / 0.70 | 0.05 / 1.03 | 0.0014 | 0.0014 | 0.000852 / 0.000909 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k | fp32 | 11.8 / 12.4 | 0.95 | 0.06 / 0.99 | 225 / 302 | 0.75 | 192 / 192 | 12.4 / 13.1 | 0.82 / 0.70 | 0.06 / 1.00 | 5e-07 | 5e-07 | 5e-06 / 5.01e-06 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k_384 | fp16 | 21.0 / 21.2 | 0.99 | 0.05 / 1.05 | 143 / 233 | 0.61 | 96 / 96 | 21.6 / 22.0 | 0.91 / 0.71 | 0.06 / 1.05 | 0.00084 | 0.00084 | 0.00128 / 0.0014 (0.0033) |  |
| convnext_small.fb_in22k_ft_in1k_384 | fp32 | 23.8 / 24.9 | 0.95 | 0.07 / 1.01 | 293 / 308 | 0.95 | 192 / 192 | 24.5 / 25.6 | 0.81 / 0.75 | 0.07 / 1.02 | 3.4e-07 | 3.4e-07 | 5.74e-06 / 5.72e-06 (0.0033) |  |
| convnext_tiny.fb_in1k | fp16 | 6.0 / 6.0 | 1.00 | 0.04 / 0.68 | 71 / 108 | 0.65 | 55 / 55 | 6.5 / 6.6 | 0.84 / 0.73 | 0.04 / 0.69 | 0.00079 | 0.00079 | 0.000808 / 0.000935 (0.0033) |  |
| convnext_tiny.fb_in1k | fp32 | 6.7 / 6.8 | 1.00 | 0.06 / 0.66 | 144 / 206 | 0.70 | 109 / 109 | 7.2 / 7.4 | 0.78 / 0.70 | 0.06 / 0.66 | 6.3e-07 | 6.3e-07 | 4.47e-06 / 4.5e-06 (0.0033) |  |
| convnext_tiny.fb_in22k_ft_in1k | fp16 | 6.0 / 6.0 | 1.01 | 0.04 / 0.70 | 70 / 108 | 0.65 | 55 / 55 | 6.5 / 6.5 | 0.83 / 0.71 | 0.04 / 0.70 | 0.0012 | 0.0012 | 0.0023 / 0.00192 (0.0033) | worse |
| convnext_tiny.fb_in22k_ft_in1k | fp32 | 6.7 / 6.8 | 0.99 | 0.05 / 0.66 | 144 / 206 | 0.70 | 109 / 109 | 7.2 / 7.3 | 0.78 / 0.70 | 0.05 / 0.67 | 5.3e-07 | 5.3e-07 | 4.78e-06 / 4.79e-06 (0.0033) |  |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp16 | 12.1 / 12.0 | 1.01 | 0.04 / 0.69 | 97 / 161 | 0.60 | 55 / 55 | 13.0 / 12.8 | 0.85 / 0.69 | 0.04 / 0.69 | 0.0013 | 0.0013 | 0.00222 / 0.00224 (0.0033) |  |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp32 | 14.0 / 14.3 | 0.98 | 0.06 / 0.68 | 190 / 221 | 0.86 | 109 / 109 | 14.7 / 15.0 | 0.77 / 0.70 | 0.07 / 0.68 | 9.3e-07 | 9.3e-07 | 6.58e-06 / 6.75e-06 (0.0033) |  |
| convnextv2_base.fcmae_ft_in22k_in1k | fp16 | 17.9 / 18.1 | 0.98 | 0.20 / 1.69 | 244 / 318 | 0.77 | 170 / 170 | 18.4 / 19.1 | 1.37 / 0.84 | 0.20 / 1.69 | 0.0011 | 0.0011 | 0.000912 / 0.00137 (0.02) |  |
| convnextv2_base.fcmae_ft_in22k_in1k | fp32 | 20.9 / 21.4 | 0.98 | 0.42 / 1.66 | 415 / 456 | 0.91 | 339 / 339 | 21.3 / 26.5 | 1.73 / 0.84 | 0.43 / 1.66 | 4.1e-07 | 4.1e-07 | 4.9e-06 / 4.8e-06 (0.02) |  |
| da3-giant-1.1 | fp16 | 598.3 / 580.7 | 1.03 | 0.66 / 5.45 | 2729 / 2889 | 0.94 | 2321 / 2364 | 605.9 / 587.7 | 4.34 / 0.86 | 0.68 / 5.47 | 0.0057 | 0.0056 | - | slower |
| da3-giant-1.1 | fp32 | 732.9 / 721.4 | 1.02 | 0.93 / 11.83 | 5376 / 5546 | 0.97 | 4571 / 4638 | 743.8 / 732.9 | 5.76 / 0.76 | 0.96 / 11.86 | 4.6e-06 | 4.2e-06 | - |  |
| da3-large-1.1 | fp16 | 214.8 / 209.0 | 1.03 | 0.25 / 3.26 | 1096 / 1096 | 1.00 | 690 / 750 | 222.7 / 216.3 | 0.29 / 0.48 | 0.27 / 3.27 | 0.0036 | 0.0032 | - | slower |
| da3-large-1.1 | fp32 | 259.1 / 249.3 | 1.04 | 0.32 / 3.25 | 2148 / 2321 | 0.93 | 1346 / 1411 | 266.7 / 256.6 | 0.12 / 0.50 | 0.34 / 3.27 | 2.1e-06 | 2.1e-06 | - | slower |
| da3metric-large | fp16 | 207.0 / 197.0 | 1.05 | 0.23 / 1.91 | 1071 / 1102 | 0.97 | 636 / 639 | 224.1 / 208.5 | 0.29 / 0.50 | 0.26 / 1.93 | 0.018 (0.017 % off) | 0.017 (0.0054 % off) | - | slower |
| da3metric-large | fp32 | 248.9 / 235.7 | 1.06 | 0.29 / 1.92 | 2082 / 2195 | 0.95 | 1273 / 1278 | 266.3 / 254.2 | 0.12 / 0.48 | 0.31 / 1.95 | 2.7e-05 | 2.7e-05 | - | slower |
| da3mono-large | fp16 | 206.8 / 196.7 | 1.05 | 0.23 / 1.91 | 1071 / 1102 | 0.97 | 636 / 639 | 224.3 / 207.3 | 0.27 / 0.49 | 0.26 / 1.93 | 0.023 (0.0027 % off) | 1 (0.0013 % off) | - | slower |
| da3mono-large | fp32 | 248.8 / 235.7 | 1.06 | 0.29 / 1.92 | 2082 / 2195 | 0.95 | 1273 / 1278 | 265.6 / 252.3 | 0.12 / 0.48 | 0.31 / 1.95 | 1.3e-05 | 1.3e-05 | - | slower |
| da3nested-giant-large-1.1 | fp16 | 802.3 / 776.6 | 1.03 | 0.86 / 7.92 | 3392 / 3528 | 0.96 | 2958 / 3003 | 829.6 / 791.7 | 5.69 / 0.80 | 0.90 / 7.96 | 0.0029 | 0.017 (0.0041 % off) | - | slower |
| da3nested-giant-large-1.1 | fp32 | 982.5 / 974.7 | 1.01 | 1.41 / 11.21 | 6652 / 6833 | 0.97 | 5844 / 5916 | 1010.0 / 1003.7 | 7.42 / 0.79 | 1.48 / 11.26 | 3.7e-06 | 2.7e-05 | - |  |
| davit_base.msft_in1k | fp16 | 19.3 / 18.5 | 1.04 | 0.05 / 2.16 | 224 / 248 | 0.90 | 168 / 168 | 19.9 / 19.5 | 1.04 / 0.84 | 0.06 / 2.16 | 0.0008 | 0.0008 | 0.000779 / 0.000801 (0.0026) | slower |
| davit_base.msft_in1k | fp32 | 22.4 / 22.5 | 1.00 | 0.09 / 2.16 | 434 / 417 | 1.04 | 336 / 336 | 23.7 / 24.0 | 0.98 / 0.97 | 0.09 / 2.16 | 3.3e-07 | 3.3e-07 | 6.87e-06 / 6.92e-06 (0.0026) | heavier |
| davit_small.msft_in1k | fp16 | 15.5 / 14.1 | 1.11 | 0.11 / 2.14 | 135 / 178 | 0.76 | 95 / 95 | 16.5 / 14.9 | 0.99 / 0.84 | 0.11 / 2.15 | 0.0015 | 0.0015 | 0.00115 / 0.00111 (0.0026) | slower |
| davit_small.msft_in1k | fp32 | 17.7 / 17.3 | 1.03 | 0.11 / 2.07 | 262 / 272 | 0.96 | 190 / 190 | 18.8 / 22.8 | 0.92 / 0.85 | 0.11 / 2.07 | 4e-07 | 4e-07 | 6.59e-06 / 6.61e-06 (0.0026) | slower |
| davit_tiny.msft_in1k | fp16 | 9.3 / 7.9 | 1.17 | 0.04 / 1.31 | 95 / 123 | 0.77 | 54 / 54 | 9.9 / 8.4 | 0.90 / 0.80 | 0.04 / 1.31 | 0.00087 | 0.00087 | 0.0013 / 0.00136 (0.0026) | slower |
| davit_tiny.msft_in1k | fp32 | 10.3 / 10.3 | 1.00 | 0.15 / 1.29 | 180 / 179 | 1.01 | 108 / 108 | 11.0 / 10.6 | 0.86 / 0.82 | 0.15 / 1.29 | 5.3e-07 | 5.3e-07 | 3.74e-06 / 3.78e-06 (0.0026) |  |
| deeplabv3-r101-d8-ade20k | fp16 | 366.9 / 377.9 | 0.97 | 0.41 / 1.69 | 682 / 574 | 1.19 | 162 / 162 | 402.1 / 410.0 | 0.22 / 0.14 | 0.45 / 1.73 | 0.0007 | 0 | - | heavier |
| deeplabv3-r101-d8-ade20k | fp32 | 540.1 / 526.5 | 1.03 | 0.57 / 1.80 | 1224 / 1312 | 0.93 | 323 / 323 | 577.0 / 563.2 | 0.22 / 0.14 | 0.61 / 1.85 | 3.4e-07 | 0 | - | slower |
| deeplabv3plus-r101-d8-ade20k | fp16 | 268.6 / 290.9 | 0.92 | 0.32 / 1.62 | 773 / 674 | 1.15 | 115 / 115 | 307.3 / 327.2 | 0.19 / 0.13 | 0.37 / 1.67 | 0.0011 | 0 | - | heavier |
| deeplabv3plus-r101-d8-ade20k | fp32 | 347.5 / 377.6 | 0.92 | 0.40 / 1.67 | 1404 / 1209 | 1.16 | 230 / 230 | 387.4 / 422.0 | 0.19 / 0.13 | 0.45 / 1.72 | 4.2e-07 | 0 | - | heavier |
| deim-l-coco | fp16 | 40.8 / 46.2 | 0.88 | 0.07 / 2.72 | 249 / 274 | 0.91 | 63 / 62 | 48.3 / 54.1 | 0.34 / 0.16 | 0.08 / 2.73 | 0.89 (48 % off) | 0.0013 | - |  |
| deim-l-coco | fp32 | 46.7 / 54.4 | 0.86 | 0.09 / 2.46 | 483 / 393 | 1.23 | 118 / 118 | 54.5 / 62.7 | 0.22 / 0.16 | 0.10 / 2.47 | 6.9e-06 | 1.1e-06 | - | heavier |
| deim-m-coco | fp16 | 27.4 / 33.6 | 0.82 | 0.09 / 2.44 | 197 / 232 | 0.85 | 40 / 40 | 35.5 / 42.5 | 0.27 / 0.13 | 0.10 / 2.45 | 0.89 (56 % off) | 0.0036 | - |  |
| deim-m-coco | fp32 | 31.7 / 40.2 | 0.79 | 0.09 / 2.26 | 381 / 376 | 1.01 | 74 / 74 | 39.4 / 48.3 | 0.18 / 0.13 | 0.10 / 2.28 | 0.49 (0.95 % off) | 2.6e-06 | - |  |
| deim-n-coco | fp16 | 9.1 / 11.9 | 0.77 | 0.06 / 1.94 | 58 / 139 | 0.41 | 8 / 8 | 17.2 / 19.9 | 0.18 / 0.09 | 0.07 / 1.96 | 0.96 (59 % off) | 0.71 (2.5 % off) | - |  |
| deim-n-coco | fp32 | 10.3 / 13.5 | 0.76 | 0.07 / 1.83 | 106 / 175 | 0.60 | 15 / 15 | 18.4 / 20.8 | 0.11 / 0.09 | 0.08 / 1.85 | 1.7e-05 | 1.5e-06 | - |  |
| deim-s-coco | fp16 | 16.3 / 20.2 | 0.80 | 0.07 / 2.10 | 129 / 193 | 0.67 | 22 / 22 | 24.1 / 28.1 | 0.21 / 0.10 | 0.09 / 2.11 | 0.97 (49 % off) | empty (3 keys differ) | - |  |
| deim-s-coco | fp32 | 18.4 / 23.9 | 0.77 | 0.08 / 2.02 | 191 / 206 | 0.93 | 40 / 40 | 26.1 / 31.9 | 0.14 / 0.10 | 0.09 / 2.04 | 4.2e-06 | 1.7e-06 | - |  |
| deim-x-coco | fp16 | 69.8 / 75.9 | 0.92 | 0.10 / 3.00 | 376 / 380 | 0.99 | 122 / 122 | 78.2 / 84.2 | 0.49 / 0.22 | 0.11 / 3.01 | 0.84 (42 % off) | 0.0068 | - |  |
| deim-x-coco | fp32 | 80.3 / 90.3 | 0.89 | 0.12 / 2.77 | 740 / 710 | 1.04 | 236 / 236 | 88.4 / 98.6 | 0.32 / 0.22 | 0.13 / 2.78 | 0.37 (0.59 % off) | 7.7e-07 | - | heavier |
| deit_base_patch16_224.fb_in1k | fp16 | 14.2 / 13.6 | 1.04 | 0.07 / 0.66 | 188 / 232 | 0.81 | 165 / 165 | 14.8 / 14.3 | 0.05 / 0.71 | 0.07 / 0.66 | 0.00093 | 0.00093 | 0.00164 / 0.00122 (0.02) | slower |
| deit_base_patch16_224.fb_in1k | fp32 | 16.3 / 16.0 | 1.02 | 0.06 / 0.64 | 366 / 420 | 0.87 | 330 / 330 | 16.8 / 16.6 | 0.06 / 0.71 | 0.06 / 0.65 | 6.8e-07 | 6.8e-07 | 7.18e-06 / 7.13e-06 (0.02) | slower |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 0.03 / 1.38 | 81 / 74 | 1.11 | 16 / 16 | 8.6 / 11.3 | 0.68 / 0.58 | 0.03 / 1.39 | 0.0017 | 0.0017 | - | heavier |
| densenet121.tv_in1k | fp32 | 9.6 / 12.4 | 0.77 | 0.03 / 1.31 | 148 / 142 | 1.04 | 30 / 31 | 10.1 / 12.9 | 0.64 / 0.58 | 0.03 / 1.31 | 5.4e-07 | 5.4e-07 | - | heavier |
| depth-pro | fp16 | 4168.0 / 4341.7 | 0.96 | 4.27 / 9.37 | 4508 / 4609 | 0.98 | 1809 / 1815 | 4170.1 / 4666.0 | 3.88 / 3.42 | 4.28 / 9.38 | 0.0017 | 0.0011 | - |  |
| depth-pro | fp32 | 5497.0 / 6831.3 | 0.80 | 5.96 / 10.71 | 9014 / 9191 | 0.98 | 3616 / 3630 | 5552.0 / 6573.6 | 4.63 / 3.41 | 5.99 / 10.74 | 1.5e-06 | 1.3e-06 | - |  |
| dfine-l-coco | fp16 | 40.9 / 46.5 | 0.88 | 0.07 / 2.62 | 249 / 274 | 0.91 | 63 / 62 | 48.1 / 54.1 | 0.29 / 0.10 | 0.08 / 2.63 | 0.97 (42 % off) | 0.97 (23 % off) | - |  |
| dfine-l-coco | fp32 | 46.5 / 54.2 | 0.86 | 0.09 / 2.44 | 488 / 393 | 1.24 | 118 / 118 | 54.4 / 62.6 | 0.17 / 0.10 | 0.10 / 2.45 | 1e-05 | 3.4e-06 | - | heavier |
| dfine-m-coco | fp16 | 27.2 / 34.0 | 0.80 | 0.08 / 2.39 | 197 / 232 | 0.85 | 40 / 40 | 34.7 / 41.8 | 0.22 / 0.08 | 0.09 / 2.40 | 0.95 (61 % off) | 0.96 (46 % off) | - |  |
| dfine-m-coco | fp32 | 31.8 / 40.2 | 0.79 | 0.09 / 2.25 | 381 / 376 | 1.01 | 74 / 74 | 39.6 / 48.8 | 0.13 / 0.08 | 0.10 / 2.26 | 0.63 (0.6 % off) | 1.4e-05 | - |  |
| dfine-n-coco | fp16 | 9.0 / 11.8 | 0.76 | 0.06 / 1.93 | 58 / 139 | 0.41 | 8 / 8 | 16.9 / 19.6 | 0.14 / 0.06 | 0.08 / 1.95 | 0.95 (53 % off) | 0.97 (30 % off) | - |  |
| dfine-n-coco | fp32 | 10.3 / 13.7 | 0.75 | 0.07 / 1.85 | 106 / 175 | 0.60 | 15 / 15 | 17.6 / 22.1 | 0.08 / 0.06 | 0.08 / 1.86 | 6e-06 | 9e-07 | - |  |
| dfine-s-coco | fp16 | 16.2 / 20.2 | 0.80 | 0.07 / 2.10 | 129 / 193 | 0.67 | 22 / 22 | 24.1 / 28.4 | 0.17 / 0.07 | 0.09 / 2.12 | 0.96 (57 % off) | 1 (38 % off) | - |  |
| dfine-s-coco | fp32 | 18.4 / 23.7 | 0.77 | 0.08 / 1.99 | 191 / 206 | 0.93 | 40 / 40 | 26.4 / 31.7 | 0.10 / 0.06 | 0.09 / 2.00 | 8.8e-06 | 2e-05 | - |  |
| dfine-x-coco | fp16 | 69.8 / 76.0 | 0.92 | 0.10 / 2.92 | 376 / 380 | 0.99 | 122 / 122 | 77.8 / 83.7 | 0.39 / 0.15 | 0.11 / 2.94 | 1.1 (52 % off) | empty (3 keys differ) | - |  |
| dfine-x-coco | fp32 | 80.4 / 90.3 | 0.89 | 0.12 / 2.75 | 740 / 710 | 1.04 | 236 / 236 | 88.5 / 98.0 | 0.24 / 0.15 | 0.13 / 2.77 | 0.00025 | 4.9e-06 | - | heavier |
| disk-lightglue | fp16 | 478.8 / 466.5 | 1.03 | 1.51 / 1.57 | 990 / 965 | 1.03 | 50 / 5 | 591.3 / 618.7 | 0.07 / 0.05 | 1.68 / 1.75 | 0 | 7.4e-06 | - | slower heavier |
| disk-lightglue | fp32 | 558.2 / 549.9 | 1.02 | 1.62 / 1.68 | 967 / 920 | 1.05 | 50 / 4 | 675.9 / 702.0 | 0.07 / 0.05 | 1.82 / 1.86 | 0 | 7.7e-06 | - | heavier |
| dlnr-middlebury | fp16 | 1609.2 / 1927.6 | 0.83 | 1.64 / 7.23 | 1647 / 1209 | 1.36 | 142 / 182 | 1658.0 / 1935.5 | 0.18 / 0.09 | 1.65 / 7.24 | 0.33 (0.15 % off) | 0.33 (0.027 % off) | - | heavier |
| dlnr-middlebury | fp32 | 2050.9 / 2774.3 | 0.74 | 2.06 / 8.03 | 2901 / 2261 | 1.28 | 229 / 314 | 2053.7 / 2758.0 | 0.18 / 0.09 | 2.07 / 8.04 | 0.02 (0.00091 % off) | 0.016 (0.00054 % off) | - | heavier |
| dlnr-sceneflow | fp16 | 1614.9 / 1942.1 | 0.83 | 1.64 / 7.23 | 1647 / 1208 | 1.36 | 142 / 181 | 1667.1 / 1933.3 | 0.18 / 0.09 | 1.65 / 7.24 | 0.019 (0.0068 % off) | 0.017 (0.0013 % off) | - | heavier |
| dlnr-sceneflow | fp32 | 2027.6 / 2769.5 | 0.73 | 2.07 / 7.81 | 2901 / 2261 | 1.28 | 229 / 314 | 2054.8 / 2760.3 | 0.18 / 0.09 | 2.08 / 7.82 | 1.2e-05 | 1.2e-05 | - | heavier |
| dwpose-l | fp16 | 56.2 / 66.0 | 0.85 | 0.07 / 1.32 | 332 / 290 | 1.15 | 81 / 78 | 65.3 / 74.5 | 0.20 / 0.08 | 0.08 / 1.34 | - | 0.44 (0.099 % off) | - | heavier |
| dwpose-l | fp32 | 62.2 / 79.0 | 0.79 | 0.09 / 1.30 | 587 / 454 | 1.29 | 128 / 128 | 71.2 / 87.5 | 0.15 / 0.08 | 0.10 / 1.31 | - | 3.3e-06 | - | heavier |
| dwpose-l-384 | fp16 | 109.3 / 132.1 | 0.83 | 0.13 / 1.47 | 409 / 376 | 1.09 | 81 / 78 | 127.1 / 147.4 | 0.21 / 0.08 | 0.15 / 1.49 | - | 0.57 (0.21 % off) | - | heavier |
| dwpose-l-384 | fp32 | 122.6 / 163.0 | 0.75 | 0.15 / 1.42 | 765 / 518 | 1.48 | 128 / 128 | 140.4 / 178.8 | 0.15 / 0.09 | 0.17 / 1.44 | - | 4e-06 | - | heavier |
| dwpose-m | fp16 | 40.1 / 42.6 | 0.94 | 0.05 / 1.21 | 243 / 237 | 1.03 | 47 / 45 | 48.7 / 50.4 | 0.14 / 0.06 | 0.06 / 1.22 | - | 0.42 (0.18 % off) | - | heavier |
| dwpose-m | fp32 | 44.2 / 50.3 | 0.88 | 0.07 / 1.21 | 400 / 429 | 0.93 | 69 / 69 | 53.0 / 58.3 | 0.10 / 0.06 | 0.08 / 1.22 | - | 2.1e-06 | - |  |
| dwpose-s | fp16 | 17.8 / 21.2 | 0.84 | 0.03 / 1.11 | 200 / 201 | 0.99 | 24 / 24 | 26.6 / 28.9 | 0.10 / 0.04 | 0.04 / 1.12 | - | 0.38 (0.017 % off) | - |  |
| dwpose-s | fp32 | 18.8 / 24.2 | 0.78 | 0.04 / 1.10 | 286 / 256 | 1.12 | 32 / 32 | 27.3 / 32.2 | 0.07 / 0.04 | 0.06 / 1.11 | - | 2.5e-06 | - | heavier |
| dwpose-t | fp16 | 16.1 / 17.9 | 0.90 | 0.03 / 1.06 | 161 / 198 | 0.81 | 17 / 17 | 24.8 / 25.9 | 0.09 / 0.04 | 0.04 / 1.08 | - | 0.57 (0.23 % off) | - |  |
| dwpose-t | fp32 | 17.1 / 20.7 | 0.83 | 0.04 / 1.05 | 224 / 283 | 0.79 | 21 / 21 | 25.8 / 28.5 | 0.06 / 0.04 | 0.06 / 1.07 | - | 3.1e-06 | - |  |
| efficientformer_l1.snap_dist_in1k | fp16 | 3.1 / 3.8 | 0.81 | 0.04 / 0.74 | 33 / 59 | 0.56 | 23 / 24 | 3.8 / 4.2 | 0.78 / 0.72 | 0.04 / 0.74 | 0.0056 | 0.0056 | 0.0034 / 0.00278 (0.0046) | worse |
| efficientformer_l1.snap_dist_in1k | fp32 | 3.4 / 4.0 | 0.86 | 0.05 / 0.71 | 64 / 96 | 0.66 | 45 / 47 | 3.9 / 4.5 | 0.76 / 0.72 | 0.05 / 0.72 | 1.3e-06 | 1.3e-06 | 1.18e-05 / 1.22e-05 (0.0046) |  |
| efficientformer_l3.snap_dist_in1k | fp16 | 5.8 / 7.3 | 0.79 | 0.04 / 1.22 | 73 / 131 | 0.56 | 59 / 60 | 6.3 / 7.9 | 0.82 / 0.73 | 0.04 / 1.22 | 0.0021 | 0.0021 | 0.00415 / 0.00135 (0.0046) | worse |
| efficientformer_l3.snap_dist_in1k | fp32 | 6.6 / 8.0 | 0.82 | 0.06 / 1.18 | 145 / 218 | 0.66 | 118 / 120 | 7.2 / 8.8 | 0.79 / 0.72 | 0.06 / 1.18 | 7.9e-07 | 7.9e-07 | 6.83e-06 / 6.76e-06 (0.0046) |  |
| efficientformer_l7.snap_dist_in1k | fp16 | 12.0 / 14.0 | 0.85 | 0.05 / 1.61 | 181 / 288 | 0.63 | 156 / 157 | 12.5 / 14.7 | 0.96 / 0.73 | 0.05 / 1.61 | 0.006 | 0.006 | 0.00244 / 0.00189 (0.0046) | worse |
| efficientformer_l7.snap_dist_in1k | fp32 | 13.6 / 15.9 | 0.85 | 0.06 / 1.58 | 368 / 442 | 0.83 | 311 / 314 | 13.8 / 16.9 | 0.90 / 0.74 | 0.07 / 1.58 | 6.9e-07 | 6.9e-07 | 4.53e-06 / 4.66e-06 (0.0046) |  |
| efficientformerv2_l.snap_dist_in1k | fp16 | 8.6 / 9.2 | 0.93 | 0.05 / 2.33 | 69 / 100 | 0.69 | 50 / 51 | 9.0 / 9.8 | 0.92 / 0.75 | 0.05 / 2.33 | 0.0027 | 0.0027 | - |  |
| efficientformerv2_l.snap_dist_in1k | fp32 | 9.0 / 10.2 | 0.88 | 0.06 / 2.27 | 130 / 189 | 0.69 | 100 / 102 | 9.7 / 10.8 | 0.83 / 0.77 | 0.07 / 2.27 | 9.3e-07 | 9.3e-07 | - |  |
| efficientformerv2_s0.snap_dist_in1k | fp16 | 2.9 / 3.0 | 0.96 | 0.04 / 1.25 | 16 / 25 | 0.63 | 7 / 8 | 3.4 / 3.5 | 0.78 / 0.73 | 0.04 / 1.25 | 0.0043 | 0.0043 | - |  |
| efficientformerv2_s0.snap_dist_in1k | fp32 | 3.2 / 4.4 | 0.74 | 0.05 / 1.20 | 27 / 45 | 0.61 | 14 / 14 | 3.7 / 4.3 | 0.73 / 0.71 | 0.05 / 1.20 | 1.5e-06 | 1.5e-06 | - |  |
| efficientformerv2_s1.snap_dist_in1k | fp16 | 3.5 / 5.1 | 0.69 | 0.04 / 1.41 | 20 / 37 | 0.56 | 12 / 13 | 4.0 / 5.8 | 0.82 / 0.71 | 0.04 / 1.41 | 0.0056 | 0.0056 | - |  |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 5.2 / 4.5 | 1.17 | 0.05 / 1.35 | 37 / 66 | 0.56 | 24 / 24 | 5.5 / 4.9 | 0.73 / 0.71 | 0.06 / 1.36 | 1.8e-06 | 1.8e-06 | - | slower |
| efficientformerv2_s2.snap_dist_in1k | fp16 | 5.6 / 6.1 | 0.93 | 0.04 / 1.78 | 40 / 56 | 0.72 | 25 / 25 | 6.1 / 6.7 | 0.84 / 0.71 | 0.04 / 1.78 | 0.0034 | 0.0034 | - |  |
| efficientformerv2_s2.snap_dist_in1k | fp32 | 6.0 / 8.4 | 0.71 | 0.06 / 1.68 | 73 / 112 | 0.65 | 48 / 50 | 6.5 / 8.7 | 0.77 / 0.71 | 0.06 / 1.68 | 1.3e-06 | 1.3e-06 | - |  |
| efficientnet_b0.tv_in1k | fp16 | 3.2 / 4.2 | 0.75 | 0.02 / 0.85 | 29 / 63 | 0.46 | 11 / 10 | 3.7 / 4.8 | 0.58 / 0.50 | 0.02 / 0.91 | 0.0013 | 0.0013 | - |  |
| efficientnet_b0.tv_in1k | fp32 | 3.3 / 4.5 | 0.75 | 0.04 / 0.83 | 45 / 94 | 0.48 | 21 / 21 | 3.8 / 4.9 | 0.53 / 0.53 | 0.04 / 0.89 | 9.5e-07 | 9.5e-07 | - |  |
| efficientnet_b4.tv_in1k | fp16 | 12.7 / 15.4 | 0.83 | 0.05 / 1.45 | 126 / 229 | 0.55 | 38 / 38 | 13.4 / 16.2 | 0.72 / 0.59 | 0.05 / 1.46 | 0.00084 | 0.00084 | - |  |
| efficientnet_b4.tv_in1k | fp32 | 13.9 / 18.0 | 0.77 | 0.05 / 1.41 | 196 / 256 | 0.77 | 75 / 75 | 14.5 / 18.9 | 0.63 / 0.59 | 0.05 / 1.41 | 5.1e-07 | 5.1e-07 | - |  |
| efficientnet_v2_s.tv_in1k | fp16 | 13.9 / 16.0 | 0.87 | 0.03 / 1.51 | 103 / 234 | 0.44 | 41 / 41 | 14.5 / 16.4 | 0.75 / 0.59 | 0.04 / 1.51 | 0.00064 | 0.00064 | - |  |
| efficientnet_v2_s.tv_in1k | fp32 | 15.0 / 17.8 | 0.84 | 0.05 / 1.46 | 157 / 260 | 0.60 | 82 / 82 | 15.6 / 18.3 | 0.64 / 0.60 | 0.05 / 1.46 | 3.9e-07 | 3.9e-07 | - |  |
| efficientvit-sam-l0 | fp32 | 28.7 / 52.3 | 0.55 | 0.06 / 1.23 | 458 / 472 | 0.97 | 120 / 117 | 57.8 / 81.1 | 0.15 / 1.17 | 0.10 / 1.27 | 4.4e-05 | 1 (0.00036 % off) | - |  |
| efficientvit-sam-l1 | fp32 | 38.7 / 62.4 | 0.62 | 0.07 / 1.38 | 615 / 624 | 0.99 | 169 / 166 | 67.8 / 91.6 | 0.18 / 1.40 | 0.10 / 1.42 | 0.00012 | 1 (0.0028 % off) | - |  |
| efficientvit-sam-l2 | fp32 | 51.8 / 76.7 | 0.68 | 0.08 / 1.50 | 732 / 778 | 0.94 | 222 / 219 | 80.7 / 105.8 | 0.23 / 1.66 | 0.12 / 1.54 | 0.0003 | 1 (0.005 % off) | - |  |
| efficientvit-sam-xl0 | fp32 | 193.3 / 203.5 | 0.95 | 0.22 / 1.50 | 1311 / 1026 | 1.28 | 443 / 431 | 214.6 / 222.6 | 0.46 / 1.41 | 0.26 / 1.53 | 3.5e-05 | 1 (0.0011 % off) | - | heavier |
| efficientvit-sam-xl1 | fp32 | 348.8 / 342.0 | 1.02 | 0.37 / 1.93 | 2197 / 1668 | 1.32 | 772 / 760 | 368.9 / 363.1 | 0.74 / 1.56 | 0.42 / 1.96 | 3.7e-05 | 1 (0.003 % off) | - | heavier |
| efficientvit_b2.r288_in1k | fp16 | 6.5 / 8.2 | 0.79 | 0.04 / 1.35 | 90 / 137 | 0.66 | 47 / 47 | 7.1 / 8.9 | 0.82 / 0.69 | 0.04 / 1.36 | 0.0015 | 0.0015 | - |  |
| efficientvit_b2.r288_in1k | fp32 | 7.7 / 10.6 | 0.72 | 0.04 / 1.33 | 144 / 194 | 0.74 | 93 / 93 | 8.1 / 11.4 | 0.79 / 0.70 | 0.04 / 1.34 | 2.1e-06 | 2.1e-06 | - |  |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp16 | 195.8 / 197.9 | 0.99 | 0.23 / 2.13 | 688 / 666 | 1.03 | 583 / 583 | 196.6 / 198.9 | 0.08 / 0.78 | 0.23 / 2.13 | 0.0012 | 0.0012 | 0.000779 / 0.000651 (0.02) | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp32 | 242.3 / 242.4 | 1.00 | 0.27 / 2.17 | 1354 / 1396 | 0.97 | 1165 / 1165 | 243.2 / 243.1 | 0.11 / 0.79 | 0.28 / 2.17 | 6e-07 | 6e-07 | 6.83e-07 / 6.35e-07 (0.02) |  |
| falconsai-nsfw-vit | fp16 | 14.2 / 13.6 | 1.04 | 0.06 / 0.67 | 185 / 231 | 0.80 | 164 / 164 | 14.4 / 14.0 | 0.05 / 0.74 | 0.06 / 0.67 | 0 | 0 | 0.000728 / 0.000728 (0.009) | slower |
| falconsai-nsfw-vit | fp32 | 16.1 / 15.9 | 1.01 | 0.06 / 0.71 | 362 / 417 | 0.87 | 327 / 327 | 16.5 / 16.4 | 0.06 / 0.78 | 0.07 / 0.72 | 1.1e-07 | 1.1e-07 | 2.86e-06 / 2.38e-06 (0.009) |  |
| faster-rcnn-r50-fpn-3x-coco | fp16 | 90.7 / 108.4 | 0.84 | 0.11 / 0.94 | 412 / 432 | 0.95 | 80 / 52 | 125.1 / 381.6 | 0.60 / 0.68 | 0.17 / 1.34 | - | 0.00015 | 0.511 / 0.511 (1.6) |  |
| faster-rcnn-r50-fpn-3x-coco | fp32 | 114.9 / 136.7 | 0.84 | 0.41 / 0.84 | 757 / 857 | 0.88 | 159 / 105 | 149.7 / 424.2 | 0.66 / 0.63 | 0.55 / 1.17 | - | 1.6e-06 | 2.81e-05 / 2.91e-05 (1.6) |  |
| fastvit_sa24.apple_dist_in1k | fp16 | 6.1 / 6.8 | 0.90 | 0.04 / 0.87 | 60 / 104 | 0.57 | 41 / 41 | 6.6 / 7.3 | 1.02 / 0.90 | 0.04 / 0.87 | 0.003 | 0.003 | - |  |
| fastvit_sa24.apple_dist_in1k | fp32 | 6.8 / 7.7 | 0.88 | 0.06 / 0.81 | 122 / 197 | 0.62 | 82 / 82 | 7.4 / 8.3 | 0.95 / 0.90 | 0.06 / 0.81 | 3.7e-07 | 3.7e-07 | - |  |
| fastvit_t12.apple_dist_in1k | fp16 | 3.2 / 3.3 | 0.97 | 0.03 / 0.53 | 26 / 47 | 0.56 | 14 / 14 | 3.6 / 3.8 | 0.91 / 0.84 | 0.03 / 0.53 | 0.0051 | 0.0051 | - |  |
| fastvit_t12.apple_dist_in1k | fp32 | 3.6 / 3.6 | 0.99 | 0.05 / 0.49 | 54 / 87 | 0.62 | 29 / 29 | 4.1 / 4.2 | 0.86 / 0.86 | 0.05 / 0.50 | 7.1e-07 | 7.1e-07 | - |  |
| fastvit_t8.apple_dist_in1k | fp16 | 2.5 / 2.8 | 0.90 | 0.03 / 0.49 | 17 / 30 | 0.57 | 8 / 8 | 3.0 / 3.3 | 0.89 / 0.83 | 0.03 / 0.50 | 0.015 (0.7 % off) | 0.015 (0.7 % off) | - |  |
| fastvit_t8.apple_dist_in1k | fp32 | 2.9 / 3.5 | 0.81 | 0.05 / 0.47 | 33 / 55 | 0.60 | 15 / 15 | 3.4 / 3.6 | 0.85 / 0.83 | 0.05 / 0.47 | 1.3e-06 | 1.3e-06 | - |  |
| fcn-r101-d8-ade20k | fp16 | 326.3 / 354.0 | 0.92 | 0.40 / 1.82 | 877 / 638 | 1.38 | 126 / 126 | 364.1 / 380.5 | 0.23 / 0.15 | 0.45 / 1.87 | 0.001 | 0 | - | heavier |
| fcn-r101-d8-ade20k | fp32 | 439.6 / 472.5 | 0.93 | 0.49 / 1.91 | 1263 / 1236 | 1.02 | 252 / 252 | 472.3 / 510.6 | 0.21 / 0.15 | 0.53 / 1.96 | 0 | 0 | - | heavier |
| focalnet_base_lrf.ms_in1k | fp16 | 19.0 / 21.6 | 0.88 | 0.06 / 1.46 | 217 / 289 | 0.75 | 171 / 170 | 19.6 / 22.7 | 0.92 / 0.72 | 0.06 / 1.47 | 0.0006 | 0.0006 | - |  |
| focalnet_base_lrf.ms_in1k | fp32 | 21.3 / 25.7 | 0.83 | 0.05 / 1.41 | 419 / 491 | 0.85 | 340 / 340 | 22.0 / 26.5 | 0.95 / 0.70 | 0.06 / 1.41 | 2.9e-07 | 2.9e-07 | - |  |
| focalnet_base_srf.ms_in1k | fp16 | 17.9 / 20.3 | 0.88 | 0.04 / 1.36 | 214 / 285 | 0.75 | 169 / 169 | 18.4 / 20.9 | 0.93 / 0.70 | 0.04 / 1.36 | 0.0008 | 0.0008 | - |  |
| focalnet_base_srf.ms_in1k | fp32 | 20.0 / 24.6 | 0.81 | 0.05 / 1.29 | 412 / 492 | 0.84 | 337 / 337 | 20.6 / 25.4 | 0.88 / 0.74 | 0.05 / 1.29 | 3.4e-07 | 3.4e-07 | - |  |
| focalnet_small_lrf.ms_in1k | fp16 | 14.6 / 14.9 | 0.98 | 0.04 / 1.46 | 131 / 195 | 0.67 | 97 / 97 | 15.1 / 15.6 | 0.89 / 0.71 | 0.04 / 1.47 | 0.00077 | 0.00077 | - |  |
| focalnet_small_lrf.ms_in1k | fp32 | 16.3 / 17.5 | 0.93 | 0.05 / 1.35 | 251 / 347 | 0.72 | 193 / 193 | 16.8 / 18.1 | 0.81 / 0.71 | 0.05 / 1.36 | 3.3e-07 | 3.3e-07 | - |  |
| focalnet_small_srf.ms_in1k | fp16 | 13.9 / 13.9 | 1.00 | 0.04 / 1.40 | 128 / 201 | 0.64 | 96 / 96 | 14.4 / 14.7 | 0.89 / 0.73 | 0.04 / 1.40 | 0.00069 | 0.00069 | - |  |
| focalnet_small_srf.ms_in1k | fp32 | 15.0 / 16.3 | 0.92 | 0.05 / 1.25 | 246 / 341 | 0.72 | 191 / 191 | 15.7 / 16.8 | 0.81 / 0.70 | 0.05 / 1.25 | 4e-07 | 4e-07 | - |  |
| focalnet_tiny_lrf.ms_in1k | fp16 | 8.5 / 8.5 | 1.00 | 0.03 / 0.85 | 89 / 143 | 0.62 | 55 / 55 | 9.1 / 9.1 | 0.82 / 0.71 | 0.03 / 0.85 | 0.00058 | 0.00058 | - |  |
| focalnet_tiny_lrf.ms_in1k | fp32 | 9.3 / 9.7 | 0.95 | 0.04 / 0.79 | 168 / 257 | 0.65 | 110 / 110 | 9.8 / 10.2 | 0.77 / 0.71 | 0.04 / 0.79 | 2.5e-07 | 2.5e-07 | - |  |
| focalnet_tiny_srf.ms_in1k | fp16 | 8.2 / 7.9 | 1.04 | 0.04 / 0.78 | 86 / 138 | 0.63 | 55 / 55 | 8.6 / 8.5 | 0.80 / 0.69 | 0.04 / 0.79 | 0.00088 | 0.00088 | - | slower |
| focalnet_tiny_srf.ms_in1k | fp32 | 8.6 / 8.8 | 0.98 | 0.04 / 0.79 | 163 / 244 | 0.67 | 109 / 109 | 9.2 / 9.5 | 0.77 / 0.76 | 0.04 / 0.79 | 4.3e-07 | 4.3e-07 | - |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp16 | 2.0 / 2.2 | 0.87 | 0.03 / 0.74 | 19 / 28 | 0.70 | 12 / 12 | 2.0 / 2.7 | 0.76 / 0.74 | 0.04 / 0.74 | 0.00081 | 0.00081 | - |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp32 | 2.5 / 3.5 | 0.72 | 0.04 / 0.69 | 36 / 52 | 0.70 | 23 / 23 | 2.4 / 3.2 | 0.72 / 0.69 | 0.05 / 0.69 | 8.1e-07 | 8.1e-07 | - |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp16 | 2.1 / 2.7 | 0.77 | 0.04 / 0.72 | 22 / 35 | 0.64 | 12 / 12 | 2.5 / 3.1 | 0.74 / 0.69 | 0.04 / 0.72 | 0.00063 | 0.00063 | - |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp32 | 2.4 / 3.0 | 0.80 | 0.04 / 0.71 | 42 / 66 | 0.65 | 24 / 24 | 2.9 / 3.5 | 0.70 / 0.71 | 0.05 / 0.71 | 7.8e-07 | 7.8e-07 | - |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp16 | 3.0 / 3.8 | 0.80 | 0.04 / 0.95 | 34 / 53 | 0.64 | 22 / 22 | 3.4 / 4.3 | 0.78 / 0.72 | 0.04 / 0.95 | 0.00053 | 0.00053 | - |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp32 | 3.5 / 4.3 | 0.82 | 0.05 / 0.91 | 65 / 100 | 0.65 | 43 / 43 | 4.2 / 4.8 | 0.75 / 0.70 | 0.06 / 0.91 | 9.1e-07 | 9.1e-07 | - |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp16 | 3.6 / 4.6 | 0.79 | 0.04 / 1.13 | 46 / 69 | 0.66 | 31 / 31 | 4.1 / 5.1 | 0.80 / 0.70 | 0.04 / 1.13 | 0.00052 | 0.00052 | - |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp32 | 4.2 / 5.1 | 0.83 | 0.06 / 1.08 | 88 / 131 | 0.67 | 62 / 62 | 4.8 / 5.8 | 0.74 / 0.70 | 0.06 / 1.08 | 5.1e-07 | 5.1e-07 | - |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp16 | 5.0 / 5.4 | 0.92 | 0.03 / 1.01 | 58 / 72 | 0.81 | 38 / 38 | 5.4 / 5.9 | 0.80 / 0.70 | 0.03 / 1.02 | 0.00064 | 0.00064 | - |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp32 | 5.7 / 6.3 | 0.90 | 0.04 / 1.01 | 111 / 137 | 0.81 | 76 / 76 | 6.2 / 6.7 | 0.74 / 0.70 | 0.04 / 1.01 | 5.4e-07 | 5.4e-07 | - |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp16 | 9.0 / 9.7 | 0.93 | 0.04 / 1.46 | 104 / 139 | 0.75 | 76 / 76 | 9.5 / 10.4 | 0.86 / 0.71 | 0.04 / 1.46 | 0.00064 | 0.00064 | - |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp32 | 10.4 / 11.3 | 0.92 | 0.04 / 1.43 | 196 / 256 | 0.76 | 151 / 151 | 10.9 / 11.9 | 0.80 / 0.71 | 0.05 / 1.43 | 3.9e-07 | 3.9e-07 | - |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp16 | 18.9 / 19.9 | 0.95 | 0.04 / 1.80 | 187 / 272 | 0.69 | 144 / 144 | 19.4 / 20.5 | 0.94 / 0.72 | 0.04 / 1.80 | 0.00088 | 0.00088 | - |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp32 | 21.7 / 23.5 | 0.92 | 0.06 / 1.77 | 368 / 493 | 0.75 | 288 / 288 | 22.3 / 24.0 | 0.83 / 0.73 | 0.06 / 1.77 | 5.9e-07 | 5.9e-07 | - |  |
| hrnet-w48-fcn-ade20k | fp16 | 152.8 / 162.0 | 0.94 | 0.18 / 2.29 | 313 / 292 | 1.07 | 129 / 128 | 194.1 / 200.5 | 0.27 / 0.20 | 0.23 / 2.33 | 0.00094 | 0.76 (0.056 % off) | - | heavier |
| hrnet-w48-fcn-ade20k | fp32 | 176.6 / 193.7 | 0.91 | 0.21 / 2.19 | 583 / 609 | 0.96 | 254 / 254 | 216.9 / 236.1 | 0.27 / 0.20 | 0.25 / 2.23 | 0 | 0 | - |  |
| igev-plusplus-eth3d | fp16 | 5696.6 / 7121.3 | 0.80 | 5.74 / 14.85 | 2414 / 2199 | 1.10 | 28 / 29 | 5712.6 / 7339.0 | 0.16 / 0.12 | 5.75 / 14.86 | 0.45 (0.33 % off) | 0.29 (0.034 % off) | 0.00067 / 0.00256 (0.02) | heavier |
| igev-plusplus-eth3d | fp32 | 6918.7 / 7331.1 | 0.94 | 6.88 / 14.77 | 4260 / 3786 | 1.13 | 55 / 56 | 7040.7 / 7292.8 | 0.13 / 0.10 | 6.89 / 14.78 | 0.0043 | 0.0043 | 1.36e-06 / 1.28e-06 (0.02) | heavier |
| igev-plusplus-kitti12 | fp16 | 5631.7 / 7096.5 | 0.79 | 5.68 / 13.00 | 2414 / 2189 | 1.10 | 28 / 29 | 5631.1 / 6993.8 | 0.15 / 0.10 | 5.69 / 13.01 | 0.023 (0.0032 % off) | 0.026 (0.00027 % off) | 0.00114 / 0.000852 (0.02) | heavier |
| igev-plusplus-kitti12 | fp32 | 6907.6 / 7268.4 | 0.95 | 7.08 / 14.58 | 4260 / 3786 | 1.13 | 55 / 56 | 6855.2 / 7366.3 | 0.14 / 0.10 | 7.10 / 14.59 | 0.00037 | 0.00053 | 9.87e-07 / 7.88e-07 (0.02) | heavier |
| igev-plusplus-kitti15 | fp16 | 5640.9 / 7267.1 | 0.78 | 5.69 / 13.30 | 2414 / 2199 | 1.10 | 28 / 29 | 5680.1 / 6981.4 | 0.16 / 0.11 | 5.70 / 13.31 | 0.053 (0.003 % off) | 0.0086 | 0.00114 / 0.00103 (0.02) | heavier |
| igev-plusplus-kitti15 | fp32 | 6913.8 / 7172.6 | 0.96 | 6.97 / 15.02 | 4260 / 3786 | 1.13 | 55 / 56 | 6892.2 / 7314.1 | 0.13 / 0.10 | 6.99 / 15.03 | 1.9e-05 | 2.1e-05 | 9.07e-07 / 8.77e-07 (0.02) | heavier |
| igev-plusplus-middlebury | fp16 | 5695.1 / 7274.0 | 0.78 | 5.77 / 12.84 | 2414 / 2189 | 1.10 | 28 / 29 | 5718.3 / 7251.8 | 0.15 / 0.10 | 5.78 / 12.85 | 0.39 (1.2 % off) | 0.41 (0.25 % off) | 0.000571 / 0.00229 (0.02) | heavier |
| igev-plusplus-middlebury | fp32 | 6861.8 / 7266.2 | 0.94 | 6.93 / 15.59 | 4260 / 3786 | 1.13 | 55 / 56 | 6887.7 / 7217.4 | 0.13 / 0.12 | 6.94 / 15.60 | 0.00055 | 0.00022 | 1.68e-06 / 1.75e-06 (0.02) | heavier |
| igev-plusplus-sceneflow | fp16 | 5654.4 / 7257.3 | 0.78 | 5.71 / 13.35 | 2414 / 2189 | 1.10 | 28 / 29 | 5648.2 / 7135.1 | 0.15 / 0.10 | 5.72 / 13.35 | 0.5 (0.023 % off) | 0.44 (0.017 % off) | 0.00137 / 0.00153 (0.02) | heavier |
| igev-plusplus-sceneflow | fp32 | 6877.3 / 7236.8 | 0.95 | 6.91 / 14.83 | 4260 / 3786 | 1.13 | 55 / 56 | 6875.2 / 7133.3 | 0.13 / 0.10 | 6.92 / 14.84 | 0.00045 | 0.00045 | 1.08e-06 / 1.09e-06 (0.02) | heavier |
| igev-stereo-eth3d | fp16 | 3183.2 / 3604.2 | 0.88 | 3.21 / 8.70 | 1686 / 1080 | 1.56 | 24 / 26 | 3195.7 / 3616.7 | 0.14 / 0.10 | 3.22 / 8.70 | 0.49 (0.38 % off) | 0.49 (0.067 % off) | 0.00144 / 0.00166 (0.02) | heavier |
| igev-stereo-eth3d | fp32 | 4165.2 / 4606.8 | 0.90 | 4.20 / 9.71 | 3122 / 2055 | 1.52 | 48 / 49 | 4143.3 / 4614.3 | 0.12 / 0.10 | 4.21 / 9.71 | 0.00084 | 0.00084 | 8.99e-06 / 8.31e-06 (0.02) | heavier |
| igev-stereo-kitti12 | fp16 | 3187.2 / 3651.0 | 0.87 | 3.21 / 8.81 | 1686 / 1080 | 1.56 | 24 / 26 | 3195.2 / 3639.8 | 0.14 / 0.09 | 3.22 / 8.82 | 0.052 (0.016 % off) | 0.0042 | 0.000965 / 0.00117 (0.02) | heavier |
| igev-stereo-kitti12 | fp32 | 4138.5 / 4637.6 | 0.89 | 4.26 / 9.73 | 3122 / 2055 | 1.52 | 48 / 49 | 4184.1 / 4596.8 | 0.12 / 0.09 | 4.27 / 9.73 | 2.7e-05 | 4.9e-06 | 1.03e-06 / 1.03e-06 (0.02) | heavier |
| igev-stereo-kitti15 | fp16 | 3184.4 / 3614.1 | 0.88 | 3.22 / 8.71 | 1686 / 1010 | 1.67 | 24 / 26 | 3193.4 / 3614.6 | 0.15 / 0.09 | 3.23 / 8.71 | 0.018 (0.00084 % off) | 0.004 | 0.000928 / 0.00102 (0.02) | heavier |
| igev-stereo-kitti15 | fp32 | 4165.1 / 4787.5 | 0.87 | 4.17 / 9.99 | 3122 / 2055 | 1.52 | 48 / 49 | 4201.9 / 4750.9 | 0.12 / 0.09 | 4.18 / 9.99 | 9.1e-06 | 1.8e-06 | 7.73e-07 / 7.74e-07 (0.02) | heavier |
| igev-stereo-middlebury | fp16 | 3192.1 / 3633.4 | 0.88 | 3.22 / 9.05 | 1686 / 1080 | 1.56 | 24 / 26 | 3203.3 / 3624.6 | 0.15 / 0.10 | 3.23 / 9.06 | 5.2 (1.7 % off) | 5.2 (0.24 % off) | 0.001 / 0.00126 (0.02) | heavier |
| igev-stereo-middlebury | fp32 | 4180.7 / 4608.6 | 0.91 | 4.18 / 9.88 | 3122 / 2055 | 1.52 | 48 / 49 | 4191.3 / 4802.9 | 0.12 / 0.10 | 4.20 / 9.88 | 0.0013 | 0.0013 | 1.35e-06 / 1.4e-06 (0.02) | heavier |
| igev-stereo-sceneflow | fp16 | 3190.5 / 3621.4 | 0.88 | 3.23 / 8.76 | 1686 / 1010 | 1.67 | 24 / 26 | 3196.6 / 3619.0 | 0.14 / 0.10 | 3.24 / 8.77 | 0.36 (0.16 % off) | 0.2 (0.03 % off) | 0.00188 / 0.00195 (0.02) | heavier |
| igev-stereo-sceneflow | fp32 | 4184.8 / 4628.8 | 0.90 | 4.20 / 9.75 | 3122 / 2055 | 1.52 | 48 / 49 | 4192.8 / 4640.4 | 0.12 / 0.09 | 4.21 / 9.76 | 3.7e-05 | 3.7e-05 | 1.65e-06 / 1.74e-06 (0.02) | heavier |
| inception_v3.tv_in1k | fp16 | 7.6 / 10.0 | 0.76 | 0.03 / 1.05 | 66 / 152 | 0.44 | 46 / 46 | 8.2 / 10.5 | 0.68 / 0.58 | 0.04 / 1.05 | 0.00084 | 0.00084 | - |  |
| inception_v3.tv_in1k | fp32 | 8.5 / 11.6 | 0.73 | 0.05 / 1.04 | 125 / 272 | 0.46 | 91 / 91 | 8.9 / 12.1 | 0.62 / 0.57 | 0.05 / 1.04 | 1e-06 | 1e-06 | - |  |
| lineart | fp16 | 74.8 / 100.1 | 0.75 | 0.12 / 1.04 | 597 / 594 | 1.00 | 8 / 8 | 85.8 / 108.8 | 0.06 / 0.03 | 0.14 / 1.05 | - | 0.028 (0.047 % off) (1 keys differ) | 6.47e-05 / 0.000117 (0.02) |  |
| lineart | fp32 | 99.0 / 117.6 | 0.84 | 0.17 / 1.04 | 1188 / 903 | 1.32 | 16 / 16 | 109.0 / 127.0 | 0.07 / 0.03 | 0.19 / 1.06 | - | 1.8e-05 (1 keys differ) | 5.61e-07 / 5.59e-07 (0.02) | heavier |
| lineart-coarse | fp16 | 70.1 / 100.1 | 0.70 | 0.08 / 1.02 | 597 / 594 | 1.00 | 8 / 8 | 79.4 / 109.0 | 0.06 / 0.03 | 0.10 / 1.04 | - | 0.022 (0.014 % off) (1 keys differ) | 0.000108 / 0.000162 (0.02) |  |
| lineart-coarse | fp32 | 110.1 / 117.4 | 0.94 | 0.14 / 1.05 | 1188 / 903 | 1.32 | 16 / 16 | 115.4 / 126.9 | 0.06 / 0.03 | 0.16 / 1.07 | - | 1.6e-05 (1 keys differ) | 7.97e-07 / 7.98e-07 (0.02) | heavier |
| loftr-indoor | fp16 | 365.9 / 1226.4 | 0.30 | 1.06 / 5.90 | 1098 / 1237 | 0.89 | 40 / 21 | 465.9 / 5319.3 | 0.11 / 0.10 | 1.32 / 8.96 | - | empty (4 keys differ) | 0.00179 r0.998 / 0.00176 r0.998 (0.0025) |  |
| loftr-indoor | fp32 | 516.9 / 1759.6 | 0.29 | 0.51 / 4.07 | 2161 / 1314 | 1.64 | 64 / 43 | 1135.8 / 2581.9 | 0.08 / 0.05 | 1.01 / 6.39 | - | 2.4e-05 | 5.5e-06 r1.000 / 5.51e-06 r1.000 (0.0025) | heavier |
| loftr-outdoor | fp16 | 345.8 / 1151.2 | 0.30 | 0.37 / 2.63 | 1098 / 1237 | 0.89 | 40 / 21 | 454.0 / 3979.9 | 0.10 / 0.04 | 0.55 / 3.96 | - | 0.0018 | 0.00219 r0.999 / 0.00233 r0.999 (0.0025) |  |
| loftr-outdoor | fp32 | 490.3 / 1366.8 | 0.36 | 0.52 / 2.94 | 2161 / 1314 | 1.64 | 64 / 43 | 716.2 / 3143.1 | 0.11 / 0.04 | 1.07 / 4.29 | - | 7.6e-06 | 5.98e-06 r1.000 / 5.99e-06 r1.000 (0.0025) | heavier |
| loma-b | fp16 | 2280.5 / 2547.8 | 0.90 | 2.30 / 7.88 | 1488 / 1856 | 0.80 | 712 / 654 | 2301.5 / 2674.6 | 1.87 / 0.23 | 2.33 / 7.94 | - | 0.98 (79 % off) (3 keys differ) | - |  |
| loma-b | fp32 | 3166.5 / 3379.9 | 0.94 | 3.16 / 9.51 | 2751 / 3714 | 0.74 | 1308 / 1308 | 3184.8 / 3464.9 | 1.44 / 0.40 | 3.19 / 9.60 | - | 0.64 (0.084 % off) (1 keys differ) | - |  |
| loma-b128 | fp16 | 800.0 / 925.8 | 0.86 | 0.80 / 4.54 | 886 / 994 | 0.89 | 112 / 60 | 801.5 / 983.5 | 0.51 / 0.10 | 0.83 / 4.60 | - | 0.98 (79 % off) (3 keys differ) | - |  |
| loma-b128 | fp32 | 1195.5 / 1312.4 | 0.91 | 1.24 / 4.74 | 1234 / 1064 | 1.16 | 119 / 120 | 1219.3 / 1359.8 | 0.17 / 0.10 | 1.26 / 4.80 | - | 0.64 (0.084 % off) (1 keys differ) | - | heavier |
| loma-b64 | fp16 | 639.4 / 639.7 | 1.00 | 0.67 / 4.10 | 817 / 907 | 0.90 | 79 / 48 | 647.4 / 691.1 | 0.40 / 0.11 | 0.70 / 4.16 | - | 0.98 (83 % off) (3 keys differ) | - |  |
| loma-b64 | fp32 | 1019.7 / 989.3 | 1.03 | 1.04 / 4.33 | 1164 / 988 | 1.18 | 96 / 96 | 1030.0 / 1027.2 | 0.17 / 0.11 | 1.06 / 4.38 | - | 0.74 (0.23 % off) (1 keys differ) | - | slower heavier |
| loma-g | fp16 | 2751.7 / 3019.5 | 0.91 | 2.77 / 8.54 | 1813 / 1916 | 0.95 | 1037 / 978 | 2770.2 / 3080.3 | 2.30 / 0.23 | 2.81 / 8.61 | - | 0.98 (79 % off) (3 keys differ) | - |  |
| loma-g | fp32 | 3791.5 / 4200.1 | 0.90 | 3.92 / 9.92 | 3398 / 4360 | 0.78 | 1954 / 1954 | 3744.8 / 4117.5 | 1.89 / 0.40 | 3.95 / 10.00 | - | 0.64 (0.084 % off) (1 keys differ) | - |  |
| loma-l | fp16 | 2390.2 / 2626.9 | 0.91 | 2.40 / 7.66 | 1553 / 1929 | 0.80 | 777 / 719 | 2407.0 / 2685.3 | 1.92 / 0.23 | 2.43 / 7.73 | - | 0.98 (79 % off) (3 keys differ) | - |  |
| loma-l | fp32 | 3299.5 / 3482.7 | 0.95 | 3.31 / 9.08 | 2881 / 3843 | 0.75 | 1437 / 1438 | 3321.7 / 3559.0 | 1.49 / 0.27 | 3.35 / 9.15 | - | 0.64 (0.084 % off) (1 keys differ) | - |  |
| loma-r | fp16 | 2272.8 / 2620.4 | 0.87 | 2.29 / 7.60 | 1488 / 1592 | 0.93 | 712 / 654 | 2290.7 / 2630.0 | 1.87 / 0.23 | 2.32 / 7.66 | - | 0.98 (79 % off) (3 keys differ) | - |  |
| loma-r | fp32 | 3145.3 / 3333.3 | 0.94 | 3.13 / 8.84 | 2751 / 3714 | 0.74 | 1308 / 1308 | 3148.5 / 3382.4 | 1.43 / 0.42 | 3.16 / 8.91 | - | 0.64 (0.084 % off) (1 keys differ) | - |  |
| lw-detr-large-coco | fp16 | 53.5 / 57.4 | 0.93 | 0.10 / 2.11 | 548 / 319 | 1.72 | 118 / 81 | 56.4 / 61.5 | 2.24 / 2.25 | 0.10 / 2.12 | 0.96 (66 % off) | 0.0072 | 0.0368 / 0.0657 (0.19) | heavier |
| lw-detr-large-coco | fp32 | 57.2 / 60.3 | 0.95 | 0.13 / 2.00 | 568 / 522 | 1.09 | 154 / 160 | 60.1 / 64.0 | 2.30 / 2.23 | 0.14 / 2.01 | 0.0015 | 2.3e-06 | 2.16e-05 / 3.12e-05 (0.19) | heavier |
| lw-detr-medium-coco | fp16 | 37.1 / 41.9 | 0.89 | 0.10 / 1.84 | 224 / 290 | 0.78 | 60 / 51 | 40.0 / 45.1 | 2.25 / 2.17 | 0.11 / 1.84 | 0.97 (76 % off) | empty (3 keys differ) | 0.0305 / 0.0343 (0.19) |  |
| lw-detr-medium-coco | fp32 | 41.5 / 41.6 | 1.00 | 0.09 / 1.76 | 228 / 231 | 0.99 | 96 / 100 | 44.5 / 44.9 | 2.15 / 2.01 | 0.09 / 1.77 | 0.00041 | 5.4e-06 | 2.73e-05 / 2.93e-05 (0.19) |  |
| lw-detr-small-coco | fp16 | 24.0 / 27.9 | 0.86 | 0.08 / 1.75 | 184 / 299 | 0.62 | 34 / 24 | 27.0 / 30.9 | 2.04 / 2.04 | 0.08 / 1.76 | 0.96 (66 % off) | empty (3 keys differ) | 0.0515 / 0.0868 (0.28) |  |
| lw-detr-small-coco | fp32 | 27.8 / 28.2 | 0.99 | 0.08 / 1.73 | 140 / 185 | 0.75 | 43 / 47 | 30.5 / 31.3 | 2.05 / 2.01 | 0.09 / 1.74 | 0.00023 | 6.2e-06 | 0.00017 / 0.000104 (0.28) |  |
| lw-detr-tiny-coco | fp16 | 17.1 / 19.6 | 0.87 | 0.08 / 1.49 | 138 / 258 | 0.54 | 30 / 19 | 20.0 / 22.8 | 2.11 / 2.00 | 0.08 / 1.50 | 0.96 (56 % off) | empty (3 keys differ) | 0.0445 / 0.0765 (0.23) |  |
| lw-detr-tiny-coco | fp32 | 19.8 / 20.0 | 0.99 | 0.06 / 1.46 | 125 / 176 | 0.71 | 36 / 37 | 22.5 / 23.1 | 2.04 / 2.02 | 0.07 / 1.46 | 0.00023 | 6.3e-06 | 3.5e-05 / 7.13e-05 (0.23) |  |
| lw-detr-xlarge-coco | fp16 | 97.2 / 107.9 | 0.90 | 0.15 / 2.07 | 722 / 445 | 1.62 | 288 / 219 | 100.2 / 111.6 | 2.34 / 2.06 | 0.15 / 2.08 | 0.97 (69 % off) | empty (3 keys differ) | 0.00829 / 0.0337 (0.14) | heavier |
| lw-detr-xlarge-coco | fp32 | 105.4 / 110.6 | 0.95 | 0.14 / 2.02 | 860 / 884 | 0.97 | 427 / 434 | 108.3 / 114.0 | 2.26 / 2.09 | 0.14 / 2.03 | 0.61 (0.78 % off) | 0.00068 | 3.66e-05 / 3.35e-05 (0.14) |  |
| mask-rcnn-r50-fpn-3x-coco | fp16 | 89.1 / 104.4 | 0.85 | 0.11 / 0.68 | 417 / 432 | 0.96 | 86 / 52 | 141.9 / 374.9 | 0.54 / 0.53 | 0.19 / 0.97 | - | 1 (0.0068 % off) | 0.0728 / 0.0717 (0.29) |  |
| mask-rcnn-r50-fpn-3x-coco | fp32 | 112.6 / 136.9 | 0.82 | 0.19 / 0.79 | 833 / 857 | 0.97 | 169 / 105 | 174.4 / 470.4 | 0.61 / 0.68 | 0.29 / 1.21 | - | 1 (7e-05 % off) | 4.77e-05 / 3.62e-05 (0.29) |  |
| mask2former-swinl-ade20k | fp16 | 590.2 / 651.5 | 0.91 | 1.22 / 6.25 | 849 / 929 | 0.91 | 458 / 455 | 647.9 / 835.7 | 4.67 / 3.13 | 1.65 / 7.21 | - | 0 (2 keys differ) | - |  |
| mask2former-swinl-ade20k | fp32 | 686.5 / 1000.8 | 0.69 | 0.72 / 6.28 | 1704 / 1760 | 0.97 | 915 / 907 | 755.4 / 1011.8 | 2.85 / 3.15 | 0.84 / 7.73 | - | 0 (2 keys differ) | - |  |
| mobilenet_v2.tv2_in1k | fp16 | 2.1 / 2.6 | 0.81 | 0.02 / 0.77 | 23 / 24 | 0.95 | 7 / 7 | 2.4 / 2.4 | 0.56 / 0.50 | 0.02 / 0.78 | 0.0012 | 0.0012 | - |  |
| mobilenet_v2.tv2_in1k | fp32 | 2.3 / 2.4 | 0.97 | 0.03 / 0.75 | 42 / 44 | 0.96 | 14 / 14 | 2.7 / 2.7 | 0.52 / 0.51 | 0.03 / 0.75 | 1.5e-06 | 1.5e-06 | - |  |
| mobilenet_v3_large.tv2_in1k | fp16 | 2.2 / 3.3 | 0.68 | 0.02 / 0.87 | 18 / 32 | 0.57 | 11 / 11 | 2.5 / 3.3 | 0.57 / 0.50 | 0.02 / 0.88 | 0.0025 | 0.0025 | - |  |
| mobilenet_v3_large.tv2_in1k | fp32 | 3.7 / 3.2 | 1.15 | 0.04 / 0.86 | 34 / 53 | 0.64 | 21 / 21 | 2.9 / 3.6 | 0.51 / 0.51 | 0.05 / 0.86 | 1.4e-06 | 1.4e-06 | - | slower |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp16 | 2.0 / 2.2 | 0.92 | 0.02 / 0.57 | 15 / 23 | 0.65 | 8 / 8 | 2.3 / 2.5 | 0.78 / 0.74 | 0.02 / 0.57 | 0.0015 | 0.0015 | 0.0465 / 0.0325 (0.08) | worse |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp32 | 2.1 / 2.1 | 1.01 | 0.04 / 0.54 | 29 / 41 | 0.72 | 16 / 16 | 2.8 / 2.6 | 0.75 / 0.74 | 0.04 / 0.55 | 1.8e-06 | 1.8e-06 | 6.54e-06 / 1.06e-05 (0.08) |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp16 | 3.0 / 3.3 | 0.91 | 0.02 / 0.91 | 30 / 41 | 0.73 | 19 / 19 | 3.5 / 3.7 | 0.79 / 0.70 | 0.03 / 0.92 | 0.00074 | 0.00074 | - |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp32 | 4.1 / 3.5 | 1.17 | 0.04 / 0.90 | 56 / 74 | 0.76 | 37 / 37 | 3.8 / 4.1 | 0.73 / 0.71 | 0.04 / 0.90 | 5.4e-07 | 5.4e-07 | - | slower |
| mvsanywhere | fp16 | 414.7 / 372.0 | 1.11 | 3.50 / 3.45 | 767 / 767 | 1.00 | 386 / 386 | 424.6 / 3490.1 | 0.10 / 0.10 | 3.51 / 6.59 | 0 | 0.045 (0.49 % off) | - | slower |
| mvsanywhere | fp32 | 502.0 / 471.5 | 1.06 | 3.54 / 3.54 | 1407 / 1407 | 1.00 | 771 / 771 | 508.6 / 3574.8 | 0.10 / 0.10 | 3.56 / 6.70 | 0 | 0.00018 | - | slower |
| nafnet-gopro | fp16 | 196.6 / 250.9 | 0.78 | 0.22 / 3.14 | 671 / 772 | 0.87 | 130 / 255 | 196.9 / 251.5 | 0.30 / 0.17 | 0.22 / 3.14 | 0.0019 | 1 (0.05 % off) | 0.0125 / 0.0142 (0.018) |  |
| nafnet-gopro | fp32 | 257.3 / 293.8 | 0.88 | 0.31 / 3.09 | 1169 / 839 | 1.39 | 259 / 384 | 258.1 / 293.8 | 0.27 / 0.17 | 0.31 / 3.09 | 2e-06 | 0.0098 | 6.55e-06 / 3.47e-06 (0.018) | heavier |
| nafnet-sidd | fp16 | 184.3 / 208.2 | 0.89 | 0.21 / 2.25 | 658 / 546 | 1.21 | 221 / 221 | 185.2 / 209.8 | 0.42 / 0.21 | 0.21 / 2.25 | 0.00088 | 0.0015 | 0.000206 / 0.000203 (0.0015) | heavier |
| nafnet-sidd | fp32 | 233.4 / 269.8 | 0.87 | 0.27 / 2.22 | 1184 / 896 | 1.32 | 442 / 442 | 234.8 / 270.1 | 0.40 / 0.21 | 0.28 / 2.23 | 7e-07 | 1.1e-06 | 9.81e-08 / 9.98e-08 (0.0015) | heavier |
| oneformer-swinl-ade20k | fp16 | 643.5 / 760.8 | 0.85 | 0.89 / 6.75 | 1364 / 1002 | 1.36 | 523 / 536 | 861.8 / 1017.3 | 2.92 / 2.80 | 1.13 / 7.07 | - | 0 (2 keys differ) | - | heavier |
| oneformer-swinl-ade20k | fp32 | 735.6 / 948.3 | 0.78 | 1.18 / 6.84 | 1894 / 2040 | 0.93 | 980 / 1070 | 990.3 / 1177.7 | 2.71 / 2.76 | 1.44 / 7.11 | - | 0 (2 keys differ) | - |  |
| pe-core-b16-224 | fp16 | 14.9 / 20.5 | 0.73 | 0.03 / 0.87 | 263 / 404 | 0.65 | 176 / 177 | 15.2 / 20.7 | 0.85 / 0.71 | 0.03 / 0.87 | 0.03 (0.024 % off) | 0.03 (0.024 % off) | - |  |
| pe-core-b16-224 | fp32 | 17.3 / 17.6 | 0.98 | 0.05 / 0.90 | 496 / 436 | 1.14 | 352 / 354 | 17.4 / 18.0 | 0.78 / 0.72 | 0.05 / 0.90 | 8.6e-05 | 8.6e-05 | - | heavier |
| pe-core-g14-448 | fp16 | 249.2 / 354.6 | 0.70 | 0.28 / 13.99 | 3664 / 3940 | 0.93 | 3580 / 3585 | 249.6 / 355.1 | 10.22 / 1.37 | 0.29 / 14.00 | 0.05 (0.014 % off) | 0.05 (0.014 % off) | - |  |
| pe-core-g14-448 | fp32 | 292.3 / 292.0 | 1.00 | 1.08 / 27.74 | 7303 / 7296 | 1.00 | 7160 / 7169 | 292.6 / 292.4 | 10.08 / 1.34 | 1.09 / 27.75 | 4e-05 | 4e-05 | - |  |
| pe-core-l14-336 | fp16 | 56.7 / 77.5 | 0.73 | 0.08 / 1.58 | 680 / 881 | 0.77 | 601 / 602 | 57.0 / 77.8 | 0.99 / 0.76 | 0.08 / 1.58 | 0.0091 | 0.0091 | - |  |
| pe-core-l14-336 | fp32 | 66.2 / 66.3 | 1.00 | 0.10 / 1.59 | 1334 / 1372 | 0.97 | 1201 / 1205 | 66.7 / 66.5 | 0.84 / 0.77 | 0.11 / 1.60 | 3e-05 | 3e-05 | - |  |
| pe-core-s16-384 | fp16 | 6.4 / 8.0 | 0.79 | 0.02 / 0.88 | 94 / 200 | 0.47 | 45 / 45 | 6.6 / 8.3 | 0.83 / 0.76 | 0.02 / 0.88 | 0.0056 | 0.0056 | - |  |
| pe-core-s16-384 | fp32 | 7.3 / 7.3 | 1.00 | 0.04 / 0.84 | 170 / 166 | 1.02 | 89 / 90 | 7.5 / 7.4 | 0.76 / 0.78 | 0.04 / 0.84 | 1.3e-05 | 1.3e-05 | - | heavier |
| pe-core-t16-384 | fp16 | 3.7 / 4.5 | 0.82 | 0.02 / 0.83 | 34 / 67 | 0.50 | 12 / 12 | 3.8 / 4.7 | 0.82 / 0.71 | 0.02 / 0.83 | 0.011 (0.0053 % off) | 0.011 (0.0053 % off) | - |  |
| pe-core-t16-384 | fp32 | 3.9 / 4.2 | 0.92 | 0.04 / 0.81 | 62 / 59 | 1.04 | 23 / 23 | 4.1 / 4.3 | 0.77 / 0.71 | 0.04 / 0.81 | 4.6e-05 | 4.6e-05 | - | heavier |
| pe-spatial-b16-512 | fp16 | 14.5 / 19.7 | 0.73 | 0.03 / 0.84 | 251 / 390 | 0.64 | 164 / 164 | 14.7 / 20.0 | 0.84 / 0.71 | 0.03 / 0.84 | 0.0021 | 0.0021 | - |  |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.2 | 0.97 | 0.05 / 0.82 | 471 / 409 | 1.15 | 327 / 327 | 17.0 / 17.3 | 0.79 / 0.72 | 0.05 / 0.83 | 7e-06 | 7e-06 | - | heavier |
| pe-spatial-g14-448 | fp16 | 248.2 / 353.0 | 0.70 | 0.31 / 6.58 | 3614 / 3886 | 0.93 | 3531 / 3531 | 248.7 / 353.2 | 9.38 / 1.35 | 0.31 / 6.58 | 0.0069 | 0.0069 | - |  |
| pe-spatial-g14-448 | fp32 | 290.7 / 290.5 | 1.00 | 0.87 / 25.75 | 7204 / 7188 | 1.00 | 7061 / 7061 | 291.4 / 291.1 | 9.94 / 1.39 | 0.88 / 25.76 | 5.8e-06 | 5.8e-06 | - |  |
| pe-spatial-l14-448 | fp16 | 56.2 / 76.4 | 0.74 | 0.08 / 1.55 | 658 / 857 | 0.77 | 578 / 578 | 56.4 / 76.6 | 1.02 / 0.76 | 0.08 / 1.55 | 0.0024 | 0.0024 | - |  |
| pe-spatial-l14-448 | fp32 | 65.4 / 65.5 | 1.00 | 0.10 / 1.51 | 1290 / 1324 | 0.97 | 1157 / 1157 | 65.7 / 65.8 | 0.85 / 0.75 | 0.10 / 1.51 | 1.7e-06 | 1.7e-06 | - |  |
| pe-spatial-s16-512 | fp16 | 6.2 / 7.6 | 0.82 | 0.03 / 0.81 | 91 / 197 | 0.46 | 42 / 41 | 6.4 / 7.8 | 0.83 / 0.70 | 0.03 / 0.81 | 0.0055 | 0.0055 | - |  |
| pe-spatial-s16-512 | fp32 | 7.0 / 6.9 | 1.03 | 0.04 / 0.80 | 163 / 159 | 1.02 | 83 / 83 | 7.3 / 7.2 | 0.75 / 0.73 | 0.04 / 0.81 | 1.4e-05 | 1.4e-05 | - | slower heavier |
| pe-spatial-t16-512 | fp16 | 3.6 / 4.2 | 0.85 | 0.02 / 0.79 | 33 / 66 | 0.50 | 11 / 11 | 3.9 / 4.4 | 0.80 / 0.70 | 0.02 / 0.79 | 0.0077 | 0.0077 | - |  |
| pe-spatial-t16-512 | fp32 | 3.8 / 3.7 | 1.01 | 0.04 / 0.76 | 59 / 58 | 1.03 | 21 / 21 | 4.0 / 4.1 | 0.77 / 0.70 | 0.04 / 0.76 | 1.3e-05 | 1.3e-05 | - | heavier |
| pspnet-r101-d8-ade20k | fp16 | 338.6 / 365.6 | 0.93 | 0.41 / 1.92 | 1042 / 768 | 1.36 | 125 / 125 | 377.3 / 405.3 | 0.21 / 0.14 | 0.45 / 1.97 | 0.00078 | 0 | - | heavier |
| pspnet-r101-d8-ade20k | fp32 | 434.8 / 486.7 | 0.89 | 0.50 / 1.99 | 1547 / 1724 | 0.90 | 250 / 250 | 470.7 / 548.4 | 0.21 / 0.15 | 0.54 / 2.04 | 2.9e-07 | 0 | - |  |
| raft-stereo-eth3d | fp16 | 3267.0 / 4104.8 | 0.80 | 3.29 / 8.84 | 1246 / 1042 | 1.20 | 35 / 82 | 3271.8 / 4116.7 | 0.08 / 0.06 | 3.31 / 8.85 | 0.058 (0.06 % off) | 0.049 (0.015 % off) | 0.00253 / 0.00243 (0.02) | heavier |
| raft-stereo-eth3d | fp32 | 4902.1 / 5253.9 | 0.93 | 4.88 / 9.90 | 1976 / 1872 | 1.06 | 42 / 116 | 4991.0 / 5184.6 | 0.08 / 0.07 | 4.90 / 9.91 | 0.00012 | 0.00012 | 1.6e-06 / 1.7e-06 (0.02) | heavier |
| raft-stereo-middlebury | fp16 | 3231.3 / 4128.3 | 0.78 | 3.24 / 8.80 | 1246 / 1042 | 1.20 | 35 / 82 | 3212.1 / 4107.9 | 0.07 / 0.06 | 3.25 / 8.81 | 0.034 (0.15 % off) | 0.036 (0.13 % off) | 0.00612 / 0.00422 (0.02) | heavier worse |
| raft-stereo-middlebury | fp32 | 4970.9 / 5321.8 | 0.93 | 4.88 / 9.85 | 2222 / 1872 | 1.19 | 42 / 116 | 4925.9 / 5173.5 | 0.08 / 0.06 | 4.89 / 9.86 | 0.00092 | 0.00098 | 3.37e-06 / 5.3e-06 (0.02) | heavier |
| raft-stereo-sceneflow | fp16 | 3238.1 / 4084.1 | 0.79 | 3.20 / 8.64 | 1246 / 1042 | 1.20 | 35 / 82 | 3264.1 / 4084.6 | 0.08 / 0.06 | 3.21 / 8.65 | 0.27 (0.11 % off) | 0.28 (0.057 % off) | 0.00178 / 0.00198 (0.02) | heavier |
| raft-stereo-sceneflow | fp32 | 4950.0 / 5270.2 | 0.94 | 4.87 / 9.77 | 1976 / 1872 | 1.06 | 42 / 116 | 4895.1 / 5186.5 | 0.08 / 0.06 | 4.88 / 9.78 | 0.00012 | 0.00012 | 1.46e-06 / 1.81e-06 (0.02) | heavier |
| regnety_160.tv2_in1k | fp16 | 17.5 / 18.6 | 0.94 | 0.04 / 0.99 | 198 / 280 | 0.71 | 160 / 160 | 18.0 / 19.0 | 0.73 / 0.57 | 0.04 / 1.00 | 0.00029 | 0.00029 | - |  |
| regnety_160.tv2_in1k | fp32 | 18.4 / 23.6 | 0.78 | 0.05 / 0.96 | 377 / 574 | 0.66 | 319 / 319 | 18.9 / 24.1 | 0.66 / 0.56 | 0.05 / 0.96 | 3.5e-07 | 3.5e-07 | - |  |
| repvit_m0_9.dist_300e_in1k | fp16 | 3.5 / 4.5 | 0.77 | 0.03 / 1.31 | 18 / 35 | 0.51 | 10 / 11 | 4.0 / 5.0 | 0.79 / 0.70 | 0.03 / 1.32 | 0.0078 | 0.0078 | - |  |
| repvit_m0_9.dist_300e_in1k | fp32 | 3.9 / 4.9 | 0.79 | 0.05 / 1.26 | 31 / 62 | 0.50 | 20 / 21 | 4.2 / 6.3 | 0.75 / 0.70 | 0.05 / 1.26 | 1.4e-06 | 1.4e-06 | - |  |
| repvit_m0_9.dist_450e_in1k | fp16 | 3.5 / 4.5 | 0.78 | 0.03 / 1.36 | 17 / 33 | 0.51 | 10 / 11 | 3.9 / 5.0 | 0.78 / 0.71 | 0.03 / 1.36 | 0.012 (0.2 % off) | 0.012 (0.2 % off) | - |  |
| repvit_m0_9.dist_450e_in1k | fp32 | 3.7 / 4.8 | 0.77 | 0.05 / 1.32 | 32 / 61 | 0.53 | 20 / 21 | 4.3 / 5.4 | 0.76 / 0.74 | 0.05 / 1.32 | 2.3e-06 | 2.3e-06 | - |  |
| repvit_m1_0.dist_300e_in1k | fp16 | 3.8 / 5.0 | 0.76 | 0.03 / 1.31 | 21 / 41 | 0.52 | 14 / 15 | 4.3 / 5.5 | 0.79 / 0.69 | 0.04 / 1.31 | 0.01 (0.1 % off) | 0.01 (0.1 % off) | - |  |
| repvit_m1_0.dist_300e_in1k | fp32 | 4.1 / 5.3 | 0.77 | 0.05 / 1.26 | 40 / 78 | 0.51 | 27 / 28 | 4.6 / 5.9 | 0.73 / 0.70 | 0.05 / 1.27 | 1.9e-06 | 1.9e-06 | - |  |
| repvit_m1_0.dist_450e_in1k | fp16 | 3.7 / 4.9 | 0.76 | 0.03 / 1.32 | 21 / 45 | 0.48 | 14 / 15 | 4.2 / 5.5 | 0.79 / 0.71 | 0.03 / 1.32 | 0.0065 | 0.0065 | - |  |
| repvit_m1_0.dist_450e_in1k | fp32 | 4.1 / 5.3 | 0.78 | 0.05 / 1.28 | 40 / 77 | 0.51 | 27 / 28 | 4.5 / 5.8 | 0.73 / 0.69 | 0.05 / 1.28 | 1.3e-06 | 1.3e-06 | - |  |
| repvit_m1_1.dist_300e_in1k | fp16 | 3.6 / 4.6 | 0.78 | 0.03 / 1.25 | 24 / 45 | 0.54 | 16 / 17 | 4.0 / 5.2 | 0.80 / 0.72 | 0.03 / 1.25 | 0.0061 | 0.0061 | - |  |
| repvit_m1_1.dist_300e_in1k | fp32 | 3.9 / 5.2 | 0.76 | 0.05 / 1.20 | 47 / 83 | 0.57 | 32 / 34 | 4.4 / 5.7 | 0.73 / 0.71 | 0.05 / 1.21 | 8.9e-07 | 8.9e-07 | - |  |
| repvit_m1_1.dist_450e_in1k | fp16 | 3.5 / 4.6 | 0.77 | 0.03 / 1.24 | 24 / 46 | 0.53 | 16 / 17 | 4.0 / 5.1 | 0.78 / 0.70 | 0.03 / 1.24 | 0.021 (5.2 % off) | 0.021 (5.2 % off) | - |  |
| repvit_m1_1.dist_450e_in1k | fp32 | 3.8 / 5.0 | 0.76 | 0.05 / 1.25 | 46 / 83 | 0.56 | 32 / 34 | 4.4 / 5.5 | 0.73 / 0.73 | 0.05 / 1.25 | 2.6e-06 | 2.6e-06 | - |  |
| repvit_m1_5.dist_300e_in1k | fp16 | 5.4 / 7.3 | 0.74 | 0.04 / 1.87 | 39 / 64 | 0.61 | 27 / 28 | 6.0 / 8.4 | 0.86 / 0.75 | 0.04 / 1.87 | 0.0073 | 0.0073 | - |  |
| repvit_m1_5.dist_300e_in1k | fp32 | 6.1 / 8.6 | 0.71 | 0.06 / 1.75 | 73 / 120 | 0.61 | 54 / 56 | 6.5 / 9.0 | 0.80 / 0.75 | 0.06 / 1.75 | 1.2e-06 | 1.2e-06 | - |  |
| repvit_m1_5.dist_450e_in1k | fp16 | 5.5 / 7.6 | 0.73 | 0.04 / 1.79 | 39 / 66 | 0.59 | 27 / 28 | 6.2 / 8.2 | 0.86 / 0.73 | 0.04 / 1.79 | 0.037 (12 % off) | 0.037 (12 % off) | - |  |
| repvit_m1_5.dist_450e_in1k | fp32 | 6.2 / 8.4 | 0.73 | 0.06 / 1.71 | 74 / 126 | 0.58 | 54 / 56 | 6.7 / 9.0 | 0.76 / 0.74 | 0.06 / 1.71 | 2e-06 | 2e-06 | - |  |
| repvit_m2_3.dist_300e_in1k | fp16 | 9.4 / 12.0 | 0.78 | 0.04 / 2.41 | 63 / 108 | 0.59 | 45 / 46 | 9.9 / 13.0 | 0.93 / 0.72 | 0.04 / 2.41 | 0.0098 | 0.0098 | - |  |
| repvit_m2_3.dist_300e_in1k | fp32 | 10.4 / 13.9 | 0.75 | 0.06 / 2.18 | 120 / 194 | 0.61 | 88 / 91 | 11.1 / 14.8 | 0.80 / 0.73 | 0.07 / 2.18 | 1.2e-06 | 1.2e-06 | - |  |
| repvit_m2_3.dist_450e_in1k | fp16 | 9.1 / 12.3 | 0.74 | 0.04 / 2.33 | 63 / 109 | 0.58 | 45 / 46 | 9.7 / 13.0 | 0.91 / 0.73 | 0.04 / 2.33 | 0.007 | 0.007 | - |  |
| repvit_m2_3.dist_450e_in1k | fp32 | 10.6 / 13.9 | 0.76 | 0.07 / 2.22 | 120 / 194 | 0.62 | 88 / 91 | 11.2 / 14.6 | 0.85 / 0.73 | 0.07 / 2.22 | 9.2e-07 | 9.2e-07 | - |  |
| resnet152.tv2_in1k | fp16 | 15.7 / 17.2 | 0.91 | 0.03 / 1.44 | 177 / 224 | 0.79 | 115 / 115 | 16.1 / 17.9 | 0.78 / 0.65 | 0.04 / 1.44 | 0.00062 | 0.00062 | - |  |
| resnet152.tv2_in1k | fp32 | 17.8 / 20.7 | 0.86 | 0.05 / 1.41 | 335 / 384 | 0.87 | 229 / 229 | 18.2 / 21.2 | 0.71 / 0.61 | 0.05 / 1.41 | 4.6e-07 | 4.6e-07 | - |  |
| resnet18.tv_in1k | fp16 | 3.4 / 3.9 | 0.88 | 0.02 / 0.49 | 30 / 52 | 0.58 | 22 / 22 | 3.7 / 4.2 | 0.54 / 0.49 | 0.02 / 0.49 | 0.00055 | 0.00055 | - |  |
| resnet18.tv_in1k | fp32 | 3.9 / 4.5 | 0.86 | 0.03 / 0.48 | 59 / 102 | 0.58 | 45 / 45 | 4.3 / 4.9 | 0.51 / 0.51 | 0.03 / 0.48 | 3.7e-07 | 3.7e-07 | - |  |
| resnet50.tv2_in1k | fp16 | 6.2 / 6.8 | 0.91 | 0.02 / 0.92 | 76 / 99 | 0.77 | 49 / 49 | 6.5 / 7.2 | 0.59 / 0.50 | 0.02 / 0.92 | 0.00082 | 0.00082 | - |  |
| resnet50.tv2_in1k | fp32 | 7.0 / 8.1 | 0.87 | 0.04 / 0.76 | 147 / 192 | 0.77 | 97 / 97 | 7.4 / 8.5 | 0.55 / 0.51 | 0.04 / 0.76 | 5.5e-07 | 5.5e-07 | - |  |
| rf-detr-keypoint-preview | fp16 | 63.7 / 61.9 | 1.03 | 0.11 / 3.19 | 244 / 225 | 1.09 | 95 / 71 | 64.7 / 63.7 | 2.03 / 1.97 | 0.11 / 3.19 | - | empty | 0.0079 / 0.0358 (0.08) | slower heavier |
| rf-detr-keypoint-preview | fp32 | 73.7 / 74.9 | 0.98 | 0.12 / 3.15 | 316 / 248 | 1.28 | 138 / 142 | 74.6 / 76.7 | 2.07 / 1.94 | 0.12 / 3.15 | - | empty | 2.36e-05 / 2.02e-05 (0.08) | heavier |
| rf-detr-l-coco | fp16 | 47.3 / 44.4 | 1.06 | 0.29 / 2.83 | 196 / 191 | 1.03 | 73 / 60 | 51.8 / 51.5 | 2.79 / 3.72 | 0.30 / 2.84 | 0.98 (69 % off) | empty | 0.124 / 0.121 (0.32) | slower heavier |
| rf-detr-l-coco | fp32 | 56.3 / 54.1 | 1.04 | 0.49 / 2.91 | 262 / 226 | 1.16 | 116 / 120 | 60.7 / 58.9 | 4.62 / 2.98 | 0.51 / 2.92 | 0.42 (3.6 % off) | empty | 3.39e-05 / 1.81e-05 (0.32) | slower heavier |
| rf-detr-m-coco | fp16 | 32.7 / 32.4 | 1.01 | 0.25 / 2.35 | 173 / 179 | 0.97 | 73 / 59 | 36.2 / 34.5 | 2.46 / 2.56 | 0.25 / 2.36 | 0.9 (70 % off) | empty | 0.036 / 0.0519 (0.13) |  |
| rf-detr-m-coco | fp32 | 36.7 / 37.2 | 0.99 | 0.32 / 2.69 | 231 / 226 | 1.02 | 115 / 119 | 40.7 / 41.7 | 2.55 / 3.46 | 0.33 / 2.70 | 0.00048 | empty | 2.91e-05 / 4.29e-05 (0.13) | heavier |
| rf-detr-n-coco | fp16 | 16.3 / 16.1 | 1.02 | 0.23 / 2.14 | 126 / 146 | 0.86 | 61 / 53 | 19.7 / 20.3 | 2.78 / 2.42 | 0.23 / 2.15 | 0.93 (51 % off) | empty | 0.00925 / 0.0127 (0.087) |  |
| rf-detr-n-coco | fp32 | 18.8 / 19.3 | 0.97 | 0.28 / 2.06 | 175 / 190 | 0.92 | 102 / 106 | 22.0 / 22.3 | 2.53 / 2.42 | 0.29 / 2.06 | 0.75 (0.56 % off) | empty | 1.99e-05 / 3.36e-05 (0.087) |  |
| rf-detr-s-coco | fp16 | 26.2 / 24.2 | 1.08 | 0.08 / 2.08 | 130 / 165 | 0.79 | 67 / 56 | 28.8 / 27.6 | 2.33 / 2.14 | 0.09 / 2.08 | 0.95 (71 % off) | empty | 0.00598 / 0.0223 (0.13) | slower |
| rf-detr-s-coco | fp32 | 28.8 / 29.4 | 0.98 | 0.16 / 1.94 | 207 / 219 | 0.95 | 109 / 113 | 31.8 / 32.6 | 2.35 / 2.22 | 0.16 / 1.95 | 0.86 (1.4 % off) | empty | 7.63e-06 / 1.53e-05 (0.13) |  |
| rf-detr-seg-2xl-coco | fp16 | 130.9 / 125.6 | 1.04 | 0.35 / 2.68 | 385 / 274 | 1.41 | 88 / 69 | 134.3 / 130.3 | 2.68 / 2.27 | 0.35 / 2.69 | 1 (73 % off) | empty | 0.278 / 0.244 (0.51) | slower heavier worse |
| rf-detr-seg-2xl-coco | fp32 | 163.5 / 164.4 | 0.99 | 0.32 / 2.81 | 482 / 433 | 1.11 | 134 / 138 | 166.6 / 168.8 | 2.65 / 2.26 | 0.33 / 2.81 | 0.00069 | empty | 7.3e-05 / 5.33e-05 (0.51) | heavier |
| rf-detr-seg-l-coco | fp16 | 49.9 / 46.8 | 1.07 | 0.21 / 3.32 | 188 / 231 | 0.81 | 80 / 64 | 53.0 / 49.3 | 2.43 / 2.38 | 0.22 / 3.33 | 0.96 (84 % off) | empty | 0.187 / 0.497 (0.52) | slower |
| rf-detr-seg-l-coco | fp32 | 58.8 / 60.1 | 0.98 | 0.34 / 2.78 | 267 / 301 | 0.89 | 124 / 127 | 60.1 / 62.8 | 2.44 / 2.51 | 0.34 / 2.79 | 0.00097 | empty | 6.07e-05 / 4.48e-05 (0.52) |  |
| rf-detr-seg-m-coco | fp16 | 39.3 / 36.3 | 1.08 | 0.10 / 2.35 | 154 / 211 | 0.73 | 80 / 63 | 41.2 / 38.8 | 2.30 / 2.12 | 0.10 / 2.36 | 0.93 (86 % off) | empty | 0.289 / 0.214 (0.37) | slower worse |
| rf-detr-seg-m-coco | fp32 | 46.3 / 42.8 | 1.08 | 0.10 / 2.07 | 243 / 289 | 0.84 | 124 / 126 | 48.1 / 45.2 | 2.37 / 1.94 | 0.10 / 2.07 | 0.0042 | empty | 2.79e-05 / 4.51e-05 (0.37) | slower |
| rf-detr-seg-n-coco | fp16 | 23.2 / 21.9 | 1.06 | 0.29 / 2.38 | 128 / 150 | 0.85 | 74 / 59 | 24.7 / 24.4 | 2.66 / 2.21 | 0.29 / 2.39 | 0.94 (77 % off) | empty | 0.0535 / 0.223 (0.34) | slower |
| rf-detr-seg-n-coco | fp32 | 26.8 / 26.1 | 1.03 | 0.29 / 2.44 | 185 / 249 | 0.74 | 117 / 118 | 29.0 / 28.2 | 2.59 / 2.27 | 0.30 / 2.45 | 0.00012 | empty | 5.99e-05 / 0.00012 (0.34) | slower |
| rf-detr-seg-s-coco | fp16 | 29.7 / 27.5 | 1.08 | 0.31 / 2.43 | 145 / 176 | 0.82 | 74 / 59 | 31.4 / 30.4 | 2.57 / 2.16 | 0.31 / 2.43 | 0.93 (79 % off) | empty | 0.102 / 0.119 (0.24) | slower |
| rf-detr-seg-s-coco | fp32 | 36.1 / 35.2 | 1.03 | 0.28 / 2.52 | 213 / 266 | 0.80 | 117 / 119 | 43.6 / 41.7 | 2.48 / 2.33 | 0.29 / 2.52 | 0.001 | empty | 5.24e-05 / 1.62e-05 (0.24) | slower |
| rf-detr-seg-xl-coco | fp16 | 83.1 / 79.8 | 1.04 | 0.36 / 2.83 | 240 / 251 | 0.96 | 87 / 68 | 85.5 / 92.3 | 2.52 / 2.27 | 0.39 / 2.84 | 0.99 (79 % off) | empty | 0.462 / 0.682 (1.1) | slower |
| rf-detr-seg-xl-coco | fp32 | 98.8 / 106.1 | 0.93 | 0.33 / 3.05 | 319 / 331 | 0.97 | 132 / 136 | 103.6 / 110.7 | 3.09 / 2.52 | 0.34 / 3.06 | 0.00081 | empty | 0.073 / 0.0305 (1.1) | worse |
| rtdetrv2-l-coco | fp16 | 46.9 / 52.8 | 0.89 | 0.07 / 2.36 | 264 / 259 | 1.02 | 83 / 83 | 54.6 / 60.8 | 0.29 / 0.12 | 0.08 / 2.38 | 0.95 (61 % off) | 0.84 (3.3 % off) | - |  |
| rtdetrv2-l-coco | fp32 | 55.3 / 63.4 | 0.87 | 0.09 / 2.19 | 455 / 424 | 1.07 | 161 / 161 | 63.0 / 71.6 | 0.20 / 0.12 | 0.10 / 2.20 | 1.3e-05 | 1.5e-06 | - | heavier |
| rtdetrv2-m-coco | fp16 | 35.7 / 39.2 | 0.91 | 0.06 / 2.00 | 224 / 258 | 0.87 | 62 / 62 | 43.8 / 47.3 | 0.23 / 0.10 | 0.07 / 2.02 | 0.97 (45 % off) | empty (3 keys differ) | - |  |
| rtdetrv2-m-coco | fp32 | 42.0 / 47.0 | 0.89 | 0.08 / 1.89 | 418 / 453 | 0.92 | 119 / 119 | 49.8 / 54.8 | 0.16 / 0.10 | 0.09 / 1.90 | 4.1e-05 | 1.8e-06 | - |  |
| rtdetrv2-ms-coco | fp16 | 37.4 / 42.2 | 0.89 | 0.06 / 1.98 | 246 / 242 | 1.02 | 65 / 65 | 45.3 / 50.1 | 0.25 / 0.12 | 0.07 / 2.00 | 0.98 (52 % off) | 0.63 (2.5 % off) | - |  |
| rtdetrv2-ms-coco | fp32 | 44.5 / 51.5 | 0.86 | 0.08 / 1.91 | 412 / 391 | 1.05 | 126 / 126 | 51.8 / 59.2 | 0.17 / 0.11 | 0.09 / 1.92 | 1.1e-05 | 1.3e-06 | - | heavier |
| rtdetrv2-s-coco | fp16 | 24.8 / 27.4 | 0.90 | 0.05 / 1.88 | 150 / 244 | 0.61 | 40 / 40 | 33.1 / 35.1 | 0.19 / 0.08 | 0.06 / 1.90 | 0.97 (29 % off) | empty (3 keys differ) | - |  |
| rtdetrv2-s-coco | fp32 | 28.7 / 33.3 | 0.86 | 0.07 / 1.72 | 271 / 362 | 0.75 | 77 / 77 | 40.2 / 41.5 | 0.13 / 0.08 | 0.08 / 1.73 | 1.6e-05 | 1.1e-05 | - |  |
| rtdetrv2-x-coco | fp16 | 79.4 / 86.6 | 0.92 | 0.17 / 2.54 | 374 / 389 | 0.96 | 146 / 146 | 88.5 / 94.9 | 0.38 / 0.17 | 0.18 / 2.55 | 0.98 (37 % off) | 0.42 (1.6 % off) | - |  |
| rtdetrv2-x-coco | fp32 | 92.5 / 103.2 | 0.90 | 0.14 / 2.40 | 736 / 679 | 1.08 | 286 / 286 | 100.3 / 110.9 | 0.28 / 0.16 | 0.14 / 2.41 | 2.4e-05 | 4.5e-07 | - | heavier |
| rtmdet-l-coco | fp16 | 52.2 / 63.1 | 0.83 | 0.08 / 1.60 | 292 / 307 | 0.95 | 112 / 109 | 58.0 / 68.3 | 0.05 / 0.17 | 0.55 / 1.67 | 0.7 (68 % off) | empty (3 keys differ) | - |  |
| rtmdet-l-coco | fp32 | 61.2 / 77.5 | 0.79 | 0.09 / 1.54 | 554 / 573 | 0.97 | 218 / 218 | 67.2 / 82.8 | 0.06 / 0.17 | 0.56 / 1.61 | 0.7 (68 % off) | 4.3e-06 | - |  |
| rtmdet-m-coco | fp16 | 33.5 / 39.7 | 0.84 | 0.07 / 1.40 | 179 / 242 | 0.74 | 54 / 53 | 39.3 / 45.2 | 0.04 / 0.12 | 0.54 / 1.47 | 0.81 (62 % off) | 0.0018 | - |  |
| rtmdet-m-coco | fp32 | 37.1 / 46.2 | 0.80 | 0.07 / 1.37 | 326 / 379 | 0.86 | 105 / 105 | 43.0 / 51.7 | 0.04 / 0.12 | 0.54 / 1.45 | 0.81 (62 % off) | 1.1e-06 | - |  |
| rtmdet-m-person | fp16 | 33.4 / 39.3 | 0.85 | 0.07 / 1.41 | 179 / 240 | 0.75 | 54 / 53 | 37.9 / 43.8 | 0.04 / 0.08 | 0.08 / 1.42 | 0.046 (0.13 % off) | empty | - |  |
| rtmdet-m-person | fp32 | 37.0 / 46.2 | 0.80 | 0.08 / 1.39 | 318 / 379 | 0.84 | 104 / 104 | 41.8 / 50.6 | 0.04 / 0.09 | 0.09 / 1.39 | 2e-05 | empty | - |  |
| rtmdet-nano-person | fp16 | 4.0 / 6.4 | 0.63 | 0.04 / 1.37 | 14 / 32 | 0.46 | 2 / 2 | 6.7 / 8.5 | 0.04 / 0.04 | 0.50 / 1.43 | 0.062 (1.3 % off) | 0.4 (14 % off) | - |  |
| rtmdet-nano-person | fp32 | 4.2 / 6.9 | 0.62 | 0.05 / 1.28 | 23 / 54 | 0.43 | 4 / 4 | 6.1 / 9.1 | 0.04 / 0.04 | 0.50 / 1.35 | 6.3e-06 | 1e-06 | - |  |
| rtmdet-s-coco | fp16 | 14.3 / 19.0 | 0.75 | 0.05 / 1.25 | 96 / 216 | 0.44 | 20 / 19 | 20.0 / 24.1 | 0.04 / 0.08 | 0.53 / 1.31 | 0.75 (67 % off) | 0.022 (17 % off) | - |  |
| rtmdet-s-coco | fp32 | 16.0 / 22.8 | 0.70 | 0.06 / 1.22 | 181 / 243 | 0.74 | 38 / 38 | 21.8 / 27.9 | 0.04 / 0.08 | 0.52 / 1.29 | 0.76 (67 % off) | 2.6e-06 | - |  |
| rtmdet-tiny-coco | fp16 | 12.9 / 16.1 | 0.80 | 0.05 / 1.21 | 66 / 180 | 0.37 | 12 / 11 | 18.7 / 21.3 | 0.04 / 0.07 | 0.51 / 1.28 | 0.72 (67 % off) | 0.55 (1.8 % off) | - |  |
| rtmdet-tiny-coco | fp32 | 15.1 / 18.9 | 0.80 | 0.05 / 1.18 | 105 / 262 | 0.40 | 22 / 22 | 21.2 / 24.5 | 0.04 / 0.07 | 0.51 / 1.25 | 0.73 (66 % off) | 1.6e-06 | - |  |
| rtmdet-x-coco | fp16 | 102.2 / 111.6 | 0.92 | 0.13 / 1.79 | 421 / 470 | 0.90 | 200 / 196 | 107.8 / 116.9 | 0.06 / 0.20 | 0.60 / 1.86 | 0.79 (65 % off) | empty (3 keys differ) | - |  |
| rtmdet-x-coco | fp32 | 116.5 / 131.7 | 0.88 | 0.17 / 1.74 | 808 / 913 | 0.89 | 390 / 390 | 122.6 / 137.4 | 0.06 / 0.21 | 0.63 / 1.81 | 0.79 (65 % off) | 1.9e-06 | - |  |
| rtmo-l | fp16 | 47.0 / 55.7 | 0.84 | 0.07 / 1.56 | 401 / 414 | 0.97 | 82 / 82 | 52.9 / 61.4 | 0.27 / 0.14 | 0.07 / 1.57 | 0.011 (0.00028 % off) | empty | - |  |
| rtmo-l | fp32 | 52.4 / 64.7 | 0.81 | 0.08 / 1.51 | 537 / 527 | 1.02 | 163 / 163 | 58.5 / 70.4 | 0.20 / 0.14 | 0.09 / 1.52 | 1e-05 | empty | - |  |
| rtmo-m | fp16 | 27.7 / 34.7 | 0.80 | 0.06 / 1.43 | 165 / 206 | 0.80 | 41 / 41 | 33.6 / 40.0 | 0.20 / 0.09 | 0.06 / 1.44 | 0.0034 | empty | - |  |
| rtmo-m | fp32 | 30.1 / 39.1 | 0.77 | 0.07 / 1.39 | 278 / 313 | 0.89 | 82 / 82 | 35.8 / 44.2 | 0.13 / 0.09 | 0.08 / 1.40 | 1.2e-05 | empty | - |  |
| rtmo-s | fp16 | 13.3 / 17.3 | 0.77 | 0.05 / 1.30 | 82 / 199 | 0.41 | 18 / 18 | 18.8 / 22.5 | 0.14 / 0.06 | 0.05 / 1.31 | 0.017 (0.0018 % off) | empty | - |  |
| rtmo-s | fp32 | 14.7 / 20.2 | 0.73 | 0.06 / 1.28 | 128 / 226 | 0.56 | 35 / 35 | 20.3 / 25.3 | 0.09 / 0.06 | 0.07 / 1.29 | 1.4e-05 | empty | - |  |
| rtmo-t | fp16 | 7.2 / 9.6 | 0.75 | 0.04 / 1.29 | 35 / 88 | 0.40 | 12 / 12 | 11.6 / 14.0 | 0.13 / 0.06 | 0.05 / 1.30 | 0.015 (0.0011 % off) | empty | - |  |
| rtmo-t | fp32 | 7.7 / 10.6 | 0.73 | 0.05 / 1.26 | 65 / 155 | 0.42 | 24 / 24 | 12.3 / 14.8 | 0.08 / 0.06 | 0.06 / 1.26 | 1.9e-05 | empty | - |  |
| rtmpose-l | fp16 | 48.6 / 60.8 | 0.80 | 0.07 / 1.34 | 310 / 267 | 1.16 | 58 / 56 | 57.0 / 68.8 | 0.19 / 0.08 | 0.08 / 1.35 | - | 0.17 (0.36 % off) | - | heavier |
| rtmpose-l | fp32 | 54.5 / 73.8 | 0.74 | 0.08 / 1.30 | 565 / 431 | 1.31 | 106 / 106 | 63.4 / 81.9 | 0.14 / 0.08 | 0.09 / 1.31 | - | 7.2e-06 | - | heavier |
| rtmpose-l-384 | fp16 | 95.3 / 118.9 | 0.80 | 0.11 / 1.38 | 387 / 354 | 1.09 | 59 / 56 | 111.7 / 135.1 | 0.19 / 0.08 | 0.13 / 1.40 | - | 0.45 (12 % off) | - | heavier |
| rtmpose-l-384 | fp32 | 107.9 / 149.1 | 0.72 | 0.13 / 1.40 | 743 / 496 | 1.50 | 106 / 106 | 125.2 / 164.2 | 0.14 / 0.08 | 0.15 / 1.41 | - | 6.3e-06 | - | heavier |
| rtmpose-l-halpe26 | fp16 | 48.6 / 61.0 | 0.80 | 0.07 / 1.32 | 311 / 269 | 1.16 | 60 / 58 | 57.1 / 68.6 | 0.19 / 0.08 | 0.08 / 1.33 | - | 0.047 (0.08 % off) | - | heavier |
| rtmpose-l-halpe26 | fp32 | 54.6 / 73.9 | 0.74 | 0.09 / 1.34 | 566 / 433 | 1.31 | 107 / 107 | 63.6 / 81.8 | 0.14 / 0.09 | 0.10 / 1.35 | - | 6.4e-06 | - | heavier |
| rtmpose-l-halpe26-384 | fp16 | 95.6 / 118.8 | 0.80 | 0.11 / 1.39 | 389 / 356 | 1.09 | 61 / 58 | 112.1 / 134.5 | 0.20 / 0.08 | 0.13 / 1.41 | - | 0.28 (4.8 % off) | - | heavier |
| rtmpose-l-halpe26-384 | fp32 | 108.4 / 149.3 | 0.73 | 0.13 / 1.40 | 745 / 510 | 1.46 | 108 / 108 | 125.4 / 165.5 | 0.14 / 0.08 | 0.15 / 1.42 | - | 9.5e-06 | - | heavier |
| rtmpose-m | fp16 | 34.2 / 39.9 | 0.86 | 0.05 / 1.22 | 211 / 220 | 0.96 | 30 / 29 | 42.5 / 47.3 | 0.13 / 0.06 | 0.06 / 1.23 | - | 0.18 (2.5 % off) | - |  |
| rtmpose-m | fp32 | 38.2 / 47.5 | 0.80 | 0.07 / 1.19 | 383 / 412 | 0.93 | 52 / 52 | 46.5 / 55.3 | 0.09 / 0.06 | 0.08 / 1.20 | - | 9e-06 | - |  |
| rtmpose-m-384 | fp16 | 67.9 / 78.9 | 0.86 | 0.09 / 1.27 | 251 / 371 | 0.68 | 31 / 29 | 84.9 / 94.3 | 0.13 / 0.06 | 0.11 / 1.29 | - | 0.042 (1.5 % off) | - |  |
| rtmpose-m-384 | fp32 | 79.6 / 98.4 | 0.81 | 0.11 / 1.25 | 440 / 622 | 0.71 | 53 / 53 | 96.7 / 114.1 | 0.09 / 0.06 | 0.13 / 1.27 | - | 5.2e-06 | - |  |
| rtmpose-m-ap10k | fp16 | 42.6 / 49.9 | 0.85 | 0.06 / 1.22 | 228 / 232 | 0.98 | 30 / 29 | 52.9 / 59.4 | 0.13 / 0.06 | 0.07 / 1.23 | - | 0.019 (0.12 % off) | - |  |
| rtmpose-m-ap10k | fp32 | 48.8 / 61.2 | 0.80 | 0.07 / 1.20 | 402 / 435 | 0.92 | 52 / 52 | 59.2 / 70.9 | 0.09 / 0.06 | 0.09 / 1.21 | - | 2.3e-06 | - |  |
| rtmpose-m-face6 | fp16 | 48.5 / 52.2 | 0.93 | 0.06 / 1.23 | 241 / 244 | 0.98 | 43 / 42 | 58.9 / 61.8 | 0.14 / 0.06 | 0.08 / 1.24 | - | 0.2 (0.23 % off) | - |  |
| rtmpose-m-face6 | fp32 | 54.2 / 63.2 | 0.86 | 0.08 / 1.21 | 414 / 448 | 0.93 | 65 / 65 | 65.3 / 72.8 | 0.10 / 0.06 | 0.10 / 1.23 | - | 2.1e-06 | - |  |
| rtmpose-m-halpe26 | fp16 | 34.2 / 39.9 | 0.86 | 0.05 / 1.20 | 212 / 222 | 0.96 | 32 / 30 | 42.8 / 47.5 | 0.15 / 0.06 | 0.06 / 1.21 | - | 0.45 (1.8 % off) | - |  |
| rtmpose-m-halpe26 | fp32 | 38.4 / 47.7 | 0.81 | 0.07 / 1.18 | 384 / 414 | 0.93 | 53 / 53 | 46.7 / 55.3 | 0.09 / 0.06 | 0.08 / 1.19 | - | 9.1e-06 | - |  |
| rtmpose-m-halpe26-384 | fp16 | 68.1 / 78.6 | 0.87 | 0.09 / 1.26 | 252 / 372 | 0.68 | 32 / 30 | 85.5 / 94.1 | 0.13 / 0.06 | 0.10 / 1.28 | - | 0.43 (2.2 % off) | - |  |
| rtmpose-m-halpe26-384 | fp32 | 80.3 / 98.8 | 0.81 | 0.11 / 1.25 | 441 / 624 | 0.71 | 54 / 54 | 96.6 / 114.4 | 0.09 / 0.06 | 0.13 / 1.27 | - | 6e-06 | - |  |
| rtmpose-m-hand5 | fp16 | 42.8 / 50.3 | 0.85 | 0.06 / 1.23 | 228 / 232 | 0.98 | 31 / 29 | 53.1 / 59.6 | 0.14 / 0.06 | 0.07 / 1.24 | - | 0.0025 | - |  |
| rtmpose-m-hand5 | fp32 | 49.1 / 61.4 | 0.80 | 0.07 / 1.24 | 402 / 436 | 0.92 | 53 / 53 | 59.4 / 71.2 | 0.09 / 0.06 | 0.09 / 1.25 | - | 3.1e-06 | - |  |
| rtmpose-s | fp16 | 13.7 / 19.0 | 0.72 | 0.03 / 1.11 | 162 / 190 | 0.85 | 13 / 13 | 21.9 / 26.6 | 0.09 / 0.04 | 0.04 / 1.12 | - | 0.16 (1.1 % off) | - |  |
| rtmpose-s | fp32 | 14.5 / 22.0 | 0.66 | 0.04 / 1.09 | 248 / 245 | 1.01 | 21 / 21 | 22.9 / 30.3 | 0.06 / 0.04 | 0.06 / 1.10 | - | 7.3e-06 | - |  |
| rtmpose-s-face6 | fp16 | 20.6 / 24.8 | 0.83 | 0.04 / 1.09 | 204 / 179 | 1.14 | 22 / 21 | 30.9 / 34.1 | 0.10 / 0.05 | 0.05 / 1.11 | - | 0.47 (0.21 % off) | - | heavier |
| rtmpose-s-face6 | fp32 | 21.9 / 29.0 | 0.76 | 0.05 / 1.10 | 329 / 248 | 1.33 | 30 / 30 | 32.3 / 38.5 | 0.07 / 0.04 | 0.06 / 1.12 | - | 2.7e-06 | - | heavier |
| rtmpose-s-halpe26 | fp16 | 13.8 / 19.2 | 0.72 | 0.03 / 1.11 | 165 / 191 | 0.86 | 14 / 14 | 22.1 / 27.0 | 0.09 / 0.04 | 0.04 / 1.13 | - | 0.4 (6.3 % off) | - |  |
| rtmpose-s-halpe26 | fp32 | 14.6 / 22.1 | 0.66 | 0.04 / 1.08 | 251 / 246 | 1.02 | 22 / 22 | 22.9 / 29.6 | 0.06 / 0.04 | 0.06 / 1.09 | - | 0.001 | - | heavier |
| rtmpose-t | fp16 | 12.8 / 16.0 | 0.80 | 0.03 / 1.06 | 125 / 189 | 0.66 | 9 / 8 | 21.2 / 23.6 | 0.08 / 0.04 | 0.04 / 1.07 | - | 0.29 (2.7 % off) | - |  |
| rtmpose-t | fp32 | 13.8 / 18.8 | 0.73 | 0.04 / 1.07 | 188 / 275 | 0.68 | 13 / 13 | 22.0 / 26.4 | 0.05 / 0.04 | 0.05 / 1.08 | - | 8.9e-06 | - |  |
| rtmpose-t-face6 | fp16 | 18.8 / 21.0 | 0.90 | 0.04 / 1.06 | 188 / 195 | 0.96 | 15 / 15 | 29.2 / 30.3 | 0.09 / 0.04 | 0.05 / 1.07 | - | 0.24 (0.35 % off) | - |  |
| rtmpose-t-face6 | fp32 | 20.1 / 24.5 | 0.82 | 0.04 / 1.06 | 270 / 288 | 0.94 | 19 / 19 | 30.4 / 33.7 | 0.06 / 0.04 | 0.06 / 1.08 | - | 3.8e-06 | - |  |
| rtmpose-t-halpe26 | fp16 | 12.9 / 16.0 | 0.81 | 0.03 / 1.06 | 128 / 190 | 0.68 | 9 / 9 | 21.4 / 23.8 | 0.08 / 0.04 | 0.04 / 1.07 | - | 0.45 (2.9 % off) | - |  |
| rtmpose-t-halpe26 | fp32 | 13.8 / 18.8 | 0.73 | 0.04 / 1.07 | 191 / 276 | 0.69 | 14 / 14 | 22.2 / 26.7 | 0.05 / 0.04 | 0.05 / 1.08 | - | 6.9e-06 | - |  |
| rtmpose-x-halpe26-384 | fp16 | 198.3 / 214.5 | 0.92 | 0.21 / 1.59 | 553 / 658 | 0.84 | 105 / 100 | 216.4 / 231.0 | 0.26 / 0.12 | 0.23 / 1.61 | - | 0.19 (0.48 % off) | - |  |
| rtmpose-x-halpe26-384 | fp32 | 259.0 / 276.7 | 0.94 | 0.28 / 1.61 | 1006 / 932 | 1.08 | 191 / 191 | 277.8 / 292.3 | 0.19 / 0.11 | 0.31 / 1.63 | - | 0.00078 | - | heavier |
| rtmw-l | fp16 | 101.4 / 108.8 | 0.93 | 0.12 / 1.61 | 393 / 350 | 1.12 | 142 / 139 | 110.2 / 116.9 | 0.28 / 0.11 | 0.13 / 1.63 | - | 0.21 (0.13 % off) (2 keys differ) | - | heavier |
| rtmw-l | fp32 | 111.7 / 125.7 | 0.89 | 0.14 / 1.65 | 678 / 544 | 1.25 | 218 / 218 | 120.6 / 134.8 | 0.21 / 0.11 | 0.16 / 1.67 | - | 3.9e-06 (2 keys differ) | - | heavier |
| rtmw-l-384 | fp16 | 206.4 / 235.8 | 0.88 | 0.23 / 1.74 | 470 / 438 | 1.07 | 142 / 140 | 224.3 / 251.8 | 0.28 / 0.11 | 0.25 / 1.75 | - | 0.43 (0.51 % off) (2 keys differ) | - | heavier |
| rtmw-l-384 | fp32 | 224.6 / 272.5 | 0.82 | 0.25 / 1.74 | 856 / 634 | 1.35 | 219 / 219 | 242.3 / 288.4 | 0.21 / 0.11 | 0.27 / 1.76 | - | 0.00078 (2 keys differ) | - | heavier |
| rtmw-m | fp16 | 71.2 / 72.4 | 0.98 | 0.09 / 1.49 | 278 / 275 | 1.01 | 85 / 84 | 80.4 / 80.4 | 0.20 / 0.08 | 0.10 / 1.50 | - | 0.45 (0.13 % off) (2 keys differ) | - |  |
| rtmw-m | fp32 | 76.8 / 82.4 | 0.93 | 0.10 / 1.54 | 468 / 484 | 0.97 | 123 / 123 | 86.3 / 90.6 | 0.14 / 0.09 | 0.12 / 1.56 | - | 2.3e-06 (2 keys differ) | - |  |
| rtmw-x | fp16 | 176.2 / 206.2 | 0.85 | 0.20 / 1.81 | 560 / 474 | 1.18 | 219 / 215 | 185.8 / 216.2 | 0.41 / 0.18 | 0.20 / 1.83 | - | 0.28 (0.21 % off) (2 keys differ) | - | heavier |
| rtmw-x | fp32 | 203.7 / 240.3 | 0.85 | 0.23 / 1.82 | 968 / 830 | 1.17 | 351 / 351 | 213.3 / 247.0 | 0.33 / 0.18 | 0.24 / 1.83 | - | 5.9e-06 (2 keys differ) | - | heavier |
| rtmw-x-384 | fp16 | 350.0 / 414.6 | 0.84 | 0.37 / 2.04 | 668 / 773 | 0.86 | 220 / 216 | 368.7 / 425.3 | 0.41 / 0.18 | 0.39 / 2.06 | - | 0.2 (5.8 % off) (2 keys differ) | - |  |
| rtmw-x-384 | fp32 | 419.5 / 490.0 | 0.86 | 0.45 / 2.08 | 1174 / 1093 | 1.07 | 352 / 352 | 437.4 / 503.6 | 0.35 / 0.18 | 0.47 / 2.11 | - | 1.3e-05 (2 keys differ) | - | heavier |
| sam-hq-vit-b | fp16 | 303.6 / 393.4 | 0.77 | 0.33 / 2.21 | 622 / 539 | 1.15 | 179 / 181 | 359.0 / 435.2 | 0.18 / 0.06 | 0.40 / 2.29 | 0.0047 | 1 (0.39 % off) | - | heavier |
| sam-hq-vit-b | fp32 | 370.9 / 484.5 | 0.77 | 0.40 / 2.29 | 879 / 1066 | 0.82 | 354 / 362 | 418.1 / 536.4 | 0.14 / 0.05 | 0.47 / 2.35 | 6.7e-06 | 1 (0.00081 % off) | - |  |
| sam-hq-vit-h | fp16 | 1538.9 / 1702.4 | 0.90 | 1.56 / 5.10 | 1762 / 1624 | 1.08 | 1227 / 1228 | 1592.5 / 1750.7 | 0.64 / 0.09 | 1.65 / 5.21 | 0.0049 | 1 (0.0077 % off) | - | heavier |
| sam-hq-vit-h | fp32 | 1881.1 / 2133.4 | 0.88 | 1.96 / 5.79 | 3059 / 3242 | 0.94 | 2449 / 2458 | 1947.3 / 2149.4 | 0.96 / 0.09 | 2.30 / 6.05 | 4.2e-06 | 2.8e-06 | - |  |
| sam-hq-vit-l | fp16 | 825.1 / 939.4 | 0.88 | 0.84 / 3.70 | 1119 / 982 | 1.14 | 597 / 599 | 880.6 / 992.8 | 0.38 / 0.07 | 0.92 / 3.78 | 0.0029 | 1 (0.0052 % off) | - | heavier |
| sam-hq-vit-l | fp32 | 976.4 / 1120.5 | 0.87 | 1.01 / 3.87 | 1778 / 1953 | 0.91 | 1190 / 1199 | 1027.3 / 1167.4 | 0.32 / 0.07 | 1.09 / 3.95 | 3.5e-06 | 4.2e-06 | - |  |
| sam-vit-b | fp16 | 299.6 / 383.8 | 0.78 | 0.32 / 2.10 | 613 / 531 | 1.15 | 175 / 179 | 323.5 / 406.6 | 0.18 / 0.05 | 0.35 / 2.13 | 0.0047 | 1 (0.015 % off) | - | heavier |
| sam-vit-b | fp32 | 365.7 / 480.0 | 0.76 | 0.40 / 2.21 | 865 / 1050 | 0.82 | 351 / 359 | 389.8 / 505.0 | 0.14 / 0.05 | 0.43 / 2.23 | 6.7e-06 | 1 (0.00057 % off) | - |  |
| sam-vit-h | fp16 | 1528.6 / 1692.6 | 0.90 | 1.54 / 4.99 | 1745 / 1611 | 1.08 | 1222 / 1226 | 1553.0 / 1715.7 | 0.62 / 0.08 | 1.58 / 5.02 | 0.0049 | 1 (0.013 % off) | - | heavier |
| sam-vit-h | fp32 | 1873.2 / 2106.1 | 0.89 | 2.26 / 5.74 | 3034 / 3210 | 0.94 | 2443 / 2452 | 1905.9 / 2109.5 | 1.39 / 0.14 | 2.31 / 5.79 | 4.2e-06 | 1 (3e-05 % off) | - |  |
| sam-vit-l | fp16 | 815.8 / 931.0 | 0.88 | 0.83 / 3.62 | 1107 / 972 | 1.14 | 593 / 597 | 837.3 / 954.6 | 0.38 / 0.07 | 0.86 / 3.65 | 0.0029 | 1 (0.00033 % off) | - | heavier |
| sam-vit-l | fp32 | 972.1 / 1110.9 | 0.88 | 1.00 / 3.88 | 1758 / 1933 | 0.91 | 1186 / 1195 | 995.8 / 1131.8 | 0.32 / 0.07 | 1.04 / 3.91 | 3.5e-06 | 3.1e-06 | - |  |
| sam2.1-hiera-base-plus | fp16 | 232.8 / 245.5 | 0.95 | 0.24 / 2.21 | 535 / 478 | 1.12 | 153 / 155 | 250.9 / 264.3 | 0.30 / 0.08 | 0.27 / 2.24 | 0.0049 | 1 (0.002 % off) | - | heavier |
| sam2.1-hiera-base-plus | fp32 | 256.2 / 268.7 | 0.95 | 0.28 / 2.22 | 871 / 808 | 1.08 | 303 / 300 | 275.4 / 290.0 | 0.23 / 0.08 | 0.31 / 2.24 | 1.3e-05 | 8.9e-06 | - | heavier |
| sam2.1-hiera-large | fp16 | 555.8 / 562.2 | 0.99 | 0.57 / 3.55 | 849 / 734 | 1.16 | 426 / 434 | 574.5 / 583.0 | 0.60 / 0.11 | 0.59 / 3.58 | 0.026 (9.5e-05 % off) | 1 (0.0013 % off) | - | heavier |
| sam2.1-hiera-large | fp32 | 600.5 / 608.8 | 0.99 | 0.63 / 3.55 | 1495 / 1317 | 1.14 | 847 / 855 | 619.5 / 628.7 | 0.53 / 0.10 | 0.66 / 3.58 | 0.00018 | 5.9e-05 | - | heavier |
| sam2.1-hiera-small | fp16 | 133.8 / 138.1 | 0.97 | 0.14 / 1.77 | 390 / 362 | 1.08 | 78 / 86 | 153.6 / 157.4 | 0.20 / 0.07 | 0.17 / 1.80 | 0.01 (4.8e-05 % off) | 1 (0.012 % off) | - | heavier |
| sam2.1-hiera-small | fp32 | 155.0 / 152.5 | 1.02 | 0.18 / 1.78 | 599 / 463 | 1.29 | 155 / 163 | 174.3 / 171.4 | 0.15 / 0.07 | 0.21 / 1.81 | 4.7e-06 | 1 (3e-05 % off) | - | heavier |
| sam2.1-hiera-tiny | fp16 | 112.1 / 116.0 | 0.97 | 0.12 / 1.58 | 377 / 349 | 1.08 | 65 / 73 | 130.9 / 134.3 | 0.18 / 0.07 | 0.15 / 1.61 | 0.0077 | 1 (0.027 % off) | - | heavier |
| sam2.1-hiera-tiny | fp32 | 130.7 / 127.7 | 1.02 | 0.15 / 1.58 | 572 / 436 | 1.31 | 128 / 136 | 149.2 / 146.9 | 0.13 / 0.07 | 0.18 / 1.60 | 6.4e-06 | 1.1e-05 | - | slower heavier |
| sam3 | fp16 | 1574.1 / 1439.5 | 1.09 | 1.60 / 5.10 | 2006 / 1385 | 1.45 | 1633 / 892 | 1590.7 / 2235.5 | 0.18 / 1.01 | 1.65 / 7.30 | 0.21 (0.84 % off) | empty | - | slower heavier |
| sam3 | fp32 | 1896.0 / 1741.0 | 1.09 | 2.48 / 5.54 | 3972 / 2551 | 1.56 | 3205 / 1776 | 1909.2 / 2695.2 | 4.13 / 1.00 | 2.58 / 11.98 | 0.19 (0.7 % off) | empty | - | slower heavier |
| sapiens-seg-0.3b-goliath | fp16 | 1140.1 / 1509.1 | 0.76 | 1.18 / 3.15 | 1822 / 1808 | 1.01 | 648 / 648 | 1144.0 / 1829.0 | 0.32 / 0.10 | 1.20 / 3.47 | 0.003 | 0 | - |  |
| sapiens-seg-0.3b-goliath | fp32 | 1399.4 / 1783.6 | 0.78 | 1.82 / 3.42 | 3631 / 3288 | 1.10 | 1295 / 1295 | 1411.4 / 2102.4 | 4.20 / 0.14 | 1.89 / 3.74 | 2.6e-06 | 0 | - | heavier |
| sapiens-seg-1b-goliath | fp16 | 2650.0 / 3004.7 | 0.88 | 2.81 / 7.35 | 3426 / 3730 | 0.92 | 2248 / 2248 | 2655.2 / 3316.0 | 23.74 / 0.34 | 2.84 / 7.72 | 0.002 | 0 | - |  |
| sapiens-seg-1b-goliath | fp32 | 3438.1 / 4241.3 | 0.81 | 3.92 / 8.73 | 6839 / 6491 | 1.05 | 4497 / 4497 | 3239.8 / 4225.5 | 22.68 / 0.33 | 3.95 / 9.07 | 2.1e-06 | 0 | - | heavier |
| segformer-b0-ade20k | fp16 | 20.0 / 17.3 | 1.16 | 0.18 / 1.19 | 158 / 114 | 1.39 | 7 / 7 | 52.1 / 53.9 | 2.31 / 2.14 | 0.22 / 1.26 | 0.001 | 0.5 (0.067 % off) | 0.00608 / 0.00589 (0.02) | slower heavier |
| segformer-b0-ade20k | fp32 | 23.9 / 22.3 | 1.07 | 0.08 / 1.03 | 221 / 149 | 1.48 | 14 / 14 | 55.9 / 58.9 | 2.01 / 2.00 | 0.12 / 1.07 | 8.9e-07 | 0 | 5.81e-06 / 5.86e-06 (0.02) | slower heavier |
| segformer-b2-clothes | fp16 | 27.4 / 24.5 | 1.12 | 0.08 / 1.43 | 181 / 158 | 1.15 | 48 / 48 | 28.7 / 25.8 | 2.11 / 2.05 | 0.09 / 1.43 | 0.00081 | 0 | 0.00874 / 0.00866 (0.02) | slower heavier |
| segformer-b2-clothes | fp32 | 32.5 / 32.1 | 1.01 | 0.10 / 1.38 | 273 / 252 | 1.09 | 96 / 96 | 33.3 / 33.0 | 2.04 / 2.05 | 0.10 / 1.39 | 6.9e-07 | 0 | 0.0086 / 0.0086 (0.02) | heavier |
| segformer-b5-ade20k | fp16 | 225.5 / 206.5 | 1.09 | 0.26 / 3.44 | 409 / 555 | 0.74 | 158 / 158 | 259.3 / 250.3 | 2.30 / 2.18 | 0.30 / 3.49 | 0.00089 | 1 (0.015 % off) | 0.00712 / 0.0066 (0.02) | slower |
| segformer-b5-ade20k | fp32 | 267.8 / 259.6 | 1.03 | 0.33 / 3.51 | 807 / 780 | 1.03 | 314 / 314 | 301.1 / 303.4 | 2.33 / 2.21 | 0.38 / 3.56 | 7.1e-07 | 0 | 3.19e-05 / 3.11e-05 (0.02) | slower heavier |
| segformer-b5-face-parsing | fp16 | 68.7 / 63.6 | 1.08 | 0.11 / 3.32 | 291 / 274 | 1.07 | 158 / 158 | 69.3 / 65.1 | 2.28 / 2.17 | 0.11 / 3.32 | 0.00088 | 0 | 0.0116 / 0.0116 (0.02) | slower heavier |
| segformer-b5-face-parsing | fp32 | 79.7 / 79.5 | 1.00 | 0.14 / 3.15 | 492 / 541 | 0.91 | 314 / 314 | 80.7 / 80.9 | 2.27 / 2.11 | 0.14 / 3.15 | 5.4e-07 | 0 | 0.0112 / 0.0112 (0.02) |  |
| segnext-mscan-b-ade20k | fp16 | 77.9 / 70.1 | 1.11 | 0.11 / 2.27 | 264 / 251 | 1.05 | 54 / 53 | 111.3 / 103.2 | 0.18 / 0.12 | 0.14 / 2.32 | 0.0076 | 0 | 0.0683 / 0.0697 (0.14) | slower heavier |
| segnext-mscan-b-ade20k | fp32 | 88.0 / 87.7 | 1.00 | 0.13 / 2.17 | 324 / 366 | 0.89 | 106 / 106 | 121.6 / 121.1 | 0.18 / 0.12 | 0.17 / 2.21 | 4.2e-06 | 0 | 1.09e-05 / 2.01e-05 (0.14) |  |
| segnext-mscan-l-ade20k | fp16 | 138.7 / 126.9 | 1.09 | 0.16 / 3.05 | 421 / 305 | 1.38 | 95 / 94 | 172.3 / 160.1 | 0.28 / 0.19 | 0.20 / 3.09 | 0.004 | 0.5 (0.0021 % off) | 0.067 / 0.0643 (0.14) | slower heavier |
| segnext-mscan-l-ade20k | fp32 | 156.3 / 158.3 | 0.99 | 0.19 / 2.90 | 518 / 497 | 1.04 | 188 / 187 | 190.5 / 192.6 | 0.27 / 0.19 | 0.23 / 2.94 | 5.5e-06 | 0 | 1.42e-05 / 2.59e-05 (0.14) | heavier |
| segnext-mscan-s-ade20k | fp16 | 43.4 / 37.6 | 1.15 | 0.07 / 1.68 | 237 / 197 | 1.20 | 27 / 27 | 76.8 / 71.1 | 0.12 / 0.08 | 0.11 / 1.72 | 0.0022 | 0.12 (0.2 % off) | 0.0706 / 0.0755 (0.14) | slower heavier |
| segnext-mscan-s-ade20k | fp32 | 49.0 / 46.7 | 1.05 | 0.09 / 1.65 | 271 / 363 | 0.75 | 53 / 53 | 82.6 / 80.8 | 0.12 / 0.07 | 0.13 / 1.69 | 3e-06 | 0 | 7.33e-06 / 1.41e-05 (0.14) | slower |
| segnext-mscan-t-ade20k | fp16 | 32.2 / 26.0 | 1.24 | 0.07 / 1.80 | 118 / 176 | 0.67 | 8 / 9 | 65.5 / 59.7 | 0.11 / 0.08 | 0.11 / 1.84 | 0.0021 | 0.92 (0.00052 % off) | 0.0362 / 0.0347 (0.14) | slower |
| segnext-mscan-t-ade20k | fp32 | 36.5 / 32.6 | 1.12 | 0.07 / 1.71 | 227 / 225 | 1.01 | 17 / 17 | 69.9 / 65.5 | 0.11 / 0.07 | 0.12 / 1.75 | 3.3e-06 | 0 | 1.07e-05 / 1.21e-05 (0.14) | slower |
| selective-igev-eth3d-finetune | fp16 | 3693.7 / 4643.9 | 0.80 | 3.95 / 11.08 | 1873 / 1102 | 1.70 | 25 / 27 | 3657.4 / 4665.0 | 0.17 / 0.10 | 3.96 / 11.09 | 0.5 (0.83 % off) | 0.5 (0.15 % off) | 0.00117 / 0.00125 (0.02) | heavier |
| selective-igev-eth3d-finetune | fp32 | 4788.7 / 6385.9 | 0.75 | 4.81 / 12.64 | 3498 / 2057 | 1.70 | 50 / 51 | 4742.2 / 6321.1 | 0.12 / 0.10 | 4.82 / 12.65 | 0.00073 | 0.00074 | 1.95e-06 / 1.92e-06 (0.02) | heavier |
| selective-igev-eth3d-train | fp16 | 3652.2 / 4644.1 | 0.79 | 3.68 / 11.06 | 1873 / 1102 | 1.70 | 25 / 27 | 3662.4 / 4637.0 | 0.14 / 0.10 | 3.69 / 11.06 | 0.99 (0.49 % off) | 0.99 (0.14 % off) | 0.000911 / 0.00111 (0.02) | heavier |
| selective-igev-eth3d-train | fp32 | 4778.2 / 6294.8 | 0.76 | 4.80 / 12.88 | 3498 / 2057 | 1.70 | 50 / 51 | 4793.4 / 6243.3 | 0.13 / 0.10 | 4.82 / 12.89 | 0.00073 | 0.00073 | 1.36e-06 / 1.38e-06 (0.02) | heavier |
| selective-igev-kitti12 | fp16 | 3643.3 / 4640.5 | 0.79 | 3.68 / 10.96 | 1873 / 1102 | 1.70 | 25 / 27 | 3655.1 / 4662.9 | 0.16 / 0.10 | 3.69 / 10.97 | 0.018 (0.00084 % off) | 0.0037 | 0.000678 / 0.000803 (0.02) | heavier |
| selective-igev-kitti12 | fp32 | 4733.6 / 6386.6 | 0.74 | 4.92 / 12.79 | 3498 / 2057 | 1.70 | 50 / 51 | 4784.6 / 6359.4 | 0.14 / 0.10 | 4.95 / 12.80 | 6.3e-06 | 1.7e-06 | 9.41e-07 / 9.34e-07 (0.02) | heavier |
| selective-igev-kitti15 | fp16 | 3672.6 / 4688.7 | 0.78 | 3.70 / 11.12 | 1873 / 1102 | 1.70 | 25 / 27 | 3676.6 / 4683.3 | 0.15 / 0.10 | 3.71 / 11.12 | 0.015 (0.00071 % off) | 0.00088 | 0.00049 / 0.00073 (0.02) | heavier |
| selective-igev-kitti15 | fp32 | 4725.9 / 6386.9 | 0.74 | 4.87 / 13.19 | 3498 / 2057 | 1.70 | 50 / 51 | 4759.5 / 6383.8 | 0.16 / 0.11 | 4.88 / 13.20 | 9.8e-06 | 1.4e-06 | 9.34e-07 / 9.13e-07 (0.02) | heavier |
| selective-igev-middlebury-finetune | fp16 | 3655.1 / 4663.3 | 0.78 | 3.69 / 11.01 | 1873 / 1102 | 1.70 | 25 / 27 | 3657.2 / 4671.3 | 0.15 / 0.10 | 3.70 / 11.02 | 0.84 (0.49 % off) | 0.84 (0.24 % off) | 0.000861 / 0.00102 (0.02) | heavier |
| selective-igev-middlebury-finetune | fp32 | 4760.9 / 6439.7 | 0.74 | 4.77 / 12.73 | 3498 / 2057 | 1.70 | 50 / 51 | 4800.8 / 6409.6 | 0.12 / 0.10 | 4.78 / 12.74 | 0.001 | 0.001 | 1.21e-06 / 1.22e-06 (0.02) | heavier |
| selective-igev-middlebury-train | fp16 | 3651.5 / 4657.9 | 0.78 | 3.68 / 11.04 | 1873 / 1102 | 1.70 | 25 / 27 | 3661.5 / 4636.4 | 0.15 / 0.10 | 3.69 / 11.05 | 0.61 (0.14 % off) | 0.61 (0.1 % off) | 0.00174 / 0.00235 (0.02) | heavier |
| selective-igev-middlebury-train | fp32 | 4728.9 / 6368.5 | 0.74 | 4.78 / 13.14 | 3498 / 2057 | 1.70 | 50 / 51 | 4759.6 / 6384.2 | 0.13 / 0.10 | 4.80 / 13.15 | 0.0012 | 0.0012 | 3.04e-06 / 2.87e-06 (0.02) | heavier |
| selective-igev-sceneflow | fp16 | 3643.8 / 4680.4 | 0.78 | 3.69 / 11.41 | 1873 / 1102 | 1.70 | 25 / 27 | 3665.5 / 4652.8 | 0.15 / 0.10 | 3.70 / 11.42 | 0.14 (0.099 % off) | 0.098 (0.064 % off) | 0.000883 / 0.00111 (0.02) | heavier |
| selective-igev-sceneflow | fp32 | 4694.9 / 6337.5 | 0.74 | 4.73 / 13.04 | 3498 / 2057 | 1.70 | 50 / 51 | 4727.2 / 6335.2 | 0.12 / 0.11 | 4.74 / 13.05 | 0.0026 | 0.0018 | 1.7e-06 / 4.11e-06 (0.02) | heavier |
| shufflenet_v2_x1_0.tv_in1k | fp16 | 2.5 / 2.5 | 1.02 | 0.02 / 0.80 | 15 / 17 | 0.90 | 4 / 4 | 2.9 / 2.9 | 0.64 / 0.52 | 0.02 / 0.80 | 0.0065 | 0.0065 | 0.00517 / 0.00363 (0.02) | slower worse |
| shufflenet_v2_x1_0.tv_in1k | fp32 | 2.7 / 2.5 | 1.08 | 0.02 / 0.80 | 27 / 30 | 0.89 | 9 / 9 | 3.1 / 2.9 | 0.58 / 0.54 | 0.02 / 0.81 | 1.2e-06 | 1.2e-06 | 8.09e-07 / 1.06e-06 (0.02) | slower |
| shvit_s1.in1k | fp16 | 3.3 / 3.0 | 1.11 | 0.02 / 1.20 | 16 / 20 | 0.80 | 12 / 12 | 4.1 / 3.5 | 0.79 / 0.71 | 0.03 / 1.21 | 0.0016 | 0.0016 | - | slower |
| shvit_s1.in1k | fp32 | 3.5 / 3.3 | 1.06 | 0.04 / 1.18 | 29 / 35 | 0.84 | 24 / 24 | 4.0 / 3.8 | 0.73 / 0.70 | 0.04 / 1.18 | 1.3e-06 | 1.3e-06 | - | slower |
| shvit_s2.in1k | fp16 | 3.5 / 3.4 | 1.06 | 0.03 / 1.22 | 26 / 31 | 0.83 | 22 / 22 | 4.0 / 3.9 | 0.78 / 0.69 | 0.04 / 1.22 | 0.0023 | 0.0023 | - | slower |
| shvit_s2.in1k | fp32 | 4.0 / 3.7 | 1.07 | 0.05 / 1.20 | 48 / 56 | 0.86 | 44 / 44 | 4.4 / 4.2 | 0.73 / 0.70 | 0.05 / 1.20 | 7.7e-07 | 7.7e-07 | - | slower |
| shvit_s3.in1k | fp16 | 4.1 / 3.9 | 1.07 | 0.03 / 1.27 | 33 / 41 | 0.82 | 28 / 28 | 4.6 / 4.4 | 0.79 / 0.71 | 0.03 / 1.28 | 0.0014 | 0.0014 | - | slower |
| shvit_s3.in1k | fp32 | 4.6 / 4.2 | 1.08 | 0.05 / 1.25 | 62 / 74 | 0.84 | 54 / 54 | 5.1 / 4.9 | 0.76 / 0.71 | 0.05 / 1.25 | 7.2e-07 | 7.2e-07 | - | slower |
| shvit_s4.in1k | fp16 | 4.9 / 4.7 | 1.04 | 0.04 / 1.46 | 40 / 52 | 0.78 | 32 / 32 | 5.5 / 5.3 | 0.81 / 0.71 | 0.04 / 1.46 | 0.002 | 0.002 | - | slower |
| shvit_s4.in1k | fp32 | 5.5 / 5.2 | 1.05 | 0.05 / 1.43 | 75 / 94 | 0.80 | 64 / 64 | 6.1 / 5.8 | 0.78 / 0.71 | 0.06 / 1.43 | 5e-07 | 5e-07 | - | slower |
| siglip2-b16-256 | fp16 | 14.5 / 14.4 | 1.01 | 0.04 / 0.70 | 197 / 244 | 0.81 | 176 / 177 | 14.8 / 14.6 | 0.84 / 0.70 | 0.05 / 0.70 | 0.0065 | 0.0065 | - |  |
| siglip2-b16-256 | fp32 | 16.8 / 16.7 | 1.00 | 0.06 / 0.70 | 387 / 444 | 0.87 | 352 / 354 | 17.0 / 17.0 | 0.78 / 0.71 | 0.06 / 0.70 | 4.8e-06 | 4.8e-06 | - |  |
| siglip2-b16-512 | fp16 | 14.5 / 14.3 | 1.02 | 0.04 / 0.72 | 197 / 244 | 0.81 | 176 / 177 | 14.8 / 14.5 | 0.85 / 0.71 | 0.04 / 0.72 | 0.0069 | 0.0069 | - |  |
| siglip2-b16-512 | fp32 | 16.8 / 16.7 | 1.01 | 0.07 / 0.70 | 387 / 444 | 0.87 | 352 / 354 | 17.0 / 17.1 | 0.77 / 0.70 | 0.07 / 0.70 | 3.6e-06 | 3.6e-06 | - |  |
| siglip2-g16-256 | fp16 | 154.2 / 148.5 | 1.04 | 0.18 / 2.02 | 2282 / 2266 | 1.01 | 2214 / 2218 | 154.5 / 149.0 | 4.44 / 0.74 | 0.18 / 2.02 | 0.076 (0.33 % off) | 0.076 (0.33 % off) | - | slower |
| siglip2-g16-256 | fp32 | 184.2 / 180.5 | 1.02 | 0.55 / 2.70 | 4497 / 4515 | 1.00 | 4429 / 4438 | 184.4 / 180.7 | 5.34 / 1.01 | 0.55 / 2.71 | 1.3e-05 | 1.3e-05 | - | slower |
| siglip2-g16-384 | fp16 | 154.2 / 148.7 | 1.04 | 0.18 / 2.00 | 2282 / 2266 | 1.01 | 2214 / 2218 | 154.4 / 148.9 | 6.37 / 0.74 | 0.18 / 2.00 | 0.029 (0.13 % off) | 0.029 (0.13 % off) | - | slower |
| siglip2-g16-384 | fp32 | 184.3 / 182.1 | 1.01 | 0.48 / 2.84 | 4497 / 4515 | 1.00 | 4429 / 4438 | 184.5 / 181.3 | 5.29 / 1.02 | 0.48 / 2.84 | 3.4e-05 | 3.4e-05 | - |  |
| siglip2-l16-256 | fp16 | 44.7 / 44.1 | 1.01 | 0.06 / 1.16 | 623 / 663 | 0.94 | 601 / 602 | 45.1 / 44.2 | 1.02 / 0.72 | 0.07 / 1.16 | 0.089 (0.57 % off) | 0.089 (0.57 % off) | - |  |
| siglip2-l16-256 | fp32 | 52.4 / 51.3 | 1.02 | 0.09 / 1.16 | 1238 / 1301 | 0.95 | 1201 / 1205 | 52.7 / 51.6 | 0.85 / 0.75 | 0.09 / 1.17 | 4.5e-05 | 4.5e-05 | - | slower |
| siglip2-l16-512 | fp16 | 46.3 / 44.2 | 1.05 | 0.06 / 1.17 | 646 / 663 | 0.97 | 601 / 602 | 46.6 / 44.5 | 1.00 / 0.72 | 0.07 / 1.17 | 0.011 (0.0005 % off) | 0.011 (0.0005 % off) | - | slower |
| siglip2-l16-512 | fp32 | 53.5 / 51.6 | 1.04 | 0.09 / 1.17 | 1247 / 1301 | 0.96 | 1201 / 1205 | 53.8 / 51.9 | 0.86 / 0.71 | 0.09 / 1.17 | 1.8e-05 | 1.8e-05 | - | slower |
| siglip2-so400m14-224 | fp16 | 63.9 / 63.2 | 1.01 | 0.08 / 1.36 | 846 / 883 | 0.96 | 814 / 816 | 64.2 / 63.4 | 1.63 / 0.70 | 0.08 / 1.37 | 0.046 (0.014 % off) | 0.046 (0.014 % off) | - |  |
| siglip2-so400m14-224 | fp32 | 73.5 / 72.0 | 1.02 | 0.11 / 1.38 | 1680 / 1740 | 0.97 | 1627 / 1632 | 73.8 / 72.1 | 0.89 / 0.71 | 0.12 / 1.38 | 2.8e-05 | 2.8e-05 | - | slower |
| siglip2-so400m14-378 | fp16 | 64.0 / 62.9 | 1.02 | 0.09 / 1.38 | 846 / 883 | 0.96 | 814 / 816 | 64.2 / 63.4 | 1.78 / 0.71 | 0.09 / 1.38 | 0.02 (0.001 % off) | 0.02 (0.001 % off) | - |  |
| siglip2-so400m14-378 | fp32 | 73.7 / 73.8 | 1.00 | 0.13 / 1.52 | 1680 / 1740 | 0.97 | 1627 / 1632 | 74.5 / 74.2 | 0.97 / 0.74 | 0.13 / 1.52 | 1.7e-05 | 1.7e-05 | - |  |
| siglip2-so400m16-256 | fp16 | 63.4 / 61.2 | 1.04 | 0.08 / 1.39 | 863 / 884 | 0.98 | 814 / 817 | 63.7 / 61.5 | 1.72 / 0.72 | 0.08 / 1.39 | 0.037 (0.0071 % off) | 0.037 (0.0071 % off) | - | slower |
| siglip2-so400m16-256 | fp32 | 72.8 / 70.3 | 1.04 | 0.11 / 1.44 | 1678 / 1740 | 0.96 | 1628 / 1633 | 73.1 / 70.6 | 0.88 / 0.81 | 0.12 / 1.44 | 4.6e-05 | 4.6e-05 | - | slower |
| siglip2-so400m16-512 | fp16 | 63.3 / 61.2 | 1.03 | 0.08 / 1.40 | 863 / 884 | 0.98 | 814 / 817 | 63.6 / 61.5 | 1.97 / 0.73 | 0.09 / 1.40 | 0.0061 | 0.0061 | - | slower |
| siglip2-so400m16-512 | fp32 | 72.0 / 69.6 | 1.03 | 0.11 / 1.45 | 1678 / 1740 | 0.96 | 1628 / 1633 | 72.3 / 69.9 | 0.88 / 0.77 | 0.11 / 1.45 | 9.6e-06 | 9.6e-06 | - | slower |
| squeezenet1_1.tv_in1k | fp16 | 1.8 / 3.0 | 0.61 | 0.02 / 0.45 | 16 / 18 | 0.86 | 2 / 2 | 2.1 / 3.3 | 0.56 / 0.54 | 0.02 / 0.45 | 0 | 0 | 0.00209 / 0.00209 (0.02) |  |
| squeezenet1_1.tv_in1k | fp32 | 1.9 / 2.2 | 0.86 | 0.04 / 0.44 | 28 / 32 | 0.86 | 5 / 5 | 2.0 / 2.3 | 0.51 / 0.53 | 0.05 / 0.45 | 9.2e-08 | 9.2e-08 | 4.19e-07 / 4.23e-07 (0.02) |  |
| starnet_s1.in1k | fp16 | 2.1 / 2.6 | 0.81 | 0.02 / 0.66 | 12 / 24 | 0.51 | 6 / 6 | 2.5 / 3.1 | 0.77 / 0.69 | 0.02 / 0.66 | 0.0015 | 0.0015 | - |  |
| starnet_s1.in1k | fp32 | 2.3 / 3.3 | 0.68 | 0.03 / 0.66 | 24 / 43 | 0.55 | 11 / 11 | 2.7 / 3.6 | 0.71 / 0.71 | 0.03 / 0.66 | 1.6e-06 | 1.6e-06 | - |  |
| starnet_s2.in1k | fp16 | 2.0 / 2.3 | 0.87 | 0.01 / 0.57 | 14 / 24 | 0.59 | 7 / 7 | 2.3 / 2.8 | 0.74 / 0.71 | 0.02 / 0.57 | 0.00093 | 0.00093 | - |  |
| starnet_s2.in1k | fp32 | 2.0 / 2.6 | 0.78 | 0.03 / 0.56 | 27 / 44 | 0.62 | 14 / 14 | 2.5 / 3.1 | 0.72 / 0.69 | 0.04 / 0.56 | 6.2e-07 | 6.2e-07 | - |  |
| starnet_s3.in1k | fp16 | 2.3 / 3.0 | 0.78 | 0.02 / 0.69 | 19 / 34 | 0.55 | 11 / 11 | 2.8 / 3.5 | 0.78 / 0.69 | 0.02 / 0.69 | 0.00078 | 0.00078 | - |  |
| starnet_s3.in1k | fp32 | 2.7 / 3.4 | 0.78 | 0.04 / 0.66 | 37 / 63 | 0.58 | 22 / 22 | 3.1 / 3.8 | 0.71 / 0.69 | 0.04 / 0.67 | 3.8e-07 | 3.8e-07 | - |  |
| starnet_s4.in1k | fp16 | 3.0 / 3.9 | 0.77 | 0.02 / 0.85 | 26 / 46 | 0.56 | 14 / 14 | 3.5 / 4.5 | 0.79 / 0.71 | 0.02 / 0.85 | 0.00086 | 0.00086 | - |  |
| starnet_s4.in1k | fp32 | 3.5 / 4.3 | 0.80 | 0.04 / 0.83 | 48 / 87 | 0.56 | 29 / 29 | 4.0 / 4.8 | 0.73 / 0.71 | 0.04 / 0.84 | 4.2e-07 | 4.2e-07 | - |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp16 | 18.4 / 18.6 | 0.99 | 0.06 / 1.95 | 221 / 263 | 0.84 | 173 / 170 | 18.9 / 19.2 | 0.88 / 0.71 | 0.06 / 1.95 | 0.0029 | 0.0029 | - |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp32 | 20.8 / 22.2 | 0.93 | 0.07 / 1.91 | 435 / 422 | 1.03 | 346 / 340 | 21.4 / 22.2 | 0.84 / 0.71 | 0.08 / 1.92 | 1.6e-06 | 1.6e-06 | - | heavier |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp16 | 4.5 / 5.0 | 0.90 | 0.04 / 1.07 | 43 / 64 | 0.66 | 25 / 25 | 5.0 / 5.4 | 0.85 / 0.81 | 0.04 / 1.07 | 0.0015 | 0.0015 | 0.00527 / 0.00519 (0.0095) |  |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp32 | 6.7 / 5.3 | 1.26 | 0.06 / 0.98 | 82 / 120 | 0.68 | 49 / 49 | 6.4 / 6.0 | 0.90 / 0.82 | 0.06 / 0.99 | 1.2e-06 | 1.2e-06 | 4.9e-06 / 4.94e-06 (0.0095) | slower |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp16 | 6.6 / 6.8 | 0.98 | 0.04 / 0.98 | 72 / 104 | 0.69 | 46 / 46 | 7.2 / 7.3 | 0.84 / 0.73 | 0.04 / 0.98 | 0.0016 | 0.0016 | 0.00254 / 0.00258 (0.0095) |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp32 | 7.5 / 8.0 | 0.93 | 0.06 / 0.95 | 148 / 192 | 0.77 | 92 / 92 | 8.0 / 8.6 | 0.81 / 0.74 | 0.06 / 0.95 | 1e-06 | 1e-06 | 3.91e-06 / 3.86e-06 (0.0095) |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp16 | 14.3 / 14.5 | 0.99 | 0.05 / 1.09 | 162 / 184 | 0.88 | 88 / 88 | 14.7 / 14.8 | 1.19 / 1.09 | 0.05 / 1.09 | 0.001 | 0.001 | 0.00251 / 0.00229 (0.0095) |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp32 | 18.4 / 17.3 | 1.06 | 0.07 / 0.98 | 252 / 289 | 0.87 | 176 / 176 | 18.6 / 17.9 | 1.09 / 1.04 | 0.07 / 0.98 | 6.8e-07 | 6.8e-07 | 4.17e-06 / 3.95e-06 (0.0095) | slower |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 25.3 / 25.2 | 1.00 | 0.06 / 1.01 | 333 / 300 | 1.11 | 191 / 191 | 25.7 / 25.5 | 1.85 / 1.75 | 0.06 / 1.01 | 0.0009 | 0.0009 | 0.00261 / 0.00268 (0.0095) | heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp32 | 32.3 / 31.8 | 1.01 | 0.12 / 0.99 | 537 / 537 | 1.00 | 381 / 381 | 33.1 / 32.1 | 1.87 / 1.70 | 0.12 / 0.99 | 6.3e-07 | 6.3e-07 | 4.43e-06 / 4.31e-06 (0.0095) |  |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp16 | 4.1 / 4.6 | 0.89 | 0.04 / 0.94 | 29 / 43 | 0.67 | 13 / 13 | 4.6 / 4.6 | 0.84 / 0.74 | 0.04 / 0.94 | 0.0014 | 0.0014 | 0.00446 / 0.00407 (0.0095) |  |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp32 | 4.4 / 4.5 | 0.97 | 0.05 / 0.92 | 56 / 81 | 0.70 | 25 / 25 | 4.9 / 5.2 | 0.85 / 0.76 | 0.05 / 0.93 | 8.5e-07 | 8.5e-07 | 3.9e-06 / 3.5e-06 (0.0095) |  |
| totalsegmentator-total | fp16 | 14612.3 / 26280.3 | 0.56 | 14.55 / 33.05 | 3464 / 3912 | 0.89 | 567 / 298 | 17992.2 / 30890.6 | 1.49 / 0.28 | 18.41 / 37.53 | - | 1 (0.0048 % off) (1 keys differ) | - |  |
| totalsegmentator-total | fp32 | 21167.3 / 38793.7 | 0.55 | 21.18 / 42.58 | 3456 / 3954 | 0.87 | 595 / 595 | 25054.6 / 42912.5 | 0.07 / 0.26 | 25.89 / 47.26 | - | 0 (1 keys differ) | - |  |
| totalsegmentator-total-3mm | fp16 | 310.8 / 433.0 | 0.72 | 0.34 / 1.59 | 2058 / 1609 | 1.28 | 45 / 32 | 490.4 / 718.0 | 0.15 / 0.79 | 0.76 / 2.18 | 0.045 (0.0023 % off) | 1 (0.29 % off) | 0.000534 / 0.00061 (0.003) | heavier |
| totalsegmentator-total-3mm | fp32 | 415.4 / 547.4 | 0.76 | 0.44 / 1.65 | 2033 / 1769 | 1.15 | 63 / 63 | 633.9 / 783.1 | 0.04 / 0.76 | 0.80 / 2.01 | 0.00012 | 0.74 (0.002 % off) | 6.61e-05 / 6.61e-05 (0.003) | heavier |
| unik3d-vitl | fp16 | 889.6 / 897.5 | 0.99 | 0.92 / 3.63 | 1523 / 1166 | 1.31 | 715 / 696 | 894.8 / 904.5 | 1.46 / 0.07 | 0.93 / 3.64 | 0.009 | 0.016 (1.1 % off) | - | heavier |
| unik3d-vitl | fp32 | 1214.4 / 1191.7 | 1.02 | 1.36 / 3.91 | 3060 / 3117 | 0.98 | 1376 / 1386 | 1221.6 / 1197.9 | 1.36 / 0.07 | 1.37 / 3.92 | 5.1e-06 | 1.2e-05 | - |  |
| vgg16.tv_in1k | fp16 | 13.2 / 13.6 | 0.97 | 0.03 / 0.47 | 322 / 353 | 0.91 | 264 / 264 | 13.6 / 14.1 | 0.84 / 0.55 | 0.03 / 0.47 | 0 | 0 | - |  |
| vgg16.tv_in1k | fp32 | 16.7 / 17.5 | 0.96 | 0.04 / 0.50 | 643 / 705 | 0.91 | 528 / 528 | 17.3 / 18.0 | 0.79 / 0.57 | 0.04 / 0.50 | 0 | 0 | - |  |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp16 | 15.0 / 14.3 | 1.05 | 0.07 / 0.70 | 186 / 232 | 0.80 | 165 / 165 | 15.4 / 14.5 | 0.06 / 0.80 | 0.07 / 0.70 | 0.0011 | 0.0011 | 0.00117 / 0.00124 (0.02) | slower |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp32 | 16.9 / 17.8 | 0.95 | 0.07 / 0.75 | 365 / 420 | 0.87 | 330 / 330 | 17.1 / 18.3 | 0.06 / 0.80 | 0.07 / 0.75 | 5.8e-07 | 5.8e-07 | 5.86e-06 / 5.9e-06 (0.02) |  |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp16 | 181.8 / 178.1 | 1.02 | 0.22 / 1.50 | 879 / 900 | 0.98 | 817 / 820 | 181.9 / 179.2 | 0.09 / 0.73 | 0.22 / 1.51 | 0.00081 | 0.00081 | 0.00249 / 0.00225 (0.02) | slower |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp32 | 214.1 / 209.9 | 1.02 | 0.30 / 1.58 | 1752 / 1796 | 0.98 | 1634 / 1639 | 214.9 / 210.9 | 0.43 / 0.71 | 0.30 / 1.58 | 5.2e-07 | 5.2e-07 | 1.48e-05 / 1.45e-05 (0.02) | slower |
| wd-convnext-tagger-v3 | fp16 | 42.7 / 43.8 | 0.98 | 0.06 / 1.09 | 274 / 328 | 0.83 | 189 / 189 | 43.2 / 44.1 | 0.97 / 0.73 | 0.06 / 1.10 | 0.00096 | 0.00096 | 0.0032 / 0.00336 (0.015) |  |
| wd-convnext-tagger-v3 | fp32 | 49.8 / 52.3 | 0.95 | 0.09 / 1.06 | 516 / 512 | 1.01 | 377 / 377 | 50.3 / 52.6 | 0.91 / 0.74 | 0.09 / 1.06 | 7.6e-07 | 7.6e-07 | 1.76e-05 / 1.61e-05 (0.015) |  |
| wd-eva02-large-tagger-v3 | fp16 | 195.5 / 198.1 | 0.99 | 0.23 / 2.14 | 699 / 686 | 1.02 | 602 / 602 | 196.3 / 198.6 | 0.07 / 0.80 | 0.23 / 2.14 | 0.001 | 0.001 | 0.0039 / 0.00337 (0.015) |  |
| wd-eva02-large-tagger-v3 | fp32 | 242.3 / 242.0 | 1.00 | 0.29 / 2.21 | 1392 / 1435 | 0.97 | 1203 / 1203 | 242.9 / 242.4 | 0.27 / 0.78 | 0.30 / 2.21 | 6.1e-07 | 6.1e-07 | 2.53e-06 / 2.39e-06 (0.015) |  |
| wd-vit-large-tagger-v3 | fp16 | 179.6 / 175.7 | 1.02 | 0.21 / 1.28 | 677 / 683 | 0.99 | 601 / 601 | 180.2 / 176.1 | 0.07 / 0.74 | 0.21 / 1.28 | 0.0009 | 0.0009 | 0.00328 / 0.00337 (0.015) | slower |
| wd-vit-large-tagger-v3 | fp32 | 203.7 / 198.3 | 1.03 | 0.26 / 1.25 | 1336 / 1401 | 0.95 | 1202 / 1202 | 203.9 / 198.8 | 0.11 / 0.73 | 0.26 / 1.26 | 3.9e-07 | 3.9e-07 | 2.86e-06 / 3.39e-06 (0.015) | slower |
| wd-vit-tagger-v3 | fp16 | 44.9 / 44.0 | 1.02 | 0.08 / 0.69 | 225 / 251 | 0.90 | 180 / 180 | 45.2 / 44.4 | 0.05 / 0.75 | 0.08 / 0.69 | 0.00088 | 0.00088 | 0.00264 / 0.00244 (0.015) | slower |
| wd-vit-tagger-v3 | fp32 | 53.7 / 53.5 | 1.00 | 0.09 / 0.74 | 448 / 444 | 1.01 | 361 / 361 | 54.2 / 54.1 | 0.06 / 0.76 | 0.09 / 0.74 | 2.7e-07 | 2.7e-07 | 2.01e-06 / 2.13e-06 (0.015) |  |
| xfeat | fp32 | 17.5 / 32.0 | 0.55 | 0.06 / 1.20 | 199 / 165 | 1.21 | 3 / 3 | 28.3 / 170.9 | 0.04 / 0.04 | 0.09 / 1.34 | - | 0.94 (0.33 % off) | 0 r1.000 / 0 r1.000 (0.0001) | heavier |
| xfeat-lighterglue | fp32 | 17.7 / 34.8 | 0.51 | 0.06 / 1.24 | 204 / 165 | 1.24 | 7 / 3 | 131.6 / 593.9 | 0.07 / 0.06 | 0.20 / 1.78 | - | 0.94 (0.36 % off) | 0 r1.000 / 0 r1.000 (0.0001) | heavier |
| yolox-l | fp16 | 51.0 / 61.3 | 0.83 | 0.07 / 1.48 | 293 / 280 | 1.05 | 103 / 103 | 56.0 / 66.1 | 0.05 / 0.14 | 0.54 / 1.55 | 0.7 (66 % off) | 0 | - | heavier |
| yolox-l | fp32 | 60.3 / 75.8 | 0.80 | 0.08 / 1.45 | 581 / 505 | 1.15 | 207 / 207 | 65.9 / 80.6 | 0.05 / 0.14 | 0.59 / 1.52 | 0.7 (66 % off) | 0 | - | heavier |
| yolox-m | fp16 | 30.0 / 37.5 | 0.80 | 0.06 / 1.35 | 170 / 207 | 0.82 | 49 / 49 | 34.8 / 41.8 | 0.04 / 0.11 | 0.07 / 1.35 | 0.74 (67 % off) | empty | - |  |
| yolox-m | fp32 | 33.1 / 42.5 | 0.78 | 0.06 / 1.38 | 368 / 441 | 0.83 | 97 / 97 | 38.3 / 46.9 | 0.04 / 0.15 | 0.07 / 1.39 | 0.74 (67 % off) | empty | - |  |
| yolox-s | fp16 | 12.4 / 17.0 | 0.73 | 0.05 / 1.25 | 83 / 178 | 0.47 | 17 / 17 | 18.0 / 21.5 | 0.04 / 0.08 | 0.54 / 1.32 | 0.77 (66 % off) | 0 | - |  |
| yolox-s | fp32 | 14.9 / 20.7 | 0.72 | 0.05 / 1.20 | 159 / 214 | 0.74 | 34 / 34 | 20.1 / 25.8 | 0.04 / 0.08 | 0.55 / 1.27 | 0.77 (66 % off) | 0 | - |  |
| yolox-tiny | fp16 | 8.8 / 12.4 | 0.71 | 0.04 / 1.23 | 37 / 65 | 0.57 | 10 / 10 | 13.4 / 17.3 | 0.04 / 0.07 | 0.53 / 1.30 | 0 | 0 | - |  |
| yolox-tiny | fp32 | 8.9 / 13.7 | 0.65 | 0.05 / 1.23 | 72 / 134 | 0.54 | 20 / 20 | 13.2 / 16.0 | 0.04 / 0.08 | 0.55 / 1.30 | 0 | 0 | - |  |
| yolox-x | fp16 | 94.4 / 106.6 | 0.89 | 0.12 / 1.73 | 391 / 441 | 0.89 | 190 / 190 | 100.1 / 111.8 | 0.06 / 0.21 | 0.64 / 1.80 | 0.58 (65 % off) | 0 | - |  |
| yolox-x | fp32 | 108.8 / 128.4 | 0.85 | 0.14 / 1.78 | 829 / 802 | 1.03 | 378 / 378 | 114.7 / 133.5 | 0.07 / 0.21 | 0.15 / 1.79 | 0.58 (65 % off) | empty | - | heavier |

Not measured (one arm failed):

- resplat-base-re10k float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-base-re10k float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
