# Brief: kernels for GPUs with full-rate fp32 accumulation (A10, L4, L40S, A100, H100)

Status (2026-11-07): design only, nothing built; revised after the GEMM shoot-out (experiment 0) and the per-call sweep
(experiment 1) on the L4, A10 and L40S. Based on the depth survey on the L4, A10 and L40S
(`docs/data/speed/*.jsonl`), a kernel trace of depth-pro on the L4 (`modal_survey.py::profile`) and a cuBLAS GEMM
test on the L4 (fp16 inputs with fp32 vs fp16 accumulation).

## What we measured

`kernels` over `portable`, fp16, 994x1482, median over the 18 depth refs: 5060 Ti 1.75x, A10 1.20x, L4 1.25x,
L40S 1.14x. The models that use only `cudnn_hacc` (Marigold, MiDaS) gain nothing or get slower (0.81-1.04x).

The GEMM test on the L4 (cuBLAS, ViT-L block shapes and 4096³):

| | fp32 accumulate | fp16 accumulate |
|---|---|---|
| throughput | 48-50 TFLOPS | 48-52 TFLOPS (0.98-1.03x) |
| mean error vs an fp32 reference | 0.009 | 0.149 |

The depth-pro trace on the L4, GPU ms per call:

| part | portable | kernels |
|---|---|---|
| GEMMs | 266 (cuBLAS, fp32 acc) | 376 (`fused_gemm_kernel`, fp16 acc, epilogue fused) |
| elementwise next to the GEMMs (bias, GELU, layer scale, residual) | ~90 | 0 |
| convolutions | 104 (cuDNN + adds) | 89 (`fused_conv_kernel`) |
| attention | 39-41 (PyTorch flash) | 49-52 (`flash64_*`, fp16 acc) |
| layer norm | 17 | 19 |
| total wall time | 560 | 516-520 |

Depth-pro's two ViT-L encoders do about 12.6 TFLOP of block GEMMs, which is about 256 ms at the L4's 49 TFLOPS
(72 W cap). `fused_gemm_kernel` also runs the DPT decoder's 1x1 projections and deconvolutions, which portable runs
in cuDNN. The two GEMM rows above are therefore not the same work (see experiment 0).

## Experiment 0: GEMM shoot-out (done)

`tools/gemm_shootout.py` (`uvx modal run tools/modal_survey.py::shootout --gpu L4 --out gemm.json`) times depth-pro's
ViT-L block GEMMs (M = 20195 tokens) with the epilogue each one has in the model. It compares every `dpt_fused` tile
configuration against cuBLAS: unfused (portable), the bare GEMM, and cuBLASLt with epilogues. TFLOPS:

| GPU | shape | `gemm_cfg` today | best `dpt_fused` cfg | cuBLAS + separate ops (portable) | bare cuBLAS GEMM | cuBLASLt epilogue |
|---|---|---|---|---|---|---|
| L4 | qkv | 45 | 55 (cfg 2) | 41 | 47 | 47 |
| L4 | proj (+res) | 43 | 48 (cfg 2) | 21 | 40 | 42 |
| L4 | fc1 (GELU) | 37 | 41 (cfg 2) | 32 | 49 | 48 |
| L4 | fc2 (+res) | 51 | 51 (rule) | 41 | 58 | 53 |
| A10 | qkv / proj / fc1 / fc2 | 71 / 65 / 56 / 66 | 73 / 65 / 66 / 66 | 56 / 37 / 48 / 54 | 63 / 63 / 67 / 65 | 66 / 58 / 65 / 68 |
| L40S | qkv / proj / fc1 / fc2 | 159 / 159 / 149 / 188 | 161 / 160 / 157 / 188 | 160 / 91 / 101 / 118 | 155 / 169 / 156 / 147 | 150 / 150 / 155 / 155 |

Max error against an fp32 reference: fp16-accumulate cfgs 0.02-0.05; fp32-accumulate cfgs (5, 6), cuBLAS and
cuBLASLt about 0.002. All fp16 cfgs give identical outputs. cuBLASLt's tanh GELU is as close to the erf reference
as the fp32 GEMM is (0.002-0.003).

Caveat: `ff_rule` and the cfg it picks run the same kernel, but they differ by up to 10-20% on fc1. So the variant
order inside a round matters, and differences below about 10% are not reliable yet. The sweep tool needs randomised
order and more rounds.

## Experiment 1: every call of a depth-pro forward (done)

`tools/call_sweep.py` (`uvx modal run tools/modal_survey.py::sweep --gpu L4 --out calls.json`) wraps `DptFused.gemm`,
`DptFused.conv` and `_flash` during one `kernels` forward. It times each call signature (52 of them, 411 calls)
in isolation: every tile cfg, the torch path (`F.linear` / channels-last `F.conv2d` plus separate epilogue ops,
cuDNN `conv_transpose2d` for the shuffles) and SDPA. Variant order is random per round. GPU ms per forward, summed
over the calls:

| GPU | part | `dpt_fused` today | best cfg per call | torch (portable path) |
|---|---|---|---|---|
| L4 | GEMMs (ViTs, 1x1) | 285 | 267 | 387 |
| L4 | shuffle GEMMs (deconv) | 7.3 | 6.8 | 11.5 |
| L4 | 3x3 convs (decoder, head) | 86 | 80 | 119 |
| L4 | attention | 41.5 (flash64) | | 36.5 (SDPA) |
| L4 | **all** | **419** | **390** (with SDPA) | **554** |
| A10 | all (GEMM / shuffle / conv / attn) | 297 (201 / 5.2 / 61 / 29) | 288 | 362 (247 / 7.0 / 79 / 28) |
| L40S | all | 117 (77 / 2.0 / 25 / 12.7) | 109 | 154 (99 / 4.0 / 39 / 12.2) |

The biggest per-call wins of a measured tile table:

- L4 qkv (M 20195): cfg 5 (fp32 accumulation, 128x64) 54.7 ms against the rule's cfg 0 71.6 ms (-24%);
- L4 768x768 3x3 convs (ReLU): conv cfg 5 21.8 against the rule's cfg 0 25.3 ms;
- L40S fc1: cfg 0 22.3 against the rule's cfg 2 26.3 ms.

Compared with the end-to-end survey times (portable / kernels, ms):

| GPU | survey | per-call sum (torch / `dpt_fused`) | outside the calls (portable / kernels) |
|---|---|---|---|
| L4 | 580 / 559 | 554 / 419 | 26 / **140** |
| A10 | 375 / 347 | 362 / 297 | 13 / **50** |
| L40S | 151 / 132 | 154 / 117 | -3 / **15** |

So per call, `kernels` is 1.3x faster than the portable ops. End to end, 50-140 ms of that disappear somewhere
outside the timed calls, and most of all on the power-capped L4. Candidates, not yet measured:

- **Power / clocks:** the isolated timings have gaps between rounds, and a forward does not. The denser fused
  kernels may pull the L4 (72 W) further down its clock curve than the portable ops do. The order L4 > A10 (150 W)
  > L40S (350 W) fits.
- **The work between the calls:** layer norms (19 ms in the L4 trace), patch extraction, bilinear resizes, the fp32
  FOV tail.
- **Side streams:** the image and FOV ViTs run next to the patch ViT. On a card with few SMs (L4: 58) the kernels
  may slow each other down more than they gain from overlap. The trace's GEMM sum (376 ms against 292 ms here) fits
  with overlapping kernels.

Next measurement (experiment 2): wall time of the kernels forward by section (encoders, decoder, head, FOV) with
`streams` on and off, and with SM clock and power logged during the run.

## Experiment 2: sections, clocks and power on the L4 (done)

`tools/section_time.py` (`modal_survey.py::sections`), depth-pro at 1536x1536, 20 forwards back to back:

| path | ms | SM MHz (mean / min) | W | gapped ms (2 s idle before each) |
|---|---|---|---|---|
| portable | 542 | 1015 / 840 | 71.8 | 524 |
| kernels | 500 | 766 / 645 | 72.1 | 473 |
| kernels, streams off | 507 | 781 / 630 | 72.3 | 484 |

- **The L4 is power-bound.** Both paths sit at the 72 W cap, and the kernels path runs its SMs at 766 MHz (the boost
  clock is 2040 MHz). The kernels path is faster because it does less work per forward, not because it runs at a
  higher rate. So a variant must be judged by sustained time under the cap, or by energy per forward (J), and never by
  isolated calls on a cold card. This probably explains experiment 1's gap: the per-call sums were taken in bursts,
  at higher clocks.
- **Side streams help a little** (500 against 507 ms). They stay on.
- **Sections (streams on):** encode 404 ms, decode 90 ms, head 11 ms, FOV 1 ms.
- **GPU time by kernel family** (one forward, streams off; 495 ms in all): `dpt_fused` GEMM 347 ms, conv 82 ms,
  `flash64` 45 ms, LayerNorm 16 ms, everything else 5 ms. Nothing outside the kernels is left to win; the GEMMs are
  the target, and on a power-capped card the cheapest work per FLOP wins.

## Experiment 3: kernel lab, sustained (in progress)

`tools/kernel_lab.py` (`modal_survey.py::lab`) loads depth-pro once and switches variants in place: the tile cfg per
problem, the `dpt_fused` library (built in the container from a modified source), the attention, the side streams.
The variants run round-robin in shuffled order after a 20 s warm-up (4 rounds of 8 forwards each), so all of them see
the same power-cap state. It reports ms, mean SM clock, W, J per forward and the error of the inverse depth (max and
mean, relative to the mean) against portable fp32 and against the first variant.

Results on the A10 (iterations 2-4; ms, J per forward):

| variant | ms | J | SM MHz | err vs fp32 (max / mean) |
|---|---|---|---|---|
| base (today's rule, erf GELU) | 357-367 | 53.7-54.6 | 872 | 0.0176 / 0.0021 |
| 128x256 tiles for the 20195-row qkv / proj / fc2 (cfg 4), fc1 cfg 0 | 350 | 52.2 | | same as base (bit-identical) |
| other tiles: K=64, 256x128, 4-stage, fc1 / conv picks | 349-406 | | | |
| **tanh GELU** (`tanh.approx.f32`) + cfg 4 + fc1 cfg 0 or 11 | **320-324** | **47.6-48.6** | 1026 | 0.0121 / 0.0019 |
| the same, flash64 `fh` / `ff` | 320 | 48.0 | | 0.0155 / 0.0025, 0.0177 / 0.0022 |
| the same, SDPA | 326 | 48.7 | | |
| the same, streams off | 324 | 48.3 | | |

- **The GELU is the lever: 11% faster and 11% less energy per forward.** The tanh form costs fewer instructions,
  so at the same 150 W the clock rises from 872 to 1026 MHz. The output is no further from fp32 than today's
  (0.0121 against 0.0176 max); against today's fp16 output it moves by 0.023 max / 0.0036 mean.
- **Tiles have plateaued.** Only cfg 4 for the large non-GELU ViT GEMMs gains (about 2%), bit-identically.
- **Attention is not a lever on the A10.** All three flash64 variants are equal; `ff` (SDPA precision) costs nothing.

## Diagnosis (revised)

1. **fp16 accumulation buys nothing on these cards.** In the cuBLAS test, fp16 and fp32 accumulation run at the
   same rate. In the shoot-out, cfg 5 (fp32) is as fast as cfg 2 (fp16, same tiles) within noise, and about 15x more
   accurate. On full-rate cards we can accumulate in fp32 for free.
2. **The fused block GEMMs are not the problem.** They match or beat bare cuBLAS on the A10 and L40S. They are about
   10% behind on the L4, mostly on fc1. They beat the portable path (cuBLAS plus separate ops) by 20-60%.
   - Weighted over a block, depth-pro's block GEMMs take about 292 ms (L4) / 200 ms (A10) fused, against about
     361 / 250 ms portable.
   - My earlier reading of the trace was wrong: `fused_gemm_kernel`'s 376 ms on the L4 also covers the DPT decoder's
     1x1 projections and stride-2 deconvolutions (pixel-shuffle GEMMs). Portable runs those as cuDNN convolutions.
     About 84 ms of the 376 ms is that decoder work.
3. **The loss is not in the decoder (experiment 1).** Per call, the fused convs and shuffle GEMMs beat cuDNN by
   25-50% on all three cards. Attention loses a little (`flash64` against SDPA: 5 / 0.7 / 0.5 ms). The gap between
   the per-call sum and the end-to-end time is the open question (see experiment 1).
4. **The tile rule is per card.** cfg 2 (128x64) and cfg 5 (the same tile, fp32 accumulation) beat today's rule pick
   (cfg 0) for qkv on the L4 (by 24% in experiment 1). On the L40S,
   cfg 0 wins fc2 by 24%. A fixed rule cannot be right on all cards, so a measured table it is (as planned for
   the long run).
5. **The erf GELU epilogue is expensive on the power-capped L4.** fc1 runs 26% below qkv with the same tiles. On the
   A10 the gap is 9%, on the L40S 3%. A cheaper GELU (tanh form with `tanh.approx.f32`, as cuBLASLt uses, or a
   cheaper erf) would close most of it. Its accuracy cost is measured above as negligible.
6. **`cudnn_hacc` only (Marigold, MiDaS) is plain loss on full-rate cards.** Use the default cuDNN engines there.

## Design (revised)

Start flexible: per-card switches and a sweep. Later, the full per-card benchmark decides every setting.

### 1. Card class, from the device name

The compute capability does not separate the classes: sm_86, sm_89 and sm_120 each contain GeForce (half-rate fp32
accumulation) and datacenter or pro cards (full rate). The rule is: `"GeForce" in name` means half rate, everything
else full rate. It lives in `fastforks._core.kernels.half_rate_fp32_acc(device)`, with an env override
(`FASTFORKS_ACC=fp16|fp32`) for experiments. A timing probe is not needed for dispatch; the sweep tool can check the
rule on each new card.

### 2. Switches on full-rate cards

| part | GeForce (today) | full rate |
|---|---|---|
| `dpt_fused` GEMM / conv | fp16 accumulation | fp32-accumulate variants of the same tiles. Today only cfg 5/6 (GEMM) and 6/7 (conv) exist; add fp32 versions of cfg 0, 2 and 4. |
| tile choice | `gemm_cfg` / `conv_cfg` rule | the measured table (item 3), else the rule |
| GELU epilogue | erf | tanh approximation (`tanh.approx.f32`), if the goldens pass; worth trying on GeForce too |
| attention | `flash64_*` (fp16 QK and PV) | `F.scaled_dot_product_attention` |
| `cudnn_hacc` | fp16-accumulate engines | default cuDNN heuristic |

### 3. Measured tile table

`tools/gemm_shootout.py` grows into the sweep:
- shapes logged from one run of each ref, instead of the four fixed ones;
- randomised variant order;
- conv cfgs as well.

It writes `tiles.toml` (key `<GPU name>/<accumulation>`, entries (epilogue, M, N, K) to cfg). The library reads the
table and falls back to the rule. One Modal run per card fills it, and the ubuntu 5060 Ti row comes from a local run.

### 4. Next: where the end-to-end time goes

The decoder was measured (experiment 1) and is fine. Next is experiment 2. If it is the side streams, `streams` becomes
a per-card switch. If it is power, the per-call tile table has to be measured under sustained load, not in isolation.

### Not pursued: new large-warp tiles, cuBLASLt

- **Large-warp tiles:** the shoot-out shows the existing tiles at cuBLAS level, and the small 128x64 tile is often
  the best. The large-warp tile set from the first version of this note is dropped.
- **cuBLASLt:** it gains nothing over our tiles for the ViT blocks. It stays an option only for Hopper.

### Later: Hopper and Blackwell datacenter (sm_90, sm_100)

The binaries there are sm_80-style `mma.sync`, without wgmma or TMA. Run the shoot-out on an H100 first. If the gap
to cuBLAS is large, use cuBLASLt there, or a CUTLASS 3.x kernel with an epilogue visitor tree.

## Expected result (depth-pro)

- **Per-card tiles plus SDPA (experiment 1):** per-call sums go down by 30 ms on the L4 (419 to 390), 9 ms on the A10 and
  8 ms on the L40S. The cheap GELU comes on top (fc1 is the largest single GEMM on the L4: 80 ms).
- **The 50-140 ms outside the calls** are worth more than all tile work together, once experiment 2 says where they are.
- **5060 Ti:** nothing changes, except the cheap GELU if it is worth it there too.
