# Marigold V2: lean port (lab result, not in the library yet)

Marigold V2 (huawei-bayerlab/marigold-v2, Apache-2.0) is Qwen-Image-Edit-2509's 20 B image transformer, quantised to
4 bit (NF4, bitsandbytes), with a rank-128 LoRA and a fine-tuned Qwen VAE, run for one rectified-flow step. With the
authors' code it did not fit the 16 GB reference card. This note records a lean re-implementation that was built and
measured on a rented RTX 3090 (24 GB, sm_86) on 2026-11-14: it needs 9 to 11 GiB with the weights on the GPU, or 1 to
3 GiB with the weights streamed from host memory, and is faster than the original. Depth variant only (stage 2
checkpoint); normals, albedo and the layered variant use the same network with other LoRA / VAE files.

## Result (RTX 3090, bf16, one square image, encode + transformer + decode)

Median of 5 warm calls; memory is torch's peak allocated / reserved in GiB.

| variant | 512² | 768² | 1024² | load |
|---|---|---|---|---|
| original (diffusers + bitsandbytes + peft) | 792 ms, 13.8 / 15.0 | 1475 ms, 15.1 / 16.6 | 2548 ms, 16.9 / 18.6 | 28.7 s |
| lean, plain PyTorch | 1106 ms, 8.9 / 9.6 | 1615 ms, 9.2 / 10.0 | 2399 ms, 9.7 / 11.1 | 2.4 s |
| lean + NF4 kernel | 705 ms, 8.9 / 9.4 | 1211 ms, 9.3 / 10.1 | 1998 ms, 9.8 / 11.2 | 2.4 s |
| lean, plain PyTorch, streamed weights | 1159 ms, 0.8 / 1.2 | 1665 ms, 0.9 / 1.8 | 2453 ms, 1.3 / 2.9 | 11.6 s |
| lean + NF4 kernel, streamed weights | 779 ms, 0.6 / 1.2 | 1265 ms, 0.9 / 1.9 | 2049 ms, 1.4 / 3.0 | 11.6 s |

- Streamed: the 60 blocks stay in pinned host memory (8.4 GB) and are uploaded one block ahead on a second CUDA
  stream while the current block computes. The GPU holds the activations and two blocks. It costs 3 to 10 % time.
- The weight file is 8.36 GiB (original: 41 GB of Qwen transformer shards + LoRA + VAE + text embeddings, quantised
  at every load).
- Plain PyTorch is slower than the original at 512² and 768² because its NF4 decode (a table gather) costs about
  430 ms per image at any resolution. The kernel does the same decode in 104 ms.
- Stage split at 768², kernel: encode 58, transformer 1058, decode 95 ms (original: 117, 1173, 186).

## Correctness

- Transformer: bit-identical to the original (`torch.equal`) at 512², 768² and 1024² when the attention is called
  with the original's all-true mask (`Block.exact_mask`). The mask only selects another attention kernel with other
  bf16 rounding; the default drops it (faster). Without it the prediction differs by 0.4 % (mean |diff| / mean |x|).
- NF4 kernel: bit-identical to bitsandbytes' decode.
- VAE as a 2D network: equal to the original 3D one in fp32 with TF32 off (encode max |diff| 1.3e-5 on values of
  8.9, decode 6.6e-6).
- End to end with the same encoder noise, default settings: depth mean |diff| 0.0015 to 0.0021 on a mean of 0.5.

## Design

1. **Fold everything that is constant at the fixed timestep** (t = 499 / 1000, fixed prompt): the 120 modulation
   linears, the time embedding, the output norm's linear and the text input layer become stored vectors and stored
   text states. The last block's text branch is never read and is dropped.
2. **NF4 weights stay packed** (4 bit + one bf16 scale per 64 values) and are decoded for each call; the LoRA stays
   a low-rank side path. Merging the LoRA into the weights and quantising again does not work: rounding to 4 bit
   removes the update (error² / update² = 0.87) and the output is garbage.
3. **Dtypes as the original computes them**: residual stream fp32, linears in bf16 with the LoRA term added in fp32,
   LayerNorm / RMS norm / RoPE / GELU in fp32, attention in bf16.
4. **VAE as a single-frame 2D network**: of each causal 3D conv only the last temporal tap sees the frame.
5. **NF4 kernel**: one thread per 8 packed bytes, high nibble first, `code * scale` in fp32, round to nearest even to
   bf16. The lab version is a CuPy RawKernel; for the library it becomes a `kind = "cubin"` kernel library.
6. **Streaming** as described above, as an option for small cards.

## What is left for the library

- **Weights.** The NF4 packing is reproducible in plain torch from the authors' bf16 Qwen files (scale = max |w| per
  64 values, nearest code; bytes and scales equal bitsandbytes' for the five layers tested), so the converter needs
  no bitsandbytes, diffusers or peft. Open: convert at first load from the authors' files (41 GB download) or host
  the 8.4 GiB file.
- Family `depth/models/marigold_v2` (card, runners `Portable` and `Kernels`, golden from the authors' code), kernel
  library `nf4`, speed and accuracy rows on the reference card. The lean model fits its 16 GB.
- More speed: at 1024² the matrix products (about 60 TFLOP) and the attention run near the 3090's bf16 peak. Left
  are the glue steps (norms, modulation, RoPE, GELU, casts: about 15 % of the transformer) as fused kernels.
- Rectangular inputs and the other modalities are untested.

## Sources

- Scripts, logs, reference tensors: `~/projects/experiments/017-fastforks-lib/results/2026-11-14_marigold_v2_3090/`
  on the Ubuntu box (`mv2_lean.py` the port, `mv2_kernel.py` the kernel, `mv2_dump.py` the converter that still uses
  the authors' stack, `mv2_up.py` the original baseline, `mv2_test.py` check + benchmark, `mv2_requant.py` the
  packing check, `mv2_merge.py` the rejected LoRA merge).
- Lean weights: `/mnt/hdd/projects/data/marigold-v2/mv2_depth_lean.safetensors` there
  (sha256 `4b4aee27acdba6ba91582b76ed374fe91bf1bb02bea3be491e04e71a54fe9f5d`).
- Earlier study on the 16 GB card: `results/2026-11-14_marigold_v2.md` in the same experiment folder.
