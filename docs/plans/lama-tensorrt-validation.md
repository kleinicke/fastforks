# Big LaMa TensorRT validation

Big LaMa's fixed-shape ONNX export now expresses the Fourier units as real matrix multiplies. TensorRT can build
that graph. The ordinary portable call uses PyTorch FFTs. A targeted regression checks forward and inverse
transforms on odd, even and rectangular inputs, including arbitrary spectra. The ONNX Runtime test uses an input
distinct from the trace.

The 512×512 engine ran in 33.867 ms on the RTX 5060 Ti, with an estimated engine device memory delta of 268 MiB.
ONNX export and engine build took 74.5 and 35.7 seconds, respectively.

The accuracy comparison uses 512×512 centre crops of every fourth DIV2K validation image, with deterministic
96×96 and 224×224 square masks on each crop. It measures RGB PSNR inside each of the 50 masks. Portable FP16 and
TensorRT averaged 17.8946 and 17.8854 dB. The largest per-case PSNR difference was 0.2677 dB. All output values
were finite and all unmasked pixels remained bit-identical to the input. Inside the masks, mean absolute
difference from portable averaged 0.00506 and the largest individual difference was 0.4546 on the 0–1 scale.

Raw benchmark and per-mask comparisons are in `results/raw/2026-11-17_lama_trt_dft/` in the experiment workspace.
