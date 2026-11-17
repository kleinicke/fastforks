# FILM TensorRT validation

The ONNX exporter previously lowered the four-tap warp's embedding-bag operation to an ONNX sequence. TensorRT
could not parse the graph. During export, FILM now gathers the four taps and sums their weighted pixel values.
The eager path still uses the existing weighted embedding bag.

Fixed exports built and ran at 128×128, 256×448, and 1080×1920 on an RTX 5060 Ti with PyTorch 2.14.1+cu130.
ONNX Runtime CPU ran the 128×128 graph. The resulting frame differed from portable by 0.000134 mean and
0.001128 maximum absolute error on that input, on the 0–1 scale.

At 1080×1920, the TensorRT and portable FP16 modules were compared on the first three frames of each of the
90 DAVIS videos. Every frame was resized to 1080×1920 and both modes predicted the middle frame. The average
PSNR to the ground truth was 28.7095 dB for TensorRT and 28.7110 dB for portable, with at most 0.0265 dB
difference on any video in this comparison.

The largest pixel difference was 0.3213 on the classic-car sequence. On that video, 0.0236% of output values
differed by more than 0.1. The export gather matched the eager warp exactly on that case. The two built
implementations of the full network produced different pixels around some edges. This comparison uses resized
clips and reports a separate validation protocol from the native-resolution DAVIS benchmark.

The experiment workspace contains the input filename and size manifest and per-video results in
`results/raw/2026-11-17_film_probe/dataset.log`. The manifest SHA-256 is
`8e4453b508a2371f2ae662c2ee450b159042bc852e2c5c5999895867fe55acac`.
