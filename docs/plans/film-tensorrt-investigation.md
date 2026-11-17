# FILM TensorRT investigation

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
differed by more than 0.1. The export gather matched the eager warp exactly on that case. The exported PyTorch graph also
matched eager on that case. TensorRT produced the larger differences around some edges. Portable FP32 and
FP16 differed by 0.000464 on average on that same clip, with 0.000514% of values over 0.1. TensorRT and
portable FP16 differed by 0.000937 on average. This comparison uses resized
clips and reports a separate validation protocol from the native-resolution DAVIS benchmark.

Separate fresh processes gave medians of 415.879 ms for portable and 188.284 ms for TensorRT, after
three warmups and ten CUDA-event runs at 1080×1920. Free device memory decreased by 4,532 MiB and
2,586 MiB after the two runs, respectively. These are retained process allocations measured after inference.

The experiment workspace contains the input filename and size manifest and per-video results in
`results/raw/2026-11-17_film_probe/dataset.log`. The manifest SHA-256 is
`8e4453b508a2371f2ae662c2ee450b159042bc852e2c5c5999895867fe55acac`.
