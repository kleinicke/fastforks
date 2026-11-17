# FILM TensorRT investigation

2026-11-17, RTX 5060 Ti, PyTorch 2.14.1+cu130. FILM's existing ONNX export contains sequence operators that
TensorRT cannot parse. An isolated experiment (`attempt/film-export`, local commit `e3abc519`) uses a four-tap
gather during export; it built TensorRT engines at 128×128, 256×448 and 1080×1920. It is not integrated into the
release branch, and TensorRT remains gated.

For the 1080×1920 synthetic module input, separate fresh-process measurements after three warmups gave medians of
415.879 ms for portable and 188.284 ms for TensorRT (10 CUDA-event runs). The device free-memory deltas after the
runs were approximately 4,532 MiB and 2,586 MiB respectively; those are retained-memory snapshots, **not peak
memory**. PyTorch's peak counters do not include TensorRT allocations.

For the first three frames of each of 90 DAVIS videos, resized to 1080×1920, outputs were finite. Mean portable and
TensorRT PSNR against the middle frame was 28.7110 and 28.7095 dB; the maximum per-video PSNR difference was 0.0265
dB. The maximum per-channel pixel difference was 0.3213 on `classic-car`: 0.0236% of its output channel values
differed by more than 0.1. On that clip the exported PyTorch graph matched eager exactly; the difference appeared
in TensorRT. A 128×128 ONNX Runtime CPU comparison on the same clip had mean/max absolute differences of
0.000134/0.001128 versus CUDA portable. These results do not establish an accepted pixel-error tolerance.
For context, on that same 1080p clip the existing portable FP32 and FP16 modes differed by 0.000464 on average,
with 0.000514% of channel values over 0.1; the TensorRT-versus-portable-FP16 comparison differed by 0.000937 on
average, with 0.0236% over 0.1. Different comparisons do not by themselves establish a pass threshold.

The dataset manifest hashes the 270 relative filenames and file sizes to
`8e4453b508a2371f2ae662c2ee450b159042bc852e2c5c5999895867fe55acac`; image contents were not independently hashed.
The frames were resized, so this is not the native-resolution DAVIS benchmark. The experimental 1080p exported
PyTorch graph also caused recoverable GPU allocation warnings during the outlier probe; its memory needs further
investigation. Six focused FILM tests and lint/format checks passed. Raw scripts and logs are under the experiment
workspace's `results/raw/2026-11-17_film_probe/`.
