# D-FINE and DEIM TensorRT export

The previous `dfine-x-coco` FP16 engine returned only NaNs on RTX 5060 Ti with TensorRT 11.4. The reported
workaround replaces GridSample with four gathers for zero-padded bilinear interpolation.
[TensorRT issue discussion](https://github.com/NVIDIA/TensorRT/issues/4813#issuecomment-6902929321)

The library now uses gather-based sampling when exporting D-FINE and DEIM to ONNX. A two-input test checks
interpolation and padding through ONNX Runtime. The first prototype emitted an invalid ONNX `Clip` node,
so index bounds now use `where`.

All three evaluations used 500 COCO val2017 images, selecting every tenth image by filename. Both modes use the
same checkpoint, image preparation and detection postprocessor. The TensorRT speed rows use the standard
`tools/speed_check.py` protocol on the prepared 640×640 FP16 network. The wall-clock build includes ONNX export.

| Variant | Portable box mAP | TensorRT box mAP | TensorRT median | Engine device memory | Build |
|---|---:|---:|---:|---:|---:|
| D-FINE-X | 56.31 | 56.32 | 5.637 ms | 182 MiB | 106.8 s |
| D-FINE-N | 42.93 | 42.96 | 1.081 ms | 56 MiB | 64.4 s |
| DEIM-N | 44.42 | 44.39 | 1.099 ms | 56 MiB | 65.6 s |

Engine device memory is the driver's free-memory delta. PyTorch buffer peaks are recorded separately in the raw
rows. On a separate synthetic 640×640 module-only run, D-FINE-X portable took 18.95 ms and TensorRT took 5.65 ms
after five warmups. Query order changes between backends, so task accuracy uses COCO scoring on the final detections.
The validated scope is static batch one at 640×640 for the three variants above.

The measurement source is `c063f114`. The same export code reached the development branch in `d2fb8db8`.
Raw scripts and logs are in the experiment workspace under `results/raw/2026-11-17_dfine_gather/`
and `results/raw/2026-11-17_dfine_export_scope/`.
