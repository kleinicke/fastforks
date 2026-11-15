# Deferred TensorRT / torch.compile rows (2026-11-15)

Rows of `docs/data/speed_compile.jsonl` that failed only because the cold build / export ran over the 5 min lock rule (`timeout after N s`, `worker rc=124`, `exceeds 5 min lock rule`, `exceeds the 175 s slice`). Their `fails` is now `deferred: measure later (build >5 min)`; gen_docs shows `deferred (build >5 min)`, gapmatrix counts `deferred` as its own class. Real failures (NaN, Myelin, ConversionError, TypeError, parser, OOM, Inductor errors) are untouched and stay `unsupported`/failed. Rewritten by a small script (only the `fails` string changed, all other fields kept); to re-measure, run the normal speed_check path and the row is replaced.

Counts: 42 rows total: tensorrt 26, compile 16.

| ref | mode | old error |
|---|---|---|
| aliked-flexglue | compile | timeout after 190 s |
| aliked-flexglue | tensorrt | timeout after 190 s |
| any2full-b | tensorrt | timeout after 190 s |
| any2full-l | tensorrt | timeout after 190 s |
| any2full-s | compile | timeout after 190 s |
| any2full-s | tensorrt | timeout after 190 s |
| dat-2-x4 | tensorrt | worker rc=124 |
| dat-x4 | tensorrt | worker rc=124 |
| depthfm-v1 | compile | timeout after 190 s |
| depthfm-v1 | tensorrt | timeout after 190 s |
| drct-l-x4 | tensorrt | worker rc=124 |
| drct-x4 | tensorrt | worker rc=124 |
| falconsai-nsfw-vit | compile | timeout after 290 s |
| foundationstereo-vit-l | compile | unsupported: cold compile build exceeds the 175 s slice of gpulock (wo |
| foundationstereo-vit-l | tensorrt | unsupported: cold tensorrt build exceeds the 175 s slice of gpulock (w |
| gimm-vfi-r | compile | worker rc=124 |
| gimm-vfi-r | tensorrt | worker rc=124 |
| gimm-vfi-r-p | compile | worker rc=124 |
| gimm-vfi-r-p | tensorrt | worker rc=124 |
| great-igev-middlebury | tensorrt | unsupported: engine build exceeds 5 min lock rule (cold ONNX export +  |
| hat-x4 | tensorrt | worker rc=124 |
| lingbot-depth | compile | timeout after 190 s |
| lingbot-map | tensorrt | timeout after 190 s |
| marigold-v2-depth | compile | worker rc=124 |
| mocha-v1-sceneflow | compile | worker rc=124 |
| mocha-v1-sceneflow | tensorrt | unsupported: engine build exceeds 5 min lock rule (cold ONNX export +  |
| omni-dc | compile | timeout after 190 s |
| real-hat-gan-x4 | tensorrt | worker rc=124 |
| resplat-small-dl3dv | compile | timeout after 190 s |
| romav1 | compile | worker rc=124 |
| s2m2-m | tensorrt | unsupported: cold tensorrt build exceeds the 175 s slice of gpulock (w |
| seedvr2-3b | compile | timeout after 290 s |
| seedvr2-7b | tensorrt | timeout after 290 s |
| segformer-b5-face-parsing | tensorrt | timeout after 290 s |
| tapnext | tensorrt | unsupported: engine build exceeds 5 min lock rule (cold ONNX export +  |
| track-on-r | compile | worker rc=124 |
| track-on2 | compile | worker rc=124 |
| waft-a1 | tensorrt | worker rc=124 |
| waft-a1-kitti | tensorrt | worker rc=124 |
| waft-a1-sintel | tensorrt | worker rc=124 |
| wd-eva02-large-tagger-v3 | tensorrt | timeout after 290 s |
| wd-vit-tagger-v3 | tensorrt | timeout after 290 s |

Not deferred: tapnext-pp compile (Triton subprocess error, real failure); tapnext compile has a time, tapnext-pp has no tensorrt row.
