# LocoTrack fixed-shape export

The current portable module exported and built in TensorRT for both LocoTrack variants. The 480×854 measurements
use a fixed eight-frame video and 16 queries per call. The benchmark's portable full-call input uses 24 frames
and 256 queries, so use the TensorRT timing for the stated clip and query count.

The correctness comparison used 28 eight-frame clips from TAP-Vid DAVIS videos. Prompts were sampled at frames
0, 2 and 4 from visible annotated points. Frames were resized to 480×854 for both modes. Scores were computed
from frames after each prompt and averaged over clips.

| Variant | Portable / TensorRT AJ | Portable / TensorRT delta average | Portable / TensorRT occlusion accuracy | TensorRT median |
|---|---:|---:|---:|---:|
| LocoTrack-B | 89.53 / 89.53 | 94.82 / 94.81 | 97.35 / 97.35 | 6.115 ms |
| LocoTrack-S | 89.00 / 89.07 | 94.41 / 94.41 | 97.48 / 97.48 | 5.658 ms |

Mean absolute coordinate differences were 0.027 and 0.031 pixels for B and S. The largest individual coordinate
differences were 30.53 and 14.44 pixels, respectively. This validates the reported aggregate scores for short fixed
clips. Raw case measurements and engine metadata are in `results/raw/2026-11-17_locotrack_trt/` in the experiment
workspace.
