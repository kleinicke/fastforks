# Multi-view API

One call is one scene: `model(images)` takes the `N` views of one scene (all the same size), plus the cameras the
model uses. Two kinds of model share one result type:

| kind | families | inputs | predicts |
|---|---|---|---|
| feed-forward | VGGT-1B, VGGT-Omega, MapAnything | images only | every view (`V = N`), cameras estimated in the frame of view `reference` |
| calibrated MVS | MVSAnywhere, PatchmatchNet | images, `intrinsics`, `cam_to_world`, `depth_range` (PatchmatchNet: required; MVSAnywhere: optional, else its heuristic) | the reference view(s) (`V = len(reference)`) |

A model that does not use an input rejects it (`InputError`), as it rejects a missing required one.
`MultiViewResult` fields have a leading view dim `V`: `depth` `[V, H, W]`, `world_points` `[V, H, W, 3]` (world
frame), `confidence`, `mask` (bool), `intrinsics` `[V, 3, 3]`, `cam_to_world` `[V, 4, 4]`, model-specific `extras`;
`views` lists the input view of each entry, `kind` says what depth means (`metric` or `scale_invariant`: one unknown
scale shared by depth, points and translations). `multiview.unproject(depth, intrinsics, cam_to_world)` gives world
points for models without a point head.

`model.module()` (portable backends) takes `images` `[N, 3, H, W]` in `[0, 1]` and the cameras the model requires, in
the order `intrinsics`, `cam_to_world`, `depth_range` (`[2]`), and returns the dict of maps the model call is built
from (before the result's casts). `export_onnx(path, h, w, batch=N)` exports it at a fixed size and view count (`batch`
is the number of views; the ONNX outputs are named after the dict keys). Checked with ONNX Runtime on the CPU for every
family; `TensorRTEngine` returns only the first output (`depth`) and is not yet tested on multi-view graphs.

`load`, `list_models` and `model_card` have the signatures documented under [task functions](index.md#task-functions).

::: fastforks.multiview
