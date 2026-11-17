"""CoTracker3 offline / online goldens from upstream's PyTorch CoTracker3 (facebookresearch/co-tracker).

Reference: `cotracker/models/core/cotracker/cotracker3_{offline,online}.py` at `COMMIT`, CPU float32, driven as
upstream's predictors (`cotracker/predictor.py`): frames resized to 384 x 512 (bilinear, align_corners=True), queries
scaled by `(511 / (W - 1), 383 / (H - 1))`, 6 iterations.
- offline (`CoTrackerPredictor._compute_sparse_tracks`): a 6 x 6 support grid of frame-0 queries, the whole clip in
  one call, tracks on the query frame set to the query, visibility 1 there; score `sigmoid(vis)` (threshold 0.9).
- online (`CoTrackerOnlinePredictor`): `is_online` calls on `video[ind : ind + 16]` for `ind` in
  `range(0, max(T - 8, 1), 8)` (upstream's loop `range(0, T - 8, 8)` runs no window on clips of <= 8 frames; one
  window here); score `sigmoid(vis) * sigmoid(conf)` (threshold 0.6).
Input: tests/synth.py `clip()` (6 frames, 100 x 150). Stored: the queries `(t, x, y)` in clip pixels, the tracks in
clip pixels `[T, Q, 2]` and the score rescaled so that the threshold is 0.5 (the runner's visibility, see
`fastforks.tracking.models.cotracker.model`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/cotracker.py COTRACKER_SRC WEIGHTS_DIR

COTRACKER_SRC is a checkout of co-tracker at `COMMIT`; WEIGHTS_DIR holds the card's weight files.
`tools/eval_tracking.py` uses `load` and `track` as the original.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import write

from synth import clip

UPSTREAM = "https://github.com/facebookresearch/co-tracker"
COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
SIZE = (384, 512)
# (t, x, y) in clip pixels: the disk centre on frame 0, background points on frames 0 and 2, a corner pixel, a point
# on the disk queried on its last frame, the opposite corner
QUERIES = np.array(
    [[0, 50, 45], [0, 120.5, 20.25], [2, 30, 80], [0, 0, 0], [5, 75, 55], [3, 149, 99]], dtype=np.float32
)
MODELS = {"cotracker3-offline@1": ("scaled_offline.pth", False), "cotracker3-online@1": ("scaled_online.pth", True)}
THRESHOLD = {False: 0.9, True: 0.6}
GRID: list = []  # upstream `get_points_on_a_grid`, set by `load`


def load(src: Path, path: Path, online: bool) -> torch.nn.Module:  # noqa: FBT001 - mirrors upstream's `offline`
    """Upstream `build_cotracker(path, offline=not online, window_len=16 if online else 60)`, in eval mode."""
    sys.path.insert(0, str(src))
    me = sys.modules.pop("cotracker", None)  # this tool, imported as `cotracker`, hides upstream's package
    try:
        from cotracker.models.build_cotracker import build_cotracker  # noqa: PLC0415
        from cotracker.models.core.model_utils import get_points_on_a_grid  # noqa: PLC0415
    finally:
        if me is not None:
            sys.modules["cotracker"] = me
    GRID.append(get_points_on_a_grid)
    return build_cotracker(str(path), offline=not online, window_len=16 if online else 60).eval()


def _calibrate(p: torch.Tensor, thr: float) -> torch.Tensor:
    return torch.where(p < thr, p * (0.5 / thr), 0.5 + (p - thr) * (0.5 / (1 - thr)))


def track(
    model: torch.nn.Module, video: torch.Tensor, queries: torch.Tensor, amp: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracks `[T, Q, 2]` and visibility `[T, Q]` of `queries` `[Q, 3]` `(t, x, y)` in `video` `[T, 3, H, W]`.

    The video in [0, 1]; queries and tracks in its pixels; visibility rescaled (threshold -> 0.5). `amp`: autocast.
    """
    online = hasattr(model, "init_video_online_processing")
    n, _, h, w = video.shape
    v = F.interpolate(video, SIZE, mode="bilinear", align_corners=True)[None] * 255
    sc = torch.tensor([(SIZE[1] - 1) / (w - 1), (SIZE[0] - 1) / (h - 1)], device=video.device)
    q = torch.cat([queries[:, :1], queries[:, 1:] * sc], 1)[None]
    nq = queries.shape[0]
    region = torch.autocast(video.device.type, amp) if amp is not None else contextlib.nullcontext()
    with torch.no_grad(), region:
        if online:
            model.init_video_online_processing()  # ty: ignore[call-non-callable]  # upstream method
            step = model.window_len // 2
            for ind in range(0, max(n - step, 1), step):
                tracks, vis, conf, _ = model(video=v[:, ind : ind + 2 * step], queries=q, iters=6, is_online=True)
            score = vis.float() * conf.float()
        else:
            grid = GRID[0](6, SIZE, device=video.device)
            q = torch.cat([q, torch.cat([torch.zeros_like(grid[:, :, :1]), grid], 2)], 1)
            tracks, score, *_ = model(video=v, queries=q, iters=6)
    tracks, vis = tracks[0, :, :nq].float() / sc, _calibrate(score[0, :, :nq].float(), THRESHOLD[online])
    if not online:  # upstream: the query points are visible at their own position on their frame
        qf, i = queries[:, 0].long(), torch.arange(nq, device=video.device)
        tracks[qf, i] = queries[:, 1:]
        vis[qf, i] = 1.0
    return tracks, vis


def main() -> None:
    src, weights = Path(sys.argv[1]), Path(sys.argv[2])
    video = torch.from_numpy(np.stack(clip())).permute(0, 3, 1, 2).float() / 255
    q = torch.from_numpy(QUERIES)
    for ref, (file, online) in MODELS.items():
        tracks, vis = track(load(src, weights / file, online), video, q)
        mode = "online (is_online windows of 16, step 8)" if online else "offline (whole clip, 6 x 6 support grid)"
        write(
            ref,
            {"queries": QUERIES, "tracks": tracks.numpy(), "visibility": vis.numpy()},
            family="cotracker",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream CoTracker3 {mode} as its predictor, 6 iterations, tests/synth.py clip() "
            f"6 x 100 x 150 bilinear (align_corners) to 384 x 512, torch {torch.__version__}",
            task="tracking",
        )


if __name__ == "__main__":
    main()
