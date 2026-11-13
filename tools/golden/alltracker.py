"""AllTracker goldens from upstream's `nets.alltracker.Net` (aharley/alltracker).

Reference: upstream `Net.forward` at `COMMIT` + `patches/alltracker-cpu-device.patch` (upstream moves the frames to
CUDA unconditionally; the patch keeps them on the model's device): 16-frame windows, stride 8, `ITERS` refinements,
CPU float32; per query frame a forward pass over the frames from it and a backward pass over the reversed frames up
to it (upstream demo.py; a 1-frame pass is the query itself, a 2-frame pass upstream's two-frame flow). Input:
tests/synth.py `clip()` (6 frames, 100 x 150) resized to 384 x 512 with `F.interpolate(bilinear)` (the runner's
resize; upstream's evaluation uses cv2 INTER_LINEAR) and scaled to [0, 255] as upstream expects. The dense flow and
`visible * confident` are sampled bilinearly at the queries (upstream's evaluation rounds them to a pixel). Stored:
the queries, the tracks in clip pixels `[T, Q, 2]` `(x, y)` and the visibility `(visible * confident) ** (ln 0.5 /
ln 0.6)` `[T, Q]` (the runner's: > 0.5 where upstream's product is > 0.6).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/alltracker.py ALLTRACKER_SRC WEIGHTS_DIR

ALLTRACKER_SRC is a patched checkout of aharley/alltracker at `COMMIT` (needs einops); WEIGHTS_DIR holds the weight
files. `tools/eval_tracking.py` and `tools/_original.py` use `load` and `track` as the original (unpatched on CUDA).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

UPSTREAM = "https://github.com/aharley/alltracker"
COMMIT = "e7553135e7b361590dbccd10e2b274b024f41cd6"
SIZE = (384, 512)  # upstream evaluation image_size
ITERS = 4  # upstream inference_iters
PATCH = "alltracker-cpu-device.patch"
VIS_POWER = math.log(0.5) / math.log(0.6)
# (t, x, y) in clip pixels: the disk centre on frame 0, background points on frames 0 and 2, a corner pixel, points on
# frames 3, 4 (a two-frame forward pass) and 5 (the last frame: backward only)
QUERIES = np.array(
    [[0, 50, 45], [0, 120.5, 20.25], [2, 30, 80], [0, 0, 0], [5, 75, 55], [3, 149, 99], [4, 60, 50]], dtype=np.float32
)
# ref -> (weight file, Net kwargs as upstream's evaluation)
MODELS = {
    "alltracker@1": ("alltracker.pth", {}),
    "alltracker-tiny@1": ("alltracker_tiny.pth", {"use_basicencoder": True, "no_split": True}),
}


def load(src: Path, file: Path, kwargs: dict[str, Any]) -> Any:  # noqa: ANN401 - upstream Net
    """Upstream `Net(16, **kwargs)` with the checkpoint's `"model"` weights, in eval mode."""
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from nets.alltracker import Net  # noqa: PLC0415

    model = Net(16, **kwargs)
    model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True)["model"])
    return model.eval().requires_grad_(False)  # noqa: FBT003 - torch's signature


def _pass(model: Any, x: torch.Tensor, pts: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
    """Tracks `[L, K, 2]` and `visible * confident` `[L, K]` of points on frame 0 of `x` `[L, 3, h, w]` in [0, 255]."""
    n, (h, w) = x.shape[0], x.shape[-2:]
    tracks, vis = pts[None].repeat(n, 1, 1), pts.new_ones(n, pts.shape[0])
    if n == 1:
        return tracks, vis
    flows, visconfs, _, _ = model(x[None], iters=iters)
    maps = torch.cat([flows.reshape(-1, 2, h, w), visconfs.reshape(-1, 2, h, w)], 1).float()
    first = 1 if n == 2 else 0  # noqa: PLR2004 - the two-frame flow has no anchor frame
    grid = (pts * pts.new_tensor((2 / (w - 1), 2 / (h - 1))) - 1)[None, None].expand(maps.shape[0], -1, -1, -1)
    f = F.grid_sample(maps, grid, align_corners=True, padding_mode="border")[:, :, 0]  # [S, 4, K]
    tracks[first:], vis[first:] = pts + f[:, :2].transpose(1, 2), f[:, 2] * f[:, 3]
    return tracks, vis


def track(
    model: Any,  # noqa: ANN401 - upstream Net
    video: torch.Tensor,
    queries: torch.Tensor,
    iters: int = ITERS,
    amp: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracks `[T, Q, 2]` and visibility `[T, Q]` (see the module docstring) of `video` `[T, 3, H, W]` in [0, 1]."""
    n, (h, w) = video.shape[0], video.shape[-2:]
    scale = torch.tensor((SIZE[1] / w, SIZE[0] / h), device=video.device)
    q = queries.to(video.device).float()
    pts = (q[:, 1:] + 0.5) * scale - 0.5
    tracks, vis = pts.new_zeros(n, len(q), 2), pts.new_zeros(n, len(q))
    with torch.no_grad(), torch.autocast(video.device.type, amp or torch.float32, enabled=amp is not None):
        x = F.interpolate(video.float(), SIZE, mode="bilinear", align_corners=False) * 255
        for t0 in q[:, 0].long().unique().tolist():
            sel = (q[:, 0] == t0).nonzero()[:, 0]
            tr, vi = _pass(model, x[t0:], pts[sel], iters)
            tracks[t0:, sel], vis[t0:, sel] = tr, vi
            if t0 > 0:
                tr, vi = _pass(model, x[: t0 + 1].flip(0), pts[sel], iters)
                tracks[:t0, sel], vis[:t0, sel] = tr[1:].flip(0), vi[1:].flip(0)
    return (tracks + 0.5) / scale - 0.5, vis.clamp(0, 1) ** VIS_POWER


def main() -> None:
    from _common import write  # noqa: PLC0415

    from synth import clip  # noqa: PLC0415

    src, weights = Path(sys.argv[1]), Path(sys.argv[2])
    video = torch.from_numpy(np.stack(clip())).permute(0, 3, 1, 2).float() / 255
    for ref, (file, kwargs) in MODELS.items():
        tracks, vis = track(load(src, weights / file, kwargs), video, torch.from_numpy(QUERIES))
        write(
            ref,
            {"queries": QUERIES, "tracks": tracks.numpy(), "visibility": vis.numpy()},
            family="alltracker",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=PATCH,
            settings=f"CPU float32, upstream Net.forward (16-frame windows, stride 8, {ITERS} iters), forward + "
            f"backward passes per query frame, tests/synth.py clip() 6 x 100 x 150 bilinear to 384 x 512, "
            f"bilinear query sampling, torch {torch.__version__}",
            task="tracking",
        )


if __name__ == "__main__":
    main()
