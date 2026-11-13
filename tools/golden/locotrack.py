"""LocoTrack-B / LocoTrack-S goldens from upstream's PyTorch LocoTrack (KU-CVLAB/LocoTrack).

Reference: `locotrack_pytorch/models/locotrack_model.py` at `COMMIT` (`LocoTrack.forward`, query chunk 64 as its
`inference`), CPU float32, the whole clip at once. Input: tests/synth.py `clip()` (6 frames, 100 x 150) resized to
256 x 256 with `F.interpolate(bilinear, antialias=True)` (the runner's resize) and scaled to [-1, 1]; `QUERIES`
`(t, x, y)` in clip pixels -> upstream `(t, y, x)` in continuous 256 x 256 coordinates. Stored: the queries, the
tracks back in clip pixels `[T, Q, 2]` `(x, y)` and upstream's visibility
`(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))` `[T, Q]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/locotrack.py LOCOTRACK_SRC WEIGHTS_DIR

LOCOTRACK_SRC is a checkout of LocoTrack at `COMMIT` (needs einops); WEIGHTS_DIR holds the card's weight files.
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

UPSTREAM = "https://github.com/KU-CVLAB/LocoTrack"
COMMIT = "3385a233a270160b253a69050c6a3934527f7775"
# (t, x, y) in clip pixels: the disk centre on frame 0, background points on frames 0 and 2, a corner pixel, a point
# on the disk queried on its last frame
QUERIES = np.array(
    [[0, 50, 45], [0, 120.5, 20.25], [2, 30, 80], [0, 0, 0], [5, 75, 55], [3, 149, 99]], dtype=np.float32
)
MODELS = {"locotrack-b@1": ("locotrack_base.ckpt", "base"), "locotrack-s@1": ("locotrack_small.ckpt", "small")}


def load(src: Path, path: Path, size: str) -> torch.nn.Module:
    """Upstream `LocoTrack(model_size=size)` with the checkpoint's `state_dict` (`model.` stripped), in eval mode."""
    sys.path.insert(0, str(src / "locotrack_pytorch"))
    from models.locotrack_model import LocoTrack  # noqa: PLC0415

    model = LocoTrack(model_size=size)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()})
    return model.eval()


def track(
    model: torch.nn.Module, video: torch.Tensor, queries: torch.Tensor, amp: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracks `[T, Q, 2]` and visibility `[T, Q]` of `queries` `[Q, 3]` `(t, x, y)` in `video` `[T, 3, H, W]`.

    Queries and tracks in the video's pixels (centres at integers); the video in [0, 1] is resized to 256 x 256 here;
    `amp`: autocast dtype.
    """
    h, w = video.shape[-2:]
    v = F.interpolate(video, (256, 256), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    scale = torch.tensor([256 / w, 256 / h], device=video.device)
    qp = torch.cat([queries[:, :1], ((queries[:, 1:] + 0.5) * scale).flip(-1)], 1)[None]  # (t, y, x)
    region = torch.autocast(video.device.type, amp) if amp is not None else contextlib.nullcontext()
    with torch.no_grad(), region:
        out = model(v.permute(0, 2, 3, 1)[None], qp, query_chunk_size=64)
    vis = (1 - out["occlusion"].float().sigmoid()) * (1 - out["expected_dist"].float().sigmoid())
    return out["tracks"][0].float().transpose(0, 1) / scale - 0.5, vis[0].T


def main() -> None:
    src, weights = Path(sys.argv[1]), Path(sys.argv[2])
    video = torch.from_numpy(np.stack(clip())).permute(0, 3, 1, 2).float() / 255
    q = torch.from_numpy(QUERIES)
    for ref, (file, size) in MODELS.items():
        tracks, vis = track(load(src, weights / file, size), video, q)
        write(
            ref,
            {"queries": QUERIES, "tracks": tracks.numpy(), "visibility": vis.numpy()},
            family="locotrack",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream LocoTrack offline (whole clip, query_chunk_size 64), tests/synth.py "
            f"clip() 6 x 100 x 150 bilinear-antialiased to 256 x 256, torch {torch.__version__}",
            task="tracking",
        )


if __name__ == "__main__":
    main()
