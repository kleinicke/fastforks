"""TAPIR / BootsTAPIR goldens from upstream's PyTorch TAPIR (google-deepmind/tapnet).

Reference: `tapnet/torch/tapir_model.py` at `COMMIT`, CPU float32, the whole clip at once as upstream's offline demo.
Input: tests/synth.py `clip()` (6 frames, 100 x 150) resized to 256 x 256 with `F.interpolate(bilinear,
antialias=True)` (the runner's resize) and scaled to [-1, 1]; `QUERIES` `(t, x, y)` in clip pixels -> upstream
`(t, y, x)` in continuous 256 x 256 coordinates. Stored: the queries, the tracks back in clip pixels `[T, Q, 2]`
`(x, y)` and upstream's visibility `(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))` `[T, Q]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/tapir.py TAPNET_SRC WEIGHTS_DIR

TAPNET_SRC is a checkout of tapnet at `COMMIT` (needs einshape and dm-tree); WEIGHTS_DIR holds the card's weight files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import write

from synth import clip

UPSTREAM = "https://github.com/google-deepmind/tapnet"
COMMIT = "ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b"
# (t, x, y) in clip pixels: the disk centre on frame 0, background points on frames 0 and 2, a corner pixel, a point
# on the disk queried on its last frame
QUERIES = np.array(
    [[0, 50, 45], [0, 120.5, 20.25], [2, 30, 80], [0, 0, 0], [5, 75, 55], [3, 149, 99]], dtype=np.float32
)
# ref -> (weight file, TAPIR kwargs as upstream's demos)
MODELS = {
    "bootstapir@1": ("bootstapir_checkpoint_v2.pt", {"pyramid_level": 1}),
    "tapir@1": ("tapir_checkpoint_panning.pt", {"pyramid_level": 0, "extra_convs": False}),
}


def main() -> None:
    src, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(src))
    from tapnet.torch.tapir_model import TAPIR  # noqa: PLC0415

    torch.manual_seed(0)  # upstream permutes the queries into chunks (order-independent result)
    frames = clip()
    h, w = frames[0].shape[:2]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
    video = F.interpolate(video, (256, 256), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    video = video.permute(0, 2, 3, 1)[None]  # [1, T, 256, 256, 3]
    scale = torch.tensor([256 / w, 256 / h])
    q = torch.from_numpy(QUERIES)
    qp = torch.cat([q[:, :1], ((q[:, 1:] + 0.5) * scale).flip(-1)], 1)[None]  # (t, y, x)
    for ref, (file, kwargs) in MODELS.items():
        model = TAPIR(**kwargs)
        model.load_state_dict(torch.load(weights / file, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            out = model(video, qp.clone())
        vis = (1 - out["occlusion"].sigmoid()) * (1 - out["expected_dist"].sigmoid())
        maps = {
            "queries": QUERIES,
            "tracks": (out["tracks"][0].transpose(0, 1) / scale - 0.5).numpy(),
            "visibility": vis[0].T.numpy(),
        }
        write(
            ref,
            maps,
            family="tapir",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream TAPIR offline (whole clip, query_chunk_size 64), tests/synth.py clip() "
            f"6 x 100 x 150 bilinear-antialiased to 256 x 256, torch {torch.__version__}",
            task="tracking",
        )


if __name__ == "__main__":
    main()
