"""TAPNext goldens from upstream's PyTorch TAPNext (google-deepmind/tapnet).

Reference: `tapnet/tapnext/tapnext_torch.py` at `COMMIT`, CPU float32, run online as upstream's demos and evaluation
(the first frame with the query points, then one frame at a time with the returned state). Input: tests/synth.py
`clip()` (6 frames, 100 x 150) resized to 256 x 256 with `F.interpolate(bilinear, antialias=True)` (the runner's
resize) and scaled to [-1, 1]; `QUERIES` `(t, x, y)` in clip pixels -> upstream `(t, y, x)` in continuous 256 x 256
coordinates. Stored: the queries, the tracks back in clip pixels `[T, Q, 2]` `(x, y)` and the sigmoid of the visible
logits `[T, Q]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/tapnext.py TAPNET_SRC WEIGHTS_DIR

TAPNET_SRC is a checkout of tapnet at `COMMIT` (needs einops); WEIGHTS_DIR holds the card's weight files.
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


def main() -> None:
    src, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(src))
    from tapnet.tapnext.tapnext_torch import TAPNext  # noqa: PLC0415
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint  # noqa: PLC0415

    frames = clip()
    h, w = frames[0].shape[:2]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
    video = F.interpolate(video, (256, 256), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    video = video.permute(0, 2, 3, 1)[None]  # [1, T, 256, 256, 3]
    scale = torch.tensor([256 / w, 256 / h])
    q = torch.from_numpy(QUERIES)
    xy = (q[:, 1:] + 0.5) * scale
    qp = torch.cat([q[:, :1], xy.flip(-1)], 1)[None]  # (t, y, x)
    for ref, file in (("tapnext@1", "bootstapnext_ckpt.npz"), ("tapnext-pp@1", "tapnextpp_ckpt.pt")):
        model = TAPNext(image_size=(256, 256))
        if file.endswith(".npz"):
            model = restore_model_from_jax_checkpoint(model, str(weights / file))
        else:
            sd = torch.load(weights / file, map_location="cpu", weights_only=True)["state_dict"]
            model.load_state_dict({k.removeprefix("tapnext."): v for k, v in sd.items()})
        model.eval()
        tracks, vis = [], []
        with torch.no_grad():
            tr, _, vl, state = model(video=video[:, :1], query_points=qp)
            tracks.append(tr), vis.append(vl)
            for t in range(1, video.shape[1]):
                tr, _, vl, state = model(video=video[:, t : t + 1], state=state)
                tracks.append(tr), vis.append(vl)
        yx = torch.cat(tracks, 1)[0]  # [T, Q, 2] (y, x) continuous 256
        maps = {
            "queries": QUERIES,
            "tracks": (yx.flip(-1) / scale - 0.5).numpy(),
            "visibility": torch.cat(vis, 1)[0, ..., 0].sigmoid().numpy(),
        }
        write(
            ref,
            maps,
            family="tapnext",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream TAPNext online (first frame with the queries, then frame by frame), "
            f"tests/synth.py clip() 6 x 100 x 150 bilinear-antialiased to 256 x 256, torch {torch.__version__}",
            task="tracking",
        )


if __name__ == "__main__":
    main()
