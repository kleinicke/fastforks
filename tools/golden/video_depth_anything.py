"""Video Depth Anything goldens from the reference implementation.

Reference: upstream DepthAnything/Video-Depth-Anything (`REF_COMMIT`), CPU float32,
`VideoDepthAnything.infer_video_depth(frames, input_size=518, device="cpu", fp32=True)` on a one-frame video (the
golden image; upstream pads it to one 32-frame window). Run this with a Python that has OpenCV, torchvision, tqdm,
einops and easydict (xFormers absent: upstream's plain-PyTorch attention).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/video_depth_anything.py VDA_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds `video_depth_anything_vits.pth` and `metric_video_depth_anything_vits.pth`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/DepthAnything/Video-Depth-Anything"
REF_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}  # upstream run.py model_configs
VARIANTS = {  # ref: (checkpoint, metric, output key)
    "video-depth-anything-s": ("video_depth_anything_vits.pth", False, "inverse_depth"),
    "video-depth-anything-metric-s": ("metric_video_depth_anything_vits.pth", True, "depth"),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    sys.path.remove(str(Path(__file__).resolve().parent))  # this script's name would shadow upstream's package
    from video_depth_anything.video_depth import VideoDepthAnything  # noqa: PLC0415 - reference checkout

    frames = image()[None]  # [1, H, W, 3] uint8 RGB (upstream reads videos as RGB)
    for ref, (file, metric, key) in VARIANTS.items():
        net = VideoDepthAnything(**CONFIG, metric=metric)
        net.load_state_dict(torch.load(weights / file, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            d, _ = net.eval().infer_video_depth(frames, 30, input_size=518, device="cpu", fp32=True)
        write(
            f"{ref}@1",
            {key: d[0]},
            family="video_depth_anything",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, infer_video_depth(input_size=518, fp32=True), one frame, torch {torch.__version__}",
            task="depth",
        )
        print(f"  {ref}: median {key} {np.median(d[0]):.4f}")


if __name__ == "__main__":
    main()
