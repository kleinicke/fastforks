"""Metric3D v2 goldens from the reference implementation.

Reference: upstream YvanYin/Metric3D (`REF_COMMIT`) plus `patches/metric3d-mmengine.patch` (import `collect_env` from
mmengine when mmcv 1.x is absent), CPU float32, the `hubconf.py` `__main__` recipe for `metric3d_vit_large`: keep-ratio
resize into 616 x 1064 (OpenCV bilinear), mean padding, ImageNet normalisation, `model.inference`, crop, bilinear
upsampling to the input size; no intrinsics, so the canonical camera (depth scale 1). Two deviations, both exact: the
image goes to OpenCV as float32 (an exact bilinear resize, padding with the exact mean) instead of uint8, and the
depth-bin anchors are built on the CPU (upstream `get_bins` hard-codes `device="cuda"`). Normals and confidence get the
same crop and upsampling as the depth (upstream's demo leaves them at network size); normals are re-normalised.
Run this with a Python that has the upstream dependencies (mmengine, opencv-python).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/metric3d.py METRIC3D_CHECKOUT metric_depth_vit_large_800k.pth
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/YvanYin/Metric3D"
REF_COMMIT = "eb5b6fac0dc155e4e52f576e304fbf11655ff339"
PATCH = "metric3d-mmengine.patch"
INPUT = (616, 1064)
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    root = reference_tree(repo, REF_COMMIT, PATCH)
    sys.path.insert(0, str(root))
    import cv2  # noqa: PLC0415 - reference environment
    import hubconf  # noqa: PLC0415  # reference checkout

    cfg = hubconf.Config.fromfile(hubconf.MODEL_TYPE["ViT-Large"]["cfg_file"])
    model = hubconf.get_configured_monodepth_model(cfg)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model_state_dict"], strict=False)
    model.eval()
    dec = model.depth_model.decoder
    dec.get_bins = lambda n: torch.exp(torch.linspace(math.log(dec.min_val), math.log(dec.max_val), n))

    rgb_origin = image().astype(np.float32)
    h, w = rgb_origin.shape[:2]
    s = min(INPUT[0] / h, INPUT[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    ph, pw = INPUT[0] - rgb.shape[0], INPUT[1] - rgb.shape[1]
    pad = [ph // 2, ph - ph // 2, pw // 2, pw - pw // 2]
    rgb = cv2.copyMakeBorder(rgb, *pad, cv2.BORDER_CONSTANT, value=MEAN)
    mean, std = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
    x = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() - mean) / std)[None]
    with torch.no_grad():
        depth, confidence, out = model.inference({"input": x})

    def restore(m: torch.Tensor) -> torch.Tensor:
        m = m[:, :, pad[0] : m.shape[2] - pad[1], pad[2] : m.shape[3] - pad[3]]
        return F.interpolate(m, (h, w), mode="bilinear")[0]

    normals = restore(out["prediction_normal"][:, :3])
    normals = normals / normals.norm(dim=0, keepdim=True)
    maps = {
        "depth": torch.clamp(restore(depth)[0] * 1.0, 0, 300).numpy(),
        "normals": normals.permute(1, 2, 0).numpy(),
        "confidence": restore(confidence)[0].numpy(),
    }
    write(
        "metric3d-vit-large@1",
        maps,
        family="metric3d",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, hubconf __main__ recipe (float32 image, CPU depth bins), no intrinsics, "
        f"torch {torch.__version__}",
        task="depth",
    )
    print(f"  median depth {np.median(maps['depth']):.4f}, median confidence {np.median(maps['confidence']):.4f}")


if __name__ == "__main__":
    main()
