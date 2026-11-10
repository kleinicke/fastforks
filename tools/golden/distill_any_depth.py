"""Distill-Any-Depth goldens (the `depth_anything_v2` family) from the reference implementation.

Reference: Westlake-AGI-Lab/Distill-Any-Depth (`REF_COMMIT`), CPU float32, the networks and checkpoints of the
authors' `app.py` (Small / Base: `distillanydepth.depth_anything_v2.dpt.DepthAnythingV2`; Large:
`distillanydepth.modeling.archs.dam.dam.DepthAnything`). Input pipeline of the library (the authors' transformers
ports, upstream DAv2 `image2tensor`): the repository's own `Resize(518, keep_aspect_ratio, multiple of 14,
lower_bound, cv2.INTER_CUBIC)` + ImageNet normalisation on the RGB image, the network, then the relative inverse depth
upsampled back bilinearly (`align_corners=True`). The repository's `infer_image` is not used: it moves the input to
CUDA and interpolates a 4-D output as 5-D. Run with a Python that has OpenCV, diffusers, timm and safetensors.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/distill_any_depth.py CHECKOUT WEIGHTS_DIR [s b l]

`WEIGHTS_DIR` holds the Hugging Face repository's `{small,base,large}/model.safetensors`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write
from safetensors.torch import load_file

UPSTREAM = "https://github.com/Westlake-AGI-Lab/Distill-Any-Depth"
REF_COMMIT = "6d8f415392eafb49c96a38cc4dedbd09a1607f50"
VARIANTS = {  # size: (Hugging Face subdirectory, app.py model_kwargs)
    "s": ("small", {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}),
    "b": ("base", {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}),
    "l": ("large", {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "use_bn": False,
                    "use_clstoken": False, "max_depth": 150.0, "mode": "disparity", "pretrain_type": "dinov2",
                    "del_mask_token": False}),
}  # fmt: skip


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2  # noqa: PLC0415 - reference checkout
    from distillanydepth.depth_anything_v2.util.transform import (  # noqa: PLC0415
        NormalizeImage,
        PrepareForNet,
        Resize,
    )
    from distillanydepth.modeling.archs.dam.dam import DepthAnything  # noqa: PLC0415

    rgb = image()
    resize = Resize(518, 518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                    resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)  # fmt: skip
    norm = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = PrepareForNet()(norm(resize({"image": rgb / 255.0})))["image"]
    x = torch.from_numpy(x)[None]
    for size in sys.argv[3:] or VARIANTS:
        sub, kw = VARIANTS[size]
        net = (DepthAnything if size == "l" else DepthAnythingV2)(**kw)
        net.load_state_dict(load_file(weights / sub / "model.safetensors"))
        with torch.inference_mode():
            d, _ = net.eval()(x)
            inv = F.interpolate(d, rgb.shape[:2], mode="bilinear", align_corners=True)[0, 0].numpy()
        write(
            f"distill-any-depth-{size}@1",
            {"inverse_depth": inv},
            family="depth_anything_v2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, DAv2 image2tensor(518) pipeline, {sub}/model.safetensors, "
            f"torch {torch.__version__}",
            task="depth",
        )
        print(f"  median inverse depth {np.median(inv):.4f}")


if __name__ == "__main__":
    main()
