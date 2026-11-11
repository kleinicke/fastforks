"""HRNet goldens from mmpose itself.

Reference: mmpose 1.3.2 (+ mmdet 3.3.0, mmcv 2.1.0, mmengine 0.10.7) as installed in another environment, appended
to `sys.path` after this one, with the import stubs of `tools/golden/rtmpose.py`: `mmpose.apis.init_model(<config>)`
builds the network from the variant's model-zoo config (`body_2d_keypoint/topdown_heatmap/coco`), the checkpoint's
`state_dict` is loaded strictly, and `inference_topdown` runs the whole test pipeline (bbox centre / scale, cv2
`warpAffine` crop, normalisation, MSRA heatmap decode, mapping back) with the configs' `flip_test` turned off (the
runner's default; `--flip` writes nothing and prints the flip-test keypoints' distance to the runner's
`flip_test=True` instead). CPU float32 on scikit-image's `astronaut.png` with the two person boxes of the RTMPose
goldens. Stored: the input image (`input`, RGB), the boxes (`x0, y0, x1, y1`), the keypoints (input pixels), their
scores, and `scores` (the mean keypoint score per box; the reference gives none), as float32.

    FASTFORKS_GOLDEN_PATH=... PYTHONPATH=src:tools/golden python tools/golden/hrnet.py MMPOSE_SITE_PACKAGES \
        [--flip] [hrnet-w32 ...]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import skimage
import torch
from _common import write
from rtmpose import BOXES, IMAGE_SHA256, OPENCV_VERSION, REF_COMMIT, UPSTREAM, _Stub, _StubFinder

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict
from fastforks.pose import load
from fastforks.pose.models.hrnet.model import extra

FAMILY = "hrnet"
CONFIGS = {  # variant -> model-zoo config (mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco)
    "hrnet-w32": "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
    "hrnet-w32-384": "td-hm_hrnet-w32_8xb64-210e_coco-384x288.py",
    "hrnet-w48": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
    "hrnet-w48-384": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
}


def reference(site: str, card: Any, variant: Any, *, flip: bool) -> Any:  # noqa: ANN401 - registry records / mmpose
    """The mmpose `TopdownPoseEstimator` of a variant (CPU, float32, eval) with its checkpoint loaded strictly."""
    if site not in sys.path:
        sys.path.append(site)
    for name in ("xtcocotools", "xtcocotools.coco", "xtcocotools.cocoeval", "xtcocotools.mask"):
        sys.modules.setdefault(name, _Stub(name))
    sys.meta_path.insert(0, _StubFinder())
    import mmpose  # noqa: PLC0415  # reference site-packages
    from mmpose.apis import init_model  # noqa: PLC0415

    config = Path(mmpose.__file__).parent / ".mim/configs/body_2d_keypoint/topdown_heatmap/coco" / CONFIGS[variant.id]
    model = init_model(str(config), None, device="cpu", cfg_options={"model.test_cfg.flip_test": flip}).eval()
    (ckpt,) = registry.weights(card, variant)
    model.load_state_dict(load_state_dict(ckpt, key="state_dict", strip="", extra=extra()), strict=True)
    return model


def main() -> None:
    args = [a for a in sys.argv[2:] if a != "--flip"]
    flip = "--flip" in sys.argv
    if not cv2.__version__.startswith(OPENCV_VERSION):
        msg = f"OpenCV {cv2.__version__}, the goldens pin {OPENCV_VERSION}"
        raise SystemExit(msg)
    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    img = skimage.io.imread(png)[..., :3]
    for card, variant in registry.cards("pose").values():
        if card.family != FAMILY or (args and variant.id not in args):
            continue
        model = reference(sys.argv[1], card, variant, flip=flip)
        from mmpose.apis import inference_topdown  # noqa: PLC0415  # on sys.path now

        with torch.inference_mode():
            res = inference_topdown(model, np.ascontiguousarray(img[..., ::-1]), BOXES, bbox_format="xyxy")
        kp = np.stack([r.pred_instances.keypoints[0] for r in res])
        ks = np.stack([r.pred_instances.keypoint_scores[0] for r in res])
        if flip:
            ours = load(f"{variant.id}@{variant.revision}", device="cpu", options={"flip_test": True})(img, boxes=BOXES)
            d = np.abs(np.asarray(ours.keypoints) - kp)
            print(f"  {variant.id} flip: |dkp| mean {d.mean():.4f} max {d.max():.4f}")
            continue
        maps = {"input": img, "boxes": BOXES, "keypoints": kp, "keypoint_scores": ks, "scores": ks.mean(1)}
        write(
            f"{variant.id}@{variant.revision}",
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, mmpose inference_topdown (coco config, flip_test off), OpenCV {cv2.__version__}, "
            f"skimage astronaut.png as BGR uint8, 2 xyxy boxes, torch {torch.__version__}",
            task="pose",
        )
        print(f"  {variant.id}: scores {np.round(ks.mean(1), 3)}")


if __name__ == "__main__":
    main()
