"""RTMO goldens from mmpose itself.

Reference: mmpose 1.3.2 (+ mmdet 3.3.0, mmcv 2.1.0, mmengine 0.10.7) as installed in another environment, appended
to `sys.path` after this one (same torch; OpenCV `OPENCV_VERSION` here), with the import stubs of
`tools/golden/rtmpose.py`: `mmpose.apis.init_model(<config>)` builds the network from the variant's body7 model-zoo
config, the checkpoint's `state_dict` is loaded strictly, and `inference_bottomup` runs the whole test pipeline
(cv2 `warpAffine` letterbox, network, NMS, coordinate classifier, mapping back). CPU float32 on scikit-image's
`astronaut.png` (public domain, NASA; `IMAGE_SHA256`, given to mmpose as BGR uint8). Stored: the input image
(`input`, RGB) and the people the reference finds: `boxes` (`x0, y0, x1, y1`, input pixels), `keypoints`,
`keypoint_scores` and `scores`, best first, as float32.

    FASTFORKS_GOLDEN_PATH=... uv run --with opencv-python-headless==4.10.0.84 --with scikit-image \
        --with pycocotools --with shapely --with terminaltables \
        python tools/golden/rtmo.py MMPOSE_SITE_PACKAGES [rtmo-s ...]
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
from rtmpose import IMAGE_SHA256, OPENCV_VERSION, REF_COMMIT, UPSTREAM, _Stub, _StubFinder

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict
from fastforks.pose.models.rtmo.model import extra

FAMILY = "rtmo"
CONFIGS = {  # variant -> model-zoo config (mmpose/.mim/configs/body_2d_keypoint/rtmo/body7)
    "rtmo-t": "rtmo-t_8xb32-600e_body7-416x416.py",
    "rtmo-s": "rtmo-s_8xb32-600e_body7-640x640.py",
    "rtmo-m": "rtmo-m_16xb16-600e_body7-640x640.py",
    "rtmo-l": "rtmo-l_16xb16-600e_body7-640x640.py",
}


def reference(site: str, card: Any, variant: Any) -> Any:  # noqa: ANN401 - registry records / mmpose
    """Build the pinned bottom-up reference and load its checkpoint strictly on CPU."""
    if site not in sys.path:
        sys.path.append(site)
    for name in ("xtcocotools", "xtcocotools.coco", "xtcocotools.cocoeval", "xtcocotools.mask"):
        sys.modules.setdefault(name, _Stub(name))
    sys.meta_path.insert(0, _StubFinder())
    import mmpose  # noqa: PLC0415
    from mmpose.apis import init_model  # noqa: PLC0415

    configs = Path(mmpose.__file__).parent / ".mim/configs/body_2d_keypoint/rtmo/body7"
    model = init_model(str(configs / CONFIGS[variant.id]), None, device="cpu").eval()
    (ckpt,) = registry.weights(card, variant)
    model.load_state_dict(load_state_dict(ckpt, key="state_dict", strip="", extra=extra()), strict=True)
    return model


def main() -> None:
    args = sys.argv[2:]
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
        model = reference(sys.argv[1], card, variant)
        from mmpose.apis import inference_bottomup  # noqa: PLC0415

        with torch.inference_mode():
            (res,) = inference_bottomup(model, np.ascontiguousarray(img[..., ::-1]))
        p = res.pred_instances
        maps = {
            "input": img,
            "boxes": p.bboxes,
            "keypoints": p.keypoints,
            "keypoint_scores": p.keypoint_scores,
            "scores": p.bbox_scores,
        }
        write(
            f"{variant.id}@{variant.revision}",
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, mmpose inference_bottomup (body7 config), OpenCV {cv2.__version__}, "
            f"skimage astronaut.png as BGR uint8, torch {torch.__version__}",
            task="pose",
        )
        print(f"  {variant.id}: {len(p.bboxes)} people, scores {np.round(p.bbox_scores, 3)}")


if __name__ == "__main__":
    main()
