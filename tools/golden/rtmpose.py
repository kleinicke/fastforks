"""RTMPose goldens from mmpose itself.

Reference: mmpose 1.3.2 (+ mmdet 3.3.0, mmcv 2.1.0, mmengine 0.10.7) as installed in another environment, appended
to `sys.path` after this one (same torch; OpenCV `OPENCV_VERSION` here): `mmpose.apis.init_model(<config>)` builds the
network from the variant's model-zoo config (body8; ubody whole-body for DWPose, cocktail14 for RTMW; `OTHER`: Halpe26,
hand5, face6, AP-10K), the checkpoint's `state_dict` is loaded strictly, and
`inference_topdown` runs the whole test pipeline (bbox centre / scale, cv2 `warpAffine` crop, normalisation, SimCC
decode, mapping back) with the configs' `flip_test` turned off (the runner's default; `--flip` writes nothing and
prints the flip-test keypoints' distance to the runner's `flip_test=True` instead). CPU float32 on scikit-image's
`astronaut.png` (public domain, NASA; `IMAGE_SHA256`, given to mmpose as BGR uint8) with the person boxes `BOXES` of
the ViTPose goldens (the second one reaches past the image: the crop's zero border; face models: the face and the
first box, hand models: the right hand at the lower edge and the first box). Stored: the input image
(`input`, RGB), the boxes (`x0, y0, x1, y1`), the keypoints (input pixels), their scores, and `scores` (the mean
keypoint score per box; the reference gives none), as float32.

The mmpose import chain needs `xtcocotools` (no wheel for every Python; only its COCO evaluation uses it) and
`mmcv.ops` (compiled ops; mmdet imports them for detectors only): both are stubbed with empty modules.

    FASTFORKS_GOLDEN_PATH=... uv run --with opencv-python-headless==4.10.0.84 --with scikit-image \
        --with pycocotools --with shapely --with terminaltables \
        python tools/golden/rtmpose.py MMPOSE_SITE_PACKAGES [--flip] [rtmpose-t ...]
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import skimage
import torch
from _common import write

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict
from fastforks.pose import load
from fastforks.pose.models.rtmpose.model import extra

FAMILY = "rtmpose"
UPSTREAM = "https://github.com/open-mmlab/mmpose"
REF_COMMIT = "5408bc76f5b848cf925a0d1857899011d8c5b497"  # v1.3.2
OPENCV_VERSION = "4.10.0"
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"  # skimage data/astronaut.png
BOXES = np.array([[120.0, 20.0, 440.0, 511.0], [250.0, 150.0, 600.0, 560.0]], dtype=np.float32)
CONFIGS = {  # variant -> model-zoo config (mmpose/.mim/configs/body_2d_keypoint/rtmpose/body8)
    "rtmpose-t": "rtmpose-t_8xb256-420e_body8-256x192.py",
    "rtmpose-s": "rtmpose-s_8xb256-420e_body8-256x192.py",
    "rtmpose-m": "rtmpose-m_8xb256-420e_body8-256x192.py",
    "rtmpose-l": "rtmpose-l_8xb256-420e_body8-256x192.py",
    "rtmpose-m-384": "rtmpose-m_8xb256-420e_body8-384x288.py",
    "rtmpose-l-384": "rtmpose-l_8xb256-420e_body8-384x288.py",
}
WHOLEBODY = {  # DWPose variant -> config (mmpose/.mim/configs/wholebody_2d_keypoint/rtmpose/ubody; the same network)
    "dwpose-t": "rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py",
    "dwpose-s": "rtmpose-s_8xb64-270e_coco-ubody-wholebody-256x192.py",
    "dwpose-m": "rtmpose-m_8xb64-270e_coco-ubody-wholebody-256x192.py",
    "dwpose-l": "rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py",
    "dwpose-l-384": "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py",
}
RTMW = {  # RTMW variant -> config (mmpose/.mim/configs/wholebody_2d_keypoint/rtmpose/cocktail14)
    "rtmw-m": "rtmw-m_8xb1024-270e_cocktail14-256x192.py",
    "rtmw-l": "rtmw-l_8xb1024-270e_cocktail14-256x192.py",
    "rtmw-x": "rtmw-x_8xb704-270e_cocktail14-256x192.py",
    "rtmw-l-384": "rtmw-l_8xb320-270e_cocktail14-384x288.py",
    "rtmw-x-384": "rtmw-x_8xb320-270e_cocktail14-384x288.py",
}
OTHER = {  # variant -> (config under mmpose/.mim/configs, its name in `settings`)
    "rtmpose-t-halpe26": ("body_2d_keypoint/rtmpose/body8/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py", "halpe26"),
    "rtmpose-s-halpe26": ("body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py", "halpe26"),
    "rtmpose-m-halpe26": ("body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py", "halpe26"),
    "rtmpose-l-halpe26": ("body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb512-700e_body8-halpe26-256x192.py", "halpe26"),
    "rtmpose-m-halpe26-384": (
        "body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py",
        "halpe26",
    ),
    "rtmpose-l-halpe26-384": (
        "body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb512-700e_body8-halpe26-384x288.py",
        "halpe26",
    ),
    "rtmpose-x-halpe26-384": (
        "body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
        "halpe26",
    ),
    "rtmpose-m-hand5": ("hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py", "hand5"),
    "rtmpose-t-face6": ("face_2d_keypoint/rtmpose/face6/rtmpose-t_8xb256-120e_face6-256x256.py", "face6"),
    "rtmpose-s-face6": ("face_2d_keypoint/rtmpose/face6/rtmpose-s_8xb256-120e_face6-256x256.py", "face6"),
    "rtmpose-m-face6": ("face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py", "face6"),
    "rtmpose-m-ap10k": ("animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py", "ap10k"),
}
PART_BOXES = {  # the first box of a golden: face / hand (from the image's RTMW-m keypoints), no feet in the image
    "halpe26": (120.0, 20.0, 440.0, 330.0),
    "face6": (165.0, 70.0, 290.0, 195.0),
    "hand5": (130.0, 470.0, 200.0, 512.0),
}


class _Stub(types.ModuleType):
    """An importable module whose every attribute is an empty class (the names mmpose imports but never calls)."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - any name
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports `mmcv.ops` and every submodule of it as a `_Stub` package."""

    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:  # noqa: ANN401, ARG002 - importlib API
        if name != "mmcv.ops" and not name.startswith("mmcv.ops."):
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec: Any) -> types.ModuleType:  # noqa: ANN401 - importlib API
        module = _Stub(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        pass


def reference(site: str, card: Any, variant: Any, *, flip: bool) -> Any:  # noqa: ANN401 - registry records / mmpose
    """The mmpose `TopdownPoseEstimator` of a variant (CPU, float32, eval) with its checkpoint loaded strictly."""
    if site not in sys.path:
        sys.path.append(site)
    for name in ("xtcocotools", "xtcocotools.coco", "xtcocotools.cocoeval", "xtcocotools.mask"):
        sys.modules.setdefault(name, _Stub(name))
    sys.meta_path.insert(0, _StubFinder())
    import mmpose  # noqa: PLC0415  # reference site-packages
    from mmpose.apis import init_model  # noqa: PLC0415

    configs = Path(mmpose.__file__).parent / ".mim/configs"
    if variant.id in OTHER:
        config = configs / OTHER[variant.id][0]
    elif variant.id in RTMW:
        config = configs / "wholebody_2d_keypoint/rtmpose/cocktail14" / RTMW[variant.id]
    elif variant.id in WHOLEBODY:
        config = configs / "wholebody_2d_keypoint/rtmpose/ubody" / WHOLEBODY[variant.id]
    else:
        config = configs / "body_2d_keypoint/rtmpose/body8" / CONFIGS[variant.id]
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

        configs = (
            OTHER[variant.id][1]
            if variant.id in OTHER
            else "cocktail14"
            if variant.id in RTMW
            else "ubody"
            if variant.id in WHOLEBODY
            else "body8"
        )
        boxes = BOXES if configs not in PART_BOXES else np.array([PART_BOXES[configs], BOXES[0]], dtype=np.float32)
        with torch.inference_mode():
            res = inference_topdown(model, np.ascontiguousarray(img[..., ::-1]), boxes, bbox_format="xyxy")
        kp = np.stack([r.pred_instances.keypoints[0] for r in res])
        ks = np.stack([r.pred_instances.keypoint_scores[0] for r in res])
        if flip:
            ours = load(f"{variant.id}@{variant.revision}", device="cpu", options={"flip_test": True})(img, boxes=boxes)
            d = np.abs(np.asarray(ours.keypoints) - kp)
            print(f"  {variant.id} flip: |dkp| mean {d.mean():.4f} max {d.max():.4f}")
            continue
        maps = {"input": img, "boxes": boxes, "keypoints": kp, "keypoint_scores": ks, "scores": ks.mean(1)}
        write(
            f"{variant.id}@{variant.revision}",
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, mmpose inference_topdown ({configs} config, flip_test off), "
            f"OpenCV {cv2.__version__}, "
            f"skimage astronaut.png as BGR uint8, 2 xyxy boxes, torch {torch.__version__}",
            task="pose",
        )
        print(f"  {variant.id}: scores {np.round(ks.mean(1), 3)}")


if __name__ == "__main__":
    main()
