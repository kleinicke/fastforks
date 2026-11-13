"""RetinaFace goldens from facexlib itself.

Reference: xinntao/facexlib (`REF_COMMIT`, v0.3.0) `facexlib/detection/retinaface.py` `RetinaFace` as
`init_detection_model` builds it (the checkpoint without its `module.` prefix, loaded strictly) and its
`detect_faces(image, conf_threshold=0.8, nms_threshold=0.4, use_origin_size=True)` on a BGR uint8 image, on the CPU in
float32. The detection files are loaded by path, unmodified (the `facexlib` package `__init__`s, which import every
tool and their dependencies, are empty stand-ins). Input: scikit-image's `astronaut.png` (public domain, NASA;
`IMAGE_SHA256`), stored as `input` (RGB). Stored: the faces it finds, best first: `boxes` (`x0, y0, x1, y1`),
`keypoints` `[P, 5, 2]`, `scores` and `keypoint_scores` (the face score per landmark, as the runner reports it).

    FASTFORKS_GOLDEN_PATH=... CUDA_VISIBLE_DEVICES= python tools/golden/retinaface.py FACEXLIB_CHECKOUT WEIGHTS_DIR \
        ASTRONAUT_PNG [retinaface-resnet50 ...]

Needs OpenCV, Pillow and torchvision (the reference's ResNet-50, `IntermediateLayerGetter` and NMS).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image

from fastforks._core import registry

FAMILY = "retinaface"
UPSTREAM = "https://github.com/xinntao/facexlib"
REF_COMMIT = "260620ae93990a300f4b16448df9bb459f1caba9"
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"  # skimage data/astronaut.png
NETWORK = {"resnet50": "resnet50", "mobilenet0.25": "mobile0.25"}  # card backbone -> facexlib network_name
FILES = ("matlab_cp2tform", "align_trans", "retinaface_net", "retinaface_utils", "retinaface")


def build(tree: Path, weights: Path, backbone: str, device: str = "cpu") -> Any:  # noqa: ANN401 - facexlib RetinaFace
    """The reference `RetinaFace` (facexlib `init_detection_model`) with the checkpoint, on `device`."""
    for name in ("facexlib", "facexlib.detection"):
        sys.modules.setdefault(name, types.ModuleType(name))
    for name in FILES:
        path = tree / "facexlib" / "detection" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"facexlib.detection.{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules[f"facexlib.detection.{name}"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    model = sys.modules["facexlib.detection.retinaface"].RetinaFace(network_name=NETWORK[backbone], device=device)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    return model.eval()


def run(model: Any, bgr: np.ndarray, threshold: float, iou: float) -> np.ndarray:  # noqa: ANN401
    """`detect_faces` on a BGR uint8 `[H, W, 3]` image: `[P, 15]` (box, score, 5 landmarks x, y)."""
    with torch.inference_mode():
        return model.detect_faces(bgr, conf_threshold=threshold, nms_threshold=iou, use_origin_size=True)


def main() -> None:
    checkout, weights, png, *only = sys.argv[1:]
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    if hashlib.sha256(Path(png).read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    tree = reference_tree(Path(checkout), REF_COMMIT)
    for card, variant in registry.cards("pose").values():
        if card.family != FAMILY or (only and variant.id not in only):
            continue
        p = {**card.params, **card.options, **variant.params}
        model = build(tree, Path(weights) / variant.weights[0].name, p["backbone"])
        det = run(model, np.ascontiguousarray(img[..., ::-1]), p["score_threshold"], p["nms_iou"])
        scores = det[:, 4]
        maps = {
            "input": img,
            "boxes": det[:, :4],
            "keypoints": det[:, 5:].reshape(-1, 5, 2),
            "keypoint_scores": np.repeat(scores[:, None], 5, 1),
            "scores": scores,
        }
        write(
            f"{variant.id}@{variant.revision}",
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, facexlib detect_faces(conf 0.8, nms 0.4, use_origin_size), skimage astronaut.png "
            f"as BGR uint8, torch {torch.__version__}",
            task="pose",
        )
        print(f"  {variant.id}: {len(det)} faces, scores {np.round(scores, 4)}")


if __name__ == "__main__":
    main()
