"""YuNet goldens from libfacedetection.train itself.

Reference: ShiqiYu/libfacedetection.train (`REF_COMMIT`) `yunet_train` `build_yunet(variant)` with the checkpoint
loaded strictly, on the input of `eval_widerface --mode origin` (BGR, 0..255 float, zero-padded bottom / right to
multiples of 32), and `YuNetPostprocessor(score_threshold, nms_threshold)` with the card's defaults, on the CPU in
float32. The package is imported from the exported tree. Input: scikit-image's `astronaut.png` (public domain, NASA;
`IMAGE_SHA256`), stored as `input` (RGB). Stored: the faces, best first: `boxes` (`x0, y0, x1, y1`), `keypoints`
`[P, 5, 2]`, `scores` and `keypoint_scores` (the face score per landmark, as the runner reports it).

    FASTFORKS_GOLDEN_PATH=... CUDA_VISIBLE_DEVICES= python tools/golden/yunet.py CHECKOUT WEIGHTS_DIR ASTRONAUT_PNG \
        [yunet-n ...]

Needs OpenCV and SciPy (imported by the `yunet_train.tasks.face` package).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image

from fastforks._core import registry

FAMILY = "yunet"
UPSTREAM = "https://github.com/ShiqiYu/libfacedetection.train"
REF_COMMIT = "dca340aa082c71081a68d17db8e58b33a58a914b"
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"  # skimage data/astronaut.png
VARIANT = {"yunet-n": "yunet_n", "yunet-s": "yunet_s"}


def _face(tree: Path) -> Any:  # noqa: ANN401 - the reference package
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from yunet_train.tasks import face  # noqa: PLC0415 - from the exported tree

    return face


def build(tree: Path, weights: Path, variant: str, device: str = "cpu") -> Any:  # noqa: ANN401 - upstream YuNet
    """Upstream `build_yunet` with the checkpoint (as `eval_widerface`: `module.` prefix dropped), in eval mode."""
    model = _face(tree).build_yunet(VARIANT[variant])
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    return model.eval().to(device)


def run(tree: Path, model: Any, bgr: torch.Tensor, threshold: float, iou: float) -> Any:  # noqa: ANN401
    """`[1, 3, H, W]` BGR 0..255 float -> upstream `DetectionResult` (padding to multiples of 32 here)."""
    h, w = bgr.shape[-2:]
    x = torch.nn.functional.pad(bgr, (0, -w % 32, 0, -h % 32))
    post = _face(tree).YuNetPostprocessor(score_threshold=threshold, nms_threshold=iou)
    with torch.inference_mode():
        return post(model(x))[0]


def main() -> None:
    checkout, weights, png, *only = sys.argv[1:]
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    if hashlib.sha256(Path(png).read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    bgr = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1)[None].float()
    tree = reference_tree(Path(checkout), REF_COMMIT)
    for card, variant in registry.cards("pose").values():
        if card.family != FAMILY or (only and variant.id not in only):
            continue
        p = {**card.params, **card.options, **variant.params}
        model = build(tree, Path(weights) / variant.weights[0].name, variant.id)
        det = run(tree, model, bgr, p["score_threshold"], p["nms_iou"])
        scores = det.scores.numpy()
        maps = {
            "input": img,
            "boxes": det.boxes.numpy(),
            "keypoints": det.keypoints.numpy().reshape(-1, 5, 2),
            "keypoint_scores": np.repeat(scores[:, None], 5, 1),
            "scores": scores,
        }
        write(
            f"{variant.id}@{variant.revision}",
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, eval_widerface origin input (BGR 0..255, pad to 32), YuNetPostprocessor(score "
            f"{p['score_threshold']}, nms {p['nms_iou']}), skimage astronaut.png, torch {torch.__version__}",
            task="pose",
        )
        print(f"  {variant.id}: {len(scores)} faces, scores {np.round(scores, 4)}")


if __name__ == "__main__":
    main()
