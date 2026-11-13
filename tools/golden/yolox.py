"""YOLOX goldens from the reference code (Megvii-BaseDetection/YOLOX).

Reference: a checkout of YOLOX at `COMMIT` on `sys.path`: `yolox.exp.get_exp(exps/default/<variant>.py).get_model()`
(BatchNorm eps 1e-3 from its `init_yolo`) with the checkpoint's `model` loaded strictly, `ValTransform(legacy=False)`
(keep-ratio cv2 resize to the exp's `test_size`, top-left on a 114 canvas, 0..255 BGR), the network (boxes decoded
in the head) and `postprocess` (objectness x best class >= the exp's `test_conf` 0.01, torchvision `batched_nms` at
`nmsthre` 0.65), boxes divided by the resize factor as `tools/demo.py`. CPU float32 on scikit-image's `coffee.png`
(CC0, Rachel Michetti) given as BGR uint8. Detections with a score of at least the midpoint of the widest gap
between consecutive scores whose midpoint is in `GAP_RANGE` are kept (the ref's `options` in golden.toml, which the
test loads with), so fp16 GPU runs keep the same set. Stored: the input image (`input`, RGB), boxes (input pixels),
scores and labels, as float32. Also needs pycocotools (the reference's data package imports it).

    FASTFORKS_GOLDEN_PATH=... uv run --with opencv-python-headless==4.10.0.84 --with scikit-image --with loguru \
        --with psutil --with tabulate --with pycocotools python tools/golden/yolox.py YOLOX_SRC [yolox-s ...]
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
from _common import goldens, write

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "yolox"
UPSTREAM = "https://github.com/Megvii-BaseDetection/YOLOX"
COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
OPENCV_VERSION = "4.10.0"
IMAGE = ("coffee.png", "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7")  # skimage file, sha256
EXPS = {
    "yolox-tiny": "yolox_tiny",
    "yolox-s": "yolox_s",
    "yolox-m": "yolox_m",
    "yolox-l": "yolox_l",
    "yolox-x": "yolox_x",
}
GAP_RANGE = (0.15, 0.6)  # range of the golden's threshold (a gap midpoint)


def reference(src: str, card: Any, variant: Any) -> tuple[Any, Any]:  # noqa: ANN401 - registry records / yolox
    """The reference exp and model of a variant (CPU, float32, eval) with its checkpoint's `model` loaded strictly."""
    if src not in sys.path:
        sys.path.insert(0, src)
    from yolox.exp import get_exp  # noqa: PLC0415  # the reference checkout

    exp = get_exp(str(Path(src) / "exps/default" / f"{EXPS[variant.id]}.py"))
    model = exp.get_model().eval()
    (ckpt,) = registry.weights(card, variant)
    model.load_state_dict(load_state_dict(ckpt, key="model", strip=""), strict=True)
    return exp, model


def detect(exp: Any, model: Any, bgr: np.ndarray, conf: float | None = None) -> tuple[torch.Tensor, ...]:  # noqa: ANN401
    """Boxes (input pixels), scores and labels of the reference demo path on a BGR uint8 image (conf: `test_conf`)."""
    from yolox.data.data_augment import ValTransform  # noqa: PLC0415
    from yolox.utils import postprocess  # noqa: PLC0415

    x, _ = ValTransform(legacy=False)(bgr, None, exp.test_size)
    r = min(exp.test_size[0] / bgr.shape[0], exp.test_size[1] / bgr.shape[1])
    dev = next(model.parameters()).device
    with torch.inference_mode():
        out = model(torch.from_numpy(x)[None].float().to(dev))
        (det,) = postprocess(out, exp.num_classes, exp.test_conf if conf is None else conf, exp.nmsthre)
    if det is None:
        return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return det[:, :4] / r, det[:, 4] * det[:, 5], det[:, 6].long()


def main() -> None:
    if not cv2.__version__.startswith(OPENCV_VERSION):
        msg = f"OpenCV {cv2.__version__}, the goldens pin {OPENCV_VERSION}"
        raise SystemExit(msg)
    png = Path(skimage.data.__file__).parent / IMAGE[0]
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE[1]:
        msg = f"{png} is not scikit-image's {IMAGE[0]}"
        raise SystemExit(msg)
    img = skimage.io.imread(png)[..., :3]
    for card, variant in registry.cards("detection").values():
        if card.family != FAMILY or (sys.argv[2:] and variant.id not in sys.argv[2:]):
            continue
        exp, model = reference(sys.argv[1], card, variant)
        boxes, scores, labels = (t.numpy() for t in detect(exp, model, np.ascontiguousarray(img[..., ::-1])))
        s = np.sort(scores)[::-1]
        mid = (s[:-1] + s[1:]) / 2
        i = int(np.argmax(np.where((mid >= GAP_RANGE[0]) & (mid <= GAP_RANGE[1]), s[:-1] - s[1:], -1)))
        threshold = round(float(mid[i]), 3)
        keep = scores >= threshold
        ref = f"{variant.id}@{variant.revision}"
        maps = {"input": img, "boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        write(
            ref,
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, reference ValTransform + model + postprocess (exps/default/{EXPS[variant.id]}.py; "
            f"test_conf {exp.test_conf}, NMS {exp.nmsthre}, test_size {exp.test_size[0]}), OpenCV {cv2.__version__}, "
            f"skimage {IMAGE[0]} as BGR uint8, score threshold {threshold}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load(FAMILY, "detection")
        store["refs"][ref]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump(FAMILY, store, "detection")
        print(f"  {ref}: {int(keep.sum())} kept of {len(scores)}, gap {s[i]:.3f} / {s[i + 1]:.3f}")


if __name__ == "__main__":
    main()
