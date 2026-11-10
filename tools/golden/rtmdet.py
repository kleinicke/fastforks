"""RTMDet goldens from mmdetection itself.

Reference: mmdet 3.3.0 (+ mmengine 0.10.7, mmcv-lite 2.1.0) as installed in another environment, appended to
`sys.path` after this one (same torch; OpenCV `OPENCV_VERSION` here): `mmdet.apis.init_detector(<config>)`
builds the detector from the variant's model-zoo config, the checkpoint's `state_dict` (what mmdet loads; the EMA
weights the test results are reported with) is loaded strictly, and `inference_detector` runs the
whole test pipeline (keep-ratio cv2 resize, padding with 114, normalisation, network, box decoding, class-wise NMS,
mapping back) with the config's test settings. The person detectors are mmpose's (`REF_COMMIT_MMPOSE`,
`projects/rtmpose/rtmdet/person`): their configs are the mmdet ones with the overrides in `PERSON` (copied from those
two files; the checkpoints then load strictly). CPU float32 on scikit-image's `coffee.png` (CC0, Rachel Michetti; the
COCO variants) and `astronaut.png` (public domain, NASA; the person detectors), given as BGR uint8. Detections with a
score of at least the variant's `THRESHOLDS` entry are kept (the ref's `options` in golden.toml, which the test loads
with): mid-gap, so fp16 GPU runs keep the same set. Stored: the input image (`input`, RGB), boxes (input pixels),
scores and labels, as float32.

mmcv-lite has no compiled ops. `mmcv.ops` is stubbed with empty modules, and the one op the detector calls,
`batched_nms`, is replaced by `batched_nms` below: mmcv 2.1.0's Python logic (boxes shifted by
`label * (largest coordinate + 1)`; one NMS call, or one per class from 10000 boxes on) over torchvision's `nms`
(the same greedy rule: a box is dropped when its IoU with a kept one is above the threshold).

    FASTFORKS_GOLDEN_PATH=... uv run --with opencv-python-headless==4.10.0.84 --with scikit-image --with matplotlib \
        python tools/golden/rtmdet.py MMDET_SITE_PACKAGES [rtmdet-tiny-coco ...]
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
import torchvision
from _common import goldens, write

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict
from fastforks.detection.models.rtmdet.model import extra

FAMILY = "rtmdet"
UPSTREAM = "https://github.com/open-mmlab/mmdetection"
REF_COMMIT = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"  # v3.3.0
REF_COMMIT_MMPOSE = "5408bc76f5b848cf925a0d1857899011d8c5b497"  # v1.3.2, the person detectors' configs
OPENCV_VERSION = "4.10.0"
IMAGES = {  # skimage data file -> sha256
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
}
CONFIGS = {  # variant -> (model-zoo config in mmdet/.mim/configs/rtmdet, image, score threshold: mid-gap)
    "rtmdet-tiny-coco": ("rtmdet_tiny_8xb32-300e_coco.py", "coffee.png", 0.16),
    "rtmdet-s-coco": ("rtmdet_s_8xb32-300e_coco.py", "coffee.png", 0.277),
    "rtmdet-m-coco": ("rtmdet_m_8xb32-300e_coco.py", "coffee.png", 0.213),
    "rtmdet-l-coco": ("rtmdet_l_8xb32-300e_coco.py", "coffee.png", 0.18),
    "rtmdet-x-coco": ("rtmdet_x_8xb32-300e_coco.py", "coffee.png", 0.159),
    "rtmdet-nano-person": ("rtmdet_l_8xb32-300e_coco.py", "astronaut.png", 0.171),
    "rtmdet-m-person": ("rtmdet_m_8xb32-300e_coco.py", "astronaut.png", 0.106),
}
_TEST = {
    "model.test_cfg.nms_pre": 1000,
    "model.test_cfg.score_thr": 0.05,
    "model.test_cfg.nms.iou_threshold": 0.6,
    "model.test_cfg.max_per_img": 100,
}
PERSON = {  # variant -> overrides of the mmdet config (mmpose projects/rtmpose/rtmdet/person/*.py)
    "rtmdet-nano-person": {
        "model.backbone.deepen_factor": 0.33,
        "model.backbone.widen_factor": 0.25,
        "model.backbone.use_depthwise": True,
        "model.neck.in_channels": [64, 128, 256],
        "model.neck.out_channels": 64,
        "model.neck.num_csp_blocks": 1,
        "model.neck.use_depthwise": True,
        "model.bbox_head.in_channels": 64,
        "model.bbox_head.feat_channels": 64,
        "model.bbox_head.share_conv": False,
        "model.bbox_head.exp_on_reg": False,
        "model.bbox_head.use_depthwise": True,
        "model.bbox_head.num_classes": 1,
        **_TEST,
    },
    "rtmdet-m-person": {"model.bbox_head.num_classes": 1, **_TEST},
}
PERSON_SIZE = {"rtmdet-nano-person": 320}  # test pipeline `Resize` scale and `Pad` size (the others: the config's 640)


class _Stub(types.ModuleType):
    """An importable module whose every attribute is an empty class (the names mmdet imports but never calls)."""

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


def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, nms_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """`mmcv.ops.batched_nms` (2.1.0) for class-wise `type="nms"`, with torchvision's NMS operator."""
    cfg = dict(nms_cfg)
    split, iou = cfg.pop("split_thr", 10000), cfg.pop("iou_threshold")
    if cfg.pop("type", "nms") != "nms" or cfg.pop("class_agnostic", False) or cfg:
        msg = f"batched_nms stand-in: unsupported settings {nms_cfg}"
        raise SystemExit(msg)
    shifted = boxes + (idxs.to(boxes) * (boxes.max() + 1))[:, None]
    if len(boxes) < split:
        keep = torchvision.ops.nms(shifted, scores, iou)
    else:  # mmcv: one NMS per class, then the kept boxes by score
        total = torch.zeros_like(scores, dtype=torch.bool)
        for label in torch.unique(idxs):
            mask = (idxs == label).nonzero().view(-1)
            total[mask[torchvision.ops.nms(shifted[mask], scores[mask], iou)]] = True
        keep = total.nonzero().view(-1)
        keep = keep[scores[keep].sort(descending=True)[1]]
    return torch.cat([boxes[keep], scores[keep, None]], -1), keep


def reference(site: str, card: Any, variant: Any) -> Any:  # noqa: ANN401 - registry records / mmdet
    """The mmdet detector of a variant (CPU, float32, eval) with its checkpoint's `state_dict` loaded strictly."""
    if site not in sys.path:
        sys.path.append(site)
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())
    import mmdet  # noqa: PLC0415  # reference site-packages
    from mmdet.apis import init_detector  # noqa: PLC0415
    from mmdet.models.dense_heads import base_dense_head  # noqa: PLC0415
    from mmengine.config import Config  # noqa: PLC0415

    base_dense_head.batched_nms = batched_nms
    cfg = Config.fromfile(str(Path(mmdet.__file__).parent / ".mim/configs/rtmdet" / CONFIGS[variant.id][0]))
    cfg.merge_from_dict(PERSON.get(variant.id, {}))
    if variant.id in PERSON:
        cfg.model.backbone.init_cfg = None  # the ImageNet backbone download; the checkpoint has every weight
    if variant.id in PERSON_SIZE:
        side = PERSON_SIZE[variant.id]
        for step in cfg.test_dataloader.dataset.pipeline:
            if step.type == "Resize":
                step.scale = (side, side)
            if step.type == "Pad":
                step.size = (side, side)
    model = init_detector(cfg, None, device="cpu").eval()
    (ckpt,) = registry.weights(card, variant)
    sd = load_state_dict(ckpt, key="state_dict", strip="", extra=extra())
    for name in (
        "mean",
        "std",
    ):  # saved by some checkpoints; not a persistent buffer in mmdet 3.3: must equal the config
        saved = sd.pop(f"data_preprocessor.{name}", None)
        if saved is not None and not torch.equal(saved, getattr(model.data_preprocessor, name)):
            msg = f"{ckpt.name}: data_preprocessor.{name} differs from the config"
            raise SystemExit(msg)
    model.load_state_dict(sd, strict=True)
    return model


def main() -> None:
    if not cv2.__version__.startswith(OPENCV_VERSION):
        msg = f"OpenCV {cv2.__version__}, the goldens pin {OPENCV_VERSION}"
        raise SystemExit(msg)
    for card, variant in registry.cards("detection").values():
        if card.family != FAMILY or (sys.argv[2:] and variant.id not in sys.argv[2:]):
            continue
        _, name, threshold = CONFIGS[variant.id]
        png = Path(skimage.data.__file__).parent / name
        if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGES[name]:
            msg = f"{png} is not scikit-image's {name}"
            raise SystemExit(msg)
        img = skimage.io.imread(png)[..., :3]
        model = reference(sys.argv[1], card, variant)
        from mmdet.apis import inference_detector  # noqa: PLC0415  # on sys.path now

        with torch.inference_mode():
            res = inference_detector(model, np.ascontiguousarray(img[..., ::-1])).pred_instances
        scores = res.scores.numpy()
        keep = scores >= threshold
        ref = f"{variant.id}@{variant.revision}"
        maps = {"input": img, "boxes": res.bboxes.numpy()[keep], "scores": scores[keep], "labels": res.labels[keep]}
        test = model.cfg.model.test_cfg
        write(
            ref,
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, mmdet inference_detector ({CONFIGS[variant.id][0]}"
            f"{' + mmpose person overrides' if variant.id in PERSON else ''}; score_thr {test.score_thr}, "
            f"NMS {test.nms.iou_threshold}, torchvision nms for mmcv's), OpenCV {cv2.__version__}, skimage {name} as "
            f"BGR uint8, score threshold {threshold}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load(FAMILY, "detection")
        store["refs"][ref]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump(FAMILY, store, "detection")
        near, n = np.sort(scores)[::-1], int(keep.sum())
        print(f"  {ref}: {n} kept of {len(scores)}, scores around the cut {np.round(near[max(0, n - 2) : n + 3], 3)}")


if __name__ == "__main__":
    main()
