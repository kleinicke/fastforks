"""The authors' detection / promptable-segmentation code behind the call interface `tools/eval_coco.py` uses.

`load(task, ref, tier)` -> an object shaped like a fastforks model of the task, so the eval loop is the same for every
tier: detection models take an RGB uint8 HWC image and return `boxes` (xyxy px), `scores`, `labels` (indices into
`class_names`), `masks` ([N, H, W] bool or None), sorted by descending score, after the card's score threshold;
promptable models have `prompts`, `get_image_embeddings(img)` and `model(emb, input_points=, input_labels=,
multimask_output=)` -> `masks` [P, C, H, W] bool, `iou_scores` [P, C]; the concept model (SAM 3) takes
`model(emb, text=, threshold=)` -> `boxes`, `scores`, `masks`.

Code paths (each family's venv, `bench/bench_upstream.py list`):

- Mask / Faster R-CNN: detectron2 `DefaultPredictor` (the pinned tree of `tools/golden/rcnn*.py`), BGR input,
  `SCORE_THRESH_TEST` 0.05 and 100 detections (the config defaults the cards copy). 007 venv.
- DETR / RT-DETRv2: `bench/detr_original.py build` (official transforms + postprocessor); DETR keeps score >= 0.05
  (card default), RT-DETRv2 all 300 queries (the official postprocessor, as the fastforks tiers of this eval). 007 venv.
- SAM: `segment_anything.SamPredictor.set_image` + `predict_torch` (point coordinates through its `ResizeLongestSide`).
- SAM 2: `SAM2ImagePredictor.set_image` + its `_prep_prompts` / `_predict` (the torch path of `predict`, masks stay on
  the GPU); the authors' README runs it under `torch.autocast("cuda", torch.bfloat16)`: that is the `upstream16` tier.
- SAM 3: transformers `Sam3Processor` + `Sam3Model` (`bench_upstream.Sam3`, geometry encoder always on as the
  golden), vision features once per image (`get_vision_features`), then one forward per text prompt with
  `vision_embeds`, `post_process_instance_segmentation(threshold, mask_threshold=0.5)`. 009-segmentation venv.

Tiers as `tools/_original.py`: `upstream` fp32, `upstream16` = the authors' option (SAM 2 bfloat16 autocast) else
`torch.autocast(float16)` around every call, `upstreamhalf` = `.half()` on every module + float16 inputs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _original
from _data import DATA

AUTHORS_BF16 = {"sam2"}  # families whose authors run inference under bfloat16 autocast
DETR_THRESHOLD = 0.05


@dataclass
class Result:
    """The fields `tools/eval_coco.py` reads."""

    boxes: Any = None
    scores: Any = None
    labels: Any = None
    masks: Any = None
    iou_scores: Any = None


class Backend:
    """`str(model.backend)` for the eval records."""

    def __init__(self, how: str, dtype: str, note: str) -> None:
        self.mode, self.how, self.dtype, self.note = "upstream", how, dtype, note

    def __str__(self) -> str:
        return f"original {self.how} ({self.dtype}){'; ' + self.note if self.note else ''}"


def coco_names() -> list[str]:
    """COCO's 80 category names in sorted category-id order."""
    cats = json.loads((DATA / "coco/annotations/instances_val2017.json").read_text())["categories"]
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def coco_ids() -> list[int]:
    cats = json.loads((DATA / "coco/annotations/instances_val2017.json").read_text())["categories"]
    return sorted(c["id"] for c in cats)


class _Tier:
    """Precision wrapper: autocast around calls (`ac` dtype or None) after optional `.half()` of `nets`."""

    def __init__(self, tier: str, family: str, nets: list[Any], flag_dtype: str | None = None) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self.ac = None
        if tier == "upstream":
            self.how, self.dtype, note = "fp32", "float32", ""
        elif tier == "upstream16" and flag_dtype:
            self.how, self.dtype, note = "authors flag", flag_dtype, f"the authors' {flag_dtype} autocast (README)"
            self.ac = getattr(torch, flag_dtype)
        elif tier in {"upstream16", "upstreamac"}:  # upstreamac: autocast also where the authors have an option
            self.how, self.dtype, note = "autocast", "float16", "torch.autocast(float16) around the call"
            self.ac = torch.float16
        else:
            n = sum(_original.halve(m) for m in nets)
            self.how, self.dtype, note = "half", "float16", f".half() on {n} modules, fp16 inputs"
        self.backend = Backend(self.how, self.dtype, note)
        _ = family

    def ctx(self) -> Any:  # noqa: ANN401
        import contextlib  # noqa: PLC0415

        torch = self.torch
        stack = contextlib.ExitStack()
        stack.enter_context(torch.no_grad())
        if self.ac is not None:
            stack.enter_context(torch.autocast("cuda", self.ac))
        return stack


class _Detector:
    def __init__(self, tier: _Tier, fn: Any, kind: str) -> None:  # noqa: ANN401
        import torch  # noqa: PLC0415

        self.torch, self.t, self.fn, self.kind = torch, tier, fn, kind
        self.backend = tier.backend
        self.class_names = coco_names()
        self._id2idx = {c: i for i, c in enumerate(coco_ids())}

    def __call__(self, img: np.ndarray) -> Result:
        torch = self.torch
        with self.t.ctx():
            boxes, scores, labels, masks = self.fn(img)
        scores = scores.float()
        if self.kind == "detr":  # labels are COCO ids (the unused ones have no name)
            keep = scores >= DETR_THRESHOLD
            idx = torch.tensor([self._id2idx.get(int(c), -1) for c in labels.tolist()], device=scores.device)
            keep &= idx >= 0
            labels = idx
        else:
            keep = torch.ones_like(scores, dtype=torch.bool)
        order = torch.argsort(scores[keep], descending=True, stable=True)
        sel = torch.nonzero(keep)[:, 0][order]
        return Result(boxes[sel].float(), scores[sel], labels[sel], None if masks is None else masks[sel])


def _rcnn(ref: str, tier: str) -> _Detector:
    bu = _original.bench()
    a = bu.FAM["rcnn"]
    a.setup()
    cfg_name, ck = a.CFG[ref]
    cfg = a.get_cfg()
    cfg.merge_from_file(str(a.tr / "configs" / f"{cfg_name}.yaml"))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = str(bu.wfile(f"{ck}.pkl"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    pred = a.DP(cfg)
    t = _Tier(tier, "rcnn", [pred.model])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        ins = pred(np.ascontiguousarray(img[..., ::-1]))["instances"]
        masks = ins.pred_masks if ins.has("pred_masks") else None
        return ins.pred_boxes.tensor, ins.scores, ins.pred_classes, masks

    return _Detector(t, fn, "rcnn")


def _detr(ref: str, tier: str) -> _Detector:
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(_original.BENCH))
    import detr_original  # noqa: PLC0415  # ty: ignore[unresolved-import]

    fn0 = detr_original.build(ref)
    nets = [c.cell_contents for c in fn0.__closure__ or () if isinstance(c.cell_contents, torch.nn.Module)]
    t = _Tier(tier, detr_original.REFS[ref][0], nets)

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        boxes, scores, labels = fn0(img)
        return boxes, scores, labels, None

    return _Detector(t, fn, "detr" if ref.startswith("detr") else "rtdetr")


class _Points:
    """SAM / SAM 2 point-prompt model."""

    prompts = frozenset({"points", "boxes", "masks"})

    def __init__(self, ref: str, tier: str, family: str) -> None:
        import torch  # noqa: PLC0415

        self.torch, self.family = torch, family
        bu = _original.bench()
        a = bu.FAM[family]
        a.setup()
        if family == "sam":
            k = ref.rsplit("-", 1)[1]
            sam = a.reg[a.KEYS[k]]()
            sam.load_state_dict(torch.load(bu.wfile(a.FILES[k]), map_location="cpu", weights_only=True))
            self.pred = a.Pred(sam.cuda().eval())
            net = sam
        else:
            size = ref.removeprefix("sam2.1-hiera-")
            cfg = f"configs/sam2.1/sam2.1_hiera_{a.CFG[size]}.yaml"
            net = a.build_sam2(cfg, str(bu.wfile(f"sam2.1_hiera_{size.replace('-', '_')}.pt")), "cuda")
            self.pred = a.Pred(net)
        self.t = _Tier(tier, family, [net], "bfloat16" if family in AUTHORS_BF16 else None)
        self.backend = self.t.backend

    def get_image_embeddings(self, img: np.ndarray) -> tuple[int, int]:
        with self.t.ctx():
            self.pred.set_image(img)
        return img.shape[:2]

    def __call__(
        self,
        size: tuple[int, int],
        *,
        input_points: Any,  # noqa: ANN401
        input_labels: Any,  # noqa: ANN401
        multimask_output: bool,
    ) -> Result:
        pts, lab = input_points.cuda().float(), input_labels.cuda().int()
        with self.t.ctx():
            if self.family == "sam":
                c = self.pred.transform.apply_coords_torch(pts, size)
                masks, iou, _ = self.pred.predict_torch(c, lab, multimask_output=multimask_output)
            else:
                _, uc, ul, _ = self.pred._prep_prompts(pts, lab, None, None, normalize_coords=True)  # noqa: SLF001
                masks, iou, _ = self.pred._predict(uc, ul, None, None, multimask_output=multimask_output)  # noqa: SLF001
        return Result(masks=masks.bool(), iou_scores=iou.float())


class _Concept:
    """SAM 3 through transformers (the bench adapter's code path) with cached vision features."""

    prompts = frozenset({"text", "boxes"})

    def __init__(self, tier: str) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        a = _original.bench().FAM["sam3"]
        a.setup()
        self.a = a
        self.model = a.ms.Sam3Model.from_pretrained(a.W, dtype=torch.float32).cuda().eval()
        self.t = _Tier(tier, "sam3", [self.model])
        self.backend = self.t.backend

    def get_image_embeddings(self, img: np.ndarray) -> Any:  # noqa: ANN401
        from PIL import Image  # noqa: PLC0415

        inp = self.a.proc(images=Image.fromarray(img), return_tensors="pt").to("cuda")
        with self.t.ctx():
            if self.t.how == "half":
                inp["pixel_values"] = inp["pixel_values"].half()
            vis = self.model.get_vision_features(pixel_values=inp["pixel_values"])
        return vis, img.shape[:2]

    def __call__(self, emb: Any, *, text: str, threshold: float) -> Result:  # noqa: ANN401
        vis, size = emb
        tx = self.a.proc(text=text, return_tensors="pt").to("cuda")
        with self.t.ctx():
            o = self.a.forward(self.model, vision_embeds=vis, input_ids=tx["input_ids"],
                               attention_mask=tx["attention_mask"])  # fmt: skip
            for k in ("pred_logits", "presence_logits", "pred_boxes", "pred_masks"):
                v = getattr(o, k, None)
                if v is not None:
                    setattr(o, k, v.float())
            r = self.a.proc.post_process_instance_segmentation(o, threshold=threshold, mask_threshold=0.5,
                                                               target_sizes=[tuple(size)])[0]  # fmt: skip
        order = self.torch.argsort(r["scores"], descending=True, stable=True)
        return Result(
            boxes=r["boxes"][order].float(), scores=r["scores"][order].float(), masks=r["masks"][order].bool()
        )


def load(task: str, ref: str, tier: str) -> Any:  # noqa: ANN401
    """The original of `ref` (id without `@rev`) in `tier`."""
    ref = ref.split("@", maxsplit=1)[0]
    if task == "detection":
        return _rcnn(ref, tier) if "rcnn" in ref else _detr(ref, tier)
    if ref.startswith("sam3"):
        return _Concept(tier)
    return _Points(ref, tier, "sam2" if ref.startswith("sam2") else "sam")


def own_option(ref: str) -> bool:
    """Whether the authors' code has its own fp16 / mixed-precision inference option (SAM 2's bfloat16 autocast)."""
    return ref.startswith("sam2")
