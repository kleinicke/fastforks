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
- D-FINE: transformers `DFineForObjectDetection` (the golden's reference) at the card's revision, 640 x 640 PIL
  bilinear input in [0, 1], the RT-DETR postprocess keeping all 300 (as the fastforks tiers of this eval).
- LW-DETR: transformers `LwDetrForObjectDetection` with the weights' image processor (CPU) and the authors'
  PostProcess (the golden's reference, `tools/golden/lwdetr.py`), all `num_select` detections kept.
- DEIM: the official DEIM code (`FF_DEIM_REPO`: a checkout at `DEIM_COMMIT`) built from
  `configs/deim_dfine/deim_hgnetv2_<size>_coco.yml`, the authors' checkpoint, `deploy()`; input and postprocess as
  D-FINE. Its training-only imports (tensorboard, faster-coco-eval, calflops) are stubbed. One DEIM model per process.
- OWLv2: transformers `Owlv2ForObjectDetection` (the golden's reference) with the card's weights, the processor's
  steps on the CPU and the 80 COCO class names as the queries; the 300 best boxes (as the fastforks tiers).
- Grounding DINO: transformers `GroundingDinoForObjectDetection` (the golden's reference) with the card's weights, the
  processor on the CPU, the coco-80 names as one caption; the authors' COCO postprocess (300 best query / class pairs).
- RTMDet: mmdet 3.3.0 `inference_detector` on the model of `tools/golden/rtmdet.py` (`FF_MMDET_SITE`: mmdet,
  mmengine, mmcv-lite; NMS through torchvision's operator), the configs' test settings (COCO: score 0.001, 300
  detections; the person detectors: 0.05, 100).
- Pose: mmpose 1.3.2 `inference_topdown` / `inference_bottomup` (RTMPose, HRNet, RTMO; `tools/_original.py`); ViTPose =
  transformers `VitPoseImageProcessor` + `VitPoseForPoseEstimation` + `post_process_pose_estimation` (the golden's pin).
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
_DEIM: list[str] = []  # the DEIM original built in this process
DEIM_COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"  # the deim-* variants' code and weights


@dataclass
class Result:
    """The fields `tools/eval_coco.py` reads."""

    boxes: Any = None
    scores: Any = None
    labels: Any = None
    masks: Any = None
    iou_scores: Any = None
    keypoints: Any = None
    keypoint_scores: Any = None


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


def _rtmdet(ref: str, tier: str) -> _Detector:
    """The mmdet `inference_detector` on the `tools/golden/rtmdet.py` model with the config's test settings."""
    import os  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _original._golden("rtmdet")  # noqa: SLF001 - the family's reference recipe
    site = os.environ.get("FF_MMDET_SITE")
    if not site:
        msg = "FF_MMDET_SITE: site-packages with mmdet 3.3.0 (+ mmcv-lite, mmengine) for the original-code tiers"
        raise RuntimeError(msg)
    card, variant = registry.resolve("detection", ref)
    net = g.reference(site, card, variant).cuda()
    from mmdet.apis import inference_detector  # noqa: PLC0415  # ty: ignore[unresolved-import]

    t = _Tier(tier, "rtmdet", [net])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        r = inference_detector(net, np.ascontiguousarray(img[..., ::-1])).pred_instances
        return r.bboxes, r.scores, r.labels, None

    det = _Detector(t, fn, "rtmdet")
    if variant.output_info["label_set"] == "person-1":
        det.class_names = ["person"]
    return det


def _dfine(ref: str, tier: str) -> _Detector:
    """The golden's transformers `DFineForObjectDetection` at the card's revision, with its input and postprocess.

    640 x 640 PIL bilinear, [0, 1]; the RT-DETR postprocess (the 300 best of the 300 x 80 sigmoid scores, all kept).
    """
    import re  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    _, variant = registry.resolve("detection", ref)
    hub = re.match(r"https://huggingface.co/([^/]+/[^/]+)/resolve/([0-9a-f]+)/", variant.weights[0].url)
    if hub is None:
        msg = f"{ref}: unexpected weight url {variant.weights[0].url}"
        raise RuntimeError(msg)
    net = transformers.DFineForObjectDetection.from_pretrained(hub[1], revision=hub[2], dtype=torch.float32)
    net = net.to("cuda").eval()
    t = _Tier(tier, "dfine", [net])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        h, w = img.shape[:2]
        x = np.asarray(Image.fromarray(img).resize((640, 640), Image.Resampling.BILINEAR))
        x = torch.from_numpy(x.copy()).cuda().permute(2, 0, 1)[None].float() / 255
        out = net(pixel_values=x.half() if t.how == "half" else x)
        scores, idx = out.logits.float().sigmoid()[0].flatten().topk(300)
        labels, query = idx % 80, idx // 80
        cx, cy, bw, bh = out.pred_boxes.float()[0][query].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)
        return boxes * torch.tensor([w, h, w, h], device=boxes.device), scores, labels, None

    return _Detector(t, fn, "rtdetr")


def _lwdetr(ref: str, tier: str) -> _Detector:
    """The golden's transformers `LwDetrForObjectDetection`, the weights' image processor, the authors' PostProcess.

    The processor runs on the CPU (as `tools/golden/lwdetr.py`); all `num_select` detections kept (coco-80 labels).
    """
    g = _original._golden("lwdetr")  # noqa: SLF001 - the family's reference recipe
    net, num_select = g.reference(ref)
    net = net.to("cuda")
    proc = g.processor(ref.removeprefix("lw-detr-").removesuffix("-coco"))
    t = _Tier(tier, "lwdetr", [net])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        x = proc(images=img, return_tensors="pt")["pixel_values"].cuda()
        d = g.detect(net, num_select, img, x.half() if t.how == "half" else x)
        return d["boxes"], d["scores"], d["labels"], None

    return _Detector(t, fn, "rtdetr")


def _deim(ref: str, tier: str) -> _Detector:
    """The official DEIM model (`FF_DEIM_REPO` at `DEIM_COMMIT`), deploy mode, with D-FINE's input and postprocess."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from unittest.mock import MagicMock  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    if _DEIM and _DEIM[0] != ref:  # the config registry keeps the first model's settings
        msg = f"one DEIM original per process ({_DEIM[0]} was built)"
        raise RuntimeError(msg)
    card, variant = registry.resolve("detection", ref)
    repo = Path(os.environ.get("FF_DEIM_REPO", ""))
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout
    if head.strip() != DEIM_COMMIT:
        msg = f"FF_DEIM_REPO must be a DEIM checkout at {DEIM_COMMIT}"
        raise RuntimeError(msg)
    for name in ("torch.utils.tensorboard", "calflops", "faster_coco_eval", "faster_coco_eval.utils",
                 "faster_coco_eval.utils.pytorch", "faster_coco_eval.core", "faster_coco_eval.core.mask"):  # fmt: skip
        sys.modules.setdefault(name, MagicMock())
    for base in ("FasterCocoDetection", "FasterCocoEvaluator"):
        setattr(sys.modules["faster_coco_eval.utils.pytorch"], base, type(base, (), {}))
    _DEIM.append(ref)
    sys.path.insert(0, str(repo))
    from engine.core import YAMLConfig  # noqa: PLC0415  # ty: ignore[unresolved-import]

    size = ref.split("-")[1]
    cfg = YAMLConfig(str(repo / f"configs/deim_dfine/deim_hgnetv2_{size}_coco.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    (weights,) = registry.weights(card, variant, allow_download=False)
    cfg.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
    net = cfg.model.deploy().cuda().eval()
    t = _Tier(tier, "dfine", [net])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        h, w = img.shape[:2]
        x = np.asarray(Image.fromarray(img).resize((640, 640), Image.Resampling.BILINEAR))
        x = torch.from_numpy(x.copy()).cuda().permute(2, 0, 1)[None].float() / 255
        out = net(x.half() if t.how == "half" else x)
        scores, idx = out["pred_logits"].float().sigmoid()[0].flatten().topk(300)
        labels, query = idx % 80, idx // 80
        cx, cy, bw, bh = out["pred_boxes"].float()[0][query].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)
        return boxes * torch.tensor([w, h, w, h], device=boxes.device), scores, labels, None

    return _Detector(t, fn, "rtdetr")


def _grounding_dino(ref: str, tier: str) -> _Detector:
    """The golden's transformers `GroundingDinoForObjectDetection` (MM: the variant's `arch`), card weights, coco-80.

    `GroundingDinoImageProcessor` on the CPU, `BertTokenizer` on `" . ".join(names) + " ."` and the text tower on every
    call; the authors' `PostProcessCocoGrounding`: mean sigmoid token score per class name, the 300 best (query, class)
    pairs (as the fastforks tiers of this eval: threshold 0, 300 detections).
    """
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    from transformers import BertTokenizer, GroundingDinoImageProcessor  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from fastforks.shared.seginfer import untie_to_checkpoint  # noqa: PLC0415

    card, variant = registry.resolve("detection", ref)
    config_json, weights, vocab = registry.weights(card, variant, allow_download=False)
    arch = variant.params.get("arch", "GroundingDinoForObjectDetection")  # MM-Grounding-DINO: its own class
    config = getattr(transformers, arch.replace("ForObjectDetection", "Config")).from_json_file(str(config_json))
    sd = load_file(str(weights))
    net = getattr(transformers, arch).from_pretrained(None, config=config, state_dict=sd).eval()
    untie_to_checkpoint(net, sd)  # the checkpoint's per-layer heads, which transformers 5.19 ties for MM
    net = net.to("cuda")
    proc = GroundingDinoImageProcessor()
    tok = BertTokenizer(str(vocab))
    names = coco_names()
    text = tok(" . ".join(names) + " .", return_tensors="pt").to("cuda")
    n_ids = text["input_ids"].shape[1]
    classes = torch.zeros(len(names), n_ids, device="cuda")
    at = 1
    for c, n in enumerate(names):
        k = len(tok(n, add_special_tokens=False)["input_ids"])
        classes[c, at : at + k] = 1 / k
        at += k + 1
    t = _Tier(tier, "grounding_dino", [net])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        h, w = img.shape[:2]
        pv = proc(images=img, return_tensors="pt")
        x = pv["pixel_values"].cuda()
        out = net(**text, pixel_values=x.half() if t.how == "half" else x, pixel_mask=pv["pixel_mask"].cuda())
        prob = out.logits.float()[0, :, :n_ids].sigmoid() @ classes.T
        scores, flat = prob.flatten().topk(300)
        cx, cy, bw, bh = out.pred_boxes.float()[0][flat // len(names)].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)
        return boxes * torch.tensor([w, h, w, h], device=boxes.device), scores, flat % len(names), None

    return _Detector(t, fn, "rtdetr")


def _owlv2(ref: str, tier: str) -> _Detector:
    """The golden's transformers `Owlv2ForObjectDetection` with the card's weights, processor and the coco-80 names.

    `Owlv2Processor` steps (CLIP tokenizer padded to 16; image processor: pad, blur, resize, normalise) on the CPU, the
    text tower on every call; `post_process_object_detection`'s scores and boxes, the 300 best kept (as the fastforks
    tiers of this eval: threshold 0, 300 detections).
    """
    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    from transformers import CLIPTokenizer, Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("detection", ref)
    config_json, weights, vocab, merges = registry.weights(card, variant, allow_download=False)
    config = Owlv2Config(**json.loads(Path(config_json).read_text()))
    net = Owlv2ForObjectDetection(config).eval()
    net.load_state_dict(load_file(str(weights)))
    net = net.to("cuda")
    size = config.vision_config.image_size
    proc = Owlv2ImageProcessor(size={"height": size, "width": size})
    text = CLIPTokenizer(str(vocab), str(merges))(
        coco_names(), padding="max_length", max_length=16, truncation=True, return_tensors="pt"
    ).to("cuda")
    t = _Tier(tier, "owlv2", [net])

    def fn(img: np.ndarray) -> tuple[Any, ...]:
        side = max(img.shape[:2])
        x = proc(images=img, return_tensors="pt")["pixel_values"].cuda()
        out = net(**text, pixel_values=x.half() if t.how == "half" else x)
        best, labels = out.logits.float()[0].max(-1)
        scores, keep = best.sigmoid().topk(300)
        cx, cy, bw, bh = out.pred_boxes.float()[0][keep].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * side
        return boxes, scores, labels[keep], None

    return _Detector(t, fn, "rtdetr")


class _Pose:
    """Pinned mmpose top-down or bottom-up inference behind the COCO pose interface."""

    def __init__(self, ref: str, tier: str) -> None:
        from fastforks._core import registry  # noqa: PLC0415

        card, _ = registry.resolve("pose", ref)
        if card.family not in {"rtmpose", "hrnet", "rtmo", "vitpose"}:
            msg = f"no original COCO pose adapter for {card.family}"
            raise NotImplementedError(msg)
        self.family = card.family
        self.original = _original.load("pose", ref, tier)
        self.backend = Backend(self.original.how, self.original.dtype, self.original.note)

    @property
    def needs_boxes(self) -> bool:
        """Top-down families take person boxes; bottom-up (RTMO) detects the people itself."""
        return self.family in {"rtmpose", "hrnet", "vitpose"}

    def __call__(self, img: np.ndarray, *, boxes: np.ndarray | None = None) -> Result:
        import torch  # noqa: PLC0415

        if self.family == "vitpose":  # transformers: RGB, `x, y, w, h` boxes, one dict per box
            xywh = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in np.asarray(boxes).tolist()]
            res = self.original(np.ascontiguousarray(img), xywh) if xywh else []
            if not res:
                return Result(
                    scores=torch.empty(0), keypoints=torch.empty(0, 17, 2), keypoint_scores=torch.empty(0, 17)
                )
            kp = torch.stack([r["keypoints"].float().cpu() for r in res])
            ks = torch.stack([r["scores"].float().cpu() for r in res])
            return Result(scores=ks.mean(1), keypoints=kp, keypoint_scores=ks)
        args = (np.ascontiguousarray(img[..., ::-1]),)
        if self.needs_boxes:
            if boxes is None:
                msg = "top-down reference requires person boxes"
                raise ValueError(msg)
            args = (*args, boxes)
        parts = [r.pred_instances for r in self.original(*args)]
        if not parts:
            return Result(scores=torch.empty(0), keypoints=torch.empty(0, 17, 2), keypoint_scores=torch.empty(0, 17))
        kp = np.concatenate([p.keypoints for p in parts])
        ks = np.concatenate([p.keypoint_scores for p in parts])
        scores = ks.mean(axis=1) if self.needs_boxes else np.concatenate([p.bbox_scores for p in parts])
        return Result(
            scores=torch.as_tensor(scores), keypoints=torch.as_tensor(kp), keypoint_scores=torch.as_tensor(ks)
        )


class _RFDETRPose:
    """Original RF-DETR person queries with all scores retained for COCO OKS evaluation."""

    needs_boxes = False
    task = "pose"

    def __init__(self, ref: str, tier: str) -> None:
        self.original = _original.load(self.task, ref, tier)
        self.backend = Backend(self.original.how, self.original.dtype, self.original.note)

    def __call__(self, img: np.ndarray) -> Result:
        import torch  # noqa: PLC0415

        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).cuda()
        return Result(**self.original(image, 0.0))


class _RFDETR(_RFDETRPose):
    """Original RF-DETR detection / Seg: the `num_select` best (query, class) pairs, `coco-80` labels, no threshold."""

    task = "detection"

    def __init__(self, ref: str, tier: str) -> None:
        super().__init__(ref, tier)
        self.class_names = coco_names()


class _Points:
    """SAM / SAM 2 / EfficientViT-SAM point-prompt model."""

    prompts = frozenset({"points", "boxes", "masks"})

    def __init__(self, ref: str, tier: str, family: str) -> None:
        import torch  # noqa: PLC0415

        self.torch, self.family = torch, family
        bu = _original.bench()
        a = bu.FAM.get(family)
        if a is not None:
            a.setup()
        if family == "efficientvit_sam":
            from fastforks._core import registry  # noqa: PLC0415

            card, variant = registry.resolve("promptable_segmentation", ref)
            weights = registry.weights(card, variant, allow_download=False)[0]
            self.pred = _original.efficientvit_sam_recipe().predictor(variant.params["arch"], weights)
            net = self.pred.model.cuda()
        elif family == "sam_hq":
            from fastforks._core import registry  # noqa: PLC0415

            card, variant = registry.resolve("promptable_segmentation", ref)
            weights = registry.weights(card, variant, allow_download=False)[0]
            key = variant.id.removeprefix("sam-hq-").replace("-", "_")
            self.pred = _original.sam_hq_recipe().predictor(key, weights)
            net = self.pred.model.cuda()
        elif family == "mobile_sam":
            from fastforks._core import registry  # noqa: PLC0415

            card, variant = registry.resolve("promptable_segmentation", ref)
            weights = registry.weights(card, variant, allow_download=False)[0]
            self.pred = _original.mobile_sam_recipe().predictor(weights)
            net = self.pred.model.cuda()
        elif family == "sam":
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
            if self.family in {"sam", "sam_hq", "efficientvit_sam", "mobile_sam"}:  # the same predictor interface
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


def load(task: str, ref: str, tier: str) -> Any:  # noqa: ANN401, C901, PLR0911, PLR0912 - one branch per family
    """The original of `ref` (id without `@rev`) in `tier`."""
    ref = ref.split("@", maxsplit=1)[0]
    if task == "pose":
        return _RFDETRPose(ref, tier) if ref.startswith("rf-detr-keypoint") else _Pose(ref, tier)
    if task == "detection":
        if ref.startswith("rtmdet"):
            return _rtmdet(ref, tier)
        if ref.startswith("dfine"):
            return _dfine(ref, tier)
        if ref.startswith("deim"):
            return _deim(ref, tier)
        if ref.startswith("lw-detr"):
            return _lwdetr(ref, tier)
        if ref.startswith("owlv2"):
            return _owlv2(ref, tier)
        if "grounding-dino" in ref:  # also mm-grounding-dino
            return _grounding_dino(ref, tier)
        if ref.startswith("rf-detr"):
            return _RFDETR(ref, tier)
        return _rcnn(ref, tier) if "rcnn" in ref else _detr(ref, tier)
    if ref.startswith("sam3"):
        return _Concept(tier)
    if ref.startswith("efficientvit-sam"):
        return _Points(ref, tier, "efficientvit_sam")
    if ref.startswith("sam-hq"):
        return _Points(ref, tier, "sam_hq")
    if ref.startswith("mobile-sam"):
        return _Points(ref, tier, "mobile_sam")
    return _Points(ref, tier, "sam2" if ref.startswith("sam2") else "sam")


def own_option(ref: str) -> bool:
    """Whether the authors' code has its own fp16 / mixed-precision inference option (SAM 2's bfloat16 autocast)."""
    return ref.startswith("sam2")
