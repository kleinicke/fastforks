"""The authors' original code for the accuracy evals, in three precisions (`bench/bench_upstream.py` adapters).

Run with the adapter's venv (`bench_upstream.py list` names it). Tiers:

- `upstream` = fp32: the adapter's `build(ref, fp16=False)` (the authors' default code path), no autocast.
- `upstream16` = the authors' own precision option where the code has one (`how` "authors flag": `build(ref, True)`,
  e.g. RAFT-Stereo `mixed_precision`; VGGT / VGGT-Omega / MapAnything / Depth-Anything-3 ship bfloat16 autocast,
  recorded as dtype bfloat16; Depth Pro / Marigold load a float16 model), else `torch.autocast("cuda", float16)`
  around the call (`how` "autocast").
- `upstreamac` = `torch.autocast("cuda", float16)` around the default (fp32) build also for families with an own option
  (the fallback where that option does not match fp32).
- `upstreamhalf` = `.half()` (families without an own option): `build(ref, False)`, then every `torch.nn.Module` the
  returned callable reaches (closure cells, bound objects, dicts / lists) is converted with `.half()`, float tensors
  held in dicts too, and every module casts float32 floating inputs to float16 (as a user's `model.half();
  model(x.half())`). TorchScript modules are wrapped (no hooks there).

The docs' "original fp16" column is the faster of `upstream16` / `upstreamhalf` whose accuracy matches fp32 (the
authors' option when it does); `own_option(task, ref)` says whether `upstreamhalf` is needed at all.

The bench module is `bench/bench_upstream.py` next to the library checkout (`FF_BENCH_DIR` overrides); it imports the
library from this checkout (`FF_UPBENCH_WT`, set here) with its metadata stand-ins, so `import fastforks` works in the
reference venvs after `bench()`.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("FF_BENCH_DIR", ROOT.parent / "bench"))
TIERS = {"upstream": "fp32", "upstream16": "autocast", "upstreamhalf": "half", "upstreamac": "autocast-plain"}
BF16 = {"vggt", "vggt_omega", "mapanything", "depth_anything_3"}  # the authors' mixed precision is bfloat16
NATIVE_HALF = {"depth_pro", "marigold", "rfdetr_keypoint"}  # build(ref, True) is a float16 model, not mixed precision
_SETUP: set[str] = set()
_MOBILE_SAM: list[Any] = []  # the golden script of MobileSAM, see `mobile_sam_recipe`


def bench() -> Any:  # noqa: ANN401 - the bench_upstream module
    """`bench_upstream` imported once (it puts this checkout's src/ and tools/golden on sys.path)."""
    os.environ.setdefault("FF_UPBENCH_WT", str(ROOT))
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    return importlib.import_module("bench_upstream")


def full_ref(task: str, ref: str) -> tuple[str, Any, Any]:
    """`id` or `id@rev` -> (`id@rev`, card, variant) of the registry."""
    from fastforks._core import registry  # noqa: PLC0415

    for k, (card, variant) in registry.cards(task).items():
        if ref in (k, k.split("@")[0], variant.ref):
            return variant.ref, card, variant  # `id@rev`: the bench adapters look variants up by it
    raise KeyError(ref)


@dataclass
class Original:
    """A built original: `prep(args)` (input conversion, untimed) then `__call__` (the timed call)."""

    adapter: Any
    fn: Any
    outer: bool  # torch.autocast(float16) around the call
    dtype: str
    how: str
    note: str = ""
    keep: list[Any] = field(default_factory=list)

    def prep(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """The adapter's conversion of uint8 CUDA CHW inputs to what its code takes."""
        return self.adapter.prep(args) if hasattr(self.adapter, "prep") else args

    def __call__(self, *args: Any) -> Any:  # noqa: ANN401
        """The original call under `no_grad` (and autocast for the autocast tier)."""
        import torch  # noqa: PLC0415

        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=self.outer):
            return self.fn(*args)


def load(task: str, ref: str, tier: str) -> Original:
    """Build `ref`'s original in `tier` (`upstream`, `upstream16`, `upstreamhalf`)."""
    bu = bench()
    full, card, _ = full_ref(task, ref)
    adapter = bu.FAM.get(card.family) or LOCAL.get(card.family)
    if adapter is None:
        msg = f"no original-code adapter for family {card.family}"
        raise NotImplementedError(msg)
    if card.family not in _SETUP:
        adapter.setup()
        _SETUP.add(card.family)
    how = TIERS[tier]
    fam = card.family
    adapter.note = ""
    build = _SPECIAL.get(full.split("@")[0], adapter.build)
    if how == "fp32":
        o = Original(adapter, build(full, False), outer=False, dtype="float32", how="fp32")  # noqa: FBT003
    elif how == "autocast" and fam in NATIVE_HALF:
        o = Original(adapter, build(full, True), outer=False, dtype="float16", how="authors flag",  # noqa: FBT003
                     note="the authors' float16 model option")  # fmt: skip
    elif how == "autocast" and adapter.flag:
        dt = "bfloat16" if fam in BF16 else "float16"
        o = Original(adapter, build(full, True), outer=False, dtype=dt, how="authors flag",  # noqa: FBT003
                     note=f"the authors' mixed-precision option ({dt})")  # fmt: skip
    elif how in {"autocast", "autocast-plain"}:  # upstreamac: autocast also where the authors have an own option
        o = Original(adapter, build(full, False), outer=True, dtype="float16", how="autocast",  # noqa: FBT003
                     note="torch.autocast(float16) around the call")  # fmt: skip
    else:
        fn = build(full, False)  # noqa: FBT003
        n = halve(fn, adapter)
        o = Original(adapter, fn, outer=False, dtype="float16", how="half", note=f".half() on {n} modules, fp16 inputs")
    if getattr(adapter, "note", ""):
        o.note = "; ".join(x for x in (o.note, adapter.note) if x)
    return o


def _moge3(full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """MoGe-3 (the bench adapter skips it): upstream `moge.model.v3` with FlexGEMM, as `tools/golden/moge.py`."""
    from moge.model.v3 import MoGeModel  # noqa: PLC0415  # ty: ignore[unresolved-import]

    # FlexGEMM at its defaults (TF32 sparse convs, adaptive autotune), as a user of the original gets it; the golden
    # reference (tools/golden/moge.py) turns TF32 off itself.
    net = MoGeModel.from_pretrained(bench().wfile(f"{full.split('@', maxsplit=1)[0]}/model.pt")).cuda().eval()
    return lambda x: net.infer(x[None].float() / 255, resolution_level=9, apply_mask=False, use_fp16=fp16)["depth"]


def _dav2_metric(full: str, fp16: bool) -> Any:  # noqa: ANN401, ARG001, FBT001 - fp16: only the autocast tiers
    """DAv2 Metric: upstream `metric_depth/`'s `DepthAnythingV2(max_depth).infer_image`.

    As `tools/golden/depth_anything_v2_metric.py`. Its package is also called `depth_anything_v2` (the relative code
    the family adapter imported): the metric one is imported with those modules moved aside, then they are put back.
    """
    import torch  # noqa: PLC0415

    bu = bench()
    _, ds, size = full.split("@", maxsplit=1)[0].rsplit("-", 2)
    arch, cfg = bu.Dav2.ARCH[f"depth-anything-v2-{size}"]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] == "depth_anything_v2"}
    root = str(bu.FAM["depth_anything_v2"].tree() / "metric_depth")
    sys.path.insert(0, root)
    try:
        net_cls = importlib.import_module("depth_anything_v2.dpt").DepthAnythingV2
    finally:
        sys.path.remove(root)
        for k in [k for k in sys.modules if k.split(".")[0] == "depth_anything_v2"]:
            del sys.modules[k]
        sys.modules.update(saved)
    net = net_cls(encoder=arch, **cfg, max_depth={"hypersim": 20.0, "vkitti": 80.0}[ds])
    bu.use_weights("depth", "depth_anything_v2", full)
    w = bu.wfile(f"depth_anything_v2_metric_{ds}_{arch}.pth")
    net.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
    net = net.cuda().eval()
    return lambda bgr: net.infer_image(bgr, 518)


def _distill(full: str, fp16: bool) -> Any:  # noqa: ANN401, ARG001, FBT001
    """Distill-Any-Depth: the authors' networks and transforms of the Westlake-AGI-Lab repository.

    Small / Base `DepthAnythingV2`, Large `DepthAnything`, as `tools/golden/distill_any_depth.py` (the repository's
    `infer_image` does not run). The checkout is `FF_DISTILL_REPO` (a clone; the commit is the golden script's).
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    bu = bench()
    golden = str(Path(__file__).resolve().parent / "golden")  # its depth_anything_v2.py shadows upstream's: not kept
    sys.path.insert(0, golden)
    try:
        g = importlib.import_module("distill_any_depth")
    finally:
        sys.path.remove(golden)
    if "distillanydepth" not in sys.modules:
        repo = os.environ.get("FF_DISTILL_REPO")
        if not repo:
            msg = "FF_DISTILL_REPO: a clone of Westlake-AGI-Lab/Distill-Any-Depth for the original-code tiers"
            raise RuntimeError(msg)
        sys.path.insert(0, str(bu.reference_tree(Path(repo), g.REF_COMMIT)))
    from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from distillanydepth.depth_anything_v2.util.transform import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        NormalizeImage,
        PrepareForNet,
        Resize,
    )
    from distillanydepth.modeling.archs.dam.dam import DepthAnything  # noqa: PLC0415  # ty: ignore[unresolved-import]

    size = full.split("@", maxsplit=1)[0].rsplit("-", 1)[1]
    _, kw = g.VARIANTS[size]
    net = (DepthAnything if size == "l" else DepthAnythingV2)(**kw)
    from fastforks._core import files, registry  # noqa: PLC0415

    card, variant = registry.resolve("depth", full)  # hash-verified local copy (bench wfile matches basenames)
    path = files.locate(
        variant.weights[0], group=card.family, env_var=registry.WEIGHTS_PATH_ENV, allow_download=False, model=full
    )
    net.load_state_dict(load_file(path))
    net = net.cuda().eval()
    resize = Resize(518, 518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                    resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)  # fmt: skip
    norm, prep = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()

    def call(bgr: Any) -> Any:  # noqa: ANN401 - the adapter's BGR uint8 HWC
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(prep(norm(resize({"image": rgb})))["image"]))[None].cuda()
        d, _ = net(x)
        return F.interpolate(d, bgr.shape[:2], mode="bilinear", align_corners=True)[0, 0]

    return call


_SPECIAL = {  # ref -> build(ref, fp16) where the bench adapter has none
    "moge-3-vitl": _moge3,
    **{f"depth-anything-v2-metric-{d}-{s}": _dav2_metric for d in ("hypersim", "vkitti") for s in "sbl"},
    **{f"distill-any-depth-{s}": _distill for s in "sbl"},
}


def _golden(name: str) -> Any:  # noqa: ANN401 - a tools/golden module
    """`tools/golden/<name>.py` as a module: the family's reference recipe (pins, configs, builders)."""
    golden = str(Path(__file__).resolve().parent / "golden")
    sys.path.insert(0, golden)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(golden)


def _raft(a: Any, full: str) -> Any:  # noqa: ANN401
    """`raft_large` / `raft_small` of torchvision as `tools/golden/raft_flow.py`: 12 flow updates, the last prediction.

    torchvision takes [-1, 1] floats of a size divisible by 8 only: the call converts the uint8 pair, replicate-pads it
    (centred, as upstream RAFT's `InputPadder`) and crops the flow back.
    """
    import torch.nn.functional as F  # noqa: PLC0415
    import torchvision  # noqa: PLC0415
    from torchvision.models.optical_flow import raft_large, raft_small  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.checkpoint import load_state_dict  # noqa: PLC0415

    card, variant = registry.resolve("flow", full)
    size = variant.params.get("size", card.params.get("size"))
    net = (raft_large if size == "large" else raft_small)(weights=None)
    net.load_state_dict(load_state_dict(registry.weights(card, variant, allow_download=False)[0]), strict=True)
    net = net.cuda().eval()
    a.note = f"torchvision {torchvision.__version__} raft_{size}, num_flow_updates=12, [-1, 1] conversion in the call"

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        h, w = im1.shape[-2:]
        ph, pw = -h % 8, -w % 8
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x1, x2 = (F.pad(im[None].float() / 255 * 2 - 1, pad, mode="replicate") for im in (im1, im2))
        return net(x1, x2, num_flow_updates=12)[-1][..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]

    return call


def _sea_raft(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream SEA-RAFT `RAFT` and evaluate.py's `calc_flow` as `tools/golden/sea_raft.py` (`FF_SEA_RAFT_REPO`)."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.checkpoint import load_state_dict  # noqa: PLC0415

    g = _golden("sea_raft")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_SEA_RAFT_REPO")
        if not repo:
            msg = "FF_SEA_RAFT_REPO: a clone of princeton-vl/SEA-RAFT for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
        sys.path[:0] = [str(a.tree / "core"), str(a.tree)]
        g._no_imagenet()  # noqa: SLF001 - the golden's own switch: no ImageNet download at construction
    from raft import RAFT  # noqa: PLC0415  # ty: ignore[unresolved-import]

    card, variant = registry.resolve("flow", full)
    size = variant.params.get("size", card.params.get("size"))
    scale = variant.params.get("scale", card.params["scale"])
    net = RAFT(argparse.Namespace(**json.loads((a.tree / "config/eval" / g.CONFIG[size]).read_text())))
    sd = load_state_dict(registry.weights(card, variant, allow_download=False)[0])
    sd |= {k.replace(".bn3.", ".downsample.1."): v for k, v in sd.items() if ".bn3." in k}
    net.load_state_dict(sd, strict=True)
    net = net.cuda().eval()
    a.note = f"upstream RAFT ({g.CONFIG[size]} architecture) + evaluate.py calc_flow, iters={g.ITERS}, scale={scale}"
    return lambda im1, im2: g.calc_flow(net, im1[None].float(), im2[None].float(), scale)


def _vitpose(a: Any, full: str) -> Any:  # noqa: ANN401
    """HF transformers `VitPoseImageProcessor` + `VitPoseForPoseEstimation` at the pin of `tools/golden/vitpose.py`.

    The uint8 image goes to the processor as upstream's example passes it; its CPU crop (scipy) and
    `post_process_pose_estimation` are inside the timed call, as the library's call covers crop and decode.
    """
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    g = _golden("vitpose")
    if transformers.__version__ != g.REF_VERSION:
        msg = f"transformers {transformers.__version__}, the ViTPose reference is {g.REF_VERSION}"
        raise RuntimeError(msg)
    repo, rev = g.REFS[full.split("@", maxsplit=1)[0]]
    proc = g.VitPoseImageProcessor.from_pretrained(repo, revision=rev)
    net = g.VitPoseForPoseEstimation.from_pretrained(repo, revision=rev).cuda().eval()
    experts = net.config.backbone_config.num_experts
    a.note = (
        f"transformers {g.REF_VERSION} VitPoseImageProcessor (CPU crop) + VitPoseForPoseEstimation + "
        "post_process_pose_estimation, all in the timed call"
    )

    def call(img: Any, boxes: Any) -> Any:  # noqa: ANN401 - uint8 HWC array, xywh lists
        x = proc(img, boxes=[boxes], return_tensors="pt")["pixel_values"].cuda()
        kw = {"dataset_index": torch.zeros(len(boxes), dtype=torch.long, device="cuda")} if experts > 1 else {}
        out = net(x, **kw)
        out.heatmaps = out.heatmaps.float()  # the decode is numpy / scipy: no float16 there
        return proc.post_process_pose_estimation(out, boxes=[boxes])[0]

    return call


def _rtmpose(a: Any, full: str) -> Any:  # noqa: ANN401
    """`inference_topdown` of mmpose (flip test off) on the `tools/golden/rtmpose.py` model (`FF_MMPOSE_SITE`).

    The test pipeline (CPU cv2 crop, normalisation) and the SimCC decode are inside the timed call.
    """
    from fastforks._core import registry  # noqa: PLC0415

    site = os.environ.get("FF_MMPOSE_SITE")
    if not site:
        msg = "FF_MMPOSE_SITE: site-packages with mmpose 1.3.2 (+ mmdet, mmcv, mmengine) for the original-code tiers"
        raise RuntimeError(msg)
    if site not in sys.path:
        sys.path.append(site)
    g = _golden("rtmpose")
    card, variant = registry.resolve("pose", full)
    net = g.reference(site, card, variant, flip=False).cuda()
    from mmpose.apis import inference_topdown  # noqa: PLC0415  # ty: ignore[unresolved-import]

    a.note = (
        "mmpose 1.3.2 inference_topdown (CPU cv2 crop, network, SimCC decode; flip_test off), all in the timed call"
    )
    return lambda bgr, boxes: inference_topdown(net, bgr, boxes, bbox_format="xyxy")


def _rtmo(a: Any, full: str) -> Any:  # noqa: ANN401
    """Time the pinned mmpose bottom-up pipeline, including preprocessing and decoding."""
    from fastforks._core import registry  # noqa: PLC0415

    site = os.environ.get("FF_MMPOSE_SITE")
    if not site:
        msg = "FF_MMPOSE_SITE: site-packages with mmpose 1.3.2 for the original-code tiers"
        raise RuntimeError(msg)
    if site not in sys.path:
        sys.path.append(site)
    g = _golden("rtmo")
    card, variant = registry.resolve("pose", full)
    net = g.reference(site, card, variant).cuda()
    from mmpose.apis import inference_bottomup  # noqa: PLC0415  # ty: ignore[unresolved-import]

    a.note = "mmpose 1.3.2 inference_bottomup (CPU cv2 letterbox, network, NMS, decode), all in the timed call"
    return lambda bgr: inference_bottomup(net, bgr)


def _rtmdet(a: Any, full: str) -> Any:  # noqa: ANN401
    """`inference_detector` of mmdet on the `tools/golden/rtmdet.py` model (`FF_MMDET_SITE`), card thresholds.

    The test pipeline (CPU cv2 resize, padding) and the box postprocessing (NMS: torchvision's operator in mmcv's
    `batched_nms` logic, mmcv-lite has no compiled ops) are inside the timed call. The score threshold and the number
    of detections are the card's defaults (0.05, 100), as the library's rows; the configs' COCO test settings (0.001,
    300) spend most of the call in the per-class NMS loop.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("rtmdet")
    site = os.environ.get("FF_MMDET_SITE")
    if not site:
        msg = "FF_MMDET_SITE: site-packages with mmdet 3.3.0 (+ mmcv-lite, mmengine) for the original-code tiers"
        raise RuntimeError(msg)
    card, variant = registry.resolve("detection", full)
    net = g.reference(site, card, variant).cuda()
    net.bbox_head.test_cfg.score_thr = card.options["score_threshold"]
    net.bbox_head.test_cfg.max_per_img = card.options["max_detections"]
    from mmdet.apis import inference_detector  # noqa: PLC0415  # ty: ignore[unresolved-import]

    a.note = (
        "mmdet 3.3.0 inference_detector (CPU cv2 resize + pad, network, decode, NMS: torchvision nms in mmcv's "
        f"batched_nms logic), score_thr {net.bbox_head.test_cfg.score_thr}, {net.bbox_head.test_cfg.max_per_img} "
        "detections (card defaults), all in the timed call"
    )
    return lambda bgr: inference_detector(net, bgr)


def _segnext(a: Any, full: str) -> Any:  # noqa: ANN401
    """Mmsegmentation `init_model` + the configs' test protocol, as `tools/golden/segnext.py` (NMF bases seeded)."""
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("segnext")
    site = os.environ.get("FF_MMSEG_SITE")
    if not site:
        msg = "FF_MMSEG_SITE: site-packages with mmsegmentation 1.2.2 (+ mmcv, mmengine) for the original-code tiers"
        raise RuntimeError(msg)
    model = g.reference(site, *registry.resolve("segmentation", full)).cuda()
    a.note = f"mmsegmentation 1.2.2 encode_decode, NMF start bases seeded ({g.SEED}) on every call"
    return lambda x: g.whole(model, x[None].float() / 255)[0]


def _convnext(a: Any, full: str) -> Any:  # noqa: ANN401
    """A timm classifier as `tools/golden/libnet_classification.py` builds it, with the card's `[io]` transform on GPU.

    The bench's classification adapter for a family it does not list: antialiased resize of the shorter side, clamp
    after bicubic, centre crop, normalisation, network, all in the timed call.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("libnet_classification")
    card, variant = registry.resolve("classification", full)
    io = variant.io
    a.sizes = ((io.crop, io.crop),)  # Benchmark each classifier at its native crop, including 384/512 variants.
    net = g._timm({**card.params, **variant.params}["arch"]).cuda().eval()  # noqa: SLF001 - the golden's builder
    tf = g.TF
    a.COMMIT = f"{card.upstream}@{card.upstream_commit}"  # as the bench's classification rows

    def fn(x: Any) -> Any:  # noqa: ANN401 - uint8 CHW
        t = tf.resize(x[None].float() / 255, [io.resize], interpolation=tf.InterpolationMode.BICUBIC, antialias=True)
        t = tf.center_crop(t.clamp(0, 1), [io.crop])
        return net(tf.normalize(t, list(g.IMAGENET_MEAN), list(g.IMAGENET_STD)))[0]

    return fn


def efficientvit_sam_recipe() -> Any:  # noqa: ANN401 - tools/golden/efficientvit_sam.py with upstream importable
    """The golden script of EfficientViT-SAM, its pinned upstream trees on `sys.path` (once).

    The checkouts are `FF_EFFICIENTVIT_REPO` (mit-han-lab/efficientvit) and `FF_SAM_REPO` (segment-anything, its
    prompt encoder and mask decoder).
    """
    g = _golden("efficientvit_sam")
    if "efficientvit_sam trees" not in _SETUP:  # `load` keys _SETUP by family (the adapter setup)
        repo, sam = os.environ.get("FF_EFFICIENTVIT_REPO"), os.environ.get("FF_SAM_REPO")
        if not repo or not sam:
            msg = "FF_EFFICIENTVIT_REPO and FF_SAM_REPO: clones of mit-han-lab/efficientvit and segment-anything"
            raise RuntimeError(msg)
        g.trees(Path(repo), Path(sam))
        _SETUP.add("efficientvit_sam trees")
    return g


def _efficientvit_sam(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `EfficientViTSamPredictor` as `tools/golden/efficientvit_sam.py`: `set_image`, then one point prompt.

    Returns the three low-resolution mask logits, as the bench's SAM call.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = efficientvit_sam_recipe()
    card, variant = registry.resolve("promptable_segmentation", full)
    pred = g.predictor(variant.params["arch"], registry.weights(card, variant, allow_download=False)[0])
    pred.model.cuda()
    a.note = "upstream EfficientViTSamPredictor.set_image + predict (centre point, multimask_output=True)"

    def call(img: Any, points: Any, labels: Any) -> Any:  # noqa: ANN401 - uint8 HWC array, [1, 2] and [1] arrays
        pred.set_image(img)
        return pred.predict(point_coords=points, point_labels=labels, multimask_output=True)[2]

    return call


def sam_hq_recipe() -> Any:  # noqa: ANN401 - tools/golden/sam_hq.py with upstream importable
    """The golden script of HQ-SAM, its pinned upstream tree on `sys.path` (once).

    The checkout is `FF_SAM_HQ_REPO` (SysCV/sam-hq). Its package is `segment_anything`, like SAM's: one of the two per
    process.
    """
    g = _golden("sam_hq")
    if "sam_hq tree" not in _SETUP:  # `load` keys _SETUP by family (the adapter setup)
        repo = os.environ.get("FF_SAM_HQ_REPO")
        if not repo:
            msg = "FF_SAM_HQ_REPO: a clone of SysCV/sam-hq for the original-code tiers"
            raise RuntimeError(msg)
        g.tree(Path(repo))
        _SETUP.add("sam_hq tree")
    return g


def _sam_hq(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `SamPredictor` of SysCV/sam-hq as `tools/golden/sam_hq.py`: `set_image`, then one point prompt.

    Returns the low-resolution mask logits (one mask: SAM's best of three plus the HQ mask, `hq_token_only=False`).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = sam_hq_recipe()
    card, variant = registry.resolve("promptable_segmentation", full)
    key = variant.id.removeprefix("sam-hq-").replace("-", "_")
    pred = g.predictor(key, registry.weights(card, variant, allow_download=False)[0])
    pred.model.cuda()
    a.note = "upstream SamPredictor.set_image + predict (centre point, multimask_output=True, hq_token_only=False)"

    def call(img: Any, points: Any, labels: Any) -> Any:  # noqa: ANN401 - uint8 HWC array, [1, 2] and [1] arrays
        pred.set_image(img)
        return pred.predict(point_coords=points, point_labels=labels, multimask_output=True, hq_token_only=False)[2]

    return call


def mobile_sam_recipe() -> Any:  # noqa: ANN401 - tools/golden/mobile_sam.py with upstream importable
    """The golden script of MobileSAM, with upstream's package imported from its pinned tree (once).

    The checkout is `FF_MOBILE_SAM_REPO` (ChaoningZhang/MobileSAM; it imports timm). The script and upstream's package
    are both called `mobile_sam`: the script is taken out of `sys.modules`, so its imports find the package.
    """
    if not _MOBILE_SAM:
        repo = os.environ.get("FF_MOBILE_SAM_REPO")
        if not repo:
            msg = "FF_MOBILE_SAM_REPO: a clone of ChaoningZhang/MobileSAM for the original-code tiers"
            raise RuntimeError(msg)
        g = _golden("mobile_sam")
        del sys.modules["mobile_sam"]
        g.tree(Path(repo))
        importlib.import_module("mobile_sam")
        _MOBILE_SAM.append(g)
    return _MOBILE_SAM[0]


def _mobile_sam(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `SamPredictor` of MobileSAM as `tools/golden/mobile_sam.py`: `set_image`, then one point prompt.

    Returns the three low-resolution mask logits, as the bench's SAM call.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = mobile_sam_recipe()
    card, variant = registry.resolve("promptable_segmentation", full)
    pred = g.predictor(registry.weights(card, variant, allow_download=False)[0])
    pred.model.cuda()
    a.note = "upstream SamPredictor.set_image + predict (centre point, multimask_output=True)"

    def call(img: Any, points: Any, labels: Any) -> Any:  # noqa: ANN401 - uint8 HWC array, [1, 2] and [1] arrays
        pred.set_image(img)
        return pred.predict(point_coords=points, point_labels=labels, multimask_output=True)[2]

    return call


def _hwc_point(a: tuple[Any, ...]) -> tuple[Any, ...]:
    """(uint8 CUDA CHW, points, labels) of the bench call -> (uint8 HWC array, [1, 2] float array, [1] int array)."""
    import numpy as np  # noqa: PLC0415

    return a[0].permute(1, 2, 0).cpu().numpy(), np.array(a[1], np.float64), np.array(a[2])


class _Local:
    """An original the bench has no adapter for, on the library's own bench inputs as uint8 CUDA tensors.

    `sizes` are the inputs of the family's docs/data/speed.jsonl rows; `fastforks.<task>` `bench_inputs` makes the
    call's arguments (flow: the synthetic pair; pose: the image and one person box per 128 px of width).
    """

    flag = False  # no mixed-precision option of its own: fp16 = torch.autocast around the call

    def __init__(
        self,
        task: str,
        build: Any,  # noqa: ANN401 - heterogeneous upstream adapter callback
        sizes: tuple[tuple[int, int], ...],
        prep: Any = None,  # noqa: ANN401 - heterogeneous upstream input conversion
        *,
        needs_boxes: bool = True,
    ) -> None:
        self.task, self._build, self.sizes, self._prep = task, build, sizes, prep
        self.needs_boxes = needs_boxes
        self.note = self.COMMIT = ""

    def setup(self) -> None:
        """Nothing to prepare: the reference checkouts and site-packages come from the environment."""

    def build(self, full: str, fp16: bool) -> Any:  # noqa: ANN401, ARG002, FBT001 - the adapter signature
        """The family's builder; `COMMIT` = the card's upstream pin."""
        self.COMMIT = full_ref(self.task, full)[1].upstream_commit
        return self._build(self, full)

    def prep(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """The untimed conversion of the uint8 CUDA inputs to what the original's entry point takes."""
        return self._prep(args) if self._prep else args

    def inputs(self, full: str) -> list[tuple[str, str, tuple[Any, ...]]]:  # noqa: ARG002 - the adapter signature
        """(label, scene, args) per size: the images as uint8 CUDA CHW, then the call's other tensors (boxes)."""
        import torch  # noqa: PLC0415

        from fastforks._core.bench import BenchSize  # noqa: PLC0415

        api = importlib.import_module(f"fastforks.{self.task}._api")
        like = types.SimpleNamespace(device=torch.device("cuda"), needs_boxes=self.needs_boxes, prompts=())
        out = []
        for h, w in self.sizes:
            args, kw = api.bench_inputs(like, BenchSize(h, w))
            images = tuple((x.squeeze(0) * 255).round().byte() for x in args)
            out.append((f"{h}x{w}", "synthetic (fastforks bench)", (*images, *kw.values())))
        return out


class _RFDETRKeypoint(_Local):
    """The authors' explicit inference dtype option, without a compiler, on the same CUDA uint8 inputs."""

    flag = True

    def __init__(self) -> None:
        super().__init__("pose", None, ((1080, 1920),), needs_boxes=False)

    def build(self, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - bench adapter contract
        import torch  # noqa: PLC0415

        from fastforks._core import registry  # noqa: PLC0415

        checkout = os.environ.get("FF_RFDETR_REPO")
        if not checkout:
            msg = "FF_RFDETR_REPO must name a checkout containing the pinned RF-DETR commit"
            raise RuntimeError(msg)
        site = os.environ.get("FF_RFDETR_SITE")
        if site and site not in sys.path:
            sys.path.append(site)
        g = _golden("rfdetr_keypoint")
        card, variant = registry.resolve("pose", full)
        (checkpoint,) = registry.weights(card, variant)
        model = g.reference(Path(checkout), checkpoint, "cuda")
        self.COMMIT = card.upstream_commit
        self.note = "pinned RF-DETR torchvision resize/normalize, original network and uncertainty-fused PostProcess"
        if fp16:
            model.inference(compile=False, dtype=torch.float16, inplace=True)
            self.note += "; authors inference(compile=False, dtype=float16, inplace=True)"

        def call(image: Any, threshold: float = 0.5) -> Any:  # noqa: ANN401 - uint8 CUDA CHW image
            x = image.unsqueeze(0) if image.ndim == 3 else image  # noqa: PLR2004 - CHW input
            x = x.float() / 255 if x.dtype == torch.uint8 else x.float()
            return g.predict(model, x, threshold)

        return call


def _xywh(boxes: Any) -> list[list[float]]:  # noqa: ANN401 - [P, 4] xyxy tensor
    return [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in boxes.tolist()]


LOCAL = {  # family -> adapter, where the bench has none (flow, pose, RTMDet, ConvNeXt, TinyViT)
    "raft": _Local("flow", _raft, ((1080, 1920), (375, 1242))),
    "sea_raft": _Local("flow", _sea_raft, ((1080, 1920), (375, 1242))),
    "vitpose": _Local("pose", _vitpose, ((1080, 1920),), lambda a: (a[0].permute(1, 2, 0).cpu().numpy(), _xywh(a[1]))),
    "rtmpose": _Local(
        "pose", _rtmpose, ((1080, 1920),), lambda a: (a[0].flip(0).permute(1, 2, 0).cpu().numpy(), a[1].cpu().numpy())
    ),
    "rfdetr_keypoint": _RFDETRKeypoint(),
    "rtmo": _Local(
        "pose",
        _rtmo,
        ((1080, 1920),),
        lambda a: (a[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy(),),
        needs_boxes=False,
    ),
    "rtmdet": _Local(
        "detection", _rtmdet, ((800, 1344),), lambda a: (a[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy(),)
    ),
    "convnext": _Local("classification", _convnext, ((224, 224),)),
    "repvit": _Local("classification", _convnext, ((224, 224),)),
    "tinyvit": _Local("classification", _convnext, ((224, 224),)),
    "hgnetv2": _Local("classification", _convnext, ((224, 224),)),
    "efficientformerv2": _Local("classification", _convnext, ((224, 224),)),
    "coatnet": _Local("classification", _convnext, ((224, 224),)),
    "efficientformer": _Local("classification", _convnext, ((224, 224),)),
    "focalnet": _Local("classification", _convnext, ((224, 224),)),
    "davit": _Local("classification", _convnext, ((224, 224),)),
    "segnext": _Local("segmentation", _segnext, ((512, 512), (1024, 2048))),
    "shvit": _Local("classification", _convnext, ((224, 224),)),
    "starnet": _Local("classification", _convnext, ((224, 224),)),
    "efficientvit_sam": _Local("promptable_segmentation", _efficientvit_sam, ((1024, 1024),), _hwc_point),
    "sam_hq": _Local("promptable_segmentation", _sam_hq, ((1024, 1024),), _hwc_point),
    "mobile_sam": _Local("promptable_segmentation", _mobile_sam, ((1024, 1024),), _hwc_point),
}


def own_option(task: str, ref: str) -> bool:
    """Whether the authors' code has its own fp16 / mixed-precision option (then `upstreamhalf` is not run)."""
    bu = bench()
    _, card, _ = full_ref(task, ref)
    a = bu.FAM.get(card.family) or LOCAL.get(card.family)
    return card.family in NATIVE_HALF or bool(a is not None and a.flag)


def _cast(x: Any, src: Any, dst: Any) -> Any:  # noqa: ANN401
    import torch  # noqa: PLC0415

    if isinstance(x, torch.Tensor):
        return x.to(dst) if x.dtype == src else x
    if isinstance(x, list):
        return [_cast(v, src, dst) for v in x]
    if isinstance(x, tuple):
        vals = [_cast(v, src, dst) for v in x]
        return type(x)(*vals) if hasattr(x, "_fields") else tuple(vals)
    if isinstance(x, dict):
        for k in list(x):
            x[k] = _cast(x[k], src, dst)
        return x
    return x


def _reach(obj: Any, out: dict[int, Any], cells: list[Any], seen: set[int], depth: int = 0) -> None:  # noqa: ANN401, C901, PLR0911, PLR0912
    """Modules reachable from `obj`; float tensors in dicts are halved on the way; script modules' cells recorded."""
    import torch  # noqa: PLC0415

    if id(obj) in seen or depth > 6:  # noqa: PLR2004
        return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        out[id(obj)] = obj
        return
    if isinstance(obj, (type, types.ModuleType, str, bytes, int, float, bool, torch.Tensor)) or obj is None:
        return
    if isinstance(obj, types.MethodType):
        _reach(obj.__self__, out, cells, seen, depth + 1)
        _reach(obj.__func__, out, cells, seen, depth + 1)
        return
    if isinstance(obj, types.FunctionType):
        for c in obj.__closure__ or ():
            try:
                v = c.cell_contents
            except ValueError:
                continue
            if isinstance(v, torch.jit.ScriptModule):
                cells.append(c)
            _reach(v, out, cells, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                obj[k] = v.half()
            else:
                _reach(v, out, cells, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _reach(v, out, cells, seen, depth + 1)
        return
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for v in list(d.values()):
            _reach(v, out, cells, seen, depth + 1)


class _HalfIn:
    """A TorchScript module called with float32 inputs cast to float16."""

    def __init__(self, m: Any) -> None:  # noqa: ANN401
        self.m = m

    def __call__(self, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        return self.m(*_cast(a, torch.float32, torch.float16), **_cast(kw, torch.float32, torch.float16))

    def __getattr__(self, k: str) -> Any:  # noqa: ANN401
        return getattr(self.m, k)


def halve(fn: Any, adapter: Any = None) -> int:  # noqa: ANN401
    """`.half()` every module `fn` (and `adapter`) reaches; float32 module inputs cast to float16. -> module count."""
    import torch  # noqa: PLC0415

    found: dict[int, Any] = {}
    cells: list[Any] = []
    seen: set[int] = set()
    _reach(fn, found, cells, seen)
    if adapter is not None:
        for v in list(vars(adapter).values()):
            _reach(v, found, cells, seen)

    def pre(_m: Any, args: Any, kwargs: Any) -> Any:  # noqa: ANN401
        return _cast(args, torch.float32, torch.float16), _cast(kwargs, torch.float32, torch.float16)

    n = 0
    for m in found.values():
        m.half()
        for sub in m.modules():
            if not isinstance(sub, torch.jit.ScriptModule):
                sub.register_forward_pre_hook(pre, with_kwargs=True)
        n += 1
    for c in cells:
        c.cell_contents = _HalfIn(c.cell_contents)
    return n
