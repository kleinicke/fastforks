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
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("FF_BENCH_DIR", ROOT.parent / "bench"))
TIERS = {"upstream": "fp32", "upstream16": "autocast", "upstreamhalf": "half", "upstreamac": "autocast-plain"}
BF16 = {
    "vggt",
    "vggt_omega",
    "mapanything",
    "depth_anything_3",
    "romav2",
    "anysplat",
    "pi3",
    "insid3",
    "megaflow",
}  # the authors' mixed precision is bfloat16
NATIVE_HALF = {
    "depth_pro",
    "marigold",
    "marigold_normals",
    "lotus",
    "birefnet",
    "rfdetr_keypoint",
    "rfdetr",
}  # build(ref, True) is a float16 model, not mixed precision
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
        """The adapter's conversion of uint8 CUDA CHW inputs to what its code takes (then the part's own, if any)."""
        args = self.adapter.prep(args) if hasattr(self.adapter, "prep") else args
        return self.fn.prep(args) if hasattr(self.fn, "prep") else args

    def __call__(self, *args: Any) -> Any:  # noqa: ANN401
        """The original call under `no_grad` (and autocast for the autocast tier)."""
        import torch  # noqa: PLC0415

        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=self.outer):
            return self.fn(*args)


def load(task: str, ref: str, tier: str, part: str = "call") -> Original:
    """Build `ref`'s original in `tier` (`upstream`, `upstream16`, `upstreamhalf`).

    `part` "network" (pose only): the crops / batch are made in the untimed `prep`, the call is the network alone.
    """
    bu = bench()
    full, card, _ = full_ref(task, ref)
    adapter = LOCAL.get(card.family) or bu.FAM.get(card.family)
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
    if part == "network":
        if fam not in NETWORK:
            msg = f"no network-only original for family {fam}"
            raise NotImplementedError(msg)
        adapter.COMMIT = card.upstream_commit
        build = lambda full, _fp16, _b=NETWORK[fam]: _b(adapter, full)  # noqa: E731
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
    """`tools/golden/<name>.py` as a module: the family's reference recipe (pins, configs, builders).

    Registered as `golden_<name>`: a recipe named like its upstream package (`romav2`) must not shadow it.
    """
    golden = Path(__file__).resolve().parent / "golden"
    key = f"golden_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, golden / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(golden))  # the recipe's own imports (_common, raft_flow)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    finally:
        sys.path.remove(str(golden))
    return module


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
    if scale == 0:
        return lambda im1, im2: g.calc_flow(net, im1[None].float(), im2[None].float(), scale)
    import torch.nn.functional as F  # noqa: PLC0415

    a.note += "; an odd size (calc_flow returns one row / column short) repeats the last row / column, as our port"

    def run(im1: Any, im2: Any) -> Any:  # noqa: ANN401
        flow = g.calc_flow(net, im1[None].float(), im2[None].float(), scale)
        h, w = im1.shape[-2:]
        return F.pad(flow, (0, w - flow.shape[-1], 0, h - flow.shape[-2]), mode="replicate")

    return run


def _memfof(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `MEMFOF` and `scale_and_forward_flow` as `tools/golden/memfof.py` (`FF_MEMFOF_REPO`).

    Frames (image1, image1, image2) in [0, 255], the forward flow of the last update. The reference environment
    supplies einops and huggingface_hub (upstream imports them).
    """
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.checkpoint import load_state_dict  # noqa: PLC0415

    g = _golden("memfof")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_MEMFOF_REPO")
        if not repo:
            msg = "FF_MEMFOF_REPO: a clone of msu-video-group/memfof for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
        sys.path[:0] = [str(a.tree)]
    import torch  # noqa: PLC0415
    from memfof.model import MEMFOF  # noqa: PLC0415  # ty: ignore[unresolved-import]

    card, variant = registry.resolve("flow", full)
    scale = variant.params.get("scale", card.params["scale"])
    net = MEMFOF(backbone_weights=None)
    net.load_state_dict(load_state_dict(registry.weights(card, variant, allow_download=False)[0]), strict=True)
    net = net.cuda().eval()
    a.note = (
        f"upstream MEMFOF + scale_and_forward_flow, iters={g.ITERS}, scale={scale}, frames (image1, image1, image2)"
    )

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        t1, t2 = im1[None].float(), im2[None].float()
        return g.scale_and_forward_flow(net, torch.stack([t1, t1, t2], dim=1), scale)

    return call


def _memflow(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream MemFlowNet + `InferenceCore.step(end=True)` per pair as `tools/golden/memflow.py` (`FF_MEMFLOW_REPO`).

    A fresh `InferenceCore` per call (empty memory: upstream's first pair of a sequence), the frames padded as upstream
    `InputPadder` (centred, replicate, multiples of 8) and the flow cropped. Upstream's inference has no
    mixed-precision path (`mixed_precision` is a training setting): the fp16 tier is autocast. The reference
    environment supplies yacs, loguru, einops and scipy.
    """
    import importlib as il  # noqa: PLC0415
    import types  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.checkpoint import load_state_dict  # noqa: PLC0415

    g = _golden("memflow")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_MEMFLOW_REPO")
        if not repo:
            msg = "FF_MEMFLOW_REPO: a clone of DQiaole/MemFlow for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
        sys.path[:0] = [str(a.tree), str(a.tree / "core")]
        sys.modules.setdefault("timm", types.ModuleType("timm"))  # only the Twins-SVT encoders use it
    from core.Networks import build_network  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from inference.inference_core_skflow import InferenceCore  # noqa: PLC0415  # ty: ignore[unresolved-import]

    card, variant = registry.resolve("flow", full)
    cfg = il.import_module(f"configs.{g.CONFIGS[variant.id]}").get_cfg()
    net = build_network(cfg).eval()
    (ckpt,) = registry.weights(card, variant, allow_download=False)
    net.load_state_dict(load_state_dict(ckpt, key="model", strip=""), strict=True)
    net = net.cuda()
    a.note = f"upstream MemFlowNet configs/{g.CONFIGS[variant.id]}.py, InferenceCore.step(end=True) per pair"

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        h, w = im1.shape[-2:]
        ph, pw = -h % 8, -w % 8
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x = torch.stack([F.pad(im[None].float(), pad, mode="replicate")[0] for im in (im1, im2)])[None]
        _, flow = InferenceCore(net, config=cfg).step(2 * (x / 255.0) - 1.0, end=True)
        return flow[..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]

    return call


def _flowformer(a: Any, full: str) -> Any:  # noqa: ANN401
    """Authors' FlowFormer Small, full-frame inference with their centred padding.

    The reference environment supplies timm 0.4.12, yacs, einops and loguru. The fp16 tier is ordinary autocast;
    upstream has no mixed-precision setting.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("flowformer")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_FLOWFORMER_REPO")
        if not repo:
            msg = "FF_FLOWFORMER_REPO: a clone of drinkingcoder/FlowFormer-Official for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
    card, variant = registry.resolve("flow", full)
    net = g.build(a.tree, registry.weights(card, variant, allow_download=False)[0]).cuda()
    m = variant.io.pad_multiple
    a.note = f"upstream FlowFormer Small, iters={g.ITERS}, full frame padded to multiples of {m}"

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        h, w = im1.shape[-2:]
        ph, pw = -h % m, -w % m
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x1, x2 = (F.pad(im[None].float(), pad, mode="replicate") for im in (im1, im2))
        return g.calc_flow(net, x1, x2)[..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]

    return call


def _gmflow(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `UniMatch` as `tools/golden/gmflow.py` (`FF_GMFLOW_REPO`), settings of scripts/gmflow_evaluate.sh.

    Upstream has no mixed-precision option. The call replicate-pads the pair to the card's multiple (centred, as
    upstream's InputPadder) and crops the flow.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("gmflow")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_GMFLOW_REPO")
        if not repo:
            msg = "FF_GMFLOW_REPO: a clone of autonomousvision/unimatch for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
    card, variant = registry.resolve("flow", full)
    params = dict(card.params | variant.params)
    net = g.build(a.tree, registry.weights(card, variant, allow_download=False)[0], params).cuda()
    m = variant.io.pad_multiple
    a.note = f"upstream UniMatch {g.settings(params)}, padded to multiples of {m}"

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        h, w = im1.shape[-2:]
        ph, pw = -h % m, -w % m
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x1, x2 = (F.pad(im[None].float(), pad, mode="replicate") for im in (im1, im2))
        return g.calc_flow(net, x1, x2, params)[..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]

    return call


def _flowseek(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `FlowSeek` (size T) as `tools/golden/flowseek.py` builds it (`FF_FLOWSEEK_REPO`), on CUDA.

    Upstream has no mixed-precision option; it scales and pads the pair itself. `iters` is its config's 4.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("flowseek")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_FLOWSEEK_REPO")
        if not repo:
            msg = "FF_FLOWSEEK_REPO: a clone of mattpoggi/flowseek for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
    card, variant = registry.resolve("flow", full)
    net = g.build(a.tree, registry.weights(card, variant, allow_download=False)[0]).cuda()
    a.note = f"upstream FlowSeek ({g.CONFIG}), iters={g.ITERS}"
    return lambda im1, im2: g.calc_flow(net, im1[None].float(), im2[None].float())


def _waft(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream WAFT `ViTWarpV8` (a1) as `tools/golden/waft.py` builds it (`FF_WAFT_REPO`), on CUDA.

    Upstream evaluates in fp32 (no mixed-precision option); it normalises and pads the pair itself. `iters` is its
    config's 5; the last iteration's flow.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("waft")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_WAFT_REPO")
        if not repo:
            msg = "FF_WAFT_REPO: a clone of princeton-vl/WAFT for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
    card, variant = registry.resolve("flow", full)
    net = g.build(a.tree, registry.weights(card, variant, allow_download=False)[0], variant.id).cuda()
    a.note = f"upstream ViTWarpV8 (config/a1/{g.CONFIG[variant.id]}), iters={g.ITERS}"
    return lambda im1, im2: net(im1[None].float(), im2[None].float(), iters=g.ITERS)["flow"][-1]


def _rife(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream `RIFE_HDv3.Model` as `tools/golden/rife.py`; fp16 = inference_video.py's `--fp16` recipe.

    `FF_RIFE_REPO`: a hzwer/Practical-RIFE clone; `FF_RIFE_RELEASES`: one unpacked release per variant id
    (`<id>/train_log`). `--fp16` makes half the default tensor type, so the model, its warp grid and the input are
    half; here `torch.set_default_dtype(torch.float16)` while building and calling. Zero pad right / bottom to 128.
    """
    import torch  # noqa: PLC0415

    g = _golden("rife")
    ref = full.split("@", maxsplit=1)[0]
    tree = g.reference_tree(Path(os.environ["FF_RIFE_REPO"]), g.REF_COMMIT)
    dtype = torch.float16 if fp16 else torch.float32
    before = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = g.build(tree, Path(os.environ["FF_RIFE_RELEASES"]) / ref)
    finally:
        torch.set_default_dtype(before)
    model.flownet.to(dtype)
    a.note = "upstream Practical-RIFE Model.inference, timestep 0.5, zero pad to 128" + (
        ", inference_video.py --fp16: half default dtype, model and input" * fp16
    )

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        torch.set_default_dtype(dtype)
        try:
            return g.interpolate(model, im1[None].to(dtype) / 255, im2[None].to(dtype) / 255)
        finally:
            torch.set_default_dtype(before)

    return call


def _gimm_vfi(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `GIMMVFI_R` as `tools/golden/gimm_vfi.py` (video_Nx.py: pad 32, one time 0.5, `ds_factor` 1).

    `FF_GIMM_VFI_REPO`: a GSeanCDAT/GIMM-VFI clone; its imports (omegaconf, einops, easydict) and cupy (the authors'
    CUDA softsplat) come from PYTHONPATH, `CUDA_HOME` its include path. Upstream has no fp16 option (fp16 tier =
    autocast).
    """
    import cupy  # noqa: PLC0415  # ty: ignore[unresolved-import]  # the authors' softsplat kernel

    from fastforks._core import registry  # noqa: PLC0415

    if not hasattr(cupy.cuda, "compile_with_cache"):  # removed in cupy 14 (the CUDA 13 builds): its RawModule
        cupy.cuda.compile_with_cache = lambda code, options=(): cupy.RawModule(code=code, options=options)
    g = _golden("gimm_vfi")
    tree = g.reference_tree(Path(os.environ["FF_GIMM_VFI_REPO"]), g.REF_COMMIT, g.PATCH)
    card, variant = registry.resolve("interpolation", full)
    model = g.build(tree, registry.weights(card, variant, allow_download=False)[0]).cuda()
    a.note = "upstream GIMMVFI_R forward as video_Nx.py, time 0.5, ds_factor 1.0, replicate pad to 32, cupy softsplat"
    return lambda im1, im2: g.interpolate(model, im1[None].float() / 255, im2[None].float() / 255)


def _ema_vfi(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `MultiScaleFlow` as `tools/golden/ema_vfi.py` (`Model.inference` without TTA, timestep 0.5).

    `FF_EMA_VFI_REPO`: an MCG-NJU/EMA-VFI clone (needs timm). BGR in and out, replicate pad to 32 (centred).
    Upstream has no fp16 option (fp16 tier = autocast).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("ema_vfi")
    tree = g.reference_tree(Path(os.environ["FF_EMA_VFI_REPO"]), g.REF_COMMIT)
    card, variant = registry.resolve("interpolation", full)
    p = {**card.params, **variant.params}
    file = registry.weights(card, variant, allow_download=False)[0]
    net = g.build(tree, file, p["width"], list(p["depths"]), device="cuda")
    a.note = "upstream MultiScaleFlow as Model.inference (no TTA), timestep 0.5, BGR, replicate pad to 32"
    return lambda im1, im2: g.interpolate(net, im1[None].float() / 255, im2[None].float() / 255)


def _film(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """The TorchScript release as `tools/golden/film.py`; fp16 = inference.py's `--fp16` (`model.half()`).

    Zero pad to 64 (centred, as `util.pad_batch`) on the GPU, `dt` 0.5, the frame clamped to [0, 1] and cropped.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("interpolation", full)
    file = registry.weights(card, variant, allow_download=False)[0]
    dtype = torch.float16 if fp16 else torch.float32
    model = torch.jit.load(file, map_location="cpu").eval().to("cuda", dtype)
    a.note = (
        "the TorchScript release as inference.py, dt 0.5, zero pad to 64 (centred)" + ", --fp16: model.half()" * fp16
    )

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        h, w = im1.shape[-2:]
        ph, pw = -h % 64, -w % 64
        pad = [pw >> 1, pw - (pw >> 1), ph >> 1, ph - (ph >> 1)]
        x0, x1 = (F.pad(x[None].to(dtype) / 255, pad) for x in (im1, im2))
        with torch.no_grad():
            pred = model(x0, x1, x0.new_full((1, 1), 0.5)).clamp(0, 1)
        return pred[..., pad[2] : pad[2] + h, pad[0] : pad[0] + w].float()

    return call


def _realesrgan(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream `RealESRGANer` as `tools/golden/realesrgan.py`; fp16 = inference_realesrgan.py's default (`half`).

    `FF_REALESRGAN_REPO` / `FF_BASICSR_REPO`: xinntao/Real-ESRGAN and XPixelGroup/BasicSR clones;
    `FF_REALESRGAN_WEIGHTS`: the release `.pth` files. `pre_pad` 0, no tile; the timed call is `enhance`'s float
    part: the numpy image to the GPU (`pre_process`, x2: reflect padding), the network, `post_process`.
    """
    import torch  # noqa: PLC0415

    g = _golden("realesrgan")
    card, variant = full_ref("restoration", full)[1:]
    up = g.build(
        Path(os.environ["FF_REALESRGAN_REPO"]),
        Path(os.environ["FF_BASICSR_REPO"]),
        Path(os.environ["FF_REALESRGAN_WEIGHTS"]) / variant.weights[0].name,
        {**card.params, **variant.params},
        half=fp16,
        device=torch.device("cuda"),
    )
    a.note = "upstream RealESRGANer pre_pad 0, no tile, on the float image" + (", half (the default)" * fp16)
    return lambda img: g.run(up, img)


def _compact(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream `RealESRGANer` on `SRVGGNetCompact` as `tools/golden/compact.py`; fp16 = upstream's default (`half`).

    `FF_COMPACT_REPO`: an xinntao/Real-ESRGAN clone; `FF_COMPACT_WEIGHTS`: the release `.pth` files. `pre_pad` 0, no
    tile, `-dn 1` (no denoise interpolation); timed as `_realesrgan`.
    """
    import torch  # noqa: PLC0415

    g = _golden("compact")
    card, variant = full_ref("restoration", full)[1:]
    up = g.build(
        Path(os.environ["FF_COMPACT_REPO"]),
        Path(os.environ["FF_COMPACT_WEIGHTS"]) / variant.weights[0].name,
        {**card.params, **variant.params},
        half=fp16,
        device=torch.device("cuda"),
    )
    a.note = "upstream RealESRGANer pre_pad 0, no tile, -dn 1, on the float image" + (", half (the default)" * fp16)
    return lambda img: g.run(up, img)


def _lama(a: Any, full: str) -> Any:  # noqa: ANN401, ARG001 - adapter signature (one variant)
    """Upstream `FFCResNetGenerator` called as `bin/predict.py` (`tools/golden/lama.py`), float32 as upstream.

    `FF_LAMA_REPO`: an advimman/lama clone; `FF_LAMA_WEIGHTS`: the unpacked big-lama.zip. The dataset step (/ 255,
    symmetric padding to 8) is untimed `prep`; timed: mask > 0, masked image + mask, generator, paste-back, crop.
    """
    import torch  # noqa: PLC0415

    g = _golden("lama")
    net, pad = g.build(
        Path(os.environ["FF_LAMA_REPO"]),
        Path(os.environ["FF_LAMA_WEIGHTS"]) / "models" / "best.ckpt",
        torch.device("cuda"),
    )
    a.note = f"advimman/lama@{g.REF_COMMIT[:12]} FFCResNetGenerator, predict.py padding / masking / paste-back"

    def prep(args: tuple[Any, ...]) -> tuple[Any, ...]:
        img, m = args[0], args[1][0, 0]
        x, mk = g.pad_inputs(pad, img.cpu().numpy(), (m * 255).byte().cpu().numpy())
        return torch.from_numpy(x)[None].cuda(), torch.from_numpy(mk)[None].cuda(), tuple(img.shape[-2:])

    a._prep = prep  # noqa: SLF001 - the pad function comes with the build
    return lambda x, mk, size: g.run(net, x, mk, size)


def _realplksr(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Neosr realplksr in eval mode as `tools/golden/realplksr.py` (`FF_REALPLKSR_REPO`, `FF_REALPLKSR_WEIGHTS`).

    As neosr's test: `net(lq)` in float32, clamped.
    """
    import torch  # noqa: PLC0415

    g = _golden("realplksr")
    card, variant = full_ref("restoration", full)[1:]
    weights = Path(os.environ["FF_REALPLKSR_WEIGHTS"]) / variant.weights[0].name
    net = g.build(
        Path(os.environ["FF_REALPLKSR_REPO"]), weights, {**card.params, **variant.params}, device=torch.device("cuda")
    )
    a.note = "neosr realplksr eval mode, float32, net(lq) clamped"
    return lambda x: g.run(net, x)


def _span(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `SPAN` in eval mode as `tools/golden/span.py` (`FF_SPAN_REPO`, `FF_SPAN_WEIGHTS`), float32.

    As BasicSR's test: `net(lq)`, clamped (its `Conv3XC` re-fuses the weights on every call, as upstream).
    """
    import torch  # noqa: PLC0415

    g = _golden("span")
    card, variant = full_ref("restoration", full)[1:]
    weights = Path(os.environ["FF_SPAN_WEIGHTS"]) / variant.weights[0].name
    net = g.build(
        Path(os.environ["FF_SPAN_REPO"]), weights, {**card.params, **variant.params}, device=torch.device("cuda")
    )
    a.note = "upstream SPAN eval mode, float32, net(lq) clamped"
    return lambda x: g.run(net, x)


def _swinir(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `SwinIR` as `tools/golden/swinir.py` (`FF_SWINIR_REPO`, `FF_SWINIR_WEIGHTS`), float32.

    As `main_test_swinir.py`: mirror padding, `model(lq)` (no tile), crop, clamp.
    """
    import torch  # noqa: PLC0415

    g = _golden("swinir")
    card, variant = full_ref("restoration", full)[1:]
    p = {**card.params, **variant.params}
    weights = Path(os.environ["FF_SWINIR_WEIGHTS"]) / variant.weights[0].name
    net = g.build(Path(os.environ["FF_SWINIR_REPO"]), weights, p, device=torch.device("cuda"))
    a.note = "upstream main_test_swinir.py, float32, no tile"
    return lambda x: g.run(net, x, p["scale"])


def _hat(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `HAT` as `tools/golden/hat.py` (`FF_HAT_REPO`, `FF_HAT_WEIGHTS`; needs `einops`), float32.

    As `hat_model.py` test: reflect padding to a multiple of 16, `net_g(img)` (no tile), crop, clamp.
    """
    import torch  # noqa: PLC0415

    g = _golden("hat")
    card, variant = full_ref("restoration", full)[1:]
    p = {**card.params, **variant.params}
    weights = Path(os.environ["FF_HAT_WEIGHTS"]) / variant.weights[0].name
    net = g.build(Path(os.environ["FF_HAT_REPO"]), weights, p, device=torch.device("cuda"))
    a.note = "upstream hat_model.py test, float32, no tile"
    return lambda x: g.run(net, x, p["scale"])


def _drct(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `DRCT` as `tools/golden/drct.py` (`FF_DRCT_REPO`, `FF_DRCT_WEIGHTS`), float32.

    As `drct_model.py` test: reflect padding to a multiple of 16, `net_g_ema(img)` (no tile), crop, clamp.
    """
    import torch  # noqa: PLC0415

    g = _golden("drct")
    card, variant = full_ref("restoration", full)[1:]
    p = {**card.params, **variant.params}
    weights = Path(os.environ["FF_DRCT_WEIGHTS"]) / variant.weights[0].name
    net = g.build(Path(os.environ["FF_DRCT_REPO"]), weights, p, device=torch.device("cuda"))
    a.note = "upstream drct_model.py test, float32, no tile"
    return lambda x: g.run(net, x, p["scale"])


def _dat(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `DAT` as `tools/golden/dat.py` (`FF_DAT_REPO`, `FF_DAT_WEIGHTS`; needs `einops`), float32.

    As `dat_model.py` test (`use_chop` False): `net_g(img)`, clamp.
    """
    import torch  # noqa: PLC0415

    g = _golden("dat")
    card, variant = full_ref("restoration", full)[1:]
    p = {**card.params, **variant.params}
    weights = Path(os.environ["FF_DAT_WEIGHTS"]) / variant.weights[0].name
    net = g.build(Path(os.environ["FF_DAT_REPO"]), weights, p, device=torch.device("cuda"))
    a.note = "upstream dat_model.py test, float32, no chop"
    return lambda x: g.run(net, x, p["scale"])


def _codeformer(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `CodeFormer` as `tools/golden/codeformer.py` (`FF_CODEFORMER_REPO`, `FF_CODEFORMER_WEIGHTS`), float32.

    As `inference_codeformer.py` on one aligned 512 x 512 face: [-1, 1], `net(x, w, adain=True)[0]`, back to [0, 1].
    """
    g = _golden("codeformer")
    card, variant = full_ref("restoration", full)[1:]
    p = {**card.params, **card.options, **variant.params}
    tree = g.reference_tree(Path(os.environ["FF_CODEFORMER_REPO"]), g.REF_COMMIT)
    net = g.build(tree, Path(os.environ["FF_CODEFORMER_WEIGHTS"]) / variant.weights[0].name, p).cuda()
    a.note = f"upstream inference_codeformer.py on the aligned face, float32, w {p['fidelity']}"
    return lambda x: g.run(net, x, p["fidelity"])


def _gfpgan(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """Upstream `GFPGANv1Clean` as `tools/golden/gfpgan.py` (`FF_GFPGAN_REPO`, `FF_GFPGAN_WEIGHTS`), float32.

    As `GFPGANer.enhance` on one aligned 512 x 512 face: [-1, 1], `net(x, return_rgb=False)[0]` (random noise per
    call, upstream's default), back to [0, 1].
    """
    g = _golden("gfpgan")
    card, variant = full_ref("restoration", full)[1:]
    p = {**card.params, **variant.params}
    tree = g.reference_tree(Path(os.environ["FF_GFPGAN_REPO"]), g.REF_COMMIT)
    net = g.build(tree, Path(os.environ["FF_GFPGAN_WEIGHTS"]) / variant.weights[0].name, p).cuda()
    a.note = "upstream GFPGANer.enhance on the aligned face, float32, random noise per call"
    return lambda x: g.run(net, x, randomize_noise=True)


def _bchw01(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """uint8 CUDA CHW -> float32 `[1, 3, H, W]` in [0, 1] (BasicSR's lq)."""
    return ((args[0].float() / 255)[None],)


def _hwc01(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """uint8 CUDA CHW -> float32 numpy HWC in [0, 1] (`enhance` divides by 255 before `pre_process`)."""
    return ((args[0].permute(1, 2, 0).float() / 255).cpu().numpy(),)


def _neuflow_v2(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001
    """Upstream `NeuFlow` as `tools/golden/neuflow_v2.py` (`FF_NEUFLOW_V2_REPO`); fp16 = upstream infer.py's recipe.

    infer.py folds the batch norms of the conv blocks into their convs, calls `model.half()` and feeds half images;
    the fp32 tier is the plain model. The call replicate-pads the pair to multiples of 16 (centred) and crops the flow.
    """
    import torch.nn.functional as F  # noqa: PLC0415
    from torch.nn.utils.fusion import fuse_conv_bn_eval  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("neuflow_v2")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_NEUFLOW_V2_REPO")
        if not repo:
            msg = "FF_NEUFLOW_V2_REPO: a clone of neufieldrobotics/NeuFlow_v2 for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
    card, variant = registry.resolve("flow", full)
    net = g.build(a.tree, registry.weights(card, variant, allow_download=False)[0]).cuda()
    if fp16:
        for m in net.modules():
            if type(m).__name__ == "ConvBlock" and hasattr(m, "norm1"):
                m.conv1, m.conv2 = fuse_conv_bn_eval(m.conv1, m.norm1), fuse_conv_bn_eval(m.conv2, m.norm2)
                m.forward = m.forward_fuse
        net.half()
    a.note = f"upstream NeuFlow, iters_s8={g.ITERS}" + (", infer.py recipe: conv-bn folded, model.half()" * fp16)

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        h, w = im1.shape[-2:]
        ph, pw = -h % 16, -w % 16
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x1, x2 = (F.pad(im[None].half() if fp16 else im[None].float(), pad, mode="replicate") for im in (im1, im2))
        return g.calc_flow(net, x1, x2, amp=fp16)[..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]

    return call


def _megaflow(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001
    """Upstream `MegaFlow` as `tools/golden/megaflow.py` (`FF_MEGAFLOW_REPO`): the pair as a two-frame clip in [0, 255].

    fp16 = the authors' recipe (README, demo_flow.py): `torch.autocast("cuda", torch.bfloat16)`; the fp32 tier is the
    plain model. Upstream pads, resizes and unpads inside its forward. The reference environment supplies
    huggingface_hub (upstream imports it).
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("megaflow")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_MEGAFLOW_REPO")
        if not repo:
            msg = "FF_MEGAFLOW_REPO: a clone of cvg/megaflow for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
    card, variant = registry.resolve("flow", full)
    net = g.build(a.tree, registry.weights(card, variant, allow_download=False)[0]).cuda()
    a.note = f"upstream MegaFlow, num_reg_refine={g.ITERS}" + (", README recipe: bf16 autocast" * fp16)

    def call(im1: Any, im2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW
        with torch.autocast("cuda", torch.bfloat16, enabled=fp16):
            return net(torch.stack([im1, im2]).float()[None], num_reg_refine=g.ITERS)["flow_preds"][-1][:, 0]

    return call


def _recover(a: Any, full: str) -> Any:  # noqa: ANN401
    """Upstream `ReCoVEr_{CX,RN,MN}` as `tools/golden/recover.py` (`FF_RECOVER_REPO`): uint8 pair as float [0, 255].

    Upstream scales, pads and unpads inside its forward (its demo.py `calc_flow` at scale 0); no fp16 option.
    """
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.checkpoint import load_state_dict  # noqa: PLC0415

    g = _golden("recover")
    if not hasattr(a, "tree"):
        repo = os.environ.get("FF_RECOVER_REPO")
        if not repo:
            msg = "FF_RECOVER_REPO: a clone of visinf/ReCoVEr for the original-code tiers"
            raise RuntimeError(msg)
        a.tree = g.reference_tree(Path(repo), g.REF_COMMIT)
        sys.path[:0] = [str(a.tree)]
        g._no_imagenet()  # noqa: SLF001 - the golden's own switch: no ImageNet download at construction
    from models import recover  # noqa: PLC0415  # ty: ignore[unresolved-import]

    card, variant = registry.resolve("flow", full)
    arch = variant.params.get("arch", card.params["arch"])
    net = getattr(recover, g.CLASSES[arch])(pretrained=False)
    net.load_state_dict(load_state_dict(registry.weights(card, variant, allow_download=False)[0]), strict=True)
    net = net.cuda().eval()
    a.note = f"upstream {g.CLASSES[arch]}, iters={g.ITERS}, test_mode"
    return lambda im1, im2: net(im1[None].float(), im2[None].float(), iters=g.ITERS, test_mode=True)["final"]


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


def _topdown(a: Any, full: str) -> Any:  # noqa: ANN401
    """`inference_topdown` of mmpose (flip test off) on the `tools/golden/<family>.py` model (`FF_MMPOSE_SITE`).

    The test pipeline (CPU cv2 crop, normalisation) and the decode (SimCC / heatmap) are inside the timed call.
    """
    from fastforks._core import registry  # noqa: PLC0415

    site = os.environ.get("FF_MMPOSE_SITE")
    if not site:
        msg = "FF_MMPOSE_SITE: site-packages with mmpose 1.3.2 (+ mmdet, mmcv, mmengine) for the original-code tiers"
        raise RuntimeError(msg)
    if site not in sys.path:
        sys.path.append(site)
    card, variant = registry.resolve("pose", full)
    net = _golden(card.family).reference(site, card, variant, flip=False).cuda()
    from mmpose.apis import inference_topdown  # noqa: PLC0415  # ty: ignore[unresolved-import]

    decode = "heatmap" if card.family == "hrnet" else "SimCC"
    a.note = (
        f"mmpose 1.3.2 inference_topdown (CPU cv2 crop, network, {decode} decode; flip_test off), all in the timed call"
    )
    return lambda bgr, boxes: inference_topdown(net, bgr, boxes, bbox_format="xyxy")


class _PoseNet:
    """The pose network alone: `prep` runs the original's own crop / normalisation (untimed), the call the network."""

    def __init__(self, prep: Any, net: Any) -> None:  # noqa: ANN401
        self.prep, self.net = prep, net

    def __call__(self, *x: Any) -> Any:  # noqa: ANN401
        return self.net(*x)


def _vitpose_net(a: Any, full: str) -> Any:  # noqa: ANN401
    """`VitPoseForPoseEstimation` on the processor's crops of all boxes as one batch (crop and decode untimed)."""
    import torch  # noqa: PLC0415

    g = _golden("vitpose")
    repo, rev = g.REFS[full.split("@", maxsplit=1)[0]]
    proc = g.VitPoseImageProcessor.from_pretrained(repo, revision=rev)
    net = g.VitPoseForPoseEstimation.from_pretrained(repo, revision=rev).cuda().eval()
    experts = net.config.backbone_config.num_experts
    a.note = f"network only: transformers {g.REF_VERSION} VitPoseForPoseEstimation on the processor's crops, one batch"

    def prep(args: tuple[Any, ...]) -> tuple[Any, ...]:
        img, boxes = args
        x = proc(img, boxes=[boxes], return_tensors="pt")["pixel_values"].cuda()
        return (x, torch.zeros(len(boxes), dtype=torch.long, device="cuda") if experts > 1 else None)

    return _PoseNet(prep, lambda x, idx: net(x, **({"dataset_index": idx} if idx is not None else {})).heatmaps)


def _rtmpose_net(a: Any, full: str) -> Any:  # noqa: ANN401
    """The mmpose model's `_forward` (backbone + SimCC head) on `inference_topdown`'s own batch (pipeline untimed)."""
    import numpy as np  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("rtmpose")
    card, variant = registry.resolve("pose", full)
    net = g.reference(os.environ["FF_MMPOSE_SITE"], card, variant, flip=False).cuda()
    from mmengine.dataset import Compose, pseudo_collate  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from mmengine.registry import init_default_scope  # noqa: PLC0415  # ty: ignore[unresolved-import]

    init_default_scope(net.cfg.get("default_scope", "mmpose"))
    pipeline = Compose(net.cfg.test_dataloader.dataset.pipeline)
    a.note = (
        "network only: mmpose 1.3.2 backbone + SimCC head on inference_topdown's batch (pipeline and decode untimed)"
    )

    def prep(args: tuple[Any, ...]) -> tuple[Any, ...]:
        bgr, boxes = args
        items = [
            pipeline({"img": bgr, "bbox": b[None], "bbox_score": np.ones(1, np.float32), **net.dataset_meta})
            for b in boxes
        ]
        return (net.data_preprocessor(pseudo_collate(items), False)["inputs"],)  # noqa: FBT003

    return _PoseNet(prep, net._forward)  # noqa: SLF001


NETWORK = {"vitpose": _vitpose_net, "rtmpose": _rtmpose_net}  # family -> network-only builder (`load(part=...)`)


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


def _retinaface(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """The facexlib `detect_faces` as `tools/golden/retinaface.py` (`FF_FACEXLIB_REPO`, `FF_RETINAFACE_WEIGHTS`).

    float32 on CUDA, the image at its own size (`use_origin_size`), card thresholds; BGR uint8 in, NMS and decoding in
    the timed call.
    """
    g = _golden("retinaface")
    card, variant = full_ref("pose", full)[1:]
    p = {**card.params, **card.options, **variant.params}
    tree = g.reference_tree(Path(os.environ["FF_FACEXLIB_REPO"]), g.REF_COMMIT)
    weights = Path(os.environ["FF_RETINAFACE_WEIGHTS"]) / variant.weights[0].name
    net = g.build(tree, weights, p["backbone"], "cuda").cuda()
    a.note = "facexlib 0.3.0 detect_faces (use_origin_size, float32): preprocessing, network, NMS, decode"
    return lambda bgr: g.run(net, bgr, p["score_threshold"], p["nms_iou"])


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


def _yolox(a: Any, full: str) -> Any:  # noqa: ANN401
    """The reference demo path of `tools/golden/yolox.py` (`ValTransform`, network, `postprocess`) on CUDA.

    Needs `YOLOX_SRC`, a Megvii-BaseDetection/YOLOX checkout at the golden's commit (with loguru, pycocotools and
    tabulate importable). CPU cv2 resize and pad, network and NMS are in the timed call; the score threshold is the
    card's default (0.05), as the library's rows; the reference keeps every box after NMS.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("yolox")
    card, variant = registry.resolve("detection", full)
    exp, model = g.reference(os.environ["YOLOX_SRC"], card, variant)
    model = model.cuda()
    conf = card.options["score_threshold"]
    a.note = (
        f"Megvii-BaseDetection/YOLOX@{g.COMMIT[:12]} ValTransform (CPU cv2) + model + postprocess (NMS "
        f"{exp.nmsthre}), score {conf} (card default), all in the timed call"
    )
    return lambda bgr: g.detect(exp, model, bgr, conf)


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


def _bisenet(a: Any, full: str) -> Any:  # noqa: ANN401
    """face-parsing.PyTorch's test.py on facexlib's network (`FF_BISENET_REPO`: a facexlib clone), float32.

    The timed call is test.py's: the uint8 image to PIL, `resize((512, 512), BILINEAR)`, `ToTensor` + `Normalize`, to
    the GPU, `net(x)[0]` (512 x 512 logits; test.py does not resize them back).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("bisenet")
    card, variant = registry.resolve("segmentation", full)
    (weights,) = registry.weights(card, variant)
    net = g.build(Path(os.environ["FF_BISENET_REPO"]), weights).cuda()
    mean, std = (torch.tensor(v, device="cuda")[:, None, None] for v in (g.MEAN, g.STD))
    a.note = "face-parsing.PyTorch test.py: PIL resize 512 x 512 on the CPU, Normalize, net(x)[0], float32"

    def call(x: Any) -> Any:  # noqa: ANN401
        im = Image.fromarray(x.permute(1, 2, 0).cpu().numpy()).resize((g.SIZE, g.SIZE), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(im)).cuda().permute(2, 0, 1).float() / 255
        return net(((t - mean) / std)[None])[0]

    return call


def _convnext(a: Any, full: str) -> Any:  # noqa: ANN401
    """A classifier as `tools/golden/libnet_classification.py` builds it, with the card's `[io]` transform on GPU.

    The bench's classification adapter for a family it does not list: the golden's `prepare` (pad to square,
    antialiased resize, clamp after bicubic, centre crop, normalisation) and network, all in the timed call.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("libnet_classification")
    card, variant = registry.resolve("classification", full)
    io = variant.io
    side = io.crop or io.resize
    a.sizes = ((side, side),)  # Benchmark each classifier at its native input, including 384/512 variants.
    net = g.reference(card, variant)[0].eval()
    if card.family == "fastvit":
        from timm.utils import reparameterize_model  # noqa: PLC0415 - authors' inference recipe

        net = reparameterize_model(net, inplace=True)
        a.note = "timm FastViT with authors' reparameterized inference recipe; fusion is outside the timed call"
    net = net.cuda()
    a.COMMIT = f"{card.upstream}@{card.upstream_commit}"  # as the bench's classification rows
    return lambda x: net(g.prepare(x[None].float() / 255, io))[0]


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


def _romav2(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream `RoMaV2.match` as `tools/golden/romav2.py` (card sizes, one direction), outputs as the API returns them.

    fp16: the authors' `enable_amp` (bfloat16 autocast); fp32 replaces `torch.autocast` by a no-op for the whole
    process (one tier per process). Needs `FF_ROMAV2_REPO` / `FF_DINOV3_REPO` checkouts with the pinned commits.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("romav2")
    card, variant = registry.resolve("matching", full)
    tree = g.reference_tree(Path(os.environ["FF_ROMAV2_REPO"]), g.REF_COMMIT)
    dino = g.reference_tree(Path(os.environ["FF_DINOV3_REPO"]), g.DINOV3_COMMIT)
    model = g.build(tree, dino, registry.weights(card, variant, allow_download=False)[0], amp=fp16).cuda()
    g.configure(model, card.options["coarse_size"], card.options["fine_size"])
    a.note = "upstream RoMaV2.match; local correlation: upstream's plain-torch path (no fused-local-corr)"

    def call(img1: Any, img2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW images
        preds = model.match(img1[None].float() / 255, img2[None].float() / 255)
        return g.to_api(preds, tuple(img1.shape[-2:]), tuple(img2.shape[-2:]))

    return call


def _romav1(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream RoMa v1 `match` as `tools/golden/romav1.py` (card sizes, one direction), outputs as the API returns.

    fp16: upstream's default `amp_dtype=torch.float16` (fp16 autocast regions, DINOv2 in fp16); fp32: `amp_dtype`
    float32. Local correlation: upstream's plain-torch path. Needs `FF_ROMAV1_REPO` (pinned commit) and einops.
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("romav1")
    card, variant = registry.resolve("matching", full)
    tree = g.reference_tree(Path(os.environ["FF_ROMAV1_REPO"]), g.REF_COMMIT)
    sizes = (card.options["coarse_size"], card.options["fine_size"])
    model = g.build(tree, list(registry.weights(card, variant, allow_download=False)), sizes, "cuda", amp=fp16)
    a.note = (
        "upstream RoMa v1 match, one direction; local correlation: upstream's plain-torch path (use_custom_corr off)"
    )

    def call(img1: Any, img2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW images
        return g.match(model, img1[None].float() / 255, img2[None].float() / 255, *sizes)

    return call


def _xfeat(a: Any, full: str, options: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    """Upstream `XFeat.detectAndCompute` + `match` / kornia `LighterGlue` as `tools/golden/xfeat.py` (card `options`).

    `options` override the card's. Upstream has no fp16 option (fp16 = autocast around the call). Needs a
    `FF_XFEAT_REPO` checkout with the pinned commit and kornia. The call returns keypoints1, keypoints2, pairs, scores.
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("xfeat")
    card, variant = registry.resolve("matching", full)
    sys.path.insert(0, str(g.reference_tree(Path(os.environ["FF_XFEAT_REPO"]), g.REF_COMMIT)))
    files = {f.name: f for f in registry.weights(card, variant, allow_download=False)}
    opts = dict(card.options) | (options or {})
    if not torch.cuda.is_available():
        msg = "the XFeat original runs on CUDA (upstream picks its device itself)"
        raise RuntimeError(msg)
    run = g.build(files, variant.params["matcher"], opts["max_keypoints"], opts["long_side"])
    a.note = f"upstream XFeat.detectAndCompute + {variant.params['matcher']}, top_k {opts['max_keypoints']}"

    def call(img1: Any, img2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW images
        return run(img1.float() / 255, img2.float() / 255)

    return call


def _lightglue(a: Any, full: str, fp16: bool, options: dict[str, Any] | None = None) -> Any:  # noqa: ANN401, FBT001
    """Upstream `match_pair` with the extractor and `LightGlue` as `tools/golden/lightglue.py` (card `options`).

    `options` override the card's. fp16: the authors' `mp` option (LightGlue's mixed precision; the extractor stays
    float32). Upstream's attention runs in float16 on CUDA either way. Needs a `FF_LIGHTGLUE_REPO` checkout with the
    pinned commit and kornia.
    """
    import functools  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("lightglue")
    card, variant = registry.resolve("matching", full)
    sys.path.insert(0, str(g.reference_tree(Path(os.environ["FF_LIGHTGLUE_REPO"]), g.REF_COMMIT)))
    from lightglue import ALIKED, DISK, LightGlue  # noqa: PLC0415  # reference checkout  # ty: ignore[unresolved-import]
    from lightglue.utils import match_pair  # noqa: PLC0415  # ty: ignore[unresolved-import]

    files = {f.name: f for f in registry.weights(card, variant, allow_download=False)}
    torch.hub.load_state_dict_from_url = functools.partial(g.from_files, files)
    params, opts = dict(variant.params), dict(card.options) | (options or {})
    extractor = {"aliked": ALIKED, "disk": DISK}[params["extractor"]]
    ex = extractor(max_num_keypoints=opts["max_keypoints"]).eval().cuda()
    lg = LightGlue(features=params["extractor"], mp=fp16).eval().cuda()
    a.note = f"upstream match_pair {params['extractor']} + LightGlue, resize {opts['long_side']}"

    def call(img1: Any, img2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW images
        return match_pair(ex, lg, img1.float() / 255, img2.float() / 255, device="cuda", resize=opts["long_side"])

    return call


def _dedode(a: Any, full: str, fp16: bool, options: dict[str, Any] | None = None) -> Any:  # noqa: ANN401, FBT001
    """Upstream DeDoDe v2 detector + descriptor + `DualSoftMaxMatcher` as `tools/golden/dedode.py` (card `options`).

    fp16: upstream's default on CUDA (float16 autocast, float16 DINOv2 for -G); off: float32. Needs a `FF_DEDODE_REPO`
    checkout with the pinned commit. The call returns keypoints1, keypoints2, pairs, scores.
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("dedode")
    card, variant = registry.resolve("matching", full)
    sys.path.insert(0, str(g.reference_tree(Path(os.environ["FF_DEDODE_REPO"]), g.REF_COMMIT)))
    files = registry.weights(card, variant, allow_download=False)
    params = dict(card.params) | dict(variant.params) | dict(card.options) | (options or {})
    run = g.build(files, params, amp=fp16, device=torch.device("cuda"))
    a.note = f"upstream DeDoDe-L v2 + descriptor {params['descriptor']}, {params['max_keypoints']} keypoints"

    def call(img1: Any, img2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW images
        return run(img1.float() / 255, img2.float() / 255)

    return call


def _loftr(a: Any, full: str, options: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    """Upstream `LoFTR` with the variant's settings as `tools/golden/loftr.py` (card `options`), float32 on CUDA.

    Upstream has no inference precision option. Needs a `FF_LOFTR_REPO` checkout with the pinned commit (einops,
    kornia < 0.8, yacs). The call returns keypoints1, keypoints2, pairs, scores.
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("loftr")
    card, variant = registry.resolve("matching", full)
    sys.path.insert(0, str(g.reference_tree(Path(os.environ["FF_LOFTR_REPO"]), g.REF_COMMIT)))
    (ckpt,) = registry.weights(card, variant, allow_download=False)
    params = dict(card.params) | dict(variant.params)
    long_side = int((dict(card.options) | (options or {}))["long_side"])
    net = g.build(ckpt, params, params["threshold"]).cuda()
    a.note = f"upstream LoFTR default_cfg, temp_bug_fix {params['temp_bug_fix']}, long side {long_side}"

    @torch.inference_mode()
    def call(img1: Any, img2: Any) -> Any:  # noqa: ANN401 - uint8 CUDA CHW images
        (g0, s0), (g1, s1) = (g.prepare(i.float() / 255, long_side) for i in (img1, img2))
        batch = {"image0": g0, "image1": g1, "scale0": s0, "scale1": s1}
        net(batch)
        idx = torch.arange(len(batch["mconf"]), device=g0.device)
        return {
            "keypoints1": batch["mkpts0_f"],
            "keypoints2": batch["mkpts1_f"],
            "pairs": torch.stack([idx, idx], -1),
            "scores": batch["mconf"],
        }

    return call


def _dfine(a: Any, full: str) -> Any:  # noqa: ANN401
    """D-FINE, DEIM, LW-DETR, OWLv2, Grounding DINO as the COCO eval runs them (`tools/_original_coco.py`), RGB in."""
    import _original_coco  # noqa: PLC0415

    ref = full.split("@", maxsplit=1)[0]
    det = _original_coco.load("detection", ref, "upstream")
    if "grounding-dino" in ref:  # also mm-grounding-dino
        a.COMMIT = "transformers-" + importlib.import_module("transformers").__version__
        arch = "MMGroundingDino" if ref.startswith("mm-") else "GroundingDino"
        a.note = f"transformers {arch}ForObjectDetection + processor (CPU), coco-80 caption, text tower per call"
    elif ref.startswith("owlv2"):
        a.COMMIT = "transformers-" + importlib.import_module("transformers").__version__
        a.note = (
            "transformers Owlv2ForObjectDetection + processor (CPU), coco-80 queries, text tower per call, 300 best"
        )
    elif ref.startswith("lw-detr"):
        a.COMMIT = "transformers-" + importlib.import_module("transformers").__version__
        a.note = "transformers LwDetrForObjectDetection + image processor (CPU), the authors' PostProcess (all kept)"
    elif ref.startswith("deim"):
        a.COMMIT = _original_coco.DEIM_COMMIT
        a.note = "official DEIM deploy model; incl. CPU PIL resize + host-to-device copy, RT-DETR postprocess (all 300)"
    else:
        a.COMMIT = "transformers-" + importlib.import_module("transformers").__version__
        a.note = "transformers DFineForObjectDetection; incl. CPU PIL resize + host-to-device copy, all 300"
    return det


def _clipseg(a: Any, full: str) -> Any:  # noqa: ANN401
    """CLIPSeg as the model card's usage: transformers CLIPSegProcessor (CPU) + CLIPSegForImageSegmentation.

    The text tower runs per call; the logits are bilinearly resized to the image and thresholded at 0 (probability
    0.5). HWC uint8 RGB in.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("promptable_segmentation", full)
    folder = registry.weights(card, variant, allow_download=False)[1].resolve().parent  # the HF snapshot folder
    proc = transformers.CLIPSegProcessor.from_pretrained(folder)
    cls = transformers.CLIPSegForImageSegmentation
    net = cls.from_pretrained(folder, dtype=torch.float32).cuda().eval()  # ty: ignore[missing-argument] - transformers' from_pretrained decorator is inferred as an unbound method
    a.COMMIT = "transformers-" + transformers.__version__
    a.note = (
        "transformers CLIPSegProcessor (CPU) + CLIPSegForImageSegmentation, text tower per call, masks at logits > 0"
    )

    @torch.inference_mode()
    def call(image: Any, text: str) -> Any:  # noqa: ANN401 - uint8 HWC array
        inputs = proc(images=image, text=[text], return_tensors="pt", padding=True).to("cuda")
        logits = net(**inputs).logits.reshape(1, 1, 352, 352)
        return F.interpolate(logits.float(), image.shape[:2], mode="bilinear")[0, 0] > 0

    return call


def _hwc_text(a: tuple[Any, ...]) -> tuple[Any, ...]:
    """(uint8 CUDA CHW, text) of the bench call -> (uint8 HWC array, text)."""
    return a[0].permute(1, 2, 0).cpu().numpy(), a[1]


def _hwc_point(a: tuple[Any, ...]) -> tuple[Any, ...]:
    """(uint8 CUDA CHW, points, labels) of the bench call -> (uint8 HWC array, [1, 2] float array, [1] int array)."""
    import numpy as np  # noqa: PLC0415

    return a[0].permute(1, 2, 0).cpu().numpy(), np.array(a[1], np.float64), np.array(a[2])


def _lotus(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream `LotusGPipeline` / `LotusDPipeline` as `tools/golden/lotus.py` (one step, seed 42, camera frame).

    `FF_LOTUS_REPO`: a clone of EnVision-Research/Lotus; `FF_LOTUS_SNAPSHOT_G` / `FF_LOTUS_SNAPSHOT_D`: the weights
    snapshot directories. fp16 = the diffusers `torch_dtype=float16` load under CUDA autocast (upstream's
    `--half_precision`; infer.py always runs under autocast, the fp32 tier here does not).
    """
    import torch  # noqa: PLC0415

    g = _golden("lotus")
    if not hasattr(a, "tree"):
        a.tree = g.reference_tree(Path(os.environ["FF_LOTUS_REPO"]), g.REF_COMMIT)
    kind = "G" if full.startswith("lotus-normal-g") else "D"
    pipe = g.pipeline(a.tree, os.environ[f"FF_LOTUS_SNAPSHOT_{kind}"], full, torch.float16 if fp16 else torch.float32)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    a.note = f"upstream Lotus-{kind} pipeline, diffusers {g.DIFFUSERS}, timestep {g.TIMESTEP}, seed {g.SEED}"
    if not fp16:
        return lambda image: g.predict(pipe, image[None].float() / 255)

    def run(image: Any) -> Any:  # noqa: ANN401 - torch tensors
        with torch.autocast("cuda"):
            return g.predict(pipe, image[None].float() / 255)

    return run


def _marigold_normals(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, ARG001, FBT001 - adapter signature
    """The diffusers `MarigoldNormalsPipeline` as `tools/golden/marigold_normals.py` (card defaults, CPU generator).

    `FF_MARIGOLD_NORMALS_SNAPSHOT`: the weights snapshot directory (configs + safetensors); fp16 = the diffusers
    `torch_dtype=float16` load (the authors' float16 option). Output: unit normals `[H, W, 3]`, camera frame.
    """
    import torch  # noqa: PLC0415

    g = _golden("marigold_normals")
    pipe = g.pipeline(os.environ["FF_MARIGOLD_NORMALS_SNAPSHOT"], torch.float16 if fp16 else torch.float32).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    a.note = f"diffusers {g.DIFFUSERS} MarigoldNormalsPipeline, {g.STEPS} LCM step, CPU generator seed {g.SEED}"
    return lambda image: g.predict(pipe, image[None].float() / 255)


def _stablenormal(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, ARG001, FBT001 - adapter signature
    """Upstream `YOSONormalsPipeline` as `tools/golden/stablenormal.py` (hubconf turbo, seeded `latents`, camera frame).

    `FF_STABLENORMAL_REPO`: a clone of Stable-X/StableNormal; `FF_STABLENORMAL_SNAPSHOT`: the yoso-normal-v0-3
    snapshot directory (configs, `controlnet/controlnetvae.py`, fp16 safetensors); fp16 = `torch_dtype=float16`.
    """
    import torch  # noqa: PLC0415

    g = _golden("stablenormal")
    if not hasattr(a, "tree"):
        a.tree = g.reference_tree(Path(os.environ["FF_STABLENORMAL_REPO"]), g.REF_COMMIT)
    dtype = torch.float16 if fp16 else torch.float32
    pipe = g.pipeline(a.tree, os.environ["FF_STABLENORMAL_SNAPSHOT"], dtype).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    a.note = f"upstream YOSONormalsPipeline (turbo, t_start 0), diffusers {g.DIFFUSERS}, latents seed {g.SEED}"
    return lambda image: g.predict(pipe, image[None].float() / 255)


def _birefnet(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """The HF repo's `birefnet.py` with the card's pre- and post-processing, as `tools/golden/birefnet.py`.

    `FF_BIREFNET_HF`: a directory with the HF snapshots as `<ref>/birefnet.py` (+ `BiRefNet_config.py`); fp16 = the
    model card's `half()` (fp16 weights and input).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("birefnet")
    ref = full.split("@", maxsplit=1)[0]
    card, variant = registry.resolve("matting", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_BIREFNET_HF"]) / ref, Path(weights)).cuda()
    if fp16:
        net = net.half()
    repo, rev = g.HF_REVISION[ref]
    a.note = f"HF {repo}@{rev[:12]} birefnet.py, the card's resize / normalisation / sigmoid / resize back"
    return lambda image: g.predict(net, image[None].float() / 255)


def _ben2(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """The authors' `BEN2.py` with float pre- and post-processing, as `tools/golden/ben2.py`.

    `FF_BEN2_REPO`: a PramaLLC/BEN2 checkout; fp16 = the forward's own CUDA fp16 autocast (fp32: without it).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("ben2")
    card, variant = registry.resolve("matting", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_BEN2_REPO"]), Path(weights)).cuda()
    a.note = f"PramaLLC/BEN2@{g.COMMIT[:12]} BEN2.py, bicubic resize / normalisation / bilinear back / min-max"
    return lambda image: g.predict(net, image[None].float() / 255, autocast=fp16)


def _rsu(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """U²-Net / IS-Net (xuebinqin) with float pre- and post-processing, as `tools/golden/{u2net,isnet}.py`, float32.

    `FF_U2NET_REPO` / `FF_DIS_REPO`: a U-2-Net / DIS checkout.
    """
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("matting", full)
    (weights,) = registry.weights(card, variant)
    g = _golden(card.family)
    p = {**card.params, **variant.params}
    if card.family == "u2net":
        net = g.network(Path(os.environ["FF_U2NET_REPO"]), Path(weights), p["arch"]).cuda()
    else:
        net = g.network(Path(os.environ["FF_DIS_REPO"]), Path(weights)).cuda()
    a.note = f"{g.UPSTREAM.removeprefix('https://github.com/')}@{g.COMMIT[:12]}, float pre- and post-processing"
    return lambda image: g.predict(net, image[None].float() / 255, p["size"])


def _edges(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """controlnet_aux detectors as `tools/golden/_edges.py`: OpenCV `resize_image` on the host.

    Network on CUDA, float post-processing (no 8-bit rounding) and `cv2.INTER_LINEAR` back to the image size, at the
    card's default resolution.

    `FF_CNAUX_REPO`: a huggingface/controlnet_aux checkout.
    """
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("edges", full)
    (weights,) = registry.weights(card, variant)
    g = _golden("_edges")
    build, predict = g.NETS[card.family]
    net = build(Path(os.environ["FF_CNAUX_REPO"]), Path(weights)).cuda()
    a.note = f"huggingface/controlnet_aux@{g.COMMIT[:12]} {card.family}, host OpenCV resize, float post-processing"

    def run(image: Any) -> Any:  # noqa: ANN401 - CUDA uint8 CHW
        img = image.permute(1, 2, 0).cpu().numpy()
        return predict(net, img, card.options["resolution"])

    return run


def _inspyrenet(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """transparent-background's `InSPyReNet_SwinB` with float pre- and post-processing, as `tools/golden/inspyrenet.py`.

    `FF_TBG_PKG`: the `transparent_background` package directory (1.3.4); float32 (`Remover` has no fp16 option).
    """
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("matting", full)
    (weights,) = registry.weights(card, variant)
    g = _golden("inspyrenet")
    size = {**card.params, **variant.params}["size"]
    net = g.network(Path(os.environ["FF_TBG_PKG"]), Path(weights), size).to("cuda")  # upstream `to` moves the kernel
    a.note = f"{g.UPSTREAM.removeprefix('https://github.com/')}@{g.COMMIT[:12]}, float pre- and post-processing"
    return lambda image: g.predict(net, image[None].float() / 255, size)


def _vitmatte(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """The transformers `VitMatteForImageMatting` with the processor's maths on tensors, as `tools/golden/vitmatte.py`.

    `FF_VITMATTE_HF`: a directory with the HF snapshots as `<ref>/` (config + weights); fp16 = `torch_dtype=float16`.
    """
    import transformers  # noqa: PLC0415

    g = _golden("vitmatte")
    net = g.network(Path(os.environ["FF_VITMATTE_HF"]) / full.split("@", maxsplit=1)[0], fp16=fp16).cuda()
    a.note = f"transformers {transformers.__version__} VitMatteForImageMatting, normalise / trimap / pad 32 / crop"
    return lambda image, trimap: g.predict(net, image[None].float() / 255, trimap)


def _megaloc(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """The authors' `megaloc_model.py` with the README's resize / normalisation, as `tools/golden/megaloc.py`.

    `FF_MEGALOC_REPO`: a gmberton/MegaLoc checkout; no precision option of its own (autocast / `half()` tiers).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("megaloc")
    card, variant = registry.resolve("retrieval", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_MEGALOC_REPO"]), Path(weights)).cuda()
    a.note = f"gmberton/MegaLoc@{g.COMMIT[:12]} megaloc_model.py, 322 x 322 bilinear antialiased resize, normalisation"
    return lambda image: g.predict(net, image[None].float() / 255)


def _boq(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """The authors' `hubconf.py` with the README's resize / normalisation, as `tools/golden/boq.py`.

    `FF_BOQ_REPO`: an amaralibey/Bag-of-Queries checkout, `FF_DINOV2_REPO`: facebookresearch/dinov2; no precision
    option of its own (autocast / `half()` tiers).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("boq")
    card, variant = registry.resolve("retrieval", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_BOQ_REPO"]), Path(weights)).cuda()
    a.note = (
        f"amaralibey/Bag-of-Queries@{g.COMMIT[:12]} hubconf.py, 322 x 322 bicubic antialiased resize, normalisation"
    )
    return lambda image: g.predict(net, image[None].float() / 255)


def _geocalib(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """The authors' `GeoCalib.calibrate` (resize, network, LM fit, camera of the input), as `tools/golden/geocalib.py`.

    `FF_GEOCALIB_REPO`: a cvg/GeoCalib checkout; no precision option of its own (autocast tier).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("geocalib")
    card, variant = registry.resolve("calibration", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_GEOCALIB_REPO"]), Path(weights)).cuda()
    a.note = f"cvg/GeoCalib@{g.COMMIT[:12]} GeoCalib.calibrate (pinhole), its kornia resize / crop / LM fit"
    return lambda image: g.predict(net, image[None].float() / 255)["intrinsics"]


def _anycalib(a: Any, full: str) -> Any:  # noqa: ANN401 - adapter signature
    """The authors' `AnyCalib.predict` (resize / crop, network, linear + GN fit), as `tools/golden/anycalib.py`.

    `FF_ANYCALIB_REPO`: a javrtg/AnyCalib checkout; no precision option of its own (autocast tier).
    """
    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("anycalib")
    card, variant = registry.resolve("calibration", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_ANYCALIB_REPO"]), Path(weights)).cuda()
    a.note = f"javrtg/AnyCalib@{g.COMMIT[:12]} AnyCalib.predict (pinhole), its resize / crop / linear + GN fit"
    return lambda image: g.predict(net, image[None].float() / 255)["intrinsics"]


class _Local:
    """An original the bench has no adapter for, on the library's own bench inputs as uint8 CUDA tensors.

    `sizes` are the inputs of the family's docs/data/speed.jsonl rows; `fastforks.<task>` `bench_inputs` makes the
    call's arguments (flow: the synthetic pair; pose: the image and one person box per 128 px of width).
    """

    flag = False  # no mixed-precision option of its own: fp16 = torch.autocast around the call
    trimap = "unused"  # matting: "required" adds the bench's synthetic trimap to the call
    cameras = "unused"  # multiview: "required" adds the bench rig's intrinsics, cam_to_world and depth range
    prompts: tuple[str, ...] = ()  # promptable segmentation: ("reference",) adds the bench's reference image and mask
    mask = "unused"  # restoration: "required" adds the bench's synthetic inpainting mask

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

    def build(self, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter signature
        """The family's builder; `COMMIT` = the card's upstream pin."""
        self.COMMIT = full_ref(self.task, full)[1].upstream_commit
        return self._build(self, full, fp16) if self.flag else self._build(self, full)

    def prep(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """The untimed conversion of the uint8 CUDA inputs to what the original's entry point takes."""
        return self._prep(args) if self._prep else args

    def inputs(self, full: str) -> list[tuple[str, str, tuple[Any, ...]]]:  # noqa: ARG002 - the adapter signature
        """(label, scene, args) per size: the images as uint8 CUDA CHW, then the call's other tensors (boxes)."""
        import torch  # noqa: PLC0415

        from fastforks._core.bench import BenchSize  # noqa: PLC0415

        api = importlib.import_module(f"fastforks.{self.task}._api")
        like = types.SimpleNamespace(
            device=torch.device("cuda"),
            needs_boxes=self.needs_boxes,
            prompts=self.prompts,
            _runner=types.SimpleNamespace(
                intrinsics=self.cameras,
                cam_to_world=self.cameras,
                depth_range=self.cameras,
                trimap=self.trimap,
                mask=self.mask,
            ),
        )
        out = []
        for h, w in self.sizes:
            args, kw = api.bench_inputs(like, BenchSize(h, w))
            images = tuple((x.squeeze(0) * 255).round().byte() for x in args)
            out.append((f"{h}x{w}", "synthetic (fastforks bench)", (*images, *kw.values())))
        return out


class _RFDETRKeypoint(_Local):
    """The authors' explicit inference dtype option, without a compiler, on the same CUDA uint8 inputs."""

    flag = True
    family = "rfdetr_keypoint"
    postprocess = "uncertainty-fused PostProcess"

    def __init__(self, task: str = "pose", sizes: tuple[tuple[int, int], ...] = ((1080, 1920),)) -> None:
        super().__init__(task, None, sizes, needs_boxes=False)

    def reference(self, g: Any, checkout: Path, full: str, checkpoint: Path) -> Any:  # noqa: ANN401 - upstream wrapper
        _ = full
        return g.reference(checkout, checkpoint, "cuda")

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
        g = _golden(self.family)
        card, variant = registry.resolve(self.task, full)
        (checkpoint,) = registry.weights(card, variant)
        model = self.reference(g, Path(checkout), full, checkpoint)
        self.COMMIT = card.upstream_commit
        self.note = f"pinned RF-DETR torchvision resize/normalize, original network and {self.postprocess}"
        if fp16:
            model.inference(compile=False, dtype=torch.float16, inplace=True)
            self.note += "; authors inference(compile=False, dtype=float16, inplace=True)"

        def call(image: Any, threshold: float = 0.5) -> Any:  # noqa: ANN401 - uint8 CUDA CHW image
            x = image.unsqueeze(0) if image.ndim == 3 else image  # noqa: PLR2004 - CHW input
            x = x.float() / 255 if x.dtype == torch.uint8 else x.float()
            return g.predict(model, x, threshold)

        return call


class _RFDETR(_RFDETRKeypoint):
    """RF-DETR detection / Seg: the golden's upstream class per variant id."""

    family = "rfdetr"
    postprocess = "PostProcess"

    def reference(self, g: Any, checkout: Path, full: str, checkpoint: Path) -> Any:  # noqa: ANN401 - upstream wrapper
        return g.reference(checkout, g.CLASSES[full.split("@", maxsplit=1)[0]], checkpoint, "cuda")


class _LocalText(_Local):
    """A text-prompted original: the bench call's text prompt follows the image."""

    prompts = ("text",)


class _LocalFp16(_Local):
    """An original with an fp16 option of its own: its builder takes `(adapter, ref, fp16)`."""

    flag = True


def _tapnext(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream `TAPNext` online (one frame per call, as the authors' online demo), the eval's resizing to 256 x 256.

    fp16: `torch.autocast("cuda", torch.float16)` as upstream's evaluation. Needs `TAPNET_SRC`, a tapnet checkout with
    the pinned commit. The call returns tracks `[1, T, Q, 2]` and the visible logits `[1, T, Q, 1]` (upstream's).
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    sys.path.insert(0, os.environ["TAPNET_SRC"])
    from tapnet.tapnext.tapnext_torch import TAPNext  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from tapnet.tapnext.tapnext_torch_utils import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        restore_model_from_jax_checkpoint,
    )

    card, variant = registry.resolve("tracking", full)
    file = registry.weights(card, variant, allow_download=False)[0]
    model = TAPNext(image_size=(256, 256))
    if variant.weights[0].name.endswith(".npz"):
        model = restore_model_from_jax_checkpoint(model, str(file))
    else:
        sd = torch.load(file, map_location="cpu", weights_only=True)["state_dict"]
        model.load_state_dict({k.removeprefix("tapnext."): v for k, v in sd.items()})
    model = model.eval().cuda()
    a.note = "upstream TAPNext, online one frame per call; frames resized to 256 x 256 (antialiased bilinear)"

    def call(video: Any, queries: Any) -> Any:  # noqa: ANN401 - uint8 CUDA [T, 3, H, W], float [Q, 3] (t, x, y)
        h, w = video.shape[-2:]
        scale = torch.tensor((256 / w, 256 / h), device=queries.device)
        qp = torch.cat([queries[:, :1], ((queries[:, 1:] + 0.5) * scale).flip(-1)], 1)[None]
        tracks, vis, state = [], [], None
        with torch.autocast("cuda", torch.float16, enabled=fp16):
            for t in range(video.shape[0]):
                x = F.interpolate(video[t : t + 1].float() / 255, (256, 256), mode="bilinear", antialias=True)
                x = (x * 2 - 1).permute(0, 2, 3, 1)[:, None]  # [1, 1, 256, 256, 3]
                kw = {"query_points": qp} if state is None else {"state": state}
                tr, _, vl, state = model(video=x, **kw)
                tracks.append(tr), vis.append(vl)
        return torch.cat(tracks, 1), torch.cat(vis, 1)

    return call


def _tapir(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream `TAPIR` offline (the whole video in one call, as the authors' offline demo), resized to 256 x 256.

    fp16: `torch.autocast("cuda", torch.float16)`. Needs `TAPNET_SRC`, a tapnet checkout with the pinned commit (with
    einshape and dm-tree). The call returns upstream's output dict (tracks `[1, Q, T, 2]`, occlusion and
    expected-distance logits).
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    sys.path.insert(0, os.environ["TAPNET_SRC"])
    from tapnet.torch.tapir_model import TAPIR  # noqa: PLC0415  # ty: ignore[unresolved-import]

    card, variant = registry.resolve("tracking", full)
    file = registry.weights(card, variant, allow_download=False)[0]
    boots = variant.id == "bootstapir"
    model = TAPIR(pyramid_level=1) if boots else TAPIR(pyramid_level=0, extra_convs=False)
    model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
    model = model.eval().cuda()
    a.note = f"upstream {'BootsTAPIR' if boots else 'TAPIR'}, offline (whole video); frames resized to 256 x 256"

    def call(video: Any, queries: Any) -> Any:  # noqa: ANN401 - uint8 CUDA [T, 3, H, W], float [Q, 3] (t, x, y)
        h, w = video.shape[-2:]
        scale = torch.tensor((256 / w, 256 / h), device=queries.device)
        qp = torch.cat([queries[:, :1], ((queries[:, 1:] + 0.5) * scale).flip(-1)], 1)[None]
        with torch.autocast("cuda", torch.float16, enabled=fp16):
            x = F.interpolate(video.float() / 255, (256, 256), mode="bilinear", antialias=True)
            return model((x * 2 - 1).permute(0, 2, 3, 1)[None], qp)

    return call


def _alltracker(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream AllTracker `Net.forward` (CUDA, unpatched) through `tools/golden/alltracker.py` `track`.

    384 x 512, 4 iters, forward + backward passes per query frame; fp16: `torch.autocast("cuda", torch.float16)`.
    Needs `ALLTRACKER_SRC`, an aharley/alltracker checkout with the pinned commit (with einops importable).
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("alltracker")
    card, variant = registry.resolve("tracking", full)
    file = registry.weights(card, variant, allow_download=False)[0]
    model = g.load(Path(os.environ["ALLTRACKER_SRC"]), Path(file), g.MODELS[f"{variant.id}@1"][1]).cuda()
    a.note = f"aharley/alltracker@{g.COMMIT[:12]} Net.forward, 384 x 512, {g.ITERS} iters"

    def call(video: Any, queries: Any) -> Any:  # noqa: ANN401 - uint8 CUDA [T, 3, H, W], float [Q, 3] (t, x, y)
        return g.track(model, video.float() / 255, queries, amp=torch.float16 if fp16 else None)

    return call


def _track_on(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream Track-On2 `Predictor` loop (CUDA, mmcv MSDA as its PyTorch path) via `tools/golden/track_on.py`.

    384 x 512, memory 72, 20 x 20 support grid; fp16: `torch.autocast("cuda", torch.float16)`. Needs `TRACK_ON_SRC`, a
    gorkaydemir/track_on checkout with the pinned commit, and the DINOv3 ViT-S+ snapshot next to the weights.
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("track_on")
    card, variant = registry.resolve("tracking", full)
    file = Path(registry.weights(card, variant, allow_download=False)[0])
    model = g.load(Path(os.environ["TRACK_ON_SRC"]), file, file.parent / g.DINOV3).cuda()
    a.note = f"gorkaydemir/track_on@{g.COMMIT[:12]} Predictor loop, 384 x 512, memory 72"

    def call(video: Any, queries: Any) -> Any:  # noqa: ANN401 - uint8 CUDA [T, 3, H, W], float [Q, 3] (t, x, y)
        return g.track(model, video.float() / 255, queries, amp=torch.float16 if fp16 else None)

    return call


def _locotrack(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """Upstream `LocoTrack.forward` offline (CUDA, unpatched) via `tools/golden/locotrack.py`, at 256 x 256.

    fp16: `torch.autocast("cuda", torch.float16)`. Needs `LOCOTRACK_SRC`, a KU-CVLAB/LocoTrack checkout with the pinned
    commit (with einops importable).
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("locotrack")
    card, variant = registry.resolve("tracking", full)
    file = Path(registry.weights(card, variant, allow_download=False)[0])
    model = g.load(Path(os.environ["LOCOTRACK_SRC"]), file, g.MODELS[f"{variant.id}@1"][1]).cuda()
    a.note = f"KU-CVLAB/LocoTrack@{g.COMMIT[:12]} forward, offline (whole video), frames resized to 256 x 256"

    def call(video: Any, queries: Any) -> Any:  # noqa: ANN401 - uint8 CUDA [T, 3, H, W], float [Q, 3] (t, x, y)
        return g.track(model, video.float() / 255, queries, amp=torch.float16 if fp16 else None)

    return call


class _Tracking(_LocalFp16):
    """The tracking bench call: the uint8 video `[T, 3, H, W]` and the float32 query grid (not scaled to uint8)."""

    def inputs(self, full: str) -> list[tuple[str, str, tuple[Any, ...]]]:  # noqa: ARG002 - the adapter signature
        """(label, scene, (video, queries)) per size, from `fastforks.tracking` `bench_inputs`."""
        import torch  # noqa: PLC0415

        from fastforks._core.bench import BenchSize  # noqa: PLC0415
        from fastforks.tracking._api import bench_inputs  # noqa: PLC0415

        like = types.SimpleNamespace(device=torch.device("cuda"))
        out = []
        for h, w in self.sizes:
            (video, queries), _ = bench_inputs(like, BenchSize(h, w))
            out.append((f"{h}x{w}", "synthetic (fastforks bench)", ((video * 255).round().byte(), queries)))
        return out


def _anysplat(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream AnySplat's inference modules as `tools/golden/anysplat.py` (long side 448, voxel 0.002 as released).

    `FF_ANYSPLAT_REPO`: a checkout with the pinned commit; `FF_ANYSPLAT_SITE`: `einops` + `jaxtyping` (upstream
    imports them). fp16 = upstream's own bf16 autocast around the aggregator (heads in float32).
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    site = os.environ.get("FF_ANYSPLAT_SITE")
    if site and site not in sys.path:
        sys.path.append(site)
    g = _golden("anysplat")
    card, variant = registry.resolve("multiview", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_ANYSPLAT_REPO"]), Path(weights))
    net.enc, net.adapter = net.enc.cuda(), net.adapter.cuda()
    a.note = f"InternRobotics/AnySplat@{g.REF_COMMIT[:12]} EncoderAnySplat inference, long side 448, voxel_size 0.002"

    def call(images: Any, *_: Any) -> Any:  # noqa: ANN401 - [V, 3, H, W] uint8 (bench) or float 0-1 (eval); no cameras
        x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
        return g.predict(net, x, 448, 0.002, bf16=fp16)["depth"]

    return call


def _pi3(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream `Pi3` as `tools/golden/pi3.py` (the runner's resizing, PIXEL_LIMIT 255000).

    `FF_PI3_REPO`: a checkout with the pinned commit. fp16 = the authors' bf16 autocast around the whole forward (their
    `example.py`).
    """
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    g = _golden("pi3")
    card, variant = registry.resolve("multiview", full)
    (weights,) = registry.weights(card, variant)
    net = g.network(Path(os.environ["FF_PI3_REPO"]), Path(weights)).cuda()
    a.note = f"yyfz/Pi3@{g.REF_COMMIT[:12]} Pi3.forward, PIXEL_LIMIT {g.PIXEL_LIMIT}"

    def call(images: Any, *_: Any) -> Any:  # noqa: ANN401 - [V, 3, H, W] uint8 (bench) or float 0-1 (eval); no cameras
        x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
        return g.predict(net, x, bf16=fp16)["depth"]

    return call


class _LocalCameras(_LocalFp16):
    """A calibrated multiview original: called with the images, intrinsics, cam_to_world and the depth range."""

    cameras = "required"


def _depthsplat(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream DepthSplat's `EncoderDepthSplat` as `tools/golden/depthsplat.py` (the runner's resizing, long side 256).

    `FF_DEPTHSPLAT_REPO`: a checkout with the pinned commit; `FF_DEPTHSPLAT_HUB`: a local copy of the
    `facebookresearch/dinov2` hub code; `FF_DEPTHSPLAT_SITE`: `einops`, `jaxtyping`, `e3nn` (upstream imports them).
    Upstream has no mixed precision of its own: fp16 = `torch.autocast(float16)` around the call.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    site = os.environ.get("FF_DEPTHSPLAT_SITE")
    if site and site not in sys.path:
        sys.path.append(site)
    g = _golden("depthsplat")
    card, variant = registry.resolve("multiview", full)
    (weights,) = registry.weights(card, variant)
    enc = g.network(Path(os.environ["FF_DEPTHSPLAT_REPO"]), Path(weights), Path(os.environ["FF_DEPTHSPLAT_HUB"]))
    enc = enc.cuda()
    a.note = f"cvg/depthsplat@{g.REF_COMMIT[:12]} EncoderDepthSplat, long side {g.LONG_SIDE}"

    def call(images: Any, k: Any, c2w: Any, depth_range: Any) -> Any:  # noqa: ANN401 - [V, 3, H, W] uint8 or 0-1
        x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
        near, far = (float(d) for d in depth_range)
        with torch.autocast("cuda", torch.float16, enabled=fp16):
            out = g.infer(enc, x, (k.float(), c2w.float(), near, far), g.LONG_SIDE)[0]
        depth = out["depths"][0][:, None].float()  # back in the input units (`infer`)
        return F.interpolate(depth, x.shape[2:], mode="bilinear", align_corners=False)[:, 0]

    return call


def _mvsplat(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - adapter signature
    """Upstream MVSplat's `EncoderCostVolume` as `tools/golden/mvsplat.py` (the runner's resizing, long side 256).

    `FF_MVSPLAT_REPO`: a checkout with the pinned commit; `FF_MVSPLAT_SITE`: `einops`, `jaxtyping`, `e3nn`,
    `omegaconf` (upstream imports them). Upstream has no mixed precision of its own: fp16 = `torch.autocast(float16)`.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    site = os.environ.get("FF_MVSPLAT_SITE")
    if site and site not in sys.path:
        sys.path.append(site)
    g = _golden("mvsplat")
    card, variant = registry.resolve("multiview", full)
    (weights,) = registry.weights(card, variant)
    enc = g.network(Path(os.environ["FF_MVSPLAT_REPO"]), Path(weights), 2).cuda()
    a.note = f"donydchen/mvsplat@{g.REF_COMMIT[:12]} EncoderCostVolume, long side {g.LONG_SIDE}"

    def call(images: Any, k: Any, c2w: Any, depth_range: Any) -> Any:  # noqa: ANN401 - [V, 3, H, W] uint8 or 0-1
        x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
        near, far = (float(d) for d in depth_range)
        with torch.autocast("cuda", torch.float16, enabled=fp16):
            _, dump, _, sc = g.infer(enc, x, (k.float(), c2w.float(), near, far), g.LONG_SIDE)
        depth = dump["depth"][0, ..., 0, 0][:, None].float() / sc
        return F.interpolate(depth, x.shape[2:], mode="bilinear", align_corners=False)[:, 0]

    return call


def _insid3(a: Any, full: str, fp16: bool) -> Any:  # noqa: ANN401, ARG001, FBT001 - adapter signature
    """Upstream `INSID3` as `tools/golden/insid3.py` (`FF_INSID3_REPO`; `FF_INSID3_WEIGHTS`: DINOv3 dir).

    fp16 = upstream as shipped (bf16 autocast around the encoder on CUDA, scikit-learn clustering on the CPU); fp32 =
    the same with that autocast off. The timed call: `set_reference`, `set_target`, `segment` (upstream's resizing).
    """
    import torch  # noqa: PLC0415

    g = _golden("insid3")
    tree = g.reference_tree(Path(os.environ["FF_INSID3_REPO"]), g.REF_COMMIT)
    sys.path.insert(0, str(tree))
    from models.insid3 import INSID3  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from transformers import DINOv3ViTModel  # noqa: PLC0415

    class Fp32(g.HubShim):
        def get_intermediate_layers(self, x: Any, n: int = 1, reshape: bool = True) -> Any:  # noqa: ANN401, FBT001, FBT002
            with torch.autocast("cuda", enabled=False):
                return super().get_intermediate_layers(x.float(), n, reshape)

    path = Path(os.environ["FF_INSID3_WEIGHTS"]) / "dinov3-vitl16-pretrain-lvd1689m"
    net = DINOv3ViTModel.from_pretrained(path, dtype=torch.float32).cuda().eval()  # ty: ignore[missing-argument] - transformers' from_pretrained decorator is inferred as an unbound method
    model = INSID3((g.HubShim if fp16 else Fp32)(net), device="cuda").eval()
    a.note = f"visinf/INSID3@{g.REF_COMMIT[:12]} set_reference + set_target + segment" + (
        ", bf16 autocast (as shipped)" if fp16 else ", its encoder autocast off"
    )

    def call(target: Any, reference: Any, mask: Any) -> Any:  # noqa: ANN401
        model.set_reference(reference, mask.cpu())
        model.set_target(target)
        return model.segment()

    return call


def _insid3_prep(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """uint8 target, float 0-1 reference (CHW, CUDA) -> upstream's ImageNet-normalised `[1, 3, H, W]` tensors."""
    import torch  # noqa: PLC0415

    target, reference, mask = args
    mean = torch.tensor((0.485, 0.456, 0.406), device=target.device).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=target.device).view(3, 1, 1)
    return ((target.float() / 255 - mean) / std)[None], ((reference.float() - mean) / std)[None], mask


class _LocalReference(_LocalFp16):
    """An in-context segmentation original: called with the image, the reference image and its mask."""

    prompts = ("reference",)


class _LocalMask(_Local):
    """An inpainting original: called with the image and the bench's synthetic mask (fp16 tier: autocast)."""

    mask = "required"


class _LocalTrimap(_LocalFp16):
    """A trimap-guided matting original: called with the image and the bench's synthetic trimap."""

    trimap = "required"


def _xywh(boxes: Any) -> list[list[float]]:  # noqa: ANN401 - [P, 4] xyxy tensor
    return [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in boxes.tolist()]


LOCAL = {  # family -> adapter where the bench has none or a local one wins (flow, pose, RTMDet, timm classifiers)
    "raft": _Local("flow", _raft, ((1080, 1920), (375, 1242))),
    "sea_raft": _Local("flow", _sea_raft, ((1080, 1920), (375, 1242))),
    "memfof": _Local("flow", _memfof, ((1080, 1920), (375, 1242))),
    "memflow": _Local("flow", _memflow, ((1080, 1920), (375, 1242))),
    "neuflow_v2": _LocalFp16("flow", _neuflow_v2, ((1080, 1920), (375, 1242))),
    "megaflow": _LocalFp16("flow", _megaflow, ((1080, 1920), (375, 1242))),
    "recover": _Local("flow", _recover, ((1080, 1920), (375, 1242))),
    "rife": _LocalFp16("interpolation", _rife, ((1080, 1920),)),
    "gimm_vfi": _Local("interpolation", _gimm_vfi, ((480, 854), (1080, 1920))),
    "ema_vfi": _Local("interpolation", _ema_vfi, ((480, 854), (1080, 1920))),
    "film": _LocalFp16("interpolation", _film, ((480, 854), (1080, 1920))),
    "realesrgan": _LocalFp16("restoration", _realesrgan, ((512, 512),), _hwc01),
    "span": _Local("restoration", _span, ((512, 512),), _bchw01),
    "compact": _LocalFp16("restoration", _compact, ((512, 512),), _hwc01),
    "lama": _LocalMask("restoration", _lama, ((512, 512),)),
    "realplksr": _Local("restoration", _realplksr, ((512, 512),), _bchw01),
    "swinir": _Local("restoration", _swinir, ((512, 512),), _bchw01),
    "hat": _Local("restoration", _hat, ((512, 512),), _bchw01),
    "dat": _Local("restoration", _dat, ((512, 512),), _bchw01),
    "drct": _Local("restoration", _drct, ((512, 512),), _bchw01),
    "codeformer": _Local("restoration", _codeformer, ((512, 512),), _bchw01),
    "gfpgan": _Local("restoration", _gfpgan, ((512, 512),), _bchw01),
    "gmflow": _Local("flow", _gmflow, ((1080, 1920), (375, 1242))),
    "flowformer": _Local("flow", _flowformer, ((1080, 1920), (375, 1242))),
    "flowseek": _Local("flow", _flowseek, ((1080, 1920), (375, 1242))),
    "waft": _Local("flow", _waft, ((1080, 1920), (375, 1242))),
    "vitpose": _Local("pose", _vitpose, ((1080, 1920),), lambda a: (a[0].permute(1, 2, 0).cpu().numpy(), _xywh(a[1]))),
    "rtmpose": _Local(
        "pose", _topdown, ((1080, 1920),), lambda a: (a[0].flip(0).permute(1, 2, 0).cpu().numpy(), a[1].cpu().numpy())
    ),
    "hrnet": _Local(
        "pose", _topdown, ((1080, 1920),), lambda a: (a[0].flip(0).permute(1, 2, 0).cpu().numpy(), a[1].cpu().numpy())
    ),
    "dfine": _Local("detection", _dfine, ((800, 1344),), lambda a: (a[0].permute(1, 2, 0).contiguous().cpu().numpy(),)),
    "lwdetr": _Local(
        "detection", _dfine, ((800, 1344),), lambda a: (a[0].permute(1, 2, 0).contiguous().cpu().numpy(),)
    ),
    "owlv2": _Local("detection", _dfine, ((800, 1344),), lambda a: (a[0].permute(1, 2, 0).contiguous().cpu().numpy(),)),
    "grounding_dino": _Local(
        "detection", _dfine, ((800, 1344),), lambda a: (a[0].permute(1, 2, 0).contiguous().cpu().numpy(),)
    ),
    "rfdetr_keypoint": _RFDETRKeypoint(),
    "rfdetr": _RFDETR("detection", ((800, 1344),)),
    "rtmo": _Local(
        "pose",
        _rtmo,
        ((1080, 1920),),
        lambda a: (a[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy(),),
        needs_boxes=False,
    ),
    "retinaface": _Local(
        "pose",
        _retinaface,
        ((1080, 1920),),
        lambda a: (a[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy(),),
        needs_boxes=False,
    ),
    "rtmdet": _Local(
        "detection", _rtmdet, ((800, 1344),), lambda a: (a[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy(),)
    ),
    "yolox": _Local(
        "detection", _yolox, ((800, 1344),), lambda a: (a[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy(),)
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
    "bisenet": _Local("segmentation", _bisenet, ((512, 512),)),
    "shvit": _Local("classification", _convnext, ((224, 224),)),
    "starnet": _Local("classification", _convnext, ((224, 224),)),
    "fastvit": _Local("classification", _convnext, ((224, 224),)),  # the authors' reparameterized inference
    "wd_tagger": _Local("classification", _convnext, ((448, 448),)),  # wdv3_timm.py: pad square, BGR
    "nsfw_vit": _Local("classification", _convnext, ((224, 224),)),  # transformers ViTForImageClassification
    "efficientvit_sam": _Local("promptable_segmentation", _efficientvit_sam, ((1024, 1024),), _hwc_point),
    "clipseg": _LocalText("promptable_segmentation", _clipseg, ((840, 1297), (1024, 1024)), _hwc_text),
    "sam_hq": _Local("promptable_segmentation", _sam_hq, ((1024, 1024),), _hwc_point),
    "mobile_sam": _Local("promptable_segmentation", _mobile_sam, ((1024, 1024),), _hwc_point),
    "romav2": _LocalFp16("matching", _romav2, ((768, 1024),)),
    "romav1": _LocalFp16("matching", _romav1, ((768, 1024),)),
    "marigold_normals": _LocalFp16("depth", _marigold_normals, ((994, 1482),)),
    "lotus": _LocalFp16("depth", _lotus, ((994, 1482),)),
    "stablenormal": _LocalFp16("depth", _stablenormal, ((994, 1482),)),
    "lightglue": _LocalFp16("matching", _lightglue, ((768, 1024),)),
    "xfeat": _Local("matching", _xfeat, ((768, 1024),)),
    "dedode": _LocalFp16("matching", _dedode, ((768, 1024),)),
    "loftr": _Local("matching", _loftr, ((768, 1024),)),
    "birefnet": _LocalFp16("matting", _birefnet, ((1080, 1920),)),
    "ben2": _LocalFp16("matting", _ben2, ((1080, 1920),)),
    "megaloc": _Local("retrieval", _megaloc, ((480, 640),)),
    "boq": _Local("retrieval", _boq, ((480, 640),)),
    "geocalib": _Local("calibration", _geocalib, ((1080, 1920),)),
    "anycalib": _Local("calibration", _anycalib, ((1080, 1920),)),
    "u2net": _Local("matting", _rsu, ((1080, 1920),)),
    "lineart": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "lineart_anime": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "hed": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "pidinet": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "teed": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "anyline": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "mlsd": _Local("edges", _edges, ((1024, 1024), (1080, 1920))),
    "isnet": _Local("matting", _rsu, ((1080, 1920),)),
    "inspyrenet": _Local("matting", _inspyrenet, ((1080, 1920),)),
    "tapnext": _Tracking("tracking", _tapnext, ((480, 854),)),
    "vitmatte": _LocalTrimap("matting", _vitmatte, ((1080, 1920),)),
    "anysplat": _LocalFp16("multiview", _anysplat, ((336, 518),)),
    "pi3": _LocalFp16("multiview", _pi3, ((336, 518),)),
    "depthsplat": _LocalCameras("multiview", _depthsplat, ((336, 518),)),
    "mvsplat": _LocalCameras("multiview", _mvsplat, ((336, 518),)),
    "insid3": _LocalReference("promptable_segmentation", _insid3, ((1024, 1024),), _insid3_prep),
    "tapir": _Tracking("tracking", _tapir, ((480, 854),)),
    "alltracker": _Tracking("tracking", _alltracker, ((480, 854),)),
    "track_on": _Tracking("tracking", _track_on, ((480, 854),)),
    "locotrack": _Tracking("tracking", _locotrack, ((480, 854),)),
}


def own_option(task: str, ref: str) -> bool:
    """Whether the authors' code has its own fp16 / mixed-precision option (then `upstreamhalf` is not run)."""
    bu = bench()
    _, card, _ = full_ref(task, ref)
    a = LOCAL.get(card.family) or bu.FAM.get(card.family)
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
