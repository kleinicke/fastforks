"""Per-model parity with golden outputs of the reference implementation (tools/golden/<family>.py).

Goldens live outside git (`FASTFORKS_GOLDEN_PATH`), identified by the SHA-256 in each family's `golden.toml`, which
also holds the family's tolerances and the kernel builds a GPU run validated (`--record`); the tests are skipped
without them. `FASTFORKS_REQUIRE_GOLDENS` turns a missing golden into a failure: `1` for all, or the path of a file
listing the required refs (CI: the goldens `tools/fetch_goldens.py --required` selected). CPU: portable float32 must
match the reference closely. GPU: every backend must stay within the fp16 tolerance of the float32 reference (the main
output; a family's `gpu_extra` tolerances cover all its other stored outputs; `gpu_min_score` (pose, matching): the
points whose reference confidence is below it are not position-compared, as the pose argmax decoders pick between
near-equal peaks of unsure keypoints and a matching warp is undefined outside the overlap), and `kernels` must not hand
any work to PyTorch (`fastforks._core.kernels.fallbacks()`). Kernels families on `cudnn_hacc` are checked on two calls:
the first call of a new size (torch convolutions until a geometry's plan is built) and one with every plan built. A
boolean option named like a stored output (MoGe `normals`) is on.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import gc
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import goldens
from fastforks import (
    classification,
    depth,
    detection,
    features,
    flow,
    matching,
    multiview,
    pose,
    promptable_segmentation,
    segmentation,
    stereo,
)
from fastforks._core import kernels, registry
from fastforks._core.errors import BackendUnavailableError, WeightsDownloadError, WeightsError
from fastforks._core.image import to_rgb_batch
from fastforks._core.preprocess import resize_crop, scale
from fastforks._core.tasks import spec
from fastforks.kernels import cudnn_hacc
from fastforks.promptable_segmentation._api import _normalised
from synth import clip, clip_prompts, concepts, image, pair, prompts, scene, size

FAMILIES = goldens.load_all(registry.TASKS)
GOLDEN = {ref: e for fam in FAMILIES.values() for ref, e in fam["refs"].items()}
FAMILY = {ref: name for name, fam in FAMILIES.items() for ref in fam["refs"]}
TASK = {ref: FAMILIES[fam]["task"] for ref, fam in FAMILY.items()}
# features goldens: the synthetic image resized to 112 x 168 (DINOv2) / 112 x 176 (DINOv3 ViT) / 128 x 160
# (ConvNeXt); HF hidden_states 0, 1 (families that have them)
FEATURES_SIZE = 112
FEATURES_LAYERS = (0, 1)
API: dict[str, Any] = {
    "stereo": stereo,
    "depth": depth,
    "classification": classification,
    "segmentation": segmentation,
    "features": features,
    "promptable_segmentation": promptable_segmentation,
    "detection": detection,
    "multiview": multiview,
    "pose": pose,
    "flow": flow,
    "matching": matching,
}
SCENE_INPUTS = ("intrinsics", "cam_to_world", "depth_range")  # multiview: passed only where the model requires them
HEAVY_CPU = "FASTFORKS_HEAVY_CPU_TESTS"  # 1: every CPU case, also those left out of the release suite for time / RAM


def _smallest(family: str) -> str:
    refs, task = list(FAMILIES[family]["refs"]), FAMILIES[family]["task"]
    return min(refs, key=lambda r: sum(w.size or 0 for w in registry.resolve(task, r)[1].weights))


SMALLEST = sorted(_smallest(f) for f in FAMILIES)  # the CPU sweeps (batch, sizes, export, weights) use these

# CPU parity runs on the smallest variant of each family plus the variants below, whose code path differs from it; the
# others differ only in size or weights (the GPU goldens check every variant; FASTFORKS_HEAVY_CPU_TESTS=1: all on CPU)
CPU_ALSO = {
    "aanet-plus-sceneflow@1": "AANet+ (plus = true)",
    "bridgedepth-rvc-pretrain@1": "ConvNeXt-T + DPT feature encoder (smallest: ResNet)",
    "c-foundationstereo-s@1": "arch fs (smallest: ffs)",
    "cas-gwcnet-gc-sceneflow@1": "arch gc (smallest: arch c)",
    "da3-large-1.1@1": "arch large (smallest: metric)",
    "da3mono-large@1": "arch mono (smallest: metric)",
    "deim-n-coco@1": "DEIM: SiLU decoder MLPs, the authors' checkpoint keys (smallest: ReLU, transformers keys)",
    "depth-anything-v2-metric-hypersim-s@1": "metric head: sigmoid * max_depth (smallest: relative, ReLU)",
    "dinov2-vits14@1": "no register tokens (smallest: registers)",
    "dinov2-vitg14-reg@1": "fused SwiGLU FFN (ViT-g only)",
    "dinov3-vith16plus@1": "gated SwiGLU FFN (ViT-H+ only)",
    "distill-any-depth-l@1": "DepthAnything checkpoint layout, renamed on load (smallest: DepthAnythingV2 keys)",
    "fast-acvnet-plus-generalization@1": "Fast-ACVNet+ (plus = true)",
    "gwcnet-gc-sceneflow@1": "concat volume (smallest: group-wise only)",
    "hitnet-middlebury-d400@1": "RGB input, 6 levels (smallest: grey, 5 levels)",
    "hrnet-w48-fcn-ade20k@1": "HRNet-W48 backbone + FCN head (smallest: ResNet-101 + separable ASPP)",
    "mask-rcnn-r50-fpn-3x-coco@1": "mask head (smallest: Faster R-CNN, boxes only)",
    "mocha-v2-sceneflow@1": "mccv = false (smallest: true)",
    "raft-large@1": "residual blocks, convex upsampling, two GRUs (smallest: bottleneck, bilinear, one)",
    "rtmo-l@1": "CSPRep blocks without the 1 x 1 output conv (smallest: with it)",
    "rtmpose-t-face6@1": "1 : 1 crop, 106 keypoints (smallest: 3 : 4 crop, 17)",
    "rtmw-m@1": "RTMW: PAFPN neck and two-scale head (smallest: backbone and SimCC head only)",
    "moge-3-vitl@1": "MoGe version 3 (smallest: version 2)",
    "resnet50.tv2_in1k@1": "bottleneck blocks (smallest: ResNet-18, basic blocks)",
}
# Smallest variants whose CPU parity takes about a minute or more (a large fixed network input): only with
# FASTFORKS_HEAVY_CPU_TESTS=1. The release suite still runs them on the CPU (test_export, test_batch, ...), and the GPU
# goldens check their parity at the fp16 tolerance.
CPU_HEAVY = {
    "vggt-1b@1": "1.2 B parameters on 4 views at 392 x 518 (CPU ~200 s)",
    "mapanything-apache@1": "4 views at 392 x 518 (CPU ~170 s)",
    "mvsanywhere@1": "4 views at 480 x 640 (CPU ~75 s)",
    "vit_so400m_patch14_siglip_378.webli_ft_in1k@1": "SigLIP So400m at 378 px (CPU ~55 s)",
    "marigold-depth-v1-1@1": "Stable Diffusion UNet, 4 steps at 768 px (CPU ~40 s)",
    "marigold-normals-lcm-v0-1@1": "Stable Diffusion UNet, 1 step at 768 px (CPU ~20 s)",
    "lotus-normal-d-v1-1@1": "Stable Diffusion UNet, 1 step at 768 px (CPU ~20 s)",
    "romav2@1": "DINOv3 ViT-L at 800 px + two-view ViT + refiners at 1280 px (CPU ~60 s)",
}
CPU_REFS = set(SMALLEST) - set(CPU_HEAVY) | set(CPU_ALSO)
# The CPU sweeps outside the goldens run with fewer iterations and smaller network inputs: the same code, less work
# (the goldens check the defaults).
_SWEEP_OPTIONS: dict[str, dict[str, Any]] = {
    "marigold": {"steps": 1, "long_side": 256},
    "marigold_normals": {"long_side": 256},
    "lotus": {"long_side": 256},
    "moge": {"resolution_level": 0},
    "unik3d": {"resolution_level": 0},
    "zoedepth": {"flip": False},
    "vggt": {"long_side": 224},
    "vggt_omega": {"area_side": 256},
    "mapanything": {"long_side": 224},
    "mvsanywhere": {"long_side": 320},
    "midas": {"input_size": 256},
    "depth_anything_v2": {"short_side": 252},
    "video_depth_anything": {"short_side": 252},
    "depth_anything_3": {"long_side": 252},
    "defom_stereo": {"iters": 2, "scale_iters": 1},
    "romav2": {"coarse_size": 256, "fine_size": 384},
}
# Refs left out of the CPU sweeps (batch, sizes, export, ...) for time: a fixed large network input, so the sweep
# options cannot shrink the work; only with FASTFORKS_HEAVY_CPU_TESTS=1. test_cpu_portable still runs.
SWEEP_HEAVY = {
    "depth-pro@1": "two DINOv2-L encoders at a fixed 1536 px (CPU ~65 s per forward; the sweeps took ~560 s)",
    "metric3d-vit-large@1": "ViT-L at a fixed network input (the sweeps took ~260 s; compile + export > 13 GB RSS)",
    "sapiens-seg-0.3b-goliath@1": "a fixed 1024 x 768 network input (the sweeps took ~105 s)",
}


def sweep_options(ref: str) -> dict[str, Any]:
    """Options of a ref in the CPU sweeps: 2 iterations (iterative models) and the family's smaller network input."""
    card = API[TASK[ref]].model_card(ref)
    return ({"iters": 2} if "iters" in card.options else {}) | _SWEEP_OPTIONS.get(FAMILY[ref], {})


SWEEP: dict[str, Any] = {}  # the model of the current ref in the CPU sweeps (at most one; conftest empties it)


def sweep_model(ref: str) -> Any:
    """The portable CPU model of `ref` with `sweep_options`, skipped if its weights are not available locally.

    Loaded once for consecutive tests of the same ref: conftest runs the ref-parametrized tests of modules that set
    `LOAD_ONCE` next to each other, and drops the model after the last of them. Tests must not modify it (monkeypatch is
    fine: it is undone before the next test).
    """
    if ref in SWEEP_HEAVY and not os.environ.get(HEAVY_CPU):
        pytest.skip(f"{SWEEP_HEAVY[ref]}; {HEAVY_CPU}=1 runs it")
    if ref not in SWEEP:
        SWEEP.clear()
        gc.collect()  # the previous model's memory first
        card, variant = registry.resolve(TASK[ref], ref)
        try:
            registry.weights(card, variant, allow_download=False)
        except WeightsError as e:
            pytest.skip(f"weights of {ref} not available locally: {e}")
        SWEEP[ref] = API[TASK[ref]].load(ref, device="cpu", options=sweep_options(ref))
    return SWEEP[ref]


def options(ref: str, want: dict[str, np.ndarray]) -> dict[str, Any]:
    """The ref's golden.toml `options` (`[name, value]` rows, e.g. R-CNN `score_threshold`), plus the boolean options
    that switch on an output the golden stores (e.g. MoGe `normals`)."""
    card = API[TASK[ref]].model_card(ref)
    rows = GOLDEN[ref].get("options", [])
    assert isinstance(rows, list)
    return dict(rows) | {k: True for k in want if isinstance(card.options.get(k), bool)}


def run(  # noqa: PLR0913 - the golden run's settings
    ref: str,
    *,
    device: str,
    mode: str = "portable",
    dtype: str | None = None,
    opts: dict[str, Any] | None = None,
    want: dict[str, np.ndarray] | None = None,
    built: bool = False,
) -> tuple[list[dict[str, torch.Tensor]], Any]:
    """The model's output maps of each call on the golden's input, and its backend.

    The input is the synthetic image / pair, or the real image a golden stores as `input` (RGB; `want` supplies it).
    `built`: a second call with every `cudnn_hacc` plan built at once (the first call of a new size runs torch
    convolutions; see `fastforks.kernels.cudnn_hacc`).
    """
    m = API[TASK[ref]].load(ref, mode=mode, device=device, dtype=dtype, options=opts)
    return calls_of(m, ref, want, built=built)


def calls_of(
    m: Any, ref: str, want: dict[str, np.ndarray] | None, *, built: bool = False
) -> tuple[list[dict[str, torch.Tensor]], Any]:
    """`run` on a loaded model `m`."""
    calls = [forward(m, ref, want)]
    if built:
        with _plans_at_once():
            calls.append(forward(m, ref, want))
    return [c for c, _ in calls], calls[-1][1]


def _scene(m: Any) -> tuple[Any, dict[str, torch.Tensor]]:
    """A multiview model on `synth.scene()` (with the cameras / depth range it requires): result and maps."""
    s = scene()
    given = {k: s[k] for k in SCENE_INPUTS if getattr(m._runner, k) == "required"}
    r = m(list(s["images"]), **given)
    return r, {k: getattr(r, k) for k in spec("multiview").standard if k not in given} | dict(r.extras)


@contextlib.contextmanager
def _plans_at_once() -> Any:
    old, cudnn_hacc.BUILD_AFTER = cudnn_hacc.BUILD_AFTER, 0
    try:
        yield
    finally:
        cudnn_hacc.BUILD_AFTER = old


def forward(m: Any, ref: str, want: dict[str, np.ndarray] | None) -> tuple[dict[str, torch.Tensor], Any]:  # noqa: C901, PLR0912, PLR0915 - one branch per task
    """One call of `m` on the golden's input: its output maps and backend."""
    task = TASK[ref]
    img = want["input"].round().astype(np.uint8) if want and "input" in want else image()
    if task == "stereo":
        r = m(*pair(*size(m.variant.io.min_size)))
        maps = {"disparity": r.disparity, "confidence": r.confidence, **r.extras}
    elif task == "depth":
        r = m(img)
        maps = {**{k: getattr(r, k) for k in spec("depth").standard}, **r.extras}
    elif task == "multiview":
        r, maps = _scene(m)
    elif task == "classification":
        r = m(img)
        maps = {"logits": r.logits}
    elif task == "features":
        r = m(img, size=FEATURES_SIZE, layers=FEATURES_LAYERS if "hidden_states" in m.variant.outputs else ())
        maps = {"last_hidden_state": r.last_hidden_state, "pooler_output": r.pooler_output}
        maps |= {f"hidden_states.{i}": r.hidden_states[i] for i in r.layers} if r.hidden_states else {}
    elif task == "promptable_segmentation" and "text" in m.prompts:  # every query's raw output, stored masks only
        assert want is not None, "concept goldens store their input image and the selected queries"
        emb = m.get_image_embeddings(img)
        maps = {}
        for prefix, p in concepts().items():  # key prefix per prompt (text, exemplar boxes, both)
            boxes, labels = m._check_exemplars(p.get("input_boxes"), p.get("input_boxes_labels"), p.get("text"))
            boxes = _normalised(boxes, emb.image_size)
            with m._session():
                out = m._runner.segment_concept(dict(emb.features), p.get("text"), boxes, labels)
            q = torch.from_numpy(want[f"{prefix}mask_query"]).long().to(m.device)
            maps |= {f"{prefix}{k}": out[k] for k in ("scores", "boxes")} | {
                f"{prefix}pred_masks": out["pred_masks"][q]
            }
        return {k: v.float().cpu() for k, v in maps.items()}, m.backend
    elif task == "promptable_segmentation":  # one embedding, the fixed prompts (key prefix per prompt)
        emb = m.get_image_embeddings(img)
        assert isinstance(emb, promptable_segmentation.ImageEmbeddings)
        maps = {}
        for prefix, p in prompts().items():
            pr = m(emb, **p)
            maps |= {f"{prefix}pred_masks": pr.pred_masks, f"{prefix}iou_scores": pr.iou_scores}
        if want is not None and "video_pred_masks" in want:  # video goldens: `track` on the synth clip per prompt
            for prefix, p in clip_prompts().items():
                tr = m.track(clip(), **p)
                maps |= {f"{prefix}pred_masks": tr.pred_masks, f"{prefix}object_score_logits": tr.object_score_logits}
        return {k: v.float().cpu() for k, v in maps.items()}, m.backend
    elif task == "detection":  # instances in score order; labels and masks stored as float
        r = m(img)
        assert isinstance(r, detection.DetectionResult)
        maps = {k: getattr(r, k) for k in spec("detection").standard}
    elif task == "pose":  # the golden's person boxes; they come back as `boxes`
        assert want is not None, "pose goldens store their input image and person boxes"
        r = m(img, boxes=want["boxes"]) if m.needs_boxes else m(img)
        maps = {k: getattr(r, k) for k in spec("pose").standard}
    elif task in {"flow", "matching"}:  # the golden's image pair
        assert want is not None, f"{task} goldens store their image pair"
        r = m(img, want["input2"].round().astype(np.uint8))
        maps = {k: getattr(r, k, None) for k in spec(task).standard}  # sparse matchers: no warp / certainty
    else:
        r = m(img, logits=True)
        assert r.logits is not None
        # mask classification: goldens hold upstream's class probabilities (log space would amplify noise near 0)
        maps = {"logits": r.logits.exp() if m._runner.probabilities else r.logits}
    return {k: v.float().cpu() for k, v in maps.items() if v is not None}, r.backend


# stored inputs of a golden (the image(s); per concept prompt, the queries whose masks are kept)
INPUTS = ("input", "input2", *(f"{prefix}mask_query" for prefix in concepts()))


def primary(ref: str, maps: dict[str, np.ndarray]) -> str:
    return next(k for k in spec(TASK[ref]).standard if k in maps)  # the map the GPU tolerances apply to


def tolerance(ref: str, mode: str | None = None) -> dict[str, float]:
    """The family's `[tolerance]`; a `[refs]` entry may override `gpu_mean`, `gpu_p99`, `gpu_extra` and `cpu_extra`.

    `gpu_mean_<mode>` / `gpu_p99_<mode>` (family or ref) bound one mode only, e.g. a kernels path that is much closer to
    the reference than portable fp16, so a bound fitted to portable would not catch a kernels regression.
    """
    own = {k: v for k, v in GOLDEN[ref].items() if k.startswith(("gpu_mean", "gpu_p99", "gpu_extra", "cpu_extra"))}
    tol = FAMILIES[FAMILY[ref]]["tolerance"] | own
    return tol | {k: tol[f"{k}_{mode}"] for k in ("gpu_mean", "gpu_p99") if f"{k}_{mode}" in tol}


def golden(ref: str) -> dict[str, np.ndarray]:
    root = os.environ.get("FASTFORKS_GOLDEN_PATH")
    path = Path(root or ".") / f"{ref}.npz"
    if not root or not path.is_file():
        why = f"golden {ref} not available (set FASTFORKS_GOLDEN_PATH)"
        if _required(ref):
            pytest.fail(why)
        pytest.skip(why)
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == GOLDEN[ref]["sha256"], f"{path} does not match golden.toml"
    with np.load(path) as z:
        return dict(z)


def _required(ref: str) -> bool:
    req = os.environ.get("FASTFORKS_REQUIRE_GOLDENS", "")
    if req in {"", "0"}:
        return False
    if req == "1":
        return True
    return ref in Path(req).read_text().split()  # a missing list file fails the test: the CI setup is broken


def run_or_skip(ref: str, **kw: Any) -> tuple[list[dict[str, torch.Tensor]], Any]:
    """`run`, skipped if the weights are gated (HTTP 401 / 403) and not in FASTFORKS_WEIGHTS_PATH (SAM 3).

    Only refs marked `gated = true` in their golden.toml skip, also with FASTFORKS_REQUIRE_GOLDENS (CI has no Hugging
    Face login); a 401 / 403 of any other ref fails.
    """
    try:
        return run(ref, **kw)
    except WeightsDownloadError as e:
        gated = GOLDEN[ref].get("gated", False) and any(f"HTTP Error {c}" in str(e) for c in (401, 403))
        if not gated:
            raise
        pytest.skip(f"{ref}: gated weights, not in FASTFORKS_WEIGHTS_PATH")


NEAR = 1.0  # px: a golden keypoint is recalled by an output keypoint this close (sparse matchers)


def sparse_parity(want: dict[str, np.ndarray], got: dict[str, torch.Tensor]) -> tuple[float, float, np.ndarray, float]:
    """Sparse matchers (`pairs`): the keypoint order (score ties) and the float noise differ between backends, so each
    golden keypoint is paired with its nearest output keypoint and vice versa. The recall (share of the golden
    keypoints of both images within `NEAR` px), the match agreement (the smaller of the shares of golden and output
    matches whose two keypoints pair up with a match of the other side), the distances of the recalled keypoints and
    the max |score diff| of the agreeing matches."""
    near, dist = [], []
    for k in ("keypoints1", "keypoints2"):
        d = torch.cdist(torch.from_numpy(want[k]).double(), got[k].double())
        if d.numel() == 0:
            return 0.0, 0.0, np.zeros(1), 0.0
        to_got, to_want = d.min(1), d.min(0)
        dist.append(to_got.values.numpy())
        near.append(
            (
                to_got.indices.numpy(),
                to_got.values.numpy() <= NEAR,
                to_want.indices.numpy(),
                to_want.values.numpy() <= NEAR,
            )
        )
    pw, pg = want["pairs"].astype(np.int64), got["pairs"].numpy().astype(np.int64)
    index_g = {(int(i), int(j)): n for n, (i, j) in enumerate(pg)}
    index_w = {(int(i), int(j)) for i, j in pw}
    (g1, ok1, w1, okw1), (g2, ok2, w2, okw2) = near
    hits = [index_g.get((int(g1[i]), int(g2[j]))) if ok1[i] and ok2[j] else None for i, j in pw]
    back = [okw1[i] and okw2[j] and (int(w1[i]), int(w2[j])) in index_w for i, j in pg]
    agree = min(np.mean([h is not None for h in hits]) if len(pw) else 1.0, np.mean(back) if len(pg) else 1.0)
    scores = got["scores"].numpy()
    ds = [abs(float(scores[h]) - float(want["scores"][n])) for n, h in enumerate(hits) if h is not None]
    d = np.concatenate(dist)
    return float((d <= NEAR).mean()), float(agree), d[d <= NEAR], max(ds, default=0.0)


def cpu_parity(ref: str, want: dict[str, np.ndarray], got: dict[str, torch.Tensor]) -> list[str]:
    """Per output: shape, max / mean |diff| against the golden and the bound (masks: the fraction of differing pixels);
    the rows of the outputs out of bounds come first, marked `FAIL`."""
    tol = tolerance(ref)  # optional cpu_rtol: maps with huge values (DINOv3 ViT-L hidden states ~1e5: 1 ulp > atol)
    rows = tol.get("cpu_extra", [])  # optional [key, atol] rows: one output bound looser than cpu_atol
    assert isinstance(rows, list)
    atol, rtol = dict(rows), tol.get("cpu_rtol", 0)
    if "pairs" in want:  # sparse matchers: keypoint and match sets, not element-wise
        recall, agree, d, ds = sparse_parity(want, got)
        bad = recall < tol["cpu_min_recall"] or agree < tol["cpu_min_match"] or ds > tol["cpu_atol"]
        line = (
            f"{'FAIL ' * bad}keypoints recall {recall:.4f} (min {tol['cpu_min_recall']}), matches {agree:.4f} (min"
            f" {tol['cpu_min_match']}), max |keypoint diff| {d.max(initial=0):.3g}, max |score diff| {ds:.3g} (atol"
            f" {tol['cpu_atol']})"
        )
        return [line]
    lines = []
    for k, v in sorted(want.items()):
        g = got[k].numpy()
        if g.shape != v.shape:
            lines.append(f"FAIL {k}: shape {g.shape}, golden {v.shape}")
            continue
        if k in spec(TASK[ref]).bool_outputs:  # thresholded maps: float noise right at the threshold flips pixels
            frac = float(((g > 0.5) != (v > 0.5)).mean())  # stored as 0 / 1
            bound = tol.get("cpu_mask_mismatch", 0.0)
            lines.append(f"{'FAIL ' * (frac > bound)}{k} {v.shape}: {frac:.2e} of the pixels differ (bound {bound})")
            continue
        a = atol.get(k, tol["cpu_atol"])
        with np.errstate(invalid="ignore"):  # inf - inf
            err = np.abs(g.astype(np.float64) - v)
            ok = (err <= a + rtol * np.abs(v)) | (g == v) | (np.isnan(g) & np.isnan(v))  # as assert_allclose
        fin = np.where(np.isfinite(err), err, -1.0)
        worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(fin)), fin.shape)) if fin.size else ()
        lines.append(
            f"{'FAIL ' * bool((~ok).any())}{k} {v.shape}: max |diff| {fin.max(initial=0):.3g} at {worst}, mean"
            f" {fin[fin >= 0].mean() if (fin >= 0).any() else 0:.3g}, {int((~ok).sum())} out of bounds (atol {a},"
            f" rtol {rtol}), |golden| max {np.abs(v).max(initial=0):.3g}"
        )
    return sorted(lines, key=lambda line: not line.startswith("FAIL"))


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", sorted(GOLDEN))
def test_cpu_portable(ref, record_property):
    """Portable float32 on the CPU matches the golden within `cpu_atol`. Release suite: the refs in `CPU_REFS`; every
    output's error is recorded (`record_property("parity", ...)`, in JUnit XML) and listed on a failure."""
    if ref not in CPU_REFS and not os.environ.get(HEAVY_CPU):
        why = CPU_HEAVY.get(ref, "same code as a CPU-checked variant (size / weights only); the GPU goldens check it")
        pytest.skip(f"{why}; {HEAVY_CPU}=1 runs it")
    want = golden(ref)
    (got,), _ = run_or_skip(ref, device="cpu", opts=options(ref, want), want=want)
    for k in INPUTS:
        want.pop(k, None)
    assert set(got) == set(want), f"outputs {sorted(got)}, golden {sorted(want)}"
    lines = cpu_parity(ref, want, got)
    record_property("parity", lines)
    assert not any(line.startswith("FAIL") for line in lines), f"{ref} vs golden:\n" + "\n".join(lines)


MIN_SURE = 1 / 3  # share of a golden's points `gpu_min_score` must keep (pose goldens: 42% .. 85% of the keypoints)
SCORE = {"pose": "keypoint_scores", "matching": "certainty"}  # the confidence `gpu_min_score` applies to


def sure_mask(score: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The mask `score` broadcast to an output with one more axis (pose `[P, K, 2]`, matching `[B, 2, H, W]`)."""
    axis = next(i for i in reversed(range(len(shape))) if shape[:i] + shape[i + 1 :] == score.shape)
    return np.broadcast_to(np.expand_dims(score, axis), shape)


def paired_detections(want: dict[str, np.ndarray], got: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """`got`'s instances in the golden's order: near-equal scores swap places between backends (fp16), so each golden
    box in turn takes the closest (L1) box not yet taken. Other counts are left to the shape check."""
    n = want["boxes"].shape[0]
    if n == 0 or tuple(got["boxes"].shape) != want["boxes"].shape:
        return got
    dist = torch.cdist(torch.from_numpy(want["boxes"]).double(), got["boxes"].double(), p=1)
    order: list[int] = []
    for row in dist:
        row[order] = float("inf")
        order.append(int(row.argmin()))
    return {k: v[order] if v.ndim and v.shape[0] == n else v for k, v in got.items()}


def _check(ref: str, mode: str, want: dict[str, np.ndarray], got: dict[str, torch.Tensor], what: str) -> np.ndarray:
    """Assert `got` within the GPU tolerances of the golden; the error map of the primary output."""
    if TASK[ref] == "detection":
        got = paired_detections(want, got)
    for k in want.keys() & got.keys() if "pairs" not in want else ():  # e.g. the same detections (not sparse counts)
        assert tuple(got[k].shape) == want[k].shape, f"{what}{k}: shape {tuple(got[k].shape)}, golden {want[k].shape}"
    if TASK[ref] == "features":  # hidden states are not compared below: at least no fp16 overflow
        assert all(bool(torch.isfinite(v).all()) for v in got.values()), f"{what}non-finite features"
    key = primary(ref, want)
    tol = tolerance(ref, mode)
    if "pairs" in want:  # sparse matchers: the distances of the recalled keypoints
        recall, agree, err, _ = sparse_parity(want, got)
        assert recall >= tol["gpu_min_recall"], f"{what}keypoints: recall {recall:.4f}"
        assert agree >= tol["gpu_min_match"], f"{what}matches: agreement {agree:.4f}"
        assert err.mean() < tol["gpu_mean"], f"{what}keypoints: mean |diff| {err.mean():.4f}"
        assert np.percentile(err, 99) < tol["gpu_p99"], f"{what}keypoints: p99 |diff| {np.percentile(err, 99):.4f}"
        return err
    unit = np.median(np.abs(want[key])) if tol.get("relative") else 1.0  # scale-free outputs (relative depth)
    err = np.abs(got[key].numpy() - want[key]) / unit
    if "gpu_min_score" in tol:  # unsure points: pose argmax ties, matching pixels outside the overlap
        sure = sure_mask(want[SCORE[TASK[ref]]] >= tol["gpu_min_score"], err.shape)
        assert sure.mean() >= MIN_SURE, f"{what}{key}: the score mask keeps {sure.mean():.0%} of the points"
        err = err[sure]
    assert err.mean() < tol["gpu_mean"], f"{what}{key}: mean |diff| {err.mean():.4f}"
    assert np.percentile(err, 99) < tol["gpu_p99"], f"{what}{key}: p99 |diff| {np.percentile(err, 99):.4f}"
    rows = tol.get("gpu_extra", [])
    assert isinstance(rows, list)
    extra = {k: (mean, p99) for k, mean, p99 in rows}
    if extra:  # [name, mean, p99] of |diff| / median of the nonzero |reference| values; family-wide, so a superset
        assert set(want) - {key} <= set(extra), f"gpu_extra covers {sorted(extra)}, the golden {sorted(want)}"
    for k, (mean, p99) in ((k, v) for k, v in extra.items() if k in want):
        nz = np.abs(want[k])[want[k] != 0]
        e = np.abs(got[k].numpy() - want[k]) / (np.median(nz) if nz.size else 1.0)  # all-zero reference: absolute
        assert e.mean() < mean, f"{what}{k}: mean |diff| {e.mean():.4f}"
        assert np.percentile(e, 99) < p99, f"{what}{k}: p99 |diff| {np.percentile(e, 99):.4f}"
    return err


def gpu_built(ref: str, mode: str) -> bool:
    """Kernels on `cudnn_hacc`: the GPU check also runs a call with every plan built (`run(..., built=True)`)."""
    return mode == "kernels" and "cudnn_hacc" in API[TASK[ref]].model_card(ref).kernel_libs


def gpu_parity(
    ref: str, mode: str, want: dict[str, np.ndarray], calls: list[dict[str, torch.Tensor]], backend: Any
) -> np.ndarray:
    """The GPU golden check of a model's `calls` (`run(..., built=...)`, fallbacks reset before them): the backend is
    the mode asked for, the kernel libraries handed no work to PyTorch, every call is within the tolerances. The error
    map of the last call's primary output. tools/merge_check.py runs it on the model it then times."""
    want = {k: v for k, v in want.items() if k not in INPUTS}
    assert backend.mode == mode
    assert bool(backend.kernels) == (mode == "kernels")
    assert kernels.fallbacks() == {}, "kernel libraries handed work to PyTorch"
    err = np.zeros(0)
    for i, got in enumerate(calls):  # kernels on cudnn_hacc: the first call of the size, then every plan built
        err = _check(ref, mode, want, got, f"call {i + 1}: ")
    return err


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("ref", sorted(GOLDEN))
def test_gpu(ref, mode, request, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    want = golden(ref)
    if mode not in API[TASK[ref]].model_card(ref).modes:
        pytest.skip(f"{ref} has no {mode}")
    kernels.reset_fallbacks()
    launched: set[str] = set()
    used_libraries: set[str] = set()
    record = mode == "kernels" and request.config.getoption("--record")
    if mode == "kernels":  # test-only hooks: actual calls, including model loading and CUDA graph capture
        from fastforks.kernels.cudnn_hacc import _backend  # noqa: PLC0415 - cuDNN is not a CubinModule

        real = kernels.CubinModule.launch
        execute = _backend.execute

        def spy(self, kernel, *a, **kw):
            library = self.tag.split("@")[0]
            launched.add(f"{library}:{kernel}")
            used_libraries.add(library)
            return real(self, kernel, *a, **kw)

        def spy_cudnn(*a, **kw):
            used_libraries.add("cudnn_hacc")
            return execute(*a, **kw)

        monkeypatch.setattr(kernels.CubinModule, "launch", spy)
        monkeypatch.setattr(_backend, "execute", spy_cudnn)
    calls, backend = run_or_skip(
        ref, mode=mode, device="cuda", opts=options(ref, want), want=want, built=gpu_built(ref, mode)
    )
    err = gpu_parity(ref, mode, want, calls, backend)
    _declared_launches(ref, set(API[TASK[ref]].model_card(ref).kernel_libs), used_libraries)
    if record:
        symbols = _used_symbols(ref)
        for lib in {x.split(":")[0] for x in launched if x.split(":")[1] not in symbols.get(x.split(":")[0], {})}:
            # launched through a library wrapper or an f-string the scan cannot see: count the whole library; a launched
            # function without a manifest hash (e.g. a macro-generated export) drops the library's hashes, so its exact
            # build stays required (test_kernels_validated)
            m = _manifest(lib)
            hashed = m is not None and all(x[len(lib) + 1 :] in m.symbols for x in launched if x.startswith(f"{lib}:"))
            if m is not None and hashed:
                symbols[lib] = dict(m.symbols)
            else:
                symbols.pop(lib, None)
        store = goldens.load(FAMILY[ref], TASK[ref])
        store["validated"][ref] = {
            "kernels": sorted(backend.kernels),
            "symbols": symbols,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "date": dt.datetime.now(dt.UTC).date().isoformat(),
            "mean_px": round(float(err.mean()), 5),
            "p99_px": round(float(np.percentile(err, 99)), 5),
        }
        goldens.dump(FAMILY[ref], store, TASK[ref])


def _declared_launches(ref: str, declared: set[str], observed: set[str]) -> None:
    """Reject executed libraries that the card cannot fetch or validate."""
    assert observed <= declared, f"{ref}: launched undeclared kernel libraries: {sorted(observed - declared)}"


@pytest.mark.model
@pytest.mark.mps
@pytest.mark.parametrize("ref", sorted(GOLDEN))
def test_mps(ref):
    """Portable on Apple MPS, in every dtype the card validates there, and the Metal kernels backend if the family has
    one, within the GPU tolerances of the golden."""
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    card = API[TASK[ref]].model_card(ref)
    if not card.portable_mps_dtypes and not card.mps_kernel_libs:
        pytest.skip(f"{ref} is not validated on MPS")
    want = golden(ref)
    runs = [("portable", d) for d in card.portable_mps_dtypes]
    runs += [("kernels", card.kernels_dtype_on(torch.device("mps")))] if card.mps_kernel_libs else []
    for mode, dtype in runs:
        (got,), backend = run_or_skip(ref, device="mps", mode=mode, dtype=dtype, opts=options(ref, want), want=want)
        assert (backend.mode, str(backend.dtype)) == (mode, dtype)
        assert len(backend.kernels) == len(card.mps_kernel_libs) if mode == "kernels" else not backend.kernels
        _check(ref, mode, {k: v for k, v in want.items() if k not in INPUTS}, got, f"{mode} {dtype}: ")


def _manifest(lib: str) -> kernels.KernelManifest | None:
    if lib == "cudnn_hacc":  # its binary only builds cuDNN's plans on the host (the ctypes path's plans, see
        return None  # test_native_build_gives_the_ctypes_plan): validated per cuDNN version, like its tag
    path = Path(kernels.__file__).parents[1] / "kernels" / lib / "manifest.toml"
    return kernels.read_manifest(path) if path.is_file() else None  # None: cuDNN libraries (tagged by version)


def _used_symbols(ref: str) -> dict[str, dict[str, str]]:
    """`{lib: {symbol: hash}}` of the kernel functions the family's code names (quoted) in its own files and in the
    `shared` modules it uses (a library the scan cannot see into, e.g. launched by a wrapper, counts as fully used);
    the launches seen while recording are checked to be inside this set."""
    card = API[TASK[ref]].model_card(ref)
    src = Path(kernels.__file__).parents[1]
    files = list((src / TASK[ref] / "models" / FAMILY[ref]).rglob("*.py"))
    for name in card.uses:
        mod = src / "shared" / name
        files += list(mod.rglob("*.py")) if mod.is_dir() else [mod.with_suffix(".py")]
    quoted = set(re.findall(r"""["'](\w+)["']""", "\n".join(f.read_text() for f in files if f.is_file())))
    used = {}
    for lib in card.kernel_libs:
        m = _manifest(lib)
        if m is not None:
            names = {n for n in m.symbols if n in quoted} or set(m.symbols)  # not named in the family: all of them
            used[lib] = {n: m.symbols[n] for n in names}
    return used


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("ref", sorted(r for r in GOLDEN if TASK[r] == "classification"))
def test_gpu_module(ref):
    """Portable `module()` runs without the fp16-accumulate switch of the call: both stay within the GPU tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    want = golden(ref)["logits"]
    m = classification.load(ref, device="cuda")
    try:
        module = m.module()
    except BackendUnavailableError as e:
        pytest.skip(str(e))
    io = m.variant.io
    with torch.inference_mode():
        got = module(scale(resize_crop(to_rgb_batch(image(), m.device), io), io))[0].float().cpu().numpy()
    tol = tolerance(ref)
    for name, ref_logits in (("golden", want), ("call", m(image()).logits.cpu().numpy())):
        err = np.abs(got - ref_logits)
        assert err.mean() < tol["gpu_mean"], f"module vs {name}: mean |diff| {err.mean():.4f}"
        assert np.percentile(err, 99) < tol["gpu_p99"], f"module vs {name}: p99 |diff| {np.percentile(err, 99):.4f}"


def _built(tag: str) -> str:
    """`lib@build:target` -> `lib@build`."""
    return tag.rsplit(":", 1)[0]


@pytest.mark.parametrize("ref", sorted(GOLDEN))
def test_kernels_validated(ref):
    """Every kernels variant was GPU-validated with the kernel builds in the current manifests (no GPU needed)."""
    card = API[TASK[ref]].model_card(ref)
    if "kernels" not in card.modes:
        return
    entry = FAMILIES[FAMILY[ref]]["validated"].get(ref)
    assert entry, f"{ref}: max never validated; run the GPU goldens with --record"
    recorded = {_built(t) for t in entry["kernels"]}
    current = _used_symbols(ref)
    for lib in card.kernel_libs:
        manifest = _manifest(lib)
        if manifest is None:
            continue
        tag = manifest.tag
        old = entry.get("symbols", {}).get(lib)
        if tag not in recorded and old is not None:  # another build: valid if every function in use is unchanged
            assert set(current[lib]) <= set(old), f"{ref}: {lib} uses unrecorded functions; rerun --record"
            changed = sorted(n for n in current[lib] if current[lib][n] != old[n])
            assert not changed, f"{ref}: {lib} functions changed since validation: {changed}; rerun --record"
            continue
        assert tag in recorded, f"{ref}: {tag} not validated (recorded {sorted(recorded)}); rerun GPU goldens --record"


def test_every_variant_has_a_golden():
    refs = {v.ref for task in registry.TASKS for card, v in registry.cards(task).values()}
    assert refs == set(GOLDEN), f"no golden: {sorted(refs - set(GOLDEN))}; stale: {sorted(set(GOLDEN) - refs)}"
    for ref, fam in FAMILY.items():
        assert API[TASK[ref]].model_card(ref).family == fam, f"{ref} is in {fam}/golden.toml"


def test_dump_writes_valid_toml(tmp_path, monkeypatch):
    """`--record` stores `symbols` (nested dicts); the file must stay valid TOML and load back unchanged."""
    monkeypatch.setattr(goldens, "SRC", tmp_path)
    (tmp_path / "stereo" / "models" / "fam").mkdir(parents=True)
    entry = {"kernels": ["lib@abc:sm_120"], "symbols": {"lib": {"k2": "22", "k1": "11"}}, "mean_px": 0.01}
    store = {"tolerance": dict(goldens.DEFAULT_TOLERANCE), "refs": {}, "validated": {"fam-a@1": entry}}
    goldens.dump("fam", store)
    assert goldens.load("fam") == store


def test_min_score_tasks():
    """`gpu_min_score` only in tasks with a confidence output, which masks both output layouts along the right axis."""
    for ref in GOLDEN:
        if "gpu_min_score" in tolerance(ref):
            assert TASK[ref] in SCORE, f"{ref}: gpu_min_score, but {TASK[ref]} has no confidence output in SCORE"
            assert SCORE[TASK[ref]] in spec(TASK[ref]).standard, f"{ref}: {SCORE[TASK[ref]]} is no {TASK[ref]} output"
    pose = sure_mask(np.array([[True, False, True]]), (1, 3, 2))  # keypoint scores [P, K] -> keypoints [P, K, 2]
    assert pose[0, :, 0].tolist() == pose[0, :, 1].tolist() == [True, False, True]
    match = sure_mask(np.array([[[True, False]]]), (1, 2, 1, 2))  # certainty [B, H, W] -> warp [B, 2, H, W]
    assert match[0, :, 0].tolist() == [[True, False], [True, False]]
