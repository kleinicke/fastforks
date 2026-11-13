"""Kernels backends on a stream of input sizes in one process (GPU): every family with a kernels backend.

The sizes grow, repeat, shrink after the growth and include odd ones, with graphs captured on a shape's second call:
cached workspaces, plans and buffers must not outlive or alias what a captured graph or a later size uses (the
accuracy campaign's ETH3D stream crashed Fast-ACVNet, IGEV++ and RT-MonSter++ with an illegal address). Every call
must give finite outputs, and a repeated size (a graph replay) the output of its first call.
`FASTFORKS_SIZE_STREAM_LARGE=1` adds a ~1000 x 1500 call (stereo) after the growth for every family, not only the
cheap ones.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from fastforks import (
    classification,
    depth,
    detection,
    features,
    flow,
    interpolation,
    matching,
    matting,
    multiview,
    pose,
    promptable_segmentation,
    retrieval,
    segmentation,
    stereo,
)
from fastforks._core import registry
from fastforks._core.errors import WeightsError
from fastforks.kernels import cudnn_hacc
from synth import clip, image, pair, prompts, scene, size, track_queries

pytestmark = [pytest.mark.model, pytest.mark.gpu]

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
    "matting": matting,
    "interpolation": interpolation,
    "retrieval": retrieval,
}
OUTPUTS = (  # the outputs checked, where a result has them
    "disparity",
    "flow",
    "frame",
    "warp",
    "certainty",
    "depth",
    "inverse_depth",
    "points",
    "logits",
    "last_hidden_state",
    "pred_masks",
    "iou_scores",
    "scores",
    "boxes",
    "keypoints",
    "keypoint_scores",
    "alpha",
    "descriptor",
)
A, B, C = (243, 333), (307, 451), (259, 397)  # odd sizes off every tile; B > A, both small enough for graphs
LARGE = (1001, 1497)
VIEWS = {A: (120, 160), B: (144, 200), C: (132, 180), LARGE: (240, 320)}  # multiview scenes (4 views)
CHEAP_LARGE = {"CoEx", "Fast-ACVNet", "HITNet", "LiteAnyStereo V2", "S2M2"}  # stereo families with a cheap large call


def _smallest() -> list[tuple[str, str]]:
    """`(task, ref)`: the smallest variant (weights) of every family with a kernels backend."""
    refs = []
    for task in API:
        fams: dict[str, list[tuple[int, str]]] = {}
        for ref, (card, variant) in registry.cards(task).items():
            if "kernels" in card.modes:
                fams.setdefault(card.name, []).append((sum(w.size or 0 for w in variant.weights), ref))
        refs += [(task, min(r)[1]) for _, r in sorted(fams.items())]
    return refs


REFS = _smallest()


def _call(m: Any, task: str, hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    """One call at `hw`: the checked outputs (float tensors)."""
    if task in ("stereo", "flow", "matching", "interpolation"):
        r = m(*pair(*size(m.variant.io.min_size, *hw)))
    elif task == "tracking":
        r = m(clip(4, *hw), track_queries())
    elif task == "multiview":
        s = scene(h=VIEWS[hw][0], w=VIEWS[hw][1])
        r = m(list(s["images"]), **{k: s[k] for k in ("intrinsics", "cam_to_world", "depth_range")
                                     if getattr(m._runner, k) == "required"})  # fmt: skip
    elif task == "promptable_segmentation":
        r = m(image(*hw), text="coin") if "text" in m.prompts else m(image(*hw), **prompts()[""])
    elif task == "segmentation":
        r = m(image(*hw), logits=True)
    elif task == "pose":
        h, w = hw
        r = m(image(*hw), boxes=[[w * 0.1, h * 0.1, w * 0.5, h * 0.9], [w * 0.5, h * 0.2, w * 0.9, h * 0.8]])
    else:
        r = m(image(*hw))
    out = {k: getattr(r, k, None) for k in OUTPUTS}
    return {k: v.float() for k, v in out.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}


@pytest.mark.parametrize(("task", "ref"), REFS, ids=[r for _, r in REFS])
def test_size_stream(task: str, ref: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 0)  # plans on a geometry's first call: replays == eager calls
    monkeypatch.setattr(cudnn_hacc, "CAPTURE_AFTER", 1)  # graphs on a shape's second call
    try:
        m = API[task].load(ref, mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")
    large = task == "stereo" and (
        API[task].model_card(ref).name in CHEAP_LARGE or os.environ.get("FASTFORKS_SIZE_STREAM_LARGE")
    )
    sizes = [A, B, A, B, *([LARGE] if large else []), A, C, B]  # eager, eager, capture, capture, ..., replays
    first: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for i, hw in enumerate(sizes):
        out = _call(m, task, hw)
        torch.cuda.synchronize()
        assert out, f"{ref}: no checked output"
        for k, v in out.items():
            assert bool(torch.isfinite(v).all()), f"{ref} call {i} at {hw}: non-finite {k}"
        ref_out = first.setdefault(hw, out)
        for k, v in out.items():
            w = ref_out[k]
            if v.shape == w.shape and v.numel():  # a detector may keep another number of instances, or none
                torch.testing.assert_close(v, w, rtol=0, atol=1e-3 + 1e-2 * w.abs().max().item(), msg=f"{ref} {k}")
