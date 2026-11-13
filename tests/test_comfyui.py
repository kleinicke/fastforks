"""The ComfyUI node pack (integrations/comfyui): tensor conversions, prompt parsing, labels and the unload hook.

Runs without ComfyUI: `comfy.model_management` is a stub. Model runs are checked against direct API calls in a
real ComfyUI (results/2026-11-13_comfyui_nodes.md).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

PACK = Path(__file__).resolve().parents[1] / "integrations" / "comfyui"


@pytest.fixture(scope="module")
def pack() -> Any:
    """The pack as a package (`comfyui_fastforks`), imported with a stub `comfy.model_management`."""
    calls: list[str] = []
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: calls.append("comfy")
    mm.soft_empty_cache = lambda: calls.append("empty")
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    saved = {k: sys.modules.get(k) for k in ("comfy", "comfy.model_management")}
    sys.modules |= {"comfy": comfy, "comfy.model_management": mm}
    try:
        spec = importlib.util.spec_from_file_location(
            "comfyui_fastforks", PACK / "__init__.py", submodule_search_locations=[str(PACK)]
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_fastforks"] = module
        spec.loader.exec_module(module)
        module.calls = calls
        module.mm = mm
        yield module
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k in [k for k in sys.modules if k.startswith("comfyui_fastforks")]:
            del sys.modules[k]


def test_image_round_trip(pack: Any) -> None:
    """IMAGE [B, H, W, C] -> fastforks [3, H, W] -> IMAGE is exact; alpha is dropped, gray becomes RGB."""
    c = pack.convert
    img = torch.rand(2, 5, 7, 3)
    frames = [c.frame(img, i, torch.device("cpu")) for i in range(2)]
    assert frames[0].shape == (3, 5, 7)
    assert torch.equal(c.to_image(frames), img)
    rgba = torch.rand(1, 5, 7, 4)
    assert torch.equal(c.frame(rgba, 0, torch.device("cpu")), rgba[0, ..., :3].permute(2, 0, 1))
    gray = torch.rand(1, 5, 7, 1)
    assert torch.equal(c.frame(gray, 0, torch.device("cpu"))[2], gray[0, ..., 0])


def test_masks(pack: Any) -> None:
    """Masks: bool and float maps stack to MASK; RGBA carries the mask; one unbatched trimap serves every image."""
    c = pack.convert
    mask = c.to_mask([torch.ones(4, 6, dtype=torch.bool), torch.full((4, 6), 0.25)])
    assert mask.shape == (2, 4, 6)
    assert mask.dtype == torch.float32
    rgba = c.with_alpha(torch.rand(2, 4, 6, 3), mask)
    assert rgba.shape == (2, 4, 6, 4)
    assert torch.equal(rgba[..., 3], mask)
    assert torch.equal(c.from_mask(mask, 5, torch.device("cpu")), mask[1])
    assert torch.equal(c.from_mask(mask[0], 1, torch.device("cpu")), mask[0])
    boxes = c.union_mask(None, torch.tensor([[1.0, 0.0, 3.0, 2.0]]), (4, 6))
    assert boxes.sum() == 4
    assert boxes[0:2, 1:3].all()
    assert c.union_mask(torch.zeros(0, 4, 6, dtype=torch.bool), torch.zeros(0, 4), (4, 6)).shape == (4, 6)


def test_depth_and_normals(pack: Any) -> None:
    """Depth: near is white, invalid is black, scaled per image; normals: facing the camera is (0.5, 0.5, 1)."""
    c = pack.convert
    depth = torch.tensor([[1.0, 2.0], [4.0, float("inf")]])
    img = c.depth_image(depth, None)
    assert img.shape == (3, 2, 2)
    assert img[0, 0, 0] == 1
    assert img[0, 1, 1] == 0
    inv = c.depth_image(None, torch.tensor([[0.0, 10.0]]))
    assert inv[:, 0, 1].tolist() == [1, 1, 1]
    facing = torch.tensor([0.0, 0.0, -1.0]).expand(2, 3, 3)
    assert torch.allclose(c.normals_image(facing)[:, 0, 0], torch.tensor([0.5, 0.5, 1.0]))
    up = torch.tensor([0.0, -1.0, 0.0]).expand(1, 1, 3)  # y down: -y is up -> green
    assert torch.allclose(c.normals_image(up)[:, 0, 0], torch.tensor([0.5, 1.0, 0.5]))


def test_prompt_parsing(pack: Any) -> None:
    """Points, boxes and detection JSON as typed into STRING widgets."""
    c = pack.convert
    assert c.points("") == []
    assert c.points("10, 20; 30,40") == [[10.0, 20.0], [30.0, 40.0]]
    assert c.points("[[1, 2]]") == [[1.0, 2.0]]
    assert c.box(" 1,2,3,4 ") == [1.0, 2.0, 3.0, 4.0]
    assert c.box("") is None
    with pytest.raises(ValueError, match="points"):
        c.points("1,2,3")
    with pytest.raises(ValueError, match="box"):
        c.box("1,2,3")
    dets = json.dumps([[{"box": [0, 0, 5, 5], "class_name": "dog"}, {"box": [1, 1, 2, 2], "class_name": "person"}]])
    assert c.person_boxes(dets, 3) == [[1.0, 1.0, 2.0, 2.0]]
    assert c.person_boxes("", 0) is None
    assert json.loads(c.to_json([{"a": torch.tensor([1.5])}])) == [{"a": [1.5]}]


def test_draw_pose(pack: Any) -> None:
    """Limbs above the threshold are drawn in colour, low-score keypoints are not drawn."""
    kp = torch.tensor([[[2.0, 2.0], [12.0, 2.0], [7.0, 12.0]]])
    out = pack.convert.draw_pose(
        kp, torch.tensor([[1.0, 1.0, 0.0]]), [(0, 1), (1, 2)], (16, 16), threshold=0.5, width=1
    )
    assert out.shape == (3, 16, 16)
    assert out[0, 2, 7] == 1  # limb 0 is red
    assert out[:, 12, 7].sum() == 0  # keypoint 2 scores 0


def test_labels_and_unload_hook(pack: Any) -> None:
    """Non-commercial ids carry a tag the node strips; ComfyUI's unload also empties our cache, wrapped once."""
    nodes = pack.nodes
    nc = next(iter(pack.nodes.fastforks.list_models(commercial="no")))
    label = nodes.labels([nc])[0]
    assert label == nc + " [non-commercial]"
    assert nodes.model_id(label) == nc
    assert nodes.labels(["depth-anything-v2-s"]) == ["depth-anything-v2-s"]
    nodes.install_unload_hook()  # second install is a no-op
    nodes._cache["x"] = object()
    pack.calls.clear()
    pack.mm.unload_all_models()
    assert pack.calls == ["comfy", "empty"]
    assert not nodes._cache
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
