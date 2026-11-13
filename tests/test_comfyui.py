"""The ComfyUI node pack (integrations/comfyui): tensor conversions, prompt parsing, labels and the unload hook.

Runs without ComfyUI: `comfy.model_management` is a stub. Model runs are checked against direct API calls in a
real ComfyUI (results/2026-11-13_comfyui_nodes.md).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

PACK = Path(__file__).resolve().parents[1] / "integrations" / "comfyui"


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The pack as a package (`comfyui_fastforks`), imported with a stub `comfy.model_management`."""
    calls: list[str] = []
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: calls.append("comfy")
    mm.soft_empty_cache = lambda: calls.append("empty")
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    fp = types.ModuleType("folder_paths")
    fp.models_dir = str(tmp_path_factory.mktemp("models"))
    fp.paths = {}
    fp.add_model_folder_path = lambda name, path: fp.paths.setdefault(name, []).append(path)
    fp.get_folder_paths = lambda name: fp.paths[name]
    names = ("comfy", "comfy.model_management", "folder_paths")
    saved = {k: sys.modules.get(k) for k in names}
    env = dict(os.environ)
    sys.modules |= {"comfy": comfy, "comfy.model_management": mm, "folder_paths": fp}
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
        module.fp = fp
        yield module
    finally:
        os.environ.clear()
        os.environ.update(env)  # the pack points FASTFORKS_CACHE at ComfyUI's models folder
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


def test_cache_setup(pack: Any, tmp_path: Path) -> None:
    """Downloads go to models/fastforks; the normal cache and extra model folders are read in place."""
    first, extra, normal = tmp_path / "first", tmp_path / "extra", tmp_path / "normal"
    for d in (extra, normal):
        d.mkdir()
    pack.fp.paths["fastforks"] = [str(first), str(extra), str(tmp_path / "missing")]
    environ = {"FASTFORKS_CACHE": str(normal)}
    pack.nodes.setup_cache(environ)
    assert environ["FASTFORKS_CACHE"] == str(first)
    assert environ["FASTFORKS_READ_CACHES"].split(os.pathsep) == [str(extra), str(normal)]
    assert pack.fp.paths["fastforks"][-1] == str(Path(pack.fp.models_dir) / "fastforks")


def test_semantic_maps(pack: Any) -> None:
    """Class colours (VOC colormap), class names to ids, label map to colours."""
    c = pack.convert
    pal = c.palette(4)
    assert pal[0].tolist() == [0, 0, 0]
    assert (pal[1] * 255).round().tolist() == [128, 0, 0]
    assert (pal[2] * 255).round().tolist() == [0, 128, 0]
    assert c.class_ids(" Sky, wall", ("wall", "sky")) == [1, 0]
    assert c.class_ids("", ("wall",)) == []
    with pytest.raises(ValueError, match="unknown classes"):
        c.class_ids("cat", ("wall",))
    img = c.labels_image(torch.tensor([[0, 2]]), pal)
    assert img.shape == (3, 1, 2)
    assert img[:, 0, 1].tolist() == pal[2].tolist()


def test_flow_image_and_warp(pack: Any) -> None:
    """No motion is white, the fastest rightward motion red; warping by a shift of +1 px reads the next column."""
    c = pack.convert
    flow = torch.zeros(2, 2, 3)
    flow[0, 0, 0] = 4.0
    img = c.flow_image(flow)
    assert img.shape == (3, 2, 3)
    assert img[:, 1, 1].tolist() == [1, 1, 1]
    assert img[0, 0, 0] == 1
    assert img[1, 0, 0] == 0
    assert c.flow_image(torch.full((2, 1, 1), float("nan"))).tolist() == [[[1.0]], [[1.0]], [[1.0]]]
    image = torch.arange(6.0).reshape(1, 2, 3, 1)
    shift = torch.zeros(1, 2, 3, 2)
    shift[..., 0] = 1
    out = c.warp(image, shift)
    assert out[0, :, :2, 0].tolist() == [[1, 2], [4, 5]]
    assert out[0, :, 2, 0].abs().max() < 0.6  # half outside: blended with the black border


def test_tracks(pack: Any) -> None:
    """Grid queries at cell centres; visible points are drawn in their colour, hidden ones are not."""
    c = pack.convert
    assert c.grid_points(2, (4, 8)) == [[2.0, 1.0], [6.0, 1.0], [2.0, 3.0], [6.0, 3.0]]
    tracks = torch.tensor([[[2.0, 2.0], [12.0, 12.0]], [[4.0, 2.0], [12.0, 12.0]]])
    visible = torch.tensor([[True, True], [True, False]])
    out = c.draw_tracks(torch.zeros(3, 16, 16), tracks, visible, 1, trail=4, width=1)
    assert out.shape == (3, 16, 16)
    assert out[:, 2, 4].tolist() == [1, 0, 0]  # point 0, red
    assert out[:, 12, 12].sum() == 0  # point 1 hidden in frame 1


def test_upscale_files(pack: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """models/upscale_models lists only files fastforks runs; the node output equals `restoration.load_file`."""
    from fastforks import restoration  # noqa: PLC0415
    from fastforks.restoration.models.span.net import SPAN  # noqa: PLC0415

    torch.manual_seed(0)
    torch.save(SPAN(3, 3, feature_channels=8, upscale=2).state_dict(), tmp_path / "2x-span.pth")
    torch.save({"body.0.weight": torch.zeros(8, 3, 3, 3)}, tmp_path / "compact.pth")  # SRVGG-like: not runnable
    fp, mm, nodes = pack.fp, pack.mm, pack.nodes
    monkeypatch.setattr(fp, "get_filename_list", lambda _: sorted(p.name for p in tmp_path.iterdir()), raising=False)
    monkeypatch.setattr(fp, "get_full_path", lambda _, name: str(tmp_path / name), raising=False)
    monkeypatch.setattr(fp, "get_full_path_or_raise", lambda _, name: str(tmp_path / name), raising=False)
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = lambda _: types.SimpleNamespace(update=lambda _: None)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setattr(sys.modules["comfy"], "utils", utils, raising=False)
    for name, value in {
        "get_torch_device": lambda: torch.device("cpu"), "intermediate_device": lambda: torch.device("cpu"),
        "free_memory": lambda *_: None, "throw_exception_if_processing_interrupted": lambda: None,
        "InterruptProcessingException": type("Interrupt", (Exception,), {}),
    }.items():  # fmt: skip
        monkeypatch.setattr(mm, name, value, raising=False)

    assert nodes.upscale_files() == ["file: 2x-span.pth"]
    assert nodes.model_id("file: 2x-span.pth") == "file: 2x-span.pth"
    inputs = nodes.Upscale.INPUT_TYPES()["required"]
    assert "file: 2x-span.pth" in inputs["model"][0]
    assert "realesrgan-x4plus" in inputs["model"][0]

    image = torch.rand(2, 16, 24, 3)
    (out,) = nodes.Upscale().run(image=image, model="file: 2x-span.pth", backend="portable", dtype="float32", tile=0)
    m = restoration.load_file(tmp_path / "2x-span.pth", mode="portable", device="cpu", dtype="float32")
    assert out.shape == (2, 32, 48, 3)
    for i in range(2):
        want = m(image[i].permute(2, 0, 1)).image.float().permute(1, 2, 0).clamp(0, 1)
        torch.testing.assert_close(out[i], want, rtol=0, atol=0)
    nodes._cache.clear()


def test_node_model_lists(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each node lists the ids of its families (cards only, nothing loads)."""
    nodes = pack.nodes
    monkeypatch.setattr(nodes, "upscale_files", list)

    def ids(node: type) -> set[str]:
        return {nodes.model_id(e) for e in node.INPUT_TYPES()["required"]["model"][0]}

    assert {"hat-x4", "hat-s-x4", "real-hat-gan-x4", "dat-x4", "dat-2-x4", "dat-light-x4"} <= ids(nodes.Upscale)
    tagger = {"wd-vit-tagger-v3", "wd-convnext-tagger-v3", "wd-vit-large-tagger-v3", "wd-eva02-large-tagger-v3"}
    assert ids(nodes.Tag) == tagger
    assert ids(nodes.NsfwCheck) == {"falconsai-nsfw-vit"}
    assert ids(nodes.Parse) == {"segformer-b2-clothes", "segformer-b5-face-parsing"}
    assert {"segformer-b2-clothes", "segnext-mscan-t-ade20k"} <= ids(nodes.SemanticSegment)


def test_wd_tags(pack: Any) -> None:
    """Ratings never; general and character tags each by their threshold, most likely first."""
    names = ("general", "sensitive", "questionable", "explicit", *[f"g{i}" for i in range(4, 8110)], "char_a", "char_b")
    probs = torch.zeros(len(names))
    probs[0], probs[4], probs[5], probs[8110], probs[8111] = 0.99, 0.4, 0.9, 0.8, 0.5
    assert pack.nodes.wd_tags(probs, names, 0.35, 0.75) == "g5, char_a, g4"
    assert pack.nodes.wd_tags(probs, names, 0.95, 0.95) == ""
