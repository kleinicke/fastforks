"""The ComfyUI node packs (integrations/comfyui, integrations/comfyui-commercial): conversions, lists, licences.

Runs without ComfyUI: `comfy.model_management` is a stub. Model runs are checked against direct API calls in a
real ComfyUI (see docs/plans/comfyui-windows-results.md).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

PACK = Path(__file__).resolve().parents[1] / "integrations" / "comfyui"
COMMERCIAL_PACK = PACK.parent / "comfyui-commercial"


def _import(name: str, folder: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, folder / "__init__.py", submodule_search_locations=[str(folder)]
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The pack as a package (`comfyui_fastforks`), imported with a stub `comfy.model_management`.

    `pack.commercial` is fastforks-commercial (`comfyui_fastforks_commercial`), imported after it (it is not in a
    custom_nodes folder, so it registers its nodes next to the fastforks pack's).
    """
    calls: list[str] = []
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: calls.append("comfy")
    mm.soft_empty_cache = lambda: calls.append("empty")
    mm.free_memory = lambda *_, **__: []
    mm.get_free_memory = lambda *_, **__: float("inf")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.intermediate_device = lambda: torch.device("cpu")
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
        module = _import("comfyui_fastforks", PACK)
        module.commercial = _import("comfyui_fastforks_commercial", COMMERCIAL_PACK)
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
        for k in [k for k in sys.modules if k.startswith(("comfyui_fastforks", "fastforks_comfyui"))]:
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


def test_stereo_scale_ignores_outliers(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stereo: a few huge disparities saturate instead of squashing the rest dark; Depth keeps min / max."""
    disp = torch.linspace(10.0, 20.0, 1000).reshape(20, 50)
    disp[0, :5] = 500.0  # 0.5 % mismatches
    plain = pack.convert.depth_image(None, disp)
    assert plain[0].median() < 0.05  # min / max: everything but the outliers is near black
    robust = pack.convert.depth_image(None, disp, clip=0.01)
    assert robust[0].median() == pytest.approx(0.5, abs=0.02)
    assert robust[0, 0, :5].tolist() == [1.0] * 5
    assert robust.min() == 0
    assert torch.equal(pack.convert.depth_image(None, disp, clip=0.0001), plain)  # under one pixel: min / max
    nodes = pack.nodes

    class Pair:
        device = torch.device("cpu")

        def __call__(self, left: torch.Tensor, right: torch.Tensor) -> types.SimpleNamespace:
            assert left.shape == right.shape == (3, 20, 50)
            return types.SimpleNamespace(disparity=disp)

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Pair())
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])  # no comfy.utils progress bar
    img = torch.rand(1, 20, 50, 3)
    (out,) = nodes.Stereo().run(left=img, right=img, model="m", backend="auto", dtype="auto")
    assert torch.equal(out[0, ..., 0], robust[0])


def test_normals_node_turns_on_moges_normal_head(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """MoGe returns `normals=None` unless loaded with `normals=True`; the other normals models take no such option."""
    nodes = pack.nodes
    loaded: list[tuple[str, Any]] = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, _image: torch.Tensor) -> types.SimpleNamespace:
            return types.SimpleNamespace(normals=torch.zeros(4, 5, 3))

    def fake_load(label: str, backend: str, dtype: str, options: Any = None) -> Model:
        loaded.append((label, options))
        return Model()

    monkeypatch.setattr(nodes, "load", fake_load)
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])
    img = torch.rand(1, 4, 5, 3)
    for mid in ("moge-2-vitb-normal", "lotus-normal-d-v1-1"):
        label = next(label for label in nodes.Normals.INPUT_TYPES()["required"]["model"][0] if mid in label)
        nodes.Normals().run(image=img, model=label, backend="auto", dtype="auto")
    assert [o for _, o in loaded] == [{"normals": True}, None]


def test_batch_size_gives_the_per_image_outputs(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """`batch_size` > 1 calls the model on chunks of the batch (last one shorter), outputs as one at a time."""
    nodes = pack.nodes
    shapes: list[tuple[int, ...]] = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, x: torch.Tensor) -> types.SimpleNamespace:
            shapes.append(tuple(x.shape))
            return types.SimpleNamespace(depth=None, inverse_depth=x.sum(-3))  # [H, W] or [B, H, W]

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Model())
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])
    img = torch.rand(5, 4, 6, 3)
    outs = [nodes.Depth().run(image=img, model="m", backend="auto", dtype="auto", batch_size=b)[0] for b in (1, 2, 5)]
    assert shapes == [(3, 4, 6)] * 5 + [(2, 3, 4, 6), (2, 3, 4, 6), (1, 3, 4, 6), (5, 3, 4, 6)]
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0], outs[2])
    assert "batch_size" in nodes.Depth.INPUT_TYPES()["optional"]


def test_detect_batch_size_passes_the_call_options(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect with `batch_size` > 1: chunks of the batch with the node's threshold, outputs as one at a time."""
    nodes = pack.nodes
    calls: list[tuple[tuple[int, ...], float]] = []

    class Result:
        masks = None
        boxes = torch.tensor([[1.0, 1.0, 3.0, 2.0]])

        def to_list(self) -> list[dict[str, Any]]:
            return [{"box": [1.0, 1.0, 3.0, 2.0], "score": 0.9, "class_name": "cat"}]

    class Model:
        device = torch.device("cpu")
        open_vocabulary = False

        def __call__(self, x: torch.Tensor, *, score_threshold: float, text: Any) -> Any:
            calls.append((tuple(x.shape), score_threshold))
            assert text is None  # closed vocabulary
            return Result() if x.ndim == 3 else [Result() for _ in range(x.shape[0])]

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Model())
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])
    img = torch.rand(3, 4, 6, 3)
    kw = {"image": img, "model": "m", "backend": "auto", "dtype": "auto", "score_threshold": 0.4, "classes": ""}
    outs = [nodes.Detect().run(**kw, batch_size=b) for b in (1, 2)]
    assert calls == [((3, 4, 6), 0.4)] * 3 + [((2, 3, 4, 6), 0.4), ((1, 3, 4, 6), 0.4)]
    assert torch.equal(outs[0][0], outs[1][0])
    assert outs[0][1] == outs[1][1]
    assert "batch_size" in nodes.Detect.INPUT_TYPES()["optional"]


def test_pose_batch_size_passes_each_images_boxes(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pose with `batch_size` > 1: chunks of the batch with one box list per image, outputs as one at a time."""
    nodes = pack.nodes
    calls: list[tuple[tuple[int, ...], Any]] = []

    class Result:
        keypoints = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
        keypoint_scores = torch.tensor([[0.9, 0.8]])
        skeleton = ((0, 1),)

        def to_list(self) -> list[dict[str, Any]]:
            return [{"box": [0.0, 0.0, 6.0, 4.0], "score": 0.9, "keypoints": {}}]

    class Model:
        device = torch.device("cpu")
        needs_boxes = True
        variant = SimpleNamespace(output_info={"label_set": "coco_keypoints"})

        def __call__(self, x: torch.Tensor, *, boxes: Any) -> Any:
            calls.append((tuple(x.shape), boxes))
            return Result() if x.ndim == 3 else [Result() for _ in range(x.shape[0])]

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Model())
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])
    img = torch.rand(3, 4, 6, 3)
    kw = {"image": img, "model": "m", "backend": "auto", "dtype": "auto", "keypoint_threshold": 0.3, "line_width": 1}
    outs = [nodes.Pose().run(**kw, batch_size=b) for b in (1, 2)]
    whole = [[0.0, 0.0, 6.0, 4.0]]
    assert calls == [((3, 4, 6), whole)] * 3 + [((2, 3, 4, 6), [whole, whole]), ((1, 3, 4, 6), [whole])]
    assert torch.equal(outs[0][0], outs[1][0])
    assert outs[0][1] == outs[1][1]
    assert "batch_size" in nodes.Pose.INPUT_TYPES()["optional"]


def test_pair_nodes_batch_size_matches_one_pair_at_a_time(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interpolate and Flow with `batch_size` > 1: chunks of consecutive pairs per call, the same outputs as 1."""
    nodes = pack.nodes
    calls: list[tuple[int, ...]] = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, a: torch.Tensor, b: torch.Tensor, timestep: float = 0.0) -> Any:
            calls.append(tuple(a.shape))
            return SimpleNamespace(frame=a * (1 - timestep) + b * timestep, flow=(b - a)[..., :2, :, :])

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Model())
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])
    frames = torch.rand(5, 4, 6, 3)
    kw = {"frames": frames, "model": "m", "backend": "auto", "dtype": "auto"}
    one, three = (nodes.Interpolate().run(**kw, multiplier=3, batch_size=b)[0] for b in (1, 3))
    assert one.shape == (13, 4, 6, 3)
    torch.testing.assert_close(three, one)
    assert calls[-4:] == [(3, 3, 4, 6), (3, 3, 4, 6), (1, 3, 4, 6), (1, 3, 4, 6)]  # 2 timesteps x 2 chunks
    calls.clear()
    one_f, three_f = (nodes.Flow().run(**kw, batch_size=b) for b in (1, 3))
    assert calls == [(3, 4, 6)] * 4 + [(3, 3, 4, 6), (1, 3, 4, 6)]
    torch.testing.assert_close(three_f[1], one_f[1])
    torch.testing.assert_close(three_f[0], one_f[0])
    assert "batch_size" in nodes.Interpolate.INPUT_TYPES()["optional"]
    assert "batch_size" in nodes.Flow.INPUT_TYPES()["optional"]


def test_remove_background_batch_size(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """RemoveBackground with `batch_size` > 1 batches the images; with a trimap it stays one image per call."""
    nodes = pack.nodes
    calls: list[tuple[int, ...]] = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, x: torch.Tensor, *, trimap: Any = None) -> Any:
            calls.append(tuple(x.shape))
            alpha = x[..., 0, :, :] if trimap is None else trimap
            return SimpleNamespace(alpha=alpha)

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Model())
    monkeypatch.setattr(nodes, "each", lambda n, fn: [fn(i) for i in range(n)])
    img = torch.rand(3, 4, 6, 3)
    kw = {"image": img, "model": "m", "backend": "auto", "dtype": "auto"}
    one, two = (nodes.RemoveBackground().run(**kw, batch_size=b) for b in (1, 2))
    assert calls == [(3, 4, 6)] * 3 + [(2, 3, 4, 6), (1, 3, 4, 6)]
    torch.testing.assert_close(two[1], one[1])
    torch.testing.assert_close(two[0], one[0])
    calls.clear()
    nodes.RemoveBackground().run(**kw, trimap=torch.rand(3, 4, 6), batch_size=2)
    assert calls == [(3, 4, 6)] * 3
    assert "batch_size" in nodes.RemoveBackground.INPUT_TYPES()["optional"]


def test_errors_name_the_backend_input(pack: Any) -> None:
    """The library's `mode=` is the node's `backend` input: errors say so, keeping their type and other words."""
    import fastforks  # noqa: PLC0415

    nodes = pack.nodes
    err = fastforks.KernelUnavailableError("cutlass_conv", "no binary", model="coex")
    with pytest.raises(fastforks.KernelUnavailableError) as info, nodes.node_errors():
        raise err
    assert str(info.value) == (
        "coex (backend='kernels'): kernel library 'cutlass_conv' unavailable: no binary; use backend='auto' to fall"
        " back to portable"
    )
    assert info.value.library == "cutlass_conv"
    assert nodes.backend_words("padding_mode='zeros', mode='portable'") == "padding_mode='zeros', backend='portable'"
    msg = "mode='x'"
    with pytest.raises(ValueError, match=msg), nodes.node_errors():  # not a fastforks error: unchanged
        raise ValueError(msg)


def test_licence_errors_in_each_pack(pack: Any) -> None:
    """The library's messages stay whole; fastforks-commercial adds the fastforks pack, never the variable."""
    import fastforks  # noqa: PLC0415
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.errors import install_hint  # noqa: PLC0415

    nodes = pack.nodes
    card = fastforks.model_card("segformer-b2-clothes")
    variant = next(v for v in card.variants if v.ref == card.variant_ref)
    gate = registry._nc_weights_error(variant, card)
    library = str(gate)
    text = nodes.node_words(gate, library, commercial_only=True)
    assert text.startswith(library)
    added = text.removeprefix(library)
    assert added == f". In ComfyUI: this model {nodes.ONLY_FULL}."
    assert "comfy node install fastforks`" in added
    assert "FASTFORKS_NONCOMMERCIAL" not in added
    assert ".bat" not in added

    missing = fastforks.ModelNotFoundError(
        "stereo model 's2m2-s' is in the package fastforks-nc, which is not installed: "
        'pip install "fastforks-core[nc]" (or fastforks; check its licence first)'
    )
    text = nodes.node_words(missing, str(missing), commercial_only=True)
    assert text == f"{missing}. In ComfyUI: this model {nodes.ONLY_FULL}."
    full = nodes.node_words(missing, str(missing))  # the fastforks pack: install the missing package
    assert '`python_embeded\\python.exe -m pip install "fastforks-core[nc]"`' in full
    assert nodes.FULL_PACK + " pack" not in full

    dep = fastforks.MissingDependencyError(
        f"x needs timm, which is not installed: {install_hint('timm', upgrade=True)}"
    )
    for commercial_only in (False, True):
        text = nodes.node_words(dep, str(dep), commercial_only=commercial_only)
        assert '`python_embeded\\python.exe -m pip install -U "fastforks-core[timm]"`' in text
        assert "non-commercial" not in text  # a permissive extra: no licence words

    only = fastforks.LicenceError("x: commercial use is 'no', and FASTFORKS_COMMERCIAL_ONLY=1 refuses models")
    assert nodes.node_words(only, str(only)) == str(only)  # the user's own commercial-only switch: nothing to add


def test_unlisted_models_say_why(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved id the node does not list: in fastforks-commercial one sentence naming the fastforks pack."""
    from fastforks._core import registry  # noqa: PLC0415

    nodes = pack.nodes
    monkeypatch.setattr(
        registry, "_known_models", lambda t: {"fake-nc-stereo": "fastforks-nc"} if t == "stereo" else {}
    )
    stereo, cstereo = nodes.Stereo, pack.commercial.NODE_CLASS_MAPPINGS["FastforksStereo"]
    entries = cstereo.INPUT_TYPES()["required"]["model"][0]
    assert cstereo.VALIDATE_INPUTS(entries[0]) is True
    assert cstereo.VALIDATE_INPUTS(None) is True  # linked: checked when it runs
    assert cstereo.VALIDATE_INPUTS(entries[0] + " [non-commercial]") is True  # saved under another licence tag
    for mid in ("s2m2-l", "s2m2-l [non-commercial]", "fake-nc-stereo"):  # installed and not cleared; not installed
        why = cstereo.VALIDATE_INPUTS(mid)
        assert why == f"'{nodes.model_id(mid)}' {nodes.ONLY_FULL}.", mid
        assert why.count(". ") == 0  # one sentence
    assert "only in the fastforks pack" in nodes.ONLY_FULL
    assert "comfy node install fastforks`" in nodes.ONLY_FULL
    assert stereo.VALIDATE_INPUTS("s2m2-l [non-commercial]") is True  # the fastforks pack lists it
    why = str(stereo.VALIDATE_INPUTS("fake-nc-stereo"))  # published, not installed: install the package
    assert why.startswith("stereo model 'fake-nc-stereo' is in the package fastforks-nc, which is not installed")
    assert "python_embeded" in why
    assert "did you mean" in str(cstereo.VALIDATE_INPUTS("raft-stereo-middlebur"))
    assert "inputs the node does not pass" in str(cstereo.VALIDATE_INPUTS("depth-anything-v2-s"))
    assert nodes.Warp.VALIDATE_INPUTS() is True

    tip = cstereo.INPUT_TYPES()["required"]["model"][1]["tooltip"]
    n = len([m for m in stereo.ids() if not nodes.commercial(m)]) + 1
    assert f"{n} more stereo models (e.g." in tip
    assert "are only in the fastforks pack" in tip
    full_tip = stereo.INPUT_TYPES()["required"]["model"][1]["tooltip"]
    assert nodes.NC_NOTICE in full_tip
    assert "Not installed: 1 more stereo models" in full_tip
    assert "FASTFORKS_NONCOMMERCIAL" not in tip + full_tip

    line = nodes.licence_state(commercial_only=True)
    assert line.startswith("fastforks: fastforks-commercial lists only models cleared for commercial use")
    assert "other models are only in the fastforks pack" in line
    line = nodes.licence_state()
    assert "installing this pack is the opt-in" in line
    assert "your responsibility" in line
    assert "1 models with non-commercial or GPL code (e.g. fake-nc-stereo) are not installed" in line
    assert "FASTFORKS_NONCOMMERCIAL" not in line


def test_commercial_pack_never_runs_nc_models(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """fastforks-commercial refuses an nc id before any download (a linked input skips validation); the fastforks
    pack's run allows non-commercial weights, and only while it runs."""
    import fastforks  # noqa: PLC0415
    from fastforks._core import registry  # noqa: PLC0415

    nodes = pack.nodes
    monkeypatch.delenv(registry.NONCOMMERCIAL_ENV, raising=False)
    monkeypatch.setattr(pack.mm, "get_torch_device", lambda: torch.device("cpu"), raising=False)
    monkeypatch.setattr(pack.mm, "free_memory", lambda *_: pytest.fail("got as far as freeing memory"), raising=False)
    monkeypatch.setattr(fastforks, "load", lambda *_a, **_k: pytest.fail("loaded"))
    seen: list[tuple[bool, bool]] = []

    def run(self: Any, **inputs: Any) -> tuple[Any, ...]:
        seen.append((registry.noncommercial_enabled(), nodes._COMMERCIAL_RUN.get()))
        if inputs.get("load"):
            nodes.load(inputs["load"], "auto", "auto")
        return (inputs,)

    monkeypatch.setattr(nodes.SemanticSegment, "run", run)
    full = pack.NODE_CLASS_MAPPINGS["FastforksSemanticSegment"]
    commercial = pack.commercial.NODE_CLASS_MAPPINGS["FastforksSemanticSegment"]
    assert full.FUNCTION == commercial.FUNCTION == "execute"
    with pytest.raises(fastforks.LicenceError) as info:
        commercial().execute(load="segformer-b2-clothes [non-commercial]")
    assert str(info.value) == f"segformer-b2-clothes {nodes.ONLY_FULL}."
    assert full().execute(model="m") == ({"model": "m"},)
    assert seen == [(False, True), (True, False)]
    assert not registry.noncommercial_enabled()  # only during the run; the environment is untouched
    assert not nodes._COMMERCIAL_RUN.get()
    assert registry.NONCOMMERCIAL_ENV not in os.environ


def test_pack_split(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fastforks pack lists every model (nc ones tagged), fastforks-commercial commercial ones; same classes."""
    nodes = pack.nodes
    monkeypatch.setattr(nodes, "upscale_files", list)
    full, commercial = pack.NODE_CLASS_MAPPINGS, pack.commercial.NODE_CLASS_MAPPINGS
    assert set(commercial) == set(full) - {"FastforksParse"}  # Parse: both of its models are non-commercial
    assert all(k.startswith("Fastforks") for k in full)
    assert set(pack.NODE_DISPLAY_NAME_MAPPINGS) == set(full)
    assert set(pack.commercial.NODE_DISPLAY_NAME_MAPPINGS) == set(commercial)
    for key, name in pack.commercial.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert name.endswith((" (fastforks, commercial)", ", fastforks, commercial)")), name
        assert name != pack.NODE_DISPLAY_NAME_MAPPINGS[key]
    assert all(v.COMMERCIAL_ONLY for v in commercial.values())
    assert not any(v.COMMERCIAL_ONLY for v in full.values())
    assert pack.commercial.nodes is nodes  # one implementation

    def entries(node: type) -> list[str]:
        model = node.INPUT_TYPES()["required"].get("model")
        return model[0] if model else []

    def ids(node: type) -> list[str]:
        return [nodes.model_id(e) for e in entries(node)]

    for name, node in commercial.items():
        if node.TASK:
            assert ids(node), name
            assert all(nodes.commercial(m) for m in ids(node)), name
            assert not any(e.endswith("]") for e in entries(node)), name
            assert set(ids(node)) <= set(ids(full[name])), name
    for name, node in full.items():
        for e in entries(node):
            assert e.endswith(("[non-commercial]", "[licence unclear]")) != nodes.commercial(nodes.model_id(e)), e
        if any(e.endswith("]") for e in entries(node)):
            assert nodes.NC_NOTICE in node.DESCRIPTION, name
    assert nodes.NC_NOTICE not in commercial["FastforksStereo"].DESCRIPTION
    assert {"s2m2-l", "foundationstereo-vit-s"} <= set(ids(full["FastforksStereo"]))
    assert not {"s2m2-l", "foundationstereo-vit-s"} & set(ids(commercial["FastforksStereo"]))
    assert ids(full["FastforksParse"]) == ["segformer-b2-clothes", "segformer-b5-face-parsing"]
    assert full["FastforksFaceRestore"].INPUT_TYPES()["required"]["model"][1]["default"].startswith("codeformer")
    assert ids(commercial["FastforksFaceRestore"]) == ["gfpgan-v1.4"]
    monkeypatch.setattr(nodes, "upscale_files", lambda: ["file: x.pth"])  # your own files: both packs
    assert "file: x.pth" in ids(full["FastforksUpscale"])
    assert "file: x.pth" in ids(commercial["FastforksUpscale"])


def test_commercial_pack_steps_aside(pack: Any, tmp_path: Path) -> None:
    """With the fastforks pack next to it in custom_nodes, fastforks-commercial registers nothing; else it finds the
    implementation (imported, bundled or the clone's integrations/comfyui) or says what is missing."""
    find, impl = pack.commercial.full_pack, pack.commercial.implementation
    assert find(COMMERCIAL_PACK) is None  # integrations/ is not custom_nodes, though integrations/comfyui is there
    custom = tmp_path / "custom_nodes"
    here, other, ours, off = (custom / n for n in ("fastforks-commercial", "a-pack", "renamed", "z.disabled"))
    for d in (here, other, ours, off):
        d.mkdir(parents=True)
        (d / "nodes.py").write_text("")
    (other / "pyproject.toml").write_text('[project]\nname = "a-pack"\n')
    full_meta = (PACK / "pyproject.toml").read_text(encoding="utf-8")
    (off / "pyproject.toml").write_text(full_meta, encoding="utf-8")
    assert find(here) is None  # a disabled fastforks pack does not count
    (ours / "pyproject.toml").write_text(full_meta, encoding="utf-8")
    assert find(here) == ours
    # ComfyUI that skips the full pack (--disable-all-custom-nodes without it in --whitelist-custom-nodes) does not
    # load it, so fastforks-commercial must not step aside for it
    fake = types.SimpleNamespace(args=types.SimpleNamespace(disable_all_custom_nodes=True, whitelist_custom_nodes=[]))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
        mp.setitem(sys.modules, "comfy.cli_args", fake)
        assert find(here) is None
        fake.args.whitelist_custom_nodes = ["fastforks-commercial"]
        assert find(here) is None
        fake.args.whitelist_custom_nodes = ["renamed"]
        assert find(here) == ours
        fake.args.disable_all_custom_nodes = False
        fake.args.whitelist_custom_nodes = []
        assert find(here) == ours
    assert impl(here) is pack.nodes  # already imported: the one copy
    saved = sys.modules.pop("fastforks_comfyui")
    try:
        with pytest.raises(ImportError, match="has no nodes"):
            impl(tmp_path / "elsewhere" / "fastforks-commercial")
        assert "fastforks_comfyui" not in sys.modules
    finally:
        sys.modules["fastforks_comfyui"] = saved


def test_pack_metadata() -> None:
    """Both packs: registry metadata, their own requirements, no env-var or .bat instructions; the licence notice."""
    import tomllib  # noqa: PLC0415

    meta = {d.name: tomllib.loads((d / "pyproject.toml").read_text(encoding="utf-8")) for d in (PACK, COMMERCIAL_PACK)}
    assert {d: m["project"]["name"] for d, m in meta.items()} == {
        "comfyui": "fastforks",
        "comfyui-commercial": "fastforks-commercial",
    }
    for d, m in meta.items():
        comfy = m["tool"]["comfy"]
        assert {"PublisherId", "DisplayName", "Icon"} <= set(comfy), d
        assert comfy["DisplayName"].startswith("fastforks \N{EN DASH} fast vision models"), d
        assert m["project"]["license"] == {"text": "Apache-2.0"}, d
        requirements = (PACK.parent / d / "requirements.txt").read_text(encoding="utf-8").split()
        assert requirements == m["project"]["dependencies"], d
        assert m["project"]["description"].startswith("Fast vision nodes: "), d
    assert meta["comfyui"]["project"]["dependencies"] == ["fastforks>=0.2"]
    assert meta["comfyui-commercial"]["project"]["dependencies"] == ["fastforks-core[timm,transformers]>=0.2"]
    assert meta["comfyui-commercial"]["tool"]["comfy"]["DisplayName"].endswith("(commercial use)")
    assert meta["comfyui-commercial"]["tool"]["comfy"]["includes"] == ["fastforks_comfyui"]
    assert "your responsibility" in meta["comfyui"]["project"]["description"]
    assert "[non-commercial]" in meta["comfyui"]["project"]["description"]
    docs = Path(__file__).resolve().parents[1] / "docs" / "comfyui.md"
    for path in [docs, *PACK.glob("*.*"), *COMMERCIAL_PACK.glob("*.*")]:
        text = path.read_text(encoding="utf-8")
        assert "FASTFORKS_NONCOMMERCIAL=1" not in text, path
        assert "run_nvidia_gpu.bat" not in text, path


def test_readme_model_lists_match_the_registry(pack: Any) -> None:
    """Each pack's README lists its nodes and the models of their dropdowns (tools/gen_comfyui_readme.py)."""
    nodes = pack.nodes
    if nodes.not_installed():
        pytest.skip("the README lists are written with every model family installed")
    for commercial_only, folder in ((False, PACK), (True, COMMERCIAL_PACK)):
        text = (folder / "README.md").read_text(encoding="utf-8")
        block = text[text.index(nodes.README_START) : text.index(nodes.README_END) + len(nodes.README_END)]
        assert block == nodes.readme_models(commercial_only=commercial_only), (
            f"{folder / 'README.md'} is out of date: run tools/gen_comfyui_readme.py"
        )
    full = nodes.readme_models()
    commercial = nodes.readme_models(commercial_only=True)
    assert "`s2m2-l [non-commercial]`" in full
    assert "s2m2-l" not in commercial
    assert "[non-commercial]" not in commercial
    assert "Parse clothes / face" in full
    assert "Parse clothes / face" not in commercial


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
    assert c.subject_boxes(dets, 3, "coco-keypoints-17") == [[1.0, 1.0, 2.0, 2.0]]
    assert c.subject_boxes(dets, 0, "ap10k-17") == [[0.0, 0.0, 5.0, 5.0]]  # the animal, not the person
    assert c.subject_boxes(dets, 0, "lapa-106") == [[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 2.0, 2.0]]  # face boxes: all
    cars = json.dumps([[{"box": [0, 0, 5, 5], "class_name": "car"}]])
    assert c.subject_boxes(cars, 0, "ap10k-17") == [[0.0, 0.0, 5.0, 5.0]]  # no animal: every box
    assert c.subject_boxes("", 0, "halpe-26") is None
    from fastforks.pose._api import SKELETONS  # noqa: PLC0415 - only this check needs the pose task

    assert set(c.SUBJECTS) <= set(SKELETONS)  # every named label set exists
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


def test_comfyui_making_room_frees_our_models(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """ComfyUI's free_memory, short after unloading its own models: our CUDA graphs go first, then our models (least
    recently used first), only on that device and only until enough is free."""
    nodes, mm = pack.nodes, pack.mm
    cuda, other = torch.device("cuda", 0), torch.device("cuda", 1)

    class Held:
        def __init__(self, device: torch.device, weights: int, graphs: int) -> None:
            self.device, self.weights, self.graphs = device, weights, graphs

        def release_graphs(self) -> None:
            self.graphs = 0

    monkeypatch.setattr(nodes, "_cache", nodes.OrderedDict())
    nodes._cache[("old",)] = Held(cuda, 3, 2)
    nodes._cache[("new",)] = Held(cuda, 3, 2)
    nodes._cache[("elsewhere",)] = Held(other, 3, 2)
    total = 20

    def used(dev: torch.device, *, graphs: bool) -> int:
        return sum(m.weights + graphs * m.graphs for m in nodes._cache.values() if m.device == dev)

    # as ComfyUI: a graph's pool is PyTorch's cached memory, counted as free; the device itself does not have it
    monkeypatch.setattr(mm, "get_free_memory", lambda dev, **_: total - used(dev, graphs=False))
    monkeypatch.setattr(nodes, "_device_free", lambda dev: total - used(dev, graphs=True))
    monkeypatch.setattr(mm, "free_memory", lambda *_, **__: ["comfy"])
    nodes.install_unload_hook()  # wraps the stub just set

    assert mm.free_memory(9, cuda) == ["comfy"]  # 10 free on the device: nothing to do
    assert all(m.graphs == 2 for m in nodes._cache.values())
    mm.free_memory(13, cuda)  # ComfyUI sees 14 free, the device has 10: the graphs go, both models stay
    assert list(nodes._cache) == [("old",), ("new",), ("elsewhere",)]
    assert [m.graphs for m in nodes._cache.values()] == [0, 0, 2]  # the other device's graphs stay
    mm.free_memory(17, cuda)  # one model (3) more is enough: the least recently used goes
    assert list(nodes._cache) == [("new",), ("elsewhere",)]
    mm.free_memory(100, cuda)  # never enough: every model on that device goes, the other device's stays
    assert list(nodes._cache) == [("elsewhere",)]


def test_load_keeps_models_while_memory_lasts(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A workflow cycling several nodes reloads none of them; only memory (or `_MAX_CACHED`) drops the oldest."""
    nodes, mm = pack.nodes, pack.mm
    loads: list[str] = []
    asked: list[float] = []

    def fake_load(mid: str, **kw: Any) -> Any:
        loads.append(mid)
        return types.SimpleNamespace(device=kw["device"], backend="portable", release_graphs=lambda: None)

    monkeypatch.setattr(nodes, "_cache", nodes.OrderedDict())
    monkeypatch.setattr(nodes.fastforks, "load", fake_load)
    monkeypatch.setattr(nodes, "_weights_bytes", lambda _: 2**20)
    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"), raising=False)
    monkeypatch.setattr(mm, "free_memory", lambda need, *_, **__: asked.append(need) or [])
    ids = ["depth-anything-v2-s", "birefnet-lite", "sam2.1-hiera-small", "rife-4.26", "realesr-general-x4v3"]
    for _ in range(3):
        for mid in ids:
            nodes.load(mid, "auto", "auto")
    assert loads == ids  # each loaded once
    assert asked == [2 * 2**20 + 2**30] * len(ids)  # room asked before each load, never on a cache hit
    monkeypatch.setattr(nodes, "_MAX_CACHED", 2)
    nodes.load("rtdetrv2-r18", "auto", "auto")
    assert [k[0] for k in nodes._cache] == ["realesr-general-x4v3", "rtdetrv2-r18"]  # oldest dropped first


def test_cache_setup(pack: Any, tmp_path: Path) -> None:
    """Downloads go to models/fastforks; the normal cache and extra model folders are read in place."""
    first, extra, normal = tmp_path / "first", tmp_path / "extra", tmp_path / "normal"
    for d in (extra, normal):
        d.mkdir()
    pack.fp.paths["fastforks"] = [str(first), str(extra), str(tmp_path / "missing")]
    again = str(tmp_path / "extra" / ".." / "normal")  # the same folder spelled another way is read once
    environ = {"FASTFORKS_CACHE": str(normal), "FASTFORKS_READ_CACHES": os.pathsep.join([again, str(first)])}
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
    """models/upscale_models lists only files fastforks runs; the node output equals `restoration.load(path)`."""
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
    m = restoration.load(tmp_path / "2x-span.pth", mode="portable", device="cpu", dtype="float32")
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
    assert not {"codeformer", "gfpgan-v1.4", "big-lama"} & ids(nodes.Upscale)  # aligned faces; inpainting (mask)
    assert ids(nodes.FaceRestore) == {"codeformer", "gfpgan-v1.4"}  # CodeFormer: not in fastforks-commercial
    # inputs the nodes do not have: a depth prompt, a first-frame mask (video matting), a reference (in-context)
    assert not {"prompt-depth-anything-s", "lingbot-depth", "omni-dc"} & ids(nodes.Depth)
    assert "matanyone" not in ids(nodes.RemoveBackground)
    assert "vitmatte-s" in ids(nodes.RemoveBackground)  # trimap: an optional input of the node
    assert "insid3-vitl16" not in ids(nodes.Segment)
    assert {"sam2.1-hiera-small", "clipseg", "sam3"} <= ids(nodes.Segment)  # points / boxes; text
    tagger = {"wd-vit-tagger-v3", "wd-convnext-tagger-v3", "wd-vit-large-tagger-v3", "wd-eva02-large-tagger-v3"}
    assert ids(nodes.Tag) == tagger
    assert ids(nodes.NsfwCheck) == {"falconsai-nsfw-vit"}
    assert ids(nodes.Parse) == {"segformer-b2-clothes", "segformer-b5-face-parsing"}  # not in fastforks-commercial
    assert "segnext-mscan-t-ade20k" in ids(nodes.SemanticSegment)
    assert "segformer-b2-clothes" in ids(nodes.SemanticSegment)
    clips = ids(nodes.VideoDepth)
    assert clips == {"video-depth-anything-s", "video-depth-anything-metric-s"}
    assert not clips & ids(nodes.Depth)  # video models run as a clip in Video depth, not frame by frame
    assert "depth-anything-v2-s" in ids(nodes.Depth)


def test_video_depth_scales_the_clip(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """One min / max for the whole clip (per-frame scaling would flicker); the model runs the frames as one video."""
    nodes = pack.nodes
    seen: list[tuple[int, ...]] = []

    class Clip:
        device = torch.device("cpu")

        def video(self, frames: torch.Tensor) -> types.SimpleNamespace:
            seen.append(tuple(frames.shape))
            inv = torch.stack([torch.full((4, 6), 1.0), torch.full((4, 6), 2.0), torch.full((4, 6), 3.0)])
            return types.SimpleNamespace(depth=None, inverse_depth=inv)

    monkeypatch.setattr(nodes, "load", lambda *_a, **_k: Clip())
    (out,) = nodes.VideoDepth().run(frames=torch.rand(3, 4, 6, 3), model="m", backend="auto", dtype="auto")
    assert seen == [(3, 3, 4, 6)]
    assert out.shape == (3, 4, 6, 3)
    assert [out[t].mean().item() for t in range(3)] == pytest.approx([0.0, 0.5, 1.0])


def test_wd_tags(pack: Any) -> None:
    """Ratings never; general and character tags each by their threshold, most likely first."""
    names = ("general", "sensitive", "questionable", "explicit", *[f"g{i}" for i in range(4, 8110)], "char_a", "char_b")
    probs = torch.zeros(len(names))
    probs[0], probs[4], probs[5], probs[8110], probs[8111] = 0.99, 0.4, 0.9, 0.8, 0.5
    assert pack.nodes.wd_tags(probs, names, 0.35, 0.75) == "g5, char_a, g4"
    assert pack.nodes.wd_tags(probs, names, 0.95, 0.95) == ""


def test_every_node_matches_the_library(pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every node: a non-empty model list of cards whose required inputs it passes, `run` arguments = inputs, a name."""
    import inspect  # noqa: PLC0415

    import fastforks  # noqa: PLC0415

    nodes = pack.nodes
    monkeypatch.setattr(nodes, "upscale_files", list)
    for name, node in [*pack.NODE_CLASS_MAPPINGS.items(), *pack.commercial.NODE_CLASS_MAPPINGS.items()]:
        spec = node.INPUT_TYPES()
        assert name in pack.NODE_DISPLAY_NAME_MAPPINGS
        assert set(inspect.signature(node.run).parameters) - {"self"} == set(spec["required"]) | set(
            spec.get("optional", {})
        ), name
        if "model" in spec["required"]:
            labels = spec["required"]["model"][0]
            assert labels, f"{name}: no models listed"
            for label in labels:
                mid = nodes.model_id(label)
                fastforks.model_card(mid)
                assert node.runs(mid), f"{name} lists {mid}, which needs inputs the node does not pass"
