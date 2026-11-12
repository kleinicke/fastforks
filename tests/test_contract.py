"""`model_card(id).contract()`, option bounds, classification `labels=` and the MPS hook (CPU, no weights)."""

from __future__ import annotations

import dataclasses
import json
import types

import pytest
import torch

from fastforks import InputError, ModelNotFoundError, OptionsError, base_type, classification, depth
from fastforks._core import model as core
from fastforks._core import mps, registry
from fastforks._core.model import check_options, model_card
from fastforks.classification._runner import Runner as ClassificationRunner
from fastforks.detection._api import DetectionModel


def test_every_contract_resolves_without_weights():
    """Every installed id has a JSON-ready contract whose outputs are the variant's and whose options are the card's."""
    n = 0
    for task in registry.TASKS:
        for mid in registry.cards(task):
            card = model_card(task, mid)
            c = card.contract()
            d = json.loads(json.dumps(c.to_dict()))
            assert d["model"] == card.variant_ref == f"{mid}@{registry.resolve(task, mid)[1].revision}"
            assert list(d["outputs"])[: len(card.outputs)] == list(card.outputs)  # then the derived fields
            assert set(d["options"]) == set(card.options)
            assert d["export"]["dynamic"] is not bool(d["export"]["fixed_reason"])
            for mode in card.modes:
                assert mode in d["modes"]
            n += 1
    assert n > 100


def test_family_card_has_no_contract():
    card, _ = registry.resolve("depth", "moge-2-vitl-normal")  # the family's card, no variant resolved
    with pytest.raises(ModelNotFoundError, match="model_card"):
        card.contract()


def test_contract_inputs_follow_the_variant():
    """DA3: metric variants take optional intrinsics, mono ones none; the card says it, not the docs."""
    metric = depth.model_card("da3metric-large").contract()
    mono = depth.model_card("da3mono-large").contract()
    assert metric.inputs == {"image": "required", "intrinsics": "optional"}
    assert mono.inputs == {"image": "required", "intrinsics": "unused"}
    assert metric.camera["pixels"] == "opencv"
    assert metric.outputs["depth"]["unit"] == "metres"
    assert mono.outputs["depth"]["unit"] == "unknown scale and shift"
    with pytest.raises(TypeError):
        metric.inputs["x"] = 1


def test_contract_multiview_and_modes():
    c = model_card("multiview", "mvsanywhere").contract()
    assert c.inputs["intrinsics"] == c.inputs["cam_to_world"] == "required"
    assert c.inputs["depth_range"] == "optional"
    assert c.inputs["min_views"] == 2
    assert c.camera["pixels"] == "opencv"
    assert c.modes["kernels"]["inputs"]["max_views"] > 2  # a per-mode limit
    assert c.export["inputs"] == ("images", "intrinsics", "cam_to_world")
    assert not c.export["dynamic"]
    assert model_card("multiview", "vggt-1b").contract().inputs["intrinsics"] == "unused"
    s = model_card("stereo", "hitnet-eth3d").contract()
    assert s.inputs == {"left": "required", "right": "required"}
    assert "dynamic export is not supported" in s.export["fixed_reason"]
    assert model_card("stereo", "raft-stereo-middlebury").contract().export["dynamic"]
    assert s.modes["portable"]["cpu"] == {"implemented": ("float32",), "validated": ("float32",)}


def test_contract_promptable_and_classification():
    sam = model_card("promptable_segmentation", "sam-vit-b").contract()
    assert set(sam.inputs["prompts"]) == {"points", "boxes", "masks"}
    assert sam.export["default_size"] == (1024, 1024)
    assert "mask decoder" in sam.export["outside"]
    cls = model_card("classification", "resnet50").contract()
    assert cls.export["default_size"] == (224, 224)
    assert "imagenet-1k" in cls.outputs["logits"]["unit"]


def test_contract_options():
    c = depth.model_card("moge-2-vitl-normal").contract()
    assert c.options["resolution_level"] == {
        "default": 9,
        "min": 0,
        "max": 9,
        "choices": (),
        "effect": "output",
        "aliases": (),
    }
    graphs = model_card("stereo", "hitnet-eth3d").contract().options["cuda_graphs"]
    assert graphs["effect"] == "speed"


@pytest.mark.parametrize("value", [10, 100, -1])
def test_option_max_is_checked(value):
    """MoGe `resolution_level` 100 was accepted and cost 7.8x the tokens; now 0-9 only, at load."""
    card = depth.model_card("moge-2-vitl-normal")
    with pytest.raises(OptionsError, match=r"'resolution_level' must be 0 to 9"):
        check_options(card, {"resolution_level": value})
    assert check_options(card, {"resolution_level": 3}) == {"resolution_level": 3}


def test_option_choices_are_checked():
    card = dataclasses.replace(
        depth.model_card("moge-2-vitl-normal"), option_choices=types.MappingProxyType({"resolution_level": (0, 9)})
    )
    with pytest.raises(OptionsError, match="must be one of 0, 9, got 3"):
        check_options(card, {"resolution_level": 3})


def test_bounded_options_on_cards():
    """The options the review found unbounded have bounds now."""
    bounds = {
        ("depth", "moge-2-vitl-normal", "resolution_level"): "0 to 9",
        ("depth", "unik3d-vitl", "resolution_level"): "0 to 9",
        ("depth", "marigold-depth-v1-1", "steps"): "1 to 1000",
        ("detection", "detr-r50-coco", "score_threshold"): "0.0 to 1.0",
    }
    for (task, mid, name), words in bounds.items():
        assert model_card(task, mid).option_bounds(name) == words, (mid, name)


def test_detection_threshold_uses_the_card_bounds():
    fake = types.SimpleNamespace(card=model_card("detection", "detr-r50-coco"))
    assert DetectionModel._threshold(fake, 0.5) == 0.5
    for bad in (1.5, -0.1, float("nan"), True):
        with pytest.raises(InputError, match=r"from 0\.0 to 1\.0"):
            DetectionModel._threshold(fake, bad)


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (["cat", "dog"], "labels has 2 names, the classifier head has 1000 classes"),
        ("cat", "sequence of class-name strings"),
        ([1] * 1000, "sequence of class-name strings"),
        ([], "sequence of class-name strings"),
    ],
)
def test_labels_are_checked_before_loading(labels, match, monkeypatch):
    monkeypatch.setattr(core, "load_model", lambda *_a, **_k: pytest.fail("loaded before the labels check"))
    with pytest.raises(OptionsError, match=match):
        classification.load("resnet50", device="cpu", labels=labels)


def test_labels_rename_the_classes(monkeypatch):
    card, v = registry.resolve("classification", "resnet50")
    names = [*card.params, *v.params, *card.options]
    params = dataclasses.make_dataclass("Params", names, frozen=True)

    class Dummy(ClassificationRunner):
        Params = params

        def __call__(self, image):
            logits = torch.zeros(image.shape[0], 1000)
            logits[:, 7] = 5.0
            return {"logits": logits}

    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Dummy}))
    labels = (f"c{i}" for i in range(1000))  # any iterable of strings
    m = classification.load("resnet50", mode="portable", device="cpu", labels=labels)
    assert m.class_names[7] == "c7"
    assert m.id2label[999] == "c999"
    r = m(torch.rand(3, 64, 64))
    assert r.class_names[7] == "c7"
    assert r.top(1)[0][0] == "c7"


class _Two(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(4, 4, 2, stride=2)


def test_mps_adapts_every_network(monkeypatch):
    """`adapt_to_device` adapts `runner.networks()`, not only `module()` (CPU: a fake MPS device, no Metal)."""
    monkeypatch.setattr(mps, "sync_at_exit", lambda: None)
    nets = (_Two(), _Two())
    runner = types.SimpleNamespace(networks=lambda: nets)
    core.adapt_to_device(runner, "kernels", torch.device("mps"))
    assert all(base_type(n.up) is type(n.up) for n in nets)  # kernels runners adapt themselves
    core.adapt_to_device(runner, "portable", torch.device("cpu"))
    assert all(base_type(n.up) is type(n.up) for n in nets)
    core.adapt_to_device(runner, "portable", torch.device("mps"))
    assert all(base_type(n.up) is not type(n.up) for n in nets)


def test_default_networks_is_module():
    from fastforks._core.runner import Runner  # noqa: PLC0415

    net = _Two()
    fake = types.SimpleNamespace(module=lambda: net)
    assert Runner.networks(fake) == (net,)


def test_sam_networks_include_the_mask_decoder(monkeypatch):
    """SAM's `module()` is the image encoder only; its MPS adaptation must also reach the mask decoder's upscaling."""
    from fastforks.promptable_segmentation.models.sam import model as sam_model  # noqa: PLC0415

    card = model_card("promptable_segmentation", "sam-vit-b")
    _, v = registry.resolve("promptable_segmentation", "sam-vit-b")
    params = core.make_params(sam_model.Params, card, v, {})
    with torch.device("meta"):
        net = sam_model.build(params)
    runner = sam_model.Portable.__new__(sam_model.Portable)
    runner.sam = net
    ups = [m for m in net.mask_decoder.modules() if isinstance(m, torch.nn.ConvTranspose2d)]
    assert ups
    monkeypatch.setattr(mps, "sync_at_exit", lambda: None)
    core.adapt_to_device(runner, "portable", torch.device("mps"))
    assert all(base_type(m) is not type(m) for m in ups)


def test_input_use_of_the_runners():
    from fastforks.depth.models.depth_anything_3 import model as da3  # noqa: PLC0415

    for cls in (da3.Portable, da3.Kernels):
        arch = {f.name: None for f in dataclasses.fields(cls.Params)}
        metric = types.SimpleNamespace(**{**arch, "arch": "metric"})
        mono = types.SimpleNamespace(**{**arch, "arch": "mono"})
        assert cls.input_use(metric) == {"intrinsics": "optional"}
        assert cls.input_use(mono) == {"intrinsics": "unused"}


_NOT_OUTPUTS = {  # result fields that are metadata, not outputs
    "model", "backend", "options", "class_names", "kind", "extras", "views", "layers", "grid",
    "num_prefix_tokens", "num_register_tokens", "labels", "keypoint_names", "skeleton", "to_list",
}  # fmt: skip
_RESULTS = {
    "detection": "DetectionResult",
    "pose": "PoseResult",
    "flow": "FlowResult",
    "matching": "MatchingResult SparseMatchingResult",
    "matting": "MattingResult",
    "depth": "DepthResult",
    "features": "FeaturesResult",
    "multiview": "MultiViewResult",
    "classification": "ClassificationResult",
    "stereo": "StereoResult",
    "segmentation": "SegmentationResult",
    "promptable_segmentation": "PromptableSegmentationResult",
    "tracking": "TrackingResult",
}


def test_contract_lists_every_result_field():
    """Every output field (and derived property / method) of each task's result is in some contract of the task."""
    import importlib  # noqa: PLC0415

    for task, name in ((t, n) for t, names in _RESULTS.items() for n in names.split()):
        cls = getattr(importlib.import_module(f"fastforks.{task}._api"), name)
        names = {f.name for f in dataclasses.fields(cls)} | {
            n for n, v in vars(cls).items() if not n.startswith("_") and (isinstance(v, property) or callable(v))
        }
        listed = {k for mid in registry.cards(task) for k in model_card(task, mid).contract().outputs}
        keep = names - _NOT_OUTPUTS - {"save", "save_scene", "depth"}  # writers; StereoResult.depth raises (no rig)
        assert keep <= listed, (task, keep - listed)


def test_contract_output_presence_and_call_arguments():
    seg = model_card("segmentation", "deeplabv3-r101-d8-ade20k").contract()
    assert seg.outputs["labels"]["present"] == "always"
    assert "logits=True" in seg.outputs["logits"]["present"]
    assert seg.call["logits"] == {"default": False, "type": "bool", "min": None, "max": None}
    sam = model_card("promptable_segmentation", "sam-vit-b").contract()
    assert "masks" in sam.outputs
    assert sam.call["multimask_output"]["default"] is True
    det = model_card("detection", "detr-r101-coco").contract()
    assert det.call["score_threshold"]["default"] is None
    assert "reference" in model_card("multiview", "vggt-1b").contract().call


def test_contract_unit_strings():
    inv = model_card("depth", "depth-anything-v2-b").contract().outputs["inverse_depth"]
    assert inv["unit"].startswith("unknown scale and shift")
    assert "1 /" not in inv["unit"]
    mv = model_card("multiview", "vggt-1b").contract()
    for out in mv.outputs.values():
        assert out["frame"] != "extras"
    s2 = model_card("stereo", "s2m2-l").contract().outputs["occlusion"]
    assert s2["frame"] == ""


def test_option_aliases():
    """The old upstream name maps to the canonical one; both together, or a typo, is an `OptionsError`."""
    card = depth.model_card("da3mono-large")
    assert card.contract().options["long_side"]["aliases"] == ("process_res",)
    assert check_options(card, {"process_res": 252}) == {"long_side": 252}
    with pytest.raises(OptionsError, match="given twice"):
        check_options(card, {"process_res": 252, "long_side": 252})
    with pytest.raises(OptionsError, match="did you mean 'long_side'"):
        check_options(card, {"process_re": 252})
    with pytest.raises(OptionsError, match=r"must be >= 14"):
        check_options(card, {"process_res": 3})
