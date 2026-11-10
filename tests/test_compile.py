"""`Model.compile()` on stub runners (CPU, no weights): every task's default call path runs the compiled entry points,
nested entry points trace inline, a failed first compiled call leaves the model eager, a repeat compile replaces.

Real families: `test_export.test_model_compile_equals_eager`; here, every runner's `compiled_methods` exist.
"""

from __future__ import annotations

import dataclasses
import threading
import types
from typing import Any

import pytest
import torch
from torch._dynamo.exc import Unsupported

from fastforks import multiview, promptable_segmentation, segmentation, stereo
from fastforks._core import model as core_model
from fastforks._core import registry
from fastforks._core.errors import BackendUnavailableError
from fastforks.multiview._runner import Runner as MultiViewRunner
from fastforks.promptable_segmentation._runner import Runner as PromptRunner
from fastforks.segmentation._runner import Runner as SegRunner
from fastforks.stereo._runner import Runner as StereoRunner

pytestmark = pytest.mark.usefixtures("card_min_size")  # small images through stub runners


class Graphs:
    """A `torch.compile` backend that counts the graphs it compiles and runs them eagerly."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, gm: torch.fx.GraphModule, _inputs: Any) -> Any:
        self.n += 1
        return gm.forward


def _params(task: str, model: str) -> type:
    """A `Params` dataclass with the card's parameter names (stub runners ignore the values)."""
    card, v = registry.resolve(task, model)
    names = {**card.params, **v.params, **card.options}
    return dataclasses.make_dataclass("Params", [(n, Any) for n in names], frozen=True)


class _Net:
    def module(self) -> torch.nn.Module:
        return torch.nn.Identity()


def _load(monkeypatch: pytest.MonkeyPatch, api: Any, model: str, runner: type) -> Any:
    runner.Params = _params(api.__name__.rsplit(".", 1)[-1], model)
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": runner}))
    return api.load(model, mode="portable", device="cpu")


class Seg(_Net, SegRunner):
    def __call__(self, image):
        return {"logits": torch.cat([image, image[:, :1]], 1).repeat(1, 38, 1, 1)[:, :150] * 2 + 1}

    # `labels` is the base's: `self(image)["logits"].argmax(1)`, a nested entry point


@pytest.mark.slow
def test_segmentation_default_call_is_compiled(monkeypatch):
    m = _load(monkeypatch, segmentation, "fcn-r101-d8-ade20k", Seg)
    x = torch.rand(1, 3, 32, 32)
    want = m(x[0]).labels
    graphs = Graphs()
    m.compile(backend=graphs)
    torch.testing.assert_close(m(x[0]).labels, want)  # the default call: `runner.labels`
    assert graphs.n == 1  # one graph: the nested `__call__` traced inline, not compiled on its own
    m(x[0], logits=True)
    assert graphs.n == 2  # `__call__` compiles on its first direct call


class Prompt(_Net, PromptRunner):
    mask_size = (8, 8)

    def embed(self, image):
        return {"f": torch.nn.functional.interpolate(image, (8, 8)).mean(1, keepdim=True)}

    def predict(self, features, size, *, points, labels, boxes, masks, multimask):  # noqa: ARG002, PLR0913 - stub
        p = points.shape[0]
        f = features["f"].expand(p, 3 if multimask else 1, 8, 8) * 2 - 1
        return {"pred_masks": f, "iou_scores": f.mean((2, 3))}

    def upsample(self, pred_masks, size):
        return torch.nn.functional.interpolate(pred_masks, size)


def test_promptable_embed_and_predict_are_compiled(monkeypatch):
    m = _load(monkeypatch, promptable_segmentation, "sam2.1-hiera-tiny", Prompt)
    image, pts = torch.rand(3, 32, 32), torch.tensor([[[4.0, 5.0]]])
    want = m(image, input_points=pts, input_labels=torch.ones(1, 1)).masks
    graphs = Graphs()
    m.compile(backend=graphs)
    emb = m.get_image_embeddings(image)
    assert graphs.n == 1
    torch.testing.assert_close(m(emb, input_points=pts, input_labels=torch.ones(1, 1)).masks, want)
    assert graphs.n == 2  # the prompt decoder


class Scene(_Net, MultiViewRunner):
    intrinsics = cam_to_world = depth_range = "required"
    per_view = False
    min_views = 2
    compiled_methods = ("__call__", "prepare", "predict")

    def prepare(self, images):
        return images * 2

    def predict(self, x, k, depth_range):  # noqa: ARG002 - stub
        d = x[:1].mean(1) + depth_range[0]
        return {"depth": d, "confidence": d.sigmoid()}

    def __call__(self, images, intrinsics, cam_to_world, depth_range):  # noqa: ARG002 - stub
        return self.predict(self.prepare(images), intrinsics, depth_range)

    def scene(self, images, intrinsics, cam_to_world, depth_range):  # noqa: ARG002 - stub
        x = self.prepare(images)  # once per scene
        return lambda order: self.predict(x if order is None else x[order], intrinsics, depth_range)


def test_multiview_scene_is_compiled(monkeypatch):
    m = _load(monkeypatch, multiview, "patchmatchnet", Scene)
    images = torch.rand(3, 3, 32, 32)
    k, c2w, rng = torch.eye(3).expand(3, 3, 3), torch.eye(4).expand(3, 4, 4), (0.5, 4.0)
    want = m(images, intrinsics=k, cam_to_world=c2w, depth_range=rng, reference=[0, 2]).depth
    graphs = Graphs()
    m.compile(backend=graphs)
    m(images, intrinsics=k, cam_to_world=c2w, depth_range=rng)  # one reference: `__call__`
    assert graphs.n == 1
    got = m(images, intrinsics=k, cam_to_world=c2w, depth_range=rng, reference=[0, 2]).depth  # `scene`
    torch.testing.assert_close(got, want)
    assert graphs.n == 3  # `prepare` and `predict` of the scene path


class Stereo(_Net, StereoRunner):
    calls = 0

    def __call__(self, left, right):
        Stereo.calls += 1
        if Stereo.calls == 1:
            torch._dynamo.graph_break()  # fails under fullgraph=True
        return {"disparity": (left - right)[:, 0]}


def test_failed_first_compiled_call_restores_eager(monkeypatch):
    m = _load(monkeypatch, stereo, "s2m2-s", Stereo)
    base = type(m._runner)
    left, right = torch.rand(3, 32, 64), torch.rand(3, 32, 64)
    m.compile(fullgraph=True, backend="eager")
    Stereo.calls = 0
    with pytest.raises(Unsupported) as err:
        m(left, right)
    assert "runs eager again" in "".join(getattr(err.value, "__notes__", ()))
    assert type(m._runner) is base
    m(left, right)  # eager
    graphs = Graphs()
    assert m.compile(backend=graphs) is m  # compiling again works
    m(left, right)
    assert graphs.n >= 1


def test_repeat_and_concurrent_compile_wrap_once(monkeypatch):
    m = _load(monkeypatch, stereo, "s2m2-s", Stereo)
    base = type(m._runner)
    m.compile(backend="eager")
    graphs = Graphs()
    threads = [threading.Thread(target=m.compile, kwargs={"backend": graphs}) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert type(m._runner).__bases__ == (base,)
    Stereo.calls = 1  # no graph break
    m(torch.rand(3, 32, 64), torch.rand(3, 32, 64))
    assert graphs.n == 1  # the last compile's backend


@pytest.mark.parametrize("task", registry.TASKS)
def test_compiled_methods_exist(task):
    """Every portable runner defines the entry points it declares; a multi-view `scene` override declares more."""
    for card, _ in registry.cards(task).values():
        try:
            cls = core_model.runner_class(card, "portable")
        except BackendUnavailableError as e:  # an optional library is missing: check the other families
            if not isinstance(e.__cause__, ImportError):
                raise
            continue
        for name in cls.compiled_methods:
            assert callable(getattr(cls, name, None)), (card.family, name)
        if issubclass(cls, MultiViewRunner) and cls.scene is not MultiViewRunner.scene:
            assert len(cls.compiled_methods) > 1, card.family


@pytest.mark.parametrize(
    "module",
    [
        "fastforks.stereo.models.foundationstereo.net.core.extractor",
        "fastforks.stereo.models.monster_plusplus.net.submodule",
    ],
)
def test_resize_sizes_trace_as_ints(module):
    """FoundationStereo's ViT resize size is plain int math: dynamo traced `np.ceil` / `np.lcm` into tensors, and
    torch 2.14's guard on them failed under `inference_mode` ("Guard failed on the same frame it was created")."""
    import importlib  # noqa: PLC0415
    import math  # noqa: PLC0415

    resize = importlib.import_module(module).get_resize_keep_aspect_ratio

    def fn(x: torch.Tensor) -> torch.Tensor:
        h, w = resize(x.shape[-2], x.shape[-1], divider=math.lcm(14, 16), max_H=1344, max_W=1344)
        return x.new_zeros(h, w)

    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    with torch.inference_mode():
        for h, w in ((100, 150), (1500, 900)):
            assert compiled(torch.zeros(h, w)).shape == fn(torch.zeros(h, w)).shape
    assert resize(1500, 900, divider=112, max_H=1344, max_W=1344) == (1344, 896)


def test_great_igev_compiled_outlook_attention_matches_eager():
    """GREAT-IGEV's outlook attention under torch.compile (explicit softmax: Inductor's softmax(x * scale) lowering
    crashes Triton on its 49-wide rows) equals eager."""
    from fastforks.stereo.models.great_igev.net.transformers import OutlookAttentionLayer  # noqa: PLC0415

    torch.manual_seed(0)
    layer = OutlookAttentionLayer(8, 8, 8, kernel_size=7, padding=3).eval()
    x = torch.randn(1, 8, 9, 11)
    seen = []

    def backend(gm, _inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes)
        return gm

    with torch.no_grad():
        want = layer(x)
        got = torch.compile(layer, backend=backend, fullgraph=True)(x)
    assert seen
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) > (2, 8):  # torch 2.8's Dynamo takes the eager branch
        assert not [t for t in seen if "softmax" in t]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
