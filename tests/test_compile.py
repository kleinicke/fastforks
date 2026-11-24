"""`Model.compile()` on stub runners (CPU, no weights): every task's default call path runs the compiled entry points,
nested entry points trace inline, a failed first compiled call leaves the model eager, a repeat compile replaces.

Real families: `test_export.test_model_compile_equals_eager`; here, every runner's `compiled_methods` exist and every
override of a task entry point is compiled or marked `@host`.
"""

from __future__ import annotations

import dataclasses
import importlib
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
from fastforks._core.runner import host
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


def test_pipeline_can_compile_without_an_exportable_module(monkeypatch):
    class Pipeline(SegRunner):
        compiled_methods = ("_scores",)

        def __init__(self, ctx):
            super().__init__(ctx)
            self.net = torch.nn.Conv2d(3, 150, 1)

        def networks(self):
            return (self.net,)

        def _scores(self, image):
            return self.net(image)

        @host
        def __call__(self, image):
            return {"logits": self._scores(image)}

    model = _load(monkeypatch, segmentation, "fcn-r101-d8-ade20k", Pipeline)
    image = torch.rand(3, 32, 32)
    expected = model(image).labels
    graphs = Graphs()
    model.compile(backend=graphs)
    torch.testing.assert_close(model(image).labels, expected)
    assert graphs.n == 1
    assert not any(p.requires_grad for p in model._runner.net.parameters())
    with pytest.raises(BackendUnavailableError, match="no exportable module"):
        model.module()


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
    k = torch.tensor([[32.0, 0, 16], [0, 32, 16], [0, 0, 1]]).expand(3, 3, 3)  # pixel intrinsics of a 32 x 32 view
    c2w, rng = torch.eye(4).expand(3, 4, 4), (0.5, 4.0)
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


# host code around the entry points
HELPERS = {"module", "input_use", "pad", "resize", "restore", "upsample", "binary_masks", "scene", "segment_concept"}


def _is_host(cls: type, name: str) -> bool:
    return bool(getattr(getattr(cls, name), "ff_host", False))


@pytest.mark.parametrize("task", registry.TASKS)
def test_overridden_entry_points_are_compiled(task):
    """A family override of a task entry point (`__call__`, segmentation `labels`, ...) is in `compiled_methods`,
    unless it is marked `@host` (loops, tokenizers, solvers: it lists the methods it calls instead).

    The 2026-11-04 segmentation compile rows timed eager code: the default call ran `labels` overrides, not `__call__`;
    the 2026-11-14 ports listed private network methods (or the `net` attribute: `compile()` raised) instead.
    """
    for card, _ in registry.cards(task).values():
        try:
            cls = core_model.runner_class(card, "portable")
        except BackendUnavailableError as e:
            if not isinstance(e.__cause__, ImportError):
                raise
            continue
        base = importlib.import_module(f"fastforks.{task}._runner").Runner
        names = {n for n, v in vars(base).items() if callable(v) and (n == "__call__" or not n.startswith("_"))}
        if set(getattr(cls, "prompts", ())) - {"points", "boxes", "masks"}:  # concept / reference models: all
            names.discard("predict")  # prompts go to `segment_concept` / `segment_reference`, `predict` only raises
        overridden = {n for n in names - HELPERS if getattr(cls, n) is not getattr(base, n)}
        missed = {n for n in overridden if not _is_host(cls, n)} - set(cls.compiled_methods)
        assert not missed, (card.family, missed)
        hosts = {n for n in cls.compiled_methods if _is_host(cls, n)}
        assert not hosts, (card.family, hosts)  # host code is not compiled


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


@pytest.mark.parametrize("scheduler", ["ddim", "lcm"])
def test_marigold_scheduler_steps_trace_without_cpu_selects(scheduler):
    """The DDIM / LCM coefficients enter the graph as 0-d constants, inlined as literals: diffusers selects them from
    the CPU `alphas_cumprod` per step, and Inductor fuses that select into the CUDA kernel with the CPU buffer as a
    device pointer ("CUDA driver error: invalid argument", torch 2.14). The traced steps hold no select / index and
    no tensor constant with a dimension."""
    if scheduler == "ddim":
        from fastforks.depth.models.marigold.net.ddim import DDIM  # noqa: PLC0415

        ddim = DDIM(4)

        def step(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            for t in ddim.timesteps:
                x = ddim.step(v, t, x)
            return x

    else:
        from fastforks.depth.models.marigold_normals.net.lcm import LCM  # noqa: PLC0415

        lcm = LCM(4)

        def step(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            for t in lcm.timesteps:
                x = lcm.step(v, t, x, None if t == lcm.timesteps[-1] else v)
            return x

    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm

    v, x = (torch.randn(1, 4, 8, 8, dtype=torch.float16) for _ in range(2))
    got = torch.compile(step, backend=backend, fullgraph=True)(v, x)
    assert len(graphs) == 1
    gm = graphs[0]
    bad = [n for n in gm.graph.nodes if n.op != "get_attr" and ("select" in str(n.target) or "index" in str(n.target))]
    bad += [n for n in gm.graph.nodes if n.op == "get_attr" and getattr(gm, n.target).dim() != 0]
    assert not bad, bad
    torch.testing.assert_close(got, step(v, x), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.slow
def test_triton_scaled_softmax_rows_bitcast(tmp_path):
    """Sentinel for the GREAT-IGEV outlook workaround: Inductor's `softmax(x * scale)` lowering (a masked boolean
    any-reduction beside two max-reductions and an exp; `_partial_softmax_pattern`) aborts Triton 3.8's LLVM on
    49-wide rows with XBLOCK=8 / 2 warps ("Cannot BITCAST between types of different sizes!"). The kernel runs in a
    subprocess because the LLVM assertion aborts the process. Triton 3.9 compiles it. The workaround remains
    for the release toolchain, which uses Triton 3.8."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    triton = pytest.importorskip("triton")
    release = tuple(int(part) for part in triton.__version__.split(".")[:2])
    if release not in {(3, 8), (3, 9)}:
        pytest.skip(f"scaled-softmax behavior verified on Triton 3.8 and 3.9, got {triton.__version__}")

    src = textwrap.dedent(
        """
        import torch, triton, triton.language as tl

        @triton.jit
        def _or(a, b):
            return a | b

        @triton.jit
        def _max(a, b):
            return tl.maximum(a, b)

        @triton.jit
        def k(in0, in1, out, xnumel, XBLOCK: tl.constexpr):
            xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
            r = tl.arange(0, 64)[None, :]
            mask = (r < 49) & (xindex < xnumel)
            x = tl.load(in0 + (r + 49 * xindex), mask, other=0.0).to(tl.float32)
            x = x + tl.load(in1 + xindex // 49, mask, other=0.0).to(tl.float32)
            s = x * 0.3535533905932738
            bad = ((s == s) & (tl.abs(s) != float("inf"))) == 0
            nonfinite = tl.reduce(tl.where(mask, bad, False), 1, _or)[:, None]
            m1 = tl.reduce(tl.where(mask, x, float("-inf")), 1, _max)[:, None]
            m2 = tl.reduce(tl.where(mask, s, float("-inf")), 1, _max)[:, None]
            e = tl.where(nonfinite == 0, (x - m1) * 0.3535533905932738, s - m2)
            tl.store(out + (r + 49 * xindex), tl.exp(e), mask)

        n = 265825
        a = torch.randn(n * 49, device="cuda", dtype=torch.float16)
        b = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.zeros(n * 49, device="cuda")
        k[(triton.cdiv(n, 8),)](a, b, out, n, 8, num_warps=2)
        torch.cuda.synchronize()
        print("compiled")
        """
    )
    script = tmp_path / "scaled_softmax.py"
    script.write_text(src, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=600, check=False
    )
    if release == (3, 8):
        assert r.returncode != 0, r.stdout[-2000:]
        assert "BITCAST" in r.stderr, r.stderr[-2000:]
    else:
        assert r.returncode == 0, r.stderr[-2000:]
        assert "compiled" in r.stdout, r.stdout[-2000:]
