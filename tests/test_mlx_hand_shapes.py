"""Hand-written MLX ports against a reference at an odd input size and at batch 2.

A port's goldens run at one size and batch 1, so a shape-specialized `mx.compile`, a size-keyed RoPE or positional
table, a padding path or a batch folded into another axis can pass them and still be wrong elsewhere. Each `SHAPES`
entry runs the hand port in float32 on the task's `bench_inputs` at its odd size and, unless `batch` is off, on a batch
of two (the input and its up-down mirror stacked on the batch axis: a left-right mirror leaves a stereo pair without a
valid disparity). The reference is the family's graph arm
(`FASTFORKS_MLX_GRAPH=1`) where it has one, else the PyTorch portable network on the CPU, both in float32. Every output
must agree within `tol` of the reference output's largest magnitude; an integer output (a label map) may differ in at
most `LABELS` of its elements.

Add a family: one `Shape(ref, (h, w))` line (`batch=False` when the task's call does not take a batch, `tol` for an
iterative network whose float noise grows). Needs mlx and the weights: `pytest tests/test_mlx_hand_shapes.py --slow`
(a GPU job: under the GPU lock).
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.model, pytest.mark.mlx, pytest.mark.slow]


@dataclass(frozen=True)
class Shape:
    """One family's check: a small ref, an odd `(h, w)` input, whether batch 2 is checked, the relative tolerance."""

    ref: str
    size: tuple[int, int]
    batch: bool = True
    tol: float = 5e-3
    views: int = 2


LABELS = 1e-3  # integer outputs (label maps): at most this share of elements may differ (ties at a boundary)

SHAPES = [
    Shape("depth-anything-v2-s", (251, 333)),
    Shape("dinov2-vits14-reg", (227, 301)),
    Shape("crestereo-eth3d", (243, 325), tol=2e-2),
    Shape("rife-4.25", (247, 331)),
    Shape("ema-vfi-small", (245, 339)),
    Shape("depth-pro", (251, 333)),
    Shape("totalsegmentator-total", (61, 75), batch=False),
]


def _flat(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    import torch  # noqa: PLC0415

    if isinstance(tree, torch.Tensor):
        tree = (tree.float() if tree.is_floating_point() else tree).detach().cpu().numpy()  # bfloat16: no numpy
    if isinstance(tree, np.ndarray) and tree.dtype != object:
        return {prefix or "out": tree if tree.dtype.kind in "iub" else tree.astype(np.float32)}
    if hasattr(tree, "__dataclass_fields__"):
        tree = {k: getattr(tree, k) for k in tree.__dataclass_fields__}
    if isinstance(tree, (list, tuple)):
        tree = dict(enumerate(tree))
    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out |= _flat(v, f"{prefix}.{k}" if prefix else str(k))
    return out


def _batch2(x: Any) -> Any:
    """A batch-1 image tensor as the batch of it and its up-down mirror; anything else unchanged."""
    import torch  # noqa: PLC0415

    if isinstance(x, torch.Tensor) and x.ndim >= 3 and x.shape[0] == 1:
        return torch.cat([x, x.flip(-2)])
    return x


def _calls(model: Any, inputs: tuple, *, batch: bool) -> list[dict[str, np.ndarray] | None]:
    """The flattened outputs at batch 1, and at batch 2 when `batch` (else None)."""
    import torch  # noqa: PLC0415

    args, kw = inputs
    with torch.inference_mode():
        one = _flat(model(*args, **kw))
        two = _flat(model(*[_batch2(a) for a in args], **{k: _batch2(v) for k, v in kw.items()})) if batch else None
    return [one, two]


def _check(got: dict[str, np.ndarray], want: dict[str, np.ndarray], tol: float, what: str) -> list[str]:
    bad = []
    for k, w in want.items():
        g = got.get(k)
        if g is None or g.shape != w.shape:
            bad.append(f"{what} {k}: shape {None if g is None else g.shape} vs {w.shape}")
            continue
        if w.dtype.kind in "iub":  # labels or masks: the share of elements that differ
            share = float(np.mean(g != w))
            if not share <= LABELS:
                bad.append(f"{what} {k}: {share:.2e} of the elements differ (tol {LABELS:g})")
            continue
        rel = float(np.max(np.abs(g.astype(np.float64) - w))) / max(float(np.max(np.abs(w))), 1e-12)
        if not rel <= tol:
            bad.append(f"{what} {k}: max |diff| {rel:.2e} of the reference's largest magnitude (tol {tol:g})")
    return bad


@pytest.mark.parametrize("case", SHAPES, ids=lambda c: c.ref)
def test_hand_matches_reference_at_odd_size_and_batch_2(case: Shape, monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("no mlx")
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    card, _ = registry.resolve_any(case.ref)
    task = importlib.import_module(f"fastforks.{card.task}")
    graph = "portable_mlx" in getattr(registry.import_model(card), "GRAPH_RUNNERS", {})
    monkeypatch.setenv("FASTFORKS_MLX_GRAPH", "0")
    model = task.load(case.ref, mode="portable", device="mlx", dtype="float32")
    inputs = bench_inputs(model, BenchSize(*case.size, case.views))
    hand = _calls(model, inputs, batch=case.batch)
    del model
    gc.collect()
    if graph:
        monkeypatch.setenv("FASTFORKS_MLX_GRAPH", "1")
        ref = task.load(case.ref, mode="portable", device="mlx", dtype="float32")
    else:
        ref = task.load(case.ref, mode="portable", device="cpu", dtype="float32")
    want = _calls(ref, inputs, batch=case.batch)
    name = "graph" if graph else "torch cpu"
    h, w = case.size
    bad = _check(hand[0], want[0] or {}, case.tol, f"vs {name} {h}x{w}:")
    if hand[1] is not None:
        bad += _check(hand[1], want[1] or {}, case.tol, f"vs {name} {h}x{w} batch 2:")
    assert not bad, "\n".join(bad)
