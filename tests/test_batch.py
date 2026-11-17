"""A batch of two gives the same maps as two single calls (CPU, portable, smallest variant per family, all tasks)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from synth import image, pair, prompt_kwargs, size
from test_golden import SMALLEST, TASK, sweep_model

LOAD_ONCE = True  # conftest runs the tests of one ref together: `sweep_model` loads it once

_BATCH_COUPLED = {  # the batch is one problem, in the pinned upstream design
    "geocalib-pinhole@1": "the Levenberg-Marquardt damping, singularity check and early stop are batch-wide",
}
SIZE = (67, 99)  # small and odd: exercises padding and cropping


def _maps(result: object) -> dict[str, torch.Tensor]:
    """Every tensor of a result (standard fields and extras)."""
    maps = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    maps |= dict(maps.pop("extras", None) or {})
    return {k: v for k, v in maps.items() if isinstance(v, torch.Tensor)}


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.filterwarnings("ignore:.*without intrinsics")  # metric3d / da3metric: the assumed camera is fine here
@pytest.mark.filterwarnings("ignore:.*is a video model but a batch of")  # compare independent images, not clips
@pytest.mark.parametrize("ref", SMALLEST)
def test_batch_equals_singles(ref):  # noqa: C901 - one branch per task family
    task = TASK[ref]
    if task == "promptable_segmentation":
        pytest.skip("one image per call with prompts (batched prompts: tests/test_tasks.py)")
    if task == "tracking":
        pytest.skip("tracking: one video per call (its frames are the leading dim)")
    if task == "volume":
        pytest.skip("volume: one volume per call")
    if task == "multiview":
        pytest.skip("multiview: the leading dim holds the views of one scene (they attend to each other), not a batch")
    if ref in _BATCH_COUPLED:
        pytest.xfail(_BATCH_COUPLED[ref])
    model = sweep_model(ref)
    hw = size(model.variant.io.min_size, *SIZE)
    inputs = [
        pair(*hw, seed=s) if task in ("stereo", "flow", "matching", "interpolation") else (image(*hw, seed=s),)
        for s in (0, 1)
    ]
    boxes = [[0.1 * hw[1], 0.1 * hw[0], 0.9 * hw[1], 0.9 * hw[0]]]  # one person box per image
    extra = {"boxes": [boxes, boxes]} if task == "pose" and model.needs_boxes else {}
    guide = np.zeros(hw, np.uint8)  # inpainting / matting models that require a mask or trimap: the centre block
    guide[hw[0] // 4 : 3 * hw[0] // 4, hw[1] // 4 : 3 * hw[1] // 4] = 255
    guides = {n: guide for n in ("mask", "trimap") if getattr(model._runner, n, None) == "required"}
    kws = [prompt_kwargs(model, s) | guides for s in (0, 1)]  # a depth prompt per image, if the model needs one
    extra |= {k: np.stack([kw[k] for kw in kws]) for k in kws[0]}
    extra |= {k: v[..., None] for k, v in extra.items() if k in guides}  # a gray batch is BHW1
    batch = model(*(np.stack(xs) for xs in zip(*inputs, strict=True)), **extra)
    if isinstance(batch, list):  # per-image results (detection, pose, sparse matching: their counts differ)
        assert isinstance(batch, list)
        assert len(batch) == len(inputs)
        for got, x in zip(batch, inputs, strict=True):
            single = _maps(model(*x, **({"boxes": boxes} if task == "pose" and model.needs_boxes else {})))
            assert set(single) == set(_maps(got))
            for k, v in single.items():
                torch.testing.assert_close(_maps(got)[k], v, rtol=1e-4, atol=1e-3)
        return
    ragged = {
        f.name: getattr(batch, f.name)
        for f in dataclasses.fields(batch)
        if isinstance(getattr(batch, f.name), list)
        and getattr(batch, f.name)
        and all(isinstance(v, torch.Tensor) for v in getattr(batch, f.name))
    }
    batch = _maps(batch)
    for i, single in enumerate(_maps(model(*x, **kw)) for x, kw in zip(inputs, kws, strict=True)):
        assert set(single) == set(batch) | set(ragged)
        for k, v in single.items():
            if k in ragged:
                assert len(ragged[k]) == len(inputs)
                torch.testing.assert_close(ragged[k][i], v, rtol=1e-4, atol=1e-3)
                continue
            assert batch[k].shape == (2, *v.shape)
            torch.testing.assert_close(batch[k][i], v, rtol=1e-4, atol=1e-3)
