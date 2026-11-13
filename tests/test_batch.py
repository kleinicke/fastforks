"""A batch of two gives the same maps as two single calls (CPU, portable, smallest variant per family, all tasks)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from synth import image, pair, prompt_kwargs, size
from test_golden import FAMILY, SMALLEST, TASK, sweep_model

LOAD_ONCE = True  # conftest runs the tests of one ref together: `sweep_model` loads it once

SIZE = (67, 99)  # small and odd: exercises padding and cropping


def _maps(result: object) -> dict[str, torch.Tensor]:
    """Every tensor of a result (standard fields and extras)."""
    maps = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    maps |= dict(maps.pop("extras", None) or {})
    return {k: v for k, v in maps.items() if isinstance(v, torch.Tensor)}


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", SMALLEST)
def test_batch_equals_singles(ref):
    task = TASK[ref]
    if task == "promptable_segmentation":
        pytest.skip("one image per call with prompts (batched prompts: tests/test_tasks.py)")
    if task == "tracking":
        pytest.skip("tracking: one video per call (its frames are the leading dim)")
    if task == "multiview":
        pytest.skip("multiview: the leading dim holds the views of one scene (they attend to each other), not a batch")
    if FAMILY[ref] == "video_depth_anything":
        pytest.skip("video depth: the batch is one video in time order (tests/test_depth_video_depth_anything.py)")
    model = sweep_model(ref)
    hw = size(model.variant.io.min_size, *SIZE)
    inputs = [
        pair(*hw, seed=s) if task in ("stereo", "flow", "matching", "interpolation") else (image(*hw, seed=s),)
        for s in (0, 1)
    ]
    boxes = [[0.1 * hw[1], 0.1 * hw[0], 0.9 * hw[1], 0.9 * hw[0]]]  # one person box per image
    extra = {"boxes": [boxes, boxes]} if task == "pose" and model.needs_boxes else {}
    kws = [prompt_kwargs(model, s) for s in (0, 1)]  # a depth prompt per image, if the model needs one
    extra |= {k: np.stack([kw[k] for kw in kws]) for k in kws[0]}
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
    batch = _maps(batch)
    for i, single in enumerate(_maps(model(*x, **kw)) for x, kw in zip(inputs, kws, strict=True)):
        assert set(single) == set(batch)
        for k, v in single.items():
            assert batch[k].shape == (2, *v.shape)
            torch.testing.assert_close(batch[k][i], v, rtol=1e-4, atol=1e-3)
