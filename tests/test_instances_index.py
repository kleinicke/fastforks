"""Indexing `Instances` with ints, 0-d integer tensors and numpy integer scalars keeps the instance dimension."""

import numpy as np
import torch

from fastforks._core.instances import Instances


def _inst(n=3):
    return Instances(torch.rand(n, 4), torch.tensor([0.2, 0.9, 0.5])[:n], torch.arange(n), None, (), "m@1", None)


def test_scalar_indices_keep_the_instance_dimension():
    r = _inst()
    for idx in (1, np.int64(1), torch.tensor(1), r.scores.argmax()):
        best = r[idx]
        assert len(best) == 1
        assert best.boxes.shape == (1, 4)
        assert best.scores.shape == (1,)
    assert r[-1].scores.shape == (1,)
    assert len(r[r.scores > 0.4]) == 2
    assert len(r[torch.tensor([0, 2])]) == 2
