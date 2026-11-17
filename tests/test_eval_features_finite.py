"""Feature accuracy must not turn non-finite vectors into a reported kNN score."""

import numpy as np
import pytest
import torch

import eval_features


def test_knn_rejects_nonfinite_cached_features() -> None:
    labels = torch.tensor([3, 4, 3])
    valid = {"feat": np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float16), "nonfinite": 0}
    assert eval_features._score_parts([valid], labels, 2, 1, weighted=True, device="cpu") == 100.0
    invalid = {"feat": valid["feat"].copy(), "nonfinite": 1}
    invalid["feat"][2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite features"):
        eval_features._score_parts([invalid], labels, 2, 1, weighted=True, device="cpu")
    invalid["nonfinite"] = 0  # catch corrupt or stale chunk metadata too
    with pytest.raises(ValueError, match="non-finite features"):
        eval_features._score_parts([invalid], labels, 2, 1, weighted=True, device="cpu")
