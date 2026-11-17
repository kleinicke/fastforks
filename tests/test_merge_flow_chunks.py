"""Full flow metrics must never be assembled from partial or overlapping chunks."""

import pytest
from tools.merge_flow_chunks import merge


def test_merge_flow_chunks():
    base = {
        "ref": "example",
        "source_id": "source",
        "task": "flow",
        "dataset": "kitti2015-train",
        "n": 1,
        "peak_alloc_mb": 4,
        "peak_reserved_mb": 6,
        "date": "today",
        "command": "run",
    }
    a = {**base, "pair_stats": [{"id": "a", "epe": 1.0, "outliers": 1, "valid": 10, "ms": 2.0}]}
    b = {**base, "pair_stats": [{"id": "b", "epe": 3.0, "outliers": 1, "valid": 30, "ms": 4.0}]}
    row = merge([a, b], {"a", "b"})
    assert row["metrics"] == {"epe": 2.0, "fl_all": 5.0}
    assert row["ms_per_img"] == 3.0
    with pytest.raises(ValueError, match="duplicate"):
        merge([a, a], {"a", "b"})
    with pytest.raises(ValueError, match="coverage"):
        merge([a], {"a", "b"})
    with pytest.raises(ValueError, match="incompatible"):
        merge([a, {**b, "ref": "different"}], {"a", "b"})
    with pytest.raises(ValueError, match="source identifier"):
        merge([{**a, "source_id": None}], {"a"})
