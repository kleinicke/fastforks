"""Validate the byte thresholds and the aggregation used by the BSDS500 scorer."""

import numpy as np
import pytest

from eval_edges import CUTOFFS, score_image, summarize


def test_byte_cutoffs_match_berkeley_score_thresholds():
    confidence = np.arange(256)
    thresholds = np.arange(1, 100) / 100
    assert np.array_equal(confidence[None] >= CUTOFFS[:, None], confidence[None] / 255 >= thresholds[:, None])


def test_ods_ois_and_ap_use_all_boundary_counts():
    # Both images have the same ideal threshold, with half their predictions on the true boundary.
    one = [[2, 2, 2, 4], [2, 2, 2, 4], [0, 2, 0, 0]]
    values = np.array([one, one], dtype=np.int64)
    assert summarize(values) == {"ods": 66.67, "ois": 66.67, "ap": 25.25}


def test_resume_uses_counts_published_by_another_worker(monkeypatch, tmp_path):
    scipy = pytest.importorskip("scipy.io")
    predicted = tmp_path / "input.npy"
    labels = tmp_path / "reference.mat"
    np.save(predicted, np.zeros((2, 2), dtype=np.uint8))
    scipy.savemat(labels, {"groundTruth": {"Boundaries": np.ones((2, 2), dtype=np.uint8)}})
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = np.tile([0, 4, 0, 0], (len(CUTOFFS), 1))
    load = np.load

    def concurrent_load(path, **options):
        if path.name.endswith(".partial.npy"):
            np.save(path.with_name(path.name.replace(".partial.npy", ".npy")), expected)
            raise FileNotFoundError(path)
        return load(path, **options)

    monkeypatch.setattr(np, "load", concurrent_load)
    np.testing.assert_array_equal(score_image((predicted, labels, cache, tmp_path / "unused.so"), "test"), expected)
