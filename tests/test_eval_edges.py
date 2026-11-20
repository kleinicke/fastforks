"""Validate the byte thresholds and the aggregation used by the BSDS500 scorer."""

import numpy as np

from eval_edges import CUTOFFS, summarize


def test_byte_cutoffs_match_berkeley_score_thresholds():
    confidence = np.arange(256)
    thresholds = np.arange(1, 100) / 100
    assert np.array_equal(confidence[None] >= CUTOFFS[:, None], confidence[None] / 255 >= thresholds[:, None])


def test_ods_ois_and_ap_use_all_boundary_counts():
    # Both images have the same ideal threshold, with half their predictions on the true boundary.
    one = [[2, 2, 2, 4], [2, 2, 2, 4], [0, 2, 0, 0]]
    values = np.array([one, one], dtype=np.int64)
    assert summarize(values) == {"ods": 66.67, "ois": 66.67, "ap": 25.25}
