"""Calibration accuracy must not silently score invalid model predictions."""

import numpy as np
import pytest

from eval_calibration import available_angles


def test_available_angles_rejects_nonfinite_predictions():
    errors = np.array([[np.nan, np.nan, 2.0], [np.nan, np.nan, 1.0]])
    assert available_angles(errors) == [2]
    errors[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        available_angles(errors)
    errors[:, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        available_angles(errors)
