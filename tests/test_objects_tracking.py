"""Multi-object tracking (`tracking.ByteTracker`, `shared.box_kalman`): assignment, Kalman filter, reference values."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from fastforks._core.errors import InputError
from fastforks.shared import box_kalman
from fastforks.tracking import ByteTracker, ObjectTracks, track_objects
from fastforks.tracking.objects import linear_assignment


def _brute(cost: np.ndarray, limit: float) -> float:
    """Best objective `sum(cost - limit)` over matched pairs with `cost < limit` (exhaustive)."""
    n, m = cost.shape
    best = 0.0
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] < limit]
                best = min(best, sum(cost[r, c] - limit for r, c in pairs))
    return best


def test_linear_assignment_is_optimal() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = rng.integers(0, 5, 2)
        cost = rng.random((n, m))
        cost[rng.random((n, m)) < 0.4] = 1.0
        limit = float(rng.choice([0.5, 0.9]))
        matches, rows, cols = linear_assignment(cost, limit)
        assert sorted(rows + [r for r, _ in matches]) == list(range(n))
        assert sorted(cols + [c for _, c in matches]) == list(range(m))
        assert all(cost[r, c] < limit for r, c in matches)
        assert sum(cost[r, c] - limit for r, c in matches) == pytest.approx(_brute(cost, limit), abs=1e-12)


def test_kalman_follows_constant_velocity() -> None:
    truth = np.array([[100.0, 50.0, 0.5, 80.0]])
    mean, cov = box_kalman.initiate(truth)
    for _ in range(30):
        truth = truth + np.array([[3.0, -1.0, 0.0, 0.5]])
        mean, cov = box_kalman.predict(mean, cov)
        mean, cov = box_kalman.update(mean, cov, truth)
    np.testing.assert_allclose(mean[0, 4:], [3.0, -1.0, 0.0, 0.5], atol=0.05)
    np.testing.assert_allclose(cov, cov.transpose(0, 2, 1), atol=1e-9)
    xyxy = np.array([[10.0, 20.0, 50.0, 100.0]])
    np.testing.assert_allclose(box_kalman.to_xyxy(box_kalman.to_xyah(xyxy)), xyxy)


def _scene(seed: int, frames: int = 120, n: int = 12) -> list[np.ndarray]:
    """Random walks with misses, noise, low scores and false positives (as the reference parity run)."""
    r = np.random.default_rng(seed)
    pos = r.random((n, 2)) * 800
    vel = r.normal(0, 3, (n, 2))
    size = r.uniform(30, 150, (n, 1)) * [0.4, 1]
    out = []
    for _ in range(frames):
        pos = pos + vel
        vis = r.random(n) > 0.1
        b = np.concatenate([pos, pos + size], 1)[vis] + r.normal(0, 2, (vis.sum(), 4))
        s = r.uniform(0.05, 1.0, vis.sum())
        fp = r.random((3, 4)) * 800
        fp[:, 2:] = fp[:, :2] + 60
        det = np.concatenate([np.concatenate([b, s[:, None]], 1), np.concatenate([fp, r.uniform(0, 0.7, (3, 1))], 1)])
        out.append(det)
    return out


# The official ByteTrack BYTETracker (d1bf0191, MOT17 defaults) on `_scene(seed)`: track rows, sum of ids, largest id,
# sum of box coordinates over all frames.
REFERENCE = {0: (895, 11782, 30, 1627030.98), 1: (939, 11555, 31, 1427041.81), 2: (972, 14044, 42, 1677609.3)}


@pytest.mark.parametrize("seed", sorted(REFERENCE))
def test_matches_reference_bytetrack(seed: int) -> None:
    tracker = ByteTracker()
    rows = idsum = maxid = 0
    boxsum = 0.0
    for det in _scene(seed):
        t = tracker.update(det)
        assert isinstance(t, ObjectTracks)
        assert t.labels is None
        rows += len(t)
        idsum += int(t.ids.sum())
        maxid = max(maxid, int(t.ids.max()) if len(t) else 0)
        boxsum += float(t.boxes.double().sum())
    assert (rows, idsum, maxid) == REFERENCE[seed][:3]
    assert boxsum == pytest.approx(REFERENCE[seed][3], abs=0.1 * rows)  # float32 boxes


def test_identity_survives_a_gap() -> None:
    box = np.array([100.0, 100.0, 140.0, 200.0])
    step = np.array([2.0, 0, 2, 0])
    frames = []
    for f in range(40):
        d = [[*(box + f * step), 0.9], [*(box + 200 * step - f * step), 0.9]]
        frames.append(np.array(d[:1] if 15 <= f < 20 else d))  # the second object is missed 5 frames
    out = track_objects(frames)
    assert out[0].ids.tolist() == [1, 2]
    assert out[17].ids.tolist() == [1]
    assert sorted(out[-1].ids.tolist()) == [1, 2]


def test_low_score_detections_extend_tracks() -> None:
    tracker = ByteTracker()
    box = [100.0, 100.0, 140.0, 200.0]
    tracker.update(np.array([[*box, 0.9]]))
    assert tracker.update(np.array([[*box, 0.3]])).ids.tolist() == [1]  # second association
    tracker.reset()
    assert len(tracker.update(np.array([[*box, 0.65]]))) == 0  # below track_thresh + 0.1: no new track


def test_detection_result_and_labels() -> None:
    class Dets:
        def __init__(self, boxes: list, scores: list, labels: list) -> None:
            self.boxes, self.scores, self.labels = torch.tensor(boxes), torch.tensor(scores), torch.tensor(labels)

        def __getitem__(self, keep: torch.Tensor) -> Dets:
            return Dets(self.boxes[keep].tolist(), self.scores[keep].tolist(), self.labels[keep].tolist())

    frames = [Dets([[0.0, 0, 10, 20], [50, 50, 70, 90]], [0.9, 0.9], [0, 2])] * 3
    out = track_objects(frames, labels=[2])
    assert out[-1].labels.tolist() == [2]
    assert out[-1].boxes.tolist() == [[50.0, 50.0, 70.0, 90.0]]
    with pytest.raises(InputError):
        track_objects([np.zeros((0, 5))], labels=[0])


@pytest.mark.parametrize("bad", [np.zeros((3, 4)), [[0, 0, 1, 1, np.nan]], "boxes"])
def test_bad_detections(bad: object) -> None:
    with pytest.raises(InputError):
        ByteTracker().update(bad)
