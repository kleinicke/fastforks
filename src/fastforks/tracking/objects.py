"""Multi-object tracking by detection: ByteTrack association of per-frame detections into tracks with identities.

```python
from fastforks import detection, tracking

detector = detection.load("dfine-x-coco", options={"score_threshold": 0.1})  # ByteTrack also uses low-score boxes
frames = tracking.track_objects(video, detector, labels=[0])  # one ObjectTracks per frame (COCO class 0: person)
frames[10].ids, frames[10].boxes  # [K] int64 track ids, [K, 4] (x0, y0, x1, y1) pixels

tracker = tracking.ByteTracker(frame_rate=30)  # online: one update per frame
for frame in video:
    tracks = tracker.update(detector(frame))  # a DetectionResult, or [N, 5] (x0, y0, x1, y1, score)
```

The algorithm is ByteTrack's (Zhang et al. 2022, MIT): a constant-velocity Kalman filter per track
(`fastforks.shared.box_kalman`, all tracks of a frame at once), a first association of the high-score detections to
the tracked and lost tracks (IoU fused with the score, optimal assignment), a second one of the low-score detections
to the remaining tracked tracks, then the tentative tracks of the last frame, new tracks from unmatched high-score
detections, and lost tracks dropped after `track_buffer` frames. It reproduces the reference `BYTETracker` exactly,
including its IoU with the `+1` pixel convention. The optimal assignment (as `lap.lapjv` with a cost limit) splits
the cost matrix into its connected blocks and solves each with a shortest augmenting path in numpy (no scipy); a
frame costs well under a millisecond on the CPU, so the tracker runs there (a GPU round trip per frame costs more).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.errors import InputError
from fastforks.shared import box_kalman

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = ["ByteTracker", "ObjectTracks", "linear_assignment", "track_objects"]

LOW_SCORE = 0.1  # fixed: reference floor of the second association's detections
SECOND_MATCH = 0.5  # fixed: reference IoU-distance limit of the second association
TENTATIVE_MATCH = 0.7  # fixed: reference limit for the tentative (one-frame) tracks
DUPLICATE = 0.15  # fixed: reference IoU distance below which a tracked and a lost track are duplicates
NEW_TRACK_MARGIN = 0.1  # fixed: a new track needs a score above track_thresh + this (reference det_thresh)
_TRACKED, _LOST, _REMOVED = 1, 2, 3


@dataclass(frozen=True)
class ObjectTracks:
    """The confirmed tracks of one frame (CPU tensors).

    Attributes:
        ids: `[K]` int64 track identities (from 1, unique within a tracker).
        boxes: `[K, 4]` float32 `(x0, y0, x1, y1)` in pixels: the Kalman filter's box after this frame's detection.
        scores: `[K]` float32 score of the detection the track last matched.
        labels: `[K]` int64 class of that detection, `None` for `[N, 5]` detections without classes.
    """

    ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor | None

    def __len__(self) -> int:
        """Number of tracks in the frame."""
        return int(self.ids.shape[0])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`[N, M]` IoU of `[N, 4]` and `[M, 4]` boxes, with the reference's inclusive `+1` pixel extents."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return np.where(inter > 0, inter / (area_a[:, None] + area_b[None] - inter), 0.0)


def _square_assignment(cost: np.ndarray) -> np.ndarray:
    """Column of every row of the min-cost perfect matching of a square matrix (shortest augmenting paths)."""
    n = cost.shape[0]
    u, v = np.zeros(n), np.zeros(n)
    col4row, row4col = np.full(n, -1), np.full(n, -1)
    for start in range(n):
        shortest = np.full(n, np.inf)
        path = np.full(n, -1)
        seen = np.zeros(n, dtype=bool)
        rows = []
        i, low, sink = start, 0.0, -1
        while sink < 0:
            rows.append(i)
            reduced = low + cost[i] - u[i] - v
            better = ~seen & (reduced < shortest)
            path[better] = i
            shortest[better] = reduced[better]
            free = np.flatnonzero(~seen)
            j = int(free[np.argmin(shortest[free])])
            low = shortest[j]
            seen[j] = True
            if row4col[j] < 0:
                sink = j
            else:
                i = int(row4col[j])
        u[start] += low
        for r in rows[1:]:
            u[r] += low - shortest[col4row[r]]
        v[seen] -= low - shortest[seen]
        j = sink
        while True:
            i = int(path[j])
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == start:
                break
    return col4row


def linear_assignment(  # noqa: C901 - union-find, blocks, one solve per block
    cost: np.ndarray, limit: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Min-cost matching of rows to columns where only pairs with `cost < limit` may match.

    It minimises the summed cost of the matched pairs plus `limit / 2` per unmatched row and column, as
    `lap.lapjv(cost, extend_cost=True, cost_limit=limit)` does: the pairs that can match form independent blocks
    (connected components), and each block is solved exactly.

    Returns:
        `(matches, unmatched_rows, unmatched_cols)`: `(row, col)` pairs sorted by row, and the sorted leftovers.
    """
    n, m = cost.shape
    rows, cols = np.nonzero(cost < limit)
    if len(rows) == 0:
        return [], list(range(n)), list(range(m))
    parent = list(range(n + m))  # union-find over rows 0..n-1 and columns n..n+m-1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        parent[find(r)] = find(n + c)
    blocks: dict[int, tuple[list[int], list[int]]] = {}
    for r in sorted(set(rows.tolist())):
        blocks.setdefault(find(r), ([], []))[0].append(r)
    for c in sorted(set(cols.tolist())):
        blocks[find(n + c)][1].append(c)
    match_of = {}
    for br, bc in blocks.values():
        if len(br) == 1 and len(bc) == 1:
            match_of[br[0]] = bc[0]
            continue
        sub = np.minimum(cost[np.ix_(br, bc)] - limit, 0.0)  # a pair is worth only its saving below the limit
        k = len(br) + len(bc)
        square = np.zeros((k, k))  # rows / columns may stay unmatched at no cost (their dummy partners)
        square[: len(br), : len(bc)] = sub
        for r, c in enumerate(_square_assignment(square)[: len(br)].tolist()):
            if c < len(bc) and cost[br[r], bc[c]] < limit:
                match_of[br[r]] = bc[c]
    matches = sorted(match_of.items())
    used = set(match_of.values())
    return matches, [r for r in range(n) if r not in match_of], [c for c in range(m) if c not in used]


class _Track:
    """One track: Kalman state, bookkeeping as the reference `STrack`."""

    __slots__ = ("activated", "cov", "frame", "id", "label", "mean", "score", "start", "state")

    def __init__(self, mean: np.ndarray, cov: np.ndarray, score: float, label: int) -> None:
        self.mean, self.cov = mean, cov
        self.score, self.label = score, label
        self.id = 0
        self.state = _TRACKED
        self.activated = False
        self.frame = self.start = 0

    def box(self) -> np.ndarray:
        """`(x0, y0, x1, y1)` of the current state."""
        return box_kalman.to_xyxy(self.mean[None])[0]


def _boxes(tracks: Sequence[_Track]) -> np.ndarray:
    return np.stack([t.box() for t in tracks]) if tracks else np.zeros((0, 4))


def _join(a: list[_Track], b: Iterable[_Track]) -> list[_Track]:
    """`a` then the tracks of `b` whose id is not in `a` (reference `joint_stracks`)."""
    ids = {t.id for t in a}
    out = list(a)
    for t in b:
        if t.id not in ids:
            ids.add(t.id)
            out.append(t)
    return out


class ByteTracker:
    """Online ByteTrack: `update(detections)` once per frame gives the confirmed tracks of that frame.

    The defaults are the reference MOT17 settings (`track_thresh 0.6`, `match_thresh 0.9`, `track_buffer 30`).
    """

    def __init__(
        self, *, track_thresh: float = 0.6, match_thresh: float = 0.9, track_buffer: float = 30, frame_rate: float = 30
    ) -> None:
        """Set the association thresholds.

        Args:
        track_thresh: Detections above this score start and extend tracks; those between 0.1 and it only extend.
        match_thresh: Limit of the fused IoU distance `1 - IoU * score` of the first association.
        track_buffer: Frames (at 30 fps) a lost track is kept for re-identification.
        frame_rate: Frame rate of the video; scales `track_buffer`.
        """
        self.track_thresh, self.match_thresh = track_thresh, match_thresh
        self.max_lost = int(frame_rate / 30.0 * track_buffer)
        self.reset()

    def reset(self) -> None:
        """Forget all tracks and restart the ids at 1 (a new video)."""
        self.frame = 0
        self.next_id = 1
        self.tracked: list[_Track] = []
        self.lost: list[_Track] = []
        self.removed: set[int] = set()

    def update(  # noqa: C901, PLR0912, PLR0915 - the reference's one-frame step, in its order
        self,
        detections: Any,  # noqa: ANN401 - DetectionResult or array-like
    ) -> ObjectTracks:
        """Advance one frame.

        Args:
            detections: This frame's detections: a `fastforks.detection.DetectionResult` (class-agnostic tracking;
                filter its classes before, e.g. `det[det.labels == 0]`) or an `[N, 5]` `(x0, y0, x1, y1, score)`
                array / tensor.

        Returns:
            The confirmed tracks after this frame.

        Raises:
            InputError: Detections that are not `[N, 5]` or have non-finite values.
        """
        boxes, scores, labels = _detections(detections)
        self.frame += 1
        frame = self.frame
        high = scores > self.track_thresh
        second = (scores > LOW_SCORE) & (scores < self.track_thresh)
        dets = [_Det(b, s, lab) for b, s, lab in zip(boxes[high], scores[high], labels[high], strict=True)]
        dets_low = [_Det(b, s, lab) for b, s, lab in zip(boxes[second], scores[second], labels[second], strict=True)]
        tentative = [t for t in self.tracked if not t.activated]
        pool = _join([t for t in self.tracked if t.activated], self.lost)
        if pool:
            mean = np.stack([t.mean for t in pool])
            mean[[t.state != _TRACKED for t in pool], 7] = 0.0  # a lost track stops growing
            mean, cov = box_kalman.predict(mean, np.stack([t.cov for t in pool]))
            for t, m, c in zip(pool, mean, cov, strict=True):
                t.mean, t.cov = m, c
        updates: list[tuple[_Track, _Det]] = []  # Kalman updates, applied together after the associations
        activated, refound, lost = [], [], []

        def matched(track: _Track, det: _Det) -> None:
            updates.append((track, det))
            (activated if track.state == _TRACKED else refound).append(track)
            track.state, track.activated, track.frame = _TRACKED, True, frame
            track.score, track.label = det.score, det.label

        cost = _fused(_boxes(pool), dets)
        pairs, left_tracks, left_dets = linear_assignment(cost, self.match_thresh)
        for i, j in pairs:
            matched(pool[i], dets[j])
        remaining = [pool[i] for i in left_tracks if pool[i].state == _TRACKED]
        cost = 1 - box_iou(_boxes(remaining), _det_boxes(dets_low))
        pairs, left_tracks2, _ = linear_assignment(cost, SECOND_MATCH)
        for i, j in pairs:
            matched(remaining[i], dets_low[j])
        for i in left_tracks2:
            remaining[i].state = _LOST
            lost.append(remaining[i])
        dets = [dets[j] for j in left_dets]
        pairs, left_tentative, left_dets = linear_assignment(_fused(_boxes(tentative), dets), TENTATIVE_MATCH)
        for i, j in pairs:
            matched(tentative[i], dets[j])
        removed = []
        for i in left_tentative:
            tentative[i].state = _REMOVED
            removed.append(tentative[i])
        if updates:
            tracks = [t for t, _ in updates]
            mean, cov = box_kalman.update(
                np.stack([t.mean for t in tracks]),
                np.stack([t.cov for t in tracks]),
                box_kalman.to_xyah(np.stack([d.box for _, d in updates])),
            )
            for t, m, c in zip(tracks, mean, cov, strict=True):
                t.mean, t.cov = m, c
        new = [dets[j] for j in left_dets if dets[j].score >= self.track_thresh + NEW_TRACK_MARGIN]
        if new:
            mean, cov = box_kalman.initiate(box_kalman.to_xyah(np.stack([d.box for d in new])))
            for d, m, c in zip(new, mean, cov, strict=True):
                t = _Track(m, c, d.score, d.label)
                t.id, self.next_id = self.next_id, self.next_id + 1
                t.activated = frame == 1
                t.frame = t.start = frame
                activated.append(t)
        for t in self.lost:
            if frame - t.frame > self.max_lost:
                t.state = _REMOVED
                removed.append(t)
        tracked = _join(_join([t for t in self.tracked if t.state == _TRACKED], activated), refound)
        tracked_ids = {t.id for t in tracked}
        # As the reference: the removed set is applied before this frame's removals join it, so a lost track that
        # times out stays in the lost pool (state removed) for one more frame and can still be re-found there.
        keep = [t for t in self.lost if t.id not in tracked_ids] + lost
        keep = [t for t in keep if t.id not in self.removed]
        self.removed.update(t.id for t in removed)
        self.tracked, self.lost = _dedup(tracked, keep)
        out = [t for t in self.tracked if t.activated]
        return ObjectTracks(
            torch.tensor([t.id for t in out], dtype=torch.int64),
            torch.from_numpy(_boxes(out).astype(np.float32)),
            torch.tensor([t.score for t in out], dtype=torch.float32),
            None if labels.dtype == object else torch.tensor([t.label for t in out], dtype=torch.int64),
        )


@dataclass
class _Det:
    box: np.ndarray
    score: float
    label: Any


def _det_boxes(dets: Sequence[_Det]) -> np.ndarray:
    return np.stack([d.box for d in dets]) if dets else np.zeros((0, 4))


def _fused(track_boxes: np.ndarray, dets: Sequence[_Det]) -> np.ndarray:
    """First-association cost `1 - IoU * score` (reference `fuse_score`)."""
    iou = box_iou(track_boxes, _det_boxes(dets))
    return 1 - iou * np.array([d.score for d in dets])[None]


def _dedup(tracked: list[_Track], lost: list[_Track]) -> tuple[list[_Track], list[_Track]]:
    """Drop the younger of a tracked / lost pair that overlap (reference `remove_duplicate_stracks`)."""
    dist = 1 - box_iou(_boxes(tracked), _boxes(lost))
    drop_a, drop_b = set(), set()
    for p, q in zip(*np.nonzero(dist < DUPLICATE), strict=True):
        if tracked[p].frame - tracked[p].start > lost[q].frame - lost[q].start:
            drop_b.add(int(q))
        else:
            drop_a.add(int(p))
    return [t for i, t in enumerate(tracked) if i not in drop_a], [t for i, t in enumerate(lost) if i not in drop_b]


def _detections(detections: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN401 - see update
    """float64 `[N, 4]` boxes, `[N]` scores and `[N]` labels (an object array of `None` without classes)."""
    if hasattr(detections, "boxes") and hasattr(detections, "scores"):
        boxes = detections.boxes.detach().cpu().double().numpy()
        scores = detections.scores.detach().cpu().double().numpy()
        lab = getattr(detections, "labels", None)
        labels = np.full(len(scores), None, dtype=object) if lab is None else lab.detach().cpu().numpy()
    else:
        try:
            arr = (
                detections.detach().cpu().double().numpy()
                if isinstance(detections, torch.Tensor)
                else np.asarray(detections, dtype=np.float64)
            )
        except (TypeError, ValueError) as e:
            msg = f"detections: not a numeric array or a DetectionResult ({e})"
            raise InputError(msg) from e
        if arr.size == 0:
            arr = arr.reshape(0, 5)
        if arr.ndim != 2 or arr.shape[1] != 5:  # noqa: PLR2004 - [N, (x0, y0, x1, y1, score)]
            msg = f"detections must be [N, 5] (x0, y0, x1, y1, score), got shape {arr.shape}"
            raise InputError(msg)
        boxes, scores = arr[:, :4], arr[:, 4]
        labels = np.full(len(scores), None, dtype=object)
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        msg = "detections have non-finite values"
        raise InputError(msg)
    return boxes, scores, labels


def track_objects(
    video: Iterable[Any],
    detector: Callable[..., Any] | None = None,
    *,
    labels: Sequence[int] | None = None,
    **tracker: float,
) -> list[ObjectTracks]:
    """Track the objects of a whole video (ByteTrack); `ByteTracker` is the online version.

    Args:
        video: The frames (anything `detector` takes), or with `detector=None` the detections of every frame
            (`DetectionResult`s or `[N, 5]` arrays).
        detector: A `fastforks.detection` model (load it with a low `score_threshold` option, e.g. 0.1: the second
            association uses detections down to 0.1), called once per frame.
        labels: Track only detections of these classes (e.g. `[0]`: COCO persons); default all.
        **tracker: `ByteTracker` settings (`track_thresh`, `match_thresh`, `track_buffer`, `frame_rate`).

    Returns:
        One `ObjectTracks` per frame.

    Raises:
        InputError: `labels` for detections without classes (`[N, 5]` arrays).
    """
    bt = ByteTracker(**tracker)
    keep = None if labels is None else torch.tensor(list(labels))
    out = []
    for item in video:
        det = item if detector is None else detector(item)
        if keep is not None:
            if getattr(det, "labels", None) is None:
                msg = "labels: these detections have no classes ([N, 5] arrays); filter them before"
                raise InputError(msg)
            det = det[torch.isin(det.labels, keep.to(det.labels.device))]
        out.append(bt.update(det))
    return out
