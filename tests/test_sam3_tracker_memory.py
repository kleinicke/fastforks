"""Only live spatial features and object pointers should survive between SAM 3 video frames."""

import pytest
import torch

from fastforks.promptable_segmentation.models.sam3 import video


@pytest.mark.parametrize("length", [1, 36])
def test_tracker_keeps_memory_until_its_last_reader(length, monkeypatch) -> None:
    tracker = object.__new__(video.Tracker)
    tracker.no_mem = torch.zeros(1)
    monkeypatch.setattr(video, "SIZE", 2)
    encoded = []

    def encode(raw, _high, _obj, **_kwargs):
        encoded.append(raw.item())
        return raw

    def condition(raw, t, _n_frames, memories, pointers):
        for window, history in [(video.NUM_MASKMEM, memories), (video.MAX_PTRS, pointers)]:
            expected = {0, *range(max(1, t - window + 1), t)}
            assert history.keys() == expected
            assert all(history[k].item() == k for k in expected)
        return raw

    monkeypatch.setattr(tracker, "condition", condition)
    monkeypatch.setattr(tracker, "encode", encode)
    monkeypatch.setattr(tracker, "heads", lambda raw, *_args: (raw, raw, raw))
    frames = [{k: torch.full((1, 1, 1, 1), float(i)) for k in ("raw", "feat_s0", "feat_s1")} for i in range(length)]
    seen = []
    tracker.run(frames, length, coords=torch.zeros(1, 1, 2), labels=torch.ones(1, 1), mask=None,
                emit=lambda _t, low, _obj: seen.append(low.item()))  # fmt: skip
    assert seen == list(range(length))
    assert encoded == list(range(length - 1))
