"""The committed speed floor keeps its mode order: kernels <= portable <= the faster original, per ref and input.

Exceptions need an entry with the measured reason in docs/data/speed_allowlist.json; an entry that no longer matches a
violation fails too, so the list shrinks with each fix. Re-measuring against the floor needs a GPU:
`python tools/speed_check.py run --out rows.jsonl --check` (see CONTRIBUTING.md).
"""

import json

import speed_check


def test_floor_mode_order_or_allowlisted():
    allow = json.loads(speed_check.ALLOW.read_text())
    found = dict(speed_check.static_violations())
    assert not sorted(set(found) - set(allow)), "slower than a mode it must not lose to; fix it or allowlist it"
    assert not sorted(set(allow) - set(found)), "stale allowlist entries: remove them"


def test_allowlist_entries_give_a_reason():
    for key, reason in json.loads(speed_check.ALLOW.read_text()).items():
        assert isinstance(reason, str), key
        assert len(reason) > 20, key


def test_tensorrt_rows_fail_on_all_nan_outputs():
    """A TensorRT engine that builds but outputs only NaN (mvsanywhere: a fused-graph TensorRT bug) is a failed row."""
    import torch  # noqa: PLC0415

    from fastforks._core.export import all_nan  # noqa: PLC0415 - speed_check's check

    nan = torch.full((2, 3), float("nan"))
    assert all_nan(nan) == "0"
    assert all_nan((torch.ones(2), nan)) == "1"
    assert all_nan({"depth": nan}) == "depth"
    assert all_nan((torch.ones(2), torch.tensor([float("nan"), 1.0]), torch.zeros(0))) is None
    assert all_nan(torch.zeros(3, dtype=torch.long)) is None
