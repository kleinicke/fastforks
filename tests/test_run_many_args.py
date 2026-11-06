"""`run_many` checks its arguments before it starts any thread or touches the model."""

import pytest

from fastforks import InputError
from fastforks.stereo.pipeline import run_many


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"writers": -1}, "writers must be >= 0"),
        ({"readers": -2}, "readers must be >= 0"),
        ({"save_to": "x", "bits": 30}, "bits must be in 1..23"),
        ({"save_to": "x", "bits": 0}, "bits must be in 1..23"),
        ({"bits": 10}, "pass save_to="),
        ({"skip_existing": True}, "pass save_to="),
        ({"save_format": ".npy"}, "pass save_to="),
    ],
)
def test_run_many_checks_arguments_up_front(kwargs, match):
    with pytest.raises(InputError, match=match):
        run_many(None, "l", "r", **kwargs)  # type: ignore[arg-type]


def test_closed_pipeline_raises_a_library_error():

    from fastforks import FastforksError  # noqa: PLC0415
    from test_pipeline import Fake, _pairs  # noqa: PLC0415

    lefts, rights = _pairs(4)
    it = run_many(Fake(), lefts, rights)
    next(it)
    it.close()
    assert it.stats.total_s is not None
    from fastforks._core.pipeline import PipelineClosedError  # noqa: PLC0415

    assert issubclass(PipelineClosedError, FastforksError)
