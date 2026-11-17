"""List in, list out (`Model._list_images`, `_per_input`): one call per image, per-image arguments split."""

import inspect

import pytest

import fastforks
from fastforks import ArgumentError, InputError
from fastforks._core import registry
from fastforks._core.model import Model

SETS = {"multiview", "tracking", "volume"}  # the list is one input (views, frames, slices)
PAIRS = {"stereo", "flow", "interpolation", "matching"}


class Pair(Model):
    _list_images = 2
    _per_input = ("boxes",)

    def __init__(self) -> None:
        pass

    def __call__(self, a, b, *, boxes=None, scale=1):
        return (a, b, boxes, scale)


def test_lists_run_one_call_per_image_and_split_per_image_arguments():
    m = Pair()
    assert m(1, 2) == (1, 2, None, 1)
    assert m([1, 3], [2, 4], boxes=["x", "y"], scale=5) == [(1, 2, "x", 5), (3, 4, "y", 5)]
    assert m(a=[1], b=[2]) == [(1, 2, None, 1)]
    with pytest.raises(InputError, match="got 2 a, one b"):
        m([1, 3], 2)
    with pytest.raises(InputError, match="got 0 a, 0 b"):
        m([], [])
    with pytest.raises(InputError, match=r"boxes= as a list needs 2 items, one per image .* got 1"):
        m([1, 3], [2, 4], boxes=["x"])
    with pytest.raises(ArgumentError, match="scal"):
        m([1], [2], scal=2)  # ty: ignore[unknown-argument]


@pytest.mark.parametrize("task", registry.TASKS)
def test_every_single_image_and_pair_task_takes_lists(task):
    classes = [c for c in vars(getattr(fastforks, task)).values() if inspect.isclass(c) and issubclass(c, Model)]
    assert classes
    for cls in classes:
        want = 0 if task in SETS else 2 if task in PAIRS else 1
        assert cls._list_images == want, cls
        assert set(cls._per_input) <= set(inspect.signature(cls.__call__).parameters), cls


def test_only_a_list_is_split_per_image():
    """A tuple (or array) for a per-image argument is one shared value, even when its length equals the image count."""
    m = Pair()
    shared = ("x", "y")
    assert m([1, 3], [2, 4], boxes=shared) == [(1, 2, shared, 1), (3, 4, shared, 1)]
    assert m([1, 3], [2, 4], boxes=[shared, shared]) == [(1, 2, shared, 1), (3, 4, shared, 1)]


@pytest.mark.parametrize("task", registry.TASKS)
def test_declared_per_input_names_are_call_parameters(task):
    for cls in (c for c in vars(getattr(fastforks, task)).values() if inspect.isclass(c) and issubclass(c, Model)):
        assert all(k in inspect.signature(cls.__call__).parameters for k in cls._per_input), cls
