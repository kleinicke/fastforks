"""Edge / line maps (`fastforks.edges`): ControlNet-style preprocessors, a per-pixel line map in [0, 1] (1 = line)."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("edges", "lines"),
    extras=False,
    required=frozenset({"edges"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=36,
)
