"""`TaskSpec`: what a card of one task may declare, plus the `[io]` / `[output]` building blocks the specs share."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    """Card rules of one task.

    Attributes:
        standard: Outputs with their own result field, the main one first. Others go to the result's `extras`.
        extras: Whether a card may list outputs outside `standard`.
        required: The card lists (and a runner returns) at least one of these.
        output_keys: Required `[output]` keys and their allowed values (`None`: a label set id).
        optional_output_keys: Optional `[output]` keys and their allowed values, the first the default.
        io_keys: `[io]` keys a card may set. Who applies them is in the task's `_runner.py`.
        order: Position of the task in `registry.TASKS` (listings, CLI lookups); ties are broken by name.
        bool_outputs: Outputs a runner returns as `torch.bool` (e.g. validity masks); the core checks them and casts
            every other output to float32 (`Model._check_outputs`).
        inputs: Names of the tensors `module()` takes, in order (ONNX / TensorRT input names).
        export_size: Default `(H, W)` of `export_onnx` / `fastforks export`; `None`: a size is required (stereo,
            depth, segmentation: no legacy default; classification: the card's crop; features: 224).
    """

    standard: tuple[str, ...]
    extras: bool
    required: frozenset[str]
    output_keys: dict[str, tuple[str, ...] | None]
    io_keys: frozenset[str]
    order: int
    optional_output_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)
    bool_outputs: frozenset[str] = field(default_factory=frozenset)
    inputs: tuple[str, ...] = ("image",)
    export_size: tuple[int, int] | None = None


VALUE_KEYS = frozenset({"input", "min_size", "mean", "std"})
"""`[io]` keys every task allows."""

LABELS: dict[str, tuple[str, ...] | None] = {"label_set": None}
"""`[output]` table of tasks whose outputs index a label set (`fastforks._core.labels`)."""
