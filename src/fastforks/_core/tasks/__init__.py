"""What a card of each task may declare: outputs, the `[output]` table and the `[io]` keys.

One spec per task, used by the registry, the card parser, the task APIs and the goldens. Each task has its own module
here (`<task>.py` with a `SPEC = TaskSpec(...)`, the same name as the `fastforks.<task>` package), so adding a task
adds a file and edits no shared list: `TASK_SPECS` and `registry.TASKS` are discovered from these modules, ordered by
`TaskSpec.order`. A task without a spec is a `KeyError`, never an unchecked card.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import MappingProxyType
from typing import TYPE_CHECKING

from fastforks._core.tasks._base import LABELS, VALUE_KEYS, TaskSpec
from fastforks._core.tasks.depth import DepthKind

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["LABELS", "TASK_SPECS", "VALUE_KEYS", "DepthKind", "TaskSpec", "spec"]


def _discover() -> dict[str, TaskSpec]:
    """`{task: spec}` of every public module in this package, ordered by `(spec.order, task)`."""
    found: dict[str, TaskSpec] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        sp = getattr(importlib.import_module(f"{__name__}.{info.name}"), "SPEC", None)
        if not isinstance(sp, TaskSpec):
            msg = f"{__name__}.{info.name} defines no SPEC = TaskSpec(...)"
            raise TypeError(msg)
        found[info.name] = sp
    return dict(sorted(found.items(), key=lambda kv: (kv[1].order, kv[0])))


TASK_SPECS: Mapping[str, TaskSpec] = MappingProxyType(_discover())
"""Every task's spec, in `registry.TASKS` order."""


def spec(task: str) -> TaskSpec:
    """The spec of `task` (`KeyError` for a task without one)."""
    return TASK_SPECS[task]
