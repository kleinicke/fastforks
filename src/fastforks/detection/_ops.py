"""torchvision's detection operators (NMS, ROIAlign), from the optional `torchvision` extra.

Upstream detectors (detectron2) call these exact operators, so a family uses them rather than a re-implementation;
their results (box order after NMS, pooled features) are then those of upstream on every device.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from fastforks._core.errors import MissingDependencyError, install_hint

if TYPE_CHECKING:
    from types import ModuleType


@functools.cache
def ops() -> ModuleType:
    """`torchvision.ops`.

    Raises:
        MissingDependencyError: torchvision is not installed.
    """
    try:
        import torchvision.ops  # noqa: PLC0415 - optional dependency (extra `torchvision`)
    except ImportError as e:
        msg = (
            f"detection models need torchvision's NMS / ROIAlign operators;"
            f" torchvision is not installed or cannot be imported ({e}): {install_hint('torchvision')}"
        )
        raise MissingDependencyError(msg) from e
    return torchvision.ops
