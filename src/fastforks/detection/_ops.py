"""torchvision's detection operators (NMS, ROIAlign), from the optional `torchvision` extra.

Upstream detectors (detectron2) call these exact operators, so a family uses them rather than a re-implementation;
their results (box order after NMS, pooled features) are then those of upstream on every device.
"""

from __future__ import annotations

import functools
import sys
import threading
from importlib.machinery import SourceFileLoader
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.errors import MissingDependencyError, install_hint

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_NMS_WHEELS = frozenset(("0.29.1+cu130", "0.29.2+cu132"))  # tested wheels using torchvision/_C_stable.so
_NMS_LOCK = threading.Lock()


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


@functools.cache
def _native_cuda_nms() -> Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None:
    """Resolve the tested wheel's operator directly, or let torchvision handle any other installation."""
    cuda = torch.version.cuda
    if "torchvision" in sys.modules or cuda is None:
        return None
    try:
        wheel = distribution("torchvision")
        spec = find_spec("torchvision")
        if (
            wheel.version not in _NMS_WHEELS
            or spec is None
            or spec.origin is None
            or not isinstance(spec.loader, SourceFileLoader)
            or Path(spec.origin).resolve() != Path(str(wheel.locate_file("torchvision/__init__.py"))).resolve()
        ):
            return None
        path = str(Path(spec.origin).with_name("_C_stable.so").resolve(strict=True))
        if hasattr(torch.ops.torchvision, "nms") and path not in torch.ops.loaded_libraries:
            return None  # a different registration may already own this namespace
        if path not in torch.ops.loaded_libraries:
            torch.ops.load_library(path)
        version = torch.ops.torchvision._cuda_version()  # noqa: SLF001 - torchvision's own CUDA version query
        if (
            version < 12000  # noqa: PLR2004 - torchvision's encoded CUDA version starts at 12.0
            or version // 1000 != int(cuda.split(".", maxsplit=1)[0])
            or not torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "CUDA")  # noqa: SLF001
        ):
            return None
        nms = torch.ops.torchvision.nms.default
    except (PackageNotFoundError, AttributeError, ImportError, OSError, RuntimeError, ValueError):
        return None
    else:
        return nms


def cuda_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """RTMDet's NMS, deferring torchvision's Python import on the supported CUDA wheels."""
    if (
        sys.platform == "linux"
        and boxes.is_cuda
        and boxes.device == scores.device
        and boxes.dtype == scores.dtype == torch.float32
        and "torchvision" not in sys.modules
        and not torch.compiler.is_compiling()
        and not torch.jit.is_tracing()
        and not torch.jit.is_scripting()
    ):
        with _NMS_LOCK:
            direct = _native_cuda_nms()
        if direct is not None:
            return direct(boxes, scores, threshold)
    return ops().nms(boxes, scores, threshold)
