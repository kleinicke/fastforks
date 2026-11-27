"""Torch tensors or NumPy arrays: the result container option, and array helpers that never import torch.

`fastforks.set_outputs("numpy")` (or `FASTFORKS_OUTPUTS=numpy`, or `model(..., outputs="numpy")` per call) makes the
results hold NumPy arrays instead of torch tensors. On an MLX backend whose runner computes on arrays (`Runner.arrays
= "numpy"`) a warm-cache run then never imports torch. The values are the same either way; only the container
differs. Other backends compute in torch and convert their results at the end (`.cpu().numpy()`).

Torch is looked up in `sys.modules`, never imported: an input that is a tensor means torch is loaded already.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, get_args

import numpy as np

from fastforks._core.errors import OptionsError

if TYPE_CHECKING:
    import torch

__all__ = ["HOST", "Device", "Outputs", "get_outputs", "is_tensor", "outputs_default", "set_outputs"]

Outputs = Literal["torch", "numpy"]
_KINDS: tuple[str, ...] = get_args(Outputs)
_ENV = "FASTFORKS_OUTPUTS"


def outputs_default() -> Outputs:
    """The container when neither `set_outputs` nor `FASTFORKS_OUTPUTS` chose one: torch tensors.

    (Step 2 of the torch-free plan makes this `"numpy"` when torch is not installed; `docs/plans/torch-free.md`.)
    """
    return "torch"


_chosen: Outputs | None = None


def _checked(kind: object, what: str) -> Outputs:
    if kind not in _KINDS:
        msg = f"{what} must be one of {', '.join(map(repr, _KINDS))}, got {kind!r}"
        raise OptionsError(msg)
    return kind  # ty: ignore[invalid-return-type] - checked above


def set_outputs(kind: Outputs | None) -> Outputs:
    """Set the container of every result: `"torch"` tensors (the default) or `"numpy"` arrays; `None` resets it.

    Process-wide; `model(..., outputs=)` overrides it for one call and `FASTFORKS_OUTPUTS` sets it at start-up.

    Returns:
        The previous setting (as `get_outputs` gave it), to restore it.

    Raises:
        OptionsError: Unknown kind.
    """
    global _chosen  # noqa: PLW0603 - the process-wide choice
    old = get_outputs()
    _chosen = None if kind is None else _checked(kind, "set_outputs")
    return old


def get_outputs() -> Outputs:
    """The current container: `set_outputs`, else `FASTFORKS_OUTPUTS`, else `"torch"`."""
    if _chosen is not None:
        return _chosen
    env = os.environ.get(_ENV)
    return _checked(env, _ENV) if env else outputs_default()


def resolve_outputs(kind: object) -> Outputs:
    """A call's `outputs=` (`None`: the process-wide setting), checked."""
    return get_outputs() if kind is None else _checked(kind, "outputs")


def torch_module() -> Any:  # noqa: ANN401 - the torch module or None
    """Torch if something imported it, else `None`."""
    return sys.modules.get("torch")


def is_tensor(x: object) -> TypeGuard[torch.Tensor]:
    """Whether `x` is a torch tensor (never imports torch: without it nothing is one)."""
    t = sys.modules.get("torch")
    return t is not None and isinstance(x, t.Tensor)


def is_array(x: object) -> TypeGuard[np.ndarray]:
    """Whether `x` is a NumPy array."""
    return isinstance(x, np.ndarray)


@dataclass(frozen=True)
class Device:
    """The device of a model without torch loaded: `type` and `index` as `torch.device` has them.

    Equal to the `torch.device` (or string) of the same name; `torch.device(model.device)` converts it.
    """

    type: str
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self) -> str:
        return f"device(type={self.type!r}" + ("" if self.index is None else f", index={self.index}") + ")"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Device):
            return str(self) == str(other)
        if isinstance(other, str) or type(other).__name__ == "device":
            return str(self) == str(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))


HOST = Device("numpy")  # the input "device" of an array runner: NumPy arrays on the host


def device(name: str) -> Any:  # noqa: ANN401 - torch.device or Device
    """`torch.device(name)` when torch is loaded, else a `Device` of that name (never imports torch)."""
    t = sys.modules.get("torch")
    if t is not None and name != HOST.type:
        return t.device(name)
    kind, _, index = name.partition(":")
    return Device(kind, int(index) if index else None)


def to_numpy(x: Any) -> np.ndarray:  # noqa: ANN401 - tensor or array
    """A host NumPy array of a tensor (bfloat16 -> float32; no copy on the CPU) or of an array (unchanged)."""
    if isinstance(x, np.ndarray):
        return x
    if is_tensor(x):
        t = x.detach().cpu()
        return (t.float() if t.dtype == sys.modules["torch"].bfloat16 else t).numpy()
    if type(x).__module__.partition(".")[0] == "mlx":
        return np.asarray(x)
    return np.asarray(x)


def to_torch(x: np.ndarray, device: Any = None) -> torch.Tensor:  # noqa: ANN401 - torch.device or None
    """A torch tensor of an array: shares its memory on the CPU (read-only arrays are copied), else moved."""
    import torch  # noqa: PLC0415 - the torch results only

    t = torch.from_numpy(np.require(x, requirements="W"))  # torch warns on read-only memory
    return t if device is None or str(device) == "cpu" else t.to(str(device))


def cat(parts: list[Any]) -> Any:  # noqa: ANN401 - tensors or arrays
    """`torch.cat` / `np.concatenate` along dim 0, by the type of the parts."""
    if isinstance(parts[0], np.ndarray):
        return np.concatenate(parts)
    return sys.modules["torch"].cat(parts)


def is_bool(x: Any) -> bool:  # noqa: ANN401 - tensor or array
    """Whether `x` is a bool tensor or array."""
    if isinstance(x, np.ndarray):
        return x.dtype == np.bool_
    return x.dtype == sys.modules["torch"].bool


def float32(x: Any) -> Any:  # noqa: ANN401 - tensor or array
    """`x` as float32 (no copy if it is already)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return x.float()


def convert(value: Any, kind: Outputs, device: Any = None) -> Any:  # noqa: ANN401, PLR0911 - any value, by kind
    """`value` with its arrays / tensors as `kind`: results (`NumpyOutputs`), tensors, arrays, lists, tuples, dicts.

    `device`: where torch tensors go (`None`: where they are; NumPy arrays become CPU tensors there).
    """
    from fastforks._core.outputs import NumpyOutputs  # noqa: PLC0415 - outputs imports this module

    if isinstance(value, NumpyOutputs):
        return value._converted(kind, device)  # noqa: SLF001 - its own conversion
    if isinstance(value, np.ndarray):
        return value if kind == "numpy" else to_torch(value, device)
    if is_tensor(value):
        if kind == "numpy":
            return to_numpy(value)
        return value if device is None else value.to(device)
    if isinstance(value, list):
        return [convert(v, kind, device) for v in value]
    if isinstance(value, tuple) and type(value) is tuple:
        return tuple(convert(v, kind, device) for v in value)
    if isinstance(value, dict):
        return {k: convert(v, kind, device) for k, v in value.items()}
    return value
