"""What every model family implements, whatever the task: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

A task package (`fastforks.stereo`, `fastforks.depth`, ...) subclasses `Runner` once to fix the call signature and
the input handling of its task (see the `_runner.py` of each task); families subclass that.

```python
# <task>/models/<family>/model.py
@dataclass(frozen=True)
class Params:  # card [params] + variant params + [options] (+ load(options=)), checked at load
    iters: int


class Portable(TaskRunner):
    dtypes = ("float32", "float16")  # CUDA dtypes this code implements; the card picks the validated one
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...  # build from ctx.state_dict(), move to ctx.device


RUNNERS = {"portable": Portable, "kernels": Kernels}
```
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict
from fastforks._core.errors import WeightsError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from fastforks._core.cards import Dtype, Io, Mode, ModelCard, Variant


@dataclass(frozen=True)
class NoParams:
    """Params of a family without parameters."""


@dataclass(frozen=True)
class Context:
    """Everything a runner is built from.

    Attributes:
        card: The family's card.
        variant: The variant to load.
        mode: The backend.
        device: Target device.
        dtype: Main compute dtype (portable: the card's validated CUDA dtype, or float32 on CPU; kernels: the card's).
        params: The runner's `Params`, filled from the card and the load options.
        kernels: Kernel library tags of `kernels` (`name@build:target`), empty for portable.
        user_weights: The files of `load(weights=...)`, used instead of the variant's (not hash-checked).
    """

    card: ModelCard
    variant: Variant
    mode: Mode
    device: torch.device
    dtype: Dtype
    params: Any
    kernels: tuple[str, ...] = ()
    user_weights: tuple[Path, ...] | None = None

    @property
    def io(self) -> Io:
        """Effective input handling of the variant (the card's `[io]` plus the variant's overrides)."""
        return self.variant.io

    @property
    def torch_dtype(self) -> torch.dtype:
        """`dtype` as a `torch.dtype`."""
        return getattr(torch, self.dtype)

    def weights(self) -> list[Path]:
        """Local paths of the variant's weight files: verified (downloaded if needed), or the user's."""
        if self.user_weights is not None:
            return list(self.user_weights)
        return registry.weights(self.card, self.variant)

    def state_dict(
        self,
        index: int = 0,
        *,
        key: str | tuple[str, ...] | None = None,
        strip: str = "module.",
        extra: Mapping[tuple[str, str], object] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Weight file `index` (hash-checked, or the user's) as a flat state dict on the CPU.

        Zip files are memory-mapped and the legacy format is read normally (detected, not flagged); safetensors files
        are read with `safetensors`; TorchScript archives give their tensors only (no code is loaded). Only tensors and
        plain containers are unpickled, plus the globals in `extra` (see `fastforks._core.checkpoint`).

        Args:
            index: Which of the variant's weight files.
            key: Entry of the checkpoint that holds the state dict (`"model"`, `"state_dict"`; a tuple for nested
                entries, e.g. `("ema", "module")`); `None` if the file is the state dict.
            strip: Prefix removed from every key (`DataParallel`'s `"module."`); `""` keeps the keys.
            extra: Further allowed globals, `{(module, name): object}` (`Inert` stands in for config classes).

        Raises:
            WeightsError: Missing, unverifiable or unsafe weights, or a user's file that cannot be read (empty,
                truncated, another format): the message names it and the variant.
        """
        path = self.weights()[index]
        if self.user_weights is None:
            return load_state_dict(path, key=key, strip=strip, extra=extra)
        try:
            return load_state_dict(path, key=key, strip=strip, extra=extra)
        except (MemoryError, torch.OutOfMemoryError):
            raise
        except Exception as e:  # any reader's error: the file is the user's
            detail = str(e) if isinstance(e, WeightsError) else f"{type(e).__name__}: {e}"
            msg = f"weights={path} for {self.variant.ref} cannot be read as its checkpoint: {detail}"
            raise WeightsError(msg) from e


class Runner(abc.ABC):  # noqa: B024 - abstract `__call__` is declared by each task's Runner
    """One backend of a model family (task-agnostic part).

    Class attributes:
        dtypes: Compute dtypes the portable code implements on CUDA (the card's `cuda_dtype` must be one of them).
        batch: Inputs per call (`None`: any number); the core splits larger batches.
        Params: Frozen dataclass of the family's parameters.
        deterministic_cudnn: Run each CUDA call with `torch.backends.cudnn.deterministic = True` (restored after).
            Set on the portable runners where cuDNN's default algorithms (atomics in transposed convs) made repeated
            calls differ and the deterministic ones cost nothing (measured on IGEV++, LAS2 and the other GRU families).
        compiled_methods: The methods the task API calls, which `Model.compile()` wraps in `torch.compile` (one
            calling another runs inside the caller's graph). A family whose API path calls other methods (a
            multi-view `scene` override) lists those.
    """

    dtypes: ClassVar[tuple[Dtype, ...]] = ("float32",)
    batch: ClassVar[int | None] = 1
    Params: ClassVar[type[Any]] = NoParams
    deterministic_cudnn: ClassVar[bool] = False
    compiled_methods: ClassVar[tuple[str, ...]] = ("__call__",)

    def __init__(self, ctx: Context) -> None:
        """Build the network for `ctx`; subclasses call this first."""
        self.ctx = ctx

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        The task's `_runner.py` says which inputs and outputs. Portable runners implement it.

        Raises:
            NotImplementedError: The backend has no exportable module (e.g. `kernels`).
        """
        msg = f"{type(self).__name__} has no exportable module"
        raise NotImplementedError(msg)

    @classmethod
    def input_use(cls, params: Any) -> dict[str, Any]:  # noqa: ANN401, ARG003 - the family's `Params`
        """Which optional inputs a variant takes and how many (the task's class attributes; no weights are loaded).

        Read by `ModelCard.contract()`. Task runners return their class attributes (depth: `intrinsics`; multi-view:
        the cameras and view counts; promptable segmentation: `prompts`); a family whose use differs per variant
        overrides this and sets the attributes from it in `__init__`.

        Args:
            params: The variant's `Params`.
        """
        return {}

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """Every network a call runs, for device adaptations (Apple MPS: `mps.adapt` on the portable runners).

        Default: `(module(),)`. Runners whose `module()` covers only part of a call (e.g. SAM's image encoder without
        its mask decoder) override it.
        """
        return (self.module(),)


class Forward(torch.nn.Module):
    """`nn.Module` wrapper for `Runner.module()`: `forward(*inputs) = fn(net, *inputs)`."""

    def __init__(
        self,
        net: torch.nn.Module,
        fn: Callable[..., Any],
        output_names: tuple[str, ...] = (),
        query_scores: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Wrap `net`; `fn` is called as `fn(net, *inputs)`. `output_names` name a tuple output in an ONNX export.

        `query_scores` (detection transformers): `logits [B, Q, C]` -> the per-query score `[B, Q]`; the CLI's
        TensorRT check then matches the scored queries by box (an encoder top-k may pick or order them differently).
        """
        super().__init__()
        self.net = net
        self.fn = fn
        self.output_names = output_names
        self.query_scores = query_scores
        self.training = net.training  # a new Module starts in training mode; only this flag, no recursion

    def forward(self, *inputs: torch.Tensor) -> Any:  # noqa: ANN401 - whatever the task's module returns
        """`fn(net, *inputs)`."""
        return self.fn(self.net, *inputs)
