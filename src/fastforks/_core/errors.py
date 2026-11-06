"""Typed errors. Every error raised on purpose by fastforks derives from `FastforksError`."""

from __future__ import annotations

from typing import Any


class FastforksError(Exception):
    """Base class of all fastforks errors."""


class ModelNotFoundError(FastforksError, KeyError):
    """No installed model card has this id."""

    def __str__(self) -> str:
        # KeyError.__str__ would print the repr of the message.
        return str(self.args[0]) if self.args else ""


class CardError(FastforksError, ValueError):
    """A model card or kernel manifest is malformed (a packaging bug, not a user error)."""


class InputError(FastforksError, ValueError):
    """Input images have an unsupported type, shape, dtype or value range."""


class ArgumentError(InputError, TypeError):
    """A model call got arguments that do not fit its signature (still a `TypeError`, as Python raises it)."""


class OptionsError(FastforksError, ValueError):
    """A load option is unknown or has the wrong type."""


class BackendUnavailableError(FastforksError, RuntimeError):
    """The requested backend (e.g. `mode="kernels"`) cannot run here. There is no silent fallback."""


class MissingDependencyError(BackendUnavailableError):
    """An optional library (a pip extra such as `torchvision`) is not installed; the message names the extra."""


def _make_kernel_unavailable(library: str, reason: str, model: str | None) -> KernelUnavailableError:
    return KernelUnavailableError(library, reason, model=model)


class KernelUnavailableError(BackendUnavailableError):
    """A custom-kernel library cannot be loaded on this machine.

    Attributes:
        library: Kernel library name, e.g. `"s2m2"`.
        reason: Why it cannot be loaded (arch, driver, missing binary, ...).
        model: The model whose `kernels` backend needs it, or `None`.
    """

    def __init__(self, library: str, reason: str, *, model: str | None = None) -> None:
        """Create the error.

        Args:
            library: Kernel library name.
            reason: Human-readable cause, ideally with a remedy.
            model: The model whose `kernels` backend needs the library, if known.
        """
        who = f"{model} (mode='kernels'): " if model else ""
        super().__init__(
            f"{who}kernel library {library!r} unavailable: {reason}; use mode='auto' to fall back to portable"
        )
        self.library = library
        self.reason = reason
        self.model = model

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by the constructor arguments (the base class would pass the message alone)."""
        return (_make_kernel_unavailable, (self.library, self.reason, self.model))


class LicenceError(FastforksError, RuntimeError):
    """The weights of a model forbid commercial use and non-commercial models were not enabled (see the message)."""


class WeightsError(FastforksError, RuntimeError):
    """Model weights cannot be provided."""


class CheckpointMismatchError(WeightsError):
    """A state dict does not fit the network; `details` lists the first missing, unexpected and reshaped keys."""

    def __init__(self, what: str, details: str) -> None:
        """`what` (e.g. `"s2m2-s@1 checkpoint"`) does not fit the network: `details` (see `mismatch_details`)."""
        super().__init__(f"{what} does not fit the network: {details}")
        self.what = what
        self.details = details

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by the constructor arguments (the base class would pass the message alone)."""
        return (type(self), (self.what, self.details))


class WeightsDownloadError(WeightsError):
    """Downloading a weights or kernel file failed."""


class WeightsHashError(WeightsError):
    """A downloaded or local file does not match its recorded SHA-256."""


def install_hint(*extras: str, upgrade: bool = False) -> str:
    """The pip command that installs `extras` (`upgrade`: replacing an unsupported version), for error messages."""
    u = "-U " if upgrade else ""
    pick = " ".join(f"'fastforks-core[{e}]'" for e in extras)
    return (
        f"pip install {u}{pick} (or 'pip install fastforks' for everything;"
        f" from a git checkout: pip install {u}-e '.[{','.join(extras)}]')"
    )
