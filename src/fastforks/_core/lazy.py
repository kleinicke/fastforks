"""Torch on demand: a module stand-in that imports the real module on first use, and steps run after an import.

An MLX run on a warm conversion cache never touches torch, so the modules on that path refer to torch through
`lazy("torch")` (and `lazy("torch.nn.functional")`, ...) instead of importing it. The stand-in imports the module on
the first attribute access, so a torch mode pays the import where it used to (at load) and an MLX run never does::

    if TYPE_CHECKING:
        import torch
    else:
        torch = lazy("torch")

Class bodies, decorators and default arguments run at import: they must not touch the stand-in (a `torch.nn.Module`
base class belongs in a module only torch modes import, or behind a function).

`after_import("torch", step)` runs `step` right after whoever imports torch (now, if it already is): the
`torch.serialization.add_safe_globals` registrations of the result types.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["after_import", "imported", "lazy"]


class _Lazy(types.ModuleType):
    """A module that imports `name` on its first attribute access and forwards every access to it."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__dict__["_ff_target"] = None

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - the real module's attribute
        target = self.__dict__["_ff_target"]
        if target is None:
            if name.startswith("__"):  # introspection (copy, pickle, doctest) on a stand-in never imported
                raise AttributeError(name)
            target = importlib.import_module(self.__name__)
            self.__dict__["_ff_target"] = target
        return getattr(target, name)

    def __repr__(self) -> str:
        return f"<lazy module {self.__name__!r}>"


_STANDINS: dict[str, _Lazy] = {}


def lazy(name: str) -> Any:  # noqa: ANN401 - stands for the module
    """A stand-in for module `name` that imports it on first use (one per name)."""
    standin = _STANDINS.get(name)
    if standin is None:
        standin = _STANDINS[name] = _Lazy(name)
    return standin


def imported(name: str = "torch") -> types.ModuleType | None:
    """Module `name` if something imported it already, else `None` (never imports it)."""
    return sys.modules.get(name)


_AFTER: dict[str, list[Callable[[], None]]] = {}


class _Imported(importlib.abc.MetaPathFinder):
    """Runs the waiting `_AFTER[name]` steps right after module `name` is imported, by whoever imports it."""

    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:  # noqa: ANN401 - the finder protocol
        steps = _AFTER.get(name)
        if not steps:
            return None
        spec = None
        for finder in sys.meta_path:  # the regular finders after this one: a blocking finder before it still blocks
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        run = spec.loader.exec_module

        def exec_module(module: Any) -> None:  # noqa: ANN401 - the loader protocol
            run(module)
            for step in _AFTER.pop(name, ()):
                step()

        spec.loader.exec_module = exec_module  # ty: ignore[invalid-assignment] - this spec's own loader instance
        return spec


_FINDER = _Imported()


def after_import(name: str, step: Callable[[], None]) -> None:
    """Run `step` now if module `name` is imported, else right after its import."""
    if name in sys.modules:
        step()
        return
    _AFTER.setdefault(name, []).append(step)
    with contextlib.suppress(ValueError):
        sys.meta_path.remove(_FINDER)
    sys.meta_path.insert(0, _FINDER)
