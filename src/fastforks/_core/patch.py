"""Per-instance method overrides of `nn.Module`s without reference cycles.

`m.forward = fn.__get__(m)` stores a bound method in the module's `__dict__`: module -> bound method -> module is a
cycle, so `del model` frees the weights only when the cyclic garbage collector runs (seconds later, or never with
`gc.disable()`). `override` instead swaps the module's class for a cached subclass of its original class that defines
the method. The subclass keeps the original class name, so `isinstance`, `repr`, `state_dict`, `torch.compile`,
`torch.export` and ONNX export see the same module. Exact type checks use `base_type(m)`: `type(m)` is the subclass.
Pickling (`torch.save(module)`) stores the base class and the override functions and re-applies them on load.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

_T = TypeVar("_T")
_CLASSES: dict[tuple[type, tuple[tuple[str, Any], ...]], type] = {}
_LOCK = threading.Lock()


def base_type(module: object) -> type[Any]:
    """The class `module` had before any `override` (`type(module)` for anything not overridden).

    The base is a class attribute and so is inherited: if a class derived from an overridden module's class is made
    later (e.g. by `torch.nn.utils.parametrize`), its instances also report the original base, and a later `override`
    builds its class from that base, dropping the derived class. No fastforks code path does this.

    Args:
        module: Any object, usually a module.
    """
    cls: type[Any] = type(module)
    return getattr(cls, "_fastforks_base", cls)


def is_base(module: object, cls: type[_T]) -> TypeGuard[_T]:
    """`base_type(module) is cls`, as a type guard (`module` is a `cls` afterwards).

    Args:
        module: Any object, usually a module.
        cls: The exact class to test for.
    """
    return base_type(module) is cls


def _rebuild(base: type[nn.Module], methods: tuple[tuple[str, Any], ...]) -> nn.Module:
    """Unpickle step of an overridden module: an empty `base` instance with the same overrides (state comes after)."""
    module = base.__new__(base)
    for name, fn in methods:
        override(module, fn, name)
    return module


def _reduce_ex(self: nn.Module, protocol: int) -> tuple[Any, ...]:
    """Pickle (`torch.save(module)`, `copy.deepcopy`) as the base class plus the overrides.

    Pickle finds classes by name, and the per-instance subclass has the base's name. The override functions must be
    module-level (picklable).
    """
    del protocol
    methods = getattr(type(self), "_fastforks_methods", ())
    return _rebuild, (base_type(self), methods), self.__getstate__()


def override(module: nn.Module, fn: Callable[..., Any], name: str = "forward") -> None:
    """Make `fn(module, ...)` the method `name` of `module` (only this instance), without a reference cycle.

    A second `override` of the same `name` replaces the first; it does not wrap it. A wrapper that calls the original
    method through `base_type(self).<name>` skips any earlier override of that method.

    Args:
        module: The module to change in place.
        fn: A function taking the module as its first argument. Use a module-level function: each distinct `fn` makes
            one cached class per module class, so a new closure per call would create a class per call.
        name: The method to override.
    """
    base = base_type(module)
    methods = dict(getattr(type(module), "_fastforks_methods", ()))
    methods[name] = fn
    key = (base, tuple(sorted(methods.items(), key=lambda kv: kv[0])))
    with _LOCK:
        cls = _CLASSES.get(key)
        if cls is None:
            attrs = {
                **methods,
                "_fastforks_base": base,
                "_fastforks_methods": key[1],
                "__module__": base.__module__,
                "__qualname__": base.__qualname__,
                "__doc__": base.__doc__,
                "__reduce_ex__": _reduce_ex,
            }
            cls = _CLASSES[key] = type(base.__name__, (base,), attrs)
    module.__dict__.pop(name, None)  # an earlier instance-level override would shadow the class method
    module.__class__ = cls


def on_load(module: nn.Module, hook: Callable[[Any, Any], None]) -> None:
    """Run `hook(module, incompatible_keys)` after every `load_state_dict` that reaches `module` (once per hook).

    For passes that keep derived copies of weights (fused / folded / packed buffers, not in the state dict): the hook
    re-derives them once `module` and its children are loaded, at no cost per forward. Use a module-level `hook` so
    the patched model still pickles.
    """
    if hook not in module._load_state_dict_post_hooks.values():  # noqa: SLF001 - torch's hook table
        module.register_load_state_dict_post_hook(hook)
