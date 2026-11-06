"""Parameter initialisation deferred past checkpoint loading.

A network built from upstream code initialises every parameter (`torch.nn.init`, e.g. `kaiming_uniform_` in each
`Conv2d`/`Linear`) and the checkpoint then overwrites nearly all of them: on the CPU that random fill costs ~0.1 s for a
50M-parameter stereo net and seconds for a ViT-L or a 1B-parameter network, per load. Inside `deferred_init()`,
`torch.nn.init`'s random in-place initialisers only zero the tensor (a memset, far cheaper than a random fill) and
record their call on the calling thread; the deterministic ones (`ones_`, `constant_`, ...) run as usual. When the block
ends, each recorded tensor that nothing has written to since (same version counter and same storage, e.g. a tensor the
checkpoint lacks) is initialised then. Tensors written in place (`load_state_dict`), given new data (`weight.data =
fused`, a reparameterisation; also a `.to()` move or cast, which leaves an unloaded randomly initialised tensor at
zero), replaced (`load_state_dict(assign=True)`) or freed are skipped. Loaded tensors therefore hold exactly the
checkpoint's values, as before; only the random draws of tensors the checkpoint does not cover come from a later point
of the global RNG stream (they were never seeded).
"""

from __future__ import annotations

import contextlib
import functools
import threading
import weakref
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

__all__ = ["deferred_init", "float32_default", "trunc_normal_"]

# The random in-place initialisers of `torch.nn.init` (each takes the tensor first and returns it). The deterministic
# ones (`constant_`, `ones_`, `zeros_`, `eye_`, `dirac_`) cost about as much as the memset that replaces a deferred
# call and run at once: a tensor the checkpoint lacks (a LayerScale gamma, a non-persistent buffer) then keeps its 1
# or c even when the build moves or casts it before the block ends.
_INITS = (
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "orthogonal_",
    "sparse_",
    "trunc_normal_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
)

_Pending = list[
    tuple["weakref.ref[torch.Tensor]", int, "weakref.ref[torch.UntypedStorage]", "Callable[..., Any]", tuple, dict]
]

_local = threading.local()  # .pending: the calling thread's recorded calls, or absent outside `deferred_init`
_lock = threading.Lock()
_users = 0  # threads inside `deferred_init`; the patch is installed while > 0
_originals: dict[str, Callable[..., Any]] = {}


def _deferring(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def init(tensor: object, *args: object, **kwargs: object) -> object:
        pending: _Pending | None = getattr(_local, "pending", None)
        if pending is None or not isinstance(tensor, torch.Tensor) or tensor.is_inference():
            return fn(tensor, *args, **kwargs)  # another thread, or no version counter to check later
        with torch.no_grad():
            tensor.zero_()  # a memset: tensors moved or cast before the block ends hold zeros, never stale memory
        # The storage's Python object lives exactly as long as the storage (torch preserves it), so unlike `data_ptr()`,
        # whose address the allocator can hand to the copy a `.data =` or `.to()` makes, it identifies the memory.
        storage = weakref.ref(tensor.untyped_storage())
        pending.append((weakref.ref(tensor), tensor._version, storage, fn, args, kwargs))  # noqa: SLF001
        return tensor

    init.__name__ = name
    return init


def _install() -> None:
    global _users  # noqa: PLW0603 - process-wide patch refcount
    with _lock:
        if _users == 0:
            for name in _INITS:
                _originals[name] = fn = getattr(torch.nn.init, name)
                setattr(torch.nn.init, name, _deferring(name, fn))
        _users += 1


def _uninstall() -> None:
    global _users  # noqa: PLW0603 - process-wide patch refcount
    with _lock:
        _users -= 1
        if _users == 0:
            for name, fn in _originals.items():
                setattr(torch.nn.init, name, fn)
            _originals.clear()


def trunc_normal_(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401 - forwarded as given
    """`torch.nn.init.trunc_normal_`, looked up per call: for code that imported the function by name."""
    return torch.nn.init.trunc_normal_(tensor, *args, **kwargs)


@contextlib.contextmanager
def deferred_init() -> Generator[None]:
    """Build and load a network inside this block: initialisers run only for tensors still unwritten at its end.

    Nested blocks on one thread share the outer block's record. If the block raises, nothing is initialised.
    """
    if getattr(_local, "pending", None) is not None:
        yield
        return
    pending: _Pending = []
    _install()
    _local.pending = pending
    try:
        yield
    finally:
        del _local.pending
        _uninstall()
    with torch.no_grad():
        for ref, version, storage, fn, args, kwargs in pending:
            tensor = ref()
            if tensor is None or tensor._version != version:  # noqa: SLF001 - torch's counter
                continue
            if tensor.untyped_storage() is storage():
                fn(tensor, *args, **kwargs)


_dt_lock = threading.Lock()
_dt_users = 0  # threads inside `float32_default`
_dt_saved = torch.float32  # the default dtype before the first one entered


@contextlib.contextmanager
def float32_default() -> Generator[None]:
    """`torch.get_default_dtype()` is float32 inside the block (a network built here gets float32 parameters).

    Builds must not depend on the caller's `torch.set_default_dtype` (e.g. float64): the requested dtype is what runs.
    The default dtype is process-wide: refcounted under a lock, the last thread out restores the value from before the
    first one entered (another thread building its own tensors at that moment also gets float32 defaults).
    """
    global _dt_users, _dt_saved  # noqa: PLW0603 - process-wide refcount
    with _dt_lock:
        if _dt_users == 0:
            _dt_saved = torch.get_default_dtype()
            if _dt_saved != torch.float32:
                torch.set_default_dtype(torch.float32)
        _dt_users += 1
    try:
        yield
    finally:
        with _dt_lock:
            _dt_users -= 1
            if _dt_users == 0 and _dt_saved != torch.float32:
                torch.set_default_dtype(_dt_saved)
