"""Opt-in on-disk cache of the MLX graph translation's slow first-call steps (`FASTFORKS_MLX_PLAN_CACHE=<dir>`).

The first call of an `MlxModule` at an input signature exports the network (`torch.export` and the core-ATen
decomposition: 1 to 7 s on the M1 Pro), translates it and, on its first replay, times each big compiled segment once
eagerly and once compiled (the tuning pass). A new process pays all of it again. With the cache on:

- the decomposed `ExportedProgram` is saved (`torch.export.save`) without the network's weights: every parameter and
  buffer of the module is a one-element stand-in in the file and is bound to the module's own tensor (same name, shape
  and dtype, else the entry is not used) when it is loaded; only the constants the forward builds are stored.
- the tuning pass's decisions (compiled or eager per segment) are saved beside it and replace the timing trials.

An entry's key is the load (model ref, mode, dtype, load options, user weight files: `scope`, set while an MLX runner is
built), the module's index among that load's MLX modules, the module's class and parameter shapes, the input
signature, and the versions of fastforks (plus a fingerprint of its source files), torch and mlx: anything else is a
miss and the slow path runs (and writes the entry). Nothing is cached when no scope is set (an `MlxModule` built outside
a model load) or when the export needed the eager forward first (lazily built buffers: the loaded graph could not
rebuild them). Writes are atomic (temp file + rename); an unreadable entry is a miss. The cache is never pruned; delete
the directory to reset it.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from collections.abc import Iterator

ENV = "FASTFORKS_MLX_PLAN_CACHE"
FORMAT = 1  # bump when the stored layout or what the translation reads from the program changes

_SCOPE: contextvars.ContextVar[tuple[str, list[int]] | None] = contextvars.ContextVar("mlx_plan_scope", default=None)
# env switches that change the segmentation or the compile policy: part of a tuning record's key
_TUNE_ENV = ("FF_MLX_SEG_MB", "FF_MLX_CHUNK_MB", "FF_MLX_COMPILE", "FF_MLX_NO_BAND", "FF_MLX_NO_CL", "FF_MLX_NO_TUNE")


def directory() -> Path | None:
    """The cache directory (`FASTFORKS_MLX_PLAN_CACHE`), or None when the cache is off."""
    d = os.environ.get(ENV, "").strip()
    return Path(d).expanduser() if d else None


@contextlib.contextmanager
def scope(name: str) -> Iterator[None]:
    """While an MLX runner is built: its `MlxModule`s get cache keys `name#0`, `name#1`, ... in creation order."""
    token = _SCOPE.set((name, [0]))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def next_scope() -> str | None:
    """The cache scope of an `MlxModule` built now (None outside a model load)."""
    cur = _SCOPE.get()
    if cur is None:
        return None
    name, counter = cur
    counter[0] += 1
    return f"{name}#{counter[0] - 1}"


@functools.cache
def _versions() -> str:
    """The fastforks / torch / mlx versions and a fingerprint (path, size, mtime) of fastforks' source files."""
    import fastforks  # noqa: PLC0415

    root = Path(fastforks.__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        st = p.stat()
        h.update(f"{p.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    try:
        mlx = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx = ""
    return f"{fastforks.__version__}|{torch.__version__}|{mlx}|{h.hexdigest()[:16]}|{FORMAT}"


def _named_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter and buffer of `module` by its state-dict name (shared tensors under each of their names)."""
    out: dict[str, torch.Tensor] = {}
    for prefix, host in module.named_modules(remove_duplicate=False):
        for leaf, t in (*host._parameters.items(), *host._buffers.items()):  # noqa: SLF001 - the registries
            if t is not None:
                out[f"{prefix}.{leaf}" if prefix else leaf] = t
    return out


class Entry:
    """A cache entry: the exported program (`<key>.pt2`), call metadata (`<key>.pkl`), tuning (`<key>.tune.json`)."""

    def __init__(self, base: Path) -> None:
        """The entry whose files are `base` plus a suffix."""
        self.base = base

    @classmethod
    def make(cls, scope_name: str | None, module: torch.nn.Module, signature: Any) -> Entry | None:  # noqa: ANN401
        """The entry of `module` (its scope, structure) at `signature`, or None when the cache is off or unscoped."""
        d = directory()
        if d is None or scope_name is None:
            return None
        shapes = sorted((k, tuple(t.shape), str(t.dtype)) for k, t in _named_tensors(module).items())
        inner = getattr(module, "module", module)
        text = repr((scope_name, type(inner).__qualname__, type(inner).__module__, shapes, signature, _versions()))
        return cls(d / hashlib.sha256(text.encode()).hexdigest()[:32])

    # The exported program.
    def load_program(self, module: torch.nn.Module) -> tuple[Any, Any] | None:
        """`(ExportedProgram with the module's own tensors bound, metadata)`, or None (miss or unusable entry)."""
        pt2, meta = self.base.with_suffix(".pt2"), self.base.with_suffix(".pkl")
        if not (pt2.is_file() and meta.is_file()):
            return None
        try:
            info = pickle.loads(meta.read_bytes())  # noqa: S301 - our own cache directory
            with warnings.catch_warnings():  # the loader's notes must not fail a run that turns warnings into errors
                warnings.simplefilter("ignore", UserWarning)
                ep = torch.export.load(pt2)
        except Exception:  # noqa: BLE001 - a stale or partly written entry is a miss
            return None
        own = _named_tensors(module)
        for attr, names in (("_state_dict", info["bound_state"]), ("_constants", info["bound_constants"])):
            store = dict(getattr(ep, attr))
            for name in names:
                t, saved = own.get(name), store.get(name)
                if t is None or saved is None or tuple(t.shape) != tuple(saved.shape) or t.dtype != saved.dtype:
                    return None
                store[name] = t
            setattr(ep, attr, store)
        # the loader adds a getitem for every output of a multi-output op (batch norm's unused mean and invstd): drop
        # them, the translation's pattern matches (batch-norm folding) count users
        ep.graph_module.graph.eliminate_dead_code()
        ep.graph_module.recompile()
        return ep, info["extra"]

    def save_program(self, ep: Any, module: torch.nn.Module, extra: Any) -> None:  # noqa: ANN401
        """Store `ep` with the module's tensors as stand-ins, and `extra` (picklable call metadata); best effort."""
        own = _named_tensors(module)

        def stand_ins(store: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            out, bound = dict(store), []
            for name, v in store.items():
                t = own.get(name)
                if (  # an empty or one-element tensor is stored as is (an empty stand-in breaks torch.export.save)
                    isinstance(v, torch.Tensor)
                    and v.numel() > 1
                    and t is not None
                    and tuple(t.shape) == tuple(v.shape)
                    and t.dtype == v.dtype
                ):
                    stand = torch.zeros((), dtype=v.dtype).expand(v.shape)
                    out[name] = (
                        torch.nn.Parameter(stand, requires_grad=False) if isinstance(v, torch.nn.Parameter) else stand
                    )
                    bound.append(name)
            return out, bound

        state, consts, example = ep._state_dict, ep._constants, ep._example_inputs  # noqa: SLF001 - swapped back below
        try:
            ep._state_dict, bound_state = stand_ins(state)  # noqa: SLF001
            ep._constants, bound_constants = stand_ins(consts)  # noqa: SLF001
            if example is not None:  # the sample inputs: shapes only (a stereo pair is tens of MB)
                ep._example_inputs = pytree.tree_map_only(  # noqa: SLF001
                    torch.Tensor, lambda t: torch.zeros((), dtype=t.dtype).expand(t.shape), example
                )
            buf = io.BytesIO()
            with warnings.catch_warnings():  # constants that are partial views of one storage: CPU, saved as they are
                warnings.simplefilter("ignore", UserWarning)
                torch.export.save(ep, buf)
            info = pickle.dumps({"bound_state": bound_state, "bound_constants": bound_constants, "extra": extra})
        except Exception:  # noqa: BLE001 - an unserialisable program is simply not cached
            return
        finally:
            ep._state_dict, ep._constants, ep._example_inputs = state, consts, example  # noqa: SLF001
        _write(self.base.with_suffix(".pt2"), buf.getvalue())
        _write(self.base.with_suffix(".pkl"), info)

    # The tuning decisions.
    @staticmethod
    def _tune_key(extra: Any) -> str:  # noqa: ANN401
        return repr((extra, [(k, os.environ.get(k, "")) for k in _TUNE_ENV]))

    def load_tune(self, extra: Any) -> list[bool | None] | None:  # noqa: ANN401
        """Per segment: True compiled, False eager, None not tuned; None when no record matches `extra`."""
        path = self.base.with_suffix(".tune.json")
        try:
            got = json.loads(path.read_text()).get(self._tune_key(extra))
        except (OSError, ValueError):
            return None
        return got if isinstance(got, list) else None

    def save_tune(self, extra: Any, use: list[bool | None]) -> None:  # noqa: ANN401
        """Record the per-segment decisions `use` (see `load_tune`) under `extra`; best effort."""
        path = self.base.with_suffix(".tune.json")
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        data[self._tune_key(extra)] = use
        _write(path, json.dumps(data).encode())


def _write(path: Path, data: bytes) -> None:
    """Atomic write (temp file in the same directory, then rename); best effort: a full disk is no error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError:
        pass
