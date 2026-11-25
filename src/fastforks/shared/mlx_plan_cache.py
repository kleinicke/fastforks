"""Opt-in on-disk cache of the MLX graph translation's slow first-call steps (`FASTFORKS_MLX_PLAN_CACHE=<dir>`).

The first call of an `MlxModule` at an input signature exports the network (`torch.export` and the core-ATen
decomposition: 1 to 7 s on the M1 Pro), translates it and, on its first replay, times each big compiled segment once
eagerly and once compiled (the tuning pass). A new process pays all of it again. With the cache on:

- the decomposed graph is saved as the little the translation reads (`<key>.graph`, a pickle of each node's op, name,
  target, arguments and output shape/dtype plus the input signature), not as an `ExportedProgram`:
  `torch.export.load` took 1 to 2 s, more than the export it replaced saved. The network's weights are not stored:
  every parameter and buffer is bound to the module's own tensor (same name, shape and dtype, else the entry is not
  used) when it is loaded; only the constants the forward builds are stored.
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
import json
import os
import pickle
import tempfile
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from collections.abc import Generator

ENV = "FASTFORKS_MLX_PLAN_CACHE"
FORMAT = 2  # bump when the stored layout or what the translation reads from the program changes

_SCOPE: contextvars.ContextVar[tuple[str, list[int]] | None] = contextvars.ContextVar("mlx_plan_scope", default=None)
# env switches that change the segmentation or the compile policy: part of a tuning record's key
_TUNE_ENV = ("FF_MLX_SEG_MB", "FF_MLX_CHUNK_MB", "FF_MLX_COMPILE", "FF_MLX_NO_BAND", "FF_MLX_NO_CL", "FF_MLX_NO_TUNE")


def directory() -> Path | None:
    """The cache directory (`FASTFORKS_MLX_PLAN_CACHE`), or None when the cache is off."""
    d = os.environ.get(ENV, "").strip()
    return Path(d).expanduser() if d else None


@contextlib.contextmanager
def scope(name: str) -> Generator[None]:
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
    """A cache entry: the exported program and call metadata (`<key>.graph`), tuning decisions (`<key>.tune.json`)."""

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
        """`(the stored program with the module's own tensors bound, metadata)`, or None (miss or unusable entry).

        The program is rebuilt from `_GraphFile` as a plain `torch.fx.Graph` (no `torch.export.load`, whose import and
        deserialisation cost about as much as the trace on a big net); see `_Program` for what it provides.
        """
        path = self.base.with_suffix(".graph")
        if not path.is_file():
            return None
        try:
            stored = pickle.loads(path.read_bytes())  # noqa: S301 - our own cache directory
            if not isinstance(stored, _GraphFile):
                return None
            own = _named_tensors(module)
            stores: list[dict[str, Any]] = []
            for kept, bound in ((stored.state, stored.bound_state), (stored.constants, stored.bound_constants)):
                store = dict(kept)
                for name, (shape, dtype) in bound.items():
                    t = own.get(name)
                    if t is None or tuple(t.shape) != shape or t.dtype != dtype:
                        return None
                    store[name] = t
                stores.append(store)
            graph = stored.graph()
        except Exception:  # noqa: BLE001 - a stale or partly written entry is a miss
            return None
        return _Program(graph, stored.inputs, *stores), stored.extra

    def save_program(self, ep: Any, module: torch.nn.Module, extra: Any) -> None:  # noqa: ANN401
        """Store `ep` without the module's tensors (bound by name on load) and `extra` (picklable); best effort."""
        own = _named_tensors(module)

        def split(store: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], torch.dtype]]]:
            kept, bound = {}, {}
            for name, v in store.items():
                t = own.get(name)
                if (
                    isinstance(v, torch.Tensor)
                    and t is not None
                    and tuple(t.shape) == tuple(v.shape)
                    and t.dtype == v.dtype
                ):
                    bound[name] = (tuple(v.shape), v.dtype)
                else:  # a constant the forward built (a grid): stored as it is
                    kept[name] = v.detach().clone()
            return kept, bound

        try:
            state, bound_state = split(dict(ep.state_dict))
            constants, bound_constants = split(dict(ep.constants))
            inputs = [(s.arg.name, s.target) for s in ep.graph_signature.input_specs]
            stored = _GraphFile.of(ep.graph_module.graph, inputs, state, bound_state, constants, bound_constants, extra)
            data = pickle.dumps(stored, protocol=pickle.HIGHEST_PROTOCOL)
            if [n.name for n in stored.graph().nodes] != [n.name for n in ep.graph_module.graph.nodes]:
                return  # the rebuilt graph must have the exported one's names
        except Exception:  # noqa: BLE001 - an unserialisable program is simply not cached
            return
        _write(self.base.with_suffix(".graph"), data)

    # The tuning decisions.
    @staticmethod
    def _tune_key(extra: Any) -> str:  # noqa: ANN401
        return repr((extra, [(k, os.environ.get(k, "")) for k in _TUNE_ENV]))

    def load_tune(self, extra: Any) -> list[bool | None] | None:  # noqa: ANN401
        """Per segment: True compiled, False eager, None not tuned; None when no record matches `extra`."""
        path = self.base.with_suffix(".tune.json")
        try:
            got = json.loads(path.read_text(encoding="utf-8")).get(self._tune_key(extra))
        except (OSError, ValueError):
            return None
        return got if isinstance(got, list) else None

    def save_tune(self, extra: Any, use: list[bool | None]) -> None:  # noqa: ANN401
        """Record the per-segment decisions `use` (see `load_tune`) under `extra`; best effort."""
        path = self.base.with_suffix(".tune.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[self._tune_key(extra)] = use
        _write(path, json.dumps(data).encode())


class _Ref:
    """A graph node in a stored node's arguments, by name."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class _Meta:
    """A node's `meta["val"]` tensor in a stored graph: shape and dtype (rebuilt on the meta device)."""

    __slots__ = ("dtype", "shape")

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        self.shape, self.dtype = shape, dtype


class _OpRef:
    """An ATen overload in a stored graph, by namespace, name and overload name."""

    __slots__ = ("path",)

    def __init__(self, target: torch._ops.OpOverload) -> None:
        ns, name = target._schema.name.split("::")  # noqa: SLF001
        self.path = (ns, name, target._overloadname)  # noqa: SLF001

    def resolve(self) -> Any:  # noqa: ANN401
        ns, name, overload = self.path
        return getattr(getattr(getattr(torch.ops, ns), name), overload)


class _GraphFile:
    """A decomposed program's graph and signature in plain picklable values (`Entry.save_program`).

    Per node: op kind, name, target, args and kwargs (nodes as `_Ref`), and `meta["val"]` with every tensor as `_Meta`;
    that is all the translation reads. The constants the forward built are stored; the module's own tensors are names
    with shape and dtype (`bound_*`), bound on load.
    """

    def __init__(self) -> None:
        self.nodes: list[tuple[Any, ...]] = []
        self.inputs: list[tuple[str, str | None]] = []
        self.state: dict[str, torch.Tensor] = {}
        self.constants: dict[str, torch.Tensor] = {}
        self.bound_state: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self.bound_constants: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self.extra: Any = None

    @classmethod
    def of(cls, graph: torch.fx.Graph, inputs: list[tuple[str, str | None]], *stores: Any) -> _GraphFile:  # noqa: ANN401
        """The stored form of `graph` (stores: state, bound state, constants, bound constants, extra)."""
        out = cls()
        out.inputs = inputs
        out.state, out.bound_state, out.constants, out.bound_constants, out.extra = stores

        def ref(x: Any) -> Any:  # noqa: ANN401
            return _Ref(x.name) if isinstance(x, torch.fx.Node) else x

        def meta(x: Any) -> Any:  # noqa: ANN401
            return _Meta(tuple(int(d) for d in x.shape), x.dtype) if isinstance(x, torch.Tensor) else x

        for n in graph.nodes:
            args, kwargs = torch.fx.node.map_aggregate((n.args, n.kwargs), ref)
            val = pytree.tree_map(meta, n.meta.get("val"))
            target = _OpRef(n.target) if isinstance(n.target, torch._ops.OpOverload) else n.target  # noqa: SLF001
            out.nodes.append((n.op, n.name, target, args, kwargs, val))
        return out

    def graph(self) -> torch.fx.Graph:
        """The graph rebuilt (node names as stored; `meta["val"]` tensors on the meta device)."""
        g = torch.fx.Graph()
        env: dict[str, torch.fx.Node] = {}

        def node(x: Any) -> Any:  # noqa: ANN401
            return env[x.name] if isinstance(x, _Ref) else x

        def meta(x: Any) -> Any:  # noqa: ANN401
            return torch.empty(x.shape, dtype=x.dtype, device="meta") if isinstance(x, _Meta) else x

        is_meta = lambda x: isinstance(x, _Meta)  # noqa: E731
        for op, name, target, args, kwargs, val in self.nodes:
            a, kw = pytree.tree_map(node, (args, kwargs), is_leaf=lambda x: isinstance(x, _Ref))
            fn = target.resolve() if isinstance(target, _OpRef) else target
            n = g.create_node(op, fn, tuple(a), dict(kw), name=name)
            if n.name != name:
                msg = f"node {name} became {n.name}"
                raise ValueError(msg)
            if val is not None:
                n.meta["val"] = pytree.tree_map(meta, val, is_leaf=is_meta)
            env[name] = n
        return g


class _Program:
    """What the translation reads of an `ExportedProgram`, for a stored graph: the graph, signature, tensors."""

    def __init__(
        self,
        graph: torch.fx.Graph,
        inputs: list[tuple[str, str | None]],
        state: dict[str, torch.Tensor],
        constants: dict[str, torch.Tensor],
    ) -> None:
        self.graph_module = types.SimpleNamespace(graph=graph)
        self.graph_signature = types.SimpleNamespace(
            input_specs=[types.SimpleNamespace(arg=types.SimpleNamespace(name=n), target=t) for n, t in inputs]
        )
        self.state_dict = state
        self.constants = constants


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
