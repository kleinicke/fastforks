"""The authors' original code for the accuracy evals, in three precisions (`bench/bench_upstream.py` adapters).

Run with the adapter's venv (`bench_upstream.py list` names it). Tiers:

- `upstream` = fp32: the adapter's `build(ref, fp16=False)` (the authors' default code path), no autocast.
- `upstream16` = the authors' own precision option where the code has one (`how` "authors flag": `build(ref, True)`,
  e.g. RAFT-Stereo `mixed_precision`; VGGT / VGGT-Omega / MapAnything / Depth-Anything-3 ship bfloat16 autocast,
  recorded as dtype bfloat16; Depth Pro / Marigold load a float16 model), else `torch.autocast("cuda", float16)`
  around the call (`how` "autocast").
- `upstreamac` = `torch.autocast("cuda", float16)` around the default (fp32) build also for families with an own option
  (the fallback where that option does not match fp32).
- `upstreamhalf` = `.half()` (families without an own option): `build(ref, False)`, then every `torch.nn.Module` the
  returned callable reaches (closure cells, bound objects, dicts / lists) is converted with `.half()`, float tensors
  held in dicts too, and every module casts float32 floating inputs to float16 (as a user's `model.half();
  model(x.half())`). TorchScript modules are wrapped (no hooks there).

The docs' "original fp16" column is the faster of `upstream16` / `upstreamhalf` whose accuracy matches fp32 (the
authors' option when it does); `own_option(task, ref)` says whether `upstreamhalf` is needed at all.

The bench module is `bench/bench_upstream.py` next to the library checkout (`FF_BENCH_DIR` overrides); it imports the
library from this checkout (`FF_UPBENCH_WT`, set here) with its metadata stand-ins, so `import fastforks` works in the
reference venvs after `bench()`.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("FF_BENCH_DIR", ROOT.parent / "bench"))
TIERS = {"upstream": "fp32", "upstream16": "autocast", "upstreamhalf": "half", "upstreamac": "autocast-plain"}
BF16 = {"vggt", "vggt_omega", "mapanything", "depth_anything_3"}  # the authors' mixed precision is bfloat16
NATIVE_HALF = {"depth_pro", "marigold"}  # build(ref, True) is a float16 model, not mixed precision
_SETUP: set[str] = set()


def bench() -> Any:  # noqa: ANN401 - the bench_upstream module
    """`bench_upstream` imported once (it puts this checkout's src/ and tools/golden on sys.path)."""
    os.environ.setdefault("FF_UPBENCH_WT", str(ROOT))
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    return importlib.import_module("bench_upstream")


def full_ref(task: str, ref: str) -> tuple[str, Any, Any]:
    """`id` or `id@rev` -> (`id@rev`, card, variant) of the registry."""
    from fastforks._core import registry  # noqa: PLC0415

    for k, (card, variant) in registry.cards(task).items():
        if ref in (k, k.split("@")[0], variant.ref):
            return variant.ref, card, variant  # `id@rev`: the bench adapters look variants up by it
    raise KeyError(ref)


@dataclass
class Original:
    """A built original: `prep(args)` (input conversion, untimed) then `__call__` (the timed call)."""

    adapter: Any
    fn: Any
    outer: bool  # torch.autocast(float16) around the call
    dtype: str
    how: str
    note: str = ""
    keep: list[Any] = field(default_factory=list)

    def prep(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """The adapter's conversion of uint8 CUDA CHW inputs to what its code takes."""
        return self.adapter.prep(args) if hasattr(self.adapter, "prep") else args

    def __call__(self, *args: Any) -> Any:  # noqa: ANN401
        """The original call under `no_grad` (and autocast for the autocast tier)."""
        import torch  # noqa: PLC0415

        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=self.outer):
            return self.fn(*args)


def load(task: str, ref: str, tier: str) -> Original:
    """Build `ref`'s original in `tier` (`upstream`, `upstream16`, `upstreamhalf`)."""
    bu = bench()
    full, card, _ = full_ref(task, ref)
    adapter = bu.FAM.get(card.family)
    if adapter is None:
        msg = f"no original-code adapter for family {card.family}"
        raise NotImplementedError(msg)
    if card.family not in _SETUP:
        adapter.setup()
        _SETUP.add(card.family)
    how = TIERS[tier]
    fam = card.family
    adapter.note = ""
    build = _SPECIAL.get(full.split("@")[0], adapter.build)
    if how == "fp32":
        o = Original(adapter, build(full, False), outer=False, dtype="float32", how="fp32")  # noqa: FBT003
    elif how == "autocast" and fam in NATIVE_HALF:
        o = Original(adapter, build(full, True), outer=False, dtype="float16", how="authors flag",  # noqa: FBT003
                     note="the authors' float16 model option")  # fmt: skip
    elif how == "autocast" and adapter.flag:
        dt = "bfloat16" if fam in BF16 else "float16"
        o = Original(adapter, build(full, True), outer=False, dtype=dt, how="authors flag",  # noqa: FBT003
                     note=f"the authors' mixed-precision option ({dt})")  # fmt: skip
    elif how in {"autocast", "autocast-plain"}:  # upstreamac: autocast also where the authors have an own option
        o = Original(adapter, build(full, False), outer=True, dtype="float16", how="autocast",  # noqa: FBT003
                     note="torch.autocast(float16) around the call")  # fmt: skip
    else:
        fn = build(full, False)  # noqa: FBT003
        n = halve(fn, adapter)
        o = Original(adapter, fn, outer=False, dtype="float16", how="half", note=f".half() on {n} modules, fp16 inputs")
    if getattr(adapter, "note", ""):
        o.note = "; ".join(x for x in (o.note, adapter.note) if x)
    return o


def _moge3(full: str, fp16: bool) -> Any:  # noqa: ANN401, FBT001 - the adapter's build(ref, fp16) signature
    """MoGe-3 (the bench adapter skips it): upstream `moge.model.v3` with FlexGEMM, as `tools/golden/moge.py`."""
    import flex_gemm  # noqa: PLC0415 - reference-only dependency (004 venv)  # ty: ignore[unresolved-import]
    from moge.model.v3 import MoGeModel  # noqa: PLC0415  # ty: ignore[unresolved-import]

    flex_gemm.config.SPCONV_ALLOW_TF32 = False
    net = MoGeModel.from_pretrained(bench().wfile(f"{full.split('@', maxsplit=1)[0]}/model.pt")).cuda().eval()
    return lambda x: net.infer(x[None].float() / 255, resolution_level=9, apply_mask=False, use_fp16=fp16)["depth"]


_SPECIAL = {"moge-3-vitl": _moge3}  # ref -> build(ref, fp16) where the bench adapter has none


def own_option(task: str, ref: str) -> bool:
    """Whether the authors' code has its own fp16 / mixed-precision option (then `upstreamhalf` is not run)."""
    bu = bench()
    _, card, _ = full_ref(task, ref)
    a = bu.FAM.get(card.family)
    return card.family in NATIVE_HALF or bool(a is not None and a.flag)


def _cast(x: Any, src: Any, dst: Any) -> Any:  # noqa: ANN401
    import torch  # noqa: PLC0415

    if isinstance(x, torch.Tensor):
        return x.to(dst) if x.dtype == src else x
    if isinstance(x, list):
        return [_cast(v, src, dst) for v in x]
    if isinstance(x, tuple):
        vals = [_cast(v, src, dst) for v in x]
        return type(x)(*vals) if hasattr(x, "_fields") else tuple(vals)
    if isinstance(x, dict):
        for k in list(x):
            x[k] = _cast(x[k], src, dst)
        return x
    return x


def _reach(obj: Any, out: dict[int, Any], cells: list[Any], seen: set[int], depth: int = 0) -> None:  # noqa: ANN401, C901, PLR0911, PLR0912
    """Modules reachable from `obj`; float tensors in dicts are halved on the way; script modules' cells recorded."""
    import torch  # noqa: PLC0415

    if id(obj) in seen or depth > 6:  # noqa: PLR2004
        return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        out[id(obj)] = obj
        return
    if isinstance(obj, (type, types.ModuleType, str, bytes, int, float, bool, torch.Tensor)) or obj is None:
        return
    if isinstance(obj, types.MethodType):
        _reach(obj.__self__, out, cells, seen, depth + 1)
        _reach(obj.__func__, out, cells, seen, depth + 1)
        return
    if isinstance(obj, types.FunctionType):
        for c in obj.__closure__ or ():
            try:
                v = c.cell_contents
            except ValueError:
                continue
            if isinstance(v, torch.jit.ScriptModule):
                cells.append(c)
            _reach(v, out, cells, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                obj[k] = v.half()
            else:
                _reach(v, out, cells, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _reach(v, out, cells, seen, depth + 1)
        return
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for v in list(d.values()):
            _reach(v, out, cells, seen, depth + 1)


class _HalfIn:
    """A TorchScript module called with float32 inputs cast to float16."""

    def __init__(self, m: Any) -> None:  # noqa: ANN401
        self.m = m

    def __call__(self, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        return self.m(*_cast(a, torch.float32, torch.float16), **_cast(kw, torch.float32, torch.float16))

    def __getattr__(self, k: str) -> Any:  # noqa: ANN401
        return getattr(self.m, k)


def halve(fn: Any, adapter: Any = None) -> int:  # noqa: ANN401
    """`.half()` every module `fn` (and `adapter`) reaches; float32 module inputs cast to float16. -> module count."""
    import torch  # noqa: PLC0415

    found: dict[int, Any] = {}
    cells: list[Any] = []
    seen: set[int] = set()
    _reach(fn, found, cells, seen)
    if adapter is not None:
        for v in list(vars(adapter).values()):
            _reach(v, found, cells, seen)

    def pre(_m: Any, args: Any, kwargs: Any) -> Any:  # noqa: ANN401
        return _cast(args, torch.float32, torch.float16), _cast(kwargs, torch.float32, torch.float16)

    n = 0
    for m in found.values():
        m.half()
        for sub in m.modules():
            if not isinstance(sub, torch.jit.ScriptModule):
                sub.register_forward_pre_hook(pre, with_kwargs=True)
        n += 1
    for c in cells:
        c.cell_contents = _HalfIn(c.cell_contents)
    return n
