# ruff: noqa: SLF001 - patches library modules in place
"""Generic passes: re-parameterization, BatchNorm folding, fx flattening, fp16-accumulate cuDNN copy."""

from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import threading
from collections import Counter
from typing import TYPE_CHECKING

import torch
from torch import fx, nn

from fastforks._core.patch import base_type, is_base, override

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

log = logging.getLogger(__name__)

# modules with data-dependent Python control flow: kept as leaf calls inside the fx graph
FX_LEAVES = ("LiteMLA",)


_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
# transformers' output-capture hooks only record outputs (inactive unless asked); `output_hidden_states` through a
# rewritten block works as before this check
_RECORDER = "transformers.utils.output_capturing"


def hooked(m: nn.Module) -> bool:
    """`m` has forward or backward hooks of its own (e.g. weight_norm): a pass must keep the module as is."""
    return any(getattr(f, "__module__", None) != _RECORDER for h in _HOOKS for f in getattr(m, h, {}).values())


def _reparam_tree(module: nn.Module) -> None:
    """The walk of timm's `reparameterize_model` (timm.utils.model), without importing timm."""
    for name, child in module.named_children():
        if hasattr(child, "fuse"):  # blocks that return a fused module
            setattr(module, name, child.fuse())  # ty: ignore[call-non-callable]
        elif hasattr(child, "reparameterize"):
            child.reparameterize()
        elif hasattr(child, "switch_to_deploy"):
            child.switch_to_deploy()
        _reparam_tree(child)


def _timm_fuse(m: nn.Module) -> bool:
    """A timm block with a `fuse()` method (the class's, not a submodule named `fuse` as other libraries have)."""
    return type(m).__module__.startswith("timm.") and callable(getattr(type(m), "fuse", None))


def reparam(model: nn.Module) -> bool:
    """Re-parameterize timm blocks (FastViT / MobileOne: `reparameterize` / `switch_to_deploy`; RepViT: `fuse`)."""
    if any(hasattr(m, "reparameterize") or hasattr(m, "switch_to_deploy") or _timm_fuse(m) for m in model.modules()):
        _reparam_tree(model)
        return True
    return False


def _probe(model: nn.Module, size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    t = next(itertools.chain(model.parameters(), model.buffers()), None)
    dev = t.device if t is not None else torch.device("cpu")
    for sz in (size, 224):
        try:
            x0 = torch.randn(1, 3, sz, sz, device=dev, generator=torch.Generator(dev).manual_seed(0))
            with torch.inference_mode():
                ref = model(x0)
        except Exception:  # noqa: BLE001, S112 - fixed-size models fail at the small size
            continue
        if isinstance(ref, torch.Tensor):
            return x0, ref
        return None
    return None


class _PairTracer(fx.Tracer):
    """Conv2d and BatchNorm2d (with subclasses such as timm BatchNormAct2d) as leaf calls, and the modules `leaves`."""

    def __init__(self, leaves: set[int]) -> None:
        super().__init__()
        self.leaves = leaves

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return (
            isinstance(m, (nn.Conv2d, nn.BatchNorm2d))
            or type(m).__name__ in FX_LEAVES
            or id(m) in self.leaves
            or super().is_leaf_module(m, module_qualified_name)
        )


def _sequential_pairs(model: nn.Module) -> set[tuple[int, int]]:
    """(id(conv), id(bn)): a conv directly followed by its BN in an `nn.Sequential`, each module used only there."""
    seen: dict[int, int] = {}
    for parent in model.modules():
        for ch in parent._modules.values():
            if ch is not None:
                seen[id(ch)] = seen.get(id(ch), 0) + 1
    proven: set[tuple[int, int]] = set()
    for parent in model.modules():
        # nn.Sequential or a subclass with its forward (torchvision Conv2dNormActivation), not replaced on the instance
        if (
            isinstance(parent, nn.Sequential)
            and base_type(parent).forward is nn.Sequential.forward
            and ("forward" not in parent.__dict__)
        ):
            for c, bn in itertools.pairwise(parent._modules.values()):
                if is_base(c, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and seen[id(c)] == seen[id(bn)] == 1:
                    proven.add((id(c), id(bn)))
    return proven


def _graph_pairs(model: nn.Module, graph: fx.Graph) -> set[tuple[int, int]]:
    """(id(conv), id(bn)) from an fx graph: each called once, the BN's only argument the conv's only consumer."""
    mods = dict(model.named_modules(remove_duplicate=False))
    calls: dict[int, int] = {}
    for n in graph.nodes:
        if n.op == "call_module":
            calls[id(mods[n.target])] = calls.get(id(mods[n.target]), 0) + 1
    proven: set[tuple[int, int]] = set()
    for n in graph.nodes:
        if n.op != "call_module" or not isinstance(bn := mods[n.target], nn.BatchNorm2d) or len(n.args) != 1:
            continue
        src = n.args[0]
        if n.kwargs or not isinstance(src, fx.Node) or src.op != "call_module":
            continue
        c = mods[src.target]
        if is_base(c, nn.Conv2d) and len(src.users) == 1 and calls[id(c)] == calls[id(bn)] == 1:
            proven.add((id(c), id(bn)))
    return proven


_CONV_BN = (nn.Conv2d, nn.BatchNorm2d)


def _pair_graph(m: nn.Module, leaves: set[int]) -> fx.Graph | None:
    try:
        return _PairTracer(leaves).trace(m)
    except Exception as e:  # noqa: BLE001 - untraceable (e.g. input-shape dependent control flow)
        log.debug("BN fold: fx trace of %s failed (%s)", type(m).__name__, type(e).__name__)
        return None


def _proven_pairs(model: nn.Module) -> set[tuple[int, int]]:
    """(id(conv), id(bn)) pairs whose BN provably reads only the conv's output, for every input.

    Structural, not from a probe input: fx graphs with Conv2d / BatchNorm2d as leaves (`_graph_pairs`) and plain
    `nn.Sequential` pairs (`_sequential_pairs`). A module whose trace fails is traced again with its untraceable
    descendants as leaves; if it still fails, its own Python code (e.g. input-shape dependent control flow) may call any
    module below it directly, on a path no probe saw, so no pair below it counts. Only the untraceable FX_LEAVES (known
    code) and containers without a forward (ModuleList) have their children traced instead. Leaves of a graph are
    traced on their own. Without pairs whose conv or BN parameters a traced forward also reads directly (`get_attr`).
    """
    proven = _sequential_pairs(model)
    unsafe: set[int] = set()
    opaque: set[int] = set()  # modules whose own code is untraceable
    done: set[int] = set()

    def visit(m: nn.Module) -> None:
        nonlocal proven
        if id(m) in done:
            return
        done.add(id(m))
        graph = _pair_graph(m, set())
        if graph is None:
            for c in m.children():
                if not isinstance(c, _CONV_BN):
                    visit(c)
            if type(m).__name__ in FX_LEAVES or type(m).forward is nn.Module.forward:
                return  # known library code calling each child once per path, or no code (ModuleList)
            below = opaque.intersection(id(d) for d in m.modules())
            graph = _pair_graph(m, below) if below else None
            if graph is None:
                opaque.add(id(m))
                unsafe.update(id(d) for d in m.modules())
                return
        proven |= _graph_pairs(m, graph)
        unsafe.update(id(m.get_submodule(str(n.target).rpartition(".")[0])) for n in graph.nodes if n.op == "get_attr")
        for n in graph.nodes:  # leaves with children (FX_LEAVES, e.g. timm LiteMLA): traced on their own
            leaf = m.get_submodule(n.target) if n.op == "call_module" else None
            if leaf is not None and not isinstance(leaf, _CONV_BN) and next(leaf.children(), None) is not None:
                visit(leaf)

    visit(model)
    return {p for p in proven if p[0] not in unsafe and p[1] not in unsafe}


def fold_bn_hooks(model: nn.Module, size: int = 64) -> int:  # noqa: C901, PLR0912, PLR0915 - one search + validate loop
    """Fold every eval BatchNorm2d whose input is provably exactly the output of a Conv2d.

    Candidates come from one tiny fp32 forward with hooks (works for timm BatchNormAct2d); each must also be proven
    structurally (`_proven_pairs`: fx graph or plain `nn.Sequential`), since one probe input does not show
    input-dependent paths. The BN module is replaced by its activation (timm BatchNormAct2d: drop + act) or
    nn.Identity. A fold that changes the probe output is undone (an extra check). Returns the number folded.
    """
    convs: dict[int, tuple[nn.Module, torch.Tensor]] = {}
    pairs: list[tuple[nn.Conv2d, nn.BatchNorm2d]] = []
    hooks = []

    def conv_hook(m: nn.Module, _inp: object, out: torch.Tensor) -> None:
        convs[id(out)] = (m, out)

    def bn_hook(m: nn.Module, inp: tuple[torch.Tensor, ...], _out: object) -> None:
        c = convs.get(id(inp[0]))
        if c is not None and c[1] is inp[0]:
            pairs.append((c[0], m))  # ty: ignore[invalid-argument-type] - registered on Conv2d / BatchNorm2d only

    for m in model.modules():
        if is_base(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(bn_hook))
    try:
        probe = _probe(model, size)
    finally:
        for h in hooks:
            h.remove()
    if probe is None:
        return 0
    x0, ref = probe
    parent_of: dict[int, list[tuple[nn.Module, str]]] = {}
    for parent in model.modules():
        for k, ch in parent._modules.items():
            if ch is not None:
                parent_of.setdefault(id(ch), []).append((parent, k))
    nparam = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    nbn: dict[int, int] = {}
    ncv: dict[int, int] = {}
    for c, bn in pairs:
        nbn[id(bn)] = nbn.get(id(bn), 0) + 1
        ncv[id(c)] = ncv.get(id(c), 0) + 1
    cand: list[tuple[nn.Conv2d, nn.BatchNorm2d]] = []
    seen: set[int] = set()
    proven = _proven_pairs(model) if pairs else set()
    for c, bn in pairs:
        if bn.running_var is None or bn.running_mean is None:
            continue  # no running stats (track_running_stats=False): normalizes with batch stats, stays as is
        if hooked(c) or hooked(bn):
            continue  # a hook sees (or changes) the conv output / the BN module: both stay as they are
        if any(nparam[id(p)] > 1 for p in (c.weight, c.bias) if p is not None):
            continue  # a Parameter shared with another module (tied weights): folding would change that module too
        if nbn[id(bn)] == 1 and ncv[id(c)] == 1 and id(bn) not in seen and (id(c), id(bn)) in proven:
            seen.add(id(bn))
            cand.append((c, bn))

    def apply(c: nn.Conv2d, bn: nn.BatchNorm2d) -> Callable[[], None]:
        old_w, old_b = c.weight.data, c.bias
        with torch.no_grad():
            rv, rm = bn.running_var, bn.running_mean
            assert rv is not None  # noqa: S101 - eval BN with running stats (hooked above)
            assert rm is not None  # noqa: S101
            sc = (bn.weight.float() if bn.affine else 1) / torch.sqrt(rv.float() + bn.eps)
            b = (bn.bias.float() if bn.affine else 0) - rm.float() * sc
            c.weight.data = (c.weight.float() * sc.view(-1, 1, 1, 1)).to(c.weight.dtype)
            cb = (c.bias.float() * sc + b) if c.bias is not None else b
            c.bias = nn.Parameter(cb.to(c.weight.dtype))
        act, drop = getattr(bn, "act", None), getattr(bn, "drop", None)
        if isinstance(act, nn.Module) and not isinstance(act, nn.Identity):
            r = act if not isinstance(drop, nn.Module) or isinstance(drop, nn.Identity) else nn.Sequential(drop, act)
        else:
            r = nn.Identity()
        for parent, k in parent_of.get(id(bn), []):
            parent._modules[k] = r

        def undo() -> None:
            c.weight.data, c.bias = old_w, old_b
            for parent, k in parent_of.get(id(bn), []):
                parent._modules[k] = bn

        return undo

    def ok() -> bool:
        with torch.inference_mode():
            y = model(x0)
        return bool(((y - ref).abs().max() <= 1e-3 * (ref.abs().max() + 1e-6)).item())

    undos = [apply(c, bn) for c, bn in cand]
    if ok():
        return len(cand)
    for u in reversed(undos):  # some conv output has a second consumer: fold one by one, keep the exact ones
        u()
    n = 0
    for c, bn in cand:
        u = apply(c, bn)
        if ok():
            n += 1
        else:
            u()
    return n


class _Tracer(fx.Tracer):
    """Modules with hooks stay calls (their hooks run), as do FX_LEAVES."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return type(m).__name__ in FX_LEAVES or hooked(m) or super().is_leaf_module(m, module_qualified_name)


def flatten(model: nn.Module) -> fx.GraphModule | None:
    """Fx symbolic trace without nn.Identity calls (no nn.Module call overhead), `view` as `reshape`.

    None if untraceable or if the model itself has hooks (a GraphModule would not run them).
    """
    if hooked(model):
        return None
    try:
        graph = _Tracer().trace(model)
        gm = fx.GraphModule(model, graph)
    except Exception as e:  # noqa: BLE001 - any trace failure keeps the eager module
        log.debug("fx trace failed (%s: %s), eager modules kept", type(e).__name__, str(e)[:80])
        return None
    mods = dict(gm.named_modules())
    for n in list(gm.graph.nodes):
        if n.op == "call_module" and isinstance(mods.get(n.target), nn.Identity) and not hooked(mods[n.target]):
            n.replace_all_uses_with(n.args[0])
            gm.graph.erase_node(n)
        elif (
            n.op == "call_method"
            and n.target == "view"
            and not any(isinstance(a, torch.dtype) for a in (*n.args, *n.kwargs.values()))
        ):
            n.target = "reshape"  # channels-last activations: `view` can fail where `reshape` (a view if it can) works
    gm.graph.lint()
    gm.recompile()
    return gm


@contextlib.contextmanager
def _no_tf32() -> Generator[None]:
    """True float32 convs / GEMMs inside the block (restored after).

    The fold check compares fp32 forwards at 1e-3 of the output range; with PyTorch's default TF32 convolutions on CUDA
    the rounding alone failed it for some convs (ResNet-18: 5 of 20), which fx `fuse` then folded into new modules,
    leaving the user's fp32 originals on the device (+20 MB). Process-wide flags: a concurrent float32 call only gets
    more precise for the moment.
    """
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


_FOLD_LOCK = threading.Lock()  # fx tracing patches `nn.Module.__call__` process-wide: loads fold one at a time


def fold_bn(model: nn.Module) -> tuple[nn.Module, dict[str, int]]:
    """Re-parameterize (timm), fold BN by hooks, fx flatten, fx conv/BN fuse for anything left that fx sees."""
    with _FOLD_LOCK:
        return _fold_bn(model)


def module_uses(graph: fx.Graph) -> Counter[str]:
    """Graph nodes per module path: its calls plus its parameter / buffer reads (`get_attr`).

    A pass that changes a module's weights needs exactly one: a shared (tied) module would change its other uses too.
    """
    return Counter(
        str(n.target) if n.op == "call_module" else str(n.target).rpartition(".")[0]
        for n in graph.nodes
        if n.op in {"call_module", "get_attr"}
    )


_FUSE_PATTERNS = (
    (nn.Conv1d, nn.BatchNorm1d),
    (nn.Conv2d, nn.BatchNorm2d),
    (nn.Conv3d, nn.BatchNorm3d),
)


def _fuse(model: fx.GraphModule) -> fx.GraphModule:
    """`torch.fx.experimental.optimization.fuse` (conv + BN), minus layers with more than one use or hooks.

    Not Linear + BatchNorm1d: for [N, C, L] input the BN normalizes axis 1, the Linear acts on the last.
    """
    from torch.fx.experimental.optimization import matches_module_pattern, replace_node_module  # noqa: PLC0415
    from torch.nn.utils.fusion import fuse_conv_bn_eval  # noqa: PLC0415

    if any(hooked(m) for m in model.modules()):
        return model  # the retrace below would inline modules with hooks
    gm = fx.symbolic_trace(model)  # as torch's fuse: the same graph, so the same fusions
    modules = dict(gm.named_modules())
    graph = copy.deepcopy(gm.graph)
    uses = module_uses(graph)
    for pattern in _FUSE_PATTERNS:
        for node in graph.nodes:
            if not matches_module_pattern(pattern, node, modules):
                continue
            src, bn = node.args[0], modules[str(node.target)]
            if len(src.users) > 1 or uses[str(src.target)] > 1 or not bn.track_running_stats:
                continue
            layer = modules[str(src.target)]
            fused = fuse_conv_bn_eval(layer, bn)
            replace_node_module(src, modules, fused)
            node.replace_all_uses_with(src)
            graph.erase_node(node)
    return fx.GraphModule(gm, graph)


def _fold_bn(model: nn.Module) -> tuple[nn.Module, dict[str, int]]:
    with _no_tf32():
        rep = {"reparam": int(reparam(model)), "bn_folded": fold_bn_hooks(model)}
    gm = flatten(model)
    rep["fx"] = int(gm is not None)
    if gm is None:
        return model, rep
    try:
        gm = _fuse(gm)
    except Exception as e:  # noqa: BLE001 - optional extra fold
        log.debug("fx conv/bn fuse skipped (%s)", type(e).__name__)
    return gm, rep


def _hacc_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    from fastforks.kernels.cudnn_hacc import conv_half_acc  # noqa: PLC0415 - optional cuDNN frontend

    return conv_half_acc(x, self.weight, self.bias, list(self.padding), list(self.stride))  # ty: ignore[invalid-argument-type]


def hacc_copy(model: nn.Module) -> nn.Module:
    """Shallow copy of the module tree whose dense fp16 convs (groups 1) run as fp16-accumulate cuDNN convs.

    Parameters are shared with `model` (no extra memory).
    """
    memo: dict[int, nn.Module] = {}

    def clone(mod: nn.Module) -> nn.Module:
        if id(mod) in memo:
            return memo[id(mod)]
        c = copy.copy(mod)
        c._modules = {k: (clone(v) if v is not None else None) for k, v in mod._modules.items()}
        memo[id(mod)] = c
        if (
            is_base(mod, nn.Conv2d)
            and mod.groups == 1
            and mod.dilation == (1, 1)
            and mod.padding_mode == "zeros"
            and not isinstance(mod.padding, str)
            and mod.weight.dtype == torch.half
        ):
            mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
            override(c, _hacc_forward)
        return c

    return clone(model)
