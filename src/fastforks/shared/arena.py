"""One contiguous allocation for a built network's weights (no allocator fragmentation in the reserved memory)."""

from __future__ import annotations

import torch
from torch import nn

_MIN_BYTES = 64 << 20  # networks below this are left alone (squeezenet / u2netp: reserved +14 MB with an arena)
_ALIGN = 512  # bytes: every tensor starts aligned (kernels load 16-byte vectors)


def _room(device: torch.device) -> int:
    """Bytes a new allocation can use on `device`: the free device memory (less a 10 % margin) plus the cached free blocks."""
    free, _ = torch.cuda.mem_get_info(device)
    return int(free * 0.9) + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def compact(module: nn.Module, min_bytes: int = _MIN_BYTES) -> nn.Module:
    """Move the storages of the parameters and buffers of `module` on a CUDA device into one arena per device (a device's tensors of any dtype share it).

    A build that casts, folds or packs tensors frees blocks between the live weights; the caching allocator cannot
    return a segment that still holds a live tensor, so the freed holes stay in the *reserved* memory for the life of
    the model (DINOv3 ViT-L: +160 MB over its 616 MB). The copies here leave the old storages unreferenced: the
    allocator's `empty_cache` after the build (done by `load`) frees their segments. A storage is moved whole, so tensors
    viewing one storage (tied weights, packed qkv) stay views of one storage; values, dtypes, shapes, strides and
    storage offsets are unchanged. Skipped: networks under `min_bytes` (64 MB), storages with a tensor of another dtype than the rest, and a device where the
    old weights and the arena do not fit at once.

    Args:
        module: The network; its tensors are replaced in place (`Parameter.data`, buffer entries).
        min_bytes: Networks with fewer weight bytes are left alone (0: always compact, as DINOv3 does).

    Returns:
        `module`.
    """
    users: dict[int, list[tuple[nn.Module, str, torch.Tensor, bool]]] = {}  # storage pointer -> the tensors viewing it
    seen: set[int] = set()
    for mod in module.modules():
        for name, t in (*mod._parameters.items(), *mod._buffers.items()):  # noqa: SLF001 - the tensor registries
            if t is None or not t.is_cuda:
                continue
            users.setdefault(t.untyped_storage().data_ptr(), []).append((mod, name, t, name in mod._parameters))  # noqa: SLF001
            seen.add(id(t))
    by_device: dict[torch.device, list[list[tuple[nn.Module, str, torch.Tensor, bool]]]] = {}
    for items in users.values():
        t = items[0][2]
        storage = t.untyped_storage()
        nbytes = storage.nbytes()
        if torch._C._storage_Use_Count(storage._cdata) - 1 != len({id(i[2]) for i in items}):  # noqa: SLF001
            continue  # a tensor outside the registries (a plain attribute, a kernel's cache) views it: moving would copy it
        if all(i[2].element_size() == t.element_size() and nbytes % t.element_size() == 0 for i in items):
            by_device.setdefault(t.device, []).append(items)
    with torch.no_grad():
        for device, groups in by_device.items():
            sizes = [-(-g[0][2].untyped_storage().nbytes() // _ALIGN) * _ALIGN for g in groups]
            if (
                sum(sizes) < min_bytes
            ):  # the holes of a small network are a few MB: the arena's own rounding costs as much
                continue
            if sum(sizes) > _room(device):  # the copy needs the old weights and the arena at once
                continue
            arena = torch.empty(sum(sizes), dtype=torch.uint8, device=device)
            at = 0
            done: dict[int, torch.Tensor] = {}  # id(old tensor) -> its new view (a tensor in two modules)
            for items, size in zip(groups, sizes, strict=True):
                nbytes = items[0][2].untyped_storage().nbytes()
                flat = arena[at : at + nbytes]
                flat.copy_(
                    torch.empty(0, dtype=torch.uint8, device=device).set_(items[0][2].untyped_storage(), 0, (nbytes,))
                )
                at += size
                for mod, name, t, is_param in items:
                    new = done.get(id(t))
                    if new is None:
                        typed = flat.view(t.dtype)
                        new = torch.as_strided(typed, t.size(), t.stride(), typed.storage_offset() + t.storage_offset())
                        done[id(t)] = new
                    if is_param:
                        mod._parameters[name].data = new  # noqa: SLF001
                    else:
                        mod._buffers[name] = new  # noqa: SLF001
    return module


def compact_runner(runner: object) -> None:
    """`compact` every network of a built runner: `runner.networks()`, else (kernels runners have no exportable module)
    the `nn.Module`s its attributes hold, directly or in a list / tuple / dict."""
    try:
        nets = list(runner.networks())  # type: ignore[attr-defined]
    except NotImplementedError:
        nets = []
        for v in vars(runner).values():
            nets += [
                x
                for x in (v.values() if isinstance(v, dict) else v if isinstance(v, (list, tuple)) else (v,))
                if isinstance(x, nn.Module)
            ]
    nested = {id(m) for n in nets for m in n.modules() if m is not n}  # compacting a part again would copy its arena
    for net in {id(n): n for n in nets if id(n) not in nested}.values():
        compact(net)
