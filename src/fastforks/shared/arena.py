"""One contiguous allocation for a built network's weights (no allocator fragmentation in the reserved memory)."""

from __future__ import annotations

import torch
from torch import nn

_ALIGN = 512  # bytes: every tensor starts aligned (kernels load 16-byte vectors)


def compact(module: nn.Module) -> nn.Module:
    """Move the parameters and buffers of `module` on a CUDA device into one arena per (device, dtype).

    A build that casts, folds or packs tensors frees blocks between the live weights; the caching allocator cannot
    return a segment that still holds a live tensor, so the freed holes stay in the *reserved* memory for the life of
    the model (DINOv3 ViT-L: +160 MB over its 616 MB). The copies here leave the old tensors unreferenced: the
    allocator's `empty_cache` after the build (done by `load`) frees their segments. Shared storage (tied weights) stays
    shared; values, dtypes, shapes and strides (contiguous only; others are left alone) are unchanged.

    Args:
        module: The network; its tensors are replaced in place (`Parameter.data`, buffer entries).

    Returns:
        `module`.
    """
    slots: dict[tuple[torch.device, torch.dtype], list[tuple[nn.Module, str, torch.Tensor, bool]]] = {}
    seen: set[int] = set()
    for mod in module.modules():
        for name, t in (*mod._parameters.items(), *mod._buffers.items()):  # noqa: SLF001 - the tensor registries
            if t is None or not t.is_cuda or not t.is_contiguous() or id(t) in seen:
                continue
            seen.add(id(t))
            slots.setdefault((t.device, t.dtype), []).append((mod, name, t, name in mod._parameters))  # noqa: SLF001
    shared = {}  # storage pointer -> the tensors viewing it must stay views of one storage: skip such storages
    for items in slots.values():
        for _, _, t, _ in items:
            shared[t.untyped_storage().data_ptr()] = shared.get(t.untyped_storage().data_ptr(), 0) + 1
    with torch.no_grad():
        for (device, dtype), items in slots.items():
            items = [i for i in items if shared[i[2].untyped_storage().data_ptr()] == 1]  # noqa: PLW2901
            step = _ALIGN // torch.empty((), dtype=dtype).element_size()
            sizes = [-(-t.numel() // step) * step for _, _, t, _ in items]
            if not items:
                continue
            arena = torch.empty(sum(sizes), dtype=dtype, device=device)
            at = 0
            for (mod, name, t, is_param), size in zip(items, sizes, strict=True):
                new = arena[at : at + t.numel()].view(t.shape)
                new.copy_(t)
                at += size
                if is_param:
                    mod._parameters[name].data = new  # noqa: SLF001
                else:
                    mod._buffers[name] = new  # noqa: SLF001
    return module
