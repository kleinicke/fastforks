"""`del model` frees the model's CUDA memory by reference counting alone, without waiting for the cyclic garbage
collector: no module -> bound method -> module cycles (see `_core.patch`). One small variant per family whose backends
override module methods, both modes."""

from __future__ import annotations

import gc

import pytest
import torch

from synth import image, pair
from test_golden import API, GOLDEN, TASK
from test_kernels_structure import held_modules

MIB = 2**20


def _weight_bytes(model) -> int:
    """Bytes of the distinct CUDA storages of the parameters and buffers of the modules the model's runner holds."""
    mods = {id(s): s for r in held_modules(model._runner) for s in r.modules()}
    tensors = [t for m in mods.values() for t in (*m.parameters(recurse=False), *m.buffers(recurse=False))]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors if t.is_cuda}
    return sum(storages.values())


REFS = [
    "aanet-sceneflow@1",
    "bridgedepth-sf@1",
    "c-fast-foundationstereo@1",
    "coex-sceneflow@1",
    "defom-stereo-vitl-sceneflow@1",
    "depth-anything-v2-s@1",
    "foundationstereo-vit-s@1",
    "gwcnet-g-sceneflow@1",
    "hitnet-eth3d@1",
    "igev-stereo-kitti12@1",
    "las2-s@1",
    "metric3d-vit-large@1",
    "midas-dpt-beit-large-512@1",
    "mocha-v1-sceneflow@1",
    "monster-mix-all@1",
    "monster-plusplus-mix-all@1",
    "nmrf-sceneflow@1",
    "rt-monster-plusplus-mix-all@1",
    "s2m2-s@1",
    "selective-igev-eth3d-finetune@1",
    "unidepth-v2-vitl14@1",
    "waft-stereo-s@1",
    "zoedepth-nk@1",
]


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("ref", REFS)
def test_del_frees_cuda_memory_without_gc(ref, mode):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if ref not in GOLDEN:
        pytest.skip(f"{ref} has no golden entry")
    task = TASK[ref]
    if mode not in API[task].model_card(ref).modes:
        pytest.skip(f"{ref} has no {mode}")
    gc.collect()
    gc.disable()
    try:
        base = torch.cuda.memory_allocated()
        m = API[task].load(ref, mode=mode, device="cuda")
        r = m(*pair()) if task in {"stereo", "flow"} else m(image())
        loaded = torch.cuda.memory_allocated() - base
        weights = _weight_bytes(m)
        del m, r
        torch.cuda.synchronize()
        without_gc = torch.cuda.memory_allocated() - base
        gc.collect()
        with_gc = torch.cuda.memory_allocated() - base
    finally:
        gc.enable()
    # process-wide workspaces (cuBLAS, cudnn_hacc) may stay; only memory that gc frees points to a reference cycle
    assert without_gc - with_gc < MIB, (
        f"{(without_gc - with_gc) / MIB:.0f} MiB of {loaded / MIB:.0f} MiB freed only by gc.collect()"
    )
    # and `del` did free the weights (a cache that keeps the model alive would pass the check above)
    assert weights > 0, f"{ref}: no CUDA weights found"
    assert loaded - without_gc > weights - MIB, (
        f"del freed {(loaded - without_gc) / MIB:.0f} MiB, the model held {weights / MIB:.0f} MiB of weights"
    )
