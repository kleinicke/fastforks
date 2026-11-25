"""ViTPose stage transfers preserve the block sequence for changing person counts."""

import copy
from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.pose.models.vitpose import model
from fastforks.shared.host_blocks import StreamedModuleList


def test_streaming_rejects_cpu_before_reading_weights() -> None:
    ctx = SimpleNamespace(device=torch.device("cpu"), params=SimpleNamespace(stream=True))
    with pytest.raises(OptionsError, match="stream=True requires CUDA"):
        model.Portable(ctx)  # ty: ignore[invalid-argument-type] - minimal context


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
def test_streamed_person_batches(mode: str) -> None:
    torch.manual_seed(85)
    arch = model.Arch(
        dim=128, depth=3, heads=4, part=0, experts=0, decoder="simple", keypoints=17, height=256, width=192
    )
    blocks = torch.nn.ModuleList(model.Block(arch) for _ in range(arch.depth)).half().eval()
    resident = copy.deepcopy(blocks).cuda()
    if mode == "kernels":
        for block in [*resident, *blocks]:
            override(block, model._staged_kernel_block)
    staged = StreamedModuleList(blocks, torch.device("cuda"))
    with torch.inference_mode(), fp16_accumulate():
        for people in (1, 3, 1):
            x = torch.randn(people, 192, arch.dim, device="cuda", dtype=torch.float16)
            expected = x
            for ref, run in zip(resident, staged, strict=True):
                expected, x = ref(expected), run(x)
            torch.testing.assert_close(x, expected, atol=0, rtol=0)
