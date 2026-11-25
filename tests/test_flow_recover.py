"""ReCoVEr's motion pack must preserve the border tile and flow strides."""

import pytest
import torch

from fastforks.kernels import recover_motion


@pytest.mark.gpu
@pytest.mark.parametrize(("height", "width"), [(5, 5), (9, 13)])
def test_recover_motion_pack(height: int, width: int) -> None:

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float16, device="cuda")

    motion = sample(2, 126, height, width)
    tile = sample(1, 126, 5, 5)
    flow = sample(2, 4, height, width * 2)[:, ::2, :, 1::2]
    rows = [0, 1, *([2] * (height - 4)), 3, 4]
    cols = [0, 1, *([2] * (width - 4)), 3, 4]
    expected = torch.cat(((motion + tile[:, :, rows][:, :, :, cols]).relu(), flow), dim=1)
    actual = recover_motion.pack(motion, tile, flow)
    assert actual.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
