"""MatAnyone: the chunked memory readout (`_read`) equals upstream's one-pass `MemoryManager.read`."""

import pytest
import torch
import torch.nn.functional as F

from fastforks.matting.models.matanyone.model import CONFIG, READ_TOKENS, _config, _read, _upsample_block
from fastforks.matting.models.matanyone.net.inference.memory_manager import MemoryManager
from fastforks.matting.models.matanyone.net.inference.object_manager import ObjectManager
from fastforks.matting.models.matanyone.net.model.modules import MaskUpsampleBlock


class _Net:
    """The three network calls of `read`, as pass-throughs."""

    @staticmethod
    def pred_uncertainty(*_args: torch.Tensor) -> None:
        return None

    @staticmethod
    def pixel_fusion(pix_feat: torch.Tensor, pixel: torch.Tensor, sensory: torch.Tensor, mask: torch.Tensor) -> tuple:
        return pix_feat, pixel, sensory, mask

    @staticmethod
    def readout_query(pixel_readout: tuple, obj_memory: torch.Tensor) -> tuple:
        return pixel_readout[1] + pixel_readout[0][:, None] + obj_memory.sum() * 0, None


def test_chunked_read_matches_one_pass() -> None:
    torch.manual_seed(0)
    h, w, ck, cv = 7, (2 * READ_TOKENS + 11) // 7 + 1, 64, 256
    objects = ObjectManager()
    objects.add_new_objects([1])
    mem = MemoryManager(_config(CONFIG), objects)
    for _ in range(3):  # three memory frames
        key, shrink = torch.randn(1, ck, h, w), torch.rand(1, 1, h, w) + 1
        mem.add_memory(key, shrink, torch.randn(1, 1, cv, h, w), torch.randn(1, 1, 16, 257), [1], as_permanent="no")
    mem.initialize_sensory_if_needed(key, [1])
    pix_feat, qk, qe = torch.randn(1, cv, h, w), torch.randn(1, ck, h, w), torch.rand(1, ck, h, w)
    last_mask, last_msk_value = torch.rand(1, 1, h, w), torch.randn(1, 1, cv, h, w)
    args = (pix_feat, qk, qe, last_mask, _Net(), None)
    kw = {"last_msk_value": last_msk_value, "ti": 3, "last_pix_feat": pix_feat, "last_pred_mask": last_mask}
    ref = MemoryManager.read(mem, *args, **kw)
    out = _read(mem, *args, **kw)
    assert out.keys() == ref.keys() == {1}
    torch.testing.assert_close(out[1], ref[1], atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("objects", [1, 3])
def test_fused_group_upsample_preserves_frame_skips(objects: int) -> None:
    """Each object's upsampled features add the skip of its own batch frame."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(9)
    b, c, h, w = 2, 8, 7, 11
    block = MaskUpsampleBlock(c, c).eval().to(device="cuda", dtype=torch.float16)
    low = torch.randn(b * objects, c, h, w, device="cuda", dtype=torch.float16)
    low = low.contiguous(memory_format=torch.channels_last)
    skip = torch.randn(b, c, 2 * h, 2 * w, device="cuda", dtype=torch.float16) * 4
    skip = skip.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        up = F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False).view(b, objects, c, 2 * h, 2 * w)
        expected = block.out_conv(up + skip[:, None], consume_input=True)
        actual = _upsample_block(block, low.view(b, objects, c, h, w), skip)
    assert torch.equal(actual, expected)
