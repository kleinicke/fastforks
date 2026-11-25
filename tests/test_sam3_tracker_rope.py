"""SAM 3 video spatial memory keys rotate per frame; trailing pointer keys do not rotate."""

import pytest
import torch

from fastforks.kernels import sam3_tracker
from fastforks.promptable_segmentation.models.sam3.fast import tracker_attention
from fastforks.promptable_segmentation.models.sam3.net.tracker import MEM_DIM, RoPEAttention, rotate_pairwise

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(("frames", "tail"), [(1, 0), (1, 4), (7, 64)])
def test_tracker_rope_matches_eager(frames, tail) -> None:
    torch.manual_seed(29)
    period, width = 17, 256
    spatial = frames * period
    x = torch.randn((2, spatial + tail, width), device="cuda", dtype=torch.float16)
    phase = torch.randn((1, period, width), device=x.device)
    cos, sin = phase.cos(), phase.sin()
    spatial_keys = x[:, :spatial].float()
    expected = torch.cat(
        [
            (spatial_keys * cos.repeat(1, frames, 1) + rotate_pairwise(spatial_keys) * sin.repeat(1, frames, 1)).half(),
            x[:, spatial:],
        ],
        dim=1,
    )
    got = sam3_tracker.rotate_(x, (cos, sin), spatial)
    assert got.data_ptr() == x.data_ptr()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [1, 3])
def test_tracker_attention_matches_upstream(frames) -> None:
    torch.manual_seed(30)
    tokens, tail = 23, 4 * frames
    m = RoPEAttention(MEM_DIM, rope_k_repeat=True).cuda().half().eval()
    q = torch.randn(2, tokens, 256, device="cuda", dtype=torch.float16)
    k, v = torch.randn(2, 2, tokens * frames + tail, MEM_DIM, device=q.device, dtype=q.dtype).unbind()
    angle = torch.randn(1, tokens, q.shape[-1], device=q.device)
    rope = angle.cos(), angle.sin()
    with torch.inference_mode():
        want = m(q, k, v, rope, tail)
        got = tracker_attention(m, q, k, v, rope, tail)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
