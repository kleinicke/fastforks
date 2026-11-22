"""SAM relative-position attention (`fastforks.shared.sam_attention`): the two forms agree, and the no-flash rule."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from fastforks.shared import sam_attention, sdpa


@pytest.fixture(autouse=True)
def _fresh() -> None:
    sdpa._resolve.cache_clear()


def _case(b: int, nh: int, d: int, h: int, w: int) -> tuple[torch.Tensor, sam_attention.Rel]:
    g = torch.Generator().manual_seed(0)
    qkv = torch.randn(b, h * w, 3 * nh * d, generator=g, dtype=torch.float64)
    rh = torch.randn(h, h, d, generator=g, dtype=torch.float64) * 0.3
    rw = torch.randn(w, w, d, generator=g, dtype=torch.float64) * 0.3
    eye_h, eye_w = torch.eye(h, dtype=torch.float64), torch.eye(w, dtype=torch.float64)
    onehot = torch.cat([eye_h.repeat_interleave(w, 0), eye_w.repeat(h, 1)], 1)
    return qkv, (rh, rw, onehot)


def _upstream(qkv: torch.Tensor, nh: int, rel: sam_attention.Rel, hw: tuple[int, int]) -> torch.Tensor:
    """Upstream SAM `Attention.forward` with `add_decomposed_rel_pos`, as `[B, heads, N, d]`."""
    rh, rw, _ = rel
    b, n, c3 = qkv.shape
    d, (h, w) = c3 // 3 // nh, hw
    q, k, v = qkv.view(b, n, 3, nh, d).permute(2, 0, 3, 1, 4)
    attn = (q * d**-0.5) @ k.transpose(-2, -1)
    r_q = q.reshape(b, nh, h, w, d)
    rel_h = torch.einsum("bnhwc,hkc->bnhwk", r_q, rh)
    rel_w = torch.einsum("bnhwc,wkc->bnhwk", r_q, rw)
    attn = (attn.view(b, nh, h, w, h, w) + rel_h[..., :, None] + rel_w[..., None, :]).view(b, nh, n, n)
    return attn.softmax(-1) @ v


@pytest.mark.parametrize(
    ("mask_bytes", "group_bytes"),
    [(1 << 30, 1 << 30), (1 << 30, 3 * 64 * 64 * 8), (1 << 30, 1), (5 * 64 * 8, 1)],  # all / 3 / 1 head, 5 rows
)
def test_biased_matches_upstream(monkeypatch: pytest.MonkeyPatch, mask_bytes: int, group_bytes: int) -> None:
    monkeypatch.setattr(sam_attention, "MASK_BYTES", mask_bytes)
    monkeypatch.setattr(sam_attention, "HEAD_GROUP_BYTES", group_bytes)
    qkv, rel = _case(2, 4, 16, 8, 8)
    want = _upstream(qkv, 4, rel, (8, 8))
    torch.testing.assert_close(sam_attention.biased(qkv, 4, rel, (8, 8)), want)
    qa, ka, va, d = sam_attention.augmented(qkv, 4, rel, (8, 8))
    torch.testing.assert_close(F.scaled_dot_product_attention(qa, ka, va, scale=1.0)[..., :d], want)


@pytest.mark.parametrize(
    ("value", "flash", "d", "hw", "want"),
    [
        ("auto", True, 64, (64, 64), False),  # flash: the augmented form, PyTorch's order
        ("auto", False, 64, (64, 64), True),  # Windows wheel, SAM global attention (192 > 128)
        ("auto", False, 64, (14, 14), False),  # Windows wheel, SAM windowed attention (96: cuDNN takes it)
        ("torch", False, 64, (64, 64), False),  # PyTorch's own choice: fastforks changes nothing
        ("cudnn,efficient", True, 64, (64, 64), True),
        ("flash,cudnn", True, 64, (64, 64), False),
    ],
)
def test_needs_bias(  # noqa: PLR0913, PLR0917 - one argument per parameter
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    flash: bool,  # noqa: FBT001
    d: int,
    hw: tuple[int, int],
    want: bool,  # noqa: FBT001
) -> None:
    monkeypatch.setenv(sdpa.ENV, value)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: flash)
    assert sam_attention.needs_bias(d, hw) is want


def test_cpu_call_keeps_augmented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sdpa.ENV, "cudnn")
    assert sdpa.backends() == (SDPBackend.CUDNN_ATTENTION,)
    qkv, _ = _case(1, 1, 64, 64, 64)
    with torch.no_grad():
        assert not sam_attention.use_biased(qkv, 64, (64, 64))
