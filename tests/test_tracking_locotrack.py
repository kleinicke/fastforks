"""LocoTrack: TF-SAME padding, the ALiBi bias, the batched CMDTop and the position encoding (no weights)."""

import math

import torch
import torch.nn.functional as F

from fastforks.tracking.models.locotrack.net.locotrack import CMDTop, alibi_bias, posenc, same_conv


def test_same_padding_is_tensorflow_same() -> None:
    torch.manual_seed(0)
    conv = same_conv(3, 4, 3, 2)
    x = torch.randn(1, 3, 8, 7)
    # TF SAME, stride 2, k 3: total pad 1 on height 8 (bottom), 2 on width 7 (one each side)
    want = F.conv2d(F.pad(x, (1, 1, 0, 1)), conv.weight, conv.bias, 2)
    torch.testing.assert_close(conv(x), want)
    assert conv(x).shape[-2:] == (math.ceil(8 / 2), math.ceil(7 / 2))


def test_alibi_bias_matches_upstream() -> None:
    heads, t = 4, 5
    bias = alibi_bias(heads, t, torch.device("cpu"))
    i, j = torch.meshgrid(torch.arange(t), torch.arange(t), indexing="ij")
    for h in range(heads // 2):
        slope = 1 / (24 ** (1 / (heads // 2))) ** (h + 1)
        fwd, bwd = bias[h], bias[heads // 2 + h]
        torch.testing.assert_close(fwd[j <= i], (slope * (j - i)[j <= i]).float())
        torch.testing.assert_close(bwd[j >= i], (slope * (i - j)[j >= i]).float())
        assert torch.isneginf(fwd[j > i]).all()
        assert torch.isneginf(bwd[j < i]).all()


def test_cmdtop_batches_both_directions() -> None:
    torch.manual_seed(0)
    m = CMDTop((64, 128), (5, 2), (4, 2)).eval()
    x = torch.randn(3, 7, 7, 7, 7)
    out = m(x)

    def run(y: torch.Tensor) -> torch.Tensor:
        for conv in m.conv:
            y = conv(y)
        return y.mean((2, 3))

    want = torch.cat([run(x.permute(0, 3, 4, 1, 2).reshape(3, 49, 7, 7)), run(x.reshape(3, 49, 7, 7))], -1)
    torch.testing.assert_close(out, want)


def test_posenc_upstream_order() -> None:
    x = torch.tensor([[0.1, -0.2, 0.3, 0.05]])
    out = posenc(x)
    assert out.shape == (1, 84)
    xb = (x[..., None, :] * 2.0 ** torch.arange(10)[:, None]).reshape(1, -1)
    torch.testing.assert_close(out, torch.cat([x, torch.sin(xb), torch.cos(xb)], -1), atol=1e-5, rtol=0)
