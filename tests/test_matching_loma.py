"""LoMa: the card, DaD's sub-pixel keypoint sampling, the match filter, the checkpoint split and the net's shapes."""

from __future__ import annotations

import pytest
import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.loma.model import PREFIXES, split_checkpoint
from fastforks.matching.models.loma.net.dad import sample_keypoints
from fastforks.matching.models.loma.net.loma import LoMaNet, filter_matches
from fastforks.shared.dedode import Descriptor, describe_keypoints
from fastforks.shared.dedode import sparse as _sparse


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "loma"}
    assert sorted(cards) == ["loma-b", "loma-b128", "loma-b64", "loma-g", "loma-l", "loma-r"]
    card, variant = cards["loma-b"]
    assert dict(card.options) == {"max_keypoints": 2048, "long_side": 1024, "size": 784}  # upstream Cfg, DaD
    assert dict(card.params) == {
        "threshold": 0.1,
        "input_dim": 256,
        "descriptor": "dedode_g",
        "normalize_descriptor": False,
    }
    assert dict(variant.params) == {"dim": 256, "heads": 4, "layers": 9}
    assert set(card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
    assert [w.name for w in variant.weights] == ["loma_B.pt"]
    assert "loma-b" in matching.list_models()


def test_sample_keypoints() -> None:
    """The best pixels after 3 x 3 NMS, each pulled towards the higher logits in its patch."""
    logits = torch.zeros(1, 1, 8, 8)
    logits[0, 0, 2, 3] = 4.0
    logits[0, 0, 2, 4] = 3.0  # suppressed by the (2, 3) peak, which it pulls to the right
    logits[0, 0, 5, 5] = 4.0
    logits[0, 0, 5, 6] = 4.0  # equal neighbour: NMS keeps both (ties), the offsets pull each towards the other
    kpts = sample_keypoints(logits, 3, nms_size=3, temp=0.5)
    px = (kpts + 1) / 2 * 8 - 0.5  # (x, y) pixel centres
    rows = px[:, 1].round()
    assert rows.tolist().count(2.0) == 1
    assert rows.tolist().count(5.0) == 2
    peak = px[rows == 2.0][0]
    assert torch.allclose(peak[1], torch.tensor(2.0), atol=1e-3)
    assert 3.0 < peak[0] < 4.0
    others = px[rows == 5.0]
    assert torch.allclose(others[:, 1], torch.full((2,), 5.0), atol=1e-3)
    assert (others[:, 0] > 5.0).all()
    assert (others[:, 0] < 6.0).all()
    assert torch.allclose(others[:, 0].sum(), torch.tensor(11.0), atol=1e-3)  # symmetric pull


def test_filter_matches() -> None:
    """Mutual maxima above the threshold; one-sided maxima and weak mutual ones are dropped."""
    s = torch.zeros(1, 3, 3)
    s[0, 0, 1] = 0.9  # mutual
    s[0, 1, 0] = 0.05  # mutual but weak
    s[0, 2, 1] = 0.5  # row 2 -> col 1, but col 1's maximum is row 0
    m0, scores = filter_matches(s, 0.1)
    assert m0.tolist() == [1, -1, -1]
    assert torch.allclose(scores, torch.tensor([0.9, 0.05, 0.0]))


def test_split_checkpoint_and_shapes() -> None:
    """A random checkpoint in upstream's layout loads strictly into the parts; the matcher gives `[1, M, N]`."""
    net = LoMaNet(32, 32, 4, 2)
    sd = {f"transformers.{k}" if k.startswith(("0.", "1.")) else k: v for k, v in net.state_dict().items()}
    sd["_detector.encoder.layers.0.weight"] = torch.zeros(1)
    sd["_descriptor.encoder.vgg.layers.0.weight"] = torch.zeros(1)
    sd["_descriptor.encoder.frozen_dinov2.dinov2_vitl14.cls_token"] = torch.zeros(1)
    sd["_descriptor.decoder.layers.8.out_conv.weight"] = torch.zeros(1)
    parts = split_checkpoint(sd)
    assert set(parts) == {*PREFIXES.values(), "matcher"}
    assert list(parts["dinov2"]) == ["cls_token"]
    assert list(parts["descriptor.decoder"]) == ["layers.8.out_conv.weight"]
    assert set(parts["matcher"]) == set(net.state_dict())
    net.load_state_dict(parts["matcher"], strict=True)
    g = torch.Generator().manual_seed(0)
    k0, k1 = torch.rand(1, 5, 2, generator=g) * 2 - 1, torch.rand(1, 7, 2, generator=g) * 2 - 1
    scores = net(k0, k1, torch.randn(1, 5, 32, generator=g), torch.randn(1, 7, 32, generator=g))
    assert scores.shape == (1, 5, 7)
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_subpixel_patches_include_zero_padding() -> None:
    """The sparse gather matches the original full unfold at corners and edges of a rectangular image."""
    h, w = 7, 11
    logits = torch.randn(1, 1, h, w, generator=torch.Generator().manual_seed(13))
    probs = logits.flatten().softmax(0).view(1, h, w)
    peaks = probs * (probs == torch.nn.functional.max_pool2d(probs, 3, stride=1, padding=1))
    indices = peaks.flatten().topk(h * w).indices  # exercise every position, including all four corners
    y, x = torch.meshgrid(
        torch.linspace(-1 + 1 / h, 1 - 1 / h, h), torch.linspace(-1 + 1 / w, 1 - 1 / w, w), indexing="ij"
    )
    centers = torch.stack((x, y), dim=-1).view(-1, 2)[indices]
    patches = torch.nn.functional.unfold(logits, 3, padding=1)[0, :, indices]
    axis = torch.linspace(-1 + 1 / 3, 1 - 1 / 3, 3)
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((dx, dy), dim=-1).view(-1, 2) * 3 / torch.tensor([w, h])
    expected = centers + (patches / 0.5).softmax(0).mT @ offsets
    torch.testing.assert_close(sample_keypoints(logits, h * w, 3, 0.5), expected, rtol=0, atol=0)


def test_sparse_descriptor_matches_dense_at_edges() -> None:
    """Odd feature sizes cover the non-integer coarse resize and zero-padded boundary samples."""
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(37)
        descriptor = Descriptor().eval().half()
        image = torch.rand(1, 3, 33, 43).half()
        points = torch.tensor([[-1.0, 1.0], [1.0, 1.0], [-1.02, -0.99], [0.3, -0.8], [0.0, 0.0]])
        expected = describe_keypoints(descriptor(image), points)
        actual = _sparse.sample(descriptor, image, points)
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=1e-3)


@pytest.mark.gpu
def test_cuda_descriptor_blend_strided() -> None:
    from fastforks.kernels.loma_descriptor import blend  # noqa: PLC0415

    prior = torch.randn(1, 9, 7, 12, dtype=torch.float16, device="cuda").contiguous(memory_format=torch.channels_last)
    delta = torch.randn(4, 31, 9, dtype=torch.float16, device="cuda").permute(2, 0, 1)
    ix = torch.randint(23, (31, 4), device="cuda").T
    iy = torch.randint(13, (31, 4), device="cuda").T
    weights = torch.rand(31, 4, device="cuda").T
    result = blend(prior, delta, ix, iy, weights, (13, 23))
    expected = _sparse._blend(prior, delta, ix, iy, weights, (13, 23))
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
