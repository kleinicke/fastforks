"""ViTMatte: the card and the portable forward (channels-last backbone, augmented-SDPA attention) against the vendored
transformers network, on a small random network (CPU)."""

from __future__ import annotations

import torch

from fastforks import matting
from fastforks._core import registry
from fastforks._core.cards import Io
from fastforks.matting.models.vitmatte import model
from fastforks.matting.models.vitmatte.net.vitmatte import Config, VitMatteForImageMatting


def test_card() -> None:
    card, variant = registry.cards("matting")["vitmatte-s"]
    assert variant.outputs == ("alpha",)
    assert card.licence.code == card.licence.weights == "Apache-2.0"
    assert {"vitmatte-s", "vitmatte-b"} <= set(matting.list_models())
    assert matting.model_card("vitmatte-b").contract().inputs["trimap"] == "required"


def test_portable_matches_network() -> None:
    """Window + global blocks, residual blocks, interpolated position embedding and relative tables; an image whose
    sides are not multiples of 32 (padding) and a token grid that is not a multiple of the window."""
    torch.manual_seed(0)
    cfg = Config(
        hidden_size=64,
        num_attention_heads=1,
        num_hidden_layers=4,
        window_block_indices=(0, 2),
        residual_block_indices=(1, 3),
        window_size=4,
        image_size=64,
        pretrain_image_size=48,
        convstream_hidden_sizes=(8, 16, 24),
        fusion_hidden_sizes=(32, 16, 8, 8),
    )
    net = VitMatteForImageMatting(cfg).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.2)
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    run = object.__new__(model.Portable)
    run.net, run.dtype = net, torch.float32
    run.geometry = model._Geometry(net, torch.float32, torch.device("cpu"))
    image, trimap = torch.rand(2, 3, 70, 120), torch.rand(2, 1, 70, 120)
    x = model.pixel_values(image, trimap, Io(input="mean_std", mean=(0.5,) * 3, std=(0.5,) * 3))
    assert x.shape == (2, 4, 96, 128)
    with torch.no_grad():
        torch.testing.assert_close(run._net(x), net(x), atol=2e-5, rtol=1e-4)
