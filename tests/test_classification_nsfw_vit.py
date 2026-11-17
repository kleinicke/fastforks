"""Falconsai NSFW ViT: the Hugging Face -> timm key mapping and the stretched resize of its processor."""

from __future__ import annotations

import pytest
import torch

from fastforks._core import cards
from fastforks._core.errors import CardError
from fastforks._core.preprocess import resize_crop
from fastforks.classification.models.nsfw_vit.model import timm_state_dict


def test_state_dict_mapping():
    sd = {
        "vit.embeddings.cls_token": torch.zeros(1),
        "vit.embeddings.patch_embeddings.projection.weight": torch.zeros(1),
        "vit.encoder.layer.11.layernorm_after.bias": torch.zeros(1),
        "vit.encoder.layer.11.intermediate.dense.weight": torch.zeros(1),
        "classifier.bias": torch.zeros(1),
    } | {
        f"vit.encoder.layer.3.attention.attention.{n}.weight": torch.full((2, 4), float(i))
        for i, n in enumerate(("query", "key", "value"))
    }
    out = timm_state_dict(sd)
    assert set(out) == {
        "cls_token",
        "patch_embed.proj.weight",
        "blocks.11.norm2.bias",
        "blocks.11.mlp.fc1.weight",
        "head.bias",
        "blocks.3.attn.qkv.weight",
    }
    assert out["blocks.3.attn.qkv.weight"][:, 0].tolist() == [0, 0, 1, 1, 2, 2]  # query, key, value rows


@pytest.mark.parametrize(("h", "w"), [(5, 9), (300, 200)])
def test_stretch_resizes_both_sides(h, w):
    out = resize_crop(torch.rand(1, 3, h, w), cards.Io(resize=7, stretch=True))
    assert out.shape[-2:] == (7, 7)


def test_stretch_must_be_bool():
    with pytest.raises(CardError, match="stretch"):
        cards._io({"stretch": 1}, "card", "classification")
