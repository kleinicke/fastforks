"""Falconsai NSFW ViT: the Hugging Face `ViTForImageClassification` checkpoint loaded into timm's ViT-B/16.

The two networks are the same (pre-norm blocks, class token, final LayerNorm, linear head on the class token); the
checkpoint's keys are renamed and its separate query / key / value projections concatenated into timm's `qkv`.
LayerNorm `eps` is the checkpoint config's 1e-12 (timm: 1e-6).
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import load_strict
from fastforks._core.labels import label_set
from fastforks.shared import libnet

if TYPE_CHECKING:
    from fastforks._core.runner import Context

_RENAME = (
    (r"^vit\.embeddings\.cls_token$", "cls_token"),
    (r"^vit\.embeddings\.position_embeddings$", "pos_embed"),
    (r"^vit\.embeddings\.patch_embeddings\.projection\.", "patch_embed.proj."),
    (r"^vit\.encoder\.layer\.(\d+)\.layernorm_before\.", r"blocks.\1.norm1."),
    (r"^vit\.encoder\.layer\.(\d+)\.layernorm_after\.", r"blocks.\1.norm2."),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"blocks.\1.attn.proj."),
    (r"^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"blocks.\1.mlp.fc1."),
    (r"^vit\.encoder\.layer\.(\d+)\.output\.dense\.", r"blocks.\1.mlp.fc2."),
    (r"^vit\.layernorm\.", "norm."),
    (r"^classifier\.", "head."),
)
_QKV = re.compile(r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)$")


def timm_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The `ViTForImageClassification` state dict with timm `VisionTransformer` keys."""
    out, qkv = {}, {}
    for k, v in sd.items():
        if m := _QKV.match(k):
            qkv[m[1], m[3], m[2]] = v
            continue
        for pat, rep in _RENAME:
            k = re.sub(pat, rep, k)  # noqa: PLW2901 - each pattern renames one key prefix
        out[k] = v
    for i, kind in {(i, kind) for i, kind, _ in qkv}:
        out[f"blocks.{i}.attn.qkv.{kind}"] = torch.cat([qkv.pop((i, kind, n)) for n in ("query", "key", "value")])
    return out


def network(ctx: Context) -> torch.nn.Module:
    """The timm ViT with the checkpoint's weights (CPU, eval)."""
    classes = len(label_set(ctx.variant.output_info["label_set"]))
    norm = functools.partial(torch.nn.LayerNorm, eps=1e-12)
    net = libnet.timm().create_model(ctx.params.arch, pretrained=False, num_classes=classes, norm_layer=norm)
    load_strict(net, timm_state_dict(ctx.state_dict(strip="")), f"{ctx.variant.ref} checkpoint")
    return net.eval()


class Portable(libnet.Portable):
    """Plain PyTorch through `fastforks._optimize`."""

    network = staticmethod(network)


class Kernels(libnet.Kernels):
    """Our kernels through `fastforks._optimize`."""

    network = staticmethod(network)


RUNNERS = {"portable": Portable, "kernels": Kernels}
