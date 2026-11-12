"""Goldens of the feature families built by timm by name (`params.arch`: SigLIP 2, CLIP, Perception Encoder, ConvNeXt).

Reference: timm itself at the version in the card's `[upstream]`, loading its own pretrained weights the way a user
would (`timm.create_model(name, pretrained=True, dynamic_img_size=True)`, the Hugging Face cache; set
`HF_HUB_OFFLINE=1` to use it as is), CPU float32. Input: the synthetic image resized as `fastforks.features` does
(shorter side 112, each side the nearest multiple of the patch, antialiased bicubic, clamped; the card's mean / std).
Outputs: `last_hidden_state` = `forward_features`, `pooler_output` = `forward_head(pre_logits=True)`,
`hidden_states.0` = the `norm_pre` output (embeddings), `.1` = block 1's output (forward hooks on timm's modules).
ConvNeXt (no hidden states): `last_hidden_state` = the last map of `timm.create_model(name, features_only=True)` as
tokens, `pooler_output` = `forward_head(pre_logits=True)` of the full network on that map.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/timm_features.py [FAMILY | REF ...]
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from _common import image, write

from fastforks._core import registry

if TYPE_CHECKING:
    from fastforks._core.cards import Io

TASK = "features"
SIZE = 112  # tests/test_golden.py FEATURES_SIZE
LAYERS = (0, 1)  # tests/test_golden.py FEATURES_LAYERS


def _input(io: Io) -> torch.Tensor:
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    s, m = SIZE / min(h, w), io.pad_multiple
    hw = (max(m, round(h * s / m) * m), max(m, round(w * s / m) * m))
    x = F.interpolate(x, size=hw, mode=io.interpolation, antialias=True).clamp(0, 1)
    return (x - torch.tensor(io.mean).view(1, 3, 1, 1)) / torch.tensor(io.std).view(1, 3, 1, 1)


def _keep(hs: dict[int, torch.Tensor], i: int) -> Any:  # noqa: ANN401 - a forward hook
    """Forward hook storing the module output as `hs[i]`."""
    return lambda _m, _i, o: hs.setdefault(i, o)


def main() -> None:
    import timm  # noqa: PLC0415 - optional dependency

    want = set(sys.argv[1:])
    for card, variant in registry.cards(TASK).values():
        params = {**card.params, **variant.params}
        if "arch" not in params or (want and not want & {card.family, variant.id}):  # built by timm by name
            continue
        arch = params["arch"]
        x = _input(variant.io)
        hs: dict[int, torch.Tensor] = {}
        if "hidden_states" in card.outputs:
            net: Any = timm.create_model(arch, pretrained=True, dynamic_img_size=True).eval()
            hooks = [m.register_forward_hook(_keep(hs, i)) for i, m in ((0, net.norm_pre), (1, net.blocks[0]))]
            with torch.inference_mode():
                last = net.forward_features(x)
                pooled = net.forward_head(last, pre_logits=True)
            for h in hooks:
                h.remove()
        else:  # ConvNeXt
            with torch.inference_mode():
                fmap = timm.create_model(arch, pretrained=True, features_only=True).eval()(x)[-1]
                full: Any = timm.create_model(arch, pretrained=True).eval()
                pooled = full.forward_head(fmap, pre_logits=True)
            last = fmap.flatten(2).transpose(1, 2)
        h, w = x.shape[-2:]
        maps = {"last_hidden_state": last, "pooler_output": pooled} | {
            f"hidden_states.{i}": hs[i] for i in LAYERS if i in hs
        }
        write(
            variant.ref,
            {k: v[0].numpy() for k, v in maps.items()},
            family=card.family,
            upstream=f"{card.upstream}@{card.upstream_commit}",
            patch=None,
            settings=f"CPU float32, timm {timm.__version__}, {h}x{w} input, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  tokens {tuple(last.shape)}, pooled {tuple(pooled.shape)}, mean |pooled| {pooled.abs().mean():.4f}")


if __name__ == "__main__":
    main()
