"""DINOv2 feature goldens from the reference implementation.

Reference: facebookresearch/dinov2 (`REF_COMMIT`), the torch.hub builders (`dinov2/hub/backbones.py`), CPU float32,
xFormers off. Input: the synthetic image resized as `fastforks.features` does (shorter side 112 -> 112 x 168, bicubic,
antialiased, clamped; ImageNet normalisation). Outputs as Hugging Face `Dinov2Model`: `last_hidden_state` (normed
tokens of `forward_features`), `pooler_output` (normed cls), `hidden_states.0` (embeddings) and `.1` (after block 1).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dinov2.py DINOV2_CHECKOUT WEIGHTS_DIR [vits14 vits14_reg4 ...]

`WEIGHTS_DIR` holds `dinov2_<arch>14[_reg4]_pretrain.pth`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/facebookresearch/dinov2"
REF_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SIZE = 112  # tests/test_golden.py FEATURES_SIZE
LAYERS = (0, 1)  # tests/test_golden.py FEATURES_LAYERS
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VARIANTS = [f"vit{a}14{r}" for a in "sblg" for r in ("", "_reg4")]


def _input() -> torch.Tensor:
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    s = SIZE / min(h, w)
    hw = (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))
    x = F.interpolate(x, size=hw, mode="bicubic", antialias=True).clamp(0, 1)
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from dinov2.hub import backbones  # noqa: PLC0415  # reference checkout

    x = _input()
    for name in sys.argv[3:] or VARIANTS:
        net = getattr(backbones, f"dinov2_{name.replace('_reg4', '_reg')}")(pretrained=False)
        sd = torch.load(weights / f"dinov2_{name}_pretrain.pth", map_location="cpu", weights_only=True, mmap=True)
        net.load_state_dict(sd, strict=True, assign=True)
        hs = []
        with torch.inference_mode():
            t = net.eval().prepare_tokens_with_masks(x)  # forward_features, recording the block outputs
            hs.append(t)
            for blk in net.blocks:
                t = blk(t)
                hs.append(t)
            f = net.forward_features(x)
            last = torch.cat([f["x_norm_clstoken"][:, None], f["x_norm_regtokens"], f["x_norm_patchtokens"]], 1)
            assert torch.equal(net.norm(hs[-1]), last)  # noqa: S101 - the recorded loop is forward_features
        maps = {"last_hidden_state": last, "pooler_output": f["x_norm_clstoken"]}
        maps |= {f"hidden_states.{i}": hs[i] for i in LAYERS}
        ref = "dinov2-" + name.replace("_reg4", "-reg")
        write(
            f"{ref}@1",
            {k: v[0].numpy() for k, v in maps.items()},
            family="dinov2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, hub builder, {x.shape[-2]}x{x.shape[-1]} input, torch {torch.__version__}",
            task="features",
        )
        print(f"  tokens {tuple(last.shape)}, mean |cls| {f['x_norm_clstoken'].abs().mean():.4f}")


if __name__ == "__main__":
    main()
