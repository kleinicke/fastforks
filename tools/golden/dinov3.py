"""DINOv3 ViT feature goldens from the reference implementation.

Reference: Hugging Face `transformers` (`REF_VERSION`) `DINOv3ViTModel`, CPU float32, eager attention, loaded from
the gated `facebook/dinov3-<arch>-pretrain-lvd1689m` repositories (local copies). Input: the synthetic image resized
as `fastforks.features` does (shorter side 112 -> 112 x 176, bilinear, antialiased, clamped; ImageNet normalisation).
Outputs: `last_hidden_state`, `pooler_output`, `hidden_states.0` (embeddings) and `.1` (after block 1).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dinov3.py WEIGHTS_DIR [vits16 vitb16 ...]

`WEIGHTS_DIR` holds `dinov3-<arch>-pretrain-lvd1689m/` (config.json, model.safetensors). Needs transformers ==
`REF_VERSION` (not a fastforks dependency: run with an environment that has it).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from _common import image, write

UPSTREAM = "https://github.com/huggingface/transformers"
REF_VERSION = "5.19.0"
SIZE = 112  # tests/test_golden.py FEATURES_SIZE
LAYERS = (0, 1)  # tests/test_golden.py FEATURES_LAYERS
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VARIANTS = ["vits16", "vitb16", "vitl16", "vith16plus"]


def _input() -> torch.Tensor:
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    s = SIZE / min(h, w)
    hw = (max(16, round(h * s / 16) * 16), max(16, round(w * s / 16) * 16))
    x = F.interpolate(x, size=hw, mode="bilinear", antialias=True).clamp(0, 1)
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def main() -> None:
    weights = Path(sys.argv[1])
    os.environ["HF_HUB_OFFLINE"] = "1"
    import transformers  # noqa: PLC0415 - reference environment
    from transformers import DINOv3ViTModel  # noqa: PLC0415

    assert transformers.__version__ == REF_VERSION, transformers.__version__  # noqa: S101
    x = _input()
    for arch in sys.argv[2:] or VARIANTS:
        net = DINOv3ViTModel.from_pretrained(
            weights / f"dinov3-{arch}-pretrain-lvd1689m", attn_implementation="eager", dtype=torch.float32
        ).eval()
        with torch.inference_mode():
            out = net(pixel_values=x, output_hidden_states=True)
            hs = out.hidden_states
            assert len(hs) == net.config.num_hidden_layers + 1  # noqa: S101 - embeddings + every block
            assert torch.equal(net.norm(hs[-1]), out.last_hidden_state)  # noqa: S101 - hidden_states are un-normed
        maps = {"last_hidden_state": out.last_hidden_state, "pooler_output": out.pooler_output}
        maps |= {f"hidden_states.{i}": hs[i] for i in LAYERS}
        write(
            f"dinov3-{arch}@1",
            {k: v[0].numpy() for k, v in maps.items()},
            family="dinov3",
            upstream=f"{UPSTREAM}@v{REF_VERSION}",
            patch=None,
            settings=f"CPU float32, DINOv3ViTModel eager, {x.shape[-2]}x{x.shape[-1]} input, torch {torch.__version__}",
            task="features",
        )
        print(f"  tokens {tuple(out.last_hidden_state.shape)}, max |hs1| {hs[1].abs().max():.1f}")


if __name__ == "__main__":
    main()
