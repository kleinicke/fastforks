"""DINOv3 ConvNeXt feature goldens from the reference implementation.

Reference: Hugging Face `transformers` (`REF_VERSION`) `DINOv3ConvNextModel`, CPU float32, loaded from the gated
`facebook/dinov3-convnext-<arch>-pretrain-lvd1689m` repositories (local copies). Input: the synthetic image resized as
`fastforks.features` does (shorter side 112 -> 128 x 160 at stride 32, bilinear, antialiased, clamped; ImageNet
normalisation). Outputs: `last_hidden_state`, `pooler_output` (no per-block hidden states in this family).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dinov3_convnext.py WEIGHTS_DIR [tiny small base large]

Needs transformers == `REF_VERSION` (not a fastforks dependency: run with an environment that has it).
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
SIZE, STRIDE = 112, 32  # tests/test_golden.py FEATURES_SIZE, the family's patch stride
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VARIANTS = ["tiny", "small", "base", "large"]


def _input() -> torch.Tensor:
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    s = SIZE / min(h, w)
    hw = (max(STRIDE, round(h * s / STRIDE) * STRIDE), max(STRIDE, round(w * s / STRIDE) * STRIDE))
    x = F.interpolate(x, size=hw, mode="bilinear", antialias=True).clamp(0, 1)
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def main() -> None:
    weights = Path(sys.argv[1])
    os.environ["HF_HUB_OFFLINE"] = "1"
    import transformers  # noqa: PLC0415 - reference environment
    from transformers import DINOv3ConvNextModel  # noqa: PLC0415

    assert transformers.__version__ == REF_VERSION, transformers.__version__  # noqa: S101
    x = _input()
    for arch in sys.argv[2:] or VARIANTS:
        path = weights / f"dinov3-convnext-{arch}-pretrain-lvd1689m"
        net = DINOv3ConvNextModel.from_pretrained(path, dtype=torch.float32).eval()
        with torch.inference_mode():
            out = net(pixel_values=x)
        maps = {"last_hidden_state": out.last_hidden_state, "pooler_output": out.pooler_output}
        write(
            f"dinov3-convnext-{arch}@1",
            {k: v[0].numpy() for k, v in maps.items()},
            family="dinov3_convnext",
            upstream=f"{UPSTREAM}@v{REF_VERSION}",
            patch=None,
            settings=f"CPU float32, DINOv3ConvNextModel, {x.shape[-2]}x{x.shape[-1]} input, torch {torch.__version__}",
            task="features",
        )
        print(f"  tokens {tuple(out.last_hidden_state.shape)}")


if __name__ == "__main__":
    main()
