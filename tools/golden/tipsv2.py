"""TIPSv2 feature goldens from the reference implementation.

Reference: google-deepmind/tips (`REF_COMMIT`), `pytorch/image_encoder.py` built as `run_image_encoder_inference.py`
does, CPU float32. Input: the synthetic image resized as `fastforks.features` does (shorter side 112 -> 112 x 168,
bilinear, antialiased; values in [0, 1], no normalisation). Outputs as the other feature families: `last_hidden_state`
(the normed first cls token, second cls token and patch tokens of the forward), `pooler_output` (first cls token),
`hidden_states.0` (embeddings) and `.1` (after block 1).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/tipsv2.py TIPS_CHECKOUT WEIGHTS_DIR [b14 l14]

`WEIGHTS_DIR` holds `tipsv2-<size>/model.safetensors` (the Hugging Face `google/tipsv2-<size>` files).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from _common import image, reference_tree, write
from safetensors.torch import load_file

UPSTREAM = "https://github.com/google-deepmind/tips"
REF_COMMIT = "bf10a73765b048edfbfcdf1f7e6ca8292542baae"
SIZE = 112  # tests/test_golden.py FEATURES_SIZE
LAYERS = (0, 1)  # tests/test_golden.py FEATURES_LAYERS
VARIANTS = {"b14": "vit_base", "l14": "vit_large"}


def _input() -> torch.Tensor:
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    s = SIZE / min(h, w)
    hw = (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))
    return F.interpolate(x, size=hw, mode="bilinear", antialias=True)


def build(repo: Path, weights: Path, name: str) -> Any:  # noqa: ANN401 - the reference module
    """The reference image encoder of `tipsv2-<name>` (CPU float32, eval) from a tips checkout and `WEIGHTS_DIR`."""
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT) / "pytorch"))
    import image_encoder  # noqa: PLC0415  # reference checkout

    net = getattr(image_encoder, VARIANTS[name])(
        img_size=448,
        patch_size=14,
        ffn_layer="mlp",
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    sd = load_file(weights / f"tipsv2-{name}" / "model.safetensors")
    sd = {k.removeprefix("vision_encoder."): v for k, v in sd.items() if k.startswith("vision_encoder.")}
    net.load_state_dict(sd, strict=True, assign=True)
    return net.eval()


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    x = _input()
    h, w = x.shape[-2:]
    for name in sys.argv[3:] or VARIANTS:
        net = build(repo, weights, name)
        hs = []
        with torch.inference_mode():
            t = net.prepare_tokens_with_masks(x)  # the forward, recording the block outputs
            hs.append(t)
            for blk in net.blocks:
                t = blk(t)
                hs.append(t)
            cls, cls2, patches = net(x)
            last = torch.cat([cls, cls2, patches], 1)
            assert torch.equal(net.norm(hs[-1]), last)  # noqa: S101 - the recorded loop is the forward
        maps = {"last_hidden_state": last, "pooler_output": cls[:, 0]}
        maps |= {f"hidden_states.{i}": hs[i] for i in LAYERS}
        write(
            f"tipsv2-{name}@1",
            {k: v[0].numpy() for k, v in maps.items()},
            family="tipsv2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, inference script model, {h}x{w} input, torch {torch.__version__}",
            task="features",
        )
        print(f"  tokens {tuple(last.shape)}, mean |cls| {cls.abs().mean():.4f}")


if __name__ == "__main__":
    main()
