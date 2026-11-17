"""DepthFM goldens from the reference implementation.

Reference: upstream CompVis/depth-fm (`REF_COMMIT`) with `patches/depthfm-cpu.patch` (the Stable Diffusion 1.5 VAE from
a local file instead of a hub download, `torch.load(weights_only=False)` of the trusted released checkpoint, the
initial noise passed in, self-attention with the standard scale as upstream's xformers path computes it: the
requirements install xformers; the fallback without it rescales by resolution), CPU float32, `DepthFM.forward`
(what `predict_depth` calls, with the noise passed in)
with the `inference.py` defaults (`num_steps=2`, `ensemble_size=4`, `processing_res=-1`: the input size, sides rounded
to multiples of 64, PIL bilinear). The noise is `torch.randn` of a CPU generator seeded with `SEED` (the runners draw
the same). The `[0, 1]` depth is resized back to the input size with PIL bilinear on the float map (`inference.py`
does it on the 8-bit colour map). Needs diffusers, torchdiffeq, einops and omegaconf.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/depthfm.py DEPTHFM_CHECKOUT depthfm-v1.ckpt vae.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write
from PIL import Image

UPSTREAM = "https://github.com/CompVis/depth-fm"
REF_COMMIT = "da632c8118e41cc64ca191a13314e2db1a7722d9"
PATCH = "depthfm-cpu.patch"
STEPS, ENSEMBLE, SEED = 2, 4, 2024


def main() -> None:
    repo, ckpt, vae = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    tree = reference_tree(repo, REF_COMMIT, PATCH)
    sys.path[:0] = [str(tree), str(tree / "depthfm")]
    from depthfm import DepthFM  # noqa: PLC0415 - reference checkout

    net = DepthFM(ckpt, vae).eval()
    img = image()
    h, w = img.shape[:2]
    size = (round(w / 64) * 64, round(h / 64) * 64)  # inference.py resize_max_res with processing_res -1
    x = np.array(Image.fromarray(img).resize(size, resample=Image.Resampling.BILINEAR))
    x = torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1
    noise = torch.randn(ENSEMBLE, 4, size[1] // 8, size[0] // 8, generator=torch.Generator().manual_seed(SEED))
    with torch.inference_mode():
        d = net(x, num_steps=STEPS, ensemble_size=ENSEMBLE, noise=noise)[0, 0].numpy()  # predict_depth without noise
    depth = np.asarray(Image.fromarray(d, mode="F").resize((w, h), resample=Image.Resampling.BILINEAR))
    write(
        "depthfm-v1@1",
        {"depth": depth.astype(np.float32)},
        family="depthfm",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=(
            f"CPU float32, {STEPS} Euler steps, ensemble {ENSEMBLE}, network input {size[1]}x{size[0]}, CPU generator "
            f"seed {SEED}, torch {torch.__version__}"
        ),
        task="depth",
    )
    print(f"  depth {depth.shape}, median {np.median(depth):.4f}")


if __name__ == "__main__":
    main()
