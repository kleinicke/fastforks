"""OMNI-DC goldens from the reference implementation.

Reference: upstream princeton-vl/OMNI-DC (`REF_COMMIT`) with `patches/omni-dc-cpu.patch` (CPU devices, no mmcv /
mmseg imports, no separate Depth Anything V2 / ResNet download: the released checkpoint holds both), CPU float32,
`OGNIDC.forward` with the `demo.sh` arguments, prepared as `demo.py` does: the ImageNet-normalised 98 x 140 image and
the sparse depth zero-padded below and on the right to 112 x 144, the output cropped back. The depth is the
12 x 18 synthetic map with holes of `tests/synth.py` (`sparse_prompt`, NaN holes as 0) nearest-resized to the image
size (as `model.prepare` does).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/omni_dc.py OMNI_DC_CHECKOUT model.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write
from safetensors.torch import load_file

from synth import sparse_prompt

UPSTREAM = "https://github.com/princeton-vl/OMNI-DC"
REF_COMMIT = "a5ecd35d087ed5e5fcc2be93989d1db3fde194fd"
PATCH = "omni-dc-cpu.patch"
HEIGHT, WIDTH = 98, 140
PADDED = (112, 144)  # demo.py: multiples of 4 * 2 ** (num_resolution - 1)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ARGS = (  # demo.sh
    "--max_depth 300 --data_normalize_median 1 --num_resolution 3 --multi_resolution_learnable_gradients_weights "
    "uniform --load_dav2 1 --GRU_iters 1 --optim_layer_input_clamp 1.0 --depth_activation_format exp "
    "--whiten_sparse_depths 1 --gru_internal_whiten_method median --backbone_mode rgbd --pred_confidence_input 1"
)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    src = reference_tree(repo, REF_COMMIT, PATCH) / "src"
    sys.path[:0] = [str(src), str(src / "model")]
    sys.argv = ["omni_dc", *ARGS.split()]
    from config import args  # noqa: PLC0415 - reference checkout
    from model.ognidc import OGNIDC  # noqa: PLC0415

    net = OGNIDC(args)
    net.load_state_dict(load_file(weights), strict=True)
    img = image(HEIGHT, WIDTH)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    d = torch.from_numpy(np.nan_to_num(sparse_prompt(), nan=0.0))[None, None].float()
    d = F.interpolate(d, (HEIGHT, WIDTH), mode="nearest")
    pads = (0, PADDED[1] - WIDTH, 0, PADDED[0] - HEIGHT)
    sample = {
        "rgb": F.pad(x, pads),
        "dep": F.pad(d, pads),
        "K": torch.eye(3)[None],
        "pattern": torch.zeros(1),
    }
    with torch.inference_mode():
        depth = net.eval()(sample)["pred"][0, 0, :HEIGHT, :WIDTH].numpy()
    write(
        "omni-dc@1",
        {"depth": depth, "input": img.astype(np.float32)},
        family="omni_dc",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, OGNIDC.forward (demo.sh arguments) on {HEIGHT}x{WIDTH}, torch {torch.__version__}",
        task="depth",
    )
    print(f"  median depth {np.median(depth):.4f} m")


if __name__ == "__main__":
    main()
