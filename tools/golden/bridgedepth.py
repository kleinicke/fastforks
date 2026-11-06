"""BridgeDepth goldens from the reference implementation.

Reference: upstream aeolusguan/BridgeDepth (`REF_COMMIT`) + `patches/bridgedepth-topk-ties.patch` (top-k ties to the
lower index, as upstream behaves on CUDA), CPU float32, `BridgeDepth.from_pretrained`'s build + strict load (an
unrestricted torch.load: trusted checkpoints only), upstream's forward (pads to a multiple of 16 itself, InputPadder
mode "nmrf"). Two changes for a CPU float32 reference: the Depth Anything V2 branch runs without its float16 autocast
(`infer(use_fp16=False)`; upstream enables it for float32 weights on any device), and the ConvNeXt backbone's ImageNet
initialisation builds torchvision's `convnext_tiny` without weights (no download; the checkpoint overwrites every
parameter). Upstream imports einops, yacs, huggingface_hub, torchvision and cv2; run this with a Python that has them
(any environment that has them).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/bridgedepth.py BRIDGEDEPTH_CHECKOUT WEIGHTS_DIR [sf rvc-pretrain]

`WEIGHTS_DIR` holds upstream's `bridge_<name>.pth`. The checkout only has to contain `REF_COMMIT`; the reference
runs from an export of that commit, in an empty working directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/aeolusguan/BridgeDepth"
REF_COMMIT = "9a231601e6981d3536fa6a78ebaccebc1edb99a4"
PATCH = "bridgedepth-topk-ties.patch"  # top-k ties to the lower index, as upstream behaves on CUDA
VARIANTS = {"sf": "bridge_sf.pth", "rvc-pretrain": "bridge_rvc_pretrain.pth"}  # ref name -> checkpoint file


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT, PATCH)
    # upstream's `bridgedepth/` has no __init__.py: drop this script's directory, whose bridgedepth.py would shadow it
    here = Path(__file__).resolve().parent
    sys.path[:] = [str(tree), *(p for p in sys.path if Path(p or ".").resolve() != here)]
    os.chdir(tempfile.mkdtemp(prefix="ff-bridgedepth-cwd-"))
    import torchvision.models  # noqa: PLC0415
    from bridgedepth.bridgedepth import BridgeDepth  # noqa: PLC0415  # reference checkout on sys.path
    from bridgedepth.monocular.depth_anything import DepthAnything  # noqa: PLC0415  # reference checkout on sys.path

    convnext_tiny = torchvision.models.convnext_tiny
    torchvision.models.convnext_tiny = lambda **_: convnext_tiny(weights=None)
    infer = DepthAnything.infer
    DepthAnything.infer = lambda self, image, use_fp16=True: infer(self, image, use_fp16=False)  # noqa: ARG005

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    for name in sys.argv[3:] or VARIANTS:
        # from_pretrained's weights_only load rejects the yacs CfgNode (a dict subclass) in torch >= 2.6: same steps
        ckpt = torch.load(weights / VARIANTS[name], map_location="cpu", weights_only=False)
        net = BridgeDepth(ckpt["model_config"])
        net.load_state_dict(ckpt["model"])
        net.eval()
        with torch.inference_mode():
            disp = net({"img1": left.clone(), "img2": right.clone()})["disp_pred"]
        disp = disp.reshape(left.shape[-2:]).numpy()
        write(
            f"bridgedepth-{name}@1",
            {"disparity": disp},
            family="bridgedepth",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"CPU float32 (mono branch without float16 autocast), torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
