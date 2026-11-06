"""MVSAnywhere golden from the reference implementation.

Reference: upstream nianticlabs/mvsanywhere (`REF_COMMIT`), CPU float32, `MVSAnywhereInference.forward` built as the
upstream inference scripts build it (`configs/models/mvsanywhere_model.yaml`, `load_model_inference`), on the
`tests/synth.py` scene: view 0 as the reference, views 1..3 as sources, `num_refinement_steps=1`, no depth range (the
upstream heuristic). Inputs are prepared as the fastforks runner does with its defaults: images resized to long side
640 (bilinear, antialiased, multiples of 16) and ImageNet-normalised, intrinsics shifted by +0.5 px (MVSAnywhere's
corner-origin pixels) and scaled; depth and sky probability are resized back to the input size (bilinear).

Run with a Python that has the upstream dependencies (loguru, antialiased_cnns, timm, torchvision; the DINOv2 hub
checkout and the antialiased ResNet-18 weights in the torch hub cache, which upstream's constructors load and the
checkpoint then overwrites):

    FASTFORKS_GOLDEN_PATH=... python tools/golden/mvsanywhere.py MVSANYWHERE_CHECKOUT mvsanywhere_hero.ckpt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write

from synth import scene

UPSTREAM = "https://github.com/nianticlabs/mvsanywhere"
REF_COMMIT = "5bd49bba0992aa56060e1aa3bb40b2150b04726f"
LONG_SIDE = 640
MULTIPLE = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main() -> None:
    repo, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    here = Path(__file__).resolve().parent  # this file would shadow upstream's `mvsanywhere` (a namespace package)
    sys.path[:] = [str(tree / "src"), str(tree), *(p for p in sys.path if Path(p or ".").resolve() != here)]
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # the Lightning checkpoint pickles its options
    from hubconf import MVSAnywhereInference  # noqa: PLC0415  # reference checkout
    from mvsanywhere.options import OptionsHandler  # noqa: PLC0415
    from mvsanywhere.utils.model_utils import (  # noqa: PLC0415
        get_model_class,
        load_model_inference,
    )

    argv, sys.argv = sys.argv, sys.argv[:1]  # the options parser reads sys.argv
    h = OptionsHandler()
    h.parse_and_merge_options(config_filepaths=str(tree / "configs/models/mvsanywhere_model.yaml"))
    sys.argv = argv
    h.options.load_weights_from_checkpoint = str(ckpt)
    m = MVSAnywhereInference.__new__(MVSAnywhereInference)
    torch.nn.Module.__init__(m)
    m.model = load_model_inference(h.options, get_model_class(h.options)).eval()
    seen: dict[str, dict] = {}  # upstream keeps the depth range it used in cur_data (updated in place per refinement)
    forward = m.model.forward

    def record(*args: object, **kwargs: dict) -> object:
        seen.setdefault("cur", kwargs["cur_data"])
        return forward(*args, **kwargs)

    m.model.forward = record

    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    sc = LONG_SIDE / max(h0, w0)
    hn, wn = (max(MULTIPLE, round(v * sc / MULTIPLE) * MULTIPLE) for v in (h0, w0))
    x = (F.interpolate(img, (hn, wn), mode="bilinear", align_corners=False, antialias=True) - MEAN) / STD
    k = torch.from_numpy(s["intrinsics"]).clone()
    k[:, :2, 2] += 0.5
    k[:, 0] *= wn / w0
    k[:, 1] *= hn / h0
    k4 = torch.eye(4).repeat(len(k), 1, 1)
    k4[:, :3, :3] = k
    c2w = torch.from_numpy(s["cam_to_world"])
    with torch.inference_mode():
        o = m(
            cur_image=x[:1],
            src_image=x[1:][None],
            cur_pose=c2w[:1],
            src_pose=c2w[1:][None],
            cur_intrinsics=k4[:1],
            src_intrinsics=k4[1:][None],
            num_refinement_steps=1,
        )
    depth = F.interpolate(o["depth_pred_s0_b1hw"], (h0, w0), mode="bilinear", align_corners=False)[:, 0]
    sky = F.interpolate(o["sky_pred_s0_b1hw"], (h0, w0), mode="bilinear", align_corners=False)[:, 0]
    maps = {
        "depth": depth.numpy(),
        "mask": (sky < 0.5).numpy(),  # noqa: PLR2004 - upstream depth_masked threshold
        "sky": sky.numpy(),
        "depth_range": torch.stack(
            [seen["cur"]["min_depth"].reshape(-1), seen["cur"]["max_depth"].reshape(-1)], 1
        ).numpy(),
    }
    write(
        "mvsanywhere@1",
        maps,
        family="mvsanywhere",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, forward(num_refinement_steps=1), synth scene (ref 0 + 3 sources) at {hn}x{wn}, "
        f"torch {torch.__version__}",
        task="multiview",
    )
    gt = s["depth"][0]
    absrel = np.mean(np.abs(maps["depth"][0] - gt) / gt)
    print(f"  depth median {np.median(maps['depth']):.4f} (GT {np.median(gt):.4f}), AbsRel {absrel:.4f}")
    print(f"  sky max {maps['sky'].max():.4f}, depth_range {maps['depth_range'].tolist()}")


if __name__ == "__main__":
    main()
