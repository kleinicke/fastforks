"""PatchmatchNet golden from the reference implementation.

Reference: upstream FangjinhuaWang/PatchmatchNet (`REF_COMMIT`), CPU float32, `PatchmatchNet.forward` in eval mode
with the `eval.py` default settings and the released checkpoint, on the `tests/synth.py` scene: view 0 as the
reference, views 1..3 as sources, the scene's depth range. Extrinsics are `inv(cam_to_world)` in float32 (as the
runner computes them); `torch.manual_seed(0)` right before the call fixes the 48 random initial hypotheses (the
runner draws them from a CPU generator seeded with 0).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/patchmatchnet.py PATCHMATCHNET_CHECKOUT params_000007.ckpt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write

from synth import scene

UPSTREAM = "https://github.com/FangjinhuaWang/PatchmatchNet"
REF_COMMIT = "8dc6cb40bdb7053e856598b378425c76a9dcf5e0"
ARGS = {
    "patchmatch_interval_scale": [0.005, 0.0125, 0.025],
    "propagation_range": [6, 4, 2],
    "patchmatch_iteration": [1, 2, 2],
    "patchmatch_num_sample": [8, 8, 16],
    "propagate_neighbors": [0, 8, 16],
    "evaluate_neighbors": [9, 9, 9],
}


def main() -> None:
    repo, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    from models.net import PatchmatchNet  # noqa: PLC0415  # reference checkout

    net = PatchmatchNet(**ARGS)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    net.eval()

    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    k = torch.from_numpy(s["intrinsics"]).float()
    e = torch.linalg.inv(torch.from_numpy(s["cam_to_world"]).float())
    dmin, dmax = (torch.tensor([float(v)]) for v in s["depth_range"])
    torch.manual_seed(0)
    with torch.no_grad():
        depth, conf, _ = net([v[None] for v in img], k[None], e[None], dmin, dmax)
    maps = {"depth": depth[:, 0].numpy(), "confidence": conf.numpy()}
    h, w = img.shape[-2:]
    write(
        "patchmatchnet@1",
        maps,
        family="patchmatchnet",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, eval.py settings, torch.manual_seed(0), synth scene (ref 0 + 3 sources) at {h}x{w}, "
        f"torch {torch.__version__}",
        task="multiview",
    )
    gt = s["depth"][0]
    absrel = np.mean(np.abs(maps["depth"][0] - gt) / gt)
    print(f"  depth median {np.median(maps['depth']):.4f} (GT {np.median(gt):.4f}), AbsRel {absrel:.4f}")
    print(f"  confidence mean {maps['confidence'].mean():.4f}, depth_range {s['depth_range'].tolist()}")


if __name__ == "__main__":
    main()
