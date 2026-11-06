"""VGGT-1B golden from the reference implementation.

Reference: upstream facebookresearch/vggt (`REF_COMMIT`), CPU float32, `VGGT.forward` (no query points: camera,
depth and point heads) on the 4 views of the `tests/synth.py` scene, with the runner's pre- and post-processing:
images resized to 392x518 (long side 518, multiples of 14, bicubic antialiased, clamped), maps resized back to the
input size (bilinear), intrinsics from `pose_encoding_to_extri_intri` rescaled to the input pixels (principal point
moved by -0.5: pixel centres at integer coordinates), `cam_to_world = inv([R | t])`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/vggt.py VGGT_CHECKOUT model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write

from synth import scene

UPSTREAM = "https://github.com/facebookresearch/vggt"
REF_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
LONG_SIDE = 518


def main() -> None:
    repo, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    here = Path(__file__).resolve().parent  # this script is `vggt.py`: it must not shadow the upstream package
    sys.path[:] = [str(tree), *(p for p in sys.path if Path(p or ".").resolve() != here)]
    sys.modules.pop("vggt", None)
    from vggt.models.vggt import VGGT  # noqa: PLC0415  # reference checkout
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: PLC0415

    net = VGGT()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    net.eval()

    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    f = LONG_SIDE / max(h0, w0)
    hn, wn = (max(14, round(v * f / 14) * 14) for v in (h0, w0))
    x = F.interpolate(img, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    with torch.no_grad():
        p = net(x[None])

    def up(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)

    ext, k = pose_encoding_to_extri_intri(p["pose_enc"], (hn, wn))
    ext, k = ext[0], k[0].clone()
    k[:, 0] *= w0 / wn
    k[:, 1] *= h0 / hn
    k[:, :2, 2] -= 0.5
    w2c = torch.eye(4).repeat(ext.shape[0], 1, 1)
    w2c[:, :3] = ext
    maps = {
        "depth": up(p["depth"][0].permute(0, 3, 1, 2))[:, 0],
        "confidence": up(p["depth_conf"][0][:, None])[:, 0],
        "world_points": up(p["world_points"][0].permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
        "points_confidence": up(p["world_points_conf"][0][:, None])[:, 0],
        "intrinsics": k,
        "cam_to_world": torch.linalg.inv(w2c),
    }
    maps = {n: v.numpy() for n, v in maps.items()}
    write(
        "vggt-1b@1",
        maps,
        family="vggt",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {hn}x{wn}, torch {torch.__version__}",
        task="multiview",
    )
    gt = s["depth"]
    d = maps["depth"]
    sc = np.median(gt) / np.median(d)
    print(f"  depth median {np.median(d):.4f} (GT {np.median(gt):.4f}), AbsRel after median scaling "
          f"{np.mean(np.abs(d * sc - gt) / gt):.4f}")  # fmt: skip
    print(f"  fx {maps['intrinsics'][:, 0, 0].tolist()} (GT {s['intrinsics'][0, 0, 0]:.2f})")
    print(f"  cam centres {maps['cam_to_world'][:, :3, 3].round(4).tolist()}")


if __name__ == "__main__":
    main()
