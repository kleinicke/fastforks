"""VGGT-Omega-1B-512 golden from the reference implementation.

Reference: upstream facebookresearch/vggt-omega (`REF_COMMIT`), CPU float32, `VGGTOmega.forward` (camera and dense
heads; its bf16 autocast is CUDA-only, so CPU runs float32) on the 4 views of the `tests/synth.py` scene, with the
runner's pre- and post-processing: images resized to the "balanced 512" size (`_balanced_target_shape`, bicubic
antialiased, clamped), maps resized back to the input size (bilinear), intrinsics from `encoding_to_camera` rescaled
to the input pixels (principal point moved by -0.5: pixel centres at integer coordinates),
`cam_to_world = inv([R | t])`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/vggt_omega.py VGGT_OMEGA_CHECKOUT vggt_omega_1b_512.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write

from synth import scene

UPSTREAM = "https://github.com/facebookresearch/vggt-omega"
REF_COMMIT = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"
RESOLUTION = 512


def main() -> None:
    repo, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    here = Path(__file__).resolve().parent  # this script is `vggt_omega.py`: it must not shadow the upstream package
    sys.path[:] = [str(tree), *(p for p in sys.path if Path(p or ".").resolve() != here)]
    sys.modules.pop("vggt_omega", None)
    from vggt_omega.models import VGGTOmega  # noqa: PLC0415  # reference checkout
    from vggt_omega.utils.load_fn import _balanced_target_shape  # noqa: PLC0415
    from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: PLC0415

    if not str(sys.modules["vggt_omega"].__file__).startswith(str(tree)):
        msg = f"vggt_omega imported from {sys.modules['vggt_omega'].__file__}, not the reference tree"
        raise RuntimeError(msg)
    net = VGGTOmega()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    net.eval()

    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    hn, wn = _balanced_target_shape(h0 / w0, RESOLUTION, 16)
    x = F.interpolate(img, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    with torch.no_grad():
        p = net(x[None])

    def up(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)

    ext, k = encoding_to_camera(p["pose_enc"], (hn, wn))
    ext, k = ext[0], k[0].clone()
    k[:, 0] *= w0 / wn
    k[:, 1] *= h0 / hn
    k[:, :2, 2] -= 0.5
    w2c = torch.eye(4).repeat(ext.shape[0], 1, 1)
    w2c[:, :3] = ext
    maps = {
        "depth": up(p["depth"][0].permute(0, 3, 1, 2))[:, 0],
        "confidence": up(p["depth_conf"][0][:, None])[:, 0],
        "intrinsics": k,
        "cam_to_world": torch.linalg.inv(w2c),
    }
    maps = {n: v.numpy() for n, v in maps.items()}
    write(
        "vggt-omega-1b-512@1",
        maps,
        family="vggt_omega",
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
