"""Pi3 golden from the reference implementation.

Reference: upstream yyfz/Pi3 (`REF_COMMIT`), CPU float32, `Pi3.forward` on the 4 views of the `tests/synth.py`
scene, with the runner's pre- and post-processing: images resized to upstream's `load_images_as_tensor` size
(PIXEL_LIMIT 255000, multiples of 14; bicubic antialiased, clamped, instead of PIL Lanczos), maps resized back to the
input size (bilinear), poses and points re-expressed in the frame of view 0, depth = z of `local_points`,
confidence = sigmoid(`conf`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/pi3.py PI3_CHECKOUT model.safetensors
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write

from synth import scene

UPSTREAM = "https://github.com/yyfz/Pi3"
REF_COMMIT = "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"
PIXEL_LIMIT = 255000


def network_size(h: int, w: int, limit: int = PIXEL_LIMIT) -> tuple[int, int]:
    """Upstream `load_images_as_tensor` target size (height, width)."""
    s = math.sqrt(limit / (w * h))
    wt, ht = w * s, h * s
    k, m = round(wt / 14), round(ht / 14)
    while (k * 14) * (m * 14) > limit:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, m) * 14, max(1, k) * 14


def network(repo: Path, weights: Path) -> Any:  # noqa: ANN401 - upstream module
    """Upstream `Pi3` at `REF_COMMIT` with the released weights (CPU, eval)."""
    tree = reference_tree(repo, REF_COMMIT)
    here = Path(__file__).resolve().parent  # this script is `pi3.py`: it must not shadow the upstream package
    sys.path[:] = [str(tree), *(p for p in sys.path if Path(p or ".").resolve() != here)]
    sys.modules.pop("pi3", None)
    from pi3.models.pi3 import Pi3  # noqa: PLC0415  # reference checkout
    from safetensors.torch import load_file  # noqa: PLC0415

    net = Pi3()
    net.load_state_dict(load_file(str(weights)), strict=True)
    return net.eval()


def predict(net: Any, img: torch.Tensor, *, bf16: bool = False) -> dict[str, torch.Tensor]:  # noqa: ANN401
    """`[V, 3, H, W]` in `[0, 1]` -> the runner's maps (the authors' bf16 autocast if `bf16`)."""
    h0, w0 = img.shape[-2:]
    hn, wn = network_size(h0, w0)
    x = F.interpolate(img, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    with torch.no_grad(), torch.autocast(img.device.type, torch.bfloat16, enabled=bf16):
        p = net(x[None])

    def up(t: torch.Tensor) -> torch.Tensor:  # [V, h, w, C] -> [V, C, h0, w0]
        return F.interpolate(t.float().permute(0, 3, 1, 2), (h0, w0), mode="bilinear", align_corners=False)

    c2w = p["camera_poses"][0].float()
    to0 = torch.linalg.inv(c2w[:1])
    pts = torch.einsum("ij,shwj->shwi", to0[0, :3, :3], p["points"][0].float()) + to0[0, :3, 3]
    return {
        "depth": up(p["local_points"][0][..., 2:])[:, 0],
        "confidence": torch.sigmoid(up(p["conf"][0])[:, 0]),
        "world_points": up(pts).permute(0, 2, 3, 1),
        "cam_to_world": to0 @ c2w,
    }


def main() -> None:
    net = network(Path(sys.argv[1]), Path(sys.argv[2]))
    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    hn, wn = network_size(h0, w0)
    maps = {n: v.numpy() for n, v in predict(net, img).items()}
    write(
        "pi3@1",
        maps,
        family="pi3",
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
    print(f"  cam centres {maps['cam_to_world'][:, :3, 3].round(4).tolist()}")


if __name__ == "__main__":
    main()
