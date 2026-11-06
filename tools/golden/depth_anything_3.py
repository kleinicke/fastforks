"""Depth Anything 3 goldens from the reference implementation.

Reference: upstream ByteDance-Seed/Depth-Anything-3 (`REF_COMMIT`) with `patches/depth_anything_3-full-quantile.patch`
(upstream estimates its quantiles from a random sample of 100k pixels on larger maps; the patch uses all pixels, as
fastforks does). CPU float32: upstream `InputProcessor` (process_res 504, upper_bound_resize, OpenCV), the network
built from upstream's config, one view, no poses. The maps are then upsampled to the input size as the runner does
(bilinear, `align_corners=False`; mask = sky < 0.3); intrinsics at the input size (Metric: a 300 px focal length at the
network input, upstream's canonical focal). Run this with a Python that has OpenCV, omegaconf, addict and safetensors.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/depth_anything_3.py DA3_CHECKOUT WEIGHTS_DIR [ref ...]

`WEIGHTS_DIR` holds `<HF repo name>/model.safetensors` (e.g. `DA3MONO-LARGE/model.safetensors`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/ByteDance-Seed/Depth-Anything-3"
REF_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
PATCH = "depth_anything_3-full-quantile.patch"
SKY = 0.3  # upstream compute_sky_mask threshold
VARIANTS = {  # ref: (upstream config, weights directory)
    "da3mono-large": ("da3mono-large", "DA3MONO-LARGE"),
    "da3metric-large": ("da3metric-large", "DA3METRIC-LARGE"),
    "da3-large-1.1": ("da3-large", "DA3-LARGE-1.1"),
    "da3-giant-1.1": ("da3-giant", "DA3-GIANT-1.1"),
    "da3nested-giant-large-1.1": ("da3nested-giant-large", "DA3NESTED-GIANT-LARGE-1.1"),
}


def up(m: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    return F.interpolate(m[None, None].float(), size, mode="bilinear", align_corners=False)[0, 0].numpy()


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT, PATCH)
    sys.path.insert(0, str(tree / "src"))
    # this script (same name as upstream's package, on sys.path) would shadow it
    sys.path.remove(str(Path(__file__).resolve().parent))
    from depth_anything_3.cfg import create_object, load_config  # noqa: PLC0415
    from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    rgb = image()
    h, w = rgb.shape[:2]
    x, _, _ = InputProcessor()([rgb], process_res=504, process_res_method="upper_bound_resize", sequential=True)
    x = x.reshape(1, 1, *x.shape[-3:])  # one batch of one view
    hh, ww = x.shape[-2:]
    for ref in sys.argv[3:] or VARIANTS:
        cfg, sub = VARIANTS[ref]
        net = create_object(load_config(str(tree / "src/depth_anything_3/configs" / f"{cfg}.yaml")))
        sd = {k.removeprefix("model."): v for k, v in load_file(weights / sub / "model.safetensors").items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "_aux." not in k]  # unused ray branch: some LayerNorms are not saved
        if missing or unexpected:
            raise RuntimeError((missing[:5], unexpected[:5]))
        with torch.inference_mode():
            out = net.eval()(x, None, None, [], infer_gs=False, use_ray_pose=False)
        maps = {"depth": up(out["depth"][0, 0], (h, w))}
        if "depth_conf" in out:
            maps["confidence"] = up(out["depth_conf"][0, 0], (h, w))
        if "intrinsics" in out:  # from the predicted field of view, at the network size: rescale to the input
            k = out["intrinsics"][0, 0].clone()
            k[0] *= w / ww
            k[1] *= h / hh
            k[:2, 2] -= 0.5  # OpenCV pixel centres (upstream: corner origin)
            maps["intrinsics"] = k.numpy()
        elif ref == "da3metric-large":
            maps["intrinsics"] = np.array([[300 * w / ww, 0, (w - 1) / 2], [0, 300 * h / hh, (h - 1) / 2], [0, 0, 1]])
        if ref.startswith("da3nested"):  # the mask comes from the metric branch's sky head (deterministic: rerun)
            with torch.inference_mode():
                out["sky"] = net.da3_metric(x)["sky"]
        if "sky" in out:
            sky = up(out["sky"][0, 0], (h, w))
            maps |= {"sky": sky, "mask": (sky < SKY).astype(np.float32)}
        write(
            f"{ref}@1",
            maps,
            family="depth_anything_3",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"CPU float32, InputProcessor(process_res=504, upper_bound_resize), torch {torch.__version__}",
            task="depth",
        )
        print(f"  {sorted(maps)} median depth {np.median(maps['depth']):.4f} net {hh}x{ww}")
        del net, sd, out


if __name__ == "__main__":
    main()
