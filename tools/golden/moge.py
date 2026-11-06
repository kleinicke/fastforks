"""MoGe goldens from the reference implementation.

Reference: upstream microsoft/MoGe (`REF_COMMIT`) with `patches/moge-converged-focal-shift.patch` (upstream's SciPy
Levenberg-Marquardt stops at a relative cost change of 1e-3; the patch solves the focal / shift fit to convergence, as
fastforks does). `MoGeModel.infer(resolution_level=9, apply_mask=False, use_fp16=False)` in float32: MoGe-2 on the
CPU; MoGe-3 on the GPU (its sparse refiner needs FlexGEMM's CUDA kernels) with TF32 off everywhere. Intrinsics are
converted to pixels; the mask is upstream's binary mask; normals where the model has a normal head. Run this with a
Python that has utils3d (MoGe's fork), SciPy, huggingface_hub and, for MoGe-3, flex_gemm.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/moge.py MOGE_CHECKOUT WEIGHTS_DIR [ref ...]

`WEIGHTS_DIR` holds `<ref>/model.pt` (e.g. `moge-2-vitl-normal/model.pt`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/microsoft/MoGe"
REF_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
PATCH = "moge-converged-focal-shift.patch"
VARIANTS = {"moge-2-vits-normal": 2, "moge-2-vitl-normal": 2, "moge-3-vitl": 3}  # ref: version


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT, PATCH)
    sys.path.insert(0, str(tree))
    # this script (same name as upstream's package, on sys.path) would shadow it
    sys.path.remove(str(Path(__file__).resolve().parent))
    os.environ.setdefault("FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE", "0")
    from moge.model.v2 import MoGeModel as MoGeV2  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rgb = image()
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    for ref in sys.argv[3:] or VARIANTS:
        if VARIANTS[ref] == 3:  # noqa: PLR2004 - MoGe-3
            import flex_gemm  # noqa: PLC0415
            from moge.model.v3 import MoGeModel  # noqa: PLC0415

            flex_gemm.config.SPCONV_ALLOW_TF32 = False
            dev, where = "cuda", f"GPU ({torch.cuda.get_device_name()}) float32, TF32 off"
        else:
            MoGeModel, dev, where = MoGeV2, "cpu", "CPU float32"  # noqa: N806 - the class
        net = MoGeModel.from_pretrained(weights / ref / "model.pt").to(dev).eval()
        with torch.inference_mode():
            out = net.infer(x.to(dev), resolution_level=9, apply_mask=False, use_fp16=False)
        k = out["intrinsics"].double().cpu()
        k[0] *= w
        k[1] *= h
        k[:2, 2] -= 0.5  # OpenCV pixel centres (upstream: corner origin)
        maps = {
            "depth": out["depth"].cpu().numpy(),
            "points": out["points"].cpu().numpy(),
            "intrinsics": k.numpy(),
            "mask": out["mask"].float().cpu().numpy(),
        }
        if "normal" in out:  # the -normal variants: the golden turns fastforks' `normals` option on
            maps["normals"] = out["normal"].cpu().numpy()
        write(
            f"{ref}@1",
            maps,
            family="moge",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"{where}, infer(resolution_level=9, apply_mask=False, use_fp16=False), torch {torch.__version__}",
            task="depth",
        )
        print(f"  median depth {np.median(maps['depth']):.4f} fx {k[0, 0]:.2f} mask {maps['mask'].mean():.3f}")
        del net, out


if __name__ == "__main__":
    main()
