"""AANet / AANet+ goldens from the reference implementation.

Reference: upstream haofeixu/aanet (`REF_COMMIT`), CPU float32, the SceneFlow inference settings of
`scripts/aanet{,+}_inference.sh`, `predict.py` preprocessing: ImageNet normalization, then zeros on top and on the
right up to a multiple of 48 (AANet) / 96 (AANet+).

Upstream's deformable convolution is a torch cpp_extension (mmdetection v1 CUDA ops: GPU only, does not build on
current torch). The reference runs with a shim that registers `nets.deform_conv` using
`torchvision.ops.deform_conv2d` (the same modulated deformable convolution, same offset layout and mask semantics;
runs on CPU). Run this with a Python that has torchvision installed.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/aanet.py AANET_CHECKOUT WEIGHTS_DIR [aanet aanet-plus]

`WEIGHTS_DIR` holds `aanet_sceneflow.pth` and `aanet+_sceneflow.pth`. The checkout only has to contain `REF_COMMIT`;
the reference runs from an export of that commit.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/haofeixu/aanet"
REF_COMMIT = "8d0d747e69bbdba43e2578b60ad7df63ba7fdac2"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VARIANTS = {
    "aanet": (
        "aanet_sceneflow.pth",
        48,
        {"feature_type": "aanet", "feature_pyramid_network": True, "refinement_type": "stereodrnet"},
    ),
    "aanet-plus": (
        "aanet+_sceneflow.pth",
        96,
        {"feature_type": "ganet", "feature_pyramid": True, "refinement_type": "hourglass"},
    ),
}


def _shim_deform_conv() -> None:
    """Register `nets.deform_conv` on top of torchvision (upstream's version needs the CUDA extension)."""
    from torchvision.ops import deform_conv2d  # noqa: PLC0415 - reference-only dependency

    def pair_(v: int | tuple[int, int]) -> tuple[int, int]:
        return (v, v) if isinstance(v, int) else v

    class DeformConv(torch.nn.Module):
        """Upstream's `DeformConv` / `ModulatedDeformConv` (same parameter names) on torchvision's op."""

        def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], **kw: float
        ) -> None:
            super().__init__()
            if kw.get("groups", 1) != 1:
                msg = "groups > 1"
                raise NotImplementedError(msg)
            self.stride = pair_(kw.get("stride", 1))  # kw values are int / pair
            self.padding = pair_(kw.get("padding", 0))  # ditto
            self.dilation = pair_(kw.get("dilation", 1))  # ditto
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, *pair_(kernel_size)))
            self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if kw.get("bias", False) else None

        def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)

    mod = types.ModuleType("nets.deform_conv")
    mod.DeformConv = mod.ModulatedDeformConv = DeformConv
    sys.modules["nets.deform_conv"] = mod


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    _shim_deform_conv()
    import nets  # noqa: PLC0415  # reference checkout on sys.path

    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in pair())
    h, w = left.shape[2:]
    for name in sys.argv[3:] or VARIANTS:
        file, mult, kw = VARIANTS[name]
        top, rpad = (-h) % mult, (-w) % mult
        lp, rp = (F.pad(x, (0, rpad, top, 0)) for x in (left, right))
        net = nets.AANet(
            192, num_downsample=2, no_feature_mdconv=False, feature_similarity="correlation",
            aggregation_type="adaptive", num_scales=3, num_fusions=6, num_stage_blocks=1, num_deform_blocks=3,
            no_intermediate_supervision=True, mdconv_dilation=2, deformable_groups=2, **kw,
        )  # fmt: skip
        sd = torch.load(weights / file, map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=False)  # upstream no_strict
        with torch.inference_mode():
            disp = net.eval()(lp, rp)[-1]
        if disp.shape[-1] < lp.shape[-1]:
            disp = F.interpolate(disp[:, None], lp.shape[-2:], mode="bilinear")[:, 0] * (lp.shape[-1] / disp.shape[-1])
        disp = disp[0, top:, :w].numpy()
        write(
            f"{name}-sceneflow@1",
            {"disparity": disp},
            family="aanet",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, deformable conv via torchvision.ops.deform_conv2d shim, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
