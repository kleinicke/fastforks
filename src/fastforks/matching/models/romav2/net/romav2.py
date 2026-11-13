# fastforks: the inference forward of upstream `romav2/romav2.py` (`RoMaV2.forward`). Weight download, image loading,
# settings, sampling and keypoint matching are left out (the runner resizes the images, the eval tool samples); the
# bf16 autocast regions take their dtype from the modules' `amp` attributes (amp.py), and `bidirectional` is an argument.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3 import DinoVisionTransformer
from .features import VGG19BN
from .geometry import bhwc_interpolate
from .matcher import Matcher
from .normalizers import imagenet
from .refiner import Refiners


def _interpolate_warp_and_confidence(
    *,
    warp: torch.Tensor,
    confidence: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    zero_out_precision: bool,
):
    warp = bhwc_interpolate(warp, size=(H // patch_size, W // patch_size), mode="bilinear", align_corners=False)
    if zero_out_precision:
        # delta at 4 is absolute, and if we
        # for the second pass we therefore can't use first pred.
        # overlap is fine since it's relative to matcher pred.
        confidence = torch.cat((confidence[..., :1], torch.zeros_like(confidence[..., 1:])), dim=-1)
    confidence = bhwc_interpolate(
        confidence, size=(H // patch_size, W // patch_size), mode="bilinear", align_corners=False
    )
    return warp, confidence


class RoMaV2(nn.Module):
    anchor_width = 512
    anchor_height = 512

    def __init__(self) -> None:
        super().__init__()
        self.f = DinoVisionTransformer(layers=(11, 17))
        self.matcher = Matcher(Matcher.Cfg())
        self.refiners = Refiners(Refiners.Cfg())
        self.refiner_features = VGG19BN()

    def describe(self, img: torch.Tensor) -> list[torch.Tensor]:
        return self.f(imagenet(img))

    def forward(
        self,
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
        bidirectional: bool = False,
        f_A: list[torch.Tensor] | None = None,  # fastforks: `describe(img_A_lr)` the caller kept from an earlier pair
        f_B: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Images in [0, 1], `[B, 3, H, W]`, A and B of the same size (lr: a multiple of 16, hr: of 4)."""
        f_A = self.describe(img_A_lr) if f_A is None else f_A
        f_B = self.describe(img_B_lr) if f_B is None else f_B
        matcher_output = self.matcher(f_A, f_B, img_A=img_A_lr, img_B=img_B_lr, bidirectional=bidirectional)
        warp_AB, confidence_AB = matcher_output["warp_AB"], matcher_output["confidence_AB"]
        warp_BA, confidence_BA = matcher_output["warp_BA"], matcher_output["confidence_BA"]
        del f_A, f_B, matcher_output  # fastforks: the descriptors and the attention maps are not needed any more
        stages = [(img_A_lr, img_B_lr)] + ([(img_A_hr, img_B_hr)] if img_A_hr is not None else [])
        for stage, (img_A, img_B) in enumerate(stages):
            B, C, H, W = img_A.shape
            scale_factor = torch.tensor((W / self.anchor_width, H / self.anchor_height), device=img_A.device)
            refiner_features_A = self.refiner_features(img_A)
            refiner_features_B = self.refiner_features(img_B)
            for patch_size_str, refiner in self.refiners.items():
                patch_size = int(patch_size_str)
                zero_out_precision = patch_size == 4 and stage == 1
                warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                    warp=warp_AB, confidence=confidence_AB, H=H, W=W, patch_size=patch_size,
                    zero_out_precision=zero_out_precision,
                )
                # fastforks: each stride map is consumed once per stage; popping frees it before the next refiner
                f_patch_A = refiner_features_A.pop(patch_size)
                f_patch_B = refiner_features_B.pop(patch_size)
                out = refiner(
                    f_A=f_patch_A, f_B=f_patch_B, prev_warp=warp_AB, prev_confidence=confidence_AB,
                    scale_factor=scale_factor,
                )
                warp_AB, confidence_AB = out["warp"], out["confidence"]
                if bidirectional:
                    warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                        warp=warp_BA, confidence=confidence_BA, H=H, W=W, patch_size=patch_size,
                        zero_out_precision=zero_out_precision,
                    )
                    out = refiner(
                        f_A=f_patch_B, f_B=f_patch_A, prev_warp=warp_BA, prev_confidence=confidence_BA,
                        scale_factor=scale_factor,
                    )
                    warp_BA, confidence_BA = out["warp"], out["confidence"]
        preds = {"warp_AB": warp_AB, "confidence_AB": confidence_AB}
        if bidirectional:
            preds |= {"warp_BA": warp_BA, "confidence_BA": confidence_BA}
        return preds


def resize(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """The input resize of upstream `RoMaV2.match` (bicubic, antialiased)."""
    return F.interpolate(img, size=size, mode="bicubic", align_corners=False, antialias=True)
