"""Sapiens segmentation goldens from the authors' Sapiens-Lite TorchScript checkpoints.

Reference: the TorchScript archive run as released (`torch.jit.load`, CPU float32) under the test-time protocol of the
Sapiens-Lite demo at `REF_COMMIT` (`lite/demo/vis_seg.py`, `adhoc_image_dataset.py`): resize to 1024 x 768 bilinear
(OpenCV `INTER_LINEAR` geometry: half-pixel centres, no antialiasing; here on the float image, so without OpenCV's
uint8 rounding), RGB normalised with mean (123.5, 116.5, 103.5) / std (58.5, 57.0, 57.5) on the 0-255 scale, the
model's logits resized bilinearly to the image size. Stored: those logits.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sapiens.py WEIGHTS_DIRS [0.3b 1b]   # WEIGHTS_DIRS: ':'-separated
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, write

UPSTREAM = "https://github.com/facebookresearch/sapiens"
REF_COMMIT = "2cb07227a740cf09896309ea3a3b8fa44429865c"
VARIANTS = {
    "0.3b": ("sapiens-seg-0.3b-goliath", "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"),
    "1b": ("sapiens-seg-1b-goliath", "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"),
}
SIZE = (1024, 768)  # vis_seg.py --shape default
MEAN = torch.tensor([123.5, 116.5, 103.5]).view(1, 3, 1, 1)
STD = torch.tensor([58.5, 57.0, 57.5]).view(1, 3, 1, 1)


def main() -> None:
    dirs = [Path(d) for d in sys.argv[1].split(":")]
    rgb = torch.from_numpy(image()).permute(2, 0, 1)[None].float()  # the demo's BGR -> RGB channel flip, 0..255
    x = (F.interpolate(rgb, SIZE, mode="bilinear", align_corners=False) - MEAN) / STD
    for key in sys.argv[2:] or VARIANTS:
        ref, file = VARIANTS[key]
        model = torch.jit.load(str(next(p for d in dirs for p in d.rglob(file))), map_location="cpu").eval()
        with torch.inference_mode():
            logits = F.interpolate(model(x), rgb.shape[-2:], mode="bilinear", align_corners=False)
        del model
        write(
            f"{ref}@1",
            {"logits": logits[0].numpy().astype(np.float32)},
            family="sapiens",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, TorchScript checkpoint + Sapiens-Lite demo protocol, torch {torch.__version__}",
            task="segmentation",
        )


if __name__ == "__main__":
    main()
