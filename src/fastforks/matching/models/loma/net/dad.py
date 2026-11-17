"""DaD (Edstedt et al. 2025, Parskatt/dad, MIT) keypoint sampling as LoMa runs it: upstream `dad/detectors/dedode_detector.py`
`DaD.forward` + `dad/utils.py` `sample_keypoints(use_nms=True, nms_size=3, sample_topk=True, subpixel=True,
subpixel_temp=0.5, increase_coverage=False, remove_borders=False)` for batch 1; the network itself is
fastforks.shared.dedode's `Detector(DETECTOR_S, VGG11_BN)` (upstream `dedode_detector_S`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastforks.shared.dedode import get_grid


def sample_keypoints(logits: torch.Tensor, num_samples: int, nms_size: int, temp: float) -> torch.Tensor:
    """The `num_samples` best non-maximum-suppressed pixels of the softmax over all pixels, refined to sub-pixel by
    the softmax (temperature `temp`) of the logits in their `nms_size` x `nms_size` patch.
    `[1, 1, H, W]` -> normalised keypoints `[N, 2]` (x, y in [-1, 1], pixel centres as `get_grid`)."""
    _, _, h, w = logits.shape
    probs = logits.reshape(1, h * w).softmax(dim=-1).reshape(1, h, w)
    probs = probs * (probs == F.max_pool2d(probs, nms_size, stride=1, padding=nms_size // 2))
    inds = torch.topk(probs.reshape(h * w), k=num_samples).indices
    kpts = get_grid(h, w, logits.device)[inds]
    # upstream `extract_patches_from_inds`: the patch's logits, in the unfold's row-major order
    patches = F.unfold(logits, nms_size, padding=nms_size // 2)[0, :, inds]  # [k*k, N]
    weights = (patches / temp).softmax(dim=0)
    offsets = get_grid(nms_size, nms_size, logits.device) * nms_size  # upstream: the k x k grid times k / size
    offsets = offsets / torch.tensor([w, h], device=logits.device, dtype=offsets.dtype)
    return kpts + weights.mT @ offsets
