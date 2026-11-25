"""Faces of a whole image for the face restorers: detect, align to the FFHQ crop, paste the restored crop back.

The steps of facexlib's `FaceRestoreHelper` (the GFPGAN / CodeFormer inference scripts, `upscale_factor=1`,
`face_size=512`), in PyTorch on the model's device (no OpenCV):

- detection with RetinaFace (ResNet-50, the restorers' detector) on the image scaled up to a short side of 640 pixels
  when it is smaller (CodeFormer's `resize=640`), faces with eyes closer than 5 pixels skipped (`eye_dist_threshold`);
- alignment: the similarity transform taking the five landmarks to facexlib's 512 x 512 template (least squares;
  facexlib calls `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)`, the same transform for five consistent points),
  the crop sampled bilinearly with facexlib's gray border;
- paste back: the restored crop warped back, blended through facexlib's soft mask (the warped crop eroded by a
  margin of a tenth of the face's size and blurred by as much).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FACE_SIZE = 512
DETECTOR = "retinaface-resnet50"
DETECT_SHORT_SIDE = 640  # CodeFormer's `get_face_landmarks_5(resize=640)`: small images are scaled up for detection
EYE_DIST_THRESHOLD = 5.0  # pixels between the eyes below which a detection is skipped (both restorers' scripts)
# facexlib `FaceRestoreHelper.face_template` for a 512 x 512 crop: eyes, nose, mouth corners (x, y)
TEMPLATE = (
    (192.98138, 239.94708),
    (318.90277, 240.1936),
    (256.63416, 314.01935),
    (201.26117, 371.41043),
    (313.08905, 371.15118),
)
BORDER = (132 / 255, 133 / 255, 135 / 255)  # facexlib's `borderValue=(135, 133, 132)` (BGR) of the aligned crop


def similarity(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """The `[2, 3]` rotation + uniform scale + translation taking the points `src` `[N, 2]` nearest `dst` (Umeyama)."""
    src, dst = src.double(), dst.double()
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    var = (s**2).sum() / len(src)
    cov = d.T @ s / len(src)
    u, sv, vt = torch.linalg.svd(cov)
    e = torch.ones(2, dtype=torch.float64, device=src.device)
    if torch.det(u) * torch.det(vt) < 0:
        e[-1] = -1  # no reflection
    r = u @ torch.diag(e) @ vt
    scale = (sv * e).sum() / var
    t = md - scale * r @ ms
    return torch.cat([scale * r, t[:, None]], 1)


def invert(m: torch.Tensor) -> torch.Tensor:
    """The inverse of a `[2, 3]` affine transform (`cv2.invertAffineTransform`)."""
    a = torch.linalg.inv(m[:, :2])
    return torch.cat([a, -(a @ m[:, 2:])], 1)


def warp(image: torch.Tensor, m: torch.Tensor, size: tuple[int, int], fill: tuple[float, ...] | None) -> torch.Tensor:
    """`cv2.warpAffine(image, m, size)`: `[C, H, W]` -> `[C, h, w]`, bilinear, outside pixels `fill` (None: zero).

    `m` maps input pixel coordinates to output ones (pixel centres at integers, as OpenCV).
    """
    c, h_in, w_in = image.shape
    h, w = size
    inv = invert(m).to(image.device, torch.float32)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32),
        torch.arange(w, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    grid = torch.stack([(2 * sx + 1) / w_in - 1, (2 * sy + 1) / h_in - 1], -1)[None]
    x = image[None].float()
    if fill is not None:  # sample a ones channel too: where it is below 1 the pixel reaches outside
        x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    if fill is None:
        return out
    inside = out[c:]
    background = torch.tensor(fill, device=out.device, dtype=out.dtype).view(c, 1, 1)
    return out[:c] + (1 - inside) * background


def landmarks(detector: object, image: torch.Tensor) -> list[torch.Tensor]:
    """Five landmarks `[5, 2]` (input pixels) of each face RetinaFace finds in `image` `[3, H, W]`, best score first."""
    h, w = image.shape[-2:]
    scale = max(1.0, DETECT_SHORT_SIDE / min(h, w))
    x = image
    if scale > 1:
        size = (int(h * scale), int(w * scale))
        x = F.interpolate(image[None].float(), size=size, mode="bilinear", align_corners=False)[0].clamp(0, 1)
    found = detector(x)  # ty: ignore[call-non-callable] - a fastforks pose model
    points = found.keypoints.float() / scale
    return [p for p in points if torch.linalg.vector_norm(p[0] - p[1]) >= EYE_DIST_THRESHOLD]


def align(image: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The aligned 512 x 512 crop `[3, 512, 512]` of the face with landmarks `points`, and its transform `[2, 3]`."""
    m = similarity(points.cpu(), torch.tensor(TEMPLATE))
    return warp(image, m, (FACE_SIZE, FACE_SIZE), BORDER), m


def _erode(mask: torch.Tensor, k: int) -> torch.Tensor:
    """`cv2.erode` with a `k` x `k` box (anchor at its centre) of a `[H, W]` mask."""
    if k <= 1:
        return mask
    lo, hi = (k - 1) // 2, k // 2
    x = F.pad(mask[None, None], (lo, hi, lo, hi), value=1.0)
    return -F.max_pool2d(-x, k, 1)[0, 0]


def _blur(mask: torch.Tensor, k: int) -> torch.Tensor:
    """`cv2.GaussianBlur(mask, (k, k), 0)` (sigma from the size, reflected borders) of a `[H, W]` mask."""
    if k <= 1:
        return mask
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    r = torch.arange(k, device=mask.device, dtype=mask.dtype) - (k - 1) / 2
    g = torch.exp(-(r**2) / (2 * sigma * sigma))
    g = g / g.sum()
    x = F.pad(mask[None, None], (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k))
    return F.conv2d(x, g.view(1, 1, k, 1))[0, 0]


def paste(image: torch.Tensor, face: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """`image` `[3, H, W]` with the restored crop `face` `[3, 512, 512]` (aligned by `m`) blended back in.

    facexlib's full-image blend, computed in the face's box only: outside the warped crop plus the reach of the erosion
    and the blur the soft mask is exactly 0, so the image is unchanged there
    (a 4K photo, RTX 5060 Ti: 30 -> 3 ms a face).
    """
    h, w = image.shape[-2:]
    inv = invert(m)
    fh, fw = face.shape[-2:]
    corners = torch.tensor([[0.0, 0.0, 1.0], [fw - 1, 0, 1], [0, fh - 1, 1], [fw - 1, fh - 1, 1]]) @ inv.cpu().float().T
    lo, hi = corners.amin(0).tolist(), corners.amax(0).tolist()  # (x, y) of the warped crop; bilinear: 1 px more
    tight = _clip((math.floor(lo[0]) - 2, math.floor(lo[1]) - 2, math.ceil(hi[0]) + 3, math.ceil(hi[1]) + 3), (h, w))
    out = image.float().clone()
    if tight[0] >= tight[2] or tight[1] >= tight[3]:  # the face lies outside the image
        return out
    ones = torch.ones_like(face[:1])
    mask = _erode(_region_warp(ones, inv, tight, (h, w))[0], 2)  # facexlib: `np.ones((2 * upscale_factor, ...))`
    edge = int(math.sqrt(float(mask.sum()))) // 20  # the full image's sum: the mask is 0 outside the box
    reach = 4 * edge + 4  # the erosion and the blur read 2 * edge px around the mask (reflected at the box): twice that
    box = _clip((tight[0] - reach, tight[1] - reach, tight[2] + reach, tight[3] + reach), (h, w))
    restored = _region_warp(face, inv, box, (h, w))
    mask = _erode(_region_warp(ones, inv, box, (h, w))[0], 2)
    soft = _blur(_erode(mask, 2 * edge), 2 * edge + 1) if edge else mask
    soft = soft.clamp(0, 1)[None]
    region = out[:, box[1] : box[3], box[0] : box[2]]
    region.copy_(soft * (mask[None] * restored) + (1 - soft) * region)
    return out


def _clip(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """`(x0, y0, x1, y1)` clipped to an `H x W` image."""
    h, w = size
    return max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)


def _region_warp(
    face: torch.Tensor, inv: torch.Tensor, box: tuple[int, int, int, int], size: tuple[int, int]
) -> torch.Tensor:
    """`warp(face, inv, size, None)` in the pixels of `box` `(x0, y0, x1, y1)` (clipped to the image `size`)."""
    x0, y0, x1, y1 = _clip(box, size)
    shifted = inv.clone()
    shifted[0, 2] -= x0
    shifted[1, 2] -= y0
    return warp(face, shifted, (max(y1 - y0, 0), max(x1 - x0, 0)), None)
