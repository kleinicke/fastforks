"""MVSplat runners: weights -> network and the portable / kernels forward passes.

Upstream `EncoderCostVolume.forward` (eval, the released re10k config: 1/4 features, 128 inverse-depth candidates
between `1 / far` and `1 / near`, cost-volume U-Net with cross-view attention at 1/16, depth-refinement U-Net, SH
degree 4) on all views at once: the UniMatch CNN + multi-view swin transformer give per-view features, every view's
cost volume averages its correlations with all other views, the U-Nets refine it to a depth, and the Gaussian head
predicts one Gaussian per pixel, which `GaussianAdapter` turns into world-space Gaussians.

Images are resized to the network size (long side `long_side`, both sides multiples of 16, bicubic antialiased;
upstream: 256 x 256 crops); the intrinsics are normalised by the input size (upstream's convention, pixel centres at
`(i + 0.5) / W`). Depth and confidence (the largest softmax weight over the depth candidates) are resized back
(bilinear); the Gaussians stay at the network resolution. The network is not scale invariant (the depth residual is
added in inverse scene units), so the scene is scaled to near = `NEAR` (upstream's re10k bounds) and depth and
centres are scaled back. As upstream, a Gaussian's centre lies at the predicted depth along its unit ray (not at that
z), its scales are proportional to the depth, and its opacity is the sigmoid of the density head (upstream's opacity
mapping with `initial = final = 0` is the identity). Upstream exports the camera-frame rotations and the world
covariance; here the rotations are rotated into the world frame (`R_c2w q`, `w x y z`, `w >= 0`), so scales +
rotations give upstream's covariance. The spherical harmonics are rotated as upstream (`e3nn`'s Wigner D matrices of
`R_c2w`; here written without `e3nn`: `D_l = Y_l(R S) pinv(Y_l(S))` on fixed directions `S`, `Y_l` `e3nn`'s
polynomials).

fp16 and kernels: fp16 weights under autocast (upstream trains in fp32), the warped features and correlations in
fp16; kernels adds cuDNN fp16-accumulate convs and fp16-accumulate GEMMs. The Gaussian adapter runs in fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels.cudnn_hacc import conv_half_acc
from fastforks.multiview._runner import Forward, Runner, f64, invert
from fastforks.shared.group_norm import fast_group_norms

from .net.backbone.multiview_transformer import MultiViewFeatureTransformer
from .net.backbone.unimatch.backbone import CNNEncoder
from .net.backbone.unimatch.position import PositionEmbeddingSine
from .net.backbone.unimatch.utils import merge_splits, split_feature
from .net.costvolume.ldm_unet.unet import QKVAttentionLegacy, UNetModel

if TYPE_CHECKING:
    import contextlib

    from fastforks.multiview._runner import Context

MULTIPLE = 16  # fixed: 1/4 features, 2 x 2 attention splits, the U-Nets' downsamplings
FEATURES = 128  # fixed: the released config (d_feature)
CANDIDATES = 128  # fixed: inverse-depth candidates (num_depth_candidates)
SPLITS = 2  # fixed: multiview_trans_attn_split
SH_DEGREE = 4  # fixed: the released config (gaussian_adapter.sh_degree)
SH = (SH_DEGREE + 1) ** 2  # fixed: SH coefficients per channel
RAW_DIM = 2 + 3 + 4 + 3 * SH  # fixed: pixel offset, scales, rotation (x y z w), SH
SCALE_MIN, SCALE_MAX = 0.5, 15.0  # fixed: gaussian_scale_min / max
NEAR = 1.0  # fixed: the scene is scaled so that near = 1 (upstream's re10k training bounds: near 1, far 100)
MEAN = (0.485, 0.456, 0.406)  # fixed: ImageNet normalisation of the UniMatch backbone
STD = (0.229, 0.224, 0.225)
DEPTH_CHUNK = 32  # fixed: a memory bound (depth candidates warped at once, [V, C, D, h, w] volume)


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (both sides rounded to multiples of 16)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: long side `long_side`, both sides multiples of 16."""
    s = long_side / max(h, w)
    return (max(MULTIPLE, round(h * s / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(w * s / MULTIPLE) * MULTIPLE))


def _unet(channels: int, attention: tuple[int, ...], mult: tuple[int, ...]) -> UNetModel:
    return UNetModel(
        image_size=None, in_channels=channels, model_channels=channels, out_channels=channels, num_res_blocks=1,
        attention_resolutions=attention, channel_mult=mult, num_head_channels=32, dims=2, postnorm=True, num_frames=2,
        use_cross_view_self_attn=True,
    )  # fmt: skip


class Backbone(nn.Module):
    """Upstream `BackboneMultiview` (released config): CNN features and the multi-view swin transformer."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.backbone = CNNEncoder(output_dim=FEATURES, num_output_scales=0)
        self.transformer = MultiViewFeatureTransformer(num_layers=6, d_model=FEATURES, nhead=1, ffn_dim_expansion=4)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[V, 3, h, w]` in `[0, 1]` -> transformer and CNN features, both `[V, C, h / 4, w / 4]`."""
        mean, std = (torch.tensor(t, device=images.device).view(1, 3, 1, 1) for t in (MEAN, STD))
        cnn = self.backbone((images - mean) / std)[0]
        views = list(cnn.unbind(0))
        splits = [split_feature(x[None], num_splits=SPLITS) for x in views]
        pos = PositionEmbeddingSine(num_pos_feats=FEATURES // 2)(splits[0])
        views = [merge_splits(x + pos, num_splits=SPLITS) for x in splits]
        return torch.cat(self.transformer(views, attn_num_splits=SPLITS), 0), cnn


class DepthPredictor(nn.Module):
    """Upstream `DepthPredictorMultiView` (released config; same module names)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        c, d, f = 128, CANDIDATES, 32  # costvolume_unet_feat_dim, candidates, depth_unet_feat_dim
        cin = d + FEATURES
        self.corr_refine_net = nn.Sequential(
            nn.Conv2d(cin, c, 3, 1, 1), nn.GroupNorm(8, c), nn.GELU(), _unet(c, (4,), (1, 1, 1)),
            nn.Conv2d(c, d, 3, 1, 1),
        )  # fmt: skip
        self.regressor_residual = nn.Conv2d(cin, d, 1, 1, 0)
        self.depth_head_lowres = nn.Sequential(nn.Conv2d(d, d * 2, 3, 1, 1), nn.GELU(), nn.Conv2d(d * 2, d, 3, 1, 1))
        self.upsampler = nn.Sequential(
            nn.Conv2d(2 * FEATURES, FEATURES, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(FEATURES, f, 3, 1, 1)
        self.refine_unet = nn.Sequential(
            nn.Conv2d(3 + f + 2, f, 3, 1, 1), nn.GroupNorm(4, f), nn.GELU(), _unet(f, (16,), (1, 1, 1, 1, 1))
        )  # fmt: skip
        gin = f + 3 + FEATURES
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gin, RAW_DIM * 2, 3, 1, 1), nn.GELU(), nn.Conv2d(RAW_DIM * 2, RAW_DIM, 3, 1, 1)
        )  # fmt: skip
        self.to_disparity = nn.Sequential(nn.Conv2d(f, f * 2, 3, 1, 1), nn.GELU(), nn.Conv2d(f * 2, 2, 3, 1, 1))


def _poses(c2w: torch.Tensor) -> list[torch.Tensor]:
    """Upstream `prepare_feat_proj_data_lists`' relative poses: for view offset `i`, `inv(E_{v+i}) E_v` for each `v`.

    Covers upstream's two-view branch too (`inv(E1) E0` and its inverse).
    """
    inv = torch.linalg.inv(c2w)
    return [inv.roll(-i, 0) @ c2w for i in range(1, c2w.shape[0])]


def correlation(
    feat: torch.Tensor,
    k: torch.Tensor,
    c2w: torch.Tensor,
    disp: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Upstream's plane-sweep cost volume, averaged over all other views.

    Args:
        feat: `[V, C, h, w]` features.
        k: `[V, 3, 3]` intrinsics in feature pixels (upstream: normalised intrinsics times `w`, `h`).
        c2w: `[V, 4, 4]` camera-to-world.
        disp: `[V, D]` inverse-depth candidates.
        dtype: dtype of the warped features and products (the sums accumulate in float32).

    Returns:
        `[V, D, h, w]` mean correlation (dot product / sqrt(C)) of every view's features with each other view's
        features warped to the candidate depths (`grid_sample`, bilinear, zeros, align_corners as upstream).
    """
    feat = feat.to(dtype)
    v, c, h, w = feat.shape
    d = disp.shape[1]
    dev = feat.device
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    grid = torch.stack((xs, ys, torch.ones_like(xs)), 0).float().view(1, 3, h * w)
    rays = invert(k.float()) @ grid  # [V, 3, hw]
    depth = (1.0 / disp.float()).view(v, 1, d, 1)
    out = feat.new_zeros(v, d, h, w, dtype=torch.float32)
    poses = _poses(c2w.float())
    for i, pose in enumerate(poses, 1):
        other = feat.roll(-i, 0)  # view j against view (j + i) % V
        pts = (pose[:, :3, :3] @ rays).unsqueeze(2) * depth + pose[:, :3, 3:, None]  # [V, 3, D, hw]
        pix = k.float() @ pts.view(v, 3, -1)
        pix = pix.view(v, 3, d, h * w)
        uv = pix[:, :2] / pix[:, 2:].clamp(min=1e-3)
        g = torch.stack((2 * uv[:, 0] / (w - 1) - 1, 2 * uv[:, 1] / (h - 1) - 1), -1)  # [V, D, hw, 2]
        for j in range(0, d, DEPTH_CHUNK):
            gj = g[:, j : j + DEPTH_CHUNK].reshape(v, -1, w, 2).to(feat.dtype)
            warped = F.grid_sample(other, gj, mode="bilinear", padding_mode="zeros", align_corners=True)
            warped = warped.view(v, c, -1, h, w)
            out[:, j : j + DEPTH_CHUNK] += (warped * feat[:, :, None]).sum(1, dtype=torch.float32)  # no layout copy
    return out / (len(poses) * c**0.5)


class Net(nn.Module):
    """`EncoderCostVolume`'s modules at inference (same attribute names: the checkpoint keys after `encoder.`)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.backbone = Backbone()
        self.depth_predictor = DepthPredictor()

    def forward(
        self, images: torch.Tensor, k_norm: torch.Tensor, c2w: torch.Tensor, near: torch.Tensor, far: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `EncoderCostVolume.forward` up to the Gaussian head (b = 1).

        Args:
            images: `[V, 3, h, w]` in `[0, 1]`.
            k_norm: `[V, 3, 3]` intrinsics normalised by the image size.
            c2w: `[V, 4, 4]` camera-to-world.
            near: `[V]` nearest depth.
            far: `[V]` farthest depth.

        Returns:
            Depth `[V, h, w]`, largest candidate probability `[V, 1, h, w]`, opacity `[V, h, w]`, raw Gaussian
            parameters `[V, RAW_DIM, h, w]`.
        """
        p = self.depth_predictor
        v = images.shape[0]
        for m in self.modules():  # cross-view attention over the views of this call (upstream: its context views)
            if isinstance(m, QKVAttentionLegacy):
                m.n_frames = v
        feat, cnn = self.backbone(images)
        h, w = feat.shape[-2:]
        k = k_norm.float().clone()
        k[:, 0] *= w
        k[:, 1] *= h
        lo, hi = 1.0 / far.float(), 1.0 / near.float()
        disp = lo[:, None] + torch.linspace(0.0, 1.0, CANDIDATES, device=feat.device)[None] * (hi - lo)[:, None]
        head: Any = p.depth_head_lowres[0]
        corr = torch.cat((correlation(feat, k, c2w, disp, dtype=head.weight.dtype).to(feat.dtype), feat), 1)
        corr = p.corr_refine_net(corr) + p.regressor_residual(corr)
        pdf = F.softmax(p.depth_head_lowres(corr).float(), dim=1)
        coarse = (disp[..., None, None] * pdf).sum(1, keepdim=True)
        pdf_max = F.interpolate(pdf.max(dim=1, keepdim=True)[0], scale_factor=4)
        disps = F.interpolate(coarse, scale_factor=4, mode="bilinear", align_corners=True)
        x = images.to(feat.dtype)
        up = p.upsampler(torch.cat((feat, cnn), 1))
        refined = p.refine_unet(torch.cat((x, p.proj_feature(up), disps.to(x.dtype), pdf_max.to(x.dtype)), 1))
        raw = p.to_gaussians(torch.cat((refined, x, up), 1))
        delta, density = p.to_disparity(refined).float().split(1, dim=1)
        fine = (disps + delta).clamp(lo.view(v, 1, 1, 1), hi.view(v, 1, 1, 1))
        return 1.0 / fine[:, 0], pdf_max, density[:, 0].sigmoid(), raw


def _net(ctx: Context) -> Net:
    sd = ctx.state_dict(key="state_dict")
    sd = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    net = Net()
    net.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def normalized_intrinsics(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """OpenCV pixel intrinsics of an `h x w` image -> upstream's normalised ones (pixel centres at `(i + 0.5) / w`)."""
    k = k.float().clone()
    k[:, :2, 2] += 0.5 * k[:, 2:, 2]
    k[:, 0] /= w
    k[:, 1] /= h
    return k


def _y(d: torch.Tensor) -> list[torch.Tensor]:
    """`e3nn`'s real spherical harmonics of degrees 1-4 (component normalisation) of unit vectors `[..., 3]`."""
    x, y, z = d.unbind(-1)
    r2, r3, r5, r6, r7 = (math.sqrt(n) for n in (2, 3, 5, 6, 7))
    r14, r15, r21, r42, r70, r105, r168, r210 = (math.sqrt(n) for n in (14, 15, 21, 42, 70, 105, 168, 210))
    y2, x2z2 = y * y, x * x + z * z
    s2 = (r15 * x * z, r15 * x * y, r5 * (y2 - x2z2 / 2), r15 * y * z, r15 / 2 * (z * z - x * x))
    s3 = (
        r42 / 6 * (s2[0] * z + s2[4] * x),
        r7 * s2[0] * y,
        r168 / 8 * (4 * y2 - x2z2) * x,
        r7 / 2 * y * (2 * y2 - 3 * x2z2),
        r168 / 8 * z * (4 * y2 - x2z2),
        r7 * s2[4] * y,
        r42 / 6 * (s2[4] * z - s2[0] * x),
    )
    s4 = (
        3 / 4 * r2 * (s3[0] * z + s3[6] * x),
        3 / 4 * s3[0] * y + 3 / 8 * r6 * s3[1] * z + 3 / 8 * r6 * s3[5] * x,
        -3 / 56 * r14 * s3[0] * z + 3 / 14 * r21 * s3[1] * y + 3 / 56 * r210 * s3[2] * z + 3 / 56 * r210 * s3[4] * x
        + 3 / 56 * r14 * s3[6] * x,
        -3 / 56 * r42 * s3[1] * z + 3 / 28 * r105 * s3[2] * y + 3 / 28 * r70 * s3[3] * x + 3 / 56 * r42 * s3[5] * x,
        -3 / 28 * r42 * s3[2] * x + 3 / 7 * r7 * s3[3] * y - 3 / 28 * r42 * s3[4] * z,
        -3 / 56 * r42 * s3[1] * x + 3 / 28 * r70 * s3[3] * z + 3 / 28 * r105 * s3[4] * y - 3 / 56 * r42 * s3[5] * z,
        -3 / 56 * r14 * s3[0] * x - 3 / 56 * r210 * s3[2] * x + 3 / 56 * r210 * s3[4] * z + 3 / 14 * r21 * s3[5] * y
        - 3 / 56 * r14 * s3[6] * z,
        -3 / 8 * r6 * s3[1] * x + 3 / 8 * r6 * s3[5] * z + 3 / 4 * s3[6] * y,
        3 / 4 * r2 * (-s3[0] * x + s3[6] * z),
    )  # fmt: skip
    return [torch.stack((r3 * x, r3 * y, r3 * z), -1), *(torch.stack(s, -1) for s in (s2, s3, s4))]


def _directions() -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Fixed unit directions `S` `[3, N]` and `pinv(Y_l(S))` `[N, 2l + 1]` for l = 1..4 (float64)."""
    s = np.random.default_rng(0).standard_normal((3, 64))
    s /= np.linalg.norm(s, axis=0)
    return torch.from_numpy(s), [torch.from_numpy(np.linalg.pinv(y.numpy().T)) for y in _y(torch.from_numpy(s.T))]


_S, _P = _directions()  # constants (torch.compile / ONNX export see plain tensors)


def sh_rotation(r: torch.Tensor) -> torch.Tensor:
    """Upstream `rotate_sh`'s per-degree Wigner D matrices of rotations `[V, 3, 3]` -> block-diagonal `[V, SH, SH]`.

    `e3nn`: `D_0 = 1`, `D_l Y_l(s) = Y_l(R s)` for every direction `s`.
    """
    d = r.new_zeros(r.shape[0], SH, SH)
    d[:, 0, 0] = 1
    for deg, (y, p) in enumerate(zip(_y((r @ _S.to(r)).transpose(1, 2)), _P, strict=True), 1):
        d[:, deg**2 : (deg + 1) ** 2, deg**2 : (deg + 1) ** 2] = y.transpose(1, 2) @ p.to(r)
    return d


def matrix_quaternion(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices `[..., 3, 3]` -> unit quaternions `[..., 4]` `(w, x, y, z)` (use float64 for accuracy)."""
    m = r.flatten(-2).unbind(-1)

    def sign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # copysign (no ONNX conversion of `signbit`)
        return torch.where(b < 0, -a, a)

    w = (1 + m[0] + m[4] + m[8]).clamp_min(0).sqrt()
    x = sign((1 + m[0] - m[4] - m[8]).clamp_min(0).sqrt(), m[7] - m[5])
    y = sign((1 - m[0] + m[4] - m[8]).clamp_min(0).sqrt(), m[2] - m[6])
    z = sign((1 - m[0] - m[4] + m[8]).clamp_min(0).sqrt(), m[3] - m[1])
    q = torch.stack((w, x, y, z), -1)
    return q / q.norm(dim=-1, keepdim=True)


def quaternion_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of `(w, x, y, z)` quaternions (`a b`: rotation `b`, then `a`)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def gaussians(
    raw: torch.Tensor, depth: torch.Tensor, opacity: torch.Tensor, k_norm: torch.Tensor, c2w: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Upstream's Gaussian activations and `GaussianAdapter` (fp32) -> the `gaussians/*` fields, view-major.

    Args:
        raw: `[V, RAW_DIM, h, w]` Gaussian head output.
        depth: `[V, h, w]` (input units).
        opacity: `[V, h, w]`.
        k_norm: `[V, 3, 3]` normalised intrinsics.
        c2w: `[V, 4, 4]` camera-to-world (input units).
    """
    v, _, h, w = raw.shape
    raw = raw.float().permute(0, 2, 3, 1).reshape(v, h * w, RAW_DIM)
    dev = raw.device
    pixel = 1 / torch.tensor((w, h), dtype=torch.float32, device=dev)
    ys, xs = ((torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n for n in (h, w))
    xy = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(1, h * w, 2)
    xy = xy + (raw[..., :2].sigmoid() - 0.5) * pixel
    scales, rot, sh = raw[..., 2:].split((3, 4, 3 * SH), -1)
    k_inv = invert(k_norm.float())
    multiplier = 0.1 * (k_inv[:, :2, :2] @ pixel).sum(-1)  # [V]
    d = depth.float().reshape(v, h * w, 1)
    scales = (SCALE_MIN + (SCALE_MAX - SCALE_MIN) * scales.sigmoid()) * d * multiplier[:, None, None]
    rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)  # x y z w, camera frame
    mask = torch.tensor([1.0] + [0.1 * 0.25**deg for deg in range(1, SH_DEGREE + 1) for _ in range(2 * deg + 1)])
    sh = sh.reshape(v, h * w, 3, SH) * mask.to(dev)
    c2w = c2w.float()
    r = c2w[:, None, :3, :3]
    rays = (k_inv[:, None] @ torch.cat((xy, torch.ones_like(xy[..., :1])), -1).expand(v, -1, -1)[..., None])[..., 0]
    rays = rays / rays.norm(dim=-1, keepdim=True)  # upstream get_world_rays: unit directions
    means = c2w[:, None, :3, 3] + (r @ rays[..., None])[..., 0] * d
    c2w64 = f64(c2w)[:, :3, :3]
    q = quaternion_product(matrix_quaternion(c2w64)[:, None], f64(rot[..., [3, 0, 1, 2]]))
    q = (q * torch.where(q[..., :1] < 0, -1.0, 1.0)).float()
    harmonics = sh @ sh_rotation(c2w64).float()[:, None].transpose(-1, -2)
    return {
        "gaussians/means": means.reshape(-1, 3),
        "gaussians/scales": scales.reshape(-1, 3),
        "gaussians/rotations": q.reshape(-1, 4),
        "gaussians/harmonics": harmonics.reshape(-1, 3, SH),
        "gaussians/opacities": opacity.float().reshape(-1),
    }


class _Base(Runner):
    intrinsics = "required"
    cam_to_world = "required"
    depth_range = "required"
    per_view = True
    min_views = 2
    batch = None
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = _net(ctx)
        fast_group_norms(self.net)  # the depth refiner's full-size GroupNorms (channels-last in float16)
        self.half = ctx.dtype == "float16"
        if self.half:
            self.net.half().to(memory_format=torch.channels_last)
            for m in self.net.modules():
                if isinstance(m, UNetModel):
                    m.dtype = torch.float16  # its input cast (upstream: use_fp16)

    def _amp(self) -> contextlib.AbstractContextManager[Any]:
        return torch.autocast(self.ctx.device.type, dtype=torch.float16, enabled=self.half)

    def forward(
        self, images: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor, depth_range: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The runner's inputs -> depth, confidence (input size) and the per-pixel Gaussians (network size)."""
        n, _, h0, w0 = images.shape
        hn, wn = network_size(h0, w0, self.ctx.params.long_side)
        x = images
        if (hn, wn) != (h0, w0):
            x = F.interpolate(images, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        k = normalized_intrinsics(intrinsics, h0, w0)
        c2w = cam_to_world.float()
        s = NEAR / depth_range[0].float()  # the network is not scale invariant: depth residuals in 1 / scene units
        net_c2w = c2w.clone()
        net_c2w[:, :3, 3] *= s
        near, far = (depth_range[i].float().mul(s).expand(n) for i in (0, 1))
        with self._amp():
            depth, prob, opacity, raw = self.net(x, k, net_c2w, near, far)
        depth = depth.float() / s

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)[:, 0]

        return {"depth": up(depth[:, None]), "confidence": up(prob)} | gaussians(raw, depth, opacity, k, c2w)

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth and confidence of every view and the scene's Gaussians (one per network pixel), in the input frame."""
        if intrinsics is None or cam_to_world is None or depth_range is None:  # the core checks required inputs
            msg = "MVSplat needs intrinsics, cam_to_world and depth_range"
            raise RuntimeError(msg)
        return self.forward(images, intrinsics, cam_to_world, depth_range)


class Portable(_Base):
    """Plain PyTorch: float32 = upstream modules; float16 (CUDA) = fp16 weights under autocast."""

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps and Gaussians of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


def _hacc_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return conv_half_acc(x.half(), self.weight, self.bias, list(self.padding), list(self.stride))  # ty: ignore


def _hacc_convs(module: nn.Module) -> int:
    """Route every fp16 Conv2d (groups 1, dilation 1, zero padding) through cuDNN fp16-accumulate convs."""
    n = 0
    for m in module.modules():
        if (
            is_base(m, nn.Conv2d)
            and m.groups == 1
            and m.dilation == (1, 1)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
            and m.weight.dtype == torch.float16
        ):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            override(m, _hacc_forward)
            n += 1
    return n


class Kernels(_Base):
    """fp16: cuDNN fp16-accumulate convs and fp16-accumulate GEMMs."""

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc  # noqa: PLC0415 - kernels only

        cudnn_hacc.load(ctx.device)
        if _hacc_convs(self.net) == 0:
            msg = "MVSplat kernels: no conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def forward(
        self, images: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor, depth_range: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """`_Base.forward` with fp16-accumulate GEMMs."""
        with fp16_accumulate():
            return super().forward(images, intrinsics, cam_to_world, depth_range)


RUNNERS = {"portable": Portable, "kernels": Kernels}
