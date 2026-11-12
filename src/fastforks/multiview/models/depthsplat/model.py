"""DepthSplat runners: weights -> network and the portable / kernels forward passes.

Upstream `EncoderDepthSplat.forward` (eval, the released re10k config: one scale, 1/4 features, ViT-S monocular
branch, SH degree 2) on all views at once: `MultiViewUniMatch` predicts every view's depth from the multi-view cost
volume (128 inverse-depth candidates between `1 / far` and `1 / near`; with more than 3 views each view matches its 2
nearest cameras, as upstream `local_mv_match = 2`), the DPT feature upsampler and the Gaussian head predict one Gaussian
per pixel, and `GaussianAdapter` turns them into world-space Gaussians.

Images are resized to the network size (long side `long_side`, both sides multiples of 16, bicubic antialiased;
upstream: 256 x 256 Lanczos crops); the intrinsics are normalised by the input size (upstream's convention, pixel
centres at `(i + 0.5) / W`), which makes them independent of the resize. Depth and confidence (upstream's match
probability: the largest softmax weight over the depth candidates) are resized back (bilinear). The Gaussians stay at
the network resolution. The network is not scale invariant (its depth residual is added in inverse scene units), so
the scene is scaled to near = `NEAR` (upstream's re10k bounds) and depth, centres and scales are scaled back; upstream
takes the scene as given (its DTU protocol divides millimetres by 200). Upstream exports the camera-frame rotations and
the world covariance; here the rotations are rotated into the world frame (`R_c2w q`, `w x y z`, `w >= 0`), so scales +
rotations give upstream's covariance. The spherical harmonics are rotated as upstream (`e3nn`'s Wigner D matrices of
`R_c2w`, applied to the 3DGS-ordered coefficients; here written without `e3nn`: `D_l = Y_l(R S) pinv(Y_l(S))` on fixed
directions `S`).

fp16 and kernels: fp16 weights under autocast (upstream trains in fp32 / bf16 mixed); kernels adds the `fastbb` DINOv2
(flash attention, fp16-accumulate GEMMs) and cuDNN fp16-accumulate convs. The Gaussian adapter runs in fp32.
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
from fastforks.shared.depth_anything_v2 import use_fast_vit

from .net.dpt_head import DPTHead
from .net.mv_unimatch import MultiViewUniMatch

if TYPE_CHECKING:
    import contextlib

    from fastforks.multiview._runner import Context

MULTIPLE = 16  # fixed: 1/4 features, two UNet downsamplings and the 2 x 2 attention splits
SH_DEGREE = 2  # fixed: the released config (gaussian_adapter.sh_degree)
SH = (SH_DEGREE + 1) ** 2  # fixed: SH coefficients per channel
RAW_DIM = 1 + 2 + 3 + 4 + 3 * SH  # fixed: opacity, pixel offset, scales, rotation (x y z w), SH
NEIGHBOURS = 2  # fixed: upstream local_mv_match (cost volume against the nearest cameras when more than 3 views)
C0 = 0.28209479177387814  # fixed: degree-0 SH constant (upstream RGB2SH)
NEAR = 1.0  # fixed: the scene is scaled so that near = 1 (upstream's re10k training bounds: near 1, far 100)


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (both sides rounded to multiples of 16)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: long side `long_side`, both sides multiples of 16."""
    s = long_side / max(h, w)
    return (max(MULTIPLE, round(h * s / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(w * s / MULTIPLE) * MULTIPLE))


class Net(nn.Module):
    """`EncoderDepthSplat`'s modules at inference (same attribute names: the checkpoint keys after `encoder.`)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.depth_predictor = MultiViewUniMatch(
            num_scales=1, upsample_factor=4, lowest_feature_resolution=4, vit_type="vits", unet_channels=128
        )
        self.feature_upsampler = DPTHead(
            in_channels=384, features=64, out_channels=[48, 96, 192, 384], downsample_factor=4, return_feature=True,
            num_scales=1,
        )  # fmt: skip
        self.gaussian_regressor = nn.Sequential(nn.Conv2d(69, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 64, 3, 1, 1))
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(132, RAW_DIM, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(RAW_DIM, RAW_DIM, 3, 1, 1, padding_mode="replicate"),
        )

    def forward(
        self, images: torch.Tensor, k_norm: torch.Tensor, c2w: torch.Tensor, near: torch.Tensor, far: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `EncoderDepthSplat.forward` up to the Gaussian head (b = 1).

        Args:
            images: `[1, V, 3, h, w]` in `[0, 1]`.
            k_norm: `[1, V, 3, 3]` intrinsics normalised by the image size.
            c2w: `[1, V, 4, 4]` camera-to-world.
            near: `[1, V]` nearest depth.
            far: `[1, V]` farthest depth.

        Returns:
            Depth `[V, h, w]`, match probability `[V, 1, h, w]`, raw Gaussian parameters `[V, RAW_DIM, h, w]`.
        """
        v = images.shape[1]
        nn_matrix = None
        if v > NEIGHBOURS + 1:
            xyz = c2w[:, :, :3, -1].float()
            dist = (xyz[:, :, None] - xyz[:, None]).norm(dim=-1)  # upstream torch.cdist (no ONNX conversion)
            nn_matrix = torch.argsort(dist)[:, :, : NEIGHBOURS + 1]
        res = self.depth_predictor(
            images, attn_splits_list=[2], min_depth=1.0 / far, max_depth=1.0 / near, intrinsics=k_norm,
            extrinsics=c2w, nn_matrix=nn_matrix,
        )  # fmt: skip
        depth = res["depth_preds"][-1][0]  # [V, h, w]
        features = self.feature_upsampler(
            res["features_mono_intermediate"], cnn_features=res["features_cnn_all_scales"][::-1],
            mv_features=res["features_mv"][0],
        )  # fmt: skip
        match_prob = res["match_probs"][-1].max(dim=1, keepdim=True)[0]
        match_prob = F.interpolate(match_prob, size=depth.shape[-2:], mode="nearest")
        img = images[0].to(features.dtype)
        out = self.gaussian_regressor(torch.cat((img, depth[:, None].to(features.dtype), match_prob, features), 1))
        raw = self.gaussian_head(torch.cat((out, img, features, match_prob), 1))
        return depth, match_prob, raw


def _net(ctx: Context) -> Net:
    sd = {k.removeprefix("encoder."): v for k, v in ctx.state_dict(key="state_dict").items()}
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


def _y2(d: torch.Tensor) -> torch.Tensor:
    """`e3nn` degree-2 real spherical harmonics (component normalisation) of unit vectors `[..., 3]` -> `[..., 5]`."""
    x, y, z = d.unbind(-1)
    s15, s5 = math.sqrt(15), math.sqrt(5)
    return torch.stack(
        (s15 * x * z, s15 * x * y, s5 * (y * y - (x * x + z * z) / 2), s15 * y * z, s15 / 2 * (z * z - x * x)), -1
    )


def _directions() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed unit directions `S` `[3, N]` and `pinv(Y_2(S))` `[N, 5]` (float64)."""
    s = np.random.default_rng(0).standard_normal((3, 32))
    s /= np.linalg.norm(s, axis=0)
    y = _y2(torch.from_numpy(s.T)).numpy().T  # [5, N]
    return torch.from_numpy(s), torch.from_numpy(np.linalg.pinv(y))


_S, _P = _directions()  # constants (torch.compile / ONNX export see plain tensors)


def sh_rotation(r: torch.Tensor) -> torch.Tensor:
    """Upstream `rotate_sh`'s per-degree Wigner D matrices of rotations `[V, 3, 3]` -> block-diagonal `[V, SH, SH]`.

    `e3nn`: `D_0 = 1`, `D_1 = R` (its degree-1 basis is `x y z`), `D_2 Y_2(s) = Y_2(R s)` for every direction `s`.
    """
    s, p = _S.to(r), _P.to(r)
    d = r.new_zeros(r.shape[0], SH, SH)
    d[:, 0, 0] = 1
    d[:, 1:4, 1:4] = r
    d[:, 4:9, 4:9] = _y2((r @ s).transpose(1, 2)).transpose(1, 2) @ p
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
    raw: torch.Tensor, depth: torch.Tensor, images: torch.Tensor, k_norm: torch.Tensor, c2w: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Upstream's Gaussian activations and `GaussianAdapter` (fp32) -> the `gaussians/*` fields, view-major.

    Args:
        raw: `[V, RAW_DIM, h, w]` Gaussian head output.
        depth: `[V, h, w]`.
        images: `[V, 3, h, w]` network input in `[0, 1]` (the SH DC initialisation).
        k_norm: `[V, 3, 3]` normalised intrinsics.
        c2w: `[V, 4, 4]` camera-to-world.
    """
    v, _, h, w = raw.shape
    raw = raw.float().permute(0, 2, 3, 1).reshape(v, h * w, RAW_DIM)
    dev = raw.device
    ys, xs = ((torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n for n in (h, w))
    xy = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(1, h * w, 2)
    xy = xy + (raw[..., 1:3].sigmoid() - 0.5) / torch.tensor((w, h), dtype=torch.float32, device=dev)
    scales, rot, sh = raw[..., 3:].split((3, 4, 3 * SH), -1)
    rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)  # x y z w, camera frame
    mask = torch.tensor([1.0] + [0.1 * 0.25**deg for deg in range(1, SH_DEGREE + 1) for _ in range(2 * deg + 1)])
    sh = sh.reshape(v, h * w, 3, SH) * mask.to(dev)
    rgb = images.float().permute(0, 2, 3, 1).reshape(v, h * w, 3)
    sh = torch.cat((sh[..., :1] + ((rgb - 0.5) / C0)[..., None], sh[..., 1:]), -1)
    c2w = c2w.float()
    r = c2w[:, None, :3, :3]
    rays = (
        invert(k_norm.float())[:, None] @ torch.cat((xy, torch.ones_like(xy[..., :1])), -1).expand(v, -1, -1)[..., None]
    )
    rays = rays[..., 0] / rays[..., 2:, 0]
    means = c2w[:, None, :3, 3] + (r @ rays[..., None])[..., 0] * depth.float().reshape(v, h * w, 1)
    c2w64 = f64(c2w)[:, :3, :3]
    q = quaternion_product(matrix_quaternion(c2w64)[:, None], f64(rot[..., [3, 0, 1, 2]]))
    q = (q * torch.where(q[..., :1] < 0, -1.0, 1.0)).float()
    harmonics = sh @ sh_rotation(c2w64).float()[:, None].transpose(-1, -2)
    return {
        "gaussians/means": means.reshape(-1, 3),
        "gaussians/scales": F.softplus(scales - 4.0).clamp(1e-10, 3.0).reshape(-1, 3),
        "gaussians/rotations": q.reshape(-1, 4),
        "gaussians/harmonics": harmonics.reshape(-1, 3, SH),
        "gaussians/opacities": raw[..., 0].sigmoid().reshape(-1),
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
        self.half = ctx.dtype == "float16"
        if self.half:
            self.net.half().to(memory_format=torch.channels_last)

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
        near, far = (depth_range[i].float().mul(s).expand(1, n) for i in (0, 1))
        with self._amp():
            depth, prob, raw = self.net(x[None], k[None], net_c2w[None], near, far)
        depth = depth.float() / s

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)[:, 0]

        maps = {"depth": up(depth[:, None]), "confidence": up(prob)}
        g = gaussians(raw, depth, x, k, c2w)
        g["gaussians/scales"] = g["gaussians/scales"] / s  # upstream's clamp (`3.0`) acts in the scaled scene
        return maps | g

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth and confidence of every view and the scene's Gaussians (one per network pixel), in the input frame."""
        if intrinsics is None or cam_to_world is None or depth_range is None:  # the core checks required inputs
            msg = "DepthSplat needs intrinsics, cam_to_world and depth_range"
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
    """fp16: the `fastbb` DINOv2 (flash attention, fp16-accumulate GEMMs), cuDNN fp16-accumulate convs."""

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        cudnn_hacc.load(ctx.device)
        use_fast_vit(self.net.depth_predictor.pretrained)
        if _hacc_convs(self.net) == 0:
            msg = "DepthSplat kernels: no conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def forward(
        self, images: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor, depth_range: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """`_Base.forward` with fp16-accumulate GEMMs."""
        with fp16_accumulate():
            return super().forward(images, intrinsics, cam_to_world, depth_range)


RUNNERS = {"portable": Portable, "kernels": Kernels}
