# ruff: noqa: PLR0913, PLR0917, PLR2004 - upstream argument lists and latent-size configurations
"""ReSplat runners: weights -> network and the portable / kernels forward passes.

Upstream `EncoderReSplat.forward` + `forward_update` at eval (b = 1, all views at once). `MultiViewUniMatch` (as
DepthSplat, with log-depth candidates) predicts every view's depth; DINOv2, CNN and cost-volume features at the latent
resolution (`latent` x downsampled) feed a CNN regressor and a kNN point transformer with multi-view attention over
the unprojected latent points, whose head predicts `gaussians_per_point` Gaussians per latent pixel (SH degree 3).
`refine` recurrent steps then render the input views (`render.py`, upstream's gsplat call), compare ResNet-18
features and colours of render and input, and a second kNN point transformer updates every Gaussian.

Images are resized to the network size (long side `long_side`, default the training size, both sides multiples of
`size_multiple`, bicubic antialiased; upstream: Lanczos resize + centre crop); intrinsics are normalised by the input
size (pixel centres at `(i + 0.5) / W`). As upstream (`pose_align_middle_view`), the poses are expressed in the frame
of the middle view `V // 2` (the network sees absolute Gaussian positions), and the Gaussians are mapped back to the
input frame: centres and rotations (`R_c2w q`, `w x y z`, `w >= 0`) exactly, the SH coefficients by gsplat-basis
rotation matrices (`D_l = B_l(R^T S) pinv(B_l(S))` on fixed directions `S`). Depth range: the given one as-is, else
upstream's training bounds (`near`, `far`); the network is not scale invariant (scenes at RealEstate10K / DL3DV scale).

fp16 and kernels: fp16 weights under autocast (upstream evaluates under bf16 autocast); Gaussian activations, the
renderer and the refinement bookkeeping run in fp32. kernels adds the `fastbb` DINOv2 (flash attention,
fp16-accumulate GEMMs) and cuDNN fp16-accumulate convs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks.multiview._runner import Forward, Runner, f64, invert
from fastforks.multiview.models.depthsplat.model import (
    _hacc_convs,
    matrix_quaternion,
    normalized_intrinsics,
    quaternion_product,
)
from fastforks.shared.depth_anything_v2 import use_fast_vit

from .net.mv_unimatch import MultiViewUniMatch
from .net.point_transformer import MultViewLowresAttn, PlainPointTransformer, PointLinearWrapper
from .net.resnet import ResNetFeatureWarpper
from .render import render, sh_basis

if TYPE_CHECKING:
    import contextlib

    from fastforks.multiview._runner import Context

SH = 16  # fixed: SH degree 3 (gaussian_adapter.sh_degree)
RAW = 4 + 3 + 1 + 2 + 3 * SH  # fixed: rotation (x y z w), scale, opacity, pixel offset, SH per Gaussian
STATE = 3 + 3 + 4 + 1 + 3 * SH  # fixed: means, scales, rotation, opacity logit, SH (the refinement's Gaussian state)
STATE_CHANNELS = 512  # fixed: state_channels
RESNET_CHANNELS = 64 + 64 + 128  # fixed: ResNet-18 conv1, layer1, layer2
NEIGHBOURS = 2  # fixed: local_mv_match (cost volume against the nearest cameras when more than 3 views)
C0 = 0.28209479177387814  # fixed: degree-0 SH constant (upstream RGB2SH)
SCALE_BIAS, SCALE_MIN, SCALE_MAX = 4.0, 1e-6, 1.0  # fixed: gaussian_adapter exp_scale_bias, clamp_min_scale, max
MONO = {"vits": 384, "vitb": 768}  # ViT channels
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@dataclass(frozen=True)
class Params:
    """Architecture (variant params) and load options."""

    vit: str  # monodepth_vit_type
    channels: int  # gaussian_regressor_channels
    latent: int  # latent_downsample
    fixed_latent: bool  # fixed_latent_size
    gaussians_per_point: int  # init_gaussian_multiple
    same_points: bool  # refine_same_num_points
    lowest: int  # lowest_feature_resolution = upsample_factor
    size_multiple: int  # network sides are multiples of this
    train_size: tuple[int, int]  # training resolution (h, w)
    near: float  # training depth bounds (used without depth_range)
    far: float
    long_side: int  # network long side (0: the training size's)
    refine: int  # recurrent refinement steps (upstream num_refine)


def network_size(h: int, w: int, p: Params) -> tuple[int, int]:
    """Network input size of an `h x w` image."""
    m = p.size_multiple
    s = (p.long_side or max(p.train_size)) / max(h, w)
    return (max(m, round(h * s / m) * m), max(m, round(w * s / m) * m))


def _feature_channels(p: Params) -> int:
    c = MONO[p.vit]
    if p.latent == 4 or (p.latent == 8 and p.fixed_latent):
        return c // 4 + 128 + 64 + 96 + 128
    if p.latent == 2:
        if p.lowest == 8:
            return c // 64 * 4 + 128 // 16 + 64 + 96 // 4 + 128 // 16
        return c // 64 * 4 + 128 // 4 + 64 + 96 + 128 // 4
    if p.latent == 8:
        return c + 128 + 64 + 96 + 128
    raise NotImplementedError(p.latent)


def quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    """Upstream `quaternion_to_matrix`: `(x, y, z, w)` quaternions `[..., 4]` -> rotations `[..., 3, 3]`."""
    i, j, k, r = q.unbind(-1)
    two_s = 2 / ((q * q).sum(-1) + 1e-8)
    m = (
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    )  # fmt: skip
    return torch.stack(m, -1).unflatten(-1, (3, 3))


def covariance(scales: torch.Tensor, rot: torch.Tensor, r_c2w: torch.Tensor) -> torch.Tensor:
    """World covariances `[V, n, 3, 3]` of camera-frame Gaussians.

    Scales `[V, n, 3]`, unit `x y z w` rotations `[V, n, 4]`, camera rotations `[V, 3, 3]` (upstream
    `build_covariance` then `R_c2w C R_c2w^T`).
    """
    r = quaternion_matrix(rot) * scales[..., None, :]
    c = r @ r.transpose(-1, -2)
    rc = r_c2w[:, None]
    return rc @ c @ rc.transpose(-1, -2)


def _pixels(n_h: int, n_w: int, dev: torch.device) -> torch.Tensor:
    """Upstream `sample_image_grid`: pixel centres `[(h w), 2]` (x, y) in `[0, 1]`."""
    ys, xs = ((torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n for n in (n_h, n_w))
    return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)


def _rays(xy: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `get_world_rays`, z-normalised directions: `xy [V or 1, n, 2]` -> origins, directions `[V, n, 3]`."""
    d = invert(k.float())[:, None] @ torch.cat((xy, torch.ones_like(xy[..., :1])), -1)[..., None]
    d = d[..., 0] / d[..., 2:, 0]
    return c2w[:, None, :3, 3].float(), (c2w[:, None, :3, :3].float() @ d[..., None])[..., 0]


def _expand(t: torch.Tensor, hl: int, wl: int, r: int) -> torch.Tensor:
    """`b (h w) (c x y) -> b (h x w y) c` for `r x r` Gaussians per latent pixel."""
    v = t.shape[0]
    return t.view(v, hl, wl, -1, r, r).permute(0, 1, 4, 2, 5, 3).reshape(v, hl * r * wl * r, -1)


class Net(nn.Module):
    """`EncoderReSplat`'s modules at inference (same attribute names: the checkpoint keys after `encoder.`)."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, p: Params) -> None:  # noqa: D107
        super().__init__()
        self.p = p
        m = p.gaussians_per_point
        self.depth_predictor = MultiViewUniMatch(
            num_scales=1, upsample_factor=p.lowest, lowest_feature_resolution=p.lowest, num_depth_candidates=128,
            vit_type=p.vit, unet_channels=128, sample_log_depth=True, return_raw_mono_features=True,
        )  # fmt: skip
        feat = _feature_channels(p)
        img = 3 * (4**2 if p.fixed_latent else p.latent**2)
        ch = p.channels
        self.gaussian_regressor = nn.Sequential(
            nn.Conv2d(img + 2 + feat, ch, 3, 1, 1), nn.GELU(), nn.Conv2d(ch, ch, 3, 1, 1)
        )
        self.proj = nn.Linear(img + feat + ch + 1, ch)
        self.pt = PlainPointTransformer(ch, 16, num_blocks=6, attn_proj_channels=64, with_mv_attn=True)
        self.gaussian_head = nn.Sequential(nn.Linear(ch, RAW * m), nn.GELU(), nn.Linear(RAW * m, RAW * m))
        self.update_proj = nn.Conv2d(ch, STATE_CHANNELS, 1)
        n_state = STATE if p.same_points else STATE * m
        self.update_rgb_error_proj = nn.Sequential(
            nn.Linear(3 if m == 4 else 3 * p.latent**2, RESNET_CHANNELS), nn.LayerNorm(RESNET_CHANNELS)
        )
        self.update_module = nn.Sequential(
            PointLinearWrapper(n_state + STATE_CHANNELS + RESNET_CHANNELS, STATE_CHANNELS),
            PlainPointTransformer(STATE_CHANNELS, 8, num_blocks=4, attn_proj_channels=64),
        )
        s = STATE_CHANNELS
        self.update_head = nn.Sequential(
            nn.Linear(s, s), nn.GELU(), nn.Linear(s, s), nn.GELU(), nn.Linear(s, s), nn.GELU(), nn.Linear(s, n_state)
        )
        self.update_feature = ResNetFeatureWarpper()
        self.update_error_attn = nn.ModuleList([MultViewLowresAttn(RESNET_CHANNELS)])
        mean, std = (torch.tensor(t).view(3, 1, 1) for t in IMAGENET)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def encode(  # noqa: C901, PLR0912, PLR0915 - upstream EncoderReSplat.forward in one piece
        self, images: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor, near: float, far: float
    ) -> dict[str, torch.Tensor]:
        """Upstream `EncoderReSplat.forward`.

        Images `[V, 3, h, w]` in `[0, 1]`, normalised intrinsics `[V, 3, 3]`, camera-to-world `[V, 4, 4]` -> depth
        `[V, h, w]`, match probability `[V, 1, h', w']`, the initial Gaussians
        (view-major, camera-frame `x y z w` rotations; fp32) and the condition features.
        """
        p, (v, _, h, w), dev = self.p, images.shape, images.device
        nn_matrix = None
        if v > NEIGHBOURS + 1:
            xyz = c2w[None, :, :3, -1].float()
            dist = (xyz[:, :, None] - xyz[:, None]).norm(dim=-1)  # upstream torch.cdist (no ONNX conversion)
            nn_matrix = torch.argsort(dist)[:, :, : NEIGHBOURS + 1]
        bounds = torch.tensor((near, far), device=dev).view(2, 1, 1).expand(2, 1, v)
        res = self.depth_predictor(
            images[None], attn_splits_list=[2], min_depth=1.0 / bounds[1], max_depth=1.0 / bounds[0],
            intrinsics=k[None], extrinsics=c2w[None], nn_matrix=nn_matrix,
        )  # fmt: skip
        depth = res["depth_preds"][-1]  # [1, V, h, w]
        mono = [
            F.interpolate(x, (h // 16, w // 16), mode="bilinear", align_corners=True) for x in res["raw_mono_features"]
        ]
        if p.fixed_latent:
            mono = torch.cat([F.pixel_shuffle(x, 4) for x in mono], 1)
            if p.latent == 8:
                mono = F.interpolate(mono, scale_factor=0.5, mode="bilinear", align_corners=True)
        else:
            mono = torch.cat([F.pixel_shuffle(x, {4: 4, 2: 8, 8: 2}[p.latent]) for x in mono], 1)
        cnn = list(res["features_cnn_all_scales"][::-1])
        mv = res["features_mv"][0]
        if p.latent == 2:
            if p.lowest == 8:
                cnn[1], cnn[2], mv = F.pixel_shuffle(cnn[1], 2), F.pixel_shuffle(cnn[2], 4), F.pixel_shuffle(mv, 4)
            else:
                cnn[2], mv = F.pixel_shuffle(cnn[2], 2), F.pixel_shuffle(mv, 2)
        else:
            size = (h // p.latent, w // p.latent)
            cnn = [F.interpolate(c, size, mode="bilinear", align_corners=True) for c in cnn]
            if mv.shape[-2:] != size:
                mv = F.interpolate(mv, size, mode="bilinear", align_corners=True)
        features = torch.cat((mono, torch.cat(cnn, 1), mv), 1)
        prob = res["match_probs"][-1].max(dim=1, keepdim=True)[0]
        x = images.to(features.dtype)
        if p.fixed_latent:
            if p.latent == 8:
                x = F.interpolate(x, scale_factor=0.5, mode="area")
            x = F.pixel_unshuffle(x, 4)
        else:
            x = F.pixel_unshuffle(x, p.latent)
        latent_depth = F.interpolate(depth, scale_factor=1.0 / p.latent, mode="bilinear", align_corners=True)
        hl, wl = latent_depth.shape[-2:]
        lprob = F.interpolate(prob, (hl, wl), mode="nearest") if prob.shape[-2:] != (hl, wl) else prob
        out = self.gaussian_regressor(
            torch.cat((x, latent_depth[0, :, None].to(x.dtype), lprob.to(x.dtype), features), 1)
        )
        out = torch.cat((out, x, features, lprob.to(x.dtype)), 1)
        tokens = self.proj(out.permute(0, 2, 3, 1).reshape(v * hl * wl, -1))
        origins, dirs = _rays(_pixels(hl, wl, dev)[None], k, c2w)
        cloud = (origins + dirs * latent_depth[0].reshape(v, -1, 1).float()).reshape(-1, 3)
        out = tokens + self.pt((cloud, tokens, None), b=1, v=v, h=hl, w=wl)
        condition = out.view(v, hl, wl, -1).permute(0, 3, 1, 2)
        raw = self.gaussian_head(out).float().view(v, hl * wl, -1)

        m = p.gaussians_per_point
        rot, scales, opac, offset, sh = raw.split([4 * m, 3 * m, m, 2 * m, 3 * SH * m], -1)
        r = round(m**0.5)
        if m > 1:
            rot, scales, opac, offset, sh = (_expand(t, hl, wl, r) for t in (rot, scales, opac, offset, sh))
        lowres = {(4, 4): 0.5, (2, 4): 1.0, (4, 16): 1.0, (8, 4): 0.25, (8, 16): 0.5}  # (latent, m): depth scale
        sf = {(4, 4): 2, (2, 4): 2, (4, 16): 4, (8, 4): 2, (8, 16): 4}.get((p.latent, m), 1)
        if m > 1:
            f = lowres[(p.latent, m)]
            depths = depth if f == 1 else F.interpolate(depth, scale_factor=f, mode="bilinear", align_corners=True)
            rgb = images if f == 1 else F.interpolate(images, scale_factor=f, mode="area")
        else:
            depths = latent_depth
            rgb = F.interpolate(images, scale_factor=1.0 / p.latent, mode="area")
        gh, gw = hl * sf, wl * sf
        xy = _pixels(gh, gw, dev)[None] + (offset.sigmoid() - 0.5) / torch.tensor((gw, gh), device=dev)
        origins, dirs = _rays(xy, k, c2w)
        sh = sh.reshape(v, -1, 3, SH).clone()
        sh[..., 0] += (rgb.float().permute(0, 2, 3, 1).reshape(v, -1, 3) - 0.5) / C0
        return {
            "depth": depth[0],
            "confidence": prob,
            "condition": condition,
            "means": origins + dirs * depths[0].reshape(v, -1, 1).float(),
            "scales": F.softplus(scales - SCALE_BIAS).clamp(SCALE_MIN, SCALE_MAX),
            "rotations": rot,  # unnormalised
            "opacities": opac[..., 0].sigmoid(),
            "harmonics": sh,
        }

    def _render(self, g: dict[str, torch.Tensor], c2w: torch.Tensor, k: torch.Tensor, size: tuple[int, ...],
                near: float, far: float) -> torch.Tensor:  # fmt: skip
        rot = g["rotations"] / (g["rotations"].norm(dim=-1, keepdim=True) + 1e-8)
        cov = covariance(g["scales"], rot, c2w[:, :3, :3].float()).flatten(0, 1)
        return render(
            g["means"].flatten(0, 1), cov, g["opacities"].flatten(), g["harmonics"].flatten(0, 1), c2w, k, *size, near,
            far,
        )  # fmt: skip

    def _features(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        dt = next(self.update_feature.parameters()).dtype
        feats = self.update_feature(((x - self.mean) / self.std).to(dt))
        out = [
            f if f.shape[-2:] == size else F.interpolate(f, size, mode="bilinear", align_corners=True) for f in feats
        ]
        return torch.cat(out, 1)

    def refine(
        self, g: dict[str, torch.Tensor], images: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor, near: float,
        far: float, steps: int,
    ) -> dict[str, torch.Tensor]:  # fmt: skip
        """Upstream `forward_update`: `steps` recurrent updates of the Gaussians of `encode`."""
        p, (v, _, h, w) = self.p, images.shape
        m = p.gaussians_per_point
        dt = g["condition"].dtype
        state = self.update_proj(g["condition"])
        if p.same_points and m in (4, 16):
            state = F.interpolate(state, scale_factor=round(m**0.5), mode="bilinear", align_corners=True)
        state = state.permute(0, 2, 3, 1).reshape(-1, STATE_CHANNELS)
        hl, wl = h // p.latent, w // p.latent
        fh, fw = (hl * round(m**0.5), wl * round(m**0.5)) if p.same_points and m in (4, 16) else (hl, wl)
        means, scales, rot = g["means"], g["scales"], g["rotations"]
        opac, sh = torch.logit(g["opacities"], eps=1e-6)[..., None], g["harmonics"].flatten(2)
        cur = {
            "means": means,
            "scales": scales,
            "rotations": rot,
            "opacities": g["opacities"],
            "harmonics": g["harmonics"],
        }
        rendered = self._render(cur, c2w, k, (h, w), near, far)
        target = None
        for i in range(steps):
            f = self._features(rendered if target is not None else torch.cat((rendered, images.float())), (fh, fw))
            if target is None:
                f, target = f[:v], f[v:]
            err = (f - target).permute(0, 2, 3, 1).reshape(1, -1, RESNET_CHANNELS)
            rgb = rendered - images.float()
            if m != 4:
                rgb = F.pixel_unshuffle(rgb, p.latent)
            err = err + self.update_rgb_error_proj(rgb.permute(0, 2, 3, 1).reshape(1, -1, rgb.shape[1]).to(dt))
            prev = torch.cat((means, scales, rot, opac, sh), -1)  # [V, n, STATE]
            cloud = means
            if m > 1 and not p.same_points:
                r = round(m**0.5)
                cloud = means.view(v, hl, r, wl, r, 3)[:, :, 0, :, 0]
                prev = prev.view(v, hl, r, wl, r, STATE).permute(0, 1, 3, 5, 2, 4)
            cloud = cloud.reshape(-1, 3)
            prev = prev.reshape(cloud.shape[0], -1)
            for blk in self.update_error_attn:
                err = blk(err, v=v, h=fh, w=fw)
            x = torch.cat((prev.to(dt), state, err[0].to(dt)), -1)
            pxo = self.update_module[0]([cloud, x, None])
            state = self.update_module[1](pxo) + state
            delta = self.update_head(state).float()
            if m > 1 and not p.same_points:
                delta = delta.view(v, hl * wl, STATE, m).transpose(-1, -2).reshape(v, -1, STATE)  # c k -> (n k) c
            else:
                delta = delta.view(v, -1, STATE)
            dm, ds, dr, do, dsh = delta.split((3, 3, 4, 1, 3 * SH), -1)
            means, scales = means + dm, (scales + ds).clamp(min=SCALE_MIN)
            rot, opac, sh = rot + dr, opac + do, sh + dsh
            cur = {
                "means": means, "scales": scales, "rotations": rot, "opacities": opac[..., 0].sigmoid(),
                "harmonics": sh.view(v, -1, 3, SH),
            }  # fmt: skip
            if i < steps - 1:  # upstream also renders after the last step (unused at inference)
                rendered = self._render(cur, c2w, k, (h, w), near, far)
        return cur

    def forward(
        self, images: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor, near: float, far: float, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Depth `[V, h, w]`, match probability `[V, 1, h', w']` and the Gaussians (view-major `[V, n, ...]`)."""
        g = self.encode(images, k, c2w, near, far)
        out = {n: g[n] for n in ("means", "scales", "rotations", "opacities", "harmonics")}
        if steps:
            out = self.refine(g, images, k, c2w, near, far, steps)
        return g["depth"], g["confidence"], out


def _net(ctx: Context) -> Net:
    sd = {k.removeprefix("encoder."): v for k, v in ctx.state_dict(key="state_dict").items()}
    net = Net(ctx.params)
    net.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


def _directions() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed unit directions `S` `[N, 3]` and `pinv(B(S))` `[N, SH]` of gsplat's SH basis (float64)."""
    s = np.random.default_rng(0).standard_normal((64, 3))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    b = sh_basis(torch.from_numpy(s)).numpy()  # [N, SH]
    return torch.from_numpy(s), torch.from_numpy(np.linalg.pinv(b.T))


_S, _P = _directions()  # constants (torch.compile / ONNX export see plain tensors)
_DEGREES = ((0, 1), (1, 4), (4, 9), (9, 16))


def sh_rotation(r: torch.Tensor) -> torch.Tensor:
    """`[SH, SH]` matrix `D` with `B(R^T d) = D B(d)` for gsplat's basis `B`.

    Coefficients `c @ D` evaluated at `R d`
    give the colour of `c` at `d` (block-diagonal per degree; float64).
    """
    s, pinv = _S.to(r), _P.to(r)
    full = sh_basis(s @ r).T @ pinv  # rows of s @ r are R^T s_i
    d = torch.zeros_like(full)
    for a, b in _DEGREES:
        d[a:b, a:b] = full[a:b, a:b]
    return d


class _Base(Runner):
    intrinsics = "required"
    cam_to_world = "required"
    depth_range = "optional"
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
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """The runner's inputs -> depth, confidence (input size) and the refined Gaussians, in the input frame."""
        p = self.ctx.params
        n, _, h0, w0 = images.shape
        hn, wn = network_size(h0, w0, p)
        x = images.float()
        if (hn, wn) != (h0, w0):
            x = F.interpolate(x, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        k = normalized_intrinsics(intrinsics, h0, w0)
        c2w = f64(cam_to_world)
        mid = c2w[n // 2]
        net_c2w = (invert(mid)[None] @ c2w).float()  # upstream pose_align_middle_view
        near, far = (p.near, p.far) if depth_range is None else (float(depth_range[0]), float(depth_range[1]))
        with self._amp():
            depth, prob, g = self.net(x, k, net_c2w, near, far, p.refine)

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)[:, 0]

        rm, tm = mid[:3, :3], mid[:3, 3]
        means = (f64(g["means"]) @ rm.T + tm).float()
        rot = g["rotations"] / (g["rotations"].norm(dim=-1, keepdim=True) + 1e-8)  # x y z w, camera frame
        q = quaternion_product(matrix_quaternion(c2w[:, :3, :3])[:, None], f64(rot[..., [3, 0, 1, 2]]))
        q = (q * torch.where(q[..., :1] < 0, -1.0, 1.0)).float()
        harmonics = g["harmonics"] @ sh_rotation(rm).float()
        return {
            "depth": up(depth[:, None]),
            "confidence": up(prob),
            "gaussians/means": means.reshape(-1, 3),
            "gaussians/scales": g["scales"].reshape(-1, 3),
            "gaussians/rotations": q.reshape(-1, 4),
            "gaussians/harmonics": harmonics.reshape(-1, 3, SH),
            "gaussians/opacities": g["opacities"].reshape(-1),
        }

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth and confidence of every view and the scene's Gaussians, in the input frame."""
        if intrinsics is None or cam_to_world is None:  # the core checks required inputs
            msg = "ReSplat needs intrinsics and cam_to_world"
            raise RuntimeError(msg)
        return self.forward(images, intrinsics, cam_to_world, depth_range)


class Portable(_Base):
    """Plain PyTorch: float32 = upstream modules; float16 (CUDA) = fp16 weights under autocast."""

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps and Gaussians of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class Kernels(_Base):
    """fp16: the `fastbb` DINOv2 (flash attention, fp16-accumulate GEMMs), cuDNN fp16-accumulate convs."""

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        cudnn_hacc.load(ctx.device)
        use_fast_vit(self.net.depth_predictor.pretrained)
        if _hacc_convs(self.net) == 0:
            msg = "ReSplat kernels: no conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """`_Base.forward` with fp16-accumulate GEMMs."""
        with fp16_accumulate():
            return super().forward(images, intrinsics, cam_to_world, depth_range)


RUNNERS = {"portable": Portable, "kernels": Kernels}
