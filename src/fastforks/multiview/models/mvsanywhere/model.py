"""MVSAnywhere runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `MVSAnywhereInference.forward` in eval mode (one reference, the other views as sources): the
images are resized to the network size (long side `long_side`, both sides multiples of 16, bilinear, antialiased) and
normalised with the ImageNet mean / std; the intrinsics are shifted to MVSAnywhere's corner-origin pixel convention
(`+0.5`, as its dataloaders) and scaled with the image. Without a depth range the upstream heuristic (from the camera
baselines) sets the 64 log-spaced depth planes; `refinement_steps` further passes refine the range from the previous
depth's quantiles. Depth and sky probability are upsampled back to the input size (bilinear); the mask is "not sky"
(probability < 0.5), as upstream's `depth_masked`.

The forward pass is ours (`fast.py`), with upstream's maths and weights: the ViTs, the conv nets and the
heads run as upstream modules; the cost volume evaluates the per-sample MLP for all views and a chunk of depth planes
at once (biases folded into the GEMMs), instead of upstream's per-plane loop. Only the scale-0 depth and sky heads run
(the others are not used at inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import Inert, load_strict
from fastforks._core.half import fp16_accumulate
from fastforks.kernels import dpt_fused
from fastforks.multiview._runner import Forward, Runner, f64, invert, scale, take
from fastforks.shared.depth_anything_v2 import DINOv2, FastViT, use_sdpa_attention
from fastforks.shared.quantile import quantile

from .net.modules import MLP, BlurPool, CostVolumePatchEmbed, DPTHead, ResnetMatchingEncoder

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable, Sequence

    from fastforks.multiview._runner import Context

IMG_LAYERS = (0, 2, 4, 5, 8, 11)  # fixed: upstream image-encoder layers (2 for the patch embed, 4 for the fusers)
CV_LAYERS = (2, 5, 8, 11)  # fixed: upstream cost-volume ViT fuser / output layers
PLANES = 64  # fixed: upstream matching_num_depth_bins
MATCH = 4  # fixed: matching features at 1/4 of the network size
MULTIPLE = 16  # fixed: the network size is a multiple of the ViT stride (16 after the 14/16 resize)
CHUNK = 4  # fixed: depth planes per cost-volume step (fastest at 448x672)
MAX_SOURCES = 16  # fixed: the cost-volume kernel keeps the per-view terms of at most 16 sources in registers
_PREFIXES = {  # our module -> checkpoint prefix
    "vit.": "encoder.model.",
    "cv_vit.": "cost_volume_net.model.",
    "cv_fusers.": "cost_volume_net.cv_feat_fusers.",
    "cv_patch_embed.": "cost_volume_net.patch_embed.",
    "dpt.": "depth_decoder.",
    "mlp.": "cost_volume.mlp.",
    "matching.": "matching_model.",
}
# checkpoint tensors upstream inference never reads: the lower-resolution depth / sky outputs (only scale 0 is
# returned) and the constant buffers of the projection, normal and loss helpers
_UNUSED = (
    *(f"dpt.{h}.{i}." for h in ("output_convs", "sky_convs") for i in (1, 2, 3)),
    "compute_normals.",
    "cost_volume.backprojector.",
    "cost_volume.linear_ramp_1d11",
    "cost_volume.projector.",
    "mv_depth_loss.",
)
# layer-1 inputs of the cost-volume MLP that vary per view and plane (warped features, mask, depth, dot, angle,
# source ray); the kernel's MlpWeights order
_DYN = [*range(16), 32, 33, 35, 36, 40, 41, 42]
_DEPTH_COL = 34  # fixed: layer-1 input of the normalised plane depth
_PIXEL_COLS, _RAY_COLS, _POSE_COLS = slice(16, 32), slice(37, 40), slice(43, 46)  # fixed: upstream input layout
_REFINE = (0.05, 0.95, 0.75, 1.25)  # fixed: upstream refine_depth_range quantiles and margins
_SKY = 0.5  # fixed: upstream sky threshold of depth_masked


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (both sides rounded to multiples of 16)
    refinement_steps: int  # depth-range refinement passes (upstream num_refinement_steps)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: long side `long_side`, both sides multiples of 16."""
    s = long_side / max(h, w)
    return (max(MULTIPLE, round(h * s / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(w * s / MULTIPLE) * MULTIPLE))


class Net(nn.Module):
    """The inference modules of upstream `DepthModelCVHint` (names as in `FastMVSAnywhere`)."""

    def __init__(self) -> None:
        """Build the modules (random init; `load` fills them)."""
        super().__init__()
        self.vit = DINOv2("vitb")
        self.cv_vit = DINOv2("vitb")
        self.cv_fusers = nn.ModuleList([nn.Sequential(nn.Linear(768, 768), nn.ReLU(inplace=True)) for _ in CV_LAYERS])
        self.cv_patch_embed = CostVolumePatchEmbed(PLANES, 768)
        self.dpt = DPTHead()
        self.mlp = MLP([46, 32, 32, 2], disable_final_activation=True)
        self.matching = ResnetMatchingEncoder(16)

    def load(self, ctx: Context) -> None:
        """Copy the weights from the Lightning checkpoint (its pickled options are skipped)."""
        sd = ctx.state_dict(
            key="state_dict",
            strip="",
            extra={("mvsanywhere.options", "Options"): Inert, ("mvsanywhere.options", "DataOptions"): Inert},
        )
        theirs = {v: k for k, v in _PREFIXES.items()}
        own = {}
        for k, t in sd.items():
            prefix = next((p for p in theirs if k.startswith(p)), None)
            own[theirs[prefix] + k[len(prefix) :] if prefix else k] = t
        load_strict(self, own, "MVSAnywhere checkpoint", unused=_UNUSED)


def _net(ctx: Context) -> Net:
    net = Net()
    net.load(ctx)
    return net.eval().requires_grad_(requires_grad=False)


def _pose_distance(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream `pose_distance`: combined, rotation and translation distance of `[K, 4, 4]` relative poses."""
    r = torch.sqrt(2 * (1 - torch.clamp(pose[:, :3, :3].diagonal(dim1=-1, dim2=-2).sum(-1), max=3.0) / 3))
    tm = torch.norm(pose[:, :3, 3], dim=1)
    return torch.sqrt(tm**2 + r**2), r, tm


def _unit_max(x: torch.Tensor) -> torch.Tensor:
    return x / torch.where((x == 0).all(), torch.ones_like(x.max()), x.max())


@dataclass
class Geometry:
    """Depth-range-independent cost-volume inputs at the matching resolution."""

    proj: torch.Tensor  # [K, 3, 4] source projection of reference-camera points (K_src @ src_T_cur)
    ray: torch.Tensor  # [3, hw] reference rays (z = 1)
    cray: torch.Tensor  # [3, hw] unit reference rays
    pose: torch.Tensor  # [K, 3] normalised pose distance, rotation, normalised translation
    cams: torch.Tensor  # [K, 3] source camera centres in the reference frame
    rng: torch.Tensor  # [2] upstream heuristic depth range


def network_intrinsics(k: torch.Tensor, size: Sequence[int], net: Sequence[int]) -> torch.Tensor:
    """OpenCV intrinsics `[N, 3, 3]` of `size` images -> MVSAnywhere's corner-origin pixels (`+0.5`) at `net`."""
    k = k.float().clone()
    k[:, :2, 2] += 0.5  # MVSAnywhere's pixel grid: corners at integers
    k[:, 0] *= net[1] / size[1]
    k[:, 1] *= net[0] / size[0]
    return k


def geometry(k: torch.Tensor, c2w: torch.Tensor, h: int, w: int) -> Geometry:
    """Cost-volume geometry from network-size intrinsics `[N, 3, 3]` and poses `[N, 4, 4]` (reference first)."""
    km = torch.eye(4, device=k.device).repeat(k.shape[0], 1, 1)
    km[:, :3, :3] = k
    km[:, :2] /= MATCH
    c = f64(c2w)  # relative poses first: independent of the world origin, then float32
    src_t_cur = (invert(c[1:]) @ c[0]).float()
    cur_t_src = (invert(c[0]) @ c[1:]).float()
    yy, xx = torch.meshgrid(torch.arange(h, device=k.device), torch.arange(w, device=k.device), indexing="ij")
    pix = torch.stack([xx.flatten() + 0.5, yy.flatten() + 0.5, torch.ones(h * w, device=k.device)])
    ray = invert(km[0])[:3, :3] @ pix
    fp, r, tm = _pose_distance(cur_t_src)
    dist = src_t_cur[:, :3, 3].norm(dim=-1)
    fx_over_w = km[1, 0, 0] / w  # upstream get_heuristic_depth_range: first source's fx at the image size
    return Geometry(
        proj=(km[1:] @ src_t_cur)[:, :3],
        ray=ray,
        cray=F.normalize(ray, dim=0),
        pose=torch.stack([_unit_max(fp), r, _unit_max(tm)], 1),
        cams=cur_t_src[:, :3, 3],
        rng=torch.stack([dist.min() * fx_over_w, dist.max() * fx_over_w / 0.01]),
    )


def planes(rng: torch.Tensor) -> torch.Tensor:
    """64 log-spaced depth planes over `rng`."""
    ramp = torch.linspace(0, 1, PLANES, device=rng.device)
    return torch.exp(torch.log(rng[0]) + torch.log(rng[1] / rng[0]) * ramp)


def refine_range(depth: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    """Upstream `refine_depth_range`, without host syncs."""
    mn, mx = rng[0], rng[1]
    n = depth.numel()
    lo, hi, shrink, grow = _REFINE
    frac_max = (depth > mx * 0.95).sum() / n
    frac_min = (depth < mn * 1.05).sum() / n
    flat = depth.flatten().float()
    if torch.compiler.is_exporting():  # fastforks: TensorRT's TopK (the exported sort) takes at most 3840 elements
        q = torch.stack([quantile(flat, lo), quantile(flat, hi)])
    else:
        q = torch.quantile(flat, torch.tensor([lo, hi], device=depth.device))
    new_min = torch.where(frac_min > lo, mn * 0.5, torch.maximum(mn, q[0] * shrink))
    new_max = torch.where(frac_max > lo, mx * 2.0, torch.minimum(mx, q[1] * grow))
    return torch.stack([new_min, new_max])


def linears(mlp: nn.Module) -> list[nn.Linear]:
    """The 3 layers of the cost-volume MLP."""
    return [m for m in mlp.modules() if isinstance(m, nn.Linear)]


def mlp_weights(mlp: nn.Module) -> tuple[torch.Tensor, ...]:
    """The cost-volume MLP with the biases folded in (fp32): `[33, 48]`, `[33, 33]`, `[2, 33]`.

    The input gets a constant-1 channel (+1 zero pad -> 48), each hidden layer a constant-1 unit (leaky(1) = 1), so
    the three layers are three plain matmuls.
    """
    (w1, b1), (w2, b2), (w3, b3) = ((m.weight.float(), m.bias.float()) for m in linears(mlp))
    a1 = torch.zeros(33, 48, device=w1.device)
    a1[:32, :46], a1[:32, 46], a1[32, 46] = w1, b1, 1
    a2 = torch.zeros(33, 33, device=w1.device)
    a2[:32, :32], a2[:32, 32], a2[32, 32] = w2, b2, 1
    return a1, a2, torch.cat([w3, b3[:, None]], 1)


def cost_volume(
    cur: torch.Tensor, src: torch.Tensor, g: Geometry, rng: torch.Tensor, weights: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Upstream's MLP cost volume in plain PyTorch, in the dtype of `weights`: `[1, 64, h, w]`.

    Per chunk of depth planes, all views at once: warp the source features (`grid_sample`), build the 46 MLP inputs
    in upstream's channel order, 3 matmuls, then the softmax over the views of the second output weights the first.
    """
    a1, a2, a3 = weights
    dt = a1.dtype
    k, c, h, w = src.shape
    hw = h * w
    mn, mx = rng[0], rng[1]
    srcd = src.to(dt).contiguous()
    curd = cur[0].to(dt).reshape(1, 1, c, hw)
    a = g.proj[:, :, :3] @ g.ray  # [K, 3, hw]
    t = g.proj[:, :, 3]
    crayd = g.cray.to(dt).view(1, 1, 3, hw)
    posed = g.pose.to(dt).view(k, 1, 3, 1)
    to_grid = torch.tensor([2.0 / w, 2.0 / h], device=src.device).view(1, 1, 2, 1)
    deps = planes(rng)
    outs = []
    for d0 in range(0, PLANES, CHUNK):
        dd = deps[d0 : d0 + CHUNK]
        d = dd.shape[0]
        xyz = dd.view(1, d, 1, 1) * a[:, None] + t.view(k, 1, 3, 1)  # [K, D, 3, hw]
        z = xyz[:, :, 2]
        zd = z + 1e-8
        uv = xyz[:, :, :2] * torch.where(z.abs() > 1e-8, 1.0 / zd, torch.ones_like(zd))[:, :, None]  # noqa: PLR2004
        grid = (uv * to_grid - 1).permute(0, 1, 3, 2).reshape(k, d * h, w, 2).to(dt)
        warped = F.grid_sample(srcd, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        warped = warped.view(k, c, d, hw).transpose(1, 2)  # [K, D, C, hw]
        mask = (zd > 0).to(dt)
        dot = (warped * curd).sum(2, dtype=torch.float32).to(dt) * mask
        sray = F.normalize(dd.view(1, d, 1, 1) * g.ray[None, None] - g.cams.view(k, 1, 3, 1), dim=2)
        ang = (sray * g.cray[None, None]).sum(2)
        dn = (zd - mn) / mx
        pn = ((dd - mn) / mx).to(dt).view(1, d, 1, 1).expand(k, d, 1, hw)
        x = torch.cat(
            [
                warped,
                curd.expand(k, d, c, hw),
                mask[:, :, None],
                dn.to(dt)[:, :, None],
                pn,
                dot[:, :, None],
                ang.to(dt)[:, :, None],
                crayd.expand(k, d, 3, hw),
                sray.to(dt),
                posed.expand(k, d, 3, hw),
                torch.ones_like(mask)[:, :, None],
                torch.zeros_like(mask)[:, :, None],
            ],
            2,
        )  # [K, D, 48, hw]
        o = F.leaky_relu_(torch.matmul(a1, x), 0.01)  # row 32 == 1
        o = F.leaky_relu_(torch.matmul(a2, o), 0.01)
        o = torch.matmul(a3, o).float()  # [K, D, 2, hw]
        outs.append((torch.softmax(o[:, :, 1], dim=0) * o[:, :, 0]).sum(0))
    return torch.cat(outs, 0).view(1, PLANES, h, w)


def _log_depth(logit: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    return torch.log(rng[0]) + torch.log(rng[1] / rng[0]) * torch.sigmoid(logit)


class _Base(Runner):
    """Shared forward: matching features, image features, cost volume passes, decoding, output resizing."""

    intrinsics = "required"
    cam_to_world = "required"
    depth_range = "optional"
    per_view = False
    min_views = 2
    batch = None
    Params = Params
    compiled_methods = ("__call__", "prepare", "predict")  # `scene` (several references) calls the last two

    net: Net

    def encode_image(self, x: torch.Tensor) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        """DINOv2 features (patch tokens, cls) of the reference image at the `IMG_LAYERS`."""
        raise NotImplementedError

    def matching(self, x: torch.Tensor) -> torch.Tensor:
        """`[N, 3, H, W]` normalised images -> matching features `[N, 16, H/4, W/4]`."""
        raise NotImplementedError

    def cost_volume(self, cur: torch.Tensor, src: torch.Tensor, g: Geometry, rng: torch.Tensor) -> torch.Tensor:
        """`[1, 64, h, w]` matching costs."""
        raise NotImplementedError

    def cv_tokens(self, cv: torch.Tensor, feats: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Cost volume -> embedded tokens with cls and positions (upstream `prepare_tokens_with_masks`)."""
        raise NotImplementedError

    def cv_block(self, i: int, t: torch.Tensor) -> torch.Tensor:
        """Block `i` of the cost-volume ViT."""
        raise NotImplementedError

    def decode(self, feats: list[torch.Tensor], ph: int, pw: int, *, sky: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """DPT head: depth logits `[1, 1, H, W]` float32 and sky logits (the depth logits if not `sky`)."""
        logit, sky_l = self.net.dpt(feats, ph, pw, sky=sky)
        return logit.float(), logit.float() if sky_l is None else sky_l.float()

    def cv_encoder(self, cv: torch.Tensor, feats: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        """Upstream `ViTCVEncoder.forward`: the patch tokens of the `CV_LAYERS` (not normed)."""
        t = self.cv_tokens(cv, feats)
        out = []
        for i in range(len(self.net.cv_vit.blocks)):
            if i in CV_LAYERS:
                j = CV_LAYERS.index(i)
                pt, cls = feats[2 + j]
                t = t + self.net.cv_fusers[j](torch.cat([cls.unsqueeze(1), pt], 1).to(t.dtype))
            t = self.cv_block(i, t)
            if i in CV_LAYERS:
                out.append(t[:, 1:])
            if i == CV_LAYERS[-1]:
                break
        return out

    def prepare(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-image work of a scene: the normalised network-size images and their matching features."""
        hn, wn = network_size(images.shape[-2], images.shape[-1], self.ctx.params.long_side)
        x = scale(F.interpolate(images, (hn, wn), mode="bilinear", align_corners=False, antialias=True), self.ctx.io)
        return x, self.matching(x)

    def predict(  # noqa: PLR0913, PLR0917 - the prepared scene and the cameras
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        size: tuple[int, int],
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        depth_range: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepared reference `x` `[1, 3, hn, wn]`, matching features `m` of all views (reference first) -> maps."""
        hn, wn = x.shape[-2:]
        k = network_intrinsics(intrinsics, size, (hn, wn))
        cur, src = m[:1], m[1:]
        h, w = m.shape[-2:]
        g = geometry(k, cam_to_world, h, w)
        rng = g.rng if depth_range is None else depth_range.float()
        feats = self.encode_image(x)
        ph, pw = hn // MULTIPLE, wn // MULTIPLE
        for _ in range(self.ctx.params.refinement_steps):
            logit, _sky = self.decode(self.cv_encoder(self.cost_volume(cur, src, g, rng), feats), ph, pw, sky=False)
            rng = refine_range(torch.exp(_log_depth(logit, rng)), rng)
        logit, sky_l = self.decode(self.cv_encoder(self.cost_volume(cur, src, g, rng), feats), ph, pw, sky=True)
        log_depth = _log_depth(logit, rng)
        depth = F.interpolate(torch.exp(log_depth), size, mode="bilinear", align_corners=False)[:, 0]
        sky = F.interpolate(torch.sigmoid(sky_l), size, mode="bilinear", align_corners=False)[:, 0]
        return {"depth": depth, "mask": sky < _SKY, "sky": sky, "depth_range": rng[None]}

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        depth_range: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One reference (`images[0]`) with the other views as sources -> maps at the input size."""
        x, m = self.prepare(images)
        return self.predict(x[:1], m, (images.shape[-2], images.shape[-1]), intrinsics, cam_to_world, depth_range)

    @staticmethod
    def _required(
        intrinsics: torch.Tensor | None, cam_to_world: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intrinsics is None or cam_to_world is None:  # the core checks required inputs
            msg = "MVSAnywhere needs intrinsics and cam_to_world"
            raise ValueError(msg)
        return intrinsics, cam_to_world

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth of the reference view (`images[0]`)."""
        k, c2w = self._required(intrinsics, cam_to_world)
        with torch.no_grad():
            return self.forward(images, k, c2w, depth_range)

    def scene(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> Callable[[torch.Tensor | None], dict[str, torch.Tensor]]:
        """See `Runner.scene`: the matching features of every view once, the rest per reference."""
        k0, c2w0 = self._required(intrinsics, cam_to_world)
        size = (images.shape[-2], images.shape[-1])
        with torch.no_grad():
            x, m = self.prepare(images)

        def predict(order: torch.Tensor | None) -> dict[str, torch.Tensor]:
            xo, mo, k, c2w = x[:1], m, k0, c2w0
            if order is not None:
                xo, mo, k, c2w = take(x, order[:1]), take(m, order), k0[order], c2w0[order]
            with torch.no_grad():
                return self.predict(xo, mo, size, k, c2w, depth_range)

        return predict


class Portable(_Base):
    """Plain PyTorch: float32, or fp16 weights + autocast (fp16 cost volume, channels-last convs), SDPA."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.device)
        use_sdpa_attention(self.net)
        self.half = ctx.dtype == "float16"
        a = mlp_weights(self.net.mlp)  # from the fp32 weights
        self.weights = tuple(w.half() for w in a) if self.half else a
        if self.half:  # fp16 weights (as friendly): half the memory, no per-call casts under autocast
            self.net.half().to(memory_format=torch.channels_last)

    def _amp(self) -> contextlib.AbstractContextManager[Any]:
        return torch.autocast(self.ctx.device.type, dtype=torch.float16, enabled=self.half)

    def encode_image(self, x: torch.Tensor) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        """Upstream `DINOv2.forward`: 14/16 resize, normed intermediate layers with cls tokens."""
        x = F.interpolate(x, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        with self._amp():
            return self.net.vit.get_intermediate_layers(x, list(IMG_LAYERS), return_class_token=True)

    def matching(self, x: torch.Tensor) -> torch.Tensor:
        """Matching encoder on all views at once (instance norm is per image)."""
        with self._amp():
            return self.net.matching(x)

    def cost_volume(self, cur: torch.Tensor, src: torch.Tensor, g: Geometry, rng: torch.Tensor) -> torch.Tensor:
        """Plain-PyTorch cost volume (`cost_volume`)."""
        return cost_volume(cur, src, g, rng, self.weights)

    def cv_tokens(self, cv: torch.Tensor, feats: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Conv patch embedding (with the image features of the first two layers), cls token, positions."""
        vit = self.net.cv_vit
        _, _, h, w = cv.shape
        with self._amp():
            t = self.net.cv_patch_embed(cv, feats[:2])
            t = torch.cat([vit.cls_token.expand(t.shape[0], -1, -1).to(t.dtype), t], 1)
            return t + vit.interpolate_pos_encoding(t, h * MATCH * 14 / 16, w * MATCH * 14 / 16)  # as upstream

    def cv_block(self, i: int, t: torch.Tensor) -> torch.Tensor:
        """Upstream block."""
        with self._amp():
            return self.net.cv_vit.blocks[i](t)

    def cv_encoder(self, cv: torch.Tensor, feats: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        """See `_Base.cv_encoder` (under autocast)."""
        with self._amp():
            return super().cv_encoder(cv, feats)

    def decode(self, feats: list[torch.Tensor], ph: int, pw: int, *, sky: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """See `_Base.decode` (under autocast)."""
        with self._amp():
            return super().decode(feats, ph, pw, sky=sky)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


_fp16_accumulate = fp16_accumulate


class _StemPool(nn.Module):
    """ReLU + MaxPool(2, 1) + ReflectionPad(1, 2, 1, 2) + BlurPool(4, stride 2) on channels-last fp16, one kernel."""

    def __init__(self, kern: Any) -> None:  # noqa: ANN401 - CubinModule
        super().__init__()
        self.kern = kern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool `[N, C, H, W]` (C even) to `[N, C, (H - 2) // 2 + 1, (W - 2) // 2 + 1]`."""
        x = x.contiguous(memory_format=torch.channels_last)
        n, c, h, w = x.shape
        ho, wo = (h - 2) // 2 + 1, (w - 2) // 2 + 1
        out = torch.empty((n, c, ho, wo), device=x.device, dtype=torch.half, memory_format=torch.channels_last)
        total = n * ho * wo * c // 2
        self.kern.launch("mvsa_stem_pool", ((total + 255) // 256, 1, 1), (256, 1, 1), x, out, n, h, w, c // 2, ho, wo)
        return out


class Kernels(_Base):
    """Kernels path: fp16-accumulate convs and GEMMs, flash attention, fused cost-volume / stem kernels.

    Both ViTs run through the fastbb flash-attention kernel (fp32 `q k^T`) with fp16-accumulate GEMMs; the conv nets
    (matching encoder, cost-volume patch embedding, DPT head) through `cnn_kernels` (fp16 channels-last, fp16-accumulate
    cuDNN / CUTLASS convs, fused BatchNorm and instance-norm kernels). The `mvsanywhere` kernels compute the whole
    cost volume per pixel and plane (warp, MLP, softmax over the views, at most 16 sources) and the matching encoder's
    stem pooling.
    """

    max_views = MAX_SOURCES + 1
    w1: torch.Tensor
    b1: torch.Tensor

    def __init__(self, ctx: Context) -> None:
        """Build the network, convert it, move it to the device."""
        from fastforks.kernels.mvsanywhere import MLP_FLOATS  # noqa: PLC0415 - kernel lib only in kernels
        from fastforks.kernels.mvsanywhere import load as load_kernels  # noqa: PLC0415
        from fastforks.shared.cnn_kernels import prepare_generic  # noqa: PLC0415

        super().__init__(ctx)
        self.kern = load_kernels(ctx.device)
        net = _net(ctx)
        l1, l2, l3 = linears(net.mlp)
        w1 = l1.weight.float()
        packed = torch.cat(
            [w1[:, _DYN].flatten(), w1[:, _DEPTH_COL], l2.weight.flatten(), l2.bias, l3.weight.flatten(), l3.bias]
        ).float()
        if packed.numel() != MLP_FLOATS:
            msg = f"MVSAnywhere cost-volume MLP: {packed.numel()} weights, the kernel takes {MLP_FLOATS}"
            raise ValueError(msg)
        self.mlp_args = packed.tolist()
        self.w1 = w1.to(ctx.device)
        self.b1 = l1.bias.float().to(ctx.device)
        self.net = net.to(ctx.device)
        fused = dpt_fused.load(ctx.device)
        self.fv = FastViT(net.vit, "fh", None, fused=fused)
        self.fcv = FastViT(net.cv_vit, "fh", None, fused=fused)
        net.cv_fusers.half()
        for mod in (net.matching, net.cv_patch_embed, net.dpt):
            prepare_generic(mod, ctx.device)
        seq = net.matching.net  # conv-BN-ReLU fused into seq[0]; the stem pooling replaces the pool + blur
        blur = next(m for m in net.matching.modules() if isinstance(m, BlurPool))
        a = torch.tensor([1.0, 3.0, 3.0, 1.0])
        if not (
            isinstance(seq, nn.Sequential)
            and isinstance(seq[2], nn.Identity)
            and blur.stride == 2  # noqa: PLR2004
            and torch.equal(torch.as_tensor(blur.filt)[0, 0].float().cpu(), torch.outer(a, a) / 64)
        ):
            msg = "MVSAnywhere matching stem does not have the layout of the stem-pool kernel"
            raise ValueError(msg)
        seq[3] = _StemPool(self.kern)

    def encode_image(self, x: torch.Tensor) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        """Upstream `DINOv2.forward` on the fast ViT."""
        x = F.interpolate(x, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        return self.fv.get_intermediate_layers(x, list(IMG_LAYERS), return_class_token=True)

    def matching(self, x: torch.Tensor) -> torch.Tensor:
        """Matching encoder (fp16 channels-last)."""
        return self.net.matching(x.half().contiguous(memory_format=torch.channels_last))

    def cost_volume(self, cur: torch.Tensor, src: torch.Tensor, g: Geometry, rng: torch.Tensor) -> torch.Tensor:
        """The fused kernel: per-pixel / per-view layer-1 terms precomputed, everything else per sample."""
        k, c, h, w = src.shape
        hw = h * w
        curf = cur[0].float().reshape(c, hw).contiguous()
        a = (self.w1[:, _PIXEL_COLS] @ curf + self.w1[:, _RAY_COLS] @ g.cray + self.b1[:, None]).contiguous()
        bk = (g.pose @ self.w1[:, _POSE_COLS].t()).contiguous()
        geo = torch.cat([g.proj.reshape(-1, 12), g.cams, torch.zeros_like(g.cams[:, :1])], 1).contiguous()
        srcl = src.permute(0, 2, 3, 1).contiguous()  # [K, h, w, 16] fp16
        deps = planes(rng).contiguous()
        out = torch.empty((PLANES, hw), device=src.device, dtype=torch.float32)
        self.kern.launch(
            "mvsa_cv_fused", ((hw + 127) // 128, PLANES, 1), (128, 1, 1),
            g.ray.contiguous(), a, curf, srcl, geo, bk, deps, rng.float().contiguous(), out, k, h, w, *self.mlp_args,
        )  # fmt: skip
        return out.view(1, PLANES, h, w)

    def cv_tokens(self, cv: torch.Tensor, feats: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Conv patch embedding (fp16 channels-last), cls token, cached positions."""
        _, _, h, w = cv.shape
        t = self.net.cv_patch_embed(cv.half().contiguous(memory_format=torch.channels_last), feats[:2])
        t = torch.cat([self.fcv.vit.cls_token.expand(t.shape[0], -1, -1), t.half()], 1)
        return t + self.fcv.pos_embed(h // MATCH, w // MATCH)

    def cv_block(self, i: int, t: torch.Tensor) -> torch.Tensor:
        """Fast block (call under `cv_encoder`'s fp16-accumulate context)."""
        return self.fcv.block(self.fcv.blocks[i], t)

    def cv_encoder(self, cv: torch.Tensor, feats: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        """See `_Base.cv_encoder` (fp16, fp16-accumulate GEMMs)."""
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            return super().cv_encoder(cv, feats)


RUNNERS = {"portable": Portable, "kernels": Kernels}
