"""PatchmatchNet runners: weights -> folded network and the portable / kernels forward passes.

Both follow upstream `PatchmatchNet.forward` in eval mode with the upstream evaluation settings (`eval.py` defaults:
three PatchMatch stages at 1/8, 1/4, 1/2 resolution, then the refinement net at full resolution). Images are resized
to multiples of 8 if needed (bilinear, intrinsics scaled, as upstream `adjust_image_dims`); the depth search range is
required. The first stage starts from 48 random inverse-depth hypotheses: they are drawn from a CPU generator with a
fixed seed (as upstream after `torch.manual_seed(0)`), so results are deterministic and the same on every device.
Outputs: the refined depth (bilinear back to the input size, as upstream) and upstream's photometric confidence
(probability mass of the 4 hypotheses around the regressed one at the last stage, nearest-upsampled).

The forward pass is ours (`fast.py`), with upstream's maths and weights: BatchNorm folded into the convs, the
feature net's FPN rewritten by linearity (its 1x1 heads applied before the bilinear upsampling, at fewer channels),
the 1x1x1 Conv3d nets as 1x1 Conv2d, the sources warped and correlated one at a time (no `[S, C, D, H, W]` volume).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.sampling import grid_sample
from fastforks.multiview._runner import Forward, NoParams, Runner, f64, invert, take

from .net.net import PatchmatchNet

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.multiview._runner import Context

ARGS = {  # upstream eval.py defaults (the published ETH3D / DTU evaluation settings)
    "patchmatch_interval_scale": [0.005, 0.0125, 0.025],
    "propagation_range": [6, 4, 2],
    "patchmatch_iteration": [1, 2, 2],
    "patchmatch_num_sample": [8, 8, 16],
    "propagate_neighbors": [0, 8, 16],
    "evaluate_neighbors": [9, 9, 9],
}
MULTIPLE = 8  # fixed: upstream resizes to multiples of 8 (three stride-2 stages)
N_INIT = 48  # fixed: upstream random hypotheses of the first iteration
MAX_HYPOTHESES = 64  # fixed: slots of the pmn_aggregate / pmn_hypo kernels (D, Db + P)
_INIT_SIZES = 8  # fixed: cache policy, initial hypotheses kept (one per input size; video at one size always hits)
SEED = 0  # fixed: seed of the random initial hypotheses (deterministic results)
_FEAT = [  # FeatureNet ConvBnReLU layers: name, stride, padding
    ("conv0", 1, 1),
    ("conv1", 1, 1),
    ("conv2", 2, 2),
    ("conv3", 1, 1),
    ("conv4", 1, 1),
    ("conv5", 2, 2),
    ("conv6", 1, 1),
    ("conv7", 1, 1),
    ("conv8", 2, 2),
    ("conv9", 1, 1),
    ("conv10", 1, 1),
]
_TAPS = {1: "c1", 4: "c4", 7: "c7"}  # FeatureNet outputs the FPN uses

Conv = tuple[torch.Tensor, torch.Tensor | None]


def _fold(w: torch.Tensor, bn: Any, *, transposed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
    """Conv weight with an eval BatchNorm folded in -> (weight, bias)."""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bn.bias - bn.running_mean * s
    if transposed:
        return w * s.view(1, -1, 1, 1), b
    return w * s.view(-1, *([1] * (w.dim() - 1))), b


def _mlp(m: Any, last: Any) -> tuple[torch.Tensor, list[Conv]]:  # noqa: ANN401 - upstream modules
    """ConvBnReLU3D(G, 16) -> ConvBnReLU3D(16, 8) -> Conv3d(8, 1): packed fp32 (kernel layout), 1x1 Conv2d weights."""
    w0, b0 = _fold(m.conv0.conv.weight.flatten(1), m.conv0.bn)
    w1, b1 = _fold(m.conv1.conv.weight.flatten(1), m.conv1.bn)
    w2, b2 = last.weight.flatten(), last.bias.flatten()
    packed = torch.cat([w0.flatten(), b0, w1.flatten(), b1, w2, b2]).float().contiguous()
    convs: list[Conv] = [(w0[:, :, None, None], b0), (w1[:, :, None, None], b1), (w2.view(1, 8, 1, 1), b2)]
    return packed, convs


def _mlp2d(x: torch.Tensor, convs: list[Conv]) -> torch.Tensor:
    """`[B, G, D, H, W]` -> `[B, D, H, W]` through the 1x1x1 nets as 1x1 Conv2d over a `[B, G, D*H, W]` view."""
    b, g, d, h, w = x.shape
    y = F.relu(F.conv2d(x.reshape(b, g, d * h, w), *convs[0]))
    y = F.relu(F.conv2d(y, *convs[1]))
    return F.conv2d(y, *convs[2]).view(b, d, h, w)


@dataclass
class Stage:
    """One PatchMatch stage: settings and folded weights."""

    stage: int
    groups: int
    dilation: int
    iters: int
    scale: float  # patchmatch_interval_scale
    nsample: int
    propagate: int  # propagation neighbours
    evaluate: int  # evaluation neighbours
    propa: Conv
    evalc: Conv
    grid: Callable[..., torch.Tensor]  # upstream PatchMatch.get_grid
    grid_type: dict[str, int]
    pw: torch.Tensor  # pixel-wise view-weight net, packed
    sim: torch.Tensor  # similarity net, packed
    fw: torch.Tensor  # feature-weight net, packed
    pw2d: list[Conv]
    sim2d: list[Conv]
    fw2d: list[Conv]


def _net(ctx: Context) -> PatchmatchNet:
    net = PatchmatchNet(**ARGS)  # ty: ignore[invalid-argument-type]
    net.load_state_dict(ctx.state_dict(key="model"), strict=True)
    return net.eval().requires_grad_(requires_grad=False)


def _no_cudnn() -> contextlib.AbstractContextManager:
    """Eager calls: cuDNN off; a no-op while compiling / exporting (dynamo cannot trace the flag context)."""
    return contextlib.nullcontext() if torch.compiler.is_compiling() else torch.backends.cudnn.flags(enabled=False)


def _proj(k: torch.Tensor, c2w: torch.Tensor, scale: float) -> torch.Tensor:
    """`[S, 12]` rotation | translation of `src_proj @ inv(ref_proj)` per source (fp32), `proj = K @ inv(c2w)`.

    The relative pose (src from ref) is composed first, in float64, so the result does not depend on the common
    world origin; then the intrinsics are applied. The stage intrinsics scale `c` by `scale` (not
    `(c + 0.5) s - 0.5`): the feature net halves with stride-2 convolutions centred on input pixel `2 i`, so a
    coarse pixel `i` sits at `c / 2` exactly.
    """
    kl = k.clone()
    kl[:, :2] *= scale
    k4 = torch.eye(4, dtype=kl.dtype, device=kl.device).repeat(len(kl), 1, 1)
    k4[:, :3, :3] = kl
    c = f64(c2w)
    pose = (invert(c[1:]) @ c[:1]).to(kl.dtype)  # src from ref
    rel = k4[1:] @ pose @ invert(k4[:1])
    return torch.cat([rel[:, :3, :3].reshape(-1, 9), rel[:, :3, 3]], 1).contiguous()


def _unnorm(grid: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream grid `[1, N*h, w, 2]` -> grid_sample pixel coordinates (align_corners=False, border-clipped)."""
    g = grid.view(-1, h, w, 2)
    nx = (((g[..., 0] + 1) * w - 1) / 2).clamp(0, w - 1)
    ny = (((g[..., 1] + 1) * h - 1) / 2).clamp(0, h - 1)
    return nx.reshape(g.shape[0], h * w).contiguous(), ny.reshape(g.shape[0], h * w).contiguous()


def rescale_intrinsics(k: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """OpenCV intrinsics `[..., 3, 3]` of an image resized by `(sx, sy)`.

    Pixel centres sit at integer coordinates, so `f' = f s` and `c' = (c + 0.5) s - 0.5`; upstream `adjust_image_dims`
    scales `c s`, which is half a pixel per unit of scale off.
    """
    k = k.clone()
    k[..., 0, :] *= sx
    k[..., 1, :] *= sy
    k[..., 0, 2] += 0.5 * (sx - 1)
    k[..., 1, 2] += 0.5 * (sy - 1)
    return k


class _Base(Runner):
    intrinsics = "required"
    cam_to_world = "required"
    depth_range = "required"
    per_view = False
    min_views = 2
    batch = None
    Params = NoParams
    compiled_methods = ("__call__", "prepare", "predict")  # `scene` (several references) calls the last two

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.ref = _net(ctx)  # upstream modules: weights folded below, `get_grid` used as is
        dev, self.dt = ctx.device, self._dtype()
        f: Any = self.ref.feature
        self.feat = []
        for name, stride, pad in _FEAT:
            c = getattr(f, name)
            w, b = _fold(c.conv.weight, c.bn)
            self.feat.append((self._w(w), b.to(dev, self.dt), stride, pad))
        # FPN by linearity: 1x1 convs and bilinear x2 upsampling commute, and bilinear weights sum to 1 (biases pass)
        o1, o2, o3 = (getattr(f, f"output{i}").weight.double()[:, :, 0, 0] for i in (1, 2, 3))
        i1w, i1b = f.inner1.weight.double()[:, :, 0, 0], f.inner1.bias.double()
        i2w, i2b = f.inner2.weight.double()[:, :, 0, 0], f.inner2.bias.double()
        self.c_out1, self.c_out2 = o1.shape[0], o2.shape[0]
        self.heads = self._w(torch.cat([o1, o2, o3])[:, :, None, None].float())
        self.c7conv = (
            self._w(torch.cat([o2 @ i1w, o3 @ i1w])[:, :, None, None].float()),
            torch.cat([o2 @ i1b, o3 @ i1b]).to(dev, self.dt),
        )
        self.c4conv = (self._w((o3 @ i2w)[:, :, None, None].float()), (o3 @ i2b).to(dev, self.dt))
        r: Any = self.ref.upsample_net
        self.r_conv0, self.r_conv1, self.r_conv2, self.r_conv3 = (
            self._wb(*_fold(c.conv.weight, c.bn)) for c in (r.conv0, r.conv1, r.conv2, r.conv3)
        )
        self.r_deconv = self._wb(*_fold(r.deconv.weight, r.bn, transposed=True))
        self.r_res = self._w(r.res.weight)
        self.stages = [self._stage(getattr(self.ref, f"patchmatch_{i}"), i) for i in (1, 2, 3)]
        self._init_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.ctx.dtype == "float16" else torch.float32

    def _w(self, w: torch.Tensor) -> torch.Tensor:
        w = w.detach().to(self.ctx.device, self.dt)
        return w.contiguous(memory_format=torch.channels_last) if w.dim() == 4 else w  # noqa: PLR2004 - conv weight

    def _wb(self, w: torch.Tensor, b: torch.Tensor) -> Conv:
        return self._w(w), b.to(self.ctx.device, self.dt)

    def _stage(self, p: Any, i: int) -> Stage:  # noqa: ANN401 - upstream PatchMatch module
        e, dev = p.evaluation, self.ctx.device
        pw, pw2d = _mlp(e.pixel_wise_net, e.pixel_wise_net.conv2)
        sim, sim2d = _mlp(e.similarity_net, e.similarity_net.similarity)
        fw, fw2d = _mlp(p.feature_weight_net, p.feature_weight_net.similarity)

        def to(convs: list[Conv]) -> list[Conv]:
            return [(w.float().to(dev), None if b is None else b.float().to(dev)) for w, b in convs]

        return Stage(
            stage=i,
            groups=p.G,
            dilation=p.dilation,
            iters=p.patchmatch_iteration,
            scale=p.patchmatch_interval_scale,
            nsample=p.depth_initialization.patchmatch_num_sample,
            propagate=p.propagate_neighbors,
            evaluate=p.evaluate_neighbors,
            propa=self._wb(p.propa_conv.weight, p.propa_conv.bias),
            evalc=self._wb(p.eval_conv.weight, p.eval_conv.bias),
            grid=p.get_grid,
            grid_type=p.grid_type,
            pw=pw.to(dev),
            sim=sim.to(dev),
            fw=fw.to(dev),
            pw2d=to(pw2d),
            sim2d=to(sim2d),
            fw2d=to(fw2d),
        )

    # ------------------------------------------------------------------ CNNs
    def _cbr(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, stride: int, pad: int) -> torch.Tensor:
        return F.relu(F.conv2d(x, w, b, stride, pad))

    def features(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """FeatureNet: stage -> `[N, C, h, w]` features (channels-last)."""
        taps = {}
        for i, (w, b, s, p) in enumerate(self.feat):
            x = self._cbr(x, w, b, s, p)
            if i in _TAPS:
                taps[_TAPS[i]] = x
        h = F.conv2d(x, self.heads)
        a, b = self.c_out1, self.c_out2
        t = F.interpolate(h[:, a:], scale_factor=2.0, mode="bilinear", align_corners=False)
        t = t + F.conv2d(taps["c7"], *self.c7conv)
        o1 = F.interpolate(t[:, b:], scale_factor=2.0, mode="bilinear", align_corners=False)
        o1 = o1 + F.conv2d(taps["c4"], *self.c4conv)
        cl = torch.channels_last
        return {
            3: h[:, :a].contiguous(memory_format=cl),
            2: t[:, :b].contiguous(memory_format=cl),
            1: o1.contiguous(memory_format=cl),
        }

    def refine(self, img: torch.Tensor, depth0: torch.Tensor, dmin: torch.Tensor, dmax: torch.Tensor) -> torch.Tensor:
        """Refinement net (upstream `Refinement`): depth `[1, 1, h, w]` at half resolution -> full resolution."""
        depth = (depth0 - dmin.view(1, 1, 1, 1)) / (dmax - dmin).view(1, 1, 1, 1)
        c0 = self._cbr(img, *self.r_conv0, 1, 1)
        d = depth.to(self.dt).contiguous(memory_format=torch.channels_last)
        d = F.relu(F.conv2d(d, *self.r_conv1, 1, 1))
        d = self._cbr(d, *self.r_conv2, 1, 1)
        d = F.relu(F.conv_transpose2d(d, self.r_deconv[0], self.r_deconv[1], 2, 1, 1))
        res = F.conv2d(self._cbr(torch.cat((d, c0), 1), *self.r_conv3, 1, 1), self.r_res, None, 1, 1)
        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res.float()
        return depth * (dmax - dmin).view(1, 1, 1, 1) + dmin.view(1, 1, 1, 1)

    # ------------------------------------------------------------------ PatchMatch helpers
    def offsets(self, st: Stage, ref: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Adaptive propagation / evaluation grids as upstream (offset convs in the CNN dtype, grids in fp32)."""
        dev, pgrid = ref.device, None
        if st.propagate > 0 and not (st.stage == 1 and st.iters == 1):
            off = F.conv2d(ref, *st.propa, 1, st.dilation, st.dilation).float().view(1, 2 * st.propagate, h * w)
            pgrid = st.grid(st.grid_type["propagation"], 1, h, w, off, dev)
        off = F.conv2d(ref, *st.evalc, 1, st.dilation, st.dilation).float().view(1, 2 * st.evaluate, h * w)
        return pgrid, st.grid(st.grid_type["evaluation"], 1, h, w, off, dev)

    def random_init(self, dmin: torch.Tensor, dmax: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Upstream first-iteration hypotheses `[1, 48, h, w]`: one random inverse depth per interval (seeded)."""
        if (h, w) not in self._init_cache:
            if len(self._init_cache) >= _INIT_SIZES:
                del self._init_cache[next(iter(self._init_cache))]
            g = torch.Generator().manual_seed(SEED)
            u = torch.rand((1, N_INIT, h, w), generator=g) + torch.arange(N_INIT).view(1, N_INIT, 1, 1)
            self._init_cache[(h, w)] = (u / N_INIT).to(dmin.device)
        imin, imax = 1.0 / dmin, 1.0 / dmax
        return 1.0 / (imax.view(1, 1, 1, 1) + self._init_cache[(h, w)] * (imin - imax).view(1, 1, 1, 1))

    def perturb(self, st: Stage, depth: torch.Tensor, dmin: torch.Tensor, dmax: torch.Tensor) -> torch.Tensor:
        """Upstream local perturbation: `nsample` inverse depths around the previous depth, clamped to the range."""
        imin, imax, k = 1.0 / dmin, 1.0 / dmax, st.nsample
        s = torch.arange(-k // 2, k // 2, 1, device=depth.device).view(1, k, 1, 1).float()
        s = 1.0 / depth + ((imin - imax) * st.scale).view(1, 1, 1, 1) * s
        return 1.0 / torch.clamp(s, min=imax[0], max=imin[0])

    def run_stage(  # noqa: PLR0913, PLR0917 - the stage state
        self,
        st: Stage,
        f: torch.Tensor,
        proj: torch.Tensor,
        dmin: torch.Tensor,
        dmax: torch.Tensor,
        depth: torch.Tensor | None,
        vw: torch.Tensor | None,
        *,
        last: bool,
        inv: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One stage on features `f` `[N, C, h, w]` -> (depth `[1, 1, h, w]`, view weights, confidence or None).

        `inv` is `host_range` of the depth range (`None` unless the runner needs it on the host).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ forward
    def prepare(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Per-image work of a scene: the network-size images (CNN dtype, channels-last) and their features."""
        _, _, h0, w0 = images.shape
        h, w = round(h0 / MULTIPLE) * MULTIPLE, round(w0 / MULTIPLE) * MULTIPLE
        if (h, w) != (h0, w0):
            images = F.interpolate(images, (h, w), mode="bilinear", align_corners=False)
        x = images.to(self.dt).contiguous(memory_format=torch.channels_last)
        return x, self.features(x)

    def host_range(self, depth_range: torch.Tensor) -> tuple[float, float] | None:
        """`(1 / near, 1 / far)` as floats if the runner's kernels take them as scalars (a host sync), else `None`."""
        del depth_range
        return None

    def predict(  # noqa: PLR0913, PLR0917 - the prepared scene and the cameras
        self,
        x: torch.Tensor,  # [>= 1, 3, h, w]: the reference first (only it is used)
        feats: dict[int, torch.Tensor],
        size: tuple[int, int],
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        depth_range: torch.Tensor,
        inv: tuple[float, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepared views (reference first; `size` = input size) and cameras -> depth and confidence `[1, H, W]`.

        `inv`: `host_range(depth_range)` if already known (one scene, several references).
        """
        (h0, w0), (h, w) = size, x.shape[-2:]
        k = intrinsics.float().clone()
        if (h, w) != (h0, w0):
            k = rescale_intrinsics(k, w / w0, h / h0)
        dmin, dmax = depth_range[:1].float(), depth_range[1:].float()
        inv = self.host_range(depth_range) if inv is None else inv
        depth = vw = conf = None
        scale = 0.125
        for i in (3, 2, 1):
            proj = _proj(k, cam_to_world, scale)
            scale *= 2.0
            st = self.stages[i - 1]
            depth, vw, conf = self.run_stage(st, feats[i], proj, dmin, dmax, depth, vw, last=i == 1, inv=inv)
            if i > 1:
                depth = F.interpolate(depth, scale_factor=2.0, mode="nearest")
                vw = F.interpolate(vw, scale_factor=2.0, mode="nearest")
        if depth is None or conf is None:
            msg = "PatchmatchNet: the last stage returned no depth / confidence"
            raise RuntimeError(msg)
        depth = self.refine(x[:1], depth, dmin, dmax)
        if (h, w) != (h0, w0):
            depth = F.interpolate(depth, (h0, w0), mode="bilinear", align_corners=False)
        conf = F.interpolate(conf, (h0, w0), mode="nearest")
        return {"depth": depth[:, 0], "confidence": conf[:, 0]}

    def forward(
        self, images: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor, depth_range: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The runner's inputs -> depth and photometric confidence of the reference view `[1, H, W]`."""
        inv = self.host_range(depth_range)  # before any GPU work, so its host sync does not wait for the features
        x, feats = self.prepare(images)
        size = (images.shape[-2], images.shape[-1])
        return self.predict(x, feats, size, intrinsics, cam_to_world, depth_range, inv=inv)

    @staticmethod
    def _required(
        intrinsics: torch.Tensor | None, cam_to_world: torch.Tensor | None, depth_range: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if intrinsics is None or cam_to_world is None or depth_range is None:  # the core checks required inputs
            msg = "PatchmatchNet needs intrinsics, cam_to_world and depth_range"
            raise ValueError(msg)
        return intrinsics, cam_to_world, depth_range

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth and confidence of `images[0]` from the other views."""
        k, c2w, rng = self._required(intrinsics, cam_to_world, depth_range)
        with torch.no_grad():
            return self.forward(images, k, c2w, rng)

    def scene(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> Callable[[torch.Tensor | None], dict[str, torch.Tensor]]:
        """See `Runner.scene`: the features of every view once, the PatchMatch stages per reference."""
        k0, c2w0, rng = self._required(intrinsics, cam_to_world, depth_range)
        size = (images.shape[-2], images.shape[-1])
        inv = self.host_range(rng)  # once per scene
        with torch.no_grad():
            x, feats = self.prepare(images)

        def predict(order: torch.Tensor | None) -> dict[str, torch.Tensor]:
            xo, fo, k, c2w = x, feats, k0, c2w0
            if order is not None:  # the refinement net needs only the reference image
                xo, fo = take(x, order[:1]), {i: take(f, order) for i, f in feats.items()}
                k, c2w = k0[order], c2w0[order]
            with torch.no_grad():
                return self.predict(xo, fo, size, k, c2w, rng, inv)

        return predict


class Portable(_Base):
    """Plain PyTorch: float32, or fp16 channels-last CNNs on CUDA (warping, correlation and depth maths in fp32)."""

    dtypes = ("float32", "float16")

    def run_stage(  # noqa: C901, PLR0913, PLR0915, PLR0917 - one upstream stage, unrolled
        self,
        st: Stage,
        f: torch.Tensor,
        proj: torch.Tensor,
        dmin: torch.Tensor,
        dmax: torch.Tensor,
        depth: torch.Tensor | None,
        vw: torch.Tensor | None,
        *,
        last: bool,
        inv: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Upstream PatchMatch.forward (eval), the sources warped one at a time (`inv` unused)."""
        del inv
        g = st.groups
        _, c, h, w = f.shape
        n_src = f.shape[0] - 1
        ref = f[:1].float()
        src = f[1:].float()
        pgrid, egrid = self.offsets(st, f[:1], h, w)
        ne = st.evaluate
        wt = F.grid_sample(ref, egrid, mode="bilinear", padding_mode="border", align_corners=False)
        wt = (wt.view(1, g, c // g, ne, h, w) * ref.view(1, g, c // g, 1, h, w)).mean(2)
        fweight = torch.sigmoid(_mlp2d(wt, st.fw2d))
        ys, xs = torch.meshgrid(
            torch.arange(h, device=f.device, dtype=torch.float32),
            torch.arange(w, device=f.device, dtype=torch.float32),
            indexing="ij",
        )
        xyz = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, device=f.device)))  # [3, hw]
        rot = proj[:, :9].view(n_src, 3, 3)
        trans = proj[:, 9:].view(n_src, 3, 1, 1)
        rxyz = torch.matmul(rot, xyz).unsqueeze(2)  # [S, 3, 1, hw]
        imin, imax = 1.0 / dmin, 1.0 / dmax
        conf = None
        for it in range(1, st.iters + 1):
            final = st.stage == 1 and it == st.iters  # inverse-depth regression, no propagation
            ds = self.random_init(dmin, dmax, h, w) if depth is None else self.perturb(st, depth, dmin, dmax)
            if pgrid is not None and not final:
                d0 = ds.shape[1]
                prop = F.grid_sample(
                    ds[:, d0 // 2 : d0 // 2 + 1], pgrid, mode="bilinear", padding_mode="border", align_corners=False
                ).view(1, st.propagate, h, w)
                ds = torch.sort(torch.cat((ds, prop), 1), 1)[0]
            d = ds.shape[1]
            # depth weight x feature weight, normalised over the evaluation neighbours: [1, D, NE, h, w]
            xn = (1.0 / ds - imax.view(1, 1, 1, 1)) / (imin - imax).view(1, 1, 1, 1)
            x1 = F.grid_sample(xn, egrid, mode="bilinear", padding_mode="border", align_corners=False)
            dw = torch.sigmoid(
                4.0 - 2.0 * (torch.abs(x1.view(1, d, ne, h, w) - xn.unsqueeze(2)) / st.scale).clamp(0, 4)
            )
            weight = dw * fweight.unsqueeze(1)
            weight = weight / weight.sum(2, keepdim=True)
            dflat = ds.view(1, 1, d, h * w)
            new_vw = vw is None
            vws = []
            sim_sum = torch.zeros((1, g, d, h, w), device=f.device)
            wsum = torch.full((1, 1, 1, h, w), 1e-5, device=f.device)
            for s in range(n_src):
                p = rxyz[s : s + 1] * dflat + trans[s : s + 1]  # [1, 3, D, hw]
                neg = p[:, 2:3] <= 1e-3  # noqa: PLR2004 - upstream differentiable_warping
                px = torch.where(neg, torch.full_like(p[:, :1], float(w)), p[:, :1])
                py = torch.where(neg, torch.full_like(p[:, 1:2], float(h)), p[:, 1:2])
                pz = torch.where(neg, torch.ones_like(p[:, 2:3]), p[:, 2:3])
                gx = (px / pz) / ((w - 1) / 2) - 1
                gy = (py / pz) / ((h - 1) / 2) - 1
                grid = torch.stack((gx[:, 0], gy[:, 0]), 3).view(1, d * h, w, 2)
                with _no_cudnn():  # the native sampler is 10-25 % faster than cuDNN's
                    warped = grid_sample(
                        src[s : s + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=True
                    )
                wv, rv = warped.view(g, c // g, d, h * w), ref.view(g, c // g, 1, h * w)
                sim = wv[:, 0] * rv[:, 0]
                for j in range(1, c // g):
                    sim.addcmul_(wv[:, j], rv[:, j])
                sim = sim.mul_(g / c).view(1, g, d, h, w)
                if new_vw:
                    v = torch.sigmoid(_mlp2d(sim, st.pw2d)).amax(1, keepdim=True)  # [1, 1, h, w]
                    vws.append(v)
                elif vw is not None:
                    v = vw[:, s : s + 1]
                sim_sum.addcmul_(sim, v.unsqueeze(1))
                wsum += v.unsqueeze(1)
            if new_vw:
                vw = torch.cat(vws, 1)
            cost = _mlp2d(sim_sum / wsum, st.sim2d)  # [1, D, h, w]
            cost = F.grid_sample(cost, egrid, mode="bilinear", padding_mode="border", align_corners=False)
            score = torch.softmax((cost.view(1, d, ne, h, w) * weight).sum(2), 1)
            if final:
                idx = (torch.arange(d, device=f.device).view(1, d, 1, 1) * score).sum(1)
                imn, imx = 1.0 / ds[:, -1], 1.0 / ds[:, 0]
                depth = (1.0 / (imx + idx / (d - 1) * (imn - imx))).unsqueeze(1)
            else:
                depth = (ds * score).sum(1, keepdim=True)
            if last and it == st.iters:
                s4 = 4 * F.avg_pool3d(F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1).squeeze(1)
                di = (score * torch.arange(d, device=f.device, dtype=torch.float).view(1, d, 1, 1)).sum(1, keepdim=True)
                conf = torch.gather(s4, 1, di.long().clamp(0, d - 1))
        if depth is None or vw is None:
            msg = "PatchmatchNet: a stage without iterations"
            raise RuntimeError(msg)
        return depth, vw, conf

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.ref, lambda _net, *inputs: self.forward(*inputs))


class Kernels(_Base):
    """Kernels path: fp16 channels-last CNNs, fused conv + bias + ReLU (cuDNN), fused PatchMatch kernels.

    The `patchmatchnet` kernels never materialise warped features: one pass per (pixel, hypothesis) warps every source,
    correlates, weights the views and runs the similarity net; one per pixel aggregates the neighbours' costs, takes the
    softmax and regresses depth and confidence; one perturbs, propagates and sorts the hypotheses.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the folded network on the device and load the kernels."""
        from fastforks.kernels.patchmatchnet import CONFIGS  # noqa: PLC0415 - kernel lib only in kernels
        from fastforks.kernels.patchmatchnet import load as load_kernels  # noqa: PLC0415

        self.kern = load_kernels(ctx.device)
        self.configs = CONFIGS
        super().__init__(ctx)
        for st in self.stages:
            c = self.feat_channels(st)
            if st.nsample + st.propagate > MAX_HYPOTHESES:
                msg = (
                    f"PatchmatchNet kernels: stage {st.stage} has {st.nsample} + {st.propagate} depth hypotheses; "
                    f"the kernels hold {MAX_HYPOTHESES}"
                )
                raise ValueError(msg)
            if (c, st.groups) not in CONFIGS or st.evaluate != 9:  # noqa: PLR2004 - aggregate kernel: 9 neighbours
                msg = f"PatchmatchNet kernels: stage {st.stage} ({c} channels, {st.groups} groups) has no kernel"
                raise ValueError(msg)

    def _dtype(self) -> torch.dtype:
        return torch.float16

    def host_range(self, depth_range: torch.Tensor) -> tuple[float, float]:
        """`(1 / near, 1 / far)` on the host for the kernels' scalar arguments: one sync per scene."""
        imin, imax = (1.0 / depth_range.float()).tolist()
        return imin, imax

    @staticmethod
    def feat_channels(st: Stage) -> int:
        """Channels of the stage features."""
        return st.evalc[0].shape[1]

    def _cbr(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, stride: int, pad: int) -> torch.Tensor:
        return torch.cudnn_convolution_relu(x, w, b, (stride, stride), (pad, pad), (1, 1), 1)

    def _launch(self, name: str, grid: tuple[int, int, int], *args: Any) -> None:  # noqa: ANN401 - kernel args
        self.kern.launch(name, grid, (128, 1, 1), *args)

    def run_stage(  # noqa: PLR0913, PLR0917 - the stage state
        self,
        st: Stage,
        f: torch.Tensor,
        proj: torch.Tensor,
        dmin: torch.Tensor,
        dmax: torch.Tensor,
        depth: torch.Tensor | None,
        vw: torch.Tensor | None,
        *,
        last: bool,
        inv: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One stage through the fused kernels (features NHWC fp16, hypotheses and costs `[D, hw]` fp32)."""
        g = st.groups
        n, c, h, w = f.shape
        n_src, hw = n - 1, h * w
        blocks = (hw + 127) // 128
        tag = f"{c}_{g}"
        nhwc = f.permute(0, 2, 3, 1).contiguous()
        ref, src = nhwc[0], nhwc[1:]
        pgrid, egrid = self.offsets(st, f[:1], h, w)
        enx, eny = _unnorm(egrid, h, w)
        pnx = pny = None
        if pgrid is not None:
            pnx, pny = _unnorm(pgrid, h, w)
        fw = torch.empty((st.evaluate, hw), device=f.device)
        self._launch(f"pmn_feat_weight_{tag}", (blocks, st.evaluate, 1), ref, enx, eny, st.fw, fw, h, w)
        imin, imax = inv if inv is not None else ((1.0 / dmin).item(), (1.0 / dmax).item())  # the kernels take scalars
        conf = None
        out = torch.empty((1, 1, h, w), device=f.device)
        for it in range(1, st.iters + 1):
            final = st.stage == 1 and it == st.iters
            n_prop = st.propagate if pgrid is not None and not final else 0
            if depth is None:
                base = self.random_init(dmin, dmax, h, w)[0].reshape(-1, hw).contiguous()
                ds = base
                if n_prop:
                    ds = torch.empty((base.shape[0] + n_prop, hw), device=f.device)
                    self._launch(
                        "pmn_hypo",
                        (blocks, 1, 1),
                        None,
                        base,
                        pnx,
                        pny,
                        ds,
                        base.shape[0],
                        n_prop,
                        imin,
                        imax,
                        0.0,
                        h,
                        w,
                    )
            else:
                ds = torch.empty((st.nsample + n_prop, hw), device=f.device)
                nx, ny = (pnx, pny) if n_prop else (enx, eny)
                prev = depth.reshape(hw).contiguous()
                interval = (imin - imax) * st.scale
                self._launch(
                    "pmn_hypo", (blocks, 1, 1), prev, None, nx, ny, ds, st.nsample, n_prop, imin, imax, interval, h, w
                )
            d = ds.shape[0]
            if d > MAX_HYPOTHESES:
                msg = f"PatchmatchNet kernels: {d} depth hypotheses at stage {st.stage}; they hold {MAX_HYPOTHESES}"
                raise ValueError(msg)
            if vw is None:
                vw = torch.zeros((1, n_src, h, w), device=f.device)
                self._launch(f"pmn_view_weight_{tag}", (blocks, d, n_src), ref, src, proj, ds, st.pw, vw, h, w)
            cost = torch.empty((d, hw), device=f.device)
            self._launch(f"pmn_cost_{tag}", (blocks, d, 1), ref, src, proj, ds, vw, st.sim, cost, n_src, h, w)
            cf = torch.empty((1, 1, h, w), device=f.device) if last and it == st.iters else None
            self._launch(
                "pmn_aggregate",
                (blocks, 1, 1),
                cost, ds, enx, eny, fw, out, cf, imax, imin - imax, 1.0 / st.scale, d, h, w, int(final),
            )  # fmt: skip
            depth = out.clone() if it < st.iters else out
            if cf is not None:
                conf = cf
        if depth is None or vw is None:
            msg = "PatchmatchNet: a stage without iterations"
            raise RuntimeError(msg)
        return depth, vw, conf


RUNNERS = {"portable": Portable, "kernels": Kernels}
