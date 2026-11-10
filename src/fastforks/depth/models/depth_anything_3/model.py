"""Depth Anything 3 runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `DepthAnything3.inference` for one view (no poses): resize so that the longer side is
`long_side` (OpenCV `INTER_CUBIC` up / `INTER_AREA` down, on uint8), round each side to the nearest multiple of 14
(a second OpenCV resize), normalise with the ImageNet mean / std, run the network, post-process as upstream, and
upsample the maps back to the input size (bilinear, `align_corners=False`). The OpenCV resizes run on the device
(`resize.py`).

Post-processing (upstream `DepthAnything3Net` / `NestedDepthAnything3Net`), per image and on the full maps (upstream
estimates the quantiles from a random sample of 100k pixels on larger maps, so its output is not deterministic there):

- sky head (Mono, Metric): non-sky = `sky < 0.3`; if both parts have more than 10 pixels, sky pixels get the 99th
  percentile of the non-sky depth
- Metric: metres = depth * focal / 300 with the focal length (mean of fx, fy) at the network input size, from the given
  intrinsics; without them a 300 px focal length is assumed there (the raw output, returned as `intrinsics`)
- camera head (Large / Giant): intrinsics from the predicted field of view, principal point at the image centre
- Nested: the any-view (Giant) depth is scaled to the Metric-Large depth (metric with the Giant's predicted
  intrinsics) by least squares over the non-sky pixels with confidence >= its median, metric > 0.01 and depth > 0.001;
  sky pixels get min(99th percentile, 200 m) and confidence 1 (with 10 or fewer non-sky pixels, where upstream raises,
  every pixel counts as non-sky)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.shared.quantile import kth_smallest

from .net.cam_dec import CameraDec
from .net.dinov2.dinov2 import DinoV2
from .net.dinov2.layers import Mlp
from .net.dpt import DPT
from .net.dualdpt import DualDPT
from .net.utils.head_utils import create_uv_grid, position_grid_to_embed
from .resize import cv2_resize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

PATCH = 14
HEAD_DIM = 64
_FOCAL = 300.0  # fixed: upstream metric depth is for a 300 px focal length (metres = output * focal / 300)
_SKY = 0.3  # fixed: upstream compute_sky_mask threshold (non-sky = sky < 0.3)
_MIN_PIXELS = 10  # fixed: upstream needs more than 10 sky and non-sky pixels to fill the sky
_SKY_QUANTILE = 0.99  # fixed: upstream sky depth = this quantile of the non-sky depth
_SKY_MAX = 200.0  # fixed: upstream nested sky depth cap (metres)
_MIN_DEPTH, _MIN_METRIC = 1e-3, 1e-2  # fixed: upstream compute_alignment_mask thresholds
_POS_RATIO = 0.1  # fixed: upstream DualDPT._add_pos_embed ratio

Arch = Literal["mono", "metric", "large", "giant", "nested"]


@dataclass(frozen=True)
class _Config:
    vit: str  # DINOv2 size
    layers: tuple[int, ...]  # blocks whose tokens feed the head
    alt: int  # first block with camera token, alternating global attention, qk-norm and RoPE (-1: none)
    dual: bool  # DualDPT (depth + confidence) and camera decoder, else DPT with sky head


_DPT_L = _Config("vitl", (4, 11, 17, 23), -1, dual=False)  # upstream da3mono-large / da3metric-large.yaml
_CONFIGS = {
    "mono": (_DPT_L,),
    "metric": (_DPT_L,),
    "large": (_Config("vitl", (11, 15, 19, 23), 8, dual=True),),  # da3-large.yaml
    "giant": (_Config("vitg", (19, 27, 33, 39), 13, dual=True),),  # da3-giant.yaml
}
_CONFIGS["nested"] = (_CONFIGS["giant"][0], _DPT_L)  # da3nested-giant-large.yaml: any-view + metric
_EMBED = {"vitl": 1024, "vitg": 1536}
_OUT_CHANNELS = (256, 512, 1024, 1024)
_FEATURES = 256


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: Arch
    long_side: int  # longer side of the network input before rounding to multiples of 14


class _Branch(nn.Module):
    """One upstream `DepthAnything3Net` without the camera encoder, the GS head and DualDPT's auxiliary ray branch."""

    def __init__(self, cfg: _Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoV2(
            cfg.vit, list(cfg.layers), alt_start=cfg.alt, qknorm_start=cfg.alt, rope_start=cfg.alt, cat_token=cfg.dual
        )
        dim = _EMBED[cfg.vit]
        self.cam_dec: CameraDec | None = None
        self.head: DPT | DualDPT
        if cfg.dual:
            self.head = DualDPT(2 * dim, output_dim=2, features=_FEATURES, out_channels=_OUT_CHANNELS)
            scratch = self.head.scratch
            for name in [n for n, _ in scratch.named_children() if n.endswith("_aux")]:
                delattr(scratch, name)  # ray branch: upstream only uses it with use_ray_pose
            self.cam_dec = CameraDec(2 * dim)
        else:
            self.head = DPT(dim, output_dim=1, features=_FEATURES, out_channels=_OUT_CHANNELS)


class _Net(nn.Module):
    """`main` (+ `metric` for Nested)."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        cfgs = _CONFIGS[arch]
        self.main = _Branch(cfgs[0])
        self.metric = _Branch(cfgs[1]) if len(cfgs) > 1 else None


def _rename(key: str, nested: bool) -> str | None:  # noqa: FBT001 - internal
    """Checkpoint key (without `model.`) -> `_Net` key; None for the modules left out."""
    if nested:
        branch, _, rest = key.partition(".")
        key = {"da3": "main", "da3_metric": "metric"}.get(branch, branch) + "." + rest  # else unexpected
    else:
        key = "main." + key
    part = key.split(".")[1]
    if part in {"cam_enc", "gs_head", "gs_adapter"} or "_aux." in key:
        return None
    return key


def _net(ctx: Context, device: torch.device | str | None = None) -> _Net:
    """The network on `device` (default: the context's)."""
    arch: Arch = ctx.params.arch
    with torch.device("meta"):
        net = _Net(arch)
    sd = {}
    for k, v in ctx.state_dict(strip="model.").items():
        name = _rename(k, arch == "nested")
        if name is not None:
            sd[name] = v
    load_strict(net, sd, f"{ctx.variant.ref} checkpoint", assign=True)
    return net.eval().to(ctx.device if device is None else device)


def network_size(h: int, w: int, long_side: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Upstream `upper_bound_resize` then `_make_divisible_by_resize`: `(h1, w1)` after the first resize, `(h2, w2)`.

    A side upstream makes 1 (not a patch multiple; aspect ratio above ~70) is one patch (stretched).
    """
    if max(h, w) == long_side:
        h1, w1 = h, w
    else:
        s = long_side / max(h, w)
        h1, w1 = max(1, round(h * s)), max(1, round(w * s))

    def nearest(x: int) -> int:
        down = x // PATCH * PATCH
        return max(PATCH, down + PATCH if down + PATCH - x <= x - down else down)

    return (h1, w1), (nearest(h1), nearest(w1))


def _prepare(ctx: Context, image: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> resized, normalised network input (float32), as upstream's uint8 OpenCV steps."""
    h, w = image.shape[-2:]
    (h1, w1), (h2, w2) = network_size(h, w, ctx.params.long_side)
    x = (image * 255).round()
    if (h1, w1) != (h, w):
        x = cv2_resize(x, h1, w1, "cubic" if ctx.params.long_side > max(h, w) else "area")
    if (h2, w2) != (h1, w1):
        x = cv2_resize(x, h2, w2, "cubic" if h2 > h1 or w2 > w1 else "area")
    return scale(x / 255, ctx.io)


# ------------------------------------------------------------------------------------------------ post-processing
def _quantile(v: torch.Tensor, valid: torch.Tensor, q: float) -> torch.Tensor:
    """Per row of `v` `[B, N]`: `torch.quantile` (linear) of the `valid` entries, without a host sync; `[B]`."""
    n = valid.sum(1)
    rank = torch.tensor(q, dtype=v.dtype, device=v.device) * (n - 1).clamp(min=0).to(v.dtype)
    lo = rank.floor().long()
    hi = rank.ceil().long()
    if torch.compiler.is_exporting():  # TensorRT's TopK (sort) takes at most 3840 elements: bisect, same values
        vlo = kth_smallest(v, lo, valid)
        le = (valid & (v <= vlo[:, None])).sum(1)  # the hi-th entry is vlo, or the next larger valid value
        vhi = torch.where(le > hi, vlo, torch.where(valid & (v > vlo[:, None]), v, torch.inf).amin(1))
        vlo, vhi = torch.where(n > 0, vlo, torch.inf), torch.where(n > 0, vhi, torch.inf)  # no valid entry: as sort
    else:
        s = torch.where(valid, v, torch.inf).sort(dim=1).values
        vlo, vhi = s.gather(1, lo[:, None])[:, 0], s.gather(1, hi[:, None])[:, 0]
    return torch.lerp(vlo, vhi, rank - lo.to(v.dtype))


def _fill_sky(depth: torch.Tensor, sky: torch.Tensor) -> torch.Tensor:
    """Upstream `_process_mono_sky_estimation` per image: `[B, h, w]`."""
    d, ns = depth.flatten(1), sky.flatten(1) < _SKY
    n = ns.sum(1)
    apply = (n > _MIN_PIXELS) & (ns.shape[1] - n > _MIN_PIXELS)
    q = _quantile(d, ns, _SKY_QUANTILE)
    return torch.where(ns | ~apply[:, None], d, q[:, None]).view_as(depth)


def _fov_intrinsics(fov: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Upstream `pose_encoding_to_extri_intri`: `[B, 2]` (fov_h, fov_w) -> `[B, 3, 3]` for an `h x w` image."""
    k = torch.zeros((fov.shape[0], 3, 3), device=fov.device)
    k[:, 0, 0] = (w / 2) / torch.tan(fov[:, 1] / 2).clamp(min=1e-6)
    k[:, 1, 1] = (h / 2) / torch.tan(fov[:, 0] / 2).clamp(min=1e-6)
    k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = (w - 1) / 2, (h - 1) / 2, 1.0  # OpenCV pixel centres
    return k


def _nested(main: dict, metric: dict, hw: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `NestedDepthAnything3Net` alignment: (metric depth, confidence) at the network size."""
    k = _fov_intrinsics(main["fov"], *hw)
    focal = (k[:, 0, 0] + k[:, 1, 1]) / 2
    m = (metric["depth"] * (focal / _FOCAL)[:, None, None]).flatten(1)
    d, conf = main["depth"].flatten(1), main["conf"].flatten(1)
    ns = metric["sky"].flatten(1) < _SKY
    ns = ns | (ns.sum(1, keepdim=True) <= _MIN_PIXELS)
    med = _quantile(conf, ns, 0.5)
    align = (conf >= med[:, None]) & ns & (m > _MIN_METRIC) & (d > _MIN_DEPTH)
    da = torch.where(align, d, 0)
    s = (da * m).sum(1) / (da * da).sum(1).clamp(min=1e-12)
    d = d * s[:, None]
    cap = _quantile(d, ns, _SKY_QUANTILE).clamp(max=_SKY_MAX)
    d = torch.where(ns, d, cap[:, None])
    conf = torch.where(ns, conf, 1.0)
    return d.view_as(main["depth"]), conf.view_as(main["depth"])


def _upsample(m: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(m[:, None].float(), size, mode="bilinear", align_corners=False)[:, 0]


def _finish(
    arch: Arch, outs: list[dict], k: torch.Tensor | None, size: Sequence[int], hw: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Branch outputs at the network size `hw` -> the runner's maps at the input `size`."""
    (h, w), (hh, ww) = size, hw
    main = outs[0]
    res: dict[str, torch.Tensor] = {}
    if arch == "nested":
        d, conf = _nested(main, outs[1], hw)
        sky = _upsample(outs[1]["sky"], size)
        res = {"depth": _upsample(d, size), "confidence": _upsample(conf, size), "mask": sky < _SKY, "sky": sky}
    elif arch in {"large", "giant"}:
        res = {"depth": _upsample(main["depth"], size), "confidence": _upsample(main["conf"], size)}
    else:
        d = _fill_sky(main["depth"], main["sky"])
        if arch == "metric":
            if k is None:
                k = torch.zeros((d.shape[0], 3, 3), device=d.device)
                k[:, 0, 0], k[:, 1, 1] = _FOCAL * w / ww, _FOCAL * h / hh
                k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = (w - 1) / 2, (h - 1) / 2, 1.0
                res["intrinsics"] = k
            focal = (k[:, 0, 0] * ww / w + k[:, 1, 1] * hh / h) / 2
            d = d * (focal / _FOCAL)[:, None, None]
        sky = _upsample(main["sky"], size)
        res |= {"depth": _upsample(d, size), "mask": sky < _SKY, "sky": sky}
    if main.get("fov") is not None:
        res["intrinsics"] = _fov_intrinsics(main["fov"], h, w)
    return res


def _activate(arch_dual: bool, logits: torch.Tensor, sky_logits: torch.Tensor | None) -> dict:  # noqa: FBT001
    """Head logits (fp32) -> depth (exp) and confidence (exp + 1) or sky (relu), `[B, h, w]`."""
    if arch_dual:
        return {"depth": logits[:, 0].exp(), "conf": logits[:, 1].exp() + 1}
    assert sky_logits is not None  # noqa: S101 - DPT has the sky head
    return {"depth": logits[:, 0].exp(), "sky": sky_logits[:, 0].relu()}


# ------------------------------------------------------------------------------------------------ portable
def _decode(head: DPT | DualDPT, feats: list[torch.Tensor], ph: int, pw: int) -> torch.Tensor:
    """Upstream DPT / DualDPT main branch up to the output convs: tokens `[B, N, C]` -> `[B, 128, 14 ph, 14 pw]`."""
    b = feats[0].shape[0]
    h, w = ph * PATCH, pw * PATCH
    rs = []
    for i, f in enumerate(feats):
        x = head.norm(f).transpose(1, 2).reshape(b, -1, ph, pw)
        x = head.projects[i](x)
        if head.pos_embed:
            x = head._add_pos_embed(x, w, h).to(x.dtype)  # noqa: SLF001 - upstream helper (built in fp32)
        rs.append(head.resize_layers[i](x))
    s: Any = head.scratch
    l1, l2, l3, l4 = s.layer1_rn(rs[0]), s.layer2_rn(rs[1]), s.layer3_rn(rs[2]), s.layer4_rn(rs[3])
    p = s.refinenet4(l4, size=l3.shape[2:])
    p = s.refinenet3(p, l3, size=l2.shape[2:])
    p = s.refinenet2(p, l2, size=l1.shape[2:])
    p = s.refinenet1(p, l1)
    p = F.interpolate(s.output_conv1(p), (h, w), mode="bilinear", align_corners=True)
    if head.pos_embed:
        p = head._add_pos_embed(p, w, h).to(p.dtype)  # noqa: SLF001 - upstream helper (built in fp32)
    return p


def _input_use(params: Params) -> dict[str, Any]:
    return {"intrinsics": "optional" if params.arch == "metric" else "unused"}


class Portable(Runner):
    """The upstream modules with fused SDPA attention: float32, or float16 (channels-last head; last convs float32)."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    intrinsics = "optional"  # Metric; set per variant in __init__

    @classmethod
    def input_use(cls, params: Params) -> dict[str, Any]:
        """Intrinsics: `"optional"` for the metric variants, else `"unused"`.

        Args:
            params: The variant's `Params`.
        """
        return _input_use(params)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.arch: Arch = ctx.params.arch
        self.intrinsics = self.input_use(ctx.params)["intrinsics"]  # ty: ignore[invalid-attribute-access] - per variant
        self.net = _net(ctx)
        self.half = ctx.dtype == "float16"
        if self.half:
            for br in self._branches():  # pure fp16 backbone + head (no autocast), the last output convs in fp32
                br.backbone.half()
                br.head.half().to(memory_format=torch.channels_last)
                s: Any = br.head.scratch
                for name in ("output_conv2", "sky_output_conv2"):
                    if hasattr(s, name):
                        getattr(s, name)[-1].float()

    def _branches(self) -> list[_Branch]:
        return [self.net.main] + ([self.net.metric] if self.net.metric is not None else [])

    def _branch(self, br: _Branch, x: torch.Tensor) -> dict:
        ph, pw = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        with torch.autocast(x.device.type, enabled=False):
            out, _ = br.backbone((x.half() if self.half else x)[:, None])
            feats = [f[0][:, 0] for f in out]
            p = _decode(br.head, feats, ph, pw)
            s: Any = br.head.scratch
            last = (lambda q: q.float()) if self.half else (lambda q: q)
            logits = s.output_conv2[-1](last(s.output_conv2[:-1](p)))
            sky = None if br.cfg.dual else s.sky_output_conv2[-1](last(s.sky_output_conv2[:-1](p)))
        res = _activate(br.cfg.dual, logits.float(), None if sky is None else sky.float())
        if br.cam_dec is not None:
            with torch.autocast(x.device.type, enabled=False):
                res["fov"] = br.cam_dec(out[-1][1].float())[:, 0, 7:9]  # (fov_h, fov_w) of the pose encoding
        return res

    def _run(self, image: torch.Tensor, k: torch.Tensor | None) -> dict[str, torch.Tensor]:
        x = _prepare(self.ctx, image)
        outs = [self._branch(br, x) for br in self._branches()]
        return _finish(self.arch, outs, k, image.shape[-2:], x.shape[-2:])

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Depth (and mask / confidence / intrinsics) of a batch."""
        return self._run(image, intrinsics)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image, None)["depth"])


# ------------------------------------------------------------------------------------------------ kernels
_GRID_SIZES = 8  # fixed: cache policy, position / RoPE tables per backbone (one per patch grid; video always hits)
_PE_SIZES = 4  # fixed: cache policy, head position tables: two per input size (~25 MB at 504 x 756)


class _FastBackbone:
    """The DA3 DINOv2 forward for one view in fp16, with the same block maths as upstream.

    `dpt_fused` GEMMs with fused epilogues (qkv + bias; proj / fc2 + bias, * LayerScale (fp32), + residual; fc1 + bias +
    GELU), fastbb flash attention, cached RoPE and position tables; LayerNorm, qk-norm, RoPE and SwiGLU's gate in
    PyTorch.
    """

    def __init__(self, br: _Branch, k: DptFused, device: torch.device) -> None:
        vit: Any = br.backbone.pretrained
        if vit.embed_dim // vit.num_heads != HEAD_DIM:
            msg = f"fast DA3 ViT needs head_dim {HEAD_DIM}"
            raise ValueError(msg)
        self.k, self.flash = k, Flash64("hh")
        self.pos32 = vit.pos_embed.detach().float().to(device)  # upstream interpolates the table in fp32
        self.gammas = [tuple(g.detach().float().to(device) for g in (b.ls1.gamma, b.ls2.gamma)) for b in vit.blocks]
        self.vit = vit.half().to(device)
        self.layers, self.alt = br.cfg.layers, br.cfg.alt
        self.cat = br.cfg.dual
        self._pos: dict[tuple[int, int], torch.Tensor] = {}
        self._rope: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}

    def pos_embed(self, h0: int, w0: int) -> torch.Tensor:
        """Upstream `interpolate_pos_encoding` for an `h0 x w0` patch grid (fp32, cached as fp16)."""
        key = (h0, w0)
        if key not in self._pos:
            if len(self._pos) >= _GRID_SIZES:
                del self._pos[next(iter(self._pos))]
            vit, pe = self.vit, self.pos32
            m = int(math.sqrt(pe.shape[1] - 1))
            if m * m == h0 * w0 and h0 == w0:
                out = pe
            else:
                p = F.interpolate(
                    pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2),
                    scale_factor=((h0 + vit.interpolate_offset) / m, (w0 + vit.interpolate_offset) / m),
                    mode="bicubic",
                    antialias=vit.interpolate_antialias,
                )
                out = torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)
            self._pos[key] = out.half()
        return self._pos[key]

    def rope(self, h0: int, w0: int) -> tuple[torch.Tensor, ...]:
        """(cos, signed sin) `[N, 1, 64]` fp16 tables for local (patch grid) and global (all patches equal) positions.

        Upstream `RotaryPositionEmbedding2D`: dims 0-31 rotate with y, 32-63 with x; per half `t cos + rot(t) sin` with
        `rot(t) = (-t[16:], t[:16])`; positions are (y + 1, x + 1) for patches and (0, 0) for the camera token in local
        blocks, (1, 1) / (0, 0) in global ones.
        """
        key = (h0, w0)
        if key not in self._rope:
            if len(self._rope) >= _GRID_SIZES:
                del self._rope[next(iter(self._rope))]
            dev = self.pos32.device
            yy, xx = torch.meshgrid(torch.arange(h0, device=dev), torch.arange(w0, device=dev), indexing="ij")
            zero = torch.zeros(1, device=dev)
            local = (torch.cat([zero, yy.flatten() + 1.0]), torch.cat([zero, xx.flatten() + 1.0]))
            glob = (torch.cat([zero, torch.ones(h0 * w0, device=dev)]),) * 2
            vit: Any = self.vit
            freq = vit.rope.base_frequency
            inv = 1.0 / (freq ** (torch.arange(0, 32, 2, device=dev).float() / 32))
            sign = torch.cat([-torch.ones(16, device=dev), torch.ones(16, device=dev)])
            tables = []
            for py, px in (local, glob):
                ang = [torch.cat([p[:, None] * inv] * 2, -1) for p in (py, px)]  # [N, 32] each, angles duplicated
                cos = torch.cat([a.cos() for a in ang], -1)
                sin = torch.cat([a.sin() * sign for a in ang], -1)
                tables += [cos[:, None].half(), sin[:, None].half()]
            self._rope[key] = tuple(tables)
        return self._rope[key]

    @staticmethod
    def _rotate(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """`t cos + swap(t) sin`, swap exchanging the 16-dim quarters within each 32-dim half (sign in `sin`)."""
        sw = t.unflatten(-1, (2, 2, 16)).flip(-2).flatten(-3)
        return torch.addcmul(t * cos, sw, sin)

    def block(self, i: int, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        """Upstream `Block` `i` (eval); returns a new tensor (`x` is kept: the dual head reads earlier tokens)."""
        blk: Any = self.vit.blocks[i]
        g1, g2 = self.gammas[i]
        a, m, k = blk.attn, blk.mlp, self.k
        b, n, c = x.shape
        qkv = k.gemm(blk.norm1(x), a.qkv.weight, a.qkv.bias).view(b, n, 3, c // HEAD_DIM, HEAD_DIM)
        q, kk, v = qkv.unbind(2)
        if not isinstance(a.q_norm, nn.Identity):
            q, kk = a.q_norm(q), a.k_norm(kk)
        if rope is not None:
            q, kk = self._rotate(q, *rope), self._rotate(kk, *rope)
        o = flash_attention(q, kk, v, HEAD_DIM**-0.5, self.flash).view(b, n, c)
        x = k.gemm(o, a.proj.weight, a.proj.bias, scale=g1, res=x)
        y = blk.norm2(x)
        if isinstance(m, Mlp):
            y = k.gemm(y, m.fc1.weight, m.fc1.bias, act=k.gelu)
            return k.gemm(y, m.fc2.weight, m.fc2.bias, scale=g2, res=x, out=x)
        y1, y2 = k.gemm(y, m.w12.weight, m.w12.bias).chunk(2, dim=-1)  # SwiGLU (Giant)
        return k.gemm(F.silu(y1) * y2, m.w3.weight, m.w3.bias, scale=g2, res=x, out=x)

    def __call__(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """`[B, 3, h, w]` -> contiguous head tokens `[B, N, C or 2C]` per layer (no cls) and the last camera token."""
        vit: Any = self.vit
        h0, w0 = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        rl_cos, rl_sin, rg_cos, rg_sin = self.rope(h0, w0) if self.alt >= 0 else (None,) * 4
        outs, cam = [], None
        with torch.autocast("cuda", enabled=False), fp16_accumulate():
            t = vit.patch_embed(x.half())
            t = torch.cat([vit.cls_token.expand(t.shape[0], -1, -1), t], 1) + self.pos_embed(h0, w0)
            local = t
            for i in range(self.layers[-1] + 1):
                if i == self.alt:
                    t[:, 0] = vit.camera_token[0, 0]
                glob = self.alt >= 0 and i >= self.alt and i % 2 == 1
                rope = None if self.alt < 0 or i < self.alt else ((rg_cos, rg_sin) if glob else (rl_cos, rl_sin))
                t = self.block(i, t, rope)  # ty: ignore[invalid-argument-type]
                if not glob:
                    local = t
                if i in self.layers:
                    if self.cat:
                        outs.append(torch.cat([local[:, 1:], vit.norm(t[:, 1:])], -1))
                        cam = torch.cat([local[:, 0], t[:, 0]], -1)
                    else:
                        outs.append(vit.norm(t[:, 1:]))
        return outs, cam


class _FusedHead:
    """The DPT / DualDPT main branch on dense NHWC fp16 tensors with `dpt_fused` convs / GEMMs.

    As the DAv2 kernels head (`depth_anything_v2.fused`): projections as GEMMs on the token rows (+ the position
    embedding in the epilogue), kernel == stride deconvolutions as pixel-shuffle GEMMs, residual conv units as fused
    convs, the fusion 1x1 `out_conv` before the bilinear upsample (both linear). The 1x1 output convs run in fp32.
    """

    def __init__(self, br: _Branch, k: DptFused, device: torch.device) -> None:
        hd: Any = br.head
        self.k, self.dual, self.device = k, br.cfg.dual, device
        self.norm = hd.norm.half().to(device)
        self.pos_embed = bool(hd.pos_embed)
        self.projects = [_linear_1x1(p, device) for p in hd.projects]
        self.resize = [_resize(r, device) for r in hd.resize_layers]
        s = hd.scratch
        self.rn = [_conv(m, device) for m in (s.layer1_rn, s.layer2_rn, s.layer3_rn, s.layer4_rn)]
        self.fusion = []
        for f in (s.refinenet1, s.refinenet2, s.refinenet3, s.refinenet4):
            u1 = f.resConfUnit1
            rcu1 = (_conv(u1.conv1, device), _conv(u1.conv2, device)) if u1 is not None else None
            rcu2 = (_conv(f.resConfUnit2.conv1, device), _conv(f.resConfUnit2.conv2, device))
            self.fusion.append((rcu1, rcu2, _linear_1x1(f.out_conv, device)))
        self.oc1 = _conv(s.output_conv1, device)
        outs = [s.output_conv2] + ([] if self.dual else [s.sky_output_conv2])
        self.out = []
        for o in outs:
            last = o[2]
            w = last.weight.detach().float().flatten(1).t().contiguous().to(device)  # [C, out]
            self.out.append((_conv(o[0], device), w, last.bias.detach().float().to(device)))
        self._pe: dict[tuple[int, ...], torch.Tensor] = {}

    def _pos(self, c: int, h: int, w: int, hw: tuple[int, int], b: int) -> torch.Tensor:
        """Upstream `_add_pos_embed` table (fp32 maths) as a contiguous NHWC fp16 `[b, h, w, c]`."""
        key = (c, h, w, *hw, b)
        if key not in self._pe:
            if len(self._pe) >= _PE_SIZES:
                del self._pe[next(iter(self._pe))]
            uv = create_uv_grid(w, h, aspect_ratio=hw[1] / hw[0], dtype=torch.float32, device=self.device)
            pe = position_grid_to_embed(uv, c) * _POS_RATIO
            self._pe[key] = pe[None].half().expand(b, -1, -1, -1).contiguous()
        return self._pe[key]

    def _conv(self, x: torch.Tensor, c: tuple, **kw: Any) -> torch.Tensor:  # noqa: ANN401 - epilogue options
        w, b, pad, stride = c
        return self.k.conv(x, w, b, padding=pad, stride=stride, **kw)

    def _fuse(
        self, i: int, x0: torch.Tensor, x1: torch.Tensor | None, r: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """FeatureFusionBlock: `x0 (+ rcu1(x1))`, rcu2, out_conv, bilinear; `r` = relu(x1), or relu(x0) without x1."""
        u1, u2, (wo, bo) = self.fusion[i]
        if x1 is not None:
            if u1 is None:
                msg = "dpt_fused head: lateral input without a residual unit"
                raise ValueError(msg)
            t = self._conv(r, u1[0], act=ACT_RELU)
            r = torch.empty_like(x1)
            x0 = self._conv(t, u1[1], res=x1, res2=x0, y2=r)
        t = self._conv(r, u2[0], act=ACT_RELU)
        z = self._conv(t, u2[1], res=x0)
        return self.k.bilinear(self.k.gemm(z, wo, bo), size, align_corners=True)

    def __call__(self, feats: list[torch.Tensor], ph: int, pw: int) -> dict:
        """Tokens `[B, ph * pw, C]` of the 4 layers -> activated maps `[B, 14 ph, 14 pw]` (fp32)."""
        k = self.k
        b, hw = feats[0].shape[0], (ph * PATCH, pw * PATCH)
        ls, rs = [], []
        for t, (wp, bp), rz, rn in zip(feats, self.projects, self.resize, self.rn, strict=True):
            pos = self._pos(wp.shape[0], ph, pw, hw, b) if self.pos_embed else None
            x = k.gemm(self.norm(t), wp, bp, res=pos).view(b, ph, pw, -1)
            if rz is not None and rz[0] == "deconv":
                _, wg, bg, s = rz
                x = k.gemm(x, wg, bg, shuffle=(ph, pw, s, s))
            elif rz is not None:
                x = self._conv(x, rz[1:])
            r = x.new_empty((*x.shape[:3], rn[0].shape[0]))
            ls.append(self._conv(x, rn, y2=r))
            rs.append(r)
        l1, l2, l3, l4 = ls
        p = self._fuse(3, l4, None, rs[3], l3.shape[1:3])
        p = self._fuse(2, p, l3, rs[2], l2.shape[1:3])
        p = self._fuse(1, p, l2, rs[1], l1.shape[1:3])
        p = self._fuse(0, p, l1, rs[0], (2 * l1.shape[1], 2 * l1.shape[2]))
        p = k.bilinear(self._conv(p, self.oc1), hw, align_corners=True)
        if self.pos_embed:
            p += self._pos(p.shape[-1], *hw, hw, b)
        maps = []
        for c3, w, bias in self.out:
            o = self._conv(p, c3, act=ACT_RELU)
            maps.append(
                torch.addmm(bias, o.view(-1, o.shape[-1]).float(), w).view(*o.shape[:3], -1).permute(0, 3, 1, 2)
            )
        return _activate(self.dual, maps[0], None if self.dual else maps[1])


class Kernels(Runner):
    """`dpt_fused` kernels: fused-epilogue CUTLASS GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 head.

    The 1x1 output convs, activations and camera decoder run in fp32.
    """

    batch = None
    Params = Params
    intrinsics = "optional"  # Metric; set per variant in __init__

    @classmethod
    def input_use(cls, params: Params) -> dict[str, Any]:
        """Intrinsics: `"optional"` for the metric variants, else `"unused"`.

        Args:
            params: The variant's `Params`.
        """
        return _input_use(params)

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert it."""
        super().__init__(ctx)
        self.arch: Arch = ctx.params.arch
        self.intrinsics = self.input_use(ctx.params)["intrinsics"]  # ty: ignore[invalid-attribute-access] - per variant
        k = load_dpt_fused(ctx.device)
        net = _net(ctx, "cpu")  # converted on the CPU: the GPU holds only the fp16 network
        self.parts = []
        for br in [net.main] + ([net.metric] if net.metric is not None else []):
            head = _FusedHead(br, k, ctx.device)
            cam = br.cam_dec.float().to(ctx.device) if br.cam_dec is not None else None
            self.parts.append((cam, _FastBackbone(br, k, ctx.device), head))

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Depth (and mask / confidence / intrinsics) of a batch."""
        x = _prepare(self.ctx, image)
        ph, pw = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        outs = []
        for cam_dec, vit, head in self.parts:
            feats, cam = vit(x)
            with torch.autocast("cuda", enabled=False):
                res = head(feats, ph, pw)
                if cam_dec is not None and cam is not None:
                    res["fov"] = cam_dec(cam[:, None].float())[:, 0, 7:9]
            outs.append(res)
        return _finish(self.arch, outs, intrinsics, image.shape[-2:], x.shape[-2:])


# after the portable part: kernels-only dependencies
from fastforks.depth.models.depth_anything_v2.fused import _conv, _linear_1x1, _resize  # noqa: E402 - shared DPT conversions
from fastforks.kernels.dpt_fused import ACT_RELU, DptFused  # noqa: E402
from fastforks.kernels.dpt_fused import load as load_dpt_fused  # noqa: E402
from fastforks.kernels.fastbb import Flash64  # noqa: E402
from fastforks.shared.depth_anything_v2 import flash_attention, fp16_accumulate  # noqa: E402

RUNNERS = {"portable": Portable, "kernels": Kernels}
