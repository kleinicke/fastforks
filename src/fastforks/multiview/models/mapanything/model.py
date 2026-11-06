"""MapAnything runners: weights -> network and the portable / kernels forward passes (images-only inference).

A lean re-implementation of upstream `MapAnything.forward` + `postprocess_model_outputs_for_inference` (defaults:
apply_mask, mask_edges) for image inputs only: the geometric input encoders (rays, depth, poses, scales) are not built
(their weights are skipped; without such inputs upstream adds nothing from them). Pieces, as the checkpoint:
- encoder: DINOv2 ViT-g/14 (24 of its 40 blocks, SwiGLU MLP, LayerScale, no registers, no final norm), bicubic
  position-embedding interpolation as DINOv2;
- fusion LayerNorm on the patch features; info sharing: 16 alternating-attention blocks (even: global over all views
  + a scale token, odd: per view) on [patches, cls] per view, the first view tagged with `view_pos_table[0]`;
- dense head: DPT on [fused encoder features, info-sharing layers 7, 11, final] -> 6 channels per pixel (ray
  direction, log depth along the ray, confidence, non-ambiguity logit); pose head (cam-to-world translation +
  quaternion per view) and scale head (one metric scale), all scaled by it.
Outputs: z depth, world points (first camera = world, OpenCV axes, metres), confidence, the mask (non-ambiguous and not
a depth / normal edge: upstream `mask_edges`, run batched on the device), intrinsics from the ray directions (least
squares on a pixel grid as upstream) and cam_to_world.

Images are resized to the network size (long side `long_side`, both sides multiples of 14, bicubic antialiased;
upstream's `load_images` instead crops to fixed aspect-ratio sizes) unless they already have it; maps are resized back
(bilinear; the mask nearest) and the intrinsics rescaled to the input pixels (pixel centres at integer coordinates).

The fp16 and kernels paths are `mapanything/fast.py` (`FastMapAnything`): encoder
fp16 (weights, stream, attention); info sharing fp16 GEMMs / attention with an fp32 residual stream, LayerScale and
final norm (upstream: bf16 autocast); DPT fp16 channels-last with the last 1x1 conv fp32; pose / scale heads fp32 (as
upstream). Kernels: the `fastbb` head-dim-64 flash kernel (fp16 accumulation) in every attention, fp16-accumulate
GEMMs in the encoder except the SwiGLU down projection (`kernels`: info sharing keeps fp32 accumulation).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks.multiview._runner import Forward, Runner

if TYPE_CHECKING:
    from fastforks.multiview._runner import Context

PATCH = 14  # fixed: the DINOv2 ViT-g patch size
DIM = 1536  # fixed: the DINOv2 ViT-g width (one checkpoint)
HEADS = 24  # fixed: the DINOv2 ViT-g heads (one checkpoint)
MEAN = (0.485, 0.456, 0.406)  # DINOv2 input normalisation
STD = (0.229, 0.224, 0.225)
INTERMEDIATE = (7, 11)  # info-sharing blocks whose (normed) output feeds the DPT
UNUSED = ("depth_encoder.", "ray_dirs_encoder.", "cam_rot_encoder.", "cam_trans_encoder.", "cam_trans_scale_encoder.",
          "depth_scale_encoder.")  # fmt: skip
_POS_SIZES = 8  # fixed: cache policy, position tables kept (one per patch grid; video at one size always hits)
EDGE_NORMAL_DEG = 5.0  # fixed: upstream infer() default
EDGE_DEPTH_RTOL = 0.03  # fixed: upstream infer() default
HIGH_RES = 1_000_000  # fixed: upstream recovers intrinsics from 5 rays above this many pixels
SMALL_WORK = 2.2e5  # fixed: B * N * C below this: torch SDPA (fastbb's threshold)


@dataclass(frozen=True)
class Params:
    """Load options."""

    long_side: int  # network long side (rounded to a multiple of 14)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Network input size for an `h x w` image: long side `long_side`, both sides multiples of 14."""
    f = long_side / max(h, w)
    hn, wn = (max(PATCH, round(v * f / PATCH) * PATCH) for v in (h, w))
    return hn, wn


# ------------------------------------------------------------------------------------- weights (checkpoint names)
class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)


class _LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(DIM))


class _SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w12 = nn.Linear(DIM, 2 * 4096)
        self.w3 = nn.Linear(4096, DIM)


class _Block(nn.Module):
    """DINOv2 / uniception self-attention block: x + ls1 * attn(norm1 x), then x + ls2 * swiglu(norm2 x)."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-6)
        self.attn = _Attn()
        self.ls1 = _LayerScale()
        self.norm2 = nn.LayerNorm(DIM, eps=1e-6)
        self.mlp = _SwiGLU()
        self.ls2 = _LayerScale()


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, PATCH, PATCH)


class _ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.empty(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.empty(1, 37 * 37 + 1, DIM))
        self.blocks = nn.ModuleList(_Block() for _ in range(24))


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ViT()


class _InfoSharing(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention_blocks = nn.ModuleList(_Block() for _ in range(16))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)
        self.view_pos_table = nn.Parameter(torch.empty(1, DIM))


class _RCU(nn.Module):
    """DPT residual conv unit: x + conv2(relu(conv1(relu(x))))."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class _Fusion(nn.Module):
    """DPT feature fusion: (x + rcu1(skip)) -> rcu2 -> 2x bilinear (align_corners) -> 1x1 conv."""

    def __init__(self, *, skip: bool) -> None:
        super().__init__()
        if skip:
            self.resConfUnit1 = _RCU()
        self.resConfUnit2 = _RCU()
        self.out_conv = nn.Conv2d(256, 256, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(x)


def _reassemble(dim: int, up: nn.Module | None) -> nn.Sequential:
    first = [nn.Conv2d(DIM, dim, 1)] + ([up] if up is not None else [])
    return nn.Sequential(nn.Sequential(*first), nn.Conv2d(dim, 256, 3, padding=1, bias=False))


class _DPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_process = nn.ModuleList(
            [
                _reassemble(96, nn.ConvTranspose2d(96, 96, 4, 4)),
                _reassemble(192, nn.ConvTranspose2d(192, 192, 2, 2)),
                _reassemble(384, None),
                _reassemble(768, nn.Conv2d(768, 768, 3, 2, 1)),
            ]
        )
        self.scratch = nn.Module()
        for i in (1, 2, 3, 4):
            self.scratch.add_module(f"refinenet{i}", _Fusion(skip=i != 4))  # noqa: PLR2004 - refinenet4 has no skip input

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """4 `[N, 1536, h, w]` features -> `[N, 256, 8h, 8w]` (2x the stride-4 level)."""
        s: Any = self.scratch
        l0, l1, l2, l3 = (p(f) for p, f in zip(self.input_process, feats, strict=True))
        p4 = s.refinenet4(l3)[:, :, : l2.shape[2], : l2.shape[3]]
        return s.refinenet1(s.refinenet2(s.refinenet3(p4, l2), l1), l0)


class _Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 6, 1))


class _ResConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.res_conv1 = nn.Conv2d(784, 784, 1)
        self.res_conv2 = nn.Conv2d(784, 784, 1)
        self.res_conv3 = nn.Conv2d(784, 784, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.res_conv3(F.relu(self.res_conv2(F.relu(self.res_conv1(x))))))


class _PoseHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(DIM, 784, 1)
        self.res_conv = nn.ModuleList([_ResConv(), _ResConv()])
        self.more_mlps = nn.Sequential(nn.Linear(784, 784), nn.ReLU(), nn.Linear(784, 784), nn.ReLU())
        self.fc_t = nn.Linear(784, 3)
        self.fc_rot = nn.Linear(784, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[V, 1536, h, w]` -> cam-to-world translation `[V, 3]` and unit quaternion (x, y, z, w) `[V, 4]`."""
        x = self.proj(x)
        for blk in self.res_conv:
            x = blk(x)
        x = self.more_mlps(x.mean((2, 3)))
        q = self.fc_rot(x)
        return self.fc_t(x), q / q.norm(dim=1, keepdim=True).clip(min=1e-8)


class _ScaleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, 196)
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(196, 196), nn.ReLU()) for _ in range(2))
        self.output_proj = nn.Linear(196, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale token `[1536]` -> metric scale (scalar tensor)."""
        x = self.proj(x)
        for layer in self.mlp:
            x = layer(x)
        return torch.exp(self.output_proj(x)[0]).clip(min=1e-8)


class Net(nn.Module):
    """The image-only parts of upstream `MapAnything` with its parameter names."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.encoder = _Encoder()
        self.fusion_norm_layer = nn.LayerNorm(DIM, eps=1e-6)
        self.scale_token = nn.Parameter(torch.empty(DIM))
        self.info_sharing = _InfoSharing()
        self.dense_head = nn.Sequential(_DPT(), _Regressor())
        self.pose_head = _PoseHead()
        self.scale_head = _ScaleHead()


def _net(ctx: Context) -> Net:
    with torch.device("meta"):
        net = Net()
    sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith(UNUSED)}
    net.load_state_dict(sd, strict=True, assign=True)
    return net.eval().requires_grad_(requires_grad=False)


# ---------------------------------------------------------------------------------------------- geometry
def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Quaternions (x, y, z, w) `[V, 4]` -> rotation matrices `[V, 3, 3]` (upstream `quaternion_to_rotation_matrix`)."""
    x, y, z, w = (q / q.norm(dim=1, keepdim=True)).unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).view(-1, 3, 3)  # fmt: skip


def intrinsics_from_rays(rays: torch.Tensor) -> torch.Tensor:
    """Pinhole intrinsics `[V, 3, 3]` from unit ray directions `[V, H, W, 3]`.

    Upstream `recover_pinhole_intrinsics_from_ray_directions`: least squares `x = cx + fx dx/dz` on a ~50 x 50 pixel
    grid, or a 5-ray estimate above 1e6 pixels; pixel centres at integer coordinates.
    """
    v, h, w, _ = rays.shape
    if h * w > HIGH_RES:
        ch, cw = h // 2, w // 2
        qw, tw, qh, th = w // 4, 3 * w // 4, h // 4, 3 * h // 4

        def r(i: int, j: int) -> torch.Tensor:
            t = rays[:, i, j]
            return t / t[:, 2:3]

        c = r(ch, cw)
        fx = ((qw - cw) / (r(ch, qw)[:, 0] - c[:, 0]) + (tw - cw) / (r(ch, tw)[:, 0] - c[:, 0])) / 2
        fy = ((qh - ch) / (r(qh, cw)[:, 1] - c[:, 1]) + (th - ch) / (r(th, cw)[:, 1] - c[:, 1])) / 2
        cx, cy = cw - fx * c[:, 0], ch - fy * c[:, 1]
    else:
        hi = torch.arange(0, h, max(1, h // 50), device=rays.device)
        wi = torch.arange(0, w, max(1, w // 50), device=rays.device)
        s = rays[:, hi[:, None], wi[None, :]].reshape(v, -1, 3)
        xs = wi.float()[None, :].expand(len(hi), -1).reshape(1, -1, 1).expand(v, -1, -1)
        ys = hi.float()[:, None].expand(-1, len(wi)).reshape(1, -1, 1).expand(v, -1, -1)
        ones = torch.ones_like(s[..., 0])

        def solve(ratio: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Least squares `b = c + f * ratio` -> (c, f): 2x2 normal equations, closed form (ONNX has no solve)."""
            a = torch.stack([ones, ratio], 2)
            m = a.transpose(1, 2) @ a  # [V, 2, 2]
            r = (a.transpose(1, 2) @ b)[..., 0]  # [V, 2]
            det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
            return (m[:, 1, 1] * r[:, 0] - m[:, 0, 1] * r[:, 1]) / det, (
                m[:, 0, 0] * r[:, 1] - m[:, 1, 0] * r[:, 0]
            ) / det

        cx, fx = solve(s[..., 0] / s[..., 2], xs)
        cy, fy = solve(s[..., 1] / s[..., 2], ys)
    k = torch.zeros(v, 3, 3, device=rays.device)
    k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = fx, fy, cx, cy, 1.0
    return k


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """`x` `[N, H + 2, W + 2, ...]` (padded) -> the `[N, H, W, ...]` window shifted by (dy, dx) in {-1, 0, 1}."""
    h, w = x.shape[1] - 2, x.shape[2] - 2
    return x[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-12)


def _maxpool_nan(x: torch.Tensor) -> torch.Tensor:
    """3x3 / stride-1 max pool ignoring NaN (numpy nanmax with NaN padding); all-NaN windows -> -inf."""
    x = torch.where(torch.isnan(x), float("-inf"), x)
    return F.max_pool2d(x[:, None], 3, 1, 1)[:, 0]


def edge_mask(pts: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Not-an-edge mask: points `[N, H, W, 3]`, z depth `[N, H, W]`, mask `[N, H, W]` bool -> `[N, H, W]` bool.

    Upstream `~(depth_edge(depth, 0.03, mask) & normals_edge(points_to_normals(pts, mask), 5 deg))` (MoGe utils),
    batched on the device (`edge_mask_gpu`).
    """
    p = F.pad(pts.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)  # zero padding
    mp = F.pad(mask[:, None].float(), (1, 1, 1, 1))[:, 0] > 0.5  # noqa: PLR2004 - False padding
    c = _shift(p, 0, 0)
    up, left, down, right = (_shift(p, -1, 0) - c, _shift(p, 0, -1) - c, _shift(p, 1, 0) - c, _shift(p, 0, 1) - c)
    mc = _shift(mp, 0, 0)
    mu, ml, md, mr = _shift(mp, -1, 0), _shift(mp, 0, -1), _shift(mp, 1, 0), _shift(mp, 0, 1)
    normal = torch.zeros_like(pts)
    nmask = torch.zeros_like(mask)
    for a, b, va in ((up, left, mu & ml), (left, down, ml & md), (down, right, md & mr), (right, up, mr & mu)):
        vm = va & mc
        normal += _unit(torch.linalg.cross(a, b, dim=-1)) * vm[..., None]
        nmask |= vm
    normal = torch.where(nmask[..., None], _unit(normal), 0.0)
    # normals_edge: replicate padding; upstream's mask window is transposed relative to the normal window
    # (sliding_window_2d over axes (-3, -2) of a 2-D mask -> axes (1, 0)), reproduced here
    n = _unit(normal)
    npad = F.pad(n.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)
    mpad = F.pad(nmask[:, None].float(), (1, 1, 1, 1), mode="replicate")[:, 0] > 0.5  # noqa: PLR2004
    ang = torch.full_like(depth, float("-inf"))
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            dot = (n * _shift(npad, a, b)).sum(-1)
            ang = torch.maximum(ang, torch.where(_shift(mpad, b, a), torch.arccos(dot), 0.0))  # NaN propagates (np.max)
    normal_edges = _maxpool_nan(ang) > math.radians(EDGE_NORMAL_DEG)
    inf = torch.tensor(float("-inf"), device=depth.device)
    diff = (F.max_pool2d(torch.where(mask, depth, inf)[:, None], 3, 1, 1)
            + F.max_pool2d(torch.where(mask, -depth, inf)[:, None], 3, 1, 1))[:, 0]  # fmt: skip
    return ~((diff / depth > EDGE_DEPTH_RTOL) & normal_edges)


# ---------------------------------------------------------------------------------------------- runners
def _fp16_accumulate(*, on: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if on else contextlib.nullcontext()


class _Blk:
    """Prepared parameters of one `_Block`: GEMM weights and norms in `dt`, LayerScales in `gdt`, on `device`."""

    def __init__(self, b: _Block, dt: torch.dtype, gdt: torch.dtype, device: torch.device) -> None:
        def cv(t: torch.Tensor, d: torch.dtype = dt) -> torch.Tensor:
            return t.detach().to(device=device, dtype=d).contiguous()

        self.n1 = (cv(b.norm1.weight), cv(b.norm1.bias))
        self.n2 = (cv(b.norm2.weight), cv(b.norm2.bias))
        self.wqkv, self.bqkv = cv(b.attn.qkv.weight), cv(b.attn.qkv.bias)
        self.wp, self.bp = cv(b.attn.proj.weight), cv(b.attn.proj.bias)
        self.w12, self.b12 = cv(b.mlp.w12.weight), cv(b.mlp.w12.bias)
        self.w3, self.b3 = cv(b.mlp.w3.weight), cv(b.mlp.w3.bias)
        self.g1, self.g2 = cv(b.ls1.gamma, gdt), cv(b.ls2.gamma, gdt)


class _Base(Runner):
    intrinsics = "unused"
    cam_to_world = "unused"
    depth_range = "unused"
    per_view = True
    min_views = 1
    batch = None
    Params = Params
    gacc = False  # fp16-accumulate GEMMs in the encoder (except the SwiGLU w3)
    dense_chunk = 8  # views per DPT chunk

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        net: Any = _net(ctx)
        dev = ctx.device
        half = ctx.dtype != "float32"
        self.dt = torch.float16 if half else torch.float32
        vit: Any = net.encoder.model
        self.enc = [_Blk(b, self.dt, self.dt, dev) for b in vit.blocks]  # encoder fully fp16 (stream, gammas)
        self.info = [_Blk(b, self.dt, torch.float32, dev) for b in net.info_sharing.self_attention_blocks]
        for mod in (vit.blocks, net.info_sharing.self_attention_blocks):  # the prepared copies replace them
            for p in mod.parameters():
                p.data = torch.empty(0, dtype=p.dtype)
        self.net = net.to(dev)
        self.pe = (vit.patch_embed.proj.weight.to(self.dt), vit.patch_embed.proj.bias.to(self.dt))
        self.cls = vit.cls_token.to(self.dt)
        self._pos: dict[tuple[int, int], torch.Tensor] = {}
        self.mean = torch.tensor(MEAN, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        reg: Any = net.dense_head[1]
        last = reg.conv2[2]
        self.head_w, self.head_b = last.weight.flatten(1).float(), last.bias.float()  # final 1x1 conv: fp32 GEMM
        reg.conv2[2] = nn.Identity().eval()
        if half:
            net.dense_head.half().to(memory_format=torch.channels_last)

    # ------------------------------------------------------------------ transformer
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`[B, N, heads, 64]` q / k / v -> `[B, N, heads * 64]` (torch SDPA)."""
        b, n, h, d = q.shape
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).reshape(b, n, h * d)

    def block(self, p: _Blk, x: torch.Tensor, *, w3_fp32acc: bool = False) -> torch.Tensor:
        """One block on `x` `[B, N, C]` (stream dtype: `dt` in the encoder, fp32 in info sharing) -> a new tensor."""
        b, n, c = x.shape
        y = F.layer_norm(x.to(self.dt), (c,), *p.n1, 1e-6)
        q, k, v = F.linear(y, p.wqkv, p.bqkv).view(b, n, 3, HEADS, c // HEADS).unbind(2)
        x = torch.addcmul(x, F.linear(self.attention(q, k, v), p.wp, p.bp), p.g1)
        x1, x2 = F.linear(F.layer_norm(x.to(self.dt), (c,), *p.n2, 1e-6), p.w12, p.b12).chunk(2, -1)
        h = F.silu(x1).mul_(x2)
        if w3_fp32acc:  # "now3": the K = 4096 down projection accumulates in fp32
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            o = F.linear(h, p.w3, p.b3)
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        else:
            o = F.linear(h, p.w3, p.b3)
        return torch.addcmul(x, o, p.g2) if torch.is_grad_enabled() else x.addcmul_(o, p.g2)

    def pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        """DINOv2 `interpolate_pos_encoding` (bicubic on the fp32 37 x 37 table, offset 0.1) in `dt`."""
        if (gh, gw) not in self._pos:
            if len(self._pos) >= _POS_SIZES:
                del self._pos[next(iter(self._pos))]
            pe = self.net.encoder.model.pos_embed.float()
            m = 37
            if gh * gw == m * m and gh == gw:
                out = pe
            else:
                grid = pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2)
                g = F.interpolate(grid, scale_factor=((gh + 0.1) / m, (gw + 0.1) / m), mode="bicubic")
                out = torch.cat([pe[:, :1], g.permute(0, 2, 3, 1).flatten(1, 2)], 1)
            self._pos[gh, gw] = out.to(self.dt)
        return self._pos[gh, gw]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised images `[V, 3, H, W]` -> DINOv2 tokens `[V, 1 + P, C]` (cls first; no final norm) in `dt`."""
        t = F.conv2d(x.to(self.dt), *self.pe, stride=PATCH)
        v, _, gh, gw = t.shape
        t = torch.cat([self.cls.expand(v, -1, -1), t.flatten(2).transpose(1, 2)], 1) + self.pos_embed(gh, gw)
        with _fp16_accumulate(on=self.gacc):
            for p in self.enc:
                t = self.block(p, t, w3_fp32acc=self.gacc)
        return t

    def share(self, feats: torch.Tensor, cls: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Info sharing on fused features `[V, P, C]` + cls `[V, C]` (fp32).

        Returns the normed outputs of blocks 7, 11, 15 as `[V, P + 1, C]` and the final scale token `[C]`.
        """
        info: Any = self.net.info_sharing
        v, n, c = feats.shape
        t = torch.cat([feats, cls[:, None]], 1)
        t[0] += info.view_pos_table[0]
        x = torch.cat([t.view(1, v * (n + 1), c), self.net.scale_token.view(1, 1, c)], 1)
        outs = []
        for i, p in enumerate(self.info):
            if i % 2 == 0:  # global: all views + the scale token
                x = self.block(p, x)
            else:  # per view: the scale token sits out
                y = self.block(p, x[:, :-1].reshape(v, n + 1, c))
                x = torch.cat([y.view(1, -1, c), x[:, -1:]], 1)
            if i in INTERMEDIATE or i == len(self.info) - 1:
                outs.append(info.norm(x))
        return [o[0, :-1].view(v, n + 1, c) for o in outs], outs[-1][0, -1]

    # ------------------------------------------------------------------ heads
    def dense(self, feats: list[torch.Tensor], h: int, w: int) -> torch.Tensor:
        """DPT on 4 `[V, C, gh, gw]` features -> raw head output `[V, h, w, 6]` (fp32)."""
        dh: Any = self.net.dense_head
        if self.dt == torch.float16:
            feats = [f.half().contiguous(memory_format=torch.channels_last) for f in feats]
        outs = []
        for s in range(0, feats[0].shape[0], self.dense_chunk):
            y = dh[0]([f[s : s + self.dense_chunk] for f in feats])
            y = F.interpolate(dh[1].conv1(y), (h, w), mode="bilinear", align_corners=True)
            y = dh[1].conv2(y)  # 3x3 conv + ReLU (the last 1x1 conv is head_w / head_b)
            outs.append(F.linear(y.permute(0, 2, 3, 1).float(), self.head_w, self.head_b))
        return torch.cat(outs) if len(outs) > 1 else outs[0]

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Images `[V, 3, h, w]` in [0, 1] (network size) -> the maps at the network size."""
        v, _, h, w = x.shape
        gh, gw = h // PATCH, w // PATCH
        tok = self.encode((x - self.mean) / self.std)
        net = self.net
        feats = F.layer_norm(tok[:, 1:].float(), (DIM,), net.fusion_norm_layer.weight, net.fusion_norm_layer.bias, 1e-6)
        layers, scale_tok = self.share(feats, tok[:, 0].float())

        def nchw(t: torch.Tensor) -> torch.Tensor:  # [V, P(+1), C] -> [V, C, gh, gw]
            return t[:, : gh * gw].reshape(v, gh, gw, DIM).permute(0, 3, 1, 2)

        final = nchw(layers[-1])
        o = self.dense([nchw(feats), *(nchw(t) for t in layers[:-1]), final], h, w)
        trans, quat = net.pose_head(final.float().contiguous())
        scale = net.scale_head(scale_tok.float())
        rays = o[..., :3] / o[..., :3].norm(dim=-1, keepdim=True).clip(min=1e-8)
        pts_cam = rays * torch.exp(o[..., 3:4])
        rot = _quat_to_rot(quat)
        pts = torch.einsum("vij,vhwj->vhwi", rot, pts_cam) + trans[:, None, None]
        pts, pts_cam = pts * scale, pts_cam * scale
        depth = pts_cam[..., 2]
        mask = torch.sigmoid(o[..., 5]) > 0.5  # noqa: PLR2004 - upstream's threshold on the non-ambiguity probability
        mask &= edge_mask(pts, depth, mask)
        c2w = torch.eye(4, device=x.device).repeat(v, 1, 1)
        c2w[:, :3, :3], c2w[:, :3, 3] = rot, trans * scale
        return {
            "depth": depth,
            "confidence": 1 + torch.exp(o[..., 4]),
            "mask": mask,
            "world_points": pts,
            "intrinsics": intrinsics_from_rays(rays),
            "cam_to_world": c2w,
            "metric_scale": scale.expand(v).clone(),
        }

    # ------------------------------------------------------------------ runner interface
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,  # noqa: ARG002 - unused input of the common signature
        cam_to_world: torch.Tensor | None = None,  # noqa: ARG002
        depth_range: torch.Tensor | None = None,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """The runner's inputs -> the maps of `__call__`."""
        h0, w0 = images.shape[-2:]
        hn, wn = network_size(h0, w0, self.ctx.params.long_side)
        x = images.float()
        if (hn, wn) != (h0, w0):
            x = F.interpolate(x, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        with torch.autocast(x.device.type, enabled=False):
            out = self.predict(x)
        if (hn, wn) == (h0, w0):
            return out

        def up(t: torch.Tensor) -> torch.Tensor:  # [V, h, w(, c)] -> input size
            y = t.unsqueeze(1) if t.ndim == 3 else t.permute(0, 3, 1, 2)  # noqa: PLR2004
            y = F.interpolate(y, (h0, w0), mode="bilinear", align_corners=False)
            return y[:, 0] if t.ndim == 3 else y.permute(0, 2, 3, 1)  # noqa: PLR2004

        k = out["intrinsics"].clone()
        sx, sy = w0 / wn, h0 / hn
        k[:, 0, 0] *= sx
        k[:, 1, 1] *= sy
        k[:, 0, 2] = (k[:, 0, 2] + 0.5) * sx - 0.5
        k[:, 1, 2] = (k[:, 1, 2] + 0.5) * sy - 0.5
        mask = F.interpolate(out["mask"][:, None].float(), (h0, w0), mode="nearest")[:, 0] > 0.5  # noqa: PLR2004
        return dict(out, depth=up(out["depth"]), confidence=up(out["confidence"]),
                    world_points=up(out["world_points"]), mask=mask, intrinsics=k)  # fmt: skip

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Depth, points and cameras of every view, in the frame of `images[0]` (metric)."""
        return self.forward(images, intrinsics, cam_to_world, depth_range)


class Portable(_Base):
    """Plain PyTorch: float32 (CPU / CUDA) or float16 (CUDA) = friendly (fp16 GEMMs, SDPA, fp32 heads)."""

    dtypes = ("float32", "float16")

    def module(self) -> torch.nn.Module:
        """The network with its input handling: the runner's inputs -> the maps of `__call__`."""
        return Forward(self.net, lambda _net, *inputs: self.forward(*inputs))


class Kernels(_Base):
    """`fastbb` flash attention (fp16 accumulation) everywhere, fp16-accumulate encoder GEMMs, not w3."""

    gacc = True

    def __init__(self, ctx: Context) -> None:  # noqa: D107
        super().__init__(ctx)
        from fastforks.kernels import fastbb  # noqa: PLC0415 - kernels only

        fastbb.load(ctx.device)
        self.flash, self.check_flash64 = fastbb.Flash64("hh"), fastbb.check_flash64

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`fastbb` flash64 (fp16, fp16 P.V accumulation) on strided `[B, N, heads, 64]` views; SDPA if small."""
        b, n, h, d = q.shape
        if b * n * h * d < SMALL_WORK:
            return super().attention(q, k, v)
        self.check_flash64(q, k, v)
        return self.flash(q, k, v, d**-0.5).view(b, n, h * d)


RUNNERS = {"portable": Portable, "kernels": Kernels}
