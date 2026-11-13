"""Any2Full network (upstream `model/ours`): Depth Anything V2 plus a sparse-depth prompt branch.

Upstream (Apache-2.0) keeps the DINOv2 blocks and DPT head and adds, at the last three of the four feature layers, a
prompt stream: `SparseDepthEmbed` turns the normalised sparse disparity into one token per patch (the patch's values
mean-pooled to 14x14 / 7x7 / 3x3 grids, empty cells filled from their neighbours, one strided conv per grid, summed,
plus a class token and positions, modulated by the ViT tokens entering the second feature layer). At each of those
layers an attention takes q / k from the ViT block's own normalised input (`prompt_depth_qk`) and the values from the
prompt (the first one masks the patches without depth), then an MLP; the layer's features are modulated by the
normalised prompt before the unchanged DPT head.

Same parameter names as upstream inside each part; `checkpoint_keys` maps the checkpoint. `Any2FullNet.forward` is the
network on prepared inputs (ImageNet-normalised image at the network size, normalised disparity); the input
preparation and the least-squares alignment of the output are in `model.py`. Exportable: the per-patch pooling are
fixed matrices, the hole filling a fixed number of steps (the most a grid needs; upstream stops when nothing changes,
the same result).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.depth_anything_v2 import DPTHead
from fastforks.shared.depth_anything_v2.net.dinov2 import DINOv2, DinoVisionTransformer
from fastforks.shared.depth_anything_v2.net.dinov2_layers.layer_scale import LayerScale
from fastforks.shared.depth_anything_v2.net.dinov2_layers.mlp import Mlp

PATCH = 14  # fixed: the DINOv2 ViT-*/14 patch
GRIDS = (14, 7, 3)  # fixed: upstream pools each patch to these grids (patch, patch // 2, patch // 4)
_MASK_EPS = 1e-6  # fixed: upstream's "has depth" test, value > per-sample minimum + eps
_POOL_EPS = 1e-5  # fixed: upstream's pooled-mean denominator eps and pooled-mask validity threshold
CONFIGS = {  # upstream model_configs + intermediate_layer_idx
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384], "layers": (2, 5, 8, 11)},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768], "layers": (2, 5, 8, 11)},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024], "layers": (4, 11, 17, 23)},
}


def _mlp2(i: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h))


def _pool_matrix(g: int) -> torch.Tensor:
    """`adaptive_avg_pool` of a 14-vector to `g` bins as a `[g, 14]` matrix (bins may overlap, as torch's)."""
    m = torch.zeros(g, PATCH)
    for i in range(g):
        lo, hi = i * PATCH // g, -(-(i + 1) * PATCH // g)
        m[i, lo:hi] = 1 / (hi - lo)
    return m


def _fill(heat: torch.Tensor, valid: torch.Tensor, steps: int) -> torch.Tensor:
    """Upstream `efficient_nearest_fill`: each step gives the cells next to valid ones their 3x3 neighbours' mean."""
    k = heat.new_ones((1, 1, 3, 3))
    for _ in range(steps):
        v = valid.to(heat.dtype)
        cnt = F.conv2d(v, k, padding=1)
        grow = (cnt > 0) & ~valid
        heat = torch.where(grow, F.conv2d(heat * v, k, padding=1) / cnt.clamp_min(1), heat)
        valid = valid | grow
    return heat


class PromptBranch(nn.Module):
    """The prompt parts of one upstream `PromptNestedTensorBlock` (attention `prompt_depth_*`, the MLP)."""

    def __init__(self, dim: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)  # attn.prompt_depth_norm (torch default eps, as upstream)
        self.bias = nn.Linear(dim, dim)  # attn.prompt_depth_bias: the values
        self.proj = nn.Linear(dim, dim)  # attn.prompt_depth_proj
        self.qk = nn.Linear(dim, 2 * dim)  # attn.prompt_depth_qk
        self.mlp = Mlp(dim, 4 * dim)  # prompt_depth_mlp
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)  # prompt_depth_norm2
        self.ls2 = LayerScale(dim, init_values=1.0)  # prompt_depth_ls2

    def forward(self, x1: torch.Tensor, p: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Prompt `p` `[B, N, C]` after this block, `x1` the block's normalised input, `mask` `[B, 1, 1, N]` keys."""
        b, n, c = p.shape
        h = self.heads
        qk = self.qk(x1).reshape(b, n, 2, h, c // h).permute(2, 0, 3, 1, 4)
        v = self.bias(self.norm(p)).reshape(b, n, h, c // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(qk[0], qk[1], v, attn_mask=mask)
        p = p + self.proj(o.transpose(1, 2).reshape(b, n, c))
        return p + self.ls2(self.mlp(self.norm2(p)))


class DepthEmbed(nn.Module):
    """Upstream `SparseDepthEmbed`: per-grid patch convs, a 16x16 position table (+ class slot), the RGB-D modulation.

    Carries the attributes `DinoVisionTransformer.interpolate_pos_encoding` reads (same interpolation as upstream).
    """

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.depth_heatmap_encoder = nn.ModuleList([nn.Conv2d(2, dim, g, stride=g) for g in GRIDS])
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // PATCH) ** 2 + 1, dim))
        self.rgbd_proj_s = _mlp2(2 * dim, dim)
        self.rgbd_proj_b = _mlp2(2 * dim, dim)
        self.patch_size = PATCH
        self.interpolate_offset = 0.1
        self.interpolate_antialias = False
        for g in GRIDS[1:]:
            self.register_buffer(f"pool{g}", _pool_matrix(g), persistent=False)

    def forward(self, d: torch.Tensor, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised disparity `[B, 1, H, W]`, ViT tokens `[B, 1 + N, C]` -> prompt tokens, key mask `[B, 1 + N]`."""
        b, _, hh, ww = d.shape
        gh, gw = hh // PATCH, ww // PATCH
        m = (d > d.amin((1, 2, 3), keepdim=True) + _MASK_EPS).to(d.dtype)
        dp = (d * m).view(b, gh, PATCH, gw, PATCH)
        mp = m.view(b, gh, PATCH, gw, PATCH)
        cols = []
        for g in GRIDS:
            if g == PATCH:
                xs, ms = dp.permute(0, 1, 3, 2, 4), mp.permute(0, 1, 3, 2, 4)
            else:
                pm = self.get_buffer(f"pool{g}")
                xs = torch.einsum("bihjw,uh,vw->bijuv", dp, pm, pm)
                ms = torch.einsum("bihjw,uh,vw->bijuv", mp, pm, pm)
            ms = ms.reshape(-1, 1, g, g)
            heat = _fill(xs.reshape(-1, 1, g, g) / (ms + _POOL_EPS), ms >= _POOL_EPS, g - 1)
            cols += [heat.view(b, gh * gw, g * g), ms.view(b, gh * gw, g * g)]
        enc: list[torch.Tensor] = [c.get_parameter(n) for c in self.depth_heatmap_encoder for n in ("weight", "bias")]
        w = torch.cat([v.flatten(1) for v in enc[::2]], 1)  # the strided convs as one linear (same products)
        t = F.linear(torch.cat(cols, -1), w, torch.stack(enc[1::2]).sum(0))
        t = torch.cat([t.mean(1, keepdim=True), t], 1)
        t = t + DinoVisionTransformer.interpolate_pos_encoding(self, t, hh, ww)  # ty: ignore[invalid-argument-type]
        z = torch.cat([rgb, t], -1)
        t = rgb * (self.rgbd_proj_s(z) + 1) + self.rgbd_proj_b(z)
        has = mp.amax((2, 4)).flatten(1) > 0
        return t, torch.cat([has.new_ones((b, 1)), has], 1)


class Any2FullNet(nn.Module):
    """ViT (`pretrained`), its prompt branches, the depth embedding, the feature fusion MLPs and the DPT head."""

    def __init__(self, encoder: str) -> None:
        """Build the network for `encoder` (`"vits"` / `"vitb"` / `"vitl"`)."""
        super().__init__()
        cfg = CONFIGS[encoder]
        self.layers: tuple[int, ...] = cfg["layers"]  # ty: ignore[invalid-assignment]
        self.pretrained = DINOv2(model_name=encoder)
        dim = self.pretrained.embed_dim
        heads = self.pretrained.blocks[0].attn.num_heads
        self.branches = nn.ModuleList([PromptBranch(dim, heads) for _ in self.layers[1:]])
        self.embed = DepthEmbed(dim)
        self.prompt_norm = nn.LayerNorm(dim, eps=1e-6)  # upstream `prompt_detph_norm`
        self.fusion = nn.ModuleList([_mlp2(2 * dim, dim) for _ in self.layers[1:]])
        self.scale = nn.ModuleList([_mlp2(dim, dim) for _ in self.layers[1:]])
        self.depth_head = DPTHead(dim, cfg["features"], use_bn=False, out_channels=cfg["out_channels"])

    def checkpoint_keys(self, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Upstream checkpoint keys -> this module's."""
        out = {}
        blocks = {layer: i for i, layer in enumerate(self.layers[1:])}
        for key, v in sd.items():
            k = key.removeprefix("module.")
            k = k.replace("pretrained.prompt_depth_embedding.patch_embed.", "embed.")
            k = k.replace("pretrained.prompt_depth_embedding.", "embed.")
            k = k.replace("pretrained.prompt_detph_norm.", "prompt_norm.")
            k = k.replace("pretrained_prompt_depth_fusion.", "fusion.")
            k = k.replace("pretrained_prompt_depth_scale.", "scale.")
            if k.startswith("pretrained.blocks.") and ".prompt_depth_" in k:
                i, rest = k.removeprefix("pretrained.blocks.").split(".", 1)
                k = f"branches.{blocks[int(i)]}.{rest.removeprefix('attn.').removeprefix('prompt_depth_')}"
            out[k] = v
        return out

    def fuse(self, j: int, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Layer `j + 1`'s features `x` modulated by its normalised prompt `p` (both `[B, N, C]`, no class token)."""
        return (self.scale[j](p) + 1) * x + self.fusion[j](torch.cat([x, p], -1))

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised image `[B, 3, 14 gh, 14 gw]`, normalised disparity (float32) -> disparity `[B, 1, H, W]`.

        The image runs in the ViT's weight dtype, the depth embedding in its own (float32 in the fp16 runner).
        """
        vit = self.pretrained
        dt = vit.cls_token.dtype
        t = vit.prepare_tokens_with_masks(x.to(dt))
        feats, pres = [], []
        for i, blk in enumerate(vit.blocks[: max(self.layers) + 1]):
            if i in self.layers[1:]:
                pres.append((blk, t))
            t = blk(t)
            if i in self.layers:
                feats.append(vit.norm(t)[:, 1:])
        et = self.embed.pos_embed.dtype
        p, mask = self.embed(d.to(et), pres[0][1].to(et))
        p = p.to(dt)
        km: torch.Tensor | None = mask[:, None, None]
        for j, (blk, pre) in enumerate(pres):
            p = self.branches[j](blk.norm1(pre), p, km)
            feats[j + 1] = self.fuse(j, feats[j + 1], self.prompt_norm(p)[:, 1:])
            km = None  # upstream: later stages get an all-ones mask
        gh, gw = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        return self.depth_head([[f] for f in feats], gh, gw)
