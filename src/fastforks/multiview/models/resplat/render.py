# ruff: noqa: PLR0913, PLR0917 - gsplat's rasterization arguments
"""Gaussian rasterizer in plain PyTorch: upstream's `gsplat.rasterization` call, forward only.

ReSplat renders its input views to refine the Gaussians (`rasterize_mode="antialiased"`, `eps2d=0.1`, covariances
given, SH colours, no background). This follows gsplat's CUDA kernels: EWA projection with the Jacobian clamped to
1.3 x the field of view, a 0.1 px blur with opacity compensation, the opacity-aware 3.33 sigma bounding box, depth
order inside 16 x 16 tiles, and per pixel `alpha = min(0.99, o exp(-sigma))` (skipped below 1 / 255) composited front
to back until the transmittance would fall to 1e-4. All tiles are composited at once, `CHUNK` Gaussians per step.
"""

from __future__ import annotations

import torch

from fastforks.multiview._runner import invert

TILE = 16  # fixed: gsplat's tile size
CHUNK = 64  # Gaussians per tile composited per step (memory: tiles x 256 x CHUNK floats)
EPS2D = 0.1  # fixed: upstream's eps2d
ALPHA_MIN = 1 / 255  # fixed: gsplat ALPHA_THRESHOLD
ALPHA_MAX = 0.99  # fixed: gsplat MAX_ALPHA
T_MIN = 1e-4  # fixed: gsplat TRANSMITTANCE_THRESHOLD
EXTEND = 3.33  # fixed: gsplat GAUSSIAN_EXTEND
MIN_COMPENSATION = 0.005  # fixed: gsplat MIN_COMPENSATION


def sh_basis(d: torch.Tensor) -> torch.Tensor:
    """The degree-3 real SH basis of gsplat (`_eval_sh_bases_fast`) of unit directions `[..., 3]` -> `[..., 16]`."""
    x, y, z = d.unbind(-1)
    z2 = z * z
    c1, s1 = x * x - y * y, 2 * x * y
    c2, s2 = x * c1 - y * s1, x * s1 + y * c1
    a, b, c = -0.48860251190292, -1.092548430592079 * z, -2.285228997322329 * z2 + 0.4570457994644658
    e = 1.445305721320277 * z
    return torch.stack(
        (
            torch.full_like(x, 0.2820947917738781),
            a * y, -a * z, a * x,
            0.5462742152960395 * s1, b * y, 0.9461746957575601 * z2 - 0.3153915652525201, b * x,
            0.5462742152960395 * c1,
            -0.5900435899266435 * s2, e * s1, c * y, z * (1.865881662950577 * z2 - 1.119528997770346), c * x, e * c1,
            -0.5900435899266435 * c2,
        ),
        -1,
    )  # fmt: skip


def _project(
    means: torch.Tensor, cov: torch.Tensor, opac: torch.Tensor, w2c: torch.Tensor, k: torch.Tensor, h: int, w: int,
    near: float, far: float,
) -> tuple[torch.Tensor, ...]:  # fmt: skip
    """One view through gsplat `fully_fused_projection` (pinhole, antialiased).

    Returns visible ids, means2d, conics, compensated opacities, depths, radii.
    """
    r, t = w2c[:3, :3], w2c[:3, 3]
    mc = means @ r.T + t
    x, y, z = mc.unbind(-1)
    cc = r @ cov @ r.T
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    tfx, tfy = 0.5 * w / fx, 0.5 * h / fy
    tx = z * (x / z).clamp(-(cx / fx + 0.3 * tfx), (w - cx) / fx + 0.3 * tfx)
    ty = z * (y / z).clamp(-(cy / fy + 0.3 * tfy), (h - cy) / fy + 0.3 * tfy)
    zero = torch.zeros_like(z)
    jac = torch.stack((fx / z, zero, -fx * tx / (z * z), zero, fy / z, -fy * ty / (z * z)), -1).view(-1, 2, 3)
    c2 = jac @ cc @ jac.transpose(1, 2)
    a, b, c = c2[:, 0, 0], c2[:, 0, 1], c2[:, 1, 1]
    det0 = a * c - b * c2[:, 1, 0]
    a, c = a + EPS2D, c + EPS2D
    det = a * c - b * c2[:, 1, 0]
    comp = (det0 / det).clamp_min(MIN_COMPENSATION**2).sqrt()
    o = opac * comp
    m2 = torch.stack((fx * x / z + cx, fy * y / z + cy), -1)
    extend = (2 * torch.log(o.clamp_min(ALPHA_MIN) / ALPHA_MIN)).sqrt().clamp_max(EXTEND)
    rx, ry = (extend * a.clamp_min(0).sqrt()).ceil(), (extend * c.clamp_min(0).sqrt()).ceil()
    ok = (z >= near) & (z <= far) & (det > 0) & (o >= ALPHA_MIN) & ((rx > 0) | (ry > 0))
    ok &= (m2[:, 0] + rx > 0) & (m2[:, 0] - rx < w) & (m2[:, 1] + ry > 0) & (m2[:, 1] - ry < h)
    ids = ok.nonzero()[:, 0]
    ids = ids[z[ids].argsort(stable=True)]  # front to back
    conic = torch.stack((c / det, -(b + c2[:, 1, 0]) / 2 / det, a / det), -1)
    return ids, m2[ids], conic[ids], o[ids], torch.stack((rx, ry), -1)[ids]


def render(  # noqa: PLR0915 - one tile pass per view
    means: torch.Tensor, covariances: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor,
    c2w: torch.Tensor, k_norm: torch.Tensor, h: int, w: int, near: float, far: float,
) -> torch.Tensor:  # fmt: skip
    """Render the views `[V, 3, h, w]` (fp32) of the Gaussians.

    Gaussians: `means [N, 3]`, `covariances [N, 3, 3]`, `opacities [N]`, `harmonics [N, 3, 16]`; cameras
    `c2w [V, 4, 4]` with intrinsics `k_norm [V, 3, 3]` normalised by the image size.
    """
    means, opacities = means.float(), opacities.float()
    cov = covariances.float().triu()
    cov = cov + cov.transpose(1, 2) - torch.diag_embed(cov.diagonal(dim1=1, dim2=2))  # gsplat reads the upper triangle
    sh = harmonics.float()
    w2c = invert(c2w.float())
    tw, th = (w + TILE - 1) // TILE, (h + TILE - 1) // TILE
    dev = means.device
    py, px = torch.meshgrid(torch.arange(TILE, device=dev), torch.arange(TILE, device=dev), indexing="ij")
    local = torch.stack((px, py), -1).view(1, TILE * TILE, 2).float()
    origin = torch.stack(torch.meshgrid(torch.arange(th, device=dev), torch.arange(tw, device=dev), indexing="ij"), -1)
    pix = origin.view(-1, 1, 2).flip(-1).float() * TILE + local + 0.5  # [tiles, 256, 2] pixel centres (x, y)
    out = []
    for v in range(c2w.shape[0]):
        k = k_norm[v].float() * torch.tensor((w, h, 1.0), device=dev)[:, None]
        ids, m2, conic, o, rad = _project(means, cov, opacities, w2c[v], k, h, w, near, far)
        d = means[ids] - c2w[v, :3, 3].float()
        col = (sh[ids] @ sh_basis(d / d.norm(dim=-1, keepdim=True))[..., None])[..., 0].add(0.5).clamp_min(0)
        # tiles touched by every Gaussian (gsplat isect_tiles), in front-to-back order, then grouped by tile
        lo = ((m2 - rad) / TILE).floor().clamp(min=0)
        hi = ((m2 + rad) / TILE).ceil()
        lo, hi = lo.long(), torch.stack((hi[:, 0].clamp(max=tw), hi[:, 1].clamp(max=th)), -1).long()
        n = (hi - lo).clamp_min(0)
        cnt = n[:, 0] * n[:, 1]
        g = torch.repeat_interleave(torch.arange(len(ids), device=dev), cnt)
        j = torch.arange(len(g), device=dev) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
        tile = (lo[g, 1] + j // n[g, 0]) * tw + lo[g, 0] + j % n[g, 0]
        tile, perm = tile.sort(stable=True)
        g = g[perm]
        per = torch.bincount(tile, minlength=tw * th)
        rank = per.argsort(descending=True, stable=True)
        pos = torch.arange(len(g), device=dev) - torch.repeat_interleave(per.cumsum(0) - per, per)
        length = int(per.max()) if len(g) else 0
        slot = torch.empty_like(rank)
        slot[rank] = torch.arange(len(rank), device=dev)
        table = torch.full((tw * th, max(length, 1)), len(ids), dtype=torch.long, device=dev)
        table[slot[tile], pos] = g
        active = per[rank]  # Gaussians per tile, descending
        m2p, conp = torch.cat((m2, m2.new_zeros(1, 2))), torch.cat((conic, conic.new_zeros(1, 3)))
        op, colp = torch.cat((o, o.new_zeros(1))), torch.cat((col, col.new_zeros(1, 3)))  # padding: opacity 0
        p = pix[rank]
        t_cur = p.new_ones(p.shape[:2])
        done = torch.zeros_like(t_cur, dtype=torch.bool)
        img = p.new_zeros(*p.shape[:2], 3)
        for s in range(0, length, CHUNK):
            nt = int((active > s).sum())
            gi = table[:nt, s : s + CHUNK]
            dx = p[:nt, :, None] - m2p[gi][:, None]  # [nt, 256, chunk, 2]
            cn = conp[gi][:, None]
            sigma = (
                0.5 * (cn[..., 0] * dx[..., 0] ** 2 + cn[..., 2] * dx[..., 1] ** 2)
                + cn[..., 1] * dx[..., 0] * dx[..., 1]
            )
            alpha = (op[gi][:, None] * torch.exp(-sigma)).clamp_max(ALPHA_MAX)
            alpha = torch.where((sigma < 0) | (alpha < ALPHA_MIN), 0.0, alpha)
            trans = t_cur[:nt, :, None] * torch.cumprod(1 - alpha, -1)
            inc = (trans > T_MIN) & ~done[:nt, :, None]
            before = torch.cat((t_cur[:nt, :, None], trans[..., :-1]), -1)
            wgt = torch.where(inc, alpha * before, 0.0)
            img[:nt] += wgt @ colp[gi]
            t_cur[:nt] = torch.where(inc, 1 - alpha, 1.0).prod(-1) * t_cur[:nt]
            done[:nt] |= trans[..., -1] <= T_MIN
        tiles = torch.empty_like(img)
        tiles[rank] = img
        out.append(tiles.view(th, tw, TILE, TILE, 3).permute(4, 0, 2, 1, 3).reshape(3, th * TILE, tw * TILE)[:, :h, :w])
    return torch.stack(out)
