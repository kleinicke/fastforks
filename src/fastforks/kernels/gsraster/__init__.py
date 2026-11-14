"""Forward 3D Gaussian splatting rasteriser for the feed-forward Gaussian families (`gsraster.cu`).

`render` is the kernels twin of `fastforks.multiview.models.resplat.render.render` (upstream's
`gsplat.rasterization(..., rasterize_mode="antialiased", eps2d=0.1, sh_degree=3, covars=...)`, forward only): the
projection and the tile compositing run in CUDA, the intersection sort and the per-tile offsets in torch.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastforks._core.kernels import CubinModule, cubin_library, linear

TILE = 16  # fixed: gsplat's tile size, TILE in gsraster.cu
CTA = 64  # threads per tile block, CTA in gsraster.cu

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in gsraster.cu)
SIGNATURES = {
    "gsr_project": "ppppppp" + "ppppppp" + "iiiiff",
    "gsr_isect": "ppppp" + "iiii",
    "gsr_rasterize": "ppppppp" + "liiii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def render(  # noqa: PLR0913, PLR0917 - the renderer's signature
    mod: CubinModule, means: torch.Tensor, covariances: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor,
    c2w: torch.Tensor, k_norm: torch.Tensor, h: int, w: int, near: float, far: float,
) -> torch.Tensor:  # fmt: skip
    """Render the views `[V, 3, h, w]` (fp32) of the Gaussians with the `gsraster` module `mod` (see `load`).

    Gaussians: `means [N, 3]`, `covariances [N, 3, 3]`, `opacities [N]`, `harmonics [N, 3, 16]`; cameras
    `c2w [V, 4, 4]` with intrinsics `k_norm [V, 3, 3]` normalised by the image size.
    """
    dev = means.device
    n, v = means.shape[0], c2w.shape[0]
    means, cov = means.float().contiguous(), covariances.float().contiguous()
    opac, sh = opacities.float().contiguous(), harmonics.float().contiguous()
    c2w = c2w.float()
    rt = c2w[:, :3, :3].transpose(1, 2)  # closed-form rigid inverse (no sync, unlike torch.linalg.inv)
    w2c = torch.zeros_like(c2w)
    w2c[:, :3, :3] = rt
    w2c[:, :3, 3] = -(rt @ c2w[:, :3, 3:4])[..., 0]
    w2c[:, 3, 3] = 1.0
    k = (k_norm.float() * torch.tensor((w, h, 1.0), device=dev)[:, None]).contiguous()
    campos = c2w[:, :3, 3].contiguous()
    tw, th = (w + TILE - 1) // TILE, (h + TILE - 1) // TILE
    ntiles = tw * th
    nv = n * v
    m2d = torch.empty(nv, 2, device=dev)
    conic = torch.empty(nv, 3, device=dev)
    opac_o = torch.empty(nv, device=dev)
    color = torch.empty(nv, 3, device=dev)
    depth = torch.empty(nv, device=dev)
    bbox = torch.empty(nv, 4, dtype=torch.int32, device=dev)
    cnt = torch.empty(nv, dtype=torch.int32, device=dev)
    mod.launch(
        "gsr_project", *linear(nv), means, cov, opac, sh, w2c, k, campos, m2d, conic, opac_o, color, depth, bbox,
        cnt, n, v, h, w, near, far,
    )  # fmt: skip
    off = torch.cumsum(cnt, 0, dtype=torch.int64)
    total = int(off[-1])
    off -= cnt
    keys = torch.empty(total, dtype=torch.int64, device=dev)
    vals = torch.empty(total, dtype=torch.int32, device=dev)
    out = torch.empty(v, 3, h, w, device=dev)
    if total == 0:
        return out.zero_()
    mod.launch("gsr_isect", *linear(nv), bbox, depth, off, keys, vals, n, v, ntiles, tw)
    keys, perm = keys.sort(stable=True)
    vals = vals[perm]
    offs = torch.searchsorted(keys >> 32, torch.arange(v * ntiles, device=dev))
    mod.launch(
        "gsr_rasterize", (v * ntiles, 1, 1), (CTA, 1, 1), m2d, conic, opac_o, color, offs, vals, out, total, ntiles,
        tw, h, w,
    )  # fmt: skip
    return out
