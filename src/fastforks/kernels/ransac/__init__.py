"""Essential-matrix RANSAC kernels (`ransac.cu`) for `fastforks.matching.estimate_pose`.

`e5pt_hypotheses`: one thread per five-match sample, Nister's five-point solver in registers (up to ten E per sample).
`score_many` / `score_few`: MSAC scores (truncated quadratic Sampson loss) and inlier counts of many (one thread
each) or few (one block each) candidate E against a pair's matches; `score_few` also writes the inlier mask.
`lm_refine`: Levenberg-Marquardt of (R, t) per pair on the truncated Sampson cost, all iterations in one launch.
`polish`: the whole final stage per pair in one launch: 8-point refits on the inliers, recoverPose, LM, inlier mask.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastforks._core.kernels import CubinModule, cubin_library, linear

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in ransac.cu)
SIGNATURES = {
    "e5pt_hypotheses": "pppppiii",
    "score_many": "pppppppiii",
    "score_few": "ppppppppiii",
    "lm_refine": "pppppiii",
    "polish": "pppppppppiii",
}
THREADS = 256
SMALL_THREADS, SMALL = 32, 4096  # the hypotheses kernel's block for launches of at most SMALL samples
MANY = 64  # candidates per pair from which the one-thread-per-candidate kernel wins

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def hypotheses(
    lib: CubinModule, pts: torch.Tensor, pairs: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Essential matrices `[A, H*10, 9]` and validity `[A, H*10]` of the samples `idx` `[A, H, 5]` of `pts[pairs]`."""
    a, h, _ = idx.shape
    n = pts.shape[1]
    out = torch.empty(a, h, 10, 9, device=pts.device, dtype=torch.float32)
    valid = torch.empty(a, h, 10, device=pts.device, dtype=torch.int32)
    # Small launches (one pair): one warp per block spreads the samples over the SMs, whose few fp64 units the
    # solver's Gauss-Jordan elimination otherwise shares between the warps of a block.
    grid, block = linear(a * h, SMALL_THREADS if a * h <= SMALL else THREADS)
    lib.launch("e5pt_hypotheses", grid, block, pts, pairs, idx.contiguous(), out, valid, n, h, a)
    return out.view(a, h * 10, 9), valid.view(a, h * 10).bool()


def score(  # noqa: PLR0913, PLR0917 - the kernel's arguments
    lib: CubinModule,
    pts: torch.Tensor,
    counts: torch.Tensor,
    thr2: torch.Tensor,
    pairs: torch.Tensor,
    e: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSAC scores (float32) and inlier counts (int32), `[A, C]` each, of the candidates `e` `[A, C, 9]`.

    Scored on the first `n` matches of `pts[pairs]`.
    """
    a, c, _ = e.shape
    big_n = pts.shape[1]
    e = e.contiguous()
    out = torch.empty(a, c, device=pts.device, dtype=torch.float32)
    cnt = torch.empty(a, c, device=pts.device, dtype=torch.int32)
    if c >= MANY:
        grid = ((c + THREADS - 1) // THREADS, a, 1)
        lib.launch("score_many", grid, (THREADS, 1, 1), pts, counts, thr2, pairs, e, out, cnt, big_n, c, n)
    else:
        args = (pts, counts, thr2, pairs, e, out, cnt, None, big_n, c, n)
        lib.launch("score_few", (c, a, 1), (THREADS, 1, 1), *args)
    return out, cnt


def inlier_masks(
    lib: CubinModule, pts: torch.Tensor, counts: torch.Tensor, thr2: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    """Inlier masks `[P, N]` (bool) of one model per pair `e` `[P, 9]`."""
    p, n = pts.shape[:2]
    e = e.contiguous()
    pairs = torch.arange(p, device=pts.device, dtype=torch.int32)
    out = torch.empty(p, device=pts.device, dtype=torch.float32)
    cnt = torch.empty(p, device=pts.device, dtype=torch.int32)
    mask = torch.empty(p, n, device=pts.device, dtype=torch.uint8)
    lib.launch("score_few", (1, p, 1), (THREADS, 1, 1), pts, counts, thr2, pairs, e, out, cnt, mask, n, 1, n)
    return mask.bool()


def lm_refine(  # noqa: PLR0913, PLR0917 - the kernel's arguments
    lib: CubinModule,
    pts: torch.Tensor,
    counts: torch.Tensor,
    thr2: torch.Tensor,
    pairs: torch.Tensor,
    rt: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Levenberg-Marquardt of the poses `rt` `[A, 12]` (`R` row-major, unit `t`; `E = [t]_x R`).

    Minimises the truncated Sampson cost of `pts[pairs]`, every iteration inside the kernel. Returns the refined poses.
    """
    out = rt.contiguous().clone()
    n = pts.shape[1]
    lib.launch("lm_refine", (out.shape[0], 1, 1), (THREADS, 1, 1), pts, counts, thr2, pairs, out, iterations, n, n)
    return out


def polish(  # noqa: PLR0913, PLR0917 - the kernel's arguments
    lib: CubinModule,
    pts: torch.Tensor,
    counts: torch.Tensor,
    thr2: torch.Tensor,
    e: torch.Tensor,
    refits: int,
    lm_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The final stage of one model per pair `e` `[P, 9]`: refits, `recoverPose`, LM refinement, inlier mask.

    Returns `R` `[P, 3, 3]`, `t` `[P, 3]`, the polished `E` `[P, 9]`, inlier counts `[P]` int32, `ok` `[P]` bool and
    the inlier mask `[P, N]` bool; `R`, `t` are the identity and `(0, 0, 1)` where not `ok`.
    """
    p, n = pts.shape[:2]
    e = e.contiguous().clone()
    r = torch.empty(p, 3, 3, device=pts.device, dtype=torch.float32)
    t = torch.empty(p, 3, device=pts.device, dtype=torch.float32)
    cnt = torch.empty(p, device=pts.device, dtype=torch.int32)
    ok = torch.empty(p, device=pts.device, dtype=torch.int32)
    mask = torch.empty(p, n, device=pts.device, dtype=torch.uint8)
    args = (pts, counts, thr2, e, r, t, cnt, ok, mask, refits, lm_iterations, n)
    lib.launch("polish", (p, 1, 1), (THREADS, 1, 1), *args)
    return r, t, e, cnt, ok.bool(), mask.bool()
