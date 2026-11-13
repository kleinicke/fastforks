"""RANSAC kernels (`ransac.cu`) for `matching.estimate_pose` / `estimate_fundamental` / `estimate_absolute_pose`.

`e5pt_hypotheses`: one thread per five-match sample, Nister's five-point solver in registers (up to ten E per sample).
`f7pt_hypotheses`: one thread per seven-match sample, the 7-point fundamental-matrix solver (up to three F).
`score_many` / `score_few`: MSAC scores (truncated quadratic Sampson loss) and inlier counts of many (one thread
each) or few (one block each) candidate E against a pair's matches; `score_few` also writes the inlier mask.
`lm_refine`: Levenberg-Marquardt of (R, t) per pair on the truncated Sampson cost, all iterations in one launch.
`polish`: the whole final stage per pair in one launch: 8-point refits on the inliers, recoverPose, LM, inlier mask
(for fundamental matrices: rank-2 refits, LM of the rank-2 parametrisation, the mask).
`p3p_hypotheses`: one thread per three-match sample, Lambda Twist P3P (up to four poses `[R | t]` per sample).
`score_many_abs` / `score_few_abs`: the same scoring with the truncated reprojection error of 2D-3D matches.
`polish_abs`: LM of `[R | t]` on the truncated reprojection cost and the inlier mask, per pair in one launch.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastforks._core.kernels import CubinModule, cubin_library, linear

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in ransac.cu)
SIGNATURES = {
    "e5pt_hypotheses": "pppppiii",
    "f7pt_hypotheses": "pppppiii",
    "score_many": "pppppppiii",
    "score_few": "ppppppppiii",
    "lm_refine": "pppppiii",
    "polish": "pppppppppiiiii",
    "p3p_hypotheses": "pppppiii",
    "score_many_abs": "pppppppiii",
    "score_few_abs": "ppppppppiii",
    "polish_abs": "pppppppiii",
}
THREADS = 256
SMALL_THREADS, SMALL = 32, 4096  # the hypotheses kernel's block for launches of at most SMALL samples
MANY = 64  # candidates per pair from which the one-thread-per-candidate kernel wins
# Sample size -> kernel, solutions per sample, floats per model (3: P3P on 2D-3D matches of 8 floats).
SOLVERS = {5: ("e5pt_hypotheses", 10, 9), 7: ("f7pt_hypotheses", 3, 9), 3: ("p3p_hypotheses", 4, 12)}
WIDTH = {9: 4, 12: 8}  # floats per match of a model size

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def hypotheses(
    lib: CubinModule, pts: torch.Tensor, pairs: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Models `[A, H*S, M]` and validity `[A, H*S]` of the samples `idx` `[A, H, 5 | 7 | 3]` of `pts[pairs]`.

    Five-point samples give up to `S = 10` essential matrices each (`M = 9`), seven-point samples up to `S = 3`
    fundamental matrices, three-point samples of 2D-3D matches (`pts` `[P, N, 8]`) up to `S = 4` poses (`M = 12`).
    """
    a, h, size = idx.shape
    kernel, s, m = SOLVERS[size]
    n = pts.shape[1]
    out = torch.empty(a, h, s, m, device=pts.device, dtype=torch.float32)
    valid = torch.empty(a, h, s, device=pts.device, dtype=torch.int32)
    # Small launches (one pair): one warp per block spreads the samples over the SMs, whose few fp64 units the
    # solver's Gauss-Jordan elimination otherwise shares between the warps of a block.
    grid, block = linear(a * h, SMALL_THREADS if a * h <= SMALL else THREADS)
    lib.launch(kernel, grid, block, pts, pairs, idx.contiguous(), out, valid, n, h, a)
    return out.view(a, h * s, m), valid.view(a, h * s).bool()


def score(  # noqa: PLR0913, PLR0917 - the kernel's arguments
    lib: CubinModule,
    pts: torch.Tensor,
    counts: torch.Tensor,
    thr2: torch.Tensor,
    pairs: torch.Tensor,
    e: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSAC scores (float32) and inlier counts (int32), `[A, C]` each, of the candidates `e` `[A, C, 9 | 12]`.

    Scored on the first `n` matches of `pts[pairs]` (`[P, N, 4]` for essential / fundamental matrices, `[P, N, 8]`
    2D-3D matches for the poses `[R | t]` of 12 floats).
    """
    a, c, m = e.shape
    big_n = pts.shape[1]
    e = e.contiguous()
    out = torch.empty(a, c, device=pts.device, dtype=torch.float32)
    cnt = torch.empty(a, c, device=pts.device, dtype=torch.int32)
    suffix = "_abs" if m == SOLVERS[3][2] else ""
    if c >= MANY:
        grid = ((c + THREADS - 1) // THREADS, a, 1)
        lib.launch("score_many" + suffix, grid, (THREADS, 1, 1), pts, counts, thr2, pairs, e, out, cnt, big_n, c, n)
    else:
        args = (pts, counts, thr2, pairs, e, out, cnt, None, big_n, c, n)
        lib.launch("score_few" + suffix, (c, a, 1), (THREADS, 1, 1), *args)
    return out, cnt


def inlier_masks(
    lib: CubinModule, pts: torch.Tensor, counts: torch.Tensor, thr2: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    """Inlier masks `[P, N]` (bool) of one model per pair `e` `[P, 9 | 12]`."""
    p, n = pts.shape[:2]
    e = e.contiguous()
    pairs = torch.arange(p, device=pts.device, dtype=torch.int32)
    out = torch.empty(p, device=pts.device, dtype=torch.float32)
    cnt = torch.empty(p, device=pts.device, dtype=torch.int32)
    mask = torch.empty(p, n, device=pts.device, dtype=torch.uint8)
    kernel = "score_few_abs" if e.shape[1] == SOLVERS[3][2] else "score_few"
    lib.launch(kernel, (1, p, 1), (THREADS, 1, 1), pts, counts, thr2, pairs, e, out, cnt, mask, n, 1, n)
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
    *,
    fundamental: bool = False,
    minimum: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The final stage of one model per pair `e` `[P, 9]`: refits, `recoverPose`, LM refinement, inlier mask.

    Returns `R` `[P, 3, 3]`, `t` `[P, 3]`, the polished model `[P, 9]`, inlier counts `[P]` int32, `ok` `[P]` bool
    (at least `minimum` inliers, and a valid cheirality) and the inlier mask `[P, N]` bool; `R`, `t` are the
    identity and `(0, 0, 1)` where not `ok`. `fundamental`: rank-2 refits only, no pose (`R`, `t` as where not ok).
    """
    p, n = pts.shape[:2]
    e = e.contiguous().clone()
    r = torch.empty(p, 3, 3, device=pts.device, dtype=torch.float32)
    t = torch.empty(p, 3, device=pts.device, dtype=torch.float32)
    cnt = torch.empty(p, device=pts.device, dtype=torch.int32)
    ok = torch.empty(p, device=pts.device, dtype=torch.int32)
    mask = torch.empty(p, n, device=pts.device, dtype=torch.uint8)
    args = (pts, counts, thr2, e, r, t, cnt, ok, mask, refits, lm_iterations, n, int(fundamental), minimum)
    lib.launch("polish", (p, 1, 1), (THREADS, 1, 1), *args)
    return r, t, e, cnt, ok.bool(), mask.bool()


def polish_abs(  # noqa: PLR0913, PLR0917 - the kernel's arguments
    lib: CubinModule,
    pts: torch.Tensor,
    counts: torch.Tensor,
    thr2: torch.Tensor,
    m: torch.Tensor,
    lm_iterations: int,
    minimum: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The final stage of one pose per pair `m` `[P, 12]` on the 2D-3D matches `pts` `[P, N, 8]`: LM, inlier mask.

    Returns the polished poses `[P, 12]`, inlier counts `[P]` int32, `ok` `[P]` bool (at least `minimum` inliers) and
    the inlier mask `[P, N]` bool.
    """
    p, n = pts.shape[:2]
    m = m.contiguous().clone()
    cnt = torch.empty(p, device=pts.device, dtype=torch.int32)
    ok = torch.empty(p, device=pts.device, dtype=torch.int32)
    mask = torch.empty(p, n, device=pts.device, dtype=torch.uint8)
    lib.launch("polish_abs", (p, 1, 1), (THREADS, 1, 1), pts, counts, thr2, m, cnt, ok, mask, lm_iterations, n, minimum)
    return m, cnt, ok.bool(), mask.bool()
