"""MoGe-3's sparse refiner (`sparse.SparseUNet`, `model.refine`) on Apple MLX: `mx` operations and two Metal kernels.

The trunk runs as an exported graph (`mlx_wrap.port`); the refiner's voxels depend on the predicted depth, so it is
written here by hand and runs eagerly on the GPU (one host sync per pooled level, for its voxel count):

- neighbour tables `[27, N]`: the finest level (one voxel per pixel) from the neighbour pixels' z bins, the coarser
  ones by binary search over the sorted keys (`moge_nbr` kernel)
- pooling: parent keys sorted and deduplicated (`mx.argsort`, `cumsum`), children's mean by a scatter-add
- each submanifold conv as one gather implicit GEMM (`moge_subm_conv`: `sparse_metal`'s kernel; simdgroup 8x8
  matrices, float32 accumulation). At the finest level it beats `sparse._grid_conv`'s dense conv + masked corrections
  as MLX operations (784x1168: 53 vs 156 ms per conv)

The refiner runs in float32 (as the portable runner off CUDA).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import metal_kernel, mx, to_mx, to_torch

if TYPE_CHECKING:
    import torch

    from .sparse import SparseUNet

_LN_EPS = 1e-6  # fixed: upstream SparseResBlock3d LayerNorm
_TAPS = list(itertools.product((-1, 0, 1), repeat=3))  # (di, dj, dz), FlexGEMM's order (dz fastest)
_TILE = 32  # subm_conv: output rows and channels per threadgroup

_NBR = metal_kernel(
    name="moge_nbr",
    input_names=["coords", "key", "ext"],
    output_names=["nbr"],
    source="""
        // one thread per (tap, voxel): binary search of the neighbour's key among the level's sorted keys
        uint n = thread_position_in_grid.x, v = thread_position_in_grid.y;
        uint N = key_shape[0];
        if (n >= N) return;
        int h = ext[0], w = ext[1], z = ext[2];
        int di = int(v / 9) - 1, dj = int((v / 3) % 3) - 1, dz = int(v % 3) - 1;
        int i = coords[n * 4 + 1] + di, j = coords[n * 4 + 2] + dj, k = coords[n * 4 + 3] + dz;
        int found = -1;
        if (i >= 0 && i < h && j >= 0 && j < w && k >= 0 && k < z) {
            long want = key[n] + (long(di) * w + dj) * z + dz;
            uint lo = 0, hi = N;
            while (lo < hi) {
                uint mid = (lo + hi) / 2;
                if (key[mid] < want) lo = mid + 1; else hi = mid;
            }
            if (lo < N && key[lo] == want) found = int(lo);
        }
        nbr[v * N + n] = found;
    """,
)

_SUBM = metal_kernel(
    name="moge_subm_conv",
    input_names=["x", "nbr", "w", "bias"],
    output_names=["out"],
    header="#include <metal_simdgroup_matrix>\n",
    source="""
        // out[n, o] = bias[o] + sum_v sum_c x[nbr[v, n], c] * w[v, c, o]  (sparse_metal's subm_conv)
        // one threadgroup (4 simdgroups) per 32 x 32 output tile; a tap no voxel of the tile has is skipped
        typedef vec<T, 4> T4;
        const uint BM = 32, BK = 32, LD = 36;
        const uint N = nbr_shape[1];
        const uint tg = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tiles = O / 32;
        const uint m0 = (tg / tiles) * BM, n0 = (tg % tiles) * 32;
        threadgroup float A[32 * 36], B[32 * 36];
        threadgroup int rows[32];
        threadgroup bool any;
        const uint r = tid / 4, c = (tid % 4) * 8;
        threadgroup float4* a4 = (threadgroup float4*)(A + r * LD + c);
        threadgroup float4* b4 = (threadgroup float4*)(B + r * LD + c);
        const uint sr = (sg / 2) * 16, sc = (sg % 2) * 16;
        simdgroup_float8x8 acc[2][2];
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
        for (uint v = 0; v < 27; v++) {
            if (tid < BM) {
                const uint n = m0 + tid;
                const int row = n < N ? nbr[v * N + n] : -1;
                rows[tid] = row;
                const bool hit = simd_any(row >= 0);
                if (tid == 0) any = hit;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!any) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }
            const int row = rows[r];
            const device T4* xs = (const device T4*)(x + uint(max(row, 0)) * C + c);
            const device T4* ws = (const device T4*)(w + (v * C + r) * O + n0 + c);
            for (uint k0 = 0; k0 < C; k0 += BK) {
                a4[0] = row >= 0 ? float4(xs[k0 / 4]) : 0.0f;
                a4[1] = row >= 0 ? float4(xs[k0 / 4 + 1]) : 0.0f;
                b4[0] = float4(ws[k0 * O / 4]);
                b4[1] = float4(ws[k0 * O / 4 + 1]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint kk = 0; kk < BK; kk += 8) {
                    simdgroup_float8x8 a[2], b[2];
                    for (uint i = 0; i < 2; i++) simdgroup_load(a[i], A + (sr + 8 * i) * LD + kk, LD);
                    for (uint j = 0; j < 2; j++) simdgroup_load(b[j], B + kk * LD + sc + 8 * j, LD);
                    for (uint i = 0; i < 2; i++)
                        for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], A + (sr + 8 * i) * LD + sc + 8 * j, LD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint n = m0 + r;
        if (n < N) {
            const device T4* bb = (const device T4*)(bias + n0 + c);
            device T4* o = (device T4*)(out + n * O + n0 + c);
            o[0] = T4(a4[0] + float4(bb[0]));
            o[1] = T4(a4[1] + float4(bb[1]));
        }
    """,
)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _Level:
    """One level: voxel coordinates `[N, 4]` (int32, sorted by key), keys (int64), extent (H, W, Z) and neighbours."""

    def __init__(self, coords: mx.array, key: mx.array, extent: tuple[int, int, int]) -> None:
        self.coords, self.key, self.extent = coords, key, extent
        self.nbr: mx.array | None = None  # [27, N] int32 (-1: none); coarse levels only
        self.z: mx.array | None = None  # finest level: z bins [B, H, W] (int32)

    def neighbours(self) -> mx.array:
        if self.nbr is None and self.z is not None:
            self.nbr = _grid_nbr(self.z)
        if self.nbr is None:
            n = self.key.shape[0]
            self.nbr = _NBR(
                inputs=[self.coords, self.key, mx.array(self.extent, dtype=mx.int32)],
                grid=(n, 27, 1), threadgroup=(256, 1, 1),
                output_shapes=[(27, n)], output_dtypes=[mx.int32],
            )[0]  # fmt: skip
        return self.nbr


def _key(c: mx.array, extent: tuple[int, int, int]) -> mx.array:
    h, w, z = extent
    c = c.astype(mx.int64)
    return ((c[:, 0] * h + c[:, 1]) * w + c[:, 2]) * z + c[:, 3]


def _grid_nbr(z: mx.array) -> mx.array:
    """The finest level's neighbour table `[27, B*H*W]` from its z bins `[B, H, W]`."""
    b, h, w = z.shape
    far = -(1 << 20)
    zp = mx.pad(z, [(0, 0), (1, 1), (1, 1)], constant_values=far)
    idx = mx.arange(b * h * w, dtype=mx.int32).reshape(b, h, w)
    rows = []
    for di, dj, dz in _TAPS:
        zn = zp[:, 1 + di : 1 + di + h, 1 + dj : 1 + dj + w]
        rows.append(mx.where(zn - z == dz, idx + (di * w + dj), -1).reshape(-1))
    return mx.stack(rows)


def _grid_level(z: mx.array) -> _Level:
    b, h, w = z.shape
    bb = mx.broadcast_to(mx.arange(b, dtype=mx.int32)[:, None, None], (b, h, w))
    ii = mx.broadcast_to(mx.arange(h, dtype=mx.int32)[None, :, None], (b, h, w))
    jj = mx.broadcast_to(mx.arange(w, dtype=mx.int32)[None, None, :], (b, h, w))
    coords = mx.stack([bb, ii, jj, z], -1).reshape(-1, 4)
    extent = (h, w, int(z.max().item()) + 1)
    lv = _Level(coords, _key(coords, extent), extent)
    lv.z = z
    return lv


class _Pool:
    """A 2x2x2 mean-pooling edge: the parent level, per child its parent index, children per parent."""

    def __init__(self, child: _Level) -> None:
        h, w, z = child.extent
        ext = ((h + 1) // 2, (w + 1) // 2, (z + 1) // 2)
        c = child.coords
        pc = mx.concatenate([c[:, :1], c[:, 1:] // 2], 1)
        pk = _key(pc, ext)
        order = mx.argsort(pk)
        keys = pk[order]
        first = mx.concatenate([mx.array([True]), keys[1:] != keys[:-1]])
        seg = mx.cumsum(first.astype(mx.int32)) - 1
        m = int(seg[-1].item()) + 1
        n = c.shape[0]
        self.parent = mx.zeros((n,), mx.int32).at[order].add(seg)
        last = mx.zeros((m,), mx.int32).at[seg].maximum(mx.arange(n, dtype=mx.int32))  # a child of each parent
        self.counts = mx.zeros((m,), mx.float32).at[seg].add(mx.ones((n,), mx.float32))
        self.level = _Level(pc[order[last]], keys[last], ext)

    def mean(self, x: mx.array) -> mx.array:
        s = mx.zeros((self.counts.shape[0], x.shape[1]), x.dtype).at[self.parent].add(x)
        return s / self.counts[:, None].astype(x.dtype)


class MlxRefiner:
    """`SparseUNet` with its parameters on MLX (float32)."""

    def __init__(self, net: SparseUNet) -> None:
        """Convert the parameters of the PyTorch refiner."""

        def lin(m: Any) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - torch modules of the vendored net
            return to_mx(m.weight.detach().float().t()), to_mx(m.bias.detach().float())

        def conv(m: Any) -> dict[str, mx.array]:  # noqa: ANN401
            w = m.weight.detach().float()  # [C_out, 3 (i), 3 (j), 3 (z), C_in]
            o, c = w.shape[0], w.shape[-1]
            return {"w": to_mx(w.reshape(o, 27, c).permute(1, 2, 0).contiguous()), "b": to_mx(m.bias.detach().float())}

        def block(stage: Any) -> list[dict[str, mx.array]]:  # noqa: ANN401
            out = []
            for blk in stage.children():
                ln = blk.norm1
                out.append(
                    {
                        "ln_w": to_mx(ln.weight.detach().float()),
                        "ln_b": to_mx(ln.bias.detach().float()),
                        "c1": conv(blk.conv1),
                        "c2": conv(blk.conv2),
                    }
                )
            return out

        self.input_proj = lin(net.input_proj)
        self.encoder_fuse = lin(net.encoder_fuse)
        self.fuse = (lin(net.fuse_proj[0]), lin(net.fuse_proj[2]))
        self.down = [block(s) for s in net.down_stages]
        self.downsample = [lin(m.get_submodule("linear")) for m in net.downsample_blocks]
        self.bottleneck = block(net.bottleneck_stage)
        self.upsample = [lin(m.get_submodule("linear")) for m in net.upsample_blocks]
        self.up = [block(s) for s in net.up_stages]
        self.out_proj = lin(net.out_proj)
        mx.eval(self.__dict__)

    @staticmethod
    def _conv(x: mx.array, lv: _Level, p: dict[str, mx.array]) -> mx.array:
        n, c = x.shape
        o = p["w"].shape[2]
        return _SUBM(
            inputs=[x, lv.neighbours(), p["w"], p["b"]],
            template=[("T", x.dtype), ("C", c), ("O", o)],
            grid=(-(-n // _TILE) * (o // _TILE) * 128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(n, o)], output_dtypes=[x.dtype],
        )[0]  # fmt: skip

    def _block(self, blocks: list[dict[str, mx.array]], x: mx.array, lv: _Level) -> mx.array:
        for p in blocks:
            h = _silu(mx.fast.layer_norm(x, p["ln_w"], p["ln_b"], _LN_EPS))
            x = self._conv(_silu(self._conv(h, lv, p["c1"])), lv, p["c2"]) + x
            mx.eval(x)
        return x

    def __call__(self, x: mx.array, z: mx.array, enc: mx.array) -> mx.array:
        """`SparseUNet.forward`: `x [B*H*W, 3]`, z bins `[B, H, W]` (int32), `enc [B, h/16, w/16, C]` (NHWC)."""
        levels = [_grid_level(z)]
        pools: list[_Pool] = []
        skips: list[mx.array] = []
        x = x @ self.input_proj[0] + self.input_proj[1]
        for i, blocks in enumerate(self.down):
            x = self._block(blocks, x, levels[i])
            if i < len(self.downsample):
                skips.append(x)
                p = _Pool(levels[i])
                pools.append(p)
                levels.append(p.level)
                ws, bs = self.downsample[i]
                x = p.mean(x) @ ws + bs
        c = levels[-1].coords
        _, eh, ew, ec = enc.shape
        e = enc.reshape(-1, ec)[(c[:, 0] * eh + c[:, 1]) * ew + c[:, 2]]
        (w1, b1), (w2, b2) = self.fuse
        x = mx.concatenate([x, e @ self.encoder_fuse[0] + self.encoder_fuse[1]], -1)
        x = _silu(x @ w1 + b1) @ w2 + b2
        x = self._block(self.bottleneck, x, levels[-1])
        for i, ((wu, bu), blocks) in enumerate(zip(self.upsample, self.up, strict=True)):
            t = len(levels) - 2 - i
            x = (x @ wu + bu)[pools[t].parent] + skips[t]
            x = self._block(blocks, x, levels[t])
        return x @ self.out_proj[0] + self.out_proj[1]


def refine(refiner: MlxRefiner, coords: torch.Tensor, enc: torch.Tensor, steps: int, depth_res: int) -> torch.Tensor:
    """`model.refine` on MLX: `coords [B, 3, h, w]`, `enc [B, C, h/16, w/16]` (float32 CPU) -> refined coords."""
    b, _, h, w = coords.shape
    c = mx.transpose(to_mx(coords.float()), (0, 2, 3, 1))
    e = mx.transpose(to_mx(enc.float()), (0, 2, 3, 1))
    for _ in range(steps):
        logz = c[..., 2]
        zq = mx.round(logz * depth_res).astype(mx.int32)
        zq = zq - zq.min(axis=(1, 2), keepdims=True)
        out = refiner(c.reshape(-1, 3), zq, e)
        c = mx.concatenate([c[..., :2], (logz + out.reshape(b, h, w))[..., None]], -1)
        mx.eval(c)
    return to_torch(mx.transpose(c, (0, 3, 1, 2)))
