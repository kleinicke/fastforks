# ruff: noqa: PLR0913, PLR0917, PLR2004 - upstream argument lists and fixed sizes
"""ReSplat's refinement (`Net.refine`, upstream `forward_update`) on Apple MLX: `mx` operations and three Metal kernels.

The encoder runs as an exported graph (`mlx_wrap.port`); the render-and-compare loop has data-dependent shapes (the
rasteriser's tile lists, the kNN of moving points), so it is written here by hand and runs eagerly on the GPU:

- `render`: `render.py` (gsplat's antialiased rasterisation) as a projection kernel (covariance from scale and
  rotation, EWA projection, opacity compensation, tile rectangle, SH colour; one thread per Gaussian and view), an
  intersection kernel writing `(view, tile, depth rank)` keys, one `mx.sort` of the keys, and a compositing kernel
  (one thread per pixel, one threadgroup per 16 x 16 tile, the tile's Gaussians staged through threadgroup memory,
  front to back until the transmittance would fall to 1e-4).
- `knn`: exact k nearest neighbours by brute force over z-ordered tiles of 64 points, skipping a tile whose bounding
  box is farther than the current k-th distance (the query's own tile is searched first).
- `knn_attention`: the point transformer's attention over those neighbours, fused (no `[N, K, C]` gathers).
- the ResNet-18 features, the multi-view low-resolution attention (`mx.fast.scaled_dot_product_attention`), the point
  transformer and the update head in `mx`; per-point layers run in row chunks with an evaluation per chunk.

Precision: the renderer, the Gaussian state and the updates are float32; the networks run in the run dtype `dt`
(float32, or float16 with float32 LayerNorm / softmax statistics), as the PyTorch half path does under autocast.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import metal_kernel, mx, to_mx
from fastforks.multiview._runner import invert

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

F32 = mx.float32
SH = 16  # fixed: SH degree 3
STATE = 3 + 3 + 4 + 1 + 3 * SH  # fixed: means, scales, rotation, opacity logit, SH
STATE_CHANNELS = 512  # fixed
RESNET_CHANNELS = 256  # fixed: ResNet-18 conv1, layer1, layer2
SCALE_MIN = 1e-6  # fixed: gaussian_adapter clamp_min_scale
TILE = 16  # fixed: gsplat's tile size
KNN_TILE = 64  # points per z-order tile of the kNN search
ROWS = 65536  # points per chunk of the per-point layers (bounds the [rows, 4C] hidden layer)

_RENDER_CONSTS = (
    "constant constexpr float EPS2D = 0.1f;\n"
    "constant constexpr float ALPHA_MIN = 1.0f / 255.0f;\n"
    "constant constexpr float ALPHA_MAX = 0.99f;\n"
    "constant constexpr float T_MIN = 1e-4f;\n"
    "constant constexpr float EXTEND = 3.33f;\n"
    "constant constexpr float MIN_COMP = 0.005f;\n"
)

_PROJECT = metal_kernel(
    name="resplat_project",
    header=_RENDER_CONSTS,
    input_names=["means", "scales", "rot", "opac", "sh", "src_r", "w2c", "kin", "cam", "nf"],
    output_names=["depth", "data", "rect"],
    source="""
        // one thread: Gaussian i in view v (render.py `_project` and the SH colour)
        uint i = thread_position_in_grid.x, v = thread_position_in_grid.y;
        int n = means_shape[0];
        if (int(i) >= n) return;
        size_t o = size_t(v) * n + i;
        depth[o] = INFINITY;
        for (int t = 0; t < 4; ++t) rect[o * 4 + t] = 0;
        for (int t = 0; t < 9; ++t) data[o * 9 + t] = 0.0f;
        float px = means[3 * i], py = means[3 * i + 1], pz = means[3 * i + 2];
        const device float* wv = w2c + 12 * v;
        float x = wv[0] * px + wv[1] * py + wv[2] * pz + wv[3];
        float y = wv[4] * px + wv[5] * py + wv[6] * pz + wv[7];
        float z = wv[8] * px + wv[9] * py + wv[10] * pz + wv[11];
        float near = nf[0], far = nf[1];
        if (!(z >= near && z <= far)) return;
        // world covariance: R_c2w(source view) (R(q) S)(R(q) S)^T R_c2w^T, upper triangle mirrored (gsplat)
        float qi = rot[4 * i], qj = rot[4 * i + 1], qk = rot[4 * i + 2], qr = rot[4 * i + 3];
        float qn = sqrt(qi * qi + qj * qj + qk * qk + qr * qr) + 1e-8f;
        qi /= qn; qj /= qn; qk /= qn; qr /= qn;
        float s2 = 2.0f / (qi * qi + qj * qj + qk * qk + qr * qr + 1e-8f);
        float q[3][3] = {
            {1 - s2 * (qj * qj + qk * qk), s2 * (qi * qj - qk * qr), s2 * (qi * qk + qj * qr)},
            {s2 * (qi * qj + qk * qr), 1 - s2 * (qi * qi + qk * qk), s2 * (qj * qk - qi * qr)},
            {s2 * (qi * qk - qj * qr), s2 * (qj * qk + qi * qr), 1 - s2 * (qi * qi + qj * qj)}};
        float sc[3] = {scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]};
        float m[3][3], c[3][3], a[3][3], w[3][3], rc[3][3];
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) m[r][s] = q[r][s] * sc[s];
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) c[r][s] = m[r][0] * m[s][0] + m[r][1] * m[s][1] + m[r][2] * m[s][2];
        const device float* sr = src_r + 9 * (i / NPV);
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) rc[r][s] = sr[3 * r + s];
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) a[r][s] = rc[r][0] * c[0][s] + rc[r][1] * c[1][s] + rc[r][2] * c[2][s];
        for (int r = 0; r < 3; ++r)
            for (int s = r; s < 3; ++s) w[r][s] = a[r][0] * rc[s][0] + a[r][1] * rc[s][1] + a[r][2] * rc[s][2];
        for (int r = 1; r < 3; ++r)
            for (int s = 0; s < r; ++s) w[r][s] = w[s][r];
        // camera covariance R W R^T
        float rv[3][3] = {{wv[0], wv[1], wv[2]}, {wv[4], wv[5], wv[6]}, {wv[8], wv[9], wv[10]}};
        float t1[3][3], cc[3][3];
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) t1[r][s] = rv[r][0] * w[0][s] + rv[r][1] * w[1][s] + rv[r][2] * w[2][s];
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s) cc[r][s] = t1[r][0] * rv[s][0] + t1[r][1] * rv[s][1] + t1[r][2] * rv[s][2];
        const device float* kv = kin + 4 * v;
        float fx = kv[0], fy = kv[1], cx = kv[2], cy = kv[3];
        float tfx = 0.5f * float(W) / fx, tfy = 0.5f * float(H) / fy;
        float tx = z * clamp(x / z, -(cx / fx + 0.3f * tfx), (float(W) - cx) / fx + 0.3f * tfx);
        float ty = z * clamp(y / z, -(cy / fy + 0.3f * tfy), (float(H) - cy) / fy + 0.3f * tfy);
        float j00 = fx / z, j02 = -fx * tx / (z * z), j11 = fy / z, j12 = -fy * ty / (z * z);
        float u0[3], u1[3];  // J cc
        for (int s = 0; s < 3; ++s) {
            u0[s] = j00 * cc[0][s] + j02 * cc[2][s];
            u1[s] = j11 * cc[1][s] + j12 * cc[2][s];
        }
        float ca = u0[0] * j00 + u0[2] * j02, cb = u0[1] * j11 + u0[2] * j12;
        float cb2 = u1[0] * j00 + u1[2] * j02, ccc = u1[1] * j11 + u1[2] * j12;
        float det0 = ca * ccc - cb * cb2;
        ca += EPS2D; ccc += EPS2D;
        float det = ca * ccc - cb * cb2;
        if (!(det > 0.0f)) return;
        float comp = sqrt(max(det0 / det, MIN_COMP * MIN_COMP));
        float op = opac[i] * comp;
        if (!(op >= ALPHA_MIN)) return;
        float mx_ = fx * x / z + cx, my_ = fy * y / z + cy;
        float ext = min(sqrt(2.0f * log(max(op, ALPHA_MIN) / ALPHA_MIN)), EXTEND);
        float rx = ceil(ext * sqrt(max(ca, 0.0f))), ry = ceil(ext * sqrt(max(ccc, 0.0f)));
        if (!(rx > 0.0f || ry > 0.0f)) return;
        if (!(mx_ + rx > 0.0f && mx_ - rx < float(W) && my_ + ry > 0.0f && my_ - ry < float(H))) return;
        int x0 = int(max(floor((mx_ - rx) / TILE), 0.0f)), y0 = int(max(floor((my_ - ry) / TILE), 0.0f));
        int x1 = min(int(ceil((mx_ + rx) / TILE)), TW), y1 = min(int(ceil((my_ + ry) / TILE)), TH);
        int nx = max(x1 - x0, 0), ny = max(y1 - y0, 0);
        if (nx * ny == 0) return;
        // colour: SH at the direction from the camera centre
        float dx = px - cam[3 * v], dy = py - cam[3 * v + 1], dz = pz - cam[3 * v + 2];
        float dn = sqrt(dx * dx + dy * dy + dz * dz);
        dx /= dn; dy /= dn; dz /= dn;
        float z2 = dz * dz, c1 = dx * dx - dy * dy, s1 = 2.0f * dx * dy;
        float c2 = dx * c1 - dy * s1, s2b = dx * s1 + dy * c1;
        float ba = -0.48860251190292f, bb = -1.092548430592079f * dz;
        float bc = -2.285228997322329f * z2 + 0.4570457994644658f, be = 1.445305721320277f * dz;
        float basis[16] = {
            0.2820947917738781f, ba * dy, -ba * dz, ba * dx,
            0.5462742152960395f * s1, bb * dy, 0.9461746957575601f * z2 - 0.3153915652525201f, bb * dx,
            0.5462742152960395f * c1, -0.5900435899266435f * s2b, be * s1, bc * dy,
            dz * (1.865881662950577f * z2 - 1.119528997770346f), bc * dx, be * c1, -0.5900435899266435f * c2};
        const device float* shi = sh + size_t(i) * 48;
        float col[3];
        for (int ch = 0; ch < 3; ++ch) {
            float acc = 0.0f;
            for (int s = 0; s < 16; ++s) acc += shi[ch * 16 + s] * basis[s];
            col[ch] = max(acc + 0.5f, 0.0f);
        }
        depth[o] = z;
        float d9[9] = {mx_, my_, ccc / det, -(cb + cb2) / 2.0f / det, ca / det, op, col[0], col[1], col[2]};
        for (int t = 0; t < 9; ++t) data[o * 9 + t] = d9[t];
        rect[o * 4] = x0; rect[o * 4 + 1] = y0; rect[o * 4 + 2] = nx; rect[o * 4 + 3] = ny;
    """,
)

_ISECT = metal_kernel(
    name="resplat_isect",
    input_names=["order", "rect", "off"],
    output_names=["keys"],
    source="""
        // one thread: the depth-rank r Gaussian of view v writes one key per touched tile
        uint r = thread_position_in_grid.x, v = thread_position_in_grid.y;
        int n = order_shape[1];
        if (int(r) >= n) return;
        size_t o = size_t(v) * n + r;
        size_t g = size_t(v) * n + order[o];
        int x0 = rect[g * 4], y0 = rect[g * 4 + 1], nx = rect[g * 4 + 2], ny = rect[g * 4 + 3];
        size_t start = off[o];
        for (int j = 0; j < nx * ny; ++j) {
            KT tile = KT(v) * (TW * TH) + KT((y0 + j / nx) * TW + x0 + j % nx);
            keys[start + j] = tile * KT(n) + KT(r);
        }
    """,
)

_RASTER = metal_kernel(
    name="resplat_raster",
    header=_RENDER_CONSTS,
    input_names=["keys", "ranges", "order", "data"],
    output_names=["image"],
    source="""
        // one threadgroup: one 16 x 16 tile of view v; one thread: one pixel, composited front to back
        uint2 gp = thread_position_in_grid.xy;
        uint v = thread_position_in_grid.z;
        uint2 tg = threadgroup_position_in_grid.xy;
        uint lid = thread_position_in_threadgroup.y * TILE + thread_position_in_threadgroup.x;
        int n = order_shape[1];
        size_t tile = size_t(v) * (TW * TH) + tg.y * TW + tg.x;
        int start = ranges[2 * tile], end = ranges[2 * tile + 1];
        float px = float(gp.x) + 0.5f, py = float(gp.y) + 0.5f;
        threadgroup float4 sa[TILE * TILE];
        threadgroup float4 sb[TILE * TILE];
        threadgroup float sc[TILE * TILE];
        threadgroup int alive[TILE * TILE / 32];
        float tr = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f;
        bool done = false;
        for (int base = start; base < end; base += TILE * TILE) {
            int e = base + int(lid);
            if (e < end) {
                int rank = int(keys[e] % KT(n));
                const device float* d = data + (size_t(v) * n + order[size_t(v) * n + rank]) * 9;
                sa[lid] = float4(d[0], d[1], d[2], d[3]);
                sb[lid] = float4(d[4], d[5], d[6], d[7]);
                sc[lid] = d[8];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int cnt = min(TILE * TILE, end - base);
            for (int t = 0; t < cnt && !done; ++t) {
                float4 ga = sa[t], gb = sb[t];
                float dx = px - ga.x, dy = py - ga.y;
                float sigma = 0.5f * (ga.z * dx * dx + gb.x * dy * dy) + ga.w * dx * dy;
                if (sigma < 0.0f) continue;
                float alpha = min(ALPHA_MAX, gb.y * exp(-sigma));
                if (alpha < ALPHA_MIN) continue;
                float nt = tr * (1.0f - alpha);
                if (nt <= T_MIN) { done = true; break; }
                float wgt = alpha * tr;
                r += wgt * gb.z; g += wgt * gb.w; b += wgt * sc[t];
                tr = nt;
            }
            bool any = simd_any(!done);
            if (thread_index_in_simdgroup == 0) alive[simdgroup_index_in_threadgroup] = any ? 1 : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int live = 0;
            for (int s = 0; s < TILE * TILE / 32; ++s) live += alive[s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live == 0) break;
        }
        size_t p = ((size_t(v) * TH * TILE + gp.y) * (TW * TILE) + gp.x) * 3;
        image[p] = r; image[p + 1] = g; image[p + 2] = b;
    """,
)

_KNN = metal_kernel(
    name="resplat_knn",
    input_names=["ps", "perm", "box"],
    output_names=["idx"],
    source="""
        // one thread: query point q (z-ordered); exact K nearest by squared distance, tiles culled by bounding box
        uint q = thread_position_in_grid.x;
        int n = ps_shape[0];
        if (int(q) >= n) return;
        float qx = ps[3 * q], qy = ps[3 * q + 1], qz = ps[3 * q + 2];
        float bd[K];
        int bi[K];
        for (int s = 0; s < K; ++s) { bd[s] = INFINITY; bi[s] = int(q); }
        int nt = (n + TS - 1) / TS, own = int(q) / TS;
        for (int tt = 0; tt < nt; ++tt) {
            int t = tt == 0 ? own : (tt <= own ? tt - 1 : tt);
            const device float* bx = box + 6 * t;
            float gx = max(max(bx[0] - qx, qx - bx[3]), 0.0f);
            float gy = max(max(bx[1] - qy, qy - bx[4]), 0.0f);
            float gz = max(max(bx[2] - qz, qz - bx[5]), 0.0f);
            if (gx * gx + gy * gy + gz * gz > bd[K - 1] * 1.000001f) continue;
            int e = min(n, (t + 1) * TS);
            for (int j = t * TS; j < e; ++j) {
                float dx = ps[3 * j] - qx, dy = ps[3 * j + 1] - qy, dz = ps[3 * j + 2] - qz;
                float d = dx * dx + dy * dy + dz * dz;
                if (d < bd[K - 1]) {
                    int c = j;
                    for (int s = 0; s < K; ++s) {
                        if (d < bd[s]) {
                            float td = bd[s]; bd[s] = d; d = td;
                            int ti = bi[s]; bi[s] = c; c = ti;
                        }
                    }
                }
            }
        }
        size_t o = size_t(perm[q]) * K;
        for (int s = 0; s < K; ++s) idx[o + s] = perm[bi[s]];
    """,
)

_KNN_ATTN = metal_kernel(
    name="resplat_knn_attention",
    input_names=["qkv", "nbr"],
    output_names=["out"],
    source="""
        // one thread: point i attends over its K neighbours (softmax(q.k / sqrt(C)) v, float32 accumulation)
        uint i = thread_position_in_grid.x;
        int n = qkv_shape[0];
        if (int(i) >= n) return;
        const device vec<T, 4>* row = (const device vec<T, 4>*)(qkv + size_t(i) * 3 * C);
        float s[K];
        float hi = -INFINITY;
        for (int k = 0; k < K; ++k) {
            const device vec<T, 4>* kr = (const device vec<T, 4>*)(qkv + size_t(nbr[size_t(i) * K + k]) * 3 * C + C);
            float acc = 0.0f;
            for (int c = 0; c < C / 4; ++c) acc += dot(float4(row[c]), float4(kr[c]));
            s[k] = acc * rsqrt(float(C));
            hi = max(hi, s[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) { s[k] = exp(s[k] - hi); sum += s[k]; }
        for (int c = 0; c < C / 4; ++c) {
            float4 acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const device vec<T, 4>* vr = (const device vec<T, 4>*)(qkv + size_t(nbr[size_t(i) * K + k]) * 3 * C + 2 * C);
                acc += s[k] * float4(vr[c]);
            }
            ((device vec<T, 4>*)out)[size_t(i) * (C / 4) + c] = vec<T, 4>(acc / sum);
        }
    """,
)


# --- Gaussian rasterisation -----------------------------------------------------------------------------------------


def render(
    means: mx.array, scales: mx.array, rot: mx.array, opac: mx.array, sh: mx.array, cams: dict[str, mx.array],
    h: int, w: int,
) -> mx.array:  # fmt: skip
    """Rendered views `[V, h, w, 3]` (float32) of the view-major Gaussians (`render.py` + `Net._render`).

    `means`, `scales` `[V, n, 3]`, unnormalised `x y z w` rotations `[V, n, 4]` in the frame of their view's camera,
    opacities `[V, n]`, SH `[V, n, 3, 16]`; `cams` from `cameras`.
    """
    vs, npv = means.shape[:2]
    n = vs * npv
    tw, th = -(-w // TILE), -(-h // TILE)
    v = cams["w2c"].shape[0]
    consts = [
        ("H", h), ("W", w), ("TW", tw), ("TH", th), ("NPV", npv), ("TILE", TILE),
    ]  # fmt: skip
    depth, data, rect = _PROJECT(
        inputs=[
            means.reshape(n, 3),
            scales.reshape(n, 3),
            rot.reshape(n, 4),
            opac.reshape(n),
            sh.reshape(n, 3 * SH),
            cams["src_r"],
            cams["w2c"],
            cams["kin"],
            cams["cam"],
            cams["nf"],
        ],
        template=consts,
        grid=(n, v, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(v, n), (v, n, 9), (v, n, 4)],
        output_dtypes=[F32, F32, mx.int32],
    )
    order = mx.argsort(depth, axis=1).astype(mx.int32)  # front to back; culled Gaussians (inf) last
    cnt = mx.take_along_axis(rect[..., 2] * rect[..., 3], order, axis=1).reshape(-1)
    end = mx.cumsum(cnt.astype(mx.int64))
    total = int(end[-1].item()) if cnt.size else 0
    kt = mx.uint32 if v * tw * th * n < 2**32 else mx.uint64
    tiles = v * tw * th
    if total:
        (keys,) = _ISECT(
            inputs=[order, rect, (end - cnt).astype(mx.int64)],
            template=[("TW", tw), ("TH", th), ("KT", kt)],
            grid=(n, v, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(total,)],
            output_dtypes=[kt],
        )
        keys = mx.sort(keys)
        tile = (keys // n).astype(mx.int32)
        per = mx.zeros((tiles,), mx.int32).at[tile].add(mx.ones_like(tile))
    else:
        keys = mx.zeros((1,), kt)
        per = mx.zeros((tiles,), mx.int32)
    hi = mx.cumsum(per)
    ranges = mx.stack((hi - per, hi), -1)
    (image,) = _RASTER(
        inputs=[keys, ranges, order, data],
        template=[("TW", tw), ("TH", th), ("TILE", TILE), ("KT", kt)],
        grid=(tw * TILE, th * TILE, v),
        threadgroup=(TILE, TILE, 1),
        output_shapes=[(v, th * TILE, tw * TILE, 3)],
        output_dtypes=[F32],
    )
    return image[:, :h, :w]


def cameras(c2w: torch.Tensor, k_norm: torch.Tensor, h: int, w: int, near: float, far: float) -> dict[str, mx.array]:
    """The renderer's per-view camera arrays: source rotations, world-to-camera, pixel intrinsics, centres, depth bounds."""
    c2w = c2w.float()
    w2c = invert(c2w)
    k = k_norm.float() * torch.tensor((w, h, 1.0))[:, None]
    return {
        "src_r": to_mx(c2w[:, :3, :3].reshape(-1, 9)),
        "w2c": to_mx(w2c[:, :3, :4].reshape(-1, 12)),
        "kin": to_mx(torch.stack((k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]), -1)),
        "cam": to_mx(c2w[:, :3, 3].contiguous()),
        "nf": mx.array([near, far], F32),
    }


# --- kNN and point attention ----------------------------------------------------------------------------------------


def _morton(p: mx.array) -> mx.array:
    """z-order codes of 10-bit cells of the bounding box (`point_transformer._morton`)."""
    lo, hi = p.min(0), p.max(0)
    v = ((p - lo) / mx.maximum(hi - lo, 1e-12) * 1023).astype(mx.uint32)
    for shift, mask in ((16, 0x030000FF), (8, 0x0300F00F), (4, 0x030C30C3), (2, 0x09249249)):
        v = (v | (v << shift)) & mask
    return v[:, 0] | (v[:, 1] << 1) | (v[:, 2] << 2)


def knn(p: mx.array, k: int) -> mx.array:
    """Indices `[N, k]` (int32) of the `k` nearest points of every point of `p` `[N, 3]` (itself included), exact."""
    n = p.shape[0]
    p = p.astype(F32)
    perm = mx.argsort(_morton(p)).astype(mx.int32)
    ps = p[perm]
    pad = -n % KNN_TILE
    tiles = mx.concatenate((ps, mx.repeat(ps[-1:], pad, 0))) if pad else ps
    tiles = tiles.reshape(-1, KNN_TILE, 3)
    box = mx.concatenate((tiles.min(1), tiles.max(1)), -1)
    (idx,) = _KNN(
        inputs=[ps, perm, box],
        template=[("K", k), ("TS", KNN_TILE)],
        grid=(n, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n, k)],
        output_dtypes=[mx.int32],
    )
    return idx


def knn_attention(qkv: mx.array, nbr: mx.array) -> mx.array:
    """`KNNAttention` without its projections: `qkv` `[N, 3C]`, neighbours `[N, K]` -> `[N, C]`."""
    n, c3 = qkv.shape
    c = c3 // 3
    (out,) = _KNN_ATTN(
        inputs=[qkv, nbr],
        template=[("T", qkv.dtype), ("C", c), ("K", nbr.shape[1])],
        grid=(n, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, c)],
        output_dtypes=[qkv.dtype],
    )
    return out


# --- networks (NHWC maps, `[rows, C]` tokens) -----------------------------------------------------------------------


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU, evaluated in float32."""
    y = x.astype(F32)
    return (0.5 * y * (1.0 + mx.erf(y * (1.0 / math.sqrt(2.0))))).astype(x.dtype)


def _resize(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bilinear `align_corners=True` resize of NHWC maps (`F.interpolate`)."""
    for axis, out in ((1, size[0]), (2, size[1])):
        n = x.shape[axis]
        if n == out:
            continue
        scale = (n - 1) / (out - 1) if out > 1 else 0.0
        src = mx.arange(out, dtype=F32) * np.float32(scale)
        i0 = mx.minimum(mx.floor(src).astype(mx.int32), n - 1)
        i1 = mx.minimum(i0 + 1, n - 1)
        shape = [1, 1, 1, 1]
        shape[axis] = out
        f = (src - i0.astype(F32)).reshape(shape).astype(x.dtype)
        a, b = mx.take(x, i0, axis=axis), mx.take(x, i1, axis=axis)
        x = a + (b - a) * f
    return x


def _unshuffle(x: mx.array, r: int) -> mx.array:
    """`F.pixel_unshuffle` of NHWC maps (channel `c r^2 + i r + j`)."""
    b, h, w, c = x.shape
    return x.reshape(b, h // r, r, w // r, r, c).transpose(0, 1, 3, 5, 2, 4).reshape(b, h // r, w // r, c * r * r)


def _shuffle(x: mx.array, r: int) -> mx.array:
    """`F.pixel_shuffle` of NHWC maps."""
    b, h, w, c = x.shape
    c //= r * r
    return x.reshape(b, h, w, c, r, r).transpose(0, 1, 4, 2, 5, 3).reshape(b, h * r, w * r, c)


def _maxpool(x: mx.array) -> mx.array:
    """`MaxPool2d(3, 2, 1)` of NHWC maps."""
    _, h, w, _ = x.shape
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    x = mx.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-math.inf)
    out = None
    for i in range(3):
        for j in range(3):
            s = x[:, i : i + 2 * oh - 1 : 2, j : j + 2 * ow - 1 : 2]
            out = s if out is None else mx.maximum(out, s)
    return out


def _linear(m: nn.Module, dt: mx.Dtype) -> tuple[mx.array, mx.array | None]:
    w = to_mx(m.weight.detach().float().T.contiguous()).astype(dt)
    b = None if m.bias is None else to_mx(m.bias.detach().float()).astype(dt)
    return w, b


def _apply(lin: tuple[mx.array, mx.array | None], x: mx.array) -> mx.array:
    w, b = lin
    return x @ w if b is None else mx.addmm(b, x, w)


def _norm(m: nn.Module) -> tuple[mx.array, mx.array, float]:
    return to_mx(m.weight.detach().float()), to_mx(m.bias.detach().float()), float(m.eps)


def _ln(norm: tuple[mx.array, mx.array, float], x: mx.array) -> mx.array:
    """LayerNorm with float32 statistics and affine, cast back to `x`'s dtype."""
    w, b, eps = norm
    return mx.fast.layer_norm(x.astype(F32), w, b, eps).astype(x.dtype)


def _conv(m: nn.Module, dt: mx.Dtype, bn: nn.Module | None = None) -> tuple[mx.array, mx.array | None]:
    """A conv's OHWI weight and bias, with the BatchNorm after it folded in."""
    w = m.weight.detach().float()
    b = None if m.bias is None else m.bias.detach().float()
    if bn is not None:
        s = bn.weight.detach().float() / (bn.running_var.detach().float() + bn.eps).sqrt()
        w = w * s[:, None, None, None]
        b = bn.bias.detach().float() - bn.running_mean.detach().float() * s + (0 if b is None else b * s)
    w_mx = to_mx(w.permute(0, 2, 3, 1).contiguous()).astype(dt)
    return w_mx, None if b is None else to_mx(b).astype(dt)


def _run_conv(c: tuple[mx.array, mx.array | None], x: mx.array, stride: int = 1) -> mx.array:
    w, b = c
    y = mx.conv2d(x, w, stride=stride, padding=w.shape[1] // 2)
    return y if b is None else y + b


class _ResNet:
    """`ResNetFeatureWarpper`: conv1 (output 0), bn1 + relu + maxpool, layer1 (output 1), layer2 (output 2)."""

    def __init__(self, m: nn.Module, dt: mx.Dtype) -> None:
        self.conv1 = _conv(m.conv1, dt)
        s = m.bn1.weight.detach().float() / (m.bn1.running_var.detach().float() + m.bn1.eps).sqrt()
        self.bn1 = (to_mx(s).astype(dt), to_mx(m.bn1.bias.detach().float() - m.bn1.running_mean.float() * s).astype(dt))
        self.layers = [
            [
                (
                    _conv(blk.conv1, dt, blk.bn1),
                    _conv(blk.conv2, dt, blk.bn2),
                    None if blk.downsample is None else _conv(blk.downsample[0], dt, blk.downsample[1]),
                    blk.conv1.stride[0],
                )
                for blk in layer
            ]
            for layer in (m.layer1, m.layer2)
        ]

    def __call__(self, x: mx.array) -> list[mx.array]:
        out = [_run_conv(self.conv1, x, 2)]
        x = _maxpool(mx.maximum(out[0] * self.bn1[0] + self.bn1[1], 0))
        for layer in self.layers:
            for c1, c2, down, stride in layer:
                idn = x if down is None else _run_conv(down, x, stride)
                y = mx.maximum(_run_conv(c1, x, stride), 0)
                x = mx.maximum(_run_conv(c2, y) + idn, 0)
            out.append(x)
        return out


class _LowresAttn:
    """`MultViewLowresAttn(256, down_factor=4)`: pixel-unshuffled tokens through one DINOv2 block, back, conv."""

    def __init__(self, m: nn.Module, dt: mx.Dtype) -> None:
        self.r = m.down_factor
        self.proj1, self.norm1 = _linear(m.proj1, dt), _norm(m.norm1)
        self.proj2, self.norm2 = _linear(m.proj2, dt), _norm(m.norm2)
        self.conv = _conv(m.conv, dt)
        blk = m.attn
        self.heads = blk.attn.num_heads
        self.bn1, self.bn2 = _norm(blk.norm1), _norm(blk.norm2)
        self.qkv, self.proj = _linear(blk.attn.qkv, dt), _linear(blk.attn.proj, dt)
        self.fc1, self.fc2 = _linear(blk.mlp.fc1, dt), _linear(blk.mlp.fc2, dt)

    def __call__(self, x: mx.array, v: int, h: int, w: int) -> mx.array:
        """`x` `[v h w, C]` -> same."""
        r, c = self.r, x.shape[-1]
        t = _unshuffle(x.reshape(v, h, w, c), r).reshape(-1, c * r * r)
        t = _ln(self.norm1, _apply(self.proj1, t))
        n = t.shape[0]
        qkv = _apply(self.qkv, _ln(self.bn1, t)).reshape(1, n, 3, self.heads, -1).transpose(2, 0, 3, 1, 4)
        hd = qkv.shape[-1]
        o = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=hd**-0.5)
        t = t + _apply(self.proj, o.transpose(0, 2, 1, 3).reshape(n, -1))
        t = t + _apply(self.fc2, _gelu(_apply(self.fc1, _ln(self.bn2, t))))
        t = _ln(self.norm2, _apply(self.proj2, t))
        y = _run_conv(self.conv, _shuffle(t.reshape(v, h // r, w // r, -1), r))
        return y.reshape(-1, c) + x


def _rows(fn: Callable[..., mx.array], *xs: mx.array) -> mx.array:
    """`fn` over row chunks of `ROWS` of the arrays (evaluated per chunk: bounds the hidden layers' memory)."""
    n = xs[0].shape[0]
    if n <= ROWS:
        out = fn(*xs)
        mx.eval(out)
        return out
    parts = []
    for s in range(0, n, ROWS):
        part = fn(*(x[s : s + ROWS] for x in xs))
        mx.eval(part)
        parts.append(part)
    return mx.concatenate(parts)


class _PointTransformer:
    """`PlainPointTransformer(512, 8, 4 blocks, attn_proj_channels=64)` on fixed kNN indices."""

    def __init__(self, m: nn.Module, dt: mx.Dtype) -> None:
        self.k = m.knn_samples
        self.blocks = []
        for b in m.blocks:
            n1, qkv, proj = _norm(b.norm1), _linear(b.attn.qkv, dt), _linear(b.attn.proj, dt)
            n2, fc1, fc2 = _norm(b.norm2), _linear(b.mlp.fc1, dt), _linear(b.mlp.fc2, dt)

            def head(xs: mx.array, n1=n1, qkv=qkv) -> mx.array:  # noqa: ANN001
                return _apply(qkv, _ln(n1, xs))

            def tail(xs: mx.array, a: mx.array, proj=proj, n2=n2, fc1=fc1, fc2=fc2) -> mx.array:  # noqa: ANN001
                y = xs + _apply(proj, a)
                return y + _apply(fc2, _gelu(_apply(fc1, _ln(n2, y))))

            # compiled: the GELU and LayerNorms fuse (the MLP chunk ~30 % faster)
            self.blocks.append((mx.compile(head), mx.compile(tail)))

    def __call__(self, p: mx.array, x: mx.array) -> mx.array:
        nbr = knn(p, self.k)
        for head, tail in self.blocks:
            q = _rows(head, x)
            a = knn_attention(q, nbr)
            del q
            x = _rows(tail, x, a)
        return x


class Refine:
    """`Net.refine` on MLX (same call and result: PyTorch tensors in and out)."""

    def __init__(self, net: nn.Module, dtype: str) -> None:  # noqa: D107
        dt = mx.float16 if dtype == "float16" else F32
        self.p, self.dt = net.p, dt
        self.update_proj = _conv(net.update_proj, dt)
        self.rgb_proj, self.rgb_norm = _linear(net.update_rgb_error_proj[0], dt), _norm(net.update_rgb_error_proj[1])
        self.lin = _linear(net.update_module[0].linear, dt)
        self.pt = _PointTransformer(net.update_module[1], dt)
        self.head = [_linear(net.update_head[i], dt) for i in (0, 2, 4, 6)]
        self._head_compiled = mx.compile(self._head)
        self.resnet = _ResNet(net.update_feature, dt)
        self.attn = [_LowresAttn(b, dt) for b in net.update_error_attn]
        self.mean = to_mx(net.mean.float().reshape(3))
        self.std = to_mx(net.std.float().reshape(3))

    def _features(self, x: mx.array, size: tuple[int, int]) -> mx.array:
        feats = self.resnet(((x - self.mean) / self.std).astype(self.dt))
        return mx.concatenate([_resize(f, size) for f in feats], -1)

    def _head(self, s: mx.array) -> mx.array:
        for lin in self.head[:-1]:
            s = _gelu(_apply(lin, s))
        return _apply(self.head[-1], s).astype(F32)

    def __call__(
        self, g: dict[str, torch.Tensor], images: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor, near: float,
        far: float, steps: int,
    ) -> dict[str, torch.Tensor]:  # fmt: skip
        """Upstream `forward_update`: `steps` recurrent updates of the Gaussians of `encode`."""
        p, (v, _, h, w), dt = self.p, images.shape, self.dt
        m = p.gaussians_per_point
        r = round(m**0.5)
        if m > 1 and not p.same_points:
            msg = "ReSplat MLX refine: only one state per Gaussian (refine_same_num_points)"
            raise NotImplementedError(msg)
        state = _run_conv(self.update_proj, to_mx(g["condition"].float().permute(0, 2, 3, 1)).astype(dt))
        if m in (4, 16):
            hl, wl = state.shape[1:3]
            state = _resize(state, (hl * r, wl * r))
        fh, fw = state.shape[1:3]
        state = state.reshape(-1, STATE_CHANNELS)
        mx.eval(state)
        cams = cameras(c2w, k, h, w, near, far)
        imgs = to_mx(images.float().permute(0, 2, 3, 1))
        means, scales, rot = (to_mx(g[n].float()) for n in ("means", "scales", "rotations"))
        o = mx.clip(to_mx(g["opacities"].float()), 1e-6, 1 - 1e-6)
        opac = mx.log(o / (1 - o))[..., None]
        sh = to_mx(g["harmonics"].float()).reshape(v, -1, 3 * SH)
        alpha = to_mx(g["opacities"].float())
        target = None
        for _ in range(steps):
            rendered = render(means, scales, rot, alpha, sh, cams, h, w)
            mx.eval(rendered)
            if target is None:
                f = self._features(mx.concatenate((rendered, imgs)), (fh, fw))
                f, target = f[:v], f[v:]
                mx.eval(target)
            else:
                f = self._features(rendered, (fh, fw))
            err = (f - target).reshape(-1, f.shape[-1])
            rgb = rendered - imgs
            if m != 4:
                rgb = _unshuffle(rgb, p.latent)
            err = err + _ln(self.rgb_norm, _apply(self.rgb_proj, rgb.reshape(err.shape[0], -1).astype(dt)))
            mx.eval(err)
            del f, rgb, rendered
            for blk in self.attn:
                err = blk(err, v, fh, fw)
            mx.eval(err)
            prev = mx.concatenate((means, scales, rot, opac, sh), -1).reshape(-1, STATE)
            cloud = means.reshape(-1, 3)
            pxo = _rows(lambda a, b, c: _apply(self.lin, mx.concatenate((a.astype(dt), b, c), -1)), prev, state, err)
            del prev, err
            state = self.pt(cloud, pxo) + state
            mx.eval(state)
            del pxo
            delta = _rows(self._head_compiled, state).reshape(v, -1, STATE)
            means = means + delta[..., :3]
            scales = mx.maximum(scales + delta[..., 3:6], SCALE_MIN)
            rot = rot + delta[..., 6:10]
            opac = opac + delta[..., 10:11]
            sh = sh + delta[..., 11:]
            alpha = mx.sigmoid(opac[..., 0])
            mx.eval(means, scales, rot, opac, sh, alpha)
            del delta
        to_t = _to_torch
        return {
            "means": to_t(means),
            "scales": to_t(scales),
            "rotations": to_t(rot),
            "opacities": to_t(alpha),
            "harmonics": to_t(sh).view(v, -1, 3, SH),
        }


def _to_torch(a: mx.array) -> torch.Tensor:
    a = a.astype(F32)
    mx.eval(a)
    return torch.from_numpy(np.array(a))


def _register() -> None:
    """The encoder's exported `fastforks::resplat_knn` (`point_transformer.knn_op`) lowers to `knn`."""
    from fastforks.shared.mlx_graph import OPS  # noqa: PLC0415 - the MLX graph translator

    from . import point_transformer  # noqa: F401, PLC0415 - registers the op

    OPS[torch.ops.fastforks.resplat_knn.default] = knn


_register()
