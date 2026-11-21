"""MVSAnywhere's MLP cost volume as one Apple Metal kernel (the MLX port; the Metal twin of `mvsa_cv_fused`).

One thread per (pixel, depth plane): per source view, project the plane point, bilinear-sample the source features
(`grid_sample` bilinear, zeros padding, align_corners=False), build the 46 MLP inputs implicitly, run the 46-32-32-2
MLP in float32 and reduce over the views with an online softmax of output 1 weighting output 0. Layer 1 is split as in
the CUDA kernel: the per-pixel terms (current features, current ray, bias) and the per-view terms (pose distances)
are two small GEMMs before the kernel. The plain-PyTorch graph of `cost_volume` ran 64 planes x 7 views of
48-channel matmuls and elementwise ops (270 ms per volume at 448 x 672 on an M1 Pro); the kernel keeps everything in
registers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastforks._core.mlx import metal_kernel, mx, to_mx, to_torch

if TYPE_CHECKING:
    from torch import nn

    from .model import Geometry

NH = 32  # hidden width of the cost-volume MLP
NDYN = 23  # view- and depth-dependent layer-1 inputs (warped features 16, mask, depth, dot, angle, source ray 3)
MLP_FLOATS = NH * NDYN + NH + NH * NH + NH + 2 * NH + 2
TG = 128  # threads per threadgroup (pixels of one plane)

_CV = metal_kernel(
    name="mvsa_cv_fused",
    input_names=["ray", "A", "cur", "src", "geo", "Bk", "planes", "rng", "wt"],
    output_names=["out"],
    source="""
        // wt: w1 [NH][NDYN], w1pn [NH], w2 [NH][NH], b2 [NH], w3 [2][NH], b3 [2] (the CUDA kernel's MlpWeights)
        threadgroup float sw[MLPF];
        threadgroup float sgeo[K * 16];
        threadgroup float sB[K * NH];
        uint tid = thread_position_in_threadgroup.x;
        for (uint i = tid; i < MLPF; i += TGS) sw[i] = wt[i];
        for (uint i = tid; i < K * 16; i += TGS) sgeo[i] = geo[i];
        for (uint i = tid; i < K * NH; i += TGS) sB[i] = Bk[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int hw = HH * WW;
        const int p = thread_position_in_grid.x;
        if (p >= hw) return;
        const int di = thread_position_in_grid.y;
        const threadgroup float* w1 = sw;
        const threadgroup float* w1pn = sw + NH * NDYN;
        const threadgroup float* w2 = w1pn + NH;
        const threadgroup float* b2 = w2 + NH * NH;
        const threadgroup float* w3 = b2 + NH;
        const threadgroup float* b3 = w3 + 2 * NH;
        const float d = planes[di];
        const float mn = rng[0], mxd = rng[1];
        const float rx = ray[p], ry = ray[hw + p], rz = ray[2 * hw + p];
        const float cn = max(sqrt(rx * rx + ry * ry + rz * rz), 1e-12f);
        const float cx = rx / cn, cy = ry / cn, cz = rz / cn;
        float cf[16];
        for (int i = 0; i < 16; i++) cf[i] = cur[i * hw + p];
        const float pn = (d - mn) / mxd;
        const float wx = d * rx, wy = d * ry, wz = d * rz;
        float a1[NH];
        for (int j = 0; j < NH; j++) a1[j] = A[j * hw + p] + w1pn[j] * pn;
        float m = -INFINITY, s = 0.f, acc = 0.f;
        for (int k = 0; k < K; k++) {
            const threadgroup float* g = sgeo + k * 16;
            const float X = g[0] * wx + g[1] * wy + g[2] * wz + g[3];
            const float Y = g[4] * wx + g[5] * wy + g[6] * wz + g[7];
            const float Z = g[8] * wx + g[9] * wy + g[10] * wz + g[11];
            const float zd = Z + 1e-8f;
            const float sc = fabs(Z) > 1e-8f ? 1.f / zd : 1.f;
            const float px = X * sc, py = Y * sc;
            const float maskf = zd > 0.f ? 1.f : 0.f;
            const float u = 2.f * px / float(WW) - 1.f, v = 2.f * py / float(HH) - 1.f;
            float ix = ((u + 1.f) * float(WW) - 1.f) * 0.5f, iy = ((v + 1.f) * float(HH) - 1.f) * 0.5f;
            ix = min(max(ix, -4.f), float(WW) + 4.f);  // NaN-free; far outside reads zeros anyway
            iy = min(max(iy, -4.f), float(HH) + 4.f);
            const float fx0 = floor(ix), fy0 = floor(iy);
            const int x0 = int(fx0), y0 = int(fy0);
            const float ax = ix - fx0, ay = iy - fy0;
            float wf[16];
            for (int i = 0; i < 16; i++) wf[i] = 0.f;
            const device vec<T, 4>* sk = (const device vec<T, 4>*)(src) + size_t(k) * hw * 4;
            for (int c = 0; c < 4; c++) {
                const int xx = x0 + (c & 1), yy = y0 + (c >> 1);
                if (xx < 0 || xx >= WW || yy < 0 || yy >= HH) continue;
                const float bw = ((c & 1) ? ax : 1.f - ax) * ((c >> 1) ? ay : 1.f - ay);
                const device vec<T, 4>* q = sk + (size_t(yy) * WW + xx) * 4;
                for (int i = 0; i < 4; i++) {
                    const float4 t4 = float4(q[i]);
                    wf[4 * i] += bw * t4.x;
                    wf[4 * i + 1] += bw * t4.y;
                    wf[4 * i + 2] += bw * t4.z;
                    wf[4 * i + 3] += bw * t4.w;
                }
            }
            float dot = 0.f;
            for (int i = 0; i < 16; i++) dot += wf[i] * cf[i];
            dot *= maskf;
            const float dn = (zd - mn) / mxd;
            float sx = wx - g[12], sy = wy - g[13], sz = wz - g[14];
            const float sn = max(sqrt(sx * sx + sy * sy + sz * sz), 1e-12f);
            sx /= sn; sy /= sn; sz /= sn;
            const float ang = cx * sx + cy * sy + cz * sz;
            float in[NDYN];
            for (int i = 0; i < 16; i++) in[i] = wf[i];
            in[16] = maskf; in[17] = dn; in[18] = dot; in[19] = ang; in[20] = sx; in[21] = sy; in[22] = sz;
            float h1[NH];
            for (int j = 0; j < NH; j++) {
                float t = a1[j] + sB[k * NH + j];
                for (int i = 0; i < NDYN; i++) t += w1[j * NDYN + i] * in[i];
                h1[j] = max(t, 0.01f * t);
            }
            float o0 = b3[0], o1 = b3[1];
            for (int j = 0; j < NH; j++) {
                float t = b2[j];
                for (int i = 0; i < NH; i++) t += w2[j * NH + i] * h1[i];
                t = max(t, 0.01f * t);
                o0 += w3[j] * t;
                o1 += w3[NH + j] * t;
            }
            const float mnew = max(m, o1);
            const float sc_old = exp(m - mnew), e = exp(o1 - mnew);
            s = s * sc_old + e;
            acc = acc * sc_old + e * o0;
            m = mnew;
        }
        out[size_t(di) * hw + p] = acc / s;
    """,
)

# layer-1 inputs of the MLP (upstream order): warped 0..15, current 16..31, mask 32, depth 33, plane depth 34, dot 35,
# angle 36, current ray 37..39, source ray 40..42, pose 43..45
_DYN = [*range(16), 32, 33, 35, 36, 40, 41, 42]
_DEPTH_COL = 34
_PIXEL_COLS, _RAY_COLS, _POSE_COLS = slice(16, 32), slice(37, 40), slice(43, 46)


class CostVolume:
    """The MLP of `Net.mlp` packed for the kernel; `__call__` builds the `[1, 64, h, w]` float32 cost volume."""

    def __init__(self, mlp: nn.Module) -> None:
        """Pack the three linear layers (float32)."""
        l1, l2, l3 = (m for m in mlp.modules() if isinstance(m, torch.nn.Linear))
        w1 = l1.weight.detach().float()
        packed = torch.cat(
            [w1[:, _DYN].flatten(), w1[:, _DEPTH_COL], l2.weight.flatten(), l2.bias, l3.weight.flatten(), l3.bias]
        ).float()
        if packed.numel() != MLP_FLOATS:
            msg = f"MVSAnywhere cost-volume MLP: {packed.numel()} weights, the kernel takes {MLP_FLOATS}"
            raise ValueError(msg)
        self.wt = to_mx(packed.detach())
        self.w_pix = to_mx(w1[:, _PIXEL_COLS].contiguous())
        self.w_ray = to_mx(w1[:, _RAY_COLS].contiguous())
        self.w_pose = to_mx(w1[:, _POSE_COLS].t().contiguous())
        self.b1 = to_mx(l1.bias.detach().float())[:, None]

    def __call__(
        self, cur: torch.Tensor, src: torch.Tensor, g: Geometry, rng: torch.Tensor, planes: torch.Tensor
    ) -> torch.Tensor:
        """`cur` `[1, 16, h, w]`, `src` `[K, 16, h, w]` matching features, geometry, depth range and planes."""
        k, c, h, w = src.shape
        hw = h * w

        def f32(t: torch.Tensor) -> mx.array:
            return to_mx(t.float())

        curf = f32(cur[0].reshape(c, hw))
        a = self.w_pix @ curf + self.w_ray @ f32(g.cray) + self.b1  # [32, hw] per-pixel layer-1 terms
        bk = f32(g.pose) @ self.w_pose  # [K, 32] per-view terms
        geo = f32(torch.cat([g.proj.reshape(-1, 12), g.cams, torch.zeros_like(g.cams[:, :1])], 1))
        srcl = mx.transpose(f32(src), (0, 2, 3, 1))  # [K, h, w, 16]
        out = _CV(
            inputs=[f32(g.ray), a, curf, mx.contiguous(srcl), geo, bk, f32(planes), f32(rng), self.wt],
            template=[("T", mx.float32), ("K", k), ("HH", h), ("WW", w), ("NH", NH), ("NDYN", NDYN),
                      ("MLPF", MLP_FLOATS), ("TGS", TG)],
            grid=(-(-hw // TG) * TG, planes.shape[0], 1), threadgroup=(TG, 1, 1),
            output_shapes=[(planes.shape[0], hw)], output_dtypes=[mx.float32],
        )[0]  # fmt: skip
        return to_torch(out).view(1, planes.shape[0], h, w)
