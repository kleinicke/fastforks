"""RoMa v2's local correlation on channels-last float32 features, with zero-padded bilinear sampling."""

from __future__ import annotations

from fastforks._core.mlx import metal_kernel, mx

_CORR = metal_kernel(
    name="roma_local_correlation",
    input_names=["a", "b", "warp"],
    output_names=["out"],
    source="""
        uint lane = thread_position_in_grid.x % 32, index = thread_position_in_grid.x / 32;
        uint H = a_shape[1], W = a_shape[2], C = a_shape[3], side = 2 * R + 1;
        uint pixel = index / (side*side), offset = index % (side*side), batch = pixel / (W*H);
        float u = warp[2*pixel] + 2.0f * (int(offset % side) - R) / float(W);
        float v = warp[2*pixel+1] + 2.0f * (int(offset / side) - R) / float(H);
        float x = ((u+1)*W-1)*0.5f, y = ((v+1)*H-1)*0.5f;
        if (x <= -1 || x >= float(W) || y <= -1 || y >= float(H)) {
            if (lane == 0) out[index] = 0;
            return;
        }
        int ix = int(floor(x)), iy = int(floor(y));
        float dx = x-ix, dy = y-iy, sum = 0;
        for (uint c = lane; c < C; c += 32) {
            float sampled = 0;
            for (int j = 0; j <= 1; ++j) {
                for (int i = 0; i <= 1; ++i) {
                    int xx = ix+i, yy = iy+j;
                    if (xx >= 0 && xx < int(W) && yy >= 0 && yy < int(H))
                        sampled += b[((size_t(batch)*H+yy)*W+xx)*C+c] * (i ? dx : 1-dx) * (j ? dy : 1-dy);
                }
            }
            sum += a[size_t(pixel)*C+c] * sampled;
        }
        sum = simd_sum(sum);
        if (lane == 0) out[index] = sum * rsqrt(float(C));
    """,
)


def correlation(a: mx.array, b: mx.array, warp: mx.array, radius: int) -> mx.array:
    """One normalised channel dot product per pixel and offset (row-major window order)."""
    n, h, w, _ = a.shape
    offsets = (2 * radius + 1) ** 2
    return _CORR(
        inputs=[a, b, warp],
        template=[("R", radius)],
        grid=(32 * n * h * w * offsets, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, h, w, offsets)],
        output_dtypes=[mx.float32],
    )[0]
