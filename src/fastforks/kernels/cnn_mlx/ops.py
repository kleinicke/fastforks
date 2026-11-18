"""Global spatial average of a channels-last feature map on Apple MLX."""

from __future__ import annotations

from fastforks._core.mlx import metal_kernel, mx

_POOL = metal_kernel(
    name="cnn_global_avg",
    input_names=["x"],
    output_names=["out"],
    source="""
        size_t channel = thread_position_in_grid.x, batch = thread_position_in_grid.y;
        size_t pixels = size_t(x_shape[1]) * x_shape[2], channels = x_shape[3];
        size_t base = batch * pixels * channels + channel;
        float total = 0;
        for (size_t i = 0; i < pixels; ++i) total += float(x[base + i * channels]);
        out[batch * channels + channel] = T(total / float(pixels));
    """,
)


def pool(x: mx.array) -> mx.array:
    """Average each channel in fp32, then round to the activation dtype."""
    batch, _, _, channels = x.shape
    return _POOL(inputs=[x], template=[("T", x.dtype)], grid=(channels, batch, 1),
                 threadgroup=(min(256, channels), 1, 1), output_shapes=[(batch, channels)],
                 output_dtypes=[x.dtype])[0]  # fmt: skip
