"""Sapiens segmentation on Apple MLX, written in `mlx.core` operations (`MlxSapiens(net, dtype)`).

The maths of `net.py` (`Sapiens.forward`), channels last. Written out instead of replayed from an exported graph: the
decoder's 768-channel maps at 512 x 384 are 302 MB each in float16, and the translated graph spent more time on them
than on the 24 transformer blocks (M1 Pro, 0.3b float16: 1.05 s of 1.8 s).

- backbone: the patch convolution is one matrix product over the unfolded patches, the blocks are the shared
  `mlx_vit.block` (fused attention, `mx.fast.layer_norm`, GEMMs with the bias fused).
- 4 x 4 stride-2 transposed convolutions: each of the four output phases is a 2 x 2 convolution of the input, run as
  four accumulating GEMMs over row slices of the zero-padded map flattened to `[rows, Cin]` (a spatial shift is a row
  offset there, so no shifted copy exists), then the phases are interleaved. 256 x 192 -> 512 x 384, 768 channels,
  float16: 220 ms against 346 ms for `mx.conv_transpose2d`.
- InstanceNorm + SiLU: the float32 statistics are one pass of the shared `channel_stats` kernel, the normalisation and
  the activation one kernel that computes in float32 (8 ms per 512 x 384 map; 89 ms as separate array operations).
- 1 x 1 convolutions: GEMMs with the bias fused.

One image at a time through the decoder (its maps are the peak), each stage evaluated before the next is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import metal_kernel, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_vit

from .net import IN_EPS, LN_EPS, PATCH

if TYPE_CHECKING:
    import torch

    from .net import Sapiens

__all__ = ["MlxSapiens"]

_PHASE_TAPS = ((3, 1), (2, 0))  # kernel index of the two taps of output phase r, by increasing input offset


_NORM_SILU = metal_kernel(
    name="sapiens_norm_silu",
    input_names=["x", "a", "b"],
    output_names=["out"],
    source="""
        // silu(x * a[c] + b[c]) of a channels-last map, computed in float32 and rounded once
        uint i = thread_position_in_grid.x;
        uint c = i % C;
        float v = float(x[i]) * a[c] + b[c];
        out[i] = T(v / (1.0f + exp(-v)));
    """,
)


def _inorm_silu(x: mx.array) -> mx.array:
    """InstanceNorm (no affine) + SiLU of one channels-last map `[..., C]`; float32 statistics and arithmetic."""
    c = x.shape[-1]
    x = mx.contiguous(x)
    mean, var = ops.channel_stats(x.reshape(-1, c))
    a = mx.rsqrt(var + IN_EPS)
    return _NORM_SILU(
        inputs=[x, a, -mean * a], template=[("T", x.dtype), ("C", c)], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]  # fmt: skip


def _deconv(x: mx.array, taps: list[list[mx.array]]) -> mx.array:
    """4 x 4 stride-2 padding-1 transposed convolution (no bias) of one map `[H, W, Cin]` -> `[2 H, 2 W, Cout]`.

    `taps[2 * rh + rw]`: the four `[Cin, Cout]` matrices of output phase `(rh, rw)`, input offsets in row-major order.
    """
    h, w, c = x.shape
    wp = w + 2
    flat = mx.pad(x, ((1, 2), (1, 1), (0, 0))).reshape(-1, c)  # one spare row: every slice below stays inside
    phases = []
    for rh in (0, 1):
        for rw in (0, 1):
            acc = None
            for i, m in enumerate(taps[2 * rh + rw]):
                off = (rh + i // 2) * wp + rw + i % 2
                rows = flat[off : off + h * wp]
                acc = rows @ m if acc is None else mx.addmm(acc, rows, m)
            phase = acc.reshape(h, wp, -1)[:, :w]  # ty: ignore[unresolved-attribute] - four taps: never None
            mx.eval(phase)  # the four products of a phase are freed before the next phase's are made
            phases.append(phase)
    y = mx.stack(phases).reshape(2, 2, h, w, -1)
    return y.transpose(2, 0, 3, 1, 4).reshape(2 * h, 2 * w, -1)


def _array(t: torch.Tensor | None) -> mx.array:
    """A float32 parameter on MLX (a bias the layer has)."""
    assert t is not None  # noqa: S101 - the checkpoint's layers all have one
    return mx.array(t.detach().numpy())


def _linear(m: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - an `nn.Linear`
    """`[in, out]` weight (contiguous: MLX's GEMM copies a transposed view on every call) and bias."""
    w = mx.contiguous(mx.array(m.weight.detach().numpy()).T).astype(dtype)
    b = mx.array(m.bias.detach().numpy()).astype(dtype)
    mx.eval(w, b)
    return w, b


class MlxSapiens:
    """The network of one loaded model: weights in the run dtype, `logits(x)` channels-last on MLX."""

    def __init__(self, net: Sapiens, dtype: str) -> None:
        """Convert the float32 PyTorch network's weights (layer by layer: no second float32 copy of the model)."""
        self.dtype = dt = getattr(mx, dtype)
        bb, head = net.backbone, net.decode_head
        p: dict[str, mx.array] = {}

        def put(name: str, t: torch.Tensor) -> None:
            p[name] = mx.array(t.detach().numpy()).astype(dt)

        proj = bb.patch_embed.projection
        self.embed = proj.weight.shape[0]
        patch_w = mx.array(proj.weight.detach().numpy()).transpose(2, 3, 1, 0).reshape(PATCH * PATCH * 3, -1)
        p["patch.weight"] = mx.contiguous(patch_w).astype(dt)
        p["pos"] = (mx.array(bb.pos_embed.detach().numpy()[0]) + _array(proj.bias)).astype(dt)
        self.heads = bb.blocks[0].attn.heads
        self.depth = len(bb.blocks)
        for i, layer in enumerate(bb.blocks):
            b = f"blocks.{i}."
            for name, ln in (("norm1", layer.ln1), ("norm2", layer.ln2)):
                put(f"{b}{name}.weight", ln.weight)
                put(f"{b}{name}.bias", ln.bias)
            for name, lin in (
                ("qkv", layer.attn.qkv), ("proj", layer.attn.proj), ("fc1", layer.ffn.fc[0]), ("fc2", layer.ffn.fc[1])
            ):  # fmt: skip
                p[f"{b}{name}.weight"], p[f"{b}{name}.bias"] = _linear(lin, dt)
        put("norm.weight", bb.ln1.weight)
        put("norm.bias", bb.ln1.bias)
        mx.eval(p)
        self.p = p
        self.deconvs = []
        for m in head.deconvs:
            w = mx.array(m.weight.detach().numpy()).astype(dt)  # [Cin, Cout, kh, kw]
            taps = [
                [mx.contiguous(w[:, :, jh, jw]) for jh in _PHASE_TAPS[rh] for jw in _PHASE_TAPS[rw]]
                for rh in (0, 1)
                for rw in (0, 1)
            ]
            mx.eval(taps)
            self.deconvs.append(taps)
        self.convs = []
        for m in (*head.convs, head.conv_seg):
            w = mx.contiguous(mx.array(m.weight.detach().numpy())[:, :, 0, 0].T).astype(dt)
            b = _array(m.bias).astype(dt)
            mx.eval(w, b)
            self.convs.append((w, b))

        depth, heads = self.depth, self.heads

        def tokens(x: mx.array) -> mx.array:
            t = mlx_vit.patches(x.astype(dt), PATCH) @ p["patch.weight"] + p["pos"]
            for i in range(depth):
                t = mlx_vit.block(p, i, t, heads, LN_EPS)
            return mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], LN_EPS)

        self._tokens = mx.compile(tokens)

    def _decode(self, f: mx.array) -> mx.array:
        """One image's feature map `[gh, gw, C]` -> logits `[8 gh, 8 gw, classes]`."""
        for taps in self.deconvs:
            f = _inorm_silu(_deconv(f, taps))
            mx.eval(f)
        h, w, c = f.shape
        f = f.reshape(-1, c)
        for wm, b in self.convs[:-1]:
            f = _inorm_silu(mx.addmm(b, f, wm))
            mx.eval(f)
        wm, b = self.convs[-1]
        return mx.addmm(b, f, wm).reshape(h, w, -1)

    def logits(self, x: mx.array) -> mx.array:
        """Normalised `[B, 3, H, W]` (float32) -> logits `[B, H / 2, W / 2, classes]` in the run dtype (lazy)."""
        b, _, h, w = x.shape
        t = self._tokens(x).reshape(b, h // PATCH, w // PATCH, self.embed)
        mx.eval(t)
        return mx.stack([self._decode(t[i]) for i in range(b)])
