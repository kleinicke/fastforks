"""VGGT's alternating-attention aggregator on Apple MLX, built from a state dict for the cached hand ports.

The aggregator's `frame_blocks.<i>` / `global_blocks.<i>` (pre-norm blocks with per-head q / k LayerNorms and 2D
RoPE, LayerScale) in `mlx_vit`'s block layout, every weight read through `mlx_flow.put` (so `mlx_port.recorded`
caches them in the run dtype). Upstream's 2D RoPE rotates each half of a head (y on the first 32 of 64 channels,
x on the last 32) by `rotate_half` within it; `mlx_vit.rotate` turns the whole head (pairs `i`, `i + 32`). The q / k
output channels of each head are permuted to `[0:16, 32:48, 16:32, 48:64]` (the q / k norms with them), which pairs
the same channels, and `rope` builds the matching per-channel tables. q . k is unchanged by the common permutation.

`Aggregator` runs a frame block on `[S, P, C]` (each frame's tokens) and a global block on `[1, S P, C]` (all frames)
in turn, one compiled block function for both (a trace per shape), one block in flight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_flow as mf
from fastforks.shared.mlx_vit import block

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["Aggregator", "rope"]

QUARTER = 16  # a head of 64: y on channels 0..31, x on 32..63, each rotated within its 32
BASE = 100.0  # VGGT's RoPE frequency base


def _perm(d: int) -> np.ndarray:
    """A head's q / k channel order: `[0:q, 2q:3q, q:2q, 3q:4q]` (`q = d / 4`)."""
    q = d // 4
    return np.concatenate([np.arange(0, q), np.arange(2 * q, 3 * q), np.arange(q, 2 * q), np.arange(3 * q, 4 * q)])


def _block(sd: Mapping[str, Any], b: str, dt: mx.Dtype, heads: int) -> dict[str, mx.array]:
    """The block `b` of `sd` as `mlx_vit.block`'s `blocks.0.*` (linear weights `[in, out]`, LayerScale folded)."""
    out: dict[str, mx.array] = {}
    c = int(sd[f"{b}.norm1.weight"].shape[0])
    hidden = int(sd[f"{b}.mlp.fc1.weight"].shape[0])
    d = c // heads
    head = _perm(d)
    rows = np.concatenate([s * c + h * d + head for s in (0, 1) for h in range(heads)] + [np.arange(2 * c, 3 * c)])

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{b}.{k}"])

    for name in ("norm1", "norm2"):
        for kind in ("weight", "bias"):
            out[f"blocks.0.{name}.{kind}"] = mf.put((c,), dt, lambda k=f"{name}.{kind}": take(k))
    out["blocks.0.qkv.weight"] = mf.put((c, 3 * c), dt, lambda: take("attn.qkv.weight")[rows].T)
    out["blocks.0.qkv.bias"] = mf.put((3 * c,), dt, lambda: take("attn.qkv.bias")[rows])
    for kind in ("weight", "bias"):  # float32: `mlx_vit.attention` normalises in float32
        out[f"blocks.0.qk_norm.{kind}"] = mf.put(
            (2, d),
            mx.float32,
            lambda k=kind: np.stack([take(f"attn.q_norm.{k}")[head], take(f"attn.k_norm.{k}")[head]]),
        )
    for dst, src, shape, gamma in (
        ("proj", "attn.proj", (c, c), "ls1.gamma"),
        ("fc1", "mlp.fc1", (c, hidden), None),
        ("fc2", "mlp.fc2", (hidden, c), "ls2.gamma"),
    ):

        def scale(g: str | None = gamma) -> Any:  # noqa: ANN401 - an array or 1
            return take(g) if g is not None and f"{b}.{g}" in sd else 1.0

        out[f"blocks.0.{dst}.weight"] = mf.put(shape, dt, lambda s=src, f=scale: take(f"{s}.weight").T * f())
        out[f"blocks.0.{dst}.bias"] = mf.put((shape[1],), dt, lambda s=src, f=scale: take(f"{s}.bias") * f())
    return out


def rope(h: int, w: int, prefix: int, frames: int) -> tuple[mx.array, mx.array]:
    """Float32 `(cos, sin)` `[frames (prefix + h w), d]` for the permuted heads.

    Patch `(y, x)` of the `h` x `w` grid at position `(y + 1, x + 1)`, the `prefix` tokens before each frame's patches
    at 0 (identity rows), as VGGT's `PositionGetter` + 1 and its zero positions of the special tokens.
    """
    d, q = 4 * QUARTER, QUARTER
    inv = 1.0 / BASE ** (np.arange(0, 2 * q, 2, dtype=np.float64) / (2 * q))  # [q]
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64) + 1, np.arange(w, dtype=np.float64) + 1, indexing="ij")
    ay, ax = ys.reshape(-1, 1) * inv, xs.reshape(-1, 1) * inv  # [h w, q]
    ang = np.concatenate([ay, ax, ay, ax], axis=1)
    ang = np.concatenate([np.zeros((prefix, d)), ang])
    ang = np.tile(ang, (frames, 1))
    return mx.array(np.cos(ang).astype(np.float32)), mx.array(np.sin(ang).astype(np.float32))


class Aggregator:
    """The blocks `frame` / `global` (key prefixes in `sd`) alternating, `heads` heads, LayerNorm `eps`."""

    def __init__(self, sd: Mapping[str, Any], dt: mx.Dtype, *, heads: int, eps: float = 1e-5, depth: int) -> None:
        """Convert the first `depth` frame and global blocks."""
        self.frame = [_block(sd, f"frame_blocks.{i}", dt, heads) for i in range(depth)]
        self.glob = [_block(sd, f"global_blocks.{i}", dt, heads) for i in range(depth)]

        def run(x: mx.array, w: dict[str, mx.array], cos: mx.array, sin: mx.array) -> mx.array:
            return block(w, 0, x, heads, eps, (cos, sin))

        self.fn = mx.compile(run)

    def __call__(
        self, t: mx.array, frame_rope: tuple[mx.array, mx.array], global_rope: tuple[mx.array, mx.array],
        layers: Sequence[int],
    ) -> list[mx.array]:  # fmt: skip
        """Tokens `[S, P, C]` -> `cat[frame block output, global block output]` `[S, P, 2 C]` at each of `layers`.

        `layers`: 0-based block indices; `frame_rope` `[P, d]`, `global_rope` `[S P, d]` (`rope`).
        """
        s, p, c = t.shape
        out, prev = [], None
        for i, (wf, wg) in enumerate(zip(self.frame, self.glob, strict=True)):
            if i > max(layers):
                break
            fr = self.fn(t, wf, *frame_rope)
            mx.async_eval(fr)
            if prev is not None:
                mx.eval(prev)
            t = self.fn(fr.reshape(1, s * p, c), wg, *global_rope).reshape(s, p, c)
            mx.async_eval(t)
            mx.eval(fr)
            prev = t
            if i in layers:
                out.append(mx.concatenate([fr, t], axis=-1))
        return out
