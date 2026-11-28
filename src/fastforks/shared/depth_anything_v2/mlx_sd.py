"""Depth Anything V2's encoder and DPT head on Apple MLX, built from a state dict for the cached hand ports.

For the families that hold a Depth Anything V2 (or a ViT with its DPT head) inside their own checkpoint (FlowSeek,
WAFT): every weight is read through `mlx_flow.put`, so `mlx_port.recorded` caches them in the run dtype and a cache
hit replays them from the state-dict skeleton. The layouts are `mlx_vit`'s (`blocks.<i>.*`, linear weights
`[in, out]`) and `mlx_net.prepare`'s (`head.*`), so `mlx_vit.dinov2_layers` / `mlx_vit.Stepped` and this module's
`fusion` run them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf

from . import mlx_net as da2

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["bilinear", "blocks", "fusion", "head", "vit"]


def bilinear(x: mx.array, h: int, w: int, *, align: bool) -> mx.array:
    """`F.interpolate(mode="bilinear")` of `[N, H, W, C]` to `(h, w)` (float32 taps, one rounding)."""
    _, ih, iw, _ = x.shape

    def ratio(n: int, m: int) -> float:
        return ((n - 1) / (m - 1) if m > 1 else 0.0) if align else n / m

    return ops.bilinear_nhwc(x, (h, w), (ratio(ih, h), ratio(iw, w)), align=align)


def _linear(
    sd: Mapping[str, Any], name: str, shape: tuple[int, int], gamma: str | None, dt: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """The `Linear` `name` as an `[in, out]` weight and bias, the LayerScale `gamma` (if in `sd`) folded in."""
    g = gamma if gamma is not None and gamma in sd else None

    def scale() -> Any:  # noqa: ANN401 - an array or 1
        return mf.np64(sd[g]) if g is not None else 1.0

    w = mf.put(shape, dt, lambda: mf.np64(sd[f"{name}.weight"]).T * scale())
    return w, mf.put((shape[1],), dt, lambda: mf.np64(sd[f"{name}.bias"]) * scale())


def blocks(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, mx.array]:
    """The pre-norm ViT blocks `p.<i>` of `sd` (DINOv2's or timm's names) in `mlx_vit`'s layout, LayerScale folded."""
    out: dict[str, mx.array] = {}
    i = 0
    while f"{p}.{i}.norm1.weight" in sd:
        b, o = f"{p}.{i}.", f"blocks.{i}."
        c = int(sd[f"{b}norm1.weight"].shape[0])
        hidden = int(sd[f"{b}mlp.fc1.weight"].shape[0])
        for name in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                out[f"{o}{name}.{kind}"] = mf.put((c,), dt, lambda k=f"{b}{name}.{kind}": mf.np64(sd[k]))
        for dst, src, shape, gamma in (
            ("qkv", "attn.qkv", (c, 3 * c), None),
            ("proj", "attn.proj", (c, c), f"{b}ls1.gamma"),
            ("fc1", "mlp.fc1", (c, hidden), None),
            ("fc2", "mlp.fc2", (hidden, c), f"{b}ls2.gamma"),
        ):
            out[f"{o}{dst}.weight"], out[f"{o}{dst}.bias"] = _linear(sd, f"{b}{src}", shape, gamma, dt)
        i += 1
    return out


def vit(sd: Mapping[str, Any], p: str, dt: mx.Dtype, *, table: bool = True) -> dict[str, mx.array]:
    """The DINOv2 encoder `p` of `sd` in `mlx_vit.dinov2_params`' layout (for `mlx_vit.dinov2_layers`).

    `table`: the position table of the native grid in `dt` (`table`, cls token and patch bias in it); otherwise its
    float32 pieces for `mlx_vit.dinov2_table` at any grid (`pos.cls` `[1, C]`: class token + its position,
    `pos.grid` `[N, C]`, `pos.bias` `[C]`).
    """
    out: dict[str, mx.array] = {}
    c = int(sd[f"{p}.cls_token"].shape[-1])
    n = int(sd[f"{p}.pos_embed"].shape[1])
    patch = int(sd[f"{p}.patch_embed.proj.weight"].shape[-1])

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{p}.{k}"])

    out["patch.weight"] = mf.put(
        (patch * patch * 3, c), dt, lambda: take("patch_embed.proj.weight").transpose(2, 3, 1, 0).reshape(-1, c)
    )
    if table:

        def full() -> np.ndarray:
            pos = take("pos_embed")[0]
            return np.concatenate([pos[:1] + take("cls_token")[0], pos[1:] + take("patch_embed.proj.bias")])

        out["table"] = mf.put((n, c), dt, full)
    else:
        f32 = mx.float32
        out["pos.cls"] = mf.put((1, c), f32, lambda: take("pos_embed")[0, :1] + take("cls_token")[0])
        out["pos.grid"] = mf.put((n - 1, c), f32, lambda: take("pos_embed")[0, 1:])
        out["pos.bias"] = mf.put((c,), f32, lambda: take("patch_embed.proj.bias"))
    out.update(blocks(sd, f"{p}.blocks", dt))
    for kind in ("weight", "bias"):
        out[f"norm.{kind}"] = mf.put((c,), dt, lambda k=f"{p}.norm.{kind}": mf.np64(sd[k]))
    return out


def _projection(sd: Mapping[str, Any], p: str, i: int, dt: mx.Dtype) -> dict[str, mx.array]:
    """The head's projection `i` (with the transposed conv after it for the two finest levels) as one matmul."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{p}.{k}"])

    def shape(k: str) -> tuple[int, ...]:
        return tuple(int(n) for n in sd[f"{p}.{k}"].shape)

    c, k_in = shape(f"projects.{i}.weight")[:2]
    if i < len(da2.SHUFFLE):
        k = da2.SHUFFLE[i]
        o = shape(f"resize_layers.{i}.weight")[1]

        def wt(i: int = i) -> np.ndarray:
            w, wd = take(f"projects.{i}.weight")[:, :, 0, 0], take(f"resize_layers.{i}.weight")
            return np.einsum("ci,coab->iabo", w, wd).reshape(w.shape[1], -1)

        def bs(i: int = i) -> np.ndarray:
            b, wd = take(f"projects.{i}.bias"), take(f"resize_layers.{i}.weight")
            return (np.einsum("c,coab->abo", b, wd) + take(f"resize_layers.{i}.bias")).reshape(-1)

        out[f"head.proj{i}.weight"] = mf.put((k_in, k * k * o), dt, wt)
        out[f"head.proj{i}.bias"] = mf.put((k * k * o,), dt, bs)
    else:
        out[f"head.proj{i}.weight"] = mf.put((k_in, c), dt, lambda i=i: take(f"projects.{i}.weight")[:, :, 0, 0].T)
        out[f"head.proj{i}.bias"] = mf.put((c,), dt, lambda i=i: take(f"projects.{i}.bias"))
    return out


def head(sd: Mapping[str, Any], p: str, dt: mx.Dtype, *, depth: bool = True) -> dict[str, mx.array]:
    """The DPT head `p` of `sd` in `mlx_net.prepare`'s layout (`head.*`).

    `depth`: also the depth branch after `output_conv1` (`head.out2`, and `head.out3` in float32); without it the
    head ends at `head.out1` (WAFT's `return_intermediate` features).
    """
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{p}.{k}"])

    def shape(k: str) -> tuple[int, ...]:
        return tuple(int(n) for n in sd[f"{p}.{k}"].shape)

    def conv(dst: str, src: str, *, bias: bool = True) -> None:
        o, i, kh, kw = shape(f"{src}.weight")
        out[f"head.{dst}.weight"] = mf.put((o, kh, kw, i), dt, lambda: take(f"{src}.weight").transpose(0, 2, 3, 1))
        if bias:
            out[f"head.{dst}.bias"] = mf.put((o,), dt, lambda: take(f"{src}.bias"))

    for i in range(4):
        out.update(_projection(sd, p, i, dt))
        conv(f"rn{i}", f"scratch.layer{i + 1}_rn", bias=False)
    conv("down", "resize_layers.3")
    for i in range(4):
        src = f"scratch.refinenet{i + 1}"
        for u in (2,) if i == len(da2.SHUFFLE) + 1 else (1, 2):  # the coarsest block's unit 1: no input, unused
            for c in (1, 2):
                conv(f"fuse{i}.unit{u}.conv{c}", f"{src}.resConfUnit{u}.conv{c}")
        o, k_in = shape(f"{src}.out_conv.weight")[:2]
        out[f"head.fuse{i}.out.weight"] = mf.put(
            (k_in, o), dt, lambda s=src: take(f"{s}.out_conv.weight")[:, :, 0, 0].T
        )
        out[f"head.fuse{i}.out.bias"] = mf.put((o,), dt, lambda s=src: take(f"{s}.out_conv.bias"))
    conv("out1", "scratch.output_conv1")
    if depth:
        conv("out2", "scratch.output_conv2.0")
        k_in = shape("scratch.output_conv2.2.weight")[1]
        out["head.out3.weight"] = mf.put(
            (k_in, 1), mx.float32, lambda: take("scratch.output_conv2.2.weight")[:, :, 0, 0].T
        )
        out["head.out3.bias"] = mf.put((1,), mx.float32, lambda: take("scratch.output_conv2.2.bias"))
    return out


def _up(y: mx.array, h: int, w: int) -> mx.array:
    return bilinear(y, h, w, align=True)


def fusion(p: dict[str, mx.array], feats: list[mx.array], h0: int, w0: int) -> mx.array:
    """The head's fusion path (`path_1`, `[B, 8 h0, 8 w0, F]`) of the patch tokens `[B, h0 w0, K]` at its 4 layers."""
    b = feats[0].shape[0]
    maps = []
    for i, t in enumerate(feats):
        y = mx.addmm(p[f"head.proj{i}.bias"], t, p[f"head.proj{i}.weight"])
        if i < len(da2.SHUFFLE):
            k = da2.SHUFFLE[i]
            y = y.reshape(b, h0, w0, k, k, -1).transpose(0, 1, 3, 2, 4, 5).reshape(b, h0 * k, w0 * k, -1)
        else:
            y = y.reshape(b, h0, w0, -1)
        if i == len(feats) - 1:
            y = da2._conv(p, "head.down", y, stride=2)  # noqa: SLF001 - the DPT head's pieces
        maps.append(da2._conv(p, f"head.rn{i}", y))  # noqa: SLF001
    size = [(m.shape[1], m.shape[2]) for m in maps]
    x = da2._fuse(p, "head.fuse3", maps[3], None, size[2], _up)  # noqa: SLF001
    x = da2._fuse(p, "head.fuse2", x, maps[2], size[1], _up)  # noqa: SLF001
    x = da2._fuse(p, "head.fuse1", x, maps[1], size[0], _up)  # noqa: SLF001
    return da2._fuse(p, "head.fuse0", x, maps[0], (2 * size[0][0], 2 * size[0][1]), _up)  # noqa: SLF001
