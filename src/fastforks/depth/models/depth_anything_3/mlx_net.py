"""Depth Anything 3 on Apple MLX (`device="mlx"`): encoder, DPT / DualDPT head and camera decoder, hand-written.

The maths of `model.Portable._branch` for a resized, normalised `[B, 3, h, w]` image (both sides multiples of 14), as
`mx` operations in channels-last layout: nothing is traced, so a first call is the Metal kernels' build and no
`torch.export` (which was 1.8 s of a da3mono-large first call and ~30 s of a da3-giant one). The pieces:

- the encoder: the shared DINOv2 blocks (`fastforks.shared.mlx_vit`). The any-view backbones (Large, Giant) swap the
  class token for the camera token at block `alt` and from there on normalise q and k, rotate them (RoPE) and
  alternate local and global attention: for one view the two differ in the positions only (a global block gives
  every patch the position (1, 1)). Upstream's RoPE rotates the vertical and the horizontal half of a head's
  channels separately; the conversion reorders the q and k channels of those blocks (q . k does not change) so that
  it is `mlx_vit.rotate`'s one rotation of two halves.
- the head: the 1x1 projections as matmuls on the tokens, the transposed convolutions (stride = kernel) as a matmul
  and a depth-to-space, the fusion blocks as Depth Anything V2's (its 1x1 convolution before the resize).
- DualDPT's position embedding of the full-resolution map goes through the convolution after it on its own, once per
  input size (the convolution is linear): no 128-channel map is added per call.
- DPT's depth and sky convolutions (the same input) are one convolution of twice the channels.

Prepared once, at conversion (exact in real arithmetic, done in float64, stored float32). float16: weights and
activations fp16 (LayerNorm statistics and the attention's softmax in fp32 inside MLX's kernels); the 1x1 output
convolutions, the activations and the camera decoder run in fp32, as in the torch fp16 runner.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import mx, to_mx, tuning
from fastforks.shared.depth_anything_v2.mlx_net import _conv, _fuse, resize
from fastforks.shared.mlx_vit import block, dinov2_layers, dinov2_params, dinov2_table, patches

from .net.utils.head_utils import create_uv_grid, make_sincos_pos_embed

if TYPE_CHECKING:
    from collections.abc import Callable

    from .model import _Config

__all__ = ["LAYOUT", "Branch", "outputs", "prepare"]

LAYOUT = "v1"  # version of `prepare`'s parameter layout (for the conversion cache's file name)
PATCH = 14
EPS = 1e-6  # LayerNorm of the blocks (upstream `Block`'s `ln_eps`)
NORM_EPS = 1e-5  # the other LayerNorms: the ViT's final one, q / k and the head's (`nn.LayerNorm`'s default)
HEADS = {"vitl": 16, "vitg": 24}
HEAD_DIM = 64
SIZES = 4  # input sizes whose tables are kept (least recently used dropped)
SHUFFLE = (4, 2)  # stride (= kernel) of the transposed convolutions of the two finest levels
_POS_RATIO = 0.1  # fixed: upstream DualDPT._add_pos_embed ratio
_FP32 = ("head.out3.", "cam.")  # parameters that stay float32 in a float16 run
_HOST = ("pos.cls", "pos.grid", "patch.bias")  # float32 torch tensors for the position tables


def fp32(k: str) -> bool:
    """`prepare`'s arrays (`<branch>.<name>`) the conversion cache stores float32 whatever the run dtype."""
    name = k.partition(".")[2]
    return name in _HOST or name.startswith(_FP32)


_UNUSED = ("backbone.pretrained.mask_token", "cam_dec.fc_t.", "cam_dec.fc_qvec.")  # no part of the one-view forward
_CAST_CACHE = 64 << 20  # bytes of cast float32 weights after which the buffer cache is emptied


def _rope_order() -> np.ndarray:
    """Upstream's head channels `[v1, v2, h1, h2]` (quarters) in the order `[v1, h1, v2, h2]`."""
    q = HEAD_DIM // 4
    return np.concatenate([np.arange(q), 2 * q + np.arange(q), q + np.arange(q), 3 * q + np.arange(q)])


def _put(out: dict[str, mx.array], k: str, v: np.ndarray) -> None:
    out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))


def _encoder(vit: dict[str, torch.Tensor], cfg: _Config) -> dict[str, mx.array]:
    """The ViT's checkpoint tensors (emptied) -> `mlx_vit`'s layout, the any-view blocks' q and k in `_rope_order`."""
    out = dinov2_params(vit, PATCH)
    if cfg.alt < 0:
        return out
    _put(out, "camera", vit.pop("camera_token")[0, 0].double().numpy())
    order = _rope_order()
    heads = HEADS[cfg.vit]
    dim = heads * HEAD_DIM
    cols = (np.arange(heads)[:, None] * HEAD_DIM + order[None]).reshape(-1)
    cols = np.concatenate([cols, dim + cols, 2 * dim + np.arange(dim)])  # q and k reordered, v as it is
    for i in range(cfg.alt, 1 + max(cfg.layers)):
        b = f"blocks.{i}."
        _put(out, b + "qkv.weight", np.array(out[b + "qkv.weight"])[:, cols])
        _put(out, b + "qkv.bias", np.array(out[b + "qkv.bias"])[cols])
        for n in ("weight", "bias"):  # q's and k's LayerNorm, stacked
            qk = [vit.pop(f"{b}attn.{m}_norm.{n}").double().numpy()[order] for m in ("q", "k")]
            _put(out, f"{b}qk_norm.{n}", np.stack(qk))
    return out


def _decoder(sd: dict[str, torch.Tensor], cfg: _Config) -> dict[str, mx.array]:  # noqa: C901 - a line per layer
    """The head's checkpoint tensors (emptied) -> the float32 arrays of `_head`."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return sd.pop("head." + k).double().numpy()

    def conv(dst: str, src: str, *, bias: bool = True) -> None:
        _put(out, f"head.{dst}.weight", take(src + ".weight").transpose(0, 2, 3, 1))
        if bias:
            _put(out, f"head.{dst}.bias", take(src + ".bias"))

    def linear(dst: str, src: str) -> None:  # a 1x1 convolution as a matmul
        _put(out, f"head.{dst}.weight", take(src + ".weight")[:, :, 0, 0].T)
        _put(out, f"head.{dst}.bias", take(src + ".bias"))

    if cfg.dual:
        _put(out, "head.norm.weight", take("norm.weight"))
        _put(out, "head.norm.bias", take("norm.bias"))
    for i in range(4):
        linear(f"proj{i}", f"projects.{i}")
        if i < len(SHUFFLE):  # transposed convolution [C, O, k, k]: a matmul to (a, b, o)
            w = take(f"resize_layers.{i}.weight")
            _put(out, f"head.up{i}.weight", w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1))
            _put(out, f"head.up{i}.bias", np.tile(take(f"resize_layers.{i}.bias"), SHUFFLE[i] ** 2))
        conv(f"rn{i}", f"scratch.layer{i + 1}_rn", bias=False)
    conv("down", "resize_layers.3")
    for i in range(4):
        src = f"scratch.refinenet{i + 1}."
        for u in (1, 2) if i < 3 else (2,):  # noqa: PLR2004 - the coarsest block has no unit 1
            for c in (1, 2):
                conv(f"fuse{i}.unit{u}.conv{c}", f"{src}resConfUnit{u}.conv{c}")
        linear(f"fuse{i}.out", src + "out_conv")
    conv("out1", "scratch.output_conv1")
    tails = ["scratch.output_conv2"] + ([] if cfg.dual else ["scratch.sky_output_conv2"])
    _put(out, "head.out2.weight", np.concatenate([take(t + ".0.weight").transpose(0, 2, 3, 1) for t in tails]))
    _put(out, "head.out2.bias", np.concatenate([take(t + ".0.bias") for t in tails]))
    w1 = [take(t + ".2.weight")[:, :, 0, 0].T for t in tails]  # [32, outputs] each
    if not cfg.dual:  # depth from the first convolution's channels, sky from the second's
        z = np.zeros_like(w1[0])
        w1 = [np.concatenate([np.concatenate([w1[0], z], 1), np.concatenate([z, w1[1]], 1)])]
    _put(out, "head.out3.weight", w1[0])
    _put(out, "head.out3.bias", np.concatenate([take(t + ".2.bias") for t in tails]))
    return out


def _camera(sd: dict[str, torch.Tensor]) -> dict[str, mx.array]:
    """The camera decoder's checkpoint tensors (emptied): the two layers of its trunk and the field-of-view layer."""
    out: dict[str, mx.array] = {}
    for dst, src in (("fc0", "backbone.0"), ("fc1", "backbone.2"), ("fov", "fc_fov.0")):
        _put(out, f"cam.{dst}.weight", sd.pop(f"cam_dec.{src}.weight").double().numpy().T)
        _put(out, f"cam.{dst}.bias", sd.pop(f"cam_dec.{src}.bias").double().numpy())
    return out


def _branch(sd: dict[str, torch.Tensor], cfg: _Config) -> dict[str, mx.array]:
    """One branch's checkpoint tensors (`_Branch` names, emptied) -> the float32 arrays `forward` uses."""
    prefix = "backbone.pretrained."
    vit = {k.removeprefix(prefix): sd.pop(k) for k in list(sd) if k.startswith(prefix)}
    out = _encoder(vit, cfg)
    sd.update({prefix + k: v for k, v in vit.items()})  # what the encoder did not take: for the caller's check
    out.update(_decoder(sd, cfg))
    if cfg.dual:
        out.update(_camera(sd))
    return out


def prepare(state: dict[str, torch.Tensor], cfgs: dict[str, _Config]) -> dict[str, mx.array]:
    """Checkpoint tensors by `_Net` name (emptied) -> the float32 arrays of every branch, as `<branch>.<name>`."""
    out: dict[str, mx.array] = {}
    for name, cfg in cfgs.items():
        sd = {k.removeprefix(name + "."): state.pop(k) for k in list(state) if k.startswith(name + ".")}
        out.update({f"{name}.{k}": v for k, v in _branch(sd, cfg).items()})
        state.update({f"{name}.{k}": v for k, v in sd.items() if not k.startswith(_UNUSED)})
    if state:
        msg = f"Depth Anything 3 checkpoint: unexpected tensors {sorted(state)[:8]}"
        raise ValueError(msg)
    return out


def _tokens(  # noqa: PLR0913, PLR0917 - the backbone's pieces
    p: dict[str, mx.array], cfg: _Config, x: mx.array, table: mx.array, local: tuple, glob: tuple
) -> tuple[list[mx.array], mx.array]:
    """The any-view backbone for one view: the head's tokens `[B, N, 2 C]` of `cfg.layers` and the camera token.

    Upstream `_get_intermediate_layers_not_chunked` + `get_intermediate_layers`: tokens = (output of the last local
    block, normed output of the layer's block), without the camera token; camera token = the two, un-normed, of the
    last layer. `local` / `glob`: the `(cos, sin)` RoPE tables of the two kinds of block.
    """
    heads = HEADS[cfg.vit]
    t = patches(x.astype(table.dtype), PATCH) @ p["patch.weight"] + table[1:]
    t = mx.concatenate([mx.broadcast_to(table[:1], (t.shape[0], 1, t.shape[2])), t], axis=1)
    last = t
    outs, cam = [], t
    for i in range(1 + max(cfg.layers)):
        if i == cfg.alt:
            t = mx.concatenate([mx.broadcast_to(p["camera"], (t.shape[0], 1, t.shape[2])), t[:, 1:]], axis=1)
        everywhere = i >= cfg.alt and i % 2 == 1  # a global block
        t = block(p, i, t, heads, EPS, None if i < cfg.alt else (glob if everywhere else local))
        if not everywhere:
            last = t
        if i in cfg.layers:
            normed = mx.fast.layer_norm(t[:, 1:], p["norm.weight"], p["norm.bias"], NORM_EPS)
            outs.append(mx.concatenate([last[:, 1:], normed], axis=-1))
            cam = mx.concatenate([last[:, 0], t[:, 0]], axis=-1)
    return outs, cam


def _head(  # noqa: PLR0913, PLR0917 - the head's pieces
    p: dict[str, mx.array], feats: list[mx.array], h0: int, w0: int, pos: list[mx.array], pos_out: mx.array | None
) -> mx.array:
    """Tokens `[B, h0 * w0, K]` of the four layers -> float32 logits `[B, 14 h0, 14 w0, outputs]`.

    `pos`: DualDPT's position embedding of each level `[1, h0, w0, C]` (empty: DPT); `pos_out`: that of the
    full-resolution map after the convolution `head.out2`.
    """
    b = feats[0].shape[0]
    maps = []
    for i, t in enumerate(feats):
        if "head.norm.weight" in p:
            t = mx.fast.layer_norm(t, p["head.norm.weight"], p["head.norm.bias"], NORM_EPS)  # noqa: PLW2901 - the level's tokens
        y = mx.addmm(p[f"head.proj{i}.bias"], t, p[f"head.proj{i}.weight"]).reshape(b, h0, w0, -1)
        if pos:
            y = y + pos[i]
        if i < len(SHUFFLE):
            k = SHUFFLE[i]
            y = mx.addmm(p[f"head.up{i}.bias"], y, p[f"head.up{i}.weight"])
            y = y.reshape(b, h0, w0, k, k, -1).transpose(0, 1, 3, 2, 4, 5).reshape(b, h0 * k, w0 * k, -1)
        elif i == len(feats) - 1:
            y = _conv(p, "head.down", y, stride=2)
        maps.append(_conv(p, f"head.rn{i}", y))
    x = _fuse(p, "head.fuse3", maps[3], None, (maps[2].shape[1], maps[2].shape[2]), resize)
    x = _fuse(p, "head.fuse2", x, maps[2], (maps[1].shape[1], maps[1].shape[2]), resize)
    x = _fuse(p, "head.fuse1", x, maps[1], (maps[0].shape[1], maps[0].shape[2]), resize)
    x = _fuse(p, "head.fuse0", x, maps[0], (2 * maps[0].shape[1], 2 * maps[0].shape[2]), resize)
    x = _conv(p, "head.out2", resize(_conv(p, "head.out1", x), h0 * PATCH, w0 * PATCH))
    if pos_out is not None:
        x = x + pos_out
    return mx.maximum(x, 0).astype(mx.float32) @ p["head.out3.weight"] + p["head.out3.bias"]


def forward(  # noqa: PLR0913, PLR0917 - the network's pieces
    p: dict[str, mx.array],
    cfg: _Config,
    x: mx.array,
    table: mx.array,
    rope: tuple[mx.array, ...],
    pos: list[mx.array],
    pos_out: mx.array | None,
) -> tuple[mx.array, ...]:
    """Normalised `[B, 3, h, w]` -> float32 `(depth, sky)` or `(depth, confidence, fov)`, maps `[B, h, w]`."""
    h0, w0 = x.shape[2] // PATCH, x.shape[3] // PATCH
    if not cfg.dual:
        layers = tuple(i + 1 for i in cfg.layers)
        feats = [t[:, 1:] for t in dinov2_layers(p, HEADS[cfg.vit], x, table, layers, NORM_EPS)]
        logits = _head(p, feats, h0, w0, pos, pos_out)
        return mx.exp(logits[..., 0]), mx.maximum(logits[..., 1], 0)
    feats, cam = _tokens(p, cfg, x, table, rope[:2], rope[2:])
    logits = _head(p, feats, h0, w0, pos, pos_out)
    c = cam.astype(mx.float32)
    for k in ("cam.fc0", "cam.fc1", "cam.fov"):
        c = mx.maximum(mx.addmm(p[k + ".bias"], c, p[k + ".weight"]), 0)
    return mx.exp(logits[..., 0]), mx.exp(logits[..., 1]) + 1, c


def _rope(h0: int, w0: int, dtype: mx.Dtype) -> tuple[mx.array, ...]:
    """`(cos, sin)` `[1 + h0 * w0, 64]` of the local blocks, then of the global ones, in `_rope_order`.

    Upstream `RotaryPositionEmbedding2D` (base frequency 100): patch (y, x) has the position (y + 1, x + 1) in a
    local block and (1, 1) in a global one, the camera token (0, 0).
    """
    q = HEAD_DIM // 4
    inv = 1.0 / (100.0 ** (torch.arange(0, 2 * q, 2).float() / (2 * q)))
    yy, xx = torch.meshgrid(torch.arange(h0), torch.arange(w0), indexing="ij")
    zero = torch.zeros(1)
    local = (torch.cat([zero, yy.flatten() + 1.0]), torch.cat([zero, xx.flatten() + 1.0]))
    everywhere = (torch.cat([zero, torch.ones(h0 * w0)]),) * 2
    out = []
    for py, px in (local, everywhere):
        ang = torch.cat([py[:, None] * inv, px[:, None] * inv] * 2, -1)  # [N, 64]: v, h, v, h
        out += [to_mx(ang.cos()).astype(dtype), to_mx(ang.sin()).astype(dtype)]
    return tuple(out)


def _position(channels: int, h: int, w: int, aspect: float) -> mx.array:
    """Upstream `_add_pos_embed`'s table of an `h x w` map as float32 `[1, h, w, channels]`."""
    uv = create_uv_grid(w, h, aspect_ratio=aspect, dtype=torch.float32)
    horizontal = make_sincos_pos_embed(channels // 2, uv[0, :, 0]) * _POS_RATIO  # [w, C / 2]
    vertical = make_sincos_pos_embed(channels // 2, uv[:, 0, 1]) * _POS_RATIO  # [h, C / 2]
    half = (1, h, w, channels // 2)
    return mx.concatenate(
        [mx.broadcast_to(to_mx(horizontal)[None, None], half), mx.broadcast_to(to_mx(vertical)[None, :, None], half)],
        axis=-1,
    )


class Branch:
    """One branch's MLX network: converted weights, the tables of the last input sizes and the compiled forward."""

    def __init__(self, p32: dict[str, mx.array], cfg: _Config, dtype: str) -> None:
        """Cast the float32 parameters of one branch (`p32`, emptied) to the run dtype."""
        self.cfg = cfg
        self.dtype = getattr(mx, dtype)
        host = {k: torch.from_numpy(np.array(p32.pop(k))) for k in _HOST}
        self.pos, self.cls, self.bias = host["pos.grid"], host["pos.cls"], host["patch.bias"]  # fp32, as upstream
        self.p: dict[str, mx.array] = {}
        freed = 0
        for k in list(p32):  # one at a time: never the float32 and the run dtype's weights of the whole net at once
            v = p32.pop(k)
            self.p[k] = v if k.startswith(_FP32) else v.astype(self.dtype)
            mx.eval(self.p[k])
            freed += v.nbytes if self.p[k].dtype != v.dtype else 0
            del v
            if freed > _CAST_CACHE:  # the float32 originals would stay in MLX's buffer cache (Giant fp16: +0.7 GB)
                mx.clear_cache()
                freed = 0
        mx.clear_cache()
        self.out2 = self.p["head.out2.weight"].astype(mx.float32)

        def run(x: mx.array, table: mx.array, *rest: mx.array) -> tuple[mx.array, ...]:
            rope, pos = (rest[:4], rest[4:]) if cfg.dual else ((), ())
            return forward(self.p, cfg, x, table, rope, list(pos[:4]), pos[4] if pos else None)

        self.run: Callable[..., tuple[mx.array, ...]] = mx.compile(run) if tuning().compile_steps else run
        self.sizes: OrderedDict[tuple[int, int], tuple[mx.array, ...]] = OrderedDict()

    def tables(self, h0: int, w0: int) -> tuple[mx.array, ...]:
        """Of an `h0 x w0` patch grid: the embedding table, then (any-view) the RoPE and position-embedding tables."""
        key = (h0, w0)
        if key not in self.sizes:
            out = [to_mx(dinov2_table(self.cls, self.pos, self.bias, h0, w0)).astype(self.dtype)]
            if self.cfg.dual:
                out += _rope(h0, w0, self.dtype)
                aspect = w0 / h0
                out += [
                    _position(self.p[f"head.proj{i}.bias"].shape[0], h0, w0, aspect).astype(self.dtype)
                    for i in range(4)
                ]
                full = _position(self.out2.shape[-1], h0 * PATCH, w0 * PATCH, aspect)
                out.append(mx.conv2d(full, self.out2, padding=1).astype(self.dtype))
            mx.eval(out)
            self.sizes[key] = tuple(out)
            while len(self.sizes) > SIZES:
                self.sizes.popitem(last=False)
        self.sizes.move_to_end(key)
        return self.sizes[key]

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """Normalised `[B, 3, h, w]` -> `forward`'s lazy outputs."""
        return self.run(x, *self.tables(x.shape[2] // PATCH, x.shape[3] // PATCH))


def outputs(branch: Branch, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Normalised `[B, 3, h, w]` CPU tensor -> the branch's float32 CPU maps, by `model._finish`'s names."""
    out = branch(to_mx(image))
    mx.eval(*out)
    names = ("depth", "conf", "fov") if branch.cfg.dual else ("depth", "sky")
    return {n: torch.from_numpy(np.array(a)) for n, a in zip(names, out, strict=True)}
