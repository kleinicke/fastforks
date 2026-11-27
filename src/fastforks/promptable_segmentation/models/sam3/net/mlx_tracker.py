"""SAM 3's video tracker written in MLX, channels last: `tracker.TrackerModel` plus the state machine of `video.py`.

The same maths as `video.Tracker` on `tracker.TrackerModel`, with no PyTorch in it:

- weights: `tracker_model.*` of the checkpoint, laid out for the code here (`convert`) and kept in the conversion
  cache. The prompt encoder, mask decoder and object-pointer heads are float32 (as upstream), memory attention and
  the memory encoder run in the run dtype with a float32 residual stream.
- layout: linear weights `[in, out]` (`linear`), 1 x 1 convolutions as linears, the 2 x 2 transposed convolutions as
  one linear to the four output pixels (`_shuffle`), the other convolutions OHWI, the fuser's layer scale folded into
  its second linear.
- one frame is one lazy graph per stage: the mask of the best score is chosen before the hypernetwork product (one
  288 x 288 map per object, not four), and nothing is read back on the host between the heads, the memory encoder and
  the next frame's memory attention. The outputs of a frame are copied out while the next frame runs.
- memory attention: the cross-attention's values are a linear map of the 64-wide memory and the weights of a query
  sum to 1, so the weighted sum is taken over the memory itself, in blocks of queries, and projected after.
- the prompts of frame 0 are encoded on the host in numpy (a few tokens).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.mlx import DTYPES, attention, convert_cached, mx
from fastforks.kernels.graph_mlx.ops import bilinear
from fastforks.shared.mlx_weights import Weights

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from fastforks._core.runner import Context

F32 = mx.float32
PREFIX = "tracker_model."  # the tracker's keys in model.safetensors
LAYOUT = "sam3-mlx-tracker-v1"  # the cached layout of `convert` (this file's hash joins it)
DIM, MEM_DIM = 256, 64  # fixed: token width, memory width
SIZE, TOKENS = 1008, 72  # fixed: input side, feature grid side
LOW = 288  # fixed: low-resolution mask side
MEM_SIZE = 4 * LOW  # fixed: memory-encoder mask side (1152)
NUM_MASKMEM = 7  # fixed: frame 0 + the 6 previous frames in memory attention
MAX_PTRS = 16  # fixed: object pointers in memory attention
MEM_SCALE, MEM_BIAS = 20.0, -10.0  # fixed: sigmoid_scale / bias_for_mem_enc (also the mask-as-output logits)
NO_OBJ = -1024.0  # fixed: upstream NO_OBJ_SCORE
STABILITY_DELTA, STABILITY_THRESH = 0.05, 0.98  # fixed: dynamic_multimask_stability_delta / _thresh
SCORE_BYTES = 96 << 20  # one block of memory-attention weights (queries x memory): evaluated block by block
MASK_PIXELS = 8 << 20  # output pixels resized per group of objects

_RUN = ("memory_attention.", "memory_encoder.", "memory_temporal_positional_encoding", "occlusion_spatial_embedding_")
_TABLES = re.compile(r"(iou_token|mask_tokens|obj_score_token|no_mask_embed|point_embed|not_a_point_embed)\.weight$")
_UNUSED = ("no_memory_positional_encoding",)  # as upstream


# --- weights --------------------------------------------------------------------------------------------------------


def run_dtype(name: str) -> bool:
    """`name` runs in the run dtype (else float32): the memory modules and what is added to their maps."""
    return name.startswith(_RUN)


def _layout(key: str, a: mx.array) -> mx.array:
    """A checkpoint tensor in the layout the code here reads (see the module docstring)."""
    if key.endswith(".bias"):
        return mx.tile(a, 4) if ".upscale_conv" in key else a  # one bias per output pixel of the 2 x 2 block
    if not key.endswith(".weight"):
        return a
    if a.ndim == 4:  # noqa: PLR2004 - convolution weights
        if ".upscale_conv" in key:  # ConvTranspose2d [C_in, C_out, 2, 2] -> linear [C_in, (a, b, C_out)]
            return mx.contiguous(a.transpose(0, 2, 3, 1).reshape(a.shape[0], -1))
        if a.shape[-1] != 1:
            return mx.contiguous(a.transpose(0, 2, 3, 1))  # OIHW -> OHWI (`mx.conv2d`)
        a = a.reshape(a.shape[0], -1)  # 1 x 1 conv -> linear [O, I]
    if a.ndim == 2 and not _TABLES.search(key):  # noqa: PLR2004 - a linear layer: [out, in] -> [in, out] (`linear`)
        return mx.contiguous(a.T)
    return a


def convert(raw: Mapping[str, mx.array], dt: mx.Dtype) -> dict[str, mx.array]:
    """The tracker's tensors (names without `tracker_model.`, checkpoint layout, float32) as the code here runs them."""
    out = {k: _layout(k, a.astype(F32)) for k, a in raw.items() if not k.startswith(_UNUSED)}
    for k in [k for k in out if k.endswith(".scale")]:  # the fuser's layer scale -> its second linear
        stem, scale = k.removesuffix("scale"), out.pop(k)
        out[f"{stem}pointwise_conv2.weight"] = out[f"{stem}pointwise_conv2.weight"] * scale
        out[f"{stem}pointwise_conv2.bias"] = out[f"{stem}pointwise_conv2.bias"] * scale
    out = {k: a.astype(dt if run_dtype(k) else F32) for k, a in out.items()}
    mx.eval(out)
    return out


def cached(ctx: Context) -> dict[str, mx.array]:
    """`convert` of the variant's checkpoint in the run dtype `ctx.dtype`, from the conversion cache."""

    def build() -> tuple[dict[str, mx.array], dict[str, object]]:
        src = Weights(ctx.weights()[0])
        raw = src.take(src.prefixed(PREFIX))
        return convert({k.removeprefix(PREFIX): a for k, a in raw.items()}, DTYPES[ctx.dtype]), {}

    found = convert_cached(
        ctx, LAYOUT, build, dtype=ctx.dtype, fp32=lambda name: not run_dtype(name), sources=(Path(__file__),)
    )
    return found.arrays


# --- ops ------------------------------------------------------------------------------------------------------------


def linear(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """`F.linear` with the weight stored `[in, out]` (the bias added in the GEMM's epilogue)."""
    return mx.addmm(b, x.reshape(-1, x.shape[-1]), w).reshape(*x.shape[:-1], w.shape[1])


@mx.compile
def gelu(x: mx.array) -> mx.array:
    """Exact GELU in float32, rounded to the dtype of `x`."""
    xf = x.astype(F32)
    return (0.5 * xf * (1.0 + mx.erf(xf * (1.0 / math.sqrt(2.0))))).astype(x.dtype)


def _shuffle(y: mx.array) -> mx.array:
    """`[B, h, w, (2, 2, C)]` -> `[B, 2h, 2w, C]` (the 2 x 2 transposed convolution's output pixels)."""
    b, h, w, c4 = y.shape
    c = c4 // 4
    return y.reshape(b, h, w, 2, 2, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, c)


@mx.compile
def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Axial RoPE on `[..., N, D]`: the pairs of `x` turned by the angles `[N, D / 2]`, in float32 (as upstream)."""
    pairs = x.astype(F32).reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return mx.stack([a * cos - b * sin, b * cos + a * sin], axis=-1).reshape(x.shape).astype(x.dtype)


def resize(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bilinear resize of `[P, h, w]` maps (half-pixel centres: `F.interpolate(align_corners=False)`)."""
    h, w = x.shape[1:]
    if (h, w) == tuple(size):
        return x
    return bilinear(x[:, None], (size[0], size[1]), (h / size[0], w / size[1]), align=False)[:, 0]


def aa_matrix(inp: int, out: int) -> np.ndarray:
    """`[out, inp]` weights of PyTorch's antialiased bilinear resize along one axis (`mlx_graph._aa_matrix`)."""
    s = inp / out
    support, inv = max(s, 1.0), (1.0 / s if s >= 1.0 else 1.0)
    m = np.zeros((out, inp), dtype=np.float64)
    for i in range(out):
        c = s * (i + 0.5)
        lo, hi = max(int(c - support + 0.5), 0), min(int(c + support + 0.5), inp)
        w = np.maximum(0.0, 1.0 - np.abs((np.arange(lo, hi) - c + 0.5) * inv))
        m[i, lo:hi] = w / w.sum()
    return m.astype(np.float32)


def resize_aa(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Antialiased bilinear resize of float32 `[P, h, w]` maps (`F.interpolate(antialias=True)`)."""
    h, w = x.shape[1:]
    if (h, w) == tuple(size):
        return x
    return mx.matmul(mx.matmul(mx.array(aa_matrix(h, size[0])), x), mx.array(aa_matrix(w, size[1])).T)


def sine_1d(pos: np.ndarray, dim: int) -> np.ndarray:
    """Transformers `get_1d_sine_pe` (temperature 10000), float32."""
    half = dim // 2
    dim_t = np.float32(10000) ** (2 * (np.arange(half, dtype=np.float32) // 2) / np.float32(half))
    p = pos.astype(np.float32)[..., None] / dim_t
    return np.concatenate([np.sin(p), np.cos(p)], axis=-1).astype(np.float32)


def memory_position_encoding() -> np.ndarray:
    """The memory encoder's normalised sine encoding (32 features per axis), `[72 * 72, 64]` float32."""
    n = MEM_DIM // 2
    c = np.arange(1, TOKENS + 1, dtype=np.float32)
    c = c / (c[-1] + np.float32(1e-6)) * np.float32(2 * math.pi)
    dim_t = np.float32(10000) ** (2 * (np.arange(n) // 2).astype(np.float32) / np.float32(n))
    p = c[:, None] / dim_t  # [72, 32]
    s = np.stack((np.sin(p[:, 0::2]), np.cos(p[:, 1::2])), axis=-1).reshape(TOKENS, n)
    y = np.broadcast_to(s[:, None], (TOKENS, TOKENS, n))
    x = np.broadcast_to(s[None, :], (TOKENS, TOKENS, n))
    return np.concatenate((y, x), axis=-1).reshape(TOKENS * TOKENS, MEM_DIM).astype(np.float32)


def rope_tables() -> tuple[np.ndarray, np.ndarray]:
    """`Sam3TrackerVideoVisionRotaryEmbedding` on the 72 x 72 grid: cos, sin `[5184, 128]` (one angle per pair)."""
    inv = np.float32(1.0) / (np.float32(10000.0) ** (np.arange(0, DIM // 2, 2, dtype=np.float32) / np.float32(DIM // 2)))
    h, w = np.meshgrid(np.arange(TOKENS, dtype=np.float32), np.arange(TOKENS, dtype=np.float32), indexing="ij")
    freqs = np.concatenate([w.reshape(-1, 1) * inv, h.reshape(-1, 1) * inv], axis=-1)  # transformers' order: (w, h)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


# --- tracker --------------------------------------------------------------------------------------------------------


class Tracker:
    """The tracker's weights, its modules and the tracking state machine (frame 0 prompts, forward propagation).

    Args:
        w: The arrays of `convert` (`cached`).
        dt: The run dtype of the memory modules.
        pos: The feature grid's position encoding `[1, 72 * 72, 256]` float32 (`sam3.grid_position_encoding`).
    """

    def __init__(self, w: dict[str, mx.array], dt: mx.Dtype, pos: mx.array) -> None:
        """Build the constant tables (position encodings, RoPE angles, the prompt encoder's host copies)."""
        self.w, self.dt = w, dt
        self.pos = pos.astype(F32) * 0.1  # memory attention adds a tenth of it
        cos, sin = rope_tables()
        self.cos, self.sin = mx.array(cos), mx.array(sin)
        tpos = w["memory_temporal_positional_encoding"].reshape(NUM_MASKMEM, 1, MEM_DIM)
        self.slot_pos = mx.array(memory_position_encoding()).astype(dt)[None] + tpos  # [age - 1] -> [5184, 64]
        self.occlusion = w["occlusion_spatial_embedding_parameter"].reshape(1, 1, 1, MEM_DIM)
        self.no_mem = w["no_memory_embedding"].reshape(1, 1, DIM)
        self.no_ptr = w["no_object_pointer"]  # [1, 256]
        md = "mask_decoder."
        self.tokens = mx.concatenate([w[f"{md}{n}.weight"] for n in ("obj_score_token", "iou_token", "mask_tokens")])
        self.no_mask = w["prompt_encoder.no_mask_embed.weight"].reshape(1, 1, DIM)
        # the prompt encoder's tables on the host
        self.point_pe = np.array(w["prompt_encoder.shared_embedding.positional_embedding"])
        self.point_embed = np.array(w["prompt_encoder.point_embed.weight"])
        self.not_a_point = np.array(w["prompt_encoder.not_a_point_embed.weight"])
        c = (np.arange(TOKENS, dtype=np.float32) + 0.5) / TOKENS
        grid = np.stack([np.broadcast_to(c[None, :], (TOKENS, TOKENS)), np.broadcast_to(c[:, None], (TOKENS, TOKENS))], -1)
        pe = np.array(w["shared_image_embedding.positional_embedding"])
        self.image_pe = mx.array(self._fourier(grid, pe).reshape(1, TOKENS * TOKENS, DIM))
        mx.eval(self.pos, self.slot_pos, self.tokens, self.image_pe)

    # --- small pieces -----------------------------------------------------------------------------------------------

    def lin(self, x: mx.array, name: str) -> mx.array:
        """Linear `name` (in the dtype of its weight)."""
        return linear(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

    def ln(self, x: mx.array, name: str, eps: float = 1e-5) -> mx.array:
        """LayerNorm `name` over the last axis (float32 statistics; the output in the dtype of `x`)."""
        w, b = self.w[f"{name}.weight"], self.w[f"{name}.bias"]
        return mx.fast.layer_norm(x, w.astype(x.dtype), b.astype(x.dtype), eps)

    def conv(self, x: mx.array, name: str, **kw: int) -> mx.array:
        """Convolution `name` on an NHWC map."""
        return mx.conv2d(x, self.w[f"{name}.weight"], **kw) + self.w[f"{name}.bias"]

    def ff(self, x: mx.array, name: str, *, sigmoid: bool = False) -> mx.array:
        """`tracker.FeedForward` with three layers (ReLU between them)."""
        x = mx.maximum(self.lin(x, f"{name}.proj_in"), 0)
        x = mx.maximum(self.lin(x, f"{name}.layers.0"), 0)
        x = self.lin(x, f"{name}.proj_out")
        return mx.sigmoid(x) if sigmoid else x

    def attn(self, name: str, q: mx.array, k: mx.array, v: mx.array, heads: int = 8) -> mx.array:
        """`tracker.Attention` on `[B, N, C]` tokens."""

        def split(x: mx.array, part: str) -> mx.array:
            x = self.lin(x, f"{name}.{part}_proj")
            return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(0, 2, 1, 3)

        qh, kh, vh = split(q, "q"), split(k, "k"), split(v, "v")
        y = attention(qh, kh, vh, scale=qh.shape[-1] ** -0.5)
        return self.lin(y.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], -1), f"{name}.o_proj")

    @staticmethod
    def _fourier(coords: np.ndarray, pe: np.ndarray) -> np.ndarray:
        """`tracker.PositionalEmbedding`: `[..., 2]` in `[0, 1]` -> `[..., 256]` float32."""
        x = (2 * coords.astype(np.float32) - 1) @ pe
        x = np.float32(2 * math.pi) * x
        return np.concatenate([np.sin(x), np.cos(x)], axis=-1).astype(np.float32)

    # --- prompts ----------------------------------------------------------------------------------------------------

    def sparse(self, coords: np.ndarray | None, labels: np.ndarray | None, p: int) -> mx.array:
        """`tracker.PromptEncoder` for points on the host: `[P, N, 2]` 1008-px coordinates, labels -> `[P, N + 1, 256]`.

        Without prompts (`None`): upstream's one not-a-point click (label -1) plus the padding token.
        """
        if coords is None or labels is None:
            return mx.broadcast_to(mx.array(self.not_a_point)[None], (p, 2, DIM))
        points = np.pad(coords.astype(np.float32) + np.float32(0.5), ((0, 0), (0, 1), (0, 0)))
        labs = np.pad(labels.astype(np.int64), ((0, 0), (0, 1)), constant_values=-1)[..., None]
        emb = self._fourier(points / np.float32(SIZE), self.point_pe)
        emb = np.where(labs == -1, self.not_a_point, emb)
        emb = np.where(labs != -10, emb, np.float32(0))  # noqa: PLR2004 - upstream's padding label
        emb = emb + self.point_embed[np.clip(labs[..., 0], 0, None)] * (labs >= 0)
        return mx.array(emb.astype(np.float32))

    def mask_embed(self, m: mx.array) -> mx.array:
        """`tracker.MaskEmbedding`: mask logits `[P, 288, 288]` -> dense prompt `[P, 72 * 72, 256]`."""
        p, q = m.shape[0], "prompt_encoder.mask_embed."
        x = gelu(self.ln(self.conv(m[..., None], f"{q}conv1", stride=2), f"{q}layer_norm1", 1e-6))
        x = gelu(self.ln(self.conv(x, f"{q}conv2", stride=2), f"{q}layer_norm2", 1e-6))
        return self.lin(x, f"{q}conv3").reshape(p, TOKENS * TOKENS, DIM)

    # --- heads ------------------------------------------------------------------------------------------------------

    def feat_s0(self, level: mx.array) -> mx.array:
        """`mask_decoder.conv_s0` on the neck's 288 x 288 level (float32)."""
        return self.lin(level.astype(F32), "mask_decoder.conv_s0")

    def feat_s1(self, level: mx.array) -> mx.array:
        """`mask_decoder.conv_s1` on the neck's 144 x 144 level (float32)."""
        return self.lin(level.astype(F32), "mask_decoder.conv_s1")

    def _two_way(self, tokens: mx.array, keys: mx.array) -> tuple[mx.array, mx.array]:
        """`tracker.TwoWayTransformer`: prompt tokens `[P, T, 256]`, image tokens `[P, 5184, 256]`."""
        t, pe = "mask_decoder.transformer.", self.image_pe
        q, k = tokens, keys
        for i in range(2):
            p = f"{t}layers.{i}."
            if i == 0:
                q = self.attn(f"{p}self_attn", q, q, q)
            else:
                qq = q + tokens
                q = q + self.attn(f"{p}self_attn", qq, qq, q)
            q = self.ln(q, f"{p}layer_norm1")
            q = self.ln(q + self.attn(f"{p}cross_attn_token_to_image", q + tokens, k + pe, k), f"{p}layer_norm2")
            mlp = self.lin(mx.maximum(self.lin(q, f"{p}mlp.proj_in"), 0), f"{p}mlp.proj_out")
            q = self.ln(q + mlp, f"{p}layer_norm3")
            k = self.ln(k + self.attn(f"{p}cross_attn_image_to_token", k + pe, q + tokens, q), f"{p}layer_norm4")
        final = self.attn(f"{t}final_attn_token_to_image", q + tokens, k + pe, k)
        return self.ln(q + final, f"{t}layer_norm_final_attn"), k

    def decode(  # noqa: PLR0913 - upstream arguments
        self, image: mx.array, sparse: mx.array, dense: mx.array, s0: mx.array, s1: mx.array, *, multimask: bool
    ) -> tuple[mx.array, mx.array, mx.array]:
        """`tracker.MaskDecoder` with the mask chosen inside: logits `[P, 288, 288]`, its token, object score `[P, 1]`.

        `multimask`: the mask of the best predicted IoU among the three; else mask 0, or that best one where mask 0
        is unstable. Only the chosen masks go through the hypernetwork product.
        """
        p, md = sparse.shape[0], "mask_decoder."
        tokens = mx.concatenate([mx.broadcast_to(self.tokens[None], (p, 6, DIM)), sparse], axis=1)
        q, k = self._two_way(tokens, mx.broadcast_to(image + dense, (p, TOKENS * TOKENS, DIM)))
        mask_out = q[:, 2:6]
        up = _shuffle(self.lin(k.reshape(p, TOKENS, TOKENS, DIM), f"{md}upscale_conv1"))
        up = gelu(self.ln(up + s1, f"{md}upscale_layer_norm", 1e-6))
        up = gelu(_shuffle(self.lin(up, f"{md}upscale_conv2")) + s0).reshape(p, LOW * LOW, DIM // 8)
        hyper = mx.stack([self.ff(mask_out[:, i], f"{md}output_hypernetworks_mlps.{i}") for i in range(4)], axis=1)
        iou = self.ff(q[:, 1], f"{md}iou_prediction_head", sigmoid=True)
        obj = self.ff(q[:, 0], f"{md}pred_obj_score_head")
        best = mx.argmax(iou[:, 1:], axis=-1)[:, None, None] + 1  # [P, 1, 1]
        best_row = mx.take_along_axis(hyper, best, axis=1)  # [P, 1, 32]
        if multimask:
            token = mx.take_along_axis(mask_out, best, axis=1)[:, 0]
            return (up @ best_row.transpose(0, 2, 1)).reshape(p, LOW, LOW), token, obj
        both = up @ mx.concatenate([hyper[:, :1], best_row], axis=1).transpose(0, 2, 1)  # [P, 288 * 288, 2]
        single = both[..., 0]
        area_i = (single > STABILITY_DELTA).sum(axis=1).astype(F32)
        area_u = (single > -STABILITY_DELTA).sum(axis=1).astype(F32)
        stable = mx.where(area_u > 0, area_i / mx.maximum(area_u, 1), 1.0) >= STABILITY_THRESH
        return mx.where(stable[:, None], single, both[..., 1]).reshape(p, LOW, LOW), mask_out[:, 0], obj

    def heads(  # noqa: PLR0913 - upstream `_single_frame_forward` arguments
        self, image: mx.array, s0: mx.array, s1: mx.array, sparse: mx.array, dense: mx.array | None, *, multimask: bool
    ) -> tuple[mx.array, mx.array, mx.array]:
        """`video.Tracker.heads`: low-resolution logits `[P, 288, 288]`, object pointer `[P, 256]`, score `[P, 1]`."""
        low, token, obj = self.decode(image, sparse, self.no_mask if dense is None else dense, s0, s1, multimask=multimask)
        present = obj > 0
        low = mx.where(present[:, :, None], low, NO_OBJ)
        ptr = mx.where(present, self.ff(token, "object_pointer_proj"), self.no_ptr)
        return low, ptr, obj

    def mask_as_output(
        self, image: mx.array, s0: mx.array, s1: mx.array, mask: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Upstream `_use_mask_as_output` for a binary float32 `[P, 1008, 1008]` mask: low, high, pointer, score."""
        p = mask.shape[0]
        high = mask * MEM_SCALE + MEM_BIAS
        low = resize_aa(high, (LOW, LOW))
        down = self.conv(mask[..., None], "mask_downsample", stride=4)[..., 0]  # [P, 252, 252]
        m_in = resize(down, (LOW, LOW))  # an upscale: the antialiasing filter is the plain one
        _, ptr, _ = self.heads(image, s0, s1, self.sparse(None, None, p), self.mask_embed(m_in), multimask=True)
        lam = (mask > 0).any(axis=(1, 2))[:, None]
        obj = mx.where(lam, MEM_SCALE + MEM_BIAS, MEM_BIAS).astype(F32)
        return low, high, mx.where(lam, ptr, self.no_ptr), obj

    # --- memory -----------------------------------------------------------------------------------------------------

    def memory_encoder(self, image: mx.array, mask: mx.array) -> mx.array:
        """`tracker.MemoryEncoder`: features `[1 or P, 72, 72, 256]`, scaled mask `[P, 1152, 1152]` -> `[P, 72, 72, 64]`."""
        e, dt = "memory_encoder.", self.dt
        m = mask[..., None].astype(dt)
        for i in range(4):
            q = f"{e}mask_downsampler.layers.{i}."
            m = gelu(self.ln(self.conv(m, f"{q}conv", stride=2, padding=1), f"{q}layer_norm", 1e-6))
        x = self.lin(image.astype(dt), f"{e}feature_projection") + self.lin(m, f"{e}mask_downsampler.final_conv")
        for i in range(2):
            q = f"{e}memory_fuser.layers.{i}."
            h = self.ln(self.conv(x, f"{q}depthwise_conv", padding=3, groups=DIM), f"{q}layer_norm", 1e-6)
            x = x + self.lin(gelu(self.lin(h, f"{q}pointwise_conv1")), f"{q}pointwise_conv2")  # the scale is folded in
        return self.lin(x, f"{e}projection")

    def encode(self, image: mx.array, high: mx.array, obj: mx.array, *, binarize: bool) -> mx.array:
        """Upstream `_encode_new_memory`: bfloat16 memory tokens `[P, 72 * 72, 64]` from logits `[P, 1008, 1008]`."""
        high = resize(high, (MEM_SIZE, MEM_SIZE))  # an upscale: the antialiasing filter is the plain one
        m = (high > 0).astype(F32) if binarize else mx.sigmoid(high)
        feats = self.memory_encoder(image, m * MEM_SCALE + MEM_BIAS)
        absent = (obj <= 0).astype(feats.dtype)[:, :, None, None]  # [P, 1, 1, 1]
        return (feats + absent * self.occlusion).reshape(feats.shape[0], -1, MEM_DIM).astype(mx.bfloat16)

    def memory_attention(self, curr: mx.array, memory: mx.array, memory_pos: mx.array, plain: int) -> mx.array:
        """`tracker.MemoryAttention`: float32 `curr [1, 5184, 256]` conditioned on `memory [P, M, 64]` -> `[P, ...]`.

        `memory_pos [M, 64]`; the last `plain` memory tokens (object pointers) are not rotated. The residual stream is
        float32, attention and the MLP run in the run dtype.
        """
        dt, cos, sin = self.dt, self.cos, self.sin
        scale = DIM**-0.5
        x = curr + self.pos
        keys = memory + memory_pos
        b, n = memory.shape[:2]
        turned = n - plain  # whole frames of tokens
        rows = max(64, SCORE_BYTES // (b * n * dt.size))  # queries per block
        for i in range(4):
            p = f"memory_attention.layers.{i}."
            h = self.ln(x, f"{p}layer_norm1").astype(dt)
            q = _rotate(self.lin(h, f"{p}self_attn.q_proj"), cos, sin)
            k = _rotate(self.lin(h, f"{p}self_attn.k_proj"), cos, sin)
            y = attention(q[:, None], k[:, None], self.lin(h, f"{p}self_attn.v_proj")[:, None], scale=scale)[:, 0]
            x = x + self.lin(y, f"{p}self_attn.o_proj").astype(F32)
            h = self.ln(x, f"{p}layer_norm2").astype(dt)
            c = f"{p}cross_attn_image."
            k = self.lin(keys, f"{c}k_proj")
            rot = _rotate(k[:, :turned].reshape(b, -1, TOKENS * TOKENS, DIM), cos, sin).reshape(b, turned, DIM)
            kt = (mx.concatenate([rot, k[:, turned:]], axis=1) if plain else rot).transpose(0, 2, 1)
            q = _rotate(self.lin(h, f"{c}q_proj"), cos, sin) * scale
            parts: list[mx.array] = []
            for j in range(0, q.shape[1], rows):
                parts.append(mx.softmax(q[:, j : j + rows] @ kt, axis=-1) @ memory)
                mx.eval(parts[-1])  # one block of weights alive at a time: a lazy chain keeps several
            y = self.lin(self.lin(mx.concatenate(parts, axis=1), f"{c}v_proj"), f"{c}o_proj")
            x = x + y.astype(F32)
            h = self.ln(x, f"{p}layer_norm3").astype(dt)
            x = x + self.lin(mx.maximum(self.lin(h, f"{p}linear1"), 0), f"{p}linear2").astype(F32)
            mx.eval(x)  # a layer's attention weights (tokens x memory) do not outlive it
        return self.ln(x, "memory_attention.layer_norm")

    def condition(
        self, curr: mx.array, t: int, n_frames: int, memories: dict[int, mx.array], pointers: dict[int, mx.array]
    ) -> mx.array:
        """Upstream `_prepare_memory_conditioned_features` for frame `t >= 1`: `[P, 5184, 256]` float32."""
        dt = self.dt
        slots = [(0, NUM_MASKMEM), *((k, t - k) for k in range(max(1, t - NUM_MASKMEM + 1), t))]
        mems = [memories[k].astype(dt) for k, _ in slots]
        poss = [self.slot_pos[age - 1] for _, age in slots]
        max_ptrs = min(n_frames, MAX_PTRS)
        ptr_frames = [0, *(t - d for d in range(1, max_ptrs) if t - d >= 1)]  # frame 0, then the most recent first
        n, split = len(ptr_frames), DIM // MEM_DIM
        ptrs = mx.stack([pointers[k] for k in ptr_frames], axis=1)  # [P, n, 256]
        mems.append(ptrs.reshape(ptrs.shape[0], n * split, MEM_DIM).astype(dt))
        rel = np.array([t - k for k in ptr_frames], dtype=np.float32) / np.float32(max_ptrs - 1)
        pos = self.lin(mx.array(sine_1d(rel, DIM)), "temporal_positional_encoding_projection_layer")
        poss.append(mx.repeat(pos, split, axis=0).astype(dt))
        return self.memory_attention(curr, mx.concatenate(mems, axis=1), mx.concatenate(poss), n * split)

    # --- tracking ---------------------------------------------------------------------------------------------------

    def prompt_mask(self, masks: mx.array) -> mx.array:
        """Mask prompts `[P, H, W]` (float32, 0 / 1) at the input size: antialiased bilinear, `>= 0.5` (upstream)."""
        if masks.shape[1:] == (SIZE, SIZE):
            return masks
        return (resize_aa(masks, (SIZE, SIZE)) >= 0.5).astype(F32)  # noqa: PLR2004

    @staticmethod
    def binary(low: mx.array, size: tuple[int, int]) -> mx.array:
        """Logits `[P, 288, 288]` at the video size, `> 0` (a few objects at a time: `MASK_PIXELS`)."""
        step = max(1, MASK_PIXELS // (size[0] * size[1]))
        parts = [resize(low[i : i + step], size) > 0 for i in range(0, low.shape[0], step)]
        return parts[0] if len(parts) == 1 else mx.concatenate(parts)

    def track(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Iterable[tuple[mx.array, mx.array, mx.array]],
        n_frames: int,
        size: tuple[int, int],
        p: int,
        *,
        coords: np.ndarray | None,
        labels: np.ndarray | None,
        mask: mx.array | None,
    ) -> dict[str, np.ndarray]:
        """`video.Tracker.run` and the output handling of `model._Sam3.track`.

        `frames` yields per frame the neck's 72 x 72 level `[1, 72, 72, 256]` (float32), `feat_s0` and `feat_s1`.
        Prompts (frame 0, `p` objects): `coords [P, N, 2]` in 1008-px input coordinates + `labels [P, N]`, or `mask`
        `[P, 1008, 1008]` binary float32.

        Returns:
            `masks` bool `[T, P, H, W]`, `pred_masks` float32 `[T, P, 288, 288]`, `object_score_logits` `[T, P]`.
        """
        out = {
            "masks": np.empty((n_frames, p, *size), dtype=np.bool_),
            "pred_masks": np.empty((n_frames, p, LOW, LOW), dtype=np.float32),
            "object_score_logits": np.empty((n_frames, p), dtype=np.float32),
        }
        memories: dict[int, mx.array] = {}
        pointers: dict[int, mx.array] = {}
        pending: tuple[int, mx.array, mx.array, mx.array] | None = None

        def copy_out() -> None:
            if pending is not None:
                t, masks, low, obj = pending
                out["masks"][t], out["pred_masks"][t] = np.array(masks, copy=False), np.array(low, copy=False)
                out["object_score_logits"][t] = np.array(obj, copy=False)[:, 0]

        for t, (raw, s0, s1) in enumerate(frames):
            image = raw.reshape(1, TOKENS * TOKENS, DIM)
            high = None
            if t == 0 and mask is not None:
                low, high, ptr, obj = self.mask_as_output(image, s0, s1, mask)
            elif t == 0:
                multimask = labels is not None and labels.shape[1] <= 1
                low, ptr, obj = self.heads(
                    image + self.no_mem, s0, s1, self.sparse(coords, labels, p), None, multimask=multimask
                )
            else:
                cond = self.condition(image, t, n_frames, memories, pointers)
                low, ptr, obj = self.heads(cond, s0, s1, self.sparse(None, None, p), None, multimask=True)
            keep: list[mx.array] = []
            if t + 1 < n_frames:
                # Keep frame zero and each history window needed by the *next* frame.
                if t >= NUM_MASKMEM:
                    del memories[t - NUM_MASKMEM + 1]
                if t >= MAX_PTRS:
                    del pointers[t - MAX_PTRS + 1]
                if high is None:
                    high = resize(low, (SIZE, SIZE))
                memories[t] = self.encode(raw, high, obj, binarize=t == 0)
                pointers[t] = ptr
                keep = [memories[t], ptr]
            masks = self.binary(low, size)
            mx.async_eval(masks, low, obj, *keep)  # the GPU works on this frame while the host copies the last one
            copy_out()
            pending = (t, masks, low, obj)
        copy_out()
        return out
