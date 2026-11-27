"""SAM 2's video tracking on MLX, with no PyTorch in it: the state machine of `video.Tracker` on the MLX modules.

The same maths as `video.Tracker.run` (and `video.Motion` for SAMURAI), on `mlx_net.MlxEncoder`'s channels-last
maps and the modules of `mlx_heads`:

- one frame is a lazy graph per stage: the heads, the memory encoder and the next frame's memory attention read each
  other's MLX arrays, nothing goes through the host in between. The mask of the best predicted IoU is chosen before
  the hypernetwork product (one map per object, not four). A frame's outputs are copied out while the next one runs.
- SAMURAI: the boxes of the three candidate masks are taken on the GPU (`boxes`), the Kalman filter and the choice
  run on the host in float64 numpy (`Motion`, a few numbers per object), as the float64 reference.
- the frames are resized as upstream (uint8, PIL bicubic) and normalised in numpy (`frame`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import bilinear

from .mlx_heads import MEM_DIM, MlxMemoryAttention, MlxMemoryEncoder, _mlp, fill_holes
from .mlx_net import arr, linear

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .mlx_heads import MlxDecoder, MlxPrompts
    from .mlx_net import MlxEncoder

F32 = mx.float32
DIM = 256  # fixed: token width (`video.DIM`)
NUM_MASKMEM, MAX_PTRS = 7, 16  # fixed: `video.NUM_MASKMEM`, `video.MAX_PTRS`
MEM_SCALE, MEM_BIAS = 20.0, -10.0  # fixed: `video.MEM_SCALE`, `video.MEM_BIAS`
NO_OBJ = -1024.0  # fixed: upstream NO_OBJ_SCORE
STABLE_FRAMES, STABLE_IOU, MEMORY_IOU, SAMURAI_PRESENT = 15, 0.3, 0.5, -1.0  # fixed: SAMURAI, as `video`
MASK_PIXELS = 8 << 20  # output pixels resized per group of maps


# --- ops ------------------------------------------------------------------------------------------------------------


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
    """Upstream `get_1d_sine_pe` (temperature 10000), float32."""
    half = dim // 2
    dim_t = np.float32(10000) ** (2 * (np.arange(half, dtype=np.float32) // 2) / np.float32(half))
    p = pos.astype(np.float32)[..., None] / dim_t
    return np.concatenate([np.sin(p), np.cos(p)], axis=-1).astype(np.float32)


def sine_grid(tokens: int, dim: int) -> np.ndarray:
    """Upstream `PositionEmbeddingSine(dim, normalize=True)` of a `tokens x tokens` grid: `[tokens^2, dim]` float32."""
    n = dim // 2
    c = np.arange(1, tokens + 1, dtype=np.float32)
    c = c / (c[-1] + np.float32(1e-6)) * np.float32(2 * math.pi)
    dim_t = np.float32(10000) ** (2 * (np.arange(n) // 2).astype(np.float32) / np.float32(n))
    p = c[:, None] / dim_t  # [tokens, n]
    s = np.stack((np.sin(p[:, 0::2]), np.cos(p[:, 1::2])), axis=-1).reshape(tokens, n)
    y = np.broadcast_to(s[:, None], (tokens, tokens, n))
    x = np.broadcast_to(s[None, :], (tokens, tokens, n))
    return np.concatenate((y, x), axis=-1).reshape(tokens * tokens, dim).astype(np.float32)


def binary(low: mx.array, size: tuple[int, int]) -> mx.array:
    """Logits `[P, h, w]` at the video size, `> 0` (a few objects at a time: `MASK_PIXELS`)."""
    step = max(1, MASK_PIXELS // (size[0] * size[1]))
    parts = [resize(low[i : i + step], size) > 0 for i in range(0, low.shape[0], step)]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts)


def boxes(low: mx.array, size: int) -> mx.array:
    """`video.Motion.boxes`: `[P, M, h, w]` logits -> int32 `[P, M, 4]` boxes `(x0, y0, x1, y1)` of `> 0` at `size`."""
    p, m = low.shape[:2]
    fg = binary(low.reshape(p * m, *low.shape[2:]), (size, size))
    idx = mx.arange(size, dtype=mx.int32)
    cols, rows = fg.any(axis=1), fg.any(axis=2)
    box = mx.stack(
        [
            mx.where(cols, idx, size).min(axis=-1),
            mx.where(rows, idx, size).min(axis=-1),
            mx.where(cols, idx, -1).max(axis=-1),
            mx.where(rows, idx, -1).max(axis=-1),
        ],
        axis=-1,
    )
    return mx.where(cols.any(axis=-1, keepdims=True), box, 0).reshape(p, m, 4)


def frame(image: np.ndarray, size: int, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the normalised float32 `[1, size, size, 3]` encoder input.

    As upstream `load_video_frames` (`video.resize_frame`: uint8, PIL bicubic) and its normalisation.
    """
    x = np.clip(np.rint(image[0].astype(np.float32) * np.float32(255)), 0, 255).astype(np.uint8)
    img = Image.fromarray(np.ascontiguousarray(x.transpose(1, 2, 0))).resize((size, size), Image.Resampling.BICUBIC)
    y = np.asarray(img, np.float32) / np.float32(255)
    return ((y - np.asarray(mean, np.float32)) / np.asarray(std, np.float32))[None]


def plan(t: int, n_frames: int, memory: list[int] | None) -> tuple[list[tuple[int, int]], list[int], int]:
    """`video.Tracker.plan`: the memory `(frame, age)` slots of frame `t`, the pointer frames, the pointer window."""
    if memory is None:
        memory = list(range(max(1, t - NUM_MASKMEM + 1), t))
        ages = [t - k for k in memory]
    else:
        ages = list(range(len(memory), 0, -1))
    max_ptrs = min(n_frames, MAX_PTRS)
    ptr_frames = [0, *(t - d for d in range(1, max_ptrs) if t - d >= 1)]
    return [(0, NUM_MASKMEM), *zip(memory, ages, strict=True)], ptr_frames, max_ptrs


# --- SAMURAI --------------------------------------------------------------------------------------------------------


class Motion:
    """`video.Motion` on the host: per object the Kalman filter of the mask box, the stable count, the memory frames.

    Float64 numpy (the reference); the flags of a frame are known at once, so nothing is in flight between frames.
    """

    def __init__(self, objects: int, weight: float) -> None:
        """`objects` tracks with motion weight `weight` (`kf_score_weight`)."""
        from fastforks.shared import box_kalman  # noqa: PLC0415 - only SAMURAI needs it (it imports PyTorch)

        self.kalman: Any = box_kalman
        self.weight = weight
        self.mean = np.zeros((objects, 8), dtype=np.float64)
        self.cov = np.zeros((objects, 8, 8), dtype=np.float64)
        self.stable = np.zeros(objects, dtype=np.int64)
        self.best_iou = np.zeros(objects, dtype=np.float32)  # predicted IoU of the chosen mask (this frame)
        self.valid: list[list[int]] = [[] for _ in range(objects)]  # the last memory frames that passed, per object

    @staticmethod
    def xyah(box: np.ndarray) -> np.ndarray:
        """SAMURAI `xyxy_to_xyah` of `[N, 4]` boxes (a zero height counts as 1)."""
        w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
        h = np.where(h == 0, 1.0, h)
        return np.stack(((box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2, w / h, h), 1)

    def select(self, box: np.ndarray, iou: np.ndarray) -> np.ndarray:
        """`video.Motion.select`: the index `[P]` of the multimask output to keep, from their boxes `[P, M, 4]`.

        The filter steps run on the objects they apply to (the numpy solver raises on a new track's zero covariance,
        where `video.Motion` computes every step and keeps by `torch.where`).
        """
        kalman = self.kalman
        bx = box.astype(np.float64)
        ious = iou.astype(np.float32)
        rows = np.arange(len(bx))
        new, scored = self.stable == 0, self.stable >= STABLE_FRAMES
        run = ~new
        mean, cov = self.mean.copy(), self.cov.copy()
        if run.any():
            mean[run], cov[run] = kalman.predict(self.mean[run], self.cov[run])
        pred = kalman.to_xyxy(mean)[:, None]  # [P, 1, 4]
        iw = np.clip(np.minimum(pred[..., 2], bx[..., 2]) - np.maximum(pred[..., 0], bx[..., 0]), 0, None)
        ih = np.clip(np.minimum(pred[..., 3], bx[..., 3]) - np.maximum(pred[..., 1], bx[..., 1]), 0, None)
        inter = iw * ih
        area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        union = area + (bx[..., 2] - bx[..., 0]) * (bx[..., 3] - bx[..., 1]) - inter
        ok_box = (union != 0) & (bx != 0).any(-1)
        kf = np.where(ok_box, inter / np.where(union == 0, 1.0, union), 0.0)
        score = self.weight * kf + (1 - self.weight) * ious.astype(np.float64)  # float64, as upstream
        best = np.where(scored, score.argmax(1), ious.argmax(1))
        chosen = ious[rows, best]
        ok = np.where(scored, chosen >= np.float32(STABLE_IOU), chosen > np.float32(STABLE_IOU))
        meas = self.xyah(bx[rows, best])
        upd = run & ok
        if upd.any():
            mean[upd], cov[upd] = kalman.update(mean[upd], cov[upd], meas[upd])
        if new.any():
            mean[new], cov[new] = kalman.initiate(meas[new])
        self.mean, self.cov = mean, cov
        self.stable = np.where(new, 1, np.where(ok, np.where(scored, self.stable, self.stable + 1), 0))
        self.best_iou = chosen
        return best

    def memory(self, t: int) -> dict[tuple[int, ...], list[int]]:
        """`video.Motion.memory`: the memory frames of frame `t` (oldest first) -> the objects that use them."""
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, valid in enumerate(self.valid):
            frames = [k for k in valid if k < t - 1]
            frames = [*frames, t - 1][-(NUM_MASKMEM - 1) :] if t > 1 else []
            groups.setdefault(tuple(frames), []).append(i)
        return groups

    def kept(self, t: int, obj: np.ndarray) -> set[int]:
        """`video.Motion.kept` for the object scores `[P]` of frame `t`: the frames the memory can still select."""
        keep = {k for valid in self.valid for k in valid}
        if t >= 2:  # noqa: PLR2004 - upstream skips frame 1
            for valid, ok in zip(self.valid, (self.best_iou > MEMORY_IOU) & (obj > 0), strict=True):
                if ok:
                    valid.append(t)
                    del valid[: -(NUM_MASKMEM - 1)]
        return keep


# --- tracker --------------------------------------------------------------------------------------------------------


class MlxTracker:
    """`video.Tracker` on MLX for one loaded SAM 2: the heads, the memory modules and the tracking loop."""

    def __init__(  # noqa: PLR0913, PLR0917 - the shared modules
        self,
        mods: Any,  # noqa: ANN401 - `video.VideoModules` (a torch module)
        encoder: MlxEncoder,
        prompts: MlxPrompts,
        decoder: MlxDecoder,
        dtype_name: str,
        image_size: int,
    ) -> None:
        """Copy the video modules' weights: the memory modules in `dtype_name`, the pointer heads float32."""
        dtype = getattr(mx, dtype_name)
        self.dtype, self.size, self.low = dtype, image_size, image_size // 4
        self.encoder, self.prompts, self.dec = encoder, prompts, decoder
        self.attention = MlxMemoryAttention(mods.memory_attention, dtype)
        self.memory_encoder = MlxMemoryEncoder(mods.memory_encoder, dtype)
        tokens = image_size // 16
        self.tokens = tokens
        self.pos = mx.array(sine_grid(tokens, DIM))  # [N, 256]
        tpos = arr(mods.maskmem_tpos_enc.reshape(-1, 1, MEM_DIM), dtype)  # [7, 1, 64]
        self.slot_pos = mx.array(sine_grid(tokens, MEM_DIM)).astype(dtype)[None] + tpos  # [age - 1] -> [N, 64]
        self.no_obj = arr(mods.no_obj_embed_spatial.reshape(-1), dtype)
        self.no_ptr = arr(mods.no_obj_ptr, F32)  # [1, 256]
        down = mods.mask_downsample
        self.down = (arr(down.weight.permute(0, 2, 3, 1), F32), arr(down.bias, F32), down.stride[0])
        self.ptr = _mlp(mods.obj_ptr_proj, F32)
        self.tpos = linear(mods.obj_ptr_tpos_proj, F32)
        mx.eval(self.pos, self.slot_pos)

    # --- heads ------------------------------------------------------------------------------------------------------

    def heads(  # noqa: PLR0913 - upstream `_forward_sam_heads` arguments
        self,
        image: mx.array,
        s0: mx.array,
        s1: mx.array,
        sparse: mx.array,
        p: int,
        *,
        multimask: bool,
        motion: Motion | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """`video.Tracker.heads` without a mask prompt: logits `[P, h, w]`, object pointer `[P, 256]`, score `[P, 1]`.

        `image` `[1 or P, t, t, 256]`. `multimask`: the output of the best predicted IoU, or with `motion` (SAMURAI)
        the one it selects among the three; else the single output.
        """
        dec = self.dec
        up, hyper, iou, tokens, obj = dec.run(image, self.prompts.no_mask, sparse, s0, s1, p)
        present = obj > (0.0 if motion is None else SAMURAI_PRESENT)  # [P, 1]
        token = tokens[:, 0]
        if not multimask:
            low, _ = dec.single(up, hyper, iou)
        elif motion is None:
            best = mx.argmax(iou[:, 1:], axis=-1)[:, None, None] + 1  # [P, 1, 1]
            low = dec.masks(up, mx.take_along_axis(hyper, best, axis=1))[:, 0]
            if dec.token_per_mask:
                token = mx.take_along_axis(tokens, best, axis=1)[:, 0]
        else:
            three = mx.where(present[:, :, None, None], dec.masks(up, hyper[:, 1:]), NO_OBJ)
            box, scores = boxes(three, self.size), iou[:, 1:]
            mx.eval(box, scores, three, tokens, obj)  # the choice is made on the host
            chosen = motion.select(np.array(box), np.array(scores))
            best = mx.array(chosen.astype(np.int32))[:, None, None]
            low = mx.take_along_axis(three, best[..., None], axis=1)[:, 0]
            if dec.token_per_mask:
                token = mx.take_along_axis(tokens[:, 1:], best, axis=1)[:, 0]
        low = mx.where(present[:, :, None], low, NO_OBJ)
        return low, mx.where(present, self.ptr(token), self.no_ptr), obj

    def mask_as_output(
        self, raw: mx.array, s0: mx.array, s1: mx.array, mask: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Upstream `_use_mask_as_output` for a binary float32 `[P, S, S]` mask: low logits, pointer, object score."""
        p = mask.shape[0]
        low = resize_aa(mask * MEM_SCALE + MEM_BIAS, (self.low, self.low))
        w, b, stride = self.down
        dense = self.prompts.dense((mx.conv2d(mask[..., None], w, stride=stride) + b)[..., 0])
        # the pointer of the single output's token (its mask is not needed)
        _, _, _, tokens, score = self.dec.run(raw, dense, self.prompts.empty(p), s0, s1, p)
        ptr = mx.where(score > 0, self.ptr(tokens[:, 0]), self.no_ptr)
        lam = (mask > 0).any(axis=(1, 2))[:, None]
        obj = mx.where(lam, MEM_SCALE + MEM_BIAS, MEM_BIAS).astype(F32)
        return low, mx.where(lam, ptr, self.no_ptr), obj

    # --- memory -----------------------------------------------------------------------------------------------------

    def encode(self, raw: mx.array, high: mx.array, obj: mx.array, *, binarize: bool) -> mx.array:
        """Upstream `_encode_new_memory`: bfloat16 memory tokens `[P, N, 64]` from logits `[P, S, S]`."""
        m = (high > 0).astype(F32) if binarize else mx.sigmoid(high)
        feats = self.memory_encoder(raw, (m * MEM_SCALE + MEM_BIAS)[..., None])
        absent = (obj <= 0).astype(feats.dtype)[:, :, None, None]  # [P, 1, 1, 1]
        return (feats + absent * self.no_obj).reshape(feats.shape[0], -1, MEM_DIM).astype(mx.bfloat16)

    def condition(  # noqa: PLR0913, PLR0917 - the memory frames and the objects that use them
        self,
        curr: mx.array,
        t: int,
        n_frames: int,
        mems: dict[int, mx.array],
        ptrs: dict[int, mx.array],
        memory: list[int] | None = None,
        rows: list[int] | None = None,
    ) -> mx.array:
        """Upstream `_prepare_memory_conditioned_features` for frame `t >= 1`: float32 `[P, N, 256]`.

        `curr` `[1, N, 256]`; `memory` / `rows` as `video.Tracker.condition` (SAMURAI's frames, the objects' rows).
        """
        dt = self.dtype
        pick = None if rows is None else mx.array(rows)
        slots, ptr_frames, max_ptrs = plan(t, n_frames, memory)
        feats, poss = [], []
        for k, age in slots:
            if k not in mems:
                msg = f"SAM 2 memory for frame {k} was released before it was read"
                raise RuntimeError(msg)
            feats.append((mems[k] if pick is None else mems[k][pick]).astype(dt))
            poss.append(self.slot_pos[age - 1])
        stack = mx.stack([ptrs[k] if pick is None else ptrs[k][pick] for k in ptr_frames], axis=1)  # [P, n, 256]
        p, n, split = stack.shape[0], len(ptr_frames), DIM // MEM_DIM
        feats.append(stack.reshape(p, n * split, MEM_DIM).astype(dt))
        rel = np.array([t - k for k in ptr_frames], dtype=np.float32) / np.float32(max_ptrs - 1)
        poss.append(mx.repeat(self.tpos(mx.array(sine_1d(rel, DIM))), split, axis=0).astype(dt))
        curr = mx.broadcast_to(curr, (p, *curr.shape[1:]))
        return self.attention(curr, self.pos, mx.concatenate(feats, axis=1), mx.concatenate(poss), n * split)

    def _conditioned(  # noqa: PLR0913, PLR0917 - the frame, the bank and SAMURAI's state
        self,
        curr: mx.array,
        t: int,
        n_frames: int,
        mems: dict[int, mx.array],
        ptrs: dict[int, mx.array],
        motion: Motion | None,
    ) -> mx.array:
        """`condition` for every object: with SAMURAI per group of objects that read the same memory frames."""
        if motion is None:
            return self.condition(curr, t, n_frames, mems, ptrs)
        groups = motion.memory(t)
        if len(groups) == 1:
            return self.condition(curr, t, n_frames, mems, ptrs, list(next(iter(groups))))
        parts = [self.condition(curr, t, n_frames, mems, ptrs, list(m), rows) for m, rows in groups.items()]
        order = np.argsort(np.concatenate(list(groups.values())))  # the objects' rows in the concatenation
        return mx.concatenate(parts)[mx.array(order.astype(np.int32))]

    # --- tracking ---------------------------------------------------------------------------------------------------

    def prompt_mask(self, masks: mx.array) -> mx.array:
        """Mask prompts `[P, H, W]` (float32, 0 / 1) at the input size: antialiased bilinear, `>= 0.5` (upstream)."""
        if masks.shape[1:] == (self.size, self.size):
            return masks
        return (resize_aa(masks, (self.size, self.size)) >= 0.5).astype(F32)  # noqa: PLR2004

    def _first(  # noqa: PLR0913, PLR0917 - the frame's maps and one argument per prompt kind
        self,
        pix: mx.array,
        raw: mx.array,
        s0: mx.array,
        s1: mx.array,
        p: int,
        coords: np.ndarray | None,
        labels: np.ndarray | None,
        mask: mx.array | None,
        motion: Motion | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Frame 0: the mask prompt as the output, or the heads on the clicks / boxes (multimask for one click)."""
        if mask is not None:
            return self.mask_as_output(raw, s0, s1, mask)
        if coords is None or labels is None:
            msg = "SAM 2 video: no point / box / mask prompt"
            raise ValueError(msg)
        sparse = self.prompts.sparse(coords, labels)
        return self.heads(pix, s0, s1, sparse, p, multimask=labels.shape[1] <= 1, motion=motion)

    def track(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Iterable[np.ndarray],
        n_frames: int,
        size: tuple[int, int],
        p: int,
        *,
        coords: np.ndarray | None,
        labels: np.ndarray | None,
        mask: mx.array | None,
        kf_weight: float | None = None,
    ) -> dict[str, np.ndarray]:
        """`video.Tracker.run` and the output handling of `model._Sam2.track`.

        `frames` yields the normalised frames `[1, S, S, 3]` (`frame`). Prompts (frame 0, `p` objects): `coords`
        `[P, N, 2]` in encoder-input pixels + `labels` `[P, N]`, or `mask` `[P, S, S]` binary float32 (`prompt_mask`).
        `kf_weight`: SAMURAI with this motion weight (None: SAM 2).

        Returns:
            `masks` bool `[T, P, H, W]`, `pred_masks` float32 `[T, P, S / 4, S / 4]` (holes filled),
            `object_score_logits` `[T, P]`.
        """
        s, tk = self.size, self.tokens
        out = {
            "masks": np.empty((n_frames, p, *size), dtype=np.bool_),
            "pred_masks": np.empty((n_frames, p, self.low, self.low), dtype=np.float32),
            "object_score_logits": np.empty((n_frames, p), dtype=np.float32),
        }
        mems: dict[int, mx.array] = {}
        ptrs: dict[int, mx.array] = {}
        motion = None if kf_weight is None else Motion(p, kf_weight)
        pending: tuple[int, mx.array, mx.array, mx.array] | None = None

        def copy_out() -> None:
            if pending is not None:
                t, masks, low, obj = pending
                out["masks"][t], out["pred_masks"][t] = np.array(masks, copy=False), np.array(low, copy=False)
                out["object_score_logits"][t] = np.array(obj, copy=False)[:, 0]

        for t, x in enumerate(frames):
            pix, s1, s0 = self.encoder.features(mx.array(x))
            raw = pix - self.encoder.no_mem
            last = t + 1 == n_frames
            if t == 0:
                low, ptr, obj = self._first(pix, raw, s0, s1, p, coords, labels, mask, motion)
                low = fill_holes(low)
                seen = low  # frame 0's memory is made of the filled mask
            else:
                cond = self._conditioned(raw.reshape(1, tk * tk, DIM), t, n_frames, mems, ptrs, motion)
                cond = cond.reshape(p, tk, tk, DIM)
                seen, ptr, obj = self.heads(cond, s0, s1, self.prompts.empty(p), p, multimask=True, motion=motion)
                low = fill_holes(seen)
            keep: list[mx.array] = []
            if not last:
                mems[t] = self.encode(raw, resize(seen, (s, s)), obj, binarize=t == 0)
                ptrs[t] = ptr
                keep = [mems[t], ptr]
            if t and not last:
                held = set() if motion is None else motion.kept(t, np.array(obj)[:, 0])
                for k in [k for k in ptrs if 0 < k <= t - MAX_PTRS + 1 and k not in held]:  # no later frame reads it
                    del ptrs[k]
                    mems.pop(k, None)
                if motion is None:  # the pointer window is longer than the encoded-memory window
                    mems.pop(t - NUM_MASKMEM + 1, None) if t - NUM_MASKMEM + 1 > 0 else None
            masks = binary(low, size)
            mx.async_eval(masks, low, obj, *keep)  # the GPU works on this frame while the host copies the last one
            copy_out()
            pending = (t, masks, low, obj)
        copy_out()
        return out
