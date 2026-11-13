"""SAM 2.1 video tracking: upstream `SAM2VideoPredictor` with every prompt on frame 0 (`sam2/video.py`).

Follows `build_sam2_video_predictor` (the settings shared by all four released configs) for one call of
`add_new_points_or_box` / `add_new_mask` per object on frame 0, then `propagate_in_video` forward:

- frames: resized to the square encoder input as upstream `load_video_frames` (PIL bicubic on the uint8 frame; here
  torch's antialiased bicubic, uint8 on the CPU, float + rounding on other devices), then the image encoder
- frame 0 (conditioning frame): the prompt heads on `image_embed` (with `no_mem_embed`, upstream
  `directly_add_no_mem_embed`), multimask for at most one point; a mask prompt is used as the output directly
  (`use_mask_input_as_output_without_sam`, the object pointer from the heads on the raw features)
- later frames: memory attention over frame 0 + the 6 previous frames (`num_maskmem` 7) and the object pointers of
  frame 0 + up to 15 previous frames, then the prompt heads without prompts (multimask, best predicted IoU)
- memory: the memory encoder on the 1024 px mask (frame 0: binarised, as `propagate_in_video_preflight`; later frames:
  sigmoid), `x 20 - 10`, `no_obj_embed_spatial` for absent objects, stored as bfloat16 (as upstream)
- output: small background holes (8-connected, area <= 8) of the low-resolution logits set to 0.1 (`fill_hole_area`
  8), bilinear to the video size, `> 0`

All objects run as one batch (upstream runs them one by one on the same features; the same maths). The prompt heads
run in float32; the memory modules in the compute dtype.

SAMURAI (option `samurai`; Yang et al. 2024, Apache-2.0, `sam2_base.py` with `samurai_mode`): a constant-velocity
Kalman filter per object on the box of its mask (`fastforks.shared.box_kalman`, in encoder-input pixels) picks the
multimask output once the track is stable (15 frames with a predicted IoU above 0.3: score `w * box IoU + (1 - w) *
predicted IoU`, `w` the variant's `kf_score_weight`), and the memory bank keeps the last 6 frames whose predicted IoU
is above 0.5 with the object present (plus always the previous frame) instead of the last 6 frames; the heads keep
a mask down to an object score of -1 (SAM 2: 0). Upstream tracks
one object; here every object has its own filter and memory, objects with the same memory frames share one batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastforks._core.checkpoint import load_strict
from fastforks.shared import box_kalman

from .net.memory_attention import MemoryAttention, MemoryAttentionLayer
from .net.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from .net.position_encoding import PositionEmbeddingSine
from .net.sam2_utils import MLP, get_1d_sine_pe
from .net.transformer import RoPEAttention

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .net.mask_decoder import MaskDecoder
    from .net.prompt_encoder import PromptEncoder

DIM = 256  # fixed: SAM 2's hidden width (all released checkpoints)
MEM_DIM = 64  # fixed: memory feature width
NUM_MASKMEM = 7  # fixed: frame 0 + 6 previous frames in the memory bank
MAX_PTRS = 16  # fixed: object pointers in memory attention (max_obj_ptrs_in_encoder)
MEM_SCALE, MEM_BIAS = 20.0, -10.0  # fixed: sigmoid_scale / bias_for_mem_enc (also the mask-as-output logits)
NO_OBJ = -1024.0  # fixed: upstream NO_OBJ_SCORE
FILL_HOLE_AREA = 8  # fixed: SAM2VideoPredictor fill_hole_area
HOLE_SCORE = 0.1  # fixed: the logit given to filled holes
LOW = 256  # fixed: low-resolution mask side (image_size / 4)
STABLE_FRAMES = 15  # fixed: SAMURAI stable_frames_threshold (frames before the filter scores the masks)
STABLE_IOU = 0.3  # fixed: SAMURAI stable_ious_threshold (predicted IoU that keeps a track stable)
MEMORY_IOU = 0.5  # fixed: SAMURAI memory_bank_iou_threshold (and object score > 0)
SAMURAI_PRESENT = -1.0  # fixed: SAMURAI min_obj_score_logits (object score above which the heads keep the mask)
VIDEO_KEYS = (
    "mask_downsample.",
    "maskmem_tpos_enc",
    "memory_attention.",
    "memory_encoder.",
    "no_obj_embed_spatial",
    "no_obj_ptr",
    "obj_ptr_proj.",
    "obj_ptr_tpos_proj.",
)


class VideoModules(torch.nn.Module):
    """The video modules of upstream `SAM2Base` (same state-dict names; `no_mem_pos_enc` is not used)."""

    def __init__(self, tokens: int) -> None:
        """Upstream video config (untrained); `tokens` = the image-embedding side."""
        super().__init__()

        def rope(**kw: int | bool) -> RoPEAttention:
            return RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[tokens, tokens],
                embedding_dim=DIM,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                **kw,
            )

        layer = MemoryAttentionLayer(
            activation="relu",
            cross_attention=rope(rope_k_repeat=True, kv_in_dim=MEM_DIM),
            d_model=DIM,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            self_attention=rope(),
        )
        self.memory_attention = MemoryAttention(d_model=DIM, pos_enc_at_input=True, layer=layer, num_layers=4)
        self.memory_encoder = MemoryEncoder(
            out_dim=MEM_DIM,
            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
            fuser=Fuser(CXBlock(dim=DIM, kernel_size=7, padding=3, layer_scale_init_value=1e-6), num_layers=2),
            position_encoding=PositionEmbeddingSine(num_pos_feats=MEM_DIM, normalize=True, temperature=10000),
        )
        self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(NUM_MASKMEM, 1, 1, MEM_DIM))
        self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, DIM))
        self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, MEM_DIM))
        self.obj_ptr_proj = MLP(DIM, DIM, DIM, 3)
        self.obj_ptr_tpos_proj = torch.nn.Linear(DIM, MEM_DIM)

    @classmethod
    def load(cls, state: dict[str, torch.Tensor], tokens: int, what: str) -> VideoModules:
        """The video modules from the full SAM 2 checkpoint `state`."""
        mods = cls(tokens)
        load_strict(mods, {k: v for k, v in state.items() if k.startswith(VIDEO_KEYS)}, what)
        return mods.eval()


def fill_holes(low: torch.Tensor, max_area: int = FILL_HOLE_AREA) -> torch.Tensor:
    """Upstream `fill_holes_in_mask_scores`: 8-connected background (`<= 0`) components of area `<= max_area` -> 0.1.

    Exact without a connected-components kernel: `max_area` rounds of 3 x 3 minimum label propagation over the
    background settle every component of up to `max_area` pixels (graph diameter < `max_area`); a label group with a
    background neighbour of another label is part of a larger component (not settled) and is kept.
    """
    p, _, h, w = low.shape
    bg = low <= 0
    big = float(h * w)
    idx = torch.arange(h * w, device=low.device, dtype=torch.float32).view(1, 1, h, w).expand(p, 1, h, w)
    lab = torch.where(bg, idx, big)
    for _ in range(max_area):
        lab = torch.where(bg, -F.max_pool2d(-lab, 3, 1, 1), big)
    lo = -F.max_pool2d(-lab, 3, 1, 1)  # neighbours outside the background are `big`, never below a label
    hi = F.max_pool2d(torch.where(bg, lab, -1.0), 3, 1, 1)
    key = (torch.arange(p, device=low.device).view(p, 1, 1, 1) * (h * w) + lab.clamp(max=h * w - 1).long())[bg]
    n = p * h * w
    count = torch.bincount(key, minlength=n)
    open_ = torch.bincount(key, weights=((lo != lab) | (hi != lab))[bg].float(), minlength=n) > 0
    hole = torch.zeros_like(bg)
    hole[bg] = (count[key] <= max_area) & ~open_[key]
    return torch.where(hole, torch.tensor(HOLE_SCORE, dtype=low.dtype, device=low.device), low)


def resize_frame(frame: torch.Tensor, size: int) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> `[1, 3, size, size]` as upstream `load_video_frames` (PIL bicubic, uint8).

    CPU: PIL itself (exact; torch's uint8 bicubic rounds ~0.3 % of the pixels 1-2 levels apart, which moves the
    logits by up to ~0.2). Other devices: float bicubic, rounded to uint8 levels.
    """
    x = (frame * 255).round().clamp(0, 255)
    if x.device.type == "cpu":
        img = Image.fromarray(x[0].to(torch.uint8).permute(1, 2, 0).numpy()).resize(
            (size, size), Image.Resampling.BICUBIC
        )
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]
    x = F.interpolate(x, (size, size), mode="bicubic", align_corners=False, antialias=True)
    return x.round().clamp(0, 255) / 255


class Tracker:
    """The tracking state machine for one video (frame 0 prompts, forward propagation)."""

    def __init__(  # noqa: PLR0913, PLR0917 - the shared modules
        self,
        mods: VideoModules,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        no_mem: torch.Tensor,
        dtype: torch.dtype,
        image_size: int,
    ) -> None:
        """`mods`: memory attention / encoder in `dtype`, the rest float32 (as the prompt side); `no_mem` `[256]`."""
        self.m, self.pe, self.dec, self.no_mem, self.dtype, self.size = (
            mods,
            prompt_encoder,
            mask_decoder,
            no_mem.view(1, -1, 1, 1),
            dtype,
            image_size,
        )
        tokens = image_size // 16
        dev = no_mem.device
        z = torch.zeros(1, DIM, tokens, tokens, device=dev)
        self.pos = PositionEmbeddingSine(num_pos_feats=DIM, normalize=True, temperature=10000)(z)  # [1, 256, t, t]
        self.pos = self.pos.flatten(2).permute(2, 0, 1).to(dtype)  # [t*t, 1, 256]
        zm = torch.zeros(1, MEM_DIM, tokens, tokens, device=dev)
        self.mem_pos = PositionEmbeddingSine(num_pos_feats=MEM_DIM, normalize=True, temperature=10000)(zm)
        self.mem_pos = self.mem_pos.flatten(2).permute(2, 0, 1).to(dtype)  # [t*t, 1, 64]

    def heads(  # noqa: PLR0913, PLR0917 - upstream `_forward_sam_heads` arguments
        self,
        pix: torch.Tensor,
        hr: list[torch.Tensor],
        coords: torch.Tensor | None,
        labels: torch.Tensor | None,
        mask_in: torch.Tensor | None,
        multimask: bool,  # noqa: FBT001
        motion: Motion | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `_forward_sam_heads` (float32): low `[P, 1, 256, 256]`, high `[P, 1, S, S]`, ptr, object score.

        `motion` (SAMURAI) picks among the multimask outputs instead of the best predicted IoU.
        """
        p = pix.shape[0]
        if coords is None or labels is None:
            coords = torch.zeros(p, 1, 2, device=pix.device)
            labels = -torch.ones(p, 1, dtype=torch.int32, device=pix.device)
        sparse, dense = self.pe(points=(coords, labels), boxes=None, masks=mask_in)
        low, iou, tokens, obj = self.dec(
            image_embeddings=pix,
            image_pe=self.pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            repeat_image=False,
            high_res_features=hr,
        )
        present = obj > (0 if motion is None else SAMURAI_PRESENT)  # [P, 1]
        low = torch.where(present[:, :, None, None], low, NO_OBJ).float()
        token = tokens[:, 0]
        if multimask:
            best = iou.argmax(-1) if motion is None else motion.select(low, iou, self.size)
            rows = torch.arange(p, device=pix.device)
            low = low[rows, best][:, None]
            if tokens.shape[1] > 1:
                token = tokens[rows, best]
        high = F.interpolate(low, (self.size, self.size), mode="bilinear", align_corners=False)
        lam = present.float()
        ptr = lam * self.m.obj_ptr_proj(token) + (1 - lam) * self.m.no_obj_ptr
        return low, high, ptr, obj

    def mask_as_output(
        self, raw: torch.Tensor, hr: list[torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `_use_mask_as_output` for a binary `[P, 1, S, S]` mask: low logits, ptr, object score."""
        high = mask * MEM_SCALE + MEM_BIAS
        low = F.interpolate(high, (LOW, LOW), mode="bilinear", align_corners=False, antialias=True)
        _, _, ptr, _ = self.heads(raw, hr, None, None, self.m.mask_downsample(mask), multimask=False)
        lam = mask.flatten(1).gt(0).any(1, keepdim=True).float()
        obj = MEM_SCALE * lam + MEM_BIAS
        return low, lam * ptr + (1 - lam) * self.m.no_obj_ptr, obj

    def encode(self, raw: torch.Tensor, high: torch.Tensor, obj: torch.Tensor, *, binarize: bool) -> torch.Tensor:
        """Upstream `_encode_new_memory`: bfloat16 memory features `[P, 64, t, t]`."""
        m = (high > 0).float() if binarize else high.sigmoid()
        m = m * MEM_SCALE + MEM_BIAS
        feats = self.m.memory_encoder(raw.to(self.dtype), m.to(self.dtype), skip_mask_sigmoid=True)["vision_features"]
        absent = (obj <= 0).to(feats.dtype)[..., None, None]
        return (feats + absent * self.m.no_obj_embed_spatial.to(feats.dtype)[..., None, None]).to(torch.bfloat16)

    def condition(  # noqa: PLR0913, PLR0917 - the memory frames and the objects that use them
        self,
        raw: torch.Tensor,
        t: int,
        n_frames: int,
        bank: dict[int, tuple[torch.Tensor, torch.Tensor]],
        memory: list[int] | None = None,
        objects: list[int] | None = None,
    ) -> torch.Tensor:
        """Upstream `_prepare_memory_conditioned_features` for frame `t >= 1` (frame 0: the only conditioning frame).

        `memory`: the earlier frames in the bank (oldest first; SAMURAI's selection, the last one gets the nearest
        temporal encoding) instead of the previous 6; `objects`: the rows of the bank to use (all by default).
        """
        p, c, h, w = raw.shape
        dt = self.dtype
        tpos = self.m.maskmem_tpos_enc.to(dt)
        rows = slice(None) if objects is None else objects
        if memory is None:
            memory = list(range(max(1, t - NUM_MASKMEM + 1), t))
            ages = [t - k for k in memory]
        else:
            ages = list(range(len(memory), 0, -1))
        mems, poss = [], []
        for k, age in [(0, NUM_MASKMEM), *zip(memory, ages, strict=True)]:  # frame 0, then the others, oldest first
            feats = bank[k][0][rows].to(dt).flatten(2).permute(2, 0, 1)
            mems.append(feats)
            poss.append(self.mem_pos.expand(-1, p, -1) + tpos[age - 1])
        max_ptrs = min(n_frames, MAX_PTRS)
        ptr_frames = [0, *(t - d for d in range(1, max_ptrs) if t - d >= 1)]  # frame 0, then the most recent first
        rel = torch.tensor([t - k for k in ptr_frames], device=raw.device)
        ptrs = torch.stack([bank[k][1][rows] for k in ptr_frames])  # [n, P, 256]
        pos = self.m.obj_ptr_tpos_proj(get_1d_sine_pe(rel / (max_ptrs - 1), dim=DIM))
        split = c // MEM_DIM
        ptrs = ptrs.reshape(-1, p, split, MEM_DIM).permute(0, 2, 1, 3).flatten(0, 1).to(dt)
        pos = pos[:, None].expand(-1, p, -1).repeat_interleave(split, dim=0).to(dt)
        mems.append(ptrs)
        poss.append(pos)
        out = self.m.memory_attention(
            curr=raw.to(dt).flatten(2).permute(2, 0, 1),
            curr_pos=self.pos.expand(-1, p, -1),
            memory=torch.cat(mems),
            memory_pos=torch.cat(poss),
            num_obj_ptr_tokens=ptrs.shape[0],
        )
        return out.permute(1, 2, 0).reshape(p, c, h, w).float()

    def run(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Iterable[dict[str, torch.Tensor]],
        n_frames: int,
        *,
        coords: torch.Tensor | None,
        labels: torch.Tensor | None,
        mask: torch.Tensor | None,
        emit: Callable[[int, torch.Tensor, torch.Tensor], None],
        kf_weight: float | None = None,
    ) -> None:
        """Track: `frames` yields the encoder features per frame; `emit(t, low_filled, object_score)` per frame.

        Prompts (frame 0, `P` objects): `coords` `[P, N, 2]` in encoder-input pixels + `labels` `[P, N]`, or `mask`
        `[P, 1, S, S]` binary. `kf_weight`: SAMURAI with this motion weight (None: SAM 2).
        """
        bank: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        p = next(x for x in (coords, mask) if x is not None).shape[0]
        motion = None if kf_weight is None else Motion(p, kf_weight)
        for t, f in enumerate(frames):
            pix = f["image_embed"].expand(p, -1, -1, -1)
            raw = pix - self.no_mem
            hr = [f["feat_s0"].expand(p, -1, -1, -1), f["feat_s1"].expand(p, -1, -1, -1)]
            if t == 0:
                if mask is not None:
                    low, ptr, obj = self.mask_as_output(raw, hr, mask)
                else:
                    multimask = labels is not None and labels.shape[1] <= 1
                    low, _, ptr, obj = self.heads(pix, hr, coords, labels, None, multimask, motion)
                low = fill_holes(low)
                high = F.interpolate(low, (self.size, self.size), mode="bilinear", align_corners=False)
                bank[0] = (self.encode(raw, high, obj, binarize=True), ptr)
            else:
                if motion is None:
                    cond = self.condition(raw, t, n_frames, bank)
                else:
                    cond = torch.empty_like(raw)
                    for memory, rows in motion.memory(t).items():
                        sub = raw[rows]
                        cond[rows] = self.condition(sub, t, n_frames, bank, list(memory), rows)
                low, high, ptr, obj = self.heads(cond, hr, None, None, None, True, motion)  # noqa: FBT003
                bank[t] = (self.encode(raw, high, obj, binarize=False), ptr)
                keep = set() if motion is None else motion.kept(t, obj)
                for k in [k for k in bank if 0 < k <= t - MAX_PTRS + 1 and k not in keep]:  # no later frame reads it
                    del bank[k]
                low = fill_holes(low)
            emit(t, low, obj)


class Motion:
    """SAMURAI's per-object state: Kalman filter of the mask box, stable-frame count, memory-bank frames."""

    def __init__(self, objects: int, weight: float) -> None:
        """`objects` tracks with motion weight `weight` (`kf_score_weight`)."""
        self.weight = weight
        self.mean = np.zeros((objects, 8))
        self.cov = np.zeros((objects, 8, 8))
        self.stable = np.zeros(objects, dtype=np.int64)
        self.best_iou = np.zeros(objects, dtype=np.float32)  # predicted IoU of the chosen mask (this frame)
        self.valid: list[list[int]] = [[] for _ in range(objects)]  # the last memory frames that passed, per object

    @staticmethod
    def boxes(low: torch.Tensor, size: int) -> np.ndarray:
        """`[P, M, 256, 256]` logits -> `[P, M, 4]` float64 boxes `(x0, y0, x1, y1)` of the `> 0` pixels at `size`."""
        fg = F.interpolate(low, (size, size), mode="bilinear", align_corners=False) > 0
        idx = torch.arange(size, device=low.device)
        cols, rows = fg.any(2), fg.any(3)
        box = torch.stack(
            (
                torch.where(cols, idx, size).amin(-1),
                torch.where(rows, idx, size).amin(-1),
                torch.where(cols, idx, -1).amax(-1),
                torch.where(rows, idx, -1).amax(-1),
            ),
            -1,
        )
        return torch.where(cols.any(-1, keepdim=True), box, 0).cpu().numpy().astype(np.float64)

    @staticmethod
    def xyah(box: np.ndarray) -> np.ndarray:
        """SAMURAI `xyxy_to_xyah` of `[N, 4]` boxes (a zero height counts as 1)."""
        w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
        h = np.where(h == 0, 1.0, h)
        return np.stack(((box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2, w / h, h), 1)

    def select(self, low: torch.Tensor, iou: torch.Tensor, size: int) -> torch.Tensor:
        """Index `[P]` of the multimask output to keep (upstream `_forward_sam_heads` with `samurai_mode`)."""
        boxes = self.boxes(low, size)  # [P, M, 4]
        ious = iou.float().cpu().numpy()  # [P, M]
        p = np.arange(len(ious))
        best = ious.argmax(1)
        new, scored = self.stable == 0, self.stable >= STABLE_FRAMES
        run = ~new
        if run.any():
            self.mean[run], self.cov[run] = box_kalman.predict(self.mean[run], self.cov[run])
        if scored.any():
            pred = box_kalman.to_xyxy(self.mean[scored])[:, None]  # [S, 1, 4]
            b = boxes[scored]
            iw = np.clip(np.minimum(pred[..., 2], b[..., 2]) - np.maximum(pred[..., 0], b[..., 0]), 0, None)
            ih = np.clip(np.minimum(pred[..., 3], b[..., 3]) - np.maximum(pred[..., 1], b[..., 1]), 0, None)
            inter = iw * ih
            area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
            union = area + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
            kf = np.where((union != 0) & b.any(-1), inter / np.where(union == 0, 1, union), 0.0)
            best[scored] = (self.weight * kf + (1 - self.weight) * ious[scored].astype(np.float64)).argmax(1)  # float64
        chosen = ious[p, best]
        ok = np.where(scored, chosen >= STABLE_IOU, chosen > STABLE_IOU)
        upd = run & ok
        if upd.any():
            self.mean[upd], self.cov[upd] = box_kalman.update(
                self.mean[upd], self.cov[upd], self.xyah(boxes[upd, best[upd]])
            )
        if new.any():
            self.mean[new], self.cov[new] = box_kalman.initiate(self.xyah(boxes[new, best[new]]))
        self.stable = np.where(new, 1, np.where(ok, np.where(scored, self.stable, self.stable + 1), 0))
        self.best_iou = chosen
        return torch.from_numpy(best).to(low.device)

    def memory(self, t: int) -> dict[tuple[int, ...], list[int]]:
        """The memory frames of frame `t` (oldest first) -> the objects that use them.

        Upstream: the valid frames among `t - 1 ... 2` (frame 1 only as the previous frame), the previous frame always,
        the last `NUM_MASKMEM - 1` of them (frame 0 is the conditioning frame and comes separately).
        """
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, valid in enumerate(self.valid):
            frames = [k for k in valid if k < t - 1]
            frames = [*frames, t - 1][-(NUM_MASKMEM - 1) :] if t > 1 else []
            groups.setdefault(tuple(frames), []).append(i)
        return groups

    def kept(self, t: int, obj: torch.Tensor) -> set[int]:
        """Record whether frame `t` may be a memory frame per object; the frames the memory can still select."""
        present = (obj[:, 0] > 0).cpu().numpy()
        for i, valid in enumerate(self.valid):
            if t >= 2 and self.best_iou[i] > MEMORY_IOU and present[i]:  # noqa: PLR2004 - upstream skips frame 1
                valid.append(t)
                del valid[: -(NUM_MASKMEM - 1)]
        return {k for valid in self.valid for k in valid}
