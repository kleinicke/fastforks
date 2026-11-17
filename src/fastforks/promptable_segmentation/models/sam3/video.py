"""SAM 3 video tracking: transformers `Sam3TrackerVideoModel` with every prompt on frame 0 (`net/tracker.py`).

Follows `Sam3TrackerVideoProcessor` / `Sam3TrackerVideoModel` (v5.19.0; the SAM 3 repo's tracker) for one
`add_inputs_to_inference_session` per object on frame 0, then `propagate_in_video_iterator` forward:

- frames: the image runner's input (uint8 bilinear antialiased resize to 1008, mean / std 0.5), the shared ViT trunk,
  the tracker's own neck (`tracker_neck`, levels 288 / 144 / 72), the 288 / 144 levels through `conv_s0` / `conv_s1`
- frame 0 (conditioning frame): the heads on the 72 x 72 level plus `no_memory_embedding`, multimask for at most one
  point (a box is two points with labels 2 / 3 in front of the clicks); one mask: the single-mask output, or the best
  multimask output when it is unstable (`dynamic_multimask_via_stability`); a mask prompt is used as the output
  directly (`_use_mask_as_output`, the object pointer from the heads on the raw features with the downsampled mask)
- later frames: memory attention over frame 0 + the 6 previous frames (`num_maskmem` 7) and the object pointers of
  frame 0 + up to 15 previous frames, then the heads without prompts (multimask, best predicted IoU)
- memory: the memory encoder on the 1152 px mask (antialiased bilinear from 1008; frame 0: binarised, later frames:
  sigmoid), `x 20 - 10`, `occlusion_spatial_embedding_parameter` for absent objects, stored as bfloat16 (as upstream)
- output: the 288 x 288 low-resolution logits (no hole filling: the SAM 3 tracker has none), bilinear to the video
  size, `> 0`

All objects run as one batch (upstream runs them one by one on the same features; the same maths). The heads run in
float32; the memory modules in the compute dtype.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .net.sam3 import GRID, grid_position_encoding
from .net.tracker import (
    DIM,
    MEM_DIM,
    NUM_MASKMEM,
    SIZE,
    TrackerModel,
    get_1d_sine_pe,
    memory_position_encoding,
    rope_tables,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

MAX_PTRS = 16  # fixed: object pointers in memory attention (max_object_pointers_in_encoder)
MEM_SCALE, MEM_BIAS = 20.0, -10.0  # fixed: sigmoid_scale / bias_for_mem_enc (also the mask-as-output logits)
NO_OBJ = -1024.0  # fixed: upstream NO_OBJ_SCORE
LOW = 288  # fixed: low-resolution mask side (prompt_encoder.mask_input_size)
MEM_SIZE = 4 * LOW  # fixed: memory-encoder mask side (1152)


def to_device(values: list[int], device: torch.device) -> torch.Tensor:
    """A small int64 tensor of `values` on `device` (on a GPU via pinned memory: no host sync)."""
    x = torch.tensor(values)
    return x.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else x.to(device)


def prefetch(load: Callable[[int], torch.Tensor], n: int) -> Iterator[torch.Tensor]:
    """`load(0) ... load(n - 1)`, each loaded in a worker thread while the caller works on the previous one."""
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(load, 0)
        for t in range(n):
            x = nxt.result()
            if t + 1 < n:
                nxt = pool.submit(load, t + 1)
            yield x


class HostCopies:
    """Device tensors -> host tensors `out[name][t]` without blocking the host (a ring of pinned buffers)."""

    def __init__(self, out: dict[str, torch.Tensor], depth: int = 2) -> None:
        """Targets `out` (name -> `[T, ...]` host tensor); `depth` frames in flight."""
        self.out = out
        self.ring: list[tuple[int, dict[str, torch.Tensor], torch.cuda.Event] | None] = [None] * depth
        self.bufs: list[dict[str, torch.Tensor]] = [{} for _ in range(depth)]
        self.n = 0

    def put(self, t: int, **xs: torch.Tensor) -> None:
        """`out[name][t] = x` for each `name=x`, once the copy is done (at the latest at `flush`)."""
        if not next(iter(xs.values())).is_cuda:
            for k, x in xs.items():
                self.out[k][t] = x
            return
        i, self.n = self.n % len(self.ring), self.n + 1
        self._drain(i)
        bufs = self.bufs[i]
        for k, x in xs.items():
            if k not in bufs or bufs[k].shape != x.shape or bufs[k].dtype != x.dtype:
                bufs[k] = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
            bufs[k].copy_(x, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        self.ring[i] = (t, bufs, done)

    def _drain(self, i: int) -> None:
        if (slot := self.ring[i]) is not None:
            t, bufs, done = slot
            done.synchronize()
            for k, b in bufs.items():
                self.out[k][t] = b
            self.ring[i] = None

    def flush(self) -> None:
        """Wait for the copies in flight and write them."""
        for j in range(len(self.ring)):
            self._drain((self.n + j) % len(self.ring))


class Tracker:
    """The tracking state machine for one video (frame 0 prompts, forward propagation)."""

    def __init__(self, m: TrackerModel, dtype: torch.dtype) -> None:
        """`m`: float32 except `memory_attention` / `memory_encoder` (in `dtype`)."""
        self.m, self.dtype = m, dtype
        dev = m.no_memory_embedding.device
        self.no_mem = m.no_memory_embedding.view(1, -1, 1, 1)
        self.pos = grid_position_encoding(GRID, GRID, dev).flatten(2).transpose(1, 2).to(dtype)  # [1, HW, 256]
        self.mem_pos = memory_position_encoding(dev).flatten(2).transpose(1, 2).to(dtype)  # [1, HW, 64]
        self.rope = rope_tables(dev)
        self.image_pe = m.image_pe().detach()

    def heads(  # noqa: PLR0913, PLR0917 - upstream `_single_frame_forward` arguments
        self,
        pix: torch.Tensor,
        hr: list[torch.Tensor],
        coords: torch.Tensor | None,
        labels: torch.Tensor | None,
        mask_in: torch.Tensor | None,
        multimask: bool,  # noqa: FBT001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `_single_frame_forward` (float32): low `[P, 1, 288, 288]`, high `[P, 1, 1008, 1008]`, ptr, score."""
        p = pix.shape[0]
        if coords is None or labels is None:
            coords = torch.zeros(p, 1, 2, device=pix.device)
            labels = -torch.ones(p, 1, dtype=torch.int32, device=pix.device)
        sparse, dense = self.m.prompt_encoder(coords[:, None].float(), labels[:, None], mask_in)
        low, iou, tokens, obj = self.m.mask_decoder(pix, self.image_pe, sparse, dense, hr, multimask=multimask)
        present = obj > 0  # [P, 1]
        low = torch.where(present[:, :, None, None], low, NO_OBJ)
        token = tokens[:, 0]
        if multimask:
            best = iou.argmax(-1)
            rows = torch.arange(p, device=pix.device)
            low, token = low[rows, best][:, None], tokens[rows, best]
        high = F.interpolate(low, (SIZE, SIZE), mode="bilinear", align_corners=False)
        lam = present.float()
        ptr = lam * self.m.object_pointer_proj(token) + (1 - lam) * self.m.no_object_pointer
        return low, high, ptr, obj

    def mask_as_output(
        self, raw: torch.Tensor, hr: list[torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `_use_mask_as_output` for a binary `[P, 1, 1008, 1008]` mask: low, high, ptr, object score."""
        high = mask * MEM_SCALE + MEM_BIAS
        low = F.interpolate(high, (LOW, LOW), mode="bilinear", align_corners=False, antialias=True)
        m_in = F.interpolate(self.m.mask_downsample(mask), (LOW, LOW), mode="bilinear", antialias=True)
        _, _, ptr, _ = self.heads(raw, hr, None, None, m_in, multimask=True)
        lam = mask.flatten(1).gt(0).any(1, keepdim=True).float()
        obj = MEM_SCALE * lam + MEM_BIAS
        return low, high, lam * ptr + (1 - lam) * self.m.no_object_pointer, obj

    def encode(self, raw: torch.Tensor, high: torch.Tensor, obj: torch.Tensor, *, binarize: bool) -> torch.Tensor:
        """Upstream `_encode_new_memory`: bfloat16 memory features `[P, 64, 72, 72]`."""
        high = F.interpolate(high, (MEM_SIZE, MEM_SIZE), mode="bilinear", align_corners=False, antialias=True)
        m = (high > 0).float() if binarize else high.sigmoid()
        m = m * MEM_SCALE + MEM_BIAS
        feats = self.m.memory_encoder(raw.to(self.dtype), m.to(self.dtype))
        absent = (obj <= 0).to(feats.dtype)[..., None, None]
        occ = self.m.occlusion_spatial_embedding_parameter.to(feats.dtype)[..., None, None]
        return (feats + absent * occ).to(torch.bfloat16)

    def condition(
        self, raw: torch.Tensor, t: int, n_frames: int, bank: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Upstream `_prepare_memory_conditioned_features` for frame `t >= 1` (frame 0: the only conditioning frame)."""
        p, c, h, w = raw.shape
        dt = self.dtype
        tpos = self.m.memory_temporal_positional_encoding.to(dt)[:, 0]  # [7, 1, 64]
        mems, poss = [], []
        for k, age in [(0, NUM_MASKMEM), *((k, t - k) for k in range(max(1, t - NUM_MASKMEM + 1), t))]:
            mems.append(bank[k][0].to(dt).flatten(2).transpose(1, 2))  # [P, HW, 64]
            poss.append((self.mem_pos + tpos[age - 1]).expand(p, -1, -1))
        max_ptrs = min(n_frames, MAX_PTRS)
        ptr_frames = [0, *(t - d for d in range(1, max_ptrs) if t - d >= 1)]  # frame 0, then the most recent first
        rel = to_device([t - k for k in ptr_frames], raw.device)
        n, split = len(ptr_frames), c // MEM_DIM
        ptrs = torch.stack([bank[k][1] for k in ptr_frames])  # [n, P, 256]
        ptrs = ptrs.reshape(n, p, split, MEM_DIM).permute(1, 0, 2, 3).reshape(p, n * split, MEM_DIM)
        pos = self.m.temporal_positional_encoding_projection_layer(get_1d_sine_pe(rel / (max_ptrs - 1), DIM))
        mems.append(ptrs.to(dt))
        poss.append(pos.repeat_interleave(split, dim=0)[None].expand(p, -1, -1).to(dt))
        out = self.m.memory_attention(
            raw.to(dt).flatten(2).transpose(1, 2),
            self.pos.expand(p, -1, -1),
            torch.cat(mems, 1),
            torch.cat(poss, 1),
            self.rope,
            n * split,
        )
        return out.transpose(1, 2).reshape(p, c, h, w).float()

    def run(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Iterable[dict[str, torch.Tensor]],
        n_frames: int,
        *,
        coords: torch.Tensor | None,
        labels: torch.Tensor | None,
        mask: torch.Tensor | None,
        emit: Callable[[int, torch.Tensor, torch.Tensor], None],
    ) -> None:
        """Track: `frames` yields `raw`, `feat_s0`, `feat_s1` (float32) per frame; `emit(t, low, object_score)`.

        Prompts (frame 0, `P` objects): `coords` `[P, N, 2]` in 1008-px input coordinates + `labels` `[P, N]`, or
        `mask` `[P, 1, 1008, 1008]` binary.
        """
        bank: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        p = next(x for x in (coords, mask) if x is not None).shape[0]
        for t, f in enumerate(frames):
            raw = f["raw"].expand(p, -1, -1, -1)
            hr = [f["feat_s0"].expand(p, -1, -1, -1), f["feat_s1"].expand(p, -1, -1, -1)]
            if t == 0:
                if mask is not None:
                    low, high, ptr, obj = self.mask_as_output(raw, hr, mask)
                else:
                    multimask = labels is not None and labels.shape[1] <= 1
                    low, high, ptr, obj = self.heads(raw + self.no_mem, hr, coords, labels, None, multimask)
                bank[0] = (self.encode(raw, high, obj, binarize=True), ptr)
            else:
                cond = self.condition(raw, t, n_frames, bank)
                low, high, ptr, obj = self.heads(cond, hr, None, None, None, True)  # noqa: FBT003
                bank[t] = (self.encode(raw, high, obj, binarize=False), ptr)
                for k in [k for k in bank if 0 < k <= t - MAX_PTRS + 1]:  # no later frame reads it
                    del bank[k]
            emit(t, low, obj)
