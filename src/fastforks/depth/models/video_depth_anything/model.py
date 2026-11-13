"""Video Depth Anything runners: weights -> network and the portable / kernels forward passes of a whole video.

A call's batch is one video in time order (`[L, 3, H, W]`). Both runners follow upstream `infer_video_depth`:

- resize so that both sides are at least `short_side` and multiples of 14 (aspect kept, bicubic; videos wider than
  16:9 lower `short_side` by the aspect ratio, as upstream), normalise with the ImageNet mean / std
- windows of 32 frames every 22 frames (the video padded with copies of its last frame); in each window after the
  first, the first 10 frames are keyframes 0, 12, 24..31 of the previous window
- each window's depth: ReLU, bilinear (`align_corners=True`) to the frame size
- relative models: the window is aligned to the previous one by a least-squares scale + shift between its first two
  frames and the aligned keyframes 0 and 12 (sums in float32); the 8 overlapping frames blend linearly from the old
  to the new estimate; metric models skip the scale + shift.

The DINOv2 encoder runs per frame, so its features of frames already seen in the previous window (the keyframes, the
padding copies) are reused instead of recomputed: the same values, about a third less encoder work per window. The
aligned depth is written into one `[L, H, W]` float32 tensor on the device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import pure_half
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.shared.depth_anything_v2 import DINOv2, fold_layer_scale, use_fast_vit, use_sdpa_attention

from .net.dpt_temporal import DPTHeadTemporal
from .net.motion_module.motion_module import TemporalAttention

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

PATCH = 14  # fixed: the DINOv2 ViT-S/14 patch size of the checkpoints
INFER_LEN = 32  # fixed: upstream video_depth.py "infer settings, do not change" (the trained window)
OVERLAP = 10  # fixed: upstream infer settings
KEYFRAMES = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)  # fixed: upstream infer settings
INTERP_LEN = 8  # fixed: upstream infer settings
STEP = INFER_LEN - OVERLAP
ALIGN = OVERLAP - INTERP_LEN  # frames 0 and 1 of a window: keyframes 0 and 12 of the previous one
MAX_RATIO = 1.78  # fixed: upstream: "process video with ratio smaller than 16:9 due to memory limitation"
_CONFIGS = {  # fixed: upstream run.py model_configs (per `arch`)
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
}
_LAYERS = {"vits": [2, 5, 8, 11]}  # fixed: upstream intermediate_layer_idx (per `arch`)


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: str  # "vits"
    short_side: int  # lower bound of both resized sides (upstream input_size)
    metric: bool  # metric depth (no scale + shift alignment between windows)


def network_size(h: int, w: int, short_side: int) -> tuple[int, int]:
    """Upstream `infer_video_depth` resize for `h x w` frames (16:9 cap, then lower bound, multiples of 14)."""
    ratio = max(h, w) / min(h, w)
    if ratio > MAX_RATIO:
        short_side = round(int(short_side * 1.777 / ratio) / PATCH) * PATCH
    s = max(short_side / h, short_side / w)

    def multiple(x: float) -> int:
        y = round(x / PATCH) * PATCH
        return y if y >= short_side else math.ceil(x / PATCH) * PATCH

    return multiple(s * h), multiple(s * w)


class Net(nn.Module):
    """Upstream `VideoDepthAnything`: encoder + temporal DPT head (`forward` / `infer_video_depth`: the runners)."""

    def __init__(self, arch: str) -> None:
        """The network of one encoder size."""
        super().__init__()
        self.pretrained = DINOv2(model_name=arch)
        cfg = _CONFIGS[arch]
        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim, cfg["features"], use_bn=False, out_channels=cfg["out_channels"]
        )


class SdpaTemporalAttention(TemporalAttention):
    """`TemporalAttention` with fused `scaled_dot_product_attention` (same maths as upstream's baddbmm / softmax)."""

    def _attention(self, query, key, value, attention_mask=None):  # noqa: ANN001, ANN202 - upstream signature
        if attention_mask is not None:  # upstream TemporalAttention asserts no mask
            msg = "TemporalAttention: no attention mask"
            raise ValueError(msg)
        q, k, v = (t.unflatten(0, (-1, self.heads)) for t in (query, key, value))  # [B, heads, frames, head_dim]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.reshape_batch_dim_to_heads(out.flatten(0, 1))


def _cpu_net(ctx: Context) -> Net:
    net = Net(ctx.params.arch)
    net.load_state_dict(ctx.state_dict())
    for m in net.modules():
        if isinstance(m, TemporalAttention):
            m.__class__ = SdpaTemporalAttention
    return net.eval()


def _key(ctx: Context) -> str:
    return "depth" if ctx.params.metric else "inverse_depth"


def _scale_shift(p: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `compute_scale_and_shift_full` (all-ones mask): least squares `t ~ s * p + b`; `(1, 0)` if singular."""
    p, t = p.flatten().float(), t.flatten().float()
    a00, a01, a11 = (p * p).sum().double(), p.sum().double(), float(p.numel())
    b0, b1 = (p * t).sum().double(), t.sum().double()
    det = a00 * a11 - a01 * a01
    ok = det != 0
    safe = torch.where(ok, det, torch.ones_like(det))
    s = torch.where(ok, (a11 * b0 - a01 * b1) / safe, torch.ones_like(det))
    b = torch.where(ok, (-a01 * b0 + a00 * b1) / safe, torch.zeros_like(det))
    return s.float(), b.float()


class _Video(Runner):
    """The windowing / alignment of `infer_video_depth`; subclasses run the encoder and the head."""

    batch = None
    Params = Params
    compiled_methods = ("_encode", "_window")  # the networks; the windowing loop stays eager
    net: Net

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Normalised frames `[n, 3, h, w]` -> `[tokens, cls] * 4` of the head's encoder layers."""
        feats = self.net.pretrained.get_intermediate_layers(x, _LAYERS[self.ctx.params.arch], return_class_token=True)
        return [t for pair in feats for t in pair]

    def _window(self, feats: list[torch.Tensor], h: int, w: int) -> torch.Tensor:
        """Encoder features of the 32 frames of a window -> float32 depth `[32, h, w]` (ReLU)."""
        pairs = [(feats[i], feats[i + 1]) for i in range(0, len(feats), 2)]
        d = self.net.head(pairs, h // PATCH, w // PATCH, INFER_LEN)[0]  # [32, 1, h, w]; h, w: multiples of 14
        return F.relu(d[:, 0]).float()  # upstream resizes to (h, w) first: the same size, an exact copy

    def _video(self, image: torch.Tensor) -> torch.Tensor:
        """`[L, 3, H, W]` RGB in `[0, 1]` (one video in time order) -> aligned float32 depth `[L, H, W]`."""
        n, _, hh, ww = image.shape
        h, w = network_size(hh, ww, self.ctx.params.short_side)
        out = torch.empty((n, hh, ww), device=image.device, dtype=torch.float32)
        cache_ids: list[int] = []
        cache: list[torch.Tensor] = []
        ids: list[int] = []
        ref: list[torch.Tensor] = []
        for start in range(0, n, STEP):
            prev, ids = ids, [min(start + i, n - 1) for i in range(INFER_LEN)]  # padded with the last frame
            if prev:
                ids[:OVERLAP] = [prev[k] for k in KEYFRAMES]
            uniq = sorted(set(ids))
            new = [i for i in uniq if i not in cache_ids]
            pool_ids, pool = cache_ids + new, cache
            if new:  # the last window may hold only seen frames (keyframes and padding)
                x = F.interpolate(image[new], (h, w), mode="bicubic", align_corners=False)
                fresh = self._encode(scale(x, self.ctx.io))
                pool = [torch.cat((c, f)) for c, f in zip(cache, fresh, strict=True)] if cache else fresh
            pos = {f: k for k, f in enumerate(pool_ids)}
            sel = torch.tensor([pos[i] for i in ids], device=image.device)
            d = self._window([t[sel] for t in pool], h, w)
            keep = torch.tensor([pos[i] for i in uniq], device=image.device)
            cache_ids, cache = uniq, [t[keep] for t in pool]
            d = F.interpolate(d[:, None], (hh, ww), mode="bilinear", align_corners=True)[:, 0]
            if not ref:
                m = min(INFER_LEN, n)
                out[:m] = d[:m]
                ref = [d[0], d[KEYFRAMES[1]]]
                continue
            if self.ctx.params.metric:
                s, b = torch.ones((), device=d.device), torch.zeros((), device=d.device)
            else:
                s, b = _scale_shift(d[:ALIGN], torch.stack(ref))
            for i in range(INTERP_LEN):  # upstream get_interpolate_frames: weights 0, 1/7, ..., 1
                f = start + ALIGN + i
                if f < n:
                    wt = i / (INTERP_LEN - 1)
                    out[f] = out[f] * (1 - wt) + (d[ALIGN + i] * s + b).clamp_min(0) * wt
            m = min(INFER_LEN, n - start)
            if m > OVERLAP:
                out[start + OVERLAP : start + m] = (d[OVERLAP:m] * s + b).clamp_min(0)
            ref = [ref[0], (d[KEYFRAMES[1]] * s + b).clamp_min(0)]
        return out

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Relative inverse depth (metric depth for metric checkpoints) of one video (the batch, in time order)."""
        return {_key(self.ctx): self._video(image)}


class Portable(_Video):
    """Plain PyTorch with fused SDPA attention (encoder and temporal modules).

    float32, or on CUDA the network in fp16 (no autocast; the last output conv runs in fp32).
    """

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _cpu_net(ctx).to(ctx.device)
        use_sdpa_attention(self.net.pretrained)
        fold_layer_scale(self.net.pretrained)
        self.half = ctx.dtype == "float16"
        if self.half:
            pure_half(self.net, fp32=[self.net.head.scratch.output_conv2])

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return super()._encode(x.half() if self.half else x)

    def module(self) -> torch.nn.Module:
        """The network with its windowing: a video `[L, 3, H, W]` in `[0, 1]` -> depth `[L, H, W]`."""
        return Forward(self.net, lambda _net, image: self._video(image))


class Kernels(_Video):
    """fp16 network: the `fastbb` DINOv2 (`use_fast_vit`) and the portable runner's fp16 head.

    `use_fast_vit`: fp16-accumulate GEMMs and flash attention; the head: SDPA temporal attention, the last output conv
    in fp32.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the network, convert the encoder and the head."""
        super().__init__(ctx)
        self.net = _cpu_net(ctx).to(ctx.device)
        use_fast_vit(self.net.pretrained)
        pure_half(self.net.head, fp32=[self.net.head.scratch.output_conv2])


RUNNERS = {"portable": Portable, "kernels": Kernels}
