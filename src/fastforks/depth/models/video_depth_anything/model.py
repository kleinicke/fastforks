"""Video Depth Anything runners: weights -> network and the portable / kernels forward passes of a whole video.

A runner call takes one video in time order (`[L, 3, H, W]`; the model's `video`; a batch of the model runs frame by
frame, `batch = 1`). Both runners follow upstream `infer_video_depth`:

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
from fastforks._core.runner import host
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
DECODE_BATCH = 4  # fixed: the upstream head's decoder micro-batch size
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


def _blend(
    out: torch.Tensor, depth: torch.Tensor, reference: list[torch.Tensor], start: int, *, metric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align the window and blend the overlap in the upstream operation order."""
    if metric:
        factor = torch.ones((), device=depth.device)
        shift = torch.zeros((), device=depth.device)
    else:
        factor, shift = _scale_shift(depth[:ALIGN], torch.stack(reference))
    for i in range(min(INTERP_LEN, depth.shape[0] - ALIGN)):
        frame = start + ALIGN + i
        weight = i / (INTERP_LEN - 1)
        out[frame] = out[frame] * (1 - weight) + (depth[ALIGN + i] * factor + shift).clamp_min(0) * weight
    if depth.shape[0] > OVERLAP:
        out[start + OVERLAP : start + depth.shape[0]] = (depth[OVERLAP:] * factor + shift).clamp_min(0)
    return factor, shift


class _Video(Runner):
    """The windowing / alignment of `infer_video_depth`; subclasses run the encoder and the head."""

    batch = 1  # a batch is independent images; `DepthModel.video` passes the whole video
    video = True
    Params = Params
    compiled_methods = ("_encode", "_window")  # the networks; the windowing loop stays eager
    net: Net

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Normalised frames `[n, 3, h, w]` -> `[tokens, cls] * 4` of the head's encoder layers."""
        feats = self.net.pretrained.get_intermediate_layers(x, _LAYERS[self.ctx.params.arch], return_class_token=True)
        return [t for pair in feats for t in pair]

    def _window(self, feats: list[torch.Tensor], h: int, w: int, valid: int) -> torch.Tensor:
        """Encoder features of the 32 frames of a window -> float32 depth `[valid, h, w]` (ReLU)."""
        pairs = [(feats[i], feats[i + 1]) for i in range(0, len(feats), 2)]
        d = self.net.head(pairs, h // PATCH, w // PATCH, INFER_LEN)[0]  # [32, 1, h, w]; h, w: multiples of 14
        return F.relu(d[:valid, 0]).float()  # upstream resizes to (h, w) first: the same size, an exact copy

    def _video(self, image: torch.Tensor) -> torch.Tensor:
        """`[L, 3, H, W]` RGB in `[0, 1]` (one video in time order) -> aligned float32 depth `[L, H, W]`."""
        n, _, hh, ww = image.shape
        h, w = network_size(hh, ww, self.ctx.params.short_side)
        out = torch.empty((n, hh, ww), device=image.device, dtype=torch.float32)
        cache_ids: list[int] = []
        cache: list[torch.Tensor] = []
        ids: list[int] = []
        ref: list[torch.Tensor] = []
        # An overlap containing only the alignment anchors yields no new or blended output.
        end = max(1, n - ALIGN)
        for start in range(0, end, STEP):
            valid = min(INFER_LEN, n - start)
            more = start + STEP < end
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
                del x, fresh
            pos = {f: k for k, f in enumerate(pool_ids)}
            if ids == pool_ids:
                window = pool
            else:
                sel = torch.tensor([pos[i] for i in ids], device=image.device)
                window = [t[sel] for t in pool]
            # Only these keyframes can reappear in the next window. The head consumes
            # the assembled window; release the full encoder pool before it runs.
            cache_ids = sorted({ids[k] for k in KEYFRAMES}) if more else []
            keep = torch.tensor([pos[i] for i in cache_ids], device=image.device, dtype=torch.long)
            cache = [t[keep] for t in pool] if cache_ids else []
            del pool
            d = self._window(window, h, w, valid)
            del window
            d = F.interpolate(d[:, None], (hh, ww), mode="bilinear", align_corners=True)[:, 0]
            if ref:
                s, b = _blend(out, d, ref, start, metric=self.ctx.params.metric)
                if more:
                    ref[1] = (d[KEYFRAMES[1]] * s + b).clamp_min(0)
            else:
                out[:valid] = d
                if more:
                    ref = [d[0].clone(), d[KEYFRAMES[1]].clone()]
        return out

    @host
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


def _temporal(module: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Run a temporal module on one clip and return its refined feature maps."""
    temporal, _ = module(frames.transpose(0, 1).unsqueeze(0), None, None, None)
    return temporal[0].transpose(0, 1)


def _partial_head(head: DPTHeadTemporal, feats: list[torch.Tensor], h: int, w: int, valid: int) -> torch.Tensor:
    """Adapt ByteDance's DPT head to decode only the output prefix of a padded window.

    All four motion modules still see the full 32 frames. Only the operations after
    the last motion module can drop padded frames without changing the real frames.
    """
    ph, pw = h // PATCH, w // PATCH
    maps = []
    for i, tokens in enumerate(feats[::2]):
        x = tokens
        if head.use_clstoken:
            readout = feats[2 * i + 1].unsqueeze(1).expand_as(x)
            x = head.readout_projects[i](torch.cat((x, readout), -1))
        x = x.transpose(1, 2).reshape(INFER_LEN, x.shape[-1], ph, pw).contiguous()
        maps.append(head.resize_layers[i](head.projects[i](x)))

    low, mid, high, coarse = maps
    scratch = head.scratch
    low = scratch.layer1_rn(low)
    mid = scratch.layer2_rn(mid)
    high = scratch.layer3_rn(_temporal(head.motion_modules[0], high))
    coarse = scratch.layer4_rn(_temporal(head.motion_modules[1], coarse))
    path = scratch.refinenet4(coarse, size=high.shape[2:])
    path = _temporal(head.motion_modules[2], path)
    path = scratch.refinenet3(path, high, size=mid.shape[2:])
    path = _temporal(head.motion_modules[3], path)

    result = []
    for start in range(0, valid, DECODE_BATCH):
        section = slice(start, min(start + DECODE_BATCH, valid))
        x = scratch.refinenet2(path[section], mid[section], size=low.shape[2:])
        x = scratch.refinenet1(x, low[section])
        x = F.interpolate(scratch.output_conv1(x), (h, w), mode="bilinear", align_corners=True)
        with torch.autocast("cuda", enabled=False):
            depth = scratch.output_conv2(x.float()).to(x.dtype)
        result.append(F.relu(depth[:, 0]).float())
    return result[0] if len(result) == 1 else torch.cat(result)


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

    def _window(self, feats: list[torch.Tensor], h: int, w: int, valid: int) -> torch.Tensor:
        """Use the shortened per-frame decoder for the last, possibly padded window."""
        if valid == INFER_LEN:
            return super()._window(feats, h, w, valid)
        return _partial_head(self.net.head, feats, h, w, valid)


RUNNERS = {"portable": Portable, "kernels": Kernels}
