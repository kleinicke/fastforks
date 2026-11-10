"""Segmentation networks built by Hugging Face transformers, and the upstream test-time protocols.

We extend transformers rather than re-implement it: a family's card names the model class (`[params] arch`, e.g.
`"SegformerForSemanticSegmentation"`), the variant's files are the checkpoint's `config.json` and weights from the
authors' Hugging Face repository (hash-checked by the core), and `pretrained` builds the class from them exactly as
`from_pretrained` does (including transformers' checkpoint key conversions).

The protocol helpers reproduce how each upstream evaluates on ADE20K, on `[B, 3, H, W]` float images:

- `target_size`: mmseg `Resize(scale=(long, short), keep_ratio=True)` or detectron2 `ResizeShortestEdge(short,
  max_size)`; `resize` is OpenCV's `INTER_LINEAR` (bilinear, half-pixel centres, no antialiasing) on the device.
- `upsample_argmax`: the per-pixel argmax of upsampled logits, class chunk by chunk (no full-resolution logits).
- `slide`: mmseg `slide_inference` (overlapping crops, logits averaged).
- `eomt_windows`: EoMT's semantic windowing (square windows along the longer side, overlap spread evenly).
- `mask_semantic`: detectron2 / Mask2Former `semantic_inference` (class softmax without the void class times the
  sigmoid of the mask logits).
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core import kernels
from fastforks._core.checkpoint import mismatch_details
from fastforks._core.errors import (
    BackendUnavailableError,
    CheckpointMismatchError,
    MissingDependencyError,
    install_hint,
)
from fastforks._core.kernels import note_fallback
from fastforks._core.model import eager_in_compile

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from types import ModuleType

    from fastforks._core.runner import Context

__all__ = [
    "eomt_windows",
    "mask_semantic",
    "pad_bottom_right",
    "pretrained",
    "resize",
    "slide",
    "target_size",
    "transformers",
    "upsample",
    "upsample_argmax",
]


def transformers() -> ModuleType:
    """The installed `transformers`.

    Raises:
        BackendUnavailableError: transformers is not installed (MissingDependencyError) or is not 5.x.
    """
    try:
        import transformers  # noqa: PLC0415 - optional dependency (extra `transformers`)
    except ImportError as e:
        msg = (
            f"this model is built by transformers, which is not installed or cannot be imported ({e})"
            f": {install_hint('transformers')}"
        )
        raise MissingDependencyError(msg) from e
    if transformers.__version__.split(".", 1)[0] != "5":  # the patched Swin / Mask2Former call signatures are 5.x
        msg = f"needs transformers 5.x, found {transformers.__version__}: {install_hint('transformers')}"
        raise BackendUnavailableError(msg)
    return transformers


# The two tensor groups the Mask2Former / OneFormer Swin-L checkpoints legitimately differ in (matched in full):
# the backbone's final norm (the pixel decoder reads the per-stage norms instead; missing) and the relative-position
# index buffers (recomputed from the window size; unexpected). Any other key is a mismatch.
_UNUSED_MISSING = re.compile(r"model\.pixel_level_module\.encoder\.swin\.layernorm\.(?:weight|bias)")
_UNUSED_UNEXPECTED = re.compile(
    r"model\.pixel_level_module\.encoder\.swin\.encoder\.layers\.\d+\.blocks\.\d+\.attention\.self\.relative_position_index"
)


@contextlib.contextmanager
def _quiet(tf: ModuleType) -> Generator[None]:
    """No weight-loading progress bar (restored afterwards)."""
    log = tf.utils.logging
    was = log.is_progress_bar_enabled()
    log.disable_progress_bar()
    try:
        yield
    finally:
        if was:
            log.enable_progress_bar()


def pretrained(ctx: Context, config: int = 0, weights: int = 1) -> torch.nn.Module:
    """`transformers.<ctx.params.arch>` with the variant's `config.json` (file `config`) and weights (file `weights`).

    CPU, float32, eval mode. transformers converts old checkpoint layouts on load (e.g. SegFormer's
    `encoder.block.*` keys), so this goes through `from_pretrained` with an in-memory state dict.

    Raises:
        BackendUnavailableError: transformers is not installed, has no such class, or misses a library to build it.
    """
    tf = transformers()
    cls = getattr(tf, ctx.params.arch, None)
    if cls is None:
        msg = f"transformers {tf.__version__} has no {ctx.params.arch}"
        raise BackendUnavailableError(msg)
    paths = ctx.weights()
    cfg = cls.config_class.from_json_file(str(paths[config]))
    sd = ctx.state_dict(weights, strip="")
    try:
        with _quiet(tf):
            net, info = cls.from_pretrained(
                None, config=cfg, state_dict=sd, output_loading_info=True, ignore_mismatched_sizes=True
            )
    except ImportError as e:  # e.g. Mask2Former / OneFormer build their training loss with scipy
        msg = f"transformers cannot build {ctx.params.arch}: {e} ({install_hint('transformers')})"
        raise MissingDependencyError(msg) from e
    missing = sorted(k for k in info.get("missing_keys", ()) if not _UNUSED_MISSING.fullmatch(k))
    unexpected = sorted(k for k in info.get("unexpected_keys", ()) if not _UNUSED_UNEXPECTED.fullmatch(k))
    shapes = [(k, str(list(a)), str(list(b))) for k, a, b in info.get("mismatched_keys", ())]
    if missing or unexpected or shapes:  # transformers only warns about these (and re-initialises reshaped tensors)
        what = f"{ctx.variant.ref} checkpoint"
        raise CheckpointMismatchError(what, mismatch_details(missing, unexpected, shapes))
    return net.float().eval().requires_grad_(requires_grad=False)


def target_size(h: int, w: int, rule: Sequence[Any]) -> tuple[int, int]:
    """Network input size of an `h x w` image.

    Args:
        h: Image height.
        w: Image width.
        rule: `("mmseg", long, short)`: largest scale that fits `long x short` (mmcv `rescale` with `keep_ratio`);
            `("short", short, max_long)`: shorter side to `short`, unless the longer side then exceeds `max_long`
            (detectron2 `ResizeShortestEdge`). Sizes are rounded as both do (`int(x + 0.5)`);
            `("eomt", short)`: shorter side to `short`, sizes rounded with Python `round` (EoMT
            `scale_img_size_semantic`).
    """
    kind = rule[0]
    if kind == "mmseg":
        s = min(rule[1] / max(h, w), rule[2] / min(h, w))
    elif kind == "short":
        s = rule[1] / min(h, w)
        if max(h, w) * s > rule[2]:
            s = rule[2] / max(h, w)
    elif kind == "eomt":
        s = rule[1] / min(h, w)
        return round(h * s), round(w * s)
    else:
        msg = f"unknown resize rule {kind!r}"
        raise ValueError(msg)
    return int(h * s + 0.5), int(w * s + 0.5)


def resize(x: torch.Tensor, size: Sequence[int], *, pil: bool = False) -> torch.Tensor:
    """Bilinear resize with half-pixel centres, no-op at equal size.

    Default: OpenCV `INTER_LINEAR` (mmseg). `pil=True`: PIL `BILINEAR` (detectron2 `ResizeShortestEdge`, EoMT), which
    widens the filter when downscaling (torch `antialias`); upscaling is the same in both.
    """
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    down = size[0] < x.shape[-2] or size[1] < x.shape[-1]
    return F.interpolate(x, tuple(size), mode="bilinear", align_corners=False, antialias=pil and down)


# measured: float32 elements per class chunk of `upsample_argmax` (256 MB; fp16 scores take twice the elements in the
# same bytes). The chunks are balanced (1024 x 2048, 150 fp16 classes: 3 x 50, not 2 x 64 + 22) and merged once at the
# end from per-chunk max values + uint8 indices: SegFormer-b0 1024 x 2048 6.83 ms / 262 MB transient vs 6.87 / 340 with
# unbalanced chunks and a running max, 2^25 7.12 / 174, 2^24 7.52 / 134 (each chunk ~0.15 ms more)
_ARGMAX_CHUNK = 1 << 26


def _fused(scores: torch.Tensor, sizes: Sequence[Sequence[int]]) -> bool:
    """`seg_up` covers the resize: fp16 / fp32 CUDA scores, one step, or two whose first does not shrink."""
    if len(sizes) > 1 and (scores.shape[-2] > sizes[0][0] or scores.shape[-1] > sizes[0][1]):
        note_fallback("seg_up", "a first resize step that shrinks")
        return False
    if not scores.is_cuda or scores.dtype not in {torch.float16, torch.float32} or len(sizes) not in {1, 2}:
        note_fallback("seg_up", f"{scores.dtype} {scores.device.type} scores, {len(sizes)} resize steps")
        return False
    return True


# measured: resize steps for which channels-last scores take the per-pixel argmax kernel (`up_argmax_p*`) instead of the
# tiled / rows ones (fp16, 150 classes, 1024 x 2048: one step 0.58-0.67 vs 0.89-1.03 ms; two steps from 128 x 256 via
# 512 x 1024 (SegFormer, mmseg) 1.29 vs 0.94 ms rows)
_PIX_ARGMAX = {1}


def _launch(scores: torch.Tensor, sizes: Sequence[Sequence[int]], *, argmax: bool) -> torch.Tensor:
    from fastforks.kernels import seg_up  # noqa: PLC0415 - kernels backend only

    b, c, h, w = scores.shape
    if len(sizes) > 1 and tuple(sizes[0]) == tuple(sizes[1]):  # an identity second step: one step, the same values
        sizes = sizes[1:]
    (m, n), (oh, ow) = sizes[0], sizes[-1]
    shape = (b, oh, ow) if argmax else (b, c, oh, ow)
    out = torch.empty(shape, device=scores.device, dtype=torch.int64 if argmax else torch.float32)
    dt = "h" if scores.dtype == torch.float16 else "f"
    args = (scores, out, b, c, h, w, m, n, oh, ow, len(sizes), *scores.stride())
    sb, _, sy, sx = scores.stride()
    pairs = scores.stride(1) == 1 and (sb | sy | sx) % 2 == 0 and scores.data_ptr() % (2 * scores.element_size()) == 0
    if not argmax:  # one thread per pixel
        grid = kernels.linear(b * oh * ow)
        name = "up_" + dt
    elif pairs and len(sizes) in _PIX_ARGMAX:  # channels-last: one thread per pixel, class pairs from global memory
        grid = kernels.linear(b * oh * ow)
        name = "up_argmax_p" + dt
    elif len(sizes) > 1:  # two real steps: one 128-thread block per 128 x 8 tile
        grid = kernels.linear(b * -(-oh // 8) * -(-ow // 128) * 128, 128)
        name = "up_argmax_r" + dt
    else:  # one 256-thread block per 32 x 8 output tile (seg_up.cu)
        grid = kernels.linear(b * -(-oh // 8) * -(-ow // 32) * 256)
        name = "up_argmax_" + dt
    seg_up.load(scores.device).launch(name, *grid, *args)
    return out


def upsample(scores: torch.Tensor, sizes: Sequence[Sequence[int]], *, fused: bool = False) -> torch.Tensor:
    """`scores` resized through `sizes` (the first step `F.interpolate`, the later ones `resize`).

    Args:
        scores: Low-resolution logits `[B, C, h, w]` (any strides).
        sizes: The resize steps, the last one the output size `(H, W)`.
        fused: Kernels backends: the `seg_up` kernel, float32 math in one pass; `upsample_argmax(..., fused=True)` is
            its argmax up to near-ties (see `upsample_argmax`).

    Returns:
        Logits `[B, C, H, W]` (float32 if fused, else in the dtype of `scores`).
    """
    if fused and _fused(scores, sizes):
        return _launch(scores, sizes, argmax=False)
    y = F.interpolate(scores, tuple(sizes[0]), mode="bilinear", align_corners=False)
    for size in sizes[1:]:
        y = resize(y, size)
    return y


def upsample_argmax(scores: torch.Tensor, sizes: Sequence[Sequence[int]], *, fused: bool = False) -> torch.Tensor:
    """Per-pixel argmax of `scores` resized through `sizes`, without the full-resolution scores.

    The argmax of the resized `scores` up to near-ties: a pixel whose two best classes differ by about the rounding of
    the resize (a few ulp of its dtype; the fused kernel: of float32) may take either. Bilinear resizing treats every
    class on its own, so class chunks resize to the same values; exact ties keep the first index and NaN wins, as
    `argmax`. At most `_ARGMAX_CHUNK` float32-sized resized elements are alive (plus the per-chunk max values and
    indices). Plain PyTorch ops; the first step always resamples (`F.interpolate`, also at equal size), the later ones
    are `resize` (no-op at equal size).

    Args:
        scores: Low-resolution logits `[B, C, h, w]` (the resize runs in their dtype; NCHW-contiguous resizes 1.6x
            faster than channels-last).
        sizes: The `resize` steps, the last one the output size `(H, W)`.
        fused: Kernels backends: the `seg_up` kernel (one pass, no resized scores at all), the argmax of
            `upsample(scores, sizes, fused=True)` up to near-ties.

    Returns:
        Labels `[B, H, W]` int64.
    """
    return _argmax(scores, sizes, fused=fused)


# under `Model.compile()` eager: Inductor fuses each resize into the class max, 3x slower (segformer, 2026-11-06)
@eager_in_compile
def _argmax(scores: torch.Tensor, sizes: Sequence[Sequence[int]], *, fused: bool) -> torch.Tensor:
    if fused and _fused(scores, sizes):
        return _launch(scores, sizes, argmax=True)
    b, c = scores.shape[:2]
    step = max(1, _ARGMAX_CHUNK * 4 // (scores.element_size() * b * max(int(h) * int(w) for h, w in sizes)))
    step = -(-c // -(-c // step))  # the same chunk count, balanced
    vals, idx = [], []
    for c0 in range(0, c, step):
        y = F.interpolate(scores[:, c0 : c0 + step], tuple(sizes[0]), mode="bilinear", align_corners=False)
        for size in sizes[1:]:
            y = resize(y, size)
        if step >= c:
            return y.argmax(1)
        v, i = y.max(1)
        del y
        vals.append(v)
        idx.append(i.to(torch.uint8 if step <= torch.iinfo(torch.uint8).max + 1 else torch.int32))
        del i
    k = torch.stack(vals).argmax(0)  # the first chunk with the largest max (NaN wins), as `argmax`
    return (k * step).add_(torch.stack(idx).gather(0, k[None])[0])


def pad_bottom_right(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Zero-pad height and width up to multiples of `multiple` (below and on the right, as detectron2 batching)."""
    ph, pw = (-x.shape[-2]) % multiple, (-x.shape[-1]) % multiple
    return F.pad(x, (0, pw, 0, ph)) if ph or pw else x


def slide(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    crop: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The mmseg `slide_inference`: logits of overlapping `crop` windows (at `stride`) averaged, at `x`'s size.

    `fn` maps a crop `[B, 3, h, w]` to logits `[B, C, h, w]` at the crop's size. Windows are batched per row; the
    sums and the result are in `dtype`. One window covering `x` returns `fn(x)` itself (its average; no sum buffers).
    """
    (ch, cw), (sh, sw) = crop, stride
    b, _, h, w = x.shape
    rows = max(h - ch + sh - 1, 0) // sh + 1
    cols = max(w - cw + sw - 1, 0) // sw + 1
    if rows == cols == 1:
        return fn(x).to(dtype)
    out: torch.Tensor | None = None
    count = torch.zeros((1, 1, h, w), device=x.device, dtype=dtype)
    for i in range(rows):
        boxes = []
        for j in range(cols):
            y1, x1 = i * sh, j * sw
            y2, x2 = min(y1 + ch, h), min(x1 + cw, w)
            boxes.append((max(y2 - ch, 0), y2, max(x2 - cw, 0), x2))
        crops = torch.cat([x[:, :, y1:y2, x1:x2] for y1, y2, x1, x2 in boxes]) if len(boxes) > 1 else None
        if crops is not None and len({(y2 - y1, x2 - x1) for y1, y2, x1, x2 in boxes}) == 1:
            logits = fn(crops).to(dtype).split(b)
        else:
            logits = [fn(x[:, :, y1:y2, x1:x2]).to(dtype) for y1, y2, x1, x2 in boxes]
        for (y1, y2, x1, x2), s in zip(boxes, logits, strict=True):
            if out is None:
                out = torch.zeros((b, s.shape[1], h, w), device=x.device, dtype=dtype)
            out[:, :, y1:y2, x1:x2] += s
            count[:, :, y1:y2, x1:x2] += 1
    assert out is not None  # noqa: S101 - at least one window
    return out / count


def eomt_windows(length: int, crop: int) -> list[int]:
    """EoMT window starts along one axis: `ceil(length / crop)` windows, overlap spread evenly (`int` of the start)."""
    if length <= crop:
        return [0]
    n = -(-length // crop)
    overlap = n * crop - length
    step = crop - (overlap / (n - 1) if overlap > 0 else 0)
    return [int(j * step) for j in range(n)]


def mask_semantic(
    mask_logits: torch.Tensor, class_logits: torch.Tensor, dtype: torch.dtype | None = torch.float32
) -> torch.Tensor:
    """Semantic logits `[B, C, h, w]` from mask logits `[B, Q, h, w]` and class logits `[B, Q, C + 1]`.

    The product runs in the dtype of `mask_logits`; the result is in `dtype` (`None`: that dtype, no copy).
    """
    probs = class_logits.float().softmax(-1)[..., :-1]
    out = torch.einsum("bqc,bqhw->bchw", probs.to(mask_logits.dtype), mask_logits.sigmoid())
    return out if dtype is None else out.to(dtype)
