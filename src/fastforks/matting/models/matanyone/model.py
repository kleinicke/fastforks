"""MatAnyone runners: the network of pq-yang/MatAnyone and `InferenceCore.process_video`'s loop.

`process_video` closes the first-frame mask (`gen_dilate` then `gen_erosion`: cv2 dilate / erode with a 10 x 10
elliptical kernel; `close_mask` here, the same pixels), prepends 10 copies of frame 0 (warm-up: each predicts as a
first frame and replaces the memory), then steps through the video with the memory of `InferenceCore` (one object,
working memory every 5th frame, top-30 readout); the alpha of each real frame is the matting channel of the step's
output. Upstream runs it under CUDA fp16 autocast with fp32 regions inside; `portable` float16 does the same, float32
runs without autocast. An image is a one-frame video (with the warm-up).

The memory readout runs in query chunks (`_read`): upstream's full similarity maps (several per frame, 1 GB each at
1080p with five memory frames) never exist, and the allocator keeps no growing set of them.

Upstream's `encode_image` moves the normalisation buffers to the default device (CUDA if there is one) on every call;
here it uses the buffers where the network is (`_encode_image`). The network is built without ImageNet weights
(upstream's `pretrained_resnet` default downloads them before the checkpoint replaces them).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.patch import override
from fastforks._core.runner import host
from fastforks._core.upload import to_device
from fastforks.kernels import image_normalize
from fastforks.matting._runner import Runner

from .net.inference.image_feature_store import ImageFeatureStore
from .net.inference.inference_core import InferenceCore
from .net.inference.memory_manager import MemoryManager
from .net.model.matanyone import MatAnyone

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]` (`process_video` arguments).

    Attributes:
        warmup: Copies of frame 0 run first (`n_warmup`).
        dilate: Mask dilation kernel side in pixels (`r_dilate`).
        erode: Mask erosion kernel side in pixels after the dilation (`r_erode`).
    """

    warmup: int
    dilate: int
    erode: int


_PE3 = {"add_pe_to_qkv": [True, True, False]}
CONFIG = {  # eval_matanyone_config.yaml + model/base.yaml at the pinned commit (inference keys)
    "mem_every": 5, "max_mem_frames": 5, "use_long_term": False, "top_k": 30, "stagger_updates": 5,
    "chunk_size": -1, "save_aux": False, "max_internal_size": -1, "flip_aug": False,
    "long_term": {
        "count_usage": True, "max_mem_frames": 10, "min_mem_frames": 5, "num_prototypes": 128,
        "max_num_tokens": 10000, "buffer_tokens": 2000,
    },
    "model": {
        "pretrained_resnet": False,  # fastforks: upstream's getattr default True downloads ImageNet weights
        "pixel_mean": [0.485, 0.456, 0.406], "pixel_std": [0.229, 0.224, 0.225],
        "pixel_dim": 256, "key_dim": 64, "value_dim": 256, "sensory_dim": 256, "embed_dim": 256,
        "pixel_encoder": {"type": "resnet50", "ms_dims": [1024, 512, 256, 64, 3]},
        "mask_encoder": {"type": "resnet18", "final_dim": 256},
        "pixel_pe_scale": 32, "pixel_pe_temperature": 128,
        "object_transformer": {
            "embed_dim": 256, "ff_dim": 2048, "num_heads": 8, "num_blocks": 3, "num_queries": 16,
            "read_from_pixel": {"input_norm": False, "input_add_pe": False, **_PE3},
            "read_from_past": _PE3, "read_from_memory": _PE3,
            "read_from_query": {**_PE3, "output_norm": False},
            "query_self_attention": _PE3, "pixel_self_attention": _PE3,
        },
        "object_summarizer": {"embed_dim": 256, "num_summaries": 16, "add_pe": True},
        "aux_loss": {"sensory": {"enabled": True, "weight": 0.01}, "query": {"enabled": True, "weight": 0.01}},
        "mask_decoder": {"up_dims": [256, 128, 128, 64, 16]},
    },
}  # fmt: skip


class _Config(dict):
    """The OmegaConf config upstream reads: keys as attributes (a missing one is an `AttributeError`) and items."""

    def __getattr__(self, key: str) -> Any:  # noqa: ANN401 - config values
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _config(d: dict[str, Any]) -> _Config:
    return _Config({k: _config(v) if isinstance(v, dict) else v for k, v in d.items()})


def _ellipse(k: int, device: torch.device) -> torch.Tensor:
    """`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))` as a `[1, 1, k, k]` float kernel."""
    kernel = torch.zeros(k, k)
    r = c = k // 2
    for i in range(k):
        dy = i - r
        if abs(dy) <= r:
            dx = round(c * math.sqrt((r * r - dy * dy) / (r * r)))
            kernel[i, max(c - dx, 0) : min(c + dx + 1, k)] = 1
    return to_device(kernel, device)[None, None]


def close_mask(mask: torch.Tensor, dilate: int, erode: int) -> torch.Tensor:
    """`gen_erosion(gen_dilate(mask, dilate), erode)`: `[H, W]` nonzero = object -> `[H, W]` float 0 / 255.

    cv2 dilate / erode with an elliptical kernel anchored at its centre `k // 2`; the border counts as background
    for the dilation and as foreground for the erosion (cv2's default border value).
    """
    x = mask.ne(0).float()[None, None]
    if dilate > 0:
        k, a = _ellipse(dilate, x.device), dilate // 2
        x = F.conv2d(F.pad(x, (a, dilate - 1 - a, a, dilate - 1 - a)), k).gt(0).float()
    if erode > 0:
        k, a = _ellipse(erode, x.device), erode // 2
        x = F.conv2d(F.pad(x, (a, erode - 1 - a, a, erode - 1 - a), value=1.0), k).eq(k.sum()).float()
    return x[0, 0] * 255


def _encode_image(
    m: MatAnyone,
    image: torch.Tensor,
    seq_length: int | None = None,
    last_feats: Any = None,  # noqa: ANN401, ARG001 - unused upstream
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """`MatAnyone.encode_image` with the normalisation buffers where they are (upstream: moved per call)."""
    image = (image - m.pixel_mean) / m.pixel_std
    ms_image_feat = m.pixel_encoder(image, seq_length)
    return ms_image_feat, m.pix_feat_proj(ms_image_feat[0])


READ_TOKENS = (
    1 << 11
)  # query pixels per memory-readout chunk (1080p: 8160 queries x 6 memory frames = 0.8 GB per full map)


def _read(  # noqa: PLR0913, PLR0917 - upstream's signature
    mem: MemoryManager,
    pix_feat: torch.Tensor,
    query_key: torch.Tensor,
    selection: torch.Tensor,
    last_mask: torch.Tensor,
    network: MatAnyone,
    uncert_output: Any,  # noqa: ANN401 - upstream's untyped output dict
    *,  # `InferenceCore.step` passes the four below by keyword
    last_msk_value: torch.Tensor,
    ti: int,  # noqa: ARG001 - unused upstream
    last_pix_feat: torch.Tensor,
    last_pred_mask: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """`MemoryManager.read` (working memory, no object chunks, as configured) with the affinity in query chunks.

    Upstream builds the similarity of every query pixel to every memory token (`[1, N, HW]`, several float16
    temporaries of it, 0.8 GB each at 1080p with six memory frames) before the top-k softmax and the value readout.
    Each query column is independent, so the same ops run over `READ_TOKENS` columns at a time: the readout
    `v @ affinity` of the chunk is the readout's column block, and the full maps never exist.
    """
    assert uncert_output is None and not mem.use_long_term  # noqa: S101, PT018 - the inference config
    assert mem.chunk_size < 1  # noqa: S101 - no object chunks
    h, w = pix_feat.shape[-2:]
    bs = pix_feat.shape[0]
    query_key, selection = query_key.flatten(2), selection.flatten(2)
    out = {}
    for bucket_id, bucket in mem.work_mem.buckets.items():
        mk, ms = mem.work_mem.key[bucket_id], mem.work_mem.shrinkage[bucket_id]
        v = mem._get_visual_values_by_ids(bucket).flatten(1, 2)  # noqa: SLF001 - upstream's helpers  # bs x (objects C) x N
        ck, ms_t, vt, mk_sq = mk.shape[1], ms.flatten(1)[:, None], v.transpose(1, 2), mk.pow(2)
        reads = []
        for s in range(0, h * w, READ_TOKENS):  # similarity as [B, chunk, N]: top-k over the last dim is faster
            qk, qe = query_key[:, :, s : s + READ_TOKENS], selection[:, :, s : s + READ_TOKENS]
            sim = (
                (
                    (2 * ((qk * qe).transpose(1, 2) @ mk))
                    - (qe.transpose(1, 2) @ mk_sq)
                    - (qe * qk.pow(2)).sum(1).unsqueeze(2)
                )
                * ms_t
                / math.sqrt(ck)
            )
            val, idx = torch.topk(sim, k=mem.top_k, dim=2)
            val = val.exp_()
            val /= val.sum(2, keepdim=True)
            reads.append(sim.zero_().scatter_(2, idx, val) @ vt)  # [B, chunk, objects C]
        visual_readout = torch.cat(reads, 1).transpose(1, 2).reshape(bs, len(bucket), v.shape[1] // len(bucket), h, w)
        uncert_output = network.pred_uncertainty(
            last_pix_feat, pix_feat, last_pred_mask, visual_readout[:, 0] - last_msk_value[:, 0]
        )
        if uncert_output is not None:
            uncert_prob = uncert_output["prob"].unsqueeze(1)
            visual_readout = visual_readout * uncert_prob + last_msk_value * (1 - uncert_prob)
        pixel_readout = network.pixel_fusion(
            pix_feat,
            visual_readout,
            mem._get_sensory_by_ids(bucket),  # noqa: SLF001
            mem._get_mask_by_ids(last_mask, bucket),  # noqa: SLF001
        )
        readout_memory, _ = network.readout_query(pixel_readout, mem._get_object_mem_by_ids(bucket).unsqueeze(2))  # noqa: SLF001
        for i, obj in enumerate(bucket):
            out[obj] = readout_memory[:, i]
    return out


class _Memory(MemoryManager):
    """`MemoryManager` with the chunked readout."""

    read = _read  # ty: ignore[invalid-method-override] - keyword-only where upstream is untyped


_FrameFeatures = tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _FeatureStore(ImageFeatureStore):
    """Keep the original frame cache while running its tensor stage through the runner."""

    def __init__(self, network: MatAnyone, features: Callable[..., _FrameFeatures]) -> None:
        super().__init__(network)
        self._features = features

    def _encode_feature(self, index: int, image: torch.Tensor, last_feats: Any = None) -> None:  # noqa: ANN401 - upstream
        self._store[index] = self._features(image, last_feats)


class Portable(Runner):
    """Upstream's network and memory in plain PyTorch; float16 = CUDA fp16 autocast with fp32 weights (as upstream)."""

    dtypes = ("float32", "float16")
    Params = Params
    mask = "required"
    video = True
    compiled_methods = ("_frame_features", "_encode_mask", "_segment")

    def __init__(self, ctx: Context) -> None:
        """Build the network (one object) from the checkpoint."""
        super().__init__(ctx)
        self.cfg = _config(CONFIG)
        self.net = MatAnyone(self.cfg, single_object=True)
        load_strict(self.net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
        override(self.net, _encode_image, "encode_image")
        self.net.eval().to(ctx.device)

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The one network (device adaptations, e.g. Apple MPS)."""
        return (self.net,)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.ctx.dtype == "float32":
            return contextlib.nullcontext()
        return torch.autocast(self.ctx.device.type, getattr(torch, self.ctx.dtype))

    def _frame_features(self, image: torch.Tensor, last_feats: Any) -> _FrameFeatures:  # noqa: ANN401 - upstream
        features, pixels = self.net.encode_image(image, last_feats=last_feats)
        key, shrinkage, selection = self.net.transform_key(features[0])
        return features, pixels, key, shrinkage, selection

    def _encode_mask(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - upstream
        return self.net.encode_mask(*args, **kwargs)

    def _segment(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - upstream
        return self.net.segment(*args, **kwargs)

    def _alphas(self, frames: Callable[[int], torch.Tensor], n: int, mask: torch.Tensor) -> Iterator[torch.Tensor]:
        """`process_video`'s loop: the alpha `[H, W]` float32 of each frame, in order."""
        p = self.ctx.params
        closed = close_mask(mask, p.dilate, p.erode)
        store = _FeatureStore(self.net, self._frame_features)
        core = InferenceCore(
            self.net,
            cfg=self.cfg,
            device=self.ctx.device,
            image_feature_store=store,
            encode_mask=self._encode_mask,
            segment=self._segment,
        )
        core.memory.__class__ = _Memory  # the chunked readout (a bound method stored on it would be a reference cycle)
        first = frames(0)[0]
        # First-frame predictions reset the feature-store index to zero throughout warm-up.
        with self._autocast():
            for ti in range(p.warmup + n):
                image = first if ti <= p.warmup else frames(ti - p.warmup)[0]
                if ti == 0:
                    core.step(image, closed, objects=[1], delete_buffer=False)
                    prob = core.step(image, first_frame_pred=True, delete_buffer=p.warmup == 0)
                else:
                    prob = core.step(image, first_frame_pred=ti <= p.warmup, delete_buffer=ti >= p.warmup)
                if ti >= p.warmup:
                    yield core.output_prob_to_mask(prob).float()

    @host
    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Alpha `[B, H, W]` of each image as a one-frame video, `trimap` = the mask `[B, 1, H, W]`."""
        assert trimap is not None  # noqa: S101 - mask = "required": the core passes it
        out = [next(iter(self._alphas(lambda _t, x=x: x[None], 1, m[0]))) for x, m in zip(image, trimap, strict=True)]
        return {"alpha": torch.stack(out)}

    @host
    def track(
        self, frames: Callable[[int], torch.Tensor], n_frames: int, size: tuple[int, int], mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Alpha `[T, H, W]` float32 on the CPU (pinned on CUDA), each frame copied as it is done."""
        cuda = self.ctx.device.type == "cuda"
        out = torch.empty((n_frames, *size), dtype=torch.float32, pin_memory=cuda)
        for t, alpha in enumerate(self._alphas(frames, n_frames, mask)):
            out[t].copy_(alpha, non_blocking=cuda)
        if cuda:
            torch.cuda.synchronize(self.ctx.device)
        return {"alpha": out}


def _cuda_encode(
    m: MatAnyone,
    image: torch.Tensor,
    seq_length: int | None = None,
    last_feats: Any = None,  # noqa: ANN401, ARG001 - upstream signature
) -> tuple[list[torch.Tensor], torch.Tensor]:
    x = image_normalize.normalize(image, m.pixel_mean, m.pixel_std)
    features = m.pixel_encoder(x, seq_length)
    return features, m.pix_feat_proj(features[0])


class Kernels(Portable):
    """Use fused image normalisation before the frame encoder."""

    compiled_methods = ("__call__",)  # keep the kernel checks around the full call
    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Prepare the accelerated inference path on the model device."""
        super().__init__(ctx)
        override(self.net, _cuda_encode, "encode_image")


RUNNERS = {"portable": Portable, "kernels": Kernels}
