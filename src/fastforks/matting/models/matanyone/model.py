"""MatAnyone runners: the network of pq-yang/MatAnyone and `InferenceCore.process_video`'s loop.

`process_video` closes the first-frame mask (`gen_dilate` then `gen_erosion`: cv2 dilate / erode with a 10 x 10
elliptical kernel; `close_mask` here, the same pixels), prepends 10 copies of frame 0 (warm-up: each predicts as a
first frame and replaces the memory), then steps through the video with the memory of `InferenceCore` (one object,
working memory every 5th frame, top-30 readout); the alpha of each real frame is the matting channel of the step's
output. Upstream runs it under CUDA fp16 autocast with fp32 regions inside; `portable` float16 does the same, float32
runs without autocast. An image is a one-frame video (with the warm-up).

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
from fastforks.matting._runner import Runner

from .net.inference.inference_core import InferenceCore
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
    return kernel.to(device)[None, None]


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


class Portable(Runner):
    """Upstream's network and memory in plain PyTorch; float16 = CUDA fp16 autocast with fp32 weights (as upstream)."""

    dtypes = ("float32", "float16")
    Params = Params
    mask = "required"
    video = True

    def __init__(self, ctx: Context) -> None:
        """Build the network (one object) from the checkpoint."""
        super().__init__(ctx)
        self.cfg = _config(CONFIG)
        self.net = MatAnyone(self.cfg, single_object=True)
        load_strict(self.net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
        override(self.net, _encode_image, "encode_image")
        self.net.eval().to(ctx.device)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.ctx.dtype == "float32":
            return contextlib.nullcontext()
        return torch.autocast(self.ctx.device.type, getattr(torch, self.ctx.dtype))

    def _alphas(self, frames: Callable[[int], torch.Tensor], n: int, mask: torch.Tensor) -> Iterator[torch.Tensor]:
        """`process_video`'s loop: the alpha `[H, W]` float32 of each frame, in order."""
        p = self.ctx.params
        closed = close_mask(mask, p.dilate, p.erode)
        core = InferenceCore(self.net, cfg=self.cfg, device=self.ctx.device)
        first = frames(0)[0]
        with self._autocast():
            for ti in range(p.warmup + n):
                image = first if ti <= p.warmup else frames(ti - p.warmup)[0]
                if ti == 0:
                    core.step(image, closed, objects=[1])
                    prob = core.step(image, first_frame_pred=True)
                else:
                    prob = core.step(image, first_frame_pred=ti <= p.warmup)
                if ti >= p.warmup:
                    yield core.output_prob_to_mask(prob).float()

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Alpha `[B, H, W]` of each image as a one-frame video, `trimap` = the mask `[B, 1, H, W]`."""
        assert trimap is not None  # noqa: S101 - mask = "required": the core passes it
        out = [next(iter(self._alphas(lambda _t, x=x: x[None], 1, m[0]))) for x, m in zip(image, trimap, strict=True)]
        return {"alpha": torch.stack(out)}

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


RUNNERS = {"portable": Portable}
