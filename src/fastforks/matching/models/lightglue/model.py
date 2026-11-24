"""LightGlue runners: weights -> extractor + matcher, and the portable forward pass.

Upstream `match_pair` with `extractor.extract`: each image is resized so its longest side is `long_side` (kornia
`resize`: Gaussian blur before downscaling, then bilinear), features are detected there (ALIKED: score threshold, at
most `max_keypoints` by score; DISK: NMS 5, the best `max_keypoints` heatmap maxima), their keypoints are mapped back
to the input image (pixel centres at integers) and LightGlue matches them with upstream's defaults (adaptive depth and
width, match threshold 0.1). Returns both keypoint sets, the index pairs of the matches and their scores.

The LightGlue attention runs in float16 on CUDA, as upstream's (`flash=True`: SDPA on half q, k, v); the rest is
float32. `dtype="float16"` (CUDA, not the default) also runs the extractor's dense network (ALIKED's encoder and
aggregation, DISK's U-Net) under float16 autocast; detection, sub-pixel refinement, descriptors and the matcher stay
as in float32 (ALIKED 18.2 -> 14.5 ms, DISK 48.7 -> 38.1 ms per 1024 px image on an RTX 5060 Ti; DISK keypoints
99.5 % identical, ALIKED 99.8 % within 0.5 px).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks._core.mlx_cast import float32_call, to_float32, to_run_dtype
from fastforks._core.runner import host
from fastforks._core.upload import device_tensor
from fastforks.matching._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.aliked import ALIKED, lazy
from fastforks.shared.deform_conv import modulated_deform_conv
from fastforks.shared.lightglue import LightGlue, match_many, resize_long

from .net.disk import DISK

if TYPE_CHECKING:
    from fastforks.matching._runner import Context
    from fastforks.shared.cuda_graphs import GraphCache

_DISK_NMS = 5  # fixed: upstream DISK extractor `nms_window_size`
_DISK_PAD = 16  # fixed: the U-Net's stride (`DISK.forward`'s `pad_if_not_divisible`)
_ALIKED_NMS = 2  # fixed: upstream ALIKED `nms_radius`


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    extractor: str  # "aliked" (aliked-n16) or "disk" (depth)
    detection_threshold: float  # upstream extractor default (ALIKED 0.2, DISK 0.0)
    max_keypoints: int  # per image
    long_side: int  # longest side the features are detected at; 0 = the input size
    cuda_graphs: bool  # kernels, ALIKED: replay the dense maps from a CUDA graph per input size


class Matcher(nn.Module):
    """Extractor + LightGlue: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        p = params
        if p.extractor == "aliked":
            self.extractor = ALIKED(
                modulated_deform_conv,
                model_name="aliked-n16",
                max_num_keypoints=p.max_keypoints,
                detection_threshold=p.detection_threshold,
                nms_radius=_ALIKED_NMS,
            )
        else:
            self.extractor = DISK(desc_dim=128)
        self.matcher = LightGlue(features=p.extractor)

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `Extractor.extract`: features at `long_side`, keypoints in input pixels."""
        p = self.params
        h, w = image.shape[-2:]
        x = resize_long(image, p.long_side) if p.long_side > 0 else image
        scale = device_tensor([x.shape[-1] / w, x.shape[-2] / h], device=x.device, dtype=x.dtype)
        if p.extractor == "aliked":
            feats = self.extractor({"image": x})
            kpts, desc = feats["keypoints"], feats["descriptors"]
        else:
            ((kp, _, desc),) = self.extractor(
                x,
                n=p.max_keypoints,
                window_size=_DISK_NMS,
                score_threshold=p.detection_threshold,
                pad_if_not_divisible=True,
            )
            kpts, desc = kp[None].to(x), desc[None]
        return {
            "keypoints": (kpts + 0.5) / scale - 0.5,
            "descriptors": desc.contiguous(),
            "image_size": device_tensor([[w, h]], device=x.device, dtype=x.dtype),
        }

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        return self.match(self.extract(image1), self.extract(image2))

    def match(self, f1: dict[str, torch.Tensor], f2: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """LightGlue on the two images' `extract` results."""
        out = self.matcher({"image0": f1, "image1": f2})
        return {
            "keypoints1": f1["keypoints"][0],
            "keypoints2": f2["keypoints"][0],
            "pairs": out["matches"][0],
            "scores": out["scores"][0],
        }


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (file 0: LightGlue, file 1: the extractor), eval, on `ctx.device`."""
    net = Matcher(ctx.params)
    net.matcher.load_weights(ctx.state_dict(0), f"{ctx.variant.ref} checkpoint")
    sd = ctx.state_dict(1, key="extractor" if ctx.params.extractor == "disk" else None)
    load_strict(net.extractor, sd, f"{ctx.variant.ref} checkpoint")
    net = net.eval().requires_grad_(requires_grad=False).to(ctx.device)
    if ctx.torch_dtype == torch.float16:
        net.extractor.fp16 = True
        if ctx.params.extractor == "aliked":  # channels-last convs: 15.5 -> 14.5 ms (DISK: no gain)
            net.extractor.to(memory_format=torch.channels_last)
    return net


class Portable(Runner):
    """Plain PyTorch, float32 (LightGlue's attention in float16 on CUDA, as upstream); float16: extractor autocast."""

    Params = Params
    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores."""
        return self.net(image1, image2)

    match_reads_images = False
    matches_many = True

    def describe(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The image's keypoints and descriptors: the per-image stage."""
        return self.net.extract(image)

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Any, Any]) -> dict[str, torch.Tensor]:
        """`__call__` with the two images' `describe` results (the images are not read again)."""
        del image1, image2
        return self.net.match(*described)

    @host
    def match_many(self, described: list[tuple[Any, Any]]) -> list[dict[str, torch.Tensor]]:
        """`match` of many pairs in one packed LightGlue forward (`fastforks.shared.lightglue.match_many`)."""
        outs = match_many(self.net.matcher, described)
        return [
            {
                "keypoints1": f1["keypoints"][0],
                "keypoints2": f2["keypoints"][0],
                "pairs": o["matches"],
                "scores": o["scores"],
            }
            for (f1, f2), o in zip(described, outs, strict=True)
        ]

    def module(self) -> torch.nn.Module:
        """The extractor and matcher (data-dependent keypoint counts: compiles with graph breaks, does not export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


class Kernels(Portable):
    """Portable with DISK's U-Net on fp16 NHWC `conv_ep` convolutions; ALIKED's dense network in fp16.

    ALIKED's dense maps (`detection_maps`, fixed shape: about 260 launches) replay from a CUDA graph per input size
    (`cuda_graphs`): 16.8 -> 12.6 ms per 768 x 1024 image on an RTX 5060 Ti under Windows, bit-equal. The full-size
    descriptor map is not formed (`fastforks.shared.aliked.lazy`: the score head per feature level, the descriptors
    from the pixels SDDH reads). The graph's static outputs are handed to the keypoint selection and descriptors
    without a copy, so one extraction runs at a time (`_extract_lock`) and none of its outputs is a view of them.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the networks and switch the extractor to its kernels path."""
        super().__init__(ctx)
        self.graphs: GraphCache | None = None
        if ctx.params.extractor == "disk":
            from .kernels.inference import prepare_disk  # noqa: PLC0415 - needs the kernel libraries

            disk = self.net.extractor
            assert isinstance(disk, DISK)  # noqa: S101 - narrows the union for the type checker
            disk.fp16 = False
            prepare_disk(disk, ctx.device)
        elif torch.device(ctx.device).type == "cuda":
            aliked = self.net.extractor
            assert isinstance(aliked, ALIKED)  # noqa: S101 - narrows the union for the type checker
            lazy.install(aliked)
            if ctx.params.cuda_graphs:
                self.graphs = lazy.graph_maps(aliked)
                self._extract_lock = threading.Lock()

    def describe(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """`Portable.describe`, one at a time while the dense maps' graph outputs are read without a copy."""
        if self.graphs is None:
            return super().describe(image)
        with self._extract_lock:
            return super().describe(image)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """`Portable.__call__` through `describe`."""
        return self.net.match(self.describe(image1), self.describe(image2))

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        if self.graphs is not None:
            self.graphs.release()

    def module(self) -> torch.nn.Module:
        """No exportable module: the extractor runs kernel launches."""
        return Runner.module(self)


class _SplitHead(nn.Module):
    """DISK's U-Net for the MLX graph: `(heatmap, descriptors)` from a last conv split in two.

    The last 5 x 5 conv (80 -> 128 descriptors + 1 heatmap at full resolution) is four fifths of the U-Net's
    arithmetic, and 129 outputs run on MLX's GEMM padded to 192. Split, the descriptors are an unpadded 128-output
    conv in the run dtype and the heatmap a 1-output conv in float32. Everything before (the heatmap's inputs) stays
    float32 in a float16 replay: a float16 U-Net moves 0.8 % of the heatmap's maxima (recall 0.9917 on the golden
    pair), float16-rounded weights alone 0.07 %.
    """

    def __init__(self, unet: nn.Module) -> None:
        """Wrap `unet` (`net.disk.Unet`)."""
        super().__init__()
        self.unet = unet

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 heatmap `[B, 1, H, W]` and run-dtype dense descriptors `[B, 128, H, W]`."""
        unet = self.unet
        feats = [image]
        for layer in unet.path_down:
            feats.append(float32_call(layer, feats[-1]))
        bottom, skips = feats[-1], feats[-2::-1]
        for layer, skip in zip(unet.path_up[:-1], skips, strict=False):
            bottom = float32_call(layer, bottom, skip)
        last = unet.path_up[-1]
        norm, gate, _, conv = last.conv
        x = float32_call(gate, norm(torch.cat([last.upsample(bottom), skips[len(unet.path_up) - 1]], 1)))
        n = conv.out_channels - 1
        heat = F.conv2d(to_float32(x), to_float32(conv.weight)[n:], to_float32(conv.bias)[n:], padding="same")
        desc = F.conv2d(to_run_dtype(x, conv.weight.dtype), conv.weight[:n], conv.bias[:n], padding="same")
        return heat, desc


class _MlxMatcher(Portable):
    """`Portable` for MLX: keypoint selection, descriptors and LightGlue in MLX operations.

    A subclass queues an image's dense maps (`_maps`) and selects its features from them (`_detect`: one host read
    per image; the descriptors stay on MLX). `__call__` and `match` run the shared `MlxLightGlue`; `describe` hands
    the features out as CPU tensors (the cached per-image stage); `match_many` is `Portable`'s packed forward on the
    CPU. The results equal `Portable`'s up to float32 rounding.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch networks and copy the matcher's weights to MLX."""
        super().__init__(ctx)
        from fastforks.shared.lightglue.mlx import MlxLightGlue  # noqa: PLC0415 - imports mlx

        self._glue = MlxLightGlue(self.net.matcher)

    def _maps(self, x: torch.Tensor) -> Any:  # noqa: ANN401 - MLX arrays
        """Queue the dense maps of the float32 detection image `x` on MLX."""
        raise NotImplementedError

    def _detect(self, maps: Any) -> tuple[np.ndarray, Any]:  # noqa: ANN401 - MLX arrays
        """Keypoints `[N, 2]` (`(x, y)` pixels of the detection image) and their MLX descriptors `[N, D]`."""
        raise NotImplementedError

    def _queue(self, image: torch.Tensor) -> tuple[Any, tuple[int, ...]]:
        """An image's queued maps, with the `(width, height)` of the input and of the detection image."""
        p = self.net.params
        x = resize_long(image.float(), p.long_side) if p.long_side > 0 else image.float()
        return self._maps(x), (image.shape[-1], image.shape[-2], x.shape[-1], x.shape[-2])

    def _features(self, maps: Any, sizes: tuple[int, ...]) -> tuple[np.ndarray, Any, np.ndarray]:  # noqa: ANN401 - MLX
        """Keypoints in input pixels (`Matcher.extract`'s), their MLX descriptors and the input's `(width, height)`."""
        kpts, desc = self._detect(maps)
        w, h, dw, dh = sizes
        scale = np.array([dw / w, dh / h], np.float32)
        kpts = (kpts.astype(np.float32) + np.float32(0.5)) / scale - np.float32(0.5)
        return kpts, desc, np.array([w, h], np.float32)

    def _result(self, f1: Any, f2: Any) -> dict[str, torch.Tensor]:  # noqa: ANN401 - `_features` results
        pairs, scores = self._glue(f1, f2)
        return {
            "keypoints1": torch.from_numpy(f1[0]),
            "keypoints2": torch.from_numpy(f2[0]),
            "pairs": torch.from_numpy(pairs),
            "scores": torch.from_numpy(scores),
        }

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores (CPU tensors)."""
        # both networks are queued first: the second runs while the host selects the first image's keypoints
        q1, q2 = self._queue(image1), self._queue(image2)
        return self._result(self._features(*q1), self._features(*q2))

    def describe(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The image's keypoints and descriptors as CPU tensors (`Matcher.extract`'s dictionary)."""
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - imports mlx

        kpts, desc, size = self._features(*self._queue(image))
        return {
            "keypoints": torch.from_numpy(kpts)[None],
            "descriptors": to_torch(desc)[None],
            "image_size": torch.from_numpy(size)[None],
        }

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Any, Any]) -> dict[str, torch.Tensor]:
        """`__call__` with the two images' `describe` results (the images are not read again)."""
        from fastforks._core.mlx import to_mx  # noqa: PLC0415 - imports mlx

        del image1, image2
        f1, f2 = (
            (f["keypoints"][0].numpy(), to_mx(f["descriptors"][0]), f["image_size"][0].numpy()) for f in described
        )
        return self._result(f1, f2)


class _MlxDisk(_MlxMatcher):
    """DISK: the U-Net behind `_SplitHead` as an exported graph; NMS, top-n and the descriptor lookup in `mlx_post`.

    The U-Net's two maps stay on MLX (`MlxModule.raw`): the heatmap's candidates are one host read, the dense
    descriptors (400 MB at 768 x 1024 in float32) are never copied to the host.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the networks and wrap the U-Net."""
        super().__init__(ctx)
        self.net.extractor.unet = _SplitHead(self.net.extractor.unet)

    def _maps(self, x: torch.Tensor) -> Any:  # noqa: ANN401 - MLX arrays
        from fastforks._core.mlx import mx  # noqa: PLC0415 - imports mlx

        from . import mlx_post  # noqa: PLC0415 - imports mlx

        h, w = x.shape[-2:]
        heat, desc = self.net.extractor.unet.raw(F.pad(x, (0, -w % _DISK_PAD, 0, -h % _DISK_PAD)))
        peak = mlx_post.peaks(heat[0, 0, :h, :w], _DISK_NMS, self.net.params.detection_threshold)
        mx.async_eval(peak)
        return peak, desc[0]

    def _detect(self, maps: Any) -> tuple[np.ndarray, Any]:  # noqa: ANN401 - MLX arrays
        from . import mlx_post  # noqa: PLC0415 - imports mlx

        return mlx_post.detect(*maps, self.net.params.max_keypoints)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """`_MlxMatcher.__call__`, one image after the other.

        The U-Net is all of an image's time, so queueing both gains nothing and holds two sets of dense maps: the
        first image's descriptors are gathered, and its maps released, before the second U-Net runs.
        """
        from fastforks._core.mlx import mx  # noqa: PLC0415 - imports mlx

        f1 = self._features(*self._queue(image1))
        mx.eval(f1[1])
        return self._result(f1, self._features(*self._queue(image2)))


class _MlxAliked(_MlxMatcher):
    """ALIKED wholly on MLX (`fastforks.shared.aliked.mlx`), float32 in either dtype.

    Dense maps without the full-resolution descriptor map, keypoints from one host read per image, SDDH on MLX.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch networks and copy their weights to MLX."""
        super().__init__(ctx)
        from fastforks.shared.aliked.mlx import MlxAliked  # noqa: PLC0415 - imports mlx

        extractor = self.net.extractor
        assert isinstance(extractor, ALIKED)  # noqa: S101 - narrows the union for the type checker
        self._aliked = MlxAliked(extractor)

    def _maps(self, x: torch.Tensor) -> Any:  # noqa: ANN401 - MLX arrays
        from fastforks._core.mlx import to_mx  # noqa: PLC0415 - imports mlx

        return self._aliked.maps(to_mx(x))

    def _detect(self, maps: Any) -> tuple[np.ndarray, Any]:  # noqa: ANN401 - MLX arrays
        return self._aliked(maps)


_MLX = {
    "disk": mlx_wrap.port(_MlxDisk, attrs=("net.extractor.unet",), fp32_inputs=("image",)),
    "aliked": mlx_wrap.port(_MlxAliked, attrs=()),
}


class MlxRunner(Portable):
    """Apple MLX: DISK's U-Net as an exported graph (`_MlxDisk`) or ALIKED in MLX operations (`_MlxAliked`).

    Keypoint selection reads one map per image on the host; descriptors and LightGlue run on MLX (`_MlxMatcher`).
    """

    def __new__(cls, ctx: Context) -> Runner:  # noqa: D102
        return _MLX[ctx.params.extractor](ctx)


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": MlxRunner}
