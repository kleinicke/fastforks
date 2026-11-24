"""XFeat runners: weights -> extractor + matcher, and the portable forward pass.

Upstream `XFeat.detectAndCompute` per image (the image is bilinearly resized down to a multiple of 32, keypoints are
the heatmap maxima after a 5x5 NMS above `detection_threshold`, ranked by score, at most `max_keypoints`, mapped back
to the input pixels), then either upstream's mutual nearest neighbours of the descriptors (`match_xfeat`: no
similarity threshold; the score of a match is its cosine similarity) or LighterGlue (`match_lighterglue`: kornia's
LightGlue with XFeat's config = fastforks.shared.lightglue, match threshold 0.1). With `long_side > 0` each image is
first resized so its longest side is `long_side` (upstream LightGlue's resize; upstream XFeat does not resize).
Returns both keypoint sets, the index pairs of the matches and their scores. Float32.

`torch.export` (ONNX) of the mutual-nearest-neighbour variant traces `Matcher.forward_static` instead: the same
arithmetic with every intermediate at a fixed size (the top `max_keypoints` of the whole masked score map, a validity
mask, masked nearest neighbours) and one `nonzero` for each data-dependent output length. It returns what the eager
call returns; only keypoints of exactly equal score may be ranked differently (upstream ranks with an unstable
`argsort`, the trace with `topk`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks._core.patch import override
from fastforks._core.upload import device_tensor
from fastforks.kernels import keypoint_nms
from fastforks.matching._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import CapturedModule
from fastforks.shared.lightglue import LightGlue, resize_long

from .net.interpolator import InterpolateSparse2d
from .net.xfeat import XFeat

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

# upstream modules/lighterglue.py `default_conf_xfeat` (kornia LightGlue; `add_laf`, `scale_coef` are kornia-only, off)
LIGHTERGLUE = {
    "input_dim": 64,
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,
    "mp": False,
    "depth_confidence": -1,
    "width_confidence": 0.95,
    "filter_threshold": 0.1,
}


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    matcher: str  # "mnn" (mutual nearest neighbours) or "lighterglue"
    detection_threshold: float  # upstream default 0.05
    max_keypoints: int  # per image (upstream `top_k`)
    long_side: int  # longest side the features are detected at; 0 = the input size (upstream)


class Matcher(nn.Module):
    """XFeat + matcher: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        self.xfeat = XFeat(top_k=params.max_keypoints, detection_threshold=params.detection_threshold)
        self.matcher = LightGlue(features=None, **LIGHTERGLUE) if params.matcher == "lighterglue" else None

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `detectAndCompute` at `long_side`: keypoints and `image_size` of that resized image."""
        p = self.params
        x = resize_long(image, p.long_side) if p.long_side > 0 else image
        d = self.xfeat.detectAndCompute(x)[0]
        return {
            "keypoints": d["keypoints"],
            "descriptors": d["descriptors"],
            "image_size": torch.tensor(x.shape[-1:-3:-1]),
        }

    @staticmethod
    def to_input(kpts: torch.Tensor, size: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Keypoints of the resized image in input pixels (pixel centres at integers, as LightGlue's `extract`)."""
        return Matcher._to_input(kpts, int(size[0]), int(size[1]), image)

    @staticmethod
    def _to_input(kpts: torch.Tensor, width: int, height: int, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        if (width, height) == (w, h):
            return kpts
        scale = device_tensor([float(width) / w, float(height) / h], device=kpts.device, dtype=kpts.dtype)
        return (kpts + 0.5) / scale - 0.5

    def extract_static(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`extract` without data-dependent shapes: `min(max_keypoints, pixels)` ranked slots and which are keypoints.

        Upstream `detectAndCompute` with the 5x5 NMS as a mask of the score map instead of a position list: every
        pixel is scored (the heatmap times the reliability, with upstream's samplers), a pixel that is no maximum
        above the threshold scores -1, and `topk` ranks them. The slots with a score above 0 are upstream's keypoints
        and come first. Returns the keypoints in input pixels `[K, 2]`, the descriptors `[K, 64]` and the mask `[K]`.
        """
        p, xfeat = self.params, self.xfeat
        resized = resize_long(image, p.long_side) if p.long_side > 0 else image
        x, rh, rw = xfeat.preprocess_tensor(resized)
        height, width = x.shape[-2:]
        dense, logits, reliability = xfeat.net(x)
        dense = F.normalize(dense, dim=1)
        heatmap = xfeat.get_kpts_heatmap(logits)
        peaks = heatmap == F.max_pool2d(heatmap, kernel_size=5, stride=1, padding=2)
        keep = (peaks & (heatmap > p.detection_threshold)).flatten(1)
        keep = keep & (torch.arange(height * width, device=x.device) > 0)  # upstream drops (0, 0), its padding value
        pixels = xfeat.create_xy(height, width, x.device)[None]  # every (x, y), row-major as upstream's `nonzero`
        nearest, bilinear = InterpolateSparse2d("nearest"), InterpolateSparse2d("bilinear")
        scores = nearest(heatmap, pixels, height, width) * bilinear(reliability, pixels, height, width)
        scores = torch.where(keep, scores.squeeze(-1), -1.0)
        top, order = scores.topk(min(p.max_keypoints, height * width), dim=-1)
        kpts = torch.stack([order % width, order // width], dim=-1)
        desc = F.normalize(xfeat.interpolator(dense, kpts, H=height, W=width), dim=-1)
        kpts = kpts * torch.tensor([rw, rh], device=x.device, dtype=torch.get_default_dtype())
        kpts = self._to_input(kpts, resized.shape[-1], resized.shape[-2], image)
        return kpts[0], desc[0], top[0] > 0

    def forward_static(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """`forward` of the mutual-nearest-neighbour matcher as `torch.export` traces it (see the module docstring).

        The valid slots of `extract_static` come first, so an index into the slots is the index into the returned
        keypoints. A slot without a keypoint is nobody's neighbour (similarity -inf) and matches nothing.
        """
        if self.matcher is not None:
            msg = "the fixed-shape forward pass covers mutual nearest neighbours only, not LighterGlue"
            raise NotImplementedError(msg)
        (k1, d1, v1), (k2, d2, v2) = self.extract_static(image1), self.extract_static(image2)
        # two products, as upstream `match`: the transposed one may round differently
        match12 = (d1 @ d2.t()).masked_fill(~v2[None], -torch.inf).argmax(dim=1)
        match21 = (d2 @ d1.t()).masked_fill(~v1[None], -torch.inf).argmax(dim=1)
        mutual = (match21[match12] == torch.arange(len(match12), device=match12.device)) & v1 & v2[match12]
        idx0 = mutual.nonzero()[:, 0]
        idx1 = match12[idx0]
        return {
            "keypoints1": k1[v1.nonzero()[:, 0]],
            "keypoints2": k2[v2.nonzero()[:, 0]],
            "pairs": torch.stack([idx0, idx1], dim=-1),
            "scores": (d1[idx0] * d2[idx1]).sum(-1),
        }

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        if self.matcher is None and torch.compiler.is_exporting():  # LighterGlue's layer pruning is not traced
            return self.forward_static(image1, image2)
        f1, f2 = self.extract(image1), self.extract(image2)
        if self.matcher is None:
            idx0, idx1 = self.xfeat.match(f1["descriptors"], f2["descriptors"], min_cossim=-1)
            pairs = torch.stack([idx0, idx1], dim=-1)
            scores = (f1["descriptors"][idx0] * f2["descriptors"][idx1]).sum(-1)
        else:
            dev = f1["keypoints"].device
            out = self.matcher(
                {
                    f"image{i}": {
                        "keypoints": f["keypoints"][None],
                        "descriptors": f["descriptors"][None],
                        "image_size": f["image_size"].to(dev)[None],
                    }
                    for i, f in enumerate((f1, f2))
                }
            )
            pairs, scores = out["matches"][0], out["scores"][0]
        k1 = self.to_input(f1["keypoints"], f1["image_size"], image1)
        k2 = self.to_input(f2["keypoints"], f2["image_size"], image2)
        return {"keypoints1": k1, "keypoints2": k2, "pairs": pairs, "scores": scores}


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (file 0: XFeat, file 1: LighterGlue), eval, on `ctx.device`."""
    net = Matcher(ctx.params)
    load_strict(net.xfeat.net, ctx.state_dict(0), f"{ctx.variant.ref} checkpoint")
    if net.matcher is not None:  # upstream LighterGlue: drop the `matcher.` prefix, then LightGlue's renames
        sd = ctx.state_dict(1)  # also holds a training-time XFeat copy (`extractor.*`) upstream ignores (strict=False)
        net.matcher.load_weights(
            {k.replace("matcher.", ""): v for k, v in sd.items() if not k.startswith("extractor.")},
            f"{ctx.variant.ref} checkpoint",
        )
    # Folded BatchNorm changes the CUDA keypoint threshold decisions and the measured pose accuracy.
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


class Portable(Runner):
    """Plain PyTorch, float32."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores."""
        return self.net(image1, image2)

    def module(self) -> torch.nn.Module:
        """Extractor and matcher. The keypoint and match counts are data-dependent: `torch.compile` graph-breaks.

        `torch.export` traces the mutual-nearest-neighbour matcher's `Matcher.forward_static` (dynamic output
        lengths); LighterGlue does not export.
        """
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


def _precise_conv(layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    # XFeat's score threshold changes keypoint membership at TF32 convolution precision.
    return torch.ops.aten._convolution.default(  # noqa: SLF001 - per-call precision control
        x,
        layer.weight,
        layer.bias,
        layer.stride,
        layer.padding,
        layer.dilation,
        transposed=False,
        output_padding=[0, 0],
        groups=layer.groups,
        benchmark=False,
        deterministic=False,
        cudnn_enabled=True,
        allow_tf32=False,
    )


def _maxima(_extractor: XFeat, x: torch.Tensor, threshold: float = 0.05, kernel_size: int = 5) -> torch.Tensor:
    peaks = keypoint_nms.maxima(x, threshold, kernel_size // 2)
    positions = [plane[0].nonzero().flip(-1) for plane in peaks]
    return nn.utils.rnn.pad_sequence(positions, batch_first=True)


class Kernels(Portable):
    """Capture the fixed-shape dense extractor and fuse thresholding with local maxima."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Keep descriptors and scores in the original float32 precision."""
        super().__init__(ctx)
        extractor = self.net.xfeat
        for layer in extractor.net.modules():
            if isinstance(layer, nn.Conv2d):
                override(layer, _precise_conv)
        # The FP32 convolutions are device-bound at 768 x 1024, where a graph largely adds a private memory pool.
        dense = CapturedModule(extractor.net, pixels=2**19)
        extractor.set_submodule("net", dense)
        self.graphs = dense.graphs
        override(extractor, _maxima, name="NMS")

    def release_graphs(self) -> None:
        """Release captured extractor graphs."""
        self.graphs.release()


class _MlxPortable(Portable):
    """`Portable` for MLX: the dense network, keypoints, descriptors and matching in MLX operations.

    `mlx_net` (the network and score map), `mlx_post` (selection, sampling, nearest neighbours) and the shared
    `MlxLightGlue`. Only each image's score map and the matcher's index decisions are read on the host. The results
    equal `Portable`'s up to float32 rounding; keypoints of equal score may be ranked differently (upstream's sort is
    not stable either).
    """

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch networks and copy their weights to MLX."""
        super().__init__(ctx)
        from fastforks.shared.lightglue.mlx import MlxLightGlue  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self._dense = mlx_net.MlxDense(self.net.xfeat.net, self.net.params.detection_threshold)
        self._glue = None if self.net.matcher is None else MlxLightGlue(self.net.matcher)

    def _dense_maps(self, image: torch.Tensor) -> tuple[Any, Any, torch.Tensor]:
        """Queue an image's descriptors and score map on MLX; with the detection image's `(width, height)`."""
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - imports mlx

        p = self.net.params
        x = resize_long(image, p.long_side) if p.long_side > 0 else image
        feats, score = self._dense(to_mx(x.float()))
        mx.async_eval(feats, score)
        return feats, score, torch.tensor([x.shape[-1], x.shape[-2]])

    def _keypoints(self, feats: Any, score: Any, size: torch.Tensor) -> tuple[Any, Any]:  # noqa: ANN401 - MLX arrays
        """Keypoints (float32 `(x, y)` in the detection image) and their MLX descriptors."""
        from . import mlx_post  # noqa: PLC0415 - imports mlx

        kpts, desc = mlx_post.detect(feats, score, self.net.params.max_keypoints)
        w, h = size.tolist()
        scale = np.array([w / (w // 32 * 32), h / (h // 32 * 32)], np.float32)  # upstream `rw1`, `rh1`
        return kpts.astype(np.float32) * scale, desc

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores (CPU tensors)."""
        from . import mlx_post  # noqa: PLC0415 - imports mlx

        # both networks are queued first: the second runs while the host ranks the first image's keypoints
        (f1, m1, s1), (f2, m2, s2) = self._dense_maps(image1), self._dense_maps(image2)
        (k1, d1), (k2, d2) = self._keypoints(f1, m1, s1), self._keypoints(f2, m2, s2)
        if self._glue is None:
            pairs, scores = mlx_post.mutual_nearest(d1, d2)
        else:
            pairs, scores = self._glue((k1, d1, s1.numpy()), (k2, d2, s2.numpy()))
        return {
            "keypoints1": self.net.to_input(torch.from_numpy(k1), s1, image1),
            "keypoints2": self.net.to_input(torch.from_numpy(k2), s2, image2),
            "pairs": torch.from_numpy(pairs),
            "scores": torch.from_numpy(scores),
        }


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
