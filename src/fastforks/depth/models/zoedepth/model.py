"""ZoeDepth runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `DepthModel.infer` on one image: reflect-pad by `int(sqrt(side / 2) * 3)` per side (`pad_input`),
resize keeping the aspect ratio ("minimal" towards 384 x 512, multiples of 32, bilinear with `align_corners=True`),
normalise with mean = std = 0.5, run, upsample the metric depth to the padded size (bicubic), crop, and average with
the horizontally flipped pass (`flip`). The NK network routes every pass to its NYU (indoor, 10 m) or KITTI (outdoor,
80 m) metric head by a learned domain classifier; upstream votes over the whole batch, the runners route every image
(and flipped pass) on its own, as upstream's single-image `infer` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.errors import InputError
from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.midas_beit import HEAD_SCALE, dpt_beit_large, scale_head, use_sdpa_attention
from fastforks.shared.midas_beit import load as load_state
from fastforks.shared.midas_beit.fast import decoder, encode, fp16_accumulate, to_kernels

from .net.midas_core import MidasCore, Resize
from .net.zoedepth_nk_v1 import ZoeDepthNK

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastforks.depth._runner import Context

NET_SIZE = (384, 512)  # fixed: zoedepth_nk config img_size (h, w)
BIN_CONF = [
    {"name": "nyu", "n_bins": 64, "min_depth": 1e-3, "max_depth": 10.0},
    {"name": "kitti", "n_bins": 64, "min_depth": 1e-3, "max_depth": 80.0},
]
HEADS = ("nyu", "kitti")  # domain classifier classes


@dataclass(frozen=True)
class Params:
    """Load options."""

    pad_input: bool  # upstream's reflect padding against border artefacts
    flip: bool  # average with the horizontally flipped pass


def _net(ctx: Context) -> ZoeDepthNK:
    """`ZoeDepthNK.build(**get_config("zoedepth_nk", "infer"))` with the checkpoint's weights."""
    with torch.device(ctx.device):
        core = MidasCore(dpt_beit_large(384), keep_aspect_ratio=True, img_size=list(NET_SIZE))
        net = ZoeDepthNK(
            core,
            BIN_CONF,
            bin_centers_type="softplus",
            bin_embedding_dim=128,
            n_attractors=[16, 8, 4, 1],
            attractor_alpha=1000,
            attractor_gamma=2,
            attractor_kind="mean",
            attractor_type="inv",
            min_temp=0.0212,
            max_temp=50.0,
            memory_efficient=True,
        )
    load_state(net, ctx.state_dict(key="model"))
    # fp16 head activations of the MiDaS core reach ~9e4 otherwise: scale its `out_conv` feature down and fold the
    # inverse into the one layer that reads it (the first 1x1 conv of each metric head; its output is unchanged)
    scale_head(net.core.core)
    with torch.no_grad():
        scratch: Any = core.core.scratch
        heads: Any = net.conditional_log_binomial
        for clb in heads.values():
            clb.mlp[0].weight[:, : scratch.output_conv[2].out_channels].mul_(1 / HEAD_SCALE)
    return net.eval()


def _pads(h: int, w: int) -> tuple[int, int]:
    """Upstream `_infer_with_pad_aug` padding (fh = fw = 3): rows, columns per side."""
    return int(math.sqrt(h / 2) * 3), int(math.sqrt(w / 2) * 3)  # not np: dynamo traces NumPy into tensors


def _infer(run: Callable[[torch.Tensor], torch.Tensor], image: torch.Tensor, params: Params) -> torch.Tensor:
    """Upstream `infer(x, pad_input, with_flip_aug)`; `run` maps padded images `[n, 3, h, w]` to depth `[n, 1, h', w']`.

    Both passes go through one `run` call (the network routes them independently).

    Raises:
        InputError: `pad_input` and a side too short to reflect-pad (below 5 px).
    """
    x = image
    ph, pw = _pads(*image.shape[-2:]) if params.pad_input else (0, 0)
    if ph >= image.shape[-2] or pw >= image.shape[-1]:
        size = tuple(image.shape[-2:])
        msg = f"zoedepth: pad_input reflect-pads {ph} x {pw} px, needs sides of at least 5 px, got {size}"
        raise InputError(msg)
    if params.pad_input:
        x = F.pad(x, [pw, pw, ph, ph] if ph > 0 else [pw, pw], mode="reflect")
    xs = torch.cat([x, x.flip(3)]) if params.flip else x
    out = run(xs).float()
    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=x.shape[-2:], mode="bicubic", align_corners=False)
    out = out[:, 0, ph : out.shape[2] - ph, pw : out.shape[3] - pw]
    if params.flip:
        n = image.shape[0]
        out = (out[:n] + out[n:].flip(2)) / 2
    return out


class Portable(Runner):
    """Upstream network (SDPA attention with the relative position bias as mask): float32, or fp16 autocast."""

    dtypes = ("float32", "float16")
    batch = 1  # upstream infers one image at a time (the domain vote spans the batch)
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        use_sdpa_attention(self.net)
        self.half = ctx.dtype == "float16"

    def _pass(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.half):
            return self.net(x)["metric_depth"]

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return _infer(lambda xs: torch.cat([self._pass(x[None]) for x in xs]), image, self.ctx.params)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Metric depth of a batch."""
        return {"depth": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> metric depth `[1, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image))


def _bins(
    zoe: ZoeDepthNK, name: str, x: torch.Tensor, last: torch.Tensor, blocks: Sequence[torch.Tensor]
) -> torch.Tensor:
    """`ZoeDepthNK.forward` after the domain vote (softplus bin centres), for passes routed to head `name`."""
    _, b_prev = zoe.seed_bin_regressors[name](x)
    prev_b_embedding = zoe.seed_projector(x)
    b_centers, b_embedding = b_prev, prev_b_embedding
    for projector, attractor, block in zip(
        zoe.projectors, cast("torch.nn.ModuleList", zoe.attractors[name]), blocks, strict=True
    ):
        b_embedding = projector(block)
        b_prev, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
        prev_b_embedding = b_embedding
    b_centers = F.interpolate(b_centers, last.shape[-2:], mode="bilinear", align_corners=True)
    b_embedding = F.interpolate(b_embedding, last.shape[-2:], mode="bilinear", align_corners=True)
    probs = zoe.conditional_log_binomial[name](last, b_embedding)
    return torch.sum(probs * b_centers, dim=1, keepdim=True)


def _head(zoe: ZoeDepthNK, feats: Sequence[torch.Tensor]) -> torch.Tensor:
    """ZoeDepth NK head on MiDaS features `[out_conv, l4_rn, r4, r3, r2, r1]`, every pass routed on its own."""
    last, blocks = feats[0], feats[2:]
    x = zoe.conv2(feats[1])
    logits = zoe.mlp_classifier(zoe.patch_transformer(x)[0])
    route = logits.argmax(1).tolist()  # = argmax of upstream's softmax vote for a batch of one
    if len(set(route)) == 1:
        return _bins(zoe, HEADS[route[0]], x, last, blocks)
    out = torch.empty((x.shape[0], 1, *last.shape[-2:]), device=x.device)
    for k in set(route):
        i = torch.tensor([j for j, r in enumerate(route) if r == k], device=x.device)
        out[i] = _bins(zoe, HEADS[k], x[i], last[i], [b[i] for b in blocks]).float()
    return out


class Kernels(Runner):
    """fp16 BEiT and DPT decoder (SDPA, fp16-accumulating GEMMs and cuDNN convs), both passes as one batch.

    The NK head runs under fp16 autocast with upstream's memory-efficient attractor loop (faster and smaller than the
    vectorised path on an RTX 5060 Ti: 88.6 vs 97.8 ms, 1733 vs 2392 MiB peak on a 1482 x 994 image).
    """

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert it."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        self.net = _net(ctx)
        to_kernels(self.net.core.core)
        self.resize = Resize(
            NET_SIZE[1], NET_SIZE[0], keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal"
        )

    def _run(self, xs: torch.Tensor) -> torch.Tensor:
        w, h = self.resize.get_size(xs.shape[3], xs.shape[2])
        x = (F.interpolate(xs, (int(h), int(w)), mode="bilinear", align_corners=True) - 0.5) / 0.5
        midas = self.net.core.core
        with fp16_accumulate():
            feats = decoder(midas, encode(midas, x), int(h), int(w))
        with torch.autocast("cuda", dtype=torch.float16):
            return _head(self.net, feats)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Metric depth of a batch."""
        return {"depth": _infer(self._run, image, self.ctx.params)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
