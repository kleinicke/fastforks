"""INSID3 runners: training-free in-context segmentation on the fastforks DINOv3 backbone.

Upstream (`models/insid3.py`): DINOv3 last-layer patch features of the references and the target at 1024 x 1024,
positional debiasing (projection off the top 500 left singular vectors of a black image's features), a reference
prototype, candidate patches by forward (prototype) and backward (nearest-neighbour vote) matching, average-linkage
clustering of the target patches (cosine distance, threshold 1 - tau), then seed selection and cluster aggregation.

Here the backbone is `fastforks.features` DINOv3 in the same mode (portable: SDPA fp16; kernels: the fastbb / dpt_fused
ViT), and the clustering runs on the device (`average_linkage`: reciprocal-nearest-neighbour merges, the same
dendrogram as upstream's scikit-learn on the CPU). The per-cluster loops are scatter reductions.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn.functional as F

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict
from fastforks._core.image import checked_ranges
from fastforks._core.model import load_variant
from fastforks.features._api import TASK, FeaturesModel
from fastforks.promptable_segmentation._runner import Forward, Runner

if TYPE_CHECKING:
    from fastforks.promptable_segmentation._runner import Context

PREFIX = 5  # fixed: DINOv3 cls + 4 register tokens in front of the patch grid
HALF = 0.5  # fixed: upstream's threshold of resized binary masks


@dataclass(frozen=True)
class Params:
    """Variant `params`."""

    encoder: str  # fastforks.features DINOv3 id
    image_size: int  # square network input (upstream 1024)
    svd_components: int  # positional subspace size (upstream 500)
    tau: float  # clustering similarity threshold (upstream 0.6)
    merge_threshold: float  # cluster aggregation threshold (upstream 0.2)


def average_linkage(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Average-linkage agglomerative clustering of unit rows `x` `[N, C]` under cosine distance, cut at `threshold`.

    Clusters merge while their average distance is below `threshold` (scikit-learn's `distance_threshold`). Average
    linkage is reducible, so merging every reciprocal-nearest-neighbour pair of a round at once gives the dendrogram of
    the sequential algorithm (up to ties); distances are float64 as scikit-learn's.

    Returns:
        `[N]` int64 labels `0 .. K-1` (numbered by first point).
    """
    n = x.shape[0]
    dev = x.device
    d = (1.0 - (x @ x.T).clamp(-1, 1)).double()
    d.fill_diagonal_(math.inf)
    ids = torch.arange(n, device=dev)  # original representative of each row of `d`
    size = torch.ones(n, dtype=torch.float64, device=dev)
    owner = torch.arange(n, device=dev)  # representative of each point
    while d.shape[0] > 1:
        m = d.shape[0]
        nn_d, nn = d.min(1)
        r = torch.arange(m, device=dev)
        pair = (nn[nn] == r) & (r < nn) & (nn_d < threshold)
        i = pair.nonzero().squeeze(1)
        if i.numel() == 0:
            break
        j = nn[i]
        a = size[i] / (size[i] + size[j])
        b = 1.0 - a
        d[i] = a[:, None] * d[i] + b[:, None] * d[j]  # Lance-Williams: rows of the merged clusters
        d[:, i] = d[:, i] * a + d[:, j] * b  # then their columns (fixes merged-merged entries too)
        size[i] += size[j]
        remap = torch.arange(n, device=dev)
        remap[ids[j]] = ids[i]
        owner = remap[owner]
        keep = torch.ones(m, dtype=torch.bool, device=dev)
        keep[j] = False
        d, ids, size = d[keep][:, keep], ids[keep], size[keep]
    return torch.unique(owner, return_inverse=True)[1]


def _downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Upstream `downsample_mask`: `[1, 1, H, W]` binary -> `[h, w]` bool, never empty for a non-empty mask."""
    down = F.interpolate(mask.float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0] > HALF
    if not down.any():
        down = F.interpolate(mask.float(), size=(h, w), mode="nearest")[0, 0] > HALF
        if not down.any() and mask.any():
            centre = torch.argwhere(mask[0, 0]).float().mean(0) / (mask.shape[-1] // w)
            down[int(centre[0]), int(centre[1])] = True
    return down


def _cluster_means(x: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Per-cluster means `[K, C]` of rows `x` `[N, C]`."""
    s = torch.zeros(k, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, labels, x)
    return s / torch.bincount(labels, minlength=k).clamp_min(1)[:, None].to(x.dtype)


class Portable(Runner):
    """`fastforks.features` DINOv3 portable (SDPA, fp16 on CUDA); clustering and matching in float32 on the device."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    prompts = frozenset({"reference"})
    compiled_methods = ("embed",)
    encoder_mode: ClassVar[Literal["portable", "kernels"]] = "portable"

    def __init__(self, ctx: Context) -> None:
        """Estimate the positional subspace with a float32 backbone, then build the backbone of the mode.

        The subspace is float32 as upstream's (built outside its autocast): its trailing singular vectors are nearly
        degenerate, so fp16 features of the black image rotate them (similarity off by ~0.05). The checkpoint is
        read and moved to the device once; both backbones are built from that copy, the second while the first embeds.
        """
        super().__init__(ctx)
        p = ctx.params
        self.size = (p.image_size, p.image_size)
        card, variant = registry.resolve(TASK, p.encoder)
        path = ctx.weights()[0]
        state = {k: v.to(ctx.device) for k, v in load_state_dict(path).items()}
        encoder = functools.partial(
            load_variant,
            TASK,
            card,
            variant,
            FeaturesModel,
            device=ctx.device,
            weights=path,
            state=state,
        )
        fp32 = self.encoder_mode == "portable" and ctx.dtype == "float32"
        self.encoder = encoder(mode="portable", dtype="float32")  # assigned the device checkpoint, no copy
        e = self.embed(torch.zeros(1, 3, *self.size, device=ctx.device))["features"][0].flatten(1)  # [C, N] unit
        if not fp32:  # built on the CPU while the device still runs the float32 embed
            self.encoder = encoder(mode=self.encoder_mode, dtype=ctx.dtype if self.encoder_mode == "portable" else None)
        del state
        u = torch.linalg.svd(e - e.mean(1, keepdim=True), full_matrices=False)[0][:, : p.svd_components]
        self.p_perp = torch.eye(u.shape[0], device=ctx.device) - u @ u.T

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unit last-layer patch features `[B, C, h, w]` at `image_size` x `image_size` (upstream's square resize)."""
        with checked_ranges():  # the core checked the pixel range
            tokens = self.encoder(image, size=self.size).last_hidden_state.reshape(image.shape[0], -1, self.hidden)
        h, w = self.size[0] // 16, self.size[1] // 16
        f = tokens[:, PREFIX:].float().transpose(1, 2).reshape(image.shape[0], -1, h, w)
        return {"features": F.normalize(f, dim=1)}

    @property
    def hidden(self) -> int:
        """Backbone width."""
        return self.encoder.hidden_size

    def _debias(self, f: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f.shape
        return F.normalize((self.p_perp @ f.reshape(b, c, h * w)).reshape(b, c, h, w), dim=1)

    def segment_reference(
        self, target: torch.Tensor, references: torch.Tensor, masks: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Target mask `[h, w]` bool at the patch grid, and each patch's similarity to the reference prototype.

        Args:
            target: `embed` features of the target `[1, C, h, w]`.
            references: `embed` features of the references `[S, C, h, w]`.
            masks: Per reference `[H_s, W_s]` bool (any size; resized to `image_size` with nearest as upstream).
        """
        _, c, h, w = target.shape
        s = references.shape[0]
        deb = self._debias(torch.cat([references, target]))
        ref_deb, tgt_deb, tgt = deb[:s], deb[s], target[0]
        sq = [F.interpolate(m[None, None].float(), size=self.size, mode="nearest") > HALF for m in masks]
        small = [_downsample_mask(m, h, w) for m in sq]
        protos = [ref_deb[i][:, small[i]].mean(1) for i in range(s) if small[i].any()]
        tgt_deb_flat = tgt_deb.reshape(c, -1)  # [C, N]
        if not protos:  # every reference mask is empty
            return {
                "masks": torch.zeros(h, w, dtype=torch.bool, device=target.device),
                "similarity": target.new_zeros(h, w),
            }
        proto = F.normalize(torch.stack(protos).mean(0), dim=0)  # [C]
        fg_sim = proto @ tgt_deb_flat  # [N]
        out = {"similarity": fg_sim.reshape(h, w)}
        forward = fg_sim > 0
        if not forward.any():
            forward = fg_sim > torch.quantile(fg_sim, 0.9)
        votes = torch.zeros(h * w, dtype=torch.int32, device=target.device)
        for i in range(s):  # backward: the target patch's nearest reference patch lies in the mask
            best = (ref_deb[i].reshape(c, -1).T @ tgt_deb_flat).argmax(0)
            votes += small[i].reshape(-1)[best].to(torch.int32)
        candidate = forward & (votes >= math.ceil(s / 2))
        if not candidate.any():
            return out | {"masks": candidate.reshape(h, w)}
        tgt_flat = tgt.reshape(c, -1).T  # [N, C]
        labels = average_linkage(tgt_flat, 1.0 - self.ctx.params.tau)
        k = int(labels.max()) + 1
        protos_deb = F.normalize(_cluster_means(tgt_deb_flat.T, labels, k), dim=1)
        protos_orig = F.normalize(_cluster_means(tgt_flat, labels, k), dim=1)
        counts = torch.bincount(labels, minlength=k).float()
        matched = torch.bincount(labels[candidate], minlength=k).float()
        area = matched / counts
        cross_matched = (protos_deb @ proto).masked_fill(matched == 0, -math.inf)
        seed = int(cross_matched.argmax())  # the matched cluster closest to the reference (first on ties)
        intra = protos_orig @ protos_orig[seed]
        cross = _cluster_means(fg_sim[:, None], labels, k)[:, 0]
        area[seed] = 1.0
        score = cross * intra * area
        return out | {"masks": (score[labels] > self.ctx.params.merge_threshold).reshape(h, w)}

    def predict(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Not a point / box model."""
        msg = "INSID3 takes reference images and masks"
        raise NotImplementedError(msg)

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Feature-resolution masks `[..., h, w]` (bool or float) -> bool at `size` (upstream: bilinear > 0.5)."""
        lead = pred_masks.shape[:-2]
        x = pred_masks.float().reshape(-1, 1, *pred_masks.shape[-2:])
        return (F.interpolate(x, size=size, mode="bilinear", align_corners=False) > HALF).reshape(*lead, *size)

    def module(self) -> torch.nn.Module:
        """The backbone (`fastforks.features` portable module): normalised `[B, 3, h, w]` -> `last_hidden_state`."""
        net = self.encoder._runner.module()  # noqa: SLF001 - the backbone model's own export module
        return Forward(net, lambda n, image: n(image), output_names=("last_hidden_state",))

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The backbone's networks."""
        return self.encoder._runner.networks()  # noqa: SLF001


class Kernels(Portable):
    """`fastforks.features` DINOv3 kernels (fastbb flash attention, dpt_fused fp16-accumulate GEMMs)."""

    encoder_mode = "kernels"

    def module(self) -> torch.nn.Module:
        """Kernels export nothing."""
        raise NotImplementedError


RUNNERS = {"portable": Portable, "kernels": Kernels}
