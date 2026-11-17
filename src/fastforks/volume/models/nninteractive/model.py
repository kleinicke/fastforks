"""nnInteractive runners: DKFZ's promptable 3D segmentation (point, box and scribble prompts), nnInteractive 1.1.5.

The network is nnU-Net's `ResidualEncoderUNet` (`fastforks.volume._nnunet.UNet`, ResEnc-L: 6 stages, 32..320 channels)
on 192^3 patches of 8 channels: the z-scored image, the previous mask and the six prompt maps (box +/-, point +/-,
scribble +/-), two output heads. The session is a port of `nnInteractiveInferenceSession` (inference_session.py):

- `start`: crop to the nonzero bounding box, z-score over the crop; all state (image float32, the 7 interaction maps
  float16, the mask uint8) stays on the device (upstream keeps it in pinned CPU memory).
- every prompt: the older prompts of the same kind decay by 0.98, the new one is drawn (a point: the normalised
  Euclidean distance transform of a radius-4 ball; a box: ones; a scribble: its mask), then one prediction around it.
- prediction: a patch around the prompt, zoomed out (box / scribble size + a third of the patch, up to 4x) and grown
  1.5x while the mask still changes at the patch border (AutoZoom); a zoomed-out patch is resampled to 192^3 (prompt
  maps dilated first, area-averaged; image trilinear), its mask upsampled back, and the voxels that changed (opened
  with a 5^3 kernel) refined at full resolution by greedily placed 192^3 boxes.

Lasso prompts and initial masks are not ported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import InputError
from fastforks._core.patch import override
from fastforks._core.runner import Forward
from fastforks.volume._nnunet import Norm, Plan, UNet, _network, configuration_of
from fastforks.volume._nnunet_runner import CHECKPOINT_GLOBALS, fused_norm_act
from fastforks.volume._runner import InteractiveRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.volume._runner import Context

__all__ = ["RUNNERS", "Params", "Portable", "Session", "cover_boxes", "point_map"]

Box = list[list[int]]  # [[start, end), ...] per axis
_BORDER = (1500, 0.2, 100)  # fixed: upstream `_detect_change_at_border` (absolute, relative, minimum changed voxels)
_MAX_ZOOM = 4.0  # fixed: upstream session constant (max zoom-out factor), not in inference_session_class.json
_ZOOM_GROWTH = 1.5  # fixed: upstream session constant (zoom-out step), not in inference_session_class.json


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        configuration: The plans configuration of the checkpoint.
        point_radius: Radius of a point prompt in voxels (`inference_session_class.json`).
        decay: Factor on the older prompts of a kind when one is added (upstream's default; not in the json).
        autozoom: Zoom out while the mask changes at the patch border.
    """

    configuration: str = "3d_fullres_ps192_bs24"
    point_radius: int = 4
    decay: float = 0.98
    autozoom: bool = True


@dataclass
class Session:
    """The state of one volume: the crop, the image, the prompt maps and the mask (all on the device).

    Attributes:
        shape: The input volume's `(D, H, W)`.
        start: Offset of the nonzero crop in the input.
        image: The z-scored crop `[d, h, w]` float32.
        inter: `[7, d, h, w]` float16: the previous mask, box +/-, point +/-, scribble +/-.
        target: The mask on the input grid `[D, H, W]` uint8.
        positive_box: A positive box was added since the last refinement.
        rng: Random fallback of the box cover (upstream: `np.random`), seeded per session.
    """

    shape: tuple[int, int, int]
    start: tuple[int, int, int]
    image: torch.Tensor
    inter: torch.Tensor
    target: torch.Tensor
    positive_box: bool = False
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(0))


def _plan(plans: dict[str, Any], configuration: str) -> Plan:
    """The network plan of the checkpoint's configuration, two heads (background, object)."""
    conf = configuration_of(plans, configuration)
    net = _network(conf)
    return Plan(
        patch_size=tuple(int(p) for p in conf["patch_size"]),  # ty: ignore[invalid-argument-type]
        spacing=tuple(float(s) for s in conf["spacing"]),  # ty: ignore[invalid-argument-type]
        norms=(Norm("ZScoreNormalization"),),
        features=net.pop("features"),
        kernel_sizes=tuple(tuple(int(k) for k in ks) for ks in net.pop("kernel_sizes")),
        strides=tuple(tuple(int(s) for s in ss) for ss in net.pop("strides")),
        n_conv_encoder=net.pop("n_conv_encoder"),
        n_conv_decoder=net.pop("n_conv_decoder"),
        labels={1: "object"},
        **net,
    )


def point_map(radius: int) -> torch.Tensor:
    """`build_point((r, r, r), use_distance_transform=True)`: skimage's `ball(r)`, its exact EDT, divided by its max."""
    n = 2 * radius + 1
    g = torch.arange(n, dtype=torch.float64) - radius
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    inside = zz**2 + yy**2 + xx**2 <= radius**2
    pts = torch.stack((zz, yy, xx), -1).reshape(-1, 3)
    bg = pts[~inside.reshape(-1)]
    edt = torch.cdist(pts, bg).min(1).values.reshape(n, n, n) * inside  # distance to the nearest background voxel
    return edt / edt.max()


def _odd(x: float) -> int:
    """`round_to_nearest_odd`."""
    c, f = math.ceil(x), math.floor(x)
    return c if c % 2 else f if f % 2 else round(x) + 1


def _clip(box: Box, shape: Sequence[int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Slices of `box ∩ [0, shape)` in the volume and in the box's own frame."""
    vol = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(box, shape, strict=True))
    own = tuple(slice(s.start - a, s.stop - a) for s, (a, _) in zip(vol, box, strict=True))
    return vol, own


def _crop_pad(x: torch.Tensor, box: Box) -> torch.Tensor:
    """`x[..., box]` with zeros outside `x` (`crop_and_pad_nd` / `crop_to_valid` + `pad_cropped`)."""
    vol, _ = _clip(box, x.shape[-3:])
    out = x[(..., *vol)]
    pad = [p for (a, b), s in zip(reversed(box), reversed(vol), strict=True) for p in (s.start - a, b - s.stop)]
    return F.pad(out, pad) if any(pad) else out


def _paste(target: torch.Tensor, src: torch.Tensor, box: Box) -> None:
    """`paste_tensor`: the part of `src` (the size of `box`) inside `target`."""
    vol, own = _clip(box, target.shape)
    if all(s.stop > s.start for s in vol):
        target[vol] = src[own].to(target.dtype)


def _box_sums(sat: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Sums of the boxes `[lo, hi)` (`[N, 3]`) from a zero-padded 3D summed-area table."""
    out = torch.zeros(lo.shape[0], dtype=sat.dtype, device=sat.device)
    for corner in range(8):
        idx = [hi[:, a] if corner >> a & 1 else lo[:, a] for a in range(3)]
        sign = 1 if (3 - corner.bit_count()) % 2 == 0 else -1
        out += sign * sat[idx[0], idx[1], idx[2]]
    return out


def _sat(mask: torch.Tensor) -> torch.Tensor:
    s = F.pad(mask.to(torch.int32), (1, 0, 1, 0, 1, 0))
    for d in range(3):
        s = s.cumsum(d, dtype=torch.int32)
    return s


def _bounds(c: torch.Tensor, size: Sequence[int], margin: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
    """The covered part `[lo, hi)` (`[N, 2, 3]`) of boxes of `size` centred on `c` (`[N, 3]`), `margin` left out."""
    half = torch.tensor([s // 2 for s in size], device=c.device)
    end = torch.tensor([s - s // 2 for s in size], device=c.device)
    m = torch.tensor(list(margin), device=c.device)
    lo = (c - half + m).clamp(min=0)
    hi = torch.minimum(c + end - m, torch.tensor(list(shape), device=c.device))
    return torch.stack((lo, hi), 1)


def _box_at(c: Sequence[int], size: Sequence[int]) -> Box:
    return [[ci - s // 2, ci + s - s // 2] for ci, s in zip(c, size, strict=True)]


def _random_cover(
    mask: torch.Tensor, size: Sequence[int], margin: Sequence[int], rng: np.random.RandomState
) -> list[Box]:
    """`random_sampling_fallback`: the best of 25 random object voxels as the next centre until `mask` is covered."""
    boxes = []
    while mask.any():
        idx = torch.nonzero(mask)
        pick = idx[torch.tensor([rng.choice(len(idx)) for _ in range(25)], device=mask.device)]
        b = _bounds(pick, size, margin, mask.shape)
        best = int(torch.argmax(_box_sums(_sat(mask), b[:, 0], b[:, 1])))  # the first of equal counts
        boxes.append(_box_at(pick[best].tolist(), size))
        lo, hi = b[best].tolist()
        mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 0
    return boxes


def cover_boxes(  # noqa: PLR0913, PLR0917 - `generate_bounding_boxes`' arguments
    mask: torch.Tensor,
    size: Sequence[int],
    rng: np.random.RandomState,
    stride: Sequence[int] | None = None,
    margin: Sequence[int] = (10, 10, 10),
    max_depth: int = 3,
    depth: int = 0,
) -> list[Box]:
    """`generate_bounding_boxes(mask, size, stride="auto" if None, margin, max_depth)`: boxes covering a 0/1 mask.

    Greedy: of the object voxels on a stride grid, the centre whose box (less the margin) covers most uncovered voxels
    (the first in `(z, y, x)` order on ties), until covered; leftovers recurse at half the stride or, below a 27th of
    a box, go to the random fallback. Box counts come from a summed-area table instead of one sum per centre.
    """
    if not mask.any():
        return []
    if depth > max_depth:
        return _random_cover(mask, size, margin, rng)
    nz = [torch.nonzero(mask.any(dim=tuple(a for a in range(3) if a != d)))[:, 0] for d in range(3)]
    mn, mx = [int(n[0]) for n in nz], [int(n[-1]) for n in nz]
    if stride is None:
        stride = [max(1, round((b - a) / 4)) for a, b in zip(mn, mx, strict=True)]
    grid = mask[tuple(slice(a, b + 1, s) for a, b, s in zip(mn, mx, stride, strict=True))]
    centres = torch.nonzero(grid) * torch.tensor(list(stride), device=mask.device) + torch.tensor(
        mn, device=mask.device
    )
    half = [max(1, s // 2) for s in stride]
    if len(centres) == 0:
        return cover_boxes(mask, size, rng, half, margin, max_depth, depth + 1)
    uncovered = mask.clone().to(torch.uint8)
    boxes: list[Box] = []
    while len(centres) and uncovered.any():
        b = _bounds(centres, size, margin, mask.shape)
        counts = _box_sums(_sat(uncovered), b[:, 0], b[:, 1])
        best = int(torch.argmax(counts))
        if int(counts[best]) == 0:
            break
        boxes.append(_box_at(centres[best].tolist(), size))
        lo, hi = b[best].tolist()
        uncovered[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 0
        centres = centres[uncovered[tuple(centres.T)] > 0]
    if uncovered.any():
        if int(uncovered.sum()) < math.prod(s // 3 for s in size):
            boxes += _random_cover(uncovered, size, margin, rng)
        else:
            boxes += cover_boxes(uncovered, size, rng, half, margin, max_depth, depth + 1)
    return boxes


class Portable(InteractiveRunner):
    """Plain PyTorch: the float16 network in `channels_last_3d` on CUDA; the session state on the device."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Read the plans, build the network and load the checkpoint."""
        super().__init__(ctx)
        p: Params = ctx.params
        files = ctx.weights()
        self.plan = _plan(json.loads(Path(files[0]).read_text()), p.configuration)
        self.patch = list(self.plan.patch_size)
        self.net = self._build(UNet(self.plan, 8))
        sd = ctx.state_dict(1, key="network_weights", strip="", extra=CHECKPOINT_GLOBALS)
        load_strict(self.net, UNet.state_dict_of(sd), f"{ctx.variant.ref} checkpoint")
        self._dtype = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.net.eval().requires_grad_(requires_grad=False).to(ctx.device, self._dtype)
        self._channels_last = ctx.device.type == "cuda"
        if self._channels_last:
            self.net = self.net.to(memory_format=torch.channels_last_3d)
        self._point = point_map(p.point_radius).to(ctx.device, torch.float16)

    def _build(self, net: UNet) -> UNet:
        return net

    def module(self) -> torch.nn.Module:
        """The U-Net: `[1, 8, 192, 192, 192]` (image, previous mask, six prompt maps) -> logits `[1, 2, ...]`."""
        return Forward(self.net, lambda net, x: net(x), output_names=("logits",))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The mask (argmax, uint8) of one network input `[8, *patch]`."""
        x = x[None].to(self._dtype)
        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return self.net(x)[0].argmax(0).to(torch.uint8)

    # the session (host code: prompt maps, zoom, refinement boxes; the network runs in `__call__`)

    def start(self, volume: torch.Tensor) -> Session:
        """A session on a `[1, D, H, W]` float32 volume: crop to its nonzero box, z-score."""
        v = volume[0]
        nz = [torch.nonzero((v != 0).any(dim=tuple(a for a in range(3) if a != d)))[:, 0] for d in range(3)]
        if any(len(n) == 0 for n in nz):
            msg = "nnInteractive needs a volume with nonzero voxels (it crops to their bounding box)"
            raise InputError(msg)
        lo, hi = [int(n[0]) for n in nz], [int(n[-1]) + 1 for n in nz]
        image = v[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].float().clone()
        image -= image.mean()
        image /= image.std()
        inter = torch.zeros((7, *image.shape), dtype=torch.float16, device=v.device)
        target = torch.zeros(v.shape, dtype=torch.uint8, device=v.device)
        return Session(tuple(v.shape), tuple(lo), image, inter, target)  # ty: ignore[invalid-argument-type]

    def mask(self, s: Session) -> torch.Tensor:
        """The mask on the input grid (bool)."""
        return s.target.bool()

    def reset(self, s: Session) -> None:
        """Forget every prompt and the mask (`reset_interactions`)."""
        s.inter.zero_()
        s.target.zero_()
        s.positive_box = False

    def point(self, s: Session, zyx: Sequence[float], positive: bool) -> None:  # noqa: FBT001
        """`add_point_interaction`."""
        c = [round(x - o) for x, o in zip(zyx, s.start, strict=True)]
        s.inter[3:5] *= self.ctx.params.decay
        r = self._point.shape[0]
        box = _box_at(c, (r, r, r))
        if not any(b < 0 for _, b in box) and not any(a > n for (a, _), n in zip(box, s.image.shape, strict=True)):
            vol, own = _clip(box, s.image.shape)
            ch = s.inter[3 if positive else 4]
            torch.maximum(ch[vol], self._point[own], out=ch[vol])
        self._predict(s, c, 1.0)

    def box(self, s: Session, box: Sequence[Sequence[float]], positive: bool) -> None:  # noqa: FBT001
        """`add_bbox_interaction`: `[[z0, z1), [y0, y1), [x0, x1)]` in input voxels (one side 1: a 2D box)."""
        if positive:
            s.positive_box = True
        b = []
        for (a, e), o, n in zip(box, s.start, s.image.shape, strict=True):
            lo, hi = max(0, round(a - o)), min(n, round(e - o))
            if hi <= lo:
                lo, hi = (0, min(1, n)) if lo == 0 else (max(lo - 1, 0), hi)
            b.append([lo, hi])
        centre = [round((lo + hi) / 2) for lo, hi in b]
        zoom = max(1.0, *((hi - lo + p // 3) / p for (lo, hi), p in zip(b, self.patch, strict=True)))
        s.inter[1:3] *= self.ctx.params.decay
        s.inter[(1 if positive else 2, *(slice(lo, hi) for lo, hi in b))] = 1
        self._predict(s, centre, zoom)

    def scribble(self, s: Session, mask: torch.Tensor, positive: bool) -> None:  # noqa: FBT001
        """`add_scribble_interaction` with a `[D, H, W]` 0/1 mask on the input grid."""
        crop = _crop_pad(mask, [[o, o + n] for o, n in zip(s.start, s.image.shape, strict=True)])
        nz = torch.nonzero(crop)
        if len(nz) == 0:
            msg = "the scribble has no voxels inside the volume's nonzero region"
            raise InputError(msg)
        lo, hi = nz.min(0).values.tolist(), (nz.max(0).values + 1).tolist()
        centre = [round((a + b) / 2) for a, b in zip(lo, hi, strict=True)]
        zoom = max(1.0, *((b - a + p // 3) / p for a, b, p in zip(lo, hi, self.patch, strict=True)))
        s.inter[5:7] *= self.ctx.params.decay
        ch = s.inter[5 if positive else 6]
        torch.maximum(ch, crop.to(ch.dtype), out=ch)
        self._predict(s, centre, zoom)

    def _input(self, s: Session, centre: Sequence[int], zoom: float) -> tuple[torch.Tensor, list[int], Box]:
        """`_build_network_input`: the 8-channel input around `centre`, the zoomed patch size and its box."""
        size = [round(p * zoom) for p in self.patch]
        box = [[c - n // 2, c + n // 2 + n % 2] for c, n in zip(centre, size, strict=True)]
        img, inter = _crop_pad(s.image, box)[None], _crop_pad(s.inter, box)
        if size != self.patch:
            inter = inter.clone()  # the crop may be a view of the state
            k = _odd(zoom * 2 - 1)
            if k > 1:  # == upstream's iterated 3^3 max pools on replicate padding
                inter[3:7] = F.max_pool3d(inter[None, 3:7], k, 1, k // 2)[0]
            inter = F.interpolate(inter[None], self.patch, mode="area")[0]
            img = F.interpolate(img[None], self.patch, mode="trilinear")[0]
        return torch.cat((img, inter.float())), size, box

    def _border_change(self, pred: torch.Tensor, prev: torch.Tensor) -> bool:
        """`_detect_change_at_border`: the mask changed on a face of the patch by enough voxels."""
        faces = [(d, i) for d in range(3) for i in (0, pred.shape[d] - 1)]
        cur = [pred.select(d, i) for d, i in faces]
        old = [prev.select(d, i) for d, i in faces]
        stats = torch.stack([
            torch.stack((o.sum(dtype=torch.float32), c.sum(dtype=torch.float32), (o != c).sum().float()))
            for o, c in zip(old, cur, strict=True)
        ]).tolist()  # fmt: skip
        absolute, relative, minimum = _BORDER
        for n_prev, n_cur, n_diff in stats:
            # upstream sums the float16 map in float16
            n_old = float(np.float16(n_prev)) if prev.dtype == torch.float16 else n_prev
            rel = max(n_old, n_cur) / max(min(n_old, n_cur), 1e-5) - 1
            if n_diff > absolute or (n_diff > minimum and rel > relative):
                return True
        return False

    def _predict(self, s: Session, centre: list[int], zoom: float) -> None:
        """`_predict`: the patch around the prompt, AutoZoom, then paste or refine."""
        zoom = min(_MAX_ZOOM, zoom)
        while True:
            x, size, box = self._input(s, centre, zoom)
            pred = self(x)
            prev = _crop_pad(s.inter[0], box)
            if prev.shape != pred.shape:
                prev = F.interpolate(prev[None, None].float(), pred.shape, mode="nearest")[0, 0]
            if not (self.ctx.params.autozoom and self._border_change(pred, prev)) or zoom >= _MAX_ZOOM:
                break
            zoom = min(_MAX_ZOOM, zoom * _ZOOM_GROWTH)
        if zoom == 1:
            _paste(s.inter[0], pred, box)
            _paste(s.target, pred, [[a + o, b + o] for (a, b), o in zip(box, s.start, strict=True)])
            return
        if list(pred.shape) != size:
            pred = (F.interpolate(pred[None, None].float(), size, mode="trilinear")[0, 0] >= 0.5).to(torch.uint8)  # noqa: PLR2004
        diff = self._diff(pred, s.inter[0], box)
        _paste(s.inter[0], pred, box)
        self._refine(s, diff, centre)

    @staticmethod
    def _diff(pred: torch.Tensor, prev: torch.Tensor, box: Box) -> torch.Tensor:
        """`_compute_diff_map`: changed voxels inside the patch, opened with a 5^3 kernel, on the crop grid (uint8)."""
        vol, own = _clip(box, prev.shape)
        d = (pred[own] != prev[vol]).half()[None, None]
        d = -F.max_pool3d(-d, 5, 1, 2)  # == upstream's min / max pools on replicate padding
        d = F.max_pool3d(d, 5, 1, 2)
        out = torch.zeros(prev.shape, dtype=torch.uint8, device=prev.device)
        out[vol] = d[0, 0].to(torch.uint8)
        return out

    def _refine(self, s: Session, diff: torch.Tensor, centre: Sequence[int]) -> None:
        """`_refine_coarse`: full-resolution patches over the changed voxels."""
        if s.positive_box:
            s.inter[1][~(s.inter[0] > 0.5)] = 0  # noqa: PLR2004
            s.positive_box = False
        boxes = cover_boxes(diff, self.patch, s.rng) or [_box_at(centre, self.patch)]
        del diff
        x = torch.zeros((8, *self.patch), dtype=torch.float32, device=s.image.device)
        for box in boxes:
            vol, own = _clip(box, s.image.shape)
            x[(0, *own)] = s.image[vol]
            x[(slice(1, None), *own)] = s.inter[(slice(None), *vol)].float()
            pred = self(x)
            _paste(s.inter[0], pred, box)
            _paste(s.target, pred, [[a + o, b + o] for (a, b), o in zip(box, s.start, strict=True)])
            x.zero_()


class Kernels(Portable):
    """Portable + InstanceNorm + LeakyReLU as one in-place kernel (`vol_norm`) on the float16 feature maps."""

    def __init__(self, ctx: Context) -> None:
        """Load `vol_norm` first, then the network."""
        from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

        vol_norm.load(ctx.device)
        super().__init__(ctx)

    def _build(self, net: UNet) -> UNet:
        for m in net.modules():
            if hasattr(m, "norm_act"):
                override(m, fused_norm_act, "norm_act")
        return net


RUNNERS = {"portable": Portable, "kernels": Kernels}
