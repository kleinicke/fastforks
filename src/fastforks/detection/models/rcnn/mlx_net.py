"""detectron2 Mask / Faster R-CNN (R50-FPN) on Apple MLX, written in `mlx.core` operations (`Net`).

The three parts of `model._Rcnn` that are not box arithmetic, channels last, in the run dtype:

- `dense`: ResNet + FPN and the RPN head (compiled stages, one in flight). The levels P2..P5 stay on the GPU;
  only the RPN head's maps come back (float32, in anchor order, all levels in two arrays).
- `box_head`: ROIAlign of the proposals, the two FCs and the predictor (float32).
- `mask_head`: ROIAlign of the detections, the four convs, the deconvolution and the predicted class's mask.

ROIAlign (detectron2 `ROIAlignV2` = torchvision `roi_align(aligned=True)`, sampling ratio 0) is one Metal kernel
over all four levels: a thread writes `_CT` channels of one output bin, reading its box's level in place, so neither
the levels nor the boxes are regrouped per level and nothing is copied to the CPU. The samples are accumulated in
float32 whatever the levels' dtype.

The 1 x 1 stride-1 convolutions with more than 32 outputs are GEMMs over the `[H * W, C]` view (`mx.addmm`, 0.5-0.8 of
`mx.conv2d`'s time; a weight of 2^20 elements stays a convolution, as in `shared.mlx_graph.conv1x1_ok`), the 2 x 2
stride-2 deconvolution is a GEMM to its four output phases, and the mask predictor is evaluated for the predicted
class only.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, to_mx, to_torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from .net import Arch

__all__ = ["LAYOUT", "Net", "prepare"]

LAYOUT = "v1"  # version of `prepare`'s parameter layout (for the conversion cache's file name)

_BU, _RPN, _ROI = "backbone.bottom_up", "proposal_generator.rpn_head", "roi_heads"
FP32 = (f"{_ROI}.box_predictor", f"{_ROI}.mask_head.predictor.b")  # evaluated (and cached) in float32
_DECONV = f"{_ROI}.mask_head.deconv"
_FC1 = f"{_ROI}.box_head.fc1"
_GEMM_MIN_OUT = 33  # a 1 x 1 convolution with fewer outputs is faster as `mx.conv2d` (its narrow-output path)
_GEMM_MAX_WEIGHT = 1 << 20  # ... and a weight of this many elements leaves more in MLX's buffer cache as a GEMM
_CT = 4  # channels a ROIAlign thread writes (not tuned; every channel count here is a multiple)
_TILE = 8  # threads per threadgroup side
_MIN_ELEMENTS = 8  # MLX hands a kernel shorter arrays in Metal's `constant` address space, a different pointer type


def prepare(sd: dict[str, torch.Tensor], a: Arch) -> dict[str, mx.array]:
    """The float32 parameters of the MLX network, from the folded state dict (`net.fold_frozen_bn`).

    `<name>.w` / `<name>.b`: a convolution's weight `[O, kh, kw, I]` and bias; `<name>.m` / `<name>.b`: a linear
    layer's weight `[in, out]` (fc1's inputs in the pooled features' `[side, side, C]` order) and bias, and the
    deconvolution as the GEMM `[I, 2 * 2 * O]` to its output phases (`ky`, `kx`, channel), the bias repeated.
    """
    p: dict[str, np.ndarray] = {}
    for key, value in sd.items():
        if not key.endswith(".weight"):
            continue
        name = key.removesuffix(".weight")
        w, b = value.float().numpy(), sd[f"{name}.bias"].float().numpy()
        if name == _DECONV:  # [I, O, 2, 2]
            p[f"{name}.m"], p[f"{name}.b"] = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1), np.tile(b, 4)
        elif w.ndim == 4:  # noqa: PLR2004 - a convolution
            p[f"{name}.w"], p[f"{name}.b"] = w.transpose(0, 2, 3, 1), b
        else:
            if name == _FC1:
                w = w.reshape(len(w), a.fpn_dim, a.box_pool, a.box_pool).transpose(0, 2, 3, 1).reshape(len(w), -1)
            p[f"{name}.m"], p[f"{name}.b"] = w.T, b
    return {k: mx.array(np.ascontiguousarray(v, dtype=np.float32)) for k, v in p.items()}


_ROI_ALIGN = """
    // one thread: CT channels of one output bin of one box
    const int C = d[0], S = d[1], R = d[2];
    uint cg = thread_position_in_grid.x, bin = thread_position_in_grid.y, r = thread_position_in_grid.z;
    if (int(cg) * CT >= C || int(bin) >= S * S || int(r) >= R) return;
    const int l = lv[r], H = d[4 + 2 * l], W = d[5 + 2 * l];
    const device T* f = (l == 0 ? f0 : l == 1 ? f1 : l == 2 ? f2 : f3) + cg * CT;
    const float scale = 1.0f / float(4 << l);
    const float x1 = rois[4 * r] * scale - 0.5f, y1 = rois[4 * r + 1] * scale - 0.5f;
    const float rw = rois[4 * r + 2] * scale - 0.5f - x1, rh = rois[4 * r + 3] * scale - 0.5f - y1;
    const float bw = rw / float(S), bh = rh / float(S);
    const int gw = int(ceil(bw)), gh = int(ceil(bh));
    const int ph = int(bin) / S, pw = int(bin) % S;
    float acc[CT];
    for (int c = 0; c < CT; ++c) acc[c] = 0.0f;
    for (int iy = 0; iy < gh; ++iy) {
        float y = y1 + float(ph) * bh + (float(iy) + 0.5f) * bh / float(gh);
        if (y < -1.0f || y > float(H)) continue;
        y = max(y, 0.0f);
        int yl = int(y), yh = yl + 1;
        if (yl >= H - 1) { yl = yh = H - 1; y = float(yl); }
        const float ly = y - float(yl), hy = 1.0f - ly;
        for (int ix = 0; ix < gw; ++ix) {
            float x = x1 + float(pw) * bw + (float(ix) + 0.5f) * bw / float(gw);
            if (x < -1.0f || x > float(W)) continue;
            x = max(x, 0.0f);
            int xl = int(x), xh = xl + 1;
            if (xl >= W - 1) { xl = xh = W - 1; x = float(xl); }
            const float lx = x - float(xl), hx = 1.0f - lx;
            const device T* a = f + (size_t(yl) * W + xl) * C;
            const device T* b = f + (size_t(yl) * W + xh) * C;
            const device T* c2 = f + (size_t(yh) * W + xl) * C;
            const device T* e = f + (size_t(yh) * W + xh) * C;
            const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            for (int c = 0; c < CT; ++c)
                acc[c] += w1 * float(a[c]) + w2 * float(b[c]) + w3 * float(c2[c]) + w4 * float(e[c]);
        }
    }
    const float inv = 1.0f / float(max(gh * gw, 1));
    device T* o = out + (size_t(r) * S * S + bin) * C + cg * CT;
    for (int c = 0; c < CT; ++c) o[c] = T(acc[c] * inv);
"""


@functools.cache
def _roi_align_kernel() -> Callable[..., list[mx.array]]:
    return metal_kernel(
        name="rcnn_roi_align",
        input_names=["f0", "f1", "f2", "f3", "rois", "lv", "d"],
        output_names=["out"],
        source=_ROI_ALIGN.replace("CT", str(_CT)),
    )


def roi_align(levels: list[mx.array], boxes: torch.Tensor, lvl: torch.Tensor, side: int) -> mx.array:
    """detectron2 `ROIPooler` over P2..P5 (`[1, H, W, C]`, stride 4 on): `[R, 4]` boxes -> `[R, side, side, C]`.

    `lvl` is each box's level (`model.box_levels`); `R` is at least 1 and `C` a multiple of `_CT`.
    """
    r, c = boxes.shape[0], levels[0].shape[-1]
    rows = max(r, _MIN_ELEMENTS)
    rois = np.zeros((rows, 4), np.float32)
    rois[:r] = boxes.float().numpy()
    lv = np.zeros(rows, np.int32)
    lv[:r] = lvl.numpy()
    d = mx.array([c, side, r, 0, *(n for f in levels for n in f.shape[1:3])], dtype=mx.int32)
    dtype = levels[0].dtype
    grid = (-(-(c // _CT) // _TILE) * _TILE, -(-(side * side) // _TILE) * _TILE, r)
    return _roi_align_kernel()(
        inputs=[*levels, mx.array(rois), mx.array(lv), d],
        template=[("T", dtype)],
        grid=grid,
        threadgroup=(_TILE, _TILE, 1),
        output_shapes=[(r, side, side, c)],
        output_dtypes=[dtype],
    )[0]


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


class Net:
    """The MLX network: the converted weights and the compiled stages of the dense pass."""

    def __init__(self, p32: dict[str, mx.array], a: Arch, dtype: str) -> None:
        """`p32` is `prepare`'s output, `a` the card's network shape, `dtype` the convolutions' and FCs' dtype."""
        self.a = a
        self.dtype = dt = getattr(mx, dtype)
        # stride on the first 1 x 1 convolution (and the shortcut) of res3..res5's first block
        self.strided = {f"{_BU}.res{i + 2}.0.{c}" for i in range(1, len(a.blocks)) for c in ("conv1", "shortcut")}
        p: dict[str, mx.array] = {}
        for key, v in p32.items():
            name = key[:-2]
            if key.endswith(".w") and v.shape[1] == 1 and self._gemm(name, v):
                p[f"{name}.m"] = v[:, 0, 0, :].T.astype(dt)
            else:
                p[key] = v if key.startswith(FP32) else v.astype(dt)
        self.p = p
        mx.eval(p)
        names = [f"{_BU}.res{i + 2}.{j}" for i, n in enumerate(a.blocks) for j in range(n)]
        self._stem = mx.compile(self._stem_fn)
        self._blocks = [mx.compile(functools.partial(self._block, name=n, first=n.endswith(".0"))) for n in names]
        levels = range(2, len(a.blocks) + 2)
        self._lateral = {i: mx.compile(functools.partial(self._lateral_fn, level=i)) for i in levels}
        self._output = {i: mx.compile(functools.partial(self._conv, name=f"backbone.fpn_output{i}")) for i in levels}
        self._rpn = mx.compile(self._rpn_fn)

    def _gemm(self, name: str, w: mx.array) -> bool:
        """Whether the 1 x 1 convolution `name` (`[O, 1, 1, I]`) runs as a GEMM."""
        o, i = w.shape[0], w.shape[-1]
        if name in self.strided or name == f"{_ROI}.mask_head.predictor":  # the latter: one filter per detection
            return False
        return o >= _GEMM_MIN_OUT and o * i < _GEMM_MAX_WEIGHT

    def _conv(self, x: mx.array, name: str) -> mx.array:
        """The convolution `name` (with its bias) of NHWC `x`."""
        b = self.p[f"{name}.b"]
        m = self.p.get(f"{name}.m")
        if m is not None:
            n, h, w, c = x.shape
            return mx.addmm(b, x.reshape(-1, c), m).reshape(n, h, w, -1)
        wt = self.p[f"{name}.w"]
        return mx.conv2d(x, wt, stride=2 if name in self.strided else 1, padding=wt.shape[1] // 2) + b

    def _stem_fn(self, x: mx.array) -> mx.array:
        """The stem on the network input `[1, 3, H, W]` (float32): convolution, ReLU, max pooling, NHWC."""
        x = x.transpose(0, 2, 3, 1).astype(self.dtype)
        w = self.p[f"{_BU}.stem.conv1.w"]
        x = _relu(mx.conv2d(x, w, stride=2, padding=3) + self.p[f"{_BU}.stem.conv1.b"])
        # 3 x 3 stride-2 max pooling with padding 1: zero padding is exact after the ReLU
        n, h, wd, c = x.shape
        x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        windows = mx.as_strided(
            x,
            (n, (h + 1) // 2, (wd + 1) // 2, 3, 3, c),
            ((h + 2) * (wd + 2) * c, 2 * (wd + 2) * c, 2 * c, (wd + 2) * c, c, 1),
        )
        return mx.max(windows, axis=(3, 4))

    def _block(self, x: mx.array, name: str, first: bool) -> mx.array:  # noqa: FBT001 - a partial's keyword
        """The bottleneck block `name`."""
        out = _relu(self._conv(_relu(self._conv(x, f"{name}.conv1")), f"{name}.conv2"))
        return _relu(self._conv(out, f"{name}.conv3") + (self._conv(x, f"{name}.shortcut") if first else x))

    def _lateral_fn(self, stage: mx.array, *prev: mx.array, level: int) -> mx.array:
        """The FPN's top-down map at `level` (2..5): the lateral 1 x 1 [+ the nearest 2x upsampling of `prev`]."""
        x = self._conv(stage, f"backbone.fpn_lateral{level}")
        return x + mx.repeat(mx.repeat(prev[0], 2, axis=1), 2, axis=2) if prev else x

    def _rpn_fn(self, f: mx.array) -> tuple[mx.array, mx.array]:
        """The RPN head's objectness logits and anchor deltas of one level (flat, in the run dtype)."""
        t = _relu(self._conv(f, f"{_RPN}.conv"))
        obj = self._conv(t, f"{_RPN}.objectness_logits").reshape(-1)
        return obj, self._conv(t, f"{_RPN}.anchor_deltas").reshape(-1)

    def _forward(self, x: mx.array) -> tuple[mx.array, ...]:
        """P2..P5, and the RPN head's objectness logits and anchor deltas of P2..P6 (float32, flat, level-major).

        Compiled steps (the stem, each block, each FPN lateral and output conv, the RPN head per level), each queued
        while the one before it finishes (one in flight); the FPN's and RPN head's steps also wait for themselves.
        MLX returns a command buffer's transients only when it completes: one compiled function over the whole pass
        held every block's intermediates at once, and one over the FPN or the RPN head every level's 3 x 3 Winograd
        transforms (800 x 1344 float16: network peak 745 MB, the graph translation 432 MB; stepped, 412 MB).
        """
        prev = None

        def step(y: Any, *, wait: bool = False) -> Any:  # noqa: ANN401 - an array or a tuple of them
            nonlocal prev
            mx.async_eval(y)
            if prev is not None:
                mx.eval(prev)
            prev = None if wait else y
            if wait:
                mx.eval(y)
            return y

        x = step(self._stem(x))
        stages, ends = [], {sum(self.a.blocks[: i + 1]) - 1 for i in range(len(self.a.blocks))}
        for k, block in enumerate(self._blocks):
            x = step(block(x))
            if k in ends:
                stages.append(x)
        top, feats, down = len(stages) + 1, [], ()
        for level in range(top, 1, -1):
            down = (step(self._lateral[level](stages[level - 2], *down), wait=True),)
            feats.insert(0, step(self._output[level](down[0]), wait=True))
        del stages, down
        feats.append(feats[-1][:, ::2, ::2])  # `LastLevelMaxPool`: 1 x 1, stride 2
        heads = [step(self._rpn(f), wait=True) for f in feats]
        obj = mx.concatenate([o for o, _ in heads]).astype(mx.float32)
        return (*feats[:-1], obj, mx.concatenate([d for _, d in heads]).astype(mx.float32))

    def dense(self, x: torch.Tensor) -> tuple[list[mx.array], list[tuple[torch.Tensor, torch.Tensor, tuple[int, int]]]]:
        """The network input `[1, 3, H, W]` (float32, CPU) -> P2..P5 (on the GPU) and `model._top_proposals`' maps."""
        *levels, obj, deltas = self._forward(to_mx(x.float()))
        mx.eval(obj, deltas)
        obj_t, deltas_t = to_torch(obj), to_torch(deltas).view(-1, 4)
        maps, at = [], 0
        sizes = [tuple(f.shape[1:3]) for f in levels]
        sizes.append(((sizes[-1][0] + 1) // 2, (sizes[-1][1] + 1) // 2))
        for h, w in sizes:
            n = h * w * self.a.num_anchors
            maps.append((obj_t[at : at + n], deltas_t[at : at + n], (h, w)))
            at += n
        return list(levels), maps

    def _linear(self, x: mx.array, name: str) -> mx.array:
        return mx.addmm(self.p[f"{name}.b"], x, self.p[f"{name}.m"])

    def box_head(
        self, levels: list[mx.array], proposals: torch.Tensor, lvl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """detectron2 box head and predictor on `[R, 4]` proposals: class logits `[R, K + 1]`, deltas `[R, 4 K]`."""
        k = self.a.num_classes
        if proposals.shape[0] == 0:
            return proposals.new_zeros(0, k + 1), proposals.new_zeros(0, 4 * k)
        x = roi_align(levels, proposals, lvl, self.a.box_pool)
        x = _relu(self._linear(x.reshape(x.shape[0], -1), f"{_ROI}.box_head.fc1"))
        x = _relu(self._linear(x, f"{_ROI}.box_head.fc2")).astype(mx.float32)
        heads = [self._linear(x, f"{_ROI}.box_predictor.{n}") for n in ("cls_score", "bbox_pred")]
        out = to_torch(mx.concatenate(heads, axis=1))  # one copy
        return out[:, : k + 1], out[:, k + 1 :]

    def mask_head(
        self, levels: list[mx.array], boxes: torch.Tensor, lvl: torch.Tensor, classes: torch.Tensor
    ) -> torch.Tensor:
        """detectron2 mask head + `mask_rcnn_inference`: `[N, 1, 2 S, 2 S]` probabilities of the predicted classes."""
        n, side = boxes.shape[0], self.a.mask_pool
        if n == 0:
            return boxes.new_zeros(0, 1, 2 * side, 2 * side)
        x = roi_align(levels, boxes, lvl, side)
        for i in range(4):
            x = _relu(self._conv(x, f"{_ROI}.mask_head.mask_fcn{i + 1}"))
        c = x.shape[-1]
        # the deconvolution's four output phases per input position; the pixels stay in (y, x, ky, kx) order, which
        # the per-pixel predictor does not care about
        x = _relu(self._linear(x.reshape(-1, c), _DECONV)).reshape(n, side * side * 4, c)
        cls = mx.array(classes.numpy())
        w = self.p[f"{_ROI}.mask_head.predictor.w"][cls].reshape(n, c, 1)  # the predicted class's filter
        logits = (x @ w).astype(mx.float32).reshape(n, side, side, 2, 2)
        logits = logits + self.p[f"{_ROI}.mask_head.predictor.b"][cls].reshape(n, 1, 1, 1, 1)
        probs = mx.sigmoid(logits.transpose(0, 1, 3, 2, 4).reshape(n, 1, 2 * side, 2 * side))
        return to_torch(probs)
