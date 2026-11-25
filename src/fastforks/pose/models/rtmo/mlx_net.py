"""RTMO's network on Apple MLX, written in `mlx.core` operations (`MlxRTMO(net, dtype, size)`).

Letterboxed images -> the head's per-cell outputs: the maths of `net.RTMO.forward` (the folded checkpoint), channels
last, from the layers of `fastforks.shared.cspnext.mlx_net` (SiLU in the run dtype, the SPP's max pools, the graph
translator's conv dispatch): the 6 x 6 stem, CSPDarknet, the hybrid encoder (one post-norm transformer layer on the
stride-32 map, the CSPRep top-down and bottom-up passes) and the head in the run dtype; the person logit's 1 x 1 conv
in float32 (as the exported network for MLX), the outputs float32. The DCC stays PyTorch (the kept people only).
Written out instead of replayed from an exported graph: the export and its translation were most of the first call.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx, to_mx
from fastforks._core.patch import base_type
from fastforks.shared.cspnext import mlx_net as csp

from .net import FFN, HEADS, HIDDEN, STRIDES

if TYPE_CHECKING:
    from fastforks.shared.cspnext.mlx_net import Layer

__all__ = ["MlxRTMO"]

GRAPH = "portable"  # the conv dispatch of `cspnext.mlx_net` (RTMO's port has no kernels mode)
LN_EPS = 1e-5  # fixed: `nn.LayerNorm` default (the encoder's norms)


def _linear(layer: Any, d: mx.Dtype) -> Layer:
    w, b = csp.arr(layer.weight.t(), d), csp.arr(layer.bias, d)
    return lambda x: x @ w + b


def _encoder(enc: Any, d: mx.Dtype, pos: mx.array) -> Layer:
    """`Encoder` on tokens `[B, N, 256]`; `pos` `[1, N, 256]`."""
    c = HIDDEN
    wqk, bqk = csp.arr(enc.in_proj_weight[: 2 * c].t(), d), csp.arr(enc.in_proj_bias[: 2 * c], d)
    wv, bv = csp.arr(enc.in_proj_weight[2 * c :].t(), d), csp.arr(enc.in_proj_bias[2 * c :], d)
    out, fc1, fc2 = _linear(enc.out_proj, d), _linear(enc.fc1, d), _linear(enc.fc2, d)
    norms = [(csp.arr(n.weight, d), csp.arr(n.bias, d)) for n in (enc.norm1, enc.norm2)]
    assert enc.fc1.out_features == FFN  # noqa: S101 - the reference's width

    def heads(t: mx.array) -> mx.array:
        b, n, _ = t.shape
        return t.reshape(b, n, HEADS, c // HEADS).transpose(0, 2, 1, 3)

    def apply(x: mx.array) -> mx.array:
        b, n, _ = x.shape
        q, k = mx.split((x + pos) @ wqk + bqk, 2, axis=-1)
        v = x @ wv + bv
        y = mx.fast.scaled_dot_product_attention(heads(q), heads(k), heads(v), scale=(c // HEADS) ** -0.5)
        x = mx.fast.layer_norm(x + out(y.transpose(0, 2, 1, 3).reshape(b, n, c)), *norms[0], LN_EPS)
        h = fc1(x)
        h = h * (1 + mx.erf(h * 2**-0.5)) / 2  # exact GELU (`F.gelu`)
        return mx.fast.layer_norm(x + fc2(h), *norms[1], LN_EPS)

    return apply


def _rep(layer: Any, d: mx.Dtype) -> Layer:
    """`CSPRep`: conv3(reps(conv1(x)) + conv2(x))."""
    conv1, conv2 = csp.conv_act(layer.conv1, d, GRAPH), csp.conv_act(layer.conv2, d, GRAPH)
    reps = csp.chain(*(csp.conv_act(m, d, GRAPH) for m in layer.bottlenecks))
    conv3 = csp.conv_act(layer.conv3, d, GRAPH) if layer.conv3 is not None else None

    def apply(x: mx.array) -> mx.array:
        y = reps(conv1(x)) + conv2(x)
        return y if conv3 is None else conv3(y)

    return apply


def _stage(stage: Any, d: mx.Dtype) -> Layer:
    """The stride-2 conv, the SPP (last stage), the `CSPLayer`."""
    build = {"CSPLayer": csp.csp_layer, "SPP": csp.spp}
    return csp.chain(*(build.get(base_type(m).__name__, csp.conv_act)(m, d, GRAPH) for m in stage))


class MlxRTMO:
    """Letterboxed NHWC images in 0..255 (float32) -> float32 per-cell outputs (`net.Head.forward`).

    Class logits `[B, N]`, boxes `[B, N, 4]`, visibility logits `[B, N, K]`, pose vectors `[B, N, C]`; the cells row
    by row, stride 16 first.
    """

    def __init__(self, net: Any, dtype: str, size: int) -> None:
        """The MLX layers of `net` (an `RTMO` from the folded checkpoint) in `dtype` for `size x size` inputs."""
        d = mx.float16 if dtype == "float16" else mx.float32
        bb, neck, head = net.backbone, net.neck, net.head
        stem = csp.conv_act(bb.stem, d, GRAPH)
        stages = [_stage(getattr(bb, f"stage{i}"), d) for i in range(1, 5)]
        proj = [csp.conv(m, d, GRAPH) for m in neck.input_proj]
        side = size // STRIDES[-1]
        encoder = _encoder(neck.encoder, d, to_mx(neck.positions(side, side).float()).astype(d))
        lateral = [csp.conv_act(m, d, GRAPH) for m in neck.lateral_convs]
        fpn = [_rep(m, d) for m in neck.fpn_blocks]
        down = [csp.conv_act(m, d, GRAPH) for m in neck.downsample_convs]
        pan = [_rep(m, d) for m in neck.pan_blocks]
        projector = [csp.conv(m, d, GRAPH) for m in neck.projector]
        levels = [
            (
                csp.chain(*(csp.conv_act(m, d, GRAPH) for m in head.conv_cls[i])),
                csp.arr(head.out_cls[i].weight[:, :, 0, 0].t(), mx.float32),  # the person logit in float32
                csp.arr(head.out_cls[i].bias, mx.float32),
                csp.chain(*(csp.conv_act(m, d, GRAPH) for m in head.conv_pose[i])),
                [csp.conv(m, d, GRAPH) for m in (head.out_bbox[i], head.out_kpt_vis[i], head.out_pose[i])],
            )
            for i in range(len(STRIDES))
        ]

        def top5(c4: mx.array) -> tuple[mx.array, mx.array]:
            p5 = proj[2](stages[3](c4))
            b, h, w, c = p5.shape
            return proj[1](c4), lateral[0](encoder(p5.reshape(b, h * w, c)).reshape(b, h, w, c))

        def head_level(f: mx.array, level: tuple[Any, ...]) -> list[mx.array]:
            cls_tower, wc, bc, pose_tower, preds = level
            cls, reg = mx.split(f, 2, axis=-1)
            pose = pose_tower(reg)
            maps = [cls_tower(cls).astype(mx.float32) @ wc + bc, *(p(pose) for p in preds)]
            return [m.reshape(m.shape[0], -1, m.shape[-1]).astype(mx.float32) for m in maps]

        # each stage, neck block and head level its own compiled step (`csp.stepped`: bounded transients)
        first = csp.stepped(lambda x: stages[1](stages[0](stem(x.astype(d)))))
        s4, t5 = csp.stepped(stages[2]), csp.stepped(top5)
        t4 = csp.stepped(lambda p4, t5: lateral[1](fpn[0](mx.concatenate((csp.up2(t5), p4), axis=-1))))
        t3 = csp.stepped(lambda x, t4: fpn[1](mx.concatenate((csp.up2(t4), proj[0](x)), axis=-1)))
        b4 = csp.stepped(lambda t3, t4: pan[0](mx.concatenate((down[0](t3), t4), axis=-1)))
        b5 = csp.stepped(lambda o4, t5: pan[1](mx.concatenate((down[1](o4), t5), axis=-1)))
        heads = [csp.stepped(lambda o, i=i: head_level(projector[i](o), levels[i])) for i in range(len(levels))]

        def forward(x: mx.array) -> tuple[mx.array, ...]:
            x = first(x)
            c4 = s4(x)
            p4, f5 = t5(c4)
            f4 = t4(p4, f5)
            f3 = t3(x, f4)
            o4 = b4(f3, f4)
            o5 = b5(o4, f5)
            outs = [h(o) for h, o in zip(heads, (o4, o5), strict=True)]
            cls, box, vis, vec = (mx.concatenate(o, axis=1) for o in zip(*outs, strict=True))
            return cls[..., 0], box, vis, vec

        self.forward = forward

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """The head's per-cell outputs of letterboxed images `[B, S, S, 3]` in 0..255."""
        return csp.steps(self.forward, x)
