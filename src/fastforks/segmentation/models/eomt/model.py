"""EoMT runners: the transformers network with the authors' weights, and its portable / kernels forward passes.

Test-time protocol of the ADE20K semantic checkpoints (the authors' `window_imgs_semantic` / `revert_window_logits_
semantic`, also the transformers image processor's): resize the shorter side to 512 (bilinear), ImageNet
normalisation, square 512 windows along the longer side with the overlap spread evenly, all windows of an image in
one batch; per window the mask logits (1/4 resolution) are resized bilinearly to 512 and the semantic logits are
`sum_q softmax(class_q)[:-1] * sigmoid(mask_q)`; overlapping logits are averaged and resized to the image size.

EoMT is a plain ViT-L/16 (DINOv2 or DINOv3 pre-training) whose last 4 blocks also carry 100 query tokens, then a small
mask head; the released checkpoints switch masked attention off at test time (`attn_mask_probs <= 0`), so every
block is plain self-attention. Both passes run the transformers modules' weights with the same math in another
association or precision:

- q / k / v as one GEMM; LayerScale folded into the output projection and the second MLP layer.
- The residual stream is kept divided by `res_scale` (DINOv3: 16; its stream reaches ~1.5e5, beyond fp16's 65504).
  Every LayerNorm is scale-invariant, so LayerScale / s goes into those weights, the embeddings and queries are scaled
  once and each LayerNorm's eps becomes eps / s^2: exact in real arithmetic.
- DINOv3 RoPE: cos / sin computed once per window size in float32 (as transformers), q / k patch tokens rotated in
  place in the packed GEMM output.
- Upscale blocks channels-last, their LayerNorm2d on the NHWC view; the mask logits are one batched GEMM.

`portable`: SDPA attention, fp16 on CUDA. `kernels`: the `fastbb` flash-attention kernel (head_dim 64, fp32 `q k^T`,
reading q / k / v as strided views of the packed GEMM output) and fp16-accumulate GEMMs.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.half import fp16_accumulate
from fastforks.kernels import fastbb
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap
from fastforks.shared.seginfer import (
    eomt_windows,
    mask_semantic,
    pretrained,
    resize,
    target_size,
    upsample,
    upsample_argmax,
)

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

_Hf = Any  # transformers modules: attribute access the type checker cannot follow
_FLASH = "fh"  # fixed: fastbb flash variant, fp32 q k^T and fp16 p v (fp16 q k^T loses accuracy at 1029 tokens)
_ROPE_SIZES = 8  # fixed: cache policy, RoPE tables kept (one per patch grid; video at one size always hits)
_HEAD_DIM = 64  # fixed: the fastbb flash kernel's head size (EoMT ViT-L: 1024 / 16 heads)


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: transformers class, `"EomtForUniversalSegmentation"` or `"EomtDinov3ForUniversalSegmentation"`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), `["eomt", 512]`.
        window: Side of the square windows (the training crop).
        res_scale: The residual stream is kept divided by this (exact, see the module docstring).
    """

    arch: str
    resize: tuple[str, int]
    window: int
    res_scale: float


def _fp16_accumulate(*, enabled: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if enabled else contextlib.nullcontext()


def _fold_layer(layer: _Hf, s: float) -> None:
    """Packed q / k / v and LayerScale / `s` folded into the output projections, as float32 buffers on `layer`."""
    a, mlp = layer.attention, layer.mlp
    if hasattr(mlp, "gate_proj"):
        msg = "EoMT with a gated MLP is not supported"
        raise ValueError(msg)
    out = a.out_proj if hasattr(a, "out_proj") else a.o_proj
    fc1, fc2 = (mlp.fc1, mlp.fc2) if hasattr(mlp, "fc1") else (mlp.up_proj, mlp.down_proj)
    projs = (a.q_proj, a.k_proj, a.v_proj)
    bufs = {
        "ff_qkv_w": torch.cat([p.weight for p in projs]),
        "ff_qkv_b": torch.cat([p.bias if p.bias is not None else torch.zeros_like(p.weight[:, 0]) for p in projs]),
    }
    for name, lin, ls in (("ff_out", out, layer.layer_scale1), ("ff_fc2", fc2, layer.layer_scale2)):
        g = ls.lambda1.detach().float() / s
        b = lin.bias.detach().float() if lin.bias is not None else torch.zeros_like(g)
        bufs |= {f"{name}_w": lin.weight.detach().float() * g[:, None], f"{name}_b": b * g}
    for k, v in bufs.items():
        layer.register_buffer(k, v.detach().clone(), persistent=False)
    layer.ff_fc1 = fc1
    layer.ff_act = mlp.activation_fn if hasattr(mlp, "activation_fn") else mlp.act_fn
    layer.ff_heads, layer.ff_scale = a.num_heads, a.scaling if hasattr(a, "scaling") else a.scale


class _Net(torch.nn.Module):
    """A transformers EoMT network as a window batch `[B, 3, S, S]` -> mask logits, class logits."""

    def __init__(self, model: _Hf, dtype: torch.dtype, *, res_scale: float, kernels: bool) -> None:
        super().__init__()
        if bool((model.attn_mask_probs > 0).any()):
            msg = "EoMT with masked attention at test time (attn_mask_probs > 0) is not supported"
            raise ValueError(msg)
        self.model, self.cfg, self.kernels, self.s = model, model.config, kernels, float(res_scale)
        if kernels and self.cfg.hidden_size // self.cfg.num_attention_heads != _HEAD_DIM:
            msg = f"the fused attention kernel runs head dim {_HEAD_DIM} only; use mode='portable'"
            raise BackendUnavailableError(msg)
        rope = getattr(model, "rope_embeddings", None)
        inv_freq = None if rope is None else rope.inv_freq.detach().clone().float()
        for layer in model.layers:
            _fold_layer(layer, self.s)
        if self.s != 1:
            for ln in [m for layer in model.layers for m in (layer.norm1, layer.norm2)] + [model.layernorm]:
                ln.eps = ln.eps / self.s**2
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d | torch.nn.ConvTranspose2d):
                m.to(memory_format=torch.channels_last)
        model.to(dtype)
        if rope is not None:
            rope.inv_freq = inv_freq  # RoPE angles in float32, as transformers
        self.rope = rope
        self.flash = fastbb.Flash64(_FLASH)
        self._rope: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (x.shape[-2], x.shape[-1])
        found = self._rope.get(key)
        if found is None:
            assert self.rope is not None  # noqa: S101 - DINOv3 only
            cos, sin = self.rope(x[:1].float())
            dt = self.model.layernorm.weight.dtype
            found = (cos.to(dt)[None, :, None], sin.to(dt)[None, :, None])
            if type(found[0]) is torch.Tensor:  # not a traced one (an MLX export's fake tensor)
                if len(self._rope) >= _ROPE_SIZES:
                    del self._rope[next(iter(self._rope))]
                self._rope[key] = found
        return found

    def _block(self, layer: _Hf, x: torch.Tensor, cs: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        b, n, c = x.shape
        h = layer.ff_heads
        qkv = F.linear(layer.norm1(x), layer.ff_qkv_w, layer.ff_qkv_b).view(b, n, 3, h, c // h)
        if cs is not None:  # rotate the patch tokens of q and k (the last `cos.shape[1]` tokens)
            cos, sin = cs
            t = qkv[:, n - cos.shape[1] :, :2]
            t1, t2 = t.chunk(2, -1)
            t.copy_(t * cos[:, :, None] + torch.cat((-t2, t1), -1) * sin[:, :, None])
        if self.kernels and x.is_cuda and x.dtype == torch.float16:  # kernels: head dim checked at build
            o = self.flash(*qkv.unbind(2), layer.ff_scale).view(b, n, c)
        else:
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
            o = F.scaled_dot_product_attention(q, k, v, scale=layer.ff_scale).transpose(1, 2).reshape(b, n, c)
        x = x + F.linear(o, layer.ff_out_w, layer.ff_out_b)
        y = layer.ff_act(layer.ff_fc1(layer.norm2(x)))
        return x.add_(F.linear(y, layer.ff_fc2_w, layer.ff_fc2_b))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised windows `[B, 3, S, S]` -> mask logits `[B, Q, S/4, S/4]`, class logits `[B, Q, classes + 1]`."""
        m, cfg = self.model, self.cfg
        e = m.embeddings
        dt = m.layernorm.weight.dtype
        x = x.to(dt).contiguous(memory_format=torch.channels_last)
        pe = e.patch_embeddings if self.rope is not None else e.patch_embeddings.projection
        t = pe(x)
        b, c, gh, gw = t.shape
        t = t.permute(0, 2, 3, 1).reshape(b, gh * gw, c)
        cs = None
        if self.rope is None:
            pos = e.position_embeddings.weight
            if pos.shape[0] != gh * gw:
                msg = f"EoMT-DINOv2 runs at its training grid ({pos.shape[0]} patches), got {gh}x{gw}"
                raise ValueError(msg)
            t = t + pos
        else:
            cs = self._cos_sin(x)
        t = torch.cat([e.cls_token.expand(b, -1, -1), e.register_tokens.expand(b, -1, -1), t], 1)
        if self.s != 1:
            t = t * (1 / self.s)
        first = cfg.num_hidden_layers - cfg.num_blocks
        for i, layer in enumerate(m.layers):
            if i == first:
                t = torch.cat([(m.query.weight * (1 / self.s))[None].expand(b, -1, -1), t], 1)
            t = self._block(layer, t, cs)
        t = m.layernorm(t)
        nq = cfg.num_queries
        q = t[:, :nq]
        p = t[:, nq + 1 + cfg.num_register_tokens :].reshape(b, gh, gw, c).permute(0, 3, 1, 2)  # channels-last view
        for blk in m.upscale_block.block:
            p = blk.conv2(blk.activation(blk.conv1(p)))
            ln = blk.layernorm2d
            p = F.layer_norm(p.permute(0, 2, 3, 1), (c,), ln.weight, ln.bias, ln.eps).permute(0, 3, 1, 2)
        h4, w4 = p.shape[-2:]
        masks = torch.bmm(m.mask_head(q), p.permute(0, 2, 3, 1).reshape(b, h4 * w4, c).transpose(1, 2))
        return masks.view(b, nq, h4, w4), m.class_predictor(q)


class Portable(Runner):
    """Plain PyTorch (SDPA attention, folded LayerScale, scaled residual stream): float32, or fp16 on CUDA."""

    dtypes = ("float32", "float16")
    probabilities = True  # class softmax x mask sigmoid, summed over the queries
    batch = None
    Params = Params
    kernels = False

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        self.net = _Net(pretrained(ctx), ctx.torch_dtype, res_scale=p.res_scale, kernels=self.kernels).to(ctx.device)

    def _merged(self, image: torch.Tensor) -> torch.Tensor:
        """Semantic scores of one image `[1, 150, rh, rw]` at the network input size (windows averaged), net dtype."""
        h, w = image.shape[-2:]
        x = scale(resize(image, target_size(h, w, self.ctx.params.resize), pil=True), self.ctx.io)
        s = self.ctx.params.window
        rh, rw = x.shape[-2:]
        wins = [(y, z) for y in eomt_windows(rh, s) for z in eomt_windows(rw, s)]
        crops = torch.cat([x[..., y : y + s, z : z + s] for y, z in wins])
        with _fp16_accumulate(enabled=self.kernels and x.is_cuda and self.ctx.torch_dtype == torch.float16):
            masks, classes = self.net(crops)
        # The resized query logits and accumulated scores are disposable during CUDA inference.
        inplace = x.is_cuda and torch.is_inference_mode_enabled() and not torch.compiler.is_compiling()
        masks = resize(masks, crops.shape[-2:])
        logits = mask_semantic(masks, classes, None, inplace=inplace)
        del masks, classes, x, crops
        if len(wins) == 1 and (rh, rw) == (s, s):
            return logits
        out = logits.new_zeros((1, logits.shape[1], rh, rw))
        count = logits.new_zeros((1, 1, rh, rw))
        for (y, z), lg in zip(wins, logits, strict=True):
            out[..., y : y + s, z : z + s] += lg
            count[..., y : y + s, z : z + s] += 1
        return out.div_(count) if inplace else out / count

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]  # `resize` to the image size (no-op at equal size), fused by `seg_up` in kernels
        return torch.cat([upsample(self._merged(im[None]), ((h, w),), fused=self.kernels) for im in image])

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the scores, class chunk by chunk (no image-size scores)."""
        size = (tuple(image.shape[-2:]),)
        return torch.cat([upsample_argmax(self._merged(im[None]), size, fused=self.kernels) for im in image])

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Semantic logits `[B, 150, H, W]` of a batch (the windows of one image per forward pass)."""
        return {"logits": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


class Kernels(Portable):
    """`fastbb` flash attention (fp32 `q k^T`) on the packed qkv GEMM output, fp16-accumulate GEMMs."""

    kernels = True
    module = Runner.module


class _MlxPortable(Portable):
    """Apple MLX: the folded network written in MLX operations (`mlx_net`), the windows and their merge in MLX.

    Built from the checkpoint's tensors (a dict of meta tensors of the safetensors header on a conversion-cache hit), no
    transformers module. The input's resize stays `seginfer.resize` (PIL's antialiased downscale) on the CPU.
    """

    module = Runner.module  # the MLX network is not exportable
    CACHE_MB: ClassVar[dict[str, int]] = {}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks._core.mlx import mx  # noqa: PLC0415 - imports mlx
        from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD  # noqa: PLC0415
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        paths = ctx.weights()
        config = json.loads(Path(paths[0]).read_text())
        s = float(ctx.params.res_scale)
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda sd: mlx_net.network(sd, config, s, ctx.dtype),
            real=lambda: ctx.state_dict(1, strip=""),
            skeleton=lambda: mlx_port.header_skeleton(paths[1]),
            sources=mlx_net.SOURCES,
        )
        self.net = None
        self._mean, self._std = mx.array(IMAGENET_MEAN, mx.float32), mx.array(IMAGENET_STD, mx.float32)
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def _scores(self, image: torch.Tensor) -> Any:  # noqa: ANN401 - an mx.array
        """float32 MLX scores `[1, rh, rw, 150]` of one image `[1, 3, H, W]` at the network input size."""
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - imports mlx
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        h, w = image.shape[-2:]
        rh, rw = target_size(h, w, self.ctx.params.resize)
        x = to_mx(image.float()).transpose(0, 2, 3, 1)
        if rh < h or rw < w:  # PIL's antialiased downscale (torch's `antialias`), on the CPU
            x = to_mx(resize(image.float(), (rh, rw), pil=True)).transpose(0, 2, 3, 1)
        else:
            x = mlx_seg.resize(x, (rh, rw))
        x = (x - self._mean) / self._std
        s = self.ctx.params.window
        wins = [(y, z) for y in eomt_windows(rh, s) for z in eomt_windows(rw, s)]
        out = mlx_net.scores(self.mlx, x, wins, s)
        mx.eval(out)
        return out

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Semantic logits `[B, 150, H, W]` float32 of a batch (the windows of one image per forward pass)."""
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - imports mlx
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - imports mlx

        h, w = image.shape[-2:]
        out = []
        for im in image:
            y = mlx_seg.resize(self._scores(im[None]), (h, w)).transpose(0, 3, 1, 2)
            out.append(to_torch(y))
        return {"logits": torch.cat(out)}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the scores resized to the image, in one kernel."""
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - imports mlx

        size = tuple(image.shape[-2:])
        return torch.cat([mlx_seg.resize_argmax(self._scores(im[None]), size) for im in image])


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
    "kernels_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable), "kernels_mlx": mlx_wrap.port(Portable)}
