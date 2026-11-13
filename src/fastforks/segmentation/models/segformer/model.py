"""SegFormer runners: the transformers network with the authors' weights, and its portable / kernels forward passes.

Test-time protocol of the ADE20K checkpoints (mmseg `test_pipeline`, the HF image processor's post-processing):
resize keeping the aspect ratio to fit `2048 x short` (bilinear), ImageNet normalisation, one forward pass of the
whole image, logits (1/4 resolution) resized bilinearly to the network input size, then to the image size (mmseg).
The fine-tuned Hugging Face checkpoints (clothes, face parsing; `resize = ["fixed", 512, 512]`) follow their model
cards: `SegformerImageProcessor` (PIL bilinear resize to 512 x 512, ImageNet normalisation), logits resized bilinearly
straight to the image size.

Both passes run the transformers modules' weights with the same math, only precision, layout and association change:
fp16 weights and activations on CUDA, tokens kept as `[B, H, W, C]` (so token <-> conv-map switches are views), key
and value projections merged into one GEMM, SDPA attention, and the all-MLP head re-associated: `linear_fuse(cat(up(
proj_i(f_i))))` equals `sum_i up((W_i proj_i) f_i)` (bilinear upsampling commutes with per-pixel affine maps), so
each stage's projection and its slice of the fuse layer (and the batch norm) fold into one GEMM at the stage's own
resolution. `kernels` runs the whole network with fp16 accumulation in the cuBLAS GEMMs and the Mix-FFN depthwise 3x3 +
GELU as one kernel. The encoder linears stay cuBLAS: measured per layer, the fused CUTLASS GEMM (`lin`) was slower
than cuBLAS with fp16 accumulation at every SegFormer-B0 and B5 linear shape (small K / N; the bias / residual
epilogue does not make up for it), so those layers are a documented per-layer dispatch to the torch op, not a missing
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._optimize import fp16_accumulate
from fastforks._optimize import ops as fvm
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.seginfer import pretrained, resize, target_size, upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

# CUDA graphs (kernels): the eager forward is host-bound (~10 small launches per block); a repeated input shape
# replays a graph
_GRAPH_PIXELS = 1 << 62  # fixed: every size
_GRAPHS = 2  # fixed: graph cache policy
_GELU = 5  # fixed: fvm activation code of exact (erf) GELU, the Mix-FFN activation of every SegFormer
_MIN_SIDE = 29  # fixed: smallest network input side (stage 1: 1/4 grid >= 8, the spatial-reduction kernel)
_Hf = Any  # transformers modules: attribute access the type checker cannot follow


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: transformers class, `"SegformerForSemanticSegmentation"`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), e.g. `["mmseg", 2048, 512]`.
    """

    arch: str
    resize: tuple[str, int, int]


class _Net(torch.nn.Module):
    """The transformers `SegformerForSemanticSegmentation` modules run as `[B, H, W, C]` tokens with a folded head."""

    head_b: torch.Tensor
    cls_w: torch.Tensor
    cls_b: torch.Tensor

    def __init__(self, model: _Hf, dtype: torch.dtype, *, kernels: bool) -> None:
        super().__init__()
        self.model = model  # folds below are taken from the float32 weights; `.to(dtype)` at the end
        self.kernels = kernels
        self.stages: list[Any] = list(model.segformer.stages)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.to(memory_format=torch.channels_last)
        for st in self.stages:
            for b in st.blocks:
                a, mlp = b.attention, b.mlp
                kv_w = torch.cat([a.k_proj.weight, a.v_proj.weight])
                kv_b = torch.cat([a.k_proj.bias, a.v_proj.bias])
                bufs = {"_kv_w": kv_w, "_kv_b": kv_b}
                if kernels:
                    dw = mlp.dwconv.dwconv.weight  # [C, 1, 3, 3] -> [3, 3, C]
                    bufs["_dw"] = dw.detach()[:, 0].permute(1, 2, 0).contiguous()
                for k, v in bufs.items():
                    b.register_buffer(k, v.detach(), persistent=False)
        self._fold_head(model.decode_head)
        self.to(dtype)

    def _fold_head(self, h: _Hf) -> None:
        c = h.config.decoder_hidden_size
        n = len(h.linear_projections)
        wf = h.linear_fuse.weight.double().view(c, n * c)  # input: cat of the stages in reverse order
        bn = h.batch_norm
        sc = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        bias = bn.bias.double() - bn.running_mean.double() * sc
        for i, lp in enumerate(h.linear_projections):
            j = n - 1 - i
            wi = wf[:, j * c : (j + 1) * c]
            self.register_buffer(f"head_w{i}", (sc[:, None] * (wi @ lp.proj.weight.double())).float(), persistent=False)
            bias = bias + sc * (wi @ lp.proj.bias.double())
        self.register_buffer("head_b", bias.float(), persistent=False)
        self.register_buffer("cls_w", h.classifier.weight.detach().flatten(1).clone(), persistent=False)
        self.register_buffer("cls_b", h.classifier.bias.detach().clone(), persistent=False)

    @staticmethod
    def _reduce(a: _Hf, y: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if a.sequence_reduction_ratio <= 1:
            return y
        r = a.sequence_reduction
        yr = r.sequence_reduction(y.unflatten(1, (h, w)).permute(0, 3, 1, 2))  # channels-last view
        return r.layer_norm(yr.permute(0, 2, 3, 1).flatten(1, 2))

    def _block(self, b: _Hf, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        a, mlp = b.attention, b.mlp
        nh, hd = a.num_attention_heads, a.head_dim
        y = b.layernorm_before(x)
        q = F.linear(y, a.q_proj.weight, a.q_proj.bias).unflatten(-1, (nh, hd)).transpose(1, 2)
        kv = F.linear(self._reduce(a, y, h, w), b._kv_w, b._kv_b).unflatten(-1, (2, nh, hd))  # noqa: SLF001
        o = F.scaled_dot_product_attention(q, kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2), scale=a.scaling)
        x = x + F.linear(o.transpose(1, 2).flatten(2), a.o_proj.weight, a.o_proj.bias)
        y = F.linear(b.layernorm_after(x), mlp.fc1.weight, mlp.fc1.bias)
        y = mlp.dwconv.dwconv(y.unflatten(1, (h, w)).permute(0, 3, 1, 2))
        y = F.gelu(y.permute(0, 2, 3, 1).flatten(1, 2))
        return x + F.linear(y, mlp.fc2.weight, mlp.fc2.bias)

    def _block_kernels(self, b: _Hf, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        a, mlp = b.attention, b.mlp
        nh, hd = a.num_attention_heads, a.head_dim
        y = b.layernorm_before(x)
        q = F.linear(y, a.q_proj.weight, a.q_proj.bias).unflatten(-1, (nh, hd)).transpose(1, 2)
        kv = F.linear(self._reduce(a, y, h, w), b._kv_w, b._kv_b).unflatten(-1, (2, nh, hd))  # noqa: SLF001
        o = F.scaled_dot_product_attention(q, kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2), scale=a.scaling)
        x = x + F.linear(o.transpose(1, 2).flatten(2), a.o_proj.weight, a.o_proj.bias)
        y = F.linear(b.layernorm_after(x), mlp.fc1.weight, mlp.fc1.bias)
        y = y.unflatten(1, (h, w)).permute(0, 3, 1, 2)  # channels-last view
        y = fvm.dwconv(y, b._dw, mlp.dwconv.dwconv.bias, [1, 1], [1, 1], _GELU)  # noqa: SLF001
        return x + F.linear(y.permute(0, 2, 3, 1).flatten(1, 2), mlp.fc2.weight, mlp.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised image `[B, 3, H, W]` -> logits `[B, H/4, W/4, classes]` (model dtype)."""
        x = x.to(self.cls_w.dtype).contiguous(memory_format=torch.channels_last)
        feats = []
        for st in self.stages:
            pe = st.patch_embeddings
            e = pe.proj(x)
            bsz, c, h, w = e.shape
            t = pe.layer_norm(e.permute(0, 2, 3, 1).reshape(bsz, h * w, c))
            for b in st.blocks:
                t = self._block_kernels(b, t, h, w) if self.kernels else self._block(b, t, h, w)
            t = st.layer_norm(t)
            feats.append(t.view(bsz, h, w, c))
            x = feats[-1].permute(0, 3, 1, 2)  # channels-last view for the next patch embedding
        h4, w4 = feats[0].shape[1:3]
        acc = None
        for i, f in enumerate(feats):
            y = F.linear(f, getattr(self, f"head_w{i}"))  # at the stage's resolution
            if i and acc is not None and self._up_add(acc, y):
                continue
            if i:
                y = F.interpolate(y.permute(0, 3, 1, 2), (h4, w4), mode="bilinear", align_corners=False)
                y = y.permute(0, 2, 3, 1)
            acc = y.contiguous() if acc is None else acc.add_(y)
        assert acc is not None  # noqa: S101 - four stages
        return F.linear(F.relu_(acc.add_(self.head_b)), self.cls_w, self.cls_b)

    def _up_add(self, acc: torch.Tensor, y: torch.Tensor) -> bool:
        """Kernels: `acc += resize(y)` (fp16 NHWC) as one `seg_up` kernel, PyTorch's fp16 resize then add."""
        h4, w4 = acc.shape[1:3]
        if not (self.kernels and acc.dtype == y.dtype == torch.float16 and y.shape[-1] % 2 == 0 and y.is_contiguous()):
            return False
        from fastforks.kernels import seg_up  # noqa: PLC0415 - kernels backend only

        seg_up.up_add(y, acc, acc, (h4, w4), relu=False)
        return True


class Portable(Runner):
    """Plain PyTorch (SDPA attention, folded head): float32, or fp16 with channels-last convs on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _Net(pretrained(ctx), ctx.torch_dtype, kernels=False).to(ctx.device)
        self.acc = False
        self.fused = False  # kernels: logits resize and argmax by `seg_up`

    def _scores(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
        """Logits at 1/4 size `[B, classes, h, w]` and their resize steps (to the input size, then the image)."""
        h, w = image.shape[-2:]
        size = target_size(h, w, self.ctx.params.resize)
        size = (max(size[0], _MIN_SIDE), max(size[1], _MIN_SIDE))  # very thin images (aspect > 70): stretched
        fixed = self.ctx.params.resize[0] == "fixed"  # HF processor: PIL resize, logits straight to the image size
        x = scale(resize(image, size, pil=fixed), self.ctx.io)
        s = self._forward(x).permute(0, 3, 1, 2)
        steps = ((h, w),) if fixed else (size, (h, w))
        return (s if self.fused else s.contiguous()), steps  # NCHW: faster PyTorch bilinear upsampling

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.acc:
            with fp16_accumulate():
                return self.net(x)
        return self.net(x)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        s, sizes = self._scores(image)
        return upsample(s, sizes, fused=self.fused)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, classes, H, W]` of a batch."""
        return {"logits": self._run(image)}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no full-size logits)."""
        s, sizes = self._scores(image)
        return upsample_argmax(s, sizes, fused=self.fused)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, classes, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


class Kernels(Portable):
    """fp16-accumulate GEMMs, depthwise conv + GELU in one kernel, fused upsampling; a CUDA graph per repeated shape."""

    def __init__(self, ctx: Context) -> None:  # noqa: D107 - see Portable
        Runner.__init__(self, ctx)
        self.net = _Net(pretrained(ctx), ctx.torch_dtype, kernels=True).to(ctx.device)
        self.acc = True
        self.fused = True
        self._graphs = SmallShapeGraphs(super()._forward, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._graphs(x)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        self._graphs.release()

    module = Runner.module


RUNNERS = {"portable": Portable, "kernels": Kernels}
