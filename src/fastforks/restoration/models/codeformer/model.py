"""CodeFormer runners: the network of sczhou/CodeFormer and `inference_codeformer.py`'s call on one aligned face.

Upstream normalises the aligned 512 x 512 face crop to [-1, 1], runs `net(x, w=w, adain=True)[0]` and maps the output
from [-1, 1] back (`tensor2img(min_max=(-1, 1))`); face detection, alignment and pasting back are not part of the
network and not part of this family: the input is the aligned crop. `w` is the card's `fidelity` (upstream's
`--fidelity_weight`, 0.5 by default; 0 = the codebook face, 1 = closer to the input).

Changes: `CodeFormer.forward` without the training branches, the code index as `argmax` of the logits (upstream:
`topk(softmax(logits), 1)`, the same index) read from the codebook with an embedding lookup (upstream: a one-hot
float32 matmul), and the 16 x 16 attention blocks of the VQGAN as one `scaled_dot_product_attention` call.
float16: the transformer's logits head runs in float32, so the code indices do not flip on near ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import InputError
from fastforks._core.patch import override
from fastforks.restoration._runner import Crop, Forward, Runner

from .net import CodeFormer
from .net.codeformer_arch import adaptive_instance_normalization
from .net.vqgan_arch import AttnBlock

if TYPE_CHECKING:
    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` (`inference_codeformer.py`'s network and call)."""

    scale: int  # 1: the restored face has the input's size
    size: int  # 512: the aligned face crop (the transformer's 16 x 16 position embedding)
    fidelity: float  # w: 0.5 (upstream default)
    key: str  # checkpoint key ("params_ema")


def _attn(m: AttnBlock, x: torch.Tensor) -> torch.Tensor:
    """`AttnBlock.forward` (single-head attention over the 16 x 16 positions) with one SDPA call."""
    b, c, h, w = x.shape
    t = m.norm(x)
    q, k, v = (conv(t).flatten(2).transpose(1, 2)[:, None] for conv in (m.q, m.k, m.v))
    out = F.scaled_dot_product_attention(q, k, v)[:, 0].transpose(1, 2).reshape(b, c, h, w)
    return x + m.proj_out(out)


def _forward(m: CodeFormer, x: torch.Tensor, w: float) -> torch.Tensor:
    """`CodeFormer.forward(x, w=w, adain=True)[0]` at inference."""
    enc = {}
    taps = {m.fuse_encoder_block[s] for s in m.connect_list}
    for i, block in enumerate(m.encoder.blocks):
        x = block(x)
        if i in taps:
            enc[str(x.shape[-1])] = x
    lq = x
    b = x.shape[0]
    pos = m.position_emb.unsqueeze(1).expand(-1, b, -1)
    q = m.feat_emb(lq.flatten(2).permute(2, 0, 1))
    for layer in m.ft_layers:
        q = layer(q, query_pos=pos)
    ln, fc = m.idx_pred_layer
    logits = F.linear(F.layer_norm(q.float(), *_ln(ln)), fc.get_parameter("weight").float())  # ty: ignore[invalid-argument-type]
    idx = logits.argmax(dim=2).transpose(0, 1)  # [b, 256]: topk(softmax(logits), 1)
    quant = (
        F.embedding(idx, m.quantize.get_submodule("embedding").get_parameter("weight"))
        .view(b, 16, 16, -1)
        .permute(0, 3, 1, 2)
    )
    x = adaptive_instance_normalization(quant.to(lq.dtype), lq)
    fuse = {m.fuse_generator_block[s] for s in m.connect_list}
    for i, block in enumerate(m.generator.blocks):
        x = block(x)
        if i in fuse and w > 0:
            s = str(x.shape[-1])
            x = m.fuse_convs_dict[s](enc[s], x, w)
    return x


def _ln(ln: torch.nn.LayerNorm) -> tuple:
    return ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network (`inference_codeformer.py`'s arguments) and load `params_ema`."""
        super().__init__(ctx)
        p = ctx.params
        self.net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                              connect_list=["32", "64", "128", "256"])  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        for m in self.net.modules():
            if isinstance(m, AttnBlock):
                override(m, _attn)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """No padding: the network takes exactly the aligned `size` x `size` face crop."""
        h, w = x.shape[-2:]
        n = self.ctx.params.size
        if (h, w) != (n, n):
            msg = f"CodeFormer restores an aligned {n} x {n} face crop, got {h} x {w} (resize or align the face first)"
            raise InputError(msg)
        return x, Crop(0, 0, h, w)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored face in [0, 1] (the network works in [-1, 1])."""
        x = image.to(getattr(torch, self.ctx.dtype)) * 2 - 1
        return {"image": (_forward(self.net, x, self.ctx.params.fidelity).float() + 1) / 2}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype, [0, 1] in and out."""
        dtype, w = getattr(torch, self.ctx.dtype), self.ctx.params.fidelity
        return Forward(self.net, lambda net, image: (_forward(net, image.to(dtype) * 2 - 1, w).float() + 1) / 2)


RUNNERS = {"portable": Portable}
