"""PARSeq runners: checkpoint -> network and the portable recogniser.

Inference as upstream `read.py` (`strhub`, `SceneTextDataModule.get_transform`): the whole image resized to
`32 x 128` (Pillow bicubic, bit-exact: `fastforks.shared.pil_resize`), `/ 255`, `(x - 0.5) / 0.5`, greedy
autoregressive decoding with one refinement pass, softmax, the text up to the first end token.
The 94-character charset: digits, Latin letters, ASCII punctuation (no space).

`portable`: the network in plain PyTorch, float32 or float16. On CUDA the second call of a batch size replays the
encoder and 26 autoregressive steps as a CUDA graph (`cuda_graphs`). The refinement uses the same context length as
eager early stopping. `module()` is the plain network and exports unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.upload import device_tensor
from fastforks.ocr._runner import Forward, Runner, levels
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.pil_resize import pil_bicubic, pil_bicubic_cuda

from .net import IMAGE, PARSeq

if TYPE_CHECKING:
    from fastforks.ocr._runner import Context

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


@dataclass(frozen=True)
class Params:
    """Card `[params]`.

    Attributes:
        dim: Embedding width (384 base, 192 tiny).
        enc_heads: Encoder attention heads.
        dec_heads: Decoder attention heads.
    """

    dim: int
    enc_heads: int
    dec_heads: int
    cuda_graphs: bool  # option: replay the full 26-step decoding from a CUDA graph on CUDA (speed only)


def to_input(image: torch.Tensor, *, fused: bool = False) -> torch.Tensor:
    """`[B, 3, H, W]` RGB uint8 (or float in [0, 1]) -> the network input `[B, 3, 32, 128]` float32 in `[-1, 1]`."""
    x = levels(image).flip(1) if image.dtype != torch.uint8 else image
    resize = pil_bicubic_cuda if fused else pil_bicubic
    x = resize(x, IMAGE) if tuple(x.shape[-2:]) != IMAGE else x.float()
    return x.div_(255).sub_(0.5).div_(0.5)


class Portable(Runner):
    """PARSeq in plain PyTorch, float32 or float16."""

    fused_resize = False
    Params = Params
    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        self.dtype = ctx.torch_dtype
        net = PARSeq(p.dim, p.enc_heads, p.dec_heads)
        net.load_state_dict(ctx.state_dict())
        self.net = net.eval().to(ctx.device, self.dtype)
        self._graphs = GraphCache(self._graph_ar, graphs=2) if p.cuda_graphs and ctx.device.type == "cuda" else None

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """All 26 decoding steps, float32 logits, for the MLX port."""
        return self.net(x, stop=False).float()

    def _graph_ar(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the AR loop and save the original batch-wide early-stop length for refinement."""
        memory, queries, mask, tgt, _ = self.net._autoregressive(x, stop=False)  # noqa: SLF001 - PARSeq split path
        first = (tgt == self.net.eos).int().argmax(-1)
        # Index zero is BOS, so argmax returns zero only if this item has no EOS in the saved targets.
        stop = torch.where(first > 0, first, tgt.shape[1]).amax()
        return memory, queries, self.net._cloze_mask(mask), tgt, stop  # noqa: SLF001

    def _refine_graph(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Borrow the saved tensors within the graph lock and return freshly allocated logits."""
        memory, queries, cloze, tgt, stop = state
        context = tgt[:, : int(stop)].contiguous()
        return self.net._refine(context, memory, queries, cloze).float()  # noqa: SLF001 - PARSeq split path

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        if self._graphs is not None:
            self._graphs.release()

    def _logits(self, image: torch.Tensor) -> torch.Tensor:
        x = to_input(image, fused=self.fused_resize).to(self.dtype)
        if self._graphs is not None:
            out = self._graphs.run((x.device, x.dtype, tuple(x.shape)), x, post=self._refine_graph)
            if out is not None:
                return out
        return self.net(x).float()

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[str]]:
        """The text of a `[1, 3, H, W]` RGB crop (one line), as one quad over the whole image."""
        probs = self._logits(image).softmax(-1)[0]
        p, ids = probs.max(-1)
        ids_l = ids.tolist()
        end = ids_l.index(0) if 0 in ids_l else len(ids_l)  # the text stops at the first end token
        text = "".join(CHARSET[i - 1] for i in ids_l[:end])
        score = p[: end + 1].mean()  # the characters and the end token, as upstream's `Tokenizer._filter`
        h, w = image.shape[-2:]
        box = device_tensor([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=torch.float32, device=image.device)
        return {"boxes": box, "texts": [text], "scores": score.view(1)}

    def module(self) -> torch.nn.Module:
        """The network with its normalisation: `[B, 3, 32, 128]` RGB in `[0, 1]` -> logits `[B, 26, 95]`."""

        def forward(net: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
            return net((image * 2 - 1).to(self.dtype), stop=False).float()

        return Forward(self.net, forward)


class Kernels(Portable):
    """Use fixed-point CUDA resampling before the original encoder and decoder."""

    fused_resize = True
    module = Runner.module


class _Decoding(Portable):
    """The full 26-step decoding as one graph (no data-dependent early exit), for the MLX port."""

    def _logits(self, image: torch.Tensor) -> torch.Tensor:
        return self._decode(to_input(image).to(self.dtype))


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_Decoding, attrs=(), methods=("_decode",)),
}
