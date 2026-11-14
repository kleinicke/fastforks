"""PARSeq runners: checkpoint -> network and the portable recogniser.

Inference as upstream `read.py` (`strhub`, `SceneTextDataModule.get_transform`): the whole image resized to
`32 x 128` (Pillow bicubic, bit-exact: `fastforks.shared.pil_resize`), `/ 255`, `(x - 0.5) / 0.5`, greedy
autoregressive decoding with one refinement pass, softmax, the text up to the first end token.
The 94-character charset: digits, Latin letters, ASCII punctuation (no space).

`portable`: the network in plain PyTorch, float32 or float16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.ocr._runner import Forward, Runner, levels
from fastforks.shared.pil_resize import pil_bicubic

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


def to_input(image: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` RGB uint8 (or float in [0, 1]) -> the network input `[B, 3, 32, 128]` float32 in `[-1, 1]`."""
    x = levels(image).flip(1) if image.dtype != torch.uint8 else image
    x = pil_bicubic(x, IMAGE) if tuple(x.shape[-2:]) != IMAGE else x.float()
    return x.div_(255).sub_(0.5).div_(0.5)


class Portable(Runner):
    """PARSeq in plain PyTorch, float32 or float16."""

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

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[str]]:
        """The text of a `[1, 3, H, W]` RGB crop (one line), as one quad over the whole image."""
        probs = self.net(to_input(image).to(self.dtype)).float().softmax(-1)[0]
        p, ids = probs.max(-1)
        ids_l = ids.tolist()
        end = ids_l.index(0) if 0 in ids_l else len(ids_l)  # the text stops at the first end token
        text = "".join(CHARSET[i - 1] for i in ids_l[:end])
        score = p[: end + 1].mean()  # the characters and the end token, as upstream's `Tokenizer._filter`
        h, w = image.shape[-2:]
        box = torch.tensor([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=torch.float32, device=image.device)
        return {"boxes": box, "texts": [text], "scores": score.view(1)}

    def module(self) -> torch.nn.Module:
        """The network with its normalisation: `[B, 3, 32, 128]` RGB in `[0, 1]` -> logits `[B, 26, 95]`."""

        def forward(net: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
            return net((image * 2 - 1).to(self.dtype), stop=False).float()

        return Forward(self.net, forward)


RUNNERS = {"portable": Portable}
