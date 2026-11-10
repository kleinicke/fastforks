"""RAFT runners: weights -> network and the portable forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

from .net import LARGE, RAFT, SMALL

if TYPE_CHECKING:
    import torch

    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    size: Literal["large", "small"]  # torchvision raft_large / raft_small
    iters: int  # flow updates (torchvision's `num_flow_updates`)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the encoders and the update block run under autocast, the correlation in fp32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = RAFT(LARGE if ctx.params.size == "large" else SMALL)
        sd = ctx.state_dict()
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


class Kernels(Runner):
    """Our CUDA engine (`kernels/engine.py`): no correlation volume, fused GRU and conv epilogues.

    The lookup computes the correlation on the fly from a pooled fmap2 pyramid (O(HW) memory); fp16 activations with
    fp16-accumulate convolutions, fp32 correlation sums and flow state.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the engine."""
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, cutlass_conv, raft_flow  # noqa: PLC0415 - needs the kernel libraries

        from .kernels.engine import Engine  # noqa: PLC0415

        d = ctx.device
        sd = ctx.state_dict()
        check_fits(
            RAFT(LARGE if ctx.params.size == "large" else SMALL).state_dict(), sd, f"{ctx.variant.ref} checkpoint"
        )
        self.engine = Engine(
            sd, d, raft_flow.load(d), cutlass_conv.load(d), cudnn_hacc.load(d), ctx.params.iters,
            large=ctx.params.size == "large",
        )  # fmt: skip

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch, pair by pair."""
        return {"flow": self.engine(image1, image2)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
