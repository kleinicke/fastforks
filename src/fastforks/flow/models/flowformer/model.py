"""FlowFormer Small runners using the pinned upstream inference architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from fastforks._core.checkpoint import check_fits
from fastforks._core.patch import override
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import flow_upsample
from fastforks.shared import libnet, mlx_wrap

if TYPE_CHECKING:
    from collections.abc import KeysView

    from fastforks.flow._runner import Context

    from .net.FlowFormer.LatentCostFormer.gru import SepConvGRU


@dataclass(frozen=True)
class Params:
    """The released Small architecture and refinement iteration count."""

    library: str
    pe: str
    dropout: float
    encoder_latent_dim: int
    query_latent_dim: int
    cost_latent_input_dim: int
    cost_latent_token_num: int
    cost_latent_dim: int
    arc_type: str
    cost_heads_num: int
    pretrain: bool
    context_concat: bool
    encoder_depth: int
    feat_cross_attn: bool
    patch_size: int
    patch_embed: str
    no_pe: bool
    gma: str
    kernel_size: int
    rm_res: bool
    vert_c_dim: int
    cost_encoder_res: bool
    cnet: str
    fnet: str
    no_sc: bool
    only_global: bool
    add_flow_token: bool
    use_mlp: bool
    vertical_conv: bool
    iters: int

    @property
    def decoder_depth(self) -> int:
        """Upstream's name for refinement iterations."""
        return self.iters

    def keys(self) -> KeysView[str]:
        """Configuration field names for upstream's augmentation-presence check."""
        return vars(self).keys()


def _forward(net: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor, *, fp16: bool) -> torch.Tensor:
    """Use upstream operations, with optional autocast confined to the network."""
    with torch.autocast(image1.device.type, torch.float16, enabled=fp16):
        return net(image1, image2)[0]


class Portable(Runner):
    """Float32 upstream inference with torch-only reshape helpers."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Load the released Small checkpoint strictly."""
        super().__init__(ctx)
        libnet.timm()
        from .net.FlowFormer.LatentCostFormer.transformer import FlowFormer  # noqa: PLC0415 - optional timm

        self.net = FlowFormer(ctx.params)
        state = ctx.state_dict()
        check_fits(self.net.state_dict(), state, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(state)
        self.net.eval().to(ctx.device)
        self.net.memory_decoder.update_block.gru.fuse()
        self.fp16 = ctx.dtype == "float16"

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the full-resolution flow, excluding upstream's low-resolution auxiliary flow."""
        return {"flow": _forward(self.net, image1, image2, fp16=self.fp16)}

    def module(self) -> torch.nn.Module:
        """Export the same full-resolution flow."""
        fp16 = self.fp16
        return Forward(self.net, lambda net, a, b: _forward(net, a, b, fp16=fp16))


def _gru_fp32_state(gru: SepConvGRU, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute convolution gates in fp16, and update recurrent state in float32."""
    for gate_name, candidate in (("convzr1", gru.convq1), ("convzr2", gru.convq2)):
        gate = cast("torch.nn.Conv2d", getattr(gru, gate_name))
        gates = gate(torch.cat((h, x), 1).half()).float()
        z, r = gates.sigmoid().chunk(2, dim=1)
        q = candidate(torch.cat((r * h, x), 1).half()).float().tanh()
        h = (1 - z) * h + z * q
    return h


def _upsample(_decoder: torch.nn.Module, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return flow_upsample.convex(flow, mask)


class Kernels(Portable):
    """Use fused convex upsampling for the final prediction."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Keep the original FP32 aggregation and fuse the last prediction's upsampling."""
        super().__init__(ctx)
        gru = self.net.get_submodule("memory_decoder.update_block.gru")
        for name in ("convzr1", "convq1", "convzr2", "convq2"):
            gru.get_submodule(name).half().to(memory_format=torch.channels_last)
        override(gru, _gru_fp32_state)
        override(self.net.memory_decoder, _upsample, "upsample_flow")


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
