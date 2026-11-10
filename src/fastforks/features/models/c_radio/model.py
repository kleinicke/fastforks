"""C-RADIOv3 runners: the checkpoint on timm's `VisionTransformer`, walked by `fastforks.shared.libnet.features`.

Upstream (`hf_model.RADIOModel`, `trust_remote_code`) is a timm ViT-B / L / H / g at patch 16 whose patch embedding,
position table and prefix tokens are replaced by its `ViTPatchGenerator` (CPE, `enable_cpe_support.py`). At eval:

- patch embedding: `Im2Patches` (channel, row, column) then a bias-free `Linear(3 * 16 * 16, C)`, the same as a
  16 x 16 stride-16 convolution with that weight `.view(C, 3, 16, 16)` (bias zero);
- position table `[1, 128 * 128, C]` (2048 px): used as is at a 128 x 128 grid; otherwise resized bilinearly (float32,
  `align_corners=False`) to the square `max(h, w)`, cropped top-left to `h x w`; added to the patches only;
- prefix: 8 learnt tokens (4 summary tokens, one per teacher, then 4 registers) prepended after the position table;
- blocks: timm's (LayerScale on B / L, stored as `grandma`); no `norm_pre`, no final norm (`model_norm = false`);
- input: RGB in [0, 1] standardised with the OpenAI CLIP mean / std (`input_conditioner`, the card's `[io]`).

Outputs: `last_hidden_state` = all tokens of the last block (8 prefix tokens, then the patches row-major; upstream's
`features` are `[:, 8:]`), `pooler_output` = upstream's `summary`: the summary tokens at `summary_idxs` (clip,
siglip2, dino_v2) concatenated, `[B, 3 * C]`; `hidden_states[0]` = the embeddings (the first block's input).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import CheckpointMismatchError
from fastforks.shared.libnet import features, timm

if TYPE_CHECKING:
    from fastforks.features._runner import Context

_PATCH = 16  # fixed: upstream every C-RADIOv3 checkpoint
_GRID = 128  # fixed: upstream position table side (cpe_max_size 2048 / 16)
_SUMMARY = 4  # fixed: upstream summary tokens (one per teacher: clip, siglip2, dino_v2, sam)
_PREFIX = 8  # fixed: upstream summary tokens + registers (`cls_token.token` rows)
_LS_INIT = 1e-5  # fixed: upstream LayerScale init (B / L), overwritten by the checkpoint
_PREFIX_KEY = "radio_model."
_MODEL_KEY = "radio_model.model."


@dataclass(frozen=True)
class Params:
    """Card `[params]` / variant `params` / `[options]`."""

    library: str  # "timm": the network is timm's VisionTransformer
    embed_dim: int
    depth: int
    num_heads: int
    layer_scale: bool  # B / L: LayerScale in the blocks
    pos_align_corners: bool  # the position-table resampling's `align_corners` (B / H: true, L / g: false; upstream)
    fp16_accumulate: bool  # portable: cuBLAS fp16 accumulation in the GEMMs (option)
    fp32_accumulate: bool  # kernels: every GEMM through cuBLAS with fp32 accumulation (option)


class _RadioWalk(features._Walk):  # noqa: SLF001 - the shared walk, with upstream's position rule
    """The shared walk with the checkpoint's 128 x 128 position table and `ViTPatchGenerator`'s eval resampling."""

    def __init__(self, net: Any, pos: torch.Tensor, *, align_corners: bool) -> None:  # noqa: ANN401 - timm ViT
        """`pos`: the checkpoint's position table `[1, 128 * 128, C]`; `align_corners`: as the variant's upstream."""
        super().__init__(net)
        self.align_corners = align_corners
        self.pos32 = pos.detach().float().clone()
        self.grid = (_GRID, _GRID)
        self.npt_pos = 0

    def resample(self, pos: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Upstream `_get_pos_embeddings` at eval (CPE mode) for an `h x w` patch grid."""
        p = pos.reshape(1, _GRID, _GRID, -1).permute(0, 3, 1, 2)
        m = max(h, w)
        p = F.interpolate(p, size=(m, m), mode="bilinear", align_corners=self.align_corners)[..., :h, :w]
        return p.flatten(2).transpose(1, 2)


class _Radio(features._TimmFeatures):  # noqa: SLF001 - the shared runner with this family's network and pool
    Params = Params  # type: ignore[assignment] - this family's params

    def network(self, ctx: Context) -> Any:  # noqa: ANN401 - timm VisionTransformer
        """The ViT with the checkpoint (prefix tokens as `cls_token`); keeps the position table and `summary_idxs`."""
        p: Params = ctx.params
        what = f"{ctx.variant.ref} checkpoint"
        sd = ctx.state_dict(0, strip="")
        io = ctx.variant.io
        for name, want in (("norm_mean", io.mean), ("norm_std", io.std)):
            got = sd.pop(f"{_PREFIX_KEY}input_conditioner.{name}", None)
            if got is None or not torch.allclose(got.flatten(), torch.tensor(want), atol=1e-6):
                msg = f"input_conditioner.{name} is {None if got is None else got.flatten().tolist()}, not {want}"
                raise CheckpointMismatchError(what, msg)
        idx = sd.pop(f"{_PREFIX_KEY}summary_idxs", None)
        if idx is None or idx.numel() == 0 or int(idx.max()) >= _SUMMARY:
            raise CheckpointMismatchError(
                what, f"summary_idxs {idx} (expected indices of the {_SUMMARY} summary tokens)"
            )
        self.summary_idxs = [int(i) for i in idx]
        self._idx: dict[torch.device, torch.Tensor] = {}
        self.pooler_size = len(self.summary_idxs) * p.embed_dim
        net: Any = (
            timm()
            .models.vision_transformer.VisionTransformer(
                img_size=_GRID * _PATCH,
                patch_size=_PATCH,
                num_classes=0,
                global_pool="",
                embed_dim=p.embed_dim,
                depth=p.depth,
                num_heads=p.num_heads,
                init_values=_LS_INIT if p.layer_scale else None,
                class_token=False,
                pos_embed="none",
                no_embed_class=True,
                final_norm=False,
                dynamic_img_size=True,
                weight_init="skip",
            )
            .eval()
        )
        # other keys stay as they are and are reported by `load_strict`
        state = {k.removeprefix(_MODEL_KEY).replace(".grandma", ".gamma"): v for k, v in sd.items()}
        pos = state.pop("patch_generator.pos_embed")
        tok = state.pop("patch_generator.cls_token.token")
        emb = state.pop("patch_generator.embedder.weight")
        state.pop("reg_token", None)  # B / L: an unused timm register table (the registers are in `cls_token.token`)
        if tok.shape != (_PREFIX, p.embed_dim) or pos.shape != (1, _GRID * _GRID, p.embed_dim):
            msg = f"prefix tokens {tuple(tok.shape)}, position table {tuple(pos.shape)}"
            raise CheckpointMismatchError(what, msg)
        state["patch_embed.proj.weight"] = emb.view(p.embed_dim, 3, _PATCH, _PATCH)
        state["patch_embed.proj.bias"] = torch.zeros(p.embed_dim)
        load_strict(net, state, what)
        net.cls_token = torch.nn.Parameter(tok[None].clone())
        net.num_prefix_tokens, net.num_reg_tokens = _PREFIX, _PREFIX - _SUMMARY
        self._pos, self._align = pos, p.pos_align_corners
        return net

    def walker(self, net: Any) -> features._Walk:  # noqa: ANN401 - the network of `network`
        """The walk with upstream's position rule."""
        walk = _RadioWalk(net, self._pos, align_corners=self._align)
        del self._pos
        return walk

    def pool(self, net: Any, last: torch.Tensor) -> torch.Tensor:  # noqa: ANN401, ARG002 - the optimised network
        """Upstream's `summary`: the summary tokens at `summary_idxs`, concatenated (`[B, 3 * C]`).

        The indices as a tensor on the device, made on the first call: indexing with the list would copy it to the
        device on every call, which a CUDA-graph capture refuses (the kernels' graphs).
        """
        idx = self._idx.get(last.device)
        if idx is None:
            idx = self._idx[last.device] = torch.tensor(self.summary_idxs, device=last.device)
        return last.index_select(1, idx).flatten(1)


class Portable(_Radio, features.Portable):
    """Plain PyTorch through `fastforks._optimize` (SDPA attention, folded LayerScale): float32, or fp16 weights."""


class Kernels(_Radio, features.Kernels):
    """Our kernels through `fastforks._optimize` (fused fp16 GEMMs with bias / GELU / residual epilogues)."""


RUNNERS = {"portable": Portable, "kernels": Kernels}
