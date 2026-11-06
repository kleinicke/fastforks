"""MiDaS v3.1 DPT-BEiT-L shared by MiDaS (`dpt_beit_large_512`) and ZoeDepth (its `DPT_BEiT_L_384` core).

`net/` holds MiDaS's DPT, reassemble and fusion blocks (vendored) and timm's BEiT re-implemented with timm's names and
MiDaS's arbitrary-size relative position bias, so upstream checkpoints load unchanged apart from `UNUSED` keys.
`DPTDepthModel.forward(x, features=True)` also returns the decoder features ZoeDepth hooks. `use_sdpa_attention`
swaps the attention for `scaled_dot_product_attention` with the relative position bias as additive mask (same math).
`scale_head` folds `HEAD_SCALE` into the head so its fp16 activations stay finite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.patch import override

from .net.beit import Attention, gen_relative_position_index
from .net.dpt import DPTDepthModel

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["HEAD_SCALE", "DPTDepthModel", "dpt_beit_large", "load", "scale_head", "use_sdpa_attention"]

UNUSED = ("relative_position_index", "pretrained.model.fc_norm.", "pretrained.model.head.")
"""Checkpoint keys not in the module: derived index buffers and BEiT's classifier (MiDaS never runs it)."""


def dpt_beit_large(size: int) -> DPTDepthModel:
    """Upstream `DPTDepthModel(backbone="beitl16_<size>", non_negative=True)`, `size` 384 or 512, random weights."""
    return DPTDepthModel(backbone=f"beitl16_{size}", non_negative=True)


def load(net: torch.nn.Module, state: Mapping[str, torch.Tensor], prefix: str = "") -> None:
    """Strictly load `state` (keys `prefix` + module key) into `net`, skipping the `UNUSED` checkpoint keys."""
    keep = {k: v for k, v in state.items() if not any(u in k for u in UNUSED)}
    net.load_state_dict({k.removeprefix(prefix): v for k, v in keep.items() if k.startswith(prefix)}, strict=True)


HEAD_SCALE = 2.0**-4
"""Factor `scale_head` folds into the DPT head: its 3x3 convs reach ~1.1e5 in float32 on photos (fp16 max 65504)."""


def scale_head(net: DPTDepthModel) -> None:
    """Fold `HEAD_SCALE` into the loaded head (in place): `out_conv` and the output become `HEAD_SCALE` x upstream's.

    The head is conv3x3, bilinear x2, conv3x3, ReLU, conv1x1, ReLU: scaling the first conv's weight and bias and the
    later biases scales every activation after it (linear maps, ReLU commutes with a positive factor). A power of two,
    so float32 results stay bit-identical after multiplying back; in fp16 the 3x3 convs no longer overflow (inf / NaN
    depth on real photos). Callers multiply the output by `1 / HEAD_SCALE` in float32, consumers of `out_conv`
    fold the inverse into their weights.
    """
    oc = net.scratch.output_conv
    with torch.no_grad():
        for p in (oc[0].weight, oc[0].bias, oc[2].bias, oc[4].bias):
            p.mul_(HEAD_SCALE)


def _rel_pos_bias(attn: Attention, window: tuple[int, int]) -> torch.Tensor:
    """`Attention._get_rel_pos_bias(window)` (same values), without the `[N, N, nH]` copy.

    Gathered from the transposed table straight into the `[1, nH, N, N]` layout; the index is generated once per
    window size and device (outside `torch.inference_mode`, so `torch.export` can use it too) and shared by all blocks.
    """
    oh, ow = 2 * attn.window_size[0] - 1, 2 * attn.window_size[1] - 1
    nh, nw = 2 * window[0] - 1, 2 * window[1] - 1
    tab, nd = attn.relative_position_bias_table, attn.num_relative_distance
    sub = tab[: nd - 3].reshape(1, ow, oh, -1).permute(0, 3, 1, 2)
    sub = F.interpolate(sub, size=(nh, nw), mode="bilinear").permute(0, 2, 3, 1).reshape(nh * nw, -1)
    table = torch.cat([sub, tab[nd - 3 :]]).t()  # [nH, T]
    key = (window, tab.device)
    index = _INDEX.get(key)
    if index is None:
        if len(_INDEX) >= _INDEX_SIZES:  # int64, 18 MB at 994x1482: drop the oldest size instead of growing forever
            del _INDEX[next(iter(_INDEX))]
        with torch.inference_mode(False):  # noqa: FBT003 - positional: dynamo rejects the keyword
            index = _INDEX[key] = gen_relative_position_index(window, device=tab.device).view(-1)
    n = window[0] * window[1] + 1
    return table[:, index].view(1, -1, n, n)


_INDEX: dict[tuple[tuple[int, int], torch.device], torch.Tensor] = {}
_INDEX_SIZES = 8


def _sdpa_forward(self: Attention, x: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    b, n, _ = x.shape
    qkv_bias = torch.cat((self.q_bias, cast("torch.Tensor", self.k_bias), self.v_bias))
    qkv = F.linear(x, self.qkv.weight, qkv_bias).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    window = (int(resolution[0]) // 16, int(resolution[1]) // 16)
    bias = _rel_pos_bias(self, window).to(q.dtype).expand(b, -1, -1, -1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
    return self.proj(o.transpose(1, 2).reshape(b, n, -1))


def use_sdpa_attention(net: torch.nn.Module) -> None:
    """Run every BEiT attention of `net` through SDPA with the relative position bias as additive mask."""
    for m in net.modules():
        if isinstance(m, Attention):
            override(m, _sdpa_forward)
