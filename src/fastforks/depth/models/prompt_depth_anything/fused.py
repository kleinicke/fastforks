"""Kernels-mode Prompt Depth Anything: the Depth Anything V2 ViT and `dpt_fused` head plus the depth-prompt branches.

The ViT (`FusedViT`) and the head's projections, residual conv units, fusion 1x1 convolutions and bilinear upsamplings
are those of `depth_anything_v2.fused`. Added here, all in dense NHWC fp16 like the rest of the head:

- per fusion block, `resConfUnit_depth(prompt resized to the block)`: the prompt (padded from 1 to 8 channels, the
  kernels' channel alignment) is resized with the NHWC bilinear kernel, run through the three 3x3 convs (ReLU fused),
  and the last conv adds the block's `x` in its epilogue (`res=`)
- tail: 3x3 conv + ReLU in fp16, then the 1-channel 1x1 conv and the sigmoid in fp32 in PyTorch (the `head_dot` kernel
  has the ReLU of the Depth Anything V2 head baked in)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.depth.models.depth_anything_v2.fused import FusedHead, FusedViT, _Conv, _conv
from fastforks.kernels.dpt_fused import ACT_RELU, DptFused

__all__ = ["FusedViT", "PromptFusedHead"]

_CPAD = 8  # input channels of the first prompt conv, zero-padded (1 -> 8)


class PromptFusedHead(FusedHead):
    """`FusedHead` with the prompt branches and a sigmoid output."""

    def __init__(self, head: nn.Module, device: torch.device, k: DptFused) -> None:
        """Convert the head weights (CPU), move them to `device`."""
        super().__init__(head, device, k)
        self.prompt_convs: list[list[_Conv]] = []
        for i in range(1, 5):
            seq: nn.Sequential = getattr(head.scratch, f"refinenet{i}").resConfUnit_depth
            convs = [_conv(m, device) for m in seq if isinstance(m, nn.Conv2d)]
            w, b, pad, stride = convs[0]
            convs[0] = (F.pad(w, (0, _CPAD - w.shape[-1])).contiguous(), b, pad, stride)
            self.prompt_convs.append(convs)

    def _fuse(  # noqa: PLR0913, PLR0917 - fusion block inputs
        self,
        i: int,
        x0: torch.Tensor,
        r0: torch.Tensor | None,
        x1: torch.Tensor | None,
        r1: torch.Tensor | None,
        size: tuple[int, int],
        prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FeatureFusionBlock with the prompt: `x0 (+ rcu1(x1))`, rcu2, `+ rcu_depth(prompt)`, out_conv, bilinear."""
        assert prompt is not None  # noqa: S101 - the call below always passes it
        (u1, u2), (wo, bo) = self.fusion[i]
        if x1 is not None:
            t = self._conv(r1, u1[0], act=ACT_RELU)  # ty: ignore[invalid-argument-type]
            r0 = torch.empty_like(x1)
            x0 = self._conv(t, u1[1], res=x1, res2=x0, y2=r0)
        t = self._conv(r0, u2[0], act=ACT_RELU)  # ty: ignore[invalid-argument-type]
        z = self._conv(t, u2[1], res=x0)
        c0, c1, c2 = self.prompt_convs[i]
        t = self._conv(self.k.bilinear(prompt, (z.shape[1], z.shape[2]), align_corners=False), c0, act=ACT_RELU)
        t = self._conv(t, c1, act=ACT_RELU)
        z = self._conv(t, c2, res=z)
        return self.k.bilinear(self.k.gemm(z, wo, bo), size, align_corners=True)

    def __call__(self, feats: list[torch.Tensor], gh: int, gw: int, prompt: torch.Tensor | None = None) -> torch.Tensor:
        """Tokens of the 4 layers + the prompt in [0, 1] (`[B, 1, h, w]`) -> `[B, 14 gh, 14 gw]` (fp32, sigmoid)."""
        assert prompt is not None  # noqa: S101 - a depth prompt is required
        k = self.k
        p = F.pad(prompt.half().permute(0, 2, 3, 1), (0, _CPAD - 1)).contiguous()  # NHWC, channels padded to 8
        ls, rs = [], []
        for t, (wp, bp), rz, rn in zip(feats, self.projects, self.resize, self.rn, strict=True):
            x = k.gemm(t, wp, bp).view(t.shape[0], gh, gw, -1)
            if rz is not None and rz[0] == "deconv":
                _, wg, bg, s = rz
                x = k.gemm(x, wg, bg, shuffle=(gh, gw, s, s))
            elif rz is not None:
                x = self._conv(x, rz[1:])
            r = x.new_empty((*x.shape[:3], rn[0].shape[0]))
            ls.append(self._conv(x, rn, y2=r))
            rs.append(r)
        l1, l2, l3, l4 = ls
        z = self._fuse(3, l4, rs[3], None, None, l3.shape[1:3], p)
        z = self._fuse(2, z, None, l3, rs[2], l2.shape[1:3], p)
        z = self._fuse(1, z, None, l2, rs[1], l1.shape[1:3], p)
        z = self._fuse(0, z, None, l1, rs[0], (2 * l1.shape[1], 2 * l1.shape[2]), p)
        o = k.bilinear(self._conv(z, self.oc1), (14 * gh, 14 * gw), align_corners=True)
        o = self._conv(o, self.oc2, act=ACT_RELU)
        return torch.sigmoid(F.linear(o.float(), self.w_last[None], o.new_tensor(self.b_last, dtype=torch.float32)))[
            ..., 0
        ]
