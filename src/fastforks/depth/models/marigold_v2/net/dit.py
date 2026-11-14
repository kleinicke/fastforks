"""The Qwen-Image-Edit transformer of Marigold V2 at its one fixed timestep, with the depth LoRA folded in.

Upstream (`diffusers` `QwenImageTransformer2DModel`) runs 60 double-stream blocks (image and text tokens, joint
attention) under bf16 autocast. Here everything that depends only on the timestep and the fixed prompt is a constant
of the checkpoint: the modulation vectors of each block (`img_mod`, `txt_mod`: shift, scale, gate for attention and
MLP), the output norm vector and the text tokens after `txt_in` (`txt`). What remains per image: `img_in`, the blocks
and `proj_out`. The glue (layer norms, RMS norms of q / k, RoPE, gates, GELU) runs in float32 where upstream's
autocast leaves it in float32, the GEMMs and the attention in bf16; the modulation vectors are applied in bf16 as
upstream forms `1 + scale` in bf16.

`exact_mask` passes upstream's all-true attention mask to SDPA, which selects another attention kernel (other
rounding); it reproduces the original bit for bit on the GPU it was recorded on and costs 3-13 % time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .nf4 import BF, NF4Linear

if TYPE_CHECKING:
    from .stream import HostBlocks

HEADS, HEAD_DIM = 24, 128
AXES = (16, 56, 56)  # RoPE dims of the frame, height and width axes
MAX_POS = 4096  # RoPE table length per axis (upstream's)
N_MOD = 6  # modulation chunks per stream: shift, scale, gate for attention and MLP


def _rope_axis(index: torch.Tensor, dim: int) -> torch.Tensor:
    fr = torch.outer(index, 1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    return torch.polar(torch.ones_like(fr), fr)


def rope_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Complex RoPE tables `[MAX_POS, sum(AXES) / 2]` for positive and for negative positions (upstream's)."""
    pos = torch.arange(MAX_POS)
    neg = torch.arange(MAX_POS).flip(0) * -1 - 1
    return (
        torch.cat([_rope_axis(pos, d) for d in AXES], dim=1),
        torch.cat([_rope_axis(neg, d) for d in AXES], dim=1),
    )


def rope(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Rotate `[B, T, H, D]` (as pairs) by the complex frequencies `f [T, 1, D / 2]`; float32 out."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * f).flatten(3)


def rms_qk(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMSNorm (eps 1e-6) in place on `x` (float32, fresh from a linear)."""
    v = x.pow(2).mean(-1, keepdim=True)
    return x.mul_(torch.rsqrt(v + 1e-6)).mul_(w)


def _gelu_bf(x: torch.Tensor) -> torch.Tensor:
    """tanh GELU of the float32 `x` (in place), then bf16: as `F.gelu(x, approximate="tanh").to(bf16)`."""
    return torch.ops.aten.gelu_(x, approximate="tanh").to(BF)


def _heads(t: torch.Tensor) -> torch.Tensor:
    return t.unflatten(-1, (HEADS, HEAD_DIM))


class Block(nn.Module):
    """One double-stream block; the last block has no text output (upstream drops it)."""

    def __init__(
        self, t: dict[str, torch.Tensor], index: int, *, last: bool, fused: bool = False, exact_mask: bool = False
    ) -> None:
        """Take the block's tensors (`transformer_blocks.{index}.*`) from `t`; `fused`: see `NF4Linear`;
        `exact_mask`: pass upstream's all-true attention mask (see the module docstring)."""
        super().__init__()
        p = f"transformer_blocks.{index}."
        self.last = last
        self.exact_mask = exact_mask

        def lin(n: str) -> NF4Linear:
            return NF4Linear(t, p + n, fused=fused)

        self.to_q, self.to_k, self.to_v, self.to_out = lin("attn.to_q"), lin("attn.to_k"), lin("attn.to_v"), lin("attn.to_out.0")
        self.add_q, self.add_k, self.add_v = lin("attn.add_q_proj"), lin("attn.add_k_proj"), lin("attn.add_v_proj")
        self.img_fc1, self.img_fc2 = lin("img_mlp.net.0.proj"), lin("img_mlp.net.2")
        if not last:
            self.to_add_out = lin("attn.to_add_out")
            self.txt_fc1, self.txt_fc2 = lin("txt_mlp.net.0.proj"), lin("txt_mlp.net.2")
        for ours, theirs in (("nq", "norm_q"), ("nk", "norm_k"), ("naq", "norm_added_q"), ("nak", "norm_added_k")):
            self.register_buffer(ours, t[p + f"attn.{theirs}.weight"].float())
        self.register_buffer("img_mod", t[p + "img_mod"].to(BF))  # [6 * dim]: the folded modulation
        self.register_buffer("txt_mod", t[p + "txt_mod"].to(BF))

    @staticmethod
    def _mods(v: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """`[6 * dim]` -> `(shift, 1 + scale, gate)` for attention and for the MLP (`1 + scale` in bf16, as upstream)."""
        out = []
        for m in v.chunk(2):
            shift, scale, gate = m.chunk(3)
            out.append((shift, 1 + scale, gate))
        return out

    @staticmethod
    def _mod(x: torch.Tensor, m: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        y = F.layer_norm(x.float(), x.shape[-1:], eps=1e-6)
        return y.mul_(m[1]).add_(m[0]).to(BF)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, f_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`img [B, T_img, dim]`, `txt [B, T_txt, dim]` (float32) -> the same after the block."""
        im, tm = self._mods(self.img_mod), self._mods(self.txt_mod)
        n_txt = txt.shape[1]
        xi, xt = self._mod(img, im[0]), self._mod(txt, tm[0])
        q = rope(torch.cat([rms_qk(_heads(self.add_q(xt)), self.naq), rms_qk(_heads(self.to_q(xi)), self.nq)], 1), f_all)
        q = q.to(BF).transpose(1, 2)
        k = rope(torch.cat([rms_qk(_heads(self.add_k(xt)), self.nak), rms_qk(_heads(self.to_k(xi)), self.nk)], 1), f_all)
        k = k.to(BF).transpose(1, 2)
        v = torch.cat([_heads(self.add_v(xt)), _heads(self.to_v(xi))], dim=1).to(BF).transpose(1, 2)
        del xi, xt
        mask = torch.ones(1, 1, 1, q.shape[2], dtype=torch.bool, device=q.device) if self.exact_mask else None
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        del q, k, v
        a = a.transpose(1, 2).flatten(2, 3)
        img = self.to_out(a[:, n_txt:]).mul_(im[0][2]).add_(img)
        h = _gelu_bf(self.img_fc1(self._mod(img, im[1])))
        img += self.img_fc2(h).mul_(im[1][2])
        if not self.last:
            txt = self.to_add_out(a[:, :n_txt]).mul_(tm[0][2]).add_(txt)
            h = _gelu_bf(self.txt_fc1(self._mod(txt, tm[1])))
            txt += self.txt_fc2(h).mul_(tm[1][2])
        return img, txt


def block_keys(keys: list[str], index: int) -> list[str]:
    """The checkpoint keys of block `index`."""
    return [k for k in keys if k.startswith(f"transformer_blocks.{index}.")]


class Transformer(nn.Module):
    """`img_in`, the blocks, the folded output norm and `proj_out`; RoPE tables as buffers.

    `blocks` holds the resident blocks; with `host` (streaming) the blocks' tensors stay on the host (pinned) and
    `run_blocks` uploads one block ahead into two fixed device buffers while the current one computes.
    """

    def __init__(
        self,
        t: dict[str, torch.Tensor],
        *,
        host: HostBlocks | None = None,
        fused: bool = False,
        exact_mask: bool = False,
    ) -> None:
        """Build from the checkpoint tensors `t` (without the VAE); `host`: the streamed blocks (see `stream`).

        `fused`: the linears expand their weights with the fused kernel (`NF4Linear.dequant`); `exact_mask`: see
        `Block`.
        """
        super().__init__()
        self.fused, self.exact_mask = fused, exact_mask
        self.img_in = NF4Linear(t, "img_in", fused=fused)
        self.register_buffer("txt", t["txt"].float()[None])
        scale, shift = t["norm_out.linear"].float().chunk(2)
        self.register_buffer("out_scale1p", 1 + scale)
        self.register_buffer("out_shift", shift)
        self.register_buffer("proj_w", t["proj_out.weight"])
        self.register_buffer("proj_b", t["proj_out.bias"])
        pos, neg = rope_tables()
        self.register_buffer("pos", pos)
        self.register_buffer("neg", neg)
        self.host = host
        if host is not None:
            self.n = len(host)
            self.blocks = nn.ModuleList()
        else:
            self.n = 1 + max(int(k.split(".")[1]) for k in t if k.startswith("transformer_blocks."))
            self.blocks = nn.ModuleList(Block(t, i, last=i == self.n - 1, fused=fused, exact_mask=exact_mask) for i in range(self.n))

    def freqs(self, h: int, w: int) -> torch.Tensor:
        """RoPE frequencies of the text tokens then the `h x w` image tokens, `[T, 1, D / 2]` complex."""
        a = [s // 2 for s in AXES]
        fp, fn = self.pos.split(a, dim=1), self.neg.split(a, dim=1)
        ff = fp[0][0:1].view(1, 1, 1, -1).expand(1, h, w, -1)
        fh = torch.cat([fn[1][-(h - h // 2) :], fp[1][: h // 2]], dim=0).view(1, h, 1, -1).expand(1, h, w, -1)
        fw = torch.cat([fn[2][-(w - w // 2) :], fp[2][: w // 2]], dim=0).view(1, 1, w, -1).expand(1, h, w, -1)
        f_img = torch.cat([ff, fh, fw], dim=-1).reshape(h * w, -1)
        m = max(h // 2, w // 2)
        return torch.cat([self.pos[m : m + self.txt.shape[1]], f_img]).unsqueeze(1)

    def run_blocks(self, img: torch.Tensor, txt: torch.Tensor, f_all: torch.Tensor) -> torch.Tensor:
        """All blocks, resident or streamed."""
        if self.host is None:
            for blk in self.blocks:
                img, txt = blk(img, txt, f_all)
            return img
        for i, t in enumerate(self.host.run(img.device)):
            blk = Block(t, i, last=i == self.n - 1, fused=self.fused, exact_mask=self.exact_mask)
            img, txt = blk(img, txt, f_all)
        return img

    def forward(self, packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Packed latent tokens `[B, h * w, 64]` -> the predicted packed velocity (bf16)."""
        f_all = self.freqs(h, w)
        img = self.img_in(packed.to(BF), keep_bf=True)
        txt = self.txt.expand(packed.shape[0], -1, -1)
        img = self.run_blocks(img, txt, f_all)
        x = F.layer_norm(img.float(), img.shape[-1:], eps=1e-6) * self.out_scale1p + self.out_shift
        return F.linear(x.to(BF), self.proj_w, self.proj_b)
