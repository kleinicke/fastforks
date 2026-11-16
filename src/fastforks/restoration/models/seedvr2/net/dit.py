"""The SeedVR2 transformer (NaDiT, `models/dit_v2` for 3B, `models/dit` for 7B) at its one fixed timestep.

Upstream runs the transformer under bf16 autocast on packed variable-length sequences: video tokens (latent 1 x 2 x 2
patches) and 58 fixed text tokens, 32 (3B) / 36 (7B) blocks of joint window attention. Here everything that depends
only on the timestep (1000) and the fixed prompt is computed once at load in upstream's arithmetic: the text tokens
after `txt_in`, the time embedding and, per block, the AdaSingle sums (`emb + shift`, `emb + scale`, `emb + gate`,
float32 as upstream adds the bf16 embedding to the float32 vectors). The windows of each token grid (720p-normalised
3-D windows, every other block shifted by half a window) become one gather index per window shape, the RoPE tables are
made per window shape, and each window shape is one batched SDPA call over `[window tokens; text tokens]` (upstream:
`flash_attn_varlen_func` over all windows). The text tokens' attention output is the mean over all windows (upstream's
"text pooling"); the last block uses the text tokens only as keys and values (upstream drops its text output).

Dtypes as upstream under autocast: bf16 GEMMs, attention and residual stream; float32 RMS statistics, RoPE and
AdaSingle arithmetic, each rounded to bf16 where upstream's in-place ops on bf16 tensors round. A block linear stored
as NF4 (`.q`, `.absmax`, the `quant = "nf4"` option) is expanded to bf16 per call by `Transformer.dequant`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

BF = torch.bfloat16
HEAD_DIM = 128
EPS = 1e-5  # every RMS norm
TIMESTEP = 1000.0  # the one step of SeedVR2: t = T
SINUSOID = 256  # time embedding input width
ROPE_AXIS = 42  # 3B RoPE dims per axis (t, h, w): 126 of the 128 head dims rotate
PIXEL_AXIS = 20  # 7B: `RotaryEmbedding(dim=21, freqs_for="pixel", max_freq=256)`, 10 frequencies per axis: 60 dims
PIXEL_MAX_FREQ = 256
WINDOWS = (4, 3, 3)  # windows per (t, h, w) at 720p (45 x 80 tokens), as configured
WINDOW_AREA = 45 * 80  # token grid of 720p


@dataclass(frozen=True)
class Config:
    """The transformer's shape (`configs_{3b,7b}/main.yaml`)."""

    dim: int
    heads: int
    layers: int
    mm_layers: int  # blocks with separate video / text weights (the rest share them: `.all.`)
    swiglu: bool  # 3B: SwiGLU MLP without biases; 7B: tanh-GELU MLP with biases
    out_norm: bool  # 3B: RMS norm + AdaSingle before `vid_out`
    mm_rope: bool  # 3B: "lang" RoPE on video and text (text positions first); 7B: "pixel" RoPE on video only
    last_vid_only: bool  # 3B: the last block's AdaSingle modulates only the video tokens


CONFIGS = {
    "3b": Config(
        dim=2560, heads=20, layers=32, mm_layers=10, swiglu=True, out_norm=True, mm_rope=True, last_vid_only=True
    ),
    "7b": Config(
        dim=3072, heads=24, layers=36, mm_layers=36, swiglu=False, out_norm=False, mm_rope=False, last_vid_only=False
    ),
}


# --- windows (models/dit_v2/window.py, vendored) -----------------------------------------------------------------


def make_720p_windows(
    size: tuple[int, int, int], num_windows: tuple[int, int, int]
) -> list[tuple[slice, slice, slice]]:
    """`make_720Pwindows_bysize`."""
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt(WINDOW_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = math.ceil(resized_h / resized_nh), math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, 30) / resized_nt)
    nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]


def make_shifted_720p_windows(
    size: tuple[int, int, int], num_windows: tuple[int, int, int]
) -> list[tuple[slice, slice, slice]]:
    """`make_shifted_720Pwindows_bysize`: shifted by half a window along every axis with more than one window."""
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt(WINDOW_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = math.ceil(resized_h / resized_nh), math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, 30) / resized_nt)
    st, sh, sw = (0.5 if wt < t else 0, 0.5 if wh < h else 0, 0.5 if ww < w else 0)
    nt, nh, nw = math.ceil((t - st) / wt), math.ceil((h - sh) / wh), math.ceil((w - sw) / ww)
    nt, nh, nw = (nt + 1 if st > 0 else 1, nh + 1 if sh > 0 else 1, nw + 1 if sw > 0 else 1)
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]


# --- RoPE (rotary-embedding-torch, `freqs_for="lang"`, theta 10000) ------------------------------------------------


def _angles(pos: torch.Tensor) -> torch.Tensor:
    """`[n]` positions -> `[n, 42]` angles (each frequency twice, interleaved), float32."""
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE_AXIS, 2)[: ROPE_AXIS // 2].float() / ROPE_AXIS))
    return (pos.float()[:, None] * inv).repeat_interleave(2, dim=-1)


def _pixel_angles(n: int) -> torch.Tensor:
    """`freqs_for="pixel"` axial angles: positions `linspace(-1, 1, n)` times `linspace(1, 128, 10) * pi`, `[n, 20]`."""
    freqs = torch.linspace(1.0, PIXEL_MAX_FREQ / 2, PIXEL_AXIS // 2) * math.pi
    return (torch.linspace(-1, 1, steps=n)[:, None] * freqs).repeat_interleave(2, dim=-1)


def rope_table(ft: int, fh: int, fw: int, t0: int | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin `[ft * fh * fw, 1, 128]` (float32) of a `(ft, fh, fw)` grid: "lang" with time positions from `t0`,
    or "pixel" (`t0` None).

    The head dims past the rotated ones pass through (cos 1, sin 0: `x * 1 + y * 0 == x` exactly).
    """
    if t0 is None:
        a = [_pixel_angles(n) for n in (ft, fh, fw)]
    else:
        a = [_angles(torch.arange(n) + o) for n, o in ((ft, t0), (fh, 0), (fw, 0))]
    grid = torch.cat(
        [
            a[0][:, None, None].expand(ft, fh, fw, -1),
            a[1][None, :, None].expand(ft, fh, fw, -1),
            a[2][None, None, :].expand(ft, fh, fw, -1),
        ],
        dim=-1,
    ).reshape(ft * fh * fw, 1, -1)
    pad = HEAD_DIM - grid.shape[-1]
    cos = F.pad(grid.cos(), (0, pad), value=1.0)
    sin = F.pad(grid.sin(), (0, pad), value=0.0)
    return cos.to(device), sin.to(device)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`apply_rotary_emb` on bf16 `x [..., n, H, 128]`: float32 `x * cos + rotate_half(x) * sin` -> bf16."""
    xf = x.float()
    x1, x2 = xf.unflatten(-1, (-1, 2)).unbind(-1)
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (xf * cos + rot * sin).to(BF)


# --- window plans ----------------------------------------------------------------------------------------------------


@dataclass
class Group:
    """Windows of one shape: token indices `[n_win, n]` and the RoPE tables of the window grid."""

    idx: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass
class Plan:
    """The windows of one token grid, by shape; `count` windows in all (the text tokens' output is their mean)."""

    groups: list[Group]
    count: int


def make_plan(size: tuple[int, int, int], *, shifted: bool, txt_len: int | None, device: torch.device) -> Plan:
    """The window partition of a `(t, h, w)` token grid as gathers, grouped by window shape (upstream's order);
    `txt_len` None: pixel RoPE (see `rope_table`)."""
    wins = (make_shifted_720p_windows if shifted else make_720p_windows)(size, WINDOWS)
    grid = torch.arange(size[0] * size[1] * size[2]).view(size)
    by_shape: dict[tuple[int, ...], list[torch.Tensor]] = {}
    for s in wins:
        g = grid[s]
        by_shape.setdefault(tuple(g.shape), []).append(g.reshape(-1))
    groups = []
    for shape, idx in by_shape.items():
        cos, sin = rope_table(*shape, t0=txt_len, device=device)
        groups.append(Group(torch.stack(idx).to(device), cos, sin))
    return Plan(groups, len(wins))


# --- arithmetic ------------------------------------------------------------------------------------------------------


def rms(x: torch.Tensor) -> torch.Tensor:
    """apex `FusedRMSNorm` without affine on bf16: float32 statistics, one rounding."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)).to(BF)


def rms_affine(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """apex `FusedRMSNorm` with affine under autocast: `w_bf16 * bf16(normalised x)` (bf16 product)."""
    return w * rms(x)


def linear(t: Mapping[str, torch.Tensor], name: str, x: torch.Tensor) -> torch.Tensor:
    """`name` of `t` as a bf16 linear (bias if stored)."""
    return F.linear(x, t[name + ".weight"], t.get(name + ".bias"))


def time_embedding(t: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """`TimeEmbedding` at `TIMESTEP`: float32 sinusoid (sin then cos), bf16 MLP with SiLU -> `[6 * dim]` bf16."""
    half = SINUSOID // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
    arg = torch.tensor([TIMESTEP], device=device)[:, None] * torch.exp(exponent)[None]
    emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1).to(BF)
    emb = F.silu(linear(t, "emb_in.proj_in", emb))
    emb = F.silu(linear(t, "emb_in.proj_hid", emb))
    return linear(t, "emb_in.proj_out", emb)[0]


def ada_sums(
    emb: torch.Tensor, t: Mapping[str, torch.Tensor], prefix: str, layer: int, dim: int
) -> dict[str, torch.Tensor]:
    """AdaSingle `emb + param` (float32) for `layer` 0 (attn) / 1 (mlp) of the branch `prefix` (`...ada.vid.`)."""
    e = emb.view(dim, 2, 3)[:, layer]  # "b (d l g)"
    name = ("attn", "mlp")[layer]
    return {g: e[:, j] + t[f"{prefix}{name}_{g}"] for j, g in enumerate(("shift", "scale", "gate"))}


class Consts(nn.Module):
    """Per-block AdaSingle sums (float32 `[dim]`), resident also when the blocks stream."""

    def __init__(self, emb: torch.Tensor, t: Mapping[str, torch.Tensor], cfg: Config) -> None:
        """From the time embedding and the checkpoint's AdaSingle vectors."""
        super().__init__()
        self.cfg = cfg
        for i in range(cfg.layers):
            for branch in ("vid", "txt") if i < cfg.mm_layers else ("all",):
                for layer in (0, 1):
                    for g, v in ada_sums(emb, t, f"blocks.{i}.ada.{branch}.", layer, cfg.dim).items():
                        self.register_buffer(f"b{i}_{branch}_{layer}_{g}", v)
        if cfg.out_norm:  # `vid_out_ada` reads block 0's cached video attn chunk of the embedding (upstream's Cache)
            e = emb.view(cfg.dim, 2, 3)[:, 0]
            self.register_buffer("out_shift", e[:, 0] + t["vid_out_ada.out_shift"])
            self.register_buffer("out_scale", e[:, 1] + t["vid_out_ada.out_scale"])

    def get(self, i: int, branch: str, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(shift, scale, gate) of block `i`."""
        if i >= self.cfg.mm_layers:
            branch = "all"
        return tuple(getattr(self, f"b{i}_{branch}_{layer}_{g}") for g in ("shift", "scale", "gate"))


def block_keys(keys: Iterable[str], index: int) -> list[str]:
    """The checkpoint keys of block `index` that the forward reads (not its AdaSingle vectors: see `Consts`)."""
    p = f"blocks.{index}."
    return [k for k in keys if k.startswith(p) and ".ada." not in k]


# --- the transformer -------------------------------------------------------------------------------------------------


def _has(t: Mapping[str, torch.Tensor], name: str) -> bool:
    """`t` holds the weight `name`, in bf16 or quantised (`.q`, `.absmax`)."""
    return name + ".weight" in t or name + ".q" in t


def _branch(t: Mapping[str, torch.Tensor], name: str, branch: str) -> str:
    """`attn.proj_qkv` -> `attn.proj_qkv.vid` / `.txt` / `.all` (whichever the block has)."""
    n = f"{name}.{branch}"
    return n if _has(t, n) else f"{name}.all"


def _mlp_name(t: Mapping[str, torch.Tensor], branch: str) -> str:
    n = f"mlp.{branch}"
    return n if _has(t, n + ".proj_in") else "mlp.all"


class Transformer(nn.Module):
    """vid_in, the blocks, vid_out; constants (text tokens, AdaSingle sums) made at load.

    `blocks`: per block a module of its tensors (resident), or `host` streams them (see `fastforks.shared.host_blocks`).
    """

    def __init__(
        self,
        t: dict[str, torch.Tensor],
        cfg: Config,
        *,
        host: Callable[[torch.device], Iterable[Mapping[str, torch.Tensor]]] | None = None,
    ) -> None:
        """Build from the checkpoint tensors (on the device); `host(device)` yields the streamed blocks' tensors."""
        super().__init__()
        self.cfg, self.host = cfg, host
        with torch.no_grad():
            emb = time_embedding(t, t["txt"].device)
            self.consts = Consts(emb, t, cfg)
            self.register_buffer("txt", linear(t, "txt_in", t["txt"]))
        for k in ("vid_in.proj.weight", "vid_in.proj.bias", "vid_out.proj.weight", "vid_out.proj.bias"):
            self.register_buffer(k.replace(".", "_"), t[k])
        if cfg.out_norm:
            self.register_buffer("vid_out_norm", t["vid_out_norm.weight"])
        self.blocks = nn.ModuleList()
        if host is None:
            for i in range(cfg.layers):
                m = nn.Module()
                p = f"blocks.{i}."
                for k in block_keys(t, i):
                    m.register_buffer(k[len(p) :].replace(".", "__"), t[k])
                self.blocks.append(m)
        self.plans: dict[tuple[int, int, int, torch.device], tuple[Plan, Plan]] = {}
        self.txt_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.dequant: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None  # NF4 blocks: (q, absmax) -> bf16

    def linear(self, t: Mapping[str, torch.Tensor], name: str, x: torch.Tensor) -> torch.Tensor:
        """A block's linear `name`: its bf16 weight, or the NF4 one (`name.q`, `.absmax`) expanded by `dequant`."""
        w = t.get(name + ".weight")
        if w is None:
            if self.dequant is None:
                msg = f"{name} is quantised but no NF4 expansion is set"
                raise RuntimeError(msg)
            w = self.dequant(t[name + ".q"], t[name + ".absmax"])
        return F.linear(x, w, t.get(name + ".bias"))

    def _plans(self, size: tuple[int, int, int], device: torch.device) -> tuple[Plan, Plan]:
        key = (*size, device)
        if key not in self.plans:
            n = self.txt.shape[0] if self.cfg.mm_rope else None
            self.plans[key] = (
                make_plan(size, shifted=False, txt_len=n, device=device),
                make_plan(size, shifted=True, txt_len=n, device=device),
            )
        return self.plans[key]

    def _txt_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Text positions `0..57` on all three axes."""
        if self.txt_rope is None or self.txt_rope[0].device != self.txt.device:
            a = _angles(torch.arange(self.txt.shape[0])).repeat(1, 3)[:, None].to(self.txt.device)
            pad = HEAD_DIM - 3 * ROPE_AXIS
            self.txt_rope = (F.pad(a.cos(), (0, pad), value=1.0), F.pad(a.sin(), (0, pad), value=0.0))
        return self.txt_rope

    def _block_tensors(self, device: torch.device) -> Iterable[Mapping[str, torch.Tensor]]:
        if self.host is not None:
            for d in self.host(device):  # checkpoint keys `blocks.{i}.<name>`
                yield {k.split(".", 2)[2]: v for k, v in d.items()}
            return
        for m in self.blocks:
            yield {k.replace("__", "."): v for k, v in m.named_buffers()}

    def attention(
        self, t: Mapping[str, torch.Tensor], vid: torch.Tensor, txt: torch.Tensor, plan: Plan, *, last: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`NaSwinAttention` on modulated bf16 `vid [L, dim]`, `txt [58, dim]` -> the two outputs after `proj_out`."""
        heads = self.cfg.heads
        qkv = self.linear(t, _branch(t, "attn.proj_qkv", "vid"), vid).unflatten(-1, (3, heads, HEAD_DIM))
        tqkv = self.linear(t, _branch(t, "attn.proj_qkv", "txt"), txt).unflatten(-1, (3, heads, HEAD_DIM))
        nq, nk = _branch(t, "attn.norm_q", "vid"), _branch(t, "attn.norm_k", "vid")
        tnq, tnk = _branch(t, "attn.norm_q", "txt"), _branch(t, "attn.norm_k", "txt")
        q = rms_affine(qkv[:, 0], t[nq + ".weight"])
        k = rms_affine(qkv[:, 1], t[nk + ".weight"])
        v = qkv[:, 2]
        tq = rms_affine(tqkv[:, 0], t[tnq + ".weight"])
        tk = rms_affine(tqkv[:, 1], t[tnk + ".weight"])
        if self.cfg.mm_rope:
            tcos, tsin = self._txt_rope()
            tq, tk = rope(tq, tcos, tsin), rope(tk, tcos, tsin)
        tv = tqkv[:, 2]
        del qkv, tqkv
        n_txt = txt.shape[0]
        out = torch.empty_like(vid)
        txt_sum = None if last else torch.zeros(n_txt, heads, HEAD_DIM, dtype=torch.float32, device=vid.device)
        for g in plan.groups:
            nw, n = g.idx.shape
            gq, gk = rope(q[g.idx], g.cos, g.sin), rope(k[g.idx], g.cos, g.sin)
            gk = torch.cat([gk, tk.expand(nw, -1, -1, -1)], dim=1)
            gv = torch.cat([v[g.idx], tv.expand(nw, -1, -1, -1)], dim=1)
            if not last:
                gq = torch.cat([gq, tq.expand(nw, -1, -1, -1)], dim=1)
            o = F.scaled_dot_product_attention(gq.transpose(1, 2), gk.transpose(1, 2), gv.transpose(1, 2))
            o = o.transpose(1, 2)  # [nw, n (+ 58), H, D]
            out[g.idx.view(-1)] = o[:, :n].reshape(nw * n, -1)
            if txt_sum is not None:
                txt_sum += o[:, n:].float().sum(0)
            del gq, gk, gv, o
        vid_out = self.linear(t, _branch(t, "attn.proj_out", "vid"), out)
        if txt_sum is None:
            return vid_out, None
        txt_mean = (txt_sum / plan.count).to(BF).flatten(1)
        return vid_out, self.linear(t, _branch(t, "attn.proj_out", "txt"), txt_mean)

    def mlp(self, t: Mapping[str, torch.Tensor], x: torch.Tensor, branch: str) -> torch.Tensor:
        """SwiGLU (3B) / tanh-GELU (7B) MLP in bf16."""
        n = _mlp_name(t, branch)
        if self.cfg.swiglu:
            return self.linear(t, n + ".proj_out", F.silu(self.linear(t, n + ".proj_in_gate", x)) * self.linear(t, n + ".proj_in", x))
        return self.linear(t, n + ".proj_out", F.gelu(self.linear(t, n + ".proj_in", x), approximate="tanh"))

    def block(
        self, i: int, t: Mapping[str, torch.Tensor], vid: torch.Tensor, txt: torch.Tensor, plan: Plan
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`MMSRBlock`: attention and MLP with AdaSingle in / out (gate) and bf16 residual adds."""
        last = i == self.cfg.layers - 1
        c = self.consts
        vs, vc, vg = c.get(i, "vid", 0)
        va = rms(vid).mul_(vc).add_(vs)
        ta = rms(txt)
        if not (last and self.cfg.last_vid_only):  # 3B: the last block's text tokens enter attention unmodulated
            ts, tc, _ = c.get(i, "txt", 0)
            ta.mul_(tc).add_(ts)
        vo, to = self.attention(t, va, ta, plan, last=last)
        del va, ta
        vid = vo.mul_(vg).add_(vid)
        if to is not None:
            txt = to.mul_(c.get(i, "txt", 0)[2]).add_(txt)
        vs, vc, vg = c.get(i, "vid", 1)
        vid = self.mlp(t, rms(vid).mul_(vc).add_(vs), "vid").mul_(vg).add_(vid)
        if not last:
            ts, tc, tg = c.get(i, "txt", 1)
            txt = self.mlp(t, rms(txt).mul_(tc).add_(ts), "txt").mul_(tg).add_(txt)
        return vid, txt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[T, H, W, 33]` bf16 (noise, latent, ones; H, W even) -> the predicted velocity `[T, H, W, 16]` bf16."""
        tt, hh, ww, c = x.shape
        size = (tt, hh // 2, ww // 2)
        vid = x.view(tt, size[1], 2, size[2], 2, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4 * c)
        vid = F.linear(vid, self.vid_in_proj_weight, self.vid_in_proj_bias)
        txt = self.txt
        plans = self._plans(size, x.device)
        for i, t in enumerate(self._block_tensors(x.device)):
            vid, txt = self.block(i, t, vid, txt, plans[i % 2])
        if self.cfg.out_norm:
            vid = rms_affine(vid, self.vid_out_norm).mul_(self.consts.out_scale).add_(self.consts.out_shift)
        out = F.linear(vid, self.vid_out_proj_weight, self.vid_out_proj_bias)
        co = out.shape[-1] // 4
        return out.view(tt, size[1], size[2], 2, 2, co).permute(0, 1, 3, 2, 4, 5).reshape(tt, hh, ww, co)
