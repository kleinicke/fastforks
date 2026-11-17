"""Kernels-mode Depth Anything V2 on the `dpt_fused` kernels (CUTLASS fp16-accumulate GEMMs / convs, fused epilogues).

Same maths as upstream `infer_image` / `DepthAnythingV2.forward`, reordered where exact:

- input: bicubic resize (PyTorch / OpenCV cubic, a = -0.75) + ImageNet normalisation written straight as fp16 patch
  rows; the patch embedding is one GEMM whose epilogue adds the bias and the (cached) position embedding
- ViT blocks: qkv + bias, proj + bias * LayerScale + residual, fc1 + bias + GELU, fc2 + bias * LayerScale + residual
  as fused GEMMs (LayerScale in fp32 in the epilogue); attention = the fastbb flash kernel; LayerNorm in PyTorch.
  The patch-embedding GEMM accumulates in fp32 (K = 588 pixels; fp16 accumulation there dominated the error)
- DPT head in dense NHWC fp16: projections as GEMMs on the token rows, kernel == stride deconvolutions as one GEMM
  with a pixel-shuffle epilogue, residual conv units as two fused convs (ReLU / residual / skip add / ReLU copy in the
  epilogues), the fusion 1x1 `out_conv` before the bilinear upsample (both linear), NHWC bilinear kernel
- tail: 3x3 conv + ReLU in fp16, then the 1-channel 1x1 conv + ReLU in fp32 (one kernel)

Any structure the kernels do not cover raises `ValueError` (no PyTorch fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.kernels.dpt_fused import ACT_RELU, DptFused
from fastforks.shared.depth_anything_v2.fast_vit import POS_SIZES, FastViT, _flash

if TYPE_CHECKING:
    from collections.abc import Sequence

_Conv = tuple[torch.Tensor, torch.Tensor | None, int, int]  # weight [K, R, S, C] fp16, bias fp16, padding, stride


def _conv(m: nn.Module, device: torch.device) -> _Conv:
    if not (isinstance(m, nn.Conv2d) and m.groups == 1 and m.dilation == (1, 1) and m.padding_mode == "zeros"):
        msg = f"dpt_fused: unsupported layer {m}"
        raise ValueError(msg)
    if len(set(m.padding)) != 1 or len(set(m.stride)) != 1:
        msg = f"dpt_fused: non-square padding / stride {m}"
        raise ValueError(msg)
    w = m.weight.detach().half().permute(0, 2, 3, 1).contiguous().to(device)
    b = m.bias.detach().half().to(device) if m.bias is not None else None
    return w, b, int(m.padding[0]), int(m.stride[0])


def _linear_1x1(m: nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    w, b, pad, stride = _conv(m, device)
    if w.shape[1:3] != (1, 1) or pad or stride != 1:
        msg = f"dpt_fused: expected a 1x1 conv, got {m}"
        raise ValueError(msg)
    return w.view(w.shape[0], -1), b


class FusedViT(FastViT):
    """DINOv2 with fused-epilogue GEMMs (`features`); weights converted on the CPU, then moved to `device`."""

    def __init__(  # noqa: PLR0913 - the switches
        self,
        vit: Any,  # noqa: ANN401 - duck-typed DINOv2 module
        device: torch.device,
        k: DptFused,
        attn: str = "hh",
        gelu: int | None = None,
        *,
        res32: bool = False,
    ) -> None:
        """Check and convert the ViT (fp16, LayerScale kept in fp32), move it to `device`.

        `attn`: the flash variant; `gelu`: the MLP epilogue (default `k.gelu`, the card's). `res32`: also keep the
        LayerNorms in fp32 for `block32` / `norm32` (a float32 residual stream).
        """
        blocks = list(vit.blocks)
        ln32 = [
            [(n.weight.detach().float().to(device), n.bias.detach().float().to(device), n.eps) for n in norms]
            for norms in ([(b.norm1, b.norm2) for b in blocks] + [(vit.norm,)] if res32 else [])
        ]
        gammas = [
            tuple(
                g.detach().float().contiguous() if (g := getattr(getattr(b, n), "gamma", None)) is not None else None
                for n in ("ls1", "ls2")
            )
            for b in blocks
        ]
        super().__init__(vit, attn, None)  # structure checks, fp16 in place, fp32 position table
        if vit.chunked_blocks or len(self.blocks) != len(blocks) or vit.register_tokens is not None:
            msg = "dpt_fused ViT: chunked blocks / register tokens not supported"
            raise ValueError(msg)
        pe = vit.patch_embed
        conv = pe.proj
        ps = vit.patch_size
        if not (isinstance(pe.norm, nn.Identity) and conv.kernel_size == (ps, ps) and conv.stride == (ps, ps)):
            msg = "dpt_fused ViT: patch embedding must be a stride == kernel conv without norm"
            raise ValueError(msg)
        c = vit.embed_dim
        kin = 3 * ps * ps
        self.kp = -(-kin // 8) * 8
        wp = torch.zeros(c, self.kp, dtype=torch.float16)
        wp[:, :kin] = conv.weight.detach().reshape(c, kin)
        self.k = k
        self.gelu = k.gelu if gelu is None else gelu
        self.w_patch = wp.to(device)
        self.b_patch = conv.bias.detach().half().to(device)
        self.gammas = [tuple(g.to(device) if g is not None else None for g in gs) for gs in gammas]
        vit.patch_embed = nn.Identity()  # replaced by w_patch
        vit.to(device)
        self._tables: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.ln32 = ln32

    def _pos_tables(self, gh: int, gw: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(position embedding of the patch tokens `[gh * gw, C]`, cls token + its position `[C]`), cached."""
        key = (gh, gw)
        if key not in self._tables:
            if len(self._tables) >= POS_SIZES:
                del self._tables[next(iter(self._tables))]
            pe = self.pos_embed(gh, gw)[0]
            self._tables[key] = (pe[1:].contiguous(), (self.vit.cls_token[0, 0] + pe[0]).contiguous())
        return self._tables[key]

    def _block(self, i: int, x: torch.Tensor) -> torch.Tensor:
        blk = self.blocks[i]
        a, m = blk.attn, blk.mlp
        g1, g2 = self.gammas[i]
        k = self.k
        b, n, c = x.shape
        h = a.num_heads
        qkv = k.gemm(blk.norm1(x), a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h)
        k.gemm(_flash(qkv, (c // h) ** -0.5, self.attn), a.proj.weight, a.proj.bias, scale=g1, res=x, out=x)
        hid = k.gemm(blk.norm2(x), m.fc1.weight, m.fc1.bias, act=self.gelu)
        k.gemm(hid, m.fc2.weight, m.fc2.bias, scale=g2, res=x, out=x)
        return x

    def norm32(self, x: torch.Tensor, i: int = -1, j: int = 0) -> torch.Tensor:
        """fp16 LayerNorm (fp32 maths) of the float32 stream `x`: block `i`'s norm1 / norm2 (`j`), default the last."""
        w, b, eps = self.ln32[i][j]
        return F.layer_norm(x, w.shape, w, b, eps).half()

    def block32(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """`_block` on a float32 residual stream `x` (in place): fp16 GEMMs, residual adds and LayerNorms in fp32.

        Upstream runs fp16 / bf16 autocast, whose residual stream stays float32; models that copy metric inputs
        through the ViT (LingBot-Depth's depth tokens) lose accuracy on an fp16 stream.
        """
        blk = self.blocks[i]
        a, m = blk.attn, blk.mlp
        g1, g2 = self.gammas[i]
        k = self.k
        b, n, c = x.shape
        h = a.num_heads
        qkv = k.gemm(self.norm32(x, i, 0), a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h)
        x += k.gemm(_flash(qkv, (c // h) ** -0.5, self.attn), a.proj.weight, a.proj.bias, scale=g1)
        hid = k.gemm(self.norm32(x, i, 1), m.fc1.weight, m.fc1.bias, act=self.gelu)
        x += k.gemm(hid, m.fc2.weight, m.fc2.bias, scale=g2)
        return x

    def embed_tokens(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """The ViT's input tokens `[B, 1 + gh * gw, C]` (fp16: cls, patches + positions) of float `[B, 3, H, W]`.

        `image` in [0, 1] is resized to `size` and ImageNet-normalised by the patch kernel.
        """
        vit, k = self.vit, self.k
        ps = vit.patch_size
        gh, gw = size[0] // ps, size[1] // ps
        cols = k.patches(image, size, ps, self.kp, IMAGENET_MEAN, IMAGENET_STD)
        pos, cls = self._pos_tables(gh, gw)
        b = image.shape[0]
        t = torch.empty((b, gh * gw + 1, vit.embed_dim), device=image.device, dtype=torch.float16)
        t[:, 0] = cls
        for i in range(b):
            k.gemm(cols[i], self.w_patch, self.b_patch, res=pos, out=t[i, 1:], acc32=True)
        return t

    def features(self, image: torch.Tensor, size: tuple[int, int], layers: Sequence[int]) -> list[torch.Tensor]:
        """Normalised patch tokens `[B, gh * gw, C]` (fp16, no cls) of `layers` for float `[B, 3, H, W]` in [0, 1]."""
        vit = self.vit
        t = self.embed_tokens(image, size)
        outs = []
        last = max(layers)
        for i in range(last + 1):
            t = self._block(i, t)
            if i in layers:
                outs.append(vit.norm(t[:, 1:]))
        return outs


def _resize(r: nn.Module, device: torch.device) -> tuple | None:
    """A DPT resize layer: None (identity), `("deconv", w [s*s*Cout, Cin], b, s)` or `("conv", *_conv)`."""
    if isinstance(r, nn.Identity):
        return None
    if isinstance(r, nn.ConvTranspose2d):
        s = r.stride[0]
        if r.kernel_size != (s, s) or r.stride != (s, s) or r.padding != (0, 0) or r.groups != 1:
            msg = f"dpt_fused head: deconvolution {r} is not kernel == stride"
            raise ValueError(msg)
        w = r.weight.detach().half()  # [Cin, Cout, s, s] -> rows (a, b, cout)
        wg = w.permute(2, 3, 1, 0).reshape(s * s * w.shape[1], w.shape[0]).contiguous().to(device)
        bg = r.bias.detach().half().repeat(s * s).to(device) if r.bias is not None else None
        return ("deconv", wg, bg, s)
    return ("conv", *_conv(r, device))


class FusedHead:
    """The DPT head on dense NHWC fp16 tensors (see the module docstring)."""

    def __init__(self, head: Any, device: torch.device, k: DptFused) -> None:  # noqa: ANN401 - duck-typed DPTHead
        """Convert the head weights (CPU), move them to `device`."""
        if head.use_clstoken:
            msg = "dpt_fused head: use_clstoken not supported"
            raise ValueError(msg)
        self.k = k
        self.projects = [_linear_1x1(p, device) for p in head.projects]
        self.resize = [_resize(r, device) for r in head.resize_layers]
        s = head.scratch
        self.rn = [_conv(m, device) for m in (s.layer1_rn, s.layer2_rn, s.layer3_rn, s.layer4_rn)]
        self.fusion = []
        for f in (s.refinenet1, s.refinenet2, s.refinenet3, s.refinenet4):
            for u in (f.resConfUnit1, f.resConfUnit2):
                if u.bn or not isinstance(u.activation, nn.ReLU):
                    msg = "dpt_fused head: residual conv units with batch norm / non-ReLU not supported"
                    raise ValueError(msg)
            if f.deconv or f.expand or not f.align_corners or f.size is not None:
                msg = "dpt_fused head: unsupported fusion block options"
                raise ValueError(msg)
            rcu = [(_conv(u.conv1, device), _conv(u.conv2, device)) for u in (f.resConfUnit1, f.resConfUnit2)]
            self.fusion.append((rcu, _linear_1x1(f.out_conv, device)))
        self.oc1 = _conv(s.output_conv1, device)
        oc2 = s.output_conv2
        last = oc2[2]
        if not (isinstance(oc2[1], nn.ReLU) and isinstance(oc2[3], (nn.ReLU, nn.Sigmoid)) and last.out_channels == 1):
            msg = "dpt_fused head: unexpected output_conv2"
            raise ValueError(msg)
        self.sigmoid = isinstance(oc2[3], nn.Sigmoid)  # metric head (upstream metric_depth)
        self.oc2 = _conv(oc2[0], device)
        self.w_last = last.weight.detach().float().flatten().contiguous().to(device)
        self.b_last = float(last.bias.detach().float()[0])
        self.w_neg = -self.w_last

    def _conv(self, x: torch.Tensor, c: _Conv, **kw: Any) -> torch.Tensor:  # noqa: ANN401 - epilogue options
        w, b, pad, stride = c
        return self.k.conv(x, w, b, padding=pad, stride=stride, **kw)

    def _fuse(  # noqa: PLR0913, PLR0917 - fusion block inputs
        self,
        i: int,
        x0: torch.Tensor,
        r0: torch.Tensor | None,
        x1: torch.Tensor | None,
        r1: torch.Tensor | None,
        size: tuple[int, int],
    ) -> torch.Tensor:
        """FeatureFusionBlock: `x0 (+ rcu1(x1))`, rcu2, out_conv, bilinear; `r*` = relu of the inputs."""
        (u1, u2), (wo, bo) = self.fusion[i]
        if x1 is not None:
            t = self._conv(r1, u1[0], act=ACT_RELU)  # ty: ignore[invalid-argument-type]
            r0 = torch.empty_like(x1)
            x0 = self._conv(t, u1[1], res=x1, res2=x0, y2=r0)
        t = self._conv(r0, u2[0], act=ACT_RELU)  # ty: ignore[invalid-argument-type]
        z = self._conv(t, u2[1], res=x0)
        return self.k.bilinear(self.k.gemm(z, wo, bo), size, align_corners=True)

    def __call__(self, feats: list[torch.Tensor], gh: int, gw: int) -> torch.Tensor:
        """Tokens `[B, gh * gw, C]` of the 4 layers -> relative inverse depth `[B, 14 gh, 14 gw]` (fp32).

        A metric head returns the sigmoid instead of the ReLU of the last conv (the caller scales it by max_depth).
        """
        k = self.k
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
        p = self._fuse(3, l4, rs[3], None, None, l3.shape[1:3])
        p = self._fuse(2, p, None, l3, rs[2], l2.shape[1:3])
        p = self._fuse(1, p, None, l2, rs[1], l1.shape[1:3])
        p = self._fuse(0, p, None, l1, rs[0], (2 * l1.shape[1], 2 * l1.shape[2]))
        o = k.bilinear(self._conv(p, self.oc1), (14 * gh, 14 * gw), align_corners=True)
        o = self._conv(o, self.oc2, act=ACT_RELU)
        z = k.head_dot(o, self.w_last, self.b_last)
        if not self.sigmoid:
            return z
        # the kernel ends in a ReLU: the fp32 pre-activation is relu(z) - relu(-z), a second pass over the 32 channels
        return torch.sigmoid(z - k.head_dot(o, self.w_neg, -self.b_last))
