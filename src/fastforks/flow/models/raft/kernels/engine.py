"""RAFT kernels engine (raft-large and raft-small): batch 1, fp16 NHWC activations, fp32 flow state.

- No correlation volume: the 4-level pyramid of RAFT is avgpool^l(corr) = corr(f1, avgpool^l(f2)) (pooling is linear),
  so only the pooled fmap2 pyramid is kept (fp16, O(HW) memory) and `corr_lookup` takes the dot products of each
  pixel's (2r + 2)^2 window on the fly (fp32 accumulation) and interpolates them bilinearly, as grid_sample.
- Convolutions: CUTLASS implicit-GEMM convs with the bias / BatchNorm / activation / GRU math fused into the epilogue
  (`cutlass_conv`), cuDNN fp16-accumulate convs (`cudnn_hacc`) for the InstanceNorm layers and the RGB stems.
- GRU inputs: A = [h | context | motion | flow] for the z|r conv and B = [r*h | context | motion | flow] for the q
  conv; the context is written into both once, the motion features by the motion conv's epilogue (two outputs), the
  flow by `write_flow`, r*h by the z|r epilogue; the state update runs in place on A's h.
- Everything else (InstanceNorm, the lookup, the 7 x 7 flow conv, the flow head output + coords update, upsampling)
  is a custom kernel (`raft_flow`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.kernels import linear
from fastforks.kernels.cutlass_conv import ConvEpiArgs, narrow_cfg

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.kernels.cudnn_hacc import CudnnHalfConv
    from fastforks.kernels.cutlass_conv import FusedConvs

IN_CHUNK = 1024  # fixed: pixels per InstanceNorm partial-statistics block
_BIG = 10000  # fixed: pixel count from which the 128 x 128 CUTLASS tiles fill the GPU
MIN_CELLS = 16  # fixed: 1/8 cells per side the 4-level pyramid needs (the card's `min_size` 128; coarsest level 2 x 2)


class S:
    """Channel slice [c0, c0 + c) of an NHWC buffer [H, W, Ctot] (fp16, contiguous)."""

    __slots__ = ("buf", "c", "c0", "ctot", "h", "w")

    def __init__(self, buf: torch.Tensor, c0: int = 0, c: int | None = None) -> None:
        self.buf, self.c0 = buf, c0
        self.h, self.w, self.ctot = buf.shape
        self.c = self.ctot - c0 if c is None else c

    @property
    def ptr(self) -> int:
        return self.buf.data_ptr() + 2 * self.c0

    @property
    def p(self) -> int:
        return self.h * self.w

    @property
    def full(self) -> bool:
        return self.c0 == 0 and self.c == self.ctot

    def t4(self) -> torch.Tensor:
        """Strided `[1, C, H, W]` view of the slice (no copy)."""
        return self.buf[:, :, self.c0 : self.c0 + self.c].permute(2, 0, 1)[None]


def _krsc(w: torch.Tensor, device: torch.device) -> torch.Tensor:
    return w.detach().half().permute(0, 2, 3, 1).contiguous().to(device)


def _pad_in(w: torch.Tensor, c: int) -> torch.Tensor:
    """Zero input channels up to c (16-byte rows)."""
    return torch.cat([w, w.new_zeros(w.shape[0], c - w.shape[1], *w.shape[2:])], 1)


def _pad_out(w: torch.Tensor, b: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero output channels up to k."""
    return torch.cat([w, w.new_zeros(k - w.shape[0], *w.shape[1:])]), torch.cat([b, b.new_zeros(k - b.shape[0])])


def _up8(n: int) -> int:
    return (n + 7) // 8 * 8


class Weights:
    """Inference weights from a torchvision RAFT state dict: KRSC fp16 conv weights, BatchNorm folded into per-channel
    fp32 scale `s` / shift `t`, conv biases as `t` (InstanceNorm layers: dropped, they cancel)."""

    def __init__(self, sd: dict[str, torch.Tensor], device: torch.device, *, large: bool) -> None:
        sd = {k: v.float() for k, v in sd.items()}
        self.w: dict[str, torch.Tensor] = {}
        self.s: dict[str, torch.Tensor] = {}
        self.t: dict[str, torch.Tensor] = {}
        dev = device

        def conv(name: str, key: str) -> torch.Tensor:
            self.w[name] = _krsc(sd[key + ".weight"], dev)
            return sd[key + ".bias"]

        def cna(name: str, key: str, norm: str | None) -> None:
            """`_cna` / conv: conv at `key`.0 (or `key` itself), norm at `key`.1."""
            b = conv(name, key + ".0")
            if norm == "batch":  # BN(conv(x) + b) = conv(x) * s + ((b - mean) * s + beta)
                s = sd[key + ".1.weight"] / torch.sqrt(sd[key + ".1.running_var"] + 1e-5)
                self.s[name] = s.to(dev).contiguous()
                self.t[name] = ((b - sd[key + ".1.running_mean"]) * s + sd[key + ".1.bias"]).to(dev).contiguous()
            elif norm is None:
                self.t[name] = b.to(dev).contiguous()

        for enc, norm in (("feature_encoder", "instance"), ("context_encoder", "batch" if large else None)):
            cna(f"{enc}.stem", f"{enc}.convnormrelu", norm)
            for li in (1, 2, 3):
                for bi in (0, 1):
                    k = f"{enc}.layer{li}.{bi}"
                    for ci in (1, 2) if large else (1, 2, 3):
                        cna(f"{k}.c{ci}", f"{k}.convnormrelu{ci}", norm)
                    if bi == 0 and li > 1:
                        cna(f"{k}.ds", f"{k}.downsample", norm)
            self.t[f"{enc}.conv"] = conv(f"{enc}.conv", f"{enc}.conv").to(dev)

        m = "update_block.motion_encoder"
        look = _up8(sd[f"{m}.convcorr1.0.weight"].shape[1])
        self.look = look
        self.w["convc1"] = _krsc(_pad_in(sd[f"{m}.convcorr1.0.weight"], look), dev)
        self.t["convc1"] = sd[f"{m}.convcorr1.0.bias"].to(dev)
        if large:
            cna("convc2", f"{m}.convcorr2", None)
        self.convf1_w = sd[f"{m}.convflow1.0.weight"].contiguous().to(dev)  # fp32 [K, 2, 7, 7]
        self.convf1_b = sd[f"{m}.convflow1.0.bias"].contiguous().to(dev)
        cna("convf2", f"{m}.convflow2", None)
        w, b = sd[f"{m}.conv.0.weight"], sd[f"{m}.conv.0.bias"]
        self.motion = w.shape[0]  # 126 / 80 features, then the 2 flow channels
        w, b = _pad_out(w, b, _up8(w.shape[0]))
        self.w["mconv"], self.t["mconv"] = _krsc(w, dev), b.to(dev)

        r = "update_block.recurrent_block"
        self.gru: list[dict[str, torch.Tensor]] = []
        for g in ("convgru1", "convgru2") if large else ("convgru1",):
            k = f"{r}.{g}"
            zr = torch.cat([sd[k + ".convz.weight"], sd[k + ".convr.weight"]], 0)
            wide = _up8(zr.shape[1])
            self.gru.append(
                {
                    "zr": _krsc(_pad_in(zr, wide), dev),
                    "q": _krsc(_pad_in(sd[k + ".convq.weight"], wide), dev),
                    "bzr": torch.cat([sd[k + ".convz.bias"], sd[k + ".convr.bias"]]).half().to(dev),
                    "bzr32": torch.cat([sd[k + ".convz.bias"], sd[k + ".convr.bias"]]).to(dev),
                    "bq": sd[k + ".convq.bias"].half().to(dev),
                }
            )
        f = "update_block.flow_head"
        self.w["fh1"] = _krsc(sd[f"{f}.conv1.weight"], dev)
        self.t["fh1"] = sd[f"{f}.conv1.bias"].to(dev)
        self.fh2_w = _krsc(sd[f"{f}.conv2.weight"], dev)  # [2, 3, 3, FH]
        self.fh2_b = sd[f"{f}.conv2.bias"].half().to(dev)
        if large:
            cna("mask1", "mask_predictor.convrelu", None)
            self.w["mask2"] = _krsc(sd["mask_predictor.conv.weight"], dev)
            self.t["mask2"] = sd["mask_predictor.conv.bias"].to(dev)


def check(image1: torch.Tensor, image2: torch.Tensor) -> None:
    """Raise ValueError unless both are float32 `[B, 3, H, W]` of one shape, H and W multiples of 8 and at least
    `8 * MIN_CELLS`: the kernels index their buffers from these sizes without further checks."""
    if image1.shape != image2.shape or image1.ndim != 4 or image1.shape[1] != 3:
        raise ValueError(
            f"expected two [B, 3, H, W] batches of one shape, got {tuple(image1.shape)}, {tuple(image2.shape)}"
        )
    if image1.dtype != torch.float32 or image2.dtype != torch.float32:
        raise ValueError(f"expected float32 images, got {image1.dtype}, {image2.dtype}")
    h, w = image1.shape[-2:]
    if h % 8 or w % 8 or min(h, w) < 8 * MIN_CELLS:
        raise ValueError(f"H and W must be multiples of 8 and at least {8 * MIN_CELLS}, got {h} x {w}")


class Engine:
    """The kernels forward pass of torchvision's raft_large / raft_small."""

    def __init__(
        self,
        sd: dict[str, torch.Tensor],
        device: torch.device,
        k: CubinModule,
        fused: FusedConvs,
        convs: CudnnHalfConv,
        iters: int,
        *,
        large: bool,
    ) -> None:
        """Convert the weights; `k`, `fused`, `convs` are the loaded `raft_flow`, `cutlass_conv`, `cudnn_hacc`."""
        self.device, self.large, self.iters = device, large, iters
        self.W = Weights(sd, device, large=large)
        self.k, self.fused, self.convs = k, fused, convs
        self.fdim = 256 if large else 128  # feature channels
        self.radius = 4 if large else 3
        self.hd = 128 if large else 96  # hidden state
        self.cc = 128 if large else 64  # context
        # (layer, input height, width, stride) the CUTLASS conv refused: asked once per geometry, not per call
        self._unfusable: set[tuple[str, int, int, int]] = set()

    def _new(self, h: int, w: int, c: int) -> torch.Tensor:
        return torch.empty(h, w, c, device=self.device, dtype=torch.half)

    # ---- convolutions
    def conv_raw(self, x: S, w: torch.Tensor, y: S, stride: int = 1) -> None:
        kk, r, _, c = w.shape
        if c != x.c or kk != y.c:  # the kernel trusts these: a mismatch would read / write out of bounds
            raise ValueError(f"conv weight {tuple(w.shape)} does not fit {x.c} -> {y.c} channels")
        self.convs.conv_into(x.t4(), w.permute(0, 3, 1, 2), y.t4(), r // 2, stride)

    def conv(self, x: S, name: str, stride: int = 1, out: S | None = None) -> S:
        w = self.W.w[name]
        if out is None:
            out = S(self._new((x.h - 1) // stride + 1, (x.w - 1) // stride + 1, w.shape[0]))
        self.conv_raw(x, w, out, stride)
        return out

    def affine(self, y: S, s: torch.Tensor | None, t: torch.Tensor | None, mode: int, *, out: S, out2: S | None = None,
               r: S | None = None) -> None:  # fmt: skip
        """out (+ out2) = act(y * s + t); mode 0 none, 1 ReLU, 2 tanh, 3 ReLU(ReLU(.) + r)."""
        self.k.launch(
            "affine_act", *linear(y.p * y.c // 8), y.ptr, y.ctot, out.ptr, out.ctot,
            0 if out2 is None else out2.ptr, 0 if out2 is None else out2.ctot, s, t,
            0 if r is None else r.ptr, 0 if r is None else r.ctot, y.p, y.c, mode,
        )  # fmt: skip

    def conv_act(
        self,
        x: S,
        name: str,
        mode: int,
        stride: int = 1,
        out: S | None = None,
        out2: S | None = None,
        r: S | None = None,
    ) -> S:
        """act(conv(x) * s + t): one fused-epilogue CUTLASS conv where the operands allow it, else cuDNN + affine_act.

        mode: 0 none, 1 ReLU, 2 tanh, 3 ReLU(ReLU(.) + r)."""
        w = self.W.w[name]
        kk, _, _, c = w.shape
        ho, wo = (x.h - 1) // stride + 1, (x.w - 1) // stride + 1
        o = S(self._new(ho, wo, kk)) if out is None else out
        s, t = self.W.s.get(name), self.W.t.get(name)
        key = (name, x.h, x.w, stride)
        if x.full and c % 8 == 0 and key not in self._unfusable:
            p = ho * wo
            cfg = narrow_cfg(p, kk, x.buf.device) if kk < 128 else (0 if p >= _BIG else 1)
            e = ConvEpiArgs(
                2, mode, 0 if r is None else r.ptr, 0 if r is None else r.ctot, o.ptr, o.ctot, None,
                0 if out2 is None else out2.ctot, None if out2 is None else out2.ptr, None,
                None if s is None else s.data_ptr(), None if t is None else t.data_ptr(), None, 0,
            )  # fmt: skip
            if self.fused.conv(cfg, x.buf, w, (x.h, x.w), e, stride=stride):
                return o
            self._unfusable.add(key)
        y = self.conv(x, name, stride)
        self.affine(y, s, t, mode, out=o, out2=out2, r=r)
        return o

    def in_stats(self, y: S) -> torch.Tensor:
        if not y.full:
            raise ValueError("instance-norm statistics need a full (unpadded) tensor")
        p, c = y.p, y.c
        nblk = (p + IN_CHUNK - 1) // IN_CHUNK
        part = torch.empty(nblk * 2 * c, device=self.device, dtype=torch.float32)
        st = torch.empty(2 * c, device=self.device, dtype=torch.float32)
        cv = c // 8
        self.k.launch("in_stats_partial", (nblk, 1, 1), ((256 // cv) * cv, 1, 1), y.ptr, p, c, IN_CHUNK, part)
        self.k.launch("in_stats_final", (c, 1, 1), (32, 1, 1), part, nblk, c, p, IN_CHUNK, 1e-5, st)
        return st

    def in_apply(
        self, y: S, st: torch.Tensor, mode: int = 0, r: S | None = None, st_r: torch.Tensor | None = None
    ) -> None:
        """y = IN(y), then mode 0: ReLU(y), 1: ReLU(ReLU(y) + IN(r)), 2: ReLU(ReLU(y) + r) (in place)."""
        n = y.p * y.c
        self.k.launch("in_apply", *linear(n // 8), y.ptr, st, 0 if r is None else r.ptr, st_r, y.ptr, n, y.c, mode)

    def in_conv(self, x: S, name: str, stride: int = 1) -> S:
        """relu(IN(conv(x)))."""
        y = self.conv(x, name, stride)
        self.in_apply(y, self.in_stats(y))
        return y

    # ---- encoders
    def block(self, x: S, name: str, stride: int, *, norm: bool) -> S:
        """Residual (large) or bottleneck (small) block; norm: InstanceNorm (feature encoder), else folded BN / bias."""
        ds = name.endswith(".0") and ".layer1." not in name
        convs = ("c1", "c2") if self.large else ("c1", "c2", "c3")
        strides = (stride, 1) if self.large else (1, stride, 1)
        if not norm:
            r = self.conv_act(x, name + ".ds", 0, stride) if ds else x
            y = x
            for c, st in zip(convs[:-1], strides[:-1], strict=True):
                y = self.conv_act(y, f"{name}.{c}", 1, st)
            return self.conv_act(y, f"{name}.{convs[-1]}", 3, r=r)
        d = st_d = None
        if ds:
            d = self.conv(x, name + ".ds", stride)
            st_d = self.in_stats(d)
        y = x
        for c, st in zip(convs[:-1], strides[:-1], strict=True):
            y = self.in_conv(y, f"{name}.{c}", st)
        o = self.conv(y, f"{name}.{convs[-1]}")
        del y
        so = self.in_stats(o)
        if ds:
            self.in_apply(o, so, 1, d, st_d)
        else:
            self.in_apply(o, so, 2, x)
        return o

    def encoder(self, img: torch.Tensor, enc: str) -> S:
        """[3, H, W] float32 -> 1/8 feature map before the final 1 x 1 conv."""
        _, h, w = img.shape
        x = S(self._new(h, w, 3))
        self.k.launch("prep_image", *linear(h * w), img, x.buf, h * w)
        norm = enc == "feature_encoder"
        if norm:
            x = self.in_conv(x, f"{enc}.stem", 2)
        else:
            x = self.conv_act(x, f"{enc}.stem", 1, 2)
        for li, stride in ((1, 1), (2, 2), (3, 2)):
            x = self.block(x, f"{enc}.layer{li}.0", stride, norm=norm)
            x = self.block(x, f"{enc}.layer{li}.1", 1, norm=norm)
        return x

    # ---- update loop
    def _gru_calls(self, L: dict[str, Any]) -> list[tuple[Callable[[], None], Callable[[], None] | None]]:
        """Launchers of the GRUs (the arguments are the same every iteration: built once per call)."""
        A, B, zr = L["A"], L["B"], L["zr"]
        h, w, wide = A.shape
        p, hd = h * w, self.hd
        calls = []
        for gi, g in enumerate(self.W.gru):
            pad = ((0, 2), (2, 0))[gi] if self.large else (1, 1)
            if hd == 128:
                zr_call = self.fused.bind(  # z -> zr[:, :128], r*h -> B[:, :128]
                    0 if p >= _BIG else 1, A, g["zr"], (h, w),
                    ConvEpiArgs(0, 0, g["bzr"].data_ptr(), 0, zr.data_ptr(), 2 * hd, A.data_ptr(), wide, B.data_ptr()),
                    padding=pad,
                )  # fmt: skip
                gate = None
            else:  # raw z|r conv, then the gate kernel
                zr_call = self.fused.bind(
                    0 if p >= _BIG else 1, A, g["zr"], (h, w),
                    ConvEpiArgs(2, 0, None, 0, zr.data_ptr(), 2 * hd), padding=pad,
                )  # fmt: skip
                bz = g["bzr32"]

                def gate(bz: torch.Tensor = bz) -> None:
                    self.k.launch("gru_zr", *linear(p * hd // 8), zr, 2 * hd, bz, A, B, wide, p, hd)

            last = gi == len(self.W.gru) - 1
            q_call = self.fused.bind(  # h updated in place in A (not read by the q conv), + a packed copy for the heads
                (0 if p >= _BIG else 1) if hd == 128 else narrow_cfg(p, hd, A.device),
                B, g["q"], (h, w),
                ConvEpiArgs(1, 0, g["bq"].data_ptr(), 0, zr.data_ptr(), 2 * hd, A.data_ptr(), wide, None,
                            L["h"].data_ptr() if last else None, None, None, None, 0, hd),
                padding=pad,
            )  # fmt: skip
            calls.append((zr_call, gate, q_call))
        return calls

    def motion(self, L: dict[str, Any]) -> None:
        k, W = self.k, self.W
        h8, w8, p = L["h8"], L["w8"], L["p8"]
        A, B, M = L["A"], L["B"], L["M"]
        cx = self.hd + self.cc  # motion features start here in A / B
        kname = "corr_lookup_c256_r4" if self.large else "corr_lookup_c128_r3"
        k.launch(kname, ((p + 7) // 8, 1, 1), (256, 1, 1), L["f1"], *L["pyr"], h8, w8, L["coords"], L["look"],
                 W.look, 1.0 / math.sqrt(self.fdim))  # fmt: skip
        nc = 192 if self.large else 96
        if self.large:
            c1 = self.conv_act(S(L["look"]), "convc1", 1, out=S(L["c1"]))
            self.conv_act(c1, "convc2", 1, out=S(M, 0, nc))
        else:
            self.conv_act(S(L["look"]), "convc1", 1, out=S(M, 0, nc))
        kf = W.convf1_w.shape[0]
        grid = ((w8 + 15) // 16, (h8 + 15) // 16, kf // 64)
        k.launch("convflow1", grid, (16, 16, 1), L["coords"], h8, w8, W.convf1_w, W.convf1_b, L["fl1"], kf)
        self.conv_act(S(L["fl1"]), "convf2", 1, out=S(M, nc, M.shape[2] - nc))
        mk = W.w["mconv"].shape[0]
        self.conv_act(S(M), "mconv", 1, out=S(A, cx, mk), out2=S(B, cx, mk))
        n = 2 if self.large else 8
        k.launch("write_flow", *linear(p), L["coords"], w8, p, A.data_ptr() + 2 * (cx + W.motion),
                 B.data_ptr() + 2 * (cx + W.motion), A.shape[2], n)  # fmt: skip

    def iteration(self, L: dict[str, Any]) -> None:
        self.motion(L)
        for zr_call, gate, q_call in L["gru"]:
            zr_call()
            if gate is not None:
                gate()
            q_call()
        fh = self.conv_act(S(L["h"]), "fh1", 1, out=S(L["fh"]))
        fd = fh.c
        self.k.launch(f"flow_head_out_{fd}", ((L["p8"] * 32 + 255) // 256, 1, 1), (256, 1, 1), fh.buf, L["h8"],
                      L["w8"], self.W.fh2_w, self.W.fh2_b, L["coords"])  # fmt: skip

    # ---- full pass
    @torch.inference_mode()
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """image1, image2: float32 `[B, 3, H, W]` in [-1, 1], H and W multiples of 8. Returns float32 `[B, 2, H, W]`.

        One pair at a time: the pass is GPU-bound from KITTI sizes up, so batching would only add memory.

        Raises:
            ValueError: Inputs the kernels would read or write out of bounds with (see `check`).
        """
        check(image1, image2)
        with torch.cuda.device(self.device):
            return torch.stack([self._run(a.contiguous(), b.contiguous()) for a, b in zip(image1, image2, strict=True)])

    def _run(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        k, W = self.k, self.W
        _, hh, ww = image1.shape
        h8, w8 = hh // 8, ww // 8
        p = h8 * w8
        hd, cc = self.hd, self.cc

        # feature maps: f1 and the pooled f2 pyramid
        f1 = S(self._new(h8, w8, self.fdim))
        self.conv_act(self.encoder(image1, "feature_encoder"), "feature_encoder.conv", 0, out=f1)
        pyr = [S(self._new(h8, w8, self.fdim))]
        self.conv_act(self.encoder(image2, "feature_encoder"), "feature_encoder.conv", 0, out=pyr[0])
        for _ in range(3):
            src = pyr[-1]
            dst = S(self._new(src.h // 2, src.w // 2, self.fdim))
            k.launch("avgpool2", *linear(dst.p * self.fdim // 8), src.buf, src.h, src.w, dst.buf, dst.h, dst.w,
                     self.fdim)  # fmt: skip
            pyr.append(dst)

        # context -> h (tanh) and context (relu), into both GRU inputs
        ctx = self.conv_act(self.encoder(image1, "context_encoder"), "context_encoder.conv", 0)
        wide = _up8(hd + cc + W.motion + 2)
        A, B = self._new(h8, w8, wide), self._new(h8, w8, wide)
        self.affine(S(ctx.buf, 0, hd), None, None, 2, out=S(A, 0, hd))
        self.affine(S(ctx.buf, hd, cc), None, None, 1, out=S(A, hd, cc), out2=S(B, hd, cc))
        del ctx

        coords = torch.empty(p, 2, device=self.device, dtype=torch.float32)
        k.launch("init_coords", *linear(p), coords, w8, p)
        L: dict[str, Any] = {
            "h8": h8, "w8": w8, "p8": p, "A": A, "B": B, "coords": coords, "f1": f1.buf,
            "pyr": [s.buf for s in pyr], "look": self._new(h8, w8, W.look),
            "zr": self._new(h8, w8, 2 * hd), "h": self._new(h8, w8, hd),
            "c1": self._new(h8, w8, W.w["convc1"].shape[0]) if self.large else None,
            "M": self._new(h8, w8, W.w["mconv"].shape[3]),
            "fl1": self._new(h8, w8, W.convf1_w.shape[0]),
        }  # fmt: skip
        L["fh"] = self._new(h8, w8, W.w["fh1"].shape[0])
        L["gru"] = self._gru_calls(L)
        for _ in range(self.iters):
            self.iteration(L)
        hstate = L["h"]
        del L, A, B, pyr, f1

        out = torch.empty(2, hh, ww, device=self.device, dtype=torch.float32)
        if self.large:
            m1 = self.conv_act(S(hstate), "mask1", 1)
            m2 = self.conv_act(m1, "mask2", 0)
            k.launch("upsample_convex", *linear(hh * ww), coords, m2.buf, h8, w8, out)
        else:
            k.launch("upsample_bilinear", *linear(hh * ww), coords, h8, w8, out)
        return out
