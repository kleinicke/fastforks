"""RAFT-Stereo kernels engine: any input size, no per-size compilation, autotuning or CUDA graph.

Default architecture (3 GRU levels at 1/4, n_downsample 2), batch 1, fp16 NHWC activations, fp32 disparity state.

- Convolutions: cuDNN fp16-accumulate convs on channel slices of packed NHWC buffers (`cudnn_hacc.conv_into`),
  or CUTLASS implicit-GEMM convs with the bias / BatchNorm / activation / GRU math fused into the epilogue
  (`cutlass_conv`) where that is faster (fixed size rules, measured on the RTX 5060 Ti).
- Everything else (InstanceNorm, GRU gates and update, correlation GEMM, pyramid, lookup, resampling, flow head
  output, convex upsampling) is a custom kernel (`raft`).
- Buffers are ordinary PyTorch tensors allocated per call: the encoder activations are freed before the correlation
  volume and the loop buffers are allocated, so the peak is max(encoder, loop), not the sum.
- GRU inputs per level: A = [h | x] for the z|r conv and B = [r*h | x] for the q conv; x is written into both by its
  producers (resampling kernels, motion epilogue), r*h by the gate kernel; the state update runs in place on A's h
  and also writes a packed copy of the 1/4 h for the flow and mask heads.
- The motion branch (needs only the disparity) runs on a side stream, concurrently with the coarse GRUs.

Ported from our RAFT-Stereo fork (`cuda-kernels` branch, `core/general.py`, fused GRU path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.kernels import linear
from fastforks.kernels.cutlass_conv import ConvEpiArgs, narrow_cfg
from fastforks.shared.cuda_graphs import side_stream

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.kernels.cudnn_hacc import CudnnHalfConv
    from fastforks.kernels.cutlass_conv import FusedConvs

CX = (256, 256, 128)  # x-part channels of the GRU inputs at 1/4, 1/8, 1/16
IN_CHUNK = 1024  # pixels per InstanceNorm partial-statistics block
_PAD = 32
_FUSED_MIN_K = 128
_BIG, _MID = 10000, 4000  # pixel-count thresholds of the tile-config rules


def fused_cfg_zr(p: int) -> int | None:
    """CUTLASS tile config of the fused z|r conv, or None = cuDNN conv + `gru_zr` (faster for 4000 <= P < 10000)."""
    return None if _MID <= p < _BIG else 0


def fused_cfg_q(p: int) -> int:
    """CUTLASS tile config of the fused q conv + state update."""
    return 0 if p >= _BIG else 1


def fused_cfg_affine(p: int, k: int, device: torch.device) -> int:
    """CUTLASS tile config of conv + bias/BN + activation; K < 128 outputs use the narrow tiles."""
    if k < _FUSED_MIN_K:
        return narrow_cfg(p, k, device)
    return 0 if p >= _BIG else 1


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


class Weights:
    """Inference weights from an upstream RAFT-Stereo state dict (default architecture): KRSC fp16 conv weights,
    BatchNorm folded into per-channel fp32 scale `s` / shift `t`."""

    def __init__(self, sd: dict[str, torch.Tensor], device: torch.device) -> None:
        sd = {k.removeprefix("module."): v.float() for k, v in sd.items()}
        self.w: dict[str, torch.Tensor] = {}
        self.s: dict[str, torch.Tensor] = {}
        self.t: dict[str, torch.Tensor] = {}
        dev = device

        def conv(name: str, key: str) -> torch.Tensor | None:
            self.w[name] = _krsc(sd[key + ".weight"], dev)
            return sd.get(key + ".bias")

        def bn(name: str, conv_key: str, norm_key: str) -> None:
            # BN(conv(x) + b) = conv(x) * s + ((b - mean) * s + beta), applied in fp32
            b = conv(name, conv_key)
            s = sd[norm_key + ".weight"] / torch.sqrt(sd[norm_key + ".running_var"] + 1e-5)
            mean, beta = sd[norm_key + ".running_mean"], sd[norm_key + ".bias"]
            self.s[name] = s.to(dev).contiguous()
            self.t[name] = (((0 if b is None else b) - mean) * s + beta).to(dev).contiguous()

        def bias(name: str, key: str) -> None:
            b = conv(name, key)
            if b is None:
                raise KeyError(f"checkpoint has no bias {key}")
            self.t[name] = b.to(dev).contiguous()

        def resblock(name: str, *, batch: bool, ds: bool) -> None:
            if batch:
                bn(name + ".c1", name + ".conv1", name + ".norm1")
                bn(name + ".c2", name + ".conv2", name + ".norm2")
                if ds:
                    bn(name + ".ds", name + ".downsample.0", name + ".downsample.1")
            else:  # InstanceNorm: the conv bias cancels
                conv(name + ".c1", name + ".conv1")
                conv(name + ".c2", name + ".conv2")
                if ds:
                    conv(name + ".ds", name + ".downsample.0")

        conv("fnet.conv1", "fnet.conv1")
        bn("cnet.conv1", "cnet.conv1", "cnet.norm1")
        for net, layers in (("fnet", (1, 2, 3)), ("cnet", (1, 2, 3, 4, 5))):
            for li in layers:
                resblock(f"{net}.layer{li}.0", batch=net == "cnet", ds=li > 1)
                resblock(f"{net}.layer{li}.1", batch=net == "cnet", ds=False)
        bias("fnet.conv2", "fnet.conv2")
        for head in ("outputs08", "outputs16"):
            for j in (0, 1):
                bn_name, conv_name = f"cnet.{head}.{j}.rb", f"cnet.{head}.{j}.conv"
                for c in ("1", "2"):
                    bn(f"{bn_name}.c{c}", f"cnet.{head}.{j}.0.conv{c}", f"cnet.{head}.{j}.0.norm{c}")
                bias(conv_name, f"cnet.{head}.{j}.1")
        for j in (0, 1):
            bias(f"cnet.outputs32.{j}.conv", f"cnet.outputs32.{j}")
        for lvl in range(3):
            bias(f"zqr.{lvl}", f"context_zqr_convs.{lvl}")
        u = "update_block"
        self.gru: list[dict[str, torch.Tensor]] = []
        for gname in ("gru08", "gru16", "gru32"):
            k = f"{u}.{gname}"
            self.gru.append(
                {
                    "zr": _krsc(torch.cat([sd[k + ".convz.weight"], sd[k + ".convr.weight"]], 0), dev),
                    "q": _krsc(sd[k + ".convq.weight"], dev),
                    "bz": sd[k + ".convz.bias"].half().to(dev),
                    "br": sd[k + ".convr.bias"].half().to(dev),
                    "bq": sd[k + ".convq.bias"].half().to(dev),
                }
            )
        e = f"{u}.encoder"
        w = sd[e + ".convc1.weight"]  # input channels padded 36 -> 40 (16-byte rows)
        self.w["convc1"] = _krsc(torch.cat([w, w.new_zeros(w.shape[0], 40 - w.shape[1], 1, 1)], 1), dev)
        self.t["convc1"] = sd[e + ".convc1.bias"].to(dev)
        bias("convc2", e + ".convc2")
        bias("convf2", e + ".convf2")
        self.convf1_w = sd[e + ".convf1.weight"][:, 0].contiguous().to(dev)
        self.convf1_b = sd[e + ".convf1.bias"].contiguous().to(dev)
        w, b = sd[e + ".conv.weight"], sd[e + ".conv.bias"]  # outputs padded 126 -> 128: [motion | flow_x, 0]
        self.w["mconv"] = _krsc(torch.cat([w, w.new_zeros(128 - w.shape[0], *w.shape[1:])], 0), dev)
        self.t["mconv"] = torch.cat([b, b.new_zeros(128 - b.shape[0])]).to(dev)
        bias("fh1", f"{u}.flow_head.conv1")
        self.fh2_w = _krsc(sd[f"{u}.flow_head.conv2.weight"][:1], dev)
        self.fh2_b = sd[f"{u}.flow_head.conv2.bias"][:1].half().to(dev)
        bias("mask1", f"{u}.mask.0")
        bias("mask2", f"{u}.mask.2")


class Engine:
    """The kernels forward pass of the default RAFT-Stereo architecture."""

    def __init__(
        self,
        sd: dict[str, torch.Tensor],
        device: torch.device,
        k: CubinModule,
        fused: FusedConvs,
        convs: CudnnHalfConv,
        iters: int,
    ) -> None:
        """Convert the weights; `k`, `fused`, `convs` are the loaded `raft`, `cutlass_conv`, `cudnn_hacc` libraries."""
        self.device = device
        self.W = Weights(sd, device)
        self.iters = iters
        self.k, self.fused, self.convs = k, fused, convs
        self.side = side_stream(device)
        self._unfusable: set[str] = set()  # convs whose geometry the CUTLASS kernel rejected: cuDNN from then on

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

    def affine(
        self,
        y: S,
        name: str,
        mode: int,
        r: S | None = None,
        out: S | None = None,
        out2: S | None = None,
        flow: tuple[torch.Tensor, int] | None = None,
    ) -> S:
        """Per-channel affine (BN scale/shift or bias) + activation of conv output y, in place or into out (+ out2)."""
        o = y if out is None else out
        self.k.launch(
            "affine_act",
            *linear(y.p * y.c // 8),
            y.ptr,
            y.ctot,
            o.ptr,
            o.ctot,
            0 if out2 is None else out2.ptr,
            0 if out2 is None else out2.ctot,
            self.W.s.get(name),
            self.W.t.get(name),
            0 if r is None else r.ptr,
            0 if r is None else r.ctot,
            y.p,
            y.c,
            mode,
            None if flow is None else flow[0],
            0 if flow is None else flow[1],
        )
        return o

    def conv_act(
        self,
        x: S,
        name: str,
        mode: int,
        stride: int = 1,
        out: S | None = None,
        out2: S | None = None,
        r: S | None = None,
        flow: tuple[torch.Tensor, int] | None = None,
        tmp: S | None = None,
    ) -> S:
        """affine(conv(x)): one fused-epilogue conv where the operands allow it, else conv into `tmp` (or a new
        buffer when `out` is None) followed by `affine_act`. mode: 0 none, 1 ReLU, 2 tanh, 3 ReLU(y + r)."""
        w = self.W.w[name]
        kk, _, _, c = w.shape
        ho, wo = (x.h - 1) // stride + 1, (x.w - 1) // stride + 1
        cfg = fused_cfg_affine(ho * wo, kk, x.buf.device)
        if x.full and c % 8 == 0 and name not in self._unfusable:
            o = S(self._new(ho, wo, kk)) if out is None else out
            s, t = self.W.s.get(name), self.W.t.get(name)
            e = ConvEpiArgs(
                2,
                mode,
                0 if r is None else r.ptr,
                0 if r is None else r.ctot,
                o.ptr,
                o.ctot,
                None,
                0 if out2 is None else out2.ctot,
                None if out2 is None else out2.ptr,
                None,
                None if s is None else s.data_ptr(),
                None if t is None else t.data_ptr(),
                None if flow is None else flow[0].data_ptr(),
                0 if flow is None else flow[1],
            )
            if self.fused.conv(cfg, x.buf, w, (x.h, x.w), e, stride=stride):
                return o
            self._unfusable.add(name)  # geometry not supported by the kernel
        y = self.conv(x, name, stride, out=tmp if out is not None else None)
        return self.affine(y, name, mode, r=r, out=out, out2=out2, flow=flow)

    def in_stats(self, y: S) -> torch.Tensor:
        if not y.full:
            raise ValueError("instance-norm statistics need a full (unpadded) tensor")
        p, c = y.p, y.c
        nblk = (p + IN_CHUNK - 1) // IN_CHUNK
        part = torch.empty(nblk * 2 * c, device=self.device, dtype=torch.float32)
        st = torch.empty(2 * c, device=self.device, dtype=torch.float32)
        self.k.launch(
            "in_stats_partial", (nblk, 1, 1), ((256 // (c // 8)) * (c // 8), 1, 1), y.ptr, p, c, IN_CHUNK, part
        )
        self.k.launch("in_stats_final", (c, 1, 1), (32, 1, 1), part, nblk, c, p, IN_CHUNK, 1e-5, st)
        return st

    def in_apply(
        self, y: S, st: torch.Tensor, mode: int = 0, r: S | None = None, st_r: torch.Tensor | None = None
    ) -> None:
        """y = IN(y), then mode 0: ReLU(y), 1: ReLU(y + IN(r)), 2: ReLU(y + r) (in place)."""
        n = y.p * y.c
        self.k.launch("in_apply", *linear(n // 8), y.ptr, st, 0 if r is None else r.ptr, st_r, y.ptr, n, y.c, mode)

    # ---- encoders
    def res_block(self, x: S, name: str, stride: int, *, ds: bool, batch: bool) -> S:
        if batch:
            r = self.conv_act(x, name + ".ds", 0, stride) if ds else x
            y = self.conv_act(x, name + ".c1", 1, stride)
            return self.conv_act(y, name + ".c2", 3, r=r)
        d = st_d = None
        if ds:
            d = self.conv(x, name + ".ds", stride)
            st_d = self.in_stats(d)
        y = self.conv(x, name + ".c1", stride)
        self.in_apply(y, self.in_stats(y), 0)
        o = self.conv(y, name + ".c2")
        del y
        st = self.in_stats(o)
        if ds:
            self.in_apply(o, st, 1, d, st_d)
        else:
            self.in_apply(o, st, 2, x)
        return o

    def prep_image(self, img: torch.Tensor, hp: int, wp: int, top: int, left: int) -> S:
        """float32 [3, H, W] in [0, 1] -> replicate-padded fp16 NHWC in [-1, 1]."""
        _, h, w = img.shape
        out = self._new(hp, wp, 3)
        self.k.launch("prep_image_chw", *linear(hp * wp), img, out, h, w, hp, wp, top, left)
        return S(out)

    def fnet(self, img: S, out: S) -> None:
        x = self.conv(img, "fnet.conv1")
        self.in_apply(x, self.in_stats(x), 0)
        for li, stride in ((1, 1), (2, 2), (3, 2)):
            x = self.res_block(x, f"fnet.layer{li}.0", stride, ds=stride > 1, batch=False)
            x = self.res_block(x, f"fnet.layer{li}.1", 1, ds=False, batch=False)
        self.conv_act(x, "fnet.conv2", 0, out=out, tmp=out)

    def context_head(self, x: S, lvl: int, head: str, L: dict[str, Any]) -> None:
        # the level's long-lived buffers are allocated here, after the full-resolution encoder stage is freed
        h, w = x.h, x.w
        L["A"][lvl], L["czr"][lvl], L["cq"][lvl] = (
            self._new(h, w, 128 + CX[lvl]),
            self._new(h, w, 256),
            self._new(h, w, 128),
        )
        if lvl == 0:
            L["h0"] = self._new(h, w, 128)
        h_sl = S(L["A"][lvl], 0, 128)
        h0 = S(L["h0"]) if lvl == 0 else None
        if head == "outputs32":
            self.conv_act(x, f"cnet.{head}.0.conv", 2, out=h_sl, out2=h0)
            inp = self.conv_act(x, f"cnet.{head}.1.conv", 1)
        else:
            o = self.res_block(x, f"cnet.{head}.0.rb", 1, ds=False, batch=True)
            self.conv_act(o, f"cnet.{head}.0.conv", 2, out=h_sl, out2=h0)
            o = self.res_block(x, f"cnet.{head}.1.rb", 1, ds=False, batch=True)
            inp = self.conv_act(o, f"cnet.{head}.1.conv", 1)
            del o
        z = self.conv_act(inp, f"zqr.{lvl}", 0)
        g = self.W.gru[lvl]
        self.k.launch(
            "prep_context", *linear(z.p * 128), z.buf, g["bz"], g["br"], g["bq"], L["czr"][lvl], L["cq"][lvl], z.p
        )

    def cnet(self, img: S, L: dict[str, Any]) -> None:
        x = self.conv_act(img, "cnet.conv1", 1)
        for li, stride in ((1, 1), (2, 2), (3, 2)):
            x = self.res_block(x, f"cnet.layer{li}.0", stride, ds=stride > 1, batch=True)
            x = self.res_block(x, f"cnet.layer{li}.1", 1, ds=False, batch=True)
        self.context_head(x, 0, "outputs08", L)
        for lvl, li, head in ((1, 4, "outputs16"), (2, 5, "outputs32")):
            x = self.res_block(x, f"cnet.layer{li}.0", 2, ds=True, batch=True)
            x = self.res_block(x, f"cnet.layer{li}.1", 1, ds=False, batch=True)
            self.context_head(x, lvl, head, L)

    # ---- update loop
    def resample(self, kind: str, L: dict[str, Any], src: int, dst: int, slot: int) -> None:
        """pool2x / bilinear of level src's h into x-slot `slot` of level dst (both its [h | x] and [rh | x])."""
        sb, da, db = L["A"][src], L["A"][dst], L["B"][dst]
        hs, ws, cs = sb.shape
        hd, wd, wide = da.shape
        off = 2 * (128 + slot)
        self.k.launch(
            kind, *linear(hd * wd * 16), sb, cs, hs, ws, da.data_ptr() + off, db.data_ptr() + off, wide, hd, wd, 128
        )

    def _gru_calls(self, L: dict[str, Any], lvl: int) -> tuple[Callable[[], None] | None, Callable[[], None]]:
        """Fused launchers of one level (the launch arguments are the same every iteration: built once per pass)."""
        A, B, g = L["A"][lvl], L["B"][lvl], self.W.gru[lvl]
        h, w, wide = A.shape
        zr = L["zr"][lvl]
        cfg = fused_cfg_zr(h * w)
        zr_call = (  # z -> zr[:, :128], r*h -> B[:, :128]
            None
            if cfg is None
            else self.fused.bind(
                cfg,
                A,
                g["zr"],
                (h, w),
                ConvEpiArgs(0, 0, L["czr"][lvl].data_ptr(), 256, zr.data_ptr(), 256, A.data_ptr(), wide, B.data_ptr()),
            )
        )
        h0 = L["h0"].data_ptr() if lvl == 0 else None
        q_call = self.fused.bind(  # h updated in place in A (not read by the q conv), + packed copy h0 at 1/4
            fused_cfg_q(h * w),
            B,
            g["q"],
            (h, w),
            ConvEpiArgs(1, 0, L["cq"][lvl].data_ptr(), 128, zr.data_ptr(), 256, A.data_ptr(), wide, None, h0),
        )
        return zr_call, q_call

    def gru(self, L: dict[str, Any], lvl: int) -> None:
        zr_call, q_call = L["epi"][lvl]
        if zr_call is None:
            A, zr = L["A"][lvl], L["zr"][lvl]
            self.conv_raw(S(A), self.W.gru[lvl]["zr"], S(zr))
            p, wide = A.shape[0] * A.shape[1], A.shape[2]
            self.k.launch("gru_zr", *linear(p * 16), zr, L["czr"][lvl], A, L["B"][lvl], wide, p)
        else:
            zr_call()
        q_call()

    def motion(self, L: dict[str, Any]) -> None:
        k, W = self.k, self.W
        h2, w2, p = L["h2"], L["w2"], L["p2"]
        A0, B0, t64, t128 = L["A"][0], L["B"][0], S(L["t64"]), S(L["t128"])
        k.launch("corr_lookup", *linear(p), *L["corr"], *L["corr_w"], L["coords"], L["look"], p)
        self.conv_act(S(L["look"]), "convc1", 1, out=S(L["mc1"]), tmp=S(L["mc1"]))
        self.conv_act(S(L["mc1"]), "convc2", 1, out=S(L["mot_in"], 0, 64), tmp=t64)
        grid = ((w2 + 15) // 16, (h2 + 15) // 16, 1)
        k.launch("convf1_direct", grid, (16, 16, 1), L["coords"], h2, w2, W.convf1_w, W.convf1_b, L["f1"])
        self.conv_act(S(L["f1"]), "convf2", 1, out=S(L["mot_in"], 64, 64), tmp=t64)
        # motion features [relu(conv) (126) | flow_x, 0] into the x-slot of both level-0 GRU buffers
        self.conv_act(
            S(L["mot_in"]), "mconv", 1, out=S(A0, 128, 128), out2=S(B0, 128, 128), flow=(L["coords"], w2), tmp=t128
        )

    def flow_head(self, L: dict[str, Any]) -> None:
        self.conv(S(L["h0"]), "fh1", out=S(L["fh"]))  # bias + ReLU fused into the output kernel
        grid = ((L["p2"] * 32 + 255) // 256, 1, 1)
        self.k.launch(
            "flow_head_out_br", grid, (256, 1, 1), L["fh"], self.W.t["fh1"], L["h2"], L["w2"], self.W.fh2_w,
            self.W.fh2_b, L["coords"],
        )  # fmt: skip

    def iteration(self, L: dict[str, Any]) -> None:
        main = torch.cuda.current_stream(self.device)
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            self.motion(L)
        self.resample("pool2x_dual", L, 1, 2, 0)
        self.gru(L, 2)
        self.resample("pool2x_dual", L, 0, 1, 0)
        self.resample("interp_dual", L, 2, 1, 128)
        self.gru(L, 1)
        main.wait_stream(self.side)
        self.resample("interp_dual", L, 1, 0, 128)
        self.gru(L, 0)
        self.flow_head(L)

    # ---- full pass
    @torch.inference_mode()
    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """left, right: float32 `[1, 3, H, W]` in [0, 1] on the device. Returns float32 `[H, W]` disparity (px)."""
        with torch.cuda.device(self.device):
            return self._run(left[0].contiguous(), right[0].contiguous())

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        k = self.k
        _, h, w = left.shape
        pad_h, pad_w = (-h) % _PAD, (-w) % _PAD
        top, lft = pad_h // 2, pad_w // 2  # upstream InputPadder ('sintel' mode, divis_by=32)
        hp, wp = h + pad_h, w + pad_w
        h2, w2 = hp // 4, wp // 4
        sizes = [(h2, w2), (h2 // 2, w2 // 2), (h2 // 4, w2 // 4)]
        p = h2 * w2

        # feature encoder, one image at a time
        fmaps = [self._new(h2, w2, 256) for _ in range(2)]
        for img, fm in zip((left, right), fmaps, strict=True):
            self.fnet(self.prep_image(img, hp, wp, top, lft), S(fm))
        L: dict[str, Any] = {"h2": h2, "w2": w2, "p2": p, "A": [None] * 3, "czr": [None] * 3, "cq": [None] * 3}
        self.cnet(self.prep_image(left, hp, wp, top, lft), L)

        # correlation pyramid
        L["corr_w"] = [w2 >> level for level in range(4)]
        L["corr"] = [torch.empty(p * cw, device=self.device, dtype=torch.half) for cw in L["corr_w"]]
        t = (w2 + 127) // 128
        k.launch("corr_gemm", (t, t, h2), (256, 1, 1), fmaps[0], fmaps[1], L["corr"][0], w2, 1.0 / 16.0)
        del fmaps
        for level in range(3):
            cw0, cw1 = L["corr_w"][level], L["corr_w"][level + 1]
            k.launch("corr_pool", *linear(p * cw1), L["corr"][level], L["corr"][level + 1], p, cw0, cw1)
        L["coords"] = torch.empty(p, device=self.device, dtype=torch.float32)
        k.launch("init_coords", *linear(p), L["coords"], w2, p)

        # loop buffers
        L["B"] = [self._new(hh, ww, 128 + cx) for (hh, ww), cx in zip(sizes, CX, strict=True)]  # [r*h | x]
        L["zr"] = [self._new(hh, ww, 256) for hh, ww in sizes]
        L["q"] = [self._new(hh, ww, 128) for hh, ww in sizes]
        for name, c in (("look", 40), ("mc1", 64), ("mot_in", 128), ("f1", 64), ("t64", 64)):
            L[name] = self._new(h2, w2, c)
        L["t128"] = L["q"][0]  # motion conv output: consumed before gru(0) writes q[0]
        L["fh"] = L["zr"][0]  # flow head hidden: zr[0] is dead after gru(0)
        L["epi"] = [self._gru_calls(L, lvl) for lvl in range(3)]
        for _ in range(self.iters):
            self.iteration(L)
        fh, h0, coords = L["fh"], L["h0"], L["coords"]
        del L  # free the loop buffers before the mask head

        # mask head + convex upsampling (unpadded output)
        m1 = self.conv_act(S(h0), "mask1", 1, out=S(fh), tmp=S(fh))
        m2 = self.conv_act(m1, "mask2", 0)
        out = torch.empty(h, w, device=self.device, dtype=torch.float32)
        k.launch("upsample_out", *linear(h * w), coords, m2.buf, h2, w2, top, lft, h, w, out, 0)
        return out.neg_()  # upstream's flow_x = -disparity
