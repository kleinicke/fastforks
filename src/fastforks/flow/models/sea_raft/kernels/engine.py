"""SEA-RAFT kernels engine (S and M): batch 1, fp16 NHWC activations, fp32 flow state.

- No correlation volume: SEA-RAFT's pyramid halves fmap2 bilinearly (align_corners=False, scale 0.5), which is 2 x 2
  average pooling, and pooling commutes with the dot product, so only the pooled fmap2 pyramid is kept (fp16, O(HW)
  memory) and `corr_lookup_c256_r4` takes the dot products of each pixel's window on the fly (fp32 accumulation).
- Encoders: ResNet stages with BatchNorm folded into the convolutions; CUTLASS implicit-GEMM convs with bias,
  activation and the residual ReLU fused into the epilogue (`cutlass_conv`), cuDNN fp16-accumulate convs
  (`cudnn_hacc`) where CUTLASS does not take the operands.
- State buffer A = [hidden | context | motion | flow | MLP hidden]: the ConvNeXt blocks read their input
  [hidden | context | motion | flow] from A's first 384 channels; `dwconv7_ln_384` does the depthwise 7 x 7 conv and
  the LayerNorm, the first MLP layer writes GELU(.) into A's last 512 channels (CUTLASS epilogue), and the rest of the
  block (second MLP layer, layer scale, residual, 1 x 1 output conv: all linear) is folded into one 1 x 1 conv over
  all of A.
- The motion encoder, the flow head output + coords update and the convex upsampling are the RAFT kernels of
  `raft_flow` (SEA-RAFT keeps torchvision RAFT-large's motion encoder and 0.25-scaled upsampling mask).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.kernels import linear
from fastforks.kernels.cutlass_conv import ConvEpiArgs, narrow_cfg

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule
    from fastforks.kernels.cudnn_hacc import CudnnHalfConv
    from fastforks.kernels.cutlass_conv import FusedConvs

    from ..net import Config

_BIG = 10000  # fixed: pixel count from which the 128 x 128 CUTLASS tiles fill the GPU
DW_TX, DW_ROWS = 8, 4  # fixed: dwconv7_ln's pixels per thread and rows per block (raft_flow_kernels.cu)
BN_EPS, LN_EPS = 1e-5, 1e-6  # fixed: upstream nn.BatchNorm2d default, ConvNextBlock's LayerNorm


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
    """Inference weights from a SEA-RAFT state dict: KRSC fp16 conv weights, BatchNorm folded into per-channel fp32
    scale `s` / shift `t`, conv biases as `t`; the ConvNeXt blocks' tail folded into one 1 x 1 conv."""

    def __init__(self, sd: dict[str, torch.Tensor], cfg: Config, device: torch.device) -> None:
        sd = {k: v.double() for k, v in sd.items()}  # folds in float64, stored as fp16 / fp32
        self.w: dict[str, torch.Tensor] = {}
        self.s: dict[str, torch.Tensor] = {}
        self.t: dict[str, torch.Tensor] = {}
        dev = device

        def put(name: str, w: torch.Tensor, b: torch.Tensor, s: torch.Tensor | None = None) -> None:
            self.w[name] = _krsc(w, dev)
            self.t[name] = b.float().to(dev).contiguous()
            if s is not None:
                self.s[name] = s.float().to(dev).contiguous()

        def conv_bn(name: str, conv: str, bn: str, cin: int | None = None) -> None:
            """BN(conv(x) + b) = conv(x) * s + ((b - mean) * s + beta)."""
            w, b = sd[conv + ".weight"], sd[conv + ".bias"]
            s = sd[bn + ".weight"] / torch.sqrt(sd[bn + ".running_var"] + BN_EPS)
            put(name, w if cin is None else _pad_in(w, cin), (b - sd[bn + ".running_mean"]) * s + sd[bn + ".bias"], s)

        def conv(name: str, key: str, cin: int | None = None) -> None:
            w = sd[key + ".weight"]
            put(name, w if cin is None else _pad_in(w, cin), sd[key + ".bias"])

        for enc in ("cnet", "fnet"):
            conv_bn(f"{enc}.stem", f"{enc}.conv1", f"{enc}.bn1", cin=8)  # 6 / 3 input channels, zero-padded to 8
            for li, n in enumerate(cfg.blocks, 1):
                for j in range(n):
                    k = f"{enc}.layer{li}.{j}"
                    conv_bn(f"{k}.c1", f"{k}.conv1", f"{k}.bn1")
                    conv_bn(f"{k}.c2", f"{k}.conv2", f"{k}.bn2")
                    if f"{k}.bn3.weight" in sd:
                        conv_bn(f"{k}.ds", f"{k}.downsample.0", f"{k}.bn3")
            conv(f"{enc}.out", f"{enc}.final_conv")
        conv("init", "init_conv")

        m = "update_block.encoder"
        self.look = _up8(sd[f"{m}.convc1.weight"].shape[1])  # 4 levels x 81 lookups, padded to 328
        conv("convc1", f"{m}.convc1", cin=self.look)
        conv("convc2", f"{m}.convc2")
        self.convf1_w = sd[f"{m}.convf1.weight"].float().contiguous().to(dev)  # fp32 [K, 2, 7, 7]
        self.convf1_b = sd[f"{m}.convf1.bias"].float().contiguous().to(dev)
        conv("convf2", f"{m}.convf2")
        w, b = sd[f"{m}.conv.weight"], sd[f"{m}.conv.bias"]
        self.motion = w.shape[0]  # 126 features, then the 2 flow channels
        put("mconv", *_pad_out(w, b, _up8(w.shape[0])))

        self.dw: list[dict[str, torch.Tensor]] = []
        for i in range(cfg.num_blocks):
            r = f"update_block.refine.{i}"
            dw = sd[f"{r}.dwconv.weight"]  # [C, 1, 7, 7]
            self.dw.append(
                {
                    "w": dw[:, 0].reshape(dw.shape[0], 49).T.float().contiguous().to(dev),  # [49][C]
                    "b": sd[f"{r}.dwconv.bias"].float().to(dev),
                    "g": sd[f"{r}.norm.weight"].float().to(dev),
                    "be": sd[f"{r}.norm.bias"].float().to(dev),
                }
            )
            w1 = sd[f"{r}.pwconv1.weight"]  # [4 dim, C]
            put(f"mlp{i}", w1[:, :, None, None], sd[f"{r}.pwconv1.bias"])
            # final(x + gamma * (W2 g + b2)) = Wf x + (Wf diag(gamma) W2) g + Wf (gamma b2) + bf, as one conv on [x | g]
            wf, bf = sd[f"{r}.final.weight"][:, :, 0, 0], sd[f"{r}.final.bias"]
            gam, w2, b2 = sd[f"{r}.gamma"], sd[f"{r}.pwconv2.weight"], sd[f"{r}.pwconv2.bias"]
            put(f"tail{i}", torch.cat([wf, wf @ (gam[:, None] * w2)], 1)[:, :, None, None], wf @ (gam * b2) + bf)

        conv("fh1", "flow_head.0")
        self.fh2_w = _krsc(sd["flow_head.2.weight"][:2], dev)  # [2, 3, 3, 2 dim]: the flow (not the uncertainty)
        self.fh2_b = sd["flow_head.2.bias"][:2].half().to(dev)
        conv("up1", "upsample_weight.0")
        conv("up2", "upsample_weight.2")


class Engine:
    """The kernels forward pass of SEA-RAFT (`net.SEARAFT`)."""

    def __init__(
        self,
        sd: dict[str, torch.Tensor],
        cfg: Config,
        device: torch.device,
        k: CubinModule,
        fused: FusedConvs,
        convs: CudnnHalfConv,
        iters: int,
    ) -> None:
        """Convert the weights; `k`, `fused`, `convs` are the loaded `raft_flow`, `cutlass_conv`, `cudnn_hacc`."""
        if (cfg.dim, cfg.radius, cfg.levels) != (128, 4, 4):  # the shapes the kernels are built for
            raise ValueError(f"SEA-RAFT kernels need dim 128, radius 4, 4 levels; got {cfg}")
        self.device, self.cfg, self.iters = device, cfg, iters
        self.W = Weights(sd, cfg, device)
        self.k, self.fused, self.convs = k, fused, convs
        self.fdim = 2 * cfg.dim  # feature channels
        self.xc = 3 * cfg.dim  # ConvNeXt block input: [hidden | context | motion | flow]
        self.wide = self.xc + self.W.w["mlp0"].shape[0]  # + the MLP hidden layer
        self._unfusable: set[tuple[str, int, int, int]] = set()  # (conv, H, W, stride) CUTLASS turned down

    def _new(self, h: int, w: int, c: int) -> torch.Tensor:
        return torch.empty(h, w, c, device=self.device, dtype=torch.half)

    # ---- convolutions
    def conv(self, x: S, name: str, stride: int = 1) -> S:
        w = self.W.w[name]
        kk, r, _, c = w.shape
        if c != x.c:  # the kernel trusts these: a mismatch would read out of bounds
            raise ValueError(f"conv weight {tuple(w.shape)} does not fit {x.c} input channels")
        out = S(self._new((x.h - 1) // stride + 1, (x.w - 1) // stride + 1, kk))
        self.convs.conv_into(x.t4(), w.permute(0, 3, 1, 2), out.t4(), r // 2, stride)
        return out

    def affine(self, y: S, s: torch.Tensor | None, t: torch.Tensor | None, mode: int, *, out: S,
               r: S | None = None) -> None:  # fmt: skip
        """out = act(y * s + t); mode 0 none, 1 ReLU, 3 ReLU(ReLU(.) + r)."""
        if mode not in {0, 1, 3}:
            raise ValueError(f"affine_act has no activation {mode}")
        self.k.launch(
            "affine_act", *linear(y.p * y.c // 8), y.ptr, y.ctot, out.ptr, out.ctot, 0, 0, s, t,
            0 if r is None else r.ptr, 0 if r is None else r.ctot, y.p, y.c, mode,
        )  # fmt: skip

    def conv_act(self, x: S, name: str, mode: int, stride: int = 1, out: S | None = None, r: S | None = None) -> S:
        """act(conv(x) * s + t): one fused-epilogue CUTLASS conv where the operands allow it, else cuDNN + affine_act.

        mode: 0 none, 1 ReLU, 3 ReLU(ReLU(.) + r), 4 GELU (CUTLASS only)."""
        w = self.W.w[name]
        kk, _, _, c = w.shape
        ho, wo = (x.h - 1) // stride + 1, (x.w - 1) // stride + 1
        o = S(self._new(ho, wo, kk)) if out is None else out
        if o.c != kk or (o.h, o.w) != (ho, wo):
            raise ValueError(f"{name}: output slice {o.h}x{o.w}x{o.c} does not fit {ho}x{wo}x{kk}")
        s, t = self.W.s.get(name), self.W.t.get(name)
        key = (name, x.h, x.w, stride)  # the refusal depends on the geometry, not only on the layer
        if x.full and c == x.c and c % 8 == 0 and key not in self._unfusable:
            p = ho * wo
            cfg = narrow_cfg(p, kk, x.buf.device) if kk < 128 else (0 if p >= _BIG else 1)
            e = ConvEpiArgs(
                2, mode, 0 if r is None else r.ptr, 0 if r is None else r.ctot, o.ptr, o.ctot, None, 0, None, None,
                None if s is None else s.data_ptr(), None if t is None else t.data_ptr(), None, 0,
            )  # fmt: skip
            if self.fused.conv(cfg, x.buf, w, (x.h, x.w), e, stride=stride):
                return o
            self._unfusable.add(key)
        if mode == 4:  # GELU exists only in the CUTLASS epilogue
            raise RuntimeError(f"{name}: CUTLASS does not take this conv and affine_act has no GELU")
        y = self.conv(x, name, stride)
        self.affine(y, s, t, mode, out=o, r=r)
        return o

    # ---- encoders
    def encoder(self, x: S, enc: str) -> S:
        """[H, W, 8] fp16 input -> [H / 8, W / 8, 256] (ResNet stages and the 1 x 1 output conv)."""
        x = self.conv_act(x, f"{enc}.stem", 1, 2)
        for li, n in enumerate(self.cfg.blocks, 1):
            for j in range(n):
                name = f"{enc}.layer{li}.{j}"
                stride = 2 if li > 1 and j == 0 else 1
                r = self.conv_act(x, name + ".ds", 0, stride) if name + ".ds" in self.W.w else x
                y = self.conv_act(x, name + ".c1", 1, stride)
                x = self.conv_act(y, name + ".c2", 3, r=r)
        return self.conv_act(x, f"{enc}.out", 0)

    def _input(self, *images: torch.Tensor) -> S:
        """float32 `[3, H, W]` images -> fp16 NHWC with their channels side by side, zero-padded to 8."""
        _, h, w = images[0].shape
        x = torch.zeros(h, w, 8, device=self.device, dtype=torch.half)
        for i, img in enumerate(images):
            x[:, :, 3 * i : 3 * i + 3] = img.permute(1, 2, 0)
        return S(x)

    # ---- update loop
    def flow_head(self, L: dict[str, Any]) -> None:
        """coords += flow_head(hidden)[:2] (fp16-rounded delta)."""
        fh = self.conv_act(S(L["h"]), "fh1", 1, out=S(L["fh"]))
        self.k.launch("flow_head_out_256", ((L["p8"] * 32 + 255) // 256, 1, 1), (256, 1, 1), fh.buf, L["h8"],
                      L["w8"], self.W.fh2_w, self.W.fh2_b, L["coords"])  # fmt: skip

    def motion(self, L: dict[str, Any]) -> None:
        k, W = self.k, self.W
        h8, w8, p = L["h8"], L["w8"], L["p8"]
        A, M = L["A"], L["M"]
        k.launch("corr_lookup_c256_r4", ((p + 7) // 8, 1, 1), (256, 1, 1), L["f1"], *L["pyr"], h8, w8, L["coords"],
                 L["look"], W.look, 1.0 / math.sqrt(self.fdim))  # fmt: skip
        c1 = self.conv_act(S(L["look"]), "convc1", 1, out=S(L["c1"]))
        nc = W.w["convc2"].shape[0]
        self.conv_act(c1, "convc2", 1, out=S(M, 0, nc))
        kf = W.convf1_w.shape[0]
        grid = ((w8 + 15) // 16, (h8 + 15) // 16, kf // 64)
        k.launch("convflow1", grid, (16, 16, 1), L["coords"], h8, w8, W.convf1_w, W.convf1_b, L["fl1"], kf)
        self.conv_act(S(L["fl1"]), "convf2", 1, out=S(M, nc, M.shape[2] - nc))
        cx = 2 * self.cfg.dim  # motion features start after hidden and context
        self.conv_act(S(M), "mconv", 1, out=S(A, cx, W.w["mconv"].shape[0]))
        k.launch("write_flow", *linear(p), L["coords"], w8, p, A.data_ptr() + 2 * (cx + W.motion), 0, A.shape[2], 2)

    def refine(self, L: dict[str, Any], i: int) -> None:
        """ConvNeXt block i: hidden <- block([hidden | context | motion | flow])."""
        A, h8, w8 = L["A"], L["h8"], L["w8"]
        d = self.W.dw[i]
        segs = h8 * ((w8 + DW_TX - 1) // DW_TX)
        self.k.launch("dwconv7_ln_384", ((segs + DW_ROWS - 1) // DW_ROWS, 1, 1), (self.xc // 8, DW_ROWS, 1),
                      A, A.shape[2], h8, w8, d["w"], d["b"], d["g"], d["be"], LN_EPS, L["ln"])  # fmt: skip
        self.conv_act(S(L["ln"]), f"mlp{i}", 4, out=S(A, self.xc, self.wide - self.xc))
        self.conv_act(S(A), f"tail{i}", 0, out=S(L["h"]))
        A[:, :, : self.cfg.dim].copy_(L["h"])

    # ---- full pass
    @torch.inference_mode()
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """image1, image2: float32 `[B, 3, H, W]` in [-1, 1], H and W multiples of 8. Returns float32 `[B, 2, H, W]`.

        One pair at a time: the pass is GPU-bound from KITTI sizes up, so batching would only add memory.
        """
        with torch.cuda.device(self.device):
            return torch.stack([self._run(a.contiguous(), b.contiguous()) for a, b in zip(image1, image2, strict=True)])

    def _check(self, image1: torch.Tensor, image2: torch.Tensor) -> tuple[int, int]:
        """(H, W) of a pair the kernels can take; they trust the shapes, so anything else is refused here."""
        if image1.shape != image2.shape or image1.dim() != 3 or image1.shape[0] != 3:
            raise ValueError(f"need two [3, H, W] images, got {tuple(image1.shape)} and {tuple(image2.shape)}")
        if image1.dtype != torch.float32 or image2.dtype != torch.float32:
            raise ValueError(f"need float32 images, got {image1.dtype} and {image2.dtype}")
        _, hh, ww = image1.shape
        cells = 1 << (self.cfg.levels - 1)  # 1/8 cells one cell of the coarsest pyramid level covers
        if hh % 8 or ww % 8 or hh < 8 * cells or ww < 8 * cells:
            raise ValueError(f"need H and W as multiples of 8 of at least {8 * cells} px, got {hh}x{ww}")
        return hh, ww

    def _run(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        k, W, dim = self.k, self.W, self.cfg.dim
        hh, ww = self._check(image1, image2)
        h8, w8 = hh // 8, ww // 8
        p = h8 * w8

        # context: hidden state and context features into A, the initial flow from the hidden state
        A = self._new(h8, w8, self.wide)
        self.conv_act(self.encoder(self._input(image1, image2), "cnet"), "init", 0, out=S(A, 0, 2 * dim))
        # feature maps: f1 and the pooled f2 pyramid
        f1 = self.encoder(self._input(image1), "fnet")
        pyr = [self.encoder(self._input(image2), "fnet")]
        for _ in range(self.cfg.levels - 1):
            src = pyr[-1]
            dst = S(self._new(src.h // 2, src.w // 2, self.fdim))
            k.launch("avgpool2", *linear(dst.p * self.fdim // 8), src.buf, src.h, src.w, dst.buf, dst.h, dst.w,
                     self.fdim)  # fmt: skip
            pyr.append(dst)

        coords = torch.empty(p, 2, device=self.device, dtype=torch.float32)
        k.launch("init_coords", *linear(p), coords, w8, p)
        L: dict[str, Any] = {
            "h8": h8, "w8": w8, "p8": p, "A": A, "coords": coords, "f1": f1.buf, "pyr": [s.buf for s in pyr],
            "look": self._new(h8, w8, W.look), "c1": self._new(h8, w8, W.w["convc1"].shape[0]),
            "M": self._new(h8, w8, W.w["mconv"].shape[3]), "fl1": self._new(h8, w8, W.convf1_w.shape[0]),
            "fh": self._new(h8, w8, W.w["fh1"].shape[0]), "ln": self._new(h8, w8, self.xc),
            "h": A[:, :, :dim].contiguous(),
        }  # fmt: skip
        self.flow_head(L)  # iteration 0: the regressed initial flow
        for _ in range(self.iters):
            self.motion(L)
            for i in range(self.cfg.num_blocks):
                self.refine(L, i)
            self.flow_head(L)
        hstate = L["h"]
        del L, A, pyr, f1

        out = torch.empty(2, hh, ww, device=self.device, dtype=torch.float32)
        m2 = self.conv_act(self.conv_act(S(hstate), "up1", 1), "up2", 0)  # the 0.25 scale is in upsample_convex
        k.launch("upsample_convex", *linear(hh * ww), coords, m2.buf, h8, w8, out)
        return out
