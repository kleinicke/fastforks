"""CREStereo kernels path: upstream `test.py` inference (as `portable.py`) on our kernels, batch 1, CUDA.

What runs where (everything else is the portable math, in PyTorch):

- feature encoder: one image at a time (half the activations of a batch of 2), fp16 channels-last; convs as
  fp16-accumulate cuDNN graphs (`cudnn_hacc`); the instance norms (fp32 statistics), their ReLUs and the residual
  adds are fastbb passes; the conv biases in front of an instance norm are dropped (the norm removes them); the
  3-channel image is zero-padded to 8
- correlation: one `crestereo.agcl_corr` kernel per iteration (warp + 4-group 3 x 3 / 1 x 9 correlation, with the
  learned offsets at 1/16 and 1/8), fp16 features, fp32 flow / offsets / sampling positions
- update block: CUTLASS convs with fused epilogues (`cutlass_conv`); the motion encoder writes into the GRU input
  buffers `[h | motion]` / `[r*h | motion]` (`flow_pack` puts the flow tail there); the GRU context convs (input
  channels 128:256, constant within a stage, with the biases) run once per stage; the z|r conv computes the gates
  and r*h in its epilogue, the q conv the state update (in place, plus the copy into the next z|r input)
- the attention layers, offsets, pooling, mask softmax / convex upsampling and the stage hand-overs: fp32 PyTorch

A conv geometry the kernels do not support raises: there is no PyTorch fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks.kernels import crestereo as crestereo_kernels
from fastforks.kernels import fastbb
from fastforks.kernels.cutlass_conv import ConvEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep, conv_fused, conv_raw, cutlass, fused_cfg, prepare_convs

from ..portable import _attention, _resize

if TYPE_CHECKING:
    from ..net.crestereo import CREStereo

RELU = ACT["relu"]
HD = 128  # fixed: hidden / context width
CORR_CH = 40  # fixed: 4 groups x 9 offsets, padded to a multiple of 8
FEAT_CH = 256  # fixed: feature width agcl_corr is written for (4 groups x 64 channels)
WARP_PIXELS = 8  # fixed: agcl_corr runs one warp per pixel, 8 warps per 256-thread block
_IN_PLAIN, _IN_RELU, _IN_RELU_RES = 0, 1, 2  # fastbb in_apply modes


def _conv(w: torch.Tensor, b: torch.Tensor | None, padding: int | tuple[int, int], stride: int = 1) -> nn.Conv2d:
    """An fp32 conv module holding the derived weight `w` (and bias `b`)."""
    k, c, kh, kw = w.shape
    m: Any = nn.Conv2d(c, k, (kh, kw), stride=stride, padding=padding, bias=b is not None)
    m.weight.data = w.detach().float().clone()
    if b is not None:
        m.bias.data = b.detach().float().clone()
    return m


def _pad_in(m: nn.Module, c: int) -> nn.Module:
    """`m` with its input channels zero-padded to `c`."""
    w = m.weight.detach().float()
    w = torch.cat([w, w.new_zeros((w.shape[0], c - w.shape[1], *w.shape[2:]))], 1)
    return _conv(w, m.bias, m.padding, m.stride[0])


def _pad_out(m: nn.Module, k: int) -> nn.Module:
    """`m` with its output channels zero-padded to `k` (zero weights and bias: relu output 0)."""
    w, b = m.weight.detach().float(), m.bias.detach().float()
    w = torch.cat([w, w.new_zeros((k - w.shape[0], *w.shape[1:]))], 0)
    return _conv(w, torch.cat([b, b.new_zeros(k - b.shape[0])]), m.padding)


def _gru_convs(gru: nn.Module) -> dict[str, nn.Module]:
    """Per direction d: `ctx{d}` (context channels of z, r, q with the biases, 384 outputs), `zr{d}` (z|r on
    `[h | motion]`), `q{d}` (q on `[r*h | motion]`)."""
    out = {}
    own = (slice(0, HD), slice(2 * HD, None))  # input channels: hidden + motion; HD:2HD is the context
    for d, (z, r, q) in enumerate(((gru.convz1, gru.convr1, gru.convq1), (gru.convz2, gru.convr2, gru.convq2)), 1):
        w_all = torch.cat([z.weight, r.weight, q.weight]).detach().float()
        b_all = torch.cat([z.bias, r.bias, q.bias]).detach().float()
        out[f"ctx{d}"] = _conv(w_all[:, HD : 2 * HD], b_all, z.padding)
        out[f"zr{d}"] = _conv(torch.cat([w_all[: 2 * HD, s] for s in own], 1), None, z.padding)
        out[f"q{d}"] = _conv(torch.cat([w_all[2 * HD :, s] for s in own], 1), None, z.padding)
    return out


def _check_agcl(fmap1: torch.Tensor, fmap2: torch.Tensor, flow: torch.Tensor, extra: torch.Tensor | None) -> None:
    """Raise unless the `agcl_corr` inputs have the layout the kernel reads (16-byte loads of 256-channel NHWC fp16)."""
    _, _, h, w = flow.shape
    feats_ok = all(
        f.shape == (1, FEAT_CH, h, w)
        and f.dtype == torch.float16
        and f.is_contiguous(memory_format=CL)
        and f.data_ptr() % 16 == 0
        for f in (fmap1, fmap2)
    )
    flow_ok = flow.shape[:2] == (1, 2) and flow.dtype == torch.float32 and flow.is_contiguous()
    extra_ok = extra is None or (
        extra.shape == (1, 18, h, w) and extra.dtype == torch.float32 and extra.is_contiguous()
    )
    if not (feats_ok and flow_ok and extra_ok):
        msg = f"agcl_corr: unexpected input layout (features {tuple(fmap1.shape)}, flow {tuple(flow.shape)})"
        raise RuntimeError(msg)


class Engine:
    """Kernel state of one CREStereo network on one CUDA device."""

    def __init__(self, net: CREStereo, device: torch.device) -> None:
        """Derive and prepare the conv modules (`net` fp32, eval, on `device`).

        Raises:
            KernelUnavailableError: A kernel library is unusable (no fallback).
            RuntimeError: `device` is not a CUDA device.
        """
        if device.type != "cuda":
            msg = "CREStereo kernels runs on CUDA only"
            raise RuntimeError(msg)
        cnn_kernels.load(device)
        self.k = crestereo_kernels.load(device)
        self.fb = fastbb.load(device)
        if net.fnet.conv2.out_channels != FEAT_CH:
            msg = f"CREStereo kernels: agcl_corr needs {FEAT_CH} feature channels, the encoder has {net.fnet.conv2.out_channels}"
            raise RuntimeError(msg)
        self.net = net
        ub = net.update_block
        enc = ub.encoder
        convs: dict[str, nn.Module] = {
            "stem": _pad_in(net.fnet.conv1, 8),
            "convc1": _pad_in(enc.convc1, CORR_CH),
            "convf1": _pad_in(enc.convf1, 8),
            "motion": _pad_out(enc.conv, HD),
            **_gru_convs(ub.gru),
        }
        self.c: Any = nn.ModuleDict(convs).to(device)
        prepare_convs(self.c)
        # used as they are: encoder (cuDNN, batch 2), offsets, motion encoder, flow head, mask head
        prepare_convs(net.fnet, fuse=False)
        for m in (net.conv_offset_8, net.conv_offset_16, enc.convc2, enc.convf2, ub.flow_head, ub.mask):
            prepare_convs(m)

    # ------------------------------------------------------------------------------------------------ encoder
    def _norm(self, x: torch.Tensor, eps: float, mode: int, res: torch.Tensor | None = None) -> torch.Tensor:
        """Instance norm (no affine) of an NHWC fp16 `x` with fastbb: mode 0 norm, 1 relu(norm),
        2 relu(res + relu(norm))."""
        n, c, h, w = x.shape
        if not (x.is_contiguous(memory_format=CL) and c % 8 == 0 and x.dtype == torch.float16):
            msg = f"fastbb instance norm needs NHWC fp16 with C % 8 == 0 (got {tuple(x.shape)})"
            raise RuntimeError(msg)
        hw = h * w
        rows = max(1, 256 // (c // 8))
        nbx = max(1, min((hw + rows - 1) // rows, max(1, 288 // n)))  # blocks per image
        rpb = (hw + nbx - 1) // nbx
        buf = torch.empty(2 * n * c * (nbx + 1), device=x.device, dtype=torch.float32)
        partial, st = buf[: 2 * n * c * nbx], buf[2 * n * c * nbx :]  # per block (mean, M2); result (mean, rstd)
        fb = self.fb
        fb.launch("in_stats_h", (nbx, n, 1), (256, 1, 1), x, partial, hw, c, rpb, smem=fastbb.IN_STATS_SMEM)
        grid = ((n * c + 7) // 8, 1, 1)
        fb.launch("in_stats_final", grid, (256, 1, 1), partial, st, n, nbx, hw, c, rpb, eps, None, None)
        y = torch.empty_like(x)
        fb.launch("in_apply_h", *linear(n * hw * c // 8), x, y, st, None, None, res, n, hw, c, eps, mode)
        return y

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """`BasicEncoder` (instance norm) of an fp16 NHWC `[N, 8, H, W]` batch."""
        f: Any = self.net.fnet
        y = self._norm(conv_raw(self.c["stem"], x), f.norm1.eps, _IN_RELU)
        for layer in (f.layer1, f.layer2, f.layer3):
            for b in layer:
                y1 = self._norm(conv_raw(b.conv1, y), b.norm1.eps, _IN_RELU)
                r = self._norm(conv_raw(b.downsample[0], y), b.downsample[1].eps, _IN_PLAIN)
                y = self._norm(conv_raw(b.conv2, y1), b.norm2.eps, _IN_RELU_RES, r)
        return conv_ep(f.conv2, y)

    # ------------------------------------------------------------------------------------------------ update
    @staticmethod
    def _into(
        m: nn.Module, x: torch.Tensor, out: tuple[torch.Tensor, int], out2: tuple[torch.Tensor, int] | None = None
    ) -> None:
        """relu(conv + bias) written into channel slices of wider buffers (CUTLASS epilogue)."""
        if conv_fused(m, x, None, m._bias32, RELU, out=out, out2=out2) is None:
            msg = f"cutlass_conv: geometry {tuple(x.shape)} * {tuple(m.weight.shape)} not supported"
            raise RuntimeError(msg)

    def _offset(self, conv: nn.Module, fmap: torch.Tensor, rng: float) -> torch.Tensor:
        """`range * (sigmoid(conv_offset(fmap)) - 0.5) * 2`, fp32 NCHW `[1, 18, h, w]`."""
        return (rng * (torch.sigmoid(conv_ep(conv, fmap).float()) - 0.5) * 2.0).contiguous()

    def _stage(
        self,
        hidden: torch.Tensor,
        inp: torch.Tensor,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        extra: torch.Tensor | None,
        flow: torch.Tensor,
        iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One cascade stage (`portable._stage` with the AGCL correlation): final flow and upsampling mask (fp32).

        `hidden` (channels-last) is the GRU state and is updated in place.
        """
        c, ub, k = self.c, self.net.update_block, self.k
        _, _, h, w = hidden.shape
        p = h * w

        def buf(ch: int) -> torch.Tensor:
            return torch.empty((1, ch, h, w), device=hidden.device, dtype=torch.float16, memory_format=CL)

        hx, rhx, z, cf, corr, f8 = buf(2 * HD), buf(2 * HD), buf(HD), buf(2 * HD), buf(CORR_CH), buf(8)
        net = hidden.contiguous(memory_format=CL)  # updated in place: no caller reads `hidden` after the stage
        hx[:, :HD] = net
        ctx = {d: conv_ep(c[f"ctx{d}"], inp) for d in (1, 2)}  # [1, 384]: z | r | q context terms
        cfg = fused_cfg(p)
        corr_grid = linear(p * 32)
        flow = flow.float().contiguous().clone()
        _check_agcl(fmap1, fmap2, flow, extra)
        for itr in range(iters):
            k.launch("agcl_corr", *corr_grid, fmap1, fmap2, flow, extra, corr, 1, h, w, CORR_CH, int(itr % 2 == 1))
            cor = conv_ep(c["convc1"], corr, act=RELU)
            self._into(ub.encoder.convc2, cor, (cf, 0))
            del cor  # free before the next temporaries (peak memory)
            k.launch("flow_pack", *linear(p), flow, f8, None, None, 1, p, 2 * HD)
            flo = conv_ep(c["convf1"], f8, act=RELU)
            self._into(ub.encoder.convf2, flo, (cf, 192))
            del flo
            self._into(c["motion"], cf, (hx, HD), (rhx, HD))
            k.launch("flow_pack", *linear(p), flow, None, hx, rhx, 1, p, 2 * HD)  # motion = [conv out | flow]
            for d in (1, 2):
                zr, q, cp = c[f"zr{d}"], c[f"q{d}"], ctx[d].data_ptr()
                e0 = ConvEpiArgs(0, 0, cp, 3 * HD, z.data_ptr(), HD, hx.data_ptr(), 2 * HD, rhx.data_ptr())
                e1 = ConvEpiArgs(
                    1, 0, cp + 2 * 2 * HD, 3 * HD, z.data_ptr(), HD, net.data_ptr(), HD, None, hx.data_ptr(),
                    None, None, None, 0, 2 * HD,
                )  # fmt: skip
                if not (
                    cutlass(0, hx, zr.weight, (h, w), e0, padding=zr.padding)
                    and cutlass(cfg, rhx, q.weight, (h, w), e1, padding=q.padding)
                ):
                    msg = "cutlass_conv: GRU conv geometry not supported"
                    raise RuntimeError(msg)
            delta = conv_ep(ub.flow_head.conv2, conv_ep(ub.flow_head.conv1, net, act=RELU))
            flow.add_(delta.float())
        del hx, rhx, z, cf, corr, f8, ctx  # the stage buffers: free before the mask head
        mask = conv_ep(ub.mask[2], conv_ep(ub.mask[0], net, act=RELU))
        return flow, mask.to(torch.float32, memory_format=torch.contiguous_format).mul_(0.25)

    # ------------------------------------------------------------------------------------------------ passes
    def _pass(
        self, left: torch.Tensor, right: torch.Tensor, iters: int, flow_init: torch.Tensor | None
    ) -> torch.Tensor:
        """`CREStereo.forward` in test mode on an RGB pair (0-255, float32 `[1, 3, H, W]`): full-res flow (fp32)."""
        net = self.net
        # one image at a time: a batch of 2 holds twice the encoder activations
        fmap1, fmap2 = self._encode(self._image(left)), self._encode(self._image(right))
        hidden = torch.tanh(fmap1[:, :HD]).contiguous(memory_format=CL)
        inp = F.relu(fmap1[:, HD:]).contiguous(memory_format=CL)

        def pool(t: torch.Tensor, s: int) -> torch.Tensor:
            return F.avg_pool2d(t, s, stride=s).contiguous(memory_format=CL)

        if flow_init is None:
            f1_8, f2_8 = pool(fmap1, 2), pool(fmap2, 2)
            off8 = self._offset(net.conv_offset_8, f1_8, net.range_8)
            f1_16, f2_16 = pool(fmap1, 4), pool(fmap2, 4)
            off16 = self._offset(net.conv_offset_16, f1_16, net.range_16)
            att1, att2 = (a.half().contiguous(memory_format=CL) for a in _attention(net, f1_16.float(), f2_16.float()))
            flow = att1.new_zeros((1, 2, *att1.shape[2:]), dtype=torch.float32)
            flow, mask = self._stage(pool(hidden, 4), pool(inp, 4), att1, att2, off16, flow, iters // 2)
            flow = _resize(net.convex_upsample(flow, mask, rate=4), f1_8)
            flow, mask = self._stage(pool(hidden, 2), pool(inp, 2), f1_8, f2_8, off8, flow, iters // 2)
            flow = _resize(net.convex_upsample(flow, mask, rate=4), fmap1)
        else:
            flow = _resize(flow_init, fmap1)
        flow, mask = self._stage(hidden, inp, fmap1, fmap2, None, flow, iters)
        return -net.convex_upsample(flow, mask, rate=4)

    @staticmethod
    def _image(t: torch.Tensor) -> torch.Tensor:
        """BGR (upstream reads with cv2), `2 * x / 255 - 1`, as an fp16 NHWC `[1, 8, H, W]` zero-padded to 8 channels."""
        h, w = t.shape[-2:]
        x = torch.empty((1, 8, h, w), device=t.device, dtype=torch.float16, memory_format=CL).zero_()
        x[:, :3] = t.flip(1).div(255.0).mul_(2).sub_(1.0)
        return x

    @torch.inference_mode()
    def __call__(self, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
        """Disparity `[1, H, W]` of a padded RGB pair `[1, 3, H, W]` in 0-255 (upstream `test.py` inference).

        Raises:
            RuntimeError: Batch size other than 1.
        """
        if left.shape[0] != 1:
            msg = "CREStereo kernels runs batch 1 only"
            raise RuntimeError(msg)
        left, right = left.float(), right.float()
        h, w = left.shape[-2:]
        small = [F.interpolate(t, size=(h // 2, w // 2), mode="bilinear", align_corners=True) for t in (left, right)]
        flow = self._pass(*small, iters, None)
        del small
        return self._pass(left, right, iters, flow)[:, 0]
