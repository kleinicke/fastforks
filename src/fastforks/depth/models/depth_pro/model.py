"""Depth Pro runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `DepthPro.infer` for one image: normalise to [-1, 1], resize to 1536x1536 (bilinear,
`align_corners=False`), run the network, then `f_px = 0.5 * W / tan(fov / 2)` from the predicted field of view (or `fx`
of the given intrinsics; the FOV network is skipped then), inverse depth = network output * W / f_px, resized back to
the input size (bilinear), depth = 1 / clamp(inverse depth, 1e-4, 1e4). Estimated intrinsics have the principal point at
the image centre and square pixels.

The graph is reorganised where exact (same maths, different rounding):

- patch embedding once per pyramid level; the 35 windows are gathered as token rows and merged by a row index
  (`net.tables`), the final ViT LayerNorm only on the merged tokens (it is per token)
- decoder fusion blocks 1-4: the 2x2 / stride 2 deconvolution and the 1x1 `out_conv` as one deconvolution
- head: deconvolution + 3x3 conv at 1536x1536 as one 3x3 conv at 768x768 that produces the four output phases (the
  deconvolution bias enters through a constant channel of the preceding conv), the last 1x1 conv per phase, then the
  pixel shuffle
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.depth._runner import Forward, Runner, scale
from fastforks.shared.cuda_graphs import side_stream

from .net import DEC, HOOKS, SIZE, DepthProNet, tables

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

_MIN_INV, _MAX_INV = 1e-4, 1e4  # fixed: upstream infer's inverse-depth clamp


def _net(ctx: Context) -> DepthProNet:
    """The network on the CPU in the checkpoint's float16."""
    return DepthProNet().load(ctx.state_dict(), f"{ctx.variant.ref} checkpoint").eval()


def compose_deconv_1x1(deconv: Any, conv: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - nn layers
    """`conv1x1(deconv(x))` as one deconvolution: (weight `[Cin, Cout, 2, 2]`, bias), float32."""
    wd = deconv.weight.detach().float()
    wo = conv.weight.detach().float()[:, :, 0, 0]
    return torch.einsum("icab,kc->ikab", wd, wo), conv.bias.detach().float().clone()


def compose_head(head: Any) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - the nn.Sequential head
    """The head at 768x768 in float32: (conv0 w, b, conv1 w, b, conv2 w, b).

    conv0 `[136, 256, 3, 3]` (channel 128 = constant 1, 129-135 = 0), conv1 `[128, 136, 3, 3]` (output channel
    `4 * k + phase`, then ReLU), conv2 `[4, 128]` (one row per phase, then ReLU);
    `pixel_shuffle(relu(conv2(relu(conv1(conv0(x))))), 2)` = upstream's head.
    """
    c0, dc, c2, c4 = head[0], head[1], head[2], head[4]
    cm, k3 = c0.out_channels, c2.out_channels
    ca = cm + 8
    w0 = torch.zeros(ca, *c0.weight.shape[1:])
    w0[:cm] = c0.weight.detach().float()
    b0 = torch.zeros(ca)
    b0[:cm] = c0.bias.detach().float()
    b0[cm] = 1.0
    wd, bd = dc.weight.detach().float(), dc.bias.detach().float()  # [128, 128, 2, 2]
    w3 = c2.weight.detach().float()  # [32, 128, 3, 3]
    w1 = torch.zeros(4 * k3, ca, 3, 3)
    for a in range(2):
        for b in range(2):
            o = torch.arange(k3) * 4 + a * 2 + b
            for dy in range(3):
                py, ay = divmod(a + dy - 1, 2)
                for dx in range(3):
                    px, ax = divmod(b + dx - 1, 2)
                    w1[o, :cm, py + 1, px + 1] += w3[:, :, dy, dx] @ wd[:, :, ay, ax].t()
                    w1[o, cm, py + 1, px + 1] += w3[:, :, dy, dx] @ bd
    b1 = c2.bias.detach().float().repeat_interleave(4)
    w4 = c4.weight.detach().float()[0, :, 0, 0]
    w2 = torch.zeros(4, 4 * k3)
    for p in range(4):
        w2[p, torch.arange(k3) * 4 + p] = w4
    return w0, b0, w1, b1, w2, c4.bias.detach().float().repeat(4)


def _pixel_shuffle(z: torch.Tensor) -> torch.Tensor:
    """`[1, h, w, 4]` phases (`a * 2 + b`) -> `[1, 1, 2h, 2w]` (`F.pixel_shuffle` of the NCHW map)."""
    _, h, w, _ = z.shape
    return z.reshape(1, h, w, 2, 2).permute(0, 1, 3, 2, 4).reshape(1, 1, 2 * h, 2 * w)


def _prepare(ctx: Context, image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """`[1, 3, H, W]` in [0, 1] -> the normalised 1536x1536 network input (upstream normalises, then resizes)."""
    x = scale(image.float(), ctx.io).to(dtype)
    if x.shape[-2:] != (SIZE, SIZE):
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return x


def _finish(cid: torch.Tensor, fov: torch.Tensor | None, k: torch.Tensor | None, h: int, w: int) -> dict:
    """Upstream infer's post-processing; `cid` `[1, 1, 1536, 1536]`, `fov` (degrees) when `k` is None."""
    if k is not None:
        f = k[:, 0, 0].float().view(1, 1, 1, 1)
    else:
        assert fov is not None  # noqa: S101 - computed when no intrinsics are given
        f = 0.5 * w / torch.tan(0.5 * torch.deg2rad(fov.float().view(1, 1, 1, 1)))
    inv = cid.float() * (w / f)
    if inv.shape[-2:] != (h, w):
        inv = F.interpolate(inv, size=(h, w), mode="bilinear", align_corners=False)
    res = {"depth": 1.0 / torch.clamp(inv[:, 0], min=_MIN_INV, max=_MAX_INV)}
    if k is None:
        kk = torch.zeros(1, 3, 3, device=cid.device)
        kk[:, 0, 0] = kk[:, 1, 1] = f.view(1)
        kk[:, 0, 2], kk[:, 1, 2], kk[:, 2, 2] = (w - 1) / 2, (h - 1) / 2, 1.0  # OpenCV pixel centres
        res["intrinsics"] = kk
    return res


class _Fast(nn.Module):
    """The network plus the composed weights and index tables of the reorganised graph (buffers)."""

    tok: torch.Tensor
    m0: torch.Tensor
    m1: torch.Tensor
    m2: torch.Tensor
    h0w: torch.Tensor
    h0b: torch.Tensor
    h1w: torch.Tensor
    h1b: torch.Tensor
    h2w: torch.Tensor
    h2b: torch.Tensor

    def __init__(self, net: DepthProNet) -> None:
        super().__init__()
        self.net = net
        tok, m0, m1, m2 = tables()
        for name, t in (("tok", tok), ("m0", m0), ("m1", m1), ("m2", m2)):
            self.register_buffer(name, t, persistent=False)
        head: Any = net.head
        dt: torch.dtype = head[0].weight.dtype
        for i, fu in enumerate(net.decoder.fusions):
            if fu.deconv is not None:
                wb = compose_deconv_1x1(fu.deconv, fu.out_conv)
                self.register_buffer(f"fw{i}", wb[0].to(dt), persistent=False)
                self.register_buffer(f"fb{i}", wb[1].to(dt), persistent=False)
        for name, t in zip(("h0w", "h0b", "h1w", "h1b", "h2w", "h2b"), compose_head(net.head), strict=True):
            self.register_buffer(name, t.to(dt), persistent=False)


def _ln(norm: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - nn.LayerNorm
    """`norm(x)`; an fp16 `x` under autocast stays fp16 (fp32 statistics inside the op, one fp16 rounding).

    Autocast runs LayerNorm in fp32 and the next fp16 op rounds the result: the same values for two extra cast
    passes per call (149 LayerNorms).
    """
    if x.dtype == torch.float16 and torch.is_autocast_enabled("cuda"):
        with torch.autocast("cuda", enabled=False):
            return norm(x)
    return norm(x)


def _block(blk: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - DINOv2 block
    """A DINOv2 block (eval) with fused `scaled_dot_product_attention` (default scale head_dim**-0.5, as upstream)."""
    a = blk.attn
    b, n, c = x.shape
    qkv = a.qkv(_ln(blk.norm1, x)).reshape(b, n, 3, a.num_heads, c // a.num_heads).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(b, n, c)
    x = x + blk.ls1(a.proj(o))
    return x + blk.ls2(blk.mlp(_ln(blk.norm2, x)))


def _vit(vit: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - DINOv2 module
    """The timm `forward_features` of one 384x384 image -> normalised tokens `[1, 577, C]` (cls first)."""
    t = vit.patch_embed(x)
    t = torch.cat([vit.cls_token.expand(t.shape[0], -1, -1), t], 1) + vit.pos_embed
    for blk in vit.blocks:
        t = _block(blk, t)
    return _ln(vit.norm, t)


def forward(fast: _Fast, x: torch.Tensor, *, with_fov: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalised `[1, 3, 1536, 1536]` -> (canonical inverse depth `[1, 1, 1536, 1536]`, FOV in degrees or None)."""
    net: Any = fast.net
    enc, dec = net.encoder, net.decoder
    pe = enc.patch_encoder
    x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
    proj = pe.patch_embed.proj
    e = torch.cat([proj(v).flatten(2) for v in (x, x1, x2)], 2)[0].t()  # [96^2 + 48^2 + 24^2, C]
    t = e[fast.tok]  # [35, 576, C]
    t = torch.cat([pe.cls_token.expand(t.shape[0], -1, -1), t], 1) + pe.pos_embed
    lat = []
    for i, blk in enumerate(pe.blocks):
        t = _block(blk, t)
        if i in HOOKS:
            lat.append(t.reshape(-1, t.shape[-1])[fast.m0])
    rows = t.reshape(-1, t.shape[-1])

    def nchw(r: torch.Tensor) -> torch.Tensor:  # [h, w, C] -> [1, C, h, w]
        return r.permute(2, 0, 1).unsqueeze(0)

    f0, f1, f2 = (_ln(pe.norm, rows[m]) for m in (fast.m0, fast.m1, fast.m2))
    g = _vit(enc.image_encoder, x2)[0, 1:].reshape(f2.shape)
    lowres = torch.cat((enc.upsample2(nchw(f2)), enc.upsample_lowres(nchw(g))), dim=1)
    encs = [
        enc.upsample_latent0(nchw(lat[0])),
        enc.upsample_latent1(nchw(lat[1])),
        enc.upsample0(nchw(f0)),
        enc.upsample1(nchw(f1)),
        enc.fuse_lowres(lowres),
    ]
    feat = dec.convs[4](encs[4])
    lowres = feat
    for i in range(4, -1, -1):
        fu = dec.fusions[i]
        if i < 4:  # noqa: PLR2004 - the lowest level starts the chain
            feat = feat + fu.resnet1(dec.convs[i](encs[i]))
        feat = fu.resnet2(feat)
        if i:
            feat = F.conv_transpose2d(feat, getattr(fast, f"fw{i}"), getattr(fast, f"fb{i}"), stride=2)
        else:
            feat = fu.out_conv(feat)
    y = F.relu(F.conv2d(F.conv2d(feat, fast.h0w, fast.h0b, padding=1), fast.h1w, fast.h1b, padding=1))
    cid = F.pixel_shuffle(F.relu(F.conv2d(y, fast.h2w[:, :, None, None], fast.h2b)), 2)
    fov = None
    if with_fov:
        fv = net.fov
        z = fv.encoder[1](_vit(fv.encoder[0], x2))[:, 1:].permute(0, 2, 1)
        lr = fv.downsample(lowres)
        fov = fv.head(z.reshape_as(lr) + lr)
    return cid, fov


class Portable(Runner):
    """The reorganised graph in plain PyTorch (fused SDPA attention): float32, or fp16 weights + autocast."""

    dtypes = ("float32", "float16")
    intrinsics = "optional"  # given intrinsics replace the FOV estimate (fx)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.half = ctx.dtype == "float16"
        net = _net(ctx)
        if not self.half:
            net = net.float()
        self.fast = _Fast(net).eval().to(ctx.device)  # a new Module starts in training mode
        if self.half:  # everything after the ViTs on NHWC maps
            for m in (net.encoder, net.decoder, net.fov.downsample, net.fov.head):
                m.to(memory_format=torch.channels_last)

    def _run(self, image: torch.Tensor, k: torch.Tensor | None) -> dict[str, torch.Tensor]:
        h, w = image.shape[-2:]
        x = _prepare(self.ctx, image, torch.float16 if self.half else torch.float32)
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.half):
            cid, fov = forward(self.fast, x, with_fov=k is None)
        return _finish(cid, fov, k, h, w)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Depth (and intrinsics when none are given) of one image."""
        return self._run(image, intrinsics)

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> depth `[1, H, W]`."""
        return Forward(self.fast, lambda _net, image: self._run(image, None)["depth"])


# ------------------------------------------------------------------------------------------------ kernels
class _Up:
    """A projection chain: 1x1 conv (GEMM on token rows), then 2x2 / stride 2 deconvolutions (pixel-shuffle GEMMs)."""

    def __init__(self, seq: Any, device: torch.device) -> None:  # noqa: ANN401 - the nn.Sequential chain
        self.w = seq[0].weight.detach().half().reshape(seq[0].out_channels, -1).contiguous().to(device)
        self.dw = [_deconv_rows(m.weight.detach()).to(device) for m in seq[1:]]

    def __call__(self, k: DptFused, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        y = k.gemm(x, self.w)
        for wd in self.dw:
            y = k.gemm(y, wd, shuffle=(h, w, 2, 2))
            h, w = 2 * h, 2 * w
        return y


def _deconv_rows(wd: torch.Tensor) -> torch.Tensor:
    """ConvTranspose2d weight `[Cin, Cout, 2, 2]` -> shuffle-GEMM weight `[4 * Cout, Cin]` (rows `(a, b, cout)`)."""
    ci, co = wd.shape[:2]
    return wd.permute(2, 3, 1, 0).reshape(4 * co, ci).half().contiguous()


class _KernelsNet:
    """Depth Pro on the `dpt_fused` kernels (NHWC fp16 maps, fused epilogues), FOV tail in float32."""

    def __init__(self, net: DepthProNet, k: DptFused, device: torch.device, *, streams: bool) -> None:
        self.k, self.device = k, device
        enc: Any = net.encoder
        dec: Any = net.decoder
        tok, m0, m1, m2 = tables()
        self.tok, self.m0, self.m1, self.m2 = (t.to(device) for t in (tok, m0, m1, m2))
        self.up_l0, self.up_l1 = _Up(enc.upsample_latent0, device), _Up(enc.upsample_latent1, device)
        self.up0, self.up1, self.up2 = (_Up(s, device) for s in (enc.upsample0, enc.upsample1, enc.upsample2))
        ul = enc.upsample_lowres
        self.up_lr = (_deconv_rows(ul.weight.detach()).to(device), ul.bias.detach().half().repeat(4).to(device))
        fw, fb = _linear_1x1(enc.fuse_lowres, device)
        c = fw.shape[1] // 2
        self.fuse = (fw[:, :c].contiguous(), fw[:, c:].contiguous(), fb)
        self.convs = [None] + [_conv(m, device) for m in dec.convs[1:]]
        self.res = [
            [(_conv(rb.residual[1], device), _conv(rb.residual[3], device)) for rb in (fu.resnet1, fu.resnet2)]
            for fu in dec.fusions
        ]
        self.fdec: list[tuple[torch.Tensor, torch.Tensor]] = [
            (_deconv_rows(w).to(device), b.half().repeat(4).contiguous().to(device))
            for w, b in (compose_deconv_1x1(fu.deconv, fu.out_conv) for fu in dec.fusions[1:])
        ]  # levels 1-4 at index level - 1
        self.out0 = _linear_1x1(dec.fusions[0].out_conv, device)
        w0, b0, w1, b1, w2, b2 = compose_head(net.head)
        cm = net.head[0].out_channels
        self.h0 = (w0[:cm].permute(0, 2, 3, 1).half().contiguous().to(device), b0[:cm].half().to(device))
        self.h0_const = b0[cm:].half().to(device)  # the constant channels (1, 0, ..., 0)
        self.h1 = (w1.permute(0, 2, 3, 1).half().contiguous().to(device), b1.half().to(device))
        w2p = torch.zeros(8, w2.shape[1])  # 4 phases, padded to 8 rows
        w2p[:4] = w2
        b2p = torch.zeros(8)
        b2p[:4] = b2
        self.h2 = (w2p.half().to(device), b2p.half().to(device))
        fov = net.fov
        self.fov_linear = fov.encoder[1].float().to(device)
        self.fov_down, self.fov_head = fov.downsample.float().to(device), fov.head.float().to(device)
        # exact GELU throughout: the tanh form's bias shifts the focal length past the golden's tolerance (in every
        # encoder: the FOV head reads the patch features too)
        self.pvit = FusedViT(enc.patch_encoder, device, k, gelu=ACT_GELU)
        self.ivit = FusedViT(enc.image_encoder, device, k, gelu=ACT_GELU)
        self.fvit = FusedViT(fov.encoder[0], device, k, gelu=ACT_GELU)
        self.streams = (side_stream(device, 0), side_stream(device, 1)) if streams else None

    def _vit(self, fv: FusedViT, cols: torch.Tensor) -> torch.Tensor:
        """Patch rows `[576, kp]` of the 384x384 image -> normalised patch tokens `[576, C]`."""
        pos, cls = fv._pos_tables(24, 24)  # noqa: SLF001 - the DAv2 kernels helper's tables
        t = torch.empty((1, 577, fv.vit.embed_dim), device=self.device, dtype=torch.float16)
        t[0, 0] = cls
        self.k.gemm(cols, fv.w_patch, fv.b_patch, res=pos, out=t[0, 1:], acc32=True)
        for i in range(len(fv.blocks)):
            t = fv._block(i, t)  # noqa: SLF001 - in place
        return fv.vit.norm(t[0, 1:])

    def _fov(self, cols2: torch.Tensor) -> torch.Tensor:
        return self.fov_linear(self._vit(self.fvit, cols2).float())  # [576, 128]

    def _resblock(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        prm: tuple,
        res2: torch.Tensor | None = None,
        y2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`x + conv2(relu(conv1(r)))` for `r = relu(x)` (+ `res2`; `y2` gets relu(result))."""
        (w1, b1, _, _), (w2, b2, _, _) = prm
        k = self.k
        t = k.conv(r, w1, b1, padding=1, act=ACT_RELU)
        return k.conv(t, w2, b2, padding=1, res=x, res2=res2, y2=y2)

    def _encode(self, x: torch.Tensor, *, with_fov: bool) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Pyramid, the three ViTs and the encoder projections -> (encodings, FOV tokens or None).

        Encodings: the latent0 rows `[96, 96, C]` (projected in the decoder), NHWC levels 1-4; FOV tokens `[576, 128]`
        float32.
        """
        k, pv = self.k, self.pvit
        x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        c = pv.vit.embed_dim
        e = torch.empty((96 * 96 + 48 * 48 + 24 * 24, c), device=self.device, dtype=torch.float16)
        o = 0
        cols = e
        for v in (x, x1, x2):
            s = v.shape[-1]
            cols = k.patches(v, (s, s), 16, pv.kp, _MEAN, _MEAN)[0]  # the last: the 384x384 image's patches
            k.gemm(cols, pv.w_patch, pv.b_patch, out=e[o : o + cols.shape[0]], acc32=True)
            o += cols.shape[0]
        pos, cls = pv._pos_tables(24, 24)  # noqa: SLF001 - the DAv2 kernels helper's tables
        t = torch.empty((self.tok.shape[0], 577, c), device=self.device, dtype=torch.float16)
        t[:, 0] = cls
        torch.add(e[self.tok], pos, out=t[:, 1:])
        main = torch.cuda.current_stream(self.device)
        side = self.streams
        g = fz = None
        if side is not None:  # the two batch-1 ViTs fill the GPU poorly: overlap them with the patch encoder
            for st in side:
                st.wait_stream(main)
            with torch.cuda.stream(side[0]):
                g = self._vit(self.ivit, cols)
            if with_fov:
                with torch.cuda.stream(side[1]):
                    fz = self._fov(cols)
        lat = []
        for i in range(len(pv.blocks)):
            t = pv._block(i, t)  # noqa: SLF001 - in place
            if i in HOOKS:
                lat.append(t.view(-1, c)[self.m0])
        rows = t.view(-1, c)
        f0, f1, f2 = (pv.vit.norm(rows[m]) for m in (self.m0, self.m1, self.m2))
        if side is not None:
            for st, r in zip(side, (g, fz), strict=True):
                if r is not None:
                    main.wait_stream(st)
                    r.record_stream(main)
        g = self._vit(self.ivit, cols) if g is None else g
        fz = self._fov(cols) if with_fov and fz is None else fz
        b = k.gemm(g, *self.up_lr, shuffle=(24, 24, 2, 2))
        fa, fb, fbias = self.fuse
        enc4 = k.gemm(b, fb, fbias, res=k.gemm(self.up2(k, f2, 24, 24), fa))
        encs = [lat[0], self.up_l1(k, lat[1], 96, 96), self.up0(k, f0, 96, 96), self.up1(k, f1, 48, 48), enc4]
        return encs, fz

    def _decode(self, encs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodings -> (decoder features `[1, 768, 768, 256]`, the level-4 features `[1, 48, 48, 256]`), NHWC."""
        k = self.k
        r = torch.empty((1, 48, 48, DEC), device=self.device, dtype=torch.float16)
        feat = k.conv(encs.pop(), self.convs[4][0], padding=1, y2=r)  # ty: ignore[not-subscriptable]
        lowres = feat
        feat = k.gemm(self._resblock(feat, r, self.res[4][1]), *self.fdec[3], shuffle=(48, 48, 2, 2))
        for i in (3, 2, 1, 0):  # references dropped early: the 768x768 maps are 302 MB each
            if i:
                r = torch.empty_like(feat)
                v = k.conv(encs.pop(), self.convs[i][0], padding=1, y2=r)  # ty: ignore[not-subscriptable]
            else:  # convs[0] = identity
                v = self.up_l0(k, encs.pop(), 96, 96)
                r = torch.relu(v)
            r2 = torch.empty_like(feat)
            s = self._resblock(v, r, self.res[i][0], res2=feat, y2=r2)
            del v, r, feat
            x = self._resblock(s, r2, self.res[i][1])
            del s, r2
            hh, ww = x.shape[1:3]
            feat = k.gemm(x, *self.fdec[i - 1], shuffle=(hh, ww, 2, 2)) if i else k.gemm(x, *self.out0)
            del x
        return feat, lowres

    def _head(self, feat: torch.Tensor) -> torch.Tensor:
        """Decoder features -> canonical inverse depth `[1, 1, 1536, 1536]` float32."""
        k = self.k
        hb = torch.empty((*feat.shape[:3], self.h0[0].shape[0] + 8), device=self.device, dtype=torch.float16)
        hb[..., -8:] = self.h0_const
        k.conv(feat, *self.h0, padding=1, out=hb[..., :-8])
        y = k.conv(hb, *self.h1, padding=1, act=ACT_RELU)
        del hb
        z = k.gemm(y, *self.h2, act=ACT_RELU)
        return _pixel_shuffle(z[..., :4].float())

    def __call__(self, x: torch.Tensor, *, with_fov: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`[1, 3, 1536, 1536]` in [0, 1] -> (canonical inverse depth `[1, 1, 1536, 1536]` fp32, FOV or None)."""
        encs, fz = self._encode(x, with_fov=with_fov)
        feat, lowres = self._decode(encs)
        cid = self._head(feat)
        del feat
        fov = None
        if fz is not None:
            lr = self.fov_down(lowres.permute(0, 3, 1, 2).float())
            fov = self.fov_head(fz.t().reshape_as(lr) + lr)
        return cid, fov


class Kernels(Runner):
    """`dpt_fused` kernels: fused-epilogue CUTLASS GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 maps.

    The image and FOV ViTs run on side streams next to the patch encoder. Converted on the CPU.
    """

    intrinsics = "optional"  # given intrinsics replace the FOV estimate (fx)

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert it."""
        super().__init__(ctx)
        k = load_dpt_fused(ctx.device)  # before the weights: a kernel build / load failure costs no weight load
        self.net = _KernelsNet(_net(ctx), k, ctx.device, streams=True)

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Depth (and intrinsics when none are given) of one image."""
        h, w = image.shape[-2:]
        x = image.float()
        if x.shape[-2:] != (SIZE, SIZE):
            x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        cid, fov = self.net(x.contiguous(), with_fov=intrinsics is None)
        return _finish(cid, fov, intrinsics, h, w)


_MEAN = (0.5, 0.5, 0.5)  # fixed: upstream Normalize([0.5] * 3, [0.5] * 3) (mean = std)

# after the portable part: kernels-only dependencies
from fastforks.depth.models.depth_anything_v2.fused import FusedViT, _conv, _linear_1x1  # noqa: E402 - shared conversions
from fastforks.kernels.dpt_fused import ACT_GELU, ACT_RELU, DptFused  # noqa: E402
from fastforks.kernels.dpt_fused import load as load_dpt_fused  # noqa: E402

RUNNERS = {"portable": Portable, "kernels": Kernels}
