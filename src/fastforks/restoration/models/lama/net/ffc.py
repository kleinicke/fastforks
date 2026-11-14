# Ported from advimman/lama saicinpainting/training/modules/ffc.py (Apache-2.0, see ../LICENSE) at
# 786f5936b27fb3dacd2b1ad799e4de968ea697e7. fastforks patches: only the options of big-lama (no local Fourier unit,
# squeeze-excitation, spatial transform, gating, spectral position encoding, 3-D FFT or spatial rescaling; the
# discriminator dropped); the Fourier unit runs in float32 (cuFFT's half precision needs power-of-two sizes);
# `fold_batchnorm` (fastforks) merges the eval-mode BatchNorms into the convolutions before them; the reflect padding
# is explicit (`reflect_pad`, once per FFC input, keeps channels-last) and the Fourier unit transforms the NHWC view
# (its spectrum is then channels-last with upstream's `c * 2 + real / imag` channel order, no layout copies).
import torch
import torch.nn as nn


def reflect_pad(x, p):
    """`F.pad(x, (p, p, p, p), mode="reflect")` that keeps `x`'s memory format (slice copies into one buffer)."""
    if p == 0:
        return x
    b, c, h, w = x.shape
    fmt = torch.channels_last if x.stride(1) == 1 and c > 1 else torch.contiguous_format
    out = torch.empty((b, c, h + 2 * p, w + 2 * p), dtype=x.dtype, device=x.device, memory_format=fmt)
    out[..., p : p + h, p : p + w] = x
    if p == 1 and h >= 4 and w >= 4:  # noqa: PLR2004 - two strided copies: columns 1, w - 2 and rows 2, h - 1
        out[..., 1 : h + 1, :: w + 1] = x[..., :, 1 : w - 1 : w - 3]
        out[..., :: h + 1, :] = out[..., 2:h:h - 3, :]
        return out
    out[..., p : p + h, :p] = x[..., :, 1 : p + 1].flip(-1)
    out[..., p : p + h, p + w :] = x[..., :, w - p - 1 : w - 1].flip(-1)
    out[..., :p, :] = out[..., p + 1 : 2 * p + 1, :].flip(-2)
    out[..., p + h :, :] = out[..., h - 1 : h + p - 1, :].flip(-2)
    return out


class ReflectionPad(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return reflect_pad(x, self.p)


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        dtype = x.dtype
        nhwc = x.permute(0, 2, 3, 1).float()  # a view on channels-last input
        ffted = torch.view_as_real(torch.fft.rfftn(nhwc, dim=(1, 2), norm="ortho"))  # (b, h, w/2+1, c, 2)
        ffted = ffted.reshape(b, h, -1, 2 * c).permute(0, 3, 1, 2)  # channels c * 2 + real / imag, as upstream
        conv = self.conv_layer
        ffted = nn.functional.conv2d(ffted, conv.weight.float(), None if conv.bias is None else conv.bias.float())
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.permute(0, 2, 3, 1).reshape(b, h, -1, c, 2).contiguous()
        output = torch.fft.irfftn(torch.view_as_complex(ffted), s=(h, w), dim=(1, 2), norm="ortho")
        return output.permute(0, 3, 1, 2).to(dtype)


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        return self.conv2(x + output)


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        self.padding = padding  # reflect, applied once per input in `forward`

        def conv(i, o):
            if i == 0 or o == 0:
                return nn.Identity()
            return nn.Conv2d(i, o, kernel_size, stride, 0, dilation, bias=False)

        self.convl2l = conv(in_cl, out_cl)
        self.convl2g = conv(in_cl, out_cg)
        self.convg2l = conv(in_cg, out_cl)
        self.convg2g = nn.Identity() if in_cg == 0 or out_cg == 0 else SpectralTransform(in_cg, out_cg)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        p_l = reflect_pad(x_l, self.padding)
        p_g = reflect_pad(x_g, self.padding) if torch.is_tensor(x_g) and isinstance(self.convg2l, nn.Conv2d) else x_g
        if self.ratio_gout != 1:
            out_xl = self.convl2l(p_l) + self.convg2l(p_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(p_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation)
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.Identity() if ratio_gout == 1 else nn.BatchNorm2d(out_channels - global_channels)
        self.bn_g = nn.Identity() if ratio_gout == 0 else nn.BatchNorm2d(global_channels)
        self.act_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.act_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, ratio_gin=ratio, ratio_gout=ratio, padding=1)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, ratio_gin=ratio, ratio_gout=ratio, padding=1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    """big-lama: `input_nc` 4, `output_nc` 3, ngf 64, 3 downsamplings, `n_blocks` 18, global ratio 0.75, sigmoid."""

    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18, ratio=0.75, max_features=1024):
        super().__init__()
        model = [ReflectionPad(3), FFC_BN_ACT(input_nc, ngf, kernel_size=7, ratio_gin=0, ratio_gout=0)]
        for i in range(n_downsampling):
            mult = 2**i
            gout = ratio if i == n_downsampling - 1 else 0
            model += [
                FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3,
                           ratio_gin=0, ratio_gout=gout, stride=2, padding=1)
            ]
        feats_num_bottleneck = min(max_features, ngf * 2**n_downsampling)
        model += [FFCResnetBlock(feats_num_bottleneck, ratio) for _ in range(n_blocks)]
        model += [ConcatTupleLayer()]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)),
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(min(max_features, int(ngf * mult / 2))),
                nn.ReLU(True),
            ]
        model += [ReflectionPad(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


@torch.no_grad()
def _fold(bn, convs, out_dim=0):
    """Merge an eval BatchNorm into the convolutions whose summed outputs it normalises (bias on the first)."""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    shift = bn.bias - bn.running_mean * s
    for k, conv in enumerate(convs):
        shape = [1] * conv.weight.ndim
        shape[out_dim] = -1
        conv.weight.mul_(s.view(shape))
        bias = conv.bias * s if conv.bias is not None else torch.zeros_like(s)
        if k == 0:
            bias = bias + shift
        conv.bias = nn.Parameter(bias)


def fold_batchnorm(net):
    """Fold every BatchNorm of an eval `FFCResNetGenerator` into its convolutions (replaced by `nn.Identity`)."""
    for m in net.modules():
        if isinstance(m, FFC_BN_ACT):
            f = m.ffc
            if isinstance(m.bn_l, nn.BatchNorm2d):
                _fold(m.bn_l, [c for c in (f.convl2l, f.convg2l) if isinstance(c, nn.Conv2d)])
                m.bn_l = nn.Identity()
            if isinstance(m.bn_g, nn.BatchNorm2d):
                convs = [f.convl2g] + ([f.convg2g.conv2] if isinstance(f.convg2g, SpectralTransform) else [])
                _fold(m.bn_g, [c for c in convs if isinstance(c, nn.Conv2d)])
                m.bn_g = nn.Identity()
        elif isinstance(m, SpectralTransform):
            _fold(m.conv1[1], [m.conv1[0]])
            m.conv1[1] = nn.Identity()
        elif isinstance(m, FourierUnit):
            _fold(m.bn, [m.conv_layer])
            m.bn = nn.Identity()
    seq = net.model
    for i, layer in enumerate(seq):
        if isinstance(layer, nn.BatchNorm2d) and isinstance(seq[i - 1], nn.ConvTranspose2d):
            _fold(layer, [seq[i - 1]], out_dim=1)
            seq[i] = nn.Identity()
