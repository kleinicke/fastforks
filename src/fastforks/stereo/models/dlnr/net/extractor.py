import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


# fastforks: the LayerNorms / Attention / FeedForward below hold the parameters (same state-dict keys as upstream);
# TransformerBlock.forward runs all three on the NHWC view (see there), so their own forwards are not needed.
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def nhwc(self, x):
        """fastforks: upstream's forward (to_3d, LayerNorm over C, to_4d) on an NHWC tensor; fp32 statistics."""
        w = self.body.weight.to(x.dtype)
        if isinstance(self.body, WithBias_LayerNorm):
            return F.layer_norm(x, self.body.normalized_shape, w, self.body.bias.to(x.dtype), 1e-5)
        var = torch.var(x.float(), dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(var + 1e-5).to(x.dtype) * w


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


def _w2(conv):
    """fastforks: 1x1 conv weight as a [out, in] matrix (a view)."""
    return conv.weight.view(conv.out_channels, -1)


def _bias(conv):
    return None if conv.bias is None else conv.bias


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """fastforks: upstream's x + attn(norm1(x)), then x + ffn(norm2(x)), on the NHWC view (our fork's friendly
        restructuring, `fastbb.restormer`): the 1x1 convs as GEMMs over the channels, the depthwise convs on
        the channels-last view, the attention per head as upstream (L2-normalised q, k over the pixels, q k^T scaled by
        the temperature, softmax in fp32), then attn @ v and project_out folded into one C x C matrix per sample so
        that x + project_out(attn @ v) is one batched GEMM with the residual as its accumulator input (baddbmm); the
        GDFN's project_out also accumulates onto the residual (addmm). Returns a channels-last NCHW view."""
        B, C, H, W = x.shape
        xh = x.permute(0, 2, 3, 1).contiguous()  # free for channels-last input
        x3 = xh.reshape(B, H * W, C)

        # MDTA
        at = self.attn
        heads, c = at.num_heads, C // at.num_heads
        y = F.linear(self.norm1.nhwc(xh), _w2(at.qkv), _bias(at.qkv))  # [B, H, W, 3C]
        qkv = F.conv2d(y.permute(0, 3, 1, 2), at.qkv_dwconv.weight, at.qkv_dwconv.bias, 1, 1, 1, 3 * C)
        t = qkv.permute(0, 2, 3, 1).reshape(B, H * W, 3 * C)
        q = F.normalize(t[:, :, :C].unflatten(2, (heads, c)).permute(0, 2, 3, 1), dim=-1)  # [B, heads, c, HW]
        k = F.normalize(t[:, :, C:2 * C].unflatten(2, (heads, c)).permute(0, 2, 3, 1), dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)).float() * at.temperature.float(), dim=-1)
        wpT = _w2(at.project_out).t().float().reshape(heads, c, C)  # project_out per head: [heads, c, C]
        M = torch.matmul(attn.transpose(-1, -2), wpT).reshape(B, C, C).to(x.dtype)
        x3 = torch.baddbmm(x3, t[:, :, 2 * C:], M)
        if at.project_out.bias is not None:
            x3 = x3 + at.project_out.bias

        # GDFN
        ff = self.ffn
        z = F.linear(self.norm2.nhwc(x3.reshape(B, H, W, C)), _w2(ff.project_in), _bias(ff.project_in))
        d = F.conv2d(z.permute(0, 3, 1, 2), ff.dwconv.weight, ff.dwconv.bias, 1, 1, 1, z.shape[-1])
        a, b = d.permute(0, 2, 3, 1).chunk(2, dim=-1)
        g = (F.gelu(a) * b).reshape(B * H * W, -1)
        out = torch.addmm(x3.reshape(B * H * W, C), g, _w2(ff.project_out).t())
        if ff.project_out.bias is not None:
            out = out + ff.project_out.bias
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Channel_Attention_Transformer_Extractor(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):
        super(Channel_Attention_Transformer_Extractor, self).__init__()

        self.pixelUnShuffle = nn.PixelUnshuffle(4)

        self.patch_embed = OverlapPatchEmbed(48, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_copy = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_copy = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output16 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.outputs16 = nn.ModuleList((self.output16, self.output16))
        self.outputs08 = nn.ModuleList((self.reduce_chan_level2, self.reduce_chan_level2_copy))
        self.outputs04 = nn.ModuleList((self.decoder_level1, self.decoder_level1_copy))

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        downsampled_img = self.pixelUnShuffle(inp_img)

        inp_enc_level1 = self.patch_embed(downsampled_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3_copy = inp_dec_level3[:(inp_dec_level3.shape[0] // 2)]
        outputs16 = [f(inp_dec_level3_copy) for f in self.outputs16]
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2_copy = inp_dec_level2[:(inp_dec_level2.shape[0] // 2)]
        outputs08 = [f(inp_dec_level2_copy) for f in self.outputs08]
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        v = inp_dec_level1
        inp_dec_level1_copy = inp_dec_level1[:(inp_dec_level1.shape[0] // 2)]
        outputs04 = [f(inp_dec_level1_copy) for f in self.outputs04]
        return outputs04, outputs08, outputs16, v
