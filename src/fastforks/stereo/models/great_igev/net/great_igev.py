"""GREAT-IGEV network (upstream `models/great_stereo/great_igev_stereo.py` + the modules it uses), inference forms.

Module tree and parameter names are upstream's (the released checkpoints load strictly); the IGEV building blocks
(`MultiBasicEncoder`, `Feature` = upstream's `Mobilenetv2Encoder`, `BasicConv`, `BasicConv_IN` = `BasicConvIN`,
`Conv2x`, `Conv2x_IN` = `Conv2xIN`) are `fastforks.shared.igev_blocks`, key-identical. The forwards are rewritten for
test-mode inference (`portable.forward_fast` / `max.inference.forward_max` drive them), same math:
  - the excitive-attention cost volume (concat volume -> `cc_stem` -> x feature stem -> `fuse_stem`) is built and
    filtered in disparity chunks with a 2-plane halo (the full 192-channel concat volume at Middlebury size is
    several GB),
  - the attention layers as in `transformers.py`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.igev_blocks.net.extractor import Feature, MultiBasicEncoder
from fastforks.shared.igev_blocks.net.submodule import BasicConv, BasicConv_IN, Conv2x, Conv2x_IN, disparity_regression
from fastforks.shared.mobilenetv2 import MobileNetV2Features

from .transformers import (
    FeedForwardLayer,
    OutlookAttentionLayer,
    PositionalEmbeddingCosine2D,
    TransformerFeatureRefiner,
    VolumeTransformerBlock,
    WindowAttentionLayer,
    d_chunks,
    tokens,
    volume_softmax,
)
from .update import BasicMultiUpdateBlock

CC_CHUNK = 2**28  # concat cost volume elements per disparity chunk


class AdaptiveGlobalRefiner(nn.Module):
    """Channel + spatial attention, small / large receptive field blend (upstream `AdaptiveGlobalRefiner`)."""

    def __init__(self, channels):
        super().__init__()
        self.pre_channel_agg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        )
        self.sam_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.cam_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        self.small_reception_conv = BasicConv_IN(channels, channels, kernel_size=3, padding=1, stride=1)
        self.large_repcetion_conv = BasicConv_IN(channels, channels, kernel_size=5, padding=2, stride=1)  # sic
        self.final_fusion = nn.Sequential(
            BasicConv_IN(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        f = self.pre_channel_agg(x)
        b = f.shape[0]
        pooled = torch.cat([f.mean((2, 3), keepdim=True), f.amax((2, 3), keepdim=True)], 0)
        ca = self.cam_conv(pooled.contiguous(memory_format=torch.channels_last))
        f = f * torch.sigmoid(ca[:b] + ca[b:])
        del ca, pooled
        m = torch.cat([f.mean(1, keepdim=True), f.amax(1, keepdim=True)], 1)
        m = torch.sigmoid(self.sam_conv(m.contiguous(memory_format=torch.channels_last)))
        small = self.small_reception_conv(f)
        large = self.large_repcetion_conv(f)
        del f
        refined = small * m + large * (1 - m)
        del small, large, m
        return self.final_fusion(refined.add_(x))


class GlobalAttentionFeatureEncoder(nn.Module):
    """1x1 channel reduction -> outlook self attention -> + position -> window / shifted-window self attention."""

    def __init__(self, in_channels, cv_channels, num_splits, kernel_size, stride, padding):
        super().__init__()
        self.num_splits = num_splits
        self.channels_agg = nn.Sequential(
            BasicConv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels // 2, cv_channels, 1),
        )
        self.feature_pos = PositionalEmbeddingCosine2D(cv_channels // 2, normalize=True)
        self.outlook_self_attn = OutlookAttentionLayer(
            cv_channels, cv_channels, cv_channels, kernel_size=kernel_size, padding=padding, stride=stride
        )
        self.outlook_self_ffn = FeedForwardLayer(cv_channels, cv_channels)
        self.win_self_attn = WindowAttentionLayer(cv_channels, cv_channels, cv_channels)
        self.win_self_ffn = FeedForwardLayer(cv_channels, cv_channels)
        self.swin_self_attn = WindowAttentionLayer(cv_channels, cv_channels, cv_channels)
        self.swin_self_ffn = FeedForwardLayer(cv_channels, cv_channels)

    def forward(self, x):
        x = self.channels_agg(x)
        x = x + self.outlook_self_attn(x)
        x = x + self.outlook_self_ffn(x)
        x = x + self.feature_pos(x).to(x.dtype)
        x = x + self.win_self_attn(x, x, self.num_splits, with_shift=False)
        x = x + self.win_self_ffn(x)
        x = x + self.swin_self_attn(x, x, self.num_splits, with_shift=True)
        return x + self.swin_self_ffn(x)


class GlobalAttentionFeatureRefiner(nn.Module):
    def __init__(self, cv_channels):
        super().__init__()
        c = cv_channels
        self.feature_encoder_4 = GlobalAttentionFeatureEncoder(96, c, num_splits=8, kernel_size=7, stride=1, padding=3)
        self.feature_encoder_8 = GlobalAttentionFeatureEncoder(64, c * 2, num_splits=4, kernel_size=5, stride=1, padding=2)
        self.feature_encoder_16 = GlobalAttentionFeatureEncoder(
            192, c * 4, num_splits=2, kernel_size=3, stride=1, padding=1
        )
        self.feature_encoder_32 = GlobalAttentionFeatureEncoder(
            160, c * 6, num_splits=1, kernel_size=3, stride=1, padding=1
        )

    def forward(self, feats):
        encs = (self.feature_encoder_4, self.feature_encoder_8, self.feature_encoder_16, self.feature_encoder_32)
        return [e(f) for e, f in zip(encs, feats, strict=True)]


class ExcitiveAttentionHourglassEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        kw = {"is_3d": True, "bn": True, "relu": True, "kernel_size": 3, "padding": 1, "dilation": 1}
        up = {"deconv": True, "is_3d": True, "kernel_size": (4, 4, 4), "padding": (1, 1, 1), "stride": (2, 2, 2)}
        self.conv0 = BasicConv(c, c, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(BasicConv(c, c * 2, stride=2, **kw), BasicConv(c * 2, c * 2, stride=1, **kw))
        self.conv2 = nn.Sequential(BasicConv(c * 2, c * 4, stride=2, **kw), BasicConv(c * 4, c * 4, stride=1, **kw))
        self.conv3 = nn.Sequential(BasicConv(c * 4, c * 6, stride=2, **kw), BasicConv(c * 6, c * 6, stride=1, **kw))
        self.conv3_up = BasicConv(c * 6, c * 4, bn=True, relu=True, **up)
        self.conv2_up = BasicConv(c * 4, c * 2, bn=True, relu=True, **up)
        self.conv1_up = BasicConv(c * 2, 8, bn=False, relu=False, **up)
        self.agg_0 = nn.Sequential(
            BasicConv(c * 8, c * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(c * 4, c * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(c * 4, c * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(c * 4, c * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(c * 2, c * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(c * 2, c * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.feature_attn_4 = VolumeTransformerBlock(c, num_heads=1)
        self.feature_attn_8 = VolumeTransformerBlock(c * 2, num_heads=1)
        self.feature_attn_16 = VolumeTransformerBlock(c * 4, num_heads=1)
        self.feature_attn_32 = VolumeTransformerBlock(c * 6, num_heads=1)
        self.feature_attn_up_16 = VolumeTransformerBlock(c * 4, num_heads=1)
        self.feature_attn_up_8 = VolumeTransformerBlock(c * 2, num_heads=1)

    def hourglass(self, x, feats):
        """The filtered volume [B, 8, D, H, W]."""
        conv0 = self.feature_attn_4(self.conv0(x), feats[0])
        conv1 = self.feature_attn_8(self.conv1(conv0), feats[1])
        del conv0
        conv2 = self.feature_attn_16(self.conv2(conv1), feats[2])
        conv3 = self.feature_attn_32(self.conv3(conv2), feats[3])
        y = torch.cat((self.conv3_up(conv3), conv2), dim=1)
        del conv3, conv2
        conv2 = self.feature_attn_up_16(self.agg_0(y), feats[2])
        y = torch.cat((self.conv2_up(conv2), conv1), dim=1)
        del conv2, conv1
        conv1 = self.feature_attn_up_8(self.agg_1(y), feats[1])
        del y
        return self.conv1_up(conv1)

    def forward(self, x, feats):
        """-> (volume [B, 8, D, H, W], disparity probabilities [B, D, H, W] fp32)."""
        conv = self.hourglass(x, feats)
        t = conv.permute(0, 2, 3, 4, 1)
        prob = volume_softmax(t if t.is_contiguous() else t.contiguous(), tokens(feats[0]).float() * 8**-0.5, 1e-6)
        return conv, prob

    def disparity(self, x, feats):
        """-> (volume [B, 8, D, H, W], initial disparity [B, 1, H, W] fp32): forward + the disparity regression."""
        conv, prob = self(x, feats)
        return conv, disparity_regression(prob, prob.shape[1])


class ExcitiveAttentionVolume(nn.Module):
    def __init__(self, channels, max_disp, num_groups=8):
        super().__init__()
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.global_spatial_refiner = AdaptiveGlobalRefiner(channels)
        self.matching_pos = PositionalEmbeddingCosine2D(channels // 2, normalize=True)
        self.global_matching_refiner = TransformerFeatureRefiner(channels, 1, 4, receptive_aug="conv")
        self.global_context_refiner = GlobalAttentionFeatureRefiner(num_groups)
        self.cc_stem = nn.Sequential(
            BasicConv(channels * 2, channels, is_3d=True, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv(channels, channels // 2, is_3d=True, kernel_size=3, stride=1, padding=1),
            BasicConv(channels // 2, num_groups, is_3d=True, kernel_size=3, stride=1, padding=1),
        )
        self.feat_stem = nn.Sequential(
            BasicConv(num_groups, num_groups, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_groups, num_groups, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.fuse_stem = BasicConv(num_groups, num_groups, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.eav_agg = ExcitiveAttentionHourglassEncoder(num_groups)

    def cost_volume(self, left, right, context):
        """fuse_stem(feat_stem(context) x cc_stem(concat volume)), [B, 8, D, H, W] channels-last-3d (D = max_disp / 4).

        Built in disparity chunks: cc_stem's two 3x3x3 convs need 2 planes of halo, the rest is per plane.
        """
        b, c, h, w = left.shape
        d = self.max_disp // 4
        cl3 = torch.channels_last_3d
        feat = self.feat_stem(context).unsqueeze(2)
        out = None
        for d0, d1 in d_chunks(d, 2 * c * h * w, CC_CHUNK):
            a, e = max(0, d0 - 2), min(d, d1 + 2)
            vol = torch.empty((b, 2 * c, e - a, h, w), device=left.device, dtype=left.dtype, memory_format=cl3).zero_()
            vol[:, :c] = left.unsqueeze(2)
            for i in range(a, e):
                if i == 0:
                    vol[:, c:, 0] = right
                elif i < w:
                    vol[:, c:, i - a, :, i:] = right[:, :, :, :-i]
            y = self.cc_stem(vol)
            del vol
            y = y[:, :, d0 - a : d1 - a].contiguous(memory_format=cl3)
            y = self.fuse_stem(y.mul_(feat))
            if out is None:
                out = torch.empty((b, y.shape[1], d, h, w), device=y.device, dtype=y.dtype, memory_format=cl3)
            out[:, :, d0:d1] = y
            del y
        return out


class GREATStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        ch = args.hidden_dims
        self.cnet = MultiBasicEncoder(output_dim=[ch, ch], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=ch)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(ch[i], ch[i] * 3, 3, padding=3 // 2) for i in range(self.args.n_gru_layers)]
        )
        self.feature = Feature(MobileNetV2Features())
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        # spx / spx_2 / spx_4 upsample the initial disparity in training only (kept: the checkpoints hold them)
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))
        self.excitive_attention_volume = ExcitiveAttentionVolume(96, self.args.max_disp, 8)
