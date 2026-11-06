"""Portable inference path for MoCha-Stereo (fastforks; plain PyTorch ops only, exports to ONNX / TensorRT).

Port of our MoCha fork's `portable` tier (merged LSTM gate convs, sync-free lookup, mask features only on the last
iteration, REMP in the inference dtype) on the IGEV-family `fold_for_inference`:

fold_for_inference(model, half, channels_last):
  - pure fp16 (or fp32) weights once; BatchNorm statistics kept as an fp32 per-channel affine applied to the conv
    output (folding them into fp16 weights loses precision); layout-preserving (NHWC / NDHWC) norms
  - the four LSTM gate convs of each level as one conv (gate order i, f, c~, o); their biases go into the context
forward_fast(model, image1, image2, iters): upstream's test-mode schedule with
  - disparity state, lookup coordinates, the REMP warp and residual, softmax / regression / upsampling in fp32
  - the geometry / correlation lookup by gathers instead of grid_sample (+ its host sync)
  - MCCV's high-frequency filter without the C x C x 3 x 3 box conv (every output channel is the 3x3 box mean of the
    channel sum) and only the 1/8 and 1/16 channel-correlation volumes: the hourglass never reads the 1/4 one
  - upstream feeds the LSTMs `netC is netH` (the same list): the cell input of every step is the previous hidden
    state (pre-Mogrifier), never a carried cell state. Reproduced as such.
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import base_type, override
from . import extractor, submodule
from .geometry import FastGeoEncodingVolume
from .mccv import Mca_Camp
from .submodule import build_gwc_volume_fast, disparity_regression
from .update import interp, pool2x


def _bn_affine(m):
    scale = m.weight.float() * torch.rsqrt(m.running_var.float() + m.eps)
    shift = m.bias.float() - m.running_mean.float() * scale
    return scale, shift


def _bn_forward(self, x):
    shp = (1, -1) + (1,) * (x.dim() - 2)
    return x.mul_(self._s.to(x.dtype).view(shp)).add_(self._t.to(x.dtype).view(shp))


def _bn_act_forward(self, x):  # efficientnet BatchNormAct2d: BN -> SiLU / identity
    return self.act(_bn_forward(self, x))


def _in_forward(self, x):
    # InstanceNorm2d / 3d (affine=False): fp32 statistics, applied in place for fp32 (keeps NHWC / NDHWC)
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    if x.dtype == torch.float32:
        return x.sub_(mean).mul_(torch.rsqrt(var + self.eps))
    return ((x.float() - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)  # fp16: fp32 per element, as F.instance_norm


def _basic_conv_forward(self, x):  # submodule BasicConv / BasicConv_IN: conv -> norm -> LeakyReLU(0.01), in place
    x = self.conv(x)
    if getattr(self, "use_bn", False):
        x = self.bn(x)
    if getattr(self, "use_in", False):
        x = self.IN(x)
    if self.relu:
        x = F.leaky_relu_(x)
    return x


def _resblock_forward(self, x):
    y = self.relu(self.norm1(self.conv1(x)))
    y = self.relu(self.norm2(self.conv2(y)))
    if self.downsample is not None:
        x = self.downsample(x)
    return y.add_(x).relu_()


def _cnet_forward(self, x, num_layers=3):
    """MultiBasicEncoder.forward without upstream's `x = x.float()` after conv1 (an autocast leftover: the fp16 path
    keeps fp16)."""
    x = self.layer3(self.layer2(self.layer1(self.relu1(self.norm1(self.conv1(x))))))
    out = [[f(x) for f in self.outputs04]]
    if num_layers > 1:
        y = self.layer4(x)
        out.append([f(y) for f in self.outputs08])
        if num_layers > 2:
            z = self.layer5(y)
            out.append([f(z) for f in self.outputs16])
    return out


@torch.no_grad()  # the folded affines are constants, not part of an autograd graph
def fold_for_inference(model, half=True, channels_last=True):
    model.eval()
    dt = torch.float16 if half else torch.float32
    for g in (model.update_block.lstm04, model.update_block.lstm08, model.update_block.lstm16):
        convs = (g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot)
        c4 = nn.Conv2d(g.conv_it.in_channels, 4 * g.conv_it.out_channels, 3, padding=1, bias=False)
        c4.weight.data = torch.cat([c.weight.data for c in convs], 0)
        g.conv4 = c4.to(g.conv_it.weight.device)
        g._b4 = torch.cat([c.bias.data.float() for c in convs])
    for m in model.modules():
        t = base_type(m)
        if t in (nn.BatchNorm2d, nn.BatchNorm3d) or t.__name__ == "BatchNormAct2d":
            m._s, m._t = _bn_affine(m)
            override(m, _bn_act_forward if t.__name__ == "BatchNormAct2d" else _bn_forward)
        elif t in (nn.InstanceNorm2d, nn.InstanceNorm3d) and not m.affine and not m.track_running_stats:
            override(m, _in_forward)
        elif t in (submodule.BasicConv, submodule.BasicConv_IN):
            override(m, _basic_conv_forward)
        elif t is extractor.ResidualBlock:
            override(m, _resblock_forward)
        elif t is extractor.MultiBasicEncoder:
            override(m, _cnet_forward)
    model.to(dt)
    for m in model.modules():  # BN affine stays fp32 (cast to the activation dtype at use, tiny)
        if hasattr(m, "_s"):
            m._s, m._t = m._s.float(), m._t.float()
    if channels_last:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    model.inf_dtype, model.inf_cl = dt, channels_last
    return model


def context_upsample(disp_low, up_weights):
    """submodule.context_upsample with the 3x3 neighbourhoods as shifted slices of the zero-padded map (= F.unfold(..,
    3, 1, 1), same order; Im2Col has no ONNX / TensorRT counterpart)."""
    b, c, h, w = disp_low.shape
    dp = F.pad(disp_low, (1, 1, 1, 1))
    unf = torch.stack([dp[:, :, i : i + h, j : j + w] for i in range(3) for j in range(3)], 2).reshape(b, -1, h, w)
    unf = F.interpolate(unf, (h * 4, w * 4), mode="nearest").reshape(b, 9, h * 4, w * 4)
    return (unf * up_weights).sum(1)


def hf_filter(x):
    """Mca_Camp.high_frequency_sliding_window_filter: x - conv(x, ones(C, C, 3, 3) / 9, zero padding), i.e. minus the
    3x3 box mean (padding counted) of the channel sum in every channel; fp32 maths."""
    s = F.avg_pool2d(x.float().sum(1, keepdim=True), 3, 1, 1)
    return (x.float() - s).to(x.dtype)


def mccv(m, fl, fr, mf):
    """Mca_Camp.forward for the levels the hourglass reads ([None, 1/8, 1/16, fl[3]]; upstream also builds 1/4)."""
    out = [None]
    for i, sw, conv3d in ((1, m.sw_8x, m.Conv3d_layer_8x), (2, m.sw_16x, m.Conv3d_layer_16x)):
        nf = hf_filter(torch.cat([fl[i], fr[i]], 1)).contiguous(memory_format=mf)
        motif = F.softmax(sw(nf).float(), dim=1).to(fl[i].dtype)  # [B, 4, H, W]
        del nf
        camp = fl[i].unsqueeze(1) * motif.unsqueeze(2)  # [B, 4, C, H, W] (upstream: transpose of [B, C, 4, H, W])
        mf3 = torch.channels_last_3d if mf == torch.channels_last else torch.contiguous_format
        out.append(conv3d(camp.contiguous(memory_format=mf3)).squeeze(1).contiguous(memory_format=mf))
    out.append(fl[3])
    return out


def _lstm(g, h, ctx, x):
    """update.LSTM(c=h, h, *ctx, x) with the merged gate conv (ctx = context + gate biases, order i, f, c~, o)."""
    hm = h
    if x.shape[1] == h.shape[1]:  # Mogrifier (lstm16 only): x = 2 sigmoid(Q h) x, h = 2 sigmoid(R x) h alternately
        for ind in range(g.m.iters):
            if ind % 2 == 0:
                x = 2 * torch.sigmoid(g.m.Q(hm)) * x
            else:
                hm = 2 * torch.sigmoid(g.m.R(x)) * hm
    i, f, c, o = (g.conv4(torch.cat([hm, x], 1)) + ctx).chunk(4, 1)
    ct = h * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(c)
    return torch.sigmoid(o) * torch.tanh(ct)


def update_block(blk, net, ctx, corr, disp, compute_mask):
    """LSTMMultiUpdateBlock.forward (3 levels, test mode); net updated in place."""
    net[2] = _lstm(blk.lstm16, net[2], ctx[2], pool2x(net[1]))
    net[1] = _lstm(blk.lstm08, net[1], ctx[1], torch.cat([pool2x(net[0]), interp(net[2], net[1])], 1))
    mf = blk.encoder(disp, corr)
    net[0] = _lstm(blk.lstm04, net[0], ctx[0], torch.cat([mf, interp(net[1], net[0])], 1))
    del mf
    delta = blk.disp_head(net[0])
    return delta, blk.mask_feat_4(net[0]) if compute_mask else None


def warp(img, disp):
    """warp.disp_warp(img, disp)[0] (border padding, same coordinate formula, fp32) without the unused valid mask."""
    b, _, h, w = img.shape
    xs = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, w)
    gx = 2 * ((xs - disp[:, 0].float()) / (w - 1)) - 1
    gy = (2 * (torch.arange(h, device=img.device, dtype=torch.float32) / (h - 1)) - 1).view(1, h, 1).expand_as(gx)
    return F.grid_sample(
        img.float(), torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=False
    )  # upstream: the default (False), not passed


def remp(R, disp, left, right, dt, mf):
    """REMP.forward (disparity already at image resolution) through the module forwards in `dt`; warp, flaw and the
    residual in fp32. Returns the correction (upstream adds it to the disparity)."""
    ref = torch.cat([warp(right, disp) - left, left], 1).to(dt).contiguous(memory_format=mf)
    x = torch.cat([R.conv1(ref), R.conv2(disp.to(dt).contiguous(memory_format=mf))], 1)
    del ref
    x = R.RefinementBlock(R.conv_start(x))
    low = R.LFE(R.AP(x))
    motif = R.LMC(x)
    x = R.final_conv((1 - motif) * low + motif * x)
    return -F.relu(disp + x.float())


@torch.no_grad()
def forward_fast(self, image1, image2, iters=32):
    """Test-mode forward after fold_for_inference (see the module docstring); returns the disparity [B, 1, H, W]."""
    dt = self.inf_dtype
    mf = torch.channels_last if self.inf_cl else torch.contiguous_format
    D = self.args.max_disp // 4
    im1 = 2 * (image1 / 255.0) - 1.0
    im2 = 2 * (image2 / 255.0) - 1.0
    i1, i2 = (x.to(dt).contiguous(memory_format=mf) for x in (im1, im2))

    fr = self.feature(i2)  # right view: 1/4 matching features, 1/8 and 1/16 for the MCCV
    fr0 = torch.cat((fr[0], self.stem_4(self.stem_2(i2))), 1)
    match_right = self.desc(self.conv(fr0))
    del fr0, i2
    fl = self.feature(i1)
    stem_2x = self.stem_2(i1)
    stem_4x = self.stem_4(stem_2x)
    fl[0] = torch.cat((fl[0], stem_4x), 1)
    del stem_4x
    match_left = self.desc(self.conv(fl[0]))

    gwc_volume = build_gwc_volume_fast(match_left, match_right, D, 8, self.inf_cl)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, fl[0])
    ccv = mccv(self.mccv, fl, fr, mf) if isinstance(self.mccv, Mca_Camp) else self.mccv(fl, fr)
    del fr, fl
    volume = self.cost_agg(gwc_volume, ccv)
    del gwc_volume, ccv
    prob = F.softmax(self.classifier(volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, D)
    del prob
    # fastforks: the kernels runner on Apple MPS sets `geo_volume` (the lookups as Metal kernels)
    geo_fn = getattr(self, "geo_volume", FastGeoEncodingVolume)(
        match_left, match_right, volume, radius=self.args.corr_radius, num_levels=self.args.corr_levels, out_dtype=dt
    )
    del volume, match_left, match_right

    cnet_list = self.cnet(i1, num_layers=self.args.n_gru_layers)
    del i1
    net = [torch.tanh(x[0]) for x in cnet_list]
    lstms = (self.update_block.lstm04, self.update_block.lstm08, self.update_block.lstm16)
    ctx = [
        conv(torch.relu(x[1])).add_(g._b4.to(dt).view(1, -1, 1, 1))
        for x, conv, g in zip(cnet_list, self.context_zqr_convs, lstms, strict=True)
    ]
    del cnet_list

    disp = init_disp.float()
    mask_feat_4 = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h = disp.to(dt).contiguous(memory_format=mf)
        delta, mask_feat_4 = update_block(self.update_block, net, ctx, geo_feat, disp_h, itr == iters - 1)
        disp = disp + delta.float()

    xspx = self.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(self.spx_gru(xspx).float(), 1)
    disp_up = context_upsample(disp * 4.0, spx_pred).unsqueeze(1)
    return disp_up + remp(self.REMP, disp_up, im1, im2, dt, mf)
