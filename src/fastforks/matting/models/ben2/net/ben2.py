
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# fastforks: BEN2.py (PramaLLC/BEN2) inference network only: the Swin backbone is fastforks.shared.swin_v1 (upstream's
# copy of the same code), einops rearrange -> view / permute, timm / cv2 / video / PIL helpers removed
from fastforks.shared.swin_v1 import swin_v1_b


def swin_features(backbone, x):  # fastforks: upstream's SwinTransformer.forward, which also returns the patch embedding
    x = backbone.patch_embed(x)

    Wh, Ww = x.size(2), x.size(3)

    outs = [x.contiguous()]
    x = x.flatten(2).transpose(1, 2)
    x = backbone.pos_drop(x)


    for i in range(backbone.num_layers):
        layer = backbone.layers[i]
        x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)


        if i in backbone.out_indices:
            norm_layer = getattr(backbone, f'norm{i}')
            x_out = norm_layer(x_out)

            out = x_out.view(-1, H, W, backbone.num_features[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)



    return tuple(outs)


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "gelu":
        return F.gelu

    raise RuntimeError(F"activation should be gelu, not {activation}.")


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.InstanceNorm2d(out_dim), nn.GELU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.InstanceNorm2d(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    b, c, h, w = x.shape  # fastforks: einops rearrange -> view / permute (hg = wg = 2)
    x = x.view(b, c, 2, h // 2, 2, w // 2).permute(2, 4, 0, 1, 3, 5).reshape(4 * b, c, h // 2, w // 2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    n, c, h, w = x.shape  # fastforks: einops rearrange -> view / permute (hg = wg = 2)
    x = x.view(2, 2, n // 4, c, h, w).permute(2, 3, 0, 4, 1, 5).reshape(n // 4, c, 2 * h, 2 * w)
    return x


def _hw_b_c(x):  # fastforks: einops 'b c h w -> (h w) b c'
    return x.flatten(2).permute(2, 0, 1)


# fastforks: upstream defines this class twice (identical); one copy
class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32)

    def __call__(self, b, h, w, device=None):  # fastforks: built on the features' device (upstream: CPU, then moved)
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=device)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (2 * (self.dim_t.to(device) // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MCLM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(MCLM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('gelu')
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        # fastforks: the positional encodings are locals (upstream: self.p_poses / self.g_pos, reset every call),
        # cast to the features' dtype (upstream runs under autocast)
        p_poses = []
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = patches2image(l)

        pools = []
        for pool_ratio in self.pool_ratios:
             # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(_hw_b_c(pool))
            pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3], l.device)
            p_poses.append(_hw_b_c(pos_emb))
        pools = torch.cat(pools, 0)
        p_poses = torch.cat(p_poses, dim=0).to(l.dtype)
        g_pos = _hw_b_c(self.positional_encoding(g.shape[0], g.shape[2], g.shape[3], l.device)).to(l.dtype)


        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = _hw_b_c(g)


        # fastforks: need_weights=False (SDPA; upstream also computes the unused averaged attention weights)
        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](g_hw_b_c + g_pos, pools + p_poses, pools, need_weights=False)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = _hw_b_c(l)
        _g_hw_b_c = g_hw_b_c.view(h, w, -1, c)
        # fastforks: einops "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng = nw = 2
        _g_hw_b_c = _g_hw_b_c.view(2, h // 2, 2, w // 2, -1, c).permute(1, 3, 0, 2, 4, 5).reshape(h * w // 4, -1, c)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g, need_weights=False)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return l.view(h, w, -1, c).permute(2, 3, 0, 1)  ## (5,c,h*w)  # fastforks: einops "(h w) b c -> b c h w"


class MCRM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(MCRM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('gelu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios

    def forward(self, x):
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w

        patched_glb = image2patches(glb)

        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map, size=(2 * h, 2 * w), mode='nearest')  # fastforks: patches2image(loc).shape[-2:]
        loc = loc * image2patches(token_attention_map)

        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(pool.flatten(2))  # nl(4),c,hw

        pools = torch.cat(pools, 2).permute(0, 2, 1).unsqueeze(2)  # fastforks: einops "nl c nphw -> nl nphw 1 c"
        loc_ = loc.flatten(2).permute(0, 2, 1).unsqueeze(2)  # fastforks: einops 'nl c h w -> nl (h w) 1 c'

        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):  # traverse all local patches
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v, need_weights=False)[0])  # fastforks: need_weights=False

        outputs = torch.cat(outputs, 1)
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)
        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode='nearest')  # freshed glb

        return torch.cat((src, glb), 0), token_attention_map


class BEN_Base(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = swin_v1_b()  # fastforks: SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12)
        emb_dim = 128
        self.sideout5 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8])
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8])

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.InstanceNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.InstanceNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, emb_dim, kernel_size=3, padding=1)
        )

        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        # fastforks: in-place GELU / Dropout flags removed (no effect at inference)

    # fastforks: no inference_mode / fp16 autocast decorators (the runner chooses the dtype); returns the mask logits
    # before the sigmoid
    def forward(self, x):
        real_batch = x.size(0)

        shallow_batch = self.shallow(x)
        glb_batch = rescale_to(x, scale_factor=0.5, interpolation='bilinear')



        inputs = []  # fastforks: one cat instead of a growing one
        for i in range(real_batch):
            loc_batch = image2patches(x[i,:,:,:].unsqueeze(dim=0))
            inputs.append(torch.cat((loc_batch, glb_batch[i,:,:,:].unsqueeze(dim=0)), dim=0))
        final_input = torch.cat(inputs, dim=0)

        features = swin_features(self.backbone, final_input)  # fastforks: self.backbone(final_input)
        outputs = []

        for i in range(real_batch):

            start = i * 5
            end   = (i + 1) * 5

            f4 = features[4][start:end, :, :, :]  # shape: [5, C, H, W]
            f3 = features[3][start:end, :, :, :]
            f2 = features[2][start:end, :, :, :]
            f1 = features[1][start:end, :, :, :]
            f0 = features[0][start:end, :, :, :]
            e5 = self.output5(f4)
            e4 = self.output4(f3)
            e3 = self.output3(f2)
            e2 = self.output2(f1)
            e1 = self.output1(f0)
            loc_e5, glb_e5 = e5.split([4, 1], dim=0)
            e5 = self.multifieldcrossatt(loc_e5, glb_e5)  # (4,128,16,16)


            e4, tokenattmap4 = self.dec_blk4(e4 + resize_as(e5, e4))
            e4 = self.conv4(e4)
            e3, tokenattmap3 = self.dec_blk3(e3 + resize_as(e4, e3))
            e3 = self.conv3(e3)
            e2, tokenattmap2 = self.dec_blk2(e2 + resize_as(e3, e2))
            e2 = self.conv2(e2)
            e1, tokenattmap1 = self.dec_blk1(e1 + resize_as(e2, e1))
            e1 = self.conv1(e1)

            loc_e1, glb_e1 = e1.split([4, 1], dim=0)

            output1_cat = patches2image(loc_e1)  # (1,128,256,256)

            # add glb feat in
            output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
            # merge
            final_output = self.insmask_head(output1_cat)  # (1,128,256,256)
            # shallow feature merge
            shallow = shallow_batch[i,:,:,:].unsqueeze(dim=0)
            final_output = final_output + resize_as(shallow, final_output)
            final_output = self.upsample1(rescale_to(final_output))
            final_output = rescale_to(final_output + resize_as(shallow, final_output))
            final_output = self.upsample2(final_output)
            final_output = self.output(final_output)
            outputs.append(final_output)  # fastforks: logits (upstream: final_output.sigmoid())

        return torch.cat(outputs, dim=0)
