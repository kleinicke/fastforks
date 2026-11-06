import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .dpn import DPN
from .submodule import build_correlation_volume
from .nmp import (
    MLP,
    InferenceLayer,
    Inference,
    RefinementLayer,
    Refinement,
)


class NMRF(nn.Module):
    def __init__(self,
                 backbone,
                 dpn,
                 num_proposals,
                 max_disp,
                 num_infer_layers,
                 num_refine_layers,
                 infer_embed_dim,
                 infer_n_heads,
                 mlp_ratio,
                 window_size,
                 refine_window_size,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout=0.,
                 normalize_before=False,
                 activation="gelu"):
        """fastforks: inference only (with_refinement, compat and no aux loss, as the released models); the backbone
        and DPN are built by the caller (upstream: `configurable` from the yacs config)."""
        super().__init__()
        self.num_proposals = num_proposals
        self.max_disp = max_disp

        feat_dim = backbone.output_dim
        self.concatconv = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        self.gw = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False))

        infer_layers = nn.ModuleList([
            InferenceLayer(
                infer_embed_dim, mlp_ratio=mlp_ratio, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2, n_heads=infer_n_heads,
                activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, dropout=dropout,
                normalize_before=normalize_before
            )
            for i in range(num_infer_layers)]
        )
        infer_norm = nn.LayerNorm(infer_embed_dim)
        self.inference = Inference(32, infer_embed_dim, layers=infer_layers, norm=infer_norm)
        self.infer_head = MLP(infer_embed_dim, infer_embed_dim, 8 * 8, 3)
        self.infer_score_head = nn.Linear(infer_embed_dim, 8 * 8)

        # refinement
        refine_layers = nn.ModuleList([
            RefinementLayer(
                infer_embed_dim, mlp_ratio=mlp_ratio, window_size=refine_window_size,
                shift_size=0 if i % 2 == 0 else refine_window_size // 2, n_heads=infer_n_heads,
                activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, dropout=dropout,
                normalize_before=normalize_before,
            )
            for i in range(num_refine_layers)]
        )
        refine_norm = nn.LayerNorm(infer_embed_dim)
        self.refinement = Refinement(32, infer_embed_dim, layers=refine_layers, norm=refine_norm)
        self.refine_head = MLP(infer_embed_dim, infer_embed_dim, 4 * 4, 3)

        self.dpn = dpn
        self.backbone = backbone
        # fastforks: no `device_indicator_tensor` buffer (the caller moves the inputs) and no weight initialisation

    def extract_feature(self, img1, img2):
        img_batch = torch.cat((img1, img2), dim=0)  # [2B, C, H, W]
        features = self.backbone(img_batch)  # list of [2B, C, H, W], resolution from high to low

        # reverse resolution from low to high
        features = features[::-1]

        # split to list of tuple, res from low to high
        features = [torch.chunk(feature, 2, dim=0) for feature in features]

        feature1, feature2 = map(list, zip(*features))

        return feature1, feature2

    def forward(self, img1, img2):
        """fastforks: test-mode forward of upstream `forward(sample)`: takes images padded to a multiple of 8
        (upstream InputPadder(mode='proposal'): replicate, bottom / right) and returns `disp` [B,H,W] of the padded
        input."""
        fmap1_list, fmap2_list = self.extract_feature(img1, img2)
        cost_volume = build_correlation_volume(fmap1_list[0], fmap2_list[0], self.max_disp // 8, self.dpn.cost_group)  # [B,G,D,H,W]
        labels = self.dpn(cost_volume, fmap1_list)

        # ---- NMRF inference ---- #
        fmap1 = self.concatconv(fmap1_list[0])
        fmap2 = self.concatconv(fmap2_list[0])
        fmap1_gw = self.gw(fmap1_list[0])
        fmap2_gw = self.gw(fmap2_list[0])
        labels_curr = labels[-1].detach()

        tgt = self.inference(labels_curr, fmap1, fmap2, fmap1_gw, fmap2_gw)
        disp_delta = self.infer_head(tgt)  # [num_aux_layers,BHW,N,8*8]
        coarse_disp = F.relu(labels_curr[None].unsqueeze(-1) + disp_delta)
        mask = .25 * self.infer_score_head(tgt)  # [num_aux_layers,BHW,N,8*8]
        bs, _, ht, wd = fmap1.shape
        n = coarse_disp.shape[-2]

        def up8(x):  # fastforks: rearrange 'a (b h w) n (hs ws) -> a b (h hs) (w ws) n' of the last layer, hs=ws=8
            x = x.reshape(bs, ht, wd, n, 8, 8).permute(0, 1, 4, 2, 5, 3)
            return x.reshape(bs, ht * 8, wd * 8, n)

        # refinement
        _, indices = torch.max(up8(mask[-1]), dim=-1, keepdim=True)
        disp_curr = torch.gather(up8(coarse_disp[-1]), dim=-1, index=indices).squeeze(-1) * 2  # [B,H,W]
        # fastforks: rearrange 'b (h hs) (w ws) -> b h w (hs ws)', hs=ws=4
        disp_curr = disp_curr.reshape(bs, ht * 2, 4, wd * 2, 4).permute(0, 1, 3, 2, 4).flatten(-2)
        # fastforks: torch.median (lower median of 16) as sort + index (exports to ONNX)
        disp_curr = torch.sort(disp_curr, dim=-1)[0][..., 7]
        disp_curr = disp_curr.detach()
        fmap1 = self.concatconv(fmap1_list[1])
        fmap2 = self.concatconv(fmap2_list[1])
        fmap1_gw = self.gw(fmap1_list[1])
        fmap2_gw = self.gw(fmap2_list[1])
        tgt = self.refinement(disp_curr, fmap1, fmap2, fmap1_gw, fmap2_gw)
        disp_delta = self.refine_head(tgt)  # [num_aux_layers,BHW,4*4]
        bs, _, ht, wd = fmap1.shape
        disp_delta = disp_delta[-1].reshape(bs, ht, wd, 16)  # fastforks: rearrange 'a (b h w) p -> a b h w p', last
        disp_pred = F.relu(disp_curr.unsqueeze(-1) + disp_delta)
        # fastforks: rearrange 'b h w (hs ws) -> b (h hs) (w ws)', hs=ws=4
        disp_pred = disp_pred.reshape(bs, ht, wd, 4, 4).permute(0, 1, 3, 2, 4).reshape(bs, ht * 4, wd * 4)

        return disp_pred * 4


def build(cfg):
    """fastforks: upstream `NMRF.from_config(cfg)` + `DPN(cfg)` + `create_backbone(cfg)` (ResNet backbone) for a plain
    namespace with upstream's config keys."""
    norm = {"instance": nn.InstanceNorm2d, "batch": nn.BatchNorm2d}[cfg.BACKBONE.NORM_FN]
    backbone = Backbone(cfg.BACKBONE.OUT_CHANNELS, norm)
    nmp = cfg.NMP
    dpn = DPN(
        cost_group=cfg.DPN.COST_GROUP, num_proposals=cfg.DPN.NUM_PROPOSALS, feat_dim=cfg.BACKBONE.OUT_CHANNELS,
        context_dim=cfg.DPN.CONTEXT_DIM, num_prop_layers=nmp.NUM_PROP_LAYERS, prop_embed_dim=nmp.PROP_EMBED_DIM,
        mlp_ratio=nmp.MLP_RATIO, split_size=nmp.SPLIT_SIZE, prop_n_heads=nmp.PROP_N_HEADS,
        normalize_before=nmp.NORMALIZE_BEFORE,
    )
    return NMRF(
        backbone, dpn, num_proposals=cfg.DPN.NUM_PROPOSALS, max_disp=cfg.DPN.MAX_DISP,
        num_infer_layers=nmp.NUM_INFER_LAYERS, num_refine_layers=nmp.NUM_REFINE_LAYERS,
        infer_embed_dim=nmp.INFER_EMBED_DIM, infer_n_heads=nmp.INFER_N_HEADS, mlp_ratio=nmp.MLP_RATIO,
        window_size=nmp.WINDOW_SIZE, refine_window_size=nmp.REFINE_WINDOW_SIZE, normalize_before=nmp.NORMALIZE_BEFORE,
    )
