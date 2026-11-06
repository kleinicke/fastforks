import torch
from torch import nn
import torch.nn.functional as F
from fastforks._core.deferred_init import trunc_normal_  # fastforks: torch.nn.init.trunc_normal_ per call

from .backbone import create_backbone
from .dpn import DPN
from .submodule import build_correlation_volume
from .nmp import fourier_coord_embed, Mlp
from fastforks._core.sampling import grid_sample


class NMRF(nn.Module):
    def __init__(self,
                 backbone,
                 dpn,
                 num_proposals,
                 max_disp,
                 infer_embed_dim):
        """
        aux_loss: True if auxiliary intermediate losses (losses at each encoder/decoder layer)
        """
        super().__init__()
        self.num_proposals = num_proposals
        self.max_disp = max_disp

        feat_dim = backbone.output_dim
        self.concatconv = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        )
        self.gw = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False))

        self.ffn = Mlp(infer_embed_dim+32, infer_embed_dim, infer_embed_dim)
        
        # fastforks: no weight initialisation (`_init_weights`): the checkpoint sets every parameter

        self.dpn = dpn
        self.backbone = backbone

        # to keep track of which device the nn.Module is on
        self.register_buffer("device_indicator_tensor", torch.empty(0))

    @classmethod
    def from_config(cls, cfg):
        # backbone
        backbone = create_backbone(cfg)

        # disparity proposal network
        dpn = DPN(**DPN.from_config(cfg))  # fastforks: upstream `DPN(cfg)` through the detectron2-style `configurable`

        return {
            "backbone": backbone,
            "dpn": dpn,
            "num_proposals": cfg.DPN.NUM_PROPOSALS,
            "max_disp": cfg.DPN.MAX_DISP,
            "infer_embed_dim": cfg.NMP.INFER_EMBED_DIM,
        }
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_tensor.device
    
    @staticmethod
    def sample_fmap(fmap, disp):
        """
        fmap: [B,C,H,W]
        disp: tensor of dim [BHW,num_disp], disparity proposals
        return:
            sampled fmap feature of dim [B,C,H,W,num_disp]
        """
        bs, _, ht, wd = fmap.shape
        num_disp = disp.shape[1]
        device = fmap.device
        with torch.no_grad():
            grid_x = disp.reshape(bs, ht, wd, -1)  # [B,H,W,num_disp]
            grid_y = torch.zeros_like(grid_x)
            xs = torch.arange(0, wd, device=device, dtype=torch.float32).view(1, wd).expand(ht, wd)
            ys = torch.arange(0, ht, device=device, dtype=torch.float32).view(ht, 1).expand(ht, wd)
            grid = torch.stack((xs, ys), dim=-1).reshape(1, ht, wd, 1, 2)
            grid = grid + torch.stack((-grid_x, grid_y), dim=-1)  # [B,H,W,num_disp,2]
            grid[..., 0] = 2 * grid[..., 0].clone() / (wd - 1) - 1
            grid[..., 1] = 2 * grid[..., 1].clone() / (ht - 1) - 1
            grid = grid.reshape(bs, ht, -1, 2)  # [B,H,W*num_disp,2]
        feats = grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return feats.reshape(bs, -1, ht, wd, num_disp)
    
    def corr(self, fmap1, warped_fmap2, num_disp):
        """
        fmap1: [B,C,H,W]
        warped_fmap2: [B,C,H,W,num_disp]
        Returns:
            local cost: [B,H,W,num_disp,G]
        """
        fmap1 = fmap1.unflatten(1, (32, -1))  # fastforks: rearrange 'b (g d) h w -> b g d h w', g=32
        warped_fmap2 = warped_fmap2.unflatten(1, (32, -1))
        corr = (fmap1.unsqueeze(-1) * warped_fmap2).mean(dim=2)  # [B,G,H,W,num_disp]
        corr = corr.permute(0, 2, 3, 4, 1)  # fastforks: rearrange 'b g h w n -> b h w n g'
        return corr
    
    def construct_hypothesis_embedding(self, proposal, fmap1, fmap2, fmap1_gw, fmap2_gw, normalizer=64):
        H, W, N = fmap1.shape[2], fmap1.shape[3], proposal.shape[-1]
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, proposal)  # [B,C,H,W,N]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, N)  # [B,H,W,N,G]
        warped_fmap2 = self.sample_fmap(fmap2, proposal)  # [B,C,H,W,N]
        fmap1 = fmap1.unsqueeze(-1).expand(-1, -1, -1, -1, N)  # fastforks: repeat 'b c h w -> b c h w n'
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = feat_concat.permute(0, 2, 3, 4, 1)  # fastforks: rearrange 'b c h w n -> b h w n c'
        x = self.ffn(torch.cat((feat_concat, corr), dim=-1))
        proposal = proposal.unflatten(0, (-1, H, W))  # fastforks: rearrange '(b h w) n -> b h w n'
        pos_embed = fourier_coord_embed(proposal.unsqueeze(-1), N_freqs=15, normalizer=3.14 / normalizer)
        return x, pos_embed

    def extract_feature(self, img1, img2):
        img_batch = torch.cat((img1, img2), dim=0)  # [2B,C,H,W]
        features = self.backbone(img_batch)

        # reverse resolution from low to high
        features = features[::-1]

        # split to list of tuple, res from low to high
        features = [torch.chunk(feature, 2, dim=0) for feature in features]

        feature1, feature2 = map(list, zip(*features))

        return feature1, feature2
    
    def forward(self, input):
        """
        It returns a dict with the following elements:
            - "proposals": disparity proposals, tensor of dim [n, B*H/2*W/2, num_proposals]
            - "prob": disparity candidate probability, tensor of dim [B*H/2*W/2, W/2]
            - "disp": disparity prediction, tensor of dim [B, H, W]
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                             dictionaries containing the four above keys for each intermediate layer.
        """
        img1 = input["img1"]
        img2 = input["img2"]
        
        # We assume the input padding is not needed during training by setting adequate crop size
        fmap1_list, fmap2_list = self.extract_feature(img1, img2)

        # proposal extraction
        depth_prior = input['depth_prior']
        cost_volume = build_correlation_volume(fmap1_list[0], fmap2_list[0], self.max_disp // 8, self.dpn.cost_group)  # [B,G,D,H,W]
        prob, labels = self.dpn(cost_volume, fmap1_list[0], depth_prior)

        # construct hypothesis embedding
        fmap1_concat_list = [self.concatconv(x) for x in fmap1_list]
        fmap2_concat_list = [self.concatconv(x) for x in fmap2_list]
        fmap1_gw_list = [self.gw(x) for x in fmap1_list]
        fmap2_gw_list = [self.gw(x) for x in fmap2_list]

        proposal = labels.reshape(img1.shape[0], -1, self.num_proposals)
        out = {
            "prob": prob,
            "proposal": proposal,
            "fmap1_concat_list": fmap1_concat_list,
            "fmap2_concat_list": fmap2_concat_list,
            "fmap1_gw_list": fmap1_gw_list,
            "fmap2_gw_list": fmap2_gw_list,
        }
        
        return out


# fastforks: the training criterion (`Criterion`, `build_criterion`) is not vendored
