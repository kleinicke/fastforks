"""Vendored WAFT-Stereo feature encoder (model/encoder/dav2.py + the DAv2 branch of model/encoder/__init__.py, c534320)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# fastforks: peft -> a small local LoRA (_lora.py: same module tree and parameter names); the unmodified DINOv2 ViT comes
# from the shared Depth Anything V2 module; unused imports (timm, numpy, torchvision, sys, os) dropped
from ._lora import LoraConfig, get_peft_model
from fastforks.shared.depth_anything_v2 import DINOv2
from .dpt import UpsampleFeats, ProjFeats

DEPTH_ANYTHING_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
class DAv2Encoder(nn.Module):
    def __init__(self, model_name='vits', alpha=None, r=None):
        super().__init__()
        self.model_name = model_name
        # fastforks: only the ViT is used (the DPT depth head never runs and has no weights in the checkpoint), and the
        # DAv2 weights are part of the WAFT checkpoint, so no depth-anything-ckpts/ lookup
        depth_anything = nn.Module()
        depth_anything.pretrained = DINOv2(model_name=model_name)
        self.dpt_configs = {
            'vitl': {'n_layers': 24, "dim": 1024},
            'vitb': {'n_layers': 12, "dim": 768},
            'vits': {'n_layers': 12, "dim": 384},
        }
        self.dim = self.dpt_configs[model_name]['dim']
        self.idx = [(i + 1) * self.dpt_configs[model_name]['n_layers'] // 4 - 1 for i in range(4)]
        self.out_c = [(int)(self.dim * (2 ** (i - 3))) for i in range(4)]
        self.output_dim = self.out_c[0]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv", "proj"]
        )
        self.encoder = get_peft_model(depth_anything.pretrained, lora_config)
        self.fmap_proj = ProjFeats(self.dim, self.out_c, lvl=-3)
        self.fmap_upsample = UpsampleFeats(self.output_dim, self.out_c)
        self.hidden_proj = ProjFeats(self.dim*2, self.out_c, lvl=-3)
        self.hidden_upsample = UpsampleFeats(self.output_dim, self.out_c) 

    def forward(self, imgs):
        B, N, _, H, W = imgs.shape
        # encode by dinov2
        h, w = H // 16, W // 16
        imgs = imgs.reshape(B*N, _, H, W)
        
        imgs = F.interpolate(imgs, (h*14, w*14), mode='bilinear', align_corners=True)
        feats = self.encoder.get_intermediate_layers(imgs, self.idx, return_class_token=True)
        # fastforks: einops.rearrange as view / permute
        fmap_feats = [x[0].reshape(B*N, h, w, -1).permute(0, 3, 1, 2) for x in feats]
        fmap_feats = self.fmap_proj(fmap_feats)
        fmap_feats = self.fmap_upsample(fmap_feats)
        fmaps = fmap_feats[0].reshape(B, 2, *fmap_feats[0].shape[1:])
        fmap1 = fmaps[:, 0]
        fmap2 = fmaps[:, 1]

        hidden_feats = [x[0].reshape(B, 2, h, w, -1).permute(0, 1, 4, 2, 3).reshape(B, -1, h, w) for x in feats]
        hidden_feats = self.hidden_proj(hidden_feats)
        hidden_feats = self.hidden_upsample(hidden_feats)
        hidden = hidden_feats[0]

        return fmap1, fmap2, hidden


def fetch_feature_encoder(cfg):
    # fastforks: model/encoder/__init__.py; the dinov3 branch is not vendored (no published checkpoint uses it)
    if cfg.TYPE == 'dav2':
        factor = 16
        encoder_lora_rank = cfg.LORA_RANK
        encoder_lora_alpha = cfg.LORA_ALPHA
        encoder = DAv2Encoder(
            model_name=cfg.ARCH,
            alpha=encoder_lora_alpha,
            r=encoder_lora_rank,
        )
        encoder_dim = encoder.output_dim
    else:
        raise ValueError(f"Unknown feature encoder: {cfg.TYPE}")

    return encoder, encoder_dim, factor
