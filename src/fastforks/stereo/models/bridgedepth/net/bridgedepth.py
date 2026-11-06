from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mono import DepthAnything
from .nmrf import NMRF
from .blocks import Alignment
from .nmp import Head


class BridgeDepth(nn.Module):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        encoder = cfg.BRIDGEDEPTH.ENCODER_VIT
        self.encoder = encoder

        # monocular depth branch
        # fastforks: no `mono_pretrained` download (the BridgeDepth checkpoint holds the Depth Anything V2 weights)
        self.mono = DepthAnything(encoder)
        self.mono.freeze()
        depth_feature_dim = self.mono.feature_dim
        dim = cfg.NMP.INFER_EMBED_DIM
        self.depth_feature_down = nn.Conv2d(depth_feature_dim, dim, kernel_size=4, stride=4, padding=0)

        # stereo branch
        self.stereo_encoder = NMRF(**NMRF.from_config(cfg))  # fastforks: upstream `build_stereo_encoder(cfg)`

        # Alignment
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.align_0 = Alignment(num_blocks=cfg.NMP.NUM_INFER_LAYERS, embed_dim=dim, window_size=cfg.NMP.WINDOW_SIZE,
                                 num_heads=cfg.NMP.INFER_N_HEADS, mlp_ratio=cfg.NMP.MLP_RATIO, drop=cfg.NMP.DROPOUT,
                                 attn_drop=cfg.NMP.ATTN_DROP, drop_path=cfg.NMP.DROP_PATH, norm_layer=norm_layer,
                                 return_intermediate=cfg.NMP.RETURN_INTERMEDIATE, refine=False)
        self.align_1 = Alignment(num_blocks=cfg.NMP.NUM_REFINE_LAYERS, embed_dim=dim, window_size=cfg.NMP.REFINE_WINDOW_SIZE,
                                 num_heads=cfg.NMP.INFER_N_HEADS, mlp_ratio=cfg.NMP.MLP_RATIO, drop=cfg.NMP.DROPOUT,
                                 attn_drop=cfg.NMP.ATTN_DROP, drop_path=cfg.NMP.DROP_PATH, norm_layer=norm_layer,
                                 return_intermediate=cfg.NMP.RETURN_INTERMEDIATE, refine=True)

        # output head
        self.infer_head = Head(dim, dim, 8*8, num_layers=3)
        self.infer_score_head = nn.Linear(dim, 8*8)
        self.refine_head = Head(dim, dim, 4*4, 3)
        self.mono_output = Head(dim, dim, 8*8, num_layers=3)

        # mean and std of the pretrained dinov2 model
        self.register_buffer(
            "mean", torch.as_tensor([123.675, 116.28, 103.53], dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor([58.395, 57.12, 57.375], dtype=torch.float32)[None, :, None, None]
        )

    # fastforks: no `from_pretrained` (hub download, CfgNode unpickling), `postprocessing`, `_set_aux_loss` (training)

    def prepare_input(self, img1, img2):
        # fastforks: the caller pads to a multiple of 16 (upstream InputPadder(mode="nmrf"): replicate, top / right)
        H, W = img1.shape[-2:]
        mono_size = (H // 8 * 7, W // 8 * 7)
        mono_input = F.interpolate(img1, size=mono_size, mode="bilinear", align_corners=False)
        return mono_input.sub_(self.mean).div_(self.std)

    def forward(self, img1, img2):
        """fastforks: test-mode forward of upstream `forward(inputs)`; returns `disp_pred` [B,H,W] of the padded input."""
        image = self.prepare_input(img1, img2)

        depth_prior_raw = self.mono.infer(image)
        depth_prior = self.depth_feature_down(depth_prior_raw).permute(0, 2, 3, 1).contiguous()
        inputs = {"img1": img1, "img2": img2, "depth_prior": depth_prior}

        stereo_outputs = self.stereo_encoder(inputs)
        stereo_embed, stereo_pos_embed = self.stereo_encoder.construct_hypothesis_embedding(
            proposal=stereo_outputs["proposal"].flatten(0, 1).detach(),
            fmap1=stereo_outputs["fmap1_concat_list"][0],
            fmap2=stereo_outputs["fmap2_concat_list"][0],
            fmap1_gw=stereo_outputs["fmap1_gw_list"][0],
            fmap2_gw=stereo_outputs["fmap2_gw_list"][0],
        )

        # align: stage 0
        stereo_embeds, mono_embeds = self.align_0(
            stereo_embed,
            depth_prior,
            stereo_pos_embed
        )
        disp_update = self.infer_head(stereo_embeds)  # [num_aux_layers,B,H,W,N,8*8]
        logit = .25 * self.infer_score_head(stereo_embeds)  # [num_aux_layers,B,H,W,N,8*8]

        # prepare stereo embed for next stage
        a, b, h, w, n = disp_update.shape[:5]
        # fastforks: rearrange 'b (h w) n -> b h w n 1'
        proposal = stereo_outputs["proposal"].detach().unflatten(1, (h, -1)).unsqueeze(-1)
        disp_0 = F.relu(proposal[None] + disp_update)
        # fastforks: rearrange 'a b h w n (hs ws) -> a b (h hs) (w ws) n', hs=ws=8
        disp_0 = disp_0.unflatten(-1, (8, 8)).permute(0, 1, 2, 5, 3, 6, 4).reshape(a, b, h * 8, w * 8, n)
        logit = logit.unflatten(-1, (8, 8)).permute(0, 1, 2, 5, 3, 6, 4).reshape(a, b, h * 8, w * 8, n)
        _, indices = torch.max(logit[-1], dim=-1, keepdim=True)
        disp_est = torch.gather(disp_0[-1], dim=-1, index=indices).squeeze(-1).detach() * 2  # [B,H,W]
        # fastforks: rearrange 'b (h hs) (w ws) -> b h w (hs ws)', hs=ws=4
        disp_est = disp_est.unflatten(2, (-1, 4)).unflatten(1, (-1, 4)).permute(0, 1, 3, 2, 4).flatten(3)
        # disp_est = torch.median(disp_est.detach(), dim=-1, keepdim=True)[0]
        disp_est = torch.sort(disp_est.detach(), dim=-1)[0][..., 7:8]  # replace median for onnx export
        stereo_embed, stereo_pos_embed = self.stereo_encoder.construct_hypothesis_embedding(
            proposal=disp_est.flatten(0, 2),
            fmap1=stereo_outputs["fmap1_concat_list"][1],
            fmap2=stereo_outputs["fmap2_concat_list"][1],
            fmap1_gw=stereo_outputs["fmap1_gw_list"][1],
            fmap2_gw=stereo_outputs["fmap2_gw_list"][1],
            normalizer=128,
        )

        # align: stage 1
        stereo_embeds, mono_embeds_1 = self.align_1(
            stereo_embed,
            mono_embeds[-1],
            stereo_pos_embed
        )
        disp_update = self.refine_head(stereo_embeds.squeeze(-2))  # [num_aux_layers,B,H,W,4*4]
        disp_1 = F.relu(disp_est[None] + disp_update)
        # fastforks: rearrange 'a b h w (hs ws) -> a b (h hs) (w ws)', hs=ws=4
        a, b, h, w = disp_1.shape[:4]
        disp_1 = disp_1.unflatten(-1, (4, 4)).permute(0, 1, 2, 4, 3, 5).reshape(a, b, h * 4, w * 4) * 4
        return disp_1[-1]  # fastforks: upstream results["disp_pred"]
