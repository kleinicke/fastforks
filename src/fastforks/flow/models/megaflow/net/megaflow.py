import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
from functools import partial

from .layers import PatchEmbed
from .layers.block import Block
from .layers.rope import RotaryPositionEmbedding2D, PositionGetter
from .flow_head import FlowFeature
from .layers.attention import MemEffAttention

from .refine import ResNetFPN, RAFTUpdateBlock
from .matching import global_correlation_softmax, local_correlation_with_flow
from .model_utils import upsample_flow_with_mask, SelfAttnPropagation


logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class MegaFlow(nn.Module):
    """
    MegaFlow: Zero-Shot Large Displacement Optical Flow.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    # HuggingFace model variants
    _HF_MODELS = {
        "megaflow-flow": {"repo_id": "Kristen-Z/MegaFlow", "filename": "megaflow-flow.safetensors"},
        "megaflow-track": {"repo_id": "Kristen-Z/MegaFlow", "filename": "megaflow-track.safetensors"},
        "megaflow-chairs-things": {"repo_id": "Kristen-Z/MegaFlow", "filename": "megaflow-chairs-things.safetensors"},
    }

    # fastforks: upstream from_pretrained (huggingface_hub download) removed; fastforks loads the pinned checkpoint.
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        freeze_embed=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_self_attn_propagation=False, # use self-attn propagation for feature flow
        feature_channels=128,
        upsample_factor=4,
        cnn_blocks=2,
        reg_refine=True,  # local regression refinement
        fix_width=False,
        fuse_cnn=True,
        seq_len=16,
        use_temporal_attn=True,
    ):
        super().__init__()

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the register tokens
        self.patch_start_idx = num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

        self.use_reentrant = False # hardcoded to False

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim, freeze_embed=freeze_embed, depth=depth)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.flow_head = FlowFeature(dim_in=2*embed_dim,patch_size=patch_size, features=feature_channels, fuse_cnn=fuse_cnn, num_layers=depth) #, down_ratio=upsample_factor)
        self.seq_len = seq_len # window size for tracking
        self.fix_width = fix_width

        ######### Initialize unimatch parameters
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.refine_factor = upsample_factor
        self.use_self_attn_propagation = use_self_attn_propagation
        self.fuse_cnn = fuse_cnn

        # propagation with self-attn
        if self.use_self_attn_propagation:
            self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine:
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        else:
            # fastforks: init_weight=False (upstream True: a torchvision ResNet34 download the checkpoint overwrites)
            self.backbone = ResNetFPN(input_dim=3, output_dim=feature_channels, init_weight=False, downsample=self.refine_factor, num_blocks=cnn_blocks, return_all_feat=self.fuse_cnn)
            self.refine_proj = nn.Conv2d(feature_channels, feature_channels*2, 1)
            self.refine = RAFTUpdateBlock(corr_channels=(2 * 4 + 1) ** 2, downsample_factor=self.refine_factor, use_temporal_attn=use_temporal_attn) # 9x9 local corr

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        freeze_embed=True,
        depth=24,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        elif "dinov2" in patch_embed:
            from .layers.vision_transformer import DinoVisionTransformer
            assert patch_size == 14, "DINOv2 models use patch size 14"

            self.patch_embed = DinoVisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=1024,
                depth=depth,
                num_heads=16,
                mlp_ratio=4,
                block_fn=partial(Block, attn_class=MemEffAttention),
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values
            )

            # Disable gradient updates
            if freeze_embed:
                for p in self.patch_embed.parameters():
                    p.requires_grad = False

        elif "dinov3" in patch_embed:
            print("Loading DINOv3 as frozen backbone...")
            assert patch_size == 16, "DINOv3 models use patch size 16"
            dino_model = torch.hub.load("./dinov3", "dinov3_vitl16", source='local', weights='')

            for p in dino_model.parameters():
                p.requires_grad = False
            dino_model.eval()
            dino_model.forward = dino_model.forward_features  
            self.patch_embed = dino_model

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8, is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow.to(flow.dtype)
    
    def resize_flow_bilinear(self, flow, ori_h, ori_w, new_h, new_w=518):
        """
        Resize flow back to original resolution.
        Handles both 4D (B, 2, H, W) and 5D (B, T, 2, H, W) inputs.
        """
        # scale factors
        scale_x = ori_w / new_w
        scale_y = ori_h / new_h

        is_5d = flow.ndim == 5
        if is_5d:
            B, T, C, H, W = flow.shape
            # Fold T into B: (B, T, 2, H, W) -> (B*T, 2, H, W)
            flow = flow.view(B * T, C, H, W)

        # resize flow field
        flow = F.interpolate(flow, size=(ori_h, ori_w), mode='bilinear', align_corners=True)

        # rescale flow vectors
        flow[:, 0] *= scale_x  # dx
        flow[:, 1] *= scale_y  # dy

        # --- Start: Reshape back if needed ---
        if is_5d:
            flow = flow.view(B, T, C, ori_h, ori_w)

        return flow

    def extract_feature(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        # images = (images / 255 - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand register tokens to match batch size and sequence length
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list

    # fastforks: upstream forward_track / forward_track_sliding / get_T_padded_images (multi-frame tracking) removed.
    def forward(self, imgs, num_reg_refine=1):
        results_dict = {}
        # fastforks: `imgs` arrive normalised and padded (the runner's card [io]: ImageNet mean / std, InputPadderMF's
        # multiple 8, centred, replicate); upstream normalises `imgs / 255` and pads here, then unpads the predictions.
        B, T, C, H, W = imgs.shape

        _, _, _, H_pad, W_pad = imgs.shape

        if self.fix_width:
            # keep fix width at global matching
            resize_w = 952 if self.patch_size==14 else 960
            resize_h = round(H_pad * (resize_w / W_pad) / self.patch_size) * self.patch_size 
        else:
            resize_h, resize_w = H_pad // self.patch_size * self.patch_size, W_pad // self.patch_size * self.patch_size

        resize_input = F.interpolate(
            imgs.view(B * T, C, H_pad, W_pad), size=(resize_h, resize_w), mode="bilinear", align_corners=True
        ).view(B, T, C, resize_h, resize_w)

        aggregated_tokens_list = self.extract_feature(resize_input)
        if self.reg_refine: 
            cnn_features = self.backbone(imgs.view(B*T, C, H_pad, W_pad))

        features = self.flow_head(
            aggregated_tokens_list, images=resize_input, patch_start_idx=self.patch_start_idx, cnn_features=cnn_features
        ) # [B, T, C, H_feat, W_feat]

        del resize_input, aggregated_tokens_list
        if self.reg_refine:
            if self.fuse_cnn:
                cnn_features = cnn_features[-1]
            cnn_features = cnn_features.view(B, T, *cnn_features.shape[1:]).contiguous()
        else:
            cnn_features = None

        flows_init, flows_preds = [], []
        for t in range(T-1):
            
            feature0 = features[:, t]
            feature1 = features[:, t+1]
            
            flow, prob = global_correlation_softmax(feature0, feature1)
            if self.use_self_attn_propagation:
                flow = self.feature_flow_attn(feature0, flow.detach())
            flows_init.append(flow)
        
        flows_init = torch.stack(flows_init, dim=1).contiguous() # B, T-1, 2, H_feat, W_feat

        if self.training:
            # Use H_pad, W_pad before unpadding
            flow_bilinear = self.resize_flow_bilinear(flows_init, H_pad, W_pad, new_h=flows_init.shape[-2], new_w=flows_init.shape[-1])
            flows_preds.append(flow_bilinear)

        if self.reg_refine:
            flows = self.resize_flow_bilinear(flows_init, H_pad // self.refine_factor, W_pad // self.refine_factor, new_h=flows_init.shape[-2], new_w=flows_init.shape[-1]).contiguous()
            refine_feature = custom_interpolate(features[:, :-1].reshape(B*(T-1), *features.shape[2:]), size=flows.shape[-2:], mode='bilinear', align_corners=True)
            proj = self.refine_proj(refine_feature)  # [B*(T-1), 2C, h, w]
            net, inp = torch.chunk(proj, chunks=2, dim=1)

            del refine_feature, proj
            torch.cuda.empty_cache()

            for refine_iter_idx in range(num_reg_refine):
                flows = flows.detach() # B, T-1, 2, H, W
                corrs = []
                for t in range(T-1):
                    correlation = local_correlation_with_flow(
                        cnn_features[:, t],
                        cnn_features[:, t+1],
                        flow=flows[:, t],
                        local_radius=4,
                    )  # [B, (2R+1)^2, H, W]

                    corrs.append(correlation)
                
                corrs = torch.stack(corrs, dim=1).view(-1, *correlation.shape[1:])

                if T < 6:
                    net, up_masks, residual_flows, _ = self.refine(net, inp, corrs, flows.view(-1, *flows.shape[2:]), batch_size=B)
                else:
                    net, up_masks, residual_flows, _ = checkpoint(
                        self.refine, 
                        net, 
                        inp, 
                        corrs, 
                        flows.view(-1, *flows.shape[2:]),
                        None,
                        B,
                        use_reentrant=False
                    )

                flows = flows + residual_flows.view(flows.shape)

                if self.training or refine_iter_idx == num_reg_refine - 1:
                    flow_up = upsample_flow_with_mask(flows.view(-1, *flows.shape[2:]), up_masks, upsample_factor=self.refine_factor,)
                    flows_preds.append(flow_up.view(B, T-1, *flow_up.shape[1:]))
                
                del corrs, residual_flows, up_masks, correlation
                torch.cuda.empty_cache()

        results_dict.update({'flow_init': flows_init})
        results_dict.update({'flow_preds': flows_preds})

        return results_dict

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates
        
def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
