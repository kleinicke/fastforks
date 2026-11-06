# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monocular Depth Network Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastforks.shared.depth_anything_v2 import DINOv2, DPTHead


class RelativeDepthAnythingV2(nn.Module):
    """ Relative depth prediction model using DPT Head. """

    def __init__(
        self,
        model_config, max_depth, export=False
    ):
        """
        Initialize RelativeDepthAnythingV2.

        Args:
            model_config (dict): Model configuration dictionary.
            max_depth (float): Maximum depth value for normalization.
            export (bool, optional): Whether the model is being used for export.
                Defaults to False.
        """
        super(RelativeDepthAnythingV2, self).__init__()
        encoder = model_config['encoder']
        use_bn = model_config['mono_backbone']['use_bn']
        use_clstoken = model_config['mono_backbone']['use_clstoken']
        self.encoder = encoder
        self.max_depth = max_depth

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'intermediate_layer_idx': [2, 5, 8, 11]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'intermediate_layer_idx': [2, 5, 8, 11]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'intermediate_layer_idx': [4, 11, 17, 23]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'intermediate_layer_idx': [9, 19, 29, 39]}
        }

        features = self.model_configs[self.encoder]['features']
        out_channels = self.model_configs[self.encoder]['out_channels']

        self.pretrained = DINOv2(encoder)

        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        """
        Forward pass for relative depth prediction.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted relative depth map of shape (B, H, W).
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x.contiguous(), self.model_configs[self.encoder]['intermediate_layer_idx'], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        return depth.squeeze(1)

