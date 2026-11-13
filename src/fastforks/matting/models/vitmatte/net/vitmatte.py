# Copyright 2023 HUST-VL, Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ViTMatte = transformers v5.19.0 `models/vitdet/modeling_vitdet.py` + `models/vitmatte/modeling_vitmatte.py`.

fastforks changes: the `PreTrainedModel` / config / output-dataclass machinery is replaced by `Config` (the fields the
hustvl/vitmatte-*-composition-1k configs set or default); training-only parts (drop path, dropout, loss, gradient
checkpointing, attention outputs) and the `torch.jit` tracing switches are dropped. The eval maths and the parameter
names are transformers' (the checkpoints load strictly). `VitMatteForImageMatting.forward` returns the alphas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    """The fields of `VitDetConfig` / `VitMatteConfig` the network reads (hustvl checkpoints: S = 384 / 6, B = 768 / 12)."""

    hidden_size: int = 384
    num_attention_heads: int = 6
    num_hidden_layers: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-6
    image_size: int = 512
    pretrain_image_size: int = 224
    patch_size: int = 16
    num_channels: int = 4
    qkv_bias: bool = True
    window_block_indices: tuple[int, ...] = (0, 1, 3, 4, 6, 7, 9, 10)
    residual_block_indices: tuple[int, ...] = (2, 5, 8, 11)
    window_size: int = 14
    batch_norm_eps: float = 1e-5
    convstream_hidden_sizes: tuple[int, ...] = (48, 96, 192)
    fusion_hidden_sizes: tuple[int, ...] = (256, 128, 64, 32)


class VitDetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.pretrain_image_size, config.patch_size
        num_patches = (image_size // patch_size) ** 2
        self.num_channels = config.num_channels
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def get_absolute_positions(self, abs_pos_embeddings, has_cls_token, height, width):
        if has_cls_token:
            abs_pos_embeddings = abs_pos_embeddings[:, 1:]
        num_position = abs_pos_embeddings.shape[1]
        size = int(math.sqrt(num_position))
        if size * size != num_position:
            raise ValueError("Absolute position embeddings must be a square number.")

        if size != height or size != width:
            new_abs_pos_embeddings = nn.functional.interpolate(
                abs_pos_embeddings.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )

            return new_abs_pos_embeddings.permute(0, 2, 3, 1)
        else:
            return abs_pos_embeddings.reshape(1, height, width, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = embeddings + self.get_absolute_positions(
            self.position_embeddings, True, embeddings.shape[1], embeddings.shape[2]
        )
        embeddings = embeddings.permute(0, 3, 1, 2)
        return embeddings


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_relative_positions(attn, queries, rel_pos_h, rel_pos_w, q_size, k_size):
    queries_height, queries_width = q_size
    keys_height, keys_width = k_size
    relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
    relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)

    batch_size, _, dim = queries.shape
    r_q = queries.reshape(batch_size, queries_height, queries_width, dim)
    relative_height = torch.einsum("bhwc,hkc->bhwk", r_q, relative_height)
    relative_weight = torch.einsum("bhwc,wkc->bhwk", r_q, relative_width)

    attn = (
        attn.view(batch_size, queries_height, queries_width, keys_height, keys_width)
        + relative_height[:, :, :, :, None]
        + relative_weight[:, :, :, None, :]
    ).view(batch_size, queries_height * queries_width, keys_height * keys_width)

    return attn


class VitDetAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, input_size=None):
        super().__init__()

        dim = config.hidden_size
        num_heads = config.num_attention_heads

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_state):
        batch_size, height, width, _ = hidden_state.shape
        qkv = self.qkv(hidden_state).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.reshape(3, batch_size * self.num_heads, height * width, -1).unbind(0)

        attention_scores = (queries * self.scale) @ keys.transpose(-2, -1)

        attention_scores = add_decomposed_relative_positions(
            attention_scores, queries, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
        )

        attention_probs = attention_scores.softmax(dim=-1)

        hidden_state = attention_probs @ values
        hidden_state = hidden_state.view(batch_size, self.num_heads, height, width, -1)
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4)
        hidden_state = hidden_state.reshape(batch_size, height, width, -1)
        hidden_state = self.proj(hidden_state)

        return hidden_state


class VitDetLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of `(batch_size, channels, height, width)` inputs."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class VitDetResBottleneckBlock(nn.Module):
    """1x1, 3x3, 1x1 convs with LayerNorms, residual, no last activation."""

    def __init__(self, config, in_channels, out_channels, bottleneck_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = VitDetLayerNorm(bottleneck_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.norm2 = VitDetLayerNorm(bottleneck_channels)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = VitDetLayerNorm(out_channels)

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class VitDetMlp(nn.Module):
    def __init__(self, config, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


def window_partition(hidden_state, window_size):
    batch_size, height, width, num_channels = hidden_state.shape

    pad_height = (-height) % window_size
    pad_width = (-width) % window_size
    hidden_state = nn.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width

    n_h = padded_height // window_size
    n_w = padded_width // window_size
    hidden_state = hidden_state.view(batch_size, n_h, window_size, n_w, window_size, num_channels)
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows, (padded_height, padded_width)


def window_unpartition(windows, window_size, pad_height_width, height_width):
    padded_height, padded_width = pad_height_width
    height, width = height_width
    n_h = padded_height // window_size
    n_w = padded_width // window_size
    batch_size = windows.shape[0] // (n_h * n_w)
    hidden_state = windows.view(batch_size, n_h, n_w, window_size, window_size, -1)
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
    hidden_state = hidden_state.view(batch_size, padded_height, padded_width, -1)
    return hidden_state[:, :height, :width, :].contiguous()


class VitDetLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Config, window_size: int = 0, use_residual_block: bool = False) -> None:
        super().__init__()

        dim = config.hidden_size
        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = VitDetAttention(
            config, input_size=input_size if window_size == 0 else (window_size, window_size)
        )

        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = VitDetMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if self.use_residual_block:
            self.residual = VitDetResBottleneckBlock(
                config=config,
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 3, 1)

        shortcut = hidden_states

        hidden_states = self.norm1(hidden_states)

        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)

        hidden_states = self.attention(hidden_states)

        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))

        hidden_states = shortcut + hidden_states

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        hidden_states = hidden_states.permute(0, 3, 1, 2)

        if self.use_residual_block:
            hidden_states = self.residual(hidden_states)

        return hidden_states


class VitDetEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            VitDetLayer(
                config,
                window_size=config.window_size if i in config.window_block_indices else 0,
                use_residual_block=i in config.residual_block_indices,
            )
            for i in range(config.num_hidden_layers)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class VitDetBackbone(nn.Module):
    """`VitDetBackbone` with `out_features = ["stage12"]`: the last layer's feature map."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embeddings = VitDetEmbeddings(config)
        self.encoder = VitDetEncoder(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embeddings(pixel_values))


class VitMatteBasicConv3x3(nn.Module):
    """Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers."""

    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        self.relu = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.relu(hidden_state)

        return hidden_state


class VitMatteConvStream(nn.Module):
    """Simple ConvStream containing a series of basic conv3x3 layers to extract detail features."""

    def __init__(self, config):
        super().__init__()

        in_channels = config.num_channels

        out_channels = list(config.convstream_hidden_sizes)

        self.convs = nn.ModuleList()
        self.conv_chans = [in_channels, *out_channels]

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(VitMatteBasicConv3x3(config, in_chan_, out_chan_))

    def forward(self, pixel_values):
        out_dict = {"detailed_feature_map_0": pixel_values}
        embeddings = pixel_values
        for i in range(len(self.convs)):
            embeddings = self.convs[i](embeddings)
            name_ = "detailed_feature_map_" + str(i + 1)
            out_dict[name_] = embeddings

        return out_dict


class VitMatteFusionBlock(nn.Module):
    """Simple fusion block to fuse features from ConvStream and Plain Vision Transformer."""

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.conv = VitMatteBasicConv3x3(config, in_channels, out_channels, stride=1, padding=1)

    def forward(self, features, detailed_feature_map):
        upscaled_features = nn.functional.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.cat([detailed_feature_map, upscaled_features], dim=1)
        out = self.conv(out)

        return out


class VitMatteHead(nn.Module):
    """Simple Matting Head, containing only conv3x3 and conv1x1 layers."""

    def __init__(self, config):
        super().__init__()

        in_channels = config.fusion_hidden_sizes[-1]
        mid_channels = 16

        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, hidden_state):
        hidden_state = self.matting_convs(hidden_state)

        return hidden_state


class VitMatteDetailCaptureModule(nn.Module):
    """Simple and lightweight Detail Capture Module for ViT Matting."""

    def __init__(self, config):
        super().__init__()
        if len(config.fusion_hidden_sizes) != len(config.convstream_hidden_sizes) + 1:
            raise ValueError(
                "The length of fusion_hidden_sizes should be equal to the length of convstream_hidden_sizes + 1."
            )

        self.config = config
        self.convstream = VitMatteConvStream(config)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blocks = nn.ModuleList()
        self.fusion_channels = [config.hidden_size, *config.fusion_hidden_sizes]

        for i in range(len(self.fusion_channels) - 1):
            self.fusion_blocks.append(
                VitMatteFusionBlock(
                    config=config,
                    in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                )
            )

        self.matting_head = VitMatteHead(config)

    def forward(self, features, pixel_values):
        detail_features = self.convstream(pixel_values)
        for i in range(len(self.fusion_blocks)):
            detailed_feature_map_name = "detailed_feature_map_" + str(len(self.fusion_blocks) - i - 1)
            features = self.fusion_blocks[i](features, detail_features[detailed_feature_map_name])

        alphas = torch.sigmoid(self.matting_head(features))

        return alphas


class VitMatteForImageMatting(nn.Module):
    """ViTMatte: `[B, 4, H, W]` (normalised RGB + trimap / 255, H and W multiples of 32) -> alphas `[B, 1, H, W]`."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.backbone = VitDetBackbone(config)
        self.decoder = VitMatteDetailCaptureModule(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values)
        return self.decoder(features, pixel_values)
