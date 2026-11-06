# fastforks: timm's BEiT (`timm/models/beit.py`, Apache-2.0, see LICENSE-timm) re-implemented without timm, with the
# module / parameter names of `timm.create_model("beit_large_patch16_{384,512}")`, and MiDaS's arbitrary-window-size
# patches (`midas/backbones/beit.py`, MIT, see LICENSE: patch_embed_forward, _get_rel_pos_bias, attention_forward,
# block_forward, beit_forward_features) folded into the classes. Inference only: no dropout / drop path (identity in
# eval), no absolute position embedding / shared relative position bias (both off in the BEiT-L configs), no classifier
# (`fc_norm` / `head` are unused by MiDaS). `forward_features` returns the outputs of the hooked blocks (upstream
# registers forward hooks on them and reads a module-global dict).
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_relative_position_index(window_size: Tuple[int, int], device=None) -> torch.Tensor:
    # = timm.models.beit.gen_relative_position_index (ndgrid = meshgrid(indexing="ij"))
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(torch.meshgrid(
        torch.arange(window_size[0], device=device, dtype=torch.long),
        torch.arange(window_size[1], device=device, dtype=torch.long),
        indexing="ij",
    ))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, device=device, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class PatchEmbed(nn.Module):
    # timm PatchEmbed with MiDaS patch_embed_forward (any input size; flatten, no norm)
    def __init__(self, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.register_buffer("k_bias", torch.zeros(all_head_dim), persistent=False)
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim))

        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
        self.relative_position_indices = {}  # MiDaS: window size -> index (generated on first use)

        self.proj = nn.Linear(all_head_dim, dim)

    def _get_rel_pos_bias(self, window_size):
        """
        Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
        """
        old_height = 2 * self.window_size[0] - 1
        old_width = 2 * self.window_size[1] - 1

        new_height = 2 * window_size[0] - 1
        new_width = 2 * window_size[1] - 1

        old_relative_position_bias_table = self.relative_position_bias_table

        old_num_relative_distance = self.num_relative_distance
        new_num_relative_distance = new_height * new_width + 3

        old_sub_table = old_relative_position_bias_table[:old_num_relative_distance - 3]

        old_sub_table = old_sub_table.reshape(1, old_width, old_height, -1).permute(0, 3, 1, 2)
        new_sub_table = F.interpolate(old_sub_table, size=(new_height, new_width), mode="bilinear")
        new_sub_table = new_sub_table.permute(0, 2, 3, 1).reshape(new_num_relative_distance - 3, -1)

        new_relative_position_bias_table = torch.cat(
            [new_sub_table, old_relative_position_bias_table[old_num_relative_distance - 3:]])

        key = str(window_size[1]) + "," + str(window_size[0])
        # fastforks: the index is generated on the table's device (upstream: on the CPU, copied on every call)
        if key not in self.relative_position_indices.keys() or \
                self.relative_position_indices[key].device != old_relative_position_bias_table.device:
            self.relative_position_indices[key] = gen_relative_position_index(
                window_size, device=old_relative_position_bias_table.device)

        relative_position_bias = new_relative_position_bias_table[
            self.relative_position_indices[key].view(-1)].view(
            window_size[0] * window_size[1] + 1,
            window_size[0] * window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, resolution):
        """
        Modification of timm.models.beit.py: Attention.forward to support arbitrary window sizes.
        """
        B, N, C = x.shape

        qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            window_size = tuple(np.array(resolution) // 16)
            attn = attn + self._get_rel_pos_bias(window_size)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, init_values, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, resolution):
        """
        Modification of timm.models.beit.py: Block.forward to support arbitrary window sizes.
        """
        x = x + self.gamma_1 * self.attn(self.norm1(x), resolution)
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class Beit(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 init_values=1e-5):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        window_size = (img_size // patch_size, img_size // patch_size)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, init_values, window_size) for _ in range(depth)])
        self.patch_size = [patch_size, patch_size]

    def forward_features(self, x, hooks):
        """
        Modification of timm.models.beit.py: Beit.forward_features to support arbitrary window sizes.

        fastforks: returns the outputs of blocks `hooks` (MiDaS's forward hooks) and stops after the last one.
        """
        resolution = x.shape[2:]

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        out = []
        for i, blk in enumerate(self.blocks[:hooks[-1] + 1]):
            x = blk(x, resolution)
            if i in hooks:
                out.append(x)
        return out
