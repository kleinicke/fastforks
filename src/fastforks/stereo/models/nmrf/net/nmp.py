from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from fastforks._core.sampling import grid_sample


# fastforks: einops' rearrange / repeat as view / permute / expand (the runtime has no einops); shared helpers:
def windows(x, hs, ws, heads):
    """rearrange(x, 'b (i hs) (j ws) n (h d) -> (b i j) h (hs ws n) d', hs=hs, ws=ws, h=heads)"""
    B, H, W, N, C = x.shape
    x = x.reshape(B, H // hs, hs, W // ws, ws, N, heads, C // heads)
    return x.permute(0, 1, 3, 6, 2, 4, 5, 7).reshape(-1, heads, hs * ws * N, C // heads)


def unwindows(x, i, j, hs, ws):
    """rearrange(x, '(b i j) h (hs ws n) d -> b (i hs) (j ws) n (h d)', i=i, j=j, hs=hs, ws=ws)"""
    BIJ, h, L, d = x.shape
    x = x.reshape(BIJ // (i * j), i, j, h, hs, ws, L // (hs * ws), d)
    return x.permute(0, 1, 4, 2, 5, 6, 3, 7).reshape(BIJ // (i * j), i * hs, j * ws, L // (hs * ws), h * d)


def mask_windows_of(img_mask, hs, ws):
    """rearrange(img_mask, 'b (h hs) (w ws) c -> (b h w) (hs ws) c', hs=hs, ws=ws)"""
    b, H, W, c = img_mask.shape
    x = img_mask.reshape(b, H // hs, hs, W // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, hs * ws, c)


def cached_mask(cache, key, make):
    """fastforks: attention masks kept per module and key (shape, device) instead of rebuilt in every call (our fork's
    `fast.py` mask cache); the last few keys only, built afresh while compiling / exporting."""
    if torch.compiler.is_compiling():
        return make()
    mask = cache.get(key)
    if mask is None:
        if len(cache) >= 4:
            cache.clear()
        mask = cache[key] = make()
    return mask


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def fourier_coord_embed(coord, N_freqs, normalizer=3.14/512, logscale=True):
    """
    coord: [...]D
    returns:
        [...]dim, where dim=(2*N_freqs+1)*D
    """
    if logscale:
        freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs, device=coord.device)
    else:
        freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    coord = coord.unsqueeze(-1) * normalizer
    freq_bands = freq_bands[(*((None,) * (len(coord.shape) - 1)), Ellipsis)]
    f_coord = coord * freq_bands
    embed = torch.cat([f_coord.sin(), f_coord.cos(), coord], dim=-1)
    embed = embed.flatten(-2)  # fastforks: rearrange '... d n -> ... (d n)'

    return embed


class MLP(nn.Module):
    """ Very simple multi-layer perception (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _heads(x, h):
    """rearrange(x, 'b n (h d) -> b h n d', h=h)"""
    return x.unflatten(-1, (h, -1)).transpose(1, 2)


# Neural Message Passing along self edges of the NMRF graph.
class BasicAttention(nn.Module):
    """
    label representation:  [B, N, C]
    """
    def __init__(self, dim, qk_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0., normalize_before=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q, self.k, self.v = nn.Linear(qk_dim, dim, bias=True), nn.Linear(qk_dim, dim, bias=True), nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.normalize_before = normalize_before

    def _msg(self, label_rep, abs_encoding):
        q = k = torch.cat((label_rep, abs_encoding), dim=-1)

        # multi-head attention
        q, k, v = _heads(self.q(q), self.num_heads), _heads(self.k(k), self.num_heads), _heads(self.v(label_rep), self.num_heads)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)  # fastforks: einsum 'bhid, bhjd -> bhij'
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).flatten(-2)  # fastforks: einsum 'bhij, bhjd -> bhid', 'b h n d -> b n (h d)'
        return self.proj_drop(self.proj(out))

    def forward(self, label_rep, abs_encoding):
        """
        label_rep: [B, N, C], embedding of the candidate label
        abs_encoding: [B, N, C'], encoding of the underlying absolute disparity of the candidate label
        Returns:
            [B, N, C], aggregated message from self edges
        """
        # fastforks: upstream forward_pre / forward_post as one function
        if self.normalize_before:
            return label_rep + self._msg(self.norm1(label_rep), abs_encoding)
        return self.norm1(label_rep + self._msg(label_rep, abs_encoding))


class WindowAttention(nn.Module):
    """ Window based multi-head positional sensitive self attention (W-MSA).
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        shift_size (int): Shift size for SW-MSA.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override a default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, shift_size, num_heads, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # define a parameter table of relative position bias
        self.relative_position_enc_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), dim*3))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    @staticmethod
    def gen_window_attn_mask(window_size, device=torch.device('cuda')):
        """
        Generating attention mask to prevent message passing along self edges.

        Args:
            window_size (tuple[int]): The height, width, and depth (number of candidates) of attention window
        """
        idx = torch.arange(0, window_size[0] * window_size[1], dtype=torch.float32, device=device).view(-1, 1)
        idx = idx.expand(window_size[0] * window_size[1], window_size[2]).flatten()
        attn_mask = idx.unsqueeze(-1) - idx.unsqueeze(0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask != 0, 0.0)
        arange = torch.arange(idx.numel(), device=device)
        attn_mask[arange, arange] = 0  # fastforks: fill_diagonal_(0.0) as an indexed write (exports to ONNX)
        return attn_mask

    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        Generating attention mask for shifted window attention, modified from SWin Transformer.

        Args:
            input_resolution (tuple[int]): The height and width of input
            window_size (tuple[int]): The height, width and depth (number of candidates) of window
            shift_size (int): shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = mask_windows_of(img_mask, window_size[0], window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        n = window_size[2]  # fastforks: repeat 'b h w -> b (h h2) (w w2)', h2=w2=n
        attn_mask = attn_mask[:, :, None, :, None].expand(-1, -1, n, -1, n).flatten(3, 4).flatten(1, 2)
        attn_mask = attn_mask + WindowAttention.gen_window_attn_mask(window_size, device).unsqueeze(0)
        return attn_mask

    def forward(self, qkv, attn_mask):
        """
        qkv:   [B,H,W,N,3*C]
        mask:  [num_windows, window_size*window_size, window_size*window_size]
        Returns:
            BHWNC
        """
        bs, ht, wd, n, _ = qkv.shape
        if self.shift_size > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        window_size = self.window_size
        q, k, v = (windows(t, window_size[0], window_size[1], self.num_heads) for t in qkv.chunk(3, dim=-1))

        # positional embedding
        rpe = self.relative_position_enc_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], self.num_heads, -1)
        rpe = rpe[:, None, :, None].expand(-1, n, -1, n, -1, -1).flatten(2, 3).flatten(0, 1)  # fastforks: repeat 'i j h c -> (i hs) (j ws) h c'
        q_embed, k_embed, v_embed = rpe.chunk(3, dim=-1)

        # window attention
        q = q * self.scale
        q_embed = q_embed * self.scale
        qk = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        qr = torch.einsum('bhic,ijhc->bhij', q, k_embed)
        kr = torch.einsum('bhjc,ijhc->bhij', k, q_embed)
        attn = qk + qr + kr

        if attn_mask is not None:
            L = attn.shape[-1]  # fastforks: rearrange '(b j) h m n -> b j h m n' and back; mask in the logits' dtype
            attn = attn.view(bs, -1, self.num_heads, L, L) + attn_mask[None, :, None, :, :].to(attn.dtype)
            attn = attn.view(-1, self.num_heads, L, L)

        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = attn @ v + torch.einsum('bhij,ijhc->bhic', attn, v_embed)
        x = unwindows(x, ht // window_size[0], wd // window_size[1], window_size[0], window_size[1])

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, shift_size={self.shift_size}, num_heads={self.num_heads}'


def to_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class SwinNMP(nn.Module):
    r"""Swin Message Passing Block.

    Args:
        dim (int): Number of input channels.
        qkv_dim (int): Number of input token channels
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0 (fastforks: inference only, no DropPath)
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, qkv_dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize_before=False):
        super().__init__()
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.qkv = nn.Linear(qkv_dim, 3*dim, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), shift_size=shift_size, num_heads=num_heads,
            qk_scale=qk_scale, attn_drop=attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.normalize_before = normalize_before

    def _msg(self, label_rep, abs_encoding, attn_mask, H, W):
        # concat label embedding with absolute disparity embedding
        x = torch.cat((label_rep, abs_encoding), dim=-1)
        x = x.unflatten(0, (-1, H, W))  # fastforks: rearrange '(b h w) n c -> b h w n c'
        # window attention
        msg = self.attn(self.qkv(x), attn_mask)
        msg = self.proj_drop(self.proj(msg))
        return msg.flatten(0, 2)  # fastforks: rearrange 'b h w n c -> (b h w) n c'

    def forward(self, label_rep, abs_encoding, attn_mask, H, W):
        """
        label_rep: [B*H*W, N, C], embedding of the candidate label
        abs_encoding: [B*H*W, N, C'], encoding of the underlying
            absolute disparity of the candidate label
        attn_mask: [num_windows, window_size_h*window_size_w, window_size_h*window_size_w],
            attention mask to prevent message passing along self edges.
        H, W: fastforks: the token grid (upstream: attributes set by the caller)
        """
        # fastforks: upstream forward_pre / forward_post as one function
        if self.normalize_before:
            x = label_rep + self._msg(self.norm1(label_rep), abs_encoding, attn_mask, H, W)
            return x + self.mlp(self.norm2(x))
        x = self.norm1(label_rep + self._msg(label_rep, abs_encoding, attn_mask, H, W))
        return self.norm2(x + self.mlp(x))


class CSWinAttention(nn.Module):
    def __init__(self, dim, idx, split_size=7, num_heads=8, qk_scale=None, attn_drop=0.):
        """Attention within cross-shaped windows.
        """
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        assert idx in (-1, 0, 1), f"ERROR MODE {idx}"
        self.idx = idx
        self._masks = {}  # fastforks: see cached_mask

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def get_rpe(self, x, H_sp, W_sp):
        """fastforks: window sizes as arguments (upstream: attributes set in forward); the rearranges as reshape /
        permute and the self-edge einsum ('jk,bkdhw->bjdhw' with -1 off the diagonal) as `t - t.sum(n)`."""
        B, H, W, n, d = x.shape
        i, j = H // H_sp, W // W_sp
        # 'b (i hs) (j ws) n d -> (b i j n) d hs ws'
        x = x.reshape(B, i, H_sp, j, W_sp, n, d).permute(0, 1, 3, 5, 6, 2, 4).reshape(-1, d, H_sp, W_sp)
        rpe = self.get_v(x)  # B', C, H', W'
        rpe = torch.sum(rpe.unflatten(0, (-1, n)), dim=1, keepdim=True)  # '(b n) d hs ws -> b n d hs ws'

        # remove the positional embedding of self edges
        tmp = x.unflatten(0, (-1, n)) * ((self.get_v.weight[:, 0, 1, 1])[None, None, :, None, None])
        tmp = tmp - tmp.sum(dim=1, keepdim=True)
        rpe = rpe + tmp

        def to_tokens(t):  # 'b n (h d) hs ws -> b h (hs ws n) d'
            b = t.shape[0]
            t = t.reshape(b, n, self.num_heads, d // self.num_heads, H_sp, W_sp)
            return t.permute(0, 2, 4, 5, 1, 3).reshape(b, self.num_heads, H_sp * W_sp * n, d // self.num_heads)

        return to_tokens(x.unflatten(0, (-1, n))), to_tokens(rpe)

    def forward(self, query, key, value):
        """
        query: BHWNC
        key:   BHWNC
        value: BHWNC'
        Returns:
            BHWNC
        """
        _, ht, wd, dd, _ = query.shape

        idx = self.idx
        if idx == -1:
            H_sp, W_sp = ht, wd
        elif idx == 0:
            H_sp, W_sp = ht, self.split_size
        else:
            H_sp, W_sp = self.split_size, wd

        ### padding for split window
        H_pad = (H_sp - ht % H_sp) % H_sp
        W_pad = (W_sp - wd % W_sp) % W_sp
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        ht_ = ht + H_pad
        wd_ = wd + W_pad

        q = F.pad(query, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
        k = F.pad(key, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
        v = F.pad(value, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))

        q = windows(q, H_sp, W_sp, self.num_heads)
        k = windows(k, H_sp, W_sp, self.num_heads)
        v, rpe = self.get_rpe(v, H_sp, W_sp)

        ### Local attention
        # fastforks: fused SDPA with the cached self-edge mask as additive bias (our fork's fast.py `_cswin_forward`;
        # upstream: q * scale, q @ k^T + mask, softmax, @ v on materialized logits)
        window_size = (H_sp, W_sp, dd)
        mask = cached_mask(self._masks, (window_size, q.device), lambda: WindowAttention.gen_window_attn_mask(window_size, q.device))
        x = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=mask[None, None].to(q.dtype),
                                           dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale) + rpe
        x = unwindows(x, ht_ // H_sp, wd_ // W_sp, H_sp, W_sp)
        x = x[:, top_pad:ht+top_pad, left_pad:wd+left_pad, :, :]

        return x


class CSWinNMP(nn.Module):

    def __init__(self, dim, qk_dim, num_heads,
                 split_size=7, mlp_ratio=4., qk_scale=None,
                 attn_drop=0., proj_drop=0., dropout=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize_before=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio

        # fastforks: v_dim == dim (PropagationLayer), so upstream's fourier_grid_embed branch for v_dim > dim is dropped
        self.q, self.k, self.v = nn.Linear(qk_dim, dim, bias=True), nn.Linear(qk_dim, dim, bias=True), nn.Linear(dim, dim, bias=True)
        self.norm1 = norm_layer(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
            CSWinAttention(
                dim//2, idx=i, split_size=split_size, num_heads=num_heads//2, qk_scale=qk_scale,
                attn_drop=attn_drop)
            for i in range(2)])
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=dropout)
        self.norm2 = norm_layer(dim)

        self.normalize_before = normalize_before

    def _msg(self, tgt2, context):
        tgt2 = tgt2.unflatten(0, context.shape[:3])  # fastforks: rearrange '(b h w) n c -> b h w n c'
        q = k = torch.cat((tgt2, context), dim=-1)
        query, key, value = self.q(q), self.k(k), self.v(tgt2)

        # cross shaped window attention
        x1 = self.attns[0](query[..., :self.dim // 2], key[..., :self.dim // 2], value[..., :self.dim // 2])
        x2 = self.attns[1](query[..., self.dim // 2:], key[..., self.dim // 2:], value[..., self.dim // 2:])
        msg = torch.cat([x1, x2], dim=-1).flatten(0, 2)
        return self.proj_drop(self.proj(msg))

    def forward(self, seed_rep, context):
        """
        seed_rep: [BHW,N,C], embedding of the label seed
        context: BHWNC', visual context of the label seed
        """
        # fastforks: upstream forward_pre / forward_post as one function
        if self.normalize_before:
            x = seed_rep + self._msg(self.norm1(seed_rep), context)
            return x + self.mlp(self.norm2(x))
        x = self.norm1(seed_rep + self._msg(seed_rep, context))
        return self.norm2(x + self.mlp(x))


class Propagation(nn.Module):
    """Label seed propagation"""
    def __init__(self, embed_dim, cost_group, layers, norm=None):
        super().__init__()
        self.cost_encoder = nn.Sequential(
            nn.Linear(cost_group*9, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj = nn.Linear(embed_dim+31, embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.layers = layers
        self.norm = norm

    @staticmethod
    def sample_cost(cost_volume, label_seed):
        """
        cost_volume: [B*H*W,G,D], groupwise cost volume
        label_seed: [B*H*W,num_seed], integer disparity modals
        return:
            [B*H*W,num_seed,G*9], sampled cost values of the label seed
        """
        G, nd = cost_volume.shape[-2:]
        num_seed = label_seed.shape[1]
        offset = torch.arange(-4, 5, device=cost_volume.device, dtype=label_seed.dtype)
        idx = label_seed[..., None] + offset.view(1, 1, -1)  # [B*H*W,num_seed,9]
        idx = torch.clamp(idx, min=0, max=nd-1)
        idx = idx.reshape(-1, 1, 9*num_seed).expand(-1, G, -1)  # fastforks: expand, not repeat
        cost = torch.gather(cost_volume, dim=-1, index=idx)
        # fastforks: rearrange 'b h (n c) -> b n (h c)', n=num_seed
        cost = cost.unflatten(-1, (num_seed, 9)).transpose(1, 2).flatten(-2)
        return cost

    def forward(self, cost_volume, label_seed, context):
        """
        cost_volume: B*H*W,G,D, groupwise cost volume
        coord_seed: [B*H*W, num_seed], integer disparity modals
        context:    [ B,H,W,C], visual context
        """
        # extract cost of label seeds
        N = label_seed.shape[-1]
        cost = Propagation.sample_cost(cost_volume, label_seed)
        cost_feat = self.cost_encoder(cost)
        label_seed = label_seed.float()
        disp_encoding = fourier_coord_embed(label_seed.unsqueeze(-1), N_freqs=15, normalizer=3.14/64)
        # concat & proj
        sample_embed = self.proj(torch.cat((cost_feat, disp_encoding), dim=-1))

        context = context.unsqueeze(-2).expand(-1, -1, -1, N, -1)  # fastforks: repeat 'b h w c -> b h w n c'
        for layer in self.layers:
            sample_embed = layer(sample_embed, context)

        if self.norm is not None:
            sample_embed = self.norm(sample_embed)

        return sample_embed.unsqueeze(0), label_seed


class Inference(nn.Module):
    """Neural MRF Inference"""
    def __init__(self, cost_group, dim, layers, norm):
        super().__init__()

        self.ffn = Mlp(dim+cost_group, dim, dim)
        self.dim = dim
        self.layers = layers
        self.norm = norm
        self.cost_group = cost_group
        self._masks = {}  # fastforks: see cached_mask

    @staticmethod
    def sample_fmap(fmap, disp_sample, radius=4):
        """
        fmap: [B, C, H, W]
        disp_sample: tensor of dim [B*H*W, num_disp], disparity samples
        radius(int): 2*radius+1 samples will be sampled for each sample
        return:
            sampled fmap feature of dim [B, C, H, W, num_disp*(2*radius+1)]
        """
        bs, _, ht, wd = fmap.shape
        num_disp = disp_sample.shape[1]
        device = fmap.device
        with torch.no_grad():
            offset = torch.arange(-radius, radius+1, dtype=disp_sample.dtype, device=disp_sample.device).view(1, 1, -1)
            grid_x = disp_sample[..., None] + offset  # [B*H*W, num_disp, 2*r+1]
            grid_x = grid_x.reshape(bs, ht, wd, -1)  # [B, H, W, num_disp*(2*r+1)]
            xs = torch.arange(0, wd, device=device, dtype=torch.float32).view(1, 1, wd, 1)
            ys = torch.arange(0, ht, device=device, dtype=torch.float32).view(1, ht, 1, 1)
            # fastforks: the normalized grid without in-place writes (same arithmetic)
            gx = 2 * (xs + -grid_x) / (wd - 1) - 1
            gy = (2 * ys / (ht - 1) - 1).expand_as(gx)
            grid = torch.stack((gx, gy), dim=-1).reshape(bs, ht, -1, 2)  # [B, H, W*num_disp*(2*r+1), 2]
        samples = grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return samples.reshape(bs, -1, ht, wd, num_disp*(2*radius+1))

    def corr(self, fmap1, warped_fmap2, num_disp):
        """
        fmap1: [B, C, H, W]
        warped_fmap2: [B, C, H, W, num_disp]
        Returns:
            local cost: [B*H*W, num_disp, G]
        """
        fmap1 = fmap1.unflatten(1, (self.cost_group, -1))  # fastforks: rearrange 'b (g d) h w -> b g d h w'
        warped_fmap2 = warped_fmap2.unflatten(1, (self.cost_group, -1))
        corr = (fmap1.unsqueeze(-1) * warped_fmap2).mean(dim=2)  # [B, G, H, W, num_disp]
        corr = corr.permute(0, 2, 3, 4, 1).flatten(0, 2)  # fastforks: rearrange 'b g h w n -> (b h w) n g'
        return corr

    def _pad(self, ht, wd):
        # pad input to multiple times of window_size (assume all swin blocks have the same window size)
        window_size = self.layers[0].window_size
        H_pad = (window_size - ht % window_size) % window_size
        W_pad = (window_size - wd % window_size) % window_size
        top_pad = H_pad // 2
        left_pad = W_pad // 2
        return (0, 0, 0, 0, left_pad, W_pad - left_pad, top_pad, H_pad - top_pad)

    def _layers(self, label_rep, abs_encoding, attn_mask, ht, wd):
        """fastforks: upstream's padding, layer loop and crop (shared by Inference and Refinement)"""
        pad = self._pad(ht, wd)
        ht_, wd_ = ht + pad[6] + pad[7], wd + pad[4] + pad[5]
        if ht_ > ht or wd_ > wd:
            label_rep = F.pad(label_rep.unflatten(0, (-1, ht, wd)), pad).flatten(0, 2)
            abs_encoding = F.pad(abs_encoding.unflatten(0, (-1, ht, wd)), pad).flatten(0, 2)

        masks = attn_mask(ht_, wd_)
        for idx, layer in enumerate(self.layers):
            label_rep = layer(label_rep, abs_encoding, masks[idx % 2], ht_, wd_)

        label_rep = label_rep.unflatten(0, (-1, ht_, wd_))
        label_rep = label_rep[:, pad[6]:pad[6]+ht, pad[4]:pad[4]+wd, :, :].flatten(0, 2)
        if self.norm is not None:
            label_rep = self.norm(label_rep)
        return label_rep.unsqueeze(0)

    def forward(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw):
        """
        labels: [B*H*W,num_disp], candidate labels (disparity)
        fmap1: [B,C,H,W]
        fmap2: [B,C,H,W]
        fmap1_gw: [B,C,H,W]
        fmap2_gw: [B,C,H,W]
        Returns:
            [B, H, W]
        """
        bs, _, ht, wd = fmap1.shape
        N = labels.shape[-1]
        device = labels.device
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, labels, radius=0)  # [B,C,H,W,N]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, N)  # [B*H*W,N,G]
        warped_fmap2 = self.sample_fmap(fmap2, labels, radius=0)  # [B,C,H,W,N]
        fmap1 = fmap1.unsqueeze(-1).expand(-1, -1, -1, -1, N)  # fastforks: repeat 'b c h w -> b c h w n'
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = feat_concat.permute(0, 2, 3, 4, 1).flatten(0, 2)  # fastforks: rearrange 'b c h w n -> (b h w) n c'
        label_rep = self.ffn(torch.cat((feat_concat, corr), dim=-1))

        abs_encoding = fourier_coord_embed(labels.unsqueeze(-1), N_freqs=15, normalizer=3.14/64)

        window_size = self.layers[0].window_size
        window_size = (window_size, window_size, N)

        def masks(ht_, wd_):
            attn_mask = [cached_mask(self._masks, (window_size, device),
                                     lambda: WindowAttention.gen_window_attn_mask(window_size, device=device)[None])]
            if len(self.layers) >= 2:
                shift_size = self.layers[1].shift_size
                input_resolution = (ht_, wd_)
                attn_mask.append(cached_mask(self._masks, (input_resolution, window_size, shift_size, device),
                                             lambda: WindowAttention.gen_shift_window_attn_mask(
                                                 input_resolution, window_size, shift_size, device)))
            return attn_mask

        return self._layers(label_rep, abs_encoding, masks, ht, wd)


class Refinement(Inference):
    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        input_resolution (tuple[int]): The height and width of input
        window_size (tuple[int]): The height, width and depth of window
        shift_size (int): Shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = mask_windows_of(img_mask, window_size[0], window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        return attn_mask

    def forward(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw):
        """
        labels: [B,H,W]
        fmap1: [B,C,H,W]
        fmap2: [B,C,H,W]
        fmap1_gw: [B,C,H,W]
        fmap2_gw: [B,C,H,W]
        """
        bs, _, ht, wd = fmap1.shape
        device = labels.device
        labels = labels.reshape(-1, 1)
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, labels, radius=0)  # [B,C,H,W,1]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, 1)  # [B*H*W,1,G]
        warped_fmap2 = self.sample_fmap(fmap2, labels, radius=0).squeeze(-1)  # [B,C,H,W]
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = feat_concat.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)  # fastforks: rearrange 'b c h w -> (b h w) 1 c'
        label_rep = self.ffn(torch.cat((feat_concat, corr), dim=-1))

        abs_encoding = fourier_coord_embed(labels.unsqueeze(-1), N_freqs=15, normalizer=3.14 / 128)

        window_size = self.layers[0].window_size
        window_size = (window_size, window_size)

        def masks(ht_, wd_):
            attn_mask = [None]
            if len(self.layers) >= 2:
                shift_size = self.layers[1].shift_size
                input_resolution = (ht_, wd_)
                attn_mask.append(cached_mask(self._masks, (input_resolution, window_size, shift_size, device),
                                             lambda: self.gen_shift_window_attn_mask(
                                                 input_resolution, window_size, shift_size, device)))
            return attn_mask

        return self._layers(label_rep, abs_encoding, masks, ht, wd).squeeze(-2)


class PropagationLayer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, context_dim, split_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., dropout=0., normalize_before=False):
        super().__init__()

        # self attention
        act_layer = _get_activation_fn(activation)
        # concat seed embedding with visual context when linearly projecting to
        # query and key since visually similar pixel tends to have coherent disparities
        qk_dim = embed_dim + context_dim
        self.nmp = CSWinNMP(embed_dim, qk_dim, num_heads=n_heads,
                            split_size=split_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                            proj_drop=proj_drop, dropout=dropout, act_layer=act_layer,
                            normalize_before=normalize_before)

    def forward(self, tgt, context):
        """
        tgt:        BHW,N,C
        context:    B,H,W,N,C
        Returns:
            BHW,N,C
        """
        return self.nmp(tgt, context=context)


class InferenceLayer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, window_size, shift_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., dropout=0., normalize_before=False):
        super().__init__()

        # self attention
        act_layer = _get_activation_fn(activation)
        qk_dim = embed_dim + 31
        self.window_size = window_size
        self.shift_size = shift_size
        # attend to proposals of the same pixel to suppress non-accurate proposals
        self.self_nmp = BasicAttention(embed_dim, qk_dim, n_heads, attn_drop=attn_drop, proj_drop=proj_drop,
                                       dropout=dropout, normalize_before=normalize_before)
        # attend to neighbor pixels to extract feature
        self.nmp = SwinNMP(embed_dim, qk_dim, num_heads=n_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                           drop=dropout, act_layer=act_layer, normalize_before=normalize_before)

    def forward(self, tgt, abs_encoding, attn_mask, ht, wd):
        """
        tgt: B*H*W,N,C
        abs_encoding: B*H*W,N,C
        """
        tgt = self.self_nmp(tgt, abs_encoding=abs_encoding)
        return self.nmp(tgt, abs_encoding, attn_mask, ht, wd)


class RefinementLayer(nn.Module):
    def __init__(self, dim, mlp_ratio, window_size, shift_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., dropout=0., normalize_before=False):
        super().__init__()

        act_layer = _get_activation_fn(activation)
        qk_dim = dim + 31
        self.window_size = window_size
        self.shift_size = shift_size
        self.nmp = SwinNMP(dim, qk_dim, num_heads=n_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                           drop=dropout, act_layer=act_layer, normalize_before=normalize_before)

    def forward(self, tgt, abs_encoding, attn_mask, ht, wd):
        """
        tgt:        [B*H*W,1,C]
        context:    [B*H*W,1,C]
        """
        return self.nmp(tgt, abs_encoding, attn_mask, ht, wd)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
