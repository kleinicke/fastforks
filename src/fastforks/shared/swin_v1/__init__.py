"""Swin Transformer v1 backbone (Liu et al. 2021) as BiRefNet vendors it, with SDPA window attention.

Four normalised stage outputs, padding to the window size, shifted-window mask. `swin_v1_t` / `_s` / `_b` / `_l`
build the sizes.
"""

from .net.swin_v1 import SwinTransformer, swin_v1_b, swin_v1_l, swin_v1_s, swin_v1_t

__all__ = ["SwinTransformer", "swin_v1_b", "swin_v1_l", "swin_v1_s", "swin_v1_t"]
