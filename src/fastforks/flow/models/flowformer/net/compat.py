"""Optional timm operators loaded through the library's dependency gate."""
from fastforks.shared import libnet

timm = libnet.timm()
Mlp = timm.layers.Mlp
DropPath = timm.layers.DropPath
to_2tuple = timm.layers.to_2tuple
trunc_normal_ = timm.layers.trunc_normal_
Attention = timm.models.vision_transformer.Attention
IMAGENET_DEFAULT_MEAN = timm.data.IMAGENET_DEFAULT_MEAN
IMAGENET_DEFAULT_STD = timm.data.IMAGENET_DEFAULT_STD
