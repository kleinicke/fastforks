"""DepthSplat's depth network (cvg/depthsplat 2dad25a, `src/model/encoder/unimatch/`; MIT, see ../LICENSE).

Unchanged except (each marked `# fastforks:`): `einops.rearrange` / `repeat` written as `flatten` / `unflatten` /
`expand` / `view`; the monocular ViT built with `fastforks.shared.depth_anything_v2.DINOv2` instead of
`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` (the same ViT-S/14, no download; the checkpoint holds its
weights); `ldm_unet`: the attention block without gradient checkpointing, and ldm's `attention.py` (the spatial
transformer) and `cross_attention.py` (xformers only, the condition blocks) not vendored: DepthSplat builds neither. `feature_upsampler.py` (unused by the released model) is left out.
"""
