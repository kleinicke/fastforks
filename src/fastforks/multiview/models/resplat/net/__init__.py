"""ReSplat's networks (cvg/resplat cae7ddc, `src/model/encoder/`; MIT, see ../LICENSE).

`mv_unimatch.py` (`unimatch/mv_unimatch.py`: DepthSplat's depth network with log-depth candidates and the raw DINOv2
features returned), `point_transformer.py` (`point_transformer.py`) and `resnet.py` (the ResNet-18 of `layer.py`).
Unchanged except (each marked `# fastforks:`): DepthSplat's vendored UniMatch modules imported instead of copied (the
same upstream code); the monocular ViT built with `fastforks.shared.depth_anything_v2.DINOv2`; bf16 autocast blocks
left to the runner; `einops.rearrange` written as `flatten` / `unflatten` / `permute`; the `pointops` CUDA extension
replaced by an exact chunked `torch.cdist` kNN and indexing (one batch); `xformers` attention as PyTorch SDPA; the
ResNet without torchvision; no gradient checkpointing and the training / ablation-only options left out.
"""
