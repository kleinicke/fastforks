"""MVSplat's network modules (donydchen/mvsplat 01f9a28, `src/model/encoder/`; MIT, see ../LICENSE).

Unchanged except (each marked `# fastforks:`): `einops.rearrange` / `repeat` written as `view` / `permute` / `flatten`
/ `expand`; `multiview_transformer.py` without `forward_epipolar` (the epipolar-transformer ablation) and its
multi-view window attention as one `scaled_dot_product_attention` call (same math); `ldm_unet`: the
attention block without gradient checkpointing, ldm's `attention.py` (the spatial transformer) not vendored and the
omegaconf `context_dim` branch raising (MVSplat never builds either). `BackboneMultiview`, `DepthPredictorMultiView`
and the Gaussian adapter are rewritten in `../model.py` (same module names, so the checkpoint loads unchanged).
"""
