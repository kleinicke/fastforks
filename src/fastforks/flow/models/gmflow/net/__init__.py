"""UniMatch modules, vendored from autonomousvision/unimatch 943a990 `unimatch/` (MIT, see ../LICENSE).

One change: the three `torch.meshgrid` calls (`geometry.py`, `utils.py`) say `indexing="ij"`, what they did silently
upstream (PyTorch warns without it).

`attention.py`: the 2D attentions (full, split-window, global propagation) call `F.scaled_dot_product_attention` (`sdpa`)
instead of materialising the `[L, L]` scores; the same maths, fewer kernels, no score matrix.

`position.py` (DETR) and `trident_conv.py` (detectron2) carry their own headers (Apache-2.0, ../LICENSE-Apache-2.0).
"""
