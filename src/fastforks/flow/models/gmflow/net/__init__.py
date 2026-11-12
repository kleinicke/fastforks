"""UniMatch modules, vendored from autonomousvision/unimatch 943a990 `unimatch/` (MIT, see ../LICENSE).

One change: the three `torch.meshgrid` calls (`geometry.py`, `utils.py`) say `indexing="ij"`, what they did silently
upstream (PyTorch warns without it).

`position.py` (DETR) and `trident_conv.py` (detectron2) carry their own headers (Apache-2.0, ../LICENSE-Apache-2.0).
"""
