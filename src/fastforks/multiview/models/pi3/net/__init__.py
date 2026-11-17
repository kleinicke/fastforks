"""Vendored from yyfz/Pi3 9fa3ddb (BSD-3-Clause, see ../LICENSE).

`pi3.py` (= `pi3/models/pi3.py`), `layers/{attention,block,pos_embed,transformer_head,camera_head}.py` and `dinov2/`
(hub, layers, models); changes: `pi3.py` without `PyTorchModelHubMixin` and with `homogenize_points` inlined,
`layers/pos_embed.py` always uses the PyTorch `RoPE2D` (no cuRoPE2D import), and the Pi3X-only tails of
`layers/attention.py` (`PRopeFlashAttention`) and `layers/block.py` (`PoseInjectBlock`) are cut.
"""
