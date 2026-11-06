"""Vendored from facebookresearch/vggt a288dd0f14786c93483e45524328726ab7b1b4ce (VGGT License, see ../LICENSE).

`layers/`, `models/aggregator.py`, `heads/{camera_head,dpt_head,head_act,utils}.py`, `utils/{pose_enc,rotation}.py`;
changes: absolute `vggt.` imports made relative, and one line in `heads/dpt_head.py` (`_apply_pos_embed` casts the
position embedding to the feature dtype, keeps fp16 heads in fp16, a no-op in fp32). The `VGGT` module
itself (it imports the tracking head and huggingface_hub) is rebuilt in `../model.py` with the same attribute names.
"""
