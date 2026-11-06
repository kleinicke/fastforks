"""C-FoundationStereo networks, vendored from NVIDIA TAO PyTorch (Apache-2.0) at 77fb1a4.

Source: `nvidia_tao_pytorch/cv/depth_net/model/stereo_depth/{foundation_stereo,fast_foundation_stereo}`, plus the parts
they import from elsewhere in TAO (`depth_anything_v2/{dpt,blocks}.py`, `backbone_v2/edgenext{,_utils}.py`,
`nn/norm.py`). Changes: relative imports; DINOv2 and the DPT blocks from `fastforks.shared.depth_anything_v2` (TAO's `blocks.py` is
the upstream Depth Anything V2 code with docstrings; no pretrained-path loading: the weights come with the stereo
checkpoint); TAO-only code removed (Lightning model, registry / `BackboneBase`, timm
imports, `MetricDepthAnythingV2`, the EdgeNeXt BN-HS variants, file / dataset / image helpers with scipy, torchvision and
PIL); `DropPath` -> `nn.Identity` (inference only); the module-level `enable_flash_sdp` switch and the ONNX `sdp_kernel`
branch in `submodule.py` removed; the data-dependent `torch.unique` assert in `utils.bilinear_sampler` dropped (blocks
`torch.export`); the C-FFS checkpoint finder / loader in `ckpt_utils.py` reduced to its key remap
(`remap_state_dict`). The TAO `EdgeNeXt` backbone (`backbone_v2/`) is replaced by the shared timm-layout
`fastforks.shared.edgenext` (the checkpoint's timm keys load without a remap; outputs bit-identical). The rest of the network code is unchanged.
"""
