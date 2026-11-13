"""MegaFlow modules, vendored from cvg/megaflow 2328f8b `megaflow/model/` (Apache-2.0, see ../LICENSE).

Changes:
- `megaflow.py`: `MegaFlow.forward` takes the normalised, padded frames (the runner's card `[io]` does upstream's
  `imgs / 255` ImageNet normalisation and `InputPadderMF` padding; the flow is cropped by the runner);
  `from_pretrained` (huggingface_hub download), the tracking forwards and the unpadding are removed; the ResNet CNN is
  built with `init_weight=False` (upstream downloads torchvision's ResNet-34 weights, which the checkpoint overwrites).
- `refine.py`: the same `init_weight` note.
- `layers/attention.py`: no xFormers / FlashAttention-3 imports (upstream's plain PyTorch path without them).
- `layers/vision_transformer.py`: the stochastic-depth `linspace` on the CPU (no default-device dependence) and
  `trunc_normal_` from `fastforks._core.deferred_init` (deferred past the checkpoint load).
- `geometry.py`: the two `torch.meshgrid` calls say `indexing="ij"`, what they did silently upstream.

The DINOv2 / VGGT layers (`layers/`) carry Meta's Apache-2.0 headers.
"""
