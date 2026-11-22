"""MobileSAM: the traceable TinyViT forward (`fast.Trunk`) equals upstream `TinyViT.forward` on the same modules.

Also the kernels image-encoder CUDA graph of MobileSAM and EfficientViT-SAM against the eager runner.
"""

from __future__ import annotations

import pytest
import torch

from fastforks import promptable_segmentation
from fastforks._core.errors import WeightsError
from fastforks.promptable_segmentation.models.mobile_sam.fast import Trunk
from fastforks.promptable_segmentation.models.mobile_sam.model import Params, Sam


@pytest.mark.slow
def test_trunk_matches_upstream_forward() -> None:
    size = 1024  # upstream's forward reshapes to 64 x 64 tokens: the only size it runs at
    torch.manual_seed(0)
    enc = Sam(Params(image_size=size, cuda_graphs=True)).image_encoder.eval()
    for m in enc.modules():  # upstream zero-initialises the residual branches' last BatchNorm: make every path count
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.uniform_(m.weight, 0.5, 1.5)
            torch.nn.init.uniform_(m.running_var, 0.5, 1.5)
            torch.nn.init.normal_(m.running_mean, std=0.2)
    for name, p in enc.named_parameters():
        if name.endswith("attention_biases"):
            torch.nn.init.normal_(p)
    enc.train().eval()  # upstream `Attention.train(False)` caches the bias table `ab` from `attention_biases`
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        want = enc(x)
        got = enc.neck[3](Trunk(enc)(x))
    assert want.shape == (1, 256, 64, 64)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("ref", ["mobile-sam@1", "efficientvit-sam-l0@1"])
def test_kernels_encoder_graph_matches_eager(ref: str) -> None:
    """The graphed image encoder: bit-equal to eager, a kept embedding survives the next image, release frees it."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        graphed = promptable_segmentation.load(ref, mode="kernels", device="cuda")
        eager = promptable_segmentation.load(ref, mode="kernels", device="cuda", cuda_graphs=False)
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")
    gen = torch.Generator("cuda").manual_seed(0)
    a, b = (torch.rand(3, 600, 800, device="cuda", generator=gen) for _ in range(2))
    points, labels = torch.tensor([[400.0, 300.0]]), torch.tensor([1])
    assert eager._runner.cache is None
    for image in (a, a, a, b):  # eager, capture, replay, replay with new data
        got = graphed(image, input_points=points, input_labels=labels)
        want = eager(image, input_points=points, input_labels=labels)
        for k in ("pred_masks", "iou_scores"):
            torch.testing.assert_close(getattr(got, k), getattr(want, k), rtol=0, atol=0)
    cache = graphed._runner.encoder.cache
    assert cache is not None
    assert len(cache.graphs) == 1, f"no encoder graph captured (failed: {list(cache._failed)})"
    kept = graphed.get_image_embeddings(a)
    first = kept.features["image_embeddings"].clone()
    graphed.get_image_embeddings(b)
    assert torch.equal(kept.features["image_embeddings"], first)
    graphed.release_graphs()
    assert not cache.graphs
