"""SAM 3 exemplar boxes with weights (skipped without them): box padding is exact, kernels graph cache per bucket."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, cast

import pytest
import torch

from fastforks import BackendUnavailableError, WeightsError, promptable_segmentation
from fastforks._core import registry
from fastforks.promptable_segmentation.models.sam3.model import BOX_BUCKET, Kernels, Portable, exemplars
from synth import image

if TYPE_CHECKING:
    from fastforks._core.cards import Mode

IMAGE = image(240, 320, seed=3)  # fixed: any texture; the tests compare two paths on one image
TEXT = "coin"


def _load(mode: Mode, device: str) -> promptable_segmentation.ConceptSegmentationModel:
    card, variant = registry.resolve("promptable_segmentation", "sam3")
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"sam3: gated weights not available locally: {e}")
    try:
        return promptable_segmentation.load("sam3", mode=mode, device=device)
    except BackendUnavailableError as e:
        pytest.skip(f"sam3 {mode}: {e}")


def _boxes(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`n` normalised boxes (fractional, some past the image border) and labels with a negative every third box."""
    g = torch.Generator().manual_seed(n)
    xy = torch.rand((n, 2), generator=g) * 1.1 - 0.05
    wh = torch.rand((n, 2), generator=g) * 0.3 + 0.01
    return torch.cat([xy, xy + wh], 1), (torch.arange(n) % 3 != 2).long()


@pytest.mark.model
def test_box_padding_is_exact_on_cpu():
    """Padded exemplar boxes (as in the kernels CUDA graphs) give the unpadded result in float32: the mask is exact.

    Also the eager path of more than `BOX_BUCKET` boxes (padded here to twice the bucket).
    """
    m = _load("portable", "cpu")
    r = cast("Portable", m._runner)
    emb = m.get_image_embeddings(IMAGE)
    fpn = [emb.features[f"fpn{i}"] for i in range(3)]
    ids = torch.tensor([r.tokenizer(TEXT)])
    cpu = torch.device("cpu")
    for n, pad in ((3, BOX_BUCKET), (BOX_BUCKET + 4, 2 * BOX_BUCKET)):
        boxes, labels = _boxes(n)
        with m._session():
            plain = r.concept(ids, fpn, ex=exemplars(boxes, labels, cpu))
            padded = r.concept(ids, fpn, ex=exemplars(boxes, labels, cpu, pad))
        # bounds ~5x the float32 summation-order noise (the same boxes reordered: scores 1.3e-6, boxes 1.3e-5, mask
        # logits up to |94| 3.2e-3 max / 2e-5 mean); unmasked padding moves scores by far more
        for k, atol in (("scores", 1e-5), ("boxes", 5e-5), ("pred_masks", 1.5e-2)):
            torch.testing.assert_close(padded[k], plain[k], rtol=0, atol=atol, msg=f"{n} boxes: {k}")
        assert float((padded["pred_masks"] - plain["pred_masks"]).abs().mean()) < 1e-4


@pytest.mark.model
@pytest.mark.gpu
def test_kernels_graph_cache_per_bucket():
    """One graph per (token bucket, box bucket), reused across bucket switches; > `BOX_BUCKET` boxes run eagerly."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    m = _load("kernels", "cuda")
    r = cast("Kernels", m._runner)
    emb = m.get_image_embeddings(IMAGE)
    h, w = emb.image_size
    scale = torch.tensor([w, h, w, h])
    graphs = r.cache.graphs

    def twice(*args, **kwargs) -> None:  # a key is captured on its second call
        m(emb, *args, **kwargs)
        m(emb, *args, **kwargs)

    twice(text=TEXT)
    assert set(graphs) == {(8, 0)}
    text_only = graphs[8, 0]
    one, _ = _boxes(1)
    twice(input_boxes=one * scale)
    assert set(graphs) == {(8, 0), (8, BOX_BUCKET)}
    boxed = graphs[8, BOX_BUCKET]
    five, five_labels = _boxes(5)
    twice(text=TEXT, input_boxes=five * scale, input_boxes_labels=five_labels)  # same bucket: no new capture
    many, many_labels = _boxes(BOX_BUCKET + 1)
    with m._session():
        out = r.segment_concept(dict(emb.features), TEXT, many, many_labels)  # eager, no graph
    assert set(graphs) == {(8, 0), (8, BOX_BUCKET)}
    assert all(bool(v.isfinite().all()) for v in out.values())
    assert out["scores"].shape == (200,)  # fixed: the DETR queries
    twice(text=TEXT)  # back to text only: the first graph
    assert graphs[8, 0] is text_only
    assert graphs[8, BOX_BUCKET] is boxed
    twice(text="a yellow school bus with a long black roof rack", input_boxes=one * scale)
    assert set(graphs) == {(8, 0), (8, BOX_BUCKET), (16, BOX_BUCKET)}


@pytest.mark.model
@pytest.mark.gpu
def test_kernels_graphs_from_two_streams():
    """Two threads, each on its own stream, replaying the shared graph buffers = the sequential results.

    The lock only orders the host; the next stream must wait for the previous call's replay and copy-out on the GPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    m = _load("kernels", "cuda")
    r = cast("Kernels", m._runner)
    embs = [m.get_image_embeddings(image(240, 320, seed=s)) for s in (3, 4)]
    feats = [dict(e.features) for e in embs]
    texts = (TEXT, "a red car")
    with m._session():
        for f, t in zip(feats, texts, strict=True):  # the key's first call is eager, the second captures the graph
            r.segment_concept(f, t)
        want = [r.segment_concept(f, t) for f, t in zip(feats, texts, strict=True)]
    torch.cuda.synchronize()
    got: list[list[dict[str, torch.Tensor]]] = [[], []]
    start = threading.Barrier(2)

    def work(i: int) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.default_stream())
        start.wait()
        with torch.inference_mode(), torch.cuda.stream(stream):
            got[i] += [r.segment_concept(feats[i], texts[i]) for _ in range(12)]
        stream.synchronize()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    for i in range(2):
        assert len(got[i]) == 12
        for out in got[i]:
            for k in ("scores", "boxes", "pred_masks"):
                torch.testing.assert_close(out[k], want[i][k], rtol=0, atol=0, msg=f"call {i}: {k}")
