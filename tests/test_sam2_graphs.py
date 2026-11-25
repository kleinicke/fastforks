"""SAM 2.1: prompt coordinates without host copies; the kernels image encoder and prompt decode from CUDA graphs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastforks import WeightsError, promptable_segmentation
from fastforks.promptable_segmentation.models.sam2 import model as sam2


def test_points_match_upstream_prep_prompts() -> None:
    """`_points` = upstream `_prep_prompts` (`xy / [w, h] * S`, box corners labelled 2 / 3 first), bit for bit."""
    s, (h, w) = 1024, (731, 1297)
    runner = SimpleNamespace(ctx=SimpleNamespace(params=SimpleNamespace(image_size=s), device=torch.device("cpu")))
    g = torch.Generator().manual_seed(0)
    points, labels = torch.rand(3, 2, 2, generator=g) * torch.tensor([w, h]), torch.tensor([[1, 0], [1, 1], [0, 1]])
    boxes = torch.rand(3, 4, generator=g) * torch.tensor([w, h, w, h])
    wh = torch.tensor([w, h], dtype=torch.float32)
    want = torch.cat([boxes.view(-1, 2, 2) / wh * s, points / wh * s], 1)
    pts = sam2._Sam2._points(runner, (h, w), points, labels, boxes)
    assert pts is not None
    coords, labs = pts
    assert torch.equal(coords, want)
    assert torch.equal(labs, torch.cat([torch.tensor([[2, 3]]).expand(3, 2), labels], 1))
    assert sam2._Sam2._points(runner, (h, w), None, None, None) is None


@pytest.mark.gpu
@pytest.mark.model
def test_kernels_graphs_match_eager() -> None:
    """The encoder graph and the prompt-decode graphs give the eager numbers, fresh outputs; release and opt-out."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        model = promptable_segmentation.load(
            "sam2.1-hiera-tiny", mode="kernels", device="cuda", cuda_graphs_encoder=True
        )
    except WeightsError as e:
        pytest.skip(f"sam2.1-hiera-tiny weights not available: {e}")
    r = model._runner
    enc, dec = r.encoder_graph, r.decode_graphs
    assert enc is not None
    assert dec is not None
    g = torch.Generator().manual_seed(0)
    a, b = (torch.rand(1, 3, 600, 800, generator=g).cuda() for _ in range(2))
    size = (600, 800)
    point = (torch.tensor([[[400.0, 300.0]]], device="cuda"), torch.tensor([[1]], device="cuda"))
    layouts = {  # prompt kwargs of `predict`
        "point": {"points": point[0], "labels": point[1], "boxes": None, "masks": None},
        "box + point": {
            "points": point[0].expand(2, 1, 2),
            "labels": point[1].expand(2, 1),
            "boxes": torch.tensor([[100.0, 80.0, 500.0, 420.0]], device="cuda").expand(2, 4),
            "masks": None,
        },
        "mask": {
            "points": None,
            "labels": None,
            "boxes": None,
            "masks": torch.randn(1, 1, 256, 256, generator=g).cuda(),
        },
    }

    def eager(features: dict[str, torch.Tensor], kw: dict, multimask: bool) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: FBT001
        pts = r._points(size, kw["points"], kw["labels"], kw["boxes"])
        return sam2._Sam2._decode(r, features, pts, kw["masks"], multimask=multimask)

    with torch.inference_mode():
        want = [r.encoder(sam2.prepare(x, r.ctx.io, r.ctx.params.image_size)) for x in (a, b)]
        r.embed(a)  # first call: eager
        assert not enc.graphs
        first = r.embed(a)  # captures
        assert len(enc.graphs) == 1
        second = r.embed(b)  # replays with new data; `first` must not change
        for got, ref in ((first, want[0]), (second, want[1])):
            assert got.keys() == ref.keys()
            for k in ref:
                torch.testing.assert_close(got[k], ref[k], rtol=0, atol=0)
        assert first["image_embed"].data_ptr() != second["image_embed"].data_ptr()
        for name, kw in layouts.items():
            for multimask in (False, True):
                ref = eager(second, kw, multimask)
                for _ in range(2):  # eager, then captured (multimask False) or replayed
                    out = r.predict(second, size, multimask=multimask, **kw)
                torch.testing.assert_close(out["pred_masks"], ref[0], rtol=0, atol=0, msg=f"{name} {multimask}")
                torch.testing.assert_close(out["iou_scores"], ref[1], rtol=0, atol=0, msg=f"{name} {multimask}")
        assert len(dec.graphs) == len(layouts)  # one graph per layout serves both multimask settings
    model.release_graphs()
    assert not enc.graphs
    assert not dec.graphs
    off = promptable_segmentation.load(
        "sam2.1-hiera-tiny", mode="kernels", device="cuda", cuda_graphs=False, cuda_graphs_encoder=True
    )
    assert off._runner.encoder_graph is None
    assert off._runner.decode_graphs is None
