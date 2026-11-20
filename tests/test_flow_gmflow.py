"""GMFlow: the card's variants, the forward pass on random weights (batch, three configurations), chunked attention,
padding of odd sizes, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.gmflow import model as family
from fastforks.flow.models.gmflow.net import matching
from fastforks.flow.models.gmflow.net.attention import sdpa
from fastforks.flow.models.gmflow.net.unimatch import UniMatch
from fastforks.flow.models.gmflow.net.utils import normalize_img

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {  # id -> (scales, refine, pad multiple)
    "gmflow-scale1-things": (1, 0, 16),
    "gmflow-scale1-mixdata": (1, 0, 16),
    "gmflow-scale2-things": (2, 0, 32),
    "gmflow-scale2-sintel": (2, 0, 32),
    "gmflow-scale2-mixdata": (2, 0, 32),
    "gmflow-scale2-regrefine6-things": (2, 6, 32),
    "gmflow-scale2-regrefine6-sintelft": (2, 6, 32),
    "gmflow-scale2-regrefine6-kitti": (2, 6, 32),
    "gmflow-scale2-regrefine6-mixdata": (2, 6, 32),
}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def _net(scales: int, refine: int) -> UniMatch:
    torch.manual_seed(0)
    return UniMatch(num_scales=scales, upsample_factor=8 // scales, reg_refine=refine > 0, task="flow").eval()


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "gmflow"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (scales, refine, multiple) in VARIANTS.items():
        card, variant = cards[vid]
        params = card.params | variant.params
        assert (params["scales"], params["refine"], variant.io.pad_multiple) == (scales, refine, multiple), vid
        assert ("unstated" in variant.licence.get("training_data", ())) == vid.endswith(("-mixdata", "-kitti")), vid
    assert set(flow.list_models()) >= set(VARIANTS)


@pytest.mark.parametrize(("scales", "refine"), [(1, 0), (2, 0), (2, 2)])
def test_forward_shape(scales: int, refine: int) -> None:
    net = _net(scales, refine)
    params = family.Params(scales=scales, refine=refine)
    a, b = _pair(64, 96, batch=2)
    with torch.inference_mode():
        out = family.forward(net, a, b, params, fp16=False)
        one = family.forward(net, a[1:], b[1:], params, fp16=False)
    assert out.shape == (2, 2, 64, 96)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], one, rtol=0, atol=1e-4)  # the items of a batch do not mix


def test_forward_is_upstreams() -> None:
    """Our forward pass against the vendored `UniMatch.forward` (same weights, upstream's evaluation settings)."""
    net = _net(2, 3)
    a, b = _pair(64, 96)
    with torch.inference_mode():
        ours = family.forward(net, *normalize_img(a * 255, b * 255), family.Params(scales=2, refine=3), fp16=False)
        ref = net(
            a * 255,
            b * 255,
            attn_type="swin",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=3,
            task="flow",
        )["flow_preds"][-1]
    # the matching softmax runs as a fused SDPA kernel here and as matmul + softmax in the vendored
    # global_correlation_softmax
    # (other summation order): random weights amplify the float32 rounding through 3 refinements to about 2e-3 px
    torch.testing.assert_close(ours, ref, rtol=0, atol=5e-3)


def test_fused_attention_is_softmax_matmul() -> None:
    torch.manual_seed(0)
    query, key, value = torch.randn(2, 96, 16), torch.randn(2, 96, 16), torch.randn(2, 96, 2)
    plain = torch.softmax(query @ key.transpose(1, 2) / 16**0.5, dim=-1) @ value
    torch.testing.assert_close(family.attend(query, key, value), plain, rtol=0, atol=1e-5)


@pytest.mark.parametrize("mask_kind", ["full", "broadcast"])
def test_cpu_attention_chunks_match_upstream(mask_kind: str) -> None:
    """Multiple query chunks preserve upstream attention, including broadcast masks."""
    gen = torch.Generator().manual_seed(3)
    query = torch.randn(2, 530, 16, generator=gen)
    key = torch.randn(2, 80, 16, generator=gen)
    value = torch.randn(2, 80, 2, generator=gen)
    mask = torch.randn(2, 1 if mask_kind == "broadcast" else 530, 80, generator=gen)
    expected = torch.softmax(query @ key.transpose(-2, -1) / 16**0.5 + mask, dim=-1) @ value
    torch.testing.assert_close(sdpa(query, key, value, mask), expected, rtol=0, atol=1e-6)


@pytest.mark.gpu
def test_cuda_local_correlation_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    gen = torch.Generator(device="cuda").manual_seed(3)
    first, second = (torch.randn(2, 16, 11, 17, device="cuda", generator=gen) for _ in range(2))
    displacement = torch.randn(2, 2, 11, 17, device="cuda", generator=gen)

    with torch.inference_mode():
        expected_flow, expected_prob = matching.local_correlation_softmax(first, second, 4)
        expected_refine = matching.local_correlation_with_flow(first, second, displacement, 4)
        monkeypatch.setattr(matching, "_LOCAL_SAMPLE_BYTES", 128 * 1024)
        actual_flow, actual_prob = matching.local_correlation_softmax(first, second, 4)
        actual_refine = matching.local_correlation_with_flow(first, second, displacement, 4)

    for actual, expected in (
        (actual_flow, expected_flow),
        (actual_prob, expected_prob),
        (actual_refine, expected_refine),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=8e-6)


@pytest.mark.slow
def test_cpu_attention_exports_dynamic_query_length() -> None:
    """The bounded eager path must not specialize the exported query length."""

    class Attention(torch.nn.Module):
        def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return sdpa(query, key, value)

    model = Attention()
    query, key, value = torch.randn(2, 300, 16), torch.randn(2, 80, 16), torch.randn(2, 80, 2)
    length = torch.export.Dim("query_length", min=1, max=600)
    exported = torch.export.export(
        model, (query, key, value), dynamic_shapes={"query": {1: length}, "key": {}, "value": {}}, strict=False
    ).module()
    for size in (120, 530):
        query = torch.randn(2, size, 16)
        torch.testing.assert_close(exported(query, key, value), model(query, key, value), rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scale1 checkpoint with random weights, as `load(weights=...)` takes it."""
    path = tmp_path_factory.mktemp("gmflow") / "random.pth"
    torch.save({"model": _net(1, 0).state_dict()}, path)  # the layout of the authors' files
    return path


def test_odd_size_is_padded_and_exports(weights: Path) -> None:
    model = flow.load("gmflow-scale1-things", weights=weights, device="cpu")
    a, b = _pair(67, 93)  # neither a multiple of 16: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (1, 2, 67, 93)
    assert torch.isfinite(out).all()
    a, b = _pair(64, 96)
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, model(a, b).flow, rtol=0, atol=1e-5)


@pytest.mark.gpu
def test_cuda_single_head_attention_onnx(tmp_path: Path) -> None:
    """CUDA ONNX export accepts a rank-three head and retains its masking and output."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    class Attention(torch.nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return sdpa(q, k, v, mask)

    torch.manual_seed(14)
    q = torch.randn(2, 8, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 8, 16, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 8, 2, device="cuda", dtype=torch.float16)
    mask = torch.zeros(2, 8, 8, device="cuda", dtype=torch.float16)
    mask[:, :, -1] = -100
    model = Attention().eval()
    with torch.inference_mode():
        want = model(q, k, v, mask)
        torch.onnx.export(model, (q, k, v, mask), tmp_path / "attention.onnx", dynamo=True)
    session = ort.InferenceSession(str(tmp_path / "attention.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(
        None, {i.name: x.cpu().numpy() for i, x in zip(session.get_inputs(), (q, k, v, mask), strict=True)}
    )
    torch.testing.assert_close(torch.from_numpy(got[0]), want.cpu(), rtol=3e-3, atol=3e-3)
