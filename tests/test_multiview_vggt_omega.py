"""VGGT-Omega block preparation, ordered streaming and checkpoint dtype preservation."""

from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.multiview.models.vggt_omega import model
from fastforks.multiview.models.vggt_omega.net.models.layers.block import SelfAttentionBlock


def test_stream_rejects_cpu_before_loading_weights() -> None:
    ctx = SimpleNamespace(params=model.Params(area_side=512, stream=True), device=torch.device("cpu"))
    with pytest.raises(OptionsError, match="stream=True requires CUDA"):
        model.Portable(ctx)  # ty: ignore[invalid-argument-type]


def test_checkpoint_loader_preserves_module_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    def tiny_net(*, patch_size: int, autocast: bool) -> torch.nn.Linear:
        assert not autocast
        return torch.nn.Linear(patch_size, 2)

    monkeypatch.setattr(model, "VGGTOmega", tiny_net)
    state = torch.nn.Linear(model.PATCH, 2).half().state_dict()
    net = model._net(SimpleNamespace(state_dict=lambda: dict(state)))  # ty: ignore[invalid-argument-type]
    torch.testing.assert_close(net.state_dict()["weight"], state["weight"].float(), rtol=0, atol=0)


def test_prepare_bias_mask_and_norm_precisions() -> None:
    src = SelfAttentionBlock(
        32,
        4,
        ffn_ratio=2,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=1.0,
        mask_k_bias=True,
        use_qk_norm=True,
    )
    src.attn.q_norm.weight.data.fill_(0.1234567)
    block = model._Blk(src, torch.device("cpu"), kernels=True)
    torch.testing.assert_close(block.bqkv, (src.attn.qkv.bias * src.attn.qkv.bias_mask).half())
    assert block.qn is not None
    assert block.qnf is not None
    torch.testing.assert_close(block.qnf, block.qn.float(), rtol=0, atol=0)
    assert {"n1w", "g1", "qn", "qnf", "wqkv"} <= dict(block.named_buffers()).keys()


@pytest.mark.gpu
def test_stage_order_and_recovery_after_an_interrupted_session() -> None:
    dev = torch.device("cuda")
    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    x = torch.arange(4, device=dev, dtype=torch.float32)
    expected = [torch.nn.functional.linear(x, layer.weight.cuda(), layer.bias.cuda()) for layer in layers]
    stages = model._StagedCalls(layers, dev)
    calls = [model._Stage(stages, i) for i in range(len(layers))]
    with stages.session(), pytest.raises(RuntimeError, match="requested block 1"):
        calls[1](x)
    with stages.session():
        for call, want in zip(calls, expected, strict=True):
            torch.testing.assert_close(call(x), want, rtol=0, atol=0)
    assert len(stages.source.host.dev) == 2
