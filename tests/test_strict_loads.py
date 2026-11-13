"""The strict weight paths of the library-built vision families (timm, torchvision, transformers, DINOv2 head).

Tiny random networks stand in for real checkpoints, so these run on the CPU without downloads; the libraries are
optional dependencies, their cases skip when missing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforks._core.errors import CheckpointMismatchError
from fastforks.segmentation.models.dinov2_linear.model import _check_head
from fastforks.shared import libnet, seginfer

CHANGES = ["add", "drop", "shape"]


def _perturb(sd: dict[str, torch.Tensor], change: str) -> dict[str, torch.Tensor]:
    sd = dict(sd)
    big = max((k for k, v in sd.items() if v.is_floating_point() and v.ndim >= 1), key=lambda k: sd[k].numel())
    if change == "add":
        sd["zz_unexpected.weight"] = torch.zeros(3)
    elif change == "drop":
        del sd[big]
    else:
        sd[big] = torch.cat([sd[big], sd[big][:1]])
    return sd


def _ctx(sd: dict[str, torch.Tensor], **params: Any) -> SimpleNamespace:
    return SimpleNamespace(
        params=SimpleNamespace(**params),
        variant=SimpleNamespace(ref="tiny@1"),
        state_dict=lambda *_a, **_k: sd,
    )


def _match(change: str) -> str:
    return {"add": "1 unexpected keys", "drop": "1 missing keys", "shape": "1 shape mismatches"}[change]


@pytest.mark.parametrize("change", [None, *CHANGES])
def test_torchvision_loads_strictly(change):
    tv = pytest.importorskip("torchvision")
    sd = tv.models.get_model("shufflenet_v2_x0_5", weights=None).state_dict()
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}  # old torchvision checkpoints
    ctx = _ctx(sd if change is None else _perturb(sd, change), library="torchvision", arch="shufflenet_v2_x0_5")
    if change is None:
        assert libnet.build(ctx).training is False
        return
    with pytest.raises(CheckpointMismatchError, match=_match(change)):
        libnet.build(ctx)


@pytest.mark.parametrize("change", [None, *CHANGES])
def test_timm_loads_strictly(change):
    timm = pytest.importorskip("timm")
    sd = timm.create_model("mobilenetv3_small_050", pretrained=False).state_dict()
    ctx = _ctx(sd if change is None else _perturb(sd, change), library="timm", arch="mobilenetv3_small_050.lamb_in1k")
    if change is None:
        assert libnet.build(ctx).training is False
        return
    with pytest.raises(CheckpointMismatchError, match=_match(change)):
        libnet.build(ctx)


def _segformer(tmp_path):
    tf = pytest.importorskip("transformers")
    cfg = tf.SegformerConfig(
        depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 24, 32], num_attention_heads=[1, 1, 1, 1], decoder_hidden_size=16,
        num_labels=5,
    )  # fmt: skip
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    return tf.SegformerForSemanticSegmentation(cfg).state_dict(), path


@pytest.mark.parametrize("change", [None, *CHANGES])
def test_transformers_loads_strictly(change, tmp_path):
    sd, cfg = _segformer(tmp_path)
    ctx = _ctx(sd if change is None else _perturb(sd, change), arch="SegformerForSemanticSegmentation")
    ctx.weights = lambda: [cfg]
    if change is None:
        assert seginfer.pretrained(ctx, config=0, weights=0).training is False
        return
    with pytest.raises(CheckpointMismatchError, match=_match(change)):
        seginfer.pretrained(ctx, config=0, weights=0)


def _tied_heads(*, by_parameter: bool = False) -> torch.nn.Module:
    """Per-layer heads tied to head 0, their list shared with the decoder on purpose.

    `by_parameter`: MM-Grounding-DINO's layout in transformers 5.19, own modules sharing head 0's parameters.
    """
    net, net.dec = torch.nn.Module(), torch.nn.Module()
    heads = [torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)) for _ in range(3)]
    for h in heads[1:]:
        for j in range(2):
            if by_parameter:
                h[j].weight, h[j].bias = heads[0][j].weight, heads[0][j].bias
    net.heads = net.dec.heads = torch.nn.ModuleList(heads if by_parameter else [heads[0]] * 3)
    return net


def _checkpoint() -> dict[str, torch.Tensor]:
    """Different values per layer; layer 0 stored under the top-level name, the others under the decoder's."""
    keys = [f"{i}.{j}.{p}" for i in range(3) for j in range(2) for p in ("weight", "bias")]
    return {
        f"{'heads' if k[0] == '0' else 'dec.heads'}.{k}": torch.full((2, 2) if k.endswith("weight") else (2,), float(n))
        for n, k in enumerate(keys)
    }


@pytest.mark.parametrize("by_parameter", [False, True])
def test_untie_restores_every_checkpoint_head(by_parameter):
    """A transformers tie the config does not ask for (5.19, MM-Grounding-DINO) must not drop checkpoint values."""
    net, sd = _tied_heads(by_parameter=by_parameter), _checkpoint()
    net.load_state_dict(sd, strict=False)  # the shared head keeps the last layer's values
    seginfer.untie_to_checkpoint(net, sd)
    own = net.state_dict()
    assert all(torch.equal(own[k], v) for k, v in sd.items())
    assert all(torch.equal(own[k], own[f"dec.{k}"]) for k in own if k.startswith("heads."))
    assert net.dec.heads is net.heads
    assert seginfer.untie_to_checkpoint(net, sd) == []


def test_untie_raises_when_a_value_cannot_be_restored():
    net, sd = _tied_heads(), _checkpoint()
    sd = {k: v for k, v in sd.items() if k.startswith("heads.0")} | {"heads.0.0.weight": torch.zeros(2, 2)}
    net.register_load_state_dict_pre_hook(lambda _m, d, *_a: d.pop("heads.0.0.weight", None))
    with pytest.raises(CheckpointMismatchError, match="tensors differ after loading"):
        seginfer.untie_to_checkpoint(net, sd)


SWIN = "model.pixel_level_module.encoder.swin"


@pytest.mark.parametrize(
    "key",
    [f"{SWIN}.layernorm.weight", f"{SWIN}.layernorm.bias"],
)
def test_swin_final_norm_is_the_only_exempt_missing_key(key):
    assert seginfer._UNUSED_MISSING.fullmatch(key)
    for other in (
        "x" + key,
        key + "x",
        key.replace("swin.", ""),
        key.replace("layernorm", "layernorm2"),
        "model.pixel_level_module.decoder.layernorm.weight",
        f"{SWIN}.encoder.layers.0.blocks.0.layernorm_before.weight",
    ):
        assert not seginfer._UNUSED_MISSING.fullmatch(other), other


def test_relative_position_index_is_the_only_exempt_unexpected_key():
    ok = f"{SWIN}.encoder.layers.3.blocks.17.attention.self.relative_position_index"
    assert seginfer._UNUSED_UNEXPECTED.fullmatch(ok)
    for other in (
        "x" + ok,
        ok + "x",
        ok.replace("relative_position_index", "xrelative_position_index"),
        ok.replace("swin.encoder", "encoder"),
        "model.transformer_module.layers.0.relative_position_index",
        ok.replace("blocks.17", "blocks"),
    ):
        assert not seginfer._UNUSED_UNEXPECTED.fullmatch(other), other


def _head(dim: int = 4, classes: int = 3) -> dict[str, torch.Tensor]:
    return {
        "decode_head.conv_seg.weight": torch.zeros(classes, dim, 1, 1),
        "decode_head.conv_seg.bias": torch.zeros(classes),
        "decode_head.bn.weight": torch.ones(dim),
        "decode_head.bn.bias": torch.zeros(dim),
        "decode_head.bn.running_mean": torch.zeros(dim),
        "decode_head.bn.running_var": torch.ones(dim),
        "decode_head.bn.num_batches_tracked": torch.tensor(0),
    }


def test_dinov2_head_exact_keys_pass():
    assert _check_head(_head(), "head") is not None


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda sd: sd.pop("decode_head.bn.running_var"), "1 missing keys"),
        (lambda sd: sd.update({"decode_head.extra.weight": torch.zeros(2)}), "1 unexpected keys"),
        (lambda sd: sd.update({"decode_head.bn.bias": torch.zeros(5)}), "1 shape mismatches"),
    ],
)
def test_dinov2_head_other_keys_or_shapes_raise(edit, message):
    sd = _head()
    edit(sd)
    with pytest.raises(CheckpointMismatchError, match=message):
        _check_head(sd, "head")


@pytest.fixture(scope="module")
def raft_sd() -> dict[str, torch.Tensor]:
    from fastforks.stereo.models.raft_stereo.model import RAFTStereo, _args  # noqa: PLC0415

    return RAFTStereo(_args()).state_dict()


def test_raft_kernels_weights_exact_keys_pass(raft_sd):
    from fastforks.stereo.models.raft_stereo.model import check_weights  # noqa: PLC0415

    check_weights(raft_sd, "raft@1 checkpoint")


@pytest.mark.parametrize("change", [*CHANGES, "flow_head"])
def test_raft_kernels_weights_mismatch_raises(raft_sd, change):
    """The kernels converter reads tensors by name: extra keys, other shapes (a 3-channel flow head was sliced)."""
    from fastforks.stereo.models.raft_stereo.model import check_weights  # noqa: PLC0415

    if change == "flow_head":
        sd = dict(raft_sd)
        sd["update_block.flow_head.conv2.weight"] = torch.zeros(3, 256, 3, 3)
    else:
        sd = _perturb(raft_sd, change)
    with pytest.raises(CheckpointMismatchError, match="raft@1 checkpoint does not fit"):
        check_weights(sd, "raft@1 checkpoint")
