"""CPU tests of the timm feature walk (`fastforks.shared.libnet.features`) on small random-weight networks.

Each architecture family the SigLIP 2 / CLIP / Perception Encoder cards use, built with timm at two blocks and a small
width: the walk over the portable-optimised network matches timm's own `forward_features` / `forward_head` and its
module outputs (`norm_pre`, `blocks[0]`), at the native, a resampled and a non-square patch grid. C-RADIOv3: a small
synthetic checkpoint through the runner matches upstream's eval forward, restated from its code.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from fastforks._optimize import optimize
from fastforks.shared.libnet.features import _Walk

timm = pytest.importorskip("timm")

ARCHS = [
    "vit_base_patch16_clip_quickgelu_224",  # cls token, pre-norm, QuickGELU
    "vit_base_patch16_clip_224",  # cls token, GELU
    "vit_base_patch16_siglip_256",  # no prefix tokens, attention pool
    "vit_pe_core_base_patch16_224",  # Eva: RoPE, cls token, attention pool
    "vit_pe_spatial_base_patch16_512",  # Eva: RoPE, mean pool, LayerScale
]


@pytest.mark.parametrize("arch", ARCHS)
# the native 4x4 grid, a resampled 5x5 one, a resampled non-square 3x5 one (rows and columns must not swap)
@pytest.mark.parametrize("side", [(64, 64), (80, 80), (48, 80)])
def test_walk_matches_timm(arch: str, side: tuple[int, int]) -> None:
    torch.manual_seed(0)
    kw = {"img_size": 64, "depth": 2, "embed_dim": 64, "num_heads": 2, "num_classes": 0, "dynamic_img_size": True}
    net: Any = timm.create_model(arch, pretrained=False, **kw).eval()
    for p in net.parameters():  # non-trivial LayerScale / norm parameters
        p.data.add_(0.02 * torch.randn_like(p))
    ref = copy.deepcopy(net)
    hs: dict[int, torch.Tensor] = {}
    ref.norm_pre.register_forward_hook(lambda _m, _i, o: hs.setdefault(0, o))
    ref.blocks[0].register_forward_hook(lambda _m, _i, o: hs.setdefault(1, o))
    x = torch.randn(2, 3, *side)
    with torch.inference_mode():
        want = ref.forward_features(x)
        want_pool = ref.forward_head(want, pre_logits=True)
        walk = _Walk(net)
        fast = optimize(net, mode="portable", device="cpu")
        got, got_hs = walk.run(fast.net, x, (0, 1))
        got_pool = net.forward_head(got, pre_logits=True)  # optimised in place (`fast.net` is `net`)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_pool, want_pool, rtol=1e-4, atol=1e-5)
    for i in (0, 1):
        torch.testing.assert_close(got_hs[i], hs[i], rtol=1e-4, atol=1e-5)


def _radio_reference(
    sd: dict[str, torch.Tensor], x: torch.Tensor, blocks: Any, *, align_corners: bool
) -> tuple[Any, Any]:
    """Upstream C-RADIOv3 eval forward (`RADIOModel.forward`, `ViTPatchGenerator`, CPE mode), restated: (summary, y)."""
    m = "radio_model.model.patch_generator."
    x = (x - sd["radio_model.input_conditioner.norm_mean"]) / sd["radio_model.input_conditioner.norm_std"]
    b, _, hh, ww = x.shape
    py, px = hh // 16, ww // 16
    patches = x.reshape(b, 3, py, 16, px, 16).permute(0, 2, 4, 1, 3, 5).reshape(b, py * px, 3 * 256)  # Im2Patches
    patches = patches @ sd[m + "embedder.weight"].T
    pos = sd[m + "pos_embed"].reshape(1, 128, 128, -1).permute(0, 3, 1, 2)
    if (py, px) != (128, 128):
        mx = max(py, px)
        pos = torch.nn.functional.interpolate(pos, size=(mx, mx), align_corners=align_corners, mode="bilinear")
        pos = pos[..., :py, :px]
    y = torch.cat([sd[m + "cls_token.token"][None].expand(b, -1, -1), patches + pos.flatten(2).permute(0, 2, 1)], 1)
    y = blocks(y)
    return y[:, :4][:, sd["radio_model.summary_idxs"]].flatten(1), y


@pytest.mark.parametrize("align_corners", [False, True])  # upstream: L / g, B / H
@pytest.mark.parametrize("side", [(64, 80), (48, 48)])
def test_c_radio_matches_upstream(side: tuple[int, int], *, align_corners: bool) -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks.features.models.c_radio import model as radio  # noqa: PLC0415

    torch.manual_seed(0)
    d, mean, std = 32, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    ref_net: Any = timm.models.vision_transformer.VisionTransformer(
        img_size=64,
        embed_dim=d,
        depth=2,
        num_heads=2,
        init_values=0.1,
        num_classes=0,
        class_token=False,
        global_pool="",
    ).eval()
    sd = {
        f"radio_model.model.{k.replace('.gamma', '.grandma')}": v
        for k, v in ref_net.blocks.state_dict(prefix="blocks.").items()
    }
    pg = "radio_model.model.patch_generator."
    sd |= {
        pg + "embedder.weight": torch.randn(d, 768) * 0.05,
        pg + "pos_embed": torch.randn(1, 128 * 128, d),
        pg + "cls_token.token": torch.randn(8, d),
        "radio_model.model.reg_token": torch.randn(1, 4, d),
        "radio_model.summary_idxs": torch.tensor([0, 1, 2]),
        "radio_model.input_conditioner.norm_mean": torch.tensor(mean).view(3, 1, 1),
        "radio_model.input_conditioner.norm_std": torch.tensor(std).view(3, 1, 1),
    }
    params = radio.Params(
        library="timm",
        embed_dim=d,
        depth=2,
        num_heads=2,
        layer_scale=True,
        pos_align_corners=align_corners,
        fp16_accumulate=False,
        fp32_accumulate=False,
    )
    ctx = SimpleNamespace(
        params=params,
        mode="portable",
        device=torch.device("cpu"),
        dtype="float32",
        variant=SimpleNamespace(ref="c-radio-tiny@1", io=SimpleNamespace(mean=mean, std=std)),
        state_dict=lambda *_a, **_k: dict(sd),
    )
    run = radio.Portable(ctx)  # type: ignore[arg-type] - a stand-in context
    x = torch.rand(2, 3, *side)
    with torch.inference_mode():
        want_sum, want = _radio_reference(sd, x, ref_net.blocks, align_corners=align_corners)
        norm = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        got = run(norm, ())
    assert run.num_prefix_tokens == 8
    assert run.num_register_tokens == 4
    torch.testing.assert_close(got["last_hidden_state"], want, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got["pooler_output"], want_sum, rtol=1e-4, atol=1e-5)


def test_pe_rope_routes_every_portable_block() -> None:
    """`transformer.pe_rope` (the MPS kernels runner's hook) sends each PE block's q / k through the given function;
    with `_pe_rope` itself the output is the portable one."""
    from fastforks._optimize import transformer  # noqa: PLC0415

    torch.manual_seed(0)
    kw = {"img_size": 64, "depth": 2, "embed_dim": 64, "num_heads": 2, "num_classes": 0, "dynamic_img_size": True}
    net: Any = timm.create_model("vit_pe_core_base_patch16_224", pretrained=False, **kw).eval()
    x = torch.randn(2, 3, 64, 64)
    calls: list[int] = []

    def qk_rope(qkv: torch.Tensor, rope: torch.Tensor, npt: int) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(npt)
        return transformer._pe_rope(qkv, rope, npt)

    with torch.inference_mode():
        walk = _Walk(net)
        fast = optimize(net, mode="portable", device="cpu")
        want, _ = walk.run(fast.net, x, ())
        assert transformer.pe_rope(fast.net, qk_rope) == 2  # both blocks
        got, _ = walk.run(fast.net, x, ())
    assert calls == [1, 1]  # the cls token passes unrotated
    torch.testing.assert_close(got, want, rtol=0, atol=0)
