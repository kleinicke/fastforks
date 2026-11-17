"""Goldens of C-RADIOv3 / v4 (features).

Reference: NVIDIA's own code, `transformers.AutoModel.from_pretrained("nvidia/C-RADIOv<3|4>-<size>", revision=<the
variant's weight revision>, trust_remote_code=True)` (`hf_model.RADIOModel`; needs `einops`), CPU float32, eval.
Input: the synthetic image resized as `fastforks.features` does (shorter side 112, each side the nearest multiple of
16, antialiased bicubic, clamped) in [0, 1]: upstream standardises it itself. Outputs: `last_hidden_state` = the
final norm's output (all tokens; upstream's `features` = the patches after the prefix tokens, checked),
`pooler_output` = upstream's `summary`, `hidden_states.0` = the first block's input, `.1` = its output (hooks).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/c_radio.py [REF ...]
"""

from __future__ import annotations

import re
import sys
from typing import Any

import torch
import torch.nn.functional as F
from _common import image, write

from fastforks._core import registry

TASK = "features"
FAMILY = "c_radio"
SIZE = 112  # tests/test_golden.py FEATURES_SIZE
LAYERS = (0, 1)  # tests/test_golden.py FEATURES_LAYERS
_URL = re.compile(
    r"^https://huggingface\.co/(nvidia/C-RADIOv[34]-[A-Za-z0-9]+)/resolve/([0-9a-f]{40})/model\.safetensors$"
)


def _input(multiple: int) -> torch.Tensor:
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    s, m = SIZE / min(h, w), multiple
    hw = (max(m, round(h * s / m) * m), max(m, round(w * s / m) * m))
    return F.interpolate(x, size=hw, mode="bicubic", antialias=True).clamp(0, 1)


def _reference(model: Any, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: ANN401 - hf_model.RADIOModel
    """Upstream's outputs of `x` and the block / norm tensors (forward hooks)."""
    vit = model.radio_model.model
    hs: dict[int, torch.Tensor] = {}
    hooks = [
        vit.blocks[0].register_forward_pre_hook(lambda _m, a: hs.setdefault(0, a[0])),
        vit.blocks[0].register_forward_hook(lambda _m, _a, o: hs.setdefault(1, o)),
        vit.norm.register_forward_hook(lambda _m, _a, o: hs.setdefault(-1, o)),
    ]
    with torch.inference_mode():
        out = model(x)
    for h in hooks:
        h.remove()
    prefix = vit.patch_generator.num_cls_tokens + vit.patch_generator.num_registers
    torch.testing.assert_close(out.features, hs[-1][:, prefix:], rtol=0, atol=0)  # the token layout we document
    return {"last_hidden_state": hs[-1], "pooler_output": out.summary} | {f"hidden_states.{i}": hs[i] for i in LAYERS}


def main() -> None:
    import transformers  # noqa: PLC0415 - optional dependency

    want = set(sys.argv[1:])
    for card, variant in registry.cards(TASK).values():
        if card.family != FAMILY or (want and variant.id not in want):
            continue
        url = _URL.match(variant.weights[0].url)
        if url is None:
            msg = f"{variant.ref}: unexpected weight url {variant.weights[0].url}"
            raise SystemExit(msg)
        repo, rev = url[1], url[2]
        model: Any = transformers.AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True)
        model = model.float().eval()
        x = _input(variant.io.pad_multiple)
        maps = _reference(model, x)
        last, summary = maps["last_hidden_state"], maps["pooler_output"]
        h, w = x.shape[-2:]
        write(
            variant.ref,
            {k: v[0].numpy() for k, v in maps.items()},
            family=FAMILY,
            upstream=f"https://huggingface.co/{repo}@{rev}",
            patch=None,
            settings=f"CPU float32, transformers {transformers.__version__}, {h}x{w} input, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  {variant.ref}: tokens {tuple(last.shape)}, summary {tuple(summary.shape)}")


if __name__ == "__main__":
    main()
