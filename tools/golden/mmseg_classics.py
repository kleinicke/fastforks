"""mmseg CNN classics (ADE20K) goldens from mmsegmentation itself.

Reference: mmsegmentation v1.2.2 (+ mmcv 2.1.0, mmengine 0.10.7) as installed in another environment, appended to
`sys.path` after this one (same torch): `mmseg.apis.init_model(<config>)` builds the network from the variant's
model-zoo config, the checkpoint (`state_dict`, auxiliary head included) is loaded strictly. CPU float32 on
`tests/synth.py`'s image with the configs' test protocol written out here independently of `fastforks.shared`:
keep-ratio resize to fit `2048 x 512` (bilinear, half-pixel centres, on the float image), normalisation with the
configs' `mean` / `std`, `EncoderDecoder.encode_decode` (the decode head's logits resized to the input size by mmseg's
`predict_by_feat`), then resized to the image size (`postprocess_result`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/mmseg_classics.py MMSEG_SITE_PACKAGES [VARIANT ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from _common import image, write

from fastforks._core import registry

FAMILY = "mmseg"
TASK = "segmentation"
UPSTREAM = "https://github.com/open-mmlab/mmsegmentation"
REF_COMMIT = "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8"  # v1.2.2
CONFIGS = {  # variant -> model-zoo config (mmseg/.mim/configs)
    "fcn-r101-d8-ade20k": "fcn/fcn_r101-d8_4xb4-160k_ade20k-512x512.py",
    "pspnet-r101-d8-ade20k": "pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py",
    "deeplabv3-r101-d8-ade20k": "deeplabv3/deeplabv3_r101-d8_4xb4-160k_ade20k-512x512.py",
    "deeplabv3plus-r101-d8-ade20k": "deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py",
    "hrnet-w48-fcn-ade20k": "hrnet/fcn_hr48_4xb4-160k_ade20k-512x512.py",
}
MEAN = (123.675, 116.28, 103.53)  # fixed: the configs' data_preprocessor
STD = (58.395, 57.12, 57.375)  # fixed: the configs' data_preprocessor


def _size(h: int, w: int, long: int = 2048, short: int = 512) -> tuple[int, int]:
    """Mmcv `rescale(keep_ratio=True)` to fit `long x short`."""
    s = min(long / max(h, w), short / min(h, w))
    return int(h * s + 0.5), int(w * s + 0.5)


def reference(site: str, card: Any, variant: Any) -> Any:  # noqa: ANN401 - registry records -> an mmseg model
    """The mmseg `EncoderDecoder` of a variant (CPU, float32, eval) with its checkpoint loaded strictly."""
    if site not in sys.path:
        sys.path.append(site)
    import mmseg  # noqa: PLC0415  # reference site-packages
    from mmseg.apis import init_model  # noqa: PLC0415

    configs = Path(mmseg.__file__).parent / ".mim" / "configs"
    model = init_model(str(configs / CONFIGS[variant.id]), None, device="cpu").eval()
    (ckpt,) = registry.weights(card, variant)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    return model


def whole(model: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - an mmseg model
    """Logits `[B, 150, H, W]` of float RGB `x` in `[0, 1]` under the configs' test protocol."""
    h, w = x.shape[-2:]
    rh, rw = _size(h, w)
    y = F.interpolate(x, (rh, rw), mode="bilinear", align_corners=False) * 255
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    s = model.encode_decode((y - mean) / std, [{"img_shape": (rh, rw), "pad_shape": (rh, rw), "ori_shape": (h, w)}])
    return F.interpolate(s, (h, w), mode="bilinear", align_corners=False)


def main() -> None:
    wanted = set(sys.argv[2:])
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    for card, variant in registry.cards(TASK).values():
        if card.family != FAMILY or (wanted and variant.id not in wanted):
            continue
        model = reference(sys.argv[1], card, variant)
        import mmseg  # noqa: PLC0415  # on sys.path now

        with torch.inference_mode():
            logits = whole(model, x)
        write(
            variant.ref,
            {"logits": logits[0].numpy()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, mmsegmentation {mmseg.__version__}, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  {variant.ref}: classes {sorted(set(logits[0].argmax(0).ravel().tolist()))[:8]}")


if __name__ == "__main__":
    main()
