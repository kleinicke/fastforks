"""LaMa (big-lama) inpainting goldens from the reference implementation.

Reference: upstream advimman/lama (`REF_COMMIT`): `FFCResNetGenerator` (saicinpainting/training/modules/ffc.py) with
big-lama's `config.yaml` generator settings, on the CPU in float32, called as `bin/predict.py` with
`configs/prediction/default.yaml` (no refinement): image and mask / 255 padded with upstream `pad_img_to_modulo(.., 8)`
(numpy symmetric), the mask binarised (`> 0`), `DefaultInpaintingTrainingModule.forward`'s masked image + mask into
the generator and its paste-back (`mask * predicted + (1 - mask) * image`), cropped, no uint8 rounding. Input:
`tests/synth.py image(101, 151)` and the mask `MASK` (a box and a diagonal stroke), stored as `input` / `mask`
(0 / 255); output `image` `[3, H, W]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/lama.py LAMA_CHECKOUT BIG_LAMA_DIR

BIG_LAMA_DIR is the unpacked big-lama.zip (`config.yaml`, `models/best.ckpt`). Run with `CUDA_VISIBLE_DEVICES=`.
The modules are imported from the exported tree unmodified; pytorch_lightning, kornia and cv2 (imported by modules
the generator does not use) are stubs. The Lightning checkpoint is read with fastforks' restricted unpickler.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, reference_tree, write

from fastforks._core.checkpoint import Inert, config_globals, load_state_dict

UPSTREAM = "https://github.com/advimman/lama"
REF_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
SIZE = (101, 151)
GENERATOR = {  # big-lama config.yaml `generator` (kind ffc_resnet)
    "input_nc": 4,
    "output_nc": 3,
    "ngf": 64,
    "n_downsampling": 3,
    "n_blocks": 18,
    "add_out_act": "sigmoid",
    "init_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
    "downsample_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
    "resnet_conv_kwargs": {"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
}


def mask(h: int, w: int) -> np.ndarray:
    """uint8 `[h, w]` mask, 255 = hole: a box in the upper left and a 7 px diagonal stroke."""
    m = np.zeros((h, w), np.uint8)
    m[h // 5 : h // 2, w // 6 : w // 2] = 255
    ys, xs = np.mgrid[:h, :w]
    m[np.abs((ys / h) - (1 - xs / w)) * min(h, w) < 3.5] = 255  # noqa: PLR2004 - half the stroke width
    return m


def _imports(checkout: Path) -> tuple[Any, Any]:
    """Upstream `FFCResNetGenerator` and `pad_img_to_modulo` from the exported tree (unused imports stubbed)."""
    stubs = {
        "pytorch_lightning": {"seed_everything": None},
        "kornia": {},
        "kornia.geometry": {},
        "kornia.geometry.transform": {"rotate": None},
        "cv2": {"INTER_AREA": 3},  # a default argument of an unused function in data.py
    }
    for name, attrs in stubs.items():
        sys.modules.setdefault(name, types.ModuleType(name)).__dict__.update(attrs)
    tree = reference_tree(checkout, REF_COMMIT)
    sys.path.insert(0, str(tree))
    from saicinpainting.training.modules.ffc import FFCResNetGenerator  # noqa: PLC0415 - the exported tree

    spec = importlib.util.spec_from_file_location("lama_data", tree / "saicinpainting" / "evaluation" / "data.py")
    data = importlib.util.module_from_spec(spec)  # by path: the package's __init__ imports the evaluator stack
    spec.loader.exec_module(data)
    return FFCResNetGenerator, data.pad_img_to_modulo


def build(checkout: Path, ckpt: Path, device: torch.device) -> tuple[torch.nn.Module, Any]:
    """(generator in eval mode on `device`, upstream `pad_img_to_modulo`); `ckpt` = big-lama `models/best.ckpt`."""
    generator, pad = _imports(checkout)
    extra = {**config_globals(), ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"): Inert}
    sd = load_state_dict(ckpt, key="state_dict", strip="", extra=extra)
    net = generator(**GENERATOR).eval()
    net.load_state_dict({k.removeprefix("generator."): v for k, v in sd.items() if k.startswith("generator.")})
    return net.to(device), pad


def pad_inputs(pad: Any, img: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401
    """predict.py's dataset step: CHW float image and `[1, H, W]` mask / 255, padded symmetrically to 8."""
    return pad(img.astype(np.float32) / 255, 8), pad(m[None].astype(np.float32) / 255, 8)


def run(net: torch.nn.Module, x: torch.Tensor, mk: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """predict.py's model step on padded `[1, 3, H, W]` / `[1, 1, H, W]`: mask > 0, masked input, paste-back, crop."""
    mi = (mk > 0).float()
    pred = net(torch.cat([x * (1 - mi), mi], 1))
    return (mi * pred + (1 - mi) * x)[0, :, : size[0], : size[1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of advimman/lama")
    ap.add_argument("big_lama", type=Path, help="unpacked big-lama.zip")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    net, pad = build(a.checkout, a.big_lama / "models" / "best.ckpt", torch.device("cpu"))
    img, m = image(*SIZE), mask(*SIZE)
    x, mk = pad_inputs(pad, np.transpose(img, (2, 0, 1)), m)
    with torch.inference_mode():
        out = run(net, torch.from_numpy(x)[None], torch.from_numpy(mk)[None], SIZE).numpy()
    write(
        "big-lama@1",
        {"input": img.astype(np.float32), "mask": m.astype(np.float32), "image": out},
        family="lama",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, upstream FFCResNetGenerator (big-lama config), predict.py padding (symmetric to 8), "
        f"mask > 0, masked image + mask, paste-back, crop, no uint8 rounding; tests/synth.py image{SIZE} + "
        f"tools/golden/lama.py mask(); torch {torch.__version__}",
        task="restoration",
    )
    print(f"  big-lama: image {out.shape}, hole mean {out[:, m > 0].mean():.4f}, hole px {(m > 0).mean():.3f}")


if __name__ == "__main__":
    main()
