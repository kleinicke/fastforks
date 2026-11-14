"""BiSeNet face-parsing goldens from facexlib's network and face-parsing.PyTorch's test protocol.

Reference: facexlib (`REF_COMMIT`) `facexlib/parsing/bisenet.py` + `resnet.py`, loaded by path from an exported tree
(unmodified; the network of zllrunning/face-parsing.PyTorch), with the card's checkpoint loaded strictly. CPU float32
on `tests/synth.py`'s image with test.py's protocol written out here independently of `fastforks.shared`: each channel
resized to 512 x 512 by PIL `BILINEAR` (float images, mode `F`), `Normalize((0.485, 0.456, 0.406), (0.229, 0.224,
0.225))`, `net(x)[0]`; then the logits resized to the image size (bilinear, half-pixel centres), as fastforks returns
them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/bisenet.py FACEXLIB_CHECKOUT [VARIANT ...]
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write
from PIL import Image

from fastforks._core import registry

FAMILY = "bisenet"
TASK = "segmentation"
UPSTREAM = "https://github.com/xinntao/facexlib"
REF_COMMIT = "260620ae93990a300f4b16448df9bb459f1caba9"
SIZE = 512  # fixed: test.py
MEAN = (0.485, 0.456, 0.406)  # fixed: test.py
STD = (0.229, 0.224, 0.225)  # fixed: test.py


def build(checkout: Path, weights: Path) -> torch.nn.Module:
    """Facexlib's `BiSeNet(num_class=19)` with `weights` (CPU, float32, eval)."""
    if "ffref_parsing" not in sys.modules:
        tree = reference_tree(checkout, REF_COMMIT) / "facexlib" / "parsing"
        pkg = types.ModuleType("ffref_parsing")
        pkg.__path__ = [str(tree)]
        sys.modules["ffref_parsing"] = pkg
        for name in ("resnet", "bisenet"):
            spec = importlib.util.spec_from_file_location(f"ffref_parsing.{name}", tree / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    net = sys.modules["ffref_parsing.bisenet"].BiSeNet(num_class=19)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """Float RGB `[H, W, 3]` in [0, 1] -> the network input `[1, 3, 512, 512]` (PIL bilinear per channel)."""
    chans = [np.asarray(Image.fromarray(rgb[..., c].astype(np.float32), mode="F").resize((SIZE, SIZE), Image.BILINEAR))
             for c in range(3)]  # fmt: skip
    x = torch.from_numpy(np.stack(chans))[None]
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def logits(net: torch.nn.Module, rgb: np.ndarray) -> torch.Tensor:
    """Logits `[1, 19, H, W]` of float RGB `[H, W, 3]` in [0, 1]."""
    out = net(preprocess(rgb))[0]
    return F.interpolate(out, rgb.shape[:2], mode="bilinear", align_corners=False)


def main() -> None:
    wanted = set(sys.argv[2:])
    rgb = image().astype(np.float32) / 255
    for card, variant in registry.cards(TASK).values():
        if card.family != FAMILY or (wanted and variant.id not in wanted):
            continue
        (weights,) = registry.weights(card, variant)
        net = build(Path(sys.argv[1]), weights)
        with torch.inference_mode():
            out = logits(net, rgb)
        write(
            variant.ref,
            {"logits": out[0].numpy()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, PIL bilinear 512 x 512 per channel, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  {variant.ref}: classes {sorted(set(out[0].argmax(0).ravel().tolist()))}")


if __name__ == "__main__":
    main()
