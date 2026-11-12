"""BiRefNet goldens from the weights' own code (the HF repos' `birefnet.py`).

Reference: `birefnet.py` + `BiRefNet_config.py` of the HF snapshot (ZhengPeng7/BiRefNet `HF_REVISION["birefnet"]`,
ZhengPeng7/BiRefNet_lite `HF_REVISION["birefnet-lite"]`, unpatched; needs transformers, timm, kornia, einops), CPU
float32, the safetensors loaded strictly, on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored
as `input`; the synthetic test image is all background for one model and all foreground for the other). Pre- and
post-processing as the HF model card on float tensors: resize to 1024 x 1024 (bilinear, antialiased: torchvision
`Resize` on tensors), ImageNet normalisation, `model(x)[-1].sigmoid()`, bicubic resize back to the image size, clamped
to [0, 1] (the card resizes the 8-bit mask with PIL; float keeps the golden free of rounding).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/birefnet.py birefnet <HF snapshot> <weights.safetensors> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import write
from PIL import Image

UPSTREAM = "https://huggingface.co/ZhengPeng7"
HF_REVISION = {
    "birefnet": ("BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"),
    "birefnet-lite": ("BiRefNet_lite", "aa62cd87eafb9cc43056d08ef3615a14628b831d"),
}
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
SIZE = 1024
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def network(snapshot: Path, weights: Path) -> Any:  # noqa: ANN401 - the HF model class
    """The HF `BiRefNet` of `snapshot` (its `birefnet.py`) with `weights`, CPU float32, eval."""
    from safetensors.torch import load_file  # noqa: PLC0415

    pkg = Path(tempfile.mkdtemp(prefix="ff-birefnet-")) / "hf_birefnet"
    pkg.mkdir()
    for f in ("birefnet.py", "BiRefNet_config.py"):
        shutil.copy(snapshot / f, pkg / f)
    (pkg / "__init__.py").write_text("")
    for k in [k for k in sys.modules if k.split(".")[0] == "hf_birefnet"]:  # a second snapshot in one process
        del sys.modules[k]
    sys.path.insert(0, str(pkg.parent))
    mod = importlib.import_module("hf_birefnet.birefnet")
    cfg = importlib.import_module("hf_birefnet.BiRefNet_config")
    sys.path.remove(str(pkg.parent))
    net = mod.BiRefNet(bb_pretrained=False, config=cfg.BiRefNetConfig(bb_pretrained=False))
    net.load_state_dict(load_file(weights), strict=True)
    return net.eval()


def predict(net: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - the HF model class
    """`[B, 3, H, W]` RGB in [0, 1] -> alpha `[B, H, W]` float32 (the network's dtype and device)."""
    p = next(net.parameters())
    y = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    y = net(((y - mean) / std).to(p.dtype))[-1].float().sigmoid()
    return F.interpolate(y, x.shape[-2:], mode="bicubic", align_corners=False)[:, 0].clamp(0, 1)


def main() -> None:
    name, snapshot, weights, png = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    repo, rev = HF_REVISION[name]
    net = network(snapshot, weights)
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        alpha = predict(net, x)[0].numpy().astype(np.float32)
    write(
        f"{name}@1",
        {"input": img.astype(np.float32), "alpha": alpha},
        family="birefnet",
        upstream=f"{UPSTREAM}/{repo}@{rev}",
        patch=None,
        settings=(
            f"CPU float32, HF birefnet.py + weights {repo}@{rev}, {SIZE}x{SIZE} bilinear antialiased, ImageNet "
            f"normalisation, sigmoid, bicubic back, coffee.png, torch {torch.__version__}"
        ),
        task="matting",
    )
    print(f"  alpha {alpha.shape}, mean {alpha.mean():.4f}")


if __name__ == "__main__":
    main()
