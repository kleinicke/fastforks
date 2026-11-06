"""Goldens of the classification families built by a model library (cards with `uses = ["libnet"]`).

Reference: the library itself at the version in the card's `[upstream]`, loading its own pretrained weights the way a
user would: `torchvision.models.get_model(arch, weights=<enum>)` (torchvision's download cache) and
`timm.create_model(name, pretrained=True)` (the Hugging Face cache; set `HF_HUB_OFFLINE=1` to use it as is). DINOv2
linear adds the official head as `torch.hub` `dinov2_vitl14_lc` does. CPU float32, on `tests/synth.py`'s image
transformed as the card's `[io]` says with torchvision's tensor ops (antialiased resize of the shorter side on the
float image, clamped to [0, 1] after bicubic, centre crop, normalisation): the network is what the golden checks;
PIL vs tensor resizing is measured by the ImageNet accuracy runs instead.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/libnet_classification.py [FAMILY ...]

One file for all these families: their reference code is the same library call.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import torch
import torchvision
from _common import image, write
from torchvision.transforms.v2 import functional as TF

from fastforks._core import registry
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from fastforks._core.cards import Io

TASK = "classification"
DINOV2_HEAD = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth"


def transform(io: Io) -> torch.Tensor:
    """`tests/synth.py`'s image as the card's `[io]` transform gives it, `[1, 3, crop, crop]`."""
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    mode = {"bilinear": TF.InterpolationMode.BILINEAR, "bicubic": TF.InterpolationMode.BICUBIC}[io.interpolation]
    x = TF.resize(x, [io.resize], interpolation=mode, antialias=True)
    if io.interpolation == "bicubic":
        x = x.clamp(0, 1)
    x = TF.center_crop(x, [io.crop])
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if io.input == "imagenet" else (io.mean, io.std)
    return TF.normalize(x, list(mean), list(std))


def _torchvision(arch: str, url: str) -> torch.nn.Module:
    weights = next(w for w in torchvision.models.get_model_weights(arch) if w.url == url)
    return torchvision.models.get_model(arch, weights=weights)


def _timm(name: str) -> torch.nn.Module:
    import timm  # noqa: PLC0415 - only for the timm families

    return timm.create_model(name, pretrained=True)


def _dinov2(name: str, crop: int) -> torch.nn.Module:
    import timm  # noqa: PLC0415

    vit: Any = timm.create_model(name, pretrained=True, num_classes=0, img_size=crop)
    sd = torch.hub.load_state_dict_from_url(DINOV2_HEAD, map_location="cpu")
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head.load_state_dict(sd)

    class Lc(torch.nn.Module):  # torch.hub dinov2_vitl14_lc, layers=1
        def __init__(self) -> None:
            super().__init__()
            self.vit, self.head = vit, head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            t = self.vit.forward_features(x)
            return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=1))

    return Lc()


def main() -> None:
    families = set(sys.argv[1:])
    torch.manual_seed(0)
    for card, variant in registry.cards(TASK).values():
        if "libnet" not in card.uses or (families and card.family not in families):
            continue
        params = {**card.params, **variant.params}
        lib, arch, io = params["library"], params["arch"], variant.io
        if card.family == "dinov2_linear_cls":
            net, version = _dinov2(arch, io.crop), _version("timm")
        elif lib == "torchvision":
            net, version = _torchvision(arch, variant.weights[0].url or ""), torchvision.__version__
        else:
            net, version = _timm(arch), _version("timm")
        with torch.inference_mode():
            logits = net.eval()(transform(io))[0].numpy()
        write(
            variant.ref,
            {"logits": logits},
            family=card.family,
            upstream=f"{card.upstream}@{card.upstream_commit}",
            patch=None,
            settings=f"CPU float32, {lib} {version}, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  top-1 {int(logits.argmax())}  max logit {logits.max():.3f}")


def _version(module: str) -> str:
    return __import__(module).__version__


if __name__ == "__main__":
    main()
