"""Goldens of the classification families built by a model library (cards with `uses = ["libnet"]`).

Reference: the library itself at the version in the card's `[upstream]`, loading its own pretrained weights the way a
user would: `torchvision.models.get_model(arch, weights=<enum>)` (torchvision's download cache) and
`timm.create_model(name, pretrained=True)` (the Hugging Face cache; set `HF_HUB_OFFLINE=1` to use it as is). DINOv2
linear adds the official head as `torch.hub` `dinov2_vitl14_lc` does. CPU float32, on `tests/synth.py`'s image
transformed as the card's `[io]` says with torchvision's tensor ops (antialiased resize of the shorter side on the
float image, clamped to [0, 1] after bicubic, centre crop, normalisation): the network is what the golden checks;
PIL vs tensor resizing is measured by the ImageNet accuracy runs instead. The WD taggers are built as their reference
script does (`timm.create_model("hf-hub:SmilingWolf/<id>@<revision>")`, white square canvas, BGR input). The Falconsai
NSFW ViT is the original `transformers` `ViTForImageClassification` at the card's revision (transformers on
`PYTHONPATH`; the image stretched to 224 x 224 as its `ViTImageProcessor`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/libnet_classification.py [FAMILY | REF ...]

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
    from fastforks._core.cards import Card, Io, Variant

TASK = "classification"
DINOV2_HEAD = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth"


def transform(io: Io) -> torch.Tensor:
    """`tests/synth.py`'s image as the card's `[io]` transform gives it, `[1, 3, crop, crop]`."""
    return prepare(torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255, io)


def prepare(x: torch.Tensor, io: Io) -> torch.Tensor:
    """The card's `[io]` transform of a float `[B, 3, H, W]` batch in [0, 1]."""
    if io.pad_square is not None:  # wdv3_timm.py pil_pad_square: paste at ((side - w) // 2, (side - h) // 2)
        h, w = x.shape[-2:]
        side = max(h, w)
        x = torch.nn.functional.pad(
            x,
            ((side - w) // 2, side - w - (side - w) // 2, (side - h) // 2, side - h - (side - h) // 2),
            value=io.pad_square,
        )
    mode = {"bilinear": TF.InterpolationMode.BILINEAR, "bicubic": TF.InterpolationMode.BICUBIC}[io.interpolation]
    x = TF.resize(x, [io.resize] * (1 + io.stretch), interpolation=mode, antialias=True)
    if io.interpolation == "bicubic":
        x = x.clamp(0, 1)
    if io.crop:
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


class _Logits(torch.nn.Module):
    """A `transformers` image classifier returning its logits tensor."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=x).logits


class _Bgr(torch.nn.Module):
    """`inputs[:, [2, 1, 0]]` before the network (wdv3_timm.py)."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, [2, 1, 0]])


def reference(card: Card, variant: Variant) -> tuple[torch.nn.Module, str, str]:
    """The reference network of a variant, its library and the library's version."""
    params = {**card.params, **variant.params}
    lib, arch = params["library"], params["arch"]
    if card.family == "dinov2_linear_cls":
        net, version = _dinov2(arch, variant.io.crop), _version("timm")
    elif card.family == "wd_tagger":  # hf-hub config + weights at the card's revision; the script feeds BGR
        rev = (variant.weights[0].url or "").split("/resolve/")[1].split("/")[0]
        hub, version = _timm(f"hf-hub:SmilingWolf/{variant.id}@{rev}"), _version("timm")
        net = _Bgr(hub)
    elif card.family == "nsfw_vit":  # the original transformers model at the card's revision
        import transformers  # noqa: PLC0415 - only for this family

        hf = transformers.ViTForImageClassification.from_pretrained(
            card.upstream.removeprefix("https://huggingface.co/"), revision=card.upstream_commit
        )
        net, version = _Logits(hf), transformers.__version__
        lib = "transformers"
    elif lib == "torchvision":
        net, version = _torchvision(arch, variant.weights[0].url or ""), torchvision.__version__
    else:
        net, version = _timm(arch), _version("timm")
    return net, lib, version


def main() -> None:
    want = set(sys.argv[1:])
    torch.manual_seed(0)
    for card, variant in registry.cards(TASK).values():
        if "libnet" not in card.uses or (want and not want & {card.family, variant.id, variant.ref}):
            continue
        net, lib, version = reference(card, variant)
        with torch.inference_mode():
            logits = net.eval()(transform(variant.io))[0].numpy()
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
