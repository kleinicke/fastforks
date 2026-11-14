"""Edges goldens from controlnet_aux (huggingface/controlnet_aux `COMMIT`), shared by `tools/golden/<family>.py`.

Reference: the detector modules of `src/controlnet_aux/<module>/` (loaded by path, without the package `__init__`,
which imports every detector), CPU float32, the variant's weights loaded strictly, on scikit-image's `coffee.png` (CC0,
Rachel Michetti; `IMAGE_SHA256`; stored as `input`) at the default detect resolution. Pre-processing as the detectors:
`resize_image` (OpenCV 5 `INTER_AREA` / `INTER_CUBIC` on uint8; upstream enlarges with `INTER_LANCZOS4`), then each
detector's own scaling; post-processing in float32 without the 8-bit rounding (the map in [0, 1], 1 = line), resized to
the image size with `cv2.INTER_LINEAR`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/<family>.py <controlnet_aux checkout> <weights dir> COFFEE_PNG
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from _common import write
from PIL import Image

from fastforks._core import registry

UPSTREAM = "https://github.com/huggingface/controlnet_aux"
COMMIT = "b1aac318b15c4e65a21bc516dc236ad4bf593d46"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png


def module(checkout: Path, name: str) -> Any:  # noqa: ANN401 - an upstream module
    """`controlnet_aux.<name>` of `checkout` without running the package `__init__` (it imports every detector)."""
    if "controlnet_aux" not in sys.modules:
        pkg = types.ModuleType("controlnet_aux")
        pkg.__path__ = [str(checkout / "src" / "controlnet_aux")]
        sys.modules["controlnet_aux"] = pkg
    return importlib.import_module(f"controlnet_aux.{name}")


def detect_image(img: np.ndarray, resolution: int) -> np.ndarray:
    """controlnet_aux `resize_image` (OpenCV `INTER_CUBIC` instead of `INTER_LANCZOS4` when enlarging)."""
    h, w = img.shape[:2]
    k = resolution / min(h, w)
    hh, ww = round(h * k / 64) * 64, round(w * k / 64) * 64
    if (hh, ww) == (h, w):
        return img
    return cv2.resize(img, (ww, hh), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)


def to_image(edges: np.ndarray, h: int, w: int) -> np.ndarray:
    """A float map at the detect size -> the image size (`cv2.INTER_LINEAR`)."""
    if edges.shape == (h, w):
        return edges.astype(np.float32)
    return cv2.resize(edges.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> `[1, 3, H, W]` float32 levels 0..255."""
    return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].contiguous()


def on(net: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - the upstream model
    """`x` on the network's device and dtype (the bench's half tier)."""
    p = next(net.parameters())
    return x.to(p.device, p.dtype)


def lineart_net(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model
    """controlnet_aux `lineart.Generator(3, 1, 3)` with `weights`, CPU float32, eval."""
    net = module(checkout, "lineart").Generator(3, 1, 3)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def lineart(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`LineartDetector.__call__` in float: `1 - net(x / 255)`."""
    x = detect_image(img, resolution)
    with torch.no_grad():
        y = net(on(net, tensor(x)) / 255)[0, 0].float().cpu().numpy()
    return to_image(1 - y, *img.shape[:2])


def anime_net(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model
    """controlnet_aux `lineart_anime.UnetGenerator(3, 1, 8, 64)` (instance norm) as `from_pretrained`, CPU float32."""
    m = module(checkout, "lineart_anime")
    norm = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = m.UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    return net.eval()


def anime(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`LineartAnimeDetector.__call__` in float: cubic to a multiple of 256, [-1, 1], cubic back, clip, invert."""
    x = detect_image(img, resolution)
    h, w = x.shape[:2]
    hn, wn = 256 * int(np.ceil(h / 256)), 256 * int(np.ceil(w / 256))
    x = cv2.resize(x, (wn, hn), interpolation=cv2.INTER_CUBIC)
    with torch.no_grad():
        t = on(net, tensor(x)) / 127.5 - 1
        y = (net(t)[0, 0] * 127.5 + 127.5).float().cpu().numpy()
    y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 255)
    return to_image(1 - y / 255, *img.shape[:2])


def hed_net(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model
    """controlnet_aux `hed.ControlNetHED_Apache2` with `weights`, CPU float32, eval."""
    net = module(checkout, "hed").ControlNetHED_Apache2()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.float().eval()


def hed(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`HEDdetector.__call__` in float: side outputs `INTER_LINEAR` to the detect size, mean, sigmoid (float64)."""
    x = detect_image(img, resolution)
    h, w = x.shape[:2]
    with torch.no_grad():
        sides = [e.float().cpu().numpy()[0, 0] for e in net(on(net, tensor(x)))]
    sides = np.stack([cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) for e in sides], axis=2)
    edge = 1 / (1 + np.exp(-np.mean(sides, axis=2).astype(np.float64)))
    return to_image(edge, *img.shape[:2])


def pidi_net(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model
    """controlnet_aux `pidi.model.pidinet()` (pixel-difference convolutions) as `from_pretrained`, CPU float32."""
    net = module(checkout, "pidi.model").pidinet()
    sd = torch.load(weights, map_location="cpu", weights_only=True)["state_dict"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    return net.eval()


def pidi(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`PidiNetDetector.__call__` in float: BGR / 255, the fused (last) output."""
    x = detect_image(img, resolution)[:, :, ::-1].copy()
    with torch.no_grad():
        edge = net(on(net, tensor(x)) / 255)[-1][0, 0].float().cpu().numpy()
    return to_image(edge, *img.shape[:2])


def teed_net(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model
    """controlnet_aux `teed.ted.TED()` with `weights` (TEED or AnyLine's MTEED), CPU float32, eval."""
    net = module(checkout, "teed.ted").TED()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def teed_soft(net: Any, x: np.ndarray) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`TEEDdetector.__call__`'s map before `safe_step` at the detect size: outputs `INTER_LINEAR`, mean, sigmoid."""
    h, w = x.shape[:2]
    with torch.no_grad():
        outs = [e.float().cpu().numpy()[0, 0] for e in net(on(net, tensor(x)))]
    outs = np.stack([cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) for e in outs], axis=2)
    return 1 / (1 + np.exp(-np.mean(outs, axis=2).astype(np.float64)))


def teed(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`TEEDdetector.__call__(safe_steps=0)` in float (the golden's options row: no quantization flips)."""
    return to_image(teed_soft(net, detect_image(img, resolution)), *img.shape[:2])


def anyline(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`AnylineDetector.__call__` in float: MTEED `safe_step(2)` + the masked line art, blended as ComfyUI-Anyline.

    The blend `1 - (1 - top) * (1 - base)` on [0, 1] floats as TheMistoAI/ComfyUI-Anyline (controlnet_aux applies it
    to uint8 arrays, where it wraps around).
    """
    from skimage import morphology  # noqa: PLC0415 - only this golden needs scikit-image

    x = detect_image(img, resolution)
    base = np.minimum(np.floor(teed_soft(net, x) * 3) / 2, 1)
    xf = x.astype(np.float32)
    g = cv2.GaussianBlur(xf, (0, 0), 2.0)
    intensity = np.min(g - xf, axis=2).clip(0, 255)
    sel = intensity[intensity > 3]  # noqa: PLR2004 - upstream's intensity_threshold
    intensity /= max(16, np.median(sel)) if sel.size else 16
    lin = (intensity * 127).clip(0, 255).astype(np.uint8)
    keep = morphology.remove_small_objects(np.repeat(lin[..., None], 3, 2).astype(bool), min_size=36, connectivity=1)
    top = lin * keep[..., 0] / 255
    out = np.where(top > 0, 1 - (1 - top) * (1 - base), base)
    return to_image(out, *img.shape[:2])


def mlsd_net(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model
    """controlnet_aux `mlsd.models.mbv2_mlsd_large.MobileV2_MLSD_Large()` with `weights`, CPU float32, eval."""
    net = module(checkout, "mlsd.models.mbv2_mlsd_large").MobileV2_MLSD_Large()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def mlsd(net: Any, img: np.ndarray, resolution: int) -> np.ndarray:  # noqa: ANN401 - the upstream model
    """`MLSDdetector.__call__` (defaults `thr_v = thr_d = 0.1`): `pred_lines`, `cv2.line` 1 px at the detect size."""
    x = detect_image(img, resolution)
    out = np.zeros(x.shape[:2], np.uint8)
    try:  # no segments: upstream's `2 * np.array([])[:, 0]` raises, its `except` leaves the map empty
        with torch.no_grad():
            lines = module(Path(), "mlsd.utils").pred_lines(x, net, [x.shape[0], x.shape[1]], 0.1, 0.1)
    except IndexError:
        lines = []
    for line in lines:
        x0, y0, x1, y1 = (int(v) for v in line)
        cv2.line(out, (x0, y0), (x1, y1), 255, 1)
    return to_image(out / 255, *img.shape[:2])


NETS = {
    "pidinet": (pidi_net, pidi),
    "hed": (hed_net, hed),
    "lineart": (lineart_net, lineart),
    "lineart_anime": (anime_net, anime),
    "teed": (teed_net, teed),
    "anyline": (teed_net, anyline),
    "mlsd": (mlsd_net, mlsd),
}


def main(family: str) -> None:
    """Write the goldens of every variant of `family` (argv: checkout, weights dir, coffee.png)."""
    checkout, wdir, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    build, predict = NETS[family]
    for card, variant in registry.cards("edges").values():
        if card.family != family:
            continue
        res = card.options["resolution"]
        net = build(checkout, wdir / variant.weights[0].name)
        edges = predict(net, img, res)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "edges": edges},
            family=family,
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, controlnet_aux {family} in float (cv2 resize_image, no 8-bit rounding), "
                f"resolution {res}, coffee.png, torch {torch.__version__}"
            ),
            task="edges",
        )
        print(f"  {variant.id}: edges {edges.shape}, mean {edges.mean():.4f}")
