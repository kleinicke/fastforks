"""Input fuzzing: every accepted layout / dtype gives the canonical result, everything else is an `InputError`.

Seeded and synthetic. Most cases use a stub runner (no weights); `test_tiny_and_odd_sizes` runs the smallest variant of
every family on the CPU.
"""

# ruff: noqa: ARG005
from __future__ import annotations

import dataclasses
import types
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError, depth, stereo
from fastforks._core import registry
from fastforks._core.errors import WeightsError
from fastforks._core.model import Model
from fastforks.depth._runner import Runner as DepthRunner
from fastforks.stereo._runner import Runner
from test_api import Params
from test_golden import API, SMALLEST, TASK, sweep_model

pytestmark = pytest.mark.usefixtures("card_min_size")  # tiny images through fake runners
REAL_CHECK = Model._check_size  # before `card_min_size` replaces it
LOAD_ONCE = True  # conftest runs the tests of one ref together: `sweep_model` loads it once


RNG = np.random.default_rng(1234)
H, W = 23, 37  # odd: padding and cropping
L8 = RNG.integers(0, 256, (H, W, 3), dtype=np.uint8)
R8 = RNG.integers(0, 256, (H, W, 3), dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class DepthParams:
    arch: str
    short_side: int


class Mix(Runner):
    """Disparity = a fixed mix of all channels of both views, so any layout or channel mix-up changes it."""

    Params = Params

    def __call__(self, left, right):
        w = torch.tensor([1.0, 2.0, 4.0], device=left.device).view(1, 3, 1, 1)
        return {"disparity": (left * w).sum(1) - (right * w.flip(1)).sum(1)}


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Mix}))
    return stereo.load("s2m2-s", device="cpu")


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)


def _strided(a: np.ndarray) -> np.ndarray:
    big = np.zeros((a.shape[0], 2 * a.shape[1], 3), a.dtype)
    big[:, ::2] = a
    return big[:, ::2]


SAME: dict[str, Any] = {
    "uint16": lambda a: a.astype(np.uint16) * 257,
    "uint16 big-endian": lambda a: (a.astype(np.uint16) * 257).astype(">u2"),
    "float32": lambda a: (a / 255).astype(np.float32),
    "float64": lambda a: a / 255,
    "tensor uint8 CHW": _t,
    "tensor float32 CHW": lambda a: _t(a) / 255,
    "tensor float64 CHW": lambda a: _t(a).double() / 255,
    "BCHW": lambda a: _t(a)[None],
    "BHWC": lambda a: a[None],
    "channels_last": lambda a: _t(a)[None].contiguous(memory_format=torch.channels_last),
    "PIL RGB": Image.fromarray,
    "PIL RGBA": lambda a: Image.fromarray(a).convert("RGBA"),
    "numpy RGBA": lambda a: np.dstack([a, a[..., :1]]),
    "negative strides": lambda a: a[::-1, ::-1][::-1, ::-1],
    "Fortran order": np.asfortranarray,
    "read-only": lambda a: np.broadcast_to(a, a.shape),
    "strided view": _strided,
    "requires_grad": lambda a: (_t(a) / 255).requires_grad_(),
}
LOW_PRECISION = {
    "float16": lambda a: (a / 255).astype(np.float16),
    "tensor float16": lambda a: (_t(a) / 255).half(),
    "tensor bfloat16": lambda a: (_t(a) / 255).bfloat16(),
}


@pytest.mark.parametrize("name", sorted(SAME))
def test_layouts_match_canonical(model, name):
    ref = model(L8, R8).disparity
    f = SAME[name]
    got = model(f(L8), f(R8)).disparity
    torch.testing.assert_close(got.reshape(ref.shape), ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize("name", sorted(LOW_PRECISION))
def test_low_precision_inputs(model, name):
    """Half-precision images are read exactly: same result as the same values in float32."""
    lo = [LOW_PRECISION[name](a) for a in (L8, R8)]
    up = [x.float() if isinstance(x, torch.Tensor) else x.astype(np.float32) for x in lo]
    torch.testing.assert_close(model(*lo).disparity, model(*up).disparity, rtol=0, atol=1e-4)


def test_gray_inputs_agree(model):
    g, h = L8[..., 0], R8[..., 0]
    ref = model(g, h).disparity
    for f in (lambda a: a[..., None], lambda a: torch.from_numpy(a)[None], torch.from_numpy, Image.fromarray):
        torch.testing.assert_close(model(f(g), f(h)).disparity, ref, rtol=0, atol=1e-6)


def test_random_batches_unbatch(model):
    for b in RNG.integers(1, 5, 4):
        ls = RNG.integers(0, 256, (b, H, W, 3), dtype=np.uint8)
        rs = RNG.integers(0, 256, (b, H, W, 3), dtype=np.uint8)
        batch = model(ls, rs).disparity
        assert batch.shape == (b, H, W)
        for i in range(b):
            torch.testing.assert_close(batch[i], model(ls[i], rs[i]).disparity)


@pytest.mark.parametrize(
    ("left", "right", "match"),
    [
        (L8.astype(object), R8.astype(object), "unsupported image dtype object"),
        (L8.astype("datetime64[s]"), R8.astype("datetime64[s]"), "unsupported image dtype"),
        (L8.astype(np.int64), R8.astype(np.int64), "unsupported image dtype"),
        (L8 > 0, R8 > 0, "unsupported image dtype"),
        (L8.astype(np.float32), R8.astype(np.float32), r"in \[0, 1\].*divide 0..255"),
        ((L8 / 255) * (1 + 1e-6), R8 / 255, r"got \[0, 1\.0000\d+\]"),
        ([L8], [R8], "unsupported image type list"),
        (None, R8, "unsupported image type NoneType"),
        (L8.transpose(2, 0, 1), R8.transpose(2, 0, 1), "CHW"),
        (torch.from_numpy(L8), torch.from_numpy(R8), "numpy is HWC, torch is CHW"),
        (L8, R8[:, :-1], "differ in shape"),
        (L8[..., 0], R8, "differ in colour: 1 vs 3"),
        (Image.fromarray(L8).convert("L"), Image.fromarray(R8), "differ in colour"),
        (torch.empty(3, H, W, device="meta"), torch.empty(3, H, W, device="meta"), "meta tensor"),
        (np.zeros((0, W, 3), np.uint8), np.zeros((0, W, 3), np.uint8), "empty image"),
        (np.full((H, W, 3), np.inf, np.float32), R8 / 255, "finite"),
    ],
    ids=lambda x: None if isinstance(x, str) else type(x).__name__,
)
def test_rejected_inputs(model, left, right, match):
    with pytest.raises(InputError, match=match):
        model(left, right)


def test_unreadable_paths(model, tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    for bad, why in (
        (tmp_path / "missing.png", "No such file"),
        (tmp_path / "x.png", "cannot identify"),
        (tmp_path, ""),
    ):
        with pytest.raises(InputError, match=f"cannot read image .*{why}"):
            model(bad, R8)


def test_pil_sample_types(model, tmp_path):
    """16-bit PIL images (`I;16`) and 16-bit gray files keep their values; the lossy modes `I` and `F` raise."""
    g16 = (L8[..., 0].astype(np.uint16) * 257, R8[..., 0].astype(np.uint16) * 257)
    ref = model(*g16).disparity
    torch.testing.assert_close(model(*(Image.fromarray(a) for a in g16)).disparity, ref)
    Image.fromarray(g16[0]).save(tmp_path / "l.png")
    Image.fromarray(g16[1]).save(tmp_path / "r.png")
    torch.testing.assert_close(model(tmp_path / "l.png", tmp_path / "r.png").disparity, ref)
    with pytest.raises(InputError, match="mode I would be converted"):
        model(Image.fromarray(g16[0].astype(np.int32), mode="I"), Image.fromarray(g16[1]))
    with pytest.raises(InputError, match="mode F would be converted"):
        model(Image.fromarray(g16[0].astype(np.float32) / 65535, mode="F"), Image.fromarray(g16[1]))


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", SMALLEST)
def test_tiny_and_odd_sizes(ref, monkeypatch):
    """Tiny and odd sizes give finite maps of the input size, or an `InputError` below `size_range().reject_below`.

    Portable on the CPU, smallest variant per family. CREStereo returned all-NaN maps at <= 32 px, WAFT raised a raw
    `ValueError` at 16 x 16 px before their cards declared a `min_size`.
    """
    task = TASK[ref]
    if task not in {"stereo", "depth"}:
        pytest.skip("dense stereo / depth only")
    monkeypatch.setattr(Model, "_check_size", REAL_CHECK)  # the real size check, not `card_min_size`
    m = sweep_model(ref)
    lo = m.size_range().reject_below
    for h, w in ((1, 1), (7, 13), (5, 97), (16, 16), (32, 32), (33, 47)):
        x = RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)
        if min(h, w) < lo:
            with pytest.raises(InputError, match="at least"):
                m(x, x) if task == "stereo" else m(x)
            continue
        r = m(x, np.roll(x, -1, 1)) if task == "stereo" else m(x)
        main = r.disparity if task == "stereo" else r.depth if r.depth is not None else r.inverse_depth
        assert main.shape == (h, w), (h, w)
        assert torch.isfinite(main).all(), (h, w)


def test_depth_intrinsics_rejected_for_unused(monkeypatch):
    class Inv(DepthRunner):
        Params = DepthParams

        def __call__(self, image, intrinsics):  # noqa: ARG002
            return {"inverse_depth": image[:, 0]}

    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Inv}))
    m = depth.load("depth-anything-v2-s", device="cpu")
    inv = m(L8).inverse_depth
    assert inv is not None
    assert inv.shape == (H, W)
    for k, match in ((np.eye(3), "does not use"), ([[1, 2]], "does not use")):
        with pytest.raises(InputError, match=match):
            m(L8, intrinsics=k)


# --- real small models on the CPU: odd sizes, files, batches and tensors through the public API ---------------------

_REAL = (
    ("stereo", "s2m2-s"),
    ("depth", "depth-anything-v2-s"),
    ("classification", "squeezenet1_1.tv_in1k"),
    (
        "segmentation",
        "segformer-b0-ade20k",
    ),
)


def _main(r: Any) -> torch.Tensor:
    """The primary output of any result as a float tensor (batch axis first when batched)."""
    for name in ("disparity", "depth", "inverse_depth", "logits", "labels"):
        v = getattr(r, name, None)
        if isinstance(v, torch.Tensor):
            return v.float()
    msg = f"no main output in {type(r).__name__}"
    raise AssertionError(msg)


@pytest.fixture(scope="module", params=_REAL, ids=[r for _, r in _REAL])
def real(request):
    """`(task, model)`: a small real model of the task on the CPU; skipped when its weights are not available."""
    task, ref = request.param
    card, variant = registry.resolve(task, ref)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")
    options = {"iters": 2} if "iters" in card.options else {}
    return task, API[task].load(ref, device="cpu", options=options)


def _go(real, make: Any, base: np.ndarray) -> torch.Tensor:
    """Main output for the input `make(array, tag)` built from `base` (stereo: the right view is `base` shifted)."""
    task, m = real
    if task == "stereo":
        return _main(m(make(base, "l"), make(np.roll(base, -2, 1), "r")))
    return _main(m(make(base, "l")))


def _textured(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Smooth uint8 RGB texture: finite, stable outputs on any model, unlike pure noise."""
    g = np.random.default_rng(seed)
    small = torch.from_numpy(g.integers(0, 256, (h // 8 + 2, w // 8 + 2, 3)).astype(np.float32))
    big = torch.nn.functional.interpolate(small.permute(2, 0, 1)[None], (h, w), mode="bilinear")
    return big[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def _png(path, a: np.ndarray, mode: str | None = None):
    Image.fromarray(a, mode).save(path)
    return path


def _file_inputs(tmp_path) -> dict[str, Any]:
    """name -> `make(array, tag)` for an input that must equal the plain uint8 RGB array (the array is RGB uint8)."""
    return {
        "PNG file": lambda a, t: _png(tmp_path / f"{t}.png", a),
        "RGBA PNG file": lambda a, t: _png(tmp_path / f"{t}a.png", np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])),
        "JPEG-free str path": lambda a, t: str(_png(tmp_path / f"{t}s.png", a)),
        "PIL RGBA": lambda a, t: Image.fromarray(a).convert("RGBA"),
        "uint16 array": lambda a, t: a.astype(np.uint16) * 257,
        "float32 array": lambda a, t: a.astype(np.float32) / 255,
        "tensor CHW": lambda a, t: torch.from_numpy(a).permute(2, 0, 1),
        "tensor CHW non-contiguous": lambda a, t: (
            torch.from_numpy(np.ascontiguousarray(a[..., ::-1])).permute(2, 0, 1).flip(0)
        ),
        "tensor float64 BCHW": lambda a, t: torch.from_numpy(a).permute(2, 0, 1)[None].double() / 255,
        "tensor channels_last": lambda a, t: (
            torch.from_numpy(a).permute(2, 0, 1)[None].contiguous(memory_format=torch.channels_last)
        ),
    }


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("hw", [(97, 131), (96, 128), (95, 129), (113, 150)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_real_models_odd_sizes_and_input_kinds(real, hw, tmp_path):
    """Odd and 1-px-off sizes: every input kind gives the same output as the uint8 array, or an `InputError`."""
    base = _textured(*hw)
    ref = _go(real, lambda a, t: a, base)
    assert torch.isfinite(ref).all()
    if real[0] in {"stereo", "depth"}:
        assert ref.shape[-2:] == hw
    for name, make in _file_inputs(tmp_path).items():
        got = _go(real, make, base)
        assert got.numel() == ref.numel(), name  # a [1, ...] batch input keeps its batch axis
        got = got.reshape(ref.shape)
        tol = 2e-3 * max(ref.abs().max().item(), 1.0)  # float32 vs scaled-integer paths round alike; 0.2 % of range
        torch.testing.assert_close(got, ref, rtol=0, atol=tol, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.slow
@pytest.mark.model
def test_real_models_gray_inputs(real, tmp_path):
    """Gray (HW array, 8/16-bit PNG file, PIL `L`) equals the same gray replicated to RGB."""
    base = _textured(97, 131)[..., 0]
    rgb = np.repeat(base[..., None], 3, -1)
    ref = _go(real, lambda a, t: a, rgb)
    makes = {
        "HW": lambda a, t: a[..., 0],
        "PIL L": lambda a, t: Image.fromarray(a[..., 0]),
        "gray PNG": lambda a, t: _png(tmp_path / f"{t}g.png", a[..., 0]),
        "gray16 PNG": lambda a, t: _png(tmp_path / f"{t}g16.png", a[..., 0].astype(np.uint16) * 257),
        "gray tensor HW": lambda a, t: torch.from_numpy(a[..., 0]),
    }
    for name, make in makes.items():
        got = _go(real, make, rgb)
        torch.testing.assert_close(
            got, ref, rtol=0, atol=2e-3 * max(ref.abs().max().item(), 1.0), msg=lambda m, n=name: f"{n}: {m}"
        )


@pytest.mark.slow
@pytest.mark.model
def test_real_models_16bit_colour_png_raises(real, tmp_path):
    """A 16-bit colour PNG would be read as 8-bit by Pillow: `InputError`, never a silent wrong result."""
    import cv2  # noqa: PLC0415

    a = (_textured(97, 131).astype(np.uint16) * 257)[..., ::-1]
    p = tmp_path / "c16.png"
    assert cv2.imwrite(str(p), np.ascontiguousarray(a))
    with pytest.raises(InputError, match="16-bit colour PNG"):
        _go(real, lambda x, t: p, _textured(97, 131))


@pytest.mark.slow
@pytest.mark.model
def test_real_models_exif_orientation_is_ignored(real, tmp_path):
    """JPEG files are read as stored: the EXIF orientation tag is not applied (see `_core.image`)."""
    base = _textured(97, 131)
    ex = Image.Exif()
    ex[0x0112] = 6  # rotate 90 deg CW when displayed
    paths = {}

    def make(a, t):
        paths[t] = tmp_path / f"{t}.jpg"
        Image.fromarray(a).save(paths[t], quality=95, exif=ex)
        return paths[t]

    got = _go(real, make, base)
    plain = {t: np.asarray(Image.open(p).convert("RGB")) for t, p in paths.items()}
    assert plain["l"].shape == base.shape  # Pillow did not rotate it either
    ref = _go(real, lambda a, t: plain[t], base)
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)


@pytest.mark.slow
@pytest.mark.model
def test_real_models_batch_equals_single(real):
    """Batched BHWC / BCHW inputs give, per image, the single-image result."""
    task, m = real
    imgs = [_textured(97, 131, seed=s) for s in (1, 2, 3)]
    if task == "stereo":
        rights = [np.roll(i, -2, 1) for i in imgs]
        batch = _main(m(np.stack(imgs), np.stack(rights)))
        singles = [_main(m(a, b)) for a, b in zip(imgs, rights, strict=True)]
        tensor_batch = _main(
            m(
                torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2),
                torch.from_numpy(np.stack(rights)).permute(0, 3, 1, 2),
            )
        )
    else:
        batch = _main(m(np.stack(imgs)))
        singles = [_main(m(a)) for a in imgs]
        tensor_batch = _main(m(torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)))
    assert batch.shape[0] == 3
    atol = 1e-3 * max(batch.abs().max().item(), 1.0)
    for i, s in enumerate(singles):
        torch.testing.assert_close(batch[i], s.reshape(batch[i].shape), rtol=0, atol=atol)
    torch.testing.assert_close(tensor_batch, batch, rtol=0, atol=atol)
