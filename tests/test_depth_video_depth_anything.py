"""Video Depth Anything: the card, the input sizes, the windowing / alignment of `infer_video_depth`, `video`."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastforks import InputError, depth
from fastforks._core import registry
from fastforks.depth._runner import Runner as DepthRunner
from fastforks.depth.models.video_depth_anything import model
from fastforks.depth.models.video_depth_anything.model import INFER_LEN, KEYFRAMES, OVERLAP, Params, network_size


def test_card() -> None:
    for name, metric, key in (("video-depth-anything-s", False, "inverse_depth"),):
        _, variant = registry.cards("depth")[name]
        assert variant.outputs == (key,)
        assert bool(variant.params.get("metric", False)) is metric
        assert name in depth.list_models()
    assert registry.cards("depth")["video-depth-anything-metric-s"][1].params["metric"] is True


def test_sizes() -> None:
    """Upstream: lower bound, multiples of 14 (half to even); wider than 16:9 lowers `input_size` by the ratio."""
    assert network_size(100, 150, 518) == (518, 784)
    assert network_size(480, 640, 518) == (518, 686)
    assert network_size(480, 1280, 518) == (350, 938)  # int(518 * 1.777 / 2.667) = 345 -> 350


def _net(x: torch.Tensor) -> torch.Tensor:
    """Fake network of one window `[32, 3, h, w]` -> `[32, h, w]`: per frame, across time and by window position."""
    pos = torch.arange(x.shape[0], dtype=x.dtype)[:, None, None]
    return F.relu(x.mean(1) + 0.3 * x.mean((0, 1)) + 0.05 * pos - 0.2)


def _upstream(x: torch.Tensor, size: tuple[int, int], *, metric: bool) -> np.ndarray:
    """Upstream `infer_video_depth`'s loops with `_net` (resized, normalised frames `x`) -> `[L, H, W]`."""
    n = x.shape[0]
    step = INFER_LEN - OVERLAP
    frames = list(x) + [x[-1]] * ((step - n % step) % step + OVERLAP)
    depths, pre = [], None
    for s in range(0, n, step):
        cur = torch.stack(frames[s : s + INFER_LEN])
        if pre is not None:
            cur[:OVERLAP] = pre[list(KEYFRAMES)]
        d = F.interpolate(_net(cur)[:, None], size, mode="bilinear", align_corners=True)[:, 0]
        depths += list(d.numpy())
        pre = cur
    aligned, ref = [], []
    for s in range(0, len(depths), INFER_LEN):
        if not aligned:
            aligned += depths[:INFER_LEN]
            ref = [depths[s], depths[s + 12]]
            continue
        p, t = np.concatenate(depths[s : s + 2]).astype(np.float32), np.concatenate(ref).astype(np.float32)
        a00, a01, a11, b0, b1 = (p * p).sum(), p.sum(), np.float32(p.size), (p * t).sum(), t.sum()
        det = a00 * a11 - a01 * a01
        sc, sh = ((a11 * b0 - a01 * b1) / det, (-a01 * b0 + a00 * b1) / det) if det and not metric else (1.0, 0.0)
        post = [np.maximum(depths[s + i] * sc + sh, 0) for i in range(2, OVERLAP)]
        w = [i / 7 for i in range(8)]
        aligned[-8:] = [a * (1 - wi) + b * wi for a, b, wi in zip(aligned[-8:], post, w, strict=True)]
        aligned += [np.maximum(depths[s + i] * sc + sh, 0) for i in range(OVERLAP, INFER_LEN)]
        ref = [ref[0], np.maximum(depths[s + 12] * sc + sh, 0)]
    return np.stack(aligned[:n])


class _Fake(model._Video):  # the runner's windowing with a fake network
    def __init__(self, params: Params) -> None:
        self.ctx = type("Ctx", (), {"params": params, "io": None})()
        self.encoded = 0

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.encoded += x.shape[0]
        return [x]

    def _window(self, feats: list[torch.Tensor], h: int, w: int, valid: int) -> torch.Tensor:  # noqa: ARG002 - fake
        return _net(feats[0])[:valid]


@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("n", [1, 5, 22, 23, 24, 25, 32, 33, 50, 70])
def test_windows_match_upstream(n: int, metric: bool, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: FBT001
    """Windows, keyframes, padding, scale + shift alignment and blending as upstream; each frame encoded once."""
    monkeypatch.setattr(model, "scale", lambda x, _io: x)
    video = torch.rand(n, 3, 11, 16, generator=torch.Generator().manual_seed(n))
    runner = _Fake(Params("vits", 28, metric))
    got = runner._video(video)
    h, w = network_size(11, 16, 28)
    x = F.interpolate(video, (h, w), mode="bicubic", align_corners=False)
    np.testing.assert_allclose(got.numpy(), _upstream(x, (11, 16), metric=metric), rtol=1e-5, atol=1e-5)
    assert runner.encoded == n


class _Frames(DepthRunner):
    """Fake video runner: inverse depth = the frame's mean colour plus its index in the call; logs the call sizes."""

    Params = Params
    batch = 1
    video = True
    calls: list[int] = []  # noqa: RUF012 - test log

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - test stub
        _Frames.calls.append(image.shape[0])
        return {"inverse_depth": image.mean(1) + torch.arange(image.shape[0])[:, None, None]}


@pytest.mark.usefixtures("card_min_size")
def test_video_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`video` runs the frames as one clip, a batch each image as a one-frame video; image models refuse `video`."""
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": _Frames}))
    _Frames.calls = []
    m = depth.load("video-depth-anything-s", device="cpu")
    frames = [np.full((16, 20, 3), 10 * t, np.uint8) for t in range(3)]
    v = m.video(frames)
    assert _Frames.calls == [3]
    assert v.inverse_depth.shape == (3, 16, 20)
    torch.testing.assert_close(v.inverse_depth[:, 0, 0], torch.tensor([0.0, 10 / 255 + 1, 20 / 255 + 2]))
    torch.testing.assert_close(m.video(np.stack(frames)).inverse_depth, v.inverse_depth)
    _Frames.calls = []
    with pytest.warns(UserWarning, match=r"model\.video\(frames\)"):
        b = m(np.stack(frames))
    assert _Frames.calls == [1, 1, 1]
    torch.testing.assert_close(b.inverse_depth[:, 0, 0], torch.tensor([0.0, 10 / 255, 20 / 255]))
    with pytest.raises(InputError, match="no frames"):
        m.video([])
    with pytest.raises(InputError, match="one size"):
        m.video([frames[0], frames[1][:8]])
    with pytest.raises(InputError, match="sequence of frames"):
        m.video("clip.mp4")
    monkeypatch.setattr(_Frames, "video", False)
    with pytest.raises(InputError, match="image model"):
        m.video(frames)


def test_clip_reads_only_the_frames_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The windowing over a `Clip` (frames converted as each window asks) equals it over the whole video tensor."""
    from fastforks.depth._api import Clip  # noqa: PLC0415

    monkeypatch.setattr(model, "scale", lambda x, _io: x)
    frames = [np.asarray(np.random.default_rng(t).integers(0, 256, (11, 16, 3)), np.uint8) for t in range(50)]
    video = (torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255).contiguous()  # as to_rgb_batch
    runner = _Fake(Params("vits", 28, metric=False))
    want = runner._video(video)
    for given in (frames, np.stack(frames), video):
        clip = Clip(given, torch.device("cpu"))
        assert clip.shape == (50, 3, 11, 16)
        torch.testing.assert_close(runner._video(clip), want, rtol=0, atol=0)


class _ClipFrames(DepthRunner):
    """Fake windowed video runner: reads frames [0, 1] then [2, ...]; inverse depth = each frame's mean colour."""

    Params = Params
    batch = 1
    video = True
    clip_frames = True

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - test stub
        parts = [image[[0, 1]], image[list(range(2, image.shape[0]))]]
        return {"inverse_depth": torch.cat(parts).mean(1)}


@pytest.mark.usefixtures("card_min_size")
def test_clip_video_inputs_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `clip_frames` runner gets the frames of a list or a batch; a frame of another size fails when it is read."""
    monkeypatch.setattr(
        registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": _ClipFrames})
    )
    m = depth.load("video-depth-anything-s", device="cpu")
    frames = [np.full((16, 20, 3), 10 * t, np.uint8) for t in range(4)]
    v = m.video(frames)
    assert v.inverse_depth.shape == (4, 16, 20)
    torch.testing.assert_close(v.inverse_depth[:, 0, 0], torch.tensor([0.0, 10, 20, 30]) / 255)
    torch.testing.assert_close(m.video(np.stack(frames)).inverse_depth, v.inverse_depth)
    torch.testing.assert_close(
        m.video(torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)).inverse_depth, v.inverse_depth
    )
    with pytest.raises(InputError, match="no frames"):
        m.video([])
    with pytest.raises(InputError, match="one size"):
        m.video([frames[0], frames[1], frames[2][:8]])
    with pytest.raises(InputError, match="sequence of frames"):
        m.video("clip.mp4")
