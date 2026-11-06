"""`StereoModel.run_many` and the pipeline engine: inputs, order, errors, shutdown, and bit-identity with a loop."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import struct
import subprocess
import sys
import threading
import time
import types
import zlib
from concurrent.futures import wait
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError
from fastforks._core import pipeline as core
from fastforks.stereo import geometry as g
from fastforks.stereo.pipeline import PairResult, natural_key, run_many
from synth import pair

H, W = 48, 64
KERNEL, CAMERA, OOM = "kernel failed", "camera unplugged", "CUDA out of memory (simulated)"


class Fake:
    """A stand-in stereo model: disparity = mean(left) - mean(right) per pixel, float32, on `device`."""

    def __init__(self, device: str = "cpu", sleep_cycles: int = 0) -> None:
        self.device = torch.device(device)
        self.sleep_cycles = sleep_cycles
        self.calls = 0

    def __call__(self, left, right):
        self.calls += 1
        if self.sleep_cycles:
            torch.cuda._sleep(self.sleep_cycles)  # a slow kernel: uploads of later pairs must not overwrite inputs
        lt = torch.as_tensor(left).to(self.device).float()
        rt = torch.as_tensor(right).to(self.device).float()
        return types.SimpleNamespace(disparity=lt.mean(-3) - rt.mean(-3) + 0.5, confidence=None, extras={})


def _pairs(n: int, h: int = H, w: int = W) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ps = [pair(h, w, s) for s in range(n)]
    return [p[0] for p in ps], [p[1] for p in ps]


def _folders(tmp_path: Path, names: list[str], h: int = H, w: int = W) -> tuple[Path, Path]:
    for side in ("L", "R"):
        (tmp_path / side).mkdir(exist_ok=True)
    for i, name in enumerate(names):
        left, right = pair(h, w, i)
        for side, img in (("L", left), ("R", right)):
            p = tmp_path / side / name
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(p)
    return tmp_path / "L", tmp_path / "R"


def _expected(left, right) -> np.ndarray:
    return Fake()(_t(left), _t(right)).disparity[0].numpy()


def _t(img) -> torch.Tensor:
    from fastforks._core.image import _as_tensor  # noqa: PLC0415

    return _as_tensor(img)


def _threads() -> set[str]:
    return {
        t.name
        for t in threading.enumerate()
        if t.name.startswith("fastforks-") and not t.name.startswith("fastforks-tiff")
    }


def _no_threads(timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while _threads() and time.monotonic() < end:
        time.sleep(0.02)
    return not _threads()


# inputs -------------------------------------------------------------------------------------------------------------
def test_natural_key():
    names = ["frame_10", "frame_2", "Frame_1", "frame_1b", "frame_1a"]
    assert sorted(names, key=natural_key) == ["Frame_1", "frame_1a", "frame_1b", "frame_2", "frame_10"]


def test_numpy_lists_match_single_calls():
    lefts, rights = _pairs(5)
    out = list(run_many(Fake(), lefts, rights))
    assert [r.index for r in out] == list(range(5))
    for r, left, right in zip(out, lefts, rights, strict=True):
        assert (r.left_path, r.path, r.error) == (None, None, None)
        with pytest.raises(InputError, match="pass rig="):
            r.depth()
        np.testing.assert_array_equal(r.disparity, _expected(left, right))


def test_directories_natural_order_and_save_names(tmp_path):
    left, right = _folders(tmp_path, ["f_10.png", "f_2.png", "f_1.png"])
    (left / "notes.txt").write_text("x")
    (right / ".hidden.png").write_bytes(b"")
    out = list(run_many(Fake(), left, right, save_to=tmp_path / "out", save_format=".npy"))
    assert [r.left_path.name for r in out] == ["f_1.png", "f_2.png", "f_10.png"]
    assert [r.path.name for r in out] == ["f_1.npy", "f_2.npy", "f_10.npy"]
    for r in out:
        np.testing.assert_array_equal(np.load(r.path), _expected(r.left_path, r.right_path))
    assert not [p for p in (tmp_path / "out").iterdir() if p.name.startswith(".")]  # no temporary files left


def test_a_directory_is_not_recursive(tmp_path):
    left, right = _folders(tmp_path, ["a.png", "sub/b.png"])
    assert [r.left_path.name for r in run_many(Fake(), left, right)] == ["a.png"]


@pytest.mark.parametrize(
    ("pattern", "names"),
    [
        ("*.png", ["a1.png", "a2.png", "a10.png", "b1.png"]),
        ("a?.png", ["a1.png", "a2.png"]),
        ("[ab]1.png", ["a1.png", "b1.png"]),
        ("**/*.png", ["a1.png", "a2.png", "a10.png", "b1.png", "sub/a3.png", "sub/deep/c1.png"]),
    ],
)
def test_glob_patterns(tmp_path, pattern, names):
    left, right = _folders(tmp_path, ["a1.png", "a10.png", "a2.png", "b1.png", "sub/a3.png", "sub/deep/c1.png"])
    out = list(run_many(Fake(), str(left / pattern), str(right / pattern), save_to=tmp_path / "out"))
    assert [r.left_path.relative_to(left).as_posix() for r in out] == names
    assert [r.path.relative_to(tmp_path / "out").as_posix() for r in out] == [n.replace(".png", ".tif") for n in names]


def test_lists_of_paths_and_mixed_inputs(tmp_path):
    left, right = _folders(tmp_path, ["x.png", "y.png"])
    lefts, rights = [left / "x.png", np.asarray(Image.open(left / "y.png"))], [str(right / "x.png"), right / "y.png"]
    out = list(run_many(Fake(), lefts, rights, save_to=tmp_path / "out"))
    assert [r.path.name for r in out] == ["x.tif", "000001.tif"]
    assert (out[1].left_path, out[1].right_path) == (None, right / "y.png")


def test_generators_and_4d_batches():
    lefts, rights = _pairs(4)
    out = list(run_many(Fake(), (x for x in lefts), np.stack(rights)))
    for r, left, right in zip(out, lefts, rights, strict=True):
        np.testing.assert_array_equal(r.disparity, _expected(left, right))
    tl = torch.from_numpy(np.stack(lefts)).permute(0, 3, 1, 2)
    tr = torch.from_numpy(np.stack(rights)).permute(0, 3, 1, 2)
    out = list(run_many(Fake(), tl, tr))
    assert isinstance(out[0].disparity, torch.Tensor)  # tensors in, tensors out
    np.testing.assert_array_equal(out[3].disparity.numpy(), _expected(lefts[3], rights[3]))


@pytest.mark.parametrize(
    ("lefts", "rights", "match"),
    [
        ("missing_dir", "missing_dir", r"lefts: '.*missing_dir' is not a directory and matches no file"),
        ("nomatch*.png", "nomatch*.png", r"'.*nomatch\*\.png' is not a directory and matches no file"),
        ("empty", "empty", "lefts: no images in directory"),
        ([1, 2], [1], "2 left images but 1 right images"),
        ([], [], "no image pairs"),
        (np.zeros((H, W, 3), np.uint8), [], "one array of shape"),
        (3, [], "must be a directory, a glob pattern or a list of images"),
    ],
)
def test_input_errors(tmp_path, lefts, rights, match):
    (tmp_path / "empty").mkdir()
    fix = lambda s: str(tmp_path / s) if isinstance(s, str) else s  # noqa: E731
    with pytest.raises(InputError, match=match):
        run_many(Fake(), fix(lefts), fix(rights))


def test_unequal_generators_raise_at_the_end():
    lefts, rights = _pairs(3)
    it = run_many(Fake(), iter(lefts), iter(rights[:2]))
    assert next(it).index == 0
    assert next(it).index == 1
    with pytest.raises(InputError, match="right images ended after 2"):
        next(it)
    assert _no_threads()


def test_duplicate_output_names(tmp_path):
    left, right = _folders(tmp_path, ["a/x.png", "b/x.png"])
    with pytest.raises(InputError, match=r"both be saved as x\.tif"):
        list(
            run_many(
                Fake(),
                [left / "a/x.png", left / "b/x.png"],
                [right / "a/x.png", right / "b/x.png"],
                save_to=tmp_path / "o",
            )
        )


def test_right_view_output_names_collide(tmp_path):
    left, right = _folders(tmp_path, ["a.png", "a_right.png"])
    kw = {"save_to": tmp_path / "o"}
    with pytest.raises(InputError, match="both be saved as"):
        list(run_many(Fake(), left, right, views="both", **kw))
    assert len(list(run_many(Fake(), left, right, views="left", **kw))) == 2


def test_output_names_differing_only_in_case_collide(tmp_path):  # TM-5: one file on APFS / NTFS
    left, right = _folders(tmp_path, ["one/Photo.png", "two/photo.png"])
    lefts, rights = [left / "one/Photo.png", left / "two/photo.png"], [right / "one/Photo.png", right / "two/photo.png"]
    with pytest.raises(InputError, match="ignoring case"):
        list(run_many(Fake(), lefts, rights, save_to=tmp_path / "o"))
    globs = str(left / "**/*.png"), str(right / "**/*.png")  # names one/Photo, two/photo: distinct
    assert len(list(run_many(Fake(), *globs, save_to=tmp_path / "d"))) == 2


def test_skip_existing_needs_every_output_as_a_file(tmp_path):  # TM-7
    left, right = _folders(tmp_path, ["a.png", "b.png"])
    out = tmp_path / "o"
    out.mkdir()
    (out / "a.tif").mkdir()  # a directory named like the output is not a result
    model = Fake()
    res = list(run_many(model, left, right, save_to=out, skip_existing=True, on_error="skip"))
    assert model.calls == 2
    assert res[0].error is not None  # writing onto the directory fails visibly
    assert res[1].error is None
    (out / "b.tif").write_bytes(b"kept")  # the right view's b_right.tif is missing
    model = Fake()
    res = list(run_many(model, [left / "b.png"], [right / "b.png"], views="both", save_to=out, skip_existing=True))
    assert model.calls == 2  # both views ran
    assert (out / "b_right.tif").is_file()
    assert (out / "b.tif").read_bytes() != b"kept"


def test_options_are_checked():
    lefts, rights = _pairs(1)
    with pytest.raises(InputError, match="on_error"):
        run_many(Fake(), lefts, rights, on_error="ignore")
    with pytest.raises(InputError, match="save_format"):
        run_many(Fake(), lefts, rights, save_format="jpg")
    with pytest.raises(InputError, match="prefetch"):
        run_many(Fake(), lefts, rights, prefetch=0)
    with pytest.raises(InputError, match="rig must be"):
        run_many(Fake(), lefts, rights, rig=42)


# order, errors, shutdown --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("prefetch", "readers", "writers"), [(1, 1, 1), (2, 0, 0), (5, 3, 2), (3, 2, 0)])
def test_order_with_any_knobs(tmp_path, prefetch, readers, writers):
    left, right = _folders(tmp_path, [f"{i}.png" for i in range(9)])
    out = list(
        run_many(Fake(), left, right, save_to=tmp_path / "o", prefetch=prefetch, readers=readers, writers=writers)
    )
    assert [r.path.name for r in out] == [f"{i}.tif" for i in range(9)]
    hashes = {hashlib.sha256(r.path.read_bytes()).hexdigest() for r in out}
    assert len(hashes) == 9


def test_later_pairs_decode_after_the_first(monkeypatch):
    """Decodes compete for the cores: pair 1 starts once pair 0 is decoded, pairs 2+ once pair 0's call is done."""
    from fastforks.stereo import pipeline as sp  # noqa: PLC0415

    log: list[tuple[str, int]] = []
    host = sp._host

    def read(img, what, index):
        log.append(("start", index))
        if index == 0:
            time.sleep(0.05)
        out = host(img, what, index)
        log.append(("end", index))
        return out

    class Slow(Fake):
        def __call__(self, left, right):
            time.sleep(0.05)
            log.append(("call", self.calls))
            return super().__call__(left, right)

    monkeypatch.setattr(sp, "_host", read)
    lefts, rights = _pairs(6)
    out = list(run_many(Slow(), lefts, rights, readers=3, prefetch=5))
    assert [r.index for r in out] == list(range(6))
    starts = [k for k, (kind, i) in enumerate(log) if kind == "start"]
    second = min(k for k in starts if log[k][1] == 1)
    later = min(k for k in starts if log[k][1] > 1)
    assert log[:second].count(("end", 0)) == 2, log
    assert ("call", 0) in log[:later], log


def test_a_bad_file_names_the_pair(tmp_path):
    left, right = _folders(tmp_path, [f"{i}.png" for i in range(5)])
    (left / "2.png").write_bytes(b"not a png")
    it = run_many(Fake(), left, right)
    assert [next(it).index, next(it).index] == [0, 1]
    with pytest.raises(InputError, match=r"pair 2 \(.*2\.png, .*2\.png\): cannot read image"):
        next(it)
    assert _no_threads()


def _png16_rgb(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\0" + np.full((W, 3), 1000, ">u2").tobytes() for _ in range(H))
    header = struct.pack(">IIBBBBB", W, H, 16, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


def test_skip_bad_files(tmp_path):
    left, right = _folders(tmp_path, [f"{i}.png" for i in range(5)])
    (left / "1.png").write_bytes(b"not a png")
    _png16_rgb(right / "3.png")  # Pillow would read it as 8-bit: refused
    out = list(run_many(Fake(), left, right, on_error="skip", save_to=tmp_path / "o"))
    assert [r.error is not None for r in out] == [False, True, False, True, False]
    assert isinstance(out[1].error, InputError)
    assert out[1].path is None
    assert out[1].views == "left"
    assert out[1].calibrated is False
    assert out[4].path == tmp_path / "o" / "4.tif"


class BoomError(RuntimeError):
    pass


def test_other_errors_raise_even_with_skip():
    lefts, rights = _pairs(10)
    fake = Fake()

    class M:
        device = fake.device

        def __call__(self, left, right):
            if fake.calls == 6:
                raise BoomError(KERNEL)
            return fake(left, right)

    it = run_many(M(), lefts, rights, on_error="skip")
    assert [next(it).index for _ in range(6)] == list(range(6))
    with pytest.raises(BoomError) as info:
        next(it)
    assert any("while processing pair 6" in n for n in info.value.__notes__)
    assert _no_threads()


def test_source_errors_surface_at_their_position():
    lefts, rights = _pairs(3)

    def gen():
        yield from lefts[:2]
        raise BoomError(CAMERA)

    it = run_many(Fake(), gen(), rights)
    assert [next(it).index, next(it).index] == [0, 1]
    with pytest.raises(BoomError, match="camera unplugged"):
        next(it)
    assert _no_threads()


def test_writer_errors_name_the_pair(tmp_path):
    lefts, rights = _pairs(3)
    out = tmp_path / "o"
    out.mkdir()
    (out / "000001.tif").mkdir()  # a directory in the way of the second output
    it = run_many(Fake(), lefts, rights, save_to=out)
    assert next(it).index == 0
    with pytest.raises(InputError, match="is a directory") as info:
        next(it)
    assert "pair 1" in str(info.value)  # an InputError names the pair in its message
    assert _no_threads()


@pytest.mark.slow
@pytest.mark.parametrize("how", ["end", "close", "with_break", "gc"])
def test_no_thread_leak(how):
    lefts, rights = _pairs(12)
    if how == "end":
        list(run_many(Fake(), lefts, rights, prefetch=3))
    elif how == "close":
        it = run_many(Fake(), lefts, rights, prefetch=3)
        next(it)
        it.close()
    elif how == "with_break":
        with run_many(Fake(), lefts, rights, prefetch=3) as it:
            for _ in it:
                break
    else:
        it = run_many(Fake(), lefts, rights, prefetch=3)
        next(it)
        del it
        gc.collect()
    assert _no_threads()


def _engine_parts(**kw):
    parts = {
        "read": lambda item: item,
        "run": lambda data, _copies: data,
        "finish": lambda payload: payload,
        "on_skip": lambda _i, _item, e: e,
        "describe": lambda i, _item: f"item {i}",
        "setup": None,
        "device": torch.device("cpu"),
        "readers": 1,
        "writers": 1,
        "prefetch": 2,
        "on_error": "raise",
        "overlap": False,
    }
    return parts | kw


@pytest.mark.parametrize("taken", [0, 1])
@pytest.mark.parametrize("how", ["close", "with"])
def test_closed_pipeline_drops_results_and_raises(taken, how):  # PIPE-01
    pipe = core.Pipeline(range(4), **_engine_parts(readers=0, writers=0, prefetch=4))
    futures = list(pipe._engine.outs)
    while len(futures) < 4:  # the feeder queues the four records
        time.sleep(0.01)
        futures = list(pipe._engine.outs)
    wait(futures, timeout=10)
    assert [next(pipe) for _ in range(taken)] == list(range(taken))
    if how == "close":
        pipe.close()
    else:
        with pipe:
            pass
    with pytest.raises(core.PipelineClosedError):
        next(pipe)
    assert not pipe._engine.outs
    pipe.close()  # idempotent
    assert _no_threads()


def test_exhausted_pipeline_stays_exhausted():
    pipe = core.Pipeline(range(3), **_engine_parts())
    assert list(pipe) == [0, 1, 2]
    pipe.close()
    with pytest.raises(StopIteration):
        next(pipe)


def test_close_does_not_start_queued_writes():  # PIPE-02
    started, first, release = [], threading.Event(), threading.Event()

    def finish(payload):
        started.append(payload)
        if payload == 0:
            first.set()
            release.wait(10)
        return payload

    pipe = core.Pipeline(range(4), **_engine_parts(readers=0, writers=1, prefetch=4, finish=finish))
    assert first.wait(10)
    while pipe._engine.writers._work_queue.qsize() < 3:  # the other three writes are queued
        time.sleep(0.01)
    closer = threading.Thread(target=pipe.close)
    closer.start()
    while not pipe._engine.closing.is_set():
        time.sleep(0.01)
    while len(pipe._engine.writes) > 1:  # the queued writes are cancelled; the active one is waiting
        time.sleep(0.01)
    release.set()
    closer.join(10)
    assert not closer.is_alive()
    assert started == [0]
    assert _no_threads()


def test_close_from_a_pipeline_thread_does_not_deadlock():
    """A close in a writer (e.g. the finalizer run by a garbage collection there) must not wait for that writer."""
    holder = []

    def finish(payload):
        if payload == 2:
            holder[0].close()
        return payload

    holder.append(core.Pipeline(range(10), **_engine_parts(finish=finish)))
    got = []

    def go():
        with contextlib.suppress(RuntimeError):
            got.extend(holder[0])

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "close() in a writer deadlocked"
    assert got[:2] == [0, 1]
    assert _no_threads()


def test_writers_do_not_deadlock(tmp_path):
    """Writers call `write_tiff`, which itself waits on the shared io pool: 16 pairs, 2 writers, prefetch 2."""
    lefts, rights = _pairs(16, 256, 320)

    def go():
        out = list(run_many(Fake(), lefts, rights, save_to=tmp_path, writers=2, prefetch=2))
        assert len(out) == 16

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive(), "run_many hung"


def test_skip_existing_and_atomic_resume(tmp_path):
    left, right = _folders(tmp_path, [f"{i}.png" for i in range(4)])
    out = tmp_path / "o"
    out.mkdir()
    (out / "1.tif").write_bytes(b"kept")
    (out / ".2.fastforks-tmp.tif").write_bytes(b"half")  # a crashed earlier run
    model = Fake()
    res = list(run_many(model, left, right, save_to=out, skip_existing=True))
    assert [r.path.name for r in res] == ["0.tif", "1.tif", "2.tif", "3.tif"]
    assert model.calls == 3
    assert (out / "1.tif").read_bytes() == b"kept"
    assert sorted(p.name for p in out.iterdir()) == ["0.tif", "1.tif", "2.tif", "3.tif"]


def test_mixed_sizes():
    sizes = [(H, W), (64, 96), (H, W), (64, 96)]
    ps = [pair(h, w, i) for i, (h, w) in enumerate(sizes)]
    out = list(run_many(Fake(), [p[0] for p in ps], [p[1] for p in ps]))
    for r, (left, right) in zip(out, ps, strict=True):
        np.testing.assert_array_equal(r.disparity, _expected(left, right))


def test_rectified_rig_gives_depth():
    lefts, rights = _pairs(2)
    rig = g.RectifiedRig(f=50.0, cx=W / 2, cy=H / 2, baseline=0.1, size=(H, W))
    out = list(run_many(Fake(), lefts, rights, rig=rig))
    np.testing.assert_array_equal(out[0].disparity, _expected(lefts[0], rights[0]))
    np.testing.assert_allclose(out[0].depth(), rig.depth(_expected(lefts[0], rights[0])))


@pytest.mark.model
def test_loop_body_may_call_the_same_model():
    from fastforks import stereo  # noqa: PLC0415

    model = stereo.load("hitnet-eth3d@1", device="cpu", mode="portable")
    lefts, rights = _pairs(4, 96, 128)
    ref = [model(left, right).disparity.numpy().copy() for left, right in zip(lefts, rights, strict=True)]
    done = threading.Event()

    def go():
        for r in model.run_many(lefts, rights, prefetch=2):
            np.testing.assert_array_equal(model(lefts[r.index], rights[r.index]).disparity.numpy(), ref[r.index])
            np.testing.assert_array_equal(r.disparity, ref[r.index])
        done.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(120)
    assert done.is_set()


def test_autocast_of_the_caller_applies():
    seen = []

    class M:
        device = torch.device("cpu")

        def __call__(self, left, right):
            seen.append(torch.is_autocast_enabled("cpu"))
            return Fake()(left, right)

    lefts, rights = _pairs(2)
    with torch.autocast("cpu", torch.bfloat16):
        it = run_many(M(), lefts, rights)
    list(it)
    assert seen == [True, True]


@pytest.mark.slow
def test_dev_mode_has_no_warnings(tmp_path):
    code = (
        "import sys; sys.path.insert(0, 'tests');"
        "from test_pipeline import Fake, _pairs; from fastforks.stereo.pipeline import run_many;"
        f"l, r = _pairs(6); list(run_many(Fake(), l, r, save_to={str(tmp_path)!r}));"
        "it = run_many(Fake(), l, r); next(it); del it"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "dev", "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=Path(__file__).parent.parent,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stderr


# pinned ring --------------------------------------------------------------------------------------------------------
def test_pinned_ring_reuses_grows_and_unregisters():
    registered = {}

    def register(buf):
        registered[buf.ctypes.data] = buf.nbytes
        return lambda: registered.pop(buf.ctypes.data)

    ring = core.PinnedRing(2, register=register)
    a = ring.try_acquire(100)
    b = ring.try_acquire(100)
    assert {a.ctypes.data % 4096, b.ctypes.data % 4096} == {0}
    assert ring.try_acquire(100) is None  # all slots in use
    ring.release(a)
    assert ring.try_acquire(50) is a  # reused once collected
    ring.release(b)
    c = ring.try_acquire(1000)  # a free smaller slot is replaced by a larger one
    assert (c.nbytes, len(registered)) == (1000, 2)
    ring.release(a)
    ring.release(c)
    ring.close()
    assert not registered
    assert ring.peak_bytes == 1100


def test_registration_failure_falls_back(monkeypatch):
    def refuse(buf):
        raise core._RegisterError(str(buf.nbytes))

    monkeypatch.setattr(core, "_register", refuse)
    lefts, rights = _pairs(3)
    out = list(run_many(Fake(), lefts, rights))
    np.testing.assert_array_equal(out[2].disparity, _expected(lefts[2], rights[2]))


# GPU ----------------------------------------------------------------------------------------------------------------
gpu = pytest.mark.gpu


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


@gpu
def test_slow_compute_does_not_see_later_uploads():
    """A slow kernel per pair: the device ring must not overwrite the inputs of a pair still being computed."""
    _cuda()
    lefts, rights = _pairs(12, 256, 320)
    fake = Fake("cuda")
    ref = [fake(_t(left), _t(right)).disparity[0].cpu().numpy() for left, right in zip(lefts, rights, strict=True)]
    for prefetch in (1, 3, 8):
        out = list(run_many(Fake("cuda", sleep_cycles=20_000_000), lefts, rights, prefetch=prefetch))
        for r in out:
            np.testing.assert_array_equal(r.disparity, ref[r.index])
    assert _no_threads()


class _Syncing(Fake):
    """A model with a host sync inside (`.item()`), as several real ones have: its call returns after its compute."""

    def __call__(self, left, right):
        out = super().__call__(left, right)
        out.disparity.sum().item()
        return out


@gpu
def test_first_result_does_not_wait_for_the_next_pair():
    """The GPU thread blocks inside the next pair's call; each result must come out right after its compute.

    Pair 0 (direct path) after ~1 pair time, pair 1 (through the collector) after ~2, not one pair later each.
    """
    _cuda()
    lefts, rights = _pairs(4, 1536, 2048)  # a download that takes longer than the Python between record and query
    model = _Syncing("cuda", sleep_cycles=1)
    model(_t(lefts[0]).cuda(), _t(rights[0]).cuda())
    t = time.perf_counter()
    torch.cuda._sleep(10_000_000)
    torch.cuda.synchronize()
    model.sleep_cycles = int(10_000_000 * 0.3 / (time.perf_counter() - t))  # ~0.3 s per pair
    t = time.perf_counter()
    model(_t(lefts[0]).cuda(), _t(rights[0]).cuda())
    per_pair = time.perf_counter() - t
    with run_many(model, lefts, rights, prefetch=4) as it:
        t = time.perf_counter()
        first = next(it)
        first_s = time.perf_counter() - t
        second = next(it)
        second_s = time.perf_counter() - t
        rest = list(it)
    assert [r.index for r in [first, second, *rest]] == [0, 1, 2, 3]
    assert first_s < 1.5 * per_pair, (first_s, per_pair)
    assert second_s < 2.5 * per_pair, (second_s, per_pair)


def _hash(a) -> str:
    a = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


FA_PLUS = "fast-acvnet-plus-generalization"


@gpu
@pytest.mark.model
@pytest.mark.parametrize(
    ("ref", "mode"), [("hitnet-eth3d@1", "kernels"), ("hitnet-eth3d@1", "portable"), (FA_PLUS, "kernels")]
)
def test_bit_identical_to_a_loop(tmp_path, ref, mode):
    """Same bytes as a plain loop over single calls, with and without a rig, at several prefetch depths, twice.

    The first run_many comes before the loop: its first calls at the size are the model's first (Fast-ACVNet+ max
    builds cuDNN plans after a few calls and runs CUDA graphs: equal from call 1, in another thread).
    """
    _cuda()
    from fastforks import stereo  # noqa: PLC0415
    from test_geometry_rectify import MILD, _rig  # noqa: PLC0415

    model = stereo.load(ref, device="cuda", mode=mode)
    rect = g.rectify(_rig(MILD))
    lefts, rights = _pairs(8, 480, 640)
    cold = [_hash(r.disparity) for r in model.run_many(lefts, rights, prefetch=3)]
    plain, rig = [], []
    for left, right in zip(lefts, rights, strict=True):
        plain.append(_hash(model(left, right).disparity))
        lt, rt = (torch.from_numpy(x).permute(2, 0, 1)[None].cuda() for x in (left, right))
        lt, rt = rect.remap(lt, rt)
        rig.append(_hash(rect.unrectify(rect.depth(model(lt, rt).disparity[0]))))
    assert cold == plain
    for prefetch in (1, 2, 5):
        for _ in range(2):
            assert [_hash(r.disparity) for r in model.run_many(lefts, rights, prefetch=prefetch)] == plain
            assert [_hash(r.depth()) for r in model.run_many(lefts, rights, rig=rect, prefetch=prefetch)] == rig
    saved = [r.path for r in model.run_many(lefts, rights, rig=rect, save_to=tmp_path, save_format=".npy")]
    assert [_hash(np.load(p)) for p in saved] == rig
    assert _no_threads()


@gpu
def test_results_before_a_gpu_error_are_delivered():
    """A fatal error at pair 3 (CUDA OOM): pairs 0-2, still downloading at that moment, come out, then the error."""
    _cuda()

    class Failing(Fake):
        def __call__(self, left, right):
            if self.calls == 3:
                raise torch.cuda.OutOfMemoryError(OOM)
            return super().__call__(left, right)

    lefts, rights = _pairs(6, 256, 320)
    got = []
    with pytest.raises(torch.cuda.OutOfMemoryError):
        got.extend(r.index for r in run_many(Failing("cuda", sleep_cycles=50_000_000), lefts, rights, prefetch=4))
    assert got == [0, 1, 2]
    assert _no_threads()


@gpu
def test_failed_items_return_their_page_locked_slots():
    """`run` raises after its download (on_error="skip"): the slot must come back, or the ring runs dry and hangs."""
    _cuda()

    def run(data, copies):
        copies.download(copies.upload(torch.from_numpy(data)) * 2)
        msg = "bad item"
        raise InputError(msg)

    parts = _engine_parts(run=run, device=torch.device("cuda"), overlap=True, on_error="skip", prefetch=2)
    items = [np.full((64, 64), i, np.float32) for i in range(12)]
    got = []

    def go():
        got.extend(core.Pipeline(items, **parts))

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "the pinned ring ran dry"
    assert len(got) == 12
    assert all(isinstance(e, InputError) for e in got)
    assert _no_threads()


@gpu
@pytest.mark.model
def test_concurrent_pipelines_match_loops():
    """Two run_many iterators of two kernels models, interleaved, plus calls in the loop body: each equals a loop.

    Fast-ACVNet+ kernels builds cuDNN plans after a few calls per geometry (`kernels.begin_call` accounting); the
    process-wide CUDA call lock keeps that per call while the other pipeline's thread calls hitnet.
    """
    _cuda()
    from fastforks import stereo  # noqa: PLC0415

    models = [stereo.load(ref, device="cuda", mode="kernels") for ref in ("hitnet-eth3d@1", FA_PLUS)]
    lefts, rights = _pairs(8, 480, 640)
    its = [m.run_many(lefts, rights, prefetch=3) for m in models]
    got: list[list[str]] = [[], []]
    for _ in lefts:
        for k, it in enumerate(its):
            got[k].append(_hash(next(it).disparity))
            models[1 - k](lefts[0], rights[0])  # a call of the other model from this thread meanwhile
    for it in its:
        assert next(it, None) is None
    for k, m in enumerate(models):
        assert got[k] == [_hash(m(left, right).disparity) for left, right in zip(lefts, rights, strict=True)]
    assert _no_threads()


@gpu
def test_gpu_errors_raise_with_skip():
    _cuda()

    class M:
        device = torch.device("cuda")

        def __call__(self, *_):
            raise torch.cuda.OutOfMemoryError(OOM)

    lefts, rights = _pairs(3)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(run_many(M(), lefts, rights, on_error="skip"))
    assert _no_threads()


def test_pair_result_is_exported():
    from fastforks import stereo  # noqa: PLC0415

    assert stereo.PairResult is PairResult
