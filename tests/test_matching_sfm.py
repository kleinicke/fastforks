import sqlite3
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import matching
from fastforks._core import cli
from fastforks._core.errors import InputError, MissingDependencyError, OptionsError
from fastforks.matching import relative_pose, sfm
from fastforks.matching._api import SparseMatchingResult

N = 20  # keypoints per image


class FakeMatcher:
    """Sparse matcher stub: image i has keypoints (i, k); pair (1, 2) gets too few matches to verify."""

    sparse = True

    def match_pairs(self, _images, pairs):
        for i, j in pairs:
            k = 5 if (i, j) == (1, 2) else N
            idx = torch.arange(k)
            kp = [torch.stack([torch.full((N,), float(x)), torch.arange(N, dtype=torch.float32)], 1) for x in (i, j)]
            yield SparseMatchingResult(kp[0], kp[1], torch.stack([idx, idx], 1), torch.ones(k), "fake@1", "portable")


def fake_fundamental(p1, p2, *, threshold, mask, mode):
    f = torch.arange(9, dtype=torch.float32).reshape(3, 3).expand(len(p1), 3, 3)
    inliers = mask.clone()
    inliers[:, 0] = False  # the first match of every pair is an outlier
    return relative_pose.FundamentalMatrix(f, inliers, torch.ones(len(p1), dtype=torch.bool))


@pytest.fixture
def scene(tmp_path, monkeypatch):
    (tmp_path / "img" / "sub").mkdir(parents=True)
    for name, size in (("a.png", (32, 24)), ("b.jpg", (32, 24)), ("sub/c.png", (16, 16))):
        Image.new("RGB", size).save(tmp_path / "img" / name)
    (tmp_path / "img" / "notes.txt").write_text("not an image")
    monkeypatch.setattr(matching, "load", lambda *_a, **_k: FakeMatcher())
    monkeypatch.setattr(relative_pose, "estimate_fundamental", fake_fundamental)
    return tmp_path


def test_cli_writes_colmap_database(scene, capsys):
    db = scene / "db.db"
    assert cli.main(["sfm", str(scene / "img"), str(db), "--retrieval", "none", "--single-camera"]) == 0
    assert "3 images, 3 pairs, 2 verified" in capsys.readouterr().out
    con = sqlite3.connect(db)
    names = con.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()
    assert names == [(1, "a.png", 1), (2, "b.jpg", 1), (3, "sub/c.png", 2)]  # one camera per image size
    width, height, params = con.execute("SELECT width, height, params FROM cameras WHERE camera_id = 1").fetchone()
    assert (width, height) == (32, 24)
    np.testing.assert_allclose(np.frombuffer(params), [38.4, 16, 12, 0])
    rows, data = con.execute("SELECT rows, data FROM keypoints WHERE image_id = 2").fetchone()
    kp = np.frombuffer(data, np.float32).reshape(rows, 2)
    np.testing.assert_array_equal(kp[3], [1.5, 3.5])  # COLMAP: pixel corners at integers
    raw = dict(con.execute("SELECT pair_id, rows FROM matches").fetchall())
    pair = lambda i, j: i * 2147483647 + j  # noqa: E731
    assert raw == {pair(1, 2): N, pair(1, 3): N, pair(2, 3): 5}
    geo = con.execute("SELECT pair_id, rows, data, config, F FROM two_view_geometries ORDER BY pair_id").fetchall()
    assert [(g[0], g[1], g[3]) for g in geo] == [(pair(1, 2), N - 1, 3), (pair(1, 3), N - 1, 3)]  # 3: UNCALIBRATED
    np.testing.assert_array_equal(np.frombuffer(geo[0][2], np.uint32).reshape(-1, 2)[0], [1, 1])
    f = np.frombuffer(geo[0][4]).reshape(3, 3)  # the same epipolar constraint in COLMAP's pixels
    x1, x2 = np.array([3.0, 4.0, 1.0]), np.array([5.0, 7.0, 1.0])
    shifted = np.array([0.5, 0.5, 0.0])
    want = x2 @ np.arange(9.0).reshape(3, 3) @ x1
    assert (x2 + shifted) @ f @ (x1 + shifted) == pytest.approx(want)


def test_rejects_existing_database_and_empty_folder(scene):
    (scene / "db.db").touch()
    with pytest.raises(InputError, match="exists"):
        sfm.colmap_database(scene / "img", scene / "db.db")
    (scene / "empty").mkdir()
    with pytest.raises(InputError, match="no images"):
        sfm.colmap_database(scene / "empty", scene / "new.db")


def test_matcher_options_reach_load(scene, monkeypatch):
    seen = {}
    monkeypatch.setattr(matching, "load", lambda *_a, **k: seen.update(k) or FakeMatcher())
    sfm.colmap_database(scene / "img", scene / "db.db", retrieval=None, matcher_options={"long_side": 1600})
    assert seen["options"] == {"long_side": 1600}


class FakeReconstruction:
    def __init__(self, registered, points):
        self._registered, self._points = registered, points

    def num_reg_images(self):
        return self._registered

    def num_points3D(self):  # noqa: N802 - pycolmap's name
        return self._points


@pytest.fixture
def fake_pycolmap(monkeypatch):
    """A `pycolmap` module whose mappers record their call and return two models (the second one larger)."""
    calls = []

    def mapping(name):
        def run(database, images, out):
            calls.append((name, database, images, out))
            return {0: FakeReconstruction(2, 10), 1: FakeReconstruction(3, 40)}

        return run

    module = types.ModuleType("pycolmap")
    module.incremental_mapping = mapping("incremental")
    module.global_mapping = mapping("global")
    monkeypatch.setitem(sys.modules, "pycolmap", module)
    return calls


def test_reconstruct_writes_the_database_then_maps(scene, fake_pycolmap, capsys):
    out = scene / "out"
    rec = sfm.reconstruct(scene / "img", out, retrieval=None, mapper="global")
    assert fake_pycolmap == [("global", out / "database.db", scene / "img", out / "sparse")]
    assert (out / "database.db").is_file()
    assert (out / "sparse").is_dir()
    assert (rec.database.images, rec.database.verified) == (3, 2)
    assert (rec.models, rec.registered, rec.points, rec.model) == (2, 3, 40, out / "sparse" / "1")
    with pytest.raises(InputError, match="exists"):  # the database is there now
        sfm.reconstruct(scene / "img", out, retrieval=None)
    with pytest.raises(OptionsError, match="mapper must be"):
        sfm.reconstruct(scene / "img", scene / "other", mapper="hierarchical")
    assert cli.main(["sfm", str(scene / "img"), str(scene / "cli"), "--retrieval", "none", "--reconstruct"]) == 0
    text = capsys.readouterr().out
    assert "2 verified" in text
    assert f"3/3 images, 40 points -> {scene / 'cli' / 'sparse' / '1'}" in text
    assert fake_pycolmap[-1][0] == "incremental"


def test_reconstruct_names_the_sfm_extra_without_pycolmap(scene, monkeypatch):
    monkeypatch.setitem(sys.modules, "pycolmap", None)  # import fails
    with pytest.raises(MissingDependencyError, match=r"fastforks-core\[sfm\]"):
        sfm.reconstruct(scene / "img", scene / "out")
    assert not (scene / "out").exists()
