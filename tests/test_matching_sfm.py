import sqlite3

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import matching
from fastforks._core import cli
from fastforks._core.errors import InputError
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
