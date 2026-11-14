"""Every renamed 0.1.1 name still works and warns (`FutureWarning` naming the replacement and the removal date)."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

import fastforks
from fastforks import OptionsError, stereo
from fastforks._core import registry
from fastforks._core.arguments import REMOVAL
from fastforks.stereo._runner import Runner

pytestmark = pytest.mark.usefixtures("card_min_size")  # tiny images through fake runners


@dataclasses.dataclass(frozen=True)
class _P:
    iters: int


class _Iters(Runner):
    """Disparity = the `iters` option."""

    Params = _P

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        return {"disparity": left[:, 0] * 0 + float(self.ctx.params.iters)}


@pytest.fixture
def iters_runner(monkeypatch):
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": _Iters}))


def _warns(new: str):
    return pytest.warns(FutureWarning, match=rf"renamed to {new}.*removed after {REMOVAL}")


def test_load_options_dict(iters_runner):
    img = np.zeros((4, 5, 3), np.uint8)
    for load in (stereo.load, fastforks.load):
        with _warns(r"load\(\.\.\., \*\*options\)"):
            m = load("raft-stereo-sceneflow", device="cpu", options={"iters": 7})
        assert m.options == {"iters": 7}
        assert float(m(img, img).disparity[0, 0]) == 7
    with _warns(r"load\(\.\.\., \*\*options\)"), pytest.raises(OptionsError, match="load\\(\\) argument"):
        stereo.load("raft-stereo-sceneflow", device="cpu", options={"mode": "portable"})
    with _warns(r"load\(\.\.\., \*\*options\)"), pytest.raises(OptionsError, match="also given"):
        stereo.load("raft-stereo-sceneflow", device="cpu", iters=3, options={"iters": 7})
