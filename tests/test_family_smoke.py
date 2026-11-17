"""Real-use smoke of every family in every mode (`tools/family_smoke.py`): load the default model through the public
API, one forward on the task's minimal bench input at ~256 px, a typed finite result. Catches what only a real call
shows (a kernels-memory change once broke cotracker3 in `auto` mode while its parity tests passed)."""

from __future__ import annotations

import pytest
import torch

import family_smoke as smoke

CASES = [(m, mode) for task, m in smoke.default_models() for mode in smoke.modes_of(m, task, smoke.MODES)]


def test_every_family_is_covered() -> None:
    from fastforks._core import registry  # noqa: PLC0415

    families = {card.family for task in registry.TASKS for card, _ in registry.cards(task).values()}
    assert len(smoke.default_models()) == len(families)
    assert {m for m, _ in CASES} >= {m for _t, m in smoke.default_models()}


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("model", "mode"), CASES)
def test_default_model_runs_in_mode(model: str, mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    row = smoke.run_one(model, mode)
    if row["status"] == "skip":
        pytest.skip(row["error"])
    assert row["status"] == "ok", f"{model} [{mode}]: {row.get('error')}"
