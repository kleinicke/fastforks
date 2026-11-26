"""Card-declared Apple MLX `(mode, dtype)` combinations against the passing survey runs (no MLX needed).

A card's `mlx_dtypes` say what `load(..., device="mlx")` accepts; the `passed` key of `apple_mlx.json` (written by
`tools/gen_auto_mode.py` from the MLX survey rows whose golden check passed) says what was compared with the reference
outputs, and is what `contract()` reports as validated.
"""

from __future__ import annotations

from fastforks._core import registry
from fastforks._core.model import _apple_mlx

# Declared, loadable with an explicit `mode` / `dtype`, and without a passing MLX survey row so far. A waiver list,
# not a goal: run the MLX survey for an entry (`tools/speed_survey.py`), regenerate the table and drop it here, or
# drop the dtype from the card if its golden fails.
UNVALIDATED: dict[str, tuple[str, ...]] = {
    "foundationstereo-vit-l@1": (
        "kernels float16",
        "kernels float32",
        "portable float32",
    ),
    "foundationstereo-vit-s@1": ("kernels float32",),
    "totalsegmentator-total-6mm@1": (
        "portable float16",
        "portable float32",
    ),
    "totalsegmentator-v3-total-3mm@1": (
        "portable float16",
        "portable float32",
    ),
    "totalsegmentator-v3-total-6mm@1": (
        "portable float16",
        "portable float32",
    ),
    "totalsegmentator-v3-total-small-3mm@1": (
        "portable float16",
        "portable float32",
    ),
    "totalsegmentator-v3-total-small@1": (
        "portable float16",
        "portable float32",
    ),
    "totalsegmentator-v3-total@1": (
        "portable float16",
        "portable float32",
    ),
}


def _declared_without_a_passing_run() -> dict[str, tuple[str, ...]]:
    table, missing = _apple_mlx(), {}
    for task in registry.TASKS:
        for card, variant in registry.cards(task).values():
            if not card.portable_mlx_dtypes:
                continue
            declared = registry.variant_card(card, variant)
            passed = table.get(variant.ref, {}).get("passed", {})
            combos = tuple(
                sorted(
                    f"{mode} {dtype}"
                    for mode in ("portable", "kernels")
                    for dtype in declared.mlx_dtypes(mode)
                    if dtype not in passed.get(mode, ())
                )
            )
            if combos:
                missing[variant.ref] = combos
    return missing


def test_every_declared_mlx_combination_has_a_passing_survey_run():
    missing = _declared_without_a_passing_run()
    new = {ref: sorted(set(c) - set(UNVALIDATED.get(ref, ()))) for ref, c in missing.items()}
    new = {ref: c for ref, c in new.items() if c}
    assert not new, f"declared for MLX without a passing survey row (run the survey, or waive in UNVALIDATED): {new}"


def test_the_waiver_list_holds_no_validated_combination():
    missing = _declared_without_a_passing_run()
    stale = {ref: sorted(set(c) - set(missing.get(ref, ()))) for ref, c in UNVALIDATED.items()}
    stale = {ref: c for ref, c in stale.items() if c}
    assert not stale, f"validated (or no longer declared): remove from UNVALIDATED: {stale}"


def test_contract_reports_only_passing_mlx_runs_as_validated():
    card = registry.variant_card(*registry.resolve("stereo", "foundationstereo-vit-s"))
    modes = card.contract().modes
    waived = {c.split()[1] for c in UNVALIDATED["foundationstereo-vit-s@1"] if c.startswith("kernels ")}
    assert waived == {"float32"}
    assert modes["kernels"]["mlx"]["implemented"] == card.kernels_mlx_dtypes  # declared: `load` accepts them
    # ... and only the ones with a passing run on the reference Mac are validated
    assert set(modes["kernels"]["mlx"]["validated"]) == set(card.kernels_mlx_dtypes) - waived
    assert modes["portable"]["mlx"]["validated"] == card.portable_mlx_dtypes
