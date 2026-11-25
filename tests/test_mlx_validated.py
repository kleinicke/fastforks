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
    "davit_base.msft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "davit_small.msft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "davit_tiny.msft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "deit_base_patch16_224.fb_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformer_l1.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformer_l3.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformer_l7.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformerv2_l.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformerv2_s0.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformerv2_s1.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientformerv2_s2.snap_dist_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "efficientvit_b2.r288_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "falconsai-nsfw-vit@1": (
        "kernels float16",
        "kernels float32",
    ),
    "focalnet_base_lrf.ms_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "focalnet_base_srf.ms_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "focalnet_small_lrf.ms_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "focalnet_small_srf.ms_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "focalnet_tiny_lrf.ms_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "focalnet_tiny_srf.ms_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "foundationstereo-vit-l@1": (
        "kernels float16",
        "kernels float32",
        "portable float32",
    ),
    "foundationstereo-vit-s@1": ("kernels float32",),
    "hgnetv2_b0.ssld_stage2_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "hgnetv2_b1.ssld_stage2_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "hgnetv2_b3.ssld_stage2_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "hgnetv2_b4.ssld_stage2_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "hgnetv2_b5.ssld_stage2_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "hgnetv2_b6.ssld_stage2_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "hrnet-w48-fcn-ade20k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "inception_v3.tv_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "mobilenetv4_conv_medium.e500_r256_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "segformer-b5-face-parsing@1": (
        "kernels float16",
        "kernels float32",
    ),
    "shvit_s1.in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "shvit_s2.in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "shvit_s3.in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "shvit_s4.in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "tiny_vit_11m_224.dist_in22k_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "tiny_vit_21m_224.dist_in22k_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "tiny_vit_21m_384.dist_in22k_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "tiny_vit_21m_512.dist_in22k_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
    "tiny_vit_5m_224.dist_in22k_ft_in1k@1": (
        "kernels float16",
        "kernels float32",
    ),
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
    "yolox-l@1": (
        "kernels float16",
        "kernels float32",
    ),
    "yolox-m@1": (
        "kernels float16",
        "kernels float32",
    ),
    "yolox-x@1": (
        "kernels float16",
        "kernels float32",
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
    card = registry.variant_card(*registry.resolve("segmentation", "segformer-b5-face-parsing"))
    modes = card.contract().modes
    assert modes["kernels"]["mlx"]["implemented"] == card.kernels_mlx_dtypes  # declared: `load` accepts them
    assert modes["kernels"]["mlx"]["validated"] == ()  # ... no passing kernels run on the reference Mac
    assert modes["portable"]["mlx"]["validated"] == card.portable_mlx_dtypes
    assert set(UNVALIDATED["segformer-b5-face-parsing@1"]) == {f"kernels {d}" for d in card.kernels_mlx_dtypes}
