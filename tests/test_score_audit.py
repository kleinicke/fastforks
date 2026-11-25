"""The audit uses recorded comparisons that share the same measurement scope."""

from score_audit import _accuracy_pairs, _portable_reference


def test_backend_timing_uses_its_paired_portable_arm():
    compiled = {"input": "192x192", "ms": 7024, "paired_portable_ms": 9464}
    native = [{"input": "192x192", "ms": 1848}, {"input": "384x384", "ms": 2500}]
    assert _portable_reference(compiled, native) == 9464
    assert _portable_reference({"input": "192x192"}, native) == 1848
    assert _portable_reference({"input": "512x512"}, native) is None


def test_accuracy_pairs_use_matching_dtype_and_labeled_cohort():
    base = {"dataset": "test", "n": 200, "metrics": {"epe": 1.25}}
    original32 = base | {"dtype": "float32"}
    strict32 = original32 | {"metrics": {"epe": 1.20}, "date": "2026-11-20"}
    original16 = base | {"dtype": "float16", "dataset_manifest_sha256": "same", "date": "2026-11-18"}
    other_cohort = original16 | {"dataset_manifest_sha256": "different", "date": "2026-11-19"}
    subset = original16 | {"n": 50, "date": "2026-11-20"}
    portable32 = original32 | {"metrics": {"epe": 1.24}}
    kernels16 = original16 | {"metrics": {"epe": 1.29}}

    assert list(
        _accuracy_pairs(
            {
                "upstream": [original32],
                "upstream-fp32-strict": [strict32],
                "upstream-fp16": [original16, other_cohort, subset],
                "portable": [portable32],
                "kernels": [kernels16],
            }
        )
    ) == [("portable", portable32, original32), ("kernels", kernels16, original16)]
