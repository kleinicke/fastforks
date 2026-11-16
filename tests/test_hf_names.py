"""Repo names under `fastforks/` never collide: weights repos, family cards and the fixed repos (CPU only, fast)."""

import stage_hf


def test_weights_family_and_fixed_names_are_disjoint():
    weights = set(stage_hf.weights_repos())
    families = {stage_hf.repo_name(f) for f in stage_hf._families()}
    fixed = {"kernels", "mirrors", "reference-outputs"}
    assert len(families) == len(stage_hf._families())  # family names are unique after normalisation
    for a, b, what in ((weights, families, "weights vs family"), (weights, fixed, "weights vs fixed"),
                       (families, fixed, "family vs fixed")):  # fmt: skip
        assert not a & b, (what, sorted(a & b))
    assert fixed <= stage_hf.RESERVED_REPOS
