from __future__ import annotations

import copy
import dataclasses
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from fastforks import CardError
from fastforks._core import licences, registry
from fastforks._core.cards import parse_card

MODELS = Path(__file__).parents[1] / "src/fastforks/stereo/models"
S2M2_CARD = MODELS / "s2m2/card.toml"


@pytest.fixture
def data():
    return tomllib.loads(S2M2_CARD.read_text())


@pytest.fixture
def raft():
    d = tomllib.loads((MODELS / "raft_stereo/card.toml").read_text())
    return parse_card(d, family="raft_stereo", task="stereo", where="card")


def parse(d):
    return parse_card(d, family="s2m2", task="stereo", where="card.toml")


def test_parses(data):
    card = parse(data)
    assert card.modes == ("portable", "kernels")
    assert card.kernel_libs == ("s2m2", "s2m2_cutlass", "cudnn_hacc")
    assert card.licence.code_class == "noncommercial"
    assert card.io.pad_fill == "family"
    assert card.params["refine_iter"] == 3
    assert all(card.status(v).commercial == "no" for v in card.variants)
    assert [v.ref for v in card.variants] == ["s2m2-s@1", "s2m2-m@1", "s2m2-l@1", "s2m2-xl@1"]
    assert card.variants[0].weights[0].size == 107453000
    assert card.module == "fastforks.stereo.models.s2m2.model"


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d.pop("name"), "missing key 'name'"),
        (lambda d: d.update(schema=1), "schema must be 2"),
        (lambda d: d.update(tier="beta"), "'beta' not in"),
        (lambda d: d.update(colour="red"), r"unknown key\(s\) \['colour'\]"),
        (lambda d: d["licence"].update(commercial="no"), r"unknown key\(s\) \['commercial'\]"),
        (lambda d: d["licence"].update(code="Unknown-License-1.0"), "unknown licence id"),
        (lambda d: d["licence"].update(weights="not stated"), "assumed from code"),
        (lambda d: d["licence"].update(training_data=["imagenet"]), "bad: \\['imagenet'\\]"),
        (lambda d: d["variants"][0].update(licence={"weights": "Foo-1.0"}), "unknown licence id"),
        (lambda d: d["variants"][0].update(licence={"code": "MIT"}), "unknown key"),
        (lambda d: d["io"].update(pad={"multiple": 0}), "must be >= 1"),
        (lambda d: d["io"].update(pad={"mode": "left"}), "'left' not in"),
        (lambda d: d["io"].update(input="0-65535"), "'0-65535' not in"),
        (lambda d: d["io"].update(pad="auto"), "pad: must be a table"),
        (lambda d: d["variants"][0].update(io={"pad": {"multiple": 0}}), "must be >= 1"),
        (lambda d: d["variants"][0].update(io={"padding": 8}), r"unknown key\(s\) \['padding'\]"),
        (lambda d: d.update(outputs=["depth"]), "must include 'disparity'"),
        (lambda d: d["modes"].pop("portable"), "needs a portable backend"),
        (lambda d: d["modes"].update(turbo={}), "'turbo' not in"),
        (lambda d: d.update(variants=[]), "no variants"),
        (lambda d: d["variants"][0].update(revision="1"), "'revision' must be int"),
        (lambda d: d["variants"][0]["weights"][0].update(sha256="ABC"), "64 lower-case hex"),
        (lambda d: d["variants"][0]["weights"][0].update(member="m.pth"), "needs `archive"),
        (lambda d: d["variants"][0]["weights"][0].update(archive={"size": 1, "sha256": "0" * 64}), "only a member"),
        (
            lambda d: d["variants"][0]["weights"][0].update(member="m", archive={"size": 1, "sha256": "x"}),
            "archive: sha256 must be",
        ),
        (lambda d: d.update(outputs="disparity"), "'outputs' must be list"),
        (lambda d: d["modes"]["kernels"].update(kernels=[1]), "list of strings"),
        (lambda d: d.update(options={"iters": {"default": 0, "min": 1}}), "numeric `default` within numeric `min`"),
        (lambda d: d.update(options={"iters": {"default": 10, "max": 9}}), "numeric `default` within numeric `min`"),
        (lambda d: d.update(options={"iters": {"default": "a", "max": 9}}), "numeric `default` within numeric `min`"),
        (lambda d: d.update(options={"iters": {"default": 1, "foo": 9}}), r"unknown key\(s\) \['foo'\]"),
        (lambda d: d.update(options={"iters": {"min": 1}}), "needs a `default`"),
        (lambda d: d.update(options={"m": {"default": "a", "choices": ["b"]}}), "list holding the default"),
        (lambda d: d.update(options={"m": {"default": 1, "choices": [1, 2], "max": 2}}), "list holding the default"),
        (lambda d: d.update(options={"m": {"default": 1, "choices": [1, "2"]}}), "the default's type int"),
        (lambda d: d.update(options={"m": {"default": True, "effect": "fast"}}), "`effect` must be output"),
    ],
)
def test_rejects(data, edit, match):
    d = copy.deepcopy(data)
    edit(d)
    with pytest.raises(CardError, match=match):
        parse(d)


def test_variant_licence_overrides(raft):
    by_id = {v.id: v for v in raft.variants}
    mid, sf = by_id["raft-stereo-middlebury"], by_id["raft-stereo-sceneflow"]
    assert raft.variant_licence(mid).training_data == ("sceneflow", "middlebury")
    assert raft.variant_licence(sf).training_data == ("sceneflow",)
    st = raft.status(sf)
    assert (st.code_class, st.weights, st.training_data, st.commercial) == (
        "permissive",
        "MIT (assumed from code)",
        ("sceneflow",),
        "yes",
    )


def test_permissive_code_assumed_weights_nc_data_is_yes_with_notes(raft):
    st = raft.status(next(v for v in raft.variants if v.id == "raft-stereo-sceneflow"))
    assert licences.data_commercial(st.training_data) == "unclear"  # the data is not cleared ...
    assert st.commercial == "yes"  # ... but it does not decide the status
    assert st.reasons == ()
    assert "weights licence not stated by the original authors; assumed to be the code licence" in st.notes
    assert "data: sceneflow (non-commercial)" in st.notes


def test_permissive_code_nc_weights_is_no(raft):
    v = next(v for v in raft.variants if v.id == "raft-stereo-sceneflow")
    nc = dataclasses.replace(raft, licence=dataclasses.replace(raft.licence, weights="CC-BY-NC-4.0"))
    st = nc.status(v)
    assert st.commercial == "no"
    assert st.reasons == ("weights: CC-BY-NC-4.0",)


def test_variant_io_overrides_are_merged(data):
    d = copy.deepcopy(data)
    d["variants"][1]["io"] = {"pad": {"multiple": 96}, "min_size": 64}
    card = parse(d)
    first, second = card.variants[0], card.variants[1]
    assert first.io == card.io
    assert second.io.pad_multiple == 96
    assert second.io.min_size == 64
    assert (second.io.pad_mode, second.io.pad_fill) == (card.io.pad_mode, card.io.pad_fill)


def test_pad_none(data):
    d = copy.deepcopy(data)
    d["io"]["pad"] = "none"
    card = parse(d)
    assert (card.io.pad_mode, card.io.pad_multiple) == ("none", 1)


def test_architecture_lives_in_the_cards():
    """Per-variant padding and encoder choices are card data (no `_Padded` runner, no id -> encoder table)."""
    aanet = parse_card(
        tomllib.loads((MODELS / "aanet/card.toml").read_text()), family="aanet", task="stereo", where="card"
    )
    assert {v.id: v.io.pad_multiple for v in aanet.variants} == {"aanet-sceneflow": 48, "aanet-plus-sceneflow": 96}
    fs = parse_card(
        tomllib.loads((MODELS / "foundationstereo/card.toml").read_text()),
        family="foundationstereo",
        task="stereo",
        where="card",
    )
    assert {v.id: {**fs.params, **v.params}["encoder"] for v in fs.variants} == {
        "foundationstereo-vit-l": "vitl",
        "foundationstereo-vit-s": "vits",
    }


def test_limitations_optional_and_checked(data):
    data.pop("limitations", None)
    assert parse(data).limitations == ()
    data["limitations"] = ["Does not run below 64 px.", "Kernels kernels missing for variant x."]
    assert parse(data).limitations == tuple(data["limitations"])
    for bad in ("text", [1], [""], ["a", "a"], ["two\nlines"], ["x" * 301]):
        data["limitations"] = bad
        with pytest.raises(CardError, match="limitations"):
            parse(data)


def test_families_without_limitations_are_listed():
    """Not an error (the field is optional): the skip message names the families whose card documents none."""
    from fastforks._core import registry  # noqa: PLC0415

    missing = sorted(
        f"{card.task}/{card.family}"
        for task in registry.TASKS
        for card in {c.family: c for c, _ in registry.cards(task).values()}.values()
        if not card.limitations
    )
    if missing:
        pytest.skip(f"families without `limitations` in card.toml: {', '.join(missing)}")


def test_licence_notes_checked(data):
    data["licence_notes"] = ["A note.", "A second note."]
    assert parse(data).licence_notes == ("A note.", "A second note.")
    for bad in (["x" * 451], [""], ["a\nb"], ["dup", "dup"]):
        data["licence_notes"] = bad
        with pytest.raises(CardError, match="licence_notes"):
            parse(data)


def test_families_with_licence_notes_keep_their_explicit_nc_status():
    noted = {}
    for task in registry.TASKS:
        for card, v in registry.cards(task).values():
            if card.licence_notes:
                noted.setdefault((task, card.family), []).append(card.status(v).commercial)
    assert {f for _, f in noted} >= {"foundationstereo", "waft_stereo", "c_foundationstereo", "las2"}
    assert set(noted["stereo", "foundationstereo"]) == {"no"}  # NVIDIA NC licence: notes never turn `no` into `yes`


def _assert_frozen(value: object, where: str) -> None:
    assert not isinstance(value, (dict, list, set)), f"{where}: mutable {type(value).__name__}"
    if isinstance(value, Mapping):
        for k, v in value.items():
            _assert_frozen(v, f"{where}.{k}")
    elif isinstance(value, tuple):
        for i, v in enumerate(value):
            _assert_frozen(v, f"{where}[{i}]")


def test_cards_are_deeply_read_only():
    """Cards are shared by every load: no nested dict / list a caller could mutate."""
    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            for name in (
                "params",
                "options",
                "option_min",
                "option_max",
                "option_choices",
                "option_effect",
                "output_info",
            ):
                _assert_frozen(getattr(card, name), f"{card.family}.{name}")
            for v in card.variants:
                for name in ("params", "licence", "output_info"):
                    _assert_frozen(getattr(v, name), f"{v.id}.{name}")
    card, _ = registry.resolve_any("moge-2-vits-normal")
    with pytest.raises(TypeError):
        card.params["x"] = 1


def test_option_bounds_parse(data):
    """`{ default, min, max, choices, effect }` land in the card's mappings and `option_bounds` words."""
    d = copy.deepcopy(data)
    d["options"] = {
        "level": {"default": 9, "min": 0, "max": 9},
        "steps": {"default": 4, "min": 1},
        "cap": {"default": 0.5, "max": 1.0},
        "mode": {"default": "a", "choices": ["a", "b"]},
        "graphs": {"default": True, "effect": "speed"},
        "plain": 3,
    }
    card = parse(d)
    assert dict(card.options) == {"level": 9, "steps": 4, "cap": 0.5, "mode": "a", "graphs": True, "plain": 3}
    assert dict(card.option_min) == {"level": 0, "steps": 1}
    assert dict(card.option_max) == {"level": 9, "cap": 1.0}
    assert dict(card.option_choices) == {"mode": ("a", "b")}
    assert dict(card.option_effect) == {"graphs": "speed"}  # "output" (the default) is not stored
    words = {k: card.option_bounds(k) for k in card.options}
    assert words == {
        "level": "0 to 9",
        "steps": ">= 1",
        "cap": "<= 1.0",
        "mode": "one of 'a', 'b'",
        "graphs": "",
        "plain": "",
    }
    assert "level=9 (0 to 9)" in str(card)


def test_option_aliases_parse_and_must_be_unique(data):
    d = copy.deepcopy(data)
    d["options"] = {"a": {"default": 1, "aliases": ["old"]}, "b": 2}
    assert dict(parse(d).option_aliases) == {"old": "a"}
    for clash in ("b", "old"):
        d["options"]["b"] = {"default": 2, "aliases": [clash]}
        with pytest.raises(CardError, match="alias"):
            parse(d)
