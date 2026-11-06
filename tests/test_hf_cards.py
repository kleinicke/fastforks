"""`tools/stage_hf.py cards`: one valid model card per family (CPU only, nothing is uploaded)."""

import dataclasses
import re

import pytest
import yaml

import stage_hf
from fastforks._core import licences, registry


def _families():
    return {card.family for task in registry.TASKS for card, _ in registry.cards(task).values()}


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    out = tmp_path_factory.mktemp("cards") / "out"
    return out, stage_hf.stage_cards(out)


def test_one_readme_per_family(staged):
    out, ids = staged
    assert len(ids) == len(_families())
    assert sorted(p.parent.name for p in out.glob("*/README.md")) == sorted(i.split("/", 1)[1] for i in ids)
    assert [p for p in out.rglob("*") if p.is_file() and p.name != "README.md"] == []  # no weights


def test_repo_names_valid_and_unique(staged):
    _, ids = staged
    names = [i.split("/", 1)[1] for i in ids]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", n) for n in names)
    assert not set(names) & stage_hf.RESERVED_REPOS


def test_reserved_name_is_refused():
    with pytest.raises(SystemExit):
        stage_hf.repo_name("Kernels")


def test_front_matter_parses(staged):
    out, _ = staged
    for readme in out.glob("*/README.md"):
        text = readme.read_text()
        assert text.startswith("---\n"), readme
        front = yaml.safe_load(text.split("---\n", 2)[1])
        assert front["library_name"] == "fastforks"
        assert front["pipeline_tag"] in stage_hf._PIPELINE.values()
        assert "cuda" in front["tags"]
        assert front["license"] != "other" or {"license_name", "license_link"} <= set(front)
        assert "github.com/kleinicke/fastforks" in text
        assert "huggingface.co/fastforks/kernels" in text
        assert "## Variants" in text


def test_no_internal_text_or_base_model(staged):
    out, _ = staged
    for readme in out.glob("*/README.md"):
        text = readme.read_text()
        for bad in ("exp 0", "results/", "/home/"):
            assert bad not in text, (readme, bad)
        assert "base_model" not in text, readme


def test_assumed_weights_licence_is_not_firm(staged):
    out, _ = staged
    text = (out / "psmnet" / "README.md").read_text()
    assert 'license: "other"' in text
    assert 'license_name: "mit-code-weights-unstated"' in text
    assert "the authors do not state a separate weights licence" in text


def test_code_without_licence_is_labelled(staged):
    card, v = registry.cards("stereo")["psmnet-sceneflow"]
    bare = dataclasses.replace(card, licence=dataclasses.replace(card.licence, code="LicenseRef-none"))
    assert stage_hf._licence_front(bare, [v])["license_name"] == "no-licence-stated"


def test_repo_name_collapses():
    assert stage_hf.repo_name("a__b") == "a-b"
    assert stage_hf.repo_name("-a_b-") == "a-b"


def _api_class(card, first):
    import importlib  # noqa: PLC0415

    api = importlib.import_module(f"fastforks.{card.task}._api")
    if card.task == "promptable_segmentation":
        return api._model_class(first.id)
    return next(
        c
        for c in vars(api).values()
        if isinstance(c, type)
        and c.__module__ == api.__name__
        and "__call__" in vars(c)
        and c.__name__.endswith("Model")
    )


def test_quickstarts_match_the_call_contract():
    """Each card's example call fits its task API (known, required parameters) and the family's own inputs."""
    import ast  # noqa: PLC0415
    import importlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415

    for card, variants in stage_hf._families().values():
        if True:
            task, first = card.task, variants[0]
            text = "\n".join(stage_hf.usage_example(card, first))
            calls = [
                n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "model"
            ]
            assert calls, card.family
            call = calls[0]
            given = {k.arg for k in call.keywords}
            params = inspect.signature(_api_class(card, first).__call__).parameters
            assert given <= set(params), (card.family, given - set(params))
            required = {
                n for n, p in params.items() if n != "self" and p.default is p.empty and p.kind is p.KEYWORD_ONLY
            }
            positional = [n for n, p in params.items() if n != "self" and p.kind is p.POSITIONAL_OR_KEYWORD]
            required |= {n for n in positional[len(call.args) :] if params[n].default is params[n].empty}
            assert not required - given or card.family == "sam3", (card.family, required - given)  # sam3: text or boxes
            if task == "multiview":
                runners = [
                    c
                    for c in vars(importlib.import_module(card.module)).values()
                    if isinstance(c, type) and "intrinsics" in vars(c)
                ]
                for name in ("intrinsics", "cam_to_world", "depth_range"):
                    uses = {getattr(c, name) for c in runners}
                    assert not (uses == {"required"} and name not in given), (card.family, name)
                    assert not (uses == {"unused"} and name in given), (card.family, name)
            if card.family == "sam3":
                assert given & {"text", "input_boxes"}
                assert not given & {"input_points", "input_labels"}
            nc = licences.weights_commercial(card.variant_licence(first).weights_licence) == "no"
            assert nc == (registry.NONCOMMERCIAL_ENV in text), card.family
