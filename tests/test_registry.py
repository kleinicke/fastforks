from __future__ import annotations

import subprocess
import sys

import pytest

import fastforks
from fastforks import LicenceError, ModelNotFoundError, OptionsError, WeightsDownloadError, stereo
from fastforks._core import registry


def test_list_and_filters():
    ids = stereo.list_models()
    assert "s2m2-s" in ids
    assert ids == sorted(ids)
    assert "s2m2-s" not in stereo.list_models(commercial="yes")
    assert "s2m2-s" in stereo.list_models(commercial="no", tier="stable")


def test_resolve():
    card, variant = registry.resolve("stereo", "s2m2-xl@1")
    assert card.family == "s2m2"
    assert variant.params == {"feature_channels": 384, "num_transformer": 3}
    assert stereo.model_card("s2m2-s").name == "S2M2"


@pytest.mark.parametrize(
    ("model", "match"),
    [
        ("nope", "unknown stereo model 'nope'"),
        ("s2m2-s@9", r"s2m2-s: no revision '9' \(have 1\)"),
        ("s2m2s", r"did you mean s2m2-s.*\(\d+ installed models; fastforks.stereo.list_models\('\*s2m2s\*'\)\)"),
    ],
)
def test_resolve_errors(model, match):
    with pytest.raises(ModelNotFoundError, match=match):
        registry.resolve("stereo", model)


@pytest.mark.slow
def test_listing_imports_no_model_code():
    code = (
        "import sys; from fastforks import stereo; stereo.list_models(); stereo.model_card('s2m2-s');"
        "bad = [m for m in sys.modules if '.models.' in m or m.startswith('fastforks.kernels.')]; print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "[]"


def test_weights_missing_without_download(tmp_path, monkeypatch):
    card, variant = registry.resolve("stereo", "s2m2-s")
    monkeypatch.setenv(registry.WEIGHTS_PATH_ENV, str(tmp_path))
    with pytest.raises(WeightsDownloadError, match=r"CH128NTR1\.pth not found .* downloads are disabled"):
        registry.weights(card, variant, allow_download=False)


def test_cards_are_read_only():
    with pytest.raises(TypeError):
        registry.cards("stereo")["x"] = None  # the point of the test


def test_wrong_task_names_the_right_one():
    with pytest.raises(ModelNotFoundError, match=r"'depth-anything-v2-s' is a depth model, not a stereo model: "):
        stereo.load("depth-anything-v2-s")
    assert registry.task_of("depth-anything-v2-s@1") == "depth"
    assert registry.resolve_any("s2m2-s@1")[0].task == "stereo"


def test_unknown_in_every_task():
    with pytest.raises(ModelNotFoundError, match=r"unknown model 's2m2s' \(no task has it; .*did you mean s2m2-s"):
        registry.resolve_any("s2m2s")


def test_resolve_error_does_not_list_all_ids():
    with pytest.raises(ModelNotFoundError) as e:
        registry.resolve("stereo", "nope")
    assert len(str(e.value)) < 200


def test_list_models_filters():
    ids = stereo.list_models()
    assert ids == sorted(ids)
    assert stereo.list_models("s2m2*") == [i for i in ids if i.startswith("s2m2")]
    assert stereo.list_models("no-such-*") == []
    assert set(stereo.list_models(family="s2m2")) == set(stereo.list_models("s2m2*"))
    assert stereo.list_models("s2m2-s", commercial="yes") == []
    assert stereo.list_models(["S2M2-*", "hitnet*"], exclude_filters="*-l") == sorted(
        i for i in ids if (i.startswith(("s2m2-", "hitnet"))) and not i.endswith("-l")
    )


def test_top_level_list_models_spans_tasks():
    every = fastforks.list_models()
    assert every == sorted(every)
    assert set(every) == {i for task in registry.TASKS for i in registry.cards(task)}
    assert fastforks.list_models("*anything-v2*", task="depth", exclude_filters=["*-b", "*-l"]) == [
        "depth-anything-v2-s"
    ]
    assert fastforks.list_models(task="stereo") == stereo.list_models()
    assert fastforks.list_models("") == fastforks.list_models([]) == every  # empty filter = all, as in timm
    with pytest.raises(OptionsError, match="unknown task"):
        fastforks.list_models(task="nope")


@pytest.mark.parametrize("kw", [{"commercial": "maybe"}, {"tier": "beta"}])
def test_list_models_rejects_bad_literals(kw):
    with pytest.raises(OptionsError, match="must be one of"):
        stereo.list_models(**kw)


def test_status_reasons():
    for model_id, (card, v) in registry.cards("stereo").items():
        st = card.status(v)
        assert bool(st.reasons) == (st.commercial != "yes"), model_id
    st = registry.cards("stereo")["s2m2-s"][0].status(registry.cards("stereo")["s2m2-s"][1])
    assert any(r.startswith(("code:", "weights:", "data:")) for r in st.reasons)


def test_model_of_an_uninstalled_distribution_names_the_package(monkeypatch):
    registry._known_models.cache_clear()
    monkeypatch.setattr(registry, "_known_models", lambda _task: {"zz-model": "fastforks-nc"})
    hint = r'package fastforks-nc, .*pip install "fastforks-core\[nc\]" \(or fastforks;'
    with pytest.raises(ModelNotFoundError, match=hint):
        registry.resolve("stereo", "zz-model")


def test_resolve_is_case_insensitive():
    assert registry.resolve("stereo", "S2M2-S")[1].ref == registry.resolve("stereo", "s2m2-s")[1].ref


def test_top_level_load_dispatches_to_the_task(monkeypatch):
    import fastforks  # noqa: PLC0415
    from fastforks import stereo  # noqa: PLC0415

    seen = {}
    monkeypatch.setattr(stereo, "load", lambda model, **kw: seen.update(model=model, **kw) or "loaded")
    assert fastforks.load("s2m2-s", mode="portable", device="cpu") == "loaded"
    assert seen["model"] == "s2m2-s"
    assert seen["mode"] == "portable"
    assert fastforks.stereo is stereo  # task packages are reachable from the top level


def test_top_level_load_unknown_model_suggests():
    import fastforks  # noqa: PLC0415

    with pytest.raises(fastforks.ModelNotFoundError, match="s2m2"):
        fastforks.load("s2m2-ss")


def test_top_level_load_ambiguous_id_names_the_loaders(monkeypatch):
    import fastforks  # noqa: PLC0415

    monkeypatch.setattr(fastforks, "_has", lambda task, _model: task in {"stereo", "depth"})
    with pytest.raises(fastforks.OptionsError, match=r"fastforks\.stereo\.load.*fastforks\.depth\.load"):
        fastforks.load("anything")


def _plain_install(monkeypatch):
    """No opt-in: what every install sees, `fastforks` (with the nc code) and `fastforks-core` alike."""
    monkeypatch.delenv(registry.NONCOMMERCIAL_ENV, raising=False)


def test_nc_weights_are_not_downloaded_by_the_plain_install(monkeypatch, tmp_path):
    card, v = registry.cards("stereo")["bridgedepth-sf"]  # permissive code, non-commercial weights
    assert card.licence.code_class == "permissive"
    assert card.status(v).commercial == "no"
    _plain_install(monkeypatch)
    calls = []

    def absent(spec, *, allow_download, **_k):
        calls.append(allow_download)
        raise WeightsDownloadError(spec.name)

    monkeypatch.setattr(registry, "locate", absent)
    monkeypatch.setattr(registry, "present", lambda *_: False)  # no local copies (counted as runnable: test_review_r2b)
    with pytest.raises(LicenceError, match=r"FASTFORKS_NONCOMMERCIAL=1, or pass weights=PATH"):
        stereo.load("bridgedepth-sf", mode="portable", device="cpu")
    assert set(calls) == {False}  # looked for local copies only
    assert "bridgedepth-sf" not in stereo.list_models(runnable=True)
    assert "igev-stereo-sceneflow" in stereo.list_models(runnable=True)

    monkeypatch.setattr(registry, "locate", lambda spec, **_k: tmp_path / spec.name)
    assert registry.weights(card, v)  # a local copy (cache, FASTFORKS_WEIGHTS_PATH) is used

    monkeypatch.setattr(registry, "locate", absent)
    monkeypatch.setenv(registry.NONCOMMERCIAL_ENV, "1")
    calls.clear()
    with pytest.raises(WeightsDownloadError) as e:
        registry.weights(card, v)
    assert not isinstance(e.value, LicenceError)
    assert set(calls) == {True}


def test_download_skips_or_refuses_nc_weights(monkeypatch, capsys):
    from fastforks._core import cli  # noqa: PLC0415

    _plain_install(monkeypatch)
    fetched = []
    monkeypatch.setattr(registry, "weights", lambda _card, v, **_k: fetched.append(v.ref) or [])
    assert cli.main(["download", "bridgedepth-sf"]) != 0
    assert "FASTFORKS_NONCOMMERCIAL=1" in capsys.readouterr().err
    assert fetched == []
    assert cli.main(["download", "--dry-run"]) == 0  # fetches nothing: lists the NC models with what they need
    assert "bridgedepth-sf@1: needs FASTFORKS_NONCOMMERCIAL=1 to download" in capsys.readouterr().out
    assert fetched == []


def test_download_fetches_the_rest_past_a_failing_model(monkeypatch, capsys, tmp_path):
    """A gated file without access is reported at the end; the other models and the kernel binaries still come."""
    from fastforks._core import cli  # noqa: PLC0415

    fetched, kernels = [], []

    def weights(_card, v, **_k):
        if v.ref.startswith("hitnet-eth3d"):
            msg = "gated: no access"
            raise WeightsDownloadError(msg)
        fetched.append(v.ref)
        return [tmp_path / "w"]

    monkeypatch.setattr(registry, "weights", weights)
    monkeypatch.setattr(cli, "locate", lambda spec, **_k: kernels.append(spec.name) or tmp_path / "k")
    assert cli.main(["download", "hitnet-eth3d", "igev-stereo-sceneflow", "--kernels", "--all-archs"]) == 1
    assert fetched == ["igev-stereo-sceneflow@1"]
    assert kernels  # fetched after the failure too
    err = capsys.readouterr().err
    assert "MISSING hitnet-eth3d@1: gated: no access" in err
    assert "igev-stereo" not in err


def test_model_card_has_the_variants_outputs():
    """A variant's `[output]` overrides show on its card (the DA3 family card says metric, da3mono is affine)."""
    from fastforks import depth  # noqa: PLC0415

    mono, metric = depth.model_card("da3mono-large"), depth.model_card("da3metric-large@1")
    assert mono.output_info["kind"] == "affine_invariant"
    assert metric.output_info["kind"] == "metric"
    assert mono.family == metric.family
    assert depth.model_card("da3-large-1.1").output_info["kind"] == "scale_invariant"
    with pytest.raises(TypeError):
        mono.output_info["kind"] = "metric"


@pytest.mark.parametrize("model", [123, None, b"s2m2-s"])
def test_non_string_ids_are_not_found(model):
    with pytest.raises(ModelNotFoundError, match="must be an id string"):
        fastforks.load(model)
    with pytest.raises(ModelNotFoundError):
        registry.resolve("stereo", model)
    assert registry.task_of(model) is None


def test_revisions():
    assert registry.resolve("stereo", "s2m2-s@01") == registry.resolve("stereo", "s2m2-s")
    for bad in ("s2m2-s@", "s2m2-s@x", "s2m2-s@2", "s2m2-s@\u00b9"):
        with pytest.raises(ModelNotFoundError, match="no revision"):
            registry.resolve("stereo", bad)


def test_not_found_messages():
    with pytest.raises(ModelNotFoundError) as e:
        registry.resolve("stereo", "s2m2_xyz")
    assert "; (" not in str(e.value)
    with pytest.raises(ModelNotFoundError, match=r"'s2m2' is a model family, not an id; its models: .*s2m2-s"):
        registry.resolve("stereo", "s2m2")
    with pytest.raises(ModelNotFoundError, match=r"model family.*igev"):
        fastforks.load("igev_stereo")


def test_top_level_model_card():
    assert fastforks.model_card("s2m2-s") == stereo.model_card("s2m2-s")
    assert fastforks.model_card("s2m2-s").variant_ref == "s2m2-s@1"


def test_features_export_at_a_fixed_size():
    from fastforks._core.export import fixed_size_reason  # noqa: PLC0415
    from fastforks._core.sizes import size_range  # noqa: PLC0415

    for card, v in registry.cards("features").values():
        assert fixed_size_reason(card, size_range(v, card)), v.id
