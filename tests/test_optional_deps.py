"""Optional libraries (timm, torchvision, transformers, scipy, onnx, tensorrt) fail with an install hint, never a crash.

The model code imports them only inside guarded helpers that raise `BackendUnavailableError`; the registry derives
each model's `requires` from its card, `load` checks it before any weight is read, and the model distributions
declare the same libraries as dependencies.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

import fastforks
from fastforks import classification, segmentation, stereo
from fastforks._core import registry
from fastforks._core.cards import LIBRARY_EXTRA
from fastforks._core.errors import BackendUnavailableError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fastforks"
OPTIONAL = {
    "timm",
    "torchvision",
    "transformers",
    "scipy",
    "onnx",
    "onnxscript",
    "onnx_ir",
    "tensorrt",
    "laspy",
    "lazrs",
}
GUARDED = {  # (file under src/fastforks, function) -> the guarded helper that may import an optional library
    ("shared/libnet/__init__.py", "timm"),
    ("shared/libnet/__init__.py", "torchvision"),
    ("shared/seginfer/__init__.py", "transformers"),
    ("detection/_ops.py", "ops"),
    ("_core/export.py", "_tensorrt"),
    ("_core/export.py", "_fix_folded_weight_casts"),
    ("_core/export.py", "_probe_dynamic_onnx"),  # after an export, which needs onnx
    ("stereo/geometry/ply.py", "write_las"),
}


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _string_import(node: ast.AST) -> list[str]:
    """Top-level package named by `importlib.import_module("x.y")` / `__import__("x.y")`."""
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
            return [node.args[0].value.split(".")[0]]
    return []


def _is_type_checking(test: ast.expr) -> bool:
    """Exactly `TYPE_CHECKING` / `typing.TYPE_CHECKING` (not `not TYPE_CHECKING`, not a compound test)."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _scan(tree: ast.AST, path: str, found: list[str], func: str = "", *, typing_only: bool = False) -> None:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            _scan(node, path, found, node.name, typing_only=typing_only)
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.body:  # only the `if TYPE_CHECKING:` body is typing-only; `else:` runs
                _scan(ast.Module([child], []), path, found, func, typing_only=True)
            for child in node.orelse:
                _scan(ast.Module([child], []), path, found, func, typing_only=typing_only)
            continue
        names = [*_imported(node), *_string_import(node)]
        if not typing_only and (bad := OPTIONAL.intersection(names)) and (path, func) not in GUARDED:
            found.append(f"{path}:{node.lineno} imports {', '.join(sorted(bad))} outside a guarded helper")
        _scan(node, path, found, func, typing_only=typing_only)


def _scan_source(source: str) -> list[str]:
    found: list[str] = []
    _scan(ast.parse(source), "x.py", found)
    return found


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("import timm", True),
        ("from torchvision.ops import nms", True),
        ("import importlib\nimportlib.import_module('transformers')", True),
        ("__import__('scipy')", True),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import timm", False),
        ("if not TYPE_CHECKING:\n    import timm", True),
        ("if TYPE_CHECKING:\n    pass\nelse:\n    import timm", True),
        ("def f():\n    import onnx", True),
        ("import numpy", False),
    ],
)
def test_the_ast_check_flags_unguarded_imports(source, flagged):
    assert bool(_scan_source(source)) is flagged


def test_optional_imports_only_in_guarded_helpers():
    found: list[str] = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        _scan(ast.parse(f.read_text(), rel), rel, found)
    assert not found, "\n".join(found)


def test_guarded_helpers_exist():
    """The allowlist names real functions that really import something optional (no stale entries)."""
    for rel, func in GUARDED:
        tree = ast.parse((SRC / rel).read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
        assert any(OPTIONAL.intersection(_imported(n)) for n in ast.walk(fn)), f"{rel}:{func} imports nothing optional"


def _family_cards():
    registry.cards.cache_clear()
    seen = {}
    for task in registry.TASKS:
        for card, variant in registry.cards(task).values():
            seen.setdefault((card.task, card.family), (card, variant))
    return seen


FAMILIES = _family_cards()
NEEDY = {k: v for k, v in FAMILIES.items() if v[0].requires}


def test_requires_known_and_documented():
    assert NEEDY, "no card needs an optional library?"
    for card, _ in NEEDY.values():
        assert set(card.requires) <= set(LIBRARY_EXTRA), card.family
        assert set(card.extras) <= set(
            tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
        )
    assert stereo.model_card("s2m2-s").requires == ()
    assert classification.model_card("deit_base_patch16_224.fb_in1k").requires == ("timm",)
    assert segmentation.model_card("mask2former-swinl-ade20k").extras == ("transformers",)


@pytest.mark.parametrize(("task", "family"), sorted(NEEDY), ids=str)
def test_load_without_the_library_fails_before_weights(task, family, monkeypatch):
    card, variant = NEEDY[task, family]

    class SentinelError(Exception):
        pass

    def no_weights(*_a, **_k):
        raise SentinelError

    monkeypatch.setattr(registry, "locate", no_weights)
    for lib in card.requires:
        monkeypatch.setitem(sys.modules, lib, None)
    with pytest.raises(BackendUnavailableError) as e:
        importlib.import_module(f"fastforks.{task}").load(variant.id, mode="portable", device="cpu")
    msg = str(e.value)
    assert all(f"fastforks-core[{x}]" in msg for x in card.extras), msg
    assert "pip install fastforks' for everything" in msg
    assert f"from a git checkout: pip install -e '.[{','.join(card.extras)}]'" in msg  # TM-4
    cause = e.value.__cause__  # an import error naming the library: conftest skips such tests instead of failing them
    assert isinstance(cause, ImportError), cause
    assert cause.name in card.requires


def test_load_without_a_needed_library_does_not_block_other_models(monkeypatch):
    """The check is per model: with timm blocked a stereo model still reaches its (mocked) weight fetch."""

    class SentinelError(Exception):
        pass

    def no_weights(*_a, **_k):
        raise SentinelError

    monkeypatch.setattr(registry, "locate", no_weights)
    monkeypatch.setitem(sys.modules, "timm", None)
    with pytest.raises(SentinelError):
        stereo.load("s2m2-s", mode="portable", device="cpu")


def test_list_models_installed_filter(monkeypatch):
    everything = fastforks.list_models()
    got = fastforks.list_models(runnable=True) + fastforks.list_models(runnable=False)
    assert sorted(got) == everything
    monkeypatch.setitem(sys.modules, "timm", None)
    monkeypatch.setitem(sys.modules, "torchvision", None)
    assert classification.list_models(runnable=True) == []
    assert classification.list_models(runnable=False) == classification.list_models()
    assert "s2m2-s" in fastforks.list_models(runnable=True)


# --- the derived requirement matches what the family code calls --------------------------------------------------


def _code_needs(source: str, package: list[str]) -> tuple[set[str], bool]:  # noqa: C901
    """`(optional libraries the code reaches, whether it uses libnet)`; `package`: dotted path of the file's package.

    Reads absolute and relative imports, `import x as y`, and string imports (`importlib.import_module("...")`).
    """
    needs: set[str] = set()
    libnet = False

    def seen(mod: str, names: set[str], *, module_form: bool) -> None:
        nonlocal libnet
        top = mod.split(".", maxsplit=1)[0]
        if top in OPTIONAL:
            needs.add(top)
        if mod == "fastforks.shared.libnet" or (mod == "fastforks.shared" and "libnet" in names):
            libnet = True
        if mod == "fastforks.shared.seginfer" and (module_form or names & {"pretrained", "transformers"}):
            needs.add("transformers")
        if mod == "fastforks.shared" and "seginfer" in names:
            needs.add("transformers")
        if (mod == "fastforks.detection._ops" and "ops" in names) or (mod == "fastforks.detection" and "_ops" in names):
            needs.add("torchvision")

    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import):
            for a in n.names:
                seen(a.name, set(), module_form=True)
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if n.level:
                base = package[: len(package) - (n.level - 1)]
                mod = ".".join([*base, *([n.module] if n.module else [])])
            seen(mod, {a.name for a in n.names}, module_form=False)
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and _string_import(n):
            seen(str(n.args[0].value), set(), module_form=True)
    return needs, libnet


def _family_code(card) -> tuple[set[str], bool]:
    d = SRC / card.task / "models" / card.family
    assert d.is_dir(), d
    needs: set[str] = set()
    libnet = False
    for f in sorted(d.rglob("*.py")):
        pkg = ["fastforks", *f.parent.relative_to(SRC).parts]
        n, lib = _code_needs(f.read_text(), pkg)
        needs |= n
        libnet |= lib
    return needs, libnet


@pytest.mark.parametrize(("task", "family"), sorted(FAMILIES), ids=str)
def test_requires_matches_family_code(task, family):
    card, _ = FAMILIES[task, family]
    code, libnet = _family_code(card)
    if libnet:
        libs = {(v.params.get("library") or card.params.get("library")) for v in card.variants}
        assert None not in libs, f"{task}/{family}: uses libnet but a variant has no params.library"
        code |= libs
    assert set(card.requires) - {"scipy"} == code, f"{task}/{family}: card requires {card.requires}, code calls {code}"
    assert ("scipy" in card.requires) <= ("transformers" in card.requires)


@pytest.mark.parametrize(
    ("source", "needs", "libnet"),
    [
        ("from fastforks.shared import libnet", set(), True),
        ("import fastforks.shared.libnet as ln", set(), True),
        ("from ....shared import libnet", set(), True),
        ("from fastforks.shared.seginfer import pretrained", {"transformers"}, False),
        ("from ....shared import seginfer", {"transformers"}, False),
        ("import importlib\nimportlib.import_module('fastforks.shared.seginfer')", {"transformers"}, False),
        ("from fastforks.detection._ops import ops", {"torchvision"}, False),
        ("from fastforks.shared.seginfer import resize", set(), False),
        ("import numpy", set(), False),
    ],
)
def test_the_drift_scanner_sees_every_import_form(source, needs, libnet):
    pkg = ["fastforks", "segmentation", "models", "newfam"]
    assert _code_needs(source, pkg) == (needs, libnet)


def test_supported_versions_match_the_core_extras():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    from fastforks._core.cards import LIBRARY_SPEC  # noqa: PLC0415

    for lib, spec in LIBRARY_SPEC.items():
        reqs = [r.replace(" ", "") for r in extras[LIBRARY_EXTRA[lib]] if r.startswith(lib)]
        assert reqs == [lib + spec], (lib, reqs, spec)


def _fake_versions(monkeypatch, **versions):
    import importlib.metadata as md  # noqa: PLC0415
    import importlib.util as iu  # noqa: PLC0415

    real, real_spec = md.version, iu.find_spec
    monkeypatch.setattr(md, "version", lambda n: versions.get(n) or real(n))
    # a faked library counts as installed, so the test also runs where it is absent
    monkeypatch.setattr(iu, "find_spec", lambda n, *a: object() if n in versions else real_spec(n, *a))


def test_unsupported_library_version_fails_before_weights(monkeypatch):
    class SentinelError(Exception):
        pass

    def no_weights(*_a, **_k):
        raise SentinelError

    monkeypatch.setattr(registry, "locate", no_weights)
    _fake_versions(monkeypatch, transformers="4.57.1")
    with pytest.raises(
        BackendUnavailableError,
        match=r"needs transformers>=5.19,<6, found 4\.57\.1.*pip install -U 'fastforks-core\[transformers\]'",
    ) as e:
        segmentation.load("segformer-b0-ade20k", mode="portable", device="cpu")
    assert e.value.__cause__ is None  # a wrong version is a real failure, not a skip
    assert "segformer-b0-ade20k" not in segmentation.list_models(runnable=True)
    _fake_versions(monkeypatch, transformers="5.19.0+cu1", timm="1.0.30")
    assert "segformer-b0-ade20k" in segmentation.list_models(runnable=True)
    _fake_versions(monkeypatch, timm="1.0.29")
    assert classification.list_models(runnable=True) != classification.list_models()


def test_download_notes_missing_libraries(monkeypatch, capsys):
    from fastforks._core import cli  # noqa: PLC0415

    monkeypatch.setattr(registry, "weights", lambda *_a, **_k: [])
    monkeypatch.setitem(sys.modules, "timm", None)
    assert cli.main(["download", "deit_base_patch16_224.fb_in1k", "s2m2-s"]) == 0
    err = capsys.readouterr().err
    assert "note: deit_base_patch16_224.fb_in1k@" in err
    assert "fastforks-core[timm]" in err
    assert "s2m2-s" not in err


# --- distributions ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build_dists():
    spec = importlib.util.spec_from_file_location("build_dists", ROOT / "tools" / "build_dists.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_model_dists_declare_their_libraries(build_dists, tmp_path):
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    version = "1.2.3"
    for name, spec in dists.items():
        out = tmp_path / name
        out.mkdir()
        build_dists.stage(name, spec, version, out, extras)
        deps = tomllib.loads((out / "pyproject.toml").read_text())["project"]["dependencies"]
        assert deps == [f"fastforks-core=={version}", *build_dists.library_requirements(spec, extras)]

    def needs(task):
        return set(build_dists.library_requirements({"code_classes": ["permissive"], "tasks": [task]}, extras))

    assert set(extras["timm"]) | set(extras["torchvision"]) <= needs("classification")
    assert set(extras["transformers"]) <= needs("segmentation")  # includes scipy
    assert set(extras["torchvision"]) <= needs("detection")
    assert needs("stereo") == needs("depth") == set()


def test_fastforks_metapackage_installs_everything_but_tensorrt(build_dists):
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    text = (ROOT / "pyproject.toml").read_text()
    core = tomllib.loads(build_dists.core_pyproject(text, dists, "1.2.3"))["project"]
    assert "all" not in core["optional-dependencies"]  # `pip install fastforks` is "everything"
    # libs: every library extra and ONNX export; TensorRT stays opt-in (CUDA-version specific)
    libs = [*sorted(set(LIBRARY_EXTRA.values())), "onnx"]
    assert core["optional-dependencies"]["libs"] == [f"fastforks-core[{','.join(libs)}]"]
    meta = tomllib.loads(build_dists.meta_pyproject(text, dists, "1.2.3"))
    picks = ",".join([*(n.removeprefix("fastforks-") for n in dists), "libs"])
    assert meta["project"]["name"] == "fastforks"
    assert meta["project"]["dependencies"] == [f"fastforks-core[{picks}]==1.2.3"]
    assert meta["project"]["optional-dependencies"] == {"tensorrt": ["fastforks-core[tensorrt]==1.2.3"]}
    assert meta["tool"]["hatch"]["build"]["targets"]["wheel"] == {"bypass-selection": True}


def test_export_names_the_missing_onnx_libraries(monkeypatch):
    from fastforks._core import export  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "onnx_ir", None)
    with pytest.raises(BackendUnavailableError, match=r"fastforks-core\[onnx\].*pip install fastforks"):
        export._fix_folded_weight_casts(None)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(BackendUnavailableError, match=r"fastforks-core\[tensorrt\].*pip install fastforks"):
        export._tensorrt()
