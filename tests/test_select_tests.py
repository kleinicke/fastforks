"""tools/select_tests.py: which families and fast-tier files a diff selects (synthetic mini repo + the real tree)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

import fetch_goldens
import merge_check
import select_tests as st

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

CARD = "uses = {uses}\n\n[modes.kernels]\nkernels = {kernels}\n"
MINI = {
    "pyproject.toml": "",
    "docs/index.md": "# docs\n",
    "src/fastforks/__init__.py": "",
    "src/fastforks/_core/__init__.py": "from . import model, tasks\n",
    "src/fastforks/_core/model.py": (
        "from fastforks._core import util\n\n\n"
        "def kernel(name):\n    import fastforks.kernels.ka  # lazy: card-driven\n"
    ),
    "src/fastforks/_core/util.py": "",
    "src/fastforks/_core/table.json": '{"fa-x@1": 1, "fb-z@1": 2}\n',
    "src/fastforks/_core/tasks/__init__.py": "import pkgutil\n\nSPECS = list(pkgutil.iter_modules(__path__))\n",
    "src/fastforks/_core/tasks/stereo.py": "",
    "src/fastforks/shared/__init__.py": "",
    "src/fastforks/shared/vit.py": "",
    "src/fastforks/shared/lazyuser.py": "",
    "src/fastforks/kernels/__init__.py": "",
    "src/fastforks/kernels/ka/__init__.py": "",
    "src/fastforks/kernels/ka/ka.cu": "",
    "src/fastforks/kernels/kb/__init__.py": "",
    "src/fastforks/stereo/__init__.py": "from fastforks._core import model\n",
    "src/fastforks/stereo/models/__init__.py": "",
    "src/fastforks/stereo/models/fa/__init__.py": "from .model import Net\n",
    "src/fastforks/stereo/models/fa/model.py": "from ....shared import vit\n\nNet = vit\n",
    "src/fastforks/stereo/models/fa/card.toml": CARD.format(kernels='["ka"]', uses="[]"),
    "src/fastforks/stereo/models/fa/golden.toml": '[refs."fa-x@1"]\n[refs."fa-y@2"]\n',
    "src/fastforks/stereo/models/fb/__init__.py": "",
    "src/fastforks/stereo/models/fb/model.py": (
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from fastforks.shared import vit\n\n\n"
        "def run():\n    from fastforks.shared import lazyuser  # lazy in a family: counts\n"
    ),
    "src/fastforks/stereo/models/fb/card.toml": CARD.format(kernels="[]", uses="[]"),
    "src/fastforks/stereo/models/fb/golden.toml": '[refs."fb-z@1"]\n',
    "src/fastforks/depth/__init__.py": "import fastforks._core\n",
    "src/fastforks/depth/models/__init__.py": "",
    "src/fastforks/depth/models/fc/__init__.py": "",
    "src/fastforks/depth/models/fc/card.toml": CARD.format(kernels="[]", uses='["vit"]'),
    "tests/conftest.py": "",
    "tests/data/x.json": "{}",
    "tests/test_docs.py": "",
    "tests/test_golden.py": "import fastforks._core\nimport fastforks.kernels.kb\nimport fastforks.stereo\n",
    "tests/test_plain.py": "import fastforks._core.util\n",
    "tests/test_fa.py": "from fastforks.stereo.models import fa\n",
    "tools/select_tests.py": "",
}
ALL = {"fa", "fb", "fc"}


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mini")
    for rel, text in MINI.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.mark.parametrize(
    ("changed", "want"),
    [
        ("src/fastforks/stereo/models/fa/model.py", {"fa"}),
        ("src/fastforks/stereo/models/fa/card.toml", {"fa"}),  # not a module: its package
        ("src/fastforks/stereo/models/fa/gone.py", {"fa"}),  # deleted: nearest existing package
        ("src/fastforks/kernels/ka/ka.cu", {"fa"}),  # card kernel lib; _core's lazy import does not count
        ("src/fastforks/kernels/kb/__init__.py", set()),  # imported by the harness only
        ("src/fastforks/shared/vit.py", {"fa", "fc"}),  # relative import + card `uses`; fb's is TYPE_CHECKING only
        ("src/fastforks/shared/lazyuser.py", {"fb"}),  # imported inside a family function
        ("src/fastforks/_core/util.py", ALL),
        ("src/fastforks/_core/tasks/stereo.py", ALL),  # pkgutil-loaded submodule
        ("tests/test_golden.py", ALL),
        ("pyproject.toml", ALL),
        ("tools/select_tests.py", set()),  # a gate tool: fast tier only
        ("docs/index.md", set()),
    ],
)
def test_mini_families(mini: Path, changed: str, want: set[str]) -> None:
    assert set(st.select([changed], mini).families) == want


def test_mini_fast_tier(mini: Path) -> None:
    assert st.select([], mini).as_dict() == {"changed": [], "families": {}, "fast": [], "release": False}
    fa = st.select(["src/fastforks/stereo/models/fa/model.py"], mini)
    assert "tests/test_fa.py" in fa.fast
    assert "tests/test_plain.py" in fa.fast  # reaches no family: always in the fast tier
    docs = st.select(["docs/index.md"], mini)
    assert docs.fast == ["tests/test_docs.py"]
    assert st.select(["tests/data/x.json"], mini).fast == ["tests"]
    assert st.select(["tests/conftest.py"], mini).fast == ["tests"]
    release = st.select(["uv.lock"], mini)
    assert release.release
    assert release.fast == ["tests"]


def test_lockfile_local_version_only_does_not_select_all_families(mini: Path) -> None:
    old = '[[package]]\nname="fastforks-core"\nversion="0.1.1"\nsource={editable="."}\n'
    old += '[[package]]\nname="torch"\nversion="2.8"\nsource={registry="https://example.com"}\n'
    new = old.replace('version="0.1.1"', 'version="0.2.0"')
    assert st.same_lock_dependencies(old, new)
    assert not st.same_lock_dependencies(old, new.replace('version="2.8"', 'version="2.9"'))
    assert not st.same_lock_dependencies(old, "not valid toml")
    sel = st.select(["uv.lock"], mini, metadata_only=frozenset({"uv.lock"}))
    assert not sel.release
    assert not sel.families
    assert sel.fast != ["tests"]


def test_mini_ref_table(mini: Path) -> None:
    table = "src/fastforks/_core/table.json"
    assert set(st.select([table], mini).families) == ALL  # no diff text: the owner rule (_core)
    diff = '--- a/x\n+++ b/x\n@@ -1 +1 @@\n-{"fa-x@1": 1, "fb-z@1": 2}\n+{"fa-x@1": 1, "fb-z@1": 3, "fb-q": 1}\n'
    sel = st.select([table], mini, diff=lambda f: diff if f == table else "")
    assert set(sel.families) == ALL - {"fc"}  # whole changed line: fa-x and fb-z occur in it
    one = '@@ -1,2 +1,2 @@\n-  "fb-z",\n+  "fb-z@1",\n'
    assert set(st.select([table], mini, diff=lambda _: one).families) == {"fb"}


def test_mini_k_expression(mini: Path) -> None:
    sel = st.select(["src/fastforks/stereo/models/fa/model.py"], mini)
    assert sel.families["fa"]["refs"] == ["fa-x@1", "fa-y@2"]
    assert sel.k("fa") == "test_gpu and ([fa-x or [fa-y)"
    assert sel.as_dict()["families"]["fa"]["k"] == sel.k("fa")


def test_golden_nodeids_keep_revisions_and_similar_names_separate() -> None:
    selection = st.Selection(
        [],
        families={
            "lineart": {"task": "edges", "refs": ["lineart@1"]},
            "classification": {"task": "classification", "refs": ["example@1", "example@2"]},
        },
    )
    cpu = {nodeid.split("::")[1] for nodeid in selection.golden_nodeids("cpu")}
    gpu = {nodeid.split("::")[1] for nodeid in selection.golden_nodeids("gpu")}
    refs = ("lineart@1", "example@1", "example@2")
    assert cpu == {f"{test}[{ref}]" for test in ("test_cpu_portable", "test_kernels_validated") for ref in refs}
    assert gpu == {f"test_gpu[{ref}-{mode}]" for ref in refs for mode in ("portable", "kernels")} | {
        "test_gpu_module[example@1]",
        "test_gpu_module[example@2]",
    }
    assert st.Selection([]).golden_nodeids("cpu") == []


@pytest.fixture
def git_repo(tmp_path: Path) -> Callable[..., None]:
    def git(*args: str) -> None:
        subprocess.run(
            [  # noqa: S607 - local test repository uses the installed Git client
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.org",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )

    git("init")
    return git


def test_git_version_bump_keeps_model_scope(
    mini: Path, tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(mini, tmp_path, dirs_exist_ok=True)
    project = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    old_project = '[project]\nname="fastforks-core"\nversion="0.2.0"\ndependencies=["torch>=2.8"]\n'
    old_lock = '[[package]]\nname="fastforks-core"\nversion="0.2.0"\nsource={editable="."}\n'
    project.write_text(old_project)
    lock.write_text(old_lock)
    git_repo("add", ".")
    git_repo("commit", "-m", "Release baseline")
    monkeypatch.setattr(merge_check, "ROOT", tmp_path)
    jobs = [{"family": "fa", "ref": "fa-x@1"}]

    def receipt() -> str:
        return merge_check._golden_sources_by_ref(jobs, "HEAD")["fa-x@1"][0]

    original_receipt = receipt()
    project.write_text(old_project.replace("0.2.0", "0.2.1"))
    lock.write_text(old_lock.replace("0.2.0", "0.2.1"))
    assert receipt() == original_receipt
    model = "src/fastforks/stereo/models/fa/model.py"
    (tmp_path / model).write_text("VALUE = 1\n")
    git_repo("add", ".")
    git_repo("commit", "-m", "Bump version and update one model")

    paths = st.git_changed("HEAD~1", "HEAD", tmp_path)
    metadata = st.git_metadata_only("HEAD~1", "HEAD", paths, tmp_path)
    assert metadata == {"pyproject.toml", "uv.lock"}
    selected = st.select(paths, tmp_path, metadata_only=metadata)
    assert not selected.release
    assert set(selected.families) == {"fa"}

    model_receipt = receipt()
    assert model_receipt != original_receipt
    project.write_text(project.read_text().replace("torch>=2.8", "torch>=2.9"))
    git_repo("add", "pyproject.toml")
    git_repo("commit", "-m", "Change the Torch requirement")
    paths = st.git_changed("HEAD~1", "HEAD", tmp_path)
    assert not st.git_metadata_only("HEAD~1", "HEAD", paths, tmp_path)
    assert st.select(paths, tmp_path).release
    assert receipt() != model_receipt


def test_git_changed_reports_both_sides_of_a_moved_module(
    tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "src/fastforks/stereo/models/fa/helper with spaces.py"
    new = "src/fastforks/stereo/models/fb/helper with spaces.py"
    git_repo("config", "diff.renames", "true")
    for name in (old, new):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / old).write_text("VALUE = 1\n")
    git_repo("add", old)
    git_repo("commit", "-m", "Initial module")
    git_repo("mv", old, new)
    git_repo("commit", "-m", "Move module")

    assert set(st.git_changed("HEAD~1", "HEAD", tmp_path)) == {old, new}
    monkeypatch.setattr(fetch_goldens, "ROOT", tmp_path)
    assert fetch_goldens.affected_families("HEAD~1") == {"fa", "fb"}


def test_git_selects_only_the_changed_golden_reference(
    mini: Path, tmp_path: Path, git_repo: Callable[..., None]
) -> None:
    shutil.copytree(mini, tmp_path, dirs_exist_ok=True)
    manifest = "src/fastforks/stereo/models/fa/golden.toml"
    file = tmp_path / manifest
    fixture = '[tolerance]\ncpu_atol = 0.01\n[refs."fa-x@1"]\n[refs."fa-y@2"]\nsha256 = "old"\n'
    file.write_text(fixture)
    git_repo("add", ".")
    git_repo("commit", "-m", "Baseline")
    file.write_text(fixture.replace('"old"', '"new"'))
    git_repo("commit", "-am", "Replace one reference")

    paths = st.git_changed("HEAD~1", "HEAD", tmp_path)
    scopes = st.git_fixture_refs("HEAD~1", "HEAD", paths, tmp_path)
    assert scopes == {manifest: ("fa-y@2",)}
    selected = st.select(paths, tmp_path, fixture_refs=scopes)
    assert selected.families["fa"]["refs"] == ["fa-y@2"]
    assert selected.golden_nodeids("gpu") == [
        "tests/test_golden.py::test_gpu[fa-y@2-kernels]",
        "tests/test_golden.py::test_gpu[fa-y@2-portable]",
    ]
    assert st.select(paths, tmp_path).families["fa"]["refs"] == ["fa-x@1", "fa-y@2"]
    source = "src/fastforks/stereo/models/fa/model.py"
    assert st.select([*paths, source], tmp_path, fixture_refs=scopes).families["fa"]["refs"] == ["fa-x@1", "fa-y@2"]

    file.write_text(file.read_text().replace("0.01", "0.02"))
    git_repo("commit", "-am", "Change the shared tolerance")
    assert st.git_fixture_refs("HEAD~1", "HEAD", paths, tmp_path) == {}


def test_fetch_goldens_scopes_single_file_shared_changes(
    tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, contents in MINI.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    git_repo("add", ".")
    git_repo("commit", "-m", "Shared helper baseline")
    (tmp_path / "src/fastforks/shared/vit.py").write_text("FEATURE = 1\n")
    git_repo("add", ".")
    git_repo("commit", "-m", "Update shared helper")

    monkeypatch.setattr(fetch_goldens, "ROOT", tmp_path)
    assert fetch_goldens.affected_families("HEAD~1") == {"fa", "fc"}


def test_golden_source_survives_generated_docs_but_changes_with_fixtures(
    tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = tmp_path / "docs/data/speed.jsonl"
    golden = tmp_path / "src/fastforks/stereo/models/fa/golden.toml"
    for path in (doc, golden):
        path.parent.mkdir(parents=True)
        path.write_text("")
    git_repo("add", ".")
    git_repo("commit", "-m", "Baseline")
    monkeypatch.setattr(merge_check, "ROOT", tmp_path)
    first, legacy = merge_check._golden_sources("baseline")

    doc.write_text("{}\n")
    assert merge_check._golden_sources("baseline")[0] == first
    git_repo("add", ".")
    git_repo("commit", "-m", "Refresh timings")
    source, old_format = merge_check._golden_sources("docs")
    assert source == first
    assert old_format != legacy

    golden.write_text('[refs."fa-x@1"]\n')
    assert merge_check._golden_sources("docs")[0] != first


def test_golden_source_tracks_untracked_source(
    tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    git_repo("add", ".")
    git_repo("commit", "-m", "Baseline")
    monkeypatch.setattr(merge_check, "ROOT", tmp_path)
    initial = merge_check._golden_sources("baseline")
    (tmp_path / "selection.json").write_text("{}\n")
    assert merge_check._golden_sources("baseline") == initial

    helper = tmp_path / "src" / "helper.py"
    helper.parent.mkdir()
    helper.write_text("VALUE = 1\n")
    first = merge_check._golden_sources("baseline")
    assert len(first) == 1
    assert first[0] not in initial
    helper.write_text("VALUE = 2\n")
    assert merge_check._golden_sources("baseline") != first
    helper.unlink()
    assert merge_check._golden_sources("baseline") == initial


def test_golden_resume_reuses_a_pass_after_an_unrelated_model_change(
    tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, contents in MINI.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    dependent = tmp_path / "src/fastforks/depth/models/fc/__init__.py"
    dependent.write_text("from fastforks.stereo.models.fa import model\n")
    git_repo("add", ".")
    git_repo("commit", "-m", "Model families")
    monkeypatch.setattr(merge_check, "ROOT", tmp_path)
    jobs = [{"family": f, "ref": f"{f}-test@1"} for f in sorted(ALL)]

    def sources() -> dict[str, str]:
        return {ref: candidates[0] for ref, candidates in merge_check._golden_sources_by_ref(jobs, "HEAD").items()}

    before = sources()
    (tmp_path / "docs/index.md").write_text("# Revised guide\n")
    assert sources() == before
    fa = tmp_path / "src/fastforks/stereo/models/fa/model.py"
    fa.write_text(fa.read_text() + "\nVALUE = 1\n")
    after = sources()
    assert after["fa-test@1"] != before["fa-test@1"]
    assert after["fc-test@1"] != before["fc-test@1"]
    assert after["fb-test@1"] == before["fb-test@1"]

    sibling = fa.with_name("untracked.py")
    sibling.write_text("VALUE = 2\n")
    untracked = sources()
    assert untracked["fa-test@1"] != after["fa-test@1"]
    assert untracked["fc-test@1"] != after["fc-test@1"]
    assert untracked["fb-test@1"] == after["fb-test@1"]

    (tmp_path / "tests/data/x.json").write_text('{"changed":true}\n')
    assert all(sources()[ref] != untracked[ref] for ref in untracked)


def test_golden_receipt_tracks_only_its_fixture_record(
    mini: Path, tmp_path: Path, git_repo: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(mini, tmp_path, dirs_exist_ok=True)
    manifest = tmp_path / "src/fastforks/stereo/models/fa/golden.toml"
    fixture = '[tolerance]\ngpu_mean = 0.01\n[refs."fa-x@1"]\nsha256 = "old"\n[refs."fa-y@2"]\nsha256 = "same"\n'
    manifest.write_text(fixture)
    git_repo("add", ".")
    git_repo("commit", "-m", "Initial golden fixtures")
    monkeypatch.setattr(merge_check, "ROOT", tmp_path)
    jobs = [{"family": "fa", "ref": ref} for ref in ("fa-x@1", "fa-y@2")]

    def receipts() -> dict[str, str]:
        return {ref: candidates[0] for ref, candidates in merge_check._golden_sources_by_ref(jobs, "HEAD").items()}

    baseline = receipts()
    manifest.write_text(fixture.replace('"old"', '"replacement"'))
    changed = receipts()
    assert changed["fa-x@1"] != baseline["fa-x@1"]
    assert changed["fa-y@2"] == baseline["fa-y@2"]

    manifest.write_text(manifest.read_text().replace("0.01", "0.02"))
    assert all(receipt != changed[ref] for ref, receipt in receipts().items())


def test_golden_only_includes_each_selected_revision_without_a_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    def require_no_floor() -> None:
        pytest.fail("golden-only job selection must not load the speed floor")

    monkeypatch.setattr(merge_check, "speed_check", require_no_floor)
    families = {"fa": {"task": "stereo", "refs": ["fa-x@1", "fa-x@2"]}}
    jobs = merge_check.jobs({"families": families}, ["fa"], ["portable", "kernels"], golden_only=True)
    assert [(job["ref"], job["mode"], job["inputs"]) for job in jobs] == [
        ("fa-x@1", "portable", []),
        ("fa-x@1", "kernels", []),
        ("fa-x@2", "portable", []),
        ("fa-x@2", "kernels", []),
    ]


@pytest.fixture(scope="module")
def real_graph() -> tuple[st.Graph, dict[str, set[str]]]:
    graph = st.Graph(st.ROOT)
    return graph, st.family_closures(graph, st.families(st.ROOT))


def _real(graph_closures: tuple[st.Graph, dict[str, set[str]]], path: str) -> set[str]:
    graph, closures = graph_closures
    owner = graph.owner(path)
    return {name for name, cl in closures.items() if owner in cl}


@pytest.mark.slow
def test_real_tree(real_graph: tuple[st.Graph, dict[str, set[str]]]) -> None:
    assert _real(real_graph, "src/fastforks/stereo/models/hitnet/model.py") == {"hitnet"}
    assert _real(real_graph, "src/fastforks/kernels/hitnet/hitnet_kernels.cu") == {"hitnet"}
    dav2 = _real(real_graph, "src/fastforks/shared/depth_anything_v2/fast_vit.py")
    assert {"depth_anything_v2", "foundationstereo"} <= dav2
    assert "hitnet" not in dav2
    assert len(_real(real_graph, "src/fastforks/_core/model.py")) == len(real_graph[1])


@pytest.mark.slow
def test_real_cli_docs_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert st.main(["--files", "docs/index.md", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["families"] == {}
    assert out["fast"] == ["tests/test_docs.py"]
    assert not out["release"]


def test_documentation_only_checks() -> None:
    bench = st.select(["docs/data/speed.jsonl", "docs/models/restoration.md", "docs/benchmarks.md"])
    assert bench.fast == ["tests/test_docs.py", "tests/test_site_data.py", "tests/test_speed_floor.py"]
    assert not bench.families
    assert not bench.release
    geometry = st.select(["README.md", "docs/stereo_geometry.md"])
    assert geometry.fast == ["tests/test_docs.py", "tests/test_geometry_example.py"]
