"""tools/select_tests.py: which families and fast-tier files a diff selects (synthetic mini repo + the real tree)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

import merge_check
import select_tests as st

if TYPE_CHECKING:
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
    fa = st.select(["src/fastforks/stereo/models/fa/model.py"], mini)
    assert "tests/test_fa.py" in fa.fast
    assert "tests/test_plain.py" in fa.fast  # reaches no family: always in the fast tier
    docs = st.select(["docs/index.md"], mini)
    assert "tests/test_fa.py" not in docs.fast
    assert "tests/test_plain.py" in docs.fast
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
    sel = st.select(["uv.lock"], mini, metadata_only=st.same_lock_dependencies(old, new))
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


def test_git_changed_reports_both_sides_of_a_moved_module(tmp_path: Path) -> None:
    old = "src/fastforks/stereo/models/fa/helper.py"
    new = "src/fastforks/stereo/models/fb/helper.py"

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
    git("config", "diff.renames", "true")
    for name in (old, new):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / old).write_text("VALUE = 1\n")
    git("add", old)
    git("commit", "-m", "Initial module")
    git("mv", old, new)
    git("commit", "-m", "Move module")

    assert set(st.git_changed("HEAD~1", "HEAD", tmp_path)) == {old, new}


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
    assert out["fast"]
    assert not out["release"]
