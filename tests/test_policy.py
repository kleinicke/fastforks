"""Repository rules that keep ~hundreds of models maintainable (see AGENTS.md)."""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
import tomllib
from pathlib import Path
from typing import get_args

import pytest

import build_dists
from fastforks._core import licences, registry
from fastforks._core.licences import CodeClass, licence_class

ROOT = Path(__file__).resolve().parents[1]

SRC = Path(__file__).parents[1] / "src" / "fastforks"
# Runtime dependency allowlist (pyproject `dependencies`); optional extras in one module each.
ALLOWED = {"torch", "numpy", "PIL", "platformdirs", "safetensors", "isal", "fastforks"}
OPTIONAL = {
    "huggingface_hub": {"_core/files.py"},  # gated weights download through it when installed
    "tensorrt": {"_core/export.py"},
    "onnx_ir": {"_core/export.py"},  # comes with onnxscript (the onnx extra)
    "onnx": {"_core/export.py"},  # the dynamic-export check reads the written file
    "onnxruntime": {"_core/export.py"},  # ... and runs it in range, when installed
    "timm": {"shared/libnet/__init__.py"},  # the timm / torchvision extras: families built by those libraries
    "torchvision": {  # R-CNN ops and lazily constructed optional library networks
        "detection/_ops.py",
        "shared/libnet/__init__.py",
        "shared/rfdetr/net/utilities/box_ops.py",
        "shared/rfdetr/net/utilities/tensors.py",
    },
    "transformers": {  # optional libraries imported only when constructing their networks
        "shared/seginfer/__init__.py",
        "shared/rfdetr/net/models/backbone/dinov2.py",
        "shared/rfdetr/net/models/backbone/dinov2_with_windowed_attn.py",
    },
    "laspy": {"stereo/geometry/ply.py"},  # .las / .laz point clouds (dependencies, imported in this module only)
    "lazrs": {"stereo/geometry/ply.py"},
    "pycolmap": {"matching/sfm.py"},  # `sfm.reconstruct` (the `sfm` extra)
}
# Files allowed to import computed module names: the registry imports `fastforks.<task>.models.<family>.model`, the
# task package its own `<task>` spec modules, the top level the module of each lazily re-exported name.
DYNAMIC = {"__init__.py", "_core/registry.py", "_core/tasks/__init__.py"}
# Known violations of three rules on main, open in the depth / features workstreams. Exempt per file (and per name),
# so the rules stay active for every other file of these tasks; remove an entry when its file is fixed.
# DPT conversions shared from the DAv2 `fused` module (should move to shared/):
CROSS_FAMILY_IMPORTS: dict[str, str | tuple[str, ...]] = {
    f"depth/models/{family}/{file}.py": "fastforks.depth.models.depth_anything_v2.fused"
    for family, file in (
        ("depth_anything_3", "model"),
        ("depth_pro", "model"),
        ("moge", "model"),
        ("unik3d", "model"),
        ("prompt_depth_anything", "fused"),
    )
}
# LingBot-Depth reuses MoGe's neck / heads and its kernels-mode helpers (upstream builds on MoGe's modules):
CROSS_FAMILY_IMPORTS |= {
    f"depth/models/lingbot_depth/{file}.py": (
        "fastforks.depth.models.moge",
        "fastforks.depth.models.depth_anything_v2.fused",
    )
    for file in ("model", "net")
}
TORCH_LOAD = {"depth/models/midas/model.py", "depth/models/zoedepth/model.py"}  # per-family checkpoint loaders
UNDOCUMENTED_CONSTANTS = {
    ("depth/models/depth_anything_3/model.py", "PATCH"),
    ("depth/models/depth_anything_3/model.py", "HEAD_DIM"),
    ("depth/models/depth_anything_3/model.py", "_FEATURES"),
    ("depth/models/depth_anything_v2/model.py", "PATCH"),
    ("depth/models/midas/model.py", "MULTIPLE"),
    ("depth/models/moge/model.py", "_REFINE_STEPS"),
    ("depth/models/moge/model.py", "_DEPTH_RES"),
    ("depth/models/unik3d/model.py", "LEVELS"),
    ("depth/models/unik3d/model.py", "SLOPE"),
    ("features/models/dinov3_convnext/model.py", "EPS"),
    ("features/models/dinov3/model.py", "_WIDE"),
}


def imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and _dynamic_import(node) and node.args:
            name = _literal_prefix(node.args[0])
            if name is None:
                roots.add(f"<non-literal import at line {node.lineno}>")
            elif not name.startswith("."):
                roots.add(name.split(".")[0])
    return roots


def _dynamic_import(call: ast.Call) -> bool:
    """`importlib.import_module(...)`, `import_module(...)` or `__import__(...)`."""
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    return name in {"import_module", "__import__"}


def _literal_prefix(arg: ast.expr) -> str | None:
    """The literal module name, or the literal start of an f-string (`f"fastforks.kernels.{n}"`), else `None`."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
        return str(arg.values[0].value)
    return None


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_import_allowlist(path):
    rel = str(path.relative_to(SRC))
    bad = {
        r
        for r in imported_roots(path) - ALLOWED - set(sys.stdlib_module_names)
        if rel not in OPTIONAL.get(r, set()) and not (r.startswith("<non-literal") and rel in DYNAMIC)
    }
    assert not bad, f"{rel} imports {bad}; only {sorted(ALLOWED)} (+ stdlib) are allowed"


@pytest.mark.parametrize("task", registry.TASKS)
def test_model_directories(task):
    families = {card.family: card for card, _ in registry.cards(task).values()}
    root = SRC / task / "models"
    assert set(families) == {p.parent.name for p in root.glob("*/card.toml")}
    for fam, card in families.items():
        d = root / fam
        for f in card.licence.files:
            assert (d / f).is_file(), f"{fam}: licence file {f} missing"
        assert (d / "__init__.py").is_file()
        assert "RUNNERS = " in (d / "model.py").read_text()
        for lib in card.kernel_libs + card.mps_kernel_libs:
            assert (SRC / "kernels" / lib / "__init__.py").is_file(), f"{fam}: unknown kernel library {lib}"


def test_every_model_is_in_exactly_one_wheel():
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    specs = [{"code_classes": build_dists.CORE_CLASSES}, *dists.values()]
    shipped = [d.relative_to(SRC).as_posix() for spec in specs for d in build_dists.select(spec)]
    every = {f"{c.task}/models/{c.family}" for t in registry.TASKS for c, _ in registry.cards(t).values()}
    assert sorted(shipped) == sorted(every)


def test_distributions_are_licence_classes():
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    for name, spec in dists.items():
        assert set(spec) <= {"summary", "code_classes", "tasks"}, name
        assert set(spec["code_classes"]) <= set(get_args(CodeClass)) - set(build_dists.CORE_CLASSES), name
        assert set(spec.get("tasks", ())) <= set(registry.TASKS), name


def test_core_wheel_licence_covers_its_model_families():
    """The generated core pyproject adds the code licences and licence files of the permissive families."""
    text = build_dists.core_pyproject((ROOT / "pyproject.toml").read_text(), {}, "1.2.3")
    project = tomllib.loads(text)["project"]
    core = [c for t in registry.TASKS for c, _ in registry.cards(t).values() if c.licence.code_class == "permissive"]
    wanted = _terms(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["license"])
    for card in core:
        wanted |= _terms(card.licence.code)
    assert _terms(project["license"]) == wanted
    for card in core:
        for f in card.licence.files:
            assert f"src/fastforks/{card.task}/models/{card.family}/{f}" in project["license-files"]


def imported_modules(path: Path, *, top_only: bool = False) -> set[str]:
    """Absolute names of the `fastforks` modules `path` imports (relative resolved; `from m import x` adds m, m.x).

    `top_only`: only the module-level import statements (what importing the module runs).
    """
    package = ["fastforks", *path.relative_to(SRC).parent.parts]
    package.append("")  # level 1 resolves against the containing package, for modules and `__init__.py` alike
    out = set()
    tree = ast.parse(path.read_text(), str(path))
    for node in tree.body if top_only else ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level] if node.level else []
            mod = ".".join([*[b for b in base if b], *([node.module] if node.module else [])])
            out |= {mod, *(f"{mod}.{a.name}" for a in node.names)}  # `from pkg import mod` imports pkg.mod
    return {m for m in out if m.split(".")[0] == "fastforks"}


SHARED = sorted(p.parent.name for p in (SRC / "shared").glob("*/meta.toml"))


def _family_files(task: str, family: str) -> list[Path]:
    return sorted((SRC / task / "models" / family).rglob("*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("task", registry.TASKS)
def test_families_do_not_import_each_other(task):
    models = f"fastforks.{task}.models."
    for card, _ in registry.cards(task).values():
        own = models + card.family
        for path in _family_files(task, card.family):
            allowed = CROSS_FAMILY_IMPORTS.get(str(path.relative_to(SRC)), own)
            prefixes = tuple(p + "." for p in ((allowed,) if isinstance(allowed, str) else allowed))
            bad = {
                m
                for m in imported_modules(path)
                if m.startswith(models) and not (m + ".").startswith((own + ".", *prefixes))
            }
            assert not bad, f"{path.relative_to(SRC)} imports another family: {bad} (move shared code to shared/)"


def _task_model_imports(modules: set[str]) -> set[str]:
    """Separate forbidden task-family dependencies from upstream shared net/models packages."""
    return {
        m
        for m in modules
        if any(m == f"fastforks.{t}.models" or m.startswith(f"fastforks.{t}.models.") for t in registry.TASKS)
    }


def test_task_model_boundary_includes_all_tasks_but_not_shared_vendor_models():
    forbidden = {f"fastforks.{task}.models.family.model" for task in registry.TASKS}
    forbidden |= {"fastforks.pose.models"}
    allowed = {"fastforks.shared.rfdetr.net.models.lwdetr", "fastforks.pose.models_extra", "torchvision.models"}
    assert _task_model_imports(forbidden | allowed) == forbidden


@pytest.mark.parametrize("area", ["_core", "kernels", "shared"])
def test_core_does_not_import_models(area):
    for path in sorted((SRC / area).rglob("*.py")):
        # Vendored shared architectures may retain an upstream package named "models".
        # The boundary forbids dependencies on task-specific model families.
        bad = _task_model_imports(imported_modules(path))
        assert not bad, f"{path.relative_to(SRC)} imports model code {bad}"


@pytest.mark.parametrize("module", SHARED)
def test_shared_modules_are_permissive(module):
    d = SRC / "shared" / module
    meta = tomllib.loads((d / "meta.toml").read_text())
    assert set(meta) <= {"summary", "licence", "files", "derived_from"}, f"{module}: unknown meta.toml keys"
    assert licence_class(meta["licence"], f"shared/{module}") == "permissive", "shared code ships in the core wheel"
    for f in meta["files"]:
        assert (d / f).is_file(), f"shared/{module}: licence file {f} missing"
    shared = {
        m.split(".")[2] for p in d.rglob("*.py") for m in imported_modules(p) if m.startswith("fastforks.shared.")
    }
    shared &= set(SHARED)  # single-file helpers (`shared/cuda_graphs.py`: no meta, no third-party code) are core code
    assert shared <= {module}, f"shared/{module} imports other shared modules {shared - {module}}: list them in meta"


def test_shared_directories_have_meta():
    dirs = {p.name for p in (SRC / "shared").iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert dirs == set(SHARED), f"shared/ directories without meta.toml: {dirs - set(SHARED)}"


@pytest.mark.slow
@pytest.mark.parametrize("task", registry.TASKS)
def test_card_uses_matches_imports(task):
    cards = {card.family: card for card, _ in registry.cards(task).values()}
    for card in cards.values():
        used = {
            m.split(".")[2]
            for p in _family_files(task, card.family)
            for m in imported_modules(p)
            if m.startswith("fastforks.shared.")
        }
        assert set(card.uses) == used, f"{card.family}: card uses {sorted(card.uses)}, code imports {sorted(used)}"


@pytest.mark.parametrize("task", registry.TASKS)
def test_weight_names_are_unique_per_family(task):
    """A local weights folder holds `<family>/<name>`: two different files of one family need two names
    (e.g. `DA3MONO-LARGE/model.safetensors`, not `model.safetensors` for every variant)."""
    for card, _ in registry.cards(task).values():
        files: dict[str, set[str]] = {}
        for v in card.variants:
            for w in v.weights:
                files.setdefault(w.name, set()).add(w.sha256)
        clash = sorted(name for name, shas in files.items() if len(shas) > 1)
        assert not clash, f"{card.family}: one name for different files: {clash}"


@pytest.mark.parametrize("task", registry.TASKS)
def test_every_family_has_max_or_says_why_not(task):
    """Portable + kernels per family; one without kernels states why (`[modes.kernels] status = "none"`)."""
    for card, _ in registry.cards(task).values():
        assert ("kernels" in card.modes) != bool(card.no_kernels_reason), (
            f"{task}/{card.family}: no kernels and no reason"
        )


def test_kernels_on_a_portable_only_family_raises_with_the_reason(monkeypatch):
    """`load(mode="kernels")` on a family with `[modes.kernels] status = "none"` raises the card's reason (a test card:
    every family has a kernels backend at the moment)."""
    from fastforks import BackendUnavailableError, stereo  # noqa: PLC0415

    card, variant = next(iter(registry.cards("stereo").values()))
    reason = "our fork's only kernel gains < 2 % (test card)"
    fake = dataclasses.replace(card, modes=tuple(m for m in card.modes if m != "kernels"), no_kernels_reason=reason)
    monkeypatch.setattr(registry, "resolve", lambda _task, _model: (fake, variant))
    with pytest.raises(BackendUnavailableError, match=re.escape(reason)):
        stereo.load(variant.id, mode="kernels", device="cpu")


def _module_file(name: str) -> Path | None:
    p = SRC.joinpath(*name.split(".")[1:])
    return next((f for f in (p.with_suffix(".py"), p / "__init__.py") if f.is_file()), None)


def _optimize_imports(files: list[Path]) -> set[str]:
    """Modules imported, transitively, through the `fastforks._optimize` modules that `files` import.

    A module counts with all its imports (also those inside functions: `optimize(mode="kernels")` loads its kernel
    libraries lazily); a package that is only imported because one of its submodules is (`from
    fastforks._optimize import conv_ops`) counts with its module-level imports only, which is what Python runs.
    """
    opt = SRC / "_optimize"
    full: set[Path] = set()
    top: set[Path] = set()
    queue = [(p, True) for p in files]
    while queue:
        path, whole = queue.pop()
        tree = ast.parse(path.read_text(), str(path))
        package = ["fastforks", *path.relative_to(SRC).parent.parts, ""]
        for node in ast.walk(tree) if whole else tree.body:
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[: len(package) - node.level] if node.level else []
                mod = ".".join([*[b for b in base if b], *([node.module] if node.module else [])])
                subs = [f"{mod}.{a.name}" for a in node.names if _module_file(f"{mod}.{a.name}")]
                targets = subs if len(subs) == len(node.names) else [mod, *subs]  # names from `mod` itself: all of it
            else:
                continue
            for t in targets:
                parts = t.split(".")
                for i in range(2, len(parts) + 1):  # importing a.b.c runs a.b's module level, then a.b.c
                    name, whole_t = ".".join(parts[:i]), i == len(parts)
                    f = _module_file(name)
                    if f is None or opt not in (f.parent, *f.parents) or f in full or (not whole_t and f in top):
                        continue
                    (full if whole_t else top).add(f)
                    queue.append((f, whole_t))
    out = {m for f in full for m in imported_modules(f)}
    return out | {m for f in top - full for m in imported_modules(f, top_only=True)}


@pytest.mark.slow
@pytest.mark.parametrize("task", registry.TASKS)
def test_card_kernels_match_imports(task):
    """Declare direct dependencies; shared modules may contain alternative backends.

    Static imports give bounds, not the exact executed set: for example libnet also defines a Perception Encoder
    Metal runner that AlexNet never calls. GPU parity tests separately check the libraries actually launched for
    each exercised input, including CUDA graph capture and cuDNN half-accumulate calls.
    """
    libs = {p.name for p in (SRC / "kernels").iterdir() if (p / "__init__.py").is_file()}
    cards = {card.family: card for card, _ in registry.cards(task).values()}
    for card in cards.values():
        # Shared modules count for families with a kernels backend (a portable-only family's `uses` is checked against
        # its imports in test_card_uses_matches_imports).
        family = _family_files(task, card.family)
        files = family + [
            p for u in card.uses if "kernels" in card.modes for p in sorted((SRC / "shared" / u).rglob("*.py"))
        ]
        mods = {m for p in files for m in imported_modules(p)} | _optimize_imports(files)
        used = {m.split(".")[2] for m in mods if m.startswith("fastforks.kernels.") and m.split(".")[2] in libs}
        direct = {
            m.split(".")[2]
            for p in family
            for m in imported_modules(p)
            if m.startswith("fastforks.kernels.") and m.split(".")[2] in libs
        }
        # The Metal libraries (`*_metal`) are the MPS runner's, the others the CUDA one's: check each list on its own.
        cuda, metal = {u for u in used if not u.endswith("_metal")}, {u for u in used if u.endswith("_metal")}
        _kernel_import_bounds(card.family, set(card.kernel_libs), direct & cuda, cuda)
        _kernel_import_bounds(card.family, set(card.mps_kernel_libs), direct & metal, metal)


def _kernel_import_bounds(family: str, declared: set[str], direct: set[str], possible: set[str]) -> None:
    assert direct <= declared, f"{family}: direct kernel imports missing from card: {sorted(direct - declared)}"
    assert declared <= possible, f"{family}: declared kernels absent from import closure: {sorted(declared - possible)}"


def _calls(path: Path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Call):
            yield node


@pytest.mark.slow
def test_unsafe_unpickling_only_in_core_checkpoint():
    """`weights_only=False` and `pickle.Unpickler` exist in `_core/checkpoint.py` only (the strict unpickler)."""
    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "_core" / "checkpoint.py":
            continue
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "weights_only":
                assert not (isinstance(node.value, ast.Constant) and node.value.value is False), (
                    f"{path.relative_to(SRC)}:{node.value.lineno}: weights_only=False outside _core/checkpoint.py"
                )
            if isinstance(node, ast.Attribute) and node.attr == "Unpickler":
                msg = f"{path.relative_to(SRC)}:{node.lineno}: Unpickler outside _core/checkpoint.py"
                raise AssertionError(msg)
            if isinstance(node, ast.ImportFrom) and node.module == "pickle":
                assert not any(a.name in {"Unpickler", "load", "loads"} for a in node.names), path


@pytest.mark.parametrize("task", registry.TASKS)
def test_model_code_reads_weights_through_the_context(task):
    """`model.py` does not call `torch.load`; weights come from `ctx.state_dict` (no per-family loaders)."""
    for card, _ in registry.cards(task).values():
        path = SRC / task / "models" / card.family / "model.py"
        if str(path.relative_to(SRC)) in TORCH_LOAD:
            continue
        for call in _calls(path):
            f = call.func
            name = ast.unparse(f)
            assert name not in {"torch.load", "load"}, f"{path.relative_to(SRC)}:{call.lineno}: use ctx.state_dict()"


def _terms(expression: str) -> set[str]:
    return {t for t in re.split(r"\s+(?:AND|OR)\s+", expression.strip()) if t}


def _meta_licences(area: str) -> dict[str, str]:
    return {p.parent.name: tomllib.loads(p.read_text())["licence"] for p in sorted((SRC / area).glob("*/meta.toml"))}


@pytest.mark.parametrize("task", registry.TASKS)
def test_every_licence_file_of_a_family_is_listed_in_its_card(task):
    """The distribution metadata is built from `licence.files`: an unlisted LICENSE / NOTICE is not shipped."""
    for card, _ in registry.cards(task).values():
        d = SRC / task / "models" / card.family
        on_disk = {str(p.relative_to(d)) for g in ("LICENSE*", "NOTICE*") for p in d.rglob(g)}
        assert on_disk == set(card.licence.files), (
            f"{card.family}: LICENSE files {on_disk} vs card {card.licence.files}"
        )


@pytest.mark.parametrize("kernel", sorted(_meta_licences("kernels")))
def test_kernel_licences_are_permissive(kernel):
    """Kernels ship in the core wheel, so they must be permissive."""
    assert licence_class(_meta_licences("kernels")[kernel], f"kernels/{kernel}") == "permissive"


def test_core_licence_covers_core_shared_and_kernels():
    """The core wheel's `license` is Apache-2.0 AND every licence of the shared modules and kernels it ships."""
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["license"]
    wanted = {"Apache-2.0"}
    for area in ("shared", "kernels"):
        for expression in _meta_licences(area).values():
            wanted |= _terms(expression)
    assert _terms(declared) == wanted
    assert " OR " not in declared


def test_cutlass_kernels_are_in_the_third_party_notices():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    for kernel, expression in _meta_licences("kernels").items():
        if "BSD-3-Clause" in _terms(expression):
            assert f"`{kernel}`" in notices, f"kernel {kernel} (BSD-3-Clause) is missing in THIRD_PARTY_NOTICES.md"


def _model_constants(path: Path):
    """Module-level `NAME = <number>` assignments of a `model.py` (line, name)."""
    src = path.read_text().splitlines()
    for node in ast.parse(path.read_text(), str(path)).body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            number = isinstance(value, ast.Constant) and isinstance(value.value, int | float)
            if number and all(isinstance(t, ast.Name) and t.id.lstrip("_").isupper() for t in targets):
                yield node.lineno, ast.unparse(targets[0]), "# fixed:" in src[node.lineno - 1]


@pytest.mark.parametrize("task", registry.TASKS)
def test_no_architecture_constants_in_model_code(task):
    """Architecture numbers (max_disp, iteration counts, pad multiples) are card `[params]` / `[io]`, not code.

    A module-level number in `model.py` needs a `# fixed: <reason>` comment (a value upstream hard-codes).
    """
    for card, _ in registry.cards(task).values():
        path = SRC / task / "models" / card.family / "model.py"
        for line, name, documented in _model_constants(path):
            assert documented or (str(path.relative_to(SRC)), name) in UNDOCUMENTED_CONSTANTS, (
                f"{path.relative_to(SRC)}:{line}: {name} belongs in card.toml [params] (or add '# fixed: <reason>')"
            )


@pytest.mark.parametrize("task", registry.TASKS)
def test_kernels_runners_do_not_declare_dtypes(task):
    """The max compute dtype is the card's `[modes.kernels] dtype`; `Runner.dtypes` is for portable CUDA code only."""
    for card, _ in registry.cards(task).values():
        if "kernels" not in card.modes:
            continue
        tree = ast.parse((SRC / task / "models" / card.family / "model.py").read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Kernels"):
            names = {t.id for n in cls.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            assert "dtypes" not in names, f"{card.family}: Kernels.dtypes is never read; remove it"


@pytest.mark.parametrize("task", registry.TASKS)
def test_only_redistributable_weights_are_mirrored(task):
    """Re-host weights only under a permissive licence; never NVIDIA's (their terms restrict hosting) or gated ones."""
    for card, v in registry.cards(task).values():
        lic = card.variant_licence(v)
        if any(w.mirror for w in v.weights):
            assert licence_class(lic.weights_licence) == "permissive", v.ref
            assert "NVIDIA" not in lic.weights_licence, v.ref
            assert lic.weights_licence != "LicenseRef-SAM-License", f"{v.ref}: gated SAM License weights"


def test_mirror_paths_are_family_and_file():
    """A mirror holds `<family>/<file>`, one file per path (stage_hf copies them there)."""
    import stage_hf  # noqa: PLC0415 - tool module, imported only here

    for path, uses in stage_hf.mirrored().items():
        assert {f"{card.family}/{spec.name}" for card, _, spec in uses} == {path}
        assert len({spec.sha256 for _, _, spec in uses}) == 1, path


@pytest.mark.parametrize("task", registry.TASKS)
def test_commercial_status_follows_code_and_weights_licence_only(task):
    """User decision 2026-10-31: training data is a note, never a blocker."""
    for card, v in registry.cards(task).values():
        lic = card.variant_licence(v)
        expect = licences.worst(
            licences.code_commercial(lic.code_class), licences.weights_commercial(lic.weights_licence)
        )
        st = card.status(v)
        assert st.commercial == expect, v.ref
        assert not any(r.startswith("data:") for r in st.reasons), v.ref


def test_core_wheel_extras_pin_the_model_distributions():
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    text = build_dists.core_pyproject((ROOT / "pyproject.toml").read_text(), dists, project["version"])
    extras = tomllib.loads(text)["project"]["optional-dependencies"]
    for name in dists:
        assert extras[name.removeprefix("fastforks-")] == [f"{name}=={project['version']}"]
    assert extras["kernels"] == project["optional-dependencies"]["kernels"]


@pytest.mark.parametrize("task", registry.TASKS)
def test_task_load_is_generated_by_task_api(task):
    """Option, dtype, device and mode logic lives in `_core/model.py`; a task's `load` is made by `task_api`."""
    tree = ast.parse((SRC / task / "_api.py").read_text())
    loads = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load"]
    if not loads:
        assigns = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)]
        assert any(getattr(n.value.func, "id", "") == "task_api" for n in assigns), f"{task}: no task_api"
        return
    body = loads[-1].body  # the implementation (earlier ones are `@overload`s)
    body = body[1:] if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) else body
    calls = [n for n in ast.walk(loads[-1]) if isinstance(n, ast.Call)]
    generated = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and getattr(getattr(n.value, "func", None), "id", "") == "task_api"
        and "_load" in ast.unparse(n.targets[0])
    ]
    if generated:
        # a wrapper of the generated load that adds a task-only argument (classification `labels`): it forwards
        # mode / dtype / device / options / weights to `_load` unchanged
        forwarded = [c for c in calls if getattr(c.func, "id", "") == "_load"]
        assert len(forwarded) == 1, f"{task}: load() must forward to the task_api `_load`"
        kw = {k.arg for k in forwarded[0].keywords}
        assert kw == {"mode", "device", "dtype", "options", "weights"}, f"{task}: load() forwards {sorted(kw)}"
        return
    assert len(body) == 1, f"{task}: load() does more than forward"
    assert isinstance(body[0], ast.Return)
    assert any(isinstance(c.func, ast.Attribute) and c.func.attr == "load_model" for c in calls)
    assert len(calls) <= 2, f"{task}: load() calls {len(calls)} functions (core.load_model and a model-class lookup)"
