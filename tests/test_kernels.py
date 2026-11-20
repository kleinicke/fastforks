from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest
import torch

from fastforks import KernelUnavailableError
from fastforks._core import kernels
from fastforks._core.kernels import read_manifest, select_target
from fastforks._core.licences import licence_class

LIBS = Path(kernels.__file__).parents[1] / "kernels"
BUILT = sorted(p.parent for p in LIBS.glob("*/build.toml"))
ALL = ["sm_80", "sm_86", "sm_89", "sm_90", "sm_120", "compute_80"]


@pytest.mark.parametrize(
    ("arch", "want"),
    [
        ((8, 0), "sm_80"),
        ((8, 6), "sm_86"),
        ((8, 7), "sm_86"),  # Orin: same major, lower minor SASS
        ((8, 9), "sm_89"),
        ((9, 0), "sm_90"),
        ((10, 0), "compute_80"),  # B200: no SASS of major 10 -> PTX
        ((12, 0), "sm_120"),
        ((12, 1), "sm_120"),
        ((7, 5), None),  # Turing: older than every target
    ],
)
def test_select_target(arch, want):
    assert select_target(ALL, arch) == want


@pytest.mark.parametrize(
    ("name", "env", "want"),
    [
        ("NVIDIA GeForce RTX 5060 Ti", "", True),
        ("NVIDIA L4", "", False),
        ("NVIDIA RTX A6000", "", False),
        ("NVIDIA L4", "fp16", True),
        ("NVIDIA GeForce RTX 4090", "FP32", False),
    ],
)
def test_half_rate_fp32_acc(monkeypatch, name, env, want):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d=None: name)
    monkeypatch.setenv("FASTFORKS_ACC", env)
    assert kernels.half_rate_fp32_acc(torch.device("cuda", 0)) is want


def test_select_target_prefers_newest_ptx():
    assert select_target(["compute_80", "compute_90"], (12, 0)) == "compute_90"
    assert select_target(["compute_80", "compute_90"], (8, 9)) == "compute_80"


ALL_LIBS = sorted(p.parent for p in LIBS.glob("*/__init__.py") if p.parent != LIBS)
_LB = r"(?:__launch_bounds__\s*\([^)]*\)\s+)?"
_KERNEL = re.compile(rf'extern\s+"C"\s+__global__\s+{_LB}void\s+{_LB}(\w+)\s*\(([^)]*)\)')
_MACRO = re.compile(r"^#define\s+(\w+)\((\w+(?:\s*,\s*\w+)*)\)\s+((?:.*\\\n)*.*)$", re.MULTILINE)
_OBJ_MACRO = re.compile(r"^#define\s+(\w+)[ \t]+((?:.*\\\n)*.*)$", re.MULTILINE)  # no parameters
_CODES = {"long long": "l", "int64_t": "l", "int": "i", "int32_t": "i", "float": "f"}


def _code(param: str, structs: dict[str, str]) -> str:
    """Launch-argument code of one C parameter (`const __half* __restrict__ x` -> `p`, `long long n` -> `l`).

    By-value structs map through the library's `STRUCTS` (type name -> codes of its fields)."""
    p = " ".join(param.split())
    if "*" in p:
        return "p"
    t = p.removeprefix("const ").rsplit(" ", 1)[0]
    return {**_CODES, **structs}.get(t, f"?{p}")


def _instantiate(body: str, names: list[str], args: str) -> str | None:
    vals = [a.strip() for a in args.split(",")]
    if len(vals) != len(names):
        return None
    text = body.replace("\\\n", " ")
    for par, v in zip(names, vals, strict=True):
        text = re.sub(rf"\s*##\s*{par}\b|\b{par}\s*##\s*|\b{par}\b", v, text)
    return text


def _expand(src: str) -> str:
    """`src` plus its kernel macros instantiated (`#define K(V) ... name_v##V(...)` / `#define K(name, B) ... name(...)`
    and `K(1)` / `K(k_32, 32)` lines); a (parameter-list) macro as a kernel's whole parameter list (`k(ARGS(T))`) is
    replaced in place."""
    for name, body in _OBJ_MACRO.findall(src):  # `#define ARGS ...` used as `k(ARGS)`
        flat = "(" + body.replace("\\\n", " ") + ")"
        src = re.sub(rf"\(\s*{name}\s*\)", lambda _m, flat=flat: flat, src)
    macros = [(name, [p.strip() for p in pars.split(",")], body) for name, pars, body in _MACRO.findall(src)]
    for name, names, body in macros:

        def inline(m: re.Match[str], body: str = body, names: list[str] = names) -> str:
            return f"({_instantiate(body, names, m[1]) or m[0]})"

        src = re.sub(rf"\(\s*{name}\(([^()]*)\)\s*\)", inline, src)
    out = [src]
    for name, names, body in macros:
        uses = re.findall(rf"^{name}\(([^)]*)\)\s*$", src, re.MULTILINE)
        out += filter(None, (_instantiate(body, names, a) for a in uses))
    return "\n".join(out)


@pytest.mark.parametrize(
    "lib", [p for p in BUILT if (p / "manifest.toml").read_text(encoding="utf-8").count('kind = "cubin"')]
)
def test_signatures_match_sources(lib):
    """`SIGNATURES` (what the launcher packs) matches every `extern "C" __global__` kernel it names."""
    mod = importlib.import_module(f"fastforks.kernels.{lib.name}")
    structs = getattr(mod, "STRUCTS", {})
    found = {
        m[1]: "".join(_code(a, structs) for a in m[2].split(",") if a.strip())
        for cu in lib.glob("*.cu")
        for m in _KERNEL.finditer(
            _expand(re.sub(r"//[^\n]*", "", cu.read_text(encoding="utf-8")))
        )  # comments may hold "(" / ","
    }
    bad = {k: (v, found.get(k)) for k, v in mod.SIGNATURES.items() if found.get(k) != v}
    assert not bad, f"{lib.name}: SIGNATURES vs .cu (declared, source): {bad}"


@pytest.mark.parametrize("lib", ALL_LIBS, ids=lambda p: p.name)
def test_kernel_licences(lib):
    """Kernel libraries ship with the core: permissive, with provenance; manifests carry the same licence."""
    meta = tomllib.loads((lib / "meta.toml").read_text(encoding="utf-8"))
    assert set(meta) == {"licence", "derived_from"}
    assert licence_class(meta["licence"], f"kernels/{lib.name}") == "permissive"
    if (lib / "manifest.toml").is_file():
        assert read_manifest(lib / "manifest.toml").licence == meta["licence"], "stale manifest: run build_kernels"


# manifest.toml, and manifest.windows.toml for a library with a Windows build (a DLL, or a cubin with large parameters)
MANIFESTS = sorted(p for lib in BUILT for p in (lib / "manifest.toml", lib / "manifest.windows.toml") if p.is_file())


def _manifest_id(path: Path) -> str:
    return path.parent.name + ("-windows" if path.name == "manifest.windows.toml" else "")


@pytest.mark.parametrize("path", MANIFESTS, ids=_manifest_id)
def test_manifest_is_current(path):
    """The committed manifest belongs to the committed sources (rebuild with tools/build_kernels.py otherwise)."""
    lib = path.parent
    m = read_manifest(path)
    cfg = tomllib.loads((lib / "build.toml").read_text(encoding="utf-8"))
    h = hashlib.sha256((lib / "build.toml").read_bytes())
    for s in cfg["sources"]:
        h.update((lib / s).read_bytes())
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["sources_sha256"] == h.hexdigest(), "sources changed: run tools/build_kernels.py " + lib.name
    assert m.name == lib.name
    assert m.licence == tomllib.loads((lib / "meta.toml").read_text(encoding="utf-8"))["licence"]
    assert (m.system == "windows") == (path.name == "manifest.windows.toml")
    assert sorted(m.files) == sorted(cfg["targets"])
    assert all(f.name.startswith(f"{lib.name}-{m.build_id}-") for f in m.files.values())


@pytest.mark.parametrize("lib", [p for p in BUILT if (p / "manifest.compat.toml").is_file()], ids=lambda p: p.name)
def test_compat_build_matches(lib):
    """`manifest.compat.toml` is the same sources and targets built with an older toolkit (rebuild or delete it)."""
    m = read_manifest(lib / "manifest.toml")
    c = m.compat
    assert c is not None
    assert c.compat is None
    raw = tomllib.loads((lib / "manifest.toml").read_text(encoding="utf-8"))
    raw_c = tomllib.loads((lib / "manifest.compat.toml").read_text(encoding="utf-8"))
    assert raw_c["sources_sha256"] == raw["sources_sha256"], "sources changed: rebuild the compat build too"
    assert (c.name, c.kind, c.licence, sorted(c.files)) == (m.name, m.kind, m.licence, sorted(m.files))
    assert c.toolkit < m.toolkit
    assert c.build_id != m.build_id
    assert sorted(c.symbols) == sorted(m.symbols)
    assert all(f.name.startswith(f"{lib.name}-{c.build_id}-") for f in c.files.values())


def test_for_driver_picks_the_compat_build():
    """The compat build loads only on a driver too old for the main build but new enough for it."""
    m = read_manifest(LIBS / "fvm_conv" / "manifest.toml")
    assert m.compat is not None
    assert m.compat.toolkit < m.toolkit
    assert kernels.for_driver(m, m.toolkit) is m
    assert kernels.for_driver(m, (m.toolkit[0], m.toolkit[1] + 1)) is m
    assert kernels.for_driver(m, m.compat.toolkit) is m.compat
    too_old = (m.compat.toolkit[0], m.compat.toolkit[1] - 1)
    assert kernels.for_driver(m, too_old) is m  # too old for both: `binary` says so
    plain = read_manifest(LIBS / "raft" / "manifest.toml")
    assert plain.compat is None
    assert kernels.for_driver(plain, (11, 0)) is plain


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["fvm_conv", "igevpp", "mocha", "s2m2_cutlass"])
def test_compat_build_loads(monkeypatch, name):
    """On a driver for the compat toolkit only, a library with a compat build loads that build."""
    mod = importlib.import_module(f"fastforks.kernels.{name}")  # needs the GPU library
    path = LIBS / name / "manifest.toml"
    if kernels.platform_manifest(path) != path:
        pytest.skip("the Windows build of this library has no compat build")
    m = read_manifest(path)
    assert m.compat is not None
    monkeypatch.setattr(kernels, "driver_version", lambda: m.compat.toolkit)
    device = torch.device("cuda")
    if name == "fvm_conv":
        tag = mod.FvmLibrary(path, device, mod.SIGNATURES).tag
    elif m.kind == "cubin":
        tag = kernels.CubinModule(m, device, mod.SIGNATURES).tag  # not `load`: that one caches per device
    else:
        tag = kernels.shared_library(path, device)[1]
    assert tag.startswith(f"{m.compat.tag}:")


@pytest.mark.parametrize("path", MANIFESTS, ids=_manifest_id)
def test_manifest_symbols_are_current(path):
    """`[symbols]` matches the sources (refresh with `tools/build_kernels.py --symbols`)."""
    from fastforks._core.kernel_symbols import library_symbols  # noqa: PLC0415

    lib = path.parent
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    got = library_symbols(lib, (lib / "build.toml").read_text(encoding="utf-8"), raw["toolchain"])
    assert raw["symbols"] == got, "stale [symbols]: run tools/build_kernels.py --symbols " + lib.name


@pytest.mark.parametrize("lib", BUILT, ids=lambda p: p.name)
def test_every_export_has_a_manifest_hash(lib):
    """Every `extern "C"` function (macro-generated ones too) and every declared signature has a `[symbols]` hash."""
    from fastforks._core.kernel_symbols import exports  # noqa: PLC0415

    cfg = tomllib.loads((lib / "build.toml").read_text(encoding="utf-8"))
    raw = tomllib.loads((lib / "manifest.toml").read_text(encoding="utf-8"))
    exported = exports([(lib / s).read_text(encoding="utf-8") for s in cfg["sources"]])
    assert not set(exported) - set(raw["symbols"]), f"exports without a hash in {lib.name}/manifest.toml"
    assert not set(raw["symbols"]) - set(exported), "hash for a function the sources no longer export"
    sigs = getattr(importlib.import_module(f"fastforks.kernels.{lib.name}"), "SIGNATURES", {})
    assert not set(sigs) - set(raw["symbols"]), "declared kernel signature without a hash"


_SRC_MACRO = """
#define K_(V) extern "C" __global__ void __launch_bounds__(64) kern_v##V(int* o) { o[0] = V; }
K_(1)
K_(2)
#undef K_
"""


def test_macro_generated_exports_are_hashed():
    from fastforks._core.kernel_symbols import exports, symbol_hashes  # noqa: PLC0415

    assert exports([_SRC_MACRO]) == ["kern_v1", "kern_v2"]
    h = symbol_hashes([_SRC_MACRO], ["kern_v1", "kern_v2"], "env")
    assert h["kern_v1"] != h["kern_v2"]


_SRC = """
#include <cuda.h>
#define K 3  // comment { with brace
struct P { int a; };
__device__ int helper(int x) { return x * K; }
__device__ int other(int x) { return x + 1; }
__global__ void ka(int* o) { o[0] = helper(1); const char* s = "}{"; }
__global__ void kb(int* o) { o[0] = other(1); }
extern "C" void run_a(int* o) { ka<<<1, 1>>>(o); }
extern "C" void run_b(int* o) { kb<<<1, 1>>>(o); }
"""
_SRC_LB = """
__device__ int helper(int x) { return x + 1; }
extern "C" __global__ void __launch_bounds__(256) ka(int* o) { o[0] = helper(1); }
extern "C" __global__ void __launch_bounds__(128) kb(int* o) { o[0] = 2; }
"""


def test_symbol_hashes_follow_dependencies():
    from fastforks._core.kernel_symbols import exports, symbol_hashes  # noqa: PLC0415

    assert exports([_SRC]) == ["run_a", "run_b"]
    base = symbol_hashes([_SRC], ["run_a", "run_b"], "env")
    assert base["run_a"] != base["run_b"]
    edited = symbol_hashes([_SRC.replace("x + 1", "x + 2")], ["run_a", "run_b"], "env")
    assert edited["run_a"] == base["run_a"], "an edit to a function run_a does not call changed its hash"
    assert edited["run_b"] != base["run_b"]
    assert symbol_hashes([_SRC.replace("// comment", "// other  words")], ["run_a"], "env") == {"run_a": base["run_a"]}
    assert symbol_hashes([_SRC], ["run_a"], "env2")["run_a"] != base["run_a"], "environment is part of the hash"
    shared = symbol_hashes([_SRC.replace("#define K 3", "#define K 4")], ["run_a", "run_b"], "env")
    assert shared["run_a"] != base["run_a"], "a changed macro invalidates everything"
    with pytest.raises(ValueError, match="no definition"):
        symbol_hashes([_SRC], ["missing"], "env")


def test_symbol_hashes_skip_launch_bounds():
    from fastforks._core.kernel_symbols import exports, symbol_hashes  # noqa: PLC0415

    assert exports([_SRC_LB]) == ["ka", "kb"]
    base = symbol_hashes([_SRC_LB], ["ka", "kb"], "env")
    edited = symbol_hashes([_SRC_LB.replace("x + 1", "x + 2")], ["ka", "kb"], "env")
    assert edited["kb"] == base["kb"], "kernels that share __launch_bounds__ do not depend on each other"
    assert edited["ka"] != base["ka"]


def test_binary_needs_cuda():
    m = read_manifest(BUILT[0] / "manifest.toml")
    with pytest.raises(KernelUnavailableError, match="needs a CUDA device") as e:
        kernels.binary(m, torch.device("cpu"))
    assert e.value.library == m.name


def test_linear_launch():
    assert kernels.linear(1) == ((1, 1, 1), (256, 1, 1))
    assert kernels.linear(257)[0] == (2, 1, 1)
    assert kernels.linear(10**12)[0] == (65535 * 8, 1, 1)


def test_fallback_counters():
    kernels.reset_fallbacks()
    kernels.note_fallback("lib", "conv 3x3")
    kernels.note_fallback("lib", "conv 3x3")
    assert kernels.fallbacks() == {("lib", "conv 3x3"): 2}
    kernels.reset_fallbacks()
    assert kernels.fallbacks() == {}


def test_cuda_device_passthrough():
    assert kernels.cuda_device(torch.device("cpu")) == torch.device("cpu")
    assert kernels.cuda_device(torch.device("cuda:3")) == torch.device("cuda:3")


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"hip": "6.4"}, "NVIDIA CUDA only; torch .* is ROCm/HIP"),
        ({"platform": "darwin", "machine": "arm64"}, r"built for Linux x86-64 .*, not darwin arm64"),
        ({"machine": "aarch64"}, r"built for Linux x86-64 .*, not linux aarch64"),
        ({"platform": "win32", "machine": "ARM64"}, r"built for Linux x86-64 .*, not win32 ARM64"),
        ({"platform": "win32", "machine": "AMD64", "kind": "shared"}, r"Linux \.so .*only cubin kernel libraries"),
    ],
)
def test_binary_rejects_other_platforms(monkeypatch, patch, match):
    monkeypatch.setattr(torch.version, "hip", patch.get("hip"))
    monkeypatch.setattr(kernels.sys, "platform", patch.get("platform", "linux"))
    monkeypatch.setattr(kernels.platform, "machine", lambda: patch.get("machine", "x86_64"))
    m = read_manifest(BUILT[0] / "manifest.toml")
    m = dataclasses.replace(m, kind=patch.get("kind", m.kind))
    with pytest.raises(KernelUnavailableError, match=match):
        kernels.binary(m, torch.device("cuda", 0))


@pytest.mark.parametrize(("os_name", "machine"), [("linux", "x86_64"), ("win32", "AMD64")])
def test_cubin_libraries_pass_the_platform_gate(monkeypatch, os_name, machine):
    """A cubin is loaded by the driver API on Linux and Windows alike: the gate lets both through to the CUDA checks."""
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(kernels.sys, "platform", os_name)
    monkeypatch.setattr(kernels.platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = dataclasses.replace(read_manifest(BUILT[0] / "manifest.toml"), kind="cubin")
    assert kernels.platform_reason(m) is None
    with pytest.raises(KernelUnavailableError, match="has no usable CUDA"):
        kernels.binary(m, torch.device("cuda", 0))


@pytest.mark.parametrize(
    ("os_name", "system", "ok"),
    [("linux", "linux", True), ("linux", "windows", False), ("win32", "linux", False), ("win32", "windows", True)],
)
def test_shared_libraries_load_only_on_their_os(monkeypatch, os_name, system, ok):
    """A `shared` library is a host binary: the .so build runs on Linux, the DLL build (os = "windows") on Windows."""
    monkeypatch.setattr(kernels.sys, "platform", os_name)
    monkeypatch.setattr(kernels.platform, "machine", lambda: "AMD64" if os_name == "win32" else "x86_64")
    m = dataclasses.replace(read_manifest(BUILT[0] / "manifest.toml"), kind="shared", system=system)
    assert (kernels.platform_reason(m) is None) == ok


def _windows_manifest(tmp_path, kind="shared", name="manifest.windows.toml"):
    text = (BUILT[0] / "manifest.toml").read_text(encoding="utf-8")
    text = re.sub(r'(?m)^kind = ".*"$', f'kind = "{kind}"\nos = "windows"', text, count=1)
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path / name


def test_windows_manifest_is_read_and_preferred_on_windows(monkeypatch, tmp_path):
    """`manifest.windows.toml` (os = "windows", a DLL build) is what Windows loads next to `manifest.toml`; Linux keeps
    `manifest.toml`, and a library without a Windows build keeps its Linux manifest (rejected by the platform gate)."""
    path = _windows_manifest(tmp_path)
    assert read_manifest(path).system == "windows"
    linux = tmp_path / "manifest.toml"
    monkeypatch.setattr(kernels.sys, "platform", "linux")
    assert kernels.platform_manifest(linux) == linux
    monkeypatch.setattr(kernels.sys, "platform", "win32")
    assert kernels.platform_manifest(linux) == path
    path.unlink()
    assert kernels.platform_manifest(linux) == linux


@pytest.mark.parametrize(("os_name", "system", "ok"), [("linux", "windows", False), ("win32", "windows", True)])
def test_windows_cubin_build_runs_on_windows_only(monkeypatch, tmp_path, os_name, system, ok):
    """A cubin with a kernel parameter over 4 KB carries host-OS parameter metadata: its Windows build
    (`manifest.windows.toml`, kind cubin) is read, preferred on Windows and refused on Linux."""
    m = read_manifest(_windows_manifest(tmp_path, kind="cubin"))
    assert (m.kind, m.system) == ("cubin", system)
    monkeypatch.setattr(kernels.sys, "platform", os_name)
    monkeypatch.setattr(kernels.platform, "machine", lambda: "AMD64" if os_name == "win32" else "x86_64")
    assert (kernels.platform_reason(m) is None) == ok


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loader")
def test_dll_image_loads_from_a_private_copy():
    """Windows has no memfd: the verified bytes are written to a fresh per-process directory and loaded from there,
    with the DLL search limited to System32 (no DLL planting from the current or PATH directories)."""
    src = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "version.dll"
    lib = kernels._load_dll_image(src.read_bytes(), "ff_test_version.dll")
    assert lib.GetFileVersionInfoSizeW
    copy = kernels._dll_dir() / "ff_test_version.dll"
    assert copy.read_bytes() == src.read_bytes()
    # a second load (e.g. a second model using the library) never overwrites a loaded copy: it gets its own file
    again = kernels._load_dll_image(src.read_bytes(), "ff_test_version.dll")
    assert again.GetFileVersionInfoSizeW
    assert copy.read_bytes() == (kernels._dll_dir() / "ff_test_version.1.dll").read_bytes()


def test_binary_names_the_driver_remedy(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(kernels.sys, "platform", "linux")
    monkeypatch.setattr(kernels.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(kernels, "driver_version", lambda: (12, 4))
    m = read_manifest(BUILT[0] / "manifest.toml")
    with pytest.raises(KernelUnavailableError, match=r"driver supports CUDA 12.4, kernels need >= .*update the NVIDIA"):
        kernels.binary(m, torch.device("cuda", 0))


def _h(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float16)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda fb: fb.ln_lastdim(_h(2, 16), torch.ones(8), None, 1e-5, bias_free=True), "ln_lastdim"),
        (
            lambda fb: fb.ln_lastdim(_h(2, 16), torch.ones(16), torch.ones(16).half(), 1e-5, bias_free=False),
            "ln_lastdim",
        ),
        (lambda fb: fb.ln_lastdim(_h(2, 4096), torch.ones(4096), None, 1e-5, bias_free=True), "C <= 2048"),
        (lambda fb: fb.dwgate(_h(1, 4, 4, 8), torch.ones(9, 8), torch.ones(8).half()), "dwgate"),
        (lambda fb: fb.dwconv3(_h(1, 4, 4, 8), _h(3, 3, 8), torch.ones(8)), "dwconv3"),
        (lambda fb: fb.dwconv3(_h(1, 4, 4, 8), _h(3, 3, 8), _h(4)), "dwconv3"),
        (
            lambda fb: fb.win_rpe_attn(
                _h(1, 4, 4, 1, 96), _h(48, 96), None, heads=1, ws=4, n=1, grid=(4, 4), shift=0, scale=1.0
            ),
            r"\[49, 96\] table",
        ),
        (
            lambda fb: fb.win_rpe_attn(
                _h(1, 4, 4, 1, 96), _h(49, 96), None, heads=1, ws=4, n=1, grid=(4, 4), shift=4, scale=1.0
            ),
            "shift 4 outside",
        ),
        (
            lambda fb: fb.win_rpe_attn(
                _h(1, 4, 4, 1, 96),
                _h(49, 96),
                torch.zeros(0, 16, 16),
                heads=1,
                ws=4,
                n=1,
                grid=(4, 4),
                shift=0,
                scale=1.0,
            ),
            "nW >= 1",
        ),
    ],
)
def test_fastbb_rejects_misfit_arguments_before_launch(call, match):
    """Shapes, dtypes and window geometry the fastbb kernels would read out of bounds raise with the reason."""
    from fastforks.kernels import fastbb  # noqa: PLC0415

    with pytest.raises(ValueError, match=match):
        call(fastbb)


def test_symbol_hashes_namespace_needs_no_semicolon():
    from fastforks._core.kernel_symbols import items, symbol_hashes  # noqa: PLC0415

    src = "namespace { int h(int x) { return x; } }\nconstexpr int N = 4;\n"
    src += 'extern "C" int run(int x) { return h(x) + N; }\n'
    assert len(items(src)) == 3, "the statement after a namespace is its own item"
    assert set(symbol_hashes([src], ["run"], "env")) == {"run"}


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="memfd_create is Linux-only")
def test_in_memory_images_do_not_alias():
    """Two images loaded one after the other each get their own handle (a reused /proc/self/fd/N would not)."""
    found = [
        next((p for d in ("/usr/lib/x86_64-linux-gnu", "/lib64", "/usr/lib64") if (p := Path(d, n)).is_file()), None)
        for n in ("libm.so.6", "libz.so.1")
    ]
    if None in found:
        pytest.skip("libm / libz not found")
    first = kernels._load_image(found[0].read_bytes(), "first")
    second = kernels._load_image(found[1].read_bytes(), "second")
    assert first.cos
    assert second.zlibVersion  # the bug returned libm's handle again: AttributeError


@pytest.mark.gpu
def test_swiglu_long_sequence():
    """Check the ViT-H+ activation at 512 px, above the golden fixture's sequence length."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    mod = importlib.import_module("fastforks.kernels.swiglu")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(13)
    packed = torch.randn((1029, 10240), dtype=torch.float16, device=device, generator=generator)
    gate, up = packed.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    result = torch.empty_like(expected)
    pairs = result.numel() // 2
    mod.load(device).launch("swiglu", *kernels.linear(pairs), packed, result, result.shape[-1] // 2, pairs)
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-3)


@pytest.mark.gpu
def test_edge_component_threshold():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    components = importlib.import_module("fastforks.kernels.edge_components")
    mask = torch.zeros((2, 20, 20), dtype=torch.bool, device="cuda")
    mask[0, 2, 1:12] = True
    mask[0, 3, 12] = True  # Diagonal contact does not join components.
    mask[1, 2, 1:13] = True
    expected = torch.zeros_like(mask)
    expected[1] = mask[1]
    torch.testing.assert_close(components.keep_twelve(mask), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("size", [(13, 7), (4, 19), (32, 37), (17, 23)])
def test_pil_cuda_integer_resampling(size):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    resize = importlib.import_module("fastforks.shared.pil_resize")
    image = torch.randint(256, (2, 3, 17, 23), dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(resize.pil_bicubic_cuda(image, size), resize.pil_bicubic(image, size), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("factor", [4, 8])
def test_convex_flow_upsampling(factor):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    module = importlib.import_module("fastforks.kernels.flow_upsample")
    utils = importlib.import_module("fastforks.flow.models.gmflow.net.utils")
    flow = torch.randn(2, 2, 5, 11, device="cuda")
    mask = torch.randn(2, 9 * factor**2, 5, 11, device="cuda")
    want = utils.upsample_flow_with_mask(flow, mask, upsample_factor=factor)
    torch.testing.assert_close(module.convex(flow, mask, factor), want, atol=4e-6, rtol=2e-6)


@pytest.mark.gpu
def test_multiscale_deformable_sampling():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    module = importlib.import_module("fastforks.kernels.msda")
    portable = importlib.import_module("fastforks.detection._attention")
    levels = [(9, 12), (5, 6), (2, 3)]
    shapes = torch.tensor(levels, device="cuda")
    sizes = shapes.prod(1)
    starts = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
    args = {"dtype": torch.float16, "device": "cuda"}
    value = torch.randn(2, sum(h * w for h, w in levels), 4, 16, **args)
    locations = torch.rand(2, 19, 4, len(levels), 4, 2, **args) * 1.2 - 0.1
    weights = torch.randn(2, 19, 4, len(levels), 4, **args)
    weights = weights.flatten(-2).softmax(-1).reshape_as(weights)
    want = portable.deformable_sample(value, locations.flatten(3, 4), weights.flatten(3), levels, 4)
    got = module.sample(value, shapes, levels, starts, locations, weights, 0)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
