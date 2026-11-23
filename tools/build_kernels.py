"""Build kernel-library binaries (one per target) and regenerate their `manifest.toml`.

    uv sync --group kernels
    uv run python tools/build_kernels.py [LIB ...] [--cutlass DIR] [--jobs N] [--cuda-home DIR] [--ccbin GCC]

Binaries go to `build/kernels/<lib>/` (point `FASTFORKS_KERNEL_PATH` at `build/kernels` to use them before they
are published). The default toolkit is the pip `nvidia-cuda-nvcc` of the `kernels` dependency group (local
development); `--cuda-home` selects another one, `--ccbin` its host compiler. CUDA < 13.1 headers clash with the
`rsqrt` declarations of glibc >= 2.42 unless patched (`shared/patches/cuda13.0-glibc2.42-rsqrt.patch`); the check
below uses the glibc the host compiler builds against.

On Windows x64 it builds the `shared` libraries as DLLs into `manifest.windows.toml`. Cubin libraries are built on
Linux and load on Windows too, except one with a kernel parameter over 4 KB. Its parameter metadata (`.nv.info`)
differs by host OS, and the Windows driver rejects a Linux-built one (`CUDA_ERROR_INVALID_IMAGE`). Such a library gets
a Windows cubin build: name it explicitly on Windows (it is rebuilt by default once it has a `manifest.windows.toml`).
It needs the MSVC build tools (Visual Studio 2022 Build Tools, C++ workload),
found through vswhere; `--ccbin` is unused there. The DLLs link the CUDA runtime and the C runtime statically (/MT),
export only the `extern "C"` API (a .def file) and are linked with /Brepro (no time stamps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBS = ROOT / "src" / "fastforks" / "kernels"
sys.path.insert(0, str(ROOT / "src"))
# Provisional host: one HF repo, a folder per library.
URL = "https://huggingface.co/fastforks/kernels/resolve/main/{lib}/{file}"
# Bump when the nvcc command line changes: it is part of the build id, so other flags give other ids.
# 2: shared libraries strip their symbol table (it named nvcc's per-process temp file); 3: static libstdc++;
# 4: compile from STAGE (nvcc names internal-linkage symbols after the absolute source path)
RECIPE = "4"
WINDOWS = sys.platform == "win32"
# Fixed, so every checkout compiles the same source path; build_lib refuses it unless it is ours and private.
STAGE = Path(r"C:\fastforks-build" if WINDOWS else "/tmp/fastforks-build")  # noqa: S108 - owner (and mode) checked
# Windows stand-ins for POSIX headers some sources include (`<dlfcn.h>`): on the include path of the DLL builds that
# use them, so the shared sources and the Linux build ids stay unchanged.
WIN_INCLUDE = ROOT / "tools" / "win_include"


def _check_stage() -> None:
    r"""Refuse a STAGE someone else could write to.

    Linux: owned by us with mode 700. Windows: owned by us (a new directory under C:\ is writable by its creator only:
    the root's inherited ACL gives other users read access).
    """
    STAGE.mkdir(mode=0o700, exist_ok=True)
    st = STAGE.lstat()
    if not STAGE.is_dir() or STAGE.is_symlink():
        sys.exit(f"{STAGE} must be a directory")
    if WINDOWS:
        # an elevated process creates objects owned by Administrators, who can write anywhere anyway
        if _windows_owner(STAGE) not in {_windows_user(), "S-1-5-32-544"}:
            sys.exit(f"{STAGE} must be owned by you (it was created by another account: remove it)")
    elif st.st_uid != os.getuid() or st.st_mode & 0o077:
        sys.exit(f"{STAGE} must be a directory owned by you with mode 700")


def _windows_user() -> str:
    """The SID of the current user."""
    out = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], check=True, capture_output=True, text=True).stdout
    return out.strip().rsplit(",", 1)[-1].strip('"')


def _windows_owner(path: Path) -> str:
    """The SID of the owner of `path`."""
    import ctypes  # noqa: PLC0415 - Windows only
    from ctypes import wintypes  # noqa: PLC0415

    advapi = ctypes.WinDLL("advapi32")
    sid, sd = ctypes.c_void_p(), ctypes.c_void_p()
    # SE_FILE_OBJECT = 1, OWNER_SECURITY_INFORMATION = 1
    err = advapi.GetNamedSecurityInfoW(str(path), 1, 1, ctypes.byref(sid), None, None, None, ctypes.byref(sd))
    if err:
        raise OSError(err, f"GetNamedSecurityInfoW({path})")
    text = wintypes.LPWSTR()
    try:
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(text)):
            raise ctypes.WinError()
        return str(text.value)
    finally:
        ctypes.WinDLL("kernel32").LocalFree(text)
        ctypes.WinDLL("kernel32").LocalFree(sd)


def msvc_env() -> dict[str, str]:
    """The environment of the newest Visual Studio's x64 developer prompt (`vcvars64.bat`).

    It puts cl.exe, link.exe and the Windows SDK on PATH / INCLUDE / LIB; nvcc gets it with --use-local-env (no search
    of its own).
    """
    vswhere = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    vswhere = vswhere / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    vc_tools = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
    found = ""
    if vswhere.is_file():
        found = subprocess.run(
            [vswhere, "-latest", "-products", "*", "-requires", vc_tools, "-property", "installationPath"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if not found:
        sys.exit(
            "no MSVC found: install the Visual Studio 2022 Build Tools with the C++ workload, e.g. winget install"
            ' --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add'
            ' Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
        )
    bat = Path(found) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    out = subprocess.run(f'"{bat}" >nul && set', shell=True, check=True, capture_output=True, text=True).stdout  # noqa: S602
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def manifest_name() -> str:
    """The manifest a build writes: a Windows build (DLL, or cubin) has its own, next to the Linux one."""
    return "manifest.windows.toml" if WINDOWS else "manifest.toml"


def pip_cuda_home() -> Path:
    import nvidia  # noqa: PLC0415 - only present with the kernels group

    return Path(next(iter(nvidia.__path__))) / "cu13"


def toolkit(nvcc: Path) -> tuple[str, str]:
    """(`major.minor`, full build line) of `nvcc`."""
    out = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    m = re.search(r"release (\d+\.\d+)", out)
    if m is None:
        sys.exit(f"cannot parse nvcc version:\n{out}")
    return m[1], out.strip().splitlines()[-1]


def cuda_include(cuda: Path) -> Path:
    """Header directory of the toolkit: `include/`, or `targets/x86_64-linux/include/` in a conda-forge one."""
    conda = cuda / "targets" / "x86_64-linux" / "include"
    return conda if (conda / "cuda_runtime.h").is_file() else cuda / "include"


def host_compiler(ccbin: str, env: dict[str, str] | None = None) -> str:
    """First line of `<ccbin> --version`; on Windows the banner of `cl` (on stderr) in the MSVC environment."""
    if WINDOWS:
        cl = shutil.which("cl", path=(env or {}).get("Path") or (env or {}).get("PATH"))
        if cl is None:
            sys.exit("cl.exe is not on the MSVC environment's PATH")
        out = subprocess.run([cl], check=False, capture_output=True, text=True, env=env).stderr
        # the banner is localized (its language is part of the build id otherwise): keep the version only
        m = re.search(r"\b(\d+\.\d+\.\d+(?:\.\d+)?)\b.*\b(x64|x86|ARM64)\b", out)
        if m is None:
            sys.exit(f"cannot parse the cl banner:\n{out}")
        return f"MSVC cl {m[1]} {m[2]}"
    return subprocess.run([ccbin, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]


def host_glibc(ccbin: str) -> tuple[int, int]:
    """The glibc version of the headers `ccbin` compiles against (its sysroot), `(0, 0)` if not glibc."""
    out = subprocess.run(
        [ccbin, "-dM", "-E", "-include", "features.h", "-x", "c", "-"],
        input="",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ver = dict(re.findall(r"#define __GLIBC(_MINOR)?__ (\d+)", out))
    return (int(ver[""]), int(ver["_MINOR"])) if len(ver) == 2 else (0, 0)  # noqa: PLR2004 - major + minor


def host_toolchain(nvcc_build: str, ccbin: str, glibc_patch: bool, msvc: dict[str, str] | None) -> dict[str, str]:  # noqa: FBT001
    """The manifest's `[toolchain]` table (part of the build id)."""
    if msvc is not None:
        return {
            "nvcc": nvcc_build,
            "host_compiler": host_compiler(ccbin, msvc),
            "windows_sdk": msvc.get("WindowsSDKVersion", "").rstrip("\\"),  # the UCRT headers / static libraries
            "c_runtime": "static (/MT)",
        }
    return {
        "nvcc": nvcc_build,
        "host_compiler": host_compiler(ccbin),
        "glibc": "{}.{}".format(*host_glibc(ccbin)),  # the sysroot sets the GLIBC symbol versions a .so needs
        "glibc_rsqrt_patch": str(glibc_patch).lower(),
    }


def cutlass_version(cutlass: Path) -> tuple[str, bytes]:
    """(`major.minor.patch` + commit if a git checkout, bytes of `version.h`) of a CUTLASS checkout."""
    header = (cutlass / "include" / "cutlass" / "version.h").read_bytes()
    ver = ".".join(v.decode() for v in re.findall(rb"#define CUTLASS_(?:MAJOR|MINOR|PATCH) (\d+)", header))
    git = subprocess.run(["git", "-C", cutlass, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return (f"{ver} {git.stdout.strip()}" if git.returncode == 0 else ver), header


def gencode(target: str) -> str:
    kind, num = target.split("_")
    return f"-gencode=arch=compute_{num},code={kind}_{num}"


def c_exports(lib_dir: Path, cfg: dict) -> list[str]:
    """Names of the `extern "C"` functions in the library's sources (its ctypes API)."""
    from fastforks._core.kernel_symbols import exports  # noqa: PLC0415

    return exports((lib_dir / s).read_text(encoding="utf-8") for s in cfg["sources"])


def symbol_lines(lib_dir: Path, toolchain: dict) -> list[str]:
    """The manifest's `[symbols]` table: a hash per exported function (see `fastforks._core.kernel_symbols`)."""
    from fastforks._core.kernel_symbols import library_symbols  # noqa: PLC0415

    hashes = library_symbols(lib_dir, (lib_dir / "build.toml").read_text(encoding="utf-8"), toolchain)
    return ["", "[symbols]", *(f'{k} = "{v}"' for k, v in hashes.items())]


def refresh_symbols(name: str) -> None:
    """Rewrite only the `[symbols]` table of an existing manifest (no rebuild, same `build_id`)."""
    lib_dir = LIBS / name
    path = lib_dir / manifest_name()
    path = path if path.is_file() else lib_dir / "manifest.toml"  # a cubin library without a Windows build
    text = path.read_text(encoding="utf-8")
    text = text.split("\n\n[symbols]", 1)[0].rstrip("\n")
    toolchain = tomllib.loads(text)["toolchain"]
    path.write_text("\n".join([text, *symbol_lines(lib_dir, toolchain)]) + "\n", encoding="utf-8", newline="\n")


def nvcc_cmd(  # noqa: PLR0913, PLR0917 - internal helper, all inputs needed
    cuda: Path, src_dir: Path, cfg: dict, target: str, out: Path, cutlass: Path | None, ccbin: str, exports: Path
) -> list[str]:
    host = [] if WINDOWS else [f"-ccbin={ccbin}"]  # Windows: the vcvars environment's cl.exe (--use-local-env)
    cmd = [str(cuda / "bin" / "nvcc"), *host, *cfg["flags"], gencode(target), f"-I{cuda_include(cuda)}"]
    if cfg["kind"] == "cubin":
        cmd.append("-fatbin")
    elif WINDOWS:
        # As on Linux, the toolkit is passed explicitly; MSVC comes from the vcvars environment (--use-local-env).
        # /MT links the C runtime statically (no MSVC redistributable needed); the .def file exports only our C API;
        # /Brepro drops the PE time stamps and /d1trimfile the stage path from __FILE__, for the same bytes on rebuild.
        cmd += [
            "--use-local-env",
            "--dont-use-profile",
            "-ldir",
            str(cuda / "nvvm" / "libdevice"),
            "-isystem",
            str(cuda_include(cuda) / "cccl"),
            "-cudadevrt",
            "none",
            "-shared",
            "-cudart",
            "static",
            f"-L{cuda / 'lib' / 'x64'}",
            "-Xcompiler",
            "/MT,/EHsc,/Brepro,/Zc:__cplusplus",
            "-Xcompiler",
            f"/d1trimfile:{src_dir}\\",
            "-Xlinker",
            f"/DEF:{exports}",
            "-Xlinker",
            "/Brepro",
            "-Xlinker",
            "/NOIMPLIB",
            "-Xlinker",
            "/NOEXP",
        ]
    else:
        # pip nvcc: its nvcc.profile embeds absolute build-machine paths, so pass the toolkit explicitly; the CUDA
        # runtime is linked statically so the library needs only the driver.
        cmd += [
            "--dont-use-profile",
            "-ldir",
            str(cuda / "nvvm" / "libdevice"),
            "-isystem",
            str(cuda_include(cuda) / "cccl"),
            "-cudadevrt",
            "none",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-cudart",
            "static",
            "-Xlinker",
            f"-L{cuda}/lib",
            "-Xlinker",
            f"-L{cuda}/lib64",
            "-Xcompiler",
            f"-ffile-prefix-map={src_dir}=.",
            "-Xlinker",
            "--strip-all",  # same bytes from every checkout and run; ctypes needs only the dynamic symbols
            # CUTLASS's trace.h includes <iostream>, which ties the library to the build compiler's libstdc++ (GCC 14:
            # GLIBCXX_3.4.32, newer than e.g. Ubuntu 22.04's); the ctypes C ABI shares no C++ objects with the caller.
            "-Xcompiler",
            "-static-libstdc++",
            "-Xcompiler",
            "-static-libgcc",
            "-Xlinker",
            f"--version-script={exports}",  # export only our C API, not the static libstdc++ copy
        ]
    if "cutlass" in cfg.get("needs", []):
        if cutlass is None or not (cutlass / "include" / "cutlass" / "cutlass.h").is_file():
            sys.exit("this library needs the CUTLASS v4.7.1 headers: --cutlass DIR")
        trim = f"/d1trimfile:{cutlass}\\" if WINDOWS else f"-ffile-prefix-map={cutlass}=cutlass"
        cmd += [f"-I{cutlass / 'include'}", "-Xcompiler", trim]
    if WINDOWS and win_shims(src_dir, cfg):
        cmd.append(f"-I{WIN_INCLUDE}")
    return [*cmd, "-o", str(out), *(str(src_dir / s) for s in cfg["sources"])]


def win_shims(lib_dir: Path, cfg: dict) -> list[Path]:
    """The `tools/win_include` headers (POSIX headers MSVC lacks, e.g. `<dlfcn.h>`) the library's sources include."""
    text = "\n".join((lib_dir / s).read_text(encoding="utf-8") for s in cfg["sources"])
    return [p for p in sorted(WIN_INCLUDE.glob("*.h")) if re.search(rf"#include\s*<{re.escape(p.name)}>", text)]


def lf_bytes(path: Path) -> bytes:
    """The file with LF line ends: a Windows checkout with `core.autocrlf` hashes like a Linux one."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def build_hashes(
    lib_dir: Path, cfg: dict, tk: str, toolchain: dict[str, str], cutlass: Path | None
) -> tuple[str, hashlib._Hash]:
    """(build id, sources hash) of a library; adds the CUTLASS version to `toolchain` when it needs CUTLASS."""
    h = hashlib.sha256(lf_bytes(lib_dir / "build.toml") + tk.encode() + RECIPE.encode())
    if "cutlass" in cfg.get("needs", []) and cutlass is not None:
        toolchain["cutlass"], header = cutlass_version(cutlass)
        h.update(header)
    h.update(repr(sorted(toolchain.items())).encode())  # same id <=> same sources, settings and toolchain
    # Windows header shims, only for the libraries that include one (b"" for the others: their id is unchanged)
    h.update(b"".join(lf_bytes(p) for p in (win_shims(lib_dir, cfg) if WINDOWS else [])))
    src = hashlib.sha256(lf_bytes(lib_dir / "build.toml"))
    for s in cfg["sources"]:
        h.update(lf_bytes(lib_dir / s))
        src.update(lf_bytes(lib_dir / s))
    return h.hexdigest()[:12], src


def build_lib(  # noqa: PLR0913, PLR0917 - internal helper, all inputs needed
    name: str,
    cuda: Path,
    cutlass: Path | None,
    jobs: int,
    glibc_patch: bool,  # noqa: FBT001
    ccbin: str,
) -> None:
    lib_dir = LIBS / name
    cfg = tomllib.loads((lib_dir / "build.toml").read_text(encoding="utf-8"))
    tk, nvcc_build = toolkit(cuda / "bin" / "nvcc")
    msvc = msvc_env() if WINDOWS else None
    toolchain = host_toolchain(nvcc_build, ccbin, glibc_patch, msvc)
    build_id, src = build_hashes(lib_dir, cfg, tk, toolchain, cutlass)
    out_dir = ROOT / "build" / "kernels" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".fatbin" if cfg["kind"] == "cubin" else ".dll" if WINDOWS else ".so"
    files = {t: out_dir / f"{name}-{build_id}-{t}{ext}" for t in cfg["targets"]}
    tools = os.pathsep.join(str(d) for d in (cuda / "bin", cuda / "nvvm" / "bin"))
    if WINDOWS:  # module-definition file: the DLL's exports (the import library is not written)
        exports = out_dir / f".{name}-{build_id}.def"
        exports.write_text("EXPORTS\n" + "".join(f"    {s}\n" for s in c_exports(lib_dir, cfg)), encoding="utf-8")
    else:  # linker version script, not part of the output
        exports = out_dir / f".{name}-{build_id}.exports"
        exports.write_text(f"{{ global: {'; '.join(c_exports(lib_dir, cfg))}; local: *; }};\n", encoding="utf-8")
    src_dir = STAGE / f"{name}-{build_id}"
    _check_stage()
    src_dir.mkdir(exist_ok=True)
    for s in cfg["sources"]:  # same bytes for the same build id; replace() keeps concurrent builds safe
        tmp_src = src_dir / f".{s}.{os.getpid()}"
        tmp_src.write_bytes((lib_dir / s).read_bytes())
        tmp_src.replace(src_dir / s)
    base = dict(msvc if msvc is not None else os.environ)
    path = base.pop("Path", "") or base.pop("PATH", "")  # Windows environment keys are case-insensitive
    env = dict(base, PATH=f"{tools}{os.pathsep}{path}")

    def one(target: str) -> None:
        if files[target].is_file():
            return
        tmp = files[target].with_suffix(".tmp")
        print(f"{name} {target} ...", flush=True)
        subprocess.run(nvcc_cmd(cuda, src_dir, cfg, target, tmp, cutlass, ccbin, exports), check=True, env=env)
        tmp.replace(files[target])

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(one, cfg["targets"]))
    exports.unlink()
    meta = tomllib.loads((lib_dir / "meta.toml").read_text(encoding="utf-8"))
    lines = [
        "# Generated by tools/build_kernels.py from build.toml, meta.toml and the sources; do not edit.",
        f'name = "{name}"',
        f'build_id = "{build_id}"',
        f'toolkit = "{tk}"',
        f'sources_sha256 = "{src.hexdigest()}"  # build.toml + sources, checked by the tests',
        f'kind = "{cfg["kind"]}"',
        *(['os = "windows"'] if WINDOWS else []),
        f'url = "{URL}"',
        f"licence = {json.dumps(meta['licence'])}",
        f"derived_from = {json.dumps(meta['derived_from'])}",
        "",
        "[toolchain]",
        *(f"{k} = {json.dumps(v)}" for k, v in toolchain.items()),
        "",
        "[files]",
    ]
    for t, p in files.items():
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f'{t} = {{ name = "{p.name}", sha256 = "{digest}", size = {p.stat().st_size} }}')
    lines += symbol_lines(lib_dir, toolchain)
    (lib_dir / manifest_name()).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{name}@{build_id}: {sum(p.stat().st_size for p in files.values()) / 2**20:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("libs", nargs="*", help="library names (default: all with a build.toml)")
    ap.add_argument("--cutlass", type=Path, default=os.environ.get("CUTLASS_DIR"), help="CUTLASS checkout")
    ap.add_argument("--jobs", type=int, default=4, help="parallel nvcc processes (CUTLASS: ~3 GB RAM each)")
    ap.add_argument("--cuda-home", type=Path, help="CUDA toolkit (default: the pip nvidia-cuda-nvcc)")
    ap.add_argument("--ccbin", default="gcc", help="host compiler (default: gcc)")
    ap.add_argument("--symbols", action="store_true", help="only refresh the [symbols] tables (no build)")
    a = ap.parse_args()
    if a.symbols:
        for name in a.libs or sorted(p.parent.name for p in LIBS.glob("*/build.toml")):
            refresh_symbols(name)
        return
    cuda = a.cuda_home or pip_cuda_home()
    tk, _ = toolkit(cuda / "bin" / "nvcc")
    names = a.libs or sorted(p.parent.name for p in LIBS.glob("*/build.toml"))
    if WINDOWS:
        kinds = {n: tomllib.loads((LIBS / n / "build.toml").read_text(encoding="utf-8"))["kind"] for n in names}
        # A cubin library gets a Windows build only on request (named, or it already has one): see the docstring.
        win = {n for n, k in kinds.items() if k == "shared" or a.libs or (LIBS / n / "manifest.windows.toml").is_file()}
        if skipped := sorted(set(names) - win):
            print(f"skipping the cubin libraries (built on Linux, they load on Windows): {', '.join(skipped)}")
        for name in (n for n in names if n in win):
            build_lib(name, cuda, a.cutlass, a.jobs, False, a.ccbin)  # noqa: FBT003 - no glibc
        return
    header = (cuda_include(cuda) / "crt" / "math_functions.h").read_text(encoding="utf-8")
    patched = "_NV_RSQRT_SPECIFIER" in header
    if tuple(map(int, tk.split("."))) < (13, 1) and host_glibc(a.ccbin) >= (2, 42) and not patched:
        sys.exit(
            f"CUDA {tk} headers clash with glibc >= 2.42 (rsqrt): patch them (shared/patches/) or use a host compiler"
            " with an older sysroot"
        )
    for name in names:
        build_lib(name, cuda, a.cutlass, a.jobs, patched, a.ccbin)


if __name__ == "__main__":
    main()
