"""Build kernel-library binaries (one per target) and regenerate their `manifest.toml`.

    uv sync --group kernels
    uv run python tools/build_kernels.py [LIB ...] [--cutlass DIR] [--jobs N] [--cuda-home DIR] [--ccbin GCC]

Binaries go to `build/kernels/<lib>/` (point `FASTFORKS_KERNEL_PATH` at `build/kernels` to use them before they
are published). The default toolkit is the pip `nvidia-cuda-nvcc` of the `kernels` dependency group (local
development); `--cuda-home` selects another one, `--ccbin` its host compiler. CUDA < 13.1 headers clash with the
`rsqrt` declarations of glibc >= 2.42 unless patched (`shared/patches/cuda13.0-glibc2.42-rsqrt.patch`); the check
below uses the glibc the host compiler builds against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
# Fixed, so every checkout compiles the same source path; build_lib refuses it unless it is ours and private.
STAGE = Path("/tmp/fastforks-build")  # noqa: S108 - ownership and mode are checked before use


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


def host_compiler(ccbin: str) -> str:
    """First line of `<ccbin> --version`."""
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

    return exports((lib_dir / s).read_text() for s in cfg["sources"])


def symbol_lines(lib_dir: Path, toolchain: dict) -> list[str]:
    """The manifest's `[symbols]` table: a hash per exported function (see `fastforks._core.kernel_symbols`)."""
    from fastforks._core.kernel_symbols import library_symbols  # noqa: PLC0415

    hashes = library_symbols(lib_dir, (lib_dir / "build.toml").read_text(), toolchain)
    return ["", "[symbols]", *(f'{k} = "{v}"' for k, v in hashes.items())]


def refresh_symbols(name: str) -> None:
    """Rewrite only the `[symbols]` table of an existing manifest (no rebuild, same `build_id`)."""
    lib_dir = LIBS / name
    path = lib_dir / "manifest.toml"
    text = path.read_text()
    text = text.split("\n\n[symbols]", 1)[0].rstrip("\n")
    toolchain = tomllib.loads(text)["toolchain"]
    path.write_text("\n".join([text, *symbol_lines(lib_dir, toolchain)]) + "\n")


def nvcc_cmd(  # noqa: PLR0913, PLR0917 - internal helper, all inputs needed
    cuda: Path, src_dir: Path, cfg: dict, target: str, out: Path, cutlass: Path | None, ccbin: str, exports: Path
) -> list[str]:
    cmd = [str(cuda / "bin" / "nvcc"), f"-ccbin={ccbin}", *cfg["flags"], gencode(target), f"-I{cuda_include(cuda)}"]
    if cfg["kind"] == "cubin":
        cmd.append("-fatbin")
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
        cmd += [f"-I{cutlass}/include", "-Xcompiler", f"-ffile-prefix-map={cutlass}=cutlass"]
    return [*cmd, "-o", str(out), *(str(src_dir / s) for s in cfg["sources"])]


def build_lib(  # noqa: PLR0913, PLR0917 - internal helper, all inputs needed
    name: str,
    cuda: Path,
    cutlass: Path | None,
    jobs: int,
    glibc_patch: bool,  # noqa: FBT001
    ccbin: str,
) -> None:
    lib_dir = LIBS / name
    cfg = tomllib.loads((lib_dir / "build.toml").read_text())
    tk, nvcc_build = toolkit(cuda / "bin" / "nvcc")
    toolchain = {
        "nvcc": nvcc_build,
        "host_compiler": host_compiler(ccbin),
        "glibc": "{}.{}".format(*host_glibc(ccbin)),  # the sysroot sets the GLIBC symbol versions a .so needs
        "glibc_rsqrt_patch": str(glibc_patch).lower(),
    }
    h = hashlib.sha256((lib_dir / "build.toml").read_bytes() + tk.encode() + RECIPE.encode())
    if "cutlass" in cfg.get("needs", []) and cutlass is not None:
        toolchain["cutlass"], header = cutlass_version(cutlass)
        h.update(header)
    h.update(repr(sorted(toolchain.items())).encode())  # same id <=> same sources, settings and toolchain
    src = hashlib.sha256((lib_dir / "build.toml").read_bytes())
    for s in cfg["sources"]:
        h.update((lib_dir / s).read_bytes())
        src.update((lib_dir / s).read_bytes())
    build_id = h.hexdigest()[:12]
    out_dir = ROOT / "build" / "kernels" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".fatbin" if cfg["kind"] == "cubin" else ".so"
    files = {t: out_dir / f"{name}-{build_id}-{t}{ext}" for t in cfg["targets"]}
    tools = os.pathsep.join(str(d) for d in (cuda / "bin", cuda / "nvvm" / "bin"))
    exports = out_dir / f".{name}-{build_id}.exports"  # linker version script, not part of the output
    exports.write_text(f"{{ global: {'; '.join(c_exports(lib_dir, cfg))}; local: *; }};\n")
    src_dir = STAGE / f"{name}-{build_id}"
    STAGE.mkdir(mode=0o700, exist_ok=True)
    st = STAGE.lstat()
    if not STAGE.is_dir() or STAGE.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
        sys.exit(f"{STAGE} must be a directory owned by you with mode 700")
    src_dir.mkdir(exist_ok=True)
    for s in cfg["sources"]:  # same bytes for the same build id; replace() keeps concurrent builds safe
        tmp_src = src_dir / f".{s}.{os.getpid()}"
        tmp_src.write_bytes((lib_dir / s).read_bytes())
        tmp_src.replace(src_dir / s)
    env = dict(os.environ, PATH=f"{tools}{os.pathsep}{os.environ.get('PATH', '')}")

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
    meta = tomllib.loads((lib_dir / "meta.toml").read_text())
    lines = [
        "# Generated by tools/build_kernels.py from build.toml, meta.toml and the sources; do not edit.",
        f'name = "{name}"',
        f'build_id = "{build_id}"',
        f'toolkit = "{tk}"',
        f'sources_sha256 = "{src.hexdigest()}"  # build.toml + sources, checked by the tests',
        f'kind = "{cfg["kind"]}"',
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
    (lib_dir / "manifest.toml").write_text("\n".join(lines) + "\n")
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
    header = (cuda_include(cuda) / "crt" / "math_functions.h").read_text()
    patched = "_NV_RSQRT_SPECIFIER" in header
    if tuple(map(int, tk.split("."))) < (13, 1) and host_glibc(a.ccbin) >= (2, 42) and not patched:
        sys.exit(
            f"CUDA {tk} headers clash with glibc >= 2.42 (rsqrt): patch them (shared/patches/) or use a host compiler"
            " with an older sysroot"
        )
    for name in a.libs or sorted(p.parent.name for p in LIBS.glob("*/build.toml")):
        build_lib(name, cuda, a.cutlass, a.jobs, patched, a.ccbin)


if __name__ == "__main__":
    main()
