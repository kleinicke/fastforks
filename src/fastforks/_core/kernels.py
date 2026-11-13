"""Custom-kernel libraries: per-arch binaries found or downloaded by hash, loaded through the CUDA driver API.

A kernel library lives in `fastforks/kernels/<name>/`: the CUDA sources, `build.toml` (how to build) and
`manifest.toml` (written by `tools/build_kernels.py`: build id, toolkit, one binary per target with SHA-256).
Binaries are not in the wheel; they are looked up in `FASTFORKS_KERNEL_PATH`, then the user cache, then downloaded.

Target selection for a GPU of compute capability `X.Y` (all CPU-testable, see `select_target`):
`sm_XY` -> the highest `sm_XZ` with `Z < Y` (same major: binary compatible) -> the highest `compute_N` PTX with
`N <= XY` (JIT-compiled by the driver) -> `KernelUnavailableError`. Nothing ever falls back to another backend.

A library may also ship `manifest.compat.toml`: the same sources built with an older toolkit (its `manifest.toml`
renamed), loaded when the driver is too old for `manifest.toml` but not for it (`for_driver`).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import itertools
import logging
import math
import os
import platform
import re
import struct
import sys
import threading
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar

import torch

from fastforks._core.errors import CardError, KernelUnavailableError, WeightsError
from fastforks._core.files import FileSpec, cache_root, locate, read_caches, search_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

log = logging.getLogger(__name__)

KERNEL_PATH_ENV = "FASTFORKS_KERNEL_PATH"
_COMPAT = "manifest.compat.toml"
_TARGET = re.compile(r"(sm|compute)_(\d+)")


@dataclass(frozen=True)
class KernelManifest:
    """Binaries of one kernel library build.

    Attributes:
        name: Library name, e.g. `"s2m2"`.
        build_id: Hash of sources + build settings + toolkit; part of every file name.
        toolkit: CUDA toolkit `(major, minor)` the binaries were built with (the driver must support it).
        kind: `"cubin"` (driver-API module) or `"shared"` (a `.so` called through ctypes).
        files: Binary per target (`"sm_86"`, `"compute_80"`, ...).
        licence: SPDX expression of the sources (incl. headers compiled in, e.g. CUTLASS).
        symbols: Hash per exported function (code, what it calls, build settings); equal hashes across builds mean
            the function compiles to the same code, so a validation of it carries over.
        compat: Build of the same sources with an older toolkit (`manifest.compat.toml`), for older drivers.
    """

    name: str
    build_id: str
    toolkit: tuple[int, int]
    kind: Literal["cubin", "shared"]
    files: Mapping[str, FileSpec]
    licence: str = "unknown"
    symbols: Mapping[str, str] = field(default_factory=dict)
    compat: KernelManifest | None = None

    @property
    def tag(self) -> str:
        """`name@build_id`, recorded in every result's backend."""
        return f"{self.name}@{self.build_id}"


def read_manifest(path: Path) -> KernelManifest:
    """Read `manifest.toml`, with its `manifest.compat.toml` next to it as `compat` if there is one.

    Raises:
        CardError: Malformed manifest.
    """
    with path.open("rb") as f:
        d = tomllib.load(f)
    try:
        major, minor = (int(v) for v in d["toolkit"].split("."))
        files = {
            target: FileSpec(f["name"], f["sha256"], f["size"], d["url"].format(lib=d["name"], file=f["name"]))
            for target, f in d["files"].items()
        }
        kind = d["kind"]
        compat = path.with_name(_COMPAT)
        if kind not in ("cubin", "shared") or not all(_TARGET.fullmatch(t) for t in files):
            raise ValueError(kind)  # noqa: TRY301 - reported as a CardError below
        return KernelManifest(
            d["name"],
            d["build_id"],
            (major, minor),
            kind,
            files,
            d.get("licence", "unknown"),
            dict(d.get("symbols", {})),
            read_manifest(compat) if path.name != _COMPAT and compat.is_file() else None,
        )
    except (KeyError, ValueError, TypeError) as e:
        msg = f"{path}: malformed kernel manifest ({e!r})"
        raise CardError(msg) from e


def select_target(targets: Iterable[str], arch: tuple[int, int]) -> str | None:
    """Pick the binary for compute capability `arch` (rules in the module docstring); `None` if none fits."""
    sm = arch[0] * 10 + arch[1]
    sass: dict[int, str] = {}
    ptx: dict[int, str] = {}
    for t in targets:
        m = _TARGET.fullmatch(t)
        if m:
            (sass if m[1] == "sm" else ptx)[int(m[2])] = t
    if sm in sass:
        return sass[sm]
    same_major = [n for n in sass if n // 10 == arch[0] and n < sm]
    if same_major:
        return sass[max(same_major)]
    older_ptx = [n for n in ptx if n <= sm]
    return ptx[max(older_ptx)] if older_ptx else None


def _libcuda_path() -> str:
    """Absolute path of the driver library (`ctypes.util.find_library` knows ldconfig; no `LD_LIBRARY_PATH` search)."""
    return _system_lib(
        "libcuda.so.1",
        ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/lib/x86_64-linux-gnu", "/usr/lib/wsl/lib"),
    )


def _system_lib(name: str, dirs: Sequence[str]) -> str:
    """`dir/name` for the first of `dirs` that has it, else the bare `name` (the loader's default search)."""
    for d in dirs:
        if os.path.isfile(os.path.join(d, name)):  # noqa: PTH113, PTH118 - cheap, no Path objects at import
            return os.path.join(d, name)  # noqa: PTH118
    return name


@functools.cache
def _libcuda() -> ctypes.CDLL:
    lib = ctypes.CDLL(_libcuda_path())
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        *[ctypes.c_uint] * 7,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib


def _error_name(res: int) -> str:
    name = ctypes.c_char_p()
    if _libcuda().cuGetErrorName(res, ctypes.byref(name)) != 0 or name.value is None:
        return f"CUDA driver error {res}"
    return f"{name.value.decode()} ({res})"


def _check(res: int, what: str) -> None:
    """Raise on a failed driver call at run time (a launch failure is a bug or a sticky CUDA error, not a fallback)."""
    if res != 0:
        msg = f"{_error_name(res)} in {what}"
        raise RuntimeError(msg)


def _check_load(res: int, what: str, library: str, device: torch.device) -> None:
    """Raise `KernelUnavailableError` with arch / driver / torch CUDA context on a failed load-time driver call."""
    if res != 0:
        arch = "sm_{}{}".format(*torch.cuda.get_device_capability(device))
        drv = "{}.{}".format(*driver_version())
        reason = f"{_error_name(res)} in {what} on {arch}, driver CUDA {drv}, torch CUDA {torch.version.cuda}"
        raise KernelUnavailableError(library, reason)


def cuda_device(device: torch.device) -> torch.device:
    """`device` with an explicit index (`cuda` -> `cuda:<current>`), so caches and results name one GPU."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def half_rate_fp32_acc(device: torch.device) -> bool:
    """Whether `device` runs fp16 tensor-core math with fp32 accumulation at half rate (GeForce cards).

    The compute capability does not tell: sm_86, sm_89 and sm_120 each hold GeForce cards (half rate) and datacenter or
    pro cards (full rate), so the device name decides. `FASTFORKS_ACC=fp16` (half rate) or `fp32` (full rate)
    overrides it, for experiments.
    """
    env = os.environ.get("FASTFORKS_ACC", "").strip().lower()
    if env in {"fp16", "fp32"}:
        return env == "fp16"
    return "geforce" in torch.cuda.get_device_name(cuda_device(device)).lower()


def driver_version() -> tuple[int, int]:
    """CUDA version supported by the installed driver, e.g. `(13, 0)`."""
    v = ctypes.c_int()
    _check(_libcuda().cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
    return v.value // 1000, v.value % 1000 // 10


# Linux driver branch that first supports a CUDA toolkit (for the remedy in `binary`'s error).
_MIN_DRIVER = {(12, 8): 570, (12, 9): 575, (13, 0): 580}


def for_driver(manifest: KernelManifest, driver: tuple[int, int] | None = None) -> KernelManifest:
    """`manifest`, or its `compat` build if `driver` (default: the installed one) is too old only for `manifest`."""
    c = manifest.compat
    if c is None:
        return manifest
    if driver is None:
        try:
            driver = driver_version()
        except OSError:
            return manifest  # `binary` reports it
    if not c.toolkit <= driver < manifest.toolkit:
        return manifest
    need = _MIN_DRIVER.get(manifest.toolkit)
    faster = f" (driver >= {need} runs the faster CUDA {'.'.join(map(str, manifest.toolkit))} build)" if need else ""
    log.info("%s: driver supports CUDA %d.%d, using the CUDA %d.%d build%s", c.name, *driver, *c.toolkit, faster)
    return c


def binary(manifest: KernelManifest, device: torch.device) -> tuple[Path, str]:
    """Return the verified binary of `manifest` for `device` and its target.

    Raises:
        KernelUnavailableError: CPU device, ROCm or CPU-only torch, not Linux x86-64, no or too old NVIDIA driver,
            unsupported arch or missing binary;
            the message names the reason and the remedy.
    """
    name = manifest.name
    if device.type != "cuda":
        raise KernelUnavailableError(name, f"needs a CUDA device, got {device}")
    if torch.version.hip:
        raise KernelUnavailableError(name, f"the kernels are NVIDIA CUDA only; torch {torch.__version__} is ROCm/HIP")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise KernelUnavailableError(
            name, f"the kernels are built for Linux x86-64, not {sys.platform} {platform.machine()}"
        )
    if not torch.cuda.is_available():
        raise KernelUnavailableError(name, f"torch {torch.__version__} has no usable CUDA")
    try:
        drv = driver_version()
    except OSError as e:
        raise KernelUnavailableError(name, f"cannot load the NVIDIA driver library libcuda.so.1 ({e})") from e
    if drv < manifest.toolkit:
        need = ".".join(map(str, manifest.toolkit))
        driver = _MIN_DRIVER.get(manifest.toolkit)
        remedy = f"; update the NVIDIA driver to >= {driver}" if driver else ""
        msg = f"driver supports CUDA {drv[0]}.{drv[1]}, kernels need >= {need}{remedy}"
        raise KernelUnavailableError(name, msg)
    arch = torch.cuda.get_device_capability(device)
    target = select_target(manifest.files, arch)
    if target is None:
        have = ", ".join(sorted(manifest.files))
        raise KernelUnavailableError(name, f"no binary for sm_{arch[0]}{arch[1]} (built: {have})")
    try:
        hint = f"fastforks download MODEL --kernels --arch sm_{arch[0]}{arch[1]}"
        path = locate(manifest.files[target], group=name, env_var=KERNEL_PATH_ENV, hint=hint, burst=_call)
    except WeightsError as e:
        advice = "" if "offline mode" in str(e) else f" (run `fastforks download --kernels` or set {KERNEL_PATH_ENV})"
        raise KernelUnavailableError(name, f"{e}{advice}") from e
    if target.startswith("compute_"):
        log.warning("%s: no SASS for sm_%d%d, JIT-compiling %s PTX (slow first load)", name, *arch, target)
    return path, target


@functools.cache
def _own_group(gid: int) -> bool:
    """Whether `gid` is this user's primary group and has no other members (Ubuntu's user-private group)."""
    try:
        import grp  # noqa: PLC0415 - POSIX only (no such module on Windows)
        import pwd  # noqa: PLC0415

        if gid != os.getgid():
            return False
        members = grp.getgrgid(gid).gr_mem
    except (ImportError, AttributeError, KeyError):
        return False
    return set(members) <= {pwd.getpwuid(os.getuid()).pw_name}


def _group_ok(st: os.stat_result) -> bool:
    """No group write bit, or a group that only this user is in."""
    return not st.st_mode & 0o020 or _own_group(st.st_gid)


def _private(path: Path) -> bool:
    """Whether `path` and every directory from its cache / `FASTFORKS_KERNEL_PATH` root down are private.

    Private: owned by this user and not world-writable or writable by a group with other members (cheap `stat`s).
    Anyone who can write there already is this user; for a shared, foreign or writable location the binary is
    hashed on every load and loaded from the hashed bytes.
    """
    roots = {r.resolve() for r in (cache_root(), *read_caches(), *search_path(KERNEL_PATH_ENV))}
    uid = os.getuid()
    try:
        p = path.resolve()
        chain = [p]
        for d in p.parents:
            chain.append(d)
            if d in roots:
                break
        else:
            return False  # not below a known root
        return all(st.st_uid == uid and not st.st_mode & 0o002 and _group_ok(st) for st in map(os.stat, chain))
    except OSError:
        return False


def verified_binary(manifest: KernelManifest, device: torch.device) -> tuple[bytes | None, Path, str]:
    """`binary(manifest, device)` with its bytes: `(image, path, target)`.

    In a private location (`_private`) the stamp-verified file is trusted and `image` is `None` (the caller loads
    the path). Elsewhere the bytes are read once and hashed against the manifest on every load (stamps are not
    consulted), and those bytes are what gets loaded, so a swap after the check or a forged stamp cannot run other
    code.

    Raises:
        KernelUnavailableError: See `binary`; or the file does not match the manifest.
    """
    path, target = binary(manifest, device)
    if _private(path):
        return None, path, target
    spec = manifest.files[target]
    data = path.read_bytes()
    if len(data) != spec.size or hashlib.sha256(data).hexdigest() != spec.sha256:
        raise KernelUnavailableError(manifest.name, f"{path} does not match its manifest (size or SHA-256)")
    return data, path, target


# p: pointer (tensor, int address or None), i: int32, l: int64, f: float32
_STRUCT = {"p": "P", "i": "i", "l": "q", "f": "f"}


class CubinModule:
    """A cubin / PTX module loaded into the current context of one device; launches on torch's current stream.

    Attributes:
        tag: `name@build_id:target`, recorded in results.
    """

    def __init__(self, manifest: KernelManifest, device: torch.device, signatures: Mapping[str, str]) -> None:
        """Load the module.

        Args:
            manifest: The library's manifest (kind `"cubin"`).
            device: CUDA device; the module is bound to its primary context.
            signatures: Kernel name -> argument types (`p`, `i`, `l`, `f`).

        Raises:
            KernelUnavailableError: See `binary`.
        """
        manifest = for_driver(manifest)
        image, path, target = verified_binary(manifest, device)
        self.tag = f"{manifest.tag}:{target}"
        self.device = device
        self.signatures = dict(signatures)
        self._fmt = {k: "@" + "".join(_STRUCT[c] for c in s) for k, s in signatures.items()}
        self._smem: dict[str, int] = {}
        lib = _libcuda()
        with torch.cuda.device(device):
            torch.zeros(1, device=device)  # make torch's primary context current
            self._image = (
                image if image is not None else path.read_bytes()
            )  # the driver may keep pointers into the image
            module = ctypes.c_void_p()
            res = lib.cuModuleLoadData(ctypes.byref(module), self._image)
            _check_load(res, f"cuModuleLoadData {manifest.files[target].name}", manifest.name, device)
        self._funcs = {}
        for k in signatures:
            f = ctypes.c_void_p()
            res = lib.cuModuleGetFunction(ctypes.byref(f), module, k.encode())
            _check_load(res, f"cuModuleGetFunction {k}", manifest.name, device)
            self._funcs[k] = f

    def max_threads(self, kernel: str) -> int:
        """Largest block size `kernel` launches with on this device (its register use can keep it below 1024)."""
        v = ctypes.c_int()
        # CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        _check(_libcuda().cuFuncGetAttribute(ctypes.byref(v), 0, self._funcs[kernel]), f"cuFuncGetAttribute {kernel}")
        return v.value

    def launch(
        self,
        kernel: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        *args: torch.Tensor | float | None,
        smem: int = 0,
    ) -> None:
        """Launch `kernel` with `args` packed by its signature (tensors become their data pointers).

        Straight to the driver (~3 us). Under `torch.compile` / `torch.export` the whole kernels model is one opaque op
        (`fastforks.stereo._module`), so launches always run eagerly.
        """
        vals = [
            (a.data_ptr() if isinstance(a, torch.Tensor) else int(a or 0)) if c == "p" else a
            for c, a in zip(self.signatures[kernel], args, strict=True)
        ]
        buf = struct.pack(self._fmt[kernel], *vals)
        lib = _libcuda()
        func = self._funcs[kernel]
        if smem > 48 * 1024 and self._smem.get(kernel, 0) < smem:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _check(lib.cuFuncSetAttribute(func, 8, ctypes.c_int(smem)), f"cuFuncSetAttribute {kernel}")
            self._smem[kernel] = smem
        size = ctypes.c_size_t(len(buf))
        # CU_LAUNCH_PARAM_BUFFER_POINTER, buf, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END
        extra = (ctypes.c_void_p * 5)(
            1, ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p).value, 2, ctypes.addressof(size), 0
        )
        _check(lib.cuLaunchKernel(func, *grid, *block, smem, current_stream(), None, extra), f"launch {kernel}")


def current_stream() -> int:
    """Raw handle of torch's current CUDA stream (`torch.cuda.current_stream()` costs ~3 us more)."""
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice())  # noqa: SLF001


def linear(n: int, threads: int = 256) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """1-D launch for `n` grid-stride work items: `threads` per block, at most 65535 * 8 blocks."""
    return (min(max((n + threads - 1) // threads, 1), 65535 * 8), 1, 1), (threads, 1, 1)


def cubin_library(manifest: Path, signatures: Mapping[str, str]) -> Callable[[torch.device], CubinModule]:
    """The `load(device)` function of a cubin kernel library: one `CubinModule` per device, loaded on first use.

    Args:
        manifest: The library's `manifest.toml`.
        signatures: Kernel name -> argument types (`p`, `i`, `l`, `f`), matching the `extern "C"` kernels.
    """

    @functools.cache
    def _load(device: torch.device) -> CubinModule:
        return CubinModule(read_manifest(manifest), device, signatures)

    def load(device: torch.device) -> CubinModule:
        """Load the library on `device` (once per device).

        Raises:
            KernelUnavailableError: No usable binary for this GPU / driver (the message says why).
        """
        return _load(cuda_device(device))

    return load


# In-memory files of loaded images stay open: dlopen finds loaded objects by path, so a closed and reused
# `/proc/self/fd/N` would hand the next library the previous one's handle.
_IMAGE_FDS: list[int] = []


def _load_image(image: bytes, name: str) -> ctypes.CDLL:
    """`dlopen` the bytes `image` through an anonymous in-memory file (kept open for the process lifetime)."""
    if sys.platform != "linux":  # memfd_create is Linux-only (so are the CUDA kernel libraries)
        msg = f"in-memory kernel images need Linux, not {sys.platform}"
        raise OSError(msg)
    fd = os.memfd_create(name)
    try:
        with os.fdopen(os.dup(fd), "wb", closefd=True) as f:
            f.write(image)
        lib = ctypes.CDLL(f"/proc/self/fd/{fd}")
    except BaseException:
        os.close(fd)
        raise
    _IMAGE_FDS.append(fd)
    return lib


def shared_library(manifest: Path, device: torch.device) -> tuple[ctypes.CDLL, str]:
    """Load the `.so` of a shared kernel library for `device`; returns it and its tag `name@build_id:target`.

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver, or the library does not load.
    """
    m = for_driver(read_manifest(manifest))
    image, path, target = verified_binary(m, device)
    try:
        # a shared or writable location loads the verified bytes, not the file name (no check-to-load race)
        lib = ctypes.CDLL(str(path)) if image is None else _load_image(image, path.name)
    except OSError as e:
        raise KernelUnavailableError(m.name, f"cannot load {path.name}: {e}") from e
    return lib, f"{m.tag}:{target}"


_FALLBACKS: Counter[tuple[str, str]] = Counter()
_FALLBACK_LOCK = threading.Lock()


def note_fallback(library: str, what: str) -> None:
    """Count one call that a kernel library routed to its validated non-kernel path (e.g. an unsupported geometry).

    Such routing is part of a kernels backend's golden-checked numerics, not a silent fallback of the backend; the
    counts make it visible (`fallbacks()`), and the first occurrence of each case is logged.
    """
    with _FALLBACK_LOCK:
        _FALLBACKS[library, what] += 1
        first = _FALLBACKS[library, what] == 1
    if first:
        log.info("%s: %s runs without the custom kernel", library, what)


T = TypeVar("T")
_AUTO_FAILED: dict[tuple[object, str], KernelUnavailableError] = {}


def auto_kernels(load: Callable[[torch.device], T], device: torch.device, logger: logging.Logger) -> T | None:
    """`mode="auto"` of a per-call function: the kernel library `load(device)`, or None for the portable path.

    A library that cannot load is remembered for the process, so calls do not retry it (a missing binary costs an
    HTTP 404, ~120 ms, per try), and one WARNING says so: the portable path can be orders of magnitude slower
    (fundamental-matrix RANSAC ~400x), which an INFO hid.
    """
    key = load, str(device)
    if key in _AUTO_FAILED:
        return None
    try:
        return load(device)
    except KernelUnavailableError as e:
        with _FALLBACK_LOCK:
            first = _AUTO_FAILED.setdefault(key, e) is e
        if first:
            logger.warning(
                "%s kernels unavailable on %s (%s); mode='auto' runs the much slower portable path",
                e.library,
                device,
                e.reason,
            )
        return None


def fallbacks() -> dict[tuple[str, str], int]:
    """Counts of `note_fallback` cases since the last `reset_fallbacks()`, keyed by `(library, case)`."""
    with _FALLBACK_LOCK:
        return dict(_FALLBACKS)


def reset_fallbacks() -> None:
    """Clear the `fallbacks()` counts."""
    with _FALLBACK_LOCK:
        _FALLBACKS.clear()


_CALLS = itertools.count(1)
_call = 0


def begin_call() -> None:
    """Start one model call (`Model._session`): kernel libraries count per-shape setup work in model calls."""
    global _call  # noqa: PLW0603 - one process-wide counter
    _call = next(_CALLS)


def call_index() -> int:
    """The index of the current model call (0 before the first); see `begin_call`."""
    return _call


_SLOTS = 1024  # fixed: launch checks per fp32 buffer of a `checked` run (4 KB; more launches: one more buffer + sync)

# the output checks of the running `checked` call on this thread (a `_Checks`), or None
_CHECKS = threading.local()


class _Checks:
    """One run's launch checks: each launch's output sum in an fp32 slot, all read with one copy per buffer.

    Only the sum reduction runs on the device; `isfinite` / `stack` would load their kernels on the first call
    (+35 / +7 ms on an RTX 5060 Ti).
    """

    def __init__(self) -> None:
        self.buffers: list[torch.Tensor] = []
        self.used = 0  # slots used, over all buffers
        self.launches: list[tuple[int, Callable[[], Callable[[], None] | None], Callable[[], None]]] = []

    def add(
        self, output: torch.Tensor, widen: Callable[[], Callable[[], None] | None], refund: Callable[[], None]
    ) -> None:
        """Queue the sum of `output` (inf / NaN anywhere: a non-finite sum)."""
        if self.used == len(self.buffers) * _SLOTS or self.buffers[-1].device != output.device:
            self.used = len(self.buffers) * _SLOTS
            self.buffers.append(torch.empty(_SLOTS, device=output.device))
        slot = self.buffers[-1][self.used % _SLOTS]
        torch.sum(output, tuple(range(output.dim())), dtype=torch.float32, out=slot)
        self.launches.append((self.used, widen, refund))
        self.used += 1

    def finite(self) -> list[bool]:
        """Per launch: its output was finite (one host sync per buffer)."""
        ends = {n // _SLOTS: n % _SLOTS + 1 for n, _, _ in self.launches}
        values = {i: self.buffers[i][:k].tolist() for i, k in ends.items()}
        return [math.isfinite(values[n // _SLOTS][n % _SLOTS]) for n, _, _ in self.launches]


def checking() -> bool:
    """Whether a `checked` call runs on this thread (its launches record their checks with `defer_check`)."""
    return getattr(_CHECKS, "run", None) is not None


def defer_check(
    output: torch.Tensor, widen: Callable[[], Callable[[], None] | None], refund: Callable[[], None]
) -> None:
    """Record a launch's output check in the running `checked` call: it fails when `output` has an inf or NaN.

    `widen()` switches the launcher to a wider variant and returns how to undo that, or `None` if it has none left;
    `refund()` gives the launch back to the launcher's checked-launch budget when the call runs again.
    """
    _CHECKS.run.add(output, widen, refund)


def checked(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """`fn(*args, **kwargs)`, run again while a launch it recorded with `defer_check` failed and has a fallback.

    The model call's (`Model`'s kernels entry points) or a CUDA graph warm-up's: one host sync per run, only if a
    launch recorded a check. The first failed launch (later ones may only see its NaN) switches its launcher and the
    call runs again; a launcher with no wider variant left (non-finite inputs, not an overflow) gets its variant of
    the call's start back, and the output is returned as is. A nested call checks its own launches.
    """
    outer = getattr(_CHECKS, "run", None)
    undo: dict[Callable[[], Callable[[], None] | None], Callable[[], None]] = {}
    try:
        while True:
            _CHECKS.run = run = _Checks()
            out = fn(*args, **kwargs)
            if not run.launches:
                return out
            finite = run.finite()
            bad = next((w for (_, w, _), ok in zip(run.launches, finite, strict=True) if not ok), None)
            if bad is None:
                return out
            if (step := bad()) is None:
                if bad in undo:
                    undo[bad]()
                return out
            undo.setdefault(bad, step)
            for _, w, refund in run.launches:  # the rerun launches them again: only the returned run counts
                if w != bad:
                    refund()
    finally:
        _CHECKS.run = outer
