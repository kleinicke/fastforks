"""ComfyUI nodes over the public fastforks API: `fastforks.load(id)` once, then call the model per image.

Every node takes a model id, a backend (`auto` / `portable` / `kernels`, plus `mlx` / `mlx portable` on a Mac with MLX)
and a dtype. Loaded models stay cached across
runs on ComfyUI's device. When ComfyUI (or a node loading another model) needs memory, the least recently used
ones go first (`make_room`); `_MAX_CACHED` only bounds how many are kept. ComfyUI's "Unload models" also empties the
cache. ComfyUI is imported lazily so the module imports without it (unit tests).

Two packs share this module (one copy, `fastforks_comfyui.nodes`): the default pack `fastforks` (this folder,
`mappings()`) lists every model, non-commercial ones tagged, and runs its nodes inside `registry.allow_noncommercial()`
(installing it is the opt-in to their weights); `fastforks-commercial` (integrations/comfyui-commercial,
`mappings(commercial_only=True)`) has the same node classes with only the models cleared for commercial use and never
runs another. Same class names: a workflow saved with one pack loads in the other.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import gc
import importlib.util
import logging
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

import fastforks
from fastforks import restoration
from fastforks._core import registry  # the licence gate's variable and the published ids (no public API yet)

from . import convert

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from fastforks.detection import DetectionModel
    from fastforks.pose import PoseModel

log = logging.getLogger("fastforks.comfyui")

MLX_BACKENDS = ["mlx", "mlx portable"]  # Apple silicon with the `mlx` package: the same net through MLX instead of MPS


def _has_mlx() -> bool:
    return sys.platform == "darwin" and importlib.util.find_spec("mlx") is not None


BACKENDS = ["auto", "portable", "kernels", *(MLX_BACKENDS if _has_mlx() else [])]
DTYPES = ["auto", "float32", "float16", "bfloat16"]
_TAGS = {"no": " [non-commercial]", "unclear": " [licence unclear]"}
_MAX_CACHED = 8  # memory decides first (`make_room`): a workflow cycling a few nodes reloads none
_FILE = "file: "  # dropdown prefix of a checkpoint file of ComfyUI's `models/upscale_models`
_files: dict[str, tuple[float, str | None]] = {}  # path -> (mtime, `restoration._api._checkpoint_family`)
_cache: OrderedDict[tuple[Any, ...], fastforks.Model] = OrderedDict()
FULL_PACK = "fastforks"  # registry id of the default pack (integrations/comfyui): every model, nc ones tagged
COMMERCIAL_PACK = "fastforks-commercial"  # registry id of the commercial-only pack (integrations/comfyui-commercial)
ONLY_FULL = (
    f"is only in the {FULL_PACK} pack (it is not cleared for commercial use, so {COMMERCIAL_PACK} does not offer"
    f" it): install {FULL_PACK} with ComfyUI-Manager or `comfy node install {FULL_PACK}` instead"
)
NC_NOTICE = (
    "Models marked [non-commercial] are for non-commercial / research use only (GPL code: its copyleft terms) and"
    " [licence unclear] ones are not cleared for commercial use; using each within its licence is your"
    " responsibility (`fastforks info MODEL`)."
)
_LICENCE_TIP = " " + NC_NOTICE
_COMMERCIAL_TIP = (
    " Only models cleared for commercial use (code and weights licence) are listed. `fastforks info MODEL` shows the"
    " licences."
)
_BACKEND_TIP = (
    "auto: our CUDA kernels where they are faster, else portable PyTorch. portable: plain PyTorch (any device)."
    " kernels: our CUDA kernels (NVIDIA on Linux or Windows x86-64; fails elsewhere)."
    " mlx (Apple silicon): Apple's MLX instead of MPS, with our Metal kernels where the port has them; often leaner,"
    " but every new input size compiles once, so its first call is slow. mlx portable: MLX without the kernels."
    " Models without an MLX port fail with a message."
)
_DTYPE_TIP = "auto: the validated dtype of the backend. Others are checked against the model card at load."


def labels(ids: list[str]) -> list[str]:
    """Dropdown entries: the ids, with a licence tag on those not cleared for commercial use."""
    tagged = {m: tag for status, tag in _TAGS.items() for m in fastforks.list_models(commercial=status)}  # ty: ignore[invalid-argument-type]
    return [m + tagged.get(m, "") for m in ids]


def model_id(label: str) -> str:
    """The model id of a dropdown entry (without its licence tag); file entries unchanged."""
    return label if label.startswith(_FILE) else label.split(" [", 1)[0]


def commercial(mid: str) -> bool:
    """Whether the installed model `mid` is cleared for commercial use (code and weights licence: `yes`)."""
    card, variant = registry.resolve_any(mid)
    return card.status(variant).commercial == "yes"


def model_input(node: type[_Node]) -> tuple[list[str], dict[str, Any]]:
    """The `model` dropdown of a node: the ids its pack runs (`node.ids()`), its default and a licence tooltip.

    Computed once per node class (`_model_input`), copies returned: ComfyUI calls `INPUT_TYPES` several times per node
    and run (validation, input fetch, caching), and listing the models takes ~11 ms (every card's licence status). The
    entries change only with the installed packages, which ComfyUI reads at start.
    """
    entries, opts = _model_input(node)
    return list(entries), dict(opts)


@functools.cache
def _model_input(node: type[_Node]) -> tuple[tuple[str, ...], dict[str, Any]]:
    ids = node.ids()
    entries = labels(ids)
    pick = next((e for e in entries if model_id(e) == node.DEFAULT), entries[0] if entries else "")
    task = node.TASK or ""
    missing = not_installed(task)
    if node.COMMERCIAL_ONLY:
        others = [*(m for m in node.ids(commercial_only=False) if not commercial(m)), *missing]
        tip = _COMMERCIAL_TIP
        if others:
            tip += f" {len(others)} more {task} models (e.g. {', '.join(others[:3])}) are only in the {FULL_PACK} pack."
    else:
        tip = _LICENCE_TIP
        if missing:
            tip += (
                f" Not installed: {len(missing)} more {task} models with non-commercial or GPL code (e.g."
                f" {', '.join(missing[:3])}); they need {NC_DIST} in ComfyUI's Python (this pack's requirements.txt)."
            )
    return tuple(entries), {"default": pick, "tooltip": "fastforks model id." + tip}


def common(node: type[_Node]) -> dict[str, Any]:
    """The `model`, `backend` and `dtype` inputs every node with a model has."""
    return {
        "model": model_input(node),
        "backend": (BACKENDS, {"default": "auto", "tooltip": _BACKEND_TIP}),
        "dtype": (DTYPES, {"default": "auto", "tooltip": _DTYPE_TIP}),
    }


_MODE_ARG = re.compile(r"\bmode=")


def backend_words(text: str) -> str:
    """A fastforks message in the node's words: the library's `mode=` argument is the node's `backend` input."""
    return _MODE_ARG.sub("backend=", text)


NC_ENV = registry.NONCOMMERCIAL_ENV  # the library's own switch for library users; the packs never set it
NC_EXTRA = "fastforks-core[nc]"  # the model families with non-commercial or GPL code (dists.toml `fastforks-nc`)
NC_DIST = "fastforks"  # the PyPI metapackage (fastforks-core[nc,libs]) the default pack installs
_SPEC = re.compile(r"fastforks-core\[[\w,-]+\]")
# set while a node of the commercial pack runs (`_Node.execute`): `load` refuses other models, errors name the pack
_COMMERCIAL_RUN: contextvars.ContextVar[bool] = contextvars.ContextVar("fastforks_comfyui_commercial", default=False)


def pip_words(specs: str, *, upgrade: bool = False) -> str:
    """How to pip-install `specs` into ComfyUI's own Python (portable and venv installs), then restart."""
    u = "-U " if upgrade else ""
    return (
        f"install it into ComfyUI's Python and restart ComfyUI: ComfyUI portable (Windows, in the"
        f" ComfyUI_windows_portable folder): `python_embeded\\python.exe -m pip install {u}{specs}`; a venv or conda"
        f" install: `python -m pip install {u}{specs}` with that environment active"
    )


def comfy_hint(error: fastforks.FastforksError, text: str, *, commercial_only: bool = False) -> str:
    """What to do in ComfyUI about a fastforks error with message `text` ("" if nothing to add).

    A `pip install` of fastforks extras gets the commands for ComfyUI's Python. In the commercial pack a model of
    `fastforks-core[nc]` or with non-commercial weights gets the fastforks pack instead (`ONLY_FULL`).
    """
    specs = list(dict.fromkeys(_SPEC.findall(text))) if "pip install" in text else []
    licence = isinstance(error, fastforks.LicenceError) and NC_ENV in text
    if commercial_only and (NC_EXTRA in specs or licence):
        return f"In ComfyUI: this model {ONLY_FULL}."
    if specs:
        quoted = " ".join(f'"{s}"' for s in specs)
        return f"In ComfyUI: {pip_words(quoted, upgrade='pip install -U' in text)}."
    return ""


def node_words(error: fastforks.FastforksError, text: str, *, commercial_only: bool = False) -> str:
    """A fastforks message in the node's words: `backend=` for `mode=`, then what to do in ComfyUI (if anything)."""
    hint = comfy_hint(error, text, commercial_only=commercial_only)
    words = backend_words(text)
    return f"{words}{'' if words.endswith('.') else '.'} {hint}" if hint else words


@contextlib.contextmanager
def node_errors() -> Generator[None]:
    """Re-raise fastforks errors (same type and traceback), their message in the node's words (`node_words`)."""
    try:
        yield
    except fastforks.FastforksError as e:
        if e.args and isinstance(e.args[0], str):
            e.args = (node_words(e, e.args[0], commercial_only=_COMMERCIAL_RUN.get()), *e.args[1:])
        raise


def not_installed(task: str | None = None) -> list[str]:
    """Published ids (of `task`, or all) whose package is not installed here, e.g. `fastforks-core[nc]` ones."""
    tasks = registry.TASKS if task is None else (task,)
    # the installed cards, not `list_models`: FASTFORKS_COMMERCIAL_ONLY hides installed models from that
    known = {m for t in tasks for m in registry._known_models(t)}  # noqa: SLF001 - no public list of published ids
    return sorted(known - {m for t in tasks for m in registry.cards(t)})


def licence_state(*, commercial_only: bool = False) -> str:
    """One line for the startup log: what the pack lists and downloads, and where the other models are."""
    missing = not_installed()
    if commercial_only:
        others = len([m for m in fastforks.list_models() if not commercial(m)]) + len(missing)
        line = f"fastforks: {COMMERCIAL_PACK} lists only models cleared for commercial use and downloads no"
        line += " non-commercial weights"
        return line + (f"; {others} other models are only in the {FULL_PACK} pack" if others else "")
    nc = len([m for m in fastforks.list_models() if not commercial(m)])
    line = (
        f"fastforks: {FULL_PACK} pack: {nc} of its models are not cleared for commercial use (tagged [non-commercial]"
        " or [licence unclear]); installing this pack is the opt-in to download their weights, for its nodes only"
        f" (not the environment). Using each model within its licence is your responsibility; {COMMERCIAL_PACK} is"
        " the pack with commercial models only"
    )
    if missing:
        spec = f'"{NC_DIST}"'
        line += (
            f". {len(missing)} models with non-commercial or GPL code (e.g. {', '.join(missing[:3])}) are not"
            f" installed: to add them, {pip_words(spec)}"
        )
    return line


def _only_full(mid: str) -> bool:
    """Whether `mid` is a model the commercial pack never offers: installed and not cleared, or of the nc package."""
    if registry.task_of(mid) is not None:
        return not commercial(mid)
    return any(registry._known_models(t).get(mid.lower()) == "fastforks-nc" for t in registry.TASKS)  # noqa: SLF001


def unlisted(label: str, node: type[_Node]) -> str:
    """Why a saved `model` value is not in a node's list (a workflow from another install), in the node's words."""
    mid = model_id(label)
    if mid.startswith(_FILE):
        return f"{mid!r}: not a file of models/upscale_models that fastforks reads (fastforks.restoration.load)"
    if node.COMMERCIAL_ONLY and _only_full(mid):
        return f"{mid!r} {ONLY_FULL}."
    try:
        fastforks.model_card(mid)
    except fastforks.FastforksError as e:  # not installed (names the package), or unknown (with suggestions)
        return node_words(e, str(e), commercial_only=node.COMMERCIAL_ONLY)
    return f"{mid!r} is not one of this node's {len(node.ids())} models (it needs inputs the node does not pass)"


def _mm() -> Any:  # noqa: ANN401 - comfy.model_management (untyped)
    import comfy.model_management as mm  # noqa: PLC0415 - ComfyUI is present only inside ComfyUI

    return mm


def _weights_bytes(mid: str) -> int:
    card = fastforks.model_card(mid)
    variant = next(v for v in card.variants if v.ref == card.variant_ref)
    return sum(f.size or 0 for f in variant.weights)


def upscale_files() -> list[str]:
    """Entries for the files of `models/upscale_models` that `restoration.load` runs (each checked once)."""
    import folder_paths  # noqa: PLC0415 - ComfyUI only

    entries = []
    for name in folder_paths.get_filename_list("upscale_models"):
        path = folder_paths.get_full_path("upscale_models", name)
        mtime = Path(path).stat().st_mtime
        if _files.get(path, (None,))[0] != mtime:
            _files[path] = (mtime, restoration._api._checkpoint_family(path))  # noqa: SLF001 - no public check yet
        if _files[path][1]:
            entries.append(_FILE + name)
    return entries


def _file(mid: str) -> Path | None:
    """The path of a file entry, or None for a model id."""
    if not mid.startswith(_FILE):
        return None
    import folder_paths  # noqa: PLC0415 - ComfyUI only

    return Path(folder_paths.get_full_path_or_raise("upscale_models", mid.removeprefix(_FILE)))


def load(label: str, backend: str, dtype: str, options: dict[str, Any] | None = None) -> fastforks.Model:
    """The cached model, or load it on ComfyUI's device after asking ComfyUI to free room for it."""
    mm = _mm()
    device = mm.get_torch_device()
    mid = model_id(label)
    file = _file(mid)
    version = file.stat().st_mtime if file else None  # a replaced file loads again
    key = (mid, version, backend, dtype, str(device), tuple(sorted((options or {}).items())))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    while len(_cache) >= _MAX_CACHED:
        _cache.popitem(last=False)
        gc.collect()
        mm.soft_empty_cache()
    kw: dict[str, Any] = {"mode": backend, "device": device, "dtype": None if dtype == "auto" else dtype}
    if backend in MLX_BACKENDS:
        kw.update(mode="portable" if backend.endswith("portable") else "auto", device="mlx")
    with node_errors():  # `_weights_bytes` already raises for an id that is not installed
        if not file and _COMMERCIAL_RUN.get() and _only_full(mid):  # e.g. a linked `model` input: not validated
            msg = f"{mid} {ONLY_FULL}."
            raise fastforks.LicenceError(msg)
        # weights + activations: ComfyUI unloads its own models, then `make_room` our least recently used ones, until
        # this much is free (no-op if it already is)
        mm.free_memory(2 * (file.stat().st_size if file else _weights_bytes(mid)) + 2**30, device)
        model = restoration.load(file, **kw) if file else fastforks.load(mid, **(options or {}), **kw)
    log.info("fastforks: loaded %s (%s)", mid, model.backend)
    _cache[key] = model
    return model


def setup_cache(environ: os._Environ[str] | dict[str, str] = os.environ) -> None:
    """Download weights and kernels into ComfyUI's `models/fastforks` (the first `fastforks` model folder).

    Files already in the HF hub or torch hub cache, the normal fastforks cache or other `fastforks` folders of
    `extra_model_paths.yaml` are used in place (`FASTFORKS_READ_CACHES`); only missing files are downloaded.
    """
    import folder_paths  # noqa: PLC0415 - ComfyUI only

    folder_paths.add_model_folder_path("fastforks", str(Path(folder_paths.models_dir) / "fastforks"))
    dirs = list(folder_paths.get_folder_paths("fastforks"))
    normal = environ.get("FASTFORKS_CACHE") or fastforks_cache_default()
    reads = [*dirs[1:], normal, *environ.get("FASTFORKS_READ_CACHES", "").split(os.pathsep)]
    environ["FASTFORKS_CACHE"] = dirs[0]
    seen, keep = {Path(dirs[0]).resolve()}, []
    for d in filter(None, reads):  # one entry per folder, however it is spelled (separators, `..`, links)
        if (p := Path(d).resolve()) not in seen and p.is_dir():
            seen.add(p)
            keep.append(d)
    environ["FASTFORKS_READ_CACHES"] = os.pathsep.join(keep)
    log.info("fastforks: downloads go to %s; also reading %s", dirs[0], keep)


def fastforks_cache_default() -> str:
    """The fastforks cache when `FASTFORKS_CACHE` is unset (the platform user cache)."""
    import platformdirs  # noqa: PLC0415 - fastforks dependency

    return platformdirs.user_cache_dir("fastforks")


def unload_all() -> None:
    """Drop every cached model and return their memory to the device."""
    if _cache:
        log.info("fastforks: unloading %d cached model(s)", len(_cache))
    _cache.clear()
    gc.collect()
    _mm().soft_empty_cache()


@functools.cache
def start() -> None:
    """Once per process, by whichever pack imports this module first: the unload hook and the cache folders."""
    install_unload_hook()
    setup_cache()


def install_unload_hook() -> None:
    """Hook our cache into ComfyUI's memory management. Idempotent.

    `unload_all_models` ("Unload models", `/free`) also drops our cache; `free_memory` (called before ComfyUI loads a
    model) also frees our models' memory when unloading its own left too little (`make_room`).
    """
    mm = _mm()
    original = mm.unload_all_models
    if original.__name__ != "_unload_all_models":  # not wrapped yet

        def _unload_all_models() -> None:
            original()
            unload_all()

        mm.unload_all_models = _unload_all_models
    free = mm.free_memory
    if free.__name__ != "_free_memory":

        def _free_memory(memory_required: float, device: torch.device | None, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - ComfyUI's list of unloaded models
            unloaded = free(memory_required, device, *args, **kwargs)
            make_room(memory_required, device)
            return unloaded

        mm.free_memory = _free_memory


def _device_free(device: torch.device) -> int:
    """Bytes free on the device itself, without PyTorch's cached blocks (ComfyUI's measure elsewhere)."""
    device = torch.device(device)
    return torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else _mm().get_free_memory(device)


def make_room(memory_required: float, device: torch.device | None) -> None:
    """Free our cached models' memory on `device` while there is less than `memory_required` free there.

    First their CUDA graphs (recaptured on later calls), then the models, least recently used first. ComfyUI's
    `get_free_memory` counts PyTorch's cached blocks as free, a graph's pool among them, though only that graph can
    use it: the graphs go whenever the device itself has too little free, measured by `_device_free`.
    """
    mm = _mm()

    def short() -> bool:
        return mm.get_free_memory(device) < memory_required

    if device is None:  # ComfyUI frees every device: nothing to measure
        return
    held = [m for m in _cache.values() if m.device == torch.device(device)]
    if not held or _device_free(device) >= memory_required:
        return
    for m in held:
        m.release_graphs()
    gc.collect()
    mm.soft_empty_cache()
    if not short():  # with the graphs gone, PyTorch's cached blocks are free for anyone
        log.info("fastforks: freed the CUDA graphs of %d cached model(s) for ComfyUI", len(held))
        return
    # ComfyUI's "free everything" (`/free`, `unload_all_models`) asks for 1e30 bytes
    need = "all memory" if memory_required >= 2**60 else f"{memory_required / 2**20:.0f} MiB"
    for key in [k for k, m in _cache.items() if m.device == torch.device(device)]:
        log.info("fastforks: unloading %s for ComfyUI (%s needed)", key[0], need)
        del _cache[key]
        gc.collect()
        mm.soft_empty_cache()
        if not short():
            return


def each(n: int, fn: Callable[[int], Any]) -> list[Any]:
    """`fn(i)` for every image of a batch, with ComfyUI's progress bar and interrupt check between images."""
    import comfy.utils  # noqa: PLC0415 - ComfyUI only

    mm = _mm()
    bar = comfy.utils.ProgressBar(n)
    out = []
    try:
        for i in range(n):
            mm.throw_exception_if_processing_interrupted()
            with node_errors():  # a kernel library can also fail at the first call
                out.append(fn(i))
            bar.update(1)
    except mm.InterruptProcessingException:
        out.clear()
        mm.soft_empty_cache()  # the model stays cached; only this run's tensors are freed
        raise
    return out


_BATCH_TIP = (
    "Images per model call: larger is faster for small models (several times at 224-512 px) and needs more GPU"
    " memory; 1 runs one image at a time. Models that cannot batch raise an error."
)
BATCH = {"batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": _BATCH_TIP})}
_PAIRS_TIP = (
    "Frame pairs per model call: the library still runs a large frame one pair at a time and models that gain nothing"
    " from a batch pair by pair, so this only speeds up small frames (rife-4.26 at 384 x 384: 9.1 -> 1.6 ms per"
    " pair); 1 runs one pair at a time, so a cancel stops after the current pair."
)
PAIR_BATCH = {"batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": _PAIRS_TIP})}


def at(t: Any, k: int | None) -> Any:  # noqa: ANN401 - a result field: tensor or None
    """Image `k` of a batched result field (`None`: the field of a one-image result, as it is)."""
    return t if k is None or t is None else t[k]


def each_result(
    image: torch.Tensor,
    m: fastforks.Model,
    batch_size: int,
    fn: Callable[[Any, int | None], Any],
    call: Callable[[torch.Tensor], Any] | None = None,
) -> list[Any]:
    """`fn(result, k)` for every image of a ComfyUI batch, the model called on `batch_size` images at a time.

    With `batch_size` 1 the model takes single images and `k` is None; else `k` is the image's index in the batched
    result (read its fields with `at`). `call` runs the model on the images (default: `m` itself).
    """
    n = image.shape[0]
    call = call or m
    if batch_size <= 1:
        return each(n, lambda i: fn(call(convert.frame(image, i, m.device)), None))
    starts = range(0, n, batch_size)

    def chunk(j: int) -> list[Any]:
        stop = min(n, starts[j] + batch_size)
        r = call(convert.frames(image, starts[j], stop, m.device))
        return [fn(r, k) for k in range(stop - starts[j])]

    return [o for c in each(len(starts), chunk) for o in c]


def each_pairs(
    frames: torch.Tensor, m: fastforks.Model, batch_size: int, fn: Callable[[torch.Tensor, torch.Tensor], list[Any]]
) -> list[Any]:
    """`fn(image1, image2)` on `batch_size` consecutive pairs of a ComfyUI batch at a time: frames `i` and `i + 1`.

    `fn` gets two `[k, 3, H, W]` batches (each frame uploaded once per chunk) and returns one entry per pair.
    """
    starts = range(0, frames.shape[0] - 1, batch_size)

    def chunk(j: int) -> list[Any]:
        stop = min(frames.shape[0] - 1, starts[j] + batch_size)
        x = convert.frames(frames, starts[j], stop + 1, m.device)
        return fn(x[:-1], x[1:])

    return [o for c in each(len(starts), chunk) for o in c]


def _out_device() -> torch.device:
    return _mm().intermediate_device()


_IMAGE = frozenset({"image"})


def takes(mid: str, given: frozenset[str] = _IMAGE, prompts: frozenset[str] = frozenset()) -> bool:
    """Whether a node that passes the inputs `given` (and the prompt kinds `prompts`) can run the model `mid`.

    From the model's contract (card and runner classes, no weights): every input it requires is in `given`, and a
    promptable model takes at least one of `prompts`. E.g. a depth model that needs a depth prompt, a matting model that
    needs a first-frame mask or an inpainting model that needs a mask cannot run in a node without that input.
    """
    inputs = fastforks.model_card(mid).contract().inputs
    if any(use == "required" and name not in given for name, use in inputs.items() if name != "prompts"):
        return False
    return "prompts" not in inputs or bool(set(inputs["prompts"]) & prompts)


class _Node:
    """A node over one task: its pack (`COMMERCIAL_ONLY` or not), the ids it lists and ComfyUI's entry `execute`."""

    CATEGORY = "fastforks"
    FUNCTION = "execute"
    DESCRIPTION = ""
    RETURN_TYPES: ClassVar[tuple[str, ...]] = ()
    TASK: ClassVar[str | None] = None  # the task of the listed models; None: the node takes no model (Warp)
    DEFAULT: ClassVar[str] = ""  # the default model id ("" or not listed in the pack: the first listed)
    COMMERCIAL_ONLY: ClassVar[bool] = False  # a node of fastforks-commercial (`_commercial_class`)
    TAKES: ClassVar[frozenset[str]] = _IMAGE  # the model inputs `run` passes (contract names)
    PROMPTS: ClassVar[frozenset[str]] = frozenset()  # promptable segmentation: the prompt kinds `run` passes
    run: Callable[..., tuple[Any, ...]]  # every node defines its own

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs (every node defines its own)."""
        raise NotImplementedError

    @classmethod
    def runs(cls, mid: str) -> bool:
        """Whether `run` gives the model `mid` every input it requires (`takes`)."""
        return takes(mid, cls.TAKES, cls.PROMPTS)

    @classmethod
    def lists(cls, mid: str) -> bool:  # noqa: ARG003 - overridden by the nodes that keep only some ids of the task
        """Whether the node lists the model `mid` of its task (licence aside)."""
        return True

    @classmethod
    def ids(cls, *, commercial_only: bool | None = None) -> list[str]:
        """The ids the node lists in its pack (`commercial_only`: in that pack instead)."""
        return list(cls._listed(commercial_only=commercial_only))

    @classmethod
    def _listed(cls, *, commercial_only: bool | None = None) -> Generator[str]:
        only = cls.COMMERCIAL_ONLY if commercial_only is None else commercial_only
        ids = fastforks.list_models(task=cls.TASK) if cls.TASK else []
        return (m for m in ids if cls.lists(m) and (not only or commercial(m)))

    @classmethod
    def VALIDATE_INPUTS(cls, model: str | None = None) -> bool | str:  # noqa: N802 - ComfyUI's name
        """ComfyUI's check of the `model` widget, in place of its "Value not in list".

        Accepts a listed entry, or the same id with another licence tag (workflows saved before a tag changed). For
        any other value (a workflow from another install) returns why: in fastforks-commercial a model only the
        fastforks pack offers, else an id that is not installed or unknown. `None`: linked or absent, checked when the
        node runs (`load`).
        """
        spec = cls.INPUT_TYPES()["required"].get("model")
        if model is None or spec is None:
            return True
        entries = spec[0]
        if model in entries or model_id(model) in {model_id(e) for e in entries}:
            return True
        return unlisted(model, cls)

    def execute(self, **inputs: Any) -> tuple[Any, ...]:  # noqa: ANN401 - ComfyUI's inputs
        """ComfyUI's entry: `run` in its pack's licence scope.

        In the fastforks pack inside `registry.allow_noncommercial()`: this call (no other code, not the environment)
        may download non-commercial weights, as installing the pack asks for. In fastforks-commercial `load` refuses
        any model not cleared for commercial use, and no such weights download.
        """
        token = _COMMERCIAL_RUN.set(self.COMMERCIAL_ONLY)
        try:
            with contextlib.nullcontext() if self.COMMERCIAL_ONLY else registry.allow_noncommercial():
                return self.run(**inputs)
        finally:
            _COMMERCIAL_RUN.reset(token)


_VIDEO_DEPTH = frozenset({"video_depth_anything"})  # depth families that run a whole clip (`model.video`), not frames


def _depth(mid: str, *, video: bool) -> bool:
    """Whether a depth model returns depth / inverse depth and runs as a clip (`video`) or per image."""
    card = fastforks.model_card(mid)
    return (
        bool({"depth", "inverse_depth"} & set(card.outputs))
        and (card.family in _VIDEO_DEPTH) == video
        and takes(mid)  # not the depth-prompted / depth-completion models: the node has no depth prompt input
    )


class Depth(_Node):
    """Relative or metric depth as a gray image (near = white), e.g. for depth ControlNets."""

    DESCRIPTION = "Depth map (inverse depth scaled to 0-1 per image, near = white)."
    RETURN_TYPES = ("IMAGE",)
    TASK = "depth"
    DEFAULT = "depth-anything-v2-s"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Image depth models (not the clip ones of Video depth)."""
        return _depth(mid, video=False)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"image": ("IMAGE",), **common(cls)}, "optional": BATCH}

    def run(
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, batch_size: int = 1
    ) -> tuple[torch.Tensor]:
        """Run the model on every image."""
        m = load(model, backend, dtype)
        maps = each_result(
            image, m, batch_size, lambda r, k: convert.depth_image(at(r.depth, k), at(r.inverse_depth, k))
        )
        return (convert.to_image(maps, _out_device()),)


class VideoDepth(_Node):
    """Depth of a frame batch as one clip (Video Depth Anything): consistent over time, one scale for the clip."""

    DESCRIPTION = "Depth maps of a frame batch run as one video (near = white; one scale for all frames, no flicker)."
    RETURN_TYPES = ("IMAGE",)
    TASK = "depth"
    DEFAULT = "video-depth-anything-s"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """The clip models (Video Depth Anything)."""
        return _depth(mid, video=True)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"frames": ("IMAGE",), **common(cls)}}

    def run(self, *, frames: torch.Tensor, model: str, backend: str, dtype: str) -> tuple[torch.Tensor]:
        """Run the model on the frames as one video."""
        m = load(model, backend, dtype)
        _mm().throw_exception_if_processing_interrupted()
        video = frames[..., :3].to(m.device, torch.float32).permute(0, 3, 1, 2)
        with node_errors():
            r = m.video(video)  # ty: ignore[unresolved-attribute]
        maps = convert.depth_image(r.depth, r.inverse_depth)  # [3, T, H, W]: scaled by the clip's min and max
        return (convert.to_image([maps[:, t] for t in range(maps.shape[1])], _out_device()),)


class Normals(_Node):
    """Surface normals as a normal-map image."""

    DESCRIPTION = "Surface normals as colours (red = right, green = up, blue = towards the camera)."
    RETURN_TYPES = ("IMAGE",)
    TASK = "depth"
    DEFAULT = "stablenormal-turbo"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Depth-task models with a normals output."""
        return "normals" in fastforks.model_card(mid).outputs and cls.runs(mid)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"image": ("IMAGE",), **common(cls)}, "optional": BATCH}

    def run(
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, batch_size: int = 1
    ) -> tuple[torch.Tensor]:
        """Run the model on every image."""
        # MoGe runs its normal head only when asked (its `normals` option is off by default); the others always have it
        wants = "normals" in fastforks.model_card(model_id(model)).options
        m = load(model, backend, dtype, {"normals": True} if wants else None)
        maps = each_result(image, m, batch_size, lambda r, k: convert.normals_image(at(r.normals, k)))
        return (convert.to_image(maps, _out_device()),)


class RemoveBackground(_Node):
    """Foreground alpha (background removal / matting)."""

    DESCRIPTION = "Foreground alpha: the image with alpha (RGBA) and the alpha as a mask (1 = foreground)."
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "alpha")
    TAKES = _IMAGE | {"trimap"}
    TASK = "matting"
    DEFAULT = "birefnet-lite"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Not video matting from a first-frame mask."""
        return cls.runs(mid)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = "Only for trimap models (ViTMatte): 0 = background, 1 = foreground, 0.5 = unknown."
        return {
            "required": {"image": ("IMAGE",), **common(cls)},
            "optional": {"trimap": ("MASK", {"tooltip": tip}), **BATCH},
        }

    def run(  # noqa: PLR0913 - the node's inputs
        self,
        *,
        image: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        trimap: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model on every image (`batch_size` per model call; with a trimap one at a time)."""
        m = load(model, backend, dtype)

        def one(i: int) -> torch.Tensor:
            tri = None if trimap is None else convert.from_mask(trimap, i, m.device)
            return m(convert.frame(image, i, m.device), trimap=tri).alpha

        if trimap is None and batch_size > 1:
            maps = each_result(image, m, batch_size, lambda r, k: at(r.alpha, k))
        else:
            maps = each(image.shape[0], one)
        alpha = convert.to_mask(maps, _out_device())
        return convert.with_alpha(image.to(alpha.device), alpha), alpha


class Segment(_Node):
    """Promptable segmentation (SAM family): points / box, or text for concept models (SAM 3)."""

    DESCRIPTION = "Mask from point / box prompts (SAM, SAM 2, ...) or a text prompt (SAM 3)."
    RETURN_TYPES = ("MASK",)
    PROMPTS = frozenset({"points", "boxes", "text"})
    TASK = "promptable_segmentation"
    DEFAULT = "sam2.1-hiera-small"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Not the reference-prompted (in-context) models."""
        return cls.runs(mid)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        px = "in pixels of the input image"
        return {
            "required": {
                "image": ("IMAGE",),
                **common(cls),
                "positive_points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' foreground points {px}"}),
                "negative_points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' background points {px}"}),
                "box": ("STRING", {"default": "", "tooltip": f"'x0,y0,x1,y1' {px}"}),
                "text": ("STRING", {"default": "", "tooltip": "Concept models (SAM 3) only: what to segment"}),
            }
        }

    def run(  # noqa: PLR0913 - one argument per input
        self,
        *,
        image: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        positive_points: str,
        negative_points: str,
        box: str,
        text: str,
    ) -> tuple[torch.Tensor]:
        """Segment every image with the same prompts."""
        m = load(model, backend, dtype)
        pos, neg, bx = convert.points(positive_points), convert.points(negative_points), convert.box(box)
        kwargs: dict[str, Any] = {}
        if text.strip():
            kwargs["text"] = text.strip()
        if pos or neg:
            kwargs |= {"input_points": pos + neg, "input_labels": [1] * len(pos) + [0] * len(neg)}
        if bx is not None:
            kwargs["input_boxes"] = bx
        if not kwargs:
            msg = "give positive_points, negative_points, box or (SAM 3) text"
            raise ValueError(msg)
        if "text" not in kwargs:
            kwargs["multimask_output"] = len(pos + neg) == 1 and bx is None  # SAM's advice for one ambiguous point

        def one(i: int) -> torch.Tensor:
            x = convert.frame(image, i, m.device)
            r = m(x, **kwargs)
            if "text" in kwargs:
                return convert.union_mask(r.masks, r.boxes, (x.shape[-2], x.shape[-1]))
            return r.masks[r.iou_scores.argmax()]

        return (convert.to_mask(each(image.shape[0], one), _out_device()),)


class Detect(_Node):
    """Object detection: a mask of the detections and their boxes as JSON."""

    DESCRIPTION = "Detections as one mask (instance masks or boxes) and as JSON (box, score, class_name per image)."
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "detections")
    TASK = "detection"
    DEFAULT = "rf-detr-s-coco"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = (
            "Comma-separated class names. Open-vocabulary models (Grounding DINO): what to detect (empty: the card's"
            " label set); others: keep only these classes (empty: all)."
        )
        return {
            "required": {
                "image": ("IMAGE",),
                **common(cls),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "classes": ("STRING", {"default": "", "tooltip": tip}),
            },
            "optional": BATCH,
        }

    def run(  # noqa: PLR0913 - one argument per input
        self,
        *,
        image: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        score_threshold: float,
        classes: str,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, str]:
        """Detect in every image."""
        m = load(model, backend, dtype)
        names = [c.strip() for c in classes.split(",") if c.strip()]
        text = names if cast("DetectionModel", m).open_vocabulary else None
        size = (image.shape[1], image.shape[2])  # BHWC

        def finish(r: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:  # noqa: ANN401 - a detection result
            if names and text is None:
                r = r[torch.tensor([d.get("class_name") in names for d in r.to_list()], dtype=torch.bool)]
            dets = [{k: v for k, v in d.items() if k != "mask"} for d in r.to_list()]
            return convert.union_mask(r.masks, r.boxes, size), dets

        # a batch: one network call where the family batches (DETR-style detectors), else one image at a time
        detect = functools.partial(m, score_threshold=score_threshold, text=text)
        out = each_result(image, m, batch_size, lambda r, k: finish(at(r, k)), call=detect)
        return convert.to_mask([o[0] for o in out], _out_device()), convert.to_json([o[1] for o in out])


class Pose(_Node):
    """Human (or animal) keypoints, drawn as skeletons on black (pose ControlNet input) and as JSON."""

    DESCRIPTION = (
        "Keypoints as a skeleton image and as JSON. Top-down models use the boxes of a Detect node (people; animals for"
        " AP-10K models)."
    )
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints")
    TASK = "pose"
    DEFAULT = "vitpose-plus-small"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = (
            "Top-down models: the 'detections' output of a Detect node (person boxes; animal boxes for AP-10K models;"
            " any boxes if none is one); empty: the whole image."
        )
        return {
            "required": {
                "image": ("IMAGE",),
                **common(cls),
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 64}),
            },
            "optional": {"detections": ("STRING", {"forceInput": True, "tooltip": tip}), **BATCH},
        }

    def run(  # noqa: PLR0913 - one argument per input
        self,
        *,
        image: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        keypoint_threshold: float,
        line_width: int,
        detections: str = "",
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, str]:
        """Estimate the people of every image (`batch_size` images per model call)."""
        m = load(model, backend, dtype)
        n, h, w = image.shape[0], image.shape[1], image.shape[2]

        def boxes(i: int) -> Any:  # noqa: ANN401 - box rows or None
            if not cast("PoseModel", m).needs_boxes:
                return None
            found = convert.subject_boxes(detections, i, m.variant.output_info["label_set"])
            return [[0.0, 0.0, float(w), float(h)]] if found is None else found

        def finish(r: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:  # noqa: ANN401 - a pose result
            drawn = convert.draw_pose(
                r.keypoints, r.keypoint_scores, r.skeleton, (h, w), threshold=keypoint_threshold, width=line_width
            )
            return drawn, r.to_list()

        if batch_size <= 1:
            out = each(n, lambda i: finish(m(convert.frame(image, i, m.device), boxes=boxes(i))))
        else:
            starts = range(0, n, batch_size)

            def chunk(j: int) -> list[tuple[torch.Tensor, list[dict[str, Any]]]]:
                idx = range(starts[j], min(n, starts[j] + batch_size))
                per = [boxes(i) for i in idx]
                rs = m(convert.frames(image, idx.start, idx.stop, m.device), boxes=None if per[0] is None else per)
                return [finish(r) for r in rs]

            out = [o for c in each(len(starts), chunk) for o in c]
        return convert.to_image([o[0] for o in out], _out_device()), convert.to_json([o[1] for o in out])


class Interpolate(_Node):
    """Frame interpolation (RIFE): `multiplier - 1` new frames between each pair of input frames."""

    DESCRIPTION = "Frame interpolation: inserts multiplier - 1 frames between consecutive frames of the batch."
    RETURN_TYPES = ("IMAGE",)
    TAKES = frozenset({"image1", "image2"})
    TASK = "interpolation"
    DEFAULT = "rife-4.26"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {
                "frames": ("IMAGE",),
                **common(cls),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
            },
            "optional": PAIR_BATCH,
        }

    def run(  # noqa: PLR0913 - the node's inputs
        self, *, frames: torch.Tensor, model: str, backend: str, dtype: str, multiplier: int, batch_size: int = 8
    ) -> tuple[torch.Tensor]:
        """Interpolate between every pair of consecutive frames (`batch_size` pairs per model call)."""
        m = load(model, backend, dtype)
        n = frames.shape[0]
        if n < 2:  # noqa: PLR2004 - a pair
            return (frames,)
        if batch_size > 1:

            def pairs_of(a: torch.Tensor, b: torch.Tensor) -> list[list[torch.Tensor]]:
                mids = [m(a, b, timestep=k / multiplier).frame for k in range(1, multiplier)]
                return [[a[i]] + [t[i] for t in mids] for i in range(len(a))]

            out = [f for pair in each_pairs(frames, m, batch_size, pairs_of) for f in pair]
            return (convert.to_image([*out, convert.frame(frames, n - 1, m.device)], _out_device()),)

        last: list[torch.Tensor] = []  # the previous pair's second frame: each frame is uploaded once

        def one(i: int) -> list[torch.Tensor]:
            a = last.pop() if last else convert.frame(frames, i, m.device)
            b = convert.frame(frames, i + 1, m.device)
            last.append(b)
            return [a] + [m(a, b, timestep=k / multiplier).frame for k in range(1, multiplier)]

        pairs = each(n - 1, one)
        out = [f for pair in pairs for f in pair] + last
        return (convert.to_image(out, _out_device()),)


class Classify(_Node):
    """Image classification: the top classes as text."""

    DESCRIPTION = "Top-k classes per image as 'name: probability' lines (images separated by a blank line)."
    RETURN_TYPES = ("STRING",)
    TASK = "classification"
    DEFAULT = "convnext_tiny.fb_in22k_ft_in1k"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {
                "image": ("IMAGE",),
                **common(cls),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
            "optional": BATCH,
        }

    def run(  # noqa: PLR0913 - one argument per input
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, top_k: int, batch_size: int = 1
    ) -> tuple[str]:
        """Classify every image."""
        m = load(model, backend, dtype)

        def one(r: Any, k: int | None) -> str:  # noqa: ANN401 - a classification result
            return "\n".join(f"{name}: {p:.4f}" for name, p in at(r.top(top_k), k))

        return ("\n\n".join(each_result(image, m, batch_size, one)),)


_WD_GENERAL, _WD_CHARACTERS = 4, 8110  # WD v3 label set: ratings, then general tags, then characters (card)


def wd_tags(probs: torch.Tensor, names: tuple[str, ...], general: float, character: float) -> str:
    """WD tagger probabilities -> the general and character tags at or above their thresholds, most likely first."""
    thr = torch.full_like(probs, float("inf"))
    thr[_WD_GENERAL:_WD_CHARACTERS], thr[_WD_CHARACTERS:] = general, character
    idx = (probs >= thr).nonzero().flatten()
    return ", ".join(names[j] for j in idx[probs[idx].argsort(descending=True)].tolist())


class Tag(_Node):
    """Danbooru tags of the WD v3 taggers as prompt text."""

    DESCRIPTION = (
        "Danbooru tags (WD v3 taggers): characters and general tags at or above their thresholds, most likely first,"
        " comma-separated (one line per image). Ratings are not included."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    TASK = "classification"
    DEFAULT = "wd-eva02-large-tagger-v3"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Multi-label (sigmoid) taggers."""
        return fastforks.model_card(mid).output_info.get("probs") == "sigmoid"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        thr = {"min": 0.0, "max": 1.0, "step": 0.01}
        return {
            "required": {
                "image": ("IMAGE",),
                **common(cls),
                "general_threshold": ("FLOAT", {"default": 0.35, **thr}),
                "character_threshold": ("FLOAT", {"default": 0.75, **thr}),
            },
            "optional": BATCH,
        }

    def run(  # noqa: PLR0913 - one argument per input
        self,
        *,
        image: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        general_threshold: float,
        character_threshold: float,
        batch_size: int = 1,
    ) -> tuple[str]:
        """Tag every image."""
        m = load(model, backend, dtype)
        names = cast("tuple[str, ...]", m.class_names)  # ty: ignore[unresolved-attribute]

        def one(r: Any, k: int | None) -> str:  # noqa: ANN401 - a classification result
            return wd_tags(at(r.probs, k).float().cpu(), names, general_threshold, character_threshold)

        return ("\n".join(each_result(image, m, batch_size, one)),)


class NsfwCheck(_Node):
    """Normal / nsfw classification (Falconsai NSFW ViT)."""

    DESCRIPTION = "The class per image (normal / nsfw, one line each) and the highest nsfw probability of the batch."
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("label", "nsfw_score")
    TASK = "classification"
    DEFAULT = "falconsai-nsfw-vit"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Normal / nsfw classifiers."""
        return fastforks.model_card(mid).output_info.get("label_set") == "nsfw-2"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"image": ("IMAGE",), **common(cls)}, "optional": BATCH}

    def run(
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, batch_size: int = 1
    ) -> tuple[str, float]:
        """Classify every image."""
        m = load(model, backend, dtype)
        names = cast("tuple[str, ...]", m.class_names)  # ty: ignore[unresolved-attribute]
        probs = each_result(image, m, batch_size, lambda r, k: at(r.probs, k).float().cpu())
        labels = "\n".join(names[int(p.argmax())] for p in probs)
        return labels, max(float(p[names.index("nsfw")]) for p in probs)


_PX = "in pixels of the first frame"


def _prompt_inputs() -> dict[str, Any]:
    return {
        "positive_points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' foreground points {_PX}"}),
        "negative_points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' background points {_PX}"}),
        "box": ("STRING", {"default": "", "tooltip": f"'x0,y0,x1,y1' {_PX}"}),
    }


class SemanticSegment(_Node):
    """Semantic segmentation: a class colour map, a mask of chosen classes and the classes present."""

    DESCRIPTION = (
        "Class per pixel: colour map (one colour per class), mask of the listed classes, classes present, and one mask"
        " per listed class (image-major: image 0's classes, then image 1's)."
    )
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("colour_map", "mask", "classes_found", "class_masks")
    TASK = "segmentation"
    DEFAULT = "segnext-mscan-t-ade20k"
    LABEL_SETS: ClassVar[frozenset[str]] = frozenset()  # the label sets of the listed models; empty: all

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Whether the node lists this segmentation id."""
        return not cls.LABEL_SETS or fastforks.model_card(mid).output_info.get("label_set") in cls.LABEL_SETS

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = "Comma-separated class names for the mask (e.g. 'sky, tree'); empty: an empty mask."
        return {
            "required": {
                "image": ("IMAGE",),
                **common(cls),
                "classes": ("STRING", {"default": "", "tooltip": tip}),
            },
            "optional": BATCH,
        }

    def run(  # noqa: PLR0913 - one argument per input
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, classes: str, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """Segment every image."""
        m = load(model, backend, dtype)
        names = cast("tuple[str, ...]", m.class_names)  # ty: ignore[unresolved-attribute]
        wanted = torch.tensor(convert.class_ids(classes, names), dtype=torch.long)
        colours = convert.palette(len(names))

        def one(r: Any, k: int | None) -> tuple[torch.Tensor, torch.Tensor, list[str], list[torch.Tensor]]:  # noqa: ANN401
            lab = at(r.labels, k)
            present = [names[c] for c in lab.unique().tolist()]
            mask = torch.isin(lab, wanted.to(lab.device))
            return convert.labels_image(lab, colours), mask, present, [lab == c for c in wanted.tolist()] or [mask]

        out = each_result(image, m, batch_size, one)
        found = "\n".join(", ".join(o[2]) for o in out)
        return (
            convert.to_image([o[0] for o in out], _out_device()),
            convert.to_mask([o[1] for o in out], _out_device()),
            found,
            convert.to_mask([c for o in out for c in o[3]], _out_device()),
        )


class Parse(SemanticSegment):
    """Clothes / face parsing (SegFormer b2 clothes, b5 face parsing): masks per body or face part."""

    DESCRIPTION = "Clothes / face parsing (SegFormer b2 clothes, b5 face parsing). " + SemanticSegment.DESCRIPTION
    DEFAULT = "segformer-b2-clothes"
    LABEL_SETS = frozenset({"atr-18", "celebamask-hq-19"})


_STEREO_CLIP = 0.01  # Stereo output: quantiles 0.01 / 0.99 map to black / white (occlusion borders, mismatches)


class Stereo(_Node):
    """Stereo matching: disparity of the left image from a rectified left / right pair."""

    DESCRIPTION = (
        "Disparity of rectified left / right pairs, scaled to 0-1 per image from its 1st to 99th percentile (near ="
        " white, as Depth; the outer 1 % saturate, so a few outlier pixels do not darken the map)."
    )
    RETURN_TYPES = ("IMAGE",)
    TAKES = frozenset({"left", "right"})
    TASK = "stereo"
    DEFAULT = "raft-stereo-middlebury"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"left": ("IMAGE",), "right": ("IMAGE",), **common(cls)}, "optional": BATCH}

    def run(  # noqa: PLR0913 - the node's inputs
        self, *, left: torch.Tensor, right: torch.Tensor, model: str, backend: str, dtype: str, batch_size: int = 1
    ) -> tuple[torch.Tensor]:
        """Match every left / right pair, `batch_size` pairs per model call."""
        if left.shape != right.shape:
            msg = f"left and right must be batches of one size, got {tuple(left.shape)} and {tuple(right.shape)}"
            raise ValueError(msg)
        m = load(model, backend, dtype)
        n, step = left.shape[0], max(1, batch_size)
        starts = range(0, n, step)

        def chunk(j: int) -> list[torch.Tensor]:
            a, b = starts[j], min(n, starts[j] + step)
            if step == 1:
                r = m(convert.frame(left, a, m.device), convert.frame(right, a, m.device))
                return [convert.depth_image(None, r.disparity, clip=_STEREO_CLIP)]  # disparity is an inverse depth
            r = m(convert.frames(left, a, b, m.device), convert.frames(right, a, b, m.device))
            return [convert.depth_image(None, d, clip=_STEREO_CLIP) for d in r.disparity]

        maps = [o for c in each(len(starts), chunk) for o in c]
        return (convert.to_image(maps, _out_device()),)


class Flow(_Node):
    """Optical flow between consecutive frames: colour-wheel images and the raw flow (for Warp)."""

    DESCRIPTION = "Flow from each frame to the next (B - 1 results): colour-wheel image and raw flow in pixels."
    RETURN_TYPES = ("IMAGE", "FASTFORKS_FLOW")
    RETURN_NAMES = ("flow_image", "flow")
    TAKES = frozenset({"image1", "image2"})
    TASK = "flow"
    DEFAULT = "sea-raft-s"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"frames": ("IMAGE",), **common(cls)}, "optional": PAIR_BATCH}

    def run(
        self, *, frames: torch.Tensor, model: str, backend: str, dtype: str, batch_size: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow of every pair of consecutive frames (`batch_size` pairs per model call)."""
        if frames.shape[0] < 2:  # noqa: PLR2004 - a pair
            msg = "Flow needs a batch of at least 2 frames"
            raise ValueError(msg)
        m = load(model, backend, dtype)
        if batch_size > 1:
            flows = each_pairs(frames, m, batch_size, lambda a, b: list(m(a, b).flow))
        else:
            flows = each(
                frames.shape[0] - 1,
                lambda i: m(convert.frame(frames, i, m.device), convert.frame(frames, i + 1, m.device)).flow,
            )
        raw = torch.stack([f.detach().float().permute(1, 2, 0) for f in flows]).to(_out_device()).contiguous()
        return convert.to_image([convert.flow_image(f) for f in flows], _out_device()), raw


class Warp(_Node):
    """Backward warp with a flow: each image sampled where the flow points (frame t + 1 -> frame t)."""

    DESCRIPTION = "Warps image i by flow i: frames[1:] with the Flow output reproduce frames[:-1] (black outside)."
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"image": ("IMAGE",), "flow": ("FASTFORKS_FLOW",)}}

    def run(self, *, image: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor]:
        """Warp the first len(flow) images."""
        n = flow.shape[0]
        if image.shape[0] < n or image.shape[1:3] != flow.shape[1:3]:
            msg = f"need at least {n} images of the flow size {tuple(flow.shape[1:3])}, got {tuple(image.shape)}"
            raise ValueError(msg)
        return (convert.warp(image[:n], flow).to(_out_device()),)


class VideoSegment(_Node):
    """Video object segmentation (SAM 2 video / SAMURAI): one object prompted on the first frame, tracked."""

    DESCRIPTION = "Mask of one object over a frame batch, prompted by points / box on the first frame."
    RETURN_TYPES = ("MASK",)
    PROMPTS = frozenset({"points", "boxes"})
    TASK = "promptable_segmentation"
    DEFAULT = "sam2.1-hiera-small"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """The SAM 2 video models (they take the samurai option)."""
        return "samurai" in fastforks.model_card(mid).options

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = "SAMURAI: motion-aware memory (Kalman filter), steadier on fast motion and occlusions; best with a box."
        return {
            "required": {
                "frames": ("IMAGE",),
                **common(cls),
                **_prompt_inputs(),
                "samurai": ("BOOLEAN", {"default": False, "tooltip": tip}),
            }
        }

    def run(  # noqa: PLR0913 - one argument per input
        self,
        *,
        frames: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        positive_points: str,
        negative_points: str,
        box: str,
        samurai: bool,
    ) -> tuple[torch.Tensor]:
        """Track the prompted object through the frames."""
        pos, neg, bx = convert.points(positive_points), convert.points(negative_points), convert.box(box)
        kwargs: dict[str, Any] = {}
        if pos or neg:
            kwargs |= {"input_points": pos + neg, "input_labels": [1] * len(pos) + [0] * len(neg)}
        if bx is not None:
            kwargs["input_boxes"] = bx
        if not kwargs:
            msg = "give positive_points, negative_points or box (on the first frame)"
            raise ValueError(msg)
        m = load(model, backend, dtype, {"samurai": True} if samurai else None)
        _mm().throw_exception_if_processing_interrupted()
        video = frames[..., :3].to(m.device, torch.float32).permute(0, 3, 1, 2)
        with node_errors():
            masks = m.video(video, **kwargs).masks  # ty: ignore[unresolved-attribute]
        return (convert.to_mask(list(masks), _out_device()),)


class PointTrack(_Node):
    """Point tracking: points of the first frame followed through the frames, drawn on them and as JSON."""

    DESCRIPTION = "Tracks of a point grid (or given points) of the first frame, drawn on the frames and as JSON."
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tracks_image", "tracks")
    TAKES = frozenset({"video", "queries"})
    TASK = "tracking"
    DEFAULT = "tapir"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {
                "frames": ("IMAGE",),
                **common(cls),
                "grid_size": ("INT", {"default": 10, "min": 1, "max": 64, "tooltip": "n x n points if no points"}),
                "points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' points to track {_PX}"}),
                "trail": ("INT", {"default": 8, "min": 0, "max": 1000, "tooltip": "frames of track drawn per point"}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 32}),
            }
        }

    def run(  # noqa: PLR0913 - one argument per input
        self,
        *,
        frames: torch.Tensor,
        model: str,
        backend: str,
        dtype: str,
        grid_size: int,
        points: str,
        trail: int,
        line_width: int,
    ) -> tuple[torch.Tensor, str]:
        """Track the points through the frames."""
        m = load(model, backend, dtype)
        pts = convert.points(points) or convert.grid_points(grid_size, (frames.shape[1], frames.shape[2]))
        queries = torch.tensor([[0.0, x, y] for x, y in pts])
        _mm().throw_exception_if_processing_interrupted()
        video = frames[..., :3].to(m.device, torch.float32).permute(0, 3, 1, 2)
        with node_errors():
            r = m(video, queries)
        tracks = r.tracks.float().cpu()
        visible = r.visible.cpu() if r.visibility is not None else torch.ones(tracks.shape[:2], dtype=torch.bool)
        drawn = each(
            frames.shape[0],
            lambda t: convert.draw_tracks(
                convert.frame(frames, t, torch.device("cpu")), tracks, visible, t, trail=trail, width=line_width
            ),
        )
        return convert.to_image(drawn, _out_device()), convert.to_json({"tracks": tracks, "visible": visible})


_FACE_FAMILIES = frozenset({"codeformer", "gfpgan"})  # aligned face crops only: the FaceRestore node, not Upscale


def _face(mid: str) -> bool:
    return fastforks.model_card(mid).family in _FACE_FAMILIES


class Upscale(_Node):
    """Super-resolution / restoration (Real-ESRGAN, SPAN, SwinIR, HAT, DAT, ...), tiled for large images."""

    DESCRIPTION = (
        "Upscaled / restored image. Models: fastforks ids (Real-ESRGAN, SPAN, SwinIR, HAT, DAT, ...), or files of"
        " models/upscale_models (file: ...) whose architecture fastforks reads (fastforks.restoration.load(path):"
        " ESRGAN / Real-ESRGAN, Compact, SPAN, SwinIR, HAT, DAT, ...); other files are not listed. Keep tiling on for"
        " HAT and DAT: one pass runs out of memory at 2048 x 1024 on 16 GB."
    )
    RETURN_TYPES = ("IMAGE",)
    TASK = "restoration"
    DEFAULT = "realesrgan-x4plus"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """Not the face models (FaceRestore) nor the inpainting ones (they need a mask)."""
        return not _face(mid) and cls.runs(mid)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        inputs = common(cls)
        entries, opts = inputs["model"]  # and your own files of models/upscale_models (both packs)
        inputs["model"] = ([*entries, *upscale_files()], opts)
        tip = "Tile size in input pixels (overlap 32 px, as ComfyUI's upscaler): less memory. 0: one pass."
        return {
            "required": {
                "image": ("IMAGE",),
                **inputs,
                "tile": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 8, "tooltip": tip}),
            }
        }

    def run(self, *, image: torch.Tensor, model: str, backend: str, dtype: str, tile: int) -> tuple[torch.Tensor]:
        """Run the model on every image."""
        m = cast("restoration.RestorationModel", load(model, backend, dtype))
        maps = each(image.shape[0], lambda i: m(convert.frame(image, i, m.device), tile=tile or None).image)
        return (convert.to_image(maps, _out_device()),)


class FaceRestore(_Node):
    """Face restoration (GFPGAN, CodeFormer) of aligned face crops."""

    DESCRIPTION = (
        "Face restoration: GFPGAN v1.4, or CodeFormer (fidelity 0.5; [non-commercial], not in"
        " fastforks-commercial). Input: aligned 512 x 512 face crops (FFHQ alignment) only; this pack does not detect,"
        " align or paste faces back."
    )
    RETURN_TYPES = ("IMAGE",)
    TASK = "restoration"
    DEFAULT = "codeformer"

    @classmethod
    def lists(cls, mid: str) -> bool:
        """The aligned-face models."""
        return _face(mid)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"image": ("IMAGE",), **common(cls)}, "optional": BATCH}

    def run(
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, batch_size: int = 1
    ) -> tuple[torch.Tensor]:
        """Restore every face crop."""
        m = load(model, backend, dtype)
        return (convert.to_image(each_result(image, m, batch_size, lambda r, k: at(r.image, k)), _out_device()),)


# (key, node, display name): ComfyUI's class name is `Fastforks<key>` in both packs (a workflow loads in either; with
# both installed only the fastforks pack registers them, see integrations/comfyui-commercial); `{pack}` is "fastforks"
# or "fastforks, commercial".
_NODES: tuple[tuple[str, type[_Node], str], ...] = (
    ("Depth", Depth, "Depth ({pack})"),
    ("VideoDepth", VideoDepth, "Video depth ({pack})"),
    ("Normals", Normals, "Normals ({pack})"),
    ("RemoveBackground", RemoveBackground, "Remove background ({pack})"),
    ("Segment", Segment, "Segment (SAM, {pack})"),
    ("Detect", Detect, "Detect objects ({pack})"),
    ("Pose", Pose, "Pose ({pack})"),
    ("Interpolate", Interpolate, "Interpolate frames ({pack})"),
    ("Classify", Classify, "Classify ({pack})"),
    ("Tag", Tag, "Tag (WD tagger, {pack})"),
    ("NsfwCheck", NsfwCheck, "NSFW check ({pack})"),
    ("SemanticSegment", SemanticSegment, "Semantic segmentation ({pack})"),
    ("Parse", Parse, "Parse clothes / face ({pack})"),
    ("Stereo", Stereo, "Stereo disparity ({pack})"),
    ("Flow", Flow, "Optical flow ({pack})"),
    ("Warp", Warp, "Warp by flow ({pack})"),
    ("VideoSegment", VideoSegment, "Video segment (SAM 2 / SAMURAI, {pack})"),
    ("PointTrack", PointTrack, "Track points ({pack})"),
    ("Upscale", Upscale, "Upscale / restore ({pack})"),
    ("FaceRestore", FaceRestore, "Restore aligned faces (GFPGAN / CodeFormer, {pack})"),
)


@functools.cache
def _pack_class(node: type[_Node], *, commercial_only: bool) -> type[_Node]:
    """The node as a pack registers it.

    fastforks-commercial's lists only commercial models; the fastforks pack's says in its description that some of its
    models are non-commercial (if it lists one).
    """
    attrs: dict[str, Any] = {"__doc__": node.__doc__, "__module__": __name__}
    if commercial_only:
        attrs["COMMERCIAL_ONLY"] = True
    elif any(not commercial(m) for m in node._listed(commercial_only=False)):  # noqa: SLF001 - own class
        attrs["DESCRIPTION"] = f"{node.DESCRIPTION} {NC_NOTICE}"
    else:
        return node
    return type(node.__name__, (node,), attrs)


@functools.cache
def mappings(*, commercial_only: bool = False) -> tuple[dict[str, type[_Node]], dict[str, str]]:
    """ComfyUI's NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS of a pack (`commercial_only`: fastforks-commercial).

    A pack has the nodes that list at least one model there (and Warp, which takes no model): in
    fastforks-commercial not Parse, whose models are all non-commercial. Model files of models/upscale_models are not
    counted (they are listed when ComfyUI asks).
    """
    classes: dict[str, type[_Node]] = {}
    names: dict[str, str] = {}
    for key, node, title in _NODES:
        if node.TASK is not None and next(node._listed(commercial_only=commercial_only), None) is None:  # noqa: SLF001
            continue
        classes[f"Fastforks{key}"] = _pack_class(node, commercial_only=commercial_only)
        names[f"Fastforks{key}"] = title.format(pack="fastforks, commercial" if commercial_only else "fastforks")
    return classes, names


README_START = "<!-- models: generated by tools/gen_comfyui_readme.py from the registry, do not edit -->"
README_END = "<!-- end of models -->"


def readme_models(*, commercial_only: bool = False) -> str:
    """The README's list of a pack's nodes and the models each lists (by family), between `README_START` / `README_END`.

    From the installed cards, as the dropdowns: tools/gen_comfyui_readme.py writes it, tests/test_comfyui.py checks it.
    """
    classes, _ = mappings(commercial_only=commercial_only)
    lines = [README_START, ""]
    for key, node, title in _NODES:
        if f"Fastforks{key}" not in classes:
            continue
        name = title.replace(", {pack})", ")").replace(" ({pack})", "")
        ids = classes[f"Fastforks{key}"].ids()
        if not ids:
            lines.append(f"- **{name}**: no model (it applies a flow field)")
            continue
        families: dict[str, list[str]] = {}
        for entry in labels(ids):
            families.setdefault(registry.resolve_any(model_id(entry))[0].name, []).append(f"`{entry}`")
        own = " (and your own `.pth` / `.safetensors` files in models/upscale_models)" if node is Upscale else ""
        lines.append(f"- **{name}**{own}:")
        lines += [f"  - {family}: {', '.join(entries)}" for family, entries in families.items()]
    return "\n".join([*lines, "", README_END])
