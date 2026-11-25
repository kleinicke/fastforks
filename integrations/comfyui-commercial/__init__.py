"""fastforks-commercial: the fastforks nodes with only the models cleared for commercial use.

The same nodes as the fastforks pack (integrations/comfyui; same class names, so workflows load in either), but every
dropdown lists only models whose code and weights licences allow commercial use, and no node downloads non-commercial
weights. A workflow naming another model fails with a one-sentence error: that model is only in the fastforks pack.

With the fastforks pack installed next to this one (in ComfyUI's custom_nodes), this pack registers nothing and the
fastforks pack's nodes are used: the class names are the same, and that pack is the one that was asked for every model.

The nodes are the fastforks pack's implementation (one copy, `fastforks_comfyui`): bundled in this folder
(`fastforks_comfyui/`, copied in before `comfy node publish`, see README.md) or, in a fastforks clone, the folder
integrations/comfyui next to this one.
"""

# ruff: noqa: N999 - ComfyUI pack folders are named like the pack (comfyui-commercial), not like a module

import importlib
import importlib.machinery
import importlib.util
import logging
import sys
import tomllib
import types
from pathlib import Path

PACK = "fastforks-commercial"
FULL = "fastforks"  # the registry id of the pack with every model; its nodes win when both are installed
IMPL = "fastforks_comfyui"
BUNDLED = "fastforks_comfyui"  # the implementation copied into this folder for the registry (gitignored)

log = logging.getLogger("fastforks.comfyui")

try:
    import fastforks  # noqa: F401 - the library the nodes call
except ModuleNotFoundError as e:  # ComfyUI shows this message under the pack's IMPORT FAILED line
    if e.name != "fastforks":
        raise
    msg = (
        f"{PACK} needs fastforks-core in ComfyUI's Python: run `python -m pip install -r requirements.txt` in this"
        " folder with that Python (ComfyUI portable on Windows: `..\\..\\..\\python_embeded\\python.exe -m pip install"
        " -r requirements.txt`)"
    )
    raise ImportError(msg) from e


def _name(folder: Path) -> str | None:
    """The `[project] name` of a folder's pyproject.toml, if any."""
    try:
        return tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return None


def full_pack(here: Path) -> Path | None:
    """The fastforks pack's folder if it is installed next to this one in ComfyUI's custom_nodes (else `None`).

    Only a folder named custom_nodes counts (not a fastforks clone's integrations/), and not disabled packs
    (ComfyUI-Manager renames those to `*.disabled`).
    """
    custom = here.parent  # not resolved: this folder may be a link into a clone
    if custom.name != "custom_nodes":
        return None
    for folder in sorted(custom.iterdir()):
        other = folder.name != here.name and not folder.name.endswith(".disabled")
        if other and (folder / "nodes.py").is_file() and _name(folder) == FULL and _comfyui_loads(folder.name):
            return folder
    return None


def _comfyui_loads(folder: str) -> bool:
    """Whether ComfyUI loads the custom node folder `folder`.

    Not under `--disable-all-custom-nodes` unless it is whitelisted: the rule of ComfyUI's `load_custom_nodes`.
    Outside ComfyUI (tests) every folder counts as loaded.
    """
    try:
        from comfy.cli_args import args  # noqa: PLC0415 - ComfyUI only
    except ImportError:
        return True
    return not getattr(args, "disable_all_custom_nodes", False) or folder in (
        getattr(args, "whitelist_custom_nodes", None) or []
    )


def implementation(here: Path) -> types.ModuleType:
    """`fastforks_comfyui.nodes`: the copy already imported, else the bundled one, else a clone's integrations/comfyui.

    Raises:
        ImportError: None of them exists (a broken install); the message says what is missing.
    """
    if IMPL not in sys.modules:
        found = [f for f in (here / BUNDLED, here.resolve().parent / "comfyui") if (f / "nodes.py").is_file()]
        if not found:
            msg = (
                f"{PACK} has no nodes: neither {here / BUNDLED} nor a fastforks clone's integrations/comfyui next to"
                f" it has nodes.py; reinstall {PACK} (ComfyUI-Manager or `comfy node install {PACK}`)"
            )
            raise ImportError(msg)
        spec = importlib.machinery.ModuleSpec(IMPL, None, is_package=True)
        spec.submodule_search_locations = [str(found[0])]
        sys.modules[IMPL] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{IMPL}.nodes")


_here = Path(__file__).absolute().parent
_full = full_pack(_here)
if _full is not None:
    NODE_CLASS_MAPPINGS: dict[str, type] = {}
    NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
    log.info(
        "fastforks: %s registers no nodes: the %s pack (%s) is installed and provides the same nodes with every model;"
        " remove or disable one of the two to choose",
        PACK,
        FULL,
        _full.name,
    )
else:
    nodes = implementation(_here)
    nodes.start()  # first: the cache folder, where `mappings` keeps the models' inputs
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = nodes.mappings(commercial_only=True)
    nodes.log.info(nodes.licence_state(commercial_only=True))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
