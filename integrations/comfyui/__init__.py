"""fastforks nodes for ComfyUI: dense prediction, segmentation, detection, pose, stereo, flow and tracking.

This pack (registry id `fastforks`) lists every installed fastforks model, those not cleared for commercial use tagged
`[non-commercial]` / `[licence unclear]`: installing it is the opt-in to download their weights, for its nodes' runs
(`registry.allow_noncommercial`; no environment variable). Using each model within its licence is your responsibility.
The commercial-only pack fastforks-commercial (integrations/comfyui-commercial) has the same nodes, from this folder's
implementation.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

try:
    import fastforks  # noqa: F401 - the library the nodes call
except ModuleNotFoundError as e:  # ComfyUI shows this message under the pack's IMPORT FAILED line
    if e.name != "fastforks":
        raise
    msg = (
        "The fastforks pack needs the fastforks package (fastforks-core with every model family) in ComfyUI's Python:"
        " run `python -m pip install -r requirements.txt` in this folder with that Python (ComfyUI portable on"
        " Windows: `..\\..\\..\\python_embeded\\python.exe -m pip install -r requirements.txt`)"
    )
    raise ImportError(msg) from e

IMPL = "fastforks_comfyui"  # the one copy of nodes.py / convert.py / faces.py both packs use (fastforks-commercial too)


def implementation(folder: Path) -> types.ModuleType:
    """`fastforks_comfyui.nodes` from `folder` (this pack's), or the copy another pack already imported."""
    if IMPL not in sys.modules:
        spec = importlib.machinery.ModuleSpec(IMPL, None, is_package=True)
        spec.submodule_search_locations = [str(folder)]
        sys.modules[IMPL] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{IMPL}.nodes")


nodes = implementation(Path(__file__).resolve().parent)
convert = nodes.convert
nodes.start()  # first: the cache folder, where `mappings` keeps the models' inputs
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = nodes.mappings()
nodes.log.info(nodes.licence_state())

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
