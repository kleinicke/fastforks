"""fastforks nodes for ComfyUI: dense prediction, segmentation, detection, pose, stereo, flow and tracking."""

try:
    import fastforks  # noqa: F401 - the library the nodes call
except ModuleNotFoundError as e:  # ComfyUI shows this message under the pack's IMPORT FAILED line
    if e.name != "fastforks":
        raise
    msg = (
        "comfyui-fastforks needs the fastforks library in ComfyUI's Python: run `python -m pip install -r"
        " requirements.txt` in this folder with that Python (ComfyUI portable on Windows: `..\\..\\..\\python_embeded"
        "\\python.exe -m pip install -r requirements.txt`)"
    )
    raise ImportError(msg) from e

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, install_unload_hook, setup_cache

install_unload_hook()
setup_cache()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
