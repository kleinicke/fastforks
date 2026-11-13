"""fastforks nodes for ComfyUI: depth, normals, background removal, SAM, detection, pose, frame interpolation."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, install_unload_hook

install_unload_hook()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
