"""fastforks nodes for ComfyUI: dense prediction, segmentation, detection, pose, stereo, flow and tracking."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, install_unload_hook, setup_cache

install_unload_hook()
setup_cache()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
