"""ComfyUI nodes over the public fastforks API: `fastforks.load(id)` once, then call the model per image.

Every node takes a model id, a backend (`auto` / `portable` / `kernels`) and a dtype. Loaded models stay cached across
runs (at most `_MAX_CACHED`, least recently used dropped first) on ComfyUI's device; ComfyUI's "Unload models" also
empties the cache. ComfyUI is imported lazily so the module imports without it (unit tests).
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

import fastforks

from . import convert

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.detection import DetectionModel
    from fastforks.pose import PoseModel

log = logging.getLogger("fastforks.comfyui")

BACKENDS = ["auto", "portable", "kernels"]
DTYPES = ["auto", "float32", "float16", "bfloat16"]
_TAGS = {"no": " [non-commercial]", "unclear": " [licence unclear]"}
_MAX_CACHED = 2
_cache: OrderedDict[tuple[str, str, str, str], fastforks.Model] = OrderedDict()
_LICENCE_TIP = (
    " Ids marked [non-commercial] have a non-commercial code or weights licence (research use only; their weights"
    " download only with the environment variable FASTFORKS_NONCOMMERCIAL=1); [licence unclear] means not cleared"
    " for commercial use. `fastforks info MODEL` shows the licences."
)
_BACKEND_TIP = (
    "auto: our CUDA kernels where they are faster, else portable PyTorch. portable: plain PyTorch (any device)."
    " kernels: our CUDA kernels (Linux x86-64 + NVIDIA only; fails elsewhere)."
)
_DTYPE_TIP = "auto: the validated dtype of the backend. Others are checked against the model card at load."


def labels(ids: list[str]) -> list[str]:
    """Dropdown entries: the ids, with a licence tag on those not cleared for commercial use."""
    tagged = {m: tag for status, tag in _TAGS.items() for m in fastforks.list_models(commercial=status)}  # ty: ignore[invalid-argument-type]
    return [m + tagged.get(m, "") for m in ids]


def model_id(label: str) -> str:
    """The model id of a dropdown entry (without its licence tag)."""
    return label.split(" [", 1)[0]


def model_input(ids: list[str], default: str) -> tuple[list[str], dict[str, Any]]:
    """The `model` dropdown of a node."""
    entries = labels(ids)
    pick = next((e for e in entries if model_id(e) == default), entries[0] if entries else "")
    return entries, {"default": pick, "tooltip": "fastforks model id." + _LICENCE_TIP}


def common(ids: list[str], default: str) -> dict[str, Any]:
    """The `model`, `backend` and `dtype` inputs every node has."""
    return {
        "model": model_input(ids, default),
        "backend": (BACKENDS, {"default": "auto", "tooltip": _BACKEND_TIP}),
        "dtype": (DTYPES, {"default": "auto", "tooltip": _DTYPE_TIP}),
    }


def _mm() -> Any:  # noqa: ANN401 - comfy.model_management (untyped)
    import comfy.model_management as mm  # noqa: PLC0415 - ComfyUI is present only inside ComfyUI

    return mm


def _weights_bytes(mid: str) -> int:
    card = fastforks.model_card(mid)
    variant = next(v for v in card.variants if v.ref == card.variant_ref)
    return sum(f.size or 0 for f in variant.weights)


def load(label: str, backend: str, dtype: str) -> fastforks.Model:
    """The cached model, or load it on ComfyUI's device after asking ComfyUI to free room for it."""
    mm = _mm()
    device = mm.get_torch_device()
    mid = model_id(label)
    key = (mid, backend, dtype, str(device))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    while len(_cache) >= _MAX_CACHED:
        _cache.popitem(last=False)
        gc.collect()
        mm.soft_empty_cache()
    # weights + activations: ComfyUI unloads its own models until this much is free (no-op if it already is)
    mm.free_memory(2 * _weights_bytes(mid) + 2**30, device)
    model = fastforks.load(mid, mode=backend, device=device, dtype=None if dtype == "auto" else dtype)  # ty: ignore[invalid-argument-type]
    log.info("fastforks: loaded %s (%s)", mid, model.backend)
    _cache[key] = model
    return model


def unload_all() -> None:
    """Drop every cached model and return their memory to the device."""
    if _cache:
        log.info("fastforks: unloading %d cached model(s)", len(_cache))
    _cache.clear()
    gc.collect()
    _mm().soft_empty_cache()


def install_unload_hook() -> None:
    """Make ComfyUI's `unload_all_models` ("Unload models", `/free`) also drop our cache. Idempotent."""
    mm = _mm()
    original = mm.unload_all_models
    if original.__name__ == "_unload_all_models":  # already wrapped
        return

    def _unload_all_models() -> None:
        original()
        unload_all()

    mm.unload_all_models = _unload_all_models


def each(n: int, fn: Callable[[int], Any]) -> list[Any]:
    """`fn(i)` for every image of a batch, with ComfyUI's progress bar and interrupt check between images."""
    import comfy.utils  # noqa: PLC0415 - ComfyUI only

    mm = _mm()
    bar = comfy.utils.ProgressBar(n)
    out = []
    try:
        for i in range(n):
            mm.throw_exception_if_processing_interrupted()
            out.append(fn(i))
            bar.update(1)
    except mm.InterruptProcessingException:
        out.clear()
        mm.soft_empty_cache()  # the model stays cached; only this run's tensors are freed
        raise
    return out


def _out_device() -> torch.device:
    return _mm().intermediate_device()


def _ids(task: str, keep: Callable[[str], bool] = lambda _: True) -> list[str]:
    return [m for m in fastforks.list_models(task=task) if keep(m)]


class _Node:
    CATEGORY = "fastforks"
    FUNCTION = "run"
    DESCRIPTION = ""
    RETURN_TYPES: ClassVar[tuple[str, ...]] = ()


class Depth(_Node):
    """Relative or metric depth as a gray image (near = white), e.g. for depth ControlNets."""

    DESCRIPTION = "Depth map (inverse depth scaled to 0-1 per image, near = white)."
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        ids = _ids("depth", lambda m: bool({"depth", "inverse_depth"} & set(fastforks.model_card(m).outputs)))
        return {"required": {"image": ("IMAGE",), **common(ids, "depth-anything-v2-s")}}

    def run(self, *, image: torch.Tensor, model: str, backend: str, dtype: str) -> tuple[torch.Tensor]:
        """Run the model on every image."""
        m = load(model, backend, dtype)

        def one(i: int) -> torch.Tensor:
            r = m(convert.frame(image, i, m.device))
            return convert.depth_image(r.depth, r.inverse_depth)

        return (convert.to_image(each(image.shape[0], one), _out_device()),)


class Normals(_Node):
    """Surface normals as a normal-map image."""

    DESCRIPTION = "Surface normals as colours (red = right, green = up, blue = towards the camera)."
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        ids = _ids("depth", lambda m: "normals" in fastforks.model_card(m).outputs)
        return {"required": {"image": ("IMAGE",), **common(ids, "stablenormal-turbo")}}

    def run(self, *, image: torch.Tensor, model: str, backend: str, dtype: str) -> tuple[torch.Tensor]:
        """Run the model on every image."""
        m = load(model, backend, dtype)
        maps = each(image.shape[0], lambda i: convert.normals_image(m(convert.frame(image, i, m.device)).normals))
        return (convert.to_image(maps, _out_device()),)


class RemoveBackground(_Node):
    """Foreground alpha (background removal / matting)."""

    DESCRIPTION = "Foreground alpha: the image with alpha (RGBA) and the alpha as a mask (1 = foreground)."
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "alpha")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = "Only for trimap models (ViTMatte): 0 = background, 1 = foreground, 0.5 = unknown."
        return {
            "required": {"image": ("IMAGE",), **common(_ids("matting"), "birefnet-lite")},
            "optional": {"trimap": ("MASK", {"tooltip": tip})},
        }

    def run(
        self, image: torch.Tensor, model: str, backend: str, dtype: str, trimap: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model on every image."""
        m = load(model, backend, dtype)

        def one(i: int) -> torch.Tensor:
            tri = None if trimap is None else convert.from_mask(trimap, i, m.device)
            return m(convert.frame(image, i, m.device), trimap=tri).alpha

        alpha = convert.to_mask(each(image.shape[0], one), _out_device())
        return convert.with_alpha(image.to(alpha.device), alpha), alpha


class Segment(_Node):
    """Promptable segmentation (SAM family): points / box, or text for concept models (SAM 3)."""

    DESCRIPTION = "Mask from point / box prompts (SAM, SAM 2, ...) or a text prompt (SAM 3)."
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        px = "in pixels of the input image"
        return {
            "required": {
                "image": ("IMAGE",),
                **common(_ids("promptable_segmentation"), "sam2.1-hiera-small"),
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
                **common(_ids("detection"), "rf-detr-s-coco"),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "classes": ("STRING", {"default": "", "tooltip": tip}),
            }
        }

    def run(  # noqa: PLR0913 - one argument per input
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, score_threshold: float, classes: str
    ) -> tuple[torch.Tensor, str]:
        """Detect in every image."""
        m = load(model, backend, dtype)
        names = [c.strip() for c in classes.split(",") if c.strip()]
        text = names if cast("DetectionModel", m).open_vocabulary else None

        def one(i: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
            x = convert.frame(image, i, m.device)
            r = m(x, score_threshold=score_threshold, text=text)
            if names and text is None:
                r = r[torch.tensor([d.get("class_name") in names for d in r.to_list()], dtype=torch.bool)]
            dets = [{k: v for k, v in d.items() if k != "mask"} for d in r.to_list()]
            return convert.union_mask(r.masks, r.boxes, (x.shape[-2], x.shape[-1])), dets

        out = each(image.shape[0], one)
        return convert.to_mask([o[0] for o in out], _out_device()), convert.to_json([o[1] for o in out])


class Pose(_Node):
    """Human keypoints, drawn as skeletons on black (pose ControlNet input) and as JSON."""

    DESCRIPTION = "Keypoints as a skeleton image and as JSON. Top-down models use the person boxes of a Detect node."
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = "Top-down models: the 'detections' output of a Detect node (person boxes); empty: the whole image."
        return {
            "required": {
                "image": ("IMAGE",),
                **common(_ids("pose"), "vitpose-plus-small"),
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 64}),
            },
            "optional": {"detections": ("STRING", {"forceInput": True, "tooltip": tip})},
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
    ) -> tuple[torch.Tensor, str]:
        """Estimate the people of every image."""
        m = load(model, backend, dtype)

        def one(i: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
            x = convert.frame(image, i, m.device)
            h, w = x.shape[-2:]
            boxes = None
            if cast("PoseModel", m).needs_boxes:
                found = convert.person_boxes(detections, i)
                boxes = [[0.0, 0.0, float(w), float(h)]] if found is None else found
            r = m(x, boxes=boxes)
            drawn = convert.draw_pose(
                r.keypoints, r.keypoint_scores, r.skeleton, (h, w), threshold=keypoint_threshold, width=line_width
            )
            return drawn, r.to_list()

        out = each(image.shape[0], one)
        return convert.to_image([o[0] for o in out], _out_device()), convert.to_json([o[1] for o in out])


class Interpolate(_Node):
    """Frame interpolation (RIFE): `multiplier - 1` new frames between each pair of input frames."""

    DESCRIPTION = "Frame interpolation: inserts multiplier - 1 frames between consecutive frames of the batch."
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {
                "frames": ("IMAGE",),
                **common(_ids("interpolation"), "rife-4.26"),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
            }
        }

    def run(
        self, *, frames: torch.Tensor, model: str, backend: str, dtype: str, multiplier: int
    ) -> tuple[torch.Tensor]:
        """Interpolate between every pair of consecutive frames."""
        m = load(model, backend, dtype)
        n = frames.shape[0]
        if n < 2:  # noqa: PLR2004 - a pair
            return (frames,)

        def one(i: int) -> list[torch.Tensor]:
            a, b = convert.frame(frames, i, m.device), convert.frame(frames, i + 1, m.device)
            return [a] + [m(a, b, timestep=k / multiplier).frame for k in range(1, multiplier)]

        out = [f for pair in each(n - 1, one) for f in pair] + [convert.frame(frames, n - 1, m.device)]
        return (convert.to_image(out, _out_device()),)


class Classify(_Node):
    """Image classification: the top classes as text."""

    DESCRIPTION = "Top-k classes per image as 'name: probability' lines (images separated by a blank line)."
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {
                "image": ("IMAGE",),
                **common(_ids("classification"), "convnext_tiny.fb_in22k_ft_in1k"),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }

    def run(self, *, image: torch.Tensor, model: str, backend: str, dtype: str, top_k: int) -> tuple[str]:
        """Classify every image."""
        m = load(model, backend, dtype)

        def one(i: int) -> str:
            return "\n".join(f"{name}: {p:.4f}" for name, p in m(convert.frame(image, i, m.device)).top(top_k))

        return ("\n\n".join(each(image.shape[0], one)),)


NODE_CLASS_MAPPINGS = {
    "FastforksDepth": Depth,
    "FastforksNormals": Normals,
    "FastforksRemoveBackground": RemoveBackground,
    "FastforksSegment": Segment,
    "FastforksDetect": Detect,
    "FastforksPose": Pose,
    "FastforksInterpolate": Interpolate,
    "FastforksClassify": Classify,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastforksDepth": "Depth (fastforks)",
    "FastforksNormals": "Normals (fastforks)",
    "FastforksRemoveBackground": "Remove background (fastforks)",
    "FastforksSegment": "Segment (SAM, fastforks)",
    "FastforksDetect": "Detect objects (fastforks)",
    "FastforksPose": "Pose (fastforks)",
    "FastforksInterpolate": "Interpolate frames (fastforks)",
    "FastforksClassify": "Classify (fastforks)",
}
