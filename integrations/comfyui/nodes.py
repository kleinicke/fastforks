"""ComfyUI nodes over the public fastforks API: `fastforks.load(id)` once, then call the model per image.

Every node takes a model id, a backend (`auto` / `portable` / `kernels`) and a dtype. Loaded models stay cached across
runs (at most `_MAX_CACHED`, least recently used dropped first) on ComfyUI's device; ComfyUI's "Unload models" also
empties the cache. ComfyUI is imported lazily so the module imports without it (unit tests).
"""

from __future__ import annotations

import gc
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

import fastforks
from fastforks import restoration

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
_FILE = "file: "  # dropdown prefix of a checkpoint file of ComfyUI's `models/upscale_models`
_files: dict[str, tuple[float, str | None]] = {}  # path -> (mtime, `restoration.checkpoint_family`)
_cache: OrderedDict[tuple[Any, ...], fastforks.Model] = OrderedDict()
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
    """The model id of a dropdown entry (without its licence tag); file entries unchanged."""
    return label if label.startswith(_FILE) else label.split(" [", 1)[0]


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


def upscale_files() -> list[str]:
    """Entries for the files of `models/upscale_models` that `restoration.load_file` runs (each checked once)."""
    import folder_paths  # noqa: PLC0415 - ComfyUI only

    entries = []
    for name in folder_paths.get_filename_list("upscale_models"):
        path = folder_paths.get_full_path("upscale_models", name)
        mtime = Path(path).stat().st_mtime
        if _files.get(path, (None,))[0] != mtime:
            _files[path] = (mtime, restoration.checkpoint_family(path))
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
    # weights + activations: ComfyUI unloads its own models until this much is free (no-op if it already is)
    mm.free_memory(2 * (file.stat().st_size if file else _weights_bytes(mid)) + 2**30, device)
    kw: dict[str, Any] = {"mode": backend, "device": device, "dtype": None if dtype == "auto" else dtype}
    model = restoration.load_file(file, **kw) if file else fastforks.load(mid, options=options or None, **kw)
    log.info("fastforks: loaded %s (%s)", mid, model.backend)
    _cache[key] = model
    return model


def setup_cache(environ: os._Environ[str] | dict[str, str] = os.environ) -> None:
    """Download weights and kernels into ComfyUI's `models/fastforks` (the first `fastforks` model folder).

    Files already in the HF hub or torch hub cache, the normal fastforks cache or other `fastforks` folders of
    `extra_model_paths.yaml` are used in place (`FASTFORKS_READ_CACHES`); only missing files are downloaded.
    """
    import folder_paths  # noqa: PLC0415 - ComfyUI only

    folder_paths.add_model_folder_path("fastforks", os.path.join(folder_paths.models_dir, "fastforks"))  # noqa: PTH118
    dirs = list(folder_paths.get_folder_paths("fastforks"))
    normal = environ.get("FASTFORKS_CACHE") or fastforks_cache_default()
    reads = [*dirs[1:], normal, *environ.get("FASTFORKS_READ_CACHES", "").split(os.pathsep)]
    environ["FASTFORKS_CACHE"] = dirs[0]
    keep = [d for d in dict.fromkeys(reads) if d and d != dirs[0] and os.path.isdir(d)]  # noqa: PTH112
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

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        ids = _ids("classification", lambda m: fastforks.model_card(m).output_info.get("probs") == "sigmoid")
        thr = {"min": 0.0, "max": 1.0, "step": 0.01}
        return {
            "required": {
                "image": ("IMAGE",),
                **common(ids, "wd-eva02-large-tagger-v3"),
                "general_threshold": ("FLOAT", {"default": 0.35, **thr}),
                "character_threshold": ("FLOAT", {"default": 0.75, **thr}),
            }
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
    ) -> tuple[str]:
        """Tag every image."""
        m = load(model, backend, dtype)
        names = cast("tuple[str, ...]", m.class_names)  # ty: ignore[unresolved-attribute]

        def one(i: int) -> str:
            probs = m(convert.frame(image, i, m.device)).probs.float().cpu()
            return wd_tags(probs, names, general_threshold, character_threshold)

        return ("\n".join(each(image.shape[0], one)),)


_PX = "in pixels of the first frame"


def _prompt_inputs() -> dict[str, Any]:
    return {
        "positive_points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' foreground points {_PX}"}),
        "negative_points": ("STRING", {"default": "", "tooltip": f"'x,y; x,y' background points {_PX}"}),
        "box": ("STRING", {"default": "", "tooltip": f"'x0,y0,x1,y1' {_PX}"}),
    }


class SemanticSegment(_Node):
    """Semantic segmentation: a class colour map, a mask of chosen classes and the classes present."""

    DESCRIPTION = "Class per pixel: colour map (one colour per class), mask of the listed classes, classes present."
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("colour_map", "mask", "classes_found")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        tip = "Comma-separated class names for the mask (e.g. 'sky, tree'); empty: an empty mask."
        return {
            "required": {
                "image": ("IMAGE",),
                **common(_ids("segmentation"), "segnext-mscan-t-ade20k"),
                "classes": ("STRING", {"default": "", "tooltip": tip}),
            }
        }

    def run(
        self, *, image: torch.Tensor, model: str, backend: str, dtype: str, classes: str
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Segment every image."""
        m = load(model, backend, dtype)
        names = cast("tuple[str, ...]", m.class_names)  # ty: ignore[unresolved-attribute]
        wanted = torch.tensor(convert.class_ids(classes, names), dtype=torch.long)
        colours = convert.palette(len(names))

        def one(i: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
            lab = m(convert.frame(image, i, m.device)).labels
            present = [names[c] for c in lab.unique().tolist()]
            return convert.labels_image(lab, colours), torch.isin(lab, wanted.to(lab.device)), present

        out = each(image.shape[0], one)
        found = "\n".join(", ".join(o[2]) for o in out)
        return (
            convert.to_image([o[0] for o in out], _out_device()),
            convert.to_mask([o[1] for o in out], _out_device()),
            found,
        )


class Stereo(_Node):
    """Stereo matching: disparity of the left image from a rectified left / right pair."""

    DESCRIPTION = "Disparity of rectified left / right pairs, scaled to 0-1 per image (near = white, as Depth)."
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {"left": ("IMAGE",), "right": ("IMAGE",), **common(_ids("stereo"), "raft-stereo-middlebury")}
        }

    def run(
        self, *, left: torch.Tensor, right: torch.Tensor, model: str, backend: str, dtype: str
    ) -> tuple[torch.Tensor]:
        """Match every left / right pair."""
        if left.shape != right.shape:
            msg = f"left and right must be batches of one size, got {tuple(left.shape)} and {tuple(right.shape)}"
            raise ValueError(msg)
        m = load(model, backend, dtype)

        def one(i: int) -> torch.Tensor:
            return convert.depth_image(
                None, m(convert.frame(left, i, m.device), convert.frame(right, i, m.device)).disparity
            )

        return (convert.to_image(each(left.shape[0], one), _out_device()),)


class Flow(_Node):
    """Optical flow between consecutive frames: colour-wheel images and the raw flow (for Warp)."""

    DESCRIPTION = "Flow from each frame to the next (B - 1 results): colour-wheel image and raw flow in pixels."
    RETURN_TYPES = ("IMAGE", "FASTFORKS_FLOW")
    RETURN_NAMES = ("flow_image", "flow")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {"required": {"frames": ("IMAGE",), **common(_ids("flow"), "sea-raft-s")}}

    def run(self, *, frames: torch.Tensor, model: str, backend: str, dtype: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow of every pair of consecutive frames."""
        if frames.shape[0] < 2:  # noqa: PLR2004 - a pair
            msg = "Flow needs a batch of at least 2 frames"
            raise ValueError(msg)
        m = load(model, backend, dtype)
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

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        ids = _ids("promptable_segmentation", lambda m: "samurai" in fastforks.model_card(m).options)
        tip = "SAMURAI: motion-aware memory (Kalman filter), steadier on fast motion and occlusions; best with a box."
        return {
            "required": {
                "frames": ("IMAGE",),
                **common(ids, "sam2.1-hiera-small"),
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
        masks = m.track(video, **kwargs).masks  # ty: ignore[unresolved-attribute]
        return (convert.to_mask(list(masks), _out_device()),)


class PointTrack(_Node):
    """Point tracking: points of the first frame followed through the frames, drawn on them and as JSON."""

    DESCRIPTION = "Tracks of a point grid (or given points) of the first frame, drawn on the frames and as JSON."
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tracks_image", "tracks")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        return {
            "required": {
                "frames": ("IMAGE",),
                **common(_ids("tracking"), "tapir"),
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


class Upscale(_Node):
    """Super-resolution / restoration (Real-ESRGAN, SPAN, SwinIR, HAT, DAT, ...), tiled for large images."""

    DESCRIPTION = (
        "Upscaled image. Models: fastforks ids (Real-ESRGAN, SPAN, SwinIR, HAT, DAT, ...), or files of"
        " models/upscale_models (file: ...) of the architectures fastforks runs (Real-ESRGAN / ESRGAN, SPAN, SwinIR);"
        " other files are not listed. Keep tiling on for HAT and DAT: one pass runs out of memory at 2048 x 1024"
        " on 16 GB."
    )
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI's name
        """ComfyUI inputs."""
        inputs = common(_ids("restoration"), "realesrgan-x4plus")
        entries, opts = inputs["model"]
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


NODE_CLASS_MAPPINGS = {
    "FastforksDepth": Depth,
    "FastforksNormals": Normals,
    "FastforksRemoveBackground": RemoveBackground,
    "FastforksSegment": Segment,
    "FastforksDetect": Detect,
    "FastforksPose": Pose,
    "FastforksInterpolate": Interpolate,
    "FastforksClassify": Classify,
    "FastforksTag": Tag,
    "FastforksSemanticSegment": SemanticSegment,
    "FastforksStereo": Stereo,
    "FastforksFlow": Flow,
    "FastforksWarp": Warp,
    "FastforksVideoSegment": VideoSegment,
    "FastforksPointTrack": PointTrack,
    "FastforksUpscale": Upscale,
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
    "FastforksTag": "Tag (WD tagger, fastforks)",
    "FastforksSemanticSegment": "Semantic segmentation (fastforks)",
    "FastforksStereo": "Stereo disparity (fastforks)",
    "FastforksFlow": "Optical flow (fastforks)",
    "FastforksWarp": "Warp by flow (fastforks)",
    "FastforksVideoSegment": "Video segment (SAM 2 / SAMURAI, fastforks)",
    "FastforksPointTrack": "Track points (fastforks)",
    "FastforksUpscale": "Upscale / restore (fastforks)",
}
