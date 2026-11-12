"""`Contract`: the resolved, read-only interface of one model id, from its card and runner classes (no weights).

`model_card(id).contract()` answers what a caller or an exporter needs before loading: the inputs (and which optional
ones the variant takes), the outputs with units and coordinate conventions, the options with their bounds and what
they change, the modes and dtypes that are implemented versus checked against the reference outputs, and what an
exported `module()` contains. Every value is plain data (str, int, float, bool, tuples, read-only mappings);
`to_dict()` gives JSON-ready dicts and lists.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastforks._core import registry
from fastforks._core.cards import _freeze
from fastforks._core.errors import ModelNotFoundError
from fastforks._core.sizes import size_range
from fastforks._core.tasks import spec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks._core.cards import ModelCard, Variant

OPENCV_PIXELS = "opencv"
"""Pixel centres at integer coordinates: the image centre is `((W - 1) / 2, (H - 1) / 2)` (OpenCV, COLMAP - 0.5)."""
CAMERA_AXES = "x right, y down, z forward (OpenCV)"

_DEPTH_UNIT = {
    "metric": "metres",
    "scale_invariant": "unknown scale",
    "affine_invariant": "unknown scale and shift",
    "affine_invariant_inverse": "unknown scale and shift (inverse depth; depth and points are not defined)",
    "none": "no depth (normals only)",
}
_MULTIVIEW_UNIT = {
    "metric": "metres (estimated cameras) or the units of the given cam_to_world",
    "scale_invariant": "one unknown scale shared by every view, depth and translation alike",
}


@dataclass(frozen=True)
class Contract:
    """The resolved interface of one model id (`model_card(id).contract()`); read-only.

    Attributes:
        model: Versioned id, e.g. `"moge-2-vitl-normal@1"`.
        task: Task package.
        inputs: Input name -> `"required"`, `"optional"` or `"unused"` (unused ones raise if passed), with the
            input counts where the task has them (multi-view `min_views` / `max_views` / `per_view`; promptable
            segmentation `prompts`). Image inputs are RGB; their accepted forms are those of every task.
        sizes: `size_range()` as a mapping: `min_size`, `multiple`, `native`, `encoder`, ... (see `SizeRange`).
        outputs: Output name -> `{"shape", "dtype", "unit", "frame", "present"}` (`frame`: axes and pixel grid; empty
            where it does not apply; `present`: when the result has it, e.g. segmentation `logits`: only with
            `logits=True`). Every field of the task's result, including the derived ones (`labels`, `probs`, `top`,
            `cls_token`, `feature_map`, `extrinsics`, ...); outputs outside the task's standard fields go to `extras`.
        camera: Pixel convention of the intrinsics the model takes and returns, `"opencv"` (`OPENCV_PIXELS`, every
            task), with the axes; empty for tasks without cameras.
        options: Option name -> `{"default", "min", "max", "choices", "effect", "aliases"}` (absent bounds are
            `None` / empty; `aliases`: accepted old names; `effect`: `"output"` changes the results, `"speed"` only
            how fast, `"precision"` the rounding).
        call: Call-time keyword argument -> `{"default", "type", "min", "max"}` (from the task's `__call__`; `default`
            `"required"` where there is none; bounds are those of the load option of the same name, else `None`).
            The first argument(s), the inputs, are in `inputs`.
        modes: Mode -> device -> `{"implemented": dtypes, "validated": dtypes}` plus the mode's kernel libraries;
            validated = checked against the reference outputs (CPU float32, the card's CUDA and MPS dtypes);
            implemented = what `load(dtype=)` accepts there. Kernels run in one dtype per device and a card lists a
            device only where that dtype is validated, so for `kernels` both are equal. Mode-specific inputs (if they
            differ) as `"inputs"`.
        export: What `export_onnx` / `fastforks export` produce (portable only): `inputs` of the graph, `inside` /
            `outside` (the stages in and out of the graph), `dynamic` (height / width can be dynamic) and
            `fixed_reason` (why not), `default_size` (`None`: a size is required).
    """

    model: str
    task: str
    inputs: Mapping[str, Any]
    sizes: Mapping[str, Any]
    outputs: Mapping[str, Mapping[str, str]]
    camera: Mapping[str, str]
    options: Mapping[str, Mapping[str, Any]]
    call: Mapping[str, Mapping[str, Any]]
    modes: Mapping[str, Mapping[str, Any]]
    export: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Plain dicts and lists (JSON-ready); a new copy on every call."""
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}


def _plain(x: object) -> Any:  # noqa: ANN401 - any frozen value
    if isinstance(x, str | bytes):
        return x
    if hasattr(x, "items"):
        return {k: _plain(v) for k, v in x.items()}  # ty: ignore[call-non-callable] - a mapping
    if isinstance(x, tuple | list | frozenset | set):
        return [_plain(v) for v in x]
    return x


def resolve(card: ModelCard) -> Contract:
    """The contract of `card` (from `model_card(id)`: `card.variant_ref` set). Imports the model code; no weights."""
    from fastforks._core.model import make_params, runner_class  # noqa: PLC0415 - model imports cards

    if card.variant_ref is None:
        msg = f"{card.family}: a family card has no single contract; use {card.task}.model_card(id).contract()"
        raise ModelNotFoundError(msg)
    _, v = registry.resolve(card.task, card.variant_ref)
    uses = {}
    for mode in card.modes:
        rcls = runner_class(card, mode)
        uses[mode] = rcls.input_use(make_params(rcls.Params, card, v, {}))
    first = uses[card.modes[0]]
    rng = size_range(v, card)
    return Contract(
        model=v.ref,
        task=card.task,
        inputs=_freeze({**_task_inputs(card.task), **first}),
        sizes=_freeze(dataclasses.asdict(rng)),
        outputs=_freeze(_outputs(card, v, first)),
        camera=_freeze(_camera(card.task)),
        options=_freeze({k: _option(card, k) for k in card.options}),
        call=_freeze(_call(card)),
        modes=_freeze({m: _mode(card, m, uses[m] if uses[m] != first else None) for m in card.modes}),
        export=_freeze(_export(card, v, first)),
    )


def _task_inputs(task: str) -> dict[str, str]:
    main = (
        spec(task).inputs[:1] if task in {"multiview", "matting"} else spec(task).inputs
    )  # cameras, trimap: `input_use`
    return dict.fromkeys(main, "required")


def _camera(task: str) -> dict[str, str]:
    if task in {"depth", "multiview", "stereo"}:
        return {"pixels": OPENCV_PIXELS, "axes": CAMERA_AXES}
    return {}


def _call(card: ModelCard) -> dict[str, dict[str, Any]]:
    """The keyword arguments of the task model's `__call__` (an overloaded one: the implementation's union)."""
    mod = importlib.import_module(f"fastforks.{card.task}._api")
    cls = next(
        c
        for c in vars(mod).values()
        if isinstance(c, type)
        and c.__module__ == mod.__name__
        and "__call__" in vars(c)
        and c.__name__.endswith("Model")
    )
    out: dict[str, dict[str, Any]] = {}
    for name, p in inspect.signature(cls.__call__).parameters.items():
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        default = "required" if p.default is inspect.Parameter.empty else p.default
        out[name] = {
            "default": default if default is None or isinstance(default, str | bool | int | float) else repr(default),
            "type": p.annotation
            if isinstance(p.annotation, str)
            else getattr(p.annotation, "__name__", str(p.annotation)),
            "min": card.option_min.get(name),
            "max": card.option_max.get(name),
        }
    return out


def _option(card: ModelCard, name: str) -> dict[str, Any]:
    return {
        "default": card.options[name],
        "min": card.option_min.get(name),
        "max": card.option_max.get(name),
        "choices": card.option_choices.get(name, ()),
        "effect": card.option_effect.get(name, "output"),
        "aliases": tuple(a for a, n in card.option_aliases.items() if n == name),
    }


def _mode(card: ModelCard, mode: str, inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    from fastforks._core.model import runner_class  # noqa: PLC0415 - model imports cards

    extra = {} if inputs is None else {"inputs": inputs}
    if mode == "kernels":
        kd = card.kernels_dtype
        out: dict[str, Any] = {"cuda": {"implemented": (kd,), "validated": (kd,)}, "kernel_libs": card.kernel_libs}
        if card.mps_kernel_libs:
            md = card.mps_kernels_dtype or kd
            out["mps"] = {"implemented": (md,), "validated": (md,)}
            out["mps_kernel_libs"] = card.mps_kernel_libs
        return out | extra
    cuda = tuple(sorted({"float32", *runner_class(card, "portable").dtypes}))
    out = {
        "cpu": {"implemented": ("float32",), "validated": ("float32",)},
        "cuda": {"implemented": cuda, "validated": (card.portable_cuda_dtype,)},
    }
    mps_dtypes = tuple(dict.fromkeys(("float32", *card.portable_mps_dtypes)))
    out["mps"] = {"implemented": mps_dtypes, "validated": card.portable_mps_dtypes}
    return out | extra


_ALWAYS = "always"
_DERIVED: dict[str, dict[str, tuple[str, str, str, str, str]]] = {  # result fields / methods not listed by the cards
    "classification": {
        "probs": ("[B, classes]", "float32", "softmax of logits", "", "always (`result.probs()`)"),
        "top": ("list of (name, probability)", "-", "the k most likely classes", "", "always (`result.top(k)`)"),
    },
    "segmentation": {
        "labels": (
            "[B, H, W]",
            "int64",
            "class index into the label set (argmax of the logits)",
            "input pixels",
            _ALWAYS,
        ),
    },
    "features": {
        "cls_token": ("[B, C]", "float32", "first token of last_hidden_state", "", "always"),
        "register_tokens": ("[B, R, C]", "float32", "register tokens (R may be 0)", "", "always"),
        "patch_tokens": ("[B, h * w, C]", "float32", "patch tokens of last_hidden_state, row-major", "", "always"),
        "feature_map": (
            "[B, C, h, w]",
            "float32",
            "patch tokens as a grid (method `feature_map(layer=None)`)",
            "",
            "always",
        ),
    },
    "matching": {
        "matches": (
            "([n, 2], [n, 2])",
            "float32",
            (
                "keypoint matches in pixels of image1 and image2 (method `matches`): dense models draw n by certainty,"
                " sparse ones return their matched keypoints"
            ),
            "(x, y), pixel centres at integer coordinates",
            "always",
        ),
        "verify": (
            "VerifiedMatches",
            "float32",
            (
                "the matches with their scores, the RANSAC inlier mask and the fundamental matrix or (with intrinsics)"
                " the relative pose (method `verify`, computed when called)"
            ),
            "(x, y), pixel centres at integer coordinates",
            "always",
        ),
    },
    "tracking": {
        "visible": ("[T, Q]", "bool", "visibility > 0.5", "", "when visibility is"),
    },
    "multiview": {
        "extrinsics": (
            "[V, 3, 4]",
            "float32",
            "world-to-camera [R | t], from cam_to_world",
            "",
            "when cam_to_world is",
        ),
    },
    "promptable_segmentation": {
        "masks": (
            "[P, M, H, W]",
            "bool",
            "masks (pred_masks upsampled to the input, logits > 0)",
            "input pixels",
            _ALWAYS,
        ),
    },
}
_PRESENT: dict[tuple[str, str], str] = {
    ("segmentation", "logits"): "only with logits=True (else None)",
    ("features", "hidden_states"): "only for the layers= asked for (default none)",
}


def _outputs(card: ModelCard, v: Variant, uses: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Every field of the result: the card's outputs, then the derived ones (not already listed)."""
    out = {k: _output(card, v, k, uses) for k in v.outputs}
    concept = card.task == "promptable_segmentation" and "pred_masks" not in v.outputs  # its masks are listed
    for k, row in {} if concept else _DERIVED.get(card.task, {}).items():
        out.setdefault(k, _d(*row))
    for (task, k), present in _PRESENT.items():
        if task == card.task and k in out:
            out[k] = out[k] | {"present": present}
    return out


_FIXED_TABLES: dict[str, dict[str, tuple[str, str, str, str]]] = {  # output tables that depend on the task alone
    "stereo": {
        "disparity": ("[B, H, W]", "float32", "pixels of the input, x_left - x_right >= 0", "left view"),
        "confidence": ("[B, H, W]", "float32", "[0, 1]", "left view"),
    },
    "flow": {
        "flow": ("[B, 2, H, W]", "float32", "pixels of the input, (dx, dy)", "image1 -> image2, image1 pixels"),
    },
    "matching": {
        "warp": ("[B, 2, H1, W1]", "float32", "pixels of image2, (x, y)", "image1 pixels"),
        "certainty": ("[B, H1, W1]", "float32", "[0, 1]", "image1 pixels"),
        "keypoints1": ("[N, 2]", "float32", "pixels of image1, (x, y)", "detected keypoints, one pair per call"),
        "keypoints2": ("[M, 2]", "float32", "pixels of image2, (x, y)", "detected keypoints"),
        "pairs": ("[K, 2]", "int64", "indices into keypoints1, keypoints2", "one row per match"),
        "scores": ("[K]", "float32", "[0, 1] match confidence", ""),
    },
    "matting": {"alpha": ("[B, H, W]", "float32", "[0, 1], 1 = foreground", "input pixels")},
    "tracking": {
        "tracks": ("[T, Q, 2]", "float32", "pixels of the input, (x, y)", "every frame, one row per query"),
        "visibility": ("[T, Q]", "float32", "[0, 1] probability that the point is visible", "every frame"),
    },
    "features": {
        "last_hidden_state": ("[B, N, C]", "float32", "tokens after the final norm", "prefix tokens, then patches"),
        "pooler_output": ("[B, C]", "float32", "image embedding", ""),
        "hidden_states": ("tuple of [B, N, C]", "float32", "block outputs before the final norm", ""),
    },
    "pose": {
        "keypoints": ("[P, K, 2]", "float32", "pixels of the input image", "(x, y), corner origin"),
        "keypoint_scores": ("[P, K]", "float32", "keypoint confidence (heatmap maximum)", ""),
        "boxes": ("[P, 4]", "float32", "pixels of the input image", "(x0, y0, x1, y1); top-down: the boxes given"),
        "scores": ("[P]", "float32", "person confidence", ""),
    },
}


def _output(card: ModelCard, v: Variant, name: str, uses: Mapping[str, Any]) -> dict[str, str]:
    task, info = card.task, v.output_info
    std = name in spec(task).standard
    if task == "depth":
        unit = _DEPTH_UNIT[info["kind"]]
        inverse = unit if info["kind"] == "affine_invariant_inverse" else f"1 / ({unit})"
        cam = f"camera frame, {CAMERA_AXES}"
        table = {
            "depth": ("[B, H, W]", "float32", unit, "z along the optical axis, input pixels"),
            "inverse_depth": ("[B, H, W]", "float32", inverse, "input pixels"),
            "points": ("[B, H, W, 3]", "float32", unit, cam),
            "intrinsics": ("[B, 3, 3]", "float32", "pixels of the input", f"{OPENCV_PIXELS} pixels"),
            "confidence": ("[B, H, W]", "float32", "model-specific, higher is better", "input pixels"),
            "mask": ("[B, H, W]", "bool", "valid pixels", "input pixels"),
            "normals": ("[B, H, W, 3]", "float32", "unit vectors", cam),
        }
    elif task == "multiview":
        unit = _MULTIVIEW_UNIT[info["kind"]]
        if uses.get("cam_to_world") == "unused":  # the model cannot take cameras: only its own estimates
            unit = unit.removesuffix(" or the units of the given cam_to_world")
        world = f"world frame (given cam_to_world, else the first view's camera), {CAMERA_AXES}"
        table = {
            "depth": ("[V, H, W]", "float32", unit, "z along each view's optical axis, input pixels"),
            "world_points": ("[V, H, W, 3]", "float32", unit, world),
            "confidence": ("[V, H, W]", "float32", "model-specific, higher is better", "input pixels"),
            "mask": ("[V, H, W]", "bool", "valid pixels", "input pixels"),
            "intrinsics": ("[V, 3, 3]", "float32", "pixels of the input", f"{OPENCV_PIXELS} pixels"),
            "cam_to_world": ("[V, 4, 4]", "float32 (float64 if given as float64)", unit, world),
        }
    elif task in _FIXED_TABLES:
        table = _FIXED_TABLES[task]
    elif task == "classification":
        return _d("[B, classes]", "float32", f"logits over the label set {info['label_set']}", "")
    elif task == "segmentation":
        return _d("[B, classes, H, W]", "float32", f"logits over the label set {info['label_set']}", "input pixels")
    elif task == "detection":
        table = {
            "boxes": ("[N, 4]", "float32", "pixels of the input image", "(x0, y0, x1, y1), corner origin"),
            "scores": ("[N]", "float32", "confidence", ""),
            "labels": ("[N]", "int64", f"indices into the label set {info.get('label_set', '')}", ""),
            "masks": ("[N, H, W]", "bool", "instance masks", "input pixels"),
        }
    else:  # promptable segmentation
        table = {
            "masks": ("[P, M, H, W] or [N, H, W]", "bool", "masks (logits > 0)", "input pixels"),
            "pred_masks": ("[P, M, h, w]", "float32", "low-resolution mask logits", "encoder grid"),
            "iou_scores": ("[P, M]", "float32", "predicted mask IoU", ""),
            "boxes": ("[N, 4]", "float32", "pixels of the input image", "(x0, y0, x1, y1)"),
            "scores": ("[N]", "float32", "instance confidence", ""),
        }
    if std and name in table:
        return _d(*table[name])
    lead = {"multiview": "[V, ...]"}.get(task, "[B, ...]")
    return _d(lead, "float32", "model-specific (see the model page)", "")


def _d(shape: str, dtype: str, unit: str, frame: str, present: str = _ALWAYS) -> dict[str, str]:
    return {"shape": shape, "dtype": dtype, "unit": unit, "frame": frame, "present": present}


_STAGES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "stereo": (("scale", "pad", "network", "crop"), ("image conversion", "rig= rectification", "depth / points")),
    "flow": (("scale", "pad", "network", "crop"), ("image conversion",)),
    "matching": (("scale", "resize to the network squares", "network", "resize to image1"), ("image conversion",)),
    "tracking": (
        ("resize to the network size", "network (frame by frame)", "scale to the input"),
        ("image conversion",),
    ),
    "depth": (
        ("resize to the processing size", "pad", "network", "crop / resize to the input"),
        ("image conversion", "other maps than the main one", "intrinsics="),
    ),
    "classification": (("network",), ("image conversion", "resize", "crop", "normalise (model.data_config)", "top-k")),
    "segmentation": (("input handling", "network", "logits at the input size"), ("image conversion", "argmax")),
    "matting": (("resize to the network size", "normalise", "network", "resize to the input"), ("image conversion",)),
    "features": (("resize to the patch stride", "network"), ("image conversion", "pooler_output", "hidden_states")),
    "promptable_segmentation": (
        ("resize to the encoder input", "image encoder"),
        ("image conversion", "prompt encoder", "mask decoder", "mask thresholding"),
    ),
    "detection": (
        ("normalise", "pad", "network (R-CNN: backbone + FPN; DETR / RT-DETR: all)"),
        ("image conversion", "resize", "proposals", "NMS", "score_threshold", "box scaling", "mask pasting"),
    ),
    "multiview": (("resize to the network size", "network", "resize to the input"), ("image conversion",)),
    "pose": (
        ("normalise", "network (person crop -> heatmaps)"),
        ("image conversion", "person crops", "heatmap decoding"),
    ),
}


def _export(card: ModelCard, v: Variant, uses: Mapping[str, Any]) -> dict[str, Any]:
    from fastforks._core.export import fixed_size_reason  # noqa: PLC0415 - export imports the model module

    inside, outside = _STAGES[card.task]
    inputs: tuple[str, ...] = spec(card.task).inputs
    if card.task in {"multiview", "matting"}:  # the image(s), then the cameras / trimap the model requires
        inputs = (inputs[0], *(k for k in inputs[1:] if uses.get(k) == "required"))
    side = 0  # as `Model.export_size()`: the classification crop, the promptable encoder side, else the task's
    if card.task == "classification":
        side = v.io.crop or v.io.resize
    elif card.task == "promptable_segmentation":
        side = size_range(v, card).encoder
    reason = fixed_size_reason(card, size_range(v, card))
    return {
        "mode": "portable",
        "inputs": inputs,
        "inside": inside,
        "outside": outside,
        "dynamic": not reason,
        "fixed_reason": reason,
        "default_size": (side, side) if side else spec(card.task).export_size,
    }
