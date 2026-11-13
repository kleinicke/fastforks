"""ONNX and TensorRT export of the portable backend of any task (`Model.export_onnx` / `Model.export_tensorrt`).

The exported graph is `model.module()` at one fixed size. Its inputs are named by the task spec (`TaskSpec.inputs`:
`left`, `right` for stereo, `image` otherwise) and its outputs as `output_names` says: the variant's main output
(`disparity`, `inverse_depth`, `depth`, `logits`, ...), the names a module declares (SAM: `image_embeddings`), or a
dict's keys (multi-view); an output named like an input gets `_out` (restoration: `image_out`). Example inputs come
from the task model (`Model.example_inputs`): synthetic RGB images in `[0, 1]` (a shifted pair for stereo), the
normalised crop for classification. The graph keeps the model's compute dtype: fp32 with `dtype="float32"`, else the
card's reduced precision on CUDA (the casts are in the graph, so TensorRT builds it strongly typed). `kernels` is not
exportable: ONNX and TensorRT cannot run our kernels.

`onnx` and `onnxscript` (`pip install "fastforks-core[onnx]"`) and `tensorrt` (`"fastforks-core[tensorrt]"`) are
optional and imported only here. A TensorRT engine is tied to the GPU and TensorRT version it was built with: build it
on the target machine.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.errors import BackendUnavailableError, InputError, install_hint
from fastforks._core.export_resize import matmul_antialias
from fastforks._core.tasks import spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks._core.cards import ModelCard
    from fastforks._core.model import Model
    from fastforks._core.sizes import SizeRange

ENGINE_NOTE = "a TensorRT engine only runs on the GPU model and TensorRT version it was built with"


def _missing(module: str, extra: str) -> BackendUnavailableError:
    msg = f"{module} is not installed: {install_hint(extra)}"
    return BackendUnavailableError(msg)


def _tensorrt() -> Any:  # noqa: ANN401 - the tensorrt module
    try:
        import tensorrt  # noqa: PLC0415 - optional
    except ImportError as e:
        err = _missing("tensorrt", "tensorrt")
        raise err from e
    return tensorrt


@functools.cache
def _trt_logger() -> Any:  # noqa: ANN401 - tensorrt.ILogger
    """One TensorRT logger per process (TensorRT warns when a runtime gets another one than the first).

    Prints warnings and errors to stderr as `trt.Logger(WARNING)`; `errors` keeps the error messages, so a failed
    engine build can name its first error.
    """
    trt = _tensorrt()

    class _Logger(trt.ILogger):
        def __init__(self) -> None:
            trt.ILogger.__init__(self)
            self.errors: list[str] = []

        def log(self, severity: Any, msg: str) -> None:  # noqa: ANN401 - trt.ILogger.Severity
            level = int(severity)  # 0 internal error, 1 error, 2 warning, 3 info, 4 verbose
            if level <= int(trt.ILogger.Severity.WARNING):
                sys.stderr.write(f"[TRT] [{'FEWIV'[level]}] {msg}\n")
            if level <= int(trt.ILogger.Severity.ERROR):
                self.errors.append(msg)

    return _Logger()


def output_name(model: Model) -> str:
    """Name of the exported output: the variant's main output (first of the task's standard outputs it has)."""
    return next(k for k in spec(model.card.task).standard if k in model.variant.outputs)


def check_exportable(model: Model, height: int, width: int, batch: int) -> None:
    """Raise unless `model` runs portable and the size is supported.

    Raises:
        BackendUnavailableError: The model runs `kernels`.
        InputError: The size is below the model's minimum, or `batch < 1`.
    """
    if model.backend.mode != "portable":
        msg = (
            f"{model.variant.ref}: the kernels backend runs custom CUDA kernels, which ONNX and TensorRT cannot run;"
            " load it with mode='portable' to export (docs: export.md compares kernels with TensorRT)"
        )
        raise BackendUnavailableError(msg)
    rng = model.size_range()
    if min(height, width) < rng.min_size or batch < 1:
        msg = (
            f"{model.variant.ref}: export size {height}x{width} (batch {batch}) is outside the supported range: "
            f"{rng.describe(tested=False)}, batch >= 1"
        )
        raise InputError(msg)


SizeBounds = tuple[tuple[int, int], tuple[int, int]]
"""`((min_h, min_w), (max_h, max_w))`: the range of a dynamic-shape export."""


DYNAMIC_VERIFIED = frozenset(
    {"aanet", "cas_gwcnet", "coex", "detr", "gwcnet", "igev_stereo", "psmnet", "raft_stereo", "rcnn", "s2m2", "sapiens"}
    | {"segformer"}
)
"""Card families whose `--size-range` export was checked against eager `module()` across a size range (min, traced,
off-trace and max sizes). Every other family exports at a fixed size only."""

DYNAMIC_UNSUPPORTED = {
    "prompt_depth_anything": "the dynamic-shape spec would also resize the depth prompt, which has a size of its own",
    "depth_anything_v2": "the model rounds a symbolic size (Python `round()`), which torch.export cannot trace",
    "hitnet": "the traced graph fixes the height (the engine would only look dynamic)",
    "moge": "the traced graph fixes the input size (the file would only look dynamic)",
    "las2": "the traced graph fails at other sizes inside the range (a broadcast fixed at the traced size)",
    "rtdetrv2": "the traced graph fails at other sizes inside the range (anchor shapes fixed at the traced size)",
    "dfine": "the traced graph fails at other sizes inside the range (anchor shapes fixed at the traced size)",
    "fast_acvnet": "the traced graph gives wrong values at other sizes inside the range",
    "cfnet": "the dynamic graph does not load in onnxruntime",
    "rt_monster_plusplus": "the dynamic graph does not load in onnxruntime",
    "c_foundationstereo": "the dynamic graph does not load in onnxruntime",
    "dinov2_linear": "the sliding-window count follows the image size; the traced graph keeps the traced count",
    **dict.fromkeys(
        ("mmseg", "nmrf", "zoedepth", "eomt", "mask2former", "depth_anything_3", "midas", "metric3d", "bridgedepth"),
        "torch.export or the ONNX conversion fails with dynamic sizes",
    ),
}
"""Card families whose `--size-range` export is known not to work, with the reason (the rest of the families outside
`DYNAMIC_VERIFIED` are not verified)."""


def fixed_size_reason(card: ModelCard, rng: SizeRange) -> str:
    """Why a variant of `card` with size range `rng` exports at one fixed size only; empty if a size range works.

    A size range works for the families in `DYNAMIC_VERIFIED` only.
    """
    if rng.native or rng.encoder:
        what = (
            f"the fixed {rng.native[0]}x{rng.native[1]} input" if rng.native else f"its fixed {rng.encoder} px encoder"
        )
        return f"a size range is meaningless here, the model has {what}; export with a fixed --size"
    if card.task == "multiview":
        return "multi-view models export at a fixed size (the camera inputs have no image axes); use --size"
    if card.task == "features":
        return "feature models export at a fixed size (their dynamic-shape trace fails); use --size"
    if card.family in DYNAMIC_UNSUPPORTED:
        return f"dynamic export is not supported ({DYNAMIC_UNSUPPORTED[card.family]}); use a fixed --size"
    if card.family not in DYNAMIC_VERIFIED:
        return (
            f"dynamic export is not supported (not verified for the {card.family} family: its traced graph was not "
            "checked across a size range); use a fixed --size"
        )
    return ""


def _multiple(model: Model) -> int:
    return model.size_range().multiple


def _bounds(model: Model, height: int, width: int, bounds: SizeBounds | None, batch: int) -> None:
    """Check that `bounds` holds `(height, width)` and lies in the model's range."""
    if bounds is None:
        return
    (lo_h, lo_w), (hi_h, hi_w) = bounds
    ref = model.variant.ref
    if reason := fixed_size_reason(model.card, model.size_range()):
        msg = f"{ref}: {reason}"
        raise InputError(msg)
    if (lo_h, lo_w) == (hi_h, hi_w) or hi_h < lo_h or hi_w < lo_w:
        msg = (
            f"{ref}: the size range {lo_h}x{lo_w}:{hi_h}x{hi_w} needs max >= min in both axes and a real range "
            "(for one size use --size)"
        )
        raise InputError(msg)
    if not (lo_h <= height <= hi_h and lo_w <= width <= hi_w):
        msg = f"{ref}: the size {height}x{width} must lie in the size range {lo_h}x{lo_w}:{hi_h}x{hi_w}"
        raise InputError(msg)
    check_exportable(model, lo_h, lo_w, batch)
    m = _multiple(model)
    if lo_h == hi_h or lo_w == hi_w or min(lo_h, lo_w) // m <= 1:
        msg = (
            f"{ref}: the size range {lo_h}x{lo_w}:{hi_h}x{hi_w} needs min < max on both axes and a minimum of at "
            f"least {2 * m} px (two multiples of {m}) to be dynamic; use a fixed --size"
        )
        raise InputError(msg)
    if any(n % m for n in (height, width, lo_h, lo_w, hi_h, hi_w)):
        msg = (
            f"{ref}: a dynamic export takes sizes that are multiples of {m} (the model's padding); "
            f"got {height}x{width} in {lo_h}x{lo_w}:{hi_h}x{hi_w}"
        )
        raise InputError(msg)


def _dynamic_spec(module: torch.nn.Module, n_inputs: int, axes: dict[int, Any]) -> tuple[Any, ...]:
    """`dynamic_shapes` of `torch.onnx.export` for `n_inputs` tensors that all get `axes`.

    A `forward(*inputs)` takes one nested spec, a `forward(a, b)` one spec per argument.
    """
    per_input = tuple(dict(axes) for _ in range(n_inputs))
    varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in inspect.signature(module.forward).parameters.values()
    )
    return (per_input,) if varargs else per_input


def flat_outputs(out: Any) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - a module's output
    """The tensors of a `module()` output in export order: a dict's values, a tuple's items, or the one tensor."""
    if isinstance(out, dict):
        return tuple(out.values())
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def output_names(model: Model, module: torch.nn.Module, out: Any) -> list[str]:  # noqa: ANN401 - a module's output
    """ONNX / TensorRT output names of `module()`'s output `out`, one per `flat_outputs(out)`.

    A dict's keys; else `module.output_names` where the module declares them (DETR: `logits`, `boxes`; SAM:
    `image_embeddings`); else the variant's main output (`output_name`) for one tensor, `<main>_<i>` for a tuple.
    """
    n = len(flat_outputs(out))
    declared = list(getattr(module, "output_names", ()))
    if isinstance(out, dict):
        names = list(out)
    elif len(declared) == n:
        names = declared
    else:
        names = [output_name(model)] if n == 1 else [f"{output_name(model)}_{i}" for i in range(n)]
    inputs = set(spec(model.card.task).inputs)  # an output named like an input is a cycle for TensorRT's parser
    return [f"{name}_out" if name in inputs else name for name in names]


def export_onnx(  # noqa: PLR0913 - the size, the batch and the optional dynamic range
    model: Model,
    path: str | Path,
    height: int | None = None,
    width: int | None = None,
    *,
    batch: int = 1,
    size_range: SizeBounds | None = None,
) -> Path:
    """Write `model.module()` at `[batch, 3, height, width]` to an ONNX file.

    The size defaults to `model.export_size()` where the task has one (classification, features); else it is required.
    `size_range=((min_h, min_w), (max_h, max_w))` makes height and width dynamic axes of the graph within these bounds
    (the graph is traced at `(height, width)`); the default is a fixed size. A dynamic file is checked: its image
    inputs must have symbolic H / W, and with `onnxruntime` installed it runs at the corners of the range
    (`probe_sizes`) and must match eager `module()` there (`DYNAMIC_TOLERANCE`); a file that fails is removed
    (`InputError`). A size range is refused up front for families outside `DYNAMIC_VERIFIED` (`fixed_size_reason`).
    """
    return _export_onnx(model, path, height, width, batch=batch, size_range=size_range, run=True)


def _export_onnx(  # noqa: PLR0913 - export_onnx's arguments plus the check switch
    model: Model,
    path: str | Path,
    height: int | None,
    width: int | None,
    *,
    batch: int,
    size_range: SizeBounds | None,
    run: bool,
) -> Path:
    """`export_onnx`; `run=False` skips the onnxruntime run of a dynamic file (TensorRT runs its own check)."""
    height, width = resolve_size(model, height, width)
    check_exportable(model, height, width, batch)
    _bounds(model, height, width, size_range, batch)
    for lib in ("onnx", "onnxscript"):  # torch.onnx's dynamo exporter imports both
        if importlib.util.find_spec(lib) is None:
            err = _missing(lib, "onnx")
            raise err
    path = Path(path)
    inputs = model.example_inputs(height, width, batch=batch)
    module = model.module()
    dynamic = None
    if size_range is not None:
        (lo_h, lo_w), (hi_h, hi_w) = size_range
        m = _multiple(model)  # a multiple of m lets the exporter drop the padding arithmetic instead of fixing H, W
        dim_h = m * torch.export.Dim("height", min=lo_h // m, max=hi_h // m)
        dim_w = m * torch.export.Dim("width", min=lo_w // m, max=hi_w // m)
        dynamic = _dynamic_spec(module, len(inputs), {2: dim_h, 3: dim_w})
    with torch.no_grad(), matmul_antialias():  # TensorRT has no antialiased Resize: a fixed size traces matmuls
        try:
            out = module(*inputs)  # lazy state of the size (e.g. Marigold's seeded noise) is built outside the graph
        except AssertionError as e:
            native = model.size_range().native
            if native is None or native == (height, width) or "match model" not in str(e):
                raise
            msg = (
                f"{model.variant.ref}: the network takes exactly {native[0]}x{native[1]} (the transformed crop; the "
                f"model call resizes any image to it), not {height}x{width}: export at {native[0]}x{native[1]} "
                "(the default size)"
            )
            raise InputError(msg) from e
        try:
            program = torch.onnx.export(
                module,
                inputs,
                dynamic_shapes=dynamic,
                dynamo=True,
                input_names=list(spec(model.card.task).inputs)[: len(inputs)],  # multiview: required cameras only
                output_names=output_names(model, module, out),
                verbose=False,
            )
        except Exception as e:
            if dynamic is None:
                raise
            msg = (
                f"{model.variant.ref}: the dynamic-shape export failed ({type(e).__name__}: {str(e)[:200]}); "
                "use a fixed --size"
            )
            raise InputError(msg) from e
    if program is None:  # only the legacy exporter returns None
        msg = "torch.onnx.export returned no program"
        raise RuntimeError(msg)
    _fix_folded_weight_casts(program.model)
    _bf16_deconvolutions_as_convs(program.model)
    program.model.metadata_props["fastforks.model"] = model.variant.ref
    program.model.metadata_props["fastforks.weights"] = (
        "official" if model.weights is None else ", ".join(map(str, model.weights))
    )
    program.save(str(path))  # weights in a side file `<path>.data` if the graph exceeds protobuf's 2 GB
    if size_range is not None:
        _check_dynamic_onnx(model, path, size_range, batch, run=run)
    return path


def probe_sizes(bounds: SizeBounds) -> list[tuple[int, int]]:
    """Sizes a dynamic export is checked at: min, max and the two mixed corners of `bounds`."""
    (lo_h, lo_w), (hi_h, hi_w) = bounds
    return list(dict.fromkeys([(lo_h, lo_w), (hi_h, hi_w), (lo_h, hi_w), (hi_h, lo_w)]))


_IMAGE_RANK = 4  # [B, C, H, W]


def _check_dynamic_onnx(model: Model, path: Path, bounds: SizeBounds, batch: int, *, run: bool) -> None:
    """Raise `InputError` (and remove the file) if the ONNX file at `path` is not dynamic in H / W or is wrong.

    Wrong: it fails or gives other values than eager `module()` at `probe_sizes(bounds)` (`DYNAMIC_TOLERANCE`).

    The run needs `onnxruntime` (skipped without it) and uses its CUDA provider for a CUDA model where available.
    """
    try:
        _probe_dynamic_onnx(model, path, bounds, batch, run=run)
    except Exception:
        path.unlink(missing_ok=True)
        Path(f"{path}.data").unlink(missing_ok=True)
        raise


def _probe_dynamic_onnx(model: Model, path: Path, bounds: SizeBounds, batch: int, *, run: bool) -> None:
    import onnx  # noqa: PLC0415 - optional, present after an export

    ref = model.variant.ref
    graph = onnx.load(str(path), load_external_data=False).graph
    dims = {i.name: i.type.tensor_type.shape.dim for i in graph.input}
    fixed = [
        n for n, d in dims.items() if len(d) == _IMAGE_RANK and any(x.HasField("dim_value") for x in d[2:])
    ]  # images
    if fixed:
        msg = f"{ref}: the exported graph fixes the size of {fixed} (the export is not dynamic); use a fixed --size"
        raise InputError(msg)
    if not run or importlib.util.find_spec("onnxruntime") is None:
        return
    import onnxruntime as ort  # noqa: PLC0415 - optional

    cuda = model.device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(path), providers=providers)
    except Exception as e:  # noqa: BLE001 - an onnxruntime too old for the file: the check is optional
        warnings.warn(f"{ref}: onnxruntime cannot open the file ({e}); the size range was not run", stacklevel=3)
        return
    names = [i.name for i in session.get_inputs()]
    module = model.module()
    where = f"inside its range {bounds[0][0]}x{bounds[0][1]}:{bounds[1][0]}x{bounds[1][1]}"
    for h, w in probe_sizes(bounds):
        inputs = model.example_inputs(h, w, batch=batch)
        feed = {n: t.cpu().numpy() for n, t in zip(names, inputs, strict=False)}
        try:
            got = session.run(None, feed)
        except Exception as e:
            msg = (
                f"{ref}: the dynamic ONNX graph fails at {h}x{w}, {where} ({type(e).__name__}: {str(e)[:200]}); the "
                "file was removed. Export with a fixed --size"
            )
            raise InputError(msg) from e
        with torch.no_grad():
            want = flat_outputs(module(*inputs))
        if wrong := _mismatch(want, got):
            msg = (
                f"{ref}: the dynamic ONNX graph gives wrong values at {h}x{w}, {where} ({wrong}; eager module() is the "
                "reference); the file was removed. Export with a fixed --size"
            )
            raise InputError(msg)


DYNAMIC_TOLERANCE = (1e-2, 0.05, 1e-2)
"""`(mean, outlier, fraction)`: a dynamic file's output matches eager `module()` if its mean |difference| is at most
`mean` and at most `fraction` of its values differ by more than `outlier`, both relative to the output's max |value|.
A graph that keeps the traced size somewhere is wrong over whole rows or the whole image; float16 noise is far below."""


def _mismatch(want: tuple[torch.Tensor, ...], got: Sequence[Any]) -> str:
    """Why the onnxruntime outputs `got` do not match eager's `want` (`DYNAMIC_TOLERANCE`); empty if they match."""
    import numpy as np  # noqa: PLC0415 - onnxruntime's dependency

    mean_tol, outlier, fraction = DYNAMIC_TOLERANCE
    for i, (r, o) in enumerate(zip(want, got, strict=True)):
        ref = r.detach().float().cpu().numpy()
        out = np.asarray(o, dtype=np.float32)
        if out.shape != ref.shape:
            return f"output {i} has shape {list(out.shape)}, eager {list(ref.shape)}"
        finite = np.isfinite(ref)
        if not np.isfinite(out[finite]).all():
            return f"output {i} is not finite where eager is"
        if not finite.any():
            continue
        d = np.abs(out[finite] - ref[finite])
        scale = float(np.abs(ref[finite]).max()) or 1.0
        mean, far = float(d.mean()) / scale, float((d > outlier * scale).mean())
        if mean > mean_tol or far > fraction:
            return (
                f"output {i}: mean |difference| {mean:.2g} of the max |value| {scale:.3g}, {100 * far:.2g} % of the "
                f"values off by more than {outlier:g} of it"
            )
    return ""


def resolve_size(model: Model, height: int | None, width: int | None) -> tuple[int, int]:
    if (height is None) != (width is None):
        msg = "give both height and width (or neither, for the task's default export size where it has one)"
        raise InputError(msg)
    if height is not None and width is not None:
        return height, width
    default = model.export_size()
    if default is None:
        rng = model.size_range()
        eh, ew = rng.example()
        msg = (
            f"{model.variant.ref}: give an export size (this task has no default); valid sizes: "
            f"{rng.describe()}. Example: --size {eh}x{ew} (export_onnx(path, {eh}, {ew}))"
        )
        raise InputError(msg)
    return default


def _fix_folded_weight_casts(model: Any) -> int:  # noqa: ANN401 - an onnx_ir.Model
    """Cast float32 weight constants of fp16 / bf16 convolutions to the input's dtype; returns how many it replaced.

    Under autocast a convolution casts its float32 weights to fp16 / bf16. The exporter's constant folding (torch 2.14,
    onnxscript 0.7) stores some of these casts as float32 initializers again (seen in LAS2 fp16, on the weights the
    two feature passes share; DEFOM bf16), which onnxruntime and TensorRT reject. The replacement is the value the cast
    computes.
    """
    try:
        import onnx_ir as ir  # noqa: PLC0415 - comes with onnxscript (optional)
    except ImportError as e:
        err = _missing("onnx_ir", "onnx")
        raise err from e

    fixed = 0
    for node in model.graph.all_nodes():
        x = node.inputs[0] if node.op_type in {"Conv", "ConvTranspose"} and node.inputs else None
        if x is None or x.dtype not in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
            continue
        for i, v in enumerate(node.inputs[1:], 1):
            if v is None or v.const_value is None or v.dtype != ir.DataType.FLOAT:
                continue
            array = v.const_value.numpy().astype(x.dtype.numpy())  # round to nearest even, as the cast
            half = ir.Value(
                name=f"{v.name}_{x.dtype.name.lower()}_{fixed}",
                shape=ir.Shape(array.shape),
                type=ir.TensorType(x.dtype),
                const_value=ir.tensor(array),
            )
            model.graph.register_initializer(half)
            node.replace_input_with(i, half)
            if not v.uses() and v.name in model.graph.initializers:
                del model.graph.initializers[v.name]
            fixed += 1
    return fixed


def _const_float(v: Any) -> Any:  # noqa: ANN401 - onnx_ir.Value; returns a numpy array or None
    """The float32 array of a constant or of a Cast of one, else None."""
    cast = v.producer() if v is not None else None
    src = cast.inputs[0] if cast is not None and cast.op_type == "Cast" else v
    return None if src is None or src.const_value is None else src.const_value.numpy().astype("float32")


def _upsampling_deconv(node: Any) -> tuple[Any, Any, int] | None:  # noqa: ANN401 - an onnx_ir.Node
    """(weight, bias or None, stride) of an upsampling ConvTranspose, else None.

    Upsampling: constant weights, a square kernel equal to the stride, no padding, dilation or groups.
    """
    w = _const_float(node.inputs[1])
    has_bias = len(node.inputs) > 2 and node.inputs[2] is not None  # noqa: PLR2004 - X, W, B
    bias = _const_float(node.inputs[2]) if has_bias else None
    if w is None or w.ndim != 4 or (has_bias and bias is None) or w.shape[2] != w.shape[3]:  # noqa: PLR2004
        return None
    att = {k: a.value for k, a in node.attributes.items()}
    plain = (
        tuple(att.get("strides", (1, 1))) == tuple(w.shape[2:])
        and set(att.get("pads", (0,))) | set(att.get("output_padding", (0,))) == {0}
        and set(att.get("dilations", (1,))) == {1}
        and att.get("group", 1) == 1
        and "output_shape" not in att
        and att.get("auto_pad", "NOTSET") == "NOTSET"
    )
    return (w, bias, w.shape[2]) if plain else None


def _bf16_deconvolutions_as_convs(model: Any) -> int:  # noqa: ANN401 - an onnx_ir.Model
    """Rewrite bfloat16 ConvTranspose nodes with kernel == stride as a 1x1 Conv + DepthToSpace; returns how many.

    TensorRT 11.4 builds no bfloat16 deconvolution (strongly typed: "Could not infer output types"; in float32
    between casts: "no tactics"; RoMa v2's DPT head). With kernel == stride, no padding and one group the two are
    the same sum: out[co, i*s+a, j*s+b] = sum_ci x[ci, i, j] W[ci, co, a, b] + bias[co].
    """
    import onnx_ir as ir  # noqa: PLC0415 - _fix_folded_weight_casts checked it

    bf16 = ir.DataType.BFLOAT16
    changed = 0
    for node in list(model.graph.all_nodes()):
        if node.op_type != "ConvTranspose" or not node.inputs or node.inputs[0].dtype != bf16:
            continue
        found = _upsampling_deconv(node)
        if found is None:
            continue
        w, bias, s = found
        graph = node.graph
        arrays = [w.transpose(1, 2, 3, 0).reshape(-1, w.shape[0], 1, 1)]  # channel co*s*s + a*s + b (CRD)
        if bias is not None:
            arrays.append(bias.repeat(s * s))
        inits = [
            ir.Value(
                name=f"deconv_as_conv_{changed}_{i}",
                shape=ir.Shape(x.shape),
                type=ir.TensorType(bf16),
                const_value=ir.tensor(x.astype(bf16.numpy())),  # the bf16 values the graph used
            )
            for i, x in enumerate(arrays)
        ]
        for v in inits:
            graph.register_initializer(v)
        conv = ir.Node("", "Conv", [node.inputs[0], *inits], attributes=[ir.AttrInt64s("kernel_shape", [1, 1])])
        conv.outputs[0].dtype = bf16
        attrs = [ir.AttrInt64("blocksize", s), ir.AttrString("mode", "CRD")]
        shuffle = ir.Node("", "DepthToSpace", conv.outputs, attributes=attrs)
        shuffle.outputs[0].dtype, shuffle.outputs[0].shape = bf16, node.outputs[0].shape
        graph.insert_before(node, [conv, shuffle])
        old = list(node.inputs[1:])
        node.outputs[0].replace_all_uses_with(shuffle.outputs[0], replace_graph_outputs=True)
        graph.remove(node, safe=True)
        for v in old:
            cast = v.producer() if v is not None else None
            if cast is not None and cast.op_type == "Cast" and not v.uses():
                v = cast.inputs[0]  # noqa: PLW2901 - read before remove() detaches the inputs
                graph.remove(cast, safe=True)
            if v is not None and v.is_initializer() and not v.uses() and v.name in graph.initializers:
                del graph.initializers[v.name]
        changed += 1
    return changed


def build_tensorrt(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    workspace_gib: float | None = None,
    profile: tuple[SizeBounds, tuple[int, int], int] | None = None,
) -> Path:
    """Build a strongly typed TensorRT engine (precision as in the ONNX graph) for the current GPU.

    `profile=(((min_h, min_w), (max_h, max_w)), (opt_h, opt_w), batch)` adds an optimization profile for an ONNX graph
    with dynamic height / width: every input gets these min / opt / max shapes (TensorRT tunes for `opt`).

    Raises:
        BackendUnavailableError: `tensorrt` is not installed, or the build fails (TensorRT's first error is included).
        RuntimeError: The ONNX file does not parse.
    """
    trt = _tensorrt()
    logger = _trt_logger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):  # from the file: finds external weights (`<name>.data`) next to it
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        msg = f"TensorRT cannot parse {onnx_path}: {errors}"
        raise RuntimeError(msg)
    config = builder.create_builder_config()
    if workspace_gib is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
    if profile is not None:
        (lo, hi), opt, batch = profile
        p = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            p.set_shape(name, (batch, 3, *lo), (batch, 3, *opt), (batch, 3, *hi))
        config.add_optimization_profile(p)
    logger.errors.clear()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        first = logger.errors[0] if logger.errors else "see the TensorRT log above"
        msg = (
            f"TensorRT engine build failed: {first.splitlines()[0][:200]}. Try float32 (no --fp16), or export the "
            "ONNX file (--format onnx) and build it with another TensorRT version"
        )
        raise BackendUnavailableError(msg)
    engine_path = Path(engine_path)
    engine_path.write_bytes(bytes(plan))
    if profile is not None:
        try:
            _smoke(engine_path, profile)
        except Exception:
            engine_path.unlink()
            raise
    return engine_path


def _smoke(engine_path: Path, profile: tuple[SizeBounds, tuple[int, int], int]) -> None:
    """Run a dynamic engine once at the profile's opt size and `probe_sizes`; raise if any fails."""
    (lo, hi), opt, batch = profile
    engine = TensorRTEngine(engine_path)
    for size in dict.fromkeys([opt, *probe_sizes((lo, hi))]):
        xs = [
            torch.rand(batch, 3, *size, device=engine.device, dtype=engine._dtypes[n])  # noqa: SLF001
            for n in engine.inputs
        ]
        try:
            engine(*xs)
            torch.cuda.synchronize(engine.device)
        except Exception as e:
            msg = (
                f"the dynamic engine fails at {size[0]}x{size[1]}, inside its range {lo[0]}x{lo[1]}:{hi[0]}x{hi[1]} "
                f"({type(e).__name__}: {e}); the engine was removed. Export with a fixed --size"
            )
            raise RuntimeError(msg) from e


def export_tensorrt(  # noqa: PLR0913 - export_onnx's arguments plus the build workspace
    model: Model,
    path: str | Path,
    height: int | None = None,
    width: int | None = None,
    *,
    batch: int = 1,
    workspace_gib: float | None = None,
    size_range: SizeBounds | None = None,
) -> Path:
    """ONNX export (in a temporary directory) and TensorRT engine build in one step; returns the engine path.

    With `size_range` the engine takes any height / width inside the bounds (an optimization profile with min = lower
    bound, opt = `(height, width)`, max = upper bound); dynamic engines are not faster than fixed ones, see export.md.
    The engine is run once on `example_inputs`: an all-NaN output removes it and raises `BackendUnavailableError`.
    """
    height, width = resolve_size(model, height, width)
    check_exportable(model, height, width, batch)
    _bounds(model, height, width, size_range, batch)
    _tensorrt()
    if not torch.cuda.is_available():  # before the ONNX export: the builder fails late and cryptically without one
        msg = "a TensorRT engine needs a CUDA GPU: build it on the machine (GPU) that will run it"
        raise BackendUnavailableError(msg)
    with tempfile.TemporaryDirectory(prefix="fastforks-onnx-") as tmp:
        onnx_path = _export_onnx(
            model, Path(tmp) / "model.onnx", height, width, batch=batch, size_range=size_range, run=False
        )
        profile = None if size_range is None else (size_range, (height, width), batch)
        engine_path = build_tensorrt(onnx_path, path, workspace_gib=workspace_gib, profile=profile)
    _check_engine(model, engine_path, height, width, batch)
    return engine_path


def all_nan(outs: Any) -> str | None:  # noqa: ANN401 - a tensor, a tuple / list of them or a dict by name
    """The first floating output that is NaN everywhere (an engine that builds and runs but computes nothing)."""
    named = outs if isinstance(outs, dict) else dict(enumerate(outs if isinstance(outs, (list, tuple)) else [outs]))
    for name, t in named.items():
        if torch.is_tensor(t) and t.is_floating_point() and t.numel() > 0 and bool(t.isnan().all()):
            return str(name)
    return None


def _check_engine(model: Model, engine_path: Path, height: int, width: int, batch: int) -> None:
    """Run the engine once on the export's example inputs; remove it and raise if an output is all NaN."""
    inputs = model.example_inputs(height, width, batch=batch)
    engine = TensorRTEngine(engine_path, inputs[0].device)
    bad = all_nan(engine.run(*inputs))
    del engine
    if bad is not None:
        engine_path.unlink()
        msg = (
            f"{model.variant.ref}: the TensorRT engine outputs NaN everywhere ({bad!r}); the engine was removed. This "
            "is usually a TensorRT bug: run the ONNX export with onnxruntime instead"
        )
        raise BackendUnavailableError(msg)


def _cuda_device(device: str | torch.device) -> torch.device:
    """`device` with its index: tensors report `cuda:0`, never a bare `cuda`, so the engine's device must match that."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _torch_dtype(trt: Any, dtype: Any) -> torch.dtype:  # noqa: ANN401 - tensorrt module and DataType
    table = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in table:
        names = "float32, float16, bfloat16, int32, int64, int8, bool"
        msg = f"TensorRTEngine does not support engine tensors of type {dtype} (supported: {names})"
        raise RuntimeError(msg)
    return table[dtype]


class TensorRTEngine:
    """A TensorRT engine from `export_tensorrt`: `engine(*inputs)` -> the model's main output.

    Inputs are CUDA tensors of the engine's fixed shapes, positional in the order of `module()` or by name (stereo:
    `engine(left, right)` or `engine(left=l, right=r)` -> disparity `[B, H, W]` float32; depth: `engine(image)` -> the
    main map; classification: `engine(crop)` -> logits).

    Attributes:
        inputs: Input names, in call order.
        outputs: Output names.
        shapes: Shape of every input and output; -1 marks a dynamic axis (engine built with `size_range`).
        shape: Shape of the first input.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cuda") -> None:
        """Deserialize the engine at `path` on `device`.

        Raises:
            BackendUnavailableError: `tensorrt` is not installed.
            RuntimeError: The engine does not deserialize here (built for another GPU or TensorRT version).
        """
        trt = _tensorrt()
        self.device = _cuda_device(device)
        with torch.cuda.device(self.device):
            self._runtime = trt.Runtime(_trt_logger())
            self._engine = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self._engine is None:
                msg = f"cannot deserialize {path}: {ENGINE_NOTE} (rebuild it with `fastforks export`)"
                raise RuntimeError(msg)
            self._context = self._engine.create_execution_context()
        # one execution context: calls from several threads take turns, and a call on another stream waits for the
        # previous call's work (the context's activation memory is shared)
        self._lock = threading.Lock()
        self._stream: torch.cuda.Stream | None = None
        e = self._engine
        names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
        self.inputs = tuple(n for n in names if e.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outputs = tuple(n for n in names if n not in self.inputs)
        self.shapes = {n: tuple(e.get_tensor_shape(n)) for n in names}
        self._dtypes = {n: _torch_dtype(trt, e.get_tensor_dtype(n)) for n in names}
        self.shape = self.shapes[self.inputs[0]]

    def _same_shapes(self, inputs: tuple[torch.Tensor, ...]) -> None:
        if len({tuple(t.shape) for t in inputs}) > 1 and any(-1 in self.shapes[n] for n in self.inputs):
            msg = (
                f"the inputs {', '.join(self.inputs)} must have the same shape, got {[tuple(t.shape) for t in inputs]}"
            )
            raise InputError(msg)

    def __call__(self, *inputs: torch.Tensor, **named: torch.Tensor) -> torch.Tensor:
        """Run the engine on the current stream; returns the first output (`run` returns all, by name).

        Args:
            inputs: As `run`.
            named: As `run`.

        Returns:
            The first output tensor.
        """
        return next(iter(self.run(*inputs, **named).values()))

    def run(self, *inputs: torch.Tensor, **named: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the engine on the current stream (thread-safe: calls from several threads or streams take turns).

        Args:
            inputs: One tensor per engine input (`self.inputs`), of its shape and dtype (float32 RGB in `[0, 1]` for
                image inputs, as `module()` takes them), on the engine's device.
            named: The same by input name (`left=`, `right=`, `image=`); positional inputs fill the first names.

        Returns:
            New tensors holding the outputs, by output name.

        Raises:
            InputError: Wrong number of inputs, or a wrong shape, dtype or device.
            RuntimeError: TensorRT execution failed.
        """
        takes = f"the engine takes the inputs ({', '.join(self.inputs)})"
        given = dict(zip(self.inputs, inputs, strict=False)) if len(inputs) <= len(self.inputs) else None
        if given is None or set(named) - set(self.inputs) or set(named) & set(given):
            msg = f"{takes}, got {len(inputs)} positional + {sorted(named)}"
            raise InputError(msg)
        given |= named
        if missing := [n for n in self.inputs if n not in given]:
            msg = f"{takes}, missing {missing}"
            raise InputError(msg)
        inputs = tuple(given[n] for n in self.inputs)
        for name, t in zip(self.inputs, inputs, strict=True):
            want = (self._dtypes[name], self.shapes[name], self.device)
            if -1 in want[1]:  # dynamic engine: any shape inside the optimization profile
                lo, _, hi = self._engine.get_tensor_profile_shape(name, 0)
                inside = all(a <= b <= c for a, b, c in zip(lo, t.shape, hi, strict=True))
                want = (want[0], tuple(t.shape) if inside else f"a shape in {tuple(lo)}..{tuple(hi)}", want[2])
            if (t.dtype, tuple(t.shape), t.device) != want:
                got = f"{t.dtype} {tuple(t.shape)} on {t.device}"
                msg = f"{name}: expected {want[0]} {want[1]} on {want[2]}, got {got}"
                raise InputError(msg)
        self._same_shapes(inputs)
        return self._execute({n: t.contiguous() for n, t in zip(self.inputs, inputs, strict=True)})

    def _execute(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Bind the inputs and new outputs and enqueue the engine, locked (the one execution context is shared)."""
        with self._lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            if self._stream is not None and self._stream != stream:
                stream.wait_stream(self._stream)  # the previous call (another thread / stream) is done with the context
            self._stream = stream
            dynamic = False
            for n, t in tensors.items():
                if -1 in self.shapes[n]:
                    self._context.set_input_shape(n, tuple(t.shape))
                    dynamic = True
            try:
                shapes = {n: tuple(self._context.get_tensor_shape(n)) for n in self.outputs}
            except RuntimeError as e:  # a dynamic engine whose shape graph rejects this size
                if not dynamic:
                    raise
                shapes = dict.fromkeys(self.outputs, (-1,))
                cause: Exception | None = e
            else:
                cause = None
            if dynamic and any(d < 0 for shape in shapes.values() for d in shape):
                size = "x".join(map(str, next(iter(tensors.values())).shape[-2:]))
                msg = (
                    f"the dynamic engine cannot run at {size}: height and width must be multiples of the model's "
                    "padding multiple (`model.size_range().multiple`, as in the export's --size-range)"
                )
                raise InputError(msg) from cause
            tensors |= {n: torch.empty(shapes[n], device=self.device, dtype=self._dtypes[n]) for n in self.outputs}
            for name, t in tensors.items():
                self._context.set_tensor_address(name, t.data_ptr())
            if not self._context.execute_async_v3(stream.cuda_stream):
                msg = "TensorRT execution failed"
                raise RuntimeError(msg)
        return {n: tensors[n] for n in self.outputs}
