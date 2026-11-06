"""Model cards: one `card.toml` per model family, read without importing any model code.

The card is the single source of truth for a model's metadata: identity and revisions, licences, training data,
weights (URL + SHA-256), input handling, parameters, the backends it supports and the kernel libraries they need.
Docs tables, distribution contents and `list_models()` are all derived from it. The format is documented in
CONTRIBUTING.md ("Model cards").
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, get_args

from fastforks._core import licences, tasks
from fastforks._core.errors import CardError, install_hint
from fastforks._core.files import MIRROR_URL, FileSpec
from fastforks._core.labels import label_set
from fastforks._core.licences import CodeClass, Commercial

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import torch

    from fastforks._core.contract import Contract

__all__ = ["CodeClass", "Commercial", "Dtype", "Io", "Licence", "Mode", "ModelCard", "Status", "Tier", "Variant"]

Mode = Literal["portable", "kernels"]
Tier = Literal["stable", "experimental", "deprecated"]
Dtype = Literal["float32", "float16", "bfloat16"]
PadMode = Literal["centre", "top_right", "bottom_right", "none"]
PadFill = Literal["replicate", "zero", "family"]
InputRange = Literal["0-255", "0-1", "imagenet", "mean_std"]
Interpolation = Literal["bilinear", "bicubic", "nearest"]
OptionEffect = Literal["speed", "precision"]

SCHEMA = 2
_T = TypeVar("_T")

LIBRARY_EXTRA: Mapping[str, str] = MappingProxyType(
    {"timm": "timm", "torchvision": "torchvision", "transformers": "transformers", "scipy": "transformers"}
)
"""Optional library (import name) -> the `fastforks-core[...]` extra that installs it; the single source for the model
distributions' dependencies (`tools/build_dists.py`), the load-time check and `ModelCard.requires`."""

LIBRARY_SPEC: Mapping[str, str] = MappingProxyType(
    {"timm": ">=1.0.30", "torchvision": ">=0.23", "transformers": ">=5.19,<6", "scipy": ">=1.10"}
)
"""Supported versions of the optional libs (`>=a.b` and `<c`); `tests/test_optional_deps.py` keeps them equal to the
core pyproject extras, `registry.require` checks them against the installed metadata."""

_CODE_REQUIRES: Mapping[tuple[str, str], tuple[str, ...]] = MappingProxyType(
    {
        # families whose library is only visible in code: `shared.seginfer.pretrained` (transformers builds the network;
        # Mask2Former / OneFormer need scipy for it), `detection._ops` (torchvision NMS / ROIAlign).
        ("segmentation", "segformer"): ("transformers",),
        ("segmentation", "eomt"): ("transformers",),
        ("segmentation", "mask2former"): ("transformers", "scipy"),
        ("segmentation", "oneformer"): ("transformers", "scipy"),
        ("detection", "rcnn"): ("torchvision",),
    }
)


@dataclass(frozen=True)
class Licence:
    """Licensing of a model family; a variant may override `weights`, `evidence` and the training data.

    Attributes:
        code: SPDX expression of the code in the model directory (e.g. `"MIT"`, `"BSD-3-Clause AND CC-BY-NC-4.0"`).
        weights: Licence of the released weights (SPDX expression), or `"assumed from code"` when the original authors
            give them no licence of their own (the code licence then covers them).
        files: Verbatim licence files in the model directory.
        training_data: Dataset ids (`_core/datasets.toml`) the released weights were trained on; `"unstated"` if
            the original authors do not say.
        training_note: Free-text detail on the training data, or `None`.
        evidence: URL backing the weights entry (the original README line, issue or licence file at the pinned
            commit), or `None`.
    """

    code: str
    weights: str
    files: tuple[str, ...]
    training_data: tuple[str, ...]
    training_note: str | None = None
    evidence: str | None = None

    @property
    def code_class(self) -> CodeClass:
        """Class of the code licence; decides the distribution the family ships in."""
        return licences.licence_class(self.code)

    @property
    def weights_licence(self) -> str:
        """The licence expression that holds for the weights (`code` when `weights` is `"assumed from code"`)."""
        return licences.weights_licence(self.weights, self.code)


@dataclass(frozen=True)
class Status:
    """Effective licence status of one variant.

    Attributes:
        code_class: Class of the code licence.
        weights: Licence of the weights for display, e.g. `"MIT"` or `"MIT (assumed from code)"`.
        training_data: Dataset ids.
        commercial: `"yes"` if the code and the weights licence both allow commercial use (training data does not
            change it); `"no"` if either forbids it; else `"unclear"` (a custom licence, treat as no).
        reasons: Why `commercial` is not `"yes"`, one short phrase per blocking item (empty for `"yes"`).
        notes: Caveats that do not change `commercial`: weights licence assumed from the code, and training data with
            restrictive, unclear or unstated terms (e.g. `"data: sceneflow (non-commercial)"`).
    """

    code_class: CodeClass
    weights: str
    training_data: tuple[str, ...]
    commercial: Commercial
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Io:
    """Input handling done by the core before a runner sees the images (card `[io]`).

    Attributes:
        input: Value range the network expects: `"0-255"`, `"0-1"`, `"imagenet"` (`[0, 1]` normalised with the
            ImageNet mean / std) or `"mean_std"` (`[0, 1]` normalised with `mean` / `std`).
        pad_multiple: Height and width are padded to multiples of this.
        pad_mode: `"centre"` (original `InputPadder`), `"top_right"` (pad on top and on the right), `"bottom_right"`
            (below and on the right) or `"none"` (the core pads nothing: the net gets the raw size and pads itself;
            card syntax `pad = "none"`).
        pad_fill: `"replicate"`, `"zero"`, or `"family"` (the runner's own `pad()`, e.g. the original's blurred border).
        min_size: Smallest supported input height / width.
        mean: Per-channel mean for `input = "mean_std"`.
        std: Per-channel standard deviation for `input = "mean_std"`.
        resize: Tasks that resize first (classification): the shorter side is resized to this (0: no resizing).
        crop: Then a centre crop of this size (0: none).
        interpolation: Resampling filter of `resize` (antialiased).
    """

    input: InputRange = "0-255"
    pad_multiple: int = 32
    pad_mode: PadMode = "centre"
    pad_fill: PadFill = "replicate"
    min_size: int = 1
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    resize: int = 0
    crop: int = 0
    interpolation: Interpolation = "bilinear"


@dataclass(frozen=True)
class Variant:
    """One loadable model, e.g. `s2m2-s` revision 1.

    Attributes:
        id: Model id without revision.
        revision: Bumped whenever weights or numerics change, so `id@revision` reproduces outputs.
        weights: Files to download, in the order the model loads them.
        params: Architecture parameters of this variant (override the family's `[params]`).
        licence: Overrides of the family licence (`weights`, `evidence`, `training_data`, `training_note`).
        io: Effective input handling: the family's `[io]` with the variant's `io` overrides applied.
        outputs: Effective outputs: the family's, or the variant's `outputs` (e.g. a checkpoint with a normals head).
        output_info: Effective `[output]` table: the family's with the variant's `output` overrides applied (e.g. a
            metric checkpoint of a relative-depth family, or another label set).
        default_tag: For timm-style ids `<name>.<tag>` (e.g. `resnet50.tv2_in1k`): this variant is what the untagged
            `<name>` loads, as `timm.create_model("resnet50")` picks its default pretrained tag. At most one variant
            per `<name>` and task (checked by the registry).
    """

    id: str
    revision: int
    weights: tuple[FileSpec, ...]
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    licence: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    io: Io = field(default_factory=Io)
    outputs: tuple[str, ...] = ()
    output_info: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    default_tag: bool = False

    @property
    def ref(self) -> str:
        """Versioned id, e.g. `"s2m2-s@1"`."""
        return f"{self.id}@{self.revision}"


@dataclass(frozen=True)
class ModelCard:
    """Metadata of a model family (all variants share code; licence entries may differ per variant).

    Attributes:
        family: Directory / module name, e.g. `"s2m2"`.
        task: Task package, e.g. `"stereo"`.
        name: Display name.
        summary: One sentence.
        tier: Support tier.
        upstream: Upstream repository URL.
        upstream_commit: Upstream commit the vendored code is based on.
        licence: Licensing and training data of the family.
        io: Input handling of the family (a variant may override fields; see `Variant.io`).
        params: Architecture parameters shared by all variants (e.g. `max_disp`, the encoder); variants override them.
        options: Load-time options (`load(..., options=)`) with their validated defaults.
        option_min: Lower bounds of numeric options (card `{ default = ..., min = ... }`); checked at load.
        option_max: Upper bounds of numeric options (card `max = ...`); checked at load.
        option_choices: Allowed values of options with a fixed set (card `choices = [...]`); checked at load.
        option_aliases: Accepted alternative names of options (card `aliases = [...]`, e.g. the upstream name), mapped
            to the canonical name; giving both is an error.
        option_effect: What an option changes if not the results (card `effect`): `"speed"` (the same
            computation; results equal up to floating-point rounding) or `"precision"` (lower-precision arithmetic:
            faster, larger rounding error); options not listed change the outputs.
        uses: Shared modules (`fastforks.shared.<name>`) the model code imports.
        modes: Supported backends.
        portable_cuda_dtype: Validated dtype of the portable backend on CUDA.
        portable_mps_dtypes: Dtypes of the portable backend validated on Apple MPS (card `[modes.portable]
            mps_dtypes`, float32 first: the default); empty if the family is not validated on MPS.
        kernels_dtype: Main compute dtype of the kernels backend (on Apple MPS: `mps_kernels_dtype` if set).
        kernel_libs: Kernel libraries the kernels backend needs on CUDA (empty if kernels is unsupported).
        mps_kernel_libs: Metal kernel libraries of the kernels backend on Apple MPS (card `[modes.kernels]
            mps_kernels`); empty if kernels does not run there.
        mps_kernels_dtype: The kernels backend's dtype on Apple MPS if it differs from `kernels_dtype` (card
            `[modes.kernels] mps_dtype`).
        no_kernels_reason: Why the family has no kernels backend (card `[modes.kernels] status = "none"`, `reason`);
            empty if it has one.
        outputs: Output maps of the family, e.g. `("disparity", "confidence", "occlusion")` (variants may override
            them: `Variant.outputs`; `model_card(id)` returns the id's).
        output_info: What the outputs mean, task-specific (card `[output]`, rules in `fastforks._core.tasks`), e.g.
            `{"kind": "metric"}` or `{"label_set": "imagenet-1k"}` (variants may override keys:
            `Variant.output_info`; `model_card(id)` returns the id's).
        variant_ref: The resolved `"<id>@<revision>"` for a card from `model_card(id)` (whose `io`, `outputs` and
            `output_info` are then the variant's); `None` for a family card.
        variants: Loadable variants.
        limitations: Known limits of the family as short sentences (failure modes, input-size limits, missing
            kernels, graph breaks, nondeterminism, variants without released weights); empty if none are documented.
        licence_notes: Short factual licence notes (1-3 sentences each) on why a variant's commercial status is not a
            plain `yes`/`no` from its licence strings (inherited non-commercial terms, custom licence terms, unclear
            training-data terms); empty if there is nothing to add.
    """

    family: str
    task: str
    name: str
    summary: str
    tier: Tier
    upstream: str
    upstream_commit: str
    licence: Licence
    io: Io
    params: Mapping[str, Any]
    options: Mapping[str, Any]
    option_min: Mapping[str, float]
    uses: tuple[str, ...]
    modes: tuple[Mode, ...]
    portable_cuda_dtype: Dtype
    kernels_dtype: Dtype
    kernel_libs: tuple[str, ...]
    outputs: tuple[str, ...]
    variants: tuple[Variant, ...]
    output_info: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    no_kernels_reason: str = ""
    limitations: tuple[str, ...] = ()
    licence_notes: tuple[str, ...] = ()
    portable_mps_dtypes: tuple[Dtype, ...] = ()
    mps_kernel_libs: tuple[str, ...] = ()
    mps_kernels_dtype: Dtype | None = None
    option_max: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    option_choices: Mapping[str, tuple[Any, ...]] = field(default_factory=lambda: MappingProxyType({}))
    option_effect: Mapping[str, OptionEffect] = field(default_factory=lambda: MappingProxyType({}))
    option_aliases: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    variant_ref: str | None = None

    def option_bounds(self, name: str) -> str:
        """The allowed values of option `name` in words, e.g. `"0 to 9"`, `">= 1"`, `"one of 1, 2"`.

        Args:
            name: An option of `options`.

        Returns:
            The bounds or choices; empty if any value of the default's type is accepted.
        """
        low, high = self.option_min.get(name), self.option_max.get(name)
        if name in self.option_choices:
            return "one of " + ", ".join(repr(c) for c in self.option_choices[name])
        if low is not None and high is not None:
            return f"{low} to {high}"
        if low is not None:
            return f">= {low}"
        return "" if high is None else f"<= {high}"

    def contract(self) -> Contract:
        """The resolved, read-only interface of this model id: inputs, outputs, options, modes, export stages.

        From `<task>.model_card(id)` only (a family card has no single variant). Imports the model code; loads no
        weights. See `fastforks._core.contract.Contract`; `contract().to_dict()` is JSON-ready.

        Raises:
            ModelNotFoundError: A family card (no `variant_ref`).
        """
        from fastforks._core.contract import resolve  # noqa: PLC0415 - the contract reads the registry and runners

        return resolve(self)

    def kernel_libs_on(self, device: torch.device) -> tuple[str, ...]:
        """The kernel libraries of the kernels backend on `device`.

        Args:
            device: The device the backend runs on.

        Returns:
            The Metal libraries on Apple MPS, else the CUDA ones.
        """
        return self.mps_kernel_libs if device.type == "mps" else self.kernel_libs

    def kernels_dtype_on(self, device: torch.device) -> Dtype:
        """The main compute dtype of the kernels backend on `device`.

        Args:
            device: The device the backend runs on.

        Returns:
            `mps_kernels_dtype` on Apple MPS if set, else `kernels_dtype`.
        """
        return (self.mps_kernels_dtype if device.type == "mps" else None) or self.kernels_dtype

    @property
    def requires(self) -> tuple[str, ...]:
        """Optional libraries (import names, sorted) the family needs beyond a bare install, e.g. `("timm",)`.

        From the card's `params.library` (also per variant) plus a small explicit table for families whose library
        is only used in code; `LIBRARY_EXTRA` maps each to the extra that installs it. Empty for most families.
        """
        libs = {self.params.get("library"), *(v.params.get("library") for v in self.variants)}
        libs |= set(_CODE_REQUIRES.get((self.task, self.family), ()))
        return tuple(sorted(x for x in libs if isinstance(x, str)))

    @property
    def extras(self) -> tuple[str, ...]:
        """The `fastforks-core[...]` extras that install `requires` (sorted, no duplicates)."""
        return tuple(sorted({LIBRARY_EXTRA[x] for x in self.requires}))

    def __repr__(self) -> str:
        """Short form: task, family and variant ids (`str(card)` is the readable view)."""
        return f"ModelCard({self.task}/{self.family}, tier={self.tier}, variants={[v.id for v in self.variants]})"

    def __str__(self) -> str:
        """Readable summary: what it is, licences and commercial status per variant, modes, options, limitations."""
        lic = self.licence
        lines = [
            f"{self.name} ({self.task}, tier {self.tier}): {self.summary}",
            f"  original  {self.upstream} @ {self.upstream_commit}",
            f"  licence   code {lic.code}; weights {licences.describe_weights(lic.weights, lic.code)}",
            f"  modes     {', '.join(self.modes)}"
            + (f" (kernel libs: {', '.join(self.kernel_libs)})" if self.kernel_libs else "")
            + (f" (on Apple MPS: {', '.join(self.mps_kernel_libs)})" if self.mps_kernel_libs else ""),
            f"  outputs   {', '.join(self.outputs)}",
            *([f"  requires  {', '.join(self.requires)} ({install_hint(*self.extras)})"] if self.requires else []),
            "  options   "
            + (
                ", ".join(
                    f"{k}={v}" + (f" ({b})" if (b := self.option_bounds(k)) else "") for k, v in self.options.items()
                )
                or "none"
            ),
            "  variants",
        ]
        for v in self.variants:
            st = self.status(v)
            why = f" ({'; '.join(st.reasons)})" if st.reasons else ""
            note = f" [{'; '.join(st.notes)}]" if st.notes else ""
            lines.append(f"    {v.ref:32} commercial: {st.commercial}{why}{note}")
        lines += [f"  limit     {x}" for x in self.limitations]
        lines += [f"  licence   {x}" for x in self.licence_notes]
        lines.append("  more: fastforks info MODEL; help(fastforks.<task>.load)")
        return "\n".join(lines)

    def variant_licence(self, variant: Variant) -> Licence:
        """The family licence with the variant's overrides applied.

        Args:
            variant: A variant of this card.

        Returns:
            The licence terms that hold for that variant.
        """
        o = variant.licence
        lic = self.licence
        return Licence(
            code=lic.code,
            weights=o.get("weights", lic.weights),
            files=lic.files,
            training_data=tuple(o.get("training_data", lic.training_data)),
            training_note=o.get("training_note", lic.training_note if "training_data" not in o else None),
            evidence=o.get("evidence", lic.evidence),
        )

    def status(self, variant: Variant) -> Status:
        """Effective licence status of `variant`: code and weights licence decide, training data is only a note.

        Args:
            variant: A variant of this card.

        Returns:
            The status; `reasons` names what keeps it from `"yes"`, `notes` the caveats that do not.
        """
        lic = self.variant_licence(variant)
        cls = lic.code_class
        commercial = licences.worst(
            licences.code_commercial(cls),
            licences.weights_commercial(lic.weights_licence),
        )
        weights = licences.describe_weights(lic.weights, lic.code)
        return Status(cls, weights, lic.training_data, commercial, _reasons(lic, cls), _notes(lic))

    @property
    def module(self) -> str:
        """Import path of the model's implementation module."""
        return f"fastforks.{self.task}.models.{self.family}.model"


def _get(table: dict[str, Any], key: str, kind: type[_T], where: str) -> _T:
    if key not in table:
        msg = f"{where}: missing key {key!r}"
        raise CardError(msg)
    value = table[key]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        msg = f"{where}: {key!r} must be {kind.__name__}, got {type(value).__name__}"
        raise CardError(msg)
    return value


def _literal(value: object, literal: Any, where: str) -> Any:  # noqa: ANN401 - returns a member of `literal`
    allowed = get_args(literal)
    if value not in allowed:
        msg = f"{where}: {value!r} not in {allowed}"
        raise CardError(msg)
    return value


def _str_tuple(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f"{where}: must be a list of strings"
        raise CardError(msg)
    return tuple(value)


def _opt_str(t: dict[str, Any], key: str, where: str) -> str | None:
    return _get(t, key, str, where) if key in t else None


def _unknown_keys(t: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(t) - allowed
    if extra:
        msg = f"{where}: unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}"
        raise CardError(msg)


def _sha256(t: dict[str, Any], where: str) -> str:
    sha = _get(t, "sha256", str, where)
    if len(sha) != 64 or sha != sha.lower():  # noqa: PLR2004 - hex length of SHA-256
        msg = f"{where}: sha256 must be 64 lower-case hex digits"
        raise CardError(msg)
    return sha


def _file_spec(t: dict[str, Any], where: str, family: str) -> FileSpec:
    _unknown_keys(t, {"name", "sha256", "size", "url", "member", "archive", "mirror"}, where)
    member = _opt_str(t, "member", where)
    if (member is None) != ("archive" not in t):
        msg = (
            f"{where}: an archive member needs `archive = {{ sha256, size }}` of the archive, and only a member has one"
        )
        raise CardError(msg)
    archive = _get(t, "archive", dict, where) if member is not None else {}
    if archive:
        _unknown_keys(archive, {"sha256", "size"}, f"{where} archive")
    name = _get(t, "name", str, where)
    for key in ("url", "mirror"):
        if isinstance(t.get(key), str) and not t[key].startswith("https://"):
            msg = f"{where}: {key} must be an https:// URL"
            raise CardError(msg)
    return FileSpec(
        name=name,
        sha256=_sha256(t, where),
        size=_get(t, "size", int, where),
        url=_get(t, "url", str, where),
        member=member,
        archive_sha256=_sha256(archive, f"{where} archive") if archive else None,
        archive_size=_get(archive, "size", int, f"{where} archive") if archive else None,
        mirror=MIRROR_URL.format(group=family, name=name) if "mirror" in t and _get(t, "mirror", bool, where) else None,
    )


def _training_data(value: object, where: str) -> tuple[str, ...]:
    ids = _str_tuple(value, f"{where} training_data")
    known = licences.datasets()
    unknown = [i for i in ids if i != licences.UNSTATED and i not in known]
    if not ids or unknown:
        msg = f"{where}: training_data must list dataset ids from _core/datasets.toml (or 'unstated'); bad: {unknown}"
        raise CardError(msg)
    return ids


def _licence(t: dict[str, Any], where: str) -> Licence:
    _unknown_keys(t, {"code", "weights", "files", "training_data", "training_note", "evidence"}, where)
    lic = Licence(
        code=_get(t, "code", str, where),
        weights=_get(t, "weights", str, where),
        files=_str_tuple(_get(t, "files", list, where), f"{where} files"),
        training_data=_training_data(_get(t, "training_data", list, where), where),
        training_note=_opt_str(t, "training_note", where),
        evidence=_opt_str(t, "evidence", where),
    )
    licences.licence_class(lic.code, f"{where} code")
    licences.weights_commercial(lic.weights_licence, f"{where} weights")
    return lic


def _variant_licence(t: dict[str, Any], where: str) -> dict[str, Any]:
    _unknown_keys(t, {"weights", "training_data", "training_note", "evidence"}, where)
    out: dict[str, Any] = {k: _get(t, k, str, where) for k in ("weights", "training_note", "evidence") if k in t}
    if "weights" in out and out["weights"] != licences.ASSUMED:
        licences.weights_commercial(out["weights"], f"{where} weights")
    if "training_data" in t:
        out["training_data"] = _training_data(t["training_data"], where)
    return out


def _io(t: dict[str, Any], where: str, task: str, base: Io | None = None) -> Io:
    """Parse an `[io]` table (keys the task allows); absent keys keep the value of `base` (default: `Io()`)."""
    base = base or Io()
    _unknown_keys(t, set(tasks.spec(task).io_keys), where)
    changes = _pad(t.get("pad", {}), f"{where} pad")
    if "input" in t:
        changes["input"] = _literal(t["input"], InputRange, f"{where} input")
    for key in ("min_size", "resize", "crop"):
        if key in t:
            changes[key] = _get(t, key, int, where)
    for key in ("mean", "std"):
        if key in t:
            changes[key] = _floats3(t[key], f"{where} {key}")
    if "interpolation" in t:
        changes["interpolation"] = _literal(t["interpolation"], Interpolation, f"{where} interpolation")
    io = dataclasses.replace(base, **changes)
    if ("mean" in t or "std" in t) and io.input != "mean_std":
        msg = f"{where}: mean / std are only used with input = 'mean_std' (got input = {io.input!r})"
        raise CardError(msg)
    if io.pad_multiple < 1 or io.min_size < 1 or io.resize < 0 or io.crop < 0 or min(io.std) <= 0:
        msg = f"{where}: pad multiple and min_size must be >= 1, resize and crop >= 0, std > 0"
        raise CardError(msg)
    return io


def _pad(pad: object, where: str) -> dict[str, Any]:
    if pad == "none":
        return {"pad_mode": "none", "pad_multiple": 1}
    if not isinstance(pad, dict):
        msg = f"{where}: must be a table or the string 'none'"
        raise CardError(msg)
    _unknown_keys(pad, {"multiple", "mode", "fill"}, where)
    changes: dict[str, Any] = {}
    if "multiple" in pad:
        changes["pad_multiple"] = _get(pad, "multiple", int, where)
    if "mode" in pad:
        changes["pad_mode"] = _literal(pad["mode"], PadMode, f"{where} mode")
    if "fill" in pad:
        changes["pad_fill"] = _literal(pad["fill"], PadFill, f"{where} fill")
    return changes


def _floats3(value: object, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3 or not all(isinstance(x, int | float) for x in value):  # noqa: PLR2004 - RGB
        msg = f"{where}: must be a list of 3 numbers"
        raise CardError(msg)
    return (float(value[0]), float(value[1]), float(value[2]))


def parse_card(data: dict[str, Any], *, family: str, task: str, where: str) -> ModelCard:
    """Validate a parsed `card.toml` and build the card.

    Raises:
        CardError: Missing or unknown keys, wrong types or inconsistent values.
    """
    if data.get("schema") != SCHEMA:
        msg = f"{where}: schema must be {SCHEMA}"
        raise CardError(msg)
    _unknown_keys(
        data,
        {"schema", "name", "summary", "tier", "outputs", "uses", "upstream", "licence", "io", "params", "options"}
        | {"modes", "variants", "output", "limitations", "licence_notes"},
        where,
    )
    up = _get(data, "upstream", dict, where)
    outputs = _str_tuple(_get(data, "outputs", list, where), f"{where} outputs")
    output_t = _get(data, "output", dict, where) if "output" in data else {}
    output_info = _output(task, outputs, output_t, where)
    modes_t = _get(data, "modes", dict, where)
    max_t = _get(modes_t, "kernels", dict, f"{where} [modes.kernels]") if "kernels" in modes_t else {}
    no_kernels_reason = _no_kernels_reason(max_t, f"{where} [modes.kernels]")
    modes = tuple(_literal(m, Mode, f"{where} [modes]") for m in modes_t if not (m == "kernels" and no_kernels_reason))
    if "portable" not in modes:
        msg = f"{where}: every model needs a portable backend"
        raise CardError(msg)
    portable = _get(modes_t, "portable", dict, f"{where} [modes.portable]")
    _unknown_keys(portable, {"cuda_dtype", "mps_dtypes"}, f"{where} [modes.portable]")
    mps_dtypes = tuple(
        _literal(d, Dtype, f"{where} [modes.portable] mps_dtypes")
        for d in _str_tuple(portable.get("mps_dtypes", []), f"{where} [modes.portable] mps_dtypes")
    )
    if mps_dtypes and mps_dtypes[0] != "float32":
        msg = f"{where} [modes.portable]: mps_dtypes must start with float32 (the default on MPS)"
        raise CardError(msg)
    io = _io(_get(data, "io", dict, where) if "io" in data else {}, f"{where} [io]", task)
    variants = []
    for i, v in enumerate(_get(data, "variants", list, where)):
        vw = f"{where} [[variants]] #{i}"
        _unknown_keys(
            v, {"id", "revision", "weights", "params", "licence", "io", "outputs", "output", "default_tag"}, vw
        )
        default_tag = _get(v, "default_tag", bool, vw) if "default_tag" in v else False
        if default_tag and "." not in _get(v, "id", str, vw):
            msg = f"{vw}: default_tag needs a tagged id `<name>.<tag>` (the untagged `<name>` resolves to it)"
            raise CardError(msg)
        v_outputs = _str_tuple(_get(v, "outputs", list, vw), f"{vw} outputs") if "outputs" in v else outputs
        v_output = {**output_t, **_get(v, "output", dict, vw)} if "output" in v else output_t
        variants.append(
            Variant(
                id=_get(v, "id", str, vw),
                revision=_get(v, "revision", int, vw),
                weights=tuple(_file_spec(w, f"{vw} weights", family) for w in _get(v, "weights", list, vw)),
                params=_freeze(_get(v, "params", dict, vw)) if "params" in v else _freeze({}),
                licence=_freeze(_variant_licence(_get(v, "licence", dict, vw), f"{vw} licence"))
                if "licence" in v
                else _freeze({}),
                io=_io(_get(v, "io", dict, vw), f"{vw} io", task, io) if "io" in v else io,
                outputs=v_outputs,
                output_info=_output(task, v_outputs, v_output, vw) if "outputs" in v or "output" in v else output_info,
                default_tag=default_tag,
            )
        )
    if not variants:
        msg = f"{where}: no variants"
        raise CardError(msg)
    options = (
        {k: _option(v, f"{where} [options] {k}") for k, v in _get(data, "options", dict, where).items()}
        if "options" in data
        else {}
    )
    return ModelCard(
        family=family,
        task=task,
        name=_get(data, "name", str, where),
        summary=_get(data, "summary", str, where),
        tier=_literal(_get(data, "tier", str, where), Tier, where),
        upstream=_get(up, "repo", str, f"{where} [upstream]"),
        upstream_commit=_get(up, "commit", str, f"{where} [upstream]"),
        licence=_licence(_get(data, "licence", dict, where), f"{where} [licence]"),
        io=io,
        params=_freeze(_get(data, "params", dict, where)) if "params" in data else _freeze({}),
        options=_freeze({k: v.default for k, v in options.items()}),
        option_min=_freeze({k: v.low for k, v in options.items() if v.low is not None}),
        option_max=_freeze({k: v.high for k, v in options.items() if v.high is not None}),
        option_choices=_freeze({k: v.choices for k, v in options.items() if v.choices}),
        option_effect=_freeze({k: v.effect for k, v in options.items() if v.effect != "output"}),
        option_aliases=_aliases(options, where),
        uses=_str_tuple(data.get("uses", []), f"{where} uses"),
        modes=modes,
        portable_cuda_dtype=_literal(_get(portable, "cuda_dtype", str, where), Dtype, f"{where} [modes.portable]"),
        kernels_dtype=_literal(max_t.get("dtype", "float16"), Dtype, f"{where} [modes.kernels]"),
        kernel_libs=_str_tuple(max_t.get("kernels", []), f"{where} [modes.kernels] kernels"),
        outputs=outputs,
        variants=tuple(variants),
        output_info=output_info,
        no_kernels_reason=no_kernels_reason,
        limitations=_limitations(data.get("limitations", []), f"{where} limitations"),
        licence_notes=_limitations(data.get("licence_notes", []), f"{where} licence_notes", _MAX_LICENCE_NOTE),
        portable_mps_dtypes=mps_dtypes,
        mps_kernel_libs=_str_tuple(max_t.get("mps_kernels", []), f"{where} [modes.kernels] mps_kernels"),
        mps_kernels_dtype=_literal(max_t["mps_dtype"], Dtype, f"{where} [modes.kernels]")
        if "mps_dtype" in max_t
        else None,
    )


def _no_kernels_reason(max_t: dict[str, Any], where: str) -> str:
    """`[modes.kernels]`: either a backend (`kernels`, `dtype`) or `status = "none"` with the measured `reason`."""
    if "status" not in max_t:
        _unknown_keys(max_t, {"kernels", "dtype", "mps_kernels", "mps_dtype"}, where)
        return ""
    _unknown_keys(max_t, {"status", "reason"}, where)
    if max_t["status"] != "none" or not isinstance(max_t.get("reason"), str) or not max_t["reason"].strip():
        msg = f'{where}: status must be "none" with a non-empty reason (why there is no kernels backend)'
        raise CardError(msg)
    return max_t["reason"].strip()


_MAX_LIMITATION = 300
_MAX_LICENCE_NOTE = 450


def _limitations(value: object, where: str, limit: int = _MAX_LIMITATION) -> tuple[str, ...]:
    """Check an optional list of notes: unique, non-empty single-line strings of at most `limit` characters."""
    items = _str_tuple(value, where)
    bad = [i for i in items if not i.strip() or "\n" in i or len(i) > limit]
    if bad or len(set(items)) != len(items):
        msg = f"{where}: entries must be unique, non-empty, one line and <= {limit} characters; bad: {bad}"
        raise CardError(msg)
    return items


def _freeze(value: Any) -> Any:  # noqa: ANN401 - any TOML value
    """`value` deeply read-only: dicts become `MappingProxyType`, lists tuples (a card is shared by every load)."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _output(task: str, outputs: tuple[str, ...], info: dict[str, Any], where: str) -> Mapping[str, Any]:
    """Check `outputs` and the `[output]` table against the task's spec; returns the table read-only."""
    sp = tasks.spec(task)
    if not sp.extras and not set(outputs) <= set(sp.standard):
        msg = f"{where}: {task} outputs must be among {list(sp.standard)}, got {list(outputs)}"
        raise CardError(msg)
    if len(set(outputs)) != len(outputs):
        msg = f"{where}: outputs must be unique, got {list(outputs)}"
        raise CardError(msg)
    if not sp.required & set(outputs):
        msg = f"{where}: outputs must include {' or '.join(repr(o) for o in sorted(sp.required))}"
        raise CardError(msg)
    _unknown_keys(info, set(sp.output_keys), f"{where} [output]")
    for key, values in sp.output_keys.items():
        value = _get(info, key, str, f"{where} [output]")
        if values is None:
            label_set(value)  # raises for an unknown label set
        elif value not in values:
            msg = f"{where} [output] {key}: must be one of {values}, got {value!r}"
            raise CardError(msg)
    return _freeze(info)


class _Opt(NamedTuple):
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()
    effect: str = "output"
    aliases: tuple[str, ...] = ()


def _number(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _aliases(options: Mapping[str, _Opt], where: str) -> Mapping[str, str]:
    """`alias -> option` of all options; an alias must not be an option name or belong to two options."""
    out: dict[str, str] = {}
    for name, opt in options.items():
        for a in opt.aliases:
            if a in options or a in out:
                msg = f"{where} [options] {name}: alias {a!r} is already an option name or alias"
                raise CardError(msg)
            out[a] = name
    return _freeze(out)


def _option(value: object, where: str) -> _Opt:
    """An option as a plain value or `{ default = ..., min = ..., max = ..., choices = [...], effect = ... }`.

    `aliases = ["old_name"]` lists accepted alternative names.
    """
    if not isinstance(value, dict):
        return _Opt(value)
    _unknown_keys(value, {"default", "min", "max", "choices", "effect", "aliases"}, where)
    if "default" not in value:
        msg = f"{where}: needs a `default`"
        raise CardError(msg)
    default, low, high = value["default"], value.get("min"), value.get("max")
    if (low is not None or high is not None) and (
        not _number(default)
        or any(b is not None and not _number(b) for b in (low, high))
        or (low is not None and default < low)
        or (high is not None and default > high)
    ):
        msg = f"{where}: needs a numeric `default` within numeric `min` / `max`"
        raise CardError(msg)
    choices = value.get("choices", [])
    if not isinstance(choices, list) or (
        "choices" in value and (default not in choices or low is not None or high is not None)
    ):
        msg = f"{where}: `choices` must be a list holding the default (without `min` / `max`)"
        raise CardError(msg)
    if any(type(c) is not type(default) for c in choices):  # exact type: not a module
        msg = f"{where}: every choice must have the default's type {type(default).__name__}"
        raise CardError(msg)
    effect = value.get("effect", "output")
    if effect not in (*get_args(OptionEffect), "output"):
        msg = f"{where}: `effect` must be output (the default), speed or precision, got {effect!r}"
        raise CardError(msg)
    aliases = _str_tuple(value.get("aliases", []), f"{where} aliases")
    return _Opt(default, low, high, tuple(choices), effect, aliases)


def read_card(path: Path, *, task: str) -> ModelCard:
    """Read and validate `<family>/card.toml`."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    return parse_card(data, family=path.parent.name, task=task, where=str(path))


def _reasons(lic: Licence, cls: CodeClass) -> tuple[str, ...]:
    """Phrases naming each licence that keeps a variant from being commercially `yes`."""
    out = []
    if licences.code_commercial(cls) != "yes":
        out.append(f"code: {lic.code} ({cls})")
    if licences.weights_commercial(lic.weights_licence) != "yes":
        out.append(f"weights: {licences.describe_weights(lic.weights, lic.code)}")
    return tuple(out)


def _notes(lic: Licence) -> tuple[str, ...]:
    """Caveats that do not change the status: assumed weights licence, restrictive / unclear / unstated data."""
    out = []
    if lic.weights == licences.ASSUMED:
        out.append("weights licence not stated by the original authors; assumed to be the code licence")
    known = licences.datasets()
    out.extend(
        "data: unstated"
        if i == licences.UNSTATED
        else f"data: {i} ({'non-commercial' if known[i].commercial == 'no' else 'unclear'})"
        for i in lic.training_data
        if i == licences.UNSTATED or known[i].commercial != "yes"
    )
    return tuple(out)
